"""Hirshfeld (stockholder) density partition → environmental H/XC (+ KEDF).

Idea
----
Instead of fading ``V_Hxc[ρ_tot]`` with a geometric ``r_damp``, split the
converged PW density using free-atom UPF reference densities ``ρ̃_B``::

    w_A(r)   = Σ_{A∈active} ρ̃_A(r) / Σ_B ρ̃_B(r)   # weight of the active site
    ρ_env(r) = (1 − w_A(r)) · ρ_tot(r)              # everything else
    V_env    = V_ionic + v_H[ρ_env] + v_xc[ρ_env]
             + (δT_s[ρ_tot]/δρ − δT_s[ρ_A]/δρ)     # optional nonadditive KE

Near the edge nucleus ``w_A → 1`` so ``ρ_env → 0`` (no PP H/XC on the core).
Far away ``w_A → 0`` so ``ρ_env → ρ_tot`` (full environment).

The kinetic term is the FDE-style *nonadditive* potential from an approximate
orbital-free KEDF (DFTpy: ``TFvW``, ``LKT``, ``TF``, …).  It is **not**
``v_T[ρ_env]`` alone (that would double-count against the peel KS kinetic).

Entry point: :func:`build_hirshfeld_embedding` (also reached via
:func:`~casidapy.embed.build_ae_embedding_potential` with the default mode).
"""
from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import numpy as np

from casidapy.embed.potential import (
    DEFAULT_VLOC_SOURCE,
    density_functional_potential_ha,
    ensure_charge_grid,
    gathered_qe_array,
    ionic_vloc_residual,
    project_radial_to_grid,
    read_upf_local_potential,
    resolve_upf_path,
    ry_to_ha,
    shift_v_env_gauge_at,
)

logger = logging.getLogger(__name__)

DEFAULT_HIRSHFELD_GAUGE_TOL = 1e-6
DEFAULT_HIRSHFELD_REFERENCE = "neutral"
# Nonadditive OF-DFT kinetic in V_env (DFTpy KEDF name).  ``none`` disables.
DEFAULT_KEDF = "TFvW"
DEFAULT_KEDF_RHO_FLOOR = 1.0e-8
DEFAULT_DENSITY_PARTITION = "hirshfeld"


def _normalize_density_partition(name) -> str:
    mode = str(name or DEFAULT_DENSITY_PARTITION).lower().strip()
    aliases = {
        "hirshfeld": "hirshfeld",
        "stockholder": "hirshfeld",
        "h": "hirshfeld",
        "ihirshfeld": "ihirshfeld",
        "hirshfeld_i": "ihirshfeld",
        "hirshfeld-i": "ihirshfeld",
        "ih": "ihirshfeld",
        "iterative": "ihirshfeld",
        "mbis": "mbis",
        "mbisa": "mbis",
        "minimal_basis": "mbis",
    }
    if mode not in aliases:
        raise ValueError(
            f"Unknown density_partition={name!r}; use 'hirshfeld', "
            "'ihirshfeld', or 'mbis'"
        )
    return aliases[mode]


def _normalize_kedf_name(kedf) -> Optional[str]:
    if kedf is None:
        return None
    name = str(kedf).strip()
    if not name or name.lower() in ("none", "off", "false", "0", "no"):
        return None
    # Common aliases
    aliases = {
        "tf": "TF",
        "vw": "vW",
        "vonweizsacker": "vonWeizsacker",
        "von_weizsacker": "vonWeizsacker",
        "tfvw": "TFvW",
        "tf+vw": "TFvW",
        "lkt": "LKT",
        "ggake": "GGAKE",
        "gga": "GGA",
    }
    return aliases.get(name.lower(), name)


def _normalize_hirshfeld_reference(hirshfeld_reference) -> str:
    mode = str(hirshfeld_reference or DEFAULT_HIRSHFELD_REFERENCE).lower().strip()
    if mode in ("neutral", "upf", "pp_rhoatom", "default", ""):
        return "neutral"
    if mode in ("formal_ion", "formal", "ion", "oxidation", "ox"):
        return "formal_ion"
    raise ValueError(
        f"Unknown hirshfeld_reference={hirshfeld_reference!r}; "
        "use 'neutral' or 'formal_ion'"
    )


# ---------------------------------------------------------------------------
# UPF free-atom densities → grid
# ---------------------------------------------------------------------------

def read_upf_rhoatom(upf_path):
    """``(r_bohr, ρ_radial)`` from UPF ``PP_RHOATOM`` (e⁻/Bohr³, not 4πr²ρ)."""
    try:
        from dftpy.functional.pseudo.upf import UPF

        pp = UPF(upf_path)
        r = np.asarray(pp.r, float)
        rho = np.asarray(pp.atomic_density, float)
        if rho.size != r.size:
            n = min(rho.size, r.size)
            r, rho = r[:n], rho[:n]
        if r[0] <= 1e-10 and rho.size > 1:
            rho = rho.copy()
            rho[0] = rho[1]
        return r, rho
    except Exception as exc:
        logger.warning("dftpy UPF rhoatom failed for %r (%s); regex fallback", upf_path, exc)

    with open(upf_path) as fh:
        text = fh.read()
    m_r = re.search(r"<PP_R>(.*?)</PP_R>", text, re.S | re.I)
    m_rho = re.search(r"<PP_RHOATOM[^>]*>(.*?)</PP_RHOATOM>", text, re.S | re.I)
    if m_r is None or m_rho is None:
        raise ValueError(f"Could not find PP_R / PP_RHOATOM in {upf_path!r}")
    r = np.array(m_r.group(1).split(), dtype=float)
    rho4pi = np.array(m_rho.group(1).split(), dtype=float)
    n = min(len(r), len(rho4pi))
    r, rho4pi = r[:n], rho4pi[:n]
    rho = np.zeros_like(rho4pi)
    if r[0] > 1e-10:
        rho[:] = rho4pi / (4.0 * np.pi * r**2)
    else:
        rho[1:] = rho4pi[1:] / (4.0 * np.pi * r[1:] ** 2)
        rho[0] = rho[1]
    return r, rho


def project_radial_density_to_grid(r_rad, rho_rad, grid, atom_pos_bohr, cell_bohr):
    """Radial atomic density → 3D charge grid (zero outside the UPF table)."""
    return project_radial_to_grid(
        r_rad, rho_rad, grid, atom_pos_bohr, cell_bohr, outside="zero"
    )


def upf_rhoatom_on_grid(driver, atom_index, grid, upf_path=None):
    """UPF free-atom ``ρ̃`` for one atom on ``grid``."""
    atom_index = int(atom_index)
    sym = driver.get_ase_atoms().get_chemical_symbols()[atom_index]
    upf_path = resolve_upf_path(driver, sym, upf_path)
    r_rad, rho_rad = read_upf_rhoatom(upf_path)
    ions = driver.get_dftpy_ions()
    return project_radial_density_to_grid(
        r_rad, rho_rad, grid, ions.positions[atom_index], ions.cell
    )


def total_density_on_grid(driver, grid=None):
    """Converged PW valence density as a flat array (e⁻/Bohr³)."""
    grid = grid if grid is not None else ensure_charge_grid(driver)
    dens = gathered_qe_array(driver, driver.get_density)
    if dens.ndim == 2:
        dens = dens[:, 0]
    return np.asarray(dens, float).ravel()


# ---------------------------------------------------------------------------
# Partition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HirshfeldPartition:
    """Hirshfeld split of ``ρ_tot`` for one edge atom (flat grid arrays)."""

    rho_tot: np.ndarray
    rho_tilde_a: np.ndarray
    rho_tilde_sum: np.ndarray
    w_a: np.ndarray
    rho_active: np.ndarray
    rho_env: np.ndarray

    def as_tuple(self):
        """Legacy unpack order used by older call sites."""
        return (
            self.rho_tot,
            self.rho_tilde_a,
            self.rho_tilde_sum,
            self.w_a,
            self.rho_active,
            self.rho_env,
        )


def hirshfeld_weights_and_partition(
    driver,
    edge_atom,
    grid,
    *,
    upf_map=None,
    use_gpu: bool = False,
    comm=None,
    active_atoms=None,
    hirshfeld_reference: str = DEFAULT_HIRSHFELD_REFERENCE,
    oxidation_state=None,
    oxidation_states=None,
    formal_ion_basis: str = "aug-cc-pvdz",
    formal_ion_xc: str = "pbe",
    formal_ion_valence_only: bool = True,
):
    """Compute :class:`HirshfeldPartition` for the active site.

    ``active_atoms``
        Atom indices whose reference densities are summed as ``ρ̃_A``.
        Defaults to ``[edge_atom]``. Pass the covalently bound fragment
        (O+2H) when the AE SCF is a water molecule.

    ``hirshfeld_reference``
        ``neutral`` (default): UPF ``PP_RHOATOM`` for every atom.
        ``formal_ion``: atomic DFT density at the formal oxidation state
        from :func:`~casidapy.xas.cvs.resolve_oxidation_state` (same table
        as ``ae_electrons='atomic'``), for **every** atom in ``Σ_B ρ̃_B``.

    ``oxidation_state`` / ``oxidation_states``
        Formal charges for ``formal_ion``. ``oxidation_state`` overrides
        only the edge atom; all other atoms use the per-element table.

    Also returns the six arrays as a tuple for backward compatibility::

        rho_tot, rho_tilde_a, rho_tilde_sum, w_a, rho_active, rho_env

    ``comm``
        Optional ``mpi4py`` communicator. Atom projections are round-robin
        distributed and summed with ``Allreduce`` (each rank needs the driver).
    ``use_gpu``
        Use CuPy for the weight / density arithmetic when available.
    """
    from casidapy.adapter.qepy import build_uspp_map_from_driver
    from casidapy.utils.accel import (
        array_module,
        asnumpy,
        distributed_indices,
        mpi_allreduce_sum,
        mpi_rank_size,
    )

    edge_atom = int(edge_atom)
    symbols = driver.get_ase_atoms().get_chemical_symbols()
    n_at = len(symbols)
    active_set = {int(i) for i in (active_atoms if active_atoms is not None else [edge_atom])}
    if not active_set:
        active_set = {edge_atom}
    for ia in active_set:
        if not (0 <= ia < n_at):
            raise ValueError(f"active atom {ia} out of range [0, {n_at})")
    if upf_map is None:
        upf_map = build_uspp_map_from_driver(driver)

    ref_mode = _normalize_hirshfeld_reference(hirshfeld_reference)
    ox_resolver = None
    if ref_mode == "formal_ion":
        from casidapy.embed.formal_ion_density import formal_ion_rhoatom_on_grid
        from casidapy.xas.cvs import resolve_oxidation_state

        def ox_resolver(ia: int) -> int:
            sym_i = str(symbols[ia]).strip().capitalize()
            ox_edge = oxidation_state if ia == edge_atom else None
            return resolve_oxidation_state(sym_i, ox_edge, oxidation_states)

    rho_tot = total_density_on_grid(driver, grid=grid)
    rho_tilde_sum = np.zeros_like(rho_tot)
    rho_tilde_a_loc = np.zeros_like(rho_tot)
    have_active = False

    for ia in distributed_indices(n_at, comm):
        sym_raw = symbols[ia]
        sym = str(sym_raw).strip().capitalize()
        path = None
        if upf_map:
            path = upf_map.get(sym) or upf_map.get(sym_raw)
        if ref_mode == "formal_ion":
            rho_b = formal_ion_rhoatom_on_grid(
                driver,
                ia,
                grid,
                ox_resolver(ia),
                upf_path=path,
                basis=formal_ion_basis,
                xc=formal_ion_xc,
                valence_only=bool(formal_ion_valence_only),
            )
        else:
            rho_b = upf_rhoatom_on_grid(driver, ia, grid, upf_path=path)
        rho_tilde_sum += rho_b
        if ia in active_set:
            rho_tilde_a_loc = rho_tilde_a_loc + rho_b
            have_active = True

    rho_tilde_sum = mpi_allreduce_sum(comm, rho_tilde_sum)
    rho_tilde_a = mpi_allreduce_sum(comm, rho_tilde_a_loc)
    # Detect empty active contribution after reduce (all ranks get the same array)
    if float(np.max(np.abs(rho_tilde_a))) < 1e-30 and not have_active:
        # Another rank owned the active atoms — array was reduced; check sum
        pass
    if float(np.max(np.abs(rho_tilde_a))) < 1e-30:
        if not all(0 <= ia < n_at for ia in active_set):
            raise RuntimeError(f"Failed to build ρ̃ for active_atoms={sorted(active_set)}")

    xp = array_module(use_gpu)
    rt = xp.asarray(rho_tot, dtype=float)
    ra = xp.asarray(rho_tilde_a, dtype=float)
    rs = xp.asarray(rho_tilde_sum, dtype=float)
    w_a = ra / xp.maximum(rs, 1e-30)
    rho_active = w_a * rt
    rho_env = xp.maximum(rt - rho_active, 0.0)

    part = HirshfeldPartition(
        rho_tot=asnumpy(rt),
        rho_tilde_a=asnumpy(ra),
        rho_tilde_sum=asnumpy(rs),
        w_a=asnumpy(w_a),
        rho_active=asnumpy(rho_active),
        rho_env=asnumpy(rho_env),
    )
    rank, size = mpi_rank_size(comm)
    if size > 1 and rank == 0:
        logger.info(
            "[hirshfeld] partitioned %d atoms over %d MPI ranks (ref=%s)",
            n_at,
            size,
            ref_mode,
        )
    return part.as_tuple()


def _partition_active_env(
    driver,
    edge_atom,
    grid,
    *,
    density_partition: str = DEFAULT_DENSITY_PARTITION,
    upf_map=None,
    use_gpu: bool = False,
    comm=None,
    active_atoms=None,
    hirshfeld_reference: str = DEFAULT_HIRSHFELD_REFERENCE,
    oxidation_state=None,
    oxidation_states=None,
    formal_ion_basis: str = "aug-cc-pvdz",
    formal_ion_xc: str = "pbe",
    formal_ion_valence_only: bool = True,
):
    """Return ``(rho_tot, rho_tilde_a, rho_tilde_sum, w_a, rho_act, rho_env, extra)``.

    ``density_partition``
        ``hirshfeld`` (one-shot), ``ihirshfeld`` (Hirshfeld-I), or ``mbis``.
    """
    mode = _normalize_density_partition(density_partition)
    extra: dict = {"density_partition": mode}

    if mode == "hirshfeld":
        rho_tot, rho_ta, rho_tsum, w_a, rho_act, rho_env = hirshfeld_weights_and_partition(
            driver,
            edge_atom,
            grid,
            upf_map=upf_map,
            use_gpu=use_gpu,
            comm=comm,
            active_atoms=active_atoms,
            hirshfeld_reference=hirshfeld_reference,
            oxidation_state=oxidation_state,
            oxidation_states=oxidation_states,
            formal_ion_basis=formal_ion_basis,
            formal_ion_xc=formal_ion_xc,
            formal_ion_valence_only=formal_ion_valence_only,
        )
        return rho_tot, rho_ta, rho_tsum, w_a, rho_act, rho_env, extra

    if mode == "ihirshfeld":
        from casidapy.embed.IHirshfeld import iterative_hirshfeld_from_driver

        res = iterative_hirshfeld_from_driver(
            driver,
            edge_atom,
            grid=grid,
            active_atoms=active_atoms,
            upf_map=upf_map,
            hirshfeld_reference=hirshfeld_reference,
            oxidation_state=oxidation_state,
            oxidation_states=oxidation_states,
            formal_ion_basis=formal_ion_basis,
            formal_ion_xc=formal_ion_xc,
            formal_ion_valence_only=formal_ion_valence_only,
            use_gpu=use_gpu,
            comm=comm,
        )
        part = res.partition
        extra.update(
            {
                "ihirshfeld_converged": bool(res.converged),
                "ihirshfeld_n_iter": int(res.n_iter),
                "ihirshfeld_populations": [float(x) for x in res.populations],
                "ihirshfeld_charges": [float(x) for x in res.charges],
            }
        )
        return (
            part.rho_tot,
            part.rho_tilde_a,
            part.rho_tilde_sum,
            part.w_a,
            part.rho_active,
            part.rho_env,
            extra,
        )

    # MBIS
    from casidapy.embed.mbisa import mbis_from_dftpy_grid
    from casidapy.embed.potential import read_upf_local_potential, resolve_upf_path

    rho_tot = total_density_on_grid(driver, grid=grid)
    ions = driver.get_dftpy_ions()
    symbols = driver.get_ase_atoms().get_chemical_symbols()
    z_valence = []
    for sym_raw in symbols:
        sym = str(sym_raw).strip().capitalize()
        path = None
        if upf_map:
            path = upf_map.get(sym) or upf_map.get(sym_raw)
        path = resolve_upf_path(driver, sym, path)
        _, _, zv = read_upf_local_potential(path)
        z_valence.append(int(round(float(zv))))
    active = [int(i) for i in (active_atoms if active_atoms is not None else [edge_atom])]
    part, mbis_res = mbis_from_dftpy_grid(
        rho_tot,
        grid,
        np.asarray(ions.positions, float),
        z_valence,
        cell=np.asarray(ions.cell, float),
        active_atoms=active,
    )
    extra.update(
        {
            "mbis_converged": bool(mbis_res.converged),
            "mbis_n_iter": int(mbis_res.n_iter),
            "mbis_populations": [float(x) for x in mbis_res.populations],
            "mbis_charges": [float(x) for x in mbis_res.charges],
        }
    )
    return (
        part.rho_tot,
        part.rho_tilde_a,
        part.rho_tilde_sum,
        part.w_a,
        part.rho_active,
        part.rho_env,
        extra,
    )


# ---------------------------------------------------------------------------
# QE potentials of a trial density
# ---------------------------------------------------------------------------

@contextmanager
def _temporary_valence_density(driver, rho_r, *, zero_nlcc=True):
    """Install ``rho_r`` as the valence density; restore on exit.

    Optionally zeros NLCC so ``v_xc`` sees only the trial density (used for
    ``ρ_env``, which should not re-add core corrections twice).
    """
    rho_r = np.asarray(rho_r, float).ravel()
    dens_saved = gathered_qe_array(driver, driver.get_density)
    of_r_saved = np.array(driver.embed.rho.of_r, copy=True)
    of_g_saved = np.array(driver.embed.rho.of_g, copy=True)
    dens_new = dens_saved.copy()
    if dens_new.ndim == 1:
        dens_new = dens_new.reshape(-1, 1)
    dens_new[:, 0] = rho_r

    rho_core_saved = rhog_core_saved = None
    if zero_nlcc:
        rho_core_saved = np.array(driver.qepy_pw.scf.get_array_rho_core(), copy=True)
        rhog_core_saved = np.array(driver.qepy_pw.scf.get_array_rhog_core(), copy=True)
    try:
        driver.set_density(dens_new)
        driver.embed.rho.of_r[:, 0] = rho_r
        if zero_nlcc:
            driver.qepy_pw.scf.set_array_rho_core(np.zeros_like(rho_core_saved))
            driver.qepy_pw.scf.set_array_rhog_core(np.zeros_like(rhog_core_saved))
        yield
    finally:
        driver.set_density(dens_saved)
        driver.embed.rho.of_r[:] = of_r_saved
        driver.embed.rho.of_g[:] = of_g_saved
        if zero_nlcc and rho_core_saved is not None:
            driver.qepy_pw.scf.set_array_rho_core(rho_core_saved)
            driver.qepy_pw.scf.set_array_rhog_core(rhog_core_saved)


def hartree_potential_ha(driver, rho_r):
    """``v_H[ρ]`` on the QE charge grid (Hartree)."""
    rho_r = np.asarray(rho_r, float).ravel()
    nnr = rho_r.size
    v_out = np.zeros((nnr, 1), order="F", dtype=float)
    driver.qepy_pw.v_h_of_rho_r(rhor=rho_r.reshape(nnr, 1), v=v_out)
    return ry_to_ha(v_out[:, 0])


def xc_potential_ha(driver, rho_r, *, zero_nlcc=True):
    """``v_xc[ρ]`` for the active QE functional (Hartree)."""
    rho_r = np.asarray(rho_r, float).ravel()
    with _temporary_valence_density(driver, rho_r, zero_nlcc=zero_nlcc):
        rho_core = driver.qepy_pw.scf.get_array_rho_core()
        rhog_core = driver.qepy_pw.scf.get_array_rhog_core()
        nnr = rho_r.size
        v_out = np.zeros((nnr, 1), order="F", dtype=float)
        driver.qepy_pw.v_xc(driver.embed.rho, rho_core, rhog_core, v_out)
        return ry_to_ha(v_out[:, 0])


def kedf_potential_ha(
    rho_r,
    grid,
    *,
    kedf: str = DEFAULT_KEDF,
    rho_floor: float = DEFAULT_KEDF_RHO_FLOOR,
) -> np.ndarray:
    """Approximate KS kinetic potential ``δT_s[ρ]/δρ`` via a DFTpy KEDF (Ha).

    Returns a flat Fortran-ordered array matching the QE charge-grid layout used
    elsewhere in this module.  Densities are floored at ``rho_floor`` for vW
    stability in vacuum.
    """
    from dftpy.field import DirectField
    from dftpy.functional import Functional

    name = _normalize_kedf_name(kedf)
    if name is None:
        raise ValueError("kedf_potential_ha requires a KEDF name (got none/off)")
    nr = tuple(int(x) for x in grid.nr)
    rho = np.asarray(rho_r, float).ravel()
    if rho.size != int(np.prod(nr)):
        raise ValueError(f"rho size {rho.size} != prod(nr)={np.prod(nr)}")
    rho3 = np.maximum(rho.reshape(nr, order="F"), float(rho_floor))
    dens = DirectField(grid=grid, rank=1, griddata_3d=rho3)
    out = Functional(type="KEDF", name=name)(dens)
    pot = np.asarray(out.potential, dtype=float)
    if pot.shape != nr:
        pot = pot.reshape(nr, order="C")
    return pot.ravel(order="F")


def nonadditive_kinetic_potential_ha(
    rho_tot,
    rho_active,
    grid,
    *,
    kedf: str = DEFAULT_KEDF,
    rho_floor: float = DEFAULT_KEDF_RHO_FLOOR,
) -> tuple[np.ndarray, dict]:
    """FDE-style ``v_T^nad = δT[ρ_tot]/δρ − δT[ρ_A]/δρ`` (Hartree).

    ``ρ_active`` is the Hirshfeld (or peel) active density; ``ρ_tot`` is the
    host density.  Near the absorber ``ρ_A ≈ ρ_tot`` so ``v_T^nad → 0``; in the
    environment it supplies the missing Pauli repulsion.
    """
    name = _normalize_kedf_name(kedf)
    if name is None:
        n = int(np.asarray(rho_tot).size)
        return np.zeros(n, dtype=float), {"kedf": None, "enabled": False}

    v_tot = kedf_potential_ha(rho_tot, grid, kedf=name, rho_floor=rho_floor)
    v_act = kedf_potential_ha(rho_active, grid, kedf=name, rho_floor=rho_floor)
    v_nad = v_tot - v_act
    meta = {
        "kedf": name,
        "enabled": True,
        "rho_floor": float(rho_floor),
        "v_T_tot_mean_ha": float(np.mean(v_tot)),
        "v_T_act_mean_ha": float(np.mean(v_act)),
        "v_T_nad_mean_ha": float(np.mean(v_nad)),
        "v_T_nad_absmax_ha": float(np.max(np.abs(v_nad))),
    }
    return v_nad, meta


def shift_v_env_gauge_min_density(v_env, grid, rho_tot, frac=0.005):
    """Zero ``V_env`` in the true-vacuum region = mean over the lowest-density
    grid points (the ``frac`` fraction with smallest ``ρ_tot``).

    Unlike the Hirshfeld weight gauge (which references the *environment
    interior*) this references the potential to genuine vacuum far from all
    atoms, matching the vacuum zero of a finite all-electron cluster. Mirrors
    the flat-index convention of :func:`shift_v_env_gauge_hirshfeld`.
    """
    from dftpy.field import DirectField

    nr = tuple(int(x) for x in grid.nr)
    v3 = np.asarray(v_env, float).reshape(nr, order="F")
    vflat = v3.ravel()
    rho = np.asarray(rho_tot, float).ravel()
    n = vflat.size
    k = max(1, min(int(float(frac) * n), n - 1))
    idx = np.argpartition(rho, k)[:k]  # k lowest-density (deepest vacuum) points
    shift = float(np.mean(vflat[idx]))
    return DirectField(grid=grid, rank=1, griddata_3d=(v3 - shift)), float(shift)


def sample_field_at_point(field, grid, center_bohr):
    """Value of ``field`` (flat or 3-D) at the grid node nearest ``center_bohr``.

    Same flat/ordering convention as :func:`shift_v_env_gauge_hirshfeld` /
    ``shift_v_env_gauge_at`` so the sampled point is consistent with the gauges.
    """
    nr = tuple(int(x) for x in grid.nr)
    v3 = np.asarray(field, float).reshape(nr, order="F")
    pos = np.asarray(center_bohr, float).reshape(3)
    X = np.asarray(grid.r[0], float)
    Y = np.asarray(grid.r[1], float)
    Z = np.asarray(grid.r[2], float)
    dist = np.sqrt((X - pos[0]) ** 2 + (Y - pos[1]) ** 2 + (Z - pos[2]) ** 2)
    return float(v3.ravel()[int(np.argmin(dist.ravel()))])


def rebuild_v_env_from_active_density(
    driver,
    grid,
    v_ionic,
    rho_tot,
    rho_active,
    *,
    center_bohr,
    gauge_align="true_vacuum",
    gauge_tol=DEFAULT_HIRSHFELD_GAUGE_TOL,
    rho_tilde_a=None,
    fermi_ha=None,
    use_gpu: bool = False,
    kedf: Optional[str] = DEFAULT_KEDF,
    kedf_rho_floor: float = DEFAULT_KEDF_RHO_FLOOR,
):
    """Rebuild ``V_env = V_ionic + v_H/xc[ρ_env] + v_T^nad`` (Hartree).

    Used by self-consistent peel SCF: ``ρ_active`` is the AE fragment density
    (valence-only, charge-matched to the host), while ``V_ionic`` and ``ρ_tot``
    stay frozen from the initial Hirshfeld construction.
    """
    from dftpy.field import DirectField
    from casidapy.utils.accel import array_module, asnumpy

    nr = tuple(int(x) for x in grid.nr)
    rho_tot = np.asarray(rho_tot, float).ravel()
    rho_active = np.asarray(rho_active, float).ravel()
    if rho_tot.size != rho_active.size:
        raise ValueError(
            f"rho_tot size {rho_tot.size} != rho_active size {rho_active.size}"
        )
    rho_env = np.maximum(rho_tot - rho_active, 0.0)
    v_h = hartree_potential_ha(driver, rho_env)
    v_xc = xc_potential_ha(driver, rho_env, zero_nlcc=True)
    v_T_nad, kedf_meta = nonadditive_kinetic_potential_ha(
        rho_tot,
        rho_active,
        grid,
        kedf=kedf,
        rho_floor=kedf_rho_floor,
    )
    xp = array_module(use_gpu)
    v_env_arr = (
        xp.asarray(v_ionic, dtype=float).reshape(nr, order="F")
        + xp.asarray(v_h, dtype=float).reshape(nr, order="F")
        + xp.asarray(v_xc, dtype=float).reshape(nr, order="F")
        + xp.asarray(v_T_nad, dtype=float).reshape(nr, order="F")
    )
    v_env = DirectField(grid=grid, rank=1, griddata_3d=asnumpy(v_env_arr))

    gmode = str(gauge_align).lower().strip()
    fermi_ref_ha = None
    if gmode in ("hirshfeld", "stockholder", "weight", "default", "auto"):
        if rho_tilde_a is None:
            raise ValueError("hirshfeld gauge rebuild requires rho_tilde_a")
        v_env, gauge_shift = shift_v_env_gauge_hirshfeld(
            v_env, grid, rho_tilde_a, rho_tot, tol=gauge_tol
        )
        gmode = "hirshfeld"
    elif gmode in ("true_vacuum", "min_density", "vacuum_global"):
        v_env, gauge_shift = shift_v_env_gauge_min_density(v_env, grid, rho_tot)
        gmode = "true_vacuum"
    elif gmode in ("fermi", "ef", "efermi", "e_fermi"):
        if fermi_ha is None:
            from casidapy.adapter.qepy import fermi_energy_ha

            fermi_ha = fermi_energy_ha(driver)
        v_env, gauge_shift = shift_v_env_gauge_at(
            v_env, grid, center_bohr, mode="fermi", fermi_ha=fermi_ha
        )
        gmode = "fermi"
        fermi_ref_ha = float(fermi_ha)
    else:
        v_env, gauge_shift = shift_v_env_gauge_at(
            v_env, grid, center_bohr, mode=gmode
        )

    dV = float(grid.dV)
    meta = {
        "gauge_align": gmode,
        "gauge_shift_ha": float(gauge_shift),
        "fermi_energy_ha": fermi_ref_ha,
        "rho_active_charge": float(rho_active.sum() * dV),
        "rho_env_charge": float(rho_env.sum() * dV),
        "v_h_env_mean_ha": float(np.mean(v_h)),
        "v_xc_env_mean_ha": float(np.mean(v_xc)),
        "v_env_mean_ha": float(np.mean(np.asarray(v_env))),
        "v_h_at_nuc_ha": sample_field_at_point(v_h, grid, center_bohr),
        "v_xc_at_nuc_ha": sample_field_at_point(v_xc, grid, center_bohr),
        "v_T_nad_at_nuc_ha": sample_field_at_point(v_T_nad, grid, center_bohr),
        "v_env_at_nuc_ha": sample_field_at_point(v_env, grid, center_bohr),
        "kedf": kedf_meta.get("kedf"),
        "kedf_enabled": bool(kedf_meta.get("enabled")),
    }
    for k, v in kedf_meta.items():
        if k not in ("kedf", "enabled"):
            meta[k] = v
    return v_env, meta


def mix_v_env_fields(v_old, v_new, mix, grid):
    """``V ← (1−mix) V_old + mix V_new`` on the DFTpy grid."""
    from dftpy.field import DirectField

    a = float(np.clip(mix, 0.0, 1.0))
    nr = tuple(int(x) for x in grid.nr)
    old = np.asarray(v_old, float).reshape(nr, order="F")
    new = np.asarray(v_new, float).reshape(nr, order="F")
    mixed = (1.0 - a) * old + a * new
    return DirectField(grid=grid, rank=1, griddata_3d=mixed)


def shift_v_env_gauge_hirshfeld(v_env, grid, rho_tilde_a, rho_tot, tol=DEFAULT_HIRSHFELD_GAUGE_TOL):
    """Zero ``V_env`` where ``ρ̃_A / ρ_tot`` is below ``tol`` (else far-field)."""
    from dftpy.field import DirectField

    nr = tuple(int(x) for x in grid.nr)
    v3 = np.asarray(v_env, float).reshape(nr, order="F")
    ratio = np.asarray(rho_tilde_a, float).ravel() / np.maximum(
        np.asarray(rho_tot, float).ravel(), 1e-30
    )
    mask = ratio < float(tol)
    if int(np.count_nonzero(mask)) < 8:
        i_peak = int(np.argmax(rho_tilde_a))
        X = np.asarray(grid.r[0], float).ravel()
        Y = np.asarray(grid.r[1], float).ravel()
        Z = np.asarray(grid.r[2], float).ravel()
        dist = (X - X[i_peak]) ** 2 + (Y - Y[i_peak]) ** 2 + (Z - Z[i_peak]) ** 2
        shift = float(v3.ravel()[int(np.argmax(dist))])
    else:
        shift = float(np.mean(v3.ravel()[mask]))
    return DirectField(grid=grid, rank=1, griddata_3d=(v3 - shift)), float(shift)


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_hirshfeld_embedding(
    driver,
    edge_atom,
    *,
    upf_path=None,
    grid=None,
    verbose=False,
    gauge_align="hirshfeld",
    gauge_tol=DEFAULT_HIRSHFELD_GAUGE_TOL,
    vloc_source=DEFAULT_VLOC_SOURCE,
    use_gpu: bool = False,
    comm=None,
    active_atoms=None,
    hirshfeld_reference: str = DEFAULT_HIRSHFELD_REFERENCE,
    oxidation_state=None,
    oxidation_states=None,
    formal_ion_basis: str = "aug-cc-pvdz",
    formal_ion_xc: str = "pbe",
    formal_ion_valence_only: bool = True,
    kedf: Optional[str] = DEFAULT_KEDF,
    kedf_rho_floor: float = DEFAULT_KEDF_RHO_FLOOR,
    density_partition: str = DEFAULT_DENSITY_PARTITION,
):
    """``V_env = V_ionic + v_H[ρ_env] + v_xc[ρ_env] + v_T^nad`` (Hartree).

    Steps: ionic peel → density partition ``ρ_env`` → H/XC of ``ρ_env`` →
    optional nonadditive KEDF ``v_T[ρ_tot]−v_T[ρ_A]`` → gauge shift.
    Returns ``(v_env, grid, meta)``.

    ``density_partition``
        ``hirshfeld`` (default one-shot), ``ihirshfeld`` (Hirshfeld-I), or
        ``mbis`` (minimal-basis iterative stockholder).

    ``kedf``
        DFTpy KEDF name (``TFvW`` default), or ``none`` / ``None`` to omit
        the nonadditive kinetic potential.

    ``active_atoms``
        Atoms peeled from ``V_ionic`` and from ``ρ_env`` (default: edge atom).
        Must match the AE fragment (O+2H when the SCF is a water molecule).

    ``hirshfeld_reference``
        ``neutral`` (UPF) or ``formal_ion`` (atomic DFT at formal oxidation).
        Used by ``hirshfeld`` / ``ihirshfeld`` seed refs (ignored by MBIS).

    ``use_gpu`` / ``comm``
        Accelerate partition arithmetic (CuPy) and distribute atom UPF
        projections across MPI ranks (each rank must see the QEpy driver).
    """
    from dftpy.field import DirectField
    from pyscf.data import elements
    from casidapy.adapter.qepy import build_uspp_map_from_driver
    from casidapy.utils.accel import array_module, asnumpy, mpi_is_root

    grid = grid if grid is not None else ensure_charge_grid(driver)
    edge_atom = int(edge_atom)
    atoms = driver.get_ase_atoms()
    sym = atoms.get_chemical_symbols()[edge_atom]
    z_atomic = int(elements.charge(str(sym).capitalize()))
    upf_a = resolve_upf_path(driver, sym, upf_path)
    _, _, z_val = read_upf_local_potential(upf_a)
    center_bohr = driver.get_dftpy_ions().positions[edge_atom]
    ref_mode = _normalize_hirshfeld_reference(hirshfeld_reference)
    kedf_name = _normalize_kedf_name(kedf)
    part_mode = _normalize_density_partition(density_partition)

    active = [int(i) for i in (active_atoms if active_atoms is not None else [edge_atom])]
    if not active:
        active = [edge_atom]

    # 1) Ionic environment without the active-site local PP
    v_loc_a, v_ionic = ionic_vloc_residual(
        driver,
        edge_atom,
        grid,
        upf_path=upf_a,
        vloc_source=vloc_source,
        atom_indices=active,
    )

    # 2) Density partition → ρ_active / ρ_env
    upf_map = dict(build_uspp_map_from_driver(driver) or {})
    upf_map[str(sym).strip().capitalize()] = upf_a
    (
        rho_tot,
        rho_ta,
        _rho_tsum,
        w_a,
        rho_act,
        rho_env,
        part_extra,
    ) = _partition_active_env(
        driver,
        edge_atom,
        grid,
        density_partition=part_mode,
        upf_map=upf_map,
        use_gpu=use_gpu,
        comm=comm,
        active_atoms=active,
        hirshfeld_reference=ref_mode,
        oxidation_state=oxidation_state,
        oxidation_states=oxidation_states,
        formal_ion_basis=formal_ion_basis,
        formal_ion_xc=formal_ion_xc,
        formal_ion_valence_only=formal_ion_valence_only,
    )

    # 3) Environmental Hartree + XC (+ optional nonadditive KE)
    v_h = hartree_potential_ha(driver, rho_env)
    v_xc = xc_potential_ha(driver, rho_env, zero_nlcc=True)
    v_T_nad, kedf_meta = nonadditive_kinetic_potential_ha(
        rho_tot,
        rho_act,
        grid,
        kedf=kedf_name,
        rho_floor=kedf_rho_floor,
    )
    nr = tuple(int(x) for x in grid.nr)
    xp = array_module(use_gpu)
    v_env_arr = (
        xp.asarray(v_ionic, dtype=float).reshape(nr, order="F")
        + xp.asarray(v_h, dtype=float).reshape(nr, order="F")
        + xp.asarray(v_xc, dtype=float).reshape(nr, order="F")
        + xp.asarray(v_T_nad, dtype=float).reshape(nr, order="F")
    )
    v_env = DirectField(
        grid=grid,
        rank=1,
        griddata_3d=asnumpy(v_env_arr),
    )

    # 4) Gauge
    gmode = str(gauge_align).lower().strip()
    fermi_ref_ha = None
    if gmode in ("hirshfeld", "stockholder", "weight", "default", "auto"):
        v_env, gauge_shift = shift_v_env_gauge_hirshfeld(
            v_env, grid, rho_ta, rho_tot, tol=gauge_tol
        )
        gmode = "hirshfeld"
    elif gmode in ("true_vacuum", "min_density", "vacuum_global"):
        v_env, gauge_shift = shift_v_env_gauge_min_density(v_env, grid, rho_tot)
        gmode = "true_vacuum"
    elif gmode in ("fermi", "ef", "efermi", "e_fermi"):
        from casidapy.adapter.qepy import fermi_energy_ha

        ef = fermi_energy_ha(driver)
        v_env, gauge_shift = shift_v_env_gauge_at(
            v_env, grid, center_bohr, mode="fermi", fermi_ha=ef
        )
        gmode = "fermi"
        fermi_ref_ha = float(ef)
    else:
        v_env, gauge_shift = shift_v_env_gauge_at(v_env, grid, center_bohr, mode=gmode)
        fermi_ref_ha = None

    dV = float(grid.dV)
    meta = {
        "edge_atom": edge_atom,
        "active_atoms": list(active),
        "symbol": sym,
        "z_atomic": z_atomic,
        "upf_path": str(upf_a),
        "z_valence": float(z_val),
        "vloc_source": str(vloc_source).lower().strip(),
        "embed_mode": "hirshfeld",
        "density_partition": part_mode,
        "hirshfeld_reference": ref_mode,
        "oxidation_state": None if oxidation_state is None else int(oxidation_state),
        "formal_ion_basis": str(formal_ion_basis) if ref_mode == "formal_ion" else None,
        "formal_ion_xc": str(formal_ion_xc) if ref_mode == "formal_ion" else None,
        "formal_ion_valence_only": bool(formal_ion_valence_only) if ref_mode == "formal_ion" else None,
        "kedf": kedf_meta.get("kedf"),
        "kedf_enabled": bool(kedf_meta.get("enabled")),
        "gauge_align": gmode,
        "gauge_shift_ha": float(gauge_shift),
        "fermi_energy_ha": fermi_ref_ha,
        "gauge_tol": float(gauge_tol) if gmode == "hirshfeld" else None,
        "vhxc_scale": 1.0,
        "vhxc_scale_mode": "hirshfeld",
        "r_damp": None,
        "use_gpu": bool(use_gpu),
        "mpi": comm is not None,
        "rho_tot_charge": float(rho_tot.sum() * dV),
        "rho_active_charge": float(rho_act.sum() * dV),
        "rho_env_charge": float(rho_env.sum() * dV),
        "rho_tilde_a_charge": float(rho_ta.sum() * dV),
        "w_a_mean": float(np.mean(w_a)),
        "w_a_at_nucleus": float(w_a[int(np.argmax(rho_ta))]),
        "v_h_env_mean_ha": float(np.mean(v_h)),
        "v_xc_env_mean_ha": float(np.mean(v_xc)),
        "v_loc_a_mean_ha": float(np.mean(np.asarray(v_loc_a))),
        "v_ionic_mean_ha": float(np.mean(np.asarray(v_ionic))),
        "v_env_mean_ha": float(np.mean(np.asarray(v_env))),
        # Decomposition of V_env at the absorber nucleus — this is what rigidly
        # shifts the deep 1s. v_env(R_A) = v_ionic + v_h + v_xc + v_T^nad − gauge.
        "v_ionic_at_nuc_ha": sample_field_at_point(v_ionic, grid, center_bohr),
        "v_h_at_nuc_ha": sample_field_at_point(v_h, grid, center_bohr),
        "v_xc_at_nuc_ha": sample_field_at_point(v_xc, grid, center_bohr),
        "v_T_nad_at_nuc_ha": sample_field_at_point(v_T_nad, grid, center_bohr),
        "v_env_at_nuc_ha": sample_field_at_point(v_env, grid, center_bohr),
    }
    for k, v in kedf_meta.items():
        if k not in ("kedf", "enabled"):
            meta[k] = v
    meta.update(part_extra)
    if verbose and mpi_is_root(comm):
        logger.info(
            "[hirshfeld] %s atom %d part=%s ref=%s kedf=%s ρ_tot=%.3f ρ_act=%.3f ρ_env=%.3f "
            "w_A(nuc)=%.3f gauge=%s shift=%+.4f <V_env>=%.4f Ha "
            "v_T^nad(nuc)=%+.4f Ha gpu=%s mpi=%s",
            sym,
            edge_atom,
            part_mode,
            ref_mode,
            kedf_meta.get("kedf"),
            meta["rho_tot_charge"],
            meta["rho_active_charge"],
            meta["rho_env_charge"],
            meta["w_a_at_nucleus"],
            gmode,
            gauge_shift,
            meta["v_env_mean_ha"],
            meta["v_T_nad_at_nuc_ha"],
            use_gpu,
            comm is not None,
        )
    return v_env, grid, meta


# Soft Thomas–Fermi-like Pauli on the *environment* side only.
# Protected at the absorber by fade=(1−w_A)→0, so no nuclear TFvW blow-up.
DEFAULT_PAULI_SOFT_TF = 1.0  # Ha · (e/Bohr³)^{-2/3}


def _normalize_pauli_mode(pauli) -> str:
    m = str(pauli or "none").lower().strip()
    if m in ("", "none", "off", "false", "0"):
        return "none"
    if m in ("soft_tf", "tf", "pauli_tf", "env_tf"):
        return "soft_tf"
    raise ValueError(
        f"Unknown pauli={pauli!r}; use 'none' or 'soft_tf' "
        "(env-only λ(1−w_A)ρ^{2/3}, nucleus-safe)"
    )


def build_pot_hirshfeld_embedding(
    driver,
    edge_atom,
    *,
    upf_path=None,
    grid=None,
    verbose: bool = False,
    gauge_align: str = "hirshfeld",
    gauge_tol: float = DEFAULT_HIRSHFELD_GAUGE_TOL,
    vloc_source=DEFAULT_VLOC_SOURCE,
    use_gpu: bool = False,
    comm=None,
    active_atoms: Optional[Sequence[int]] = None,
    hirshfeld_reference: str = DEFAULT_HIRSHFELD_REFERENCE,
    oxidation_state=None,
    oxidation_states=None,
    formal_ion_basis: str = "aug-cc-pvdz",
    formal_ion_xc: str = "pbe",
    formal_ion_valence_only: bool = True,
    density_partition: str = "hirshfeld",
    pauli: str = "none",
    pauli_soft_tf: float = DEFAULT_PAULI_SOFT_TF,
):
    """Potential partition with a *density-adaptive* atom/env boundary.

    Unlike legacy ``damp_vhxc`` (fixed ``r_damp``), the fade is the Hirshfeld
    weight complement ``(1 − w_A)`` — the atom/env boundary follows the
    promolecule, not a hand-tuned radius::

        V_env = V_ionic + (1 − w_A) · v_Hxc[ρ_tot]
              + optional soft-TF Pauli on the env side

    No OFDFT ``v_T^nad`` (avoids TFvW nuclear pathology). Optional
    ``pauli='soft_tf'`` adds ``λ (1−w_A) ρ_tot^{2/3}``, which vanishes at the
    absorber (``w_A→1``) so the nucleus stays clean.
    """
    from dftpy.field import DirectField
    from pyscf.data import elements

    from casidapy.adapter.qepy import build_uspp_map_from_driver
    from casidapy.utils.accel import array_module, asnumpy, mpi_is_root

    grid = grid if grid is not None else ensure_charge_grid(driver)
    edge_atom = int(edge_atom)
    atoms = driver.get_ase_atoms()
    sym = atoms.get_chemical_symbols()[edge_atom]
    z_atomic = int(elements.charge(str(sym).capitalize()))
    upf_a = resolve_upf_path(driver, sym, upf_path)
    _, _, z_val = read_upf_local_potential(upf_a)
    center_bohr = driver.get_dftpy_ions().positions[edge_atom]
    ref_mode = _normalize_hirshfeld_reference(hirshfeld_reference)
    part_mode = _normalize_density_partition(density_partition)
    pauli_mode = _normalize_pauli_mode(pauli)
    lam = float(pauli_soft_tf)

    active = [int(i) for i in (active_atoms if active_atoms is not None else [edge_atom])]
    if not active:
        active = [edge_atom]

    v_loc_a, v_ionic = ionic_vloc_residual(
        driver,
        edge_atom,
        grid,
        upf_path=upf_a,
        vloc_source=vloc_source,
        atom_indices=active,
    )

    upf_map = dict(build_uspp_map_from_driver(driver) or {})
    upf_map[str(sym).strip().capitalize()] = upf_a
    (
        rho_tot,
        rho_ta,
        _rho_tsum,
        w_a,
        rho_act,
        rho_env,
        part_extra,
    ) = _partition_active_env(
        driver,
        edge_atom,
        grid,
        density_partition=part_mode,
        upf_map=upf_map,
        use_gpu=use_gpu,
        comm=comm,
        active_atoms=active,
        hirshfeld_reference=ref_mode,
        oxidation_state=oxidation_state,
        oxidation_states=oxidation_states,
        formal_ion_basis=formal_ion_basis,
        formal_ion_xc=formal_ion_xc,
        formal_ion_valence_only=formal_ion_valence_only,
    )

    # Potential partition: fade total Hxc with (1−w_A), not v_Hxc[ρ_env].
    v_hxc, _ = density_functional_potential_ha(driver, grid=grid)
    nr = tuple(int(x) for x in grid.nr)
    xp = array_module(use_gpu)
    w = xp.asarray(w_a, dtype=float).reshape(nr, order="F")
    fade = xp.clip(1.0 - w, 0.0, 1.0)
    vo = xp.asarray(v_ionic, dtype=float).reshape(nr, order="F")
    vh = xp.asarray(v_hxc, dtype=float).reshape(nr, order="F")
    v_env_arr = vo + fade * vh

    v_pauli_mean = 0.0
    v_pauli_nuc = 0.0
    if pauli_mode == "soft_tf":
        if lam < 0.0:
            raise ValueError(f"pauli_soft_tf must be >= 0, got {lam}")
        rho = xp.asarray(rho_tot, dtype=float).reshape(nr, order="F")
        rho_pos = xp.maximum(rho, 1.0e-12)
        v_pauli = lam * fade * (rho_pos ** (2.0 / 3.0))
        v_env_arr = v_env_arr + v_pauli
        v_pauli_np = asnumpy(v_pauli)
        v_pauli_mean = float(np.mean(v_pauli_np))
        v_pauli_nuc = float(sample_field_at_point(v_pauli_np, grid, center_bohr))

    v_env = DirectField(grid=grid, rank=1, griddata_3d=asnumpy(v_env_arr))

    gmode = str(gauge_align).lower().strip()
    fermi_ref_ha = None
    if gmode in ("hirshfeld", "stockholder", "weight", "default", "auto", ""):
        v_env, gauge_shift = shift_v_env_gauge_hirshfeld(
            v_env, grid, rho_ta, rho_tot, tol=gauge_tol
        )
        gmode = "hirshfeld"
    elif gmode in ("true_vacuum", "min_density", "vacuum_global"):
        v_env, gauge_shift = shift_v_env_gauge_min_density(v_env, grid, rho_tot)
        gmode = "true_vacuum"
    elif gmode in ("fermi", "ef", "efermi", "e_fermi"):
        from casidapy.adapter.qepy import fermi_energy_ha

        ef = fermi_energy_ha(driver)
        v_env, gauge_shift = shift_v_env_gauge_at(
            v_env, grid, center_bohr, mode="fermi", fermi_ha=ef
        )
        gmode = "fermi"
        fermi_ref_ha = float(ef)
    else:
        v_env, gauge_shift = shift_v_env_gauge_at(v_env, grid, center_bohr, mode=gmode)

    dV = float(grid.dV)
    fade_np = asnumpy(fade)
    meta = {
        "edge_atom": edge_atom,
        "active_atoms": list(active),
        "symbol": sym,
        "z_atomic": z_atomic,
        "upf_path": str(upf_a),
        "z_valence": float(z_val),
        "vloc_source": str(vloc_source).lower().strip(),
        "embed_mode": "pot_hirshfeld",
        "invert_assemble": (
            "V_ionic + (1-w_A)·v_Hxc[ρ_tot]"
            + (" + λ(1-w_A)ρ^{2/3}" if pauli_mode == "soft_tf" else "")
        ),
        "density_partition": part_mode,
        "hirshfeld_reference": ref_mode,
        "kedf": None,
        "kedf_enabled": False,
        "pauli": pauli_mode,
        "pauli_soft_tf": float(lam) if pauli_mode == "soft_tf" else None,
        "pauli_mean_ha": v_pauli_mean,
        "pauli_at_nuc_ha": v_pauli_nuc,
        "vhxc_fade": "1-w_A",
        "vhxc_fade_mean": float(np.mean(fade_np)),
        "r_damp": None,
        "gauge_align": gmode,
        "gauge_shift_ha": float(gauge_shift),
        "fermi_energy_ha": fermi_ref_ha,
        "rho_tot_charge": float(np.asarray(rho_tot).sum() * dV),
        "rho_active_charge": float(np.asarray(rho_act).sum() * dV),
        "rho_env_charge": float(np.asarray(rho_env).sum() * dV),
        "w_a_at_nucleus": float(sample_field_at_point(w_a, grid, center_bohr)),
        "v_ionic_mean_ha": float(np.mean(np.asarray(v_ionic))),
        "v_loc_a_mean_ha": float(np.mean(np.asarray(v_loc_a))),
        "v_hxc_mean_ha": float(np.mean(np.asarray(v_hxc))),
        "v_env_mean_ha": float(np.mean(np.asarray(v_env))),
        "v_env_at_nuc_ha": float(sample_field_at_point(v_env, grid, center_bohr)),
        **{f"part_{k}": v for k, v in (part_extra or {}).items()},
    }
    if verbose and mpi_is_root(comm):
        logger.info(
            "[pot_hirshfeld] %s atom=%d part=%s pauli=%s fade_mean=%.3f "
            "w_A(nuc)=%.3f Vnuc=%.4f gauge=%s",
            sym,
            edge_atom,
            part_mode,
            pauli_mode,
            meta["vhxc_fade_mean"],
            meta["w_a_at_nucleus"],
            meta["v_env_at_nuc_ha"],
            gmode,
        )
    return v_env, grid, meta
