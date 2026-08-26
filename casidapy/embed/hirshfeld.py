"""Hirshfeld (stockholder) density partition → environmental H/XC.

Idea
----
Instead of fading ``V_Hxc[ρ_tot]`` with a geometric ``r_damp``, split the
converged PW density using free-atom UPF reference densities ``ρ̃_B``::

    w_A(r)   = ρ̃_A(r) / Σ_B ρ̃_B(r)          # weight of the edge atom
    ρ_env(r) = (1 − w_A(r)) · ρ_tot(r)       # everything else
    V_env    = V_ionic + v_H[ρ_env] + v_xc[ρ_env]

Near the edge nucleus ``w_A → 1`` so ``ρ_env → 0`` (no PP H/XC on the core).
Far away ``w_A → 0`` so ``ρ_env → ρ_tot`` (full environment).

Entry point: :func:`build_hirshfeld_embedding` (also reached via
:func:`~casidapy.embed.build_ae_embedding_potential` with the default mode).
"""
from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from casidapy.embed.potential import (
    DEFAULT_VLOC_SOURCE,
    ensure_charge_grid,
    ionic_vloc_residual,
    project_radial_to_grid,
    read_upf_local_potential,
    resolve_upf_path,
    ry_to_ha,
    shift_v_env_gauge_at,
)

logger = logging.getLogger(__name__)

DEFAULT_HIRSHFELD_GAUGE_TOL = 1e-6


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
    dens = np.asarray(driver.get_density(gather=True), float)
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
):
    """Compute :class:`HirshfeldPartition` for ``edge_atom``.

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
    if upf_map is None:
        upf_map = build_uspp_map_from_driver(driver)

    rho_tot = total_density_on_grid(driver, grid=grid)
    rho_tilde_sum = np.zeros_like(rho_tot)
    rho_tilde_a_loc = np.zeros_like(rho_tot)
    have_edge = False

    for ia in distributed_indices(n_at, comm):
        sym_raw = symbols[ia]
        sym = str(sym_raw).strip().capitalize()
        path = None
        if upf_map:
            path = upf_map.get(sym) or upf_map.get(sym_raw)
        rho_b = upf_rhoatom_on_grid(driver, ia, grid, upf_path=path)
        rho_tilde_sum += rho_b
        if ia == edge_atom:
            rho_tilde_a_loc = rho_b
            have_edge = True

    rho_tilde_sum = mpi_allreduce_sum(comm, rho_tilde_sum)
    rho_tilde_a = mpi_allreduce_sum(comm, rho_tilde_a_loc)
    # Detect empty edge contribution after reduce (all ranks get the same array)
    if float(np.max(np.abs(rho_tilde_a))) < 1e-30 and not have_edge:
        # Another rank owned the edge atom — array was reduced; check sum charge
        pass
    if float(np.max(np.abs(rho_tilde_a))) < 1e-30:
        # Confirm edge atom exists
        if not (0 <= edge_atom < n_at):
            raise RuntimeError(f"Failed to build ρ̃ for edge_atom={edge_atom}")

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
        logger.info("[hirshfeld] partitioned %d atoms over %d MPI ranks", n_at, size)
    return part.as_tuple()


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
    dens_saved = driver.get_density(gather=True).copy()
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
):
    """``V_env = V_ionic + v_H[ρ_env] + v_xc[ρ_env]`` (Hartree).

    Steps: ionic peel → Hirshfeld ``ρ_env`` → H/XC of ``ρ_env`` → gauge shift.
    Returns ``(v_env, grid, meta)``.

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

    # 1) Ionic environment without the edge local PP
    v_loc_a, v_ionic = ionic_vloc_residual(
        driver, edge_atom, grid, upf_path=upf_a, vloc_source=vloc_source
    )

    # 2) Hirshfeld partition of the SCF density
    upf_map = dict(build_uspp_map_from_driver(driver) or {})
    upf_map[str(sym).strip().capitalize()] = upf_a
    rho_tot, rho_ta, _rho_tsum, w_a, rho_act, rho_env = hirshfeld_weights_and_partition(
        driver,
        edge_atom,
        grid,
        upf_map=upf_map,
        use_gpu=use_gpu,
        comm=comm,
    )

    # 3) Environmental Hartree + XC (QE host kernels; rank-independent)
    v_h = hartree_potential_ha(driver, rho_env)
    v_xc = xc_potential_ha(driver, rho_env, zero_nlcc=True)
    nr = tuple(int(x) for x in grid.nr)
    xp = array_module(use_gpu)
    v_env_arr = (
        xp.asarray(v_ionic, dtype=float).reshape(nr, order="F")
        + xp.asarray(v_h, dtype=float).reshape(nr, order="F")
        + xp.asarray(v_xc, dtype=float).reshape(nr, order="F")
    )
    v_env = DirectField(
        grid=grid,
        rank=1,
        griddata_3d=asnumpy(v_env_arr),
    )

    # 4) Gauge
    gmode = str(gauge_align).lower().strip()
    if gmode in ("hirshfeld", "stockholder", "weight", "default", "auto"):
        v_env, gauge_shift = shift_v_env_gauge_hirshfeld(
            v_env, grid, rho_ta, rho_tot, tol=gauge_tol
        )
        gmode = "hirshfeld"
    else:
        v_env, gauge_shift = shift_v_env_gauge_at(v_env, grid, center_bohr, mode=gmode)

    dV = float(grid.dV)
    meta = {
        "edge_atom": edge_atom,
        "symbol": sym,
        "z_atomic": z_atomic,
        "upf_path": str(upf_a),
        "z_valence": float(z_val),
        "vloc_source": str(vloc_source).lower().strip(),
        "embed_mode": "hirshfeld",
        "gauge_align": gmode,
        "gauge_shift_ha": float(gauge_shift),
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
    }
    if verbose and mpi_is_root(comm):
        logger.info(
            "[hirshfeld] %s atom %d ρ_tot=%.3f ρ_act=%.3f ρ_env=%.3f "
            "w_A(nuc)=%.3f gauge=%s shift=%+.4f <V_env>=%.4f Ha gpu=%s mpi=%s",
            sym,
            edge_atom,
            meta["rho_tot_charge"],
            meta["rho_active_charge"],
            meta["rho_env_charge"],
            meta["w_a_at_nucleus"],
            gmode,
            gauge_shift,
            meta["v_env_mean_ha"],
            use_gpu,
            comm is not None,
        )
    return v_env, grid, meta
