"""Frozen environment potential ``V_env`` for all-electron (AE) core SCF.

Units
-----
QE arrays are Rydberg. Everything returned here is Hartree.

Production recipe (default ``embed_mode="hirshfeld"``)
------------------------------------------------------
1. Peel the active-site local PP from QE (edge atom, or a covalently bound
   fragment such as O+2H)::

       V_loc(active) = Σ_{A∈active} setlocal with only atom A's structure factor
       V_ionic       = vltot − V_loc(active)   # 0 for a lone atom in vacuum

   Periodic crystals (production ``ae_host``): ``setlocal`` is the A
   *sublattice* (absorber plus supercell images) while PySCF ``−Z/r`` is
   one-center. Subtracting the sublattice punches charged holes at every
   image of A and shifts the 1s by a lattice-dependent Madelung (~Ha).
   ``vloc_source="home"`` peels only the home copy; clusters with
   ``assume_isolated`` keep the vacuum-null ``setlocal`` peel.

2. Partition the PW density with Hirshfeld weights from UPF ``PP_RHOATOM``::

       w_active = Σ_{A∈active} ρ̃_A / Σ_B ρ̃_B
       ρ_env    = (1 − w_active) · ρ_tot

3. Build the frozen embedding field::

       V_env = V_ionic + v_H[ρ_env] + v_xc[ρ_env]
             + (δT_s[ρ_tot]/δρ − δT_s[ρ_A]/δρ)   # nonadditive KEDF (default TFvW)

PySCF then supplies kinetic energy + (−Z/r) for the edge atom only.

Other ``embed_mode`` values (``loc_only``, ``damp_vhxc``, …) are legacy /
diagnostic; see :func:`build_ae_embedding_potential`.

Public entry point: :func:`build_ae_embedding_potential`.
"""
from __future__ import annotations

import logging
import re

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_EMBED_MODE = "hirshfeld"
DEFAULT_R_DAMP = 1.0  # Bohr; legacy damp_vhxc only
DEFAULT_GAUGE_ALIGN = "far_field"
DEFAULT_VLOC_SOURCE = "qepy"  # clusters / vacuum null; crystals use "home" via "auto"

# Frozen-core oneshot: ρ_tot (no PP_RHOATOM peel) + XC on ρ_core^AE + ρ_tot.
AE_HOST_HXC_RHO_TOT = frozenset({"rho_tot", "core_plus_tot", "xc_sum", "valence_screen"})


def normalize_ae_host_hxc(name) -> str:
    """Canonical frozen-core ``ae_host_hxc`` tag (production: ``rho_tot``)."""
    s = str(name or "rho_tot").lower().strip()
    if s in AE_HOST_HXC_RHO_TOT:
        return "rho_tot"
    raise ValueError(
        f"ae_host_hxc={name!r} is retired; production is rho_tot "
        "(v_H[ρ_tot] on FFT, v_H[core]+v_xc[sum] on Becke)"
    )

_VHXC_SCALE_ALPHA = 0.4
_VHXC_SCALE_MIN = 0.5
_VHXC_SCALE_MAX = 1.0

_EMBED_MODE_ALIASES = {
    "default": DEFAULT_EMBED_MODE,
    "hirshfeld": "hirshfeld",
    "stockholder": "hirshfeld",
    "density_ratio": "hirshfeld",
    "rho_ratio": "hirshfeld",
    "partition": "hirshfeld",
    "invert": "invert",
    "invert_env": "invert",
    "invert_xc": "invert",
    "zmp": "invert",
    "pot_hirshfeld": "pot_hirshfeld",
    "pot_partition": "pot_hirshfeld",
    "fade_vhxc": "pot_hirshfeld",
    "vhxc_fade": "pot_hirshfeld",
    "damped": "damp_vhxc",
    "radial_damp": "damp_vhxc",
    "damp_vhxc": "damp_vhxc",
    "scale": "scale_vhxc",
    "scale_vhxc": "scale_vhxc",
    "vloc_only": "loc_only",
    "local_only": "loc_only",
    "no_vhxc": "loc_only",
    "vloc": "loc_only",
    "loc_only": "loc_only",
    "zero": "zero",
    "none": "zero",
    "off": "zero",
    # AE absorber in the host's own potential (V_eff - v_loc_A + V_NL env).
    "ae_host": "ae_host",
    "host_ae": "ae_host",
    "ae_in_host": "ae_host",
    "veff": "ae_host",
    "full": "ae_host",
}


# ---------------------------------------------------------------------------
# Units / grid helpers
# ---------------------------------------------------------------------------

def ry_to_ha(arr):
    """Rydberg → Hartree."""
    return np.asarray(arr, dtype=float) * 0.5


def ensure_charge_grid(driver):
    """DFTpy grid matching the QE charge / potential mesh."""
    nr = list(driver.get_number_of_grid_points())
    if hasattr(driver, "get_dftpy_grid"):
        return driver.get_dftpy_grid(nr=nr)
    from dftpy.grid import DirectGrid

    return DirectGrid(lattice=driver.get_dftpy_ions().cell, nr=nr)


def _as_field(grid, v3):
    from dftpy.field import DirectField

    return DirectField(grid=grid, rank=1, griddata_3d=np.asarray(v3, float))


def _qe_gather_comm(driver):
    """Communicator QE gathers over, or ``None`` when QE runs on a single rank."""
    if int(getattr(driver, "nproc", 1) or 1) <= 1:
        return None
    comm = getattr(driver, "comm", None)
    if comm is not None and hasattr(comm, "Bcast"):
        return comm
    try:
        from mpi4py import MPI
    except ImportError:
        return None
    # casidapy builds the Driver without an explicit comm, so QE starts itself
    # on MPI_COMM_WORLD and ``driver.comm`` stays None.
    return MPI.COMM_WORLD if MPI.Is_initialized() else None


def gathered_qe_array(driver, getter, **kwargs):
    """A ``gather=True`` QE real-space array, replicated on every rank.

    QEpy sizes the gathered buffer only on the root rank and hands every other
    rank a one-element stub (``Driver.create_array``), which ``data2field``
    then quietly turns into ``np.zeros(1)``. Embedding is replicated work that
    assumes every rank holds the whole grid, so without this the ranks disagree
    on array shapes and desync at the next collective. ``Driver.set_density``
    broadcasts for the same reason.
    """
    data = np.asarray(getter(gather=True, **kwargs), float)
    comm = _qe_gather_comm(driver)
    if comm is None:
        return data

    shape = comm.bcast(data.shape if driver.is_root else None, root=0)
    # Fortran order: these buffers get handed straight back to QE routines.
    buf = np.empty(shape, dtype=float, order="F")
    if driver.is_root:
        buf[...] = data
    comm.Bcast(buf, root=0)
    return buf


def _qe_pot_flat_ha(driver, getter):
    """Gather a QE real-space potential and return a Fortran-flat Hartree array."""
    data = gathered_qe_array(driver, getter)
    if data.ndim == 2:
        data = data[:, 0]
    return ry_to_ha(np.asarray(data, float).ravel(order="F"))


def _qe_field_ha(driver, getter, grid=None):
    """Fetch a QE real-space potential (Rydberg) and return (field_Ha, grid)."""
    grid = grid if grid is not None else ensure_charge_grid(driver)
    data = gathered_qe_array(driver, getter)
    if data.ndim == 2:
        data = data[:, 0]
    return driver.data2field(ry_to_ha(data), grid=grid), grid


def effective_potential_ha(driver, grid=None):
    return _qe_field_ha(driver, driver.get_effective_potential, grid)


def density_functional_potential_ha(driver, grid=None):
    """``V_Hxc[ρ_tot]`` from QE (Hartree)."""
    return _qe_field_ha(driver, driver.get_density_functional_potential, grid)


def local_pp_total_ha(driver, grid=None):
    """Total local PP ``vltot`` (Hartree)."""
    return _qe_field_ha(driver, driver.get_local_pp, grid)


def _minimum_image_distances(grid, atom_pos_bohr, cell_bohr):
    """Minimum-image distances from ``atom_pos_bohr`` to every grid point."""
    from casidapy.utils.uspp import _minimum_image_displacement

    pos = np.asarray(atom_pos_bohr, float).reshape(3)
    rx, ry, rz = grid.r[0], grid.r[1], grid.r[2]
    dx, dy, dz = _minimum_image_displacement(
        pos, np.stack([rx, ry, rz], axis=0), np.asarray(cell_bohr, float)
    )
    return np.sqrt(dx**2 + dy**2 + dz**2)


def project_radial_to_grid(
    r_rad,
    y_rad,
    grid,
    atom_pos_bohr,
    cell_bohr,
    *,
    outside="coulomb",
    fill_value=0.0,
):
    """Interpolate a radial function ``y(r)`` onto the 3D charge grid.

    ``outside``:
      - ``"coulomb"`` — continue as ``y_last · r_last / r`` (local PP tail)
      - ``"zero"`` — set exterior to 0 (atomic densities)
    """
    from scipy.interpolate import interp1d

    r_rad = np.asarray(r_rad, float)
    y_rad = np.asarray(y_rad, float)
    dist = _minimum_image_distances(grid, atom_pos_bohr, cell_bohr)
    kind = "coulomb" if outside == "coulomb" else "zero"
    if kind == "coulomb":
        y3 = interp1d(r_rad, y_rad, kind="cubic", fill_value="extrapolate", bounds_error=False)(dist)
        outside_mask = dist > r_rad[-1]
        y3[outside_mask] = (y_rad[-1] * r_rad[-1]) / np.maximum(dist[outside_mask], 1e-12)
    else:
        y3 = interp1d(r_rad, y_rad, kind="cubic", fill_value=fill_value, bounds_error=False)(dist)
        y3[dist > r_rad[-1]] = 0.0
        y3 = np.maximum(y3, 0.0)
    return np.asarray(y3, float).ravel(order="F")


def project_radial_vloc_to_grid(r_rad, v_rad_ha, grid, atom_pos_bohr, cell_bohr):
    """Radial UPF ``V_loc(r)`` → 3D grid (Hartree), Coulomb tail outside table."""
    return project_radial_to_grid(
        r_rad, v_rad_ha, grid, atom_pos_bohr, cell_bohr, outside="coulomb"
    )


# ---------------------------------------------------------------------------
# UPF I/O
# ---------------------------------------------------------------------------

def resolve_upf_path(driver, atom_symbol, upf_path=None):
    """UPF path for ``atom_symbol``, from ``upf_path`` or the driver's PP map."""
    if upf_path:
        return str(upf_path)
    from casidapy.adapter.qepy import build_uspp_map_from_driver

    def _base(sym: str) -> str:
        """``Mn1`` / ``Cr2`` → ``Mn`` / ``Cr`` (QE AFM multi-type labels)."""
        letters = "".join(c for c in str(sym) if c.isalpha())
        return letters.capitalize() if letters else str(sym).strip().capitalize()

    sym = _base(atom_symbol)
    uspp_map = build_uspp_map_from_driver(driver)
    if sym in uspp_map:
        return str(uspp_map[sym])
    for key, path in uspp_map.items():
        if _base(key) == sym:
            return str(path)
    raise FileNotFoundError(f"No UPF for {sym!r}; pass upf_path=...")


def read_upf_local_potential(upf_path):
    """``(r_bohr, vloc_ha, z_valence)`` from a UPF file."""
    try:
        from dftpy.functional.pseudo.upf import UPF

        pp = UPF(upf_path)
        return np.asarray(pp.r, float), np.asarray(pp.v, float), float(pp.zval)
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("dftpy UPF parse failed for %r (%s); regex fallback", upf_path, exc)

    with open(upf_path) as fh:
        text = fh.read()
    m_v = re.search(r"<PP_LOCAL>(.*?)</PP_LOCAL>", text, re.S | re.I)
    m_r = re.search(r"<PP_R>(.*?)</PP_R>", text, re.S | re.I)
    if m_v is None or m_r is None:
        raise ValueError(f"Could not find PP_LOCAL/PP_R blocks in {upf_path!r}")
    v_ry = np.array(m_v.group(1).split(), dtype=float)
    r = np.array(m_r.group(1).split(), dtype=float)
    z_val = 0.0
    for line in text.splitlines():
        if "Z valence" in line:
            z_val = float(line.split()[0])
            break
    n = min(len(r), len(v_ry))
    return r[:n], ry_to_ha(v_ry[:n]), z_val


def read_upf_z_valence(upf_path):
    return read_upf_local_potential(upf_path)[2]


# ---------------------------------------------------------------------------
# V_loc(A): QE setlocal peel (production) and UPF interpolate (diagnostic)
# ---------------------------------------------------------------------------

def _miller_from_fft(nl, nr1, nr2, nr3, nr1x, nr2x):
    """Miller indices for QE FFT G-vectors (0-based ``nl``)."""
    n0 = np.asarray(nl, dtype=int)
    i = n0 % nr1x
    t = n0 // nr1x
    j = t % nr2x
    k = t // nr2x
    return np.stack(
        [
            np.where(i <= nr1 // 2, i, i - nr1),
            np.where(j <= nr2 // 2, j, j - nr2),
            np.where(k <= nr3 // 2, k, k - nr3),
        ],
        axis=1,
    )


def _qe_species_index(driver, atom_index):
    """0-based ntyp column for ``atom_index`` from QE ``ityp``.

    Species order is ATOMIC_SPECIES, not first appearance in the atom list.
    This repo's pw.x inputs sort species alphabetically (``O`` then ``Ti``),
    while rutile / ``(TiO2)5`` xyz files list Ti first. First-appearance
    indexing therefore peels Ti ``v_loc`` at oxygen and shifts the
    reconstructed 1s by ~6 Ha (~175 eV).
    """
    ib = driver.qepy_modules.ions_base
    ia = int(atom_index)
    nat = int(ib.get_nat())
    if ia < 0 or ia >= nat:
        raise IndexError(f"atom_index={ia} out of range for nat={nat}")
    ityp = np.asarray(ib.get_array_ityp(), dtype=int).ravel()
    if ia >= ityp.size:
        raise IndexError(f"atom_index={ia} out of range for ityp size={ityp.size}")
    idx = int(ityp[ia]) - 1  # Fortran 1-based
    if idx < 0:
        raise RuntimeError(f"QE ityp[{ia}]={ityp[ia]} is not a valid species")
    return idx


def _qe_species_label(driver, species_index):
    """Element label for 0-based QE species ``species_index``."""
    raw = driver.qepy_modules.ions_base.get_array_atm().T.view("S3")[:, 0].astype("U3")
    labels = [str(x).strip() for x in raw]
    isp = int(species_index)
    if isp < 0 or isp >= len(labels):
        raise IndexError(f"species_index={isp} out of range for nsp={len(labels)}")
    return labels[isp]


def vloc_atom_from_qepy(driver, edge_atom):
    """Single-atom local PP via QE ``strf`` + ``setlocal`` (Rydberg).

    Same machinery as ``vltot``, so ``vltot − V_loc(A)`` is exactly zero for a
    lone atom (vacuum null test). Restores the original structure factor.
    """
    edge_atom = int(edge_atom)
    symbols = list(driver.get_ions_symbols())
    isp = _qe_species_index(driver, edge_atom)
    sp_label = _qe_species_label(driver, isp)
    atom_label = str(symbols[edge_atom]).strip()
    if sp_label.capitalize() != atom_label.capitalize():
        raise RuntimeError(
            f"QE species column {isp} is {sp_label!r} but atom {edge_atom} "
            f"is {atom_label!r}; refusing to peel the wrong v_loc"
        )
    dfftp = driver.embed.dfftp
    nr1, nr2, nr3 = int(dfftp.nr1), int(dfftp.nr2), int(dfftp.nr3)
    bg = 2.0 * np.pi * np.linalg.inv(np.asarray(driver.get_ions_lattice(), float)).T
    tau = np.asarray(driver.get_ions_positions(), float)[edge_atom]
    ngm = int(dfftp.ngm)
    nl = np.asarray(dfftp.nl, dtype=int)[:ngm] - 1
    mills = _miller_from_fft(nl, nr1, nr2, nr3, int(dfftp.nr1x), int(dfftp.nr2x))
    phases = np.exp(-1j * ((mills.astype(float) @ bg) @ tau))

    vlmod = driver.qepy_pw.vlocal
    strf0 = np.array(vlmod.get_array_strf(), copy=True)
    strf_a = np.zeros_like(strf0)
    if isp >= strf_a.shape[1]:
        raise RuntimeError(
            f"QE species column {isp} >= strf nsp={strf_a.shape[1]}"
        )
    strf_a[:, isp] = phases
    try:
        vlmod.set_array_strf(strf_a)
        driver.qepy_pw.qepy_setlocal()
        v_a = gathered_qe_array(driver, driver.get_local_pp)
        if v_a.ndim == 2:
            v_a = v_a[:, 0]
        return v_a.ravel().copy()
    finally:
        vlmod.set_array_strf(strf0)
        driver.qepy_pw.qepy_setlocal()


def vloc_atom_field_upf(driver, atom_index, grid, upf_path=None):
    """UPF ``V_loc(|r−R_A|)`` on ``grid`` (Hartree). Diagnostic only."""
    atom_index = int(atom_index)
    sym = driver.get_ase_atoms().get_chemical_symbols()[atom_index]
    upf_path = resolve_upf_path(driver, sym, upf_path)
    r_rad, v_rad_ha, _ = read_upf_local_potential(upf_path)
    ions = driver.get_dftpy_ions()
    v_1d = project_radial_vloc_to_grid(
        r_rad, v_rad_ha, grid, ions.positions[atom_index], ions.cell
    )
    return driver.data2field(np.asarray(v_1d, float).ravel(), grid=grid)


_VLOC_QE = frozenset({"qepy", "qe", "strf", "setlocal"})
_VLOC_UPF = frozenset({"upf"})
_VLOC_HOME = frozenset({"home", "home_cell", "crystal", "one_center", "qepy_home"})
_VLOC_AUTO = frozenset({"auto", "default", ""})


def qe_driver_is_isolated(driver) -> bool:
    """True when QE ran with ``assume_isolated`` (cluster / Martyna–Tuckerman)."""
    opts = getattr(driver, "qe_options", None) or {}
    system = opts.get("&system", {}) or {}
    raw = system.get("assume_isolated")
    if raw is None:
        return False
    s = str(raw).strip().strip("'\"").lower()
    return bool(s) and s not in ("false", "none", "no", "off", "0")


def resolve_vloc_source(vloc_source, driver=None) -> str:
    """Canonical peel: ``qepy``, ``upf``, or ``home``.

    ``auto`` is ``qepy`` for ``assume_isolated`` hosts (vacuum null) and
    ``home`` for periodic crystals.
    """
    src = str(vloc_source if vloc_source is not None else DEFAULT_VLOC_SOURCE)
    src = src.lower().strip()
    if src in _VLOC_AUTO:
        if driver is not None and not qe_driver_is_isolated(driver):
            return "home"
        return "qepy"
    if src in _VLOC_QE:
        return "qepy"
    if src in _VLOC_UPF:
        return "upf"
    if src in _VLOC_HOME:
        return "home"
    raise ValueError(
        f"Unknown vloc_source={vloc_source!r}; use 'qepy', 'upf', 'home', or 'auto'"
    )


def upf_coulomb_match_radius(r_rad, v_rad_ha, z_val, *, atol=0.05):
    """Smallest tabulated ``r`` where ``|r V_loc + Z_val| < atol`` (Coulomb tail)."""
    r = np.asarray(r_rad, float).ravel()
    v = np.asarray(v_rad_ha, float).ravel()
    n = int(min(r.size, v.size))
    if n == 0:
        return 0.0
    resid = np.abs(r[:n] * v[:n] + abs(float(z_val)))
    ok = np.where(resid < float(atol))[0]
    return float(r[int(ok[0])]) if ok.size else float(r[n - 1])


def home_cell_image_shift_ha(delta, dist, r_in, r_out, *, min_shell=8):
    """Mean of ``setlocal − V_UPF`` in the Coulomb shell ``[r_in, r_out]``.

    That shell is outside the local-PP cutoff, so the difference is the
    supercell-image Madelung of A, not the QE vs UPF on-site well. Fewer
    than ``min_shell`` points falls back to the nucleus voxel (includes
    QE−UPF self mismatch; used only in tiny cells).
    """
    delta = np.asarray(delta, float).ravel()
    dist = np.asarray(dist, float).ravel()
    n = int(min(delta.size, dist.size))
    delta, dist = delta[:n], dist[:n]
    r_in = float(r_in)
    r_out = float(r_out)
    mask = (dist >= r_in) & (dist <= r_out)
    n_shell = int(np.sum(mask))
    if n_shell >= int(min_shell) and r_out > r_in:
        return float(np.mean(delta[mask])), n_shell
    i_nuc = int(np.argmin(dist))
    return float(delta[i_nuc]), n_shell


def _home_cell_vloc_field(driver, atom_index, grid, upf_path=None):
    """Home-cell (one-center) ``V_loc(A)`` matching QE's on-site well.

    ``setlocal`` is the A sublattice. The image field ``φ`` is the shell
    mean of ``setlocal − V_UPF``; subtracting it leaves the home copy::

        V_loc^A_home = setlocal_A − φ
        V_ionic      = vltot − V_loc^A_home
                     = (vltot − setlocal_A) + φ

    ``φ`` is spatially constant over the 1s (images are a supercell away).
    """
    atom_index = int(atom_index)
    v_q_ry = vloc_atom_from_qepy(driver, atom_index)
    v_q = driver.data2field(ry_to_ha(v_q_ry).ravel(), grid=grid)
    v_u = vloc_atom_field_upf(driver, atom_index, grid, upf_path=upf_path)
    q_arr = np.asarray(v_q, float).ravel(order="F")
    u_arr = np.asarray(v_u, float).ravel(order="F")
    n = int(min(q_arr.size, u_arr.size))
    delta = q_arr[:n] - u_arr[:n]
    ions = driver.get_dftpy_ions()
    pos = np.asarray(ions.positions[atom_index], float).reshape(3)
    cell = np.asarray(ions.cell, float)
    dist = _grid_dist_f(grid, pos, cell)[:n]
    path = resolve_upf_path(
        driver,
        driver.get_ase_atoms().get_chemical_symbols()[atom_index],
        upf_path,
    )
    r_rad, v_rad, z_val = read_upf_local_potential(path)
    r_coul = upf_coulomb_match_radius(r_rad, v_rad, z_val)
    min_len = min(float(np.linalg.norm(cell[i])) for i in range(3))
    r_in = max(float(r_coul), 1.5)
    r_out = min(0.40 * min_len, r_in + 4.0)
    phi, n_shell = home_cell_image_shift_ha(delta, dist, r_in, r_out)
    v_home = v_q - float(phi)
    meta = {
        "vloc_source": "home",
        "home_peel_image_ha": float(phi),
        "home_peel_n_shell": int(n_shell),
        "home_peel_r_in_bohr": float(r_in),
        "home_peel_r_out_bohr": float(r_out),
        "home_peel_r_coulomb_bohr": float(r_coul),
        "home_peel_z_valence": float(z_val),
        "home_peel_fallback_nucleus": bool(n_shell < 8 or r_out <= r_in),
    }
    return v_home, meta


def ionic_vloc_residual(
    driver,
    edge_atom,
    grid,
    upf_path=None,
    *,
    vloc_source=DEFAULT_VLOC_SOURCE,
    atom_indices=None,
    peel_meta=None,
):
    """``(V_loc(A), V_ionic)`` with ``V_ionic = vltot − Σ V_loc`` (Hartree).

    Peels ``atom_indices`` when given (e.g. an O+2H water fragment), otherwise
    just ``edge_atom``. The first return is ``V_loc`` of ``edge_atom`` (or the
    first peeled atom) for backward-compatible diagnostics.

    ``vloc_source="qepy"`` uses :func:`vloc_atom_from_qepy` (periodic
    ``setlocal``; vacuum-null exact). ``"upf"`` is a real-space UPF interpolate
    (fails vacuum null). ``"home"`` peels only the home-cell copy of A so
    supercell images stay PP — production for periodic ``ae_host``. ``"auto"``
    selects ``home`` for crystals and ``qepy`` when ``assume_isolated`` is set.

    ``peel_meta``, if a dict, is updated with the resolved source and home-peel
    diagnostics.
    """
    peel = [int(i) for i in (atom_indices if atom_indices is not None else [edge_atom])]
    if not peel:
        peel = [int(edge_atom)]
    src = resolve_vloc_source(vloc_source, driver)
    isolated = qe_driver_is_isolated(driver)
    v_tot, _ = local_pp_total_ha(driver, grid=grid)
    v_loc_first = None
    v_loc_sum = None
    home_acc = None
    for ia in peel:
        if src == "qepy":
            v_a_ry = vloc_atom_from_qepy(driver, ia)
            v_loc_i = driver.data2field(ry_to_ha(v_a_ry).ravel(), grid=grid)
        elif src == "upf":
            path = upf_path if ia == int(edge_atom) else None
            v_loc_i = vloc_atom_field_upf(driver, ia, grid, upf_path=path)
        elif src == "home":
            path = upf_path if ia == int(edge_atom) else None
            v_loc_i, hmeta = _home_cell_vloc_field(driver, ia, grid, upf_path=path)
            if home_acc is None:
                home_acc = dict(hmeta)
                home_acc["home_peel_image_ha"] = float(hmeta["home_peel_image_ha"])
            else:
                home_acc["home_peel_image_ha"] = float(
                    home_acc["home_peel_image_ha"] + hmeta["home_peel_image_ha"]
                )
        else:
            raise ValueError(
                f"Unknown vloc_source={vloc_source!r}; use 'qepy', 'upf', 'home', or 'auto'"
            )
        arr = np.asarray(v_loc_i, float)
        if v_loc_first is None:
            v_loc_first = v_loc_i
        v_loc_sum = arr if v_loc_sum is None else v_loc_sum + arr
    diff = np.asarray(v_tot, float) - v_loc_sum
    if peel_meta is not None:
        peel_meta["vloc_source"] = src
        peel_meta["qe_isolated"] = bool(isolated)
        if home_acc is not None:
            peel_meta.update(home_acc)
    return v_loc_first, driver.data2field(diff.ravel(order="F"), grid=grid)


def neighbor_vloc_from_residual(driver, edge_atom, grid, upf_path=None, *, vloc_source=DEFAULT_VLOC_SOURCE):
    """Ionic residual only (alias of :func:`ionic_vloc_residual` → ``V_ionic``)."""
    _, v_ionic = ionic_vloc_residual(
        driver, edge_atom, grid, upf_path=upf_path, vloc_source=vloc_source
    )
    return v_ionic


# ---------------------------------------------------------------------------
# Gauge / fade helpers
# ---------------------------------------------------------------------------

def vhxc_radial_fade(grid, center_bohr, r_damp):
    """``1 − exp(−(r/r_damp)²)``: 0 at nucleus → 1 far away (legacy damp_vhxc)."""
    rc = float(r_damp)
    if rc <= 0.0:
        raise ValueError(f"r_damp must be > 0, got {r_damp}")
    pos = np.asarray(center_bohr, float).reshape(3)
    X = np.asarray(grid.r[0], float)
    Y = np.asarray(grid.r[1], float)
    Z = np.asarray(grid.r[2], float)
    dist = np.sqrt((X - pos[0]) ** 2 + (Y - pos[1]) ** 2 + (Z - pos[2]) ** 2)
    return 1.0 - np.exp(-((dist / rc) ** 2))


def shift_v_env_gauge_at(
    v_env,
    grid,
    center_bohr,
    mode="far_field",
    *,
    fermi_ha=None,
):
    """Subtract a constant so ``V_env`` has a chosen zero (Hartree).

    ``fermi`` / ``ef`` / ``efermi``
        Crystal gauge: ``V_env ← V_env − E_F(QE)``.  Pair with automatic
        ``virt-align-fermi`` in :func:`~casidapy.xas.prepare_pw_cvs_host`.
        Optional post-SCF fragment ``μ → 0`` is opt-in via
        ``XASEmbedOptions.align_atom_mu`` and is mirrored onto PW virtuals so
        CVS ``ω`` stays invariant.
    ``true_vacuum`` / ``min_density``
        Cluster / AE-fidelity gauge (Hirshfeld path): zero ``V_env`` in the
        lowest-density fraction of the cell; pair with ``virt-align-vacuum``.
    """
    mode = str(mode).lower().strip()
    if mode in ("none", "off", "false", ""):
        return v_env, 0.0

    nr = tuple(int(x) for x in grid.nr)
    v3 = np.asarray(v_env, float).reshape(nr, order="F")

    if mode == "mean":
        shift = float(np.mean(v3))
    elif mode in (
        "far_field",
        "farfield",
        "vacuum",
        "true_vacuum",
        "min_density",
    ):
        # Cluster / molecular cells: zero V_env at the grid point farthest from
        # the absorber (true vacuum corner of a padded cell).  'true_vacuum' /
        # 'min_density' are accepted as aliases so the production CLI flag
        # works for ae_host as well as hirshfeld/invert.
        pos = np.asarray(center_bohr, float).reshape(3)
        X = np.asarray(grid.r[0], float)
        Y = np.asarray(grid.r[1], float)
        Z = np.asarray(grid.r[2], float)
        dist = np.sqrt((X - pos[0]) ** 2 + (Y - pos[1]) ** 2 + (Z - pos[2]) ** 2)
        shift = float(v3.ravel()[int(np.argmax(dist.ravel()))])
    elif mode in ("fermi", "ef", "efermi", "e_fermi"):
        if fermi_ha is None:
            raise ValueError(
                "gauge_align='fermi' requires fermi_ha (Hartree); "
                "use fermi_energy_ha(driver) from the converged QE host"
            )
        shift = float(fermi_ha)
    else:
        raise ValueError(
            f"Unknown gauge_align={mode!r}; use 'none', 'mean', 'far_field', "
            f"'true_vacuum', or 'fermi'"
        )

    from dftpy.field import DirectField

    return DirectField(grid=grid, rank=1, griddata_3d=(v3 - shift)), float(shift)


# ---------------------------------------------------------------------------
# Legacy V_Hxc scaling (scale_vhxc mode only)
# ---------------------------------------------------------------------------

def auto_vhxc_scale(
    z_atomic,
    z_valence,
    *,
    alpha: float = _VHXC_SCALE_ALPHA,
    s_min: float = _VHXC_SCALE_MIN,
    s_max: float = _VHXC_SCALE_MAX,
):
    """``s = clip(1 − α·z_valence/Z, s_min, s_max)``."""
    z = float(z_atomic)
    zv = float(z_valence)
    if z <= 0.0:
        raise ValueError(f"z_atomic must be > 0, got {z_atomic}")
    if zv < 0.0:
        raise ValueError(f"z_valence must be >= 0, got {z_valence}")
    return float(np.clip(1.0 - float(alpha) * (zv / z), float(s_min), float(s_max)))


def resolve_vhxc_scale(
    vhxc_scale,
    *,
    z_atomic,
    z_valence,
    alpha=_VHXC_SCALE_ALPHA,
    s_min=_VHXC_SCALE_MIN,
    s_max=_VHXC_SCALE_MAX,
):
    if vhxc_scale is None or (
        isinstance(vhxc_scale, str) and str(vhxc_scale).strip().lower() in ("auto", "default")
    ):
        return auto_vhxc_scale(z_atomic, z_valence, alpha=alpha, s_min=s_min, s_max=s_max)
    s = float(vhxc_scale)
    if s < 0.0:
        raise ValueError(f"vhxc_scale must be >= 0, got {vhxc_scale}")
    return s


# ---------------------------------------------------------------------------
# Build V_env
# ---------------------------------------------------------------------------

def _normalize_embed_mode(embed_mode):
    mode = str(embed_mode).lower().strip()
    return _EMBED_MODE_ALIASES.get(mode, mode)


def _edge_atom_info(driver, edge_atom, upf_path=None):
    """Symbol, Z, UPF path, z_valence, nuclear position (Bohr)."""
    from pyscf.data import elements

    edge_atom = int(edge_atom)
    atoms = driver.get_ase_atoms()
    sym = atoms.get_chemical_symbols()[edge_atom]
    z_atomic = int(elements.charge(str(sym).capitalize()))
    upf_path = resolve_upf_path(driver, sym, upf_path)
    _, _, z_val = read_upf_local_potential(upf_path)
    center_bohr = driver.get_dftpy_ions().positions[edge_atom]
    return edge_atom, sym, z_atomic, upf_path, float(z_val), center_bohr


def ae_host_environment_hxc_ha(
    driver,
    grid,
    peel_indices,
    *,
    upf_path=None,
    rho_floor: float = 0.0,
):
    """``v_H[ρ_env] + v_xc[ρ_env]`` with peel free-atom density removed (Hartree).

    Double-counting fix for ``ae_host``: the fragment SCF already rebuilds
    ``V_Hxc[ρ_frag]``, so the frozen host field must not also carry the
    absorber's (peel's) own Hartree/XC.  XC is nonlinear, so this is done in
    density space::

        ρ_env = max(ρ_tot − Σ_{A∈peel} ρ̃_A^UPF, 0)
        V     = v_H[ρ_env] + v_xc[ρ_env]

    ``ρ̃_A`` is the tabulated UPF ``PP_RHOATOM`` (same object QE uses for its
    starting guess) — known, atom-centred, not a Hirshfeld weight.  Peel NLCC
    model-core charge is removed from the XC core density in the same pass.
    """
    from casidapy.embed.hirshfeld import (
        hartree_potential_ha,
        total_density_on_grid,
        upf_rhoatom_on_grid,
        xc_potential_ha,
    )
    from casidapy.utils.uspp import parse_upf

    peel = [int(i) for i in peel_indices]
    if not peel:
        raise ValueError("ae_host_environment_hxc_ha requires at least one peel atom")

    rho_tot = np.asarray(total_density_on_grid(driver, grid), float).ravel()
    rho_peel = np.zeros_like(rho_tot)
    peel_nlcc = False
    atoms = driver.get_ase_atoms()
    ions = driver.get_dftpy_ions()
    for ia in peel:
        sym = atoms.get_chemical_symbols()[ia]
        path = resolve_upf_path(driver, sym, upf_path=upf_path)
        rho_a = np.asarray(
            upf_rhoatom_on_grid(driver, ia, grid, upf_path=path), float
        ).ravel()
        rho_peel += rho_a
        pp = parse_upf(str(path))
        if pp.get("has_nlcc") and pp.get("rho_core_rad") is not None:
            peel_nlcc = True
            # Fold peel NLCC into ρ_peel so XC (zero_nlcc) never sees the
            # absorber's model core charge.
            rho_c = project_radial_to_grid(
                np.asarray(pp["r"], float),
                np.asarray(pp["rho_core_rad"], float),
                grid,
                ions.positions[ia],
                ions.cell,
                outside="zero",
            )
            rho_peel += np.asarray(rho_c, float).ravel()

    rho_env = rho_tot - rho_peel
    n_neg = int(np.sum(rho_env < 0.0))
    neg_min = float(rho_env.min()) if rho_env.size else 0.0
    rho_env = np.maximum(rho_env, float(rho_floor))

    v_h = hartree_potential_ha(driver, rho_env)
    # zero_nlcc: peel NLCC already folded into rho_peel when present; remaining
    # host NLCC of environment atoms is dropped here too for a vacuum peel, and
    # for multi-atom hosts the env NLCC is a small valence-region correction we
    # can restore later if spectra demand it.  Matching hirshfeld's env path.
    v_xc = xc_potential_ha(driver, rho_env, zero_nlcc=True)

    meta = {
        "rho_tot_mean": float(np.mean(rho_tot)),
        "rho_peel_mean": float(np.mean(rho_peel)),
        "rho_env_mean": float(np.mean(rho_env)),
        "rho_env_nelec": float(np.sum(rho_env) * float(grid.dV)),
        "rho_env_n_neg": n_neg,
        "rho_env_neg_min": neg_min,
        "peel_nlcc_subtracted": bool(peel_nlcc),
        "n_peel": len(peel),
    }
    return np.asarray(v_h, float) + np.asarray(v_xc, float), meta


def classify_vltot_kind(v_loc, v_eff, v_hxc):
    """Is ``v_loc`` bare ionic PP (``vltot``) or already screened (``vrs``)?

    Whole-grid residual, not a vacuum sample: in vacuum ``v_Hxc → 0`` so
    ``v_eff ≈ v_loc`` either way.  At nuclei Hartree is O(Ha) and the two
    hypotheses split:

        ionic:     v_eff ≈ v_loc + v_Hxc     (get_local_pp = setlocal / vltot)
        screened:  v_eff ≈ v_loc             (get_local_pp = vrs = vltot + v_of_r)
    """
    vl = np.asarray(v_loc, float).ravel()
    ve = np.asarray(v_eff, float).ravel()
    vh = np.asarray(v_hxc, float).ravel()
    n = min(vl.size, ve.size, vh.size)
    if n == 0:
        raise ValueError("classify_vltot_kind: empty potential arrays")
    vl, ve, vh = vl[:n], ve[:n], vh[:n]

    def _rms(x):
        return float(np.sqrt(np.mean(np.square(x))))

    rms_ionic = _rms(ve - vl - vh)
    rms_screened = _rms(ve - vl)
    kind = "ionic" if rms_ionic <= rms_screened else "screened"
    return kind, rms_ionic, rms_screened


def _grid_dist_f(grid, atom_pos_bohr, cell_bohr):
    """Minimum-image distances, Fortran-raveled to match QE potential buffers."""
    dist = _minimum_image_distances(grid, atom_pos_bohr, cell_bohr)
    return np.asarray(dist, float).ravel(order="F")


def _grid_point_bohr(grid, index_f):
    """Cartesian coordinate (Bohr) of Fortran-flat index ``index_f``."""
    idx = int(index_f)
    rx = np.asarray(grid.r[0], float).ravel(order="F")
    ry = np.asarray(grid.r[1], float).ravel(order="F")
    rz = np.asarray(grid.r[2], float).ravel(order="F")
    return np.array([rx[idx], ry[idx], rz[idx]], float)


def upf_vloc_lattice_sum_at_point(driver, point_bohr, *, upf_path=None):
    """Analytic MIC lattice sum of UPF ``V_loc`` at one Cartesian point (Hartree)."""
    from scipy.interpolate import interp1d
    from casidapy.utils.uspp import _minimum_image_displacement

    ions = driver.get_dftpy_ions()
    cell = np.asarray(ions.cell, float)
    point = np.asarray(point_bohr, float).reshape(3)
    atoms = driver.get_ase_atoms()
    total = 0.0
    cache = {}
    for ia, sym in enumerate(atoms.get_chemical_symbols()):
        path = resolve_upf_path(driver, sym, upf_path)
        if path not in cache:
            r_rad, v_rad, _zval = read_upf_local_potential(path)
            cache[path] = (
                r_rad,
                v_rad,
                interp1d(
                    r_rad,
                    v_rad,
                    kind="cubic",
                    fill_value="extrapolate",
                    bounds_error=False,
                ),
            )
        r_rad, v_rad, fn = cache[path]
        dx, dy, dz = _minimum_image_displacement(
            np.asarray(ions.positions[ia], float).reshape(3),
            point,
            cell,
        )
        dist = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        if dist > float(r_rad[-1]):
            total += float(v_rad[-1] * r_rad[-1]) / max(dist, 1e-12)
        else:
            total += float(fn(dist))
    return float(total)


def classify_qe_local_potential(
    driver,
    grid=None,
    *,
    edge_atom: int = 0,
    upf_path=None,
):
    """Classify ``get_local_pp`` and compare it to the UPF ``V_loc`` lattice sum.

    Returns a JSON-safe dict.  The decisive number is
    ``rms(v_eff − v_loc − v_Hxc)`` vs ``rms(v_eff − v_loc)``.  Far-field
    ``v_loc`` vs UPF is the Ewald/G=0 check: a *constant* offset is the
    periodic convention, a nucleus-vs-vacuum *difference* in that offset
    would mean the array is carrying screening.
    """
    grid = grid if grid is not None else ensure_charge_grid(driver)
    vl = _qe_pot_flat_ha(driver, driver.get_local_pp)
    ve = _qe_pot_flat_ha(driver, driver.get_effective_potential)
    vh = _qe_pot_flat_ha(driver, driver.get_density_functional_potential)
    kind, rms_ionic, rms_screened = classify_vltot_kind(vl, ve, vh)

    ions = driver.get_dftpy_ions()
    cell = np.asarray(ions.cell, float)
    positions = np.asarray(ions.positions, float)
    dmin = None
    for pos in positions:
        d = _grid_dist_f(grid, pos, cell)
        dmin = d if dmin is None else np.minimum(dmin, d)
    i_far = int(np.argmax(dmin))
    ia = int(edge_atom)
    d_abs = _grid_dist_f(grid, positions[ia], cell)
    i_nuc = int(np.argmin(d_abs))
    pt_far = _grid_point_bohr(grid, i_far)
    pt_nuc = _grid_point_bohr(grid, i_nuc)
    pos_nuc = np.asarray(positions[ia], float).reshape(3)
    v_upf_far = upf_vloc_lattice_sum_at_point(driver, pt_far, upf_path=upf_path)
    v_upf_nuc = upf_vloc_lattice_sum_at_point(driver, pos_nuc, upf_path=upf_path)
    v_upf_nuc_voxel = upf_vloc_lattice_sum_at_point(driver, pt_nuc, upf_path=upf_path)

    return {
        "kind": kind,
        "rms_v_eff_minus_v_loc_minus_v_hxc_ha": rms_ionic,
        "rms_v_eff_minus_v_loc_ha": rms_screened,
        "v_loc_mean_ha": float(np.mean(vl)),
        "v_eff_mean_ha": float(np.mean(ve)),
        "v_hxc_mean_ha": float(np.mean(vh)),
        "far_index": i_far,
        "far_min_dist_bohr": float(dmin[i_far]),
        "v_loc_far_ha": float(vl[i_far]),
        "v_eff_far_ha": float(ve[i_far]),
        "v_hxc_far_ha": float(vh[i_far]),
        "v_upf_sum_far_ha": v_upf_far,
        "v_loc_minus_upf_far_ha": float(vl[i_far] - v_upf_far),
        "nucleus_index": i_nuc,
        "nucleus_voxel_dist_bohr": float(d_abs[i_nuc]),
        "v_loc_nucleus_ha": float(vl[i_nuc]),
        "v_eff_nucleus_ha": float(ve[i_nuc]),
        "v_hxc_nucleus_ha": float(vh[i_nuc]),
        "v_upf_sum_nucleus_ha": v_upf_nuc,
        "v_upf_sum_nucleus_voxel_ha": v_upf_nuc_voxel,
        "v_loc_minus_upf_nucleus_ha": float(vl[i_nuc] - v_upf_nuc_voxel),
        "ewald_offset_consistent": bool(
            abs(float(vl[i_far] - v_upf_far) - float(vl[i_nuc] - v_upf_nuc_voxel))
            < 0.2 * max(1.0, abs(float(vl[i_nuc] - v_upf_nuc_voxel)))
        ),
    }


def ae_host_rho_tot_hxc_ha(
    driver,
    grid,
    rho_core_ae,
    *,
    edge_atom: int = 0,
    upf_path=None,
    vltot_kind=None,
):
    """Grid piece of frozen-core Hxc with the absorber's valence left in.

    ``ρ_tot`` is QE valence (pseudopotential).  ``ρ_core^AE`` is the isolated
    1s² density on the same mesh.  Hartree is linear so ``v_H[ρ_core]`` stays
    in the AO oneshot (analytic).  XC is nonlinear::

        v_xc[ρ_core] + v_xc[ρ_tot]  ≠  v_xc[ρ_core + ρ_tot]

    so the functional is evaluated **once** on the sum.  No PP_RHOATOM peel.

    If ``get_local_pp`` is ionic ``vltot``, this returns ``v_H[ρ_tot] +
    v_xc[ρ_core+ρ_tot]``.  If it is already ``vrs`` (includes host Hxc), it
    returns only the XC replacement ``v_xc[sum] − v_xc[ρ_tot]`` so Hartree of
    ``ρ_tot`` is not added twice.
    """
    from casidapy.embed.hirshfeld import (
        hartree_potential_ha,
        total_density_on_grid,
        xc_potential_ha,
    )

    cls = classify_qe_local_potential(
        driver, grid, edge_atom=int(edge_atom), upf_path=upf_path
    )
    kind = str(vltot_kind or cls["kind"]).lower().strip()
    rho_tot = np.asarray(total_density_on_grid(driver, grid), float).ravel()
    rho_core = np.asarray(rho_core_ae, float).ravel()
    if rho_core.size != rho_tot.size:
        raise ValueError(
            f"rho_core_ae size {rho_core.size} != rho_tot size {rho_tot.size}"
        )
    dV = float(grid.dV)
    rho_sum = rho_core + rho_tot
    n_neg = int(np.sum(rho_sum < 0.0))
    rho_sum = np.maximum(rho_sum, 0.0)
    v_xc_sum = np.asarray(xc_potential_ha(driver, rho_sum, zero_nlcc=True), float).ravel()
    i_nuc = int(cls["nucleus_index"])
    v_h_tot = None
    if kind == "ionic":
        v_h_tot = np.asarray(hartree_potential_ha(driver, rho_tot), float).ravel()
        v = v_h_tot + v_xc_sum
        assemble = "v_H[rho_tot] + v_xc[rho_core_AE + rho_tot]"
        v_h_nuc = float(v_h_tot[i_nuc])
    else:
        v_xc_tot = np.asarray(
            xc_potential_ha(driver, rho_tot, zero_nlcc=False), float
        ).ravel()
        v = v_xc_sum - v_xc_tot
        assemble = "v_xc[rho_core+rho_tot] - v_xc[rho_tot] (vltot already screened)"
        v_h_nuc = None
    meta = {
        "ae_host_hxc": "rho_tot",
        "assemble": assemble,
        "vltot": cls,
        "vltot_kind": kind,
        "rho_tot_nelec": float(np.sum(rho_tot) * dV),
        "rho_core_nelec": float(np.sum(rho_core) * dV),
        "rho_sum_nelec": float(np.sum(rho_sum) * dV),
        "rho_sum_n_neg": n_neg,
        "v_h_tot_nucleus_ha": v_h_nuc,
        "v_xc_sum_nucleus_ha": float(v_xc_sum[i_nuc]),
        "v_hxc_grid_nucleus_ha": float(v[i_nuc]),
        "xc_zero_nlcc": True,
    }
    if v_h_tot is not None:
        meta["v_h_tot_mean_ha"] = float(np.mean(v_h_tot))
    return v, meta


def _build_legacy_v_env(
    driver,
    grid,
    mode,
    *,
    v_loc_a,
    v_ionic,
    center_bohr,
    z_atomic,
    z_val,
    vhxc_scale,
    vhxc_scale_alpha,
    vhxc_scale_min,
    vhxc_scale_max,
    r_damp,
    edge_atom=0,
    active_atoms=None,
    upf_path=None,
    ae_host_hxc: str = "rho_tot",
    rho_core_ae=None,
):
    """Non-Hirshfeld modes. Returns ``(v_env, scale_used, scale_mode, extras)``."""
    from dftpy.field import DirectField

    extras = {"v_eff_mean_ha": None, "vhxc_fade_mean": None, "ae_host_rho_env": None}
    nr = tuple(int(x) for x in grid.nr)

    if mode == "zero":
        v_env = DirectField(grid=grid, rank=1, griddata_3d=np.zeros(nr, dtype=float))
        return v_env, 0.0, "zero", extras

    if mode == "loc_only":
        return v_ionic, 0.0, "loc_only", extras

    if mode in ("full", "ae_host"):
        # V_ionic = vltot − Σ_{peel} v_loc keeps neighbour local PP.
        vo = np.asarray(v_ionic, float).reshape(nr, order="F")
        hxc = normalize_ae_host_hxc(ae_host_hxc)
        if hxc == "rho_tot":
            # Frozen-core oneshot: V_ionic + v_H[ρ_tot] on the QE mesh.
            # ρ_core never collocates here — v_H[ρ_core] is radial and
            # v_xc[ρ_core+ρ_tot] is Becke (see xas.core_hxc).
            from casidapy.embed.hirshfeld import (
                hartree_potential_ha,
                total_density_on_grid,
            )
            from casidapy.utils.uspp import parse_upf

            cls = classify_qe_local_potential(
                driver, grid, edge_atom=int(edge_atom), upf_path=upf_path
            )
            kind = str(cls["kind"]).lower().strip()
            rho_tot = np.asarray(total_density_on_grid(driver, grid), float).ravel()
            has_nlcc = False
            if upf_path:
                try:
                    has_nlcc = bool(parse_upf(str(upf_path)).get("has_nlcc"))
                except Exception:
                    has_nlcc = False
            rho_meta = {
                "assemble": "v_H[rho_tot] on FFT; v_H[core]+v_xc[sum] on Becke",
                "vltot": cls,
                "vltot_kind": kind,
                "rho_tot_nelec": float(np.sum(rho_tot) * float(grid.dV)),
                "has_nlcc": has_nlcc,
                "xc_on_fft": False,
            }
            extras["ae_host_hxc"] = "rho_tot"
            extras["ae_host_rho_tot"] = rho_meta
            extras["vltot_kind"] = kind
            if kind == "ionic":
                v_h = np.asarray(hartree_potential_ha(driver, rho_tot), float).ravel()
                i_nuc = int(cls["nucleus_index"])
                rho_meta["v_h_tot_nucleus_ha"] = float(v_h[i_nuc])
                rho_meta["v_h_tot_mean_ha"] = float(np.mean(v_h))
                vh = v_h.reshape(nr, order="F")
                extras["v_eff_mean_ha"] = float(np.mean(vo + vh))
                return _as_field(grid, vo + vh), 1.0, mode, extras
            extras["v_eff_mean_ha"] = float(np.mean(vo))
            rho_meta["assemble"] = (
                "vltot already screened; Becke adds v_H[core]+v_xc[sum]"
            )
            return v_ionic, 1.0, mode, extras

    if mode == "scale_vhxc":
        v_hxc, _ = density_functional_potential_ha(driver, grid=grid)
        scale = resolve_vhxc_scale(
            vhxc_scale,
            z_atomic=z_atomic,
            z_valence=z_val,
            alpha=vhxc_scale_alpha,
            s_min=vhxc_scale_min,
            s_max=vhxc_scale_max,
        )
        auto = vhxc_scale is None or (
            isinstance(vhxc_scale, str) and str(vhxc_scale).strip().lower() in ("auto", "default")
        )
        vo = np.asarray(v_ionic, float).reshape(nr, order="F")
        vh = np.asarray(v_hxc, float).reshape(nr, order="F")
        return _as_field(grid, vo + scale * vh), scale, ("auto" if auto else "manual"), extras

    if mode == "damp_vhxc":
        v_hxc, _ = density_functional_potential_ha(driver, grid=grid)
        fade = vhxc_radial_fade(grid, center_bohr, r_damp)
        extras["vhxc_fade_mean"] = float(np.mean(fade))
        vo = np.asarray(v_ionic, float).reshape(nr, order="F")
        vh = np.asarray(v_hxc, float).reshape(nr, order="F")
        return _as_field(grid, vo + vh * fade), 1.0, "damp_vhxc", extras

    raise ValueError(
        f"Unknown embed_mode={mode!r}; use 'hirshfeld', 'pot_hirshfeld', 'invert', "
        f"'ae_host', 'damp_vhxc', 'loc_only', 'scale_vhxc', or 'zero'"
    )


def build_ae_embedding_potential(
    driver,
    edge_atom,
    *,
    upf_path=None,
    grid=None,
    verbose=False,
    embed_mode=DEFAULT_EMBED_MODE,
    gauge_align=DEFAULT_GAUGE_ALIGN,
    vhxc_scale="auto",
    vhxc_scale_alpha: float = _VHXC_SCALE_ALPHA,
    vhxc_scale_min: float = _VHXC_SCALE_MIN,
    vhxc_scale_max: float = _VHXC_SCALE_MAX,
    r_damp: float = DEFAULT_R_DAMP,
    vloc_source=DEFAULT_VLOC_SOURCE,
    hirshfeld_gauge_tol: float = 1e-6,
    hirshfeld_reference: str = "neutral",
    oxidation_state=None,
    oxidation_states=None,
    formal_ion_basis: str = "aug-cc-pvdz",
    formal_ion_xc: str = "pbe",
    formal_ion_valence_only: bool = True,
    kedf: str = "TFvW",
    kedf_rho_floor: float = 1.0e-8,
    density_partition: str = "hirshfeld",
    use_gpu: bool = False,
    comm=None,
    active_atoms=None,
    invert_workdir=None,
    pauli: str = "none",
    pauli_soft_tf: float = 1.0,
    ae_host_hxc: str = "rho_tot",
    rho_core_ae=None,
):
    """Build frozen ``V_env`` for AE core reconstruction (Hartree).

    Returns ``(v_env, grid, meta)``.

    ``embed_mode``
        ``hirshfeld`` (default), ``pot_hirshfeld``, ``invert``, ``ae_host``
        (AE absorber in the host's own potential: ``V_eff − v_loc(A)`` on the
        grid, plus nonlocal PP of the environment atoms added to the AO
        Hamiltonian), ``loc_only``, ``damp_vhxc``, ``scale_vhxc``, ``zero``.
        Legacy aliases ``full`` / ``veff`` map to ``ae_host``.

    ``pauli``
        For ``pot_hirshfeld``: ``none`` or ``soft_tf``
        (``λ(1−w_A)ρ^{2/3}``, nucleus-safe). Ignored otherwise.
    """
    mode = _normalize_embed_mode(embed_mode)

    # --- production path ---------------------------------------------------
    if mode == "hirshfeld":
        from casidapy.embed.hirshfeld import build_hirshfeld_embedding

        # Cluster / molecular cells: far_field is rewritten to the Hirshfeld
        # weight gauge (zero V_env in other atoms' basins). Periodic crystals
        # have no vacuum — pass gauge_align="mean" and do not rewrite it.
        g = str(gauge_align).lower().strip()
        if g in ("far_field", "farfield", "vacuum", "default", ""):
            g = "hirshfeld"
        return build_hirshfeld_embedding(
            driver,
            edge_atom,
            upf_path=upf_path,
            grid=grid,
            verbose=verbose,
            gauge_align=g,
            gauge_tol=hirshfeld_gauge_tol,
            vloc_source=vloc_source,
            use_gpu=use_gpu,
            comm=comm,
            active_atoms=active_atoms,
            hirshfeld_reference=hirshfeld_reference,
            oxidation_state=oxidation_state,
            oxidation_states=oxidation_states,
            formal_ion_basis=formal_ion_basis,
            formal_ion_xc=formal_ion_xc,
            formal_ion_valence_only=formal_ion_valence_only,
            kedf=kedf,
            kedf_rho_floor=kedf_rho_floor,
            density_partition=density_partition,
        )

    if mode == "pot_hirshfeld":
        from casidapy.embed.hirshfeld import build_pot_hirshfeld_embedding

        g = str(gauge_align).lower().strip()
        if g in ("far_field", "farfield", "vacuum", "default", ""):
            g = "hirshfeld"
        return build_pot_hirshfeld_embedding(
            driver,
            edge_atom,
            upf_path=upf_path,
            grid=grid,
            verbose=verbose,
            gauge_align=g,
            gauge_tol=hirshfeld_gauge_tol,
            vloc_source=vloc_source,
            use_gpu=use_gpu,
            comm=comm,
            active_atoms=active_atoms,
            hirshfeld_reference=hirshfeld_reference,
            oxidation_state=oxidation_state,
            oxidation_states=oxidation_states,
            formal_ion_basis=formal_ion_basis,
            formal_ion_xc=formal_ion_xc,
            formal_ion_valence_only=formal_ion_valence_only,
            density_partition=density_partition,
            pauli=pauli,
            pauli_soft_tf=pauli_soft_tf,
        )

    if mode == "invert":
        from casidapy.embed.invert import build_invert_embedding

        g = str(gauge_align).lower().strip()
        if g in ("far_field", "farfield", "vacuum", "default", ""):
            g = "hirshfeld"
        return build_invert_embedding(
            driver,
            edge_atom,
            upf_path=upf_path,
            grid=grid,
            verbose=verbose,
            gauge_align=g,
            gauge_tol=hirshfeld_gauge_tol,
            vloc_source=vloc_source,
            use_gpu=use_gpu,
            comm=comm,
            active_atoms=active_atoms,
            hirshfeld_reference=hirshfeld_reference,
            oxidation_state=oxidation_state,
            oxidation_states=oxidation_states,
            formal_ion_basis=formal_ion_basis,
            formal_ion_xc=formal_ion_xc,
            formal_ion_valence_only=formal_ion_valence_only,
            density_partition=density_partition,
            invert_workdir=invert_workdir,
        )

    # --- legacy / diagnostic modes -----------------------------------------
    grid = grid if grid is not None else ensure_charge_grid(driver)
    edge_atom, sym, z_atomic, upf_path, z_val, center_bohr = _edge_atom_info(
        driver, edge_atom, upf_path=upf_path
    )
    peel_meta = {}
    v_loc_a, v_ionic = ionic_vloc_residual(
        driver,
        edge_atom,
        grid,
        upf_path=upf_path,
        vloc_source=vloc_source,
        atom_indices=active_atoms,
        peel_meta=peel_meta,
    )
    v_env, scale_used, scale_mode, extras = _build_legacy_v_env(
        driver,
        grid,
        mode,
        v_loc_a=v_loc_a,
        v_ionic=v_ionic,
        center_bohr=center_bohr,
        z_atomic=z_atomic,
        z_val=z_val,
        vhxc_scale=vhxc_scale,
        vhxc_scale_alpha=vhxc_scale_alpha,
        vhxc_scale_min=vhxc_scale_min,
        vhxc_scale_max=vhxc_scale_max,
        r_damp=r_damp,
        edge_atom=edge_atom,
        active_atoms=active_atoms,
        upf_path=upf_path,
        ae_host_hxc=ae_host_hxc,
        rho_core_ae=rho_core_ae,
    )
    gmode = str(gauge_align).lower().strip()
    # Production CLI uses true_vacuum for cluster cells; the legacy path's
    # far_field gauge is the matching vacuum-corner zero.
    if gmode in ("true_vacuum", "min_density", "vacuum"):
        gmode = "far_field"
    fermi_ha = None
    if gmode in ("fermi", "ef", "efermi", "e_fermi"):
        from casidapy.adapter.qepy import fermi_energy_ha

        fermi_ha = fermi_energy_ha(driver)
        gmode = "fermi"
    v_env, gauge_shift = shift_v_env_gauge_at(
        v_env, grid, center_bohr, mode=gmode, fermi_ha=fermi_ha
    )

    meta = {
        "edge_atom": edge_atom,
        "symbol": sym,
        "z_atomic": z_atomic,
        "upf_path": str(upf_path),
        "z_valence": float(z_val),
        "vloc_source": str(peel_meta.get("vloc_source") or vloc_source).lower().strip(),
        "qe_isolated": bool(peel_meta.get("qe_isolated")),
        "embed_mode": mode,
        "gauge_align": gmode,
        "gauge_shift_ha": float(gauge_shift),
        "fermi_energy_ha": float(fermi_ha) if fermi_ha is not None else None,
        "vhxc_scale": float(scale_used) if scale_used is not None else None,
        "vhxc_scale_mode": scale_mode,
        "vhxc_scale_alpha": float(vhxc_scale_alpha) if scale_mode == "auto" else None,
        "r_damp": float(r_damp) if mode == "damp_vhxc" else None,
        "vhxc_fade_mean": extras["vhxc_fade_mean"],
        "v_eff_mean_ha": extras["v_eff_mean_ha"],
        "v_loc_a_mean_ha": float(np.mean(np.asarray(v_loc_a))),
        "v_ionic_mean_ha": float(np.mean(np.asarray(v_ionic))),
        "v_env_mean_ha": float(np.mean(np.asarray(v_env))),
    }
    for key, val in peel_meta.items():
        if key.startswith("home_peel"):
            meta[key] = val
    if extras.get("ae_host_rho_env") is not None:
        meta["ae_host_rho_env"] = extras["ae_host_rho_env"]
    if extras.get("ae_host_hxc") is not None:
        meta["ae_host_hxc"] = extras["ae_host_hxc"]
    if extras.get("ae_host_g0_align_ha") is not None:
        meta["ae_host_g0_align_ha"] = float(extras["ae_host_g0_align_ha"])
    if extras.get("ae_host_rho_tot") is not None:
        meta["ae_host_rho_tot"] = extras["ae_host_rho_tot"]
    if extras.get("vltot_kind") is not None:
        meta["vltot_kind"] = extras["vltot_kind"]
    if mode == "ae_host":
        # Environment atoms keep their nonlocal projectors; the AE peel does not.
        peel = {
            int(i)
            for i in (active_atoms if active_atoms is not None else [edge_atom])
        }
        atoms = driver.get_ase_atoms()
        ions = driver.get_dftpy_ions()
        sites = []
        for ia, sym_i in enumerate(atoms.get_chemical_symbols()):
            if int(ia) in peel:
                continue
            try:
                path_i = resolve_upf_path(driver, sym_i, upf_path=None)
            except FileNotFoundError:
                # Fall back to the absorber UPF only when the neighbour is the
                # same element (common for oxides / organics).
                if str(sym_i).capitalize() == str(sym).capitalize():
                    path_i = str(upf_path)
                else:
                    raise
            sites.append(
                {
                    "atom_index": int(ia),
                    "symbol": str(sym_i),
                    "pos_bohr": [float(x) for x in np.asarray(ions.positions[ia]).ravel()[:3]],
                    "upf_path": str(path_i),
                }
            )
        meta["ae_host_nl_sites"] = sites
        meta["ae_host_peel_atoms"] = sorted(peel)
    if verbose:
        extra = ""
        phi = meta.get("home_peel_image_ha")
        if phi is not None:
            extra = f" vloc={meta.get('vloc_source')} image={float(phi):+.4f} Ha"
        logger.info(
            "[embed] %s atom %d Z=%d z_val=%.1f mode=%s gauge=%s "
            "shift=%.4f r_damp=%s <V_env>=%.4f Ha%s",
            sym,
            edge_atom,
            z_atomic,
            z_val,
            mode,
            meta["gauge_align"],
            gauge_shift,
            meta["r_damp"],
            meta["v_env_mean_ha"],
            extra,
        )
    return v_env, grid, meta
