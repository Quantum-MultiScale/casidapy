"""Frozen environment potential ``V_env`` for all-electron (AE) core SCF.

Units
-----
QE arrays are Rydberg. Everything returned here is Hartree.

Production recipe (default ``embed_mode="hirshfeld"``)
------------------------------------------------------
1. Peel the edge atom's local PP from QE::

       V_loc(A) = setlocal with only atom A's structure factor
       V_ionic  = vltot − V_loc(A)     # exactly 0 for a lone atom in vacuum

2. Partition the PW density with Hirshfeld weights from UPF ``PP_RHOATOM``::

       w_A     = ρ̃_A / Σ_B ρ̃_B
       ρ_env   = (1 − w_A) · ρ_tot

3. Build the frozen embedding field::

       V_env = V_ionic + v_H[ρ_env] + v_xc[ρ_env]

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
DEFAULT_VLOC_SOURCE = "qepy"  # "qepy" = production; "upf" = diagnostic

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
    "veff": "full",
    "full": "full",
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


def _qe_field_ha(driver, getter, grid=None):
    """Fetch a QE real-space potential (Rydberg) and return (field_Ha, grid)."""
    grid = grid if grid is not None else ensure_charge_grid(driver)
    data = np.asarray(getter(gather=True), float)
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

    sym = str(atom_symbol).strip().capitalize()
    uspp_map = build_uspp_map_from_driver(driver)
    if sym in uspp_map:
        return str(uspp_map[sym])
    for key, path in uspp_map.items():
        if key.strip().capitalize() == sym:
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


def _qe_species_index(symbols, atom_index):
    """0-based species index as QE orders unique element labels."""
    seen = []
    for s in symbols:
        key = str(s).strip().capitalize()
        if key not in seen:
            seen.append(key)
    return seen.index(str(symbols[atom_index]).strip().capitalize())


def vloc_atom_from_qepy(driver, edge_atom):
    """Single-atom local PP via QE ``strf`` + ``setlocal`` (Rydberg).

    Same machinery as ``vltot``, so ``vltot − V_loc(A)`` is exactly zero for a
    lone atom (vacuum null test). Restores the original structure factor.
    """
    edge_atom = int(edge_atom)
    symbols = list(driver.get_ions_symbols())
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
    strf_a[:, _qe_species_index(symbols, edge_atom)] = phases
    try:
        vlmod.set_array_strf(strf_a)
        driver.qepy_pw.qepy_setlocal()
        v_a = np.asarray(driver.get_local_pp(gather=True), float)
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


def ionic_vloc_residual(driver, edge_atom, grid, upf_path=None, *, vloc_source=DEFAULT_VLOC_SOURCE):
    """``(V_loc(A), V_ionic)`` with ``V_ionic = vltot − V_loc(A)`` (Hartree).

    ``vloc_source="qepy"`` (default) uses :func:`vloc_atom_from_qepy`.
    ``vloc_source="upf"`` uses a real-space UPF interpolate (fails vacuum null).
    """
    src = str(vloc_source).lower().strip()
    v_tot, _ = local_pp_total_ha(driver, grid=grid)
    if src in ("qepy", "qe", "strf", "setlocal"):
        v_a_ry = vloc_atom_from_qepy(driver, edge_atom)
        v_loc_a = driver.data2field(ry_to_ha(v_a_ry).ravel(), grid=grid)
    elif src == "upf":
        v_loc_a = vloc_atom_field_upf(driver, edge_atom, grid, upf_path=upf_path)
    else:
        raise ValueError(f"Unknown vloc_source={vloc_source!r}; use 'qepy' or 'upf'")
    diff = np.asarray(v_tot, float) - np.asarray(v_loc_a, float)
    return v_loc_a, driver.data2field(diff.ravel(order="F"), grid=grid)


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


def shift_v_env_gauge_at(v_env, grid, center_bohr, mode="far_field"):
    """Subtract a constant so ``V_env`` has a chosen zero (Hartree)."""
    mode = str(mode).lower().strip()
    if mode in ("none", "off", "false", ""):
        return v_env, 0.0

    nr = tuple(int(x) for x in grid.nr)
    v3 = np.asarray(v_env, float).reshape(nr, order="F")

    if mode == "mean":
        shift = float(np.mean(v3))
    elif mode in ("far_field", "farfield", "vacuum"):
        pos = np.asarray(center_bohr, float).reshape(3)
        X = np.asarray(grid.r[0], float)
        Y = np.asarray(grid.r[1], float)
        Z = np.asarray(grid.r[2], float)
        dist = np.sqrt((X - pos[0]) ** 2 + (Y - pos[1]) ** 2 + (Z - pos[2]) ** 2)
        shift = float(v3.ravel()[int(np.argmax(dist.ravel()))])
    else:
        raise ValueError(f"Unknown gauge_align={mode!r}; use 'none', 'mean', or 'far_field'")

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
):
    """Non-Hirshfeld modes. Returns ``(v_env, scale_used, scale_mode, extras)``."""
    from dftpy.field import DirectField

    extras = {"v_eff_mean_ha": None, "vhxc_fade_mean": None}
    nr = tuple(int(x) for x in grid.nr)

    if mode == "zero":
        v_env = DirectField(grid=grid, rank=1, griddata_3d=np.zeros(nr, dtype=float))
        return v_env, 0.0, "zero", extras

    if mode == "loc_only":
        return v_ionic, 0.0, "loc_only", extras

    if mode == "full":
        v_eff, _ = effective_potential_ha(driver, grid=grid)
        v_env = _as_field(
            grid,
            np.asarray(v_eff, float).reshape(nr, order="F")
            - np.asarray(v_loc_a, float).reshape(nr, order="F"),
        )
        extras["v_eff_mean_ha"] = float(np.mean(np.asarray(v_eff)))
        return v_env, 1.0, "full", extras

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
        f"Unknown embed_mode={mode!r}; use 'hirshfeld', 'damp_vhxc', "
        f"'loc_only', 'scale_vhxc', 'full', or 'zero'"
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
    use_gpu: bool = False,
    comm=None,
):
    """Build frozen ``V_env`` for AE core reconstruction (Hartree).

    Returns ``(v_env, grid, meta)``.

    ``embed_mode``
        ``hirshfeld`` (default), ``loc_only``, ``damp_vhxc``, ``scale_vhxc``,
        ``full``, ``zero``.

    ``vloc_source``
        ``qepy`` (default) or ``upf`` (diagnostic).

    ``gauge_align``
        For Hirshfeld, ``far_field`` is promoted to the Hirshfeld weight gauge.

    ``use_gpu`` / ``comm``
        Forwarded to Hirshfeld partition (CuPy arithmetic / MPI atom loop).
    """
    mode = _normalize_embed_mode(embed_mode)

    # --- production path ---------------------------------------------------
    if mode == "hirshfeld":
        from casidapy.embed.hirshfeld import build_hirshfeld_embedding

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
        )

    # --- legacy / diagnostic modes -----------------------------------------
    grid = grid if grid is not None else ensure_charge_grid(driver)
    edge_atom, sym, z_atomic, upf_path, z_val, center_bohr = _edge_atom_info(
        driver, edge_atom, upf_path=upf_path
    )
    v_loc_a, v_ionic = ionic_vloc_residual(
        driver, edge_atom, grid, upf_path=upf_path, vloc_source=vloc_source
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
    )
    v_env, gauge_shift = shift_v_env_gauge_at(v_env, grid, center_bohr, mode=gauge_align)

    meta = {
        "edge_atom": edge_atom,
        "symbol": sym,
        "z_atomic": z_atomic,
        "upf_path": str(upf_path),
        "z_valence": float(z_val),
        "vloc_source": str(vloc_source).lower().strip(),
        "embed_mode": mode,
        "gauge_align": str(gauge_align).lower(),
        "gauge_shift_ha": float(gauge_shift),
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
    if verbose:
        logger.info(
            "[embed] %s atom %d Z=%d z_val=%.1f mode=%s gauge=%s "
            "shift=%.4f r_damp=%s <V_env>=%.4f Ha",
            sym,
            edge_atom,
            z_atomic,
            z_val,
            mode,
            meta["gauge_align"],
            gauge_shift,
            meta["r_damp"],
            meta["v_env_mean_ha"],
        )
    return v_env, grid, meta
