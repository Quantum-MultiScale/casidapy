"""Core–virtual dipole matrices and dipole-based virtual pruning (production XAS)."""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

from casidapy.xas.xas_utils import (
    grid_charge_centroid,
    grid_projection_to_ao_coeffs,
    project_grid_orbitals_to_fragment_ao,
)

logger = logging.getLogger(__name__)

from casidapy.xas.chemical_data import DEFAULTS

RADIAL_RMAX_BOHR = DEFAULTS["radial_rmax_bohr"]
RADIAL_N_RAD = DEFAULTS["radial_n_rad"]
RADIAL_N_ANG = DEFAULTS["radial_n_ang"]

# Dense local-grid path (intensities / μ fidelity only — not for Casida A/K).
# Radius is the 1s enclosed-charge ball, not a magic floor of 1.5 Bohr:
# C/O 1s is dead by ~0.6–0.8 Bohr, and extra radius only samples the
# nodeless PP virtual against a vanishing core.
LOCAL_N_RAD = max(int(RADIAL_N_RAD), 96)
LOCAL_N_ANG = max(int(RADIAL_N_ANG), 302)
LOCAL_RMS_K = 10.0
LOCAL_RMIN_BOHR = 0.25
LOCAL_RMAX_CAP_BOHR = float(RADIAL_RMAX_BOHR)
LOCAL_ENCLOSED_FRAC = 1.0 - 1.0e-6
LOCAL_ENCLOSED_PAD = 1.25
LOCAL_RADIAL_MAPPING = "log"
LOCAL_PSI_MODE = "spline3"

_LOCAL_METHOD_ALIASES = {
    "centroid_radial": "local_grid",
    "local": "local_grid",
    "dense_local": "local_grid",
    "pw_grid": "pw_grid",
    "grid": "pw_grid",
    "fft": "pw_grid",
    "plane_wave": "pw_grid",
    "pw": "pw_grid",
    "overlap": "pw_grid",
    "paw_gipaw": "paw_gipaw",
    "gipaw": "paw_gipaw",
    "paw": "paw_gipaw",
    "reconstruct": "paw_gipaw",
}


def absorber_origin_bohr(core, fallback=None):
    """Absorber nucleus (Bohr) from a :class:`CoreOrbitals` fragment mol."""
    mol = getattr(core, "fragment_mol", None)
    if mol is None:
        if fallback is None:
            raise ValueError("core has no fragment_mol and no fallback origin")
        return np.asarray(fallback, dtype=float).reshape(3)
    idx = 0
    meta = getattr(core, "meta", None) or {}
    eidx = meta.get("edge_atom_indices")
    if eidx:
        idx = int(np.asarray(eidx).ravel()[0])
    return np.asarray(mol.atom_coords()[idx], dtype=float).reshape(3)


def _normalize_dipole_method(method: str) -> str:
    m = str(method).lower().strip()
    return _LOCAL_METHOD_ALIASES.get(m, m)


def bind_ae_core_dipole(
    kernel,
    core,
    *,
    method: str = "gto",
    gauge: str = "length",
    origin=None,
    r_max: Optional[float] = None,
    n_rad: Optional[int] = None,
    n_ang: Optional[int] = None,
    rms_k: float = LOCAL_RMS_K,
    r_min: float = LOCAL_RMIN_BOHR,
    mapping: str = LOCAL_RADIAL_MAPPING,
    psi_mode: str = LOCAL_PSI_MODE,
    enclosed_frac: float = LOCAL_ENCLOSED_FRAC,
    enclosed_pad: float = LOCAL_ENCLOSED_PAD,
    gipaw_upf=None,
    core_wfc: str = "ae",
    gipaw_rcond: float = 1e-8,
):
    """Attach AE core dipoles for PW-host XAS intensities.

    ``method``
        ``'gto'`` (default) — fragment GTO projection + ``int1e_r``.
        ``'local_grid'`` / ``'centroid_radial'`` — dense spherical quadrature
        about the core charge centroid (μ fidelity only; leave A/K on host).
        Radius defaults to the 1s enclosed-charge ball.  Host virtuals are
        sampled by tricubic spline (not the coarse FFT voxel).
        ``'pw_grid'`` / ``'grid'`` / ``'fft'`` — map the AE 1s onto the QE
        charge mesh and take ``⟨1s|r|ψ̃⟩`` as a plane-wave overlap (``dV``
        sum).  No GTO projection, no local sphere.
        ``'paw_gipaw'`` / ``'gipaw'`` — Taillefumier reconstruction of ``ψ̃``
        inside ``r_c`` from a GIPAW UPF, then ``⟨1s|r|ψ_recon⟩`` on the PAW
        sphere.  Requires ``gipaw_upf`` (same species used in the host SCF).
        ``'radial'`` — legacy absorber-centered radial path.
    """
    method = _normalize_dipole_method(method)
    host_grid = getattr(kernel, "grid", None)
    gipaw_data = None
    if method == "paw_gipaw":
        from casidapy.xas.gipaw import parse_gipaw_upf

        if gipaw_upf is None:
            raise ValueError(
                "ae dipole method 'paw_gipaw' requires gipaw_upf= (GIPAW UPF "
                "with PP_GIPAW AE/PS partial waves). ONCV cannot reconstruct."
            )
        gipaw_data = parse_gipaw_upf(gipaw_upf)

    if method in ("local_grid", "pw_grid", "paw_gipaw"):
        R0 = np.asarray(
            origin
            if origin is not None
            else core_charge_centroid_bohr(core, host_grid),
            dtype=float,
        ).reshape(3)
        if method == "pw_grid":
            if r_max is None:
                r_max = float(RADIAL_RMAX_BOHR)
            n_rad = int(RADIAL_N_RAD if n_rad is None else n_rad)
            n_ang = int(RADIAL_N_ANG if n_ang is None else n_ang)
        elif method == "paw_gipaw":
            r_paw = float(gipaw_data.r_c)
            if r_max is None:
                r_max = r_paw
                if _is_core_orbitals(core) and str(core_wfc).lower() not in (
                    "gipaw",
                    "upf",
                    "atomic",
                ):
                    r_1s = core_dipole_radius_bohr(
                        core,
                        host_grid,
                        R0,
                        r_min=r_min,
                        rms_k=rms_k,
                        enclosed_frac=enclosed_frac,
                        enclosed_pad=enclosed_pad,
                    )
                    r_max = max(r_paw, float(r_1s))
            else:
                r_max = max(float(r_max), r_paw)
            n_rad = int(LOCAL_N_RAD if n_rad is None else n_rad)
            n_ang = int(LOCAL_N_ANG if n_ang is None else n_ang)
        else:
            if r_max is None:
                r_max = core_dipole_radius_bohr(
                    core,
                    host_grid,
                    R0,
                    r_min=r_min,
                    rms_k=rms_k,
                    enclosed_frac=enclosed_frac,
                    enclosed_pad=enclosed_pad,
                )
            n_rad = int(LOCAL_N_RAD if n_rad is None else n_rad)
            n_ang = int(LOCAL_N_ANG if n_ang is None else n_ang)
    else:
        R0 = np.asarray(
            origin if origin is not None else absorber_origin_bohr(core),
            dtype=float,
        ).reshape(3)
        if r_max is None:
            r_max = float(RADIAL_RMAX_BOHR)
        n_rad = int(RADIAL_N_RAD if n_rad is None else n_rad)
        n_ang = int(RADIAL_N_ANG if n_ang is None else n_ang)

    kernel._xas_ae_core = {
        "core": core,
        "method": method,
        "gauge": str(gauge).lower().strip(),
        "origin": tuple(float(x) for x in R0),
        "r_max": float(r_max),
        "n_rad": int(n_rad),
        "n_ang": int(n_ang),
        "rms_k": float(rms_k),
        "r_min": float(r_min),
        "mapping": str(mapping),
        "psi_mode": str(psi_mode),
        "enclosed_frac": float(enclosed_frac),
        "enclosed_pad": float(enclosed_pad),
        "gipaw_upf": None if gipaw_upf is None else str(gipaw_upf),
        "gipaw": gipaw_data,
        "core_wfc": str(core_wfc).lower().strip(),
        "gipaw_rcond": float(gipaw_rcond),
    }
    kernel.dipole_origin = tuple(float(x) for x in R0)
    return kernel


# --- spherical quadrature + host-grid sampling (shared with diagnostics) ---


def _lebedev_unit_sphere(n_ang: int):
    """Unit-sphere Lebedev nodes; weights sum to 1 (``dΩ = 4π w``)."""
    from pyscf.dft.gen_grid import MakeAngularGrid

    tab = np.asarray(MakeAngularGrid(int(n_ang)), dtype=float)
    return tab[:, :3], tab[:, 3]


def absorber_radial_grid(
    center_bohr,
    *,
    r_max: float = RADIAL_RMAX_BOHR,
    n_rad: int = RADIAL_N_RAD,
    n_ang: int = RADIAL_N_ANG,
    mapping: str = "legendre",
):
    """Absorber-centered spherical quadrature ``(coords, weights)`` in Bohr.

    ``mapping``
        ``'legendre'`` — uniform Gauss–Legendre on ``[0, r_max]``.
        ``'log'`` — ``r = r_max u²`` (Ahlrichs-like), denser at the nucleus
        where a 1s lives.  ``u ∈ [0,1]`` is still Gauss–Legendre.
    """
    center = np.asarray(center_bohr, dtype=float).reshape(3)
    omega, wang = _lebedev_unit_sphere(n_ang)
    x, w = np.polynomial.legendre.leggauss(int(n_rad))
    u = 0.5 * (x + 1.0)
    wu = 0.5 * w
    rmax = float(r_max)
    mode = str(mapping or "legendre").lower().strip()
    if mode in ("log", "ahlrichs", "u2", "treutler"):
        r = rmax * u * u
        wr = rmax * 2.0 * u * wu
    elif mode in ("legendre", "linear", "uniform"):
        r = rmax * u
        wr = rmax * wu
    else:
        raise ValueError(
            f"unknown radial mapping={mapping!r}; use 'legendre' or 'log'"
        )
    coords = (r[:, None, None] * omega[None, :, :]).reshape(-1, 3) + center
    weights = (wr * r**2)[:, None] * (4.0 * np.pi * wang)[None, :]
    return coords, weights.ravel()


def build_core_dipole_grid(
    origin,
    r_max: float,
    n_rad: int = LOCAL_N_RAD,
    n_ang: int = LOCAL_N_ANG,
):
    """Dense local spherical grid for length-gauge μ (independent of PW mesh)."""
    return absorber_radial_grid(
        origin, r_max=float(r_max), n_rad=int(n_rad), n_ang=int(n_ang)
    )


def _field_on_nr(grid, field) -> np.ndarray:
    """Reshape a DFTpy field to ``grid.nr`` (C-order, matching ``grid.r``)."""
    nr = tuple(int(n) for n in grid.nr)
    arr = np.asarray(field, dtype=float)
    if arr.shape == nr:
        return np.ascontiguousarray(arr)
    flat = np.ascontiguousarray(arr).ravel()
    if flat.size != int(np.prod(nr)):
        raise ValueError(f"field size {flat.size} != grid {nr}")
    return flat.reshape(nr)


def sample_host_orbitals_at_coords(
    grid,
    orbitals,
    coords_bohr,
    *,
    psi_mode: str = LOCAL_PSI_MODE,
):
    """Evaluate stored PW orbitals at Bohr coords (default tricubic spline).

    ``psi_mode``
        ``spline3`` / ``tricubic`` (default) — band-limited reconstruction,
        same as the production PW→AO projector.
        ``trilinear`` — legacy C⁰ interpolant (too crude at a 1s nucleus).
        ``fft`` / ``gsum`` — exact Fourier interpolant (expensive).
    """
    from casidapy.xas.pw_eval import (
        orbitals_at_coords_pw_sum,
        orbitals_at_coords_spline,
        orbitals_at_coords_trilinear,
    )

    mode = str(psi_mode or LOCAL_PSI_MODE).lower().strip()
    virts = list(orbitals)
    if not virts:
        return np.zeros((np.asarray(coords_bohr).reshape(-1, 3).shape[0], 0)), {
            "source": mode
        }
    if mode in ("trilinear", "linear", "spline1"):
        psi, meta = orbitals_at_coords_trilinear(virts, grid, coords_bohr)
        return psi, meta
    if mode in ("fft", "fourier", "gsum", "pw_sum"):
        psi, metas = orbitals_at_coords_pw_sum(virts, grid, coords_bohr)
        meta = dict(metas[0] if metas else {})
        meta["source"] = "fft"
        meta["n_orb"] = int(len(virts))
        return psi, meta
    if mode in ("spline", "spline3", "tricubic", "bspline", ""):
        psi, meta = orbitals_at_coords_spline(virts, grid, coords_bohr, order=3)
        return psi, meta
    raise ValueError(
        f"unknown dipole psi_mode={psi_mode!r}; use 'spline3', 'trilinear', or 'fft'"
    )


def trilinear_sample(grid, field, coords_bohr) -> np.ndarray:
    """PBC trilinear sample of ``field`` at Bohr coordinates."""
    nr = np.array([int(n) for n in grid.nr], dtype=float)
    lattice = np.asarray(grid.lattice, dtype=float)
    vals = _field_on_nr(grid, field)
    coords = np.asarray(coords_bohr, dtype=float).reshape(-1, 3)
    frac = coords @ np.linalg.inv(lattice)
    idx = frac * nr
    nx, ny, nz = (int(n) for n in grid.nr)
    x, y, z = idx[:, 0], idx[:, 1], idx[:, 2]
    x0, y0, z0 = np.floor(x), np.floor(y), np.floor(z)
    xd, yd, zd = x - x0, y - y0, z - z0
    i0 = np.mod(x0.astype(np.int64), nx)
    j0 = np.mod(y0.astype(np.int64), ny)
    k0 = np.mod(z0.astype(np.int64), nz)
    i1 = np.mod(i0 + 1, nx)
    j1 = np.mod(j0 + 1, ny)
    k1 = np.mod(k0 + 1, nz)
    c000 = vals[i0, j0, k0]
    c100 = vals[i1, j0, k0]
    c010 = vals[i0, j1, k0]
    c110 = vals[i1, j1, k0]
    c001 = vals[i0, j0, k1]
    c101 = vals[i1, j0, k1]
    c011 = vals[i0, j1, k1]
    c111 = vals[i1, j1, k1]
    c00 = c000 * (1.0 - xd) + c100 * xd
    c01 = c001 * (1.0 - xd) + c101 * xd
    c10 = c010 * (1.0 - xd) + c110 * xd
    c11 = c011 * (1.0 - xd) + c111 * xd
    c0 = c00 * (1.0 - yd) + c10 * yd
    c1 = c01 * (1.0 - yd) + c11 * yd
    return c0 * (1.0 - zd) + c1 * zd


def fft_gradient_components(grid, field):
    """``∇field`` on the FFT mesh, three real arrays shaped ``grid.nr``."""
    nr = tuple(int(n) for n in grid.nr)
    vals = _field_on_nr(grid, field)
    vg = np.fft.fftn(vals)
    lattice = np.asarray(grid.lattice, dtype=float)
    inv_a = np.linalg.inv(lattice)
    nn = np.array(
        np.meshgrid(
            np.fft.fftfreq(nr[0]) * nr[0],
            np.fft.fftfreq(nr[1]) * nr[1],
            np.fft.fftfreq(nr[2]) * nr[2],
            indexing="ij",
        )
    )
    # r = s @ A  ⇒  G = 2π A^{-T} n
    gvec = 2.0 * np.pi * np.einsum("ba,bxyz->axyz", inv_a, nn)
    return [np.fft.ifftn(1j * gvec[a] * vg).real for a in range(3)]


def _is_core_orbitals(obj) -> bool:
    return hasattr(obj, "fragment_mol") and hasattr(obj, "mo_coeff")


def _eval_core_ao_on_coords(core, coords) -> np.ndarray:
    """AE core MO values on Bohr coordinates, shape ``(npts, n_core)``."""
    from pyscf.dft import numint

    mol = core.fragment_mol
    C = np.asarray(core.mo_coeff, dtype=float)
    if C.ndim == 1:
        C = C.reshape(-1, 1)
    return numint.eval_ao(mol, coords) @ C


def core_charge_centroid_bohr(
    core_orbitals,
    grid=None,
    *,
    origin_guess=None,
    r_max: Optional[float] = None,
    n_rad: int = LOCAL_N_RAD,
    n_ang: int = LOCAL_N_ANG,
) -> np.ndarray:
    """Core charge centroid ``∫ r |ψ_c|² / ∫ |ψ_c|²`` (Bohr).

    For :class:`CoreOrbitals`, moments are accumulated on a local spherical
    probe (not the coarse PW mesh). For grid fields, uses ``grid``.
    """
    if _is_core_orbitals(core_orbitals):
        guess = np.asarray(
            origin_guess
            if origin_guess is not None
            else absorber_origin_bohr(core_orbitals),
            dtype=float,
        ).reshape(3)
        rmax = float(r_max if r_max is not None else max(float(RADIAL_RMAX_BOHR), 3.0))
        coords, wgt = absorber_radial_grid(
            guess, r_max=rmax, n_rad=int(n_rad), n_ang=int(n_ang)
        )
        psi = _eval_core_ao_on_coords(core_orbitals, coords)
        dens = np.sum(psi * psi, axis=1)
        tot = float(np.dot(dens, wgt))
        if not np.isfinite(tot) or tot <= 0.0:
            raise ValueError("core orbitals have non-positive norm on local probe")
        return (coords.T @ (dens * wgt)) / tot

    if grid is None:
        raise ValueError("grid is required for grid-field core centroid")
    return np.asarray(grid_charge_centroid(core_orbitals, grid), dtype=float).reshape(3)


def core_rms_radius_bohr(
    core_orbitals,
    grid,
    origin,
    *,
    r_max: Optional[float] = None,
    n_rad: int = LOCAL_N_RAD,
    n_ang: int = LOCAL_N_ANG,
) -> float:
    """RMS radius ``√⟨|r−R₀|²⟩_c`` of the core density about ``origin`` (Bohr)."""
    R0 = np.asarray(origin, dtype=float).reshape(3)
    if _is_core_orbitals(core_orbitals):
        rmax = float(r_max if r_max is not None else max(float(RADIAL_RMAX_BOHR), 3.0))
        coords, wgt = absorber_radial_grid(
            R0, r_max=rmax, n_rad=int(n_rad), n_ang=int(n_ang)
        )
        psi = _eval_core_ao_on_coords(core_orbitals, coords)
        dens = np.sum(psi * psi, axis=1)
        tot = float(np.dot(dens, wgt))
        if not np.isfinite(tot) or tot <= 0.0:
            raise ValueError("core orbitals have non-positive norm on local probe")
        rr = np.sum((coords - R0) ** 2, axis=1)
        return float(np.sqrt(np.dot(dens * rr, wgt) / tot))

    if grid is None:
        raise ValueError("grid is required for grid-field core RMS")
    w = None
    for p in core_orbitals:
        arr = np.asarray(p, dtype=float).ravel()
        w = arr * arr if w is None else w + arr * arr
    if w is None:
        raise ValueError("no orbitals given for RMS radius")
    dV = float(grid.dV)
    tot = float(w.sum()) * dV
    if not np.isfinite(tot) or tot <= 0.0:
        raise ValueError("orbitals have non-positive norm on the grid")
    rr = sum((np.asarray(grid.r[a]).ravel() - R0[a]) ** 2 for a in range(3))
    return float(np.sqrt(float((rr * w).sum()) * dV / tot))


def core_enclosed_radius_bohr(
    core_orbitals,
    origin,
    *,
    frac: float = LOCAL_ENCLOSED_FRAC,
    r_probe: float = 4.0,
    n_rad: int = LOCAL_N_RAD,
    n_ang: int = LOCAL_N_ANG,
    mapping: str = LOCAL_RADIAL_MAPPING,
) -> float:
    """Smallest ``R`` with ``∫_{|r-R₀|<R} |ψ_c|² ≥ frac · ‖ψ_c‖²`` (Bohr).

    This is the ball that actually contributes to ``⟨1s|r|v⟩``.  Points
    outside it are wasted quadrature against a vanishing core.
    """
    if not _is_core_orbitals(core_orbitals):
        raise ValueError("core_enclosed_radius_bohr needs CoreOrbitals")
    R0 = np.asarray(origin, dtype=float).reshape(3)
    coords, wgt = absorber_radial_grid(
        R0,
        r_max=float(r_probe),
        n_rad=int(n_rad),
        n_ang=int(n_ang),
        mapping=mapping,
    )
    psi = _eval_core_ao_on_coords(core_orbitals, coords)
    dens = np.sum(psi * psi, axis=1)
    rr = np.linalg.norm(coords - R0, axis=1)
    order = np.argsort(rr)
    cum = np.cumsum(dens[order] * wgt[order])
    tot = float(cum[-1]) if cum.size else 0.0
    if not np.isfinite(tot) or tot <= 0.0:
        raise ValueError("core orbitals have non-positive norm on enclosed probe")
    target = float(frac) * tot
    k = int(np.searchsorted(cum, target))
    k = min(max(k, 0), int(order.size) - 1)
    return float(rr[order[k]])


def core_dipole_radius_bohr(
    core_orbitals,
    grid,
    origin,
    *,
    r_min: float = LOCAL_RMIN_BOHR,
    r_cap: float = LOCAL_RMAX_CAP_BOHR,
    rms_k: float = LOCAL_RMS_K,
    enclosed_frac: float = LOCAL_ENCLOSED_FRAC,
    enclosed_pad: float = LOCAL_ENCLOSED_PAD,
) -> float:
    """Local-grid ``r_max``: 1s enclosed-charge radius, padded, RMS-capped."""
    R0 = np.asarray(origin, dtype=float).reshape(3)
    r_enc = core_enclosed_radius_bohr(
        core_orbitals, R0, frac=float(enclosed_frac)
    )
    rms = core_rms_radius_bohr(core_orbitals, grid, R0)
    r_max = float(enclosed_pad) * float(r_enc)
    r_max = max(r_max, float(r_min), 0.5 * float(rms_k) * float(rms))
    return float(min(r_max, float(r_cap)))


def core_virtual_dipole_matrix_radial(
    core,
    virtual_orbitals,
    grid,
    *,
    origin=None,
    gauge: str = "length",
    r_max: float = RADIAL_RMAX_BOHR,
    n_rad: int = RADIAL_N_RAD,
    n_ang: int = RADIAL_N_ANG,
    gs_pass: int = 2,
    mapping: str = "legendre",
    psi_mode: str = "trilinear",
):
    """On-site ``⟨core|r−R₀|virt⟩`` (or ``⟨core|∇|virt⟩``) on a fine radial grid.

    The AE core is ``numint.eval_ao`` of the fragment GTO. Host virtuals are
    sampled off the FFT mesh (``psi_mode``: ``trilinear``, ``spline3``, ``fft``).
    Optional weight-metric projection of the core against those virtuals
    kills overlap residual.
    """
    C = np.asarray(core.mo_coeff, dtype=float)
    if C.ndim == 1:
        C = C.reshape(-1, 1)
    n_core = int(C.shape[1])
    virts = list(virtual_orbitals)
    n_virt = len(virts)
    if n_virt == 0:
        return np.zeros((3, n_core, 0), dtype=float)

    R0 = np.asarray(
        origin if origin is not None else absorber_origin_bohr(core),
        dtype=float,
    ).reshape(3)
    coords, wgt = absorber_radial_grid(
        R0, r_max=r_max, n_rad=n_rad, n_ang=n_ang, mapping=mapping
    )
    psi_c = _eval_core_ao_on_coords(core, coords)
    psi_v, _ = sample_host_orbitals_at_coords(
        grid, virts, coords, psi_mode=psi_mode
    )

    # Discrete weight metric; sampled PW virtuals are not orthonormal under it,
    # so use P = V (Vᵀ W V)⁺ (Vᵀ W) rather than naive Gram–Schmidt.
    vw = psi_v * wgt[:, None]
    G = psi_v.T @ vw
    G = 0.5 * (G + G.T)
    for _ in range(max(int(gs_pass), 0)):
        coeff = np.linalg.lstsq(G, vw.T @ psi_c, rcond=1e-10)[0]
        psi_c = psi_c - psi_v @ coeff
    norms = np.sqrt(np.einsum("pc,p,pc->c", psi_c, wgt, psi_c))
    psi_c = psi_c / np.maximum(norms, 1e-30)

    gmode = str(gauge).lower().strip()
    out = np.empty((3, n_core, n_virt), dtype=float)
    if gmode in ("velocity", "grad", "momentum"):
        for a, virt in enumerate(virts):
            gxyz = fft_gradient_components(grid, virt)
            g_s, _ = sample_host_orbitals_at_coords(
                grid, gxyz, coords, psi_mode=psi_mode
            )
            for alpha in range(3):
                out[alpha, :, a] = psi_c.T @ (g_s[:, alpha] * wgt)
    else:
        rel = coords - R0
        for alpha in range(3):
            out[alpha] = psi_c.T @ ((rel[:, alpha] * wgt)[:, None] * psi_v)
    return out


def dipole_vector_magnitude(mu) -> np.ndarray:
    """``|μ|`` from a ``(3, n_core, n_virt)`` dipole matrix → ``(n_core, n_virt)``."""
    d = np.asarray(mu, dtype=float)
    if d.ndim != 3 or d.shape[0] != 3:
        raise ValueError(f"dipole matrix must be (3, n_core, n_virt), got {d.shape}")
    return np.sqrt(np.sum(d * d, axis=0))


def pseudo_dipole_scale_report(
    mu,
    omega_ev,
    *,
    window_ev: float = 30.0,
    bright_frac: float = 0.15,
    min_bright: int = 4,
    drift_tol: float = 0.30,
    cv_tol: float = 0.35,
) -> dict:
    """Norm-conservation test: is ``|⟨1s|r|ψ̃_a⟩|`` flat across the near-edge?

    Among dipole-bright virtuals in ``[ω₀, ω₀ + window]``, the implied scale is
    ``s_a = |μ_a| / median(|μ|)``.  A linear fit of ``s`` vs energy gives the
    relative drift over the window.  If that drift is small, XANES
    normalization can absorb a constant intensity error and PAW reconstruction
    is not required yet.

    This is the integral against the *pseudo*-virtual, evaluated however ``mu``
    was built (local-grid 1s × interpolated ``ψ̃``).  It does not test
    quadrature of the 1s on the FFT mesh.
    """
    mag = np.asarray(dipole_vector_magnitude(mu)[0], dtype=float).ravel()
    w = np.asarray(omega_ev, dtype=float).ravel()
    n = int(min(mag.size, w.size))
    mag, w = mag[:n], w[:n]
    out = {
        "n": n,
        "window_ev": float(window_ev),
        "omega0_ev": float(w[0]) if n else float("nan"),
        "n_in_window": 0,
        "n_bright": 0,
        "median_mu": float("nan"),
        "cv_rel": float("nan"),
        "slope_per_ev": float("nan"),
        "rel_drift_over_window": float("nan"),
        "flat": False,
        "drift_tol": float(drift_tol),
        "cv_tol": float(cv_tol),
    }
    if n == 0:
        return out
    e0 = float(w[0])
    in_win = (w >= e0 - 1e-12) & (w <= e0 + float(window_ev))
    mag_w = mag[in_win]
    w_w = w[in_win]
    out["n_in_window"] = int(mag_w.size)
    if mag_w.size == 0:
        return out
    thresh = float(bright_frac) * float(np.max(mag_w))
    bright = mag_w >= thresh
    n_b = int(np.count_nonzero(bright))
    if n_b < int(min_bright):
        k = min(int(min_bright), int(mag_w.size))
        bright = np.zeros(mag_w.size, dtype=bool)
        bright[np.argsort(mag_w)[-k:]] = True
        n_b = k
    s = mag_w[bright]
    e_b = w_w[bright]
    s_med = float(np.median(s))
    s_rel = s / s_med if s_med > 0.0 else s
    x = e_b - e0
    coef, *_ = np.linalg.lstsq(
        np.column_stack([np.ones_like(x), x]), s_rel, rcond=None
    )
    slope = float(coef[1])
    mean_s = float(np.mean(s_rel))
    cv = float(np.std(s_rel) / mean_s) if mean_s > 0.0 else float("inf")
    drift = abs(slope) * float(window_ev)
    out.update(
        {
            "n_bright": n_b,
            "median_mu": s_med,
            "cv_rel": cv,
            "slope_per_ev": slope,
            "rel_drift_over_window": drift,
            "flat": bool(drift < float(drift_tol) and cv < float(cv_tol)),
            "bright_omega_ev": [float(x) for x in e_b],
            "bright_mu": [float(x) for x in s],
        }
    )
    return out


def core_virtual_dipole_matrix_local(
    core,
    virtual_orbitals,
    host_grid,
    *,
    origin=None,
    r_max: Optional[float] = None,
    n_rad: int = LOCAL_N_RAD,
    n_ang: int = LOCAL_N_ANG,
    gauge: str = "length",
    gs_pass: int = 2,
    rms_k: float = LOCAL_RMS_K,
    r_min: float = LOCAL_RMIN_BOHR,
    mapping: str = LOCAL_RADIAL_MAPPING,
    psi_mode: str = LOCAL_PSI_MODE,
    enclosed_frac: float = LOCAL_ENCLOSED_FRAC,
    enclosed_pad: float = LOCAL_ENCLOSED_PAD,
) -> np.ndarray:
    """Dense local-grid ``⟨c|r−R₀|v⟩`` for intensities (shape ``(3, n_core, n_virt)``).

    Gauge origin defaults to the core charge centroid. ``r_max`` defaults to
    the 1s enclosed-charge ball (padded), independent of the PW charge mesh
    used for Casida A/K. Host virtuals are tricubic-sampled. For μ fidelity
    only — do not use to build A.
    """
    if str(gauge).lower().strip() not in ("length", "r", "position"):
        pass
    R0 = np.asarray(
        origin if origin is not None else core_charge_centroid_bohr(core, host_grid),
        dtype=float,
    ).reshape(3)
    if r_max is None:
        r_max = core_dipole_radius_bohr(
            core,
            host_grid,
            R0,
            r_min=r_min,
            rms_k=rms_k,
            enclosed_frac=enclosed_frac,
            enclosed_pad=enclosed_pad,
        )
    return core_virtual_dipole_matrix_radial(
        core,
        virtual_orbitals,
        host_grid,
        origin=R0,
        gauge=gauge,
        r_max=float(r_max),
        n_rad=int(n_rad),
        n_ang=int(n_ang),
        gs_pass=gs_pass,
        mapping=mapping,
        psi_mode=psi_mode,
    )

def core_virtual_dipole_matrix_grid(
    core_orbitals,
    virtual_orbitals,
    grid,
    *,
    origin=None,
    use_gpu: bool = False,
) -> np.ndarray:
    """``<core|r_α − R₀|virt>`` on a DFTpy grid, shape ``(3, n_core, n_virt)``.

    ``origin=None`` centers ``R₀`` on the core charge centroid — the same gauge
    :func:`run_reconstruct_cvs_from_driver` uses for the Casida dipoles, which
    keeps ``(r − R₀)`` small where the tight core density lives.
    """
    from casidapy.utils.accel import array_module, asnumpy

    cores = [np.asarray(p, dtype=float).ravel() for p in core_orbitals]
    if not cores:
        raise ValueError("core_orbitals is empty")
    if origin is None:
        origin = grid_charge_centroid(cores, grid)
    orig = np.asarray(origin, dtype=float).ravel()
    if orig.shape != (3,):
        raise ValueError(f"origin must have shape (3,), got {orig.shape}")

    xp = array_module(use_gpu)
    dV = float(grid.dV)
    rxyz = [
        xp.asarray(np.asarray(grid.r[a]).ravel() - orig[a], dtype=float)
        for a in range(3)
    ]
    virts = [
        xp.asarray(np.asarray(p, dtype=float).ravel(), dtype=float)
        for p in virtual_orbitals
    ]

    if not virts:
        return np.zeros((3, len(cores), 0), dtype=float)
    C = xp.stack([xp.asarray(c, dtype=float) for c in cores], axis=0)
    V = xp.stack(virts, axis=0)
    out = np.empty((3, len(cores), len(virts)), dtype=float)
    for alpha in range(3):
        out[alpha] = asnumpy((C * rxyz[alpha]) @ V.T) * dV
    return out


def core_virtual_dipole_matrix_pw_overlap(
    core,
    virtual_orbitals,
    grid,
    *,
    origin=None,
    normalize_core: bool = True,
) -> np.ndarray:
    """``⟨1s|r−R₀|ψ̃⟩`` on the QE charge mesh (plane-wave overlap).

    Maps the AE core onto ``grid`` and dots against stored PW virtuals.
    No GTO projection and no local spherical quadrature.
    """
    from casidapy.xas.cvs import core_mos_to_pw_fields

    psi_c = core_mos_to_pw_fields(core, grid, normalize=bool(normalize_core))
    R0 = origin
    if R0 is None:
        R0 = absorber_origin_bohr(core)
    return core_virtual_dipole_matrix_grid(
        psi_c, virtual_orbitals, grid, origin=R0
    )


def core_virtual_dipole_matrix_gto(
    core_mo_coeff,
    virt_mo_coeff,
    dip_ao: np.ndarray,
) -> np.ndarray:
    """``<core|r_α|virt>`` in an AO basis; ``dip_ao`` is ``(3, nao, nao)``."""
    C_c = np.asarray(core_mo_coeff, dtype=float)
    C_v = np.asarray(virt_mo_coeff, dtype=float)
    D = np.asarray(dip_ao, dtype=float)
    if C_c.ndim == 1:
        C_c = C_c.reshape(-1, 1)
    if C_v.ndim == 1:
        C_v = C_v.reshape(-1, 1)
    if D.ndim != 3 or D.shape[0] != 3:
        raise ValueError(f"dip_ao must be (3, nao, nao), got {D.shape}")
    return np.einsum("pi,apq,qj->aij", C_c, D, C_v, optimize=True)

def _orthogonalize_core_ao_against_virt_ao(
    core_ao: np.ndarray,
    virt_ao: np.ndarray,
    overlap: np.ndarray,
    *,
    n_pass: int = 2,
) -> np.ndarray:
    """Project core columns out of the virtual AO span (``S`` metric).

    Projected PW virtuals ``C_v = S^{-1}v`` are **not** ``S``-orthonormal, so
    the naive Gram–Schmidt update ``C ← C − V(VᵀSC)`` is incorrect (it assumes
    ``VᵀSV = I``).  The proper projector onto ``span(V)`` is

        P = V (Vᵀ S V)⁺ (Vᵀ S)

    with a pseudoinverse for rank-deficient / near-dependent virtual blocks.
    """
    C = np.asarray(core_ao, dtype=float).copy()
    if C.ndim == 1:
        C = C.reshape(-1, 1)
    V = np.asarray(virt_ao, dtype=float)
    S = np.asarray(overlap, dtype=float)
    if V.size == 0:
        norms = np.sqrt(np.maximum(np.sum(C * (S @ C), axis=0), 0.0))
        return C / np.maximum(norms, 1e-30)
    # G = Vᵀ S V may be singular when virtuals are linearly dependent in AO space.
    G = V.T @ S @ V
    # Symmetrize for numerical stability of the eigensolve / pinv.
    G = 0.5 * (G + G.T)
    for _ in range(max(int(n_pass), 1)):
        # Always lstsq: projected PW virtuals are often near-dependent in AO space.
        coeff = np.linalg.lstsq(G, V.T @ S @ C, rcond=1e-10)[0]
        C = C - V @ coeff
    norms = np.sqrt(np.maximum(np.sum(C * (S @ C), axis=0), 0.0))
    return C / np.maximum(norms, 1e-30)


def _fragment_dipole_integrals(mol, origin_bohr, *, gauge: str = "length"):
    """``int1e_r`` or ``int1e_ipovlp`` centered on ``origin_bohr`` (Bohr)."""
    R0 = np.asarray(origin_bohr, dtype=float).reshape(3)
    mode = str(gauge).lower().strip()
    with mol.with_common_orig(R0):
        if mode in ("velocity", "grad", "momentum"):
            return mol.intor("int1e_ipovlp", comp=3)
        return mol.intor("int1e_r", comp=3)


def core_virtual_dipole_matrix_gto_projected(
    core,
    virtual_orbitals,
    grid,
    *,
    origin=None,
    gauge: str = "length",
    gs_pass: int = 2,
    blksize: int = 16000,
    comm=None,
    projection_method: Optional[str] = None,
    becke_level: int = 4,
    driver=None,
    band_indices: Optional[Sequence[int]] = None,
):
    """``⟨core|r−R₀|virt⟩`` via fragment GTO projection + PySCF ``int1e_r``.

    PW virtuals are expanded in the core projection basis; the core MO uses the
    stored ``mo_coeff`` embedded when ghost projection is active. This matches
    Workflow 1 dipole physics near the absorber instead of sampling the AE 1s on
    the coarse FFT mesh.

    ``projection_method``
        ``'becke'`` (default) uses atom-centered quadrature + PW G-sum; pass
        ``'qe_grid'`` for the legacy uniform charge-mesh projector.
    """
    mol = core.fragment_mol
    C_c = np.asarray(core.mo_coeff, dtype=float)
    R0 = np.asarray(
        origin if origin is not None else absorber_origin_bohr(core),
        dtype=float,
    ).reshape(3)
    abs_atom = None
    eidx = (getattr(core, "meta", None) or {}).get("edge_atom_indices")
    if eidx is not None and len(np.asarray(eidx).ravel()):
        # Fragment mol may be absorber-only (atom peel) → local index 0.
        abs_atom = 0 if mol.natm == 1 else int(np.asarray(eidx).ravel()[0])
    V_ao = project_grid_orbitals_to_fragment_ao(
        mol,
        grid,
        virtual_orbitals,
        blksize=blksize,
        comm=comm,
        method=projection_method,
        becke_level=becke_level,
        absorber_atom=abs_atom,
        driver=driver,
        band_indices=band_indices,
    )
    S = mol.intor("int1e_ovlp")
    C_v = grid_projection_to_ao_coeffs(mol, V_ao)
    # Cap only: never amplify ‖c‖_S < 1 (high spillage); shrink ‖c‖_S > 1.
    from casidapy.xas.xas_utils import normalize_ao_coeffs_s_metric

    C_v, _ = normalize_ao_coeffs_s_metric(C_v, S, max_scale=1.0)
    C_orth = _orthogonalize_core_ao_against_virt_ao(C_c, C_v, S, n_pass=gs_pass)
    dip_ao = _fragment_dipole_integrals(mol, R0, gauge=gauge)
    return core_virtual_dipole_matrix_gto(C_orth, C_v, dip_ao)

def virtual_dipole_scores(
    dipole_matrix: np.ndarray,
    *,
    reduction: str = "max",
) -> np.ndarray:
    """Per-virtual transition-dipole magnitude from a ``(3, n_core, n_virt)`` block.

    ``|d_ca| = sqrt(Σ_α d_αca²)`` is reduced over cores: ``'max'`` (default)
    keeps the strongest core, ``'sum'`` the root-sum-square over cores.
    """
    d = np.asarray(dipole_matrix, dtype=float)
    if d.ndim != 3 or d.shape[0] != 3:
        raise ValueError(f"dipole_matrix must be (3, n_core, n_virt), got {d.shape}")
    mag = np.sqrt(np.sum(d**2, axis=0))
    mode = str(reduction).lower().strip()
    if mode == "max":
        return np.max(mag, axis=0)
    if mode in ("sum", "sumsq", "l2"):
        return np.sqrt(np.sum(mag**2, axis=0))
    raise ValueError(f"unknown reduction={reduction!r}; use 'max' or 'sum'")


def _select_virtuals_by_score(
    unocc_eigs,
    virtual_orbitals,
    scores: np.ndarray,
    *,
    min_score: float,
    n_keep: Optional[int],
    grid_mode: bool,
    label: str,
    verbose: bool,
):
    """Threshold / rank a virtual manifold by a per-virtual score.

    ``n_keep`` takes the largest ``n_keep`` scores outright; otherwise every
    virtual at or above ``min_score`` is kept, sorted by descending score.
    """
    eigs = np.asarray(unocc_eigs, dtype=float).ravel()
    n_virt = int(eigs.size)
    scores = np.asarray(scores, dtype=float).ravel()

    keep = scores >= float(min_score)
    if not np.any(keep):
        raise RuntimeError(
            f"no virtuals pass min {label}={min_score}; "
            f"max score={float(scores.max()):.3e}"
        )

    if n_keep is not None:
        n_keep = int(n_keep)
        if n_keep <= 0:
            raise ValueError(f"n_keep must be > 0, got {n_keep}")
        order = np.argsort(-scores)
        idx = order[: min(n_keep, order.size)]
    else:
        idx = np.flatnonzero(keep)
        idx = idx[np.argsort(scores[idx])[::-1]]

    e_keep = eigs[idx].copy()
    if grid_mode:
        psi_keep = [virtual_orbitals[i] for i in idx]
    else:
        C_v = np.asarray(virtual_orbitals, dtype=float)
        if C_v.ndim == 1:
            C_v = C_v.reshape(-1, 1)
        psi_keep = C_v[:, idx]

    if verbose:
        print(
            f"  core–virt {label}: kept {idx.size}/{n_virt} virtuals "
            f"(min score={float(scores[idx].min()):.3e}, "
            f"max={float(scores[idx].max()):.3e})"
        )

    return e_keep, psi_keep, idx


def select_virtuals_by_core_dipole(
    unocc_eigs,
    virtual_orbitals,
    core_orbitals,
    *,
    grid=None,
    dip_ao: Optional[np.ndarray] = None,
    origin=None,
    min_abs_dipole: float = 0.0,
    n_keep: Optional[int] = None,
    reduction: str = "max",
    use_gpu: bool = False,
    verbose: bool = False,
    ae_core=None,
):
    """Keep the virtuals carrying the largest core transition dipole.

    Ranks by ``|<core|r − R₀|virt>|`` (a.u.) using GTO-projected AE cores on
    the charge grid — the quantity entering the oscillator strength.

    Returns ``(unocc_eigs_keep, virtual_orbitals_keep, dipole_matrix, kept_indices)``.
    """
    eigs = np.asarray(unocc_eigs, dtype=float).ravel()
    if eigs.size == 0:
        return eigs, list(virtual_orbitals), np.zeros((3, 0, 0)), np.zeros(0, dtype=int)

    if grid is not None:
        if ae_core is not None:
            # Ranking-only: keep the fast QE-mesh projector. Becke+G-sum is for
            # the production Casida / oscillator-strength path, not pruning 100+
            # continuum bands.
            dip = core_virtual_dipole_matrix_gto_projected(
                ae_core,
                virtual_orbitals,
                grid,
                origin=origin,
                projection_method="qe_grid",
            )
        else:
            dip = core_virtual_dipole_matrix_grid(
                core_orbitals, virtual_orbitals, grid, origin=origin, use_gpu=use_gpu
            )
        grid_mode = True
    elif dip_ao is not None:
        dip = core_virtual_dipole_matrix_gto(core_orbitals, virtual_orbitals, dip_ao)
        grid_mode = False
    else:
        raise ValueError("select_virtuals_by_core_dipole needs grid= or dip_ao=")

    scores = virtual_dipole_scores(dip, reduction=reduction)
    e_keep, psi_keep, idx = _select_virtuals_by_score(
        eigs,
        list(virtual_orbitals) if grid_mode else virtual_orbitals,
        scores,
        min_score=float(min_abs_dipole),
        n_keep=n_keep,
        grid_mode=grid_mode,
        label="dipole",
        verbose=verbose,
    )
    return e_keep, psi_keep, dip, idx


def _commit_pruned_grid_virtuals(kernel, e_keep, psi_keep, idx=None):
    kernel.set_active_orbitals(kernel._occ_e, e_keep, kernel._psi_occ, psi_keep)
    bands = getattr(kernel, "_qe_band_indices_unocc", None)
    if bands is not None and idx is not None:
        kernel._qe_band_indices_unocc = [bands[int(i)] for i in np.asarray(idx).ravel()]


def _commit_pruned_gto_virtuals(kernel, e_keep, C_keep):
    kernel._unocc_e = np.asarray(e_keep, dtype=float)
    kernel._C_v = np.asarray(C_keep, dtype=float)
    kernel._n_unocc = int(kernel._C_v.shape[1])
    kernel._n_trans = int(kernel._n_occ * kernel._n_unocc)
    kernel._dE = None
    kernel._ready = False
    kernel._K = None
    if hasattr(kernel, "_K_dev"):
        kernel._K_dev = None
    if hasattr(kernel, "_C_v_dev"):
        kernel._C_v_dev = None


def _core_dipole_ao(kernel, core_orbitals):
    """AO dipole integrals gauged on the core charge centroid."""
    mol = getattr(kernel, "mol", None) or getattr(kernel.mf, "mol", None)
    if mol is None:
        raise ValueError("GTO dipole pruning requires dip_ao= or kernel.mol")
    C_c = np.asarray(core_orbitals, dtype=float)
    if C_c.ndim == 1:
        C_c = C_c.reshape(-1, 1)
    S = mol.intor("int1e_ovlp")
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        D0 = mol.intor("int1e_r", comp=3)
    norm = float(C_c[:, 0] @ S @ C_c[:, 0])
    origin = [float(C_c[:, 0] @ D0[a] @ C_c[:, 0]) / norm for a in range(3)]
    with mol.with_common_orig(origin):
        return mol.intor("int1e_r", comp=3)


def prune_kernel_virtuals_by_core_dipole(
    kernel,
    core_orbitals,
    *,
    min_abs_dipole: float = 0.0,
    n_keep: Optional[int] = None,
    reduction: str = "max",
    dip_ao: Optional[np.ndarray] = None,
    origin=None,
    use_gpu: bool = False,
    verbose: bool = False,
    ae_core=None,
):
    """Keep the highest transition-dipole virtuals before CVS inject.

    On a plane-wave host this discards box continuum bands that carry no XAS
    intensity. Requires ``ae_core`` for GTO-projected dipole scoring.
    """
    if not getattr(kernel, "_psi_unocc", None) and not hasattr(kernel, "_C_v"):
        raise RuntimeError("kernel has no virtual active space to prune")

    if getattr(kernel, "provides_grid", False):
        e_keep, psi_keep, dip, idx = select_virtuals_by_core_dipole(
            kernel._unocc_e,
            kernel._psi_unocc,
            core_orbitals,
            grid=kernel.grid,
            origin=origin,
            min_abs_dipole=min_abs_dipole,
            n_keep=n_keep,
            reduction=reduction,
            use_gpu=use_gpu,
            verbose=verbose,
            ae_core=ae_core,
        )
        _commit_pruned_grid_virtuals(kernel, e_keep, psi_keep, idx=idx)
    elif hasattr(kernel, "_C_v"):
        if dip_ao is None:
            dip_ao = _core_dipole_ao(kernel, core_orbitals)
        e_keep, C_keep, dip, idx = select_virtuals_by_core_dipole(
            kernel._unocc_e,
            kernel._C_v,
            core_orbitals,
            dip_ao=dip_ao,
            min_abs_dipole=min_abs_dipole,
            n_keep=n_keep,
            reduction=reduction,
            verbose=verbose,
        )
        _commit_pruned_gto_virtuals(kernel, e_keep, C_keep)
    else:
        raise TypeError(
            f"prune_kernel_virtuals_by_core_dipole: unsupported {type(kernel)}"
        )

    kernel._psi_unocc_arr = None
    kernel._psi_unocc_dev = None
    return kernel, dip, idx

