"""Plane-wave evaluation off the QE FFT mesh (G-sum / Fourier interpolant).

Used to form ``V[μ,a] = ∫ φ_μ(r) ψ_a(r) dV`` on PySCF Becke grids so that
tight absorber AOs are integrated accurately without refining ``ecutwfc``.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def _field_to_nxnynz(orbital, grid, *, order: str = "C") -> np.ndarray:
    """Return orbital samples as ``(nx, ny, nz)`` on ``grid``.

    QE / DFTpy charge fields are Fortran-raveled (``order='F'``).  Stored
    PW orbitals from some paths are C-contiguous.
    """
    nr = tuple(int(x) for x in np.asarray(grid.nr).ravel()[:3])
    arr = np.asarray(orbital, dtype=np.float64)
    if arr.shape == nr:
        return np.asarray(arr, dtype=np.float64)
    ord_c = "C" if str(order).upper().startswith("C") else "F"
    flat = arr.ravel(order=ord_c)
    if flat.size != int(np.prod(nr)):
        raise ValueError(
            f"orbital size {flat.size} incompatible with grid.nr={nr}"
        )
    return flat.reshape(nr, order=ord_c)


def reciprocal_miller(nr: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integer Miller indices for ``numpy.fft.fftn`` modes (centered)."""
    nx, ny, nz = (int(nr[0]), int(nr[1]), int(nr[2]))
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    kz = np.fft.fftfreq(nz) * nz
    return kx, ky, kz


def fft_coeffs_from_orbital(
    orbital,
    grid,
    *,
    g_cutoff_rel: float = 1e-10,
    ecut_ha: Optional[float] = None,
    order: str = "C",
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Band-limited PW coeffs of a stored real-space orbital.

    Returns ``(G_cart_bohr_inv, c_G, meta)`` where

    ``ψ(r) = Σ_G c_G exp(i G·r)``

    matches ``orbital`` on the FFT mesh (Fourier interpolant).  Tiny
    coefficients are dropped via ``g_cutoff_rel * max|c|``.  Optional
    ``ecut_ha`` also drops modes with ``|G|²/2 > ecut_ha``.
    """
    psi_r = _field_to_nxnynz(orbital, grid, order=order)
    nx, ny, nz = psi_r.shape
    c_full = np.fft.fftn(psi_r)
    # numpy: psi = ifft(c) = (1/N) Σ c exp(+); so c_G_eval = c_full / N
    N = float(nx * ny * nz)
    c_full = c_full / N

    lattice = np.asarray(grid.lattice, dtype=float).reshape(3, 3)
    # G = 2π n·b ; b rows are reciprocal basis (inv(A).T)
    bg = 2.0 * np.pi * np.linalg.inv(lattice).T
    kx, ky, kz = reciprocal_miller((nx, ny, nz))
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    mill = np.stack([KX.ravel(), KY.ravel(), KZ.ravel()], axis=1)
    G = mill @ bg
    c = c_full.ravel()

    amp = np.abs(c)
    thr = float(g_cutoff_rel) * float(amp.max() if amp.size else 0.0)
    keep = amp >= thr if thr > 0.0 else np.ones(amp.shape, dtype=bool)
    if ecut_ha is not None and float(ecut_ha) > 0.0:
        g2 = np.einsum("ij,ij->i", G, G)
        keep &= g2 <= (2.0 * float(ecut_ha) * 1.05)
    meta = {
        "nr": (nx, ny, nz),
        "n_G_full": int(c.size),
        "n_G_keep": int(np.count_nonzero(keep)),
        "g_cutoff_rel": float(g_cutoff_rel),
        "ecut_ha": None if ecut_ha is None else float(ecut_ha),
        "max_abs_c": float(amp.max() if amp.size else 0.0),
        "fft_order": str(order).upper()[:1],
    }
    return G[keep], c[keep], meta


def pw_sum_at_coords(
    G: np.ndarray,
    c: np.ndarray,
    coords_bohr: np.ndarray,
    *,
    real: bool = True,
    blk_pts: int = 256,
) -> np.ndarray:
    """``ψ(r) = Σ_G c_G e^{i G·r}`` at Cartesian ``coords`` (Bohr).

    Returns shape ``(n_pts,)`` (real part if ``real``).
    """
    G = np.asarray(G, dtype=float).reshape(-1, 3)
    c = np.asarray(c, dtype=np.complex128).ravel()
    coords = np.asarray(coords_bohr, dtype=float).reshape(-1, 3)
    if G.shape[0] != c.size:
        raise ValueError(f"G ({G.shape[0]}) and c ({c.size}) length mismatch")
    n_pts = coords.shape[0]
    out = np.empty(n_pts, dtype=np.complex128)
    for p0 in range(0, n_pts, max(int(blk_pts), 1)):
        p1 = min(p0 + blk_pts, n_pts)
        # (m, n_G) = coords[m]·G[g]
        phase = coords[p0:p1] @ G.T
        out[p0:p1] = np.exp(1j * phase) @ c
    if real:
        return np.real(out)
    return out


def orbitals_at_coords_pw_sum(
    orbitals,
    grid,
    coords_bohr: np.ndarray,
    *,
    g_cutoff_rel: float = 1e-10,
    ecut_ha: Optional[float] = None,
    real: bool = True,
    blk_pts: int = 256,
) -> Tuple[np.ndarray, list]:
    """Evaluate each stored PW orbital at ``coords`` via Fourier interpolant.

    Returns ``(psi, metas)`` with ``psi.shape == (n_pts, n_orb)``.
    """
    coords = np.asarray(coords_bohr, dtype=float).reshape(-1, 3)
    orbs = list(orbitals)
    n_orb = len(orbs)
    psi = np.zeros((coords.shape[0], n_orb), dtype=float if real else np.complex128)
    metas = []
    for a, orb in enumerate(orbs):
        G, c, meta = fft_coeffs_from_orbital(
            orb, grid, g_cutoff_rel=g_cutoff_rel, ecut_ha=ecut_ha
        )
        psi[:, a] = pw_sum_at_coords(
            G, c, coords, real=real, blk_pts=blk_pts
        )
        metas.append(meta)
    return psi, metas


def trilinear_sample_field(grid, field, coords_bohr: np.ndarray) -> np.ndarray:
    """PBC trilinear sample of a real-space orbital at Bohr coordinates."""
    nr = np.array([int(n) for n in np.asarray(grid.nr).ravel()[:3]], dtype=float)
    lattice = np.asarray(grid.lattice, dtype=float).reshape(3, 3)
    vals = _field_to_nxnynz(field, grid)
    coords = np.asarray(coords_bohr, dtype=float).reshape(-1, 3)
    frac = coords @ np.linalg.inv(lattice)
    idx = frac * nr
    nx, ny, nz = (int(n) for n in nr)
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


def spline_sample_field(
    grid,
    field,
    coords_bohr: np.ndarray,
    *,
    order: int = 3,
) -> np.ndarray:
    """PBC B-spline sample of ``field`` at Bohr coordinates.

    A QE wavefunction is band-limited at ``ecutwfc``, so with the production
    mesh (``h ≈ 0.2`` bohr vs a Nyquist spacing of ``π/|G|max ≈ 0.57`` bohr)
    the stored samples determine ``ψ`` everywhere and the only question is the
    order of the reconstruction.  Trilinear is ``O(h²)`` in the value and
    ``O(h)`` in the gradient — and a 1s-weighted dipole integral is a gradient
    probe at the nucleus.  A tricubic B-spline is ``O(h⁴)`` for the same cost
    scaling, recovering the exact G-sum to ~1e-4 relative where trilinear
    manages ~3e-2.
    """
    from scipy import ndimage

    nr = np.array([int(n) for n in np.asarray(grid.nr).ravel()[:3]], dtype=float)
    lattice = np.asarray(grid.lattice, dtype=float).reshape(3, 3)
    vals = _field_to_nxnynz(field, grid)
    coords = np.asarray(coords_bohr, dtype=float).reshape(-1, 3)
    idx = (coords @ np.linalg.inv(lattice)) * nr
    # 'grid-wrap' + prefilter gives a genuinely periodic B-spline, matching the
    # PBC wrap the trilinear sampler applies.
    return ndimage.map_coordinates(
        vals, idx.T, order=int(order), mode="grid-wrap", prefilter=True
    )


def orbitals_at_coords_spline(
    orbitals,
    grid,
    coords_bohr: np.ndarray,
    *,
    order: int = 3,
) -> Tuple[np.ndarray, dict]:
    """PBC B-spline sample of stored PW orbitals at arbitrary Bohr coords."""
    coords = np.asarray(coords_bohr, dtype=float).reshape(-1, 3)
    orbs = list(orbitals)
    psi = np.empty((coords.shape[0], len(orbs)), dtype=float)
    for a, orb in enumerate(orbs):
        psi[:, a] = spline_sample_field(grid, orb, coords, order=order)
    meta = {
        "source": f"spline{int(order)}",
        "order": int(order),
        "n_orb": int(len(orbs)),
        "n_pts": int(coords.shape[0]),
    }
    return psi, meta


def orbitals_at_coords_trilinear(
    orbitals,
    grid,
    coords_bohr: np.ndarray,
) -> Tuple[np.ndarray, dict]:
    """PBC trilinear sample of stored PW orbitals at arbitrary Bohr coords.

    Cheapest option, but only ``O(h²)``; prefer
    :func:`orbitals_at_coords_spline` unless you are reproducing old runs.
    """
    coords = np.asarray(coords_bohr, dtype=float).reshape(-1, 3)
    orbs = list(orbitals)
    n_orb = len(orbs)
    psi = np.empty((coords.shape[0], n_orb), dtype=float)
    for a, orb in enumerate(orbs):
        psi[:, a] = trilinear_sample_field(grid, orb, coords)
    meta = {
        "source": "trilinear",
        "n_orb": int(n_orb),
        "n_pts": int(coords.shape[0]),
    }
    return psi, meta


def qe_evc_gvectors_coeffs(
    driver,
    band_indices: Sequence[int],
    *,
    kpt: int = 0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Pull ``(G_cart, C[:,a], meta)`` from QEpy ``evc`` for the given bands.

    ``ψ_a(r) = (1/√Ω) Σ_G C_{G a} e^{i G·r}`` (QE convention).  Gamma-only
    runs may store a packed basis; we reconstruct Cartesian ``G`` via ``igk``.
    """
    band_indices = [int(i) for i in band_indices]
    driver.qepy_pw.qepy_mod.qepy_get_evc(int(kpt) + 1)
    evc = np.asarray(
        driver.qepy_modules.wavefunctions.get_array_evc(), dtype=np.complex128
    )
    # evc shape (npwx, nbnd) typically
    if evc.ndim != 2:
        raise RuntimeError(f"unexpected evc ndim={evc.ndim}, shape={evc.shape}")

    ngk = int(np.asarray(driver.qepy_pw.klist.get_array_ngk()).ravel()[kpt])
    igk = np.asarray(driver.qepy_pw.klist.get_array_igk_k(), dtype=int)
    # igk_k(:, ik) is 1-based index into gvect
    if igk.ndim == 1:
        igk_k = igk[:ngk]
    else:
        igk_k = igk[:ngk, kpt] if igk.shape[1] > kpt else igk[:ngk, 0]
    ig = igk_k - 1

    g_cart = np.asarray(driver.qepy_modules.gvect.get_array_g(), dtype=float)
    # g is (3, ngm) in QE
    if g_cart.shape[0] == 3:
        G = g_cart[:, ig].T.copy()
    else:
        G = g_cart[ig].copy()

    omega = float(driver.get_volume())
    scale = 1.0 / np.sqrt(max(omega, 1e-30))

    C = np.empty((ngk, len(band_indices)), dtype=np.complex128)
    for j, ib in enumerate(band_indices):
        if ib < 0 or ib >= evc.shape[1]:
            raise IndexError(f"band {ib} out of range for evc nbnd={evc.shape[1]}")
        C[:, j] = evc[:ngk, ib] * scale

    meta = {
        "ngk": int(ngk),
        "npwx": int(evc.shape[0]),
        "nbnd": int(evc.shape[1]),
        "omega": omega,
        "kpt": int(kpt),
        "source": "qepy_evc",
    }
    return G, C, meta


def orbitals_at_coords_qe_evc(
    driver,
    band_indices: Sequence[int],
    coords_bohr: np.ndarray,
    *,
    kpt: int = 0,
    real: bool = True,
    blk_pts: int = 256,
) -> Tuple[np.ndarray, dict]:
    """Direct G-sum from QE ``evc`` at arbitrary Bohr coordinates."""
    G, C, meta = qe_evc_gvectors_coeffs(driver, band_indices, kpt=kpt)
    coords = np.asarray(coords_bohr, dtype=float).reshape(-1, 3)
    n_pts = coords.shape[0]
    n_orb = C.shape[1]
    out = np.empty((n_pts, n_orb), dtype=np.complex128)
    for p0 in range(0, n_pts, max(int(blk_pts), 1)):
        p1 = min(p0 + blk_pts, n_pts)
        phase = np.exp(1j * (coords[p0:p1] @ G.T))  # (m, nG)
        out[p0:p1] = phase @ C
    if real:
        return np.real(out), meta
    return out, meta
