"""Absorber-core Hartree (radial) + XC on a nuclear Becke grid.

The AE 1s density is a ~0.2 Bohr feature.  Collocating it on the QE charge
FFT makes ``v_xc`` jump by tens of eV when the nucleus sits on a voxel
(diamond 61596089 vs half-grid 61597339).  ``ρ_core`` therefore never
touches that mesh (Step C):

* ``v_H[ρ_core]`` is the spherical Poisson integral on a 1-D radial ray
* ``v_xc[ρ_core^AE + ρ_tot^QE]`` is the DFA (GGA) piece on an unpruned
  Treutler–Ahlrichs Becke grid; the pseudo valence density is splined onto
  those points
* hybrids add ``−½ [hyb K + (α−hyb) K_ω]`` of the frozen core DM — real
  exact exchange, not ``eval_xc('pbe0')`` pretending HF is local

The quadrature gate is ``∫ ρ_core dV = 2 n_core`` to several digits on that
Becke grid, checked before any eigenvalue is used.  Hybrid ``xc`` adds
1s Fock here (``+U``-like).  TDA ``A`` stays Hartree + ALDA.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Fine enough to resolve a 0.2 Bohr 1s.  Default molecular grids prune the
# nuclear region; that is exactly wrong here.
DEFAULT_N_RADIAL = 200
DEFAULT_N_ANGULAR = 590
DEFAULT_NELEC_TOL = 1.0e-3
RADIAL_HARTREE_N = 2500
RADIAL_HARTREE_RMAX = 20.0

# QE ``input_dft`` for names we actually run.  Unknown hybrids are passed
# through; QE will reject them.  Semilocal names return None → omit the tag
# and let the UPF (PBE here) set the host functional.
_QE_INPUT_DFT = {
    "pbe": None,
    "lda": None,
    "pz": None,
    "pw91": None,
    "blyp": None,
    "pbe0": "pbe0",
    "pbeh": "pbe0",
    "b3lyp": "b3lyp",
    "hse": "hse",
    "hse06": "hse",
}


def _xc_key(xc: str) -> str:
    return "".join(ch for ch in str(xc).lower() if ch.isalnum())


# Libxc does not register ``bhlyp`` / ``bhhlyp``.  Those names mean BHandHLYP
# (50% HF + 50% B88 + LYP).
_XC_ALIASES = {
    "bhlyp": "bhandhlyp",
    "bhhlyp": "bhandhlyp",
    "bhandhlyp": "bhandhlyp",
}


def canonical_xc(xc: str) -> str:
    """Libxc/PySCF name.  ``bhlyp`` → ``bhandhlyp``."""
    return _XC_ALIASES.get(_xc_key(xc), str(xc))


def hybrid_rsh_coeff(xc: str, spin: int = 0):
    """``(omega, alpha, hyb)`` as PySCF RKS uses them in ``get_veff``.

    ``hyb`` is the full-range HF fraction; range-separated functionals add
    ``(alpha − hyb) * K(ω)`` with ``ω`` the erf range-separation parameter.
    """
    from pyscf.dft import numint

    ni = numint.NumInt()
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(
        canonical_xc(xc), spin=int(spin)
    )
    return float(omega or 0.0), float(alpha or 0.0), float(hyb or 0.0)


def is_hybrid_xc(xc: str, spin: int = 0) -> bool:
    omega, alpha, hyb = hybrid_rsh_coeff(xc, spin=spin)
    return abs(hyb) > 1e-14 or abs(alpha) > 1e-14 or abs(omega) > 1e-14


def qe_input_dft(xc: str):
    """QE ``&system input_dft`` or ``None`` to leave the UPF functional."""
    key = _xc_key(xc)
    if key in _QE_INPUT_DFT:
        return _QE_INPUT_DFT[key]
    if is_hybrid_xc(xc):
        return str(xc).lower()
    return None


def qe_hybrid_system_namelist(xc: str, *, assume_isolated=None) -> dict:
    """Extra ``&system`` keys for a hybrid QE host (gamma-only supercell)."""
    dft = qe_input_dft(xc)
    if not dft:
        return {}
    keys = {
        "input_dft": f"'{dft}'",
        "nqx1": 1,
        "nqx2": 1,
        "nqx3": 1,
        "exxdiv_treatment": "'gygi-baldereschi'",
        "x_gamma_extrapolation": True,
    }
    if assume_isolated:
        # Isolated boxes: drop the crystal Madelung EXX correction.
        keys["x_gamma_extrapolation"] = False
    return keys


CASIDA_EXX_MODES = ("off", "gto", "fft", "bare", "screened")
CASIDA_EXX_IMPLEMENTED = frozenset({"off", "gto", "fft"})
_CASIDA_EXX_ALIASES = {
    "": "off",
    "false": "off",
    "0": "off",
    "none": "off",
    "no": "off",
    "true": "fft",
    "on": "fft",
    "1": "fft",
    "ij_ab": "fft",
    "ijab": "fft",
    "ao": "gto",
    "fragment": "gto",
    "projected": "gto",
}


def parse_casida_exx_mode(mode) -> str:
    """``off`` / ``gto`` / ``fft`` / ``bare`` / ``screened``.  Default ``off``.

    ``gto`` is production hybrid ``A``: fragment GTO ``(ij|ab)`` of
    GIPAW-projected virtuals.  ``fft`` is the collocated-1s Hartree
    (library tests only).  ``bare`` / ``screened`` are not wired.
    """
    raw = "off" if mode is None else str(mode).strip().lower()
    parsed = _CASIDA_EXX_ALIASES.get(raw, raw)
    if parsed not in CASIDA_EXX_MODES:
        raise ValueError(
            f"casida_exx={mode!r}; expected one of {CASIDA_EXX_MODES}"
        )
    return parsed


def casida_alda_xc(xc: str):
    """Name handed to DFTpy for ALDA ``f_xc``.

    Libxc hybrid ``f_xc`` is the DFA remainder only (honest ALDA of
    PBE0/B3LYP/HSE).  The nonlocal CIS term ``−c_x (ij|ab)`` is a separate
    kernel contribution; see :func:`casida_exx_kernel`.  Returns
    ``(name, meta)``.
    """
    omega, alpha, hyb = hybrid_rsh_coeff(xc)
    meta = {
        "xc_requested": str(xc),
        "xc_alda": str(xc),
        "hybrid_coeff": hyb,
        "rsh_omega": omega,
        "rsh_alpha": alpha,
        "casida_exx_kernel": False,
        "casida_fxc": "dfa_remainder" if is_hybrid_xc(xc) else "semilocal",
    }
    return str(xc), meta


def casida_exx_kernel(xc: str, pw_xc: str = "PBE", *, mode: str = "off"):
    """HF coefficients for CIS ``−c_x (ij|ab)`` in TDA ``A``.

    ``mode`` is ``off`` in production default: hybrid ``--xc`` is core Fock
    on the oneshot only (``+U``-like).  ``gto`` adds fragment ``(ij|ab)``
    of GIPAW-projected virtuals (honest hybrid TDA).  ``fft`` is a
    library-only collocated Hartree, not an XAS kernel.

    Prefers the core functional ``xc`` when it is a hybrid so the kernel
    matches the frozen-core Fock exchange.  Otherwise uses ``pw_xc``.
    Does not scale the exciton Hartree ``(ia|jb)``.

    Occupancy: CIS wants the orbital product ``|φ_1s|²``.  The oneshot
    ``v_H[ρ_core]`` uses ``2|φ_1s|²`` — do not reuse that field unhalved.
    """
    mode = parse_casida_exx_mode(mode)
    src = xc if is_hybrid_xc(xc) else pw_xc
    omega, alpha, hyb = hybrid_rsh_coeff(src)
    coeff_active = abs(hyb) > 1e-14 or (
        abs(omega) > 1e-14 and abs(alpha - hyb) > 1e-14
    )
    active = bool(mode != "off" and coeff_active)
    if active and mode not in CASIDA_EXX_IMPLEMENTED:
        raise NotImplementedError(
            f"casida_exx={mode!r} needs radial W[|φ_1s|²]; "
            "use off (default), gto (fragment ERIs), or fft (collocation test)."
        )
    term = None
    if active:
        term = {
            "gto": "ij_ab_gto",
            "fft": "ij_ab_fft",
            "bare": "ij_ab_bare",
            "screened": "ij_ab_screened",
        }[mode]
    mixed = (
        is_hybrid_xc(xc)
        and is_hybrid_xc(pw_xc)
        and canonical_xc(xc) != canonical_xc(pw_xc)
    )
    return {
        "exx_xc": str(src),
        "hybrid_coeff": hyb,
        "rsh_omega": omega,
        "rsh_alpha": alpha,
        "casida_exx_mode": mode if active else "off",
        "casida_exx_kernel": bool(active),
        "casida_exx_term": term,
        "casida_exx_mixed": bool(mixed),
    }


def core_exact_exchange_ao(mol, dm_core, xc: str):
    """AO exact-exchange of the frozen core, PySCF RKS hybrid convention.

    ``v = −½ [hyb K[dm] + (α − hyb) K_ω[dm]]``.  ``eval_xc`` is not used.
    Scope is ``core_only``: 1s–valence K would need reconstructed occupied
    valence orbitals, which this path does not build.
    """
    from pyscf import scf

    xc = canonical_xc(xc)
    dm = np.asarray(dm_core, float)
    if dm.ndim == 3:
        if dm.shape[0] != 2:
            raise ValueError(f"expected RKS or UKS DM, got shape {dm.shape}")
        dm = dm[0] + dm[1]
    nao = int(mol.nao)
    omega, alpha, hyb = hybrid_rsh_coeff(xc, spin=0)
    meta = {
        "xc": str(xc),
        "hybrid_coeff": hyb,
        "rsh_omega": omega,
        "rsh_alpha": alpha,
        "exx_scope": "core_only",
        "exx_frobenius_ha": 0.0,
    }
    if abs(hyb) < 1e-14 and abs(alpha) < 1e-14:
        return np.zeros((nao, nao), float), meta

    mf_k = scf.RHF(mol)
    mf_k.verbose = 0
    vk = np.asarray(mf_k.get_k(mol, dm), float) * hyb
    if abs(omega) > 1e-10:
        vklr = np.asarray(mf_k.get_k(mol, dm, omega=omega), float)
        vk = vk + vklr * (alpha - hyb)
    if vk.ndim == 3:
        vk = 0.5 * (vk[0] + vk[1])
    v_exx = -0.5 * np.asarray(vk, float)
    v_exx = 0.5 * (v_exx + v_exx.T)
    meta["exx_frobenius_ha"] = float(np.linalg.norm(v_exx))
    return v_exx, meta


def radial_hartree_spherical(r: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Spherical ``v_H[ρ](r)`` in Hartree.

    ``ρ`` is the 3-D density (e⁻/Bohr³) on a ray.  For a spherical density::

        v_H(r) = (4π/r) ∫_0^r ρ r'² dr' + 4π ∫_r^∞ ρ r' dr'

    At ``r = 0`` the first term vanishes and the second is ``4π ∫_0^∞ ρ r dr``.
    """
    from scipy.integrate import cumulative_trapezoid, trapezoid

    r = np.asarray(r, float).ravel()
    rho = np.asarray(rho, float).ravel()
    if r.size != rho.size:
        raise ValueError(f"r ({r.size}) and rho ({rho.size}) length mismatch")
    if r.size < 2:
        raise ValueError("radial_hartree_spherical needs at least 2 points")
    if np.any(np.diff(r) < -1e-15):
        raise ValueError("radial mesh must be non-decreasing")
    r2rho = rho * r * r
    rrho = rho * r
    inner = cumulative_trapezoid(r2rho, r, initial=0.0)
    outer_full = float(trapezoid(rrho, r))
    outer = outer_full - cumulative_trapezoid(rrho, r, initial=0.0)
    v = np.empty_like(r)
    tiny = r < 1.0e-14
    v[~tiny] = 4.0 * np.pi * (inner[~tiny] / r[~tiny] + outer[~tiny])
    v[tiny] = 4.0 * np.pi * outer_full
    return v


def _core_ray_density(mol, dm_c, r: np.ndarray, center_bohr: np.ndarray):
    """``ρ_core`` along ``center + r ẑ`` (e⁻/Bohr³)."""
    from pyscf.dft import numint

    center = np.asarray(center_bohr, float).reshape(3)
    r = np.asarray(r, float).ravel()
    coords = center[None, :] + np.stack(
        [np.zeros_like(r), np.zeros_like(r), r], axis=1
    )
    ao = numint.eval_ao(mol, coords)
    dm = np.asarray(dm_c, float)
    return np.einsum("pi,ij,pj->p", ao, dm, ao)


def radial_hartree_from_core_dm(mol, dm_c, *, rmax: float = RADIAL_HARTREE_RMAX, n_r: int = RADIAL_HARTREE_N):
    """``(r, ρ_core(r), v_H(r))`` for a spherical core density on ``mol`` atom 0."""
    center = np.asarray(mol.atom_coord(0), float).reshape(3)
    r = np.concatenate(
        (
            np.array([0.0]),
            np.geomspace(1.0e-6, float(rmax), int(n_r) - 1),
        )
    )
    rho = _core_ray_density(mol, dm_c, r, center)
    v_h = radial_hartree_spherical(r, rho)
    return r, rho, v_h, center


def build_absorber_core_becke_grid(
    mol,
    *,
    n_radial: int = DEFAULT_N_RADIAL,
    n_angular: int = DEFAULT_N_ANGULAR,
    absorber_atom: int = 0,
):
    """Unpruned Treutler–Ahlrichs grid, dense on the absorber only."""
    from pyscf.dft import gen_grid, radi

    grids = gen_grid.Grids(mol)
    grids.prune = None
    grids.radi_method = radi.treutler_ahlrichs
    grids.atom_grid = {int(absorber_atom): (int(n_radial), int(n_angular))}
    # Other atoms (ghosts): keep a cheap shell so Grids.build is happy.
    for ia in range(mol.natm):
        if int(ia) != int(absorber_atom):
            grids.atom_grid[int(ia)] = (20, 86)
    grids.build()
    coords = np.asarray(grids.coords, float)
    weights = np.asarray(grids.weights, float)
    if mol.natm > 1:
        R = np.asarray(mol.atom_coords(), float)
        nearest = np.argmin(np.sum((coords[:, None, :] - R[None, :, :]) ** 2, axis=2), axis=1)
        keep = nearest == int(absorber_atom)
        coords = coords[keep]
        weights = weights[keep]
    meta = {
        "n_pts": int(coords.shape[0]),
        "n_radial": int(n_radial),
        "n_angular": int(n_angular),
        "prune": None,
        "radi_method": "treutler_ahlrichs",
        "absorber_atom": int(absorber_atom),
    }
    return coords, weights, meta


def core_density_on_becke(mol, dm_c, coords, weights, *, deriv: int = 1):
    """GGA ``ρ_core`` on Becke points: shape ``(4, n)`` if ``deriv>=1`` else ``(n,)``."""
    from pyscf.dft import numint

    ni = numint.NumInt()
    ao = numint.eval_ao(mol, coords, deriv=max(int(deriv), 0))
    xctype = "GGA" if int(deriv) >= 1 else "LDA"
    rho = ni.eval_rho(mol, ao, np.asarray(dm_c, float), xctype=xctype)
    nelec = float(np.dot(np.asarray(rho if rho.ndim == 1 else rho[0], float), weights))
    return rho, nelec, ao


def _qe_density_3d(driver, grid):
    """QE valence density on the DFTpy grid layout (matches ``grid.r``)."""
    from casidapy.embed.hirshfeld import total_density_on_grid

    nr = tuple(int(x) for x in grid.nr)
    rho = np.asarray(total_density_on_grid(driver, grid), float).ravel()
    field = driver.data2field(rho, grid=grid)
    arr = np.asarray(field, float)
    if arr.ndim == 4:
        arr = np.squeeze(arr)
    if arr.shape != nr:
        arr = arr.ravel(order="F").reshape(nr, order="F")
    return arr


def rho_tot_and_grad_at_coords(driver, grid, coords_bohr, *, fd_step: float = 1.0e-3):
    """Spline ``ρ_tot^QE`` and a central-difference gradient at Becke points.

    Valence density is smooth on the charge-grid spacing; cubic spline is
    accurate.  Full Fourier interpolation over ``N_G × N_becke`` is needless.
    """
    from casidapy.xas.pw_eval import spline_sample_field

    rho3 = _qe_density_3d(driver, grid)
    coords = np.asarray(coords_bohr, float).reshape(-1, 3)
    rho = np.asarray(spline_sample_field(grid, rho3, coords), float)
    h = float(fd_step)
    grad = np.empty((3, coords.shape[0]), float)
    eye = np.eye(3)
    for ax in range(3):
        rp = spline_sample_field(grid, rho3, coords + h * eye[ax])
        rm = spline_sample_field(grid, rho3, coords - h * eye[ax])
        grad[ax] = (np.asarray(rp, float) - np.asarray(rm, float)) / (2.0 * h)
    return rho, grad


def _gga_vxc_ao(ao_deriv, weights, rho4, xc: str):
    """AO matrix of GGA ``v_xc[ρ]`` from ``eval_ao(..., deriv=1)`` and ``ρ`` ``(4, n)``.

    For a hybrid ``xc`` this is the DFA remainder only (libxc).  HF exchange
    is ``core_exact_exchange_ao``, not this local potential.
    """
    from pyscf.dft import numint

    ni = numint.NumInt()
    xc = canonical_xc(xc)
    xctype = ni._xc_type(xc)
    if str(xctype).upper() == "MGGA":
        raise ValueError(
            f"XC '{xc}' is meta-GGA; production Becke v_xc is GGA (ρ, ∇ρ). "
            "Use a GGA hybrid (PBE0, B3LYP, HSE06)."
        )
    _exc, vxc = ni.eval_xc(xc, rho4, spin=0, relativity=0, deriv=1)[:2]
    vrho = np.asarray(vxc[0], float).ravel()
    w = np.asarray(weights, float).ravel()
    ao0 = np.asarray(ao_deriv[0], float)
    vmat = (ao0.T * (vrho * w)) @ ao0
    if len(vxc) > 1 and vxc[1] is not None:
        vsigma = np.asarray(vxc[1], float).ravel()
        # 2 vσ ∇ρ · ∇(φμ φν)
        sw = 2.0 * vsigma * w
        gdot = np.einsum("xn,xni->ni", rho4[1:4], ao_deriv[1:4])
        extra = (ao0.T * sw) @ gdot
        vmat = vmat + extra + extra.T
    return 0.5 * (vmat + vmat.T)


def core_plus_tot_becke_ao(
    mol,
    dm_core,
    driver,
    grid,
    *,
    xc: str = "pbe",
    n_core_orb: int = 1,
    n_radial: int = DEFAULT_N_RADIAL,
    n_angular: int = DEFAULT_N_ANGULAR,
    nelec_tol: float = DEFAULT_NELEC_TOL,
    upf_path: Optional[str] = None,
    absorber_atom: int = 0,
):
    """``v_H[ρ_core] + v_xc^DFA[ρ_core + ρ_tot] + hyb K[ρ_core]`` in AO.

    Raises if the Becke integral of ``ρ_core`` is not ``2 n_core_orb`` to
    ``nelec_tol``.  That gate must pass before any 1s eigenvalue is trusted.
    Hybrid exact exchange is the nonlocal K of the frozen core DM, not a
    density-grid ``eval_xc('pbe0')``.
    """
    from scipy.interpolate import interp1d

    n_core = max(0, int(n_core_orb))
    n_elec_target = 2.0 * float(n_core)
    coords, weights, grid_meta = build_absorber_core_becke_grid(
        mol,
        n_radial=n_radial,
        n_angular=n_angular,
        absorber_atom=absorber_atom,
    )
    rho_c, nelec, ao_deriv = core_density_on_becke(mol, dm_core, coords, weights, deriv=1)
    if abs(nelec - n_elec_target) > float(nelec_tol):
        raise RuntimeError(
            f"Becke ∫ρ_core = {nelec:.6f} e, expected {n_elec_target:.3f} "
            f"(tol {nelec_tol:g}).  The 1s is not resolved; densify the "
            f"Treutler–Ahlrichs grid before looking at ε_1s."
        )

    r_rad, rho_ray, v_h_rad, center = radial_hartree_from_core_dm(mol, dm_core)
    dist = np.linalg.norm(coords - center.reshape(1, 3), axis=1)
    v_h_fn = interp1d(
        r_rad,
        v_h_rad,
        kind="cubic",
        bounds_error=False,
        fill_value=(float(v_h_rad[0]), float(v_h_rad[-1] * r_rad[-1] / max(r_rad[-1], 1e-12))),
    )
    v_h_pts = np.asarray(v_h_fn(dist), float)
    # Coulomb tail outside the tabulated ray.
    outside = dist > r_rad[-1]
    if np.any(outside):
        v_h_pts[outside] = float(v_h_rad[-1] * r_rad[-1]) / np.maximum(dist[outside], 1e-12)

    rho_t, grad_t = rho_tot_and_grad_at_coords(driver, grid, coords)
    rho4 = np.empty_like(rho_c)
    rho4[0] = np.maximum(np.asarray(rho_c[0], float) + rho_t, 0.0)
    rho4[1:4] = np.asarray(rho_c[1:4], float) + grad_t

    ao0 = np.asarray(ao_deriv[0], float)
    v_h_ao = (ao0.T * (v_h_pts * weights)) @ ao0
    v_h_ao = 0.5 * (v_h_ao + v_h_ao.T)
    v_xc_ao = _gga_vxc_ao(ao_deriv, weights, rho4, xc)
    v_exx, exx_meta = core_exact_exchange_ao(mol, dm_core, xc)

    has_nlcc = False
    if upf_path:
        from casidapy.utils.uspp import parse_upf

        has_nlcc = bool(parse_upf(str(upf_path)).get("has_nlcc"))
        if has_nlcc:
            logger.warning(
                "UPF %s has NLCC; 1s XC is v_xc[ρ_core^AE+ρ_tot] without the "
                "model core.  QE virtuals still see NLCC.",
                upf_path,
            )

    i_nuc = int(np.argmin(dist))
    q_tail = float(v_h_rad[-1] * r_rad[-1])
    assemble = "v_H[rho_core] radial + v_xc[rho_core+rho_tot] Becke"
    if abs(exx_meta["hybrid_coeff"]) > 1e-14 or abs(exx_meta["rsh_alpha"]) > 1e-14:
        assemble += " + hyb K[rho_core]"
    meta = {
        **grid_meta,
        "rho_core_nelec": float(nelec),
        "rho_core_nelec_target": n_elec_target,
        "rho_core_nelec_err": float(nelec - n_elec_target),
        "v_h_core_rmax_times_r": q_tail,
        "v_h_core_nucleus_ha": float(v_h_pts[i_nuc]),
        "v_h_core_r0_ha": float(v_h_rad[0]),
        "rho_tot_becke_mean": float(np.mean(rho_t)),
        "rho_sum_nucleus": float(rho4[0, i_nuc]),
        "has_nlcc": bool(has_nlcc),
        "xc": str(xc),
        "hybrid": exx_meta,
        "assemble": assemble,
    }
    return np.asarray(v_h_ao + v_xc_ao + v_exx, float), meta
