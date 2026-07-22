"""Pauli–Fierz QED-TDDFT (TDA) on top of the GTO electronic kernel.

Length gauge, dipole approximation, single cavity mode, coherent-state
(relaxed-dipole) reference, TDA only. The electronic coupling ``K`` is
consumed as-is from :class:`~casidapy.kernels.gto.GTOKernel` — this module
does not modify ``apply_K`` or the Casida algebra.

This is the physical (DSE-inclusive) route. The phenomenological
Tavis–Cummings post-processing in :mod:`casidapy.polariton_handler` is a
separate, cheaper path; the two are not expected to agree except in the
weak-coupling / DSE-negligible limit.

Ground state: ordinary converged PySCF RKS/RHF (Level A — dipole self-energy
enters the response only). Full QED-SCF is out of scope.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg

# ---------------------------------------------------------------------------
# Convention constants (single edit site if DePrince/Foley vs Flick/Rubio
# placement of closed-shell singlet factors differs). Unverified until a
# published QED-TDA number is matched — see module tests / README.
# ---------------------------------------------------------------------------
SQRT2 = np.sqrt(2.0)
DSE_DIRECT_FACTOR = 2.0

# permanent_dipole electronic part = Tr(D, r). Nuclear term uses
# NUCLEAR_DIPOLE_SIGN (physical a.u. dipole is usually el + (−1)*nuc).
# For the coherent-state oo/vv shift we use the *electronic* permanent
# dipole only — including nuclei makes ⟨μ⟩₀ gauge-invariant and then the
# diagonal shift no longer cancels the origin drift of Q_oo/Q_vv.
NUCLEAR_DIPOLE_SIGN = -1


@dataclass
class QEDOptions:
    """Physical cavity inputs for Pauli–Fierz TDA-QED (not CasidaOptions)."""

    lam_scalar: float
    polarization: Sequence[float]
    omega_c: float
    origin: Sequence[float] = (0.0, 0.0, 0.0)
    include_dse: bool = True
    coherent_state: bool = True
    # Nuclear dipole is for reporting / photonic displacement diagnostics.
    # The CS shift inside dipole_blocks uses electronic ⟨μ⟩₀ only (see there).
    include_nuclear: bool = False
    nstates: Optional[int] = None

    def lam_vec(self) -> np.ndarray:
        e = np.asarray(self.polarization, dtype=float).ravel()
        nrm = float(np.linalg.norm(e))
        if nrm < 1e-15:
            raise ValueError("polarization must be a non-zero 3-vector")
        return float(self.lam_scalar) * (e / nrm)


@dataclass
class QEDResults:
    """Eigenpairs of the dense TDA-QED matrix (electronic singles ⊕ photon)."""

    omega: np.ndarray
    X: np.ndarray
    m: np.ndarray
    f: np.ndarray
    photon_frac: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)


def permanent_dipole(
    kernel,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    include_nuclear: bool = True,
    nuclear_sign: Optional[float] = None,
) -> np.ndarray:
    """Ground-state permanent dipole ``⟨μ⟩₀`` in the same sign convention as ``q_pq``.

    Electronic part is ``Tr(D, r)`` matching ``mol.intor('int1e_r')`` (same as
    :meth:`GTOKernel.dipole_matrix`). Nuclear contribution uses
    ``NUCLEAR_DIPOLE_SIGN`` (or ``nuclear_sign`` override) so the origin-
    invariance test can resolve the relative sign empirically.
    """
    mol = kernel.mol
    orig = np.asarray(origin, dtype=float).ravel()
    mf = getattr(kernel, "_mf", None)
    if mf is not None and hasattr(mf, "make_rdm1"):
        dm = np.asarray(mf.make_rdm1(), dtype=float)
    else:
        # Closed-shell: D = C diag(n) Cᵀ
        dm = (
            kernel.mo_coeff
            * np.asarray(kernel.mo_occ, dtype=float)[None, :]
        ) @ kernel.mo_coeff.T

    with mol.with_common_orig(tuple(orig)):
        dip_ao = mol.intor("int1e_r", comp=3)  # (3, nao, nao)
    el = np.einsum("xpq,qp->x", dip_ao, dm)
    if not include_nuclear:
        return np.asarray(el, dtype=float)

    sign = NUCLEAR_DIPOLE_SIGN if nuclear_sign is None else float(nuclear_sign)
    charges = mol.atom_charges().astype(float)
    coords = mol.atom_coords() - orig
    nuc = np.einsum("A,Ax->x", charges, coords)
    return np.asarray(el + sign * nuc, dtype=float)


def dipole_blocks(
    kernel,
    lam_vec: Sequence[float],
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    include_nuclear: bool = False,
    coherent_state: bool = True,
    nuclear_sign: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """λ-contracted dipole blocks ``(q_ia, Q_oo, Q_vv)``, coherent-state shifted.

    Flattening of ``q_ia`` is ``ia = i * n_unocc + a`` (matches GTOKernel).

    The coherent-state diagonal shift uses the electronic permanent dipole
    ``Tr(D, r)`` divided by ``N_e``. Nuclei are omitted here on purpose: a
    gauge-invariant full ⟨μ⟩₀ would not cancel the origin drift of the oo/vv
    diagonals (verified by the origin-invariance tests).
    """
    mol, C_o, C_v = kernel.mol, kernel._C_o, kernel._C_v
    lam = np.asarray(lam_vec, dtype=float).ravel()
    if lam.shape != (3,):
        raise ValueError(f"lam_vec must have shape (3,), got {lam.shape}")
    orig = tuple(np.asarray(origin, dtype=float).ravel())

    with mol.with_common_orig(orig):
        dip_ao = mol.intor("int1e_r", comp=3)

    # MO blocks then contract with λ → scalars per orbital pair
    q_ov = np.einsum("x,xpq,pi,qa->ia", lam, dip_ao, C_o, C_v, optimize=True)
    Q_oo = np.einsum("x,xpq,pi,qj->ij", lam, dip_ao, C_o, C_o, optimize=True)
    Q_vv = np.einsum("x,xpq,pa,qb->ab", lam, dip_ao, C_v, C_v, optimize=True)

    if coherent_state:
        # Always electronic for the CS orbital shift (ignore include_nuclear).
        mu0_el = permanent_dipole(
            kernel,
            origin=orig,
            include_nuclear=False,
            nuclear_sign=nuclear_sign,
        )
        nelec = float(getattr(mol, "nelectron", 0) or 0)
        if nelec < 1e-12:
            nelec = float(np.sum(np.asarray(kernel.mo_occ, dtype=float)))
        shift = float(np.dot(lam, mu0_el)) / nelec
        Q_oo = Q_oo - shift * np.eye(Q_oo.shape[0])
        Q_vv = Q_vv - shift * np.eye(Q_vv.shape[0])

    return q_ov.ravel(), Q_oo, Q_vv


def dse_exchange_matrix(Q_oo: np.ndarray, Q_vv: np.ndarray) -> np.ndarray:
    """DSE exchange block ``A_exch[ia,jb] = Q_oo[i,j] * Q_vv[a,b]``.

    Explicit ``einsum`` (not ``kron``) so the ``ia = i*n_v + a`` layout is
    unambiguous.
    """
    n_o, n_v = Q_oo.shape[0], Q_vv.shape[0]
    return np.einsum("ij,ab->iajb", Q_oo, Q_vv, optimize=True).reshape(
        n_o * n_v, n_o * n_v
    )


def build_qed_tda_matrix(
    kernel,
    lam_vec: Sequence[float],
    omega_c: float,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    *,
    include_dse: bool = True,
    coherent_state: bool = True,
    include_nuclear: bool = False,
    nuclear_sign: Optional[float] = None,
) -> np.ndarray:
    """Dense TDA-QED matrix ``M`` of shape ``(n_trans+1, n_trans+1)``."""
    if not getattr(kernel, "_ready", False):
        kernel.setup(tda=True)
    if not getattr(kernel, "tda", True):
        raise NotImplementedError("QED-TDDFT in this module is TDA-only.")

    n_o, n_v = kernel.n_occ, kernel.n_unocc
    n = kernel.n_trans
    omega_c = float(omega_c)
    if omega_c <= 0.0:
        raise ValueError(f"omega_c must be positive, got {omega_c}")

    K = kernel._K if getattr(kernel, "_K", None) is not None else kernel.dense_K_rows(
        range(n)
    )
    dE = kernel.diagonal_dE()

    q, Q_oo, Q_vv = dipole_blocks(
        kernel,
        lam_vec,
        origin=origin,
        include_nuclear=include_nuclear,
        coherent_state=coherent_state,
        nuclear_sign=nuclear_sign,
    )

    A = np.array(K, dtype=float, copy=True)
    A[np.diag_indices(n)] += dE

    if include_dse:
        A += DSE_DIRECT_FACTOR * np.outer(q, q)
        A -= dse_exchange_matrix(Q_oo, Q_vv)

    g = np.sqrt(omega_c / 2.0) * SQRT2 * q

    M = np.zeros((n + 1, n + 1), dtype=float)
    M[:n, :n] = A
    M[:n, n] = g
    M[n, :n] = g
    M[n, n] = omega_c
    return M


def solve_qed_tda(
    kernel,
    lam_vec: Optional[Sequence[float]] = None,
    omega_c: Optional[float] = None,
    nstates: Optional[int] = None,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    *,
    include_dse: bool = True,
    coherent_state: bool = True,
    include_nuclear: bool = False,
    nuclear_sign: Optional[float] = None,
    options: Optional[QEDOptions] = None,
) -> QEDResults:
    """Diagonalize the dense TDA-QED matrix and return polariton observables."""
    if options is not None:
        lam_vec = options.lam_vec()
        omega_c = options.omega_c
        origin = options.origin
        include_dse = options.include_dse
        coherent_state = options.coherent_state
        include_nuclear = options.include_nuclear
        if nstates is None:
            nstates = options.nstates
    if lam_vec is None or omega_c is None:
        raise ValueError("lam_vec and omega_c are required (or pass options=QEDOptions(...))")

    M = build_qed_tda_matrix(
        kernel,
        lam_vec,
        omega_c,
        origin=origin,
        include_dse=include_dse,
        coherent_state=coherent_state,
        include_nuclear=include_nuclear,
        nuclear_sign=nuclear_sign,
    )
    w, V = scipy.linalg.eigh(M)
    n = kernel.n_trans
    X, m = V[:n, :], V[n, :]

    mu = kernel.dipole_matrix()  # (n_trans, 3) unprojected
    d = X.T @ mu  # (n+1, 3)
    f = (2.0 / 3.0) * w * np.einsum("nx,nx->n", d, d)

    if nstates is not None:
        nstates = min(int(nstates), w.shape[0])
        sl = slice(0, nstates)
        w, X, m, f = w[sl], X[:, sl], m[sl], f[sl]

    return QEDResults(
        omega=w,
        X=X,
        m=m,
        f=f,
        photon_frac=np.abs(m) ** 2,
        meta={
            "omega_c": float(omega_c),
            "lam": np.asarray(lam_vec, dtype=float).tolist(),
            "n_trans": n,
            "tda": True,
            "include_dse": bool(include_dse),
            "coherent_state": bool(coherent_state),
            "origin": list(np.asarray(origin, dtype=float).ravel()),
            "nuclear_dipole_sign": (
                NUCLEAR_DIPOLE_SIGN if nuclear_sign is None else float(nuclear_sign)
            ),
        },
    )


def _state_overlap(X_ref: np.ndarray, m_ref: np.ndarray, X: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Overlap matrix between two QED eigenbases: ``S[a,b] = X_ref[:,a]·X[:,b] + m_ref[a]*m[b]``."""
    return X_ref.T @ X + np.outer(m_ref, m)


def _match_states(S_abs: np.ndarray) -> np.ndarray:
    """Optimal bipartite matching maximizing Σ |⟨ψ_a|ψ_b⟩|.

    Returns ``perm`` such that new branch ``a`` ↔ energy-sorted column ``perm[a]``.
    Uses the Hungarian algorithm (avoids greedy collisions at avoided crossings).
    """
    from scipy.optimize import linear_sum_assignment

    # linear_sum_assignment minimizes cost → maximize overlap via negation
    row, col = linear_sum_assignment(-S_abs)
    perm = np.empty(S_abs.shape[0], dtype=int)
    perm[row] = col
    return perm


def track_states(
    results_list: Sequence[QEDResults],
) -> Tuple[np.ndarray, np.ndarray]:
    """Overlap-based branch tracking along a scan.

    At each step, branches are matched to the previous point by maximizing the
    total absolute eigenvector overlap (Hungarian assignment), then signs are
    aligned so the next overlap stays continuous. This prevents the zig-zag
    swaps that greedy matching produces at polariton avoided crossings.

    Returns
    -------
    omega_tracked : (n_pts, n_states)
    photon_tracked : (n_pts, n_states)
    """
    if not results_list:
        raise ValueError("results_list is empty")
    n_pts = len(results_list)
    n_states = results_list[0].omega.shape[0]
    omega_t = np.empty((n_pts, n_states), dtype=float)
    phot_t = np.empty((n_pts, n_states), dtype=float)

    omega_t[0] = results_list[0].omega
    phot_t[0] = results_list[0].photon_frac
    X_prev = np.array(results_list[0].X, dtype=float, copy=True)
    m_prev = np.array(results_list[0].m, dtype=float, copy=True)

    for i in range(1, n_pts):
        r = results_list[i]
        if r.omega.shape[0] != n_states:
            raise ValueError(
                f"Inconsistent nstates along scan: point 0 has {n_states}, "
                f"point {i} has {r.omega.shape[0]}"
            )
        S = _state_overlap(X_prev, m_prev, r.X, r.m)
        perm = _match_states(np.abs(S))

        X_new = r.X[:, perm]
        m_new = r.m[perm]
        # Align eigenvector phase to previous point (overlap → positive)
        signs = np.sign(np.einsum("ik,ik->k", X_prev, X_new) + m_prev * m_new)
        signs[signs == 0.0] = 1.0
        X_new = X_new * signs
        m_new = m_new * signs

        omega_t[i] = r.omega[perm]
        phot_t[i] = r.photon_frac[perm]
        X_prev, m_prev = X_new, m_new

    return omega_t, phot_t


def scan_qed_tda(
    kernels: Sequence[Any],
    lam_vec: Sequence[float],
    omega_c: float,
    *,
    nstates: Optional[int] = None,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    include_dse: bool = True,
    coherent_state: bool = True,
    track: bool = True,
) -> Dict[str, Any]:
    """Solve TDA-QED at each geometry/kernel and optionally track polariton branches.

    Parameters
    ----------
    kernels
        Sequence of ready (or setup-able) :class:`GTOKernel` instances, one per
        geometry along a coordinate.
    """
    results = [
        solve_qed_tda(
            k,
            lam_vec,
            omega_c,
            nstates=nstates,
            origin=origin,
            include_dse=include_dse,
            coherent_state=coherent_state,
        )
        for k in kernels
    ]
    out: Dict[str, Any] = {"results": results}
    if track and results:
        omega_t, phot_t = track_states(results)
        out["omega_tracked"] = omega_t
        out["photon_frac_tracked"] = phot_t
    return out


def scan_qed_lambda(
    kernel,
    lam_scalars: Sequence[float],
    polarization: Sequence[float],
    omega_c: float,
    *,
    nstates: Optional[int] = None,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    include_dse: bool = True,
    coherent_state: bool = True,
    track: bool = True,
) -> Dict[str, Any]:
    """λ-sweep at **fixed geometry** — electronic ``K`` built once, then reused.

    Each cavity strength only re-assembles the ``(n_trans+1)`` QED matrix
    (dipole blocks scale with λ; no new AO integrals). Prefer this over
    rebuilding the GTOKernel per λ. For independent geometry points, run
    separate processes / a job array instead of intra-solve MPI.
    """
    if not getattr(kernel, "_ready", False):
        kernel.setup(tda=True)

    e = np.asarray(polarization, dtype=float).ravel()
    nrm = float(np.linalg.norm(e))
    if nrm < 1e-15:
        raise ValueError("polarization must be a non-zero 3-vector")
    e = e / nrm

    results = [
        solve_qed_tda(
            kernel,
            float(lam) * e,
            omega_c,
            nstates=nstates,
            origin=origin,
            include_dse=include_dse,
            coherent_state=coherent_state,
        )
        for lam in lam_scalars
    ]
    out: Dict[str, Any] = {
        "results": results,
        "lam_scalars": np.asarray(lam_scalars, dtype=float),
        "polarization": e,
        "omega_c": float(omega_c),
    }
    if track and results:
        omega_t, phot_t = track_states(results)
        out["omega_tracked"] = omega_t
        out["photon_frac_tracked"] = phot_t
    return out
