"""Perturbative one-electron spin–orbit coupling (SI-SOC) for GTO TDDFT.

State-interaction (SI) SOC on top of closed-shell singlet and triplet TDA
roots. Spatial one-electron SOC integrals come from PySCF
``mol.intor('int1e_ia01p')`` (Breit–Pauli-like ``∇(1/r)×p``), scaled by
``1/(2c²)``. Two-electron SOC is omitted (adequate for qualitative demos;
raw 1e slightly overestimates screening).

Basis of the SI Hamiltonian (Cartesian triplets)::

    { S_1 … S_{n_s} } ∪ { T_1^x, T_1^y, T_1^z, …, T_{n_t}^x,y,z }

with optional ground-state slot ``S_0`` (energy 0) for phosphorescence tests.

Singlet–triplet matrix elements use the common CIS/TDA contraction::

    ⟨S_p|H_SO|T_q^u⟩ = √2 Σ_{ia,jb} X^S_{ia,p} X^T_{jb,q}
                       (δ_{ij} h^u_{ab} − δ_{ab} h^u_{ij})

with complex hermitian ``h^u = i · (1/(2c²)) · int1e_ia01p[u]`` in the MO
basis. Triplet–triplet SOC is neglected in this first implementation.

Oscillator strengths of mixed states are borrowed from the singlet
components (electric dipole from the singlet GS).

For cavity demos:

* :func:`solve_soc_qed_levels` — few-level Tavis–Cummings
  (SOC roots ⊕ |S₀,1⟩).
* :func:`solve_soc_qed_pf` — truncated Pauli–Fierz on the SI-SOC
  electronic eigenbasis ⊗ {0,1} photons (bilinear + optional DSE),
  with S₀ included so dark triplets can still feel DSE and bright
  roots form polaritons via borrowed dipoles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg

# CODATA 2018 inverse fine-structure constant (a.u. speed of light)
C_AU = 137.035999084
SQRT2 = np.sqrt(2.0)


@dataclass
class SOCResults:
    """Eigenpairs of the SI-SOC Hamiltonian."""

    omega: np.ndarray
    """Mixed excitation energies (Ha), ascending."""
    U: np.ndarray
    """Eigenvectors; columns are SI coefficients (complex)."""
    singlet_weight: np.ndarray
    """Σ |c_S|² over singlet slots (incl. S₀ if present)."""
    triplet_weight: np.ndarray
    """Σ |c_T|² over all triplet Cartesian slots."""
    f: np.ndarray
    """Length-gauge oscillator strengths from singlet-borrowed dipoles."""
    mu: np.ndarray
    """Transition dipoles from S₀, shape ``(n_states, 3)`` (complex→real part
    used for ``f``; stored as real after phase alignment)."""
    meta: Dict[str, Any] = field(default_factory=dict)


def soc_ao_integrals(mol, *, scale: Optional[float] = None) -> np.ndarray:
    """One-electron spatial SOC AO matrices ``h[k]`` (complex hermitian).

    Parameters
    ----------
    mol
        PySCF ``Mole``.
    scale
        Prefactor in front of ``int1e_ia01p``. Default ``1/(2c²)``.

    Returns
    -------
    hso : ndarray, shape (3, nao, nao), complex
        ``hso[k] = i * scale * int1e_ia01p[k]`` (hermitian).
    """
    if scale is None:
        scale = 0.5 / (C_AU ** 2)
    h_real = np.asarray(mol.intor("int1e_ia01p"), dtype=float)
    if h_real.shape[0] != 3:
        raise RuntimeError(
            f"int1e_ia01p expected 3 components, got shape {h_real.shape}"
        )
    # int1e_ia01p is real antisymmetric; i*h is hermitian.
    return (1j * float(scale)) * h_real


def soc_mo_blocks(
    hso_ao: np.ndarray,
    C_o: np.ndarray,
    C_v: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform SOC AO integrals into oo / vv / ov MO blocks.

    Returns
    -------
    h_oo, h_vv, h_ov : each (3, …) complex
    """
    hso_ao = np.asarray(hso_ao)
    C_o = np.asarray(C_o, dtype=float)
    C_v = np.asarray(C_v, dtype=float)
    h_oo = np.einsum("xuv,ui,vj->xij", hso_ao, C_o, C_o, optimize=True)
    h_vv = np.einsum("xuv,ua,vb->xab", hso_ao, C_v, C_v, optimize=True)
    h_ov = np.einsum("xuv,ui,va->xia", hso_ao, C_o, C_v, optimize=True)
    return h_oo, h_vv, h_ov


def _reshape_amps(Z: np.ndarray, n_o: int, n_v: int) -> np.ndarray:
    """``(n_trans, n_states)`` → ``(n_states, n_o, n_v)``."""
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError(f"amplitudes must be 2-D (n_trans, n_states), got {Z.shape}")
    n_trans, n_states = Z.shape
    if n_trans != n_o * n_v:
        raise ValueError(
            f"n_trans={n_trans} != n_occ*n_virt={n_o * n_v}"
        )
    return Z.T.reshape(n_states, n_o, n_v)


def st_soc_matrix(
    X_s: np.ndarray,
    X_t: np.ndarray,
    h_oo: np.ndarray,
    h_vv: np.ndarray,
) -> np.ndarray:
    """Singlet–triplet SOC blocks ``H[u, p, q] = ⟨S_p|H_SO|T_q^u⟩``.

    Parameters
    ----------
    X_s : (n_s, n_o, n_v)
    X_t : (n_t, n_o, n_v)
    h_oo : (3, n_o, n_o)
    h_vv : (3, n_v, n_v)

    Returns
    -------
    H_st : (3, n_s, n_t) complex
    """
    # term_vv[u,p,q] = Σ_{iab} X_s[p,i,a] X_t[q,i,b] h_vv[u,a,b]
    term_vv = np.einsum("pia,qib,uab->upq", X_s, X_t, h_vv, optimize=True)
    # term_oo[u,p,q] = Σ_{ija} X_s[p,i,a] X_t[q,j,a] h_oo[u,i,j]
    term_oo = np.einsum("pia,qja,uij->upq", X_s, X_t, h_oo, optimize=True)
    return SQRT2 * (term_vv - term_oo)


def s0_triplet_soc(
    X_t: np.ndarray,
    h_ov: np.ndarray,
) -> np.ndarray:
    """Ground-state–triplet SOC ``⟨S0|H_SO|T_q^u⟩``, shape ``(3, n_t)``."""
    # ⟨S0|h|T_q^u⟩ ~ √2 Σ_ia X_t[q,i,a] h_ov[u,i,a]
    return SQRT2 * np.einsum("qia,uia->uq", X_t, h_ov, optimize=True)


def build_soc_si_matrix(
    omega_s: np.ndarray,
    omega_t: np.ndarray,
    H_st: np.ndarray,
    *,
    include_ground: bool = False,
    H_s0t: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Assemble the complex Hermitian SI-SOC Hamiltonian.

    Ordering: optional S0, then singlets, then triplets as
    ``(T0x,T0y,T0z, T1x, …)``.
    """
    omega_s = np.asarray(omega_s, dtype=float).ravel()
    omega_t = np.asarray(omega_t, dtype=float).ravel()
    n_s = omega_s.size
    n_t = omega_t.size
    if H_st.shape != (3, n_s, n_t):
        raise ValueError(
            f"H_st shape {H_st.shape} incompatible with n_s={n_s}, n_t={n_t}"
        )

    n_s0 = 1 if include_ground else 0
    n_trip_slots = 3 * n_t
    n = n_s0 + n_s + n_trip_slots
    H = np.zeros((n, n), dtype=complex)

    # Diagonal energies
    if include_ground:
        H[0, 0] = 0.0
    for p, e in enumerate(omega_s):
        H[n_s0 + p, n_s0 + p] = e
    for q, e in enumerate(omega_t):
        for u in range(3):
            idx = n_s0 + n_s + 3 * q + u
            H[idx, idx] = e

    # S–T blocks
    for u in range(3):
        for p in range(n_s):
            for q in range(n_t):
                i = n_s0 + p
                j = n_s0 + n_s + 3 * q + u
                H[i, j] = H_st[u, p, q]
                H[j, i] = np.conjugate(H_st[u, p, q])

    # S0–T
    if include_ground:
        if H_s0t is None:
            raise ValueError("include_ground=True requires H_s0t")
        if H_s0t.shape != (3, n_t):
            raise ValueError(f"H_s0t shape {H_s0t.shape} != (3, {n_t})")
        for u in range(3):
            for q in range(n_t):
                j = n_s0 + n_s + 3 * q + u
                H[0, j] = H_s0t[u, q]
                H[j, 0] = np.conjugate(H_s0t[u, q])

    # Numerical hermiticity cleanup
    H = 0.5 * (H + H.conj().T)
    meta = {
        "n_s0": n_s0,
        "n_s": n_s,
        "n_t": n_t,
        "n_trip_slots": n_trip_slots,
        "include_ground": include_ground,
        "ordering": "S0?, S..., (Tx,Ty,Tz) per triplet",
    }
    return H, meta


def _phase_align_real_dipoles(mu_c: np.ndarray) -> np.ndarray:
    """Rotate each state's global phase so the dipole is as real as possible."""
    mu = np.array(mu_c, dtype=complex, copy=True)
    for k in range(mu.shape[0]):
        v = mu[k]
        nrm = np.linalg.norm(v)
        if nrm < 1e-16:
            mu[k] = 0.0
            continue
        # phase of the largest-magnitude component
        idx = int(np.argmax(np.abs(v)))
        phase = np.exp(-1j * np.angle(v[idx]))
        mu[k] = v * phase
    return np.real(mu)


def solve_soc_si(
    singlet_results,
    triplet_results,
    kernel,
    *,
    include_ground: bool = False,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    scale: Optional[float] = None,
) -> SOCResults:
    """Diagonalize SI-SOC on Casida singlet + triplet TDA results.

    Parameters
    ----------
    singlet_results, triplet_results
        :class:`~casidapy.casida_api.CasidaResults` from closed-shell TDA
        (``Z`` amplitudes required).
    kernel
        The **singlet** (or shared RKS) :class:`~casidapy.kernels.gto.GTOKernel`
        providing ``mol``, ``_C_o``, ``_C_v``.
    include_ground
        If True, add ``S0`` (E=0) coupled to triplets via ``h_ov``.
    """
    if getattr(kernel, "_spin_flip", False):
        raise ValueError("solve_soc_si expects a closed-shell GTOKernel (not SF).")
    if singlet_results.Z is None or triplet_results.Z is None:
        raise ValueError("CasidaResults.Z amplitudes are required for SI-SOC.")

    n_o, n_v = kernel.n_occ, kernel.n_unocc
    X_s = _reshape_amps(singlet_results.Z, n_o, n_v)
    X_t = _reshape_amps(triplet_results.Z, n_o, n_v)
    omega_s = np.asarray(singlet_results.omega, dtype=float).ravel()
    omega_t = np.asarray(triplet_results.omega, dtype=float).ravel()
    if X_s.shape[0] != omega_s.size or X_t.shape[0] != omega_t.size:
        raise ValueError("Amplitude / omega count mismatch in SI-SOC inputs.")

    hso_ao = soc_ao_integrals(kernel.mol, scale=scale)
    h_oo, h_vv, h_ov = soc_mo_blocks(hso_ao, kernel._C_o, kernel._C_v)
    H_st = st_soc_matrix(X_s, X_t, h_oo, h_vv)
    H_s0t = s0_triplet_soc(X_t, h_ov) if include_ground else None

    H, layout = build_soc_si_matrix(
        omega_s, omega_t, H_st,
        include_ground=include_ground, H_s0t=H_s0t,
    )
    evals, U = scipy.linalg.eigh(H)

    n_s0 = layout["n_s0"]
    n_s = layout["n_s"]
    # Weights
    w_s = np.sum(np.abs(U[n_s0:n_s0 + n_s, :]) ** 2, axis=0)
    if n_s0:
        w_s = w_s + np.abs(U[0, :]) ** 2
    w_t = np.sum(np.abs(U[n_s0 + n_s:, :]) ** 2, axis=0)

    # Singlet-borrowed transition dipoles from S0
    # μ_ia from kernel (singlet spatial); state dipole = Σ_ia Z_ia μ_ia
    # For mixed state k: μ_k = Σ_p U[S_p, k] * μ(S_p)
    mu_ia = kernel.dipole_matrix()  # (n_trans, 3); zeros if kernel is triplet
    if getattr(kernel, "triplet", False):
        # Need singlet spatial dipoles even if we were passed a triplet kernel
        raise ValueError(
            "Pass the singlet GTOKernel to solve_soc_si (for transition dipoles)."
        )
    # Singlet state dipoles: (n_s, 3)
    mu_s = (singlet_results.Z.T @ mu_ia).astype(complex)  # (n_s, 3)
    mu_mixed_c = np.zeros((evals.size, 3), dtype=complex)
    for p in range(n_s):
        mu_mixed_c += U[n_s0 + p, :, None] * mu_s[p, None, :]
    # S0 has no transition dipole to itself; S0 component does not add μ
    mu_real = _phase_align_real_dipoles(mu_mixed_c)

    # Oscillator strength f = (2/3) ω |μ|² (a.u.); skip near-zero / negative ω
    f = np.zeros_like(evals, dtype=float)
    for k, w in enumerate(evals):
        if w <= 1e-12:
            continue
        f[k] = (2.0 / 3.0) * float(w) * float(np.dot(mu_real[k], mu_real[k]))

    # Drop the ground-like root if include_ground (ω≈0)
    meta = {
        **layout,
        "scale": 0.5 / (C_AU ** 2) if scale is None else float(scale),
        "one_electron_only": True,
        "triplet_triplet_soc": False,
        "origin": list(np.asarray(origin, dtype=float).ravel()),
    }
    return SOCResults(
        omega=evals,
        U=U,
        singlet_weight=w_s,
        triplet_weight=w_t,
        f=f,
        mu=mu_real,
        meta=meta,
    )


def _select_soc_indices(
    soc: SOCResults,
    *,
    nstates: Optional[int],
    skip_ground: bool,
    prefer_bright: bool,
    lam: np.ndarray,
) -> np.ndarray:
    """Indices into ``soc`` for the truncated electronic manifold."""
    omega = np.asarray(soc.omega, dtype=float)
    mu = np.asarray(soc.mu, dtype=float)
    mask = np.ones(omega.size, dtype=bool)
    if skip_ground:
        mask &= omega > 1e-8
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError("No electronic states left after filtering.")
    if nstates is not None and idx.size > int(nstates):
        n_keep = int(nstates)
        if prefer_bright:
            strength = np.abs(mu[idx] @ lam)
            order = np.lexsort((omega[idx], -strength))
            idx = idx[order[:n_keep]]
            idx = idx[np.argsort(omega[idx])]
        else:
            idx = idx[:n_keep]
    return idx


def solve_soc_qed_levels(
    soc: SOCResults,
    lam_vec: Sequence[float],
    omega_c: float,
    *,
    nstates: Optional[int] = None,
    skip_ground: bool = True,
    prefer_bright: bool = True,
) -> Dict[str, Any]:
    """Few-level Jaynes–Cummings / Tavis–Cummings QED on SI-SOC states.

    Basis (size ``n + 1``)::

        {|φ_k, 0⟩ for selected SOC roots k}  ∪  {|S₀, 1⟩}

    with diagonal energies ``E_k`` and ``ω_c``, and couplings

        ⟨φ_k, 0| H |S₀, 1⟩ = √(ω_c / 2) (λ · μ_k)

    where ``μ_k`` is the S₀→SOC transition dipole (singlet-borrowed).

    For the truncated Pauli–Fierz treatment on the same manifold (electronic
    ⊗ {0,1} photons + DSE), use :func:`solve_soc_qed_pf`.
    """
    omega_c = float(omega_c)
    if omega_c <= 0.0:
        raise ValueError(f"omega_c must be positive, got {omega_c}")
    lam = np.asarray(lam_vec, dtype=float).ravel()
    if lam.shape != (3,):
        raise ValueError(f"lam_vec must have shape (3,), got {lam.shape}")

    idx = _select_soc_indices(
        soc,
        nstates=nstates,
        skip_ground=skip_ground,
        prefer_bright=prefer_bright,
        lam=lam,
    )
    e = np.asarray(soc.omega, dtype=float)[idx]
    d = np.asarray(soc.mu, dtype=float)[idx] @ lam
    g = np.sqrt(omega_c / 2.0) * d
    n = e.size
    M = np.zeros((n + 1, n + 1), dtype=float)
    M[:n, :n] = np.diag(e)
    M[n, n] = omega_c
    M[:n, n] = g
    M[n, :n] = g
    w, V = scipy.linalg.eigh(M)
    photon_frac = V[n, :] ** 2
    return {
        "omega": w,
        "photon_frac": photon_frac,
        "electronic_omega": e,
        "g": g,
        "omega_c": omega_c,
        "lam": lam,
        "V": V,
        "soc_indices": idx,
        "singlet_weight": soc.singlet_weight[idx],
        "triplet_weight": soc.triplet_weight[idx],
        "model": "tavis-cummings",
    }


def build_soc_qed_pf_matrix(
    energies: np.ndarray,
    d_mat: np.ndarray,
    omega_c: float,
    *,
    include_dse: bool = True,
) -> np.ndarray:
    """Pauli–Fierz matrix on electronic eigenstates ⊗ {0,1} photons.

    Parameters
    ----------
    energies
        Electronic energies ``E_a`` (length ``N``), typically ``[0, E_1, …]``
        with the SI-SOC ground at zero.
    d_mat
        Matrix of ``λ · μ`` in the same electronic basis, shape ``(N, N)``.
    omega_c
        Cavity frequency (Ha).
    include_dse
        Add ``½ (λ·μ)²`` to both the 0- and 1-photon electronic blocks.

    Returns
    -------
    M : (2N, 2N)
        Blocks::

            [[ E + DSE,   g  ],
             [ g,     E+ω+DSE ]]

        with ``g = √(ω_c/2) d_mat`` and ``DSE = ½ d_mat @ d_mat`` (or 0).
    """
    e = np.asarray(energies, dtype=float).ravel()
    d = np.asarray(d_mat, dtype=float)
    n = e.size
    if d.shape != (n, n):
        raise ValueError(f"d_mat shape {d.shape} incompatible with energies ({n},)")
    omega_c = float(omega_c)
    if omega_c <= 0.0:
        raise ValueError(f"omega_c must be positive, got {omega_c}")

    dse = 0.5 * (d @ d) if include_dse else np.zeros((n, n), dtype=float)
    g = np.sqrt(omega_c / 2.0) * d
    M = np.zeros((2 * n, 2 * n), dtype=float)
    M[:n, :n] = np.diag(e) + dse
    M[n:, n:] = np.diag(e) + omega_c * np.eye(n) + dse
    M[:n, n:] = g
    M[n:, :n] = g
    return M


def solve_soc_qed_pf(
    soc: SOCResults,
    lam_vec: Sequence[float],
    omega_c: float,
    *,
    nstates: Optional[int] = None,
    skip_ground: bool = True,
    prefer_bright: bool = False,
    include_ground_slot: bool = True,
    include_dse: bool = True,
    mu0_lam: float = 0.0,
) -> Dict[str, Any]:
    """Truncated Pauli–Fierz QED on the SI-SOC electronic eigenbasis.

    Electronic basis (length ``N``):

    * if ``include_ground_slot``: ``{|S₀⟩, |φ_k⟩}`` with ``E₀ = 0`` and
      selected SOC roots ``|φ_k⟩``;
    * else: selected SOC roots only (then ``d`` has no S₀ couplings unless
      excited–excited dipoles are supplied — not the default).

    Dipole matrix (default, S₀-transition / TDA-like)::

        (λ·μ)_{0k} = (λ·μ)_{k0} = λ · μ_k
        (λ·μ)_{00} = mu0_lam   (optional permanent / CS shift)
        (λ·μ)_{kj} = 0         (k,j both excited; no state–state μ yet)

    Light–matter Hilbert space: electronic ⊗ {0,1} photons (size ``2N``).
    Hamiltonian::

        H = H_el + ω_c a†a + √(ω_c/2) (λ·μ)(a+a†) + ½ (λ·μ)²

    with the last term optional (``include_dse``).

    Returned ``omega`` values are **excitation energies relative to the
    PF ground** (``w - w[0]``), so they plug into ``E_SCF + ω`` PES plots
    the same way as TDA roots.

    Oscillator strengths ``f`` are probe absorption strengths from the PF
    ground using the electronic dipole (``μ ⊗ I`` on the photon Fock space).
    Pure photonic / dark roots stay dark; polaritons borrow intensity from
    the parent bright SOC states — do **not** use ``photon_frac`` as a
    spectrum intensity.
    """
    omega_c = float(omega_c)
    if omega_c <= 0.0:
        raise ValueError(f"omega_c must be positive, got {omega_c}")
    lam = np.asarray(lam_vec, dtype=float).ravel()
    if lam.shape != (3,):
        raise ValueError(f"lam_vec must have shape (3,), got {lam.shape}")

    idx = _select_soc_indices(
        soc,
        nstates=nstates,
        skip_ground=skip_ground,
        prefer_bright=prefer_bright,
        lam=lam,
    )
    e_exc = np.asarray(soc.omega, dtype=float)[idx]
    mu_exc = np.asarray(soc.mu, dtype=float)[idx]
    d_ov = mu_exc @ lam  # (n_exc,)

    if include_ground_slot:
        e = np.concatenate([[0.0], e_exc])
        n = e.size
        d = np.zeros((n, n), dtype=float)
        d[0, 0] = float(mu0_lam)
        d[0, 1:] = d_ov
        d[1:, 0] = d_ov
    else:
        e = e_exc
        n = e.size
        d = np.zeros((n, n), dtype=float)
        # Without S₀ there is no bilinear coupling from GS transition dipoles.
        # Keep matrix well-defined; caller should pass include_ground_slot=True.

    M = build_soc_qed_pf_matrix(e, d, omega_c, include_dse=include_dse)
    w_abs, V = scipy.linalg.eigh(M)
    w = w_abs - w_abs[0]
    photon_frac = np.sum(V[n:, :] ** 2, axis=0)

    # Electronic dipole tensor in the truncated basis (TDA-like: only S₀↔exc).
    # Probe absorption uses μ ⊗ I_photon, so both 0- and 1-photon blocks contribute:
    #   μ_α = ⟨G|μ|Ψ_α⟩ with G = PF ground (column 0 of V).
    mu_el = np.zeros((n, n, 3), dtype=float)
    if include_ground_slot:
        mu_el[0, 1:, :] = mu_exc
        mu_el[1:, 0, :] = mu_exc
    mu_pol = (
        np.einsum("i,ijx,jk->kx", V[:n, 0], mu_el, V[:n, :], optimize=True)
        + np.einsum("i,ijx,jk->kx", V[n:, 0], mu_el, V[n:, :], optimize=True)
    )
    # f = (2/3) ω |μ|²; ground→ground is zero. Pure |S₀,1⟩ / dark roots stay dark.
    f = np.zeros(w.size, dtype=float)
    for k, wk in enumerate(w):
        if wk <= 1e-12:
            continue
        f[k] = (2.0 / 3.0) * float(wk) * float(np.dot(mu_pol[k], mu_pol[k]))

    return {
        "omega": w,
        "omega_absolute": w_abs,
        "photon_frac": photon_frac,
        "f": f,
        "mu": mu_pol,
        "electronic_omega": e,
        "d_mat": d,
        "omega_c": omega_c,
        "lam": lam,
        "V": V,
        "M": M,
        "soc_indices": idx,
        "singlet_weight": (
            np.concatenate([[1.0], soc.singlet_weight[idx]])
            if include_ground_slot
            else soc.singlet_weight[idx]
        ),
        "triplet_weight": (
            np.concatenate([[0.0], soc.triplet_weight[idx]])
            if include_ground_slot
            else soc.triplet_weight[idx]
        ),
        "include_dse": bool(include_dse),
        "include_ground_slot": bool(include_ground_slot),
        "model": "pauli-fierz",
    }
