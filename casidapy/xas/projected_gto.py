"""GIPAW virtuals → fragment GTO: hybrid ``(ij|ab)`` and 3d-weight TDA window.

The same projected ``C_v`` (not S-normalized) feeds two TDA knobs:

* ``casida_exx=gto`` — ``−c_x (ij|ab)`` in CVS-TDA ``A`` with the functional
  hybrid coefficient (honest hybrid kernel; not FFT collocation).
* ``weight_virtuals`` — keep QE conduction bands whose GTO weight on the
  absorber particle shell (Ti 3d, …) is large, then run ordinary CVS-TDA /
  SI-SOC in that reduced virtual window.  No SVD and no separate CIS.

``run_particle_ci`` remains a library helper (atomic-limit 2p⁵ nd¹ CIS).
Production L-edge stays in the TDA regime.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from casidapy.xas.chemical_data import (
    DEFAULT_OXIDATION_STATES,
    d_electron_count,
    valence_particle_nl,
)

logger = logging.getLogger(__name__)


def mo_ijab(mol, C_o: np.ndarray, C_v: np.ndarray) -> np.ndarray:
    """Chemist ``(ij|ab)`` for occupied ``C_o`` and virtual ``C_v``.

    Shape ``(n_o, n_o, n_v, n_v)``.  Columns of ``C_v`` are used as given
    (no ``S``-normalization — completeness is the 3d weight).
    """
    from pyscf import ao2mo

    C_o = np.asarray(C_o, float)
    C_v = np.asarray(C_v, float)
    if C_o.ndim != 2 or C_v.ndim != 2:
        raise ValueError("C_o and C_v must be 2-D AO coefficient matrices")
    n_o = int(C_o.shape[1])
    n_v = int(C_v.shape[1])
    if n_o == 0 or n_v == 0:
        return np.zeros((n_o, n_o, n_v, n_v), dtype=float)
    eri = ao2mo.general(mol, (C_o, C_o, C_v, C_v), compact=False)
    return np.asarray(eri, dtype=float).reshape(n_o, n_o, n_v, n_v)


def mo_ovov(mol, C_o: np.ndarray, C_v: np.ndarray) -> np.ndarray:
    """Chemist ``(ia|jb)`` shape ``(n_o, n_v, n_o, n_v)``."""
    from pyscf import ao2mo

    C_o = np.asarray(C_o, float)
    C_v = np.asarray(C_v, float)
    n_o = int(C_o.shape[1])
    n_v = int(C_v.shape[1])
    if n_o == 0 or n_v == 0:
        return np.zeros((n_o, n_v, n_o, n_v), dtype=float)
    eri = ao2mo.general(mol, (C_o, C_v, C_o, C_v), compact=False)
    return np.asarray(eri, dtype=float).reshape(n_o, n_v, n_o, n_v)


def hybrid_mo_ijab(
    mol,
    C_o: np.ndarray,
    C_v: np.ndarray,
    *,
    hybrid_coeff: float,
    rsh_omega: float = 0.0,
    rsh_alpha: float = 0.0,
) -> np.ndarray:
    """``c_x (ij|ab) + (α − c_x) (ij|ab)_LR(ω)`` already scaled for the kernel."""
    hyb = float(hybrid_coeff)
    omega = float(rsh_omega)
    alpha = float(rsh_alpha)
    eri = mo_ijab(mol, C_o, C_v)
    out = hyb * eri
    if abs(omega) > 1e-14 and abs(alpha - hyb) > 1e-14:
        with mol.with_range_coulomb(omega):
            eri_lr = mo_ijab(mol, C_o, C_v)
        out = out + (alpha - hyb) * eri_lr
    return out


def ijab_matvec(ijab: np.ndarray, V: np.ndarray) -> np.ndarray:
    """``(K_x V)_{ia} = Σ_{jb} (ij|ab) V_{jb}``; subtract from singlet/triplet ``Kv``."""
    V = np.asarray(V, float)
    return np.einsum("ijab,jb->ia", np.asarray(ijab, float), V, optimize=True)


def virtual_completeness(C_v: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Per-virtual ``‖c‖_S²`` (do not rescale to 1)."""
    C = np.asarray(C_v, float)
    S = np.asarray(S, float)
    return np.einsum("pi,pq,qi->i", C, S, C)


def project_kernel_virtuals_to_gto(
    core,
    kernel,
    *,
    absorber_atom: Optional[int] = 0,
    projection_method: str = "becke",
    becke_level: int = 4,
    driver=None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Project GIPAW-reconstructed kernel virtuals onto ``core.fragment_mol``.

    Uses ``kernel._psi_unocc`` (AE / GIPAW), not ``_psi_unocc_tilde``.
    Returns ``(C_v, meta)`` with ``C_v = S^{-1} V`` **not** S-normalized.
    """
    from casidapy.xas.xas_utils import (
        grid_projection_to_ao_coeffs,
        project_grid_orbitals_to_fragment_ao,
    )

    mol = core.fragment_mol
    grid = kernel.grid
    virts = list(kernel._psi_unocc)
    V_ao = project_grid_orbitals_to_fragment_ao(
        mol,
        grid,
        virts,
        method=projection_method,
        becke_level=int(becke_level),
        absorber_atom=absorber_atom,
        driver=driver,
    )
    C_v = grid_projection_to_ao_coeffs(mol, V_ao)
    S = np.asarray(mol.intor("int1e_ovlp"), dtype=float)
    comp = virtual_completeness(C_v, S)
    meta = {
        "n_virt": int(C_v.shape[1]),
        "nao": int(C_v.shape[0]),
        "completeness_mean": float(np.mean(comp)) if comp.size else 0.0,
        "completeness_max": float(np.max(comp)) if comp.size else 0.0,
        "s_normalized": False,
        "psi": "gipaw_reconstructed",
        "projection_method": str(projection_method),
    }
    return C_v, meta


def virtual_shell_weights(
    mol,
    C_v: np.ndarray,
    *,
    n: int,
    ell: int,
    absorber_atom: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Per-virtual overlap with absorber ``nl`` AOs (no SVD, no rotation).

    ``w_a = c_a^T S_{:,μ} S_{μμ}^{-1} S_{μ,:} c_a`` for ``μ`` in the shell.
    Columns of ``C_v`` are used as given (not S-normalized).  A pure
    Löwdin ``nl`` orbital scores ~1; ligand-centered CBs score near 0.
    """
    from casidapy.xas.cvs import _ao_mask_for_atoms, _ao_shell_mask

    C_v = np.asarray(C_v, float)
    if C_v.ndim != 2:
        raise ValueError("C_v must be 2-D (nao, n_virt)")
    shell = _ao_shell_mask(mol, int(n), int(ell))
    mask = _ao_mask_for_atoms(mol, [int(absorber_atom)], shell)
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise RuntimeError(
            f"No n={n}, l={ell} AOs on absorber atom {absorber_atom}; "
            "use an AE basis with that shell (e.g. aug-cc-pvdz)."
        )
    S = np.asarray(mol.intor("int1e_ovlp"), dtype=float)
    S_ss = S[np.ix_(idx, idx)]
    w_ss, U_ss = np.linalg.eigh(S_ss)
    w_ss = np.maximum(w_ss, 1e-12)
    S_ss_inv = (U_ss / w_ss) @ U_ss.T
    S_vs = C_v.T @ S[:, idx]
    w = np.einsum("ai,ij,aj->a", S_vs, S_ss_inv, S_vs, optimize=True)
    w = np.maximum(np.asarray(w, float).ravel(), 0.0)
    label = {0: "s", 1: "p", 2: "d", 3: "f"}.get(int(ell), str(int(ell)))
    meta = {
        "n": int(n),
        "ell": int(ell),
        "shell": f"{int(n)}{label}",
        "n_ao": int(idx.size),
        "n_virt": int(C_v.shape[1]),
        "weight_max": float(np.max(w)) if w.size else 0.0,
        "weight_mean": float(np.mean(w)) if w.size else 0.0,
    }
    return w, meta


def select_indices_by_score_capture(
    scores: np.ndarray,
    *,
    min_capture: float = 0.8,
    min_weight: float = 0.0,
    n_min: int = 1,
    mask: Optional[np.ndarray] = None,
    label: str = "score",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Rank by ``scores`` and keep until cumulative capture of the eligible sum.

    ``mask`` (optional bool, length ``n_virt``) restricts the eligible pool
    (e.g. an energy window) before capture.  ``min_weight`` is a relative
    floor ``w ≥ min_weight · max(w)`` on the captured set (0 = off).
    """
    w = np.asarray(scores, dtype=float).ravel()
    if w.size == 0:
        raise ValueError(f"select_indices_by_score_capture({label}): empty scores")
    if mask is None:
        eligible = np.arange(w.size, dtype=int)
    else:
        m = np.asarray(mask, dtype=bool).ravel()
        if m.size != w.size:
            raise ValueError(
                f"mask length {m.size} != scores length {w.size} ({label})"
            )
        eligible = np.flatnonzero(m)
        if eligible.size == 0:
            raise RuntimeError(
                f"No virtuals left in {label} window (mask empty)."
            )
    we = w[eligible]
    total = float(np.sum(we))
    wmax = float(np.max(we)) if we.size else 0.0
    if wmax < 1e-14 or total < 1e-14:
        raise RuntimeError(
            f"No {label} weight in the virtual window (max={wmax:.3e})."
        )
    order_loc = np.argsort(we)[::-1]
    csum = np.cumsum(we[order_loc])
    cap = min(max(float(min_capture), 0.0), 1.0)
    n_cap = int(np.searchsorted(csum, cap * total, side="left") + 1)
    n_floor = max(int(n_min), 1)
    n_keep = max(n_floor, n_cap, 1)
    n_keep = min(n_keep, int(order_loc.size))
    keep = np.sort(eligible[order_loc[:n_keep]].astype(int))
    rel = float(min_weight)
    if rel > 0.0:
        kept = keep[w[keep] >= rel * wmax]
        if kept.size >= n_floor:
            keep = kept
    captured = float(np.sum(w[keep]) / total) if total > 0 else 0.0
    top_n = int(min(15, order_loc.size))
    meta = {
        "label": str(label),
        "min_weight": rel,
        "min_capture": cap,
        "n_min": n_floor,
        "n_virt": int(w.size),
        "n_eligible": int(eligible.size),
        "n_keep": int(keep.size),
        "idx": [int(i) for i in keep],
        "weights_keep": [float(w[i]) for i in keep],
        "weight_mean_keep": float(np.mean(w[keep])),
        "weight_sum": total,
        "captured": captured,
        "weights_top": [float(we[i]) for i in order_loc[:top_n]],
        "idx_top": [int(eligible[i]) for i in order_loc[:top_n]],
    }
    return keep, meta


def select_virtuals_by_shell_weight(
    mol,
    C_v: np.ndarray,
    *,
    n: int,
    ell: int,
    absorber_atom: int = 0,
    min_weight: float = 0.0,
    min_capture: float = 0.8,
    n_min: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Keep the CBs that carry the absorber ``nl`` weight (energy order).

    Hybridized 3d is spread over many bands: an absolute per-CB cutoff
    (e.g. 0.2) can keep one orbital or none.  Rank by weight and keep
    until the running sum is ``min_capture`` of ``Σ_a w_a``, and at
    least ``n_min`` (default ``2ℓ+1``).  ``min_weight`` is a relative
    floor ``w ≥ min_weight · max(w)`` applied to that set (0 = off).
    """
    w, meta = virtual_shell_weights(
        mol, C_v, n=int(n), ell=int(ell), absorber_atom=int(absorber_atom)
    )
    total = float(np.sum(w))
    wmax = float(np.max(w)) if w.size else 0.0
    if w.size == 0 or wmax < 1e-8 or total < 1e-8:
        raise RuntimeError(
            f"No {meta['shell']} character in the virtual window "
            f"(max={wmax:.3e}). Conduction bands missed the particle shell."
        )
    n_floor = int(n_min) if n_min is not None else int(2 * ell + 1)
    keep, cap_meta = select_indices_by_score_capture(
        w,
        min_capture=float(min_capture),
        min_weight=float(min_weight),
        n_min=n_floor,
        label=str(meta.get("shell", "shell")),
    )
    return keep, w, {**meta, **cap_meta, "shell": meta.get("shell")}


def subset_pw_kernel_virtuals(kernel, idx) -> Any:
    """Restrict a PW kernel to virtuals ``idx`` (GIPAW ψ and tilde together)."""
    idx = np.asarray(idx, int).ravel()
    if idx.size == 0:
        raise ValueError("subset_pw_kernel_virtuals: empty virtual index")
    n_v = len(kernel._psi_unocc)
    if int(np.max(idx)) >= n_v or int(np.min(idx)) < 0:
        raise IndexError(
            f"virtual index {idx.tolist()} out of range for n_unocc={n_v}"
        )
    e = np.asarray(kernel._unocc_e, float).ravel()[idx]
    psi = [kernel._psi_unocc[int(i)] for i in idx]
    tilde = getattr(kernel, "_psi_unocc_tilde", None)
    kernel.set_active_orbitals(kernel._occ_e, e, list(kernel._psi_occ), psi)
    if tilde is not None:
        kernel._psi_unocc_tilde = [tilde[int(i)] for i in idx]
    bands = getattr(kernel, "_qe_band_indices_unocc", None)
    if bands is not None:
        kernel._qe_band_indices_unocc = [bands[int(i)] for i in idx]
    return kernel


def svd_particle_orbitals(
    mol,
    C_v: np.ndarray,
    eps_v: np.ndarray,
    *,
    n: int,
    ell: int,
    absorber_atom: int = 0,
    n_keep: Optional[int] = None,
    min_complete: float = 0.2,
    window_ev: float = 0.0,
    ha_to_ev: float = 27.211386245988,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """SVD projected virtuals onto absorber ``nl`` AOs; keep ``2ℓ+1``.

    One-electron energies are the PW ``ε_v`` Hamiltonian in that subspace.
    Columns are S-orthonormal (Löwdin in the small space only).

    ``window_ev > 0`` restricts the SVD pool to virtuals with
    ``ε ∈ [ε_CBM, ε_CBM + window]`` (K-edge white-line: avoid high-lying
    continuum bands that still carry ``nl`` AO weight).
    """
    from casidapy.xas.cvs import (
        _ao_mask_for_atoms,
        _ao_shell_mask,
        _lowdin_orthonormalize,
    )

    C_v = np.asarray(C_v, float)
    eps_v = np.asarray(eps_v, float).ravel()
    if C_v.shape[1] != eps_v.size:
        raise ValueError(
            f"C_v has {C_v.shape[1]} columns, eps_v has {eps_v.size}"
        )
    n_keep = int(n_keep) if n_keep is not None else int(2 * ell + 1)
    shell = _ao_shell_mask(mol, int(n), int(ell))
    mask = _ao_mask_for_atoms(mol, [int(absorber_atom)], shell)
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise RuntimeError(
            f"No n={n}, l={ell} AOs on absorber atom {absorber_atom}; "
            "use an AE basis with that shell (e.g. aug-cc-pvdz)."
        )
    pool = np.arange(eps_v.size, dtype=int)
    e_cbm = float(np.min(eps_v))
    w_ev = float(window_ev or 0.0)
    if w_ev > 0.0:
        e_rel = (eps_v - e_cbm) * float(ha_to_ev)
        pool = np.flatnonzero(e_rel <= w_ev)
        if pool.size < n_keep:
            raise RuntimeError(
                f"particle_window_ev={w_ev:g} eV leaves only {pool.size} "
                f"virtuals (need ≥{n_keep}) above CBM for nl SVD."
            )
    C_use = C_v[:, pool]
    eps_use = eps_v[pool]
    S = np.asarray(mol.intor("int1e_ovlp"), dtype=float)
    A = C_use.T @ S[:, idx]
    u, s, _vt = np.linalg.svd(A, full_matrices=False)
    s2 = s * s
    denom = float(np.sum(s2))
    k = int(min(n_keep, u.shape[1], C_use.shape[1]))
    if k < 1:
        raise RuntimeError("SVD particle space is empty")
    captured = float(np.sum(s2[:k]) / denom) if denom > 1e-30 else 0.0
    if captured < float(min_complete):
        raise RuntimeError(
            f"particle SVD completeness {captured:.3f} < min_complete="
            f"{min_complete} for n={n} l={ell} (CB missed the shell, or "
            "LUMO is ligand-centered)."
        )
    U = u[:, :k]
    # Energies live in the PW-virtual coefficient metric (Uᵀ diag(ε) U).
    # Do NOT fold AO Löwdin into H: C_v is not S-normalized (GTO projection
    # completeness ≪ 1), so L.T @ h_u @ L can push ε_p outside [ε_min, ε_max].
    h_u = (U.T * eps_use) @ U
    eps_p, rot = np.linalg.eigh(h_u)
    C_part = _lowdin_orthonormalize(C_use @ (U @ rot), S)
    label = {0: "s", 1: "p", 2: "d", 3: "f"}[int(ell)]
    meta = {
        "n": int(n),
        "ell": int(ell),
        "shell": f"{int(n)}{label}",
        "n_keep": int(k),
        "n_ao": int(idx.size),
        "singular_values": [float(x) for x in s[:k]],
        "completeness": captured,
        "min_complete": float(min_complete),
        "eps_ha": [float(x) for x in eps_p],
        "particle_window_ev": w_ev,
        "n_virt_pool": int(pool.size),
        "n_virt_full": int(eps_v.size),
        "e_cbm_ha": e_cbm,
        "eps_pool_max_ev": float(np.max((eps_use - e_cbm) * ha_to_ev)),
        "eps_p_max_ev": float(np.max((eps_p - e_cbm) * ha_to_ev)),
    }
    return C_part, np.asarray(eps_p, float), meta


def gto_dipole_ia(mol, C_o: np.ndarray, C_v: np.ndarray, origin) -> np.ndarray:
    """``μ_{ia}`` length-gauge, shape ``(n_o * n_v, 3)`` flatten ``i*n_v+a``."""
    C_o = np.asarray(C_o, float)
    C_v = np.asarray(C_v, float)
    R0 = np.asarray(origin, float).reshape(3)
    with mol.with_common_orig(R0):
        r_ao = np.asarray(mol.intor("int1e_r"), dtype=float)
    mu = np.einsum("xuv,ui,va->xia", r_ao, C_o, C_v, optimize=True)
    n_o, n_v = C_o.shape[1], C_v.shape[1]
    return np.moveaxis(mu, 0, -1).reshape(n_o * n_v, 3)


def dense_cvs_A(
    eps_o: np.ndarray,
    eps_v: np.ndarray,
    eri_ovov: np.ndarray,
    eri_oovv: np.ndarray,
    *,
    triplet: bool = False,
    slater_scale: float = 1.0,
    hartree_scale: Optional[float] = None,
) -> np.ndarray:
    """TDA ``A`` in the hole×particle space, shape ``(n_trans, n_trans)``.

    Singlet: ``Δε + 2 (ia|jb) − s (ij|ab)``.
    Triplet: ``Δε − s (ij|ab)`` (no Hartree).
    ``slater_scale`` is ``c_x`` (1 for unscreened multiplet).
    """
    eps_o = np.asarray(eps_o, float).ravel()
    eps_v = np.asarray(eps_v, float).ravel()
    n_o, n_v = eps_o.size, eps_v.size
    dE = eps_v[None, :] - eps_o[:, None]
    A4 = np.zeros((n_o, n_v, n_o, n_v), dtype=float)
    A4 += np.einsum("ia,ij,ab->iajb", dE, np.eye(n_o), np.eye(n_v))
    s = float(slater_scale)
    A4 -= s * np.einsum("ijab->iajb", np.asarray(eri_oovv, float))
    if not triplet:
        hsc = 2.0 if hartree_scale is None else float(hartree_scale)
        A4 += hsc * np.asarray(eri_ovov, float)
    n_trans = n_o * n_v
    return A4.reshape(n_trans, n_trans)


def _casida_from_A(
    A: np.ndarray,
    mu_ia: Optional[np.ndarray],
    *,
    n_states: Optional[int] = None,
) -> "Any":
    from casidapy.casida_api import CasidaResults

    evals, Z = np.linalg.eigh(A)
    n_keep = int(n_states) if n_states is not None else evals.size
    n_keep = min(n_keep, evals.size)
    omega = evals[:n_keep]
    Z = Z[:, :n_keep]
    n_trans = A.shape[0]
    if mu_ia is None:
        d_mode = np.zeros((3, n_keep), dtype=float)
        f = np.zeros(n_keep, dtype=float)
    else:
        mu = np.asarray(mu_ia, float)
        if mu.shape != (n_trans, 3):
            raise ValueError(f"mu_ia shape {mu.shape} != ({n_trans}, 3)")
        d_mode = (mu.T @ Z)
        f = np.zeros(n_keep, dtype=float)
        for k, w in enumerate(omega):
            if w <= 1e-12:
                continue
            f[k] = (2.0 / 3.0) * float(w) * float(np.dot(d_mode[:, k], d_mode[:, k]))
    return CasidaResults(
        omega=omega,
        f=f,
        Z=Z,
        mu_transition=mu_ia,
        rho_transition=None,
        d_mode=d_mode,
        xpy=Z,
        metadata={"basis": "gto_particle", "tda": True},
    )


def resolve_particle_ci_space(
    mol,
    *,
    edge: str,
    absorber_atom: int = 0,
    oxidation: Optional[int] = None,
    z: Optional[int] = None,
    particle_n: Optional[int] = None,
    particle_l: Optional[int] = None,
) -> Dict[str, Any]:
    """Hole count, particle ``(n,ℓ)``, and ionic ``d^n`` from element + edge."""
    ia = int(absorber_atom)
    z_use = int(z) if z is not None else int(mol.atom_charge(ia))
    sym = str(mol.atom_symbol(ia)).strip().capitalize()
    if particle_n is None or particle_l is None:
        n_p, l_p = valence_particle_nl(z_use, edge)
    else:
        n_p, l_p = int(particle_n), int(particle_l)
    if oxidation is None:
        if sym not in DEFAULT_OXIDATION_STATES:
            raise ValueError(
                f"No default oxidation for {sym}; pass oxidation= for particle CI."
            )
        ox = int(DEFAULT_OXIDATION_STATES[sym])
    else:
        ox = int(oxidation)
    n_d = d_electron_count(z_use, ox)
    n_part = int(2 * l_p + 1)
    return {
        "z": z_use,
        "symbol": sym,
        "oxidation": ox,
        "n_d": n_d,
        "particle_n": int(n_p),
        "particle_l": int(l_p),
        "n_particle": n_part,
        "edge": str(edge).strip().upper(),
    }


def run_particle_ci(
    mol,
    C_o: np.ndarray,
    C_v: np.ndarray,
    eps_o: np.ndarray,
    eps_v: np.ndarray,
    *,
    edge: str = "L",
    absorber_atom: int = 0,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    oxidation: Optional[int] = None,
    z: Optional[int] = None,
    particle_n: Optional[int] = None,
    particle_l: Optional[int] = None,
    slater_scale: float = 1.0,
    min_complete: float = 0.2,
    particle_window_ev: float = 0.0,
    soc: bool = True,
    n_states: Optional[int] = None,
    core=None,
) -> Tuple[Any, Dict[str, Any]]:
    """Closed-shell ``n_h^{full} n_p^{0} → n_h^{full-1} n_p^{1}`` CIS + optional SI-SOC.

    Open d^n (``n_d`` not 0) raises: that needs FCI in the 8-orbital space,
    which is not this CIS.  ``c_x = slater_scale`` (default 1), not PBE0 0.25.

    ``particle_window_ev`` (K-edge) limits the SVD pool to near-CBM virtuals
    so continuum bands with residual ``nl`` weight do not steal the exciton.
    """
    from casidapy.xas.cvs import CVSSOCResults
    from casidapy.utils.soc import solve_soc_si

    space = resolve_particle_ci_space(
        mol,
        edge=edge,
        absorber_atom=absorber_atom,
        oxidation=oxidation,
        z=z,
        particle_n=particle_n,
        particle_l=particle_l,
    )
    if int(space["n_d"]) != 0:
        raise RuntimeError(
            f"{space['symbol']}{space['oxidation']:+d} is d{space['n_d']}, not d0. "
            "Particle CIS is 2p⁵ nd¹ (closed GS). Open-shell CTM FCI is not wired."
        )
    C_o = np.asarray(C_o, float)
    C_part, eps_p, svd_meta = svd_particle_orbitals(
        mol,
        C_v,
        eps_v,
        n=int(space["particle_n"]),
        ell=int(space["particle_l"]),
        absorber_atom=int(absorber_atom),
        n_keep=int(space["n_particle"]),
        min_complete=float(min_complete),
        window_ev=float(particle_window_ev or 0.0),
    )
    eri_ovov = mo_ovov(mol, C_o, C_part)
    eri_oovv = mo_ijab(mol, C_o, C_part)
    A_s = dense_cvs_A(
        eps_o, eps_p, eri_ovov, eri_oovv, triplet=False, slater_scale=slater_scale
    )
    A_t = dense_cvs_A(
        eps_o, eps_p, eri_ovov, eri_oovv, triplet=True, slater_scale=slater_scale
    )
    mu_ia = gto_dipole_ia(mol, C_o, C_part, origin)
    n_trans = A_s.shape[0]
    n_st = int(n_states) if n_states is not None else n_trans
    res_s = _casida_from_A(A_s, mu_ia, n_states=n_st)
    res_t = _casida_from_A(A_t, None, n_states=n_st)
    meta = {
        **space,
        "svd": svd_meta,
        "slater_scale": float(slater_scale),
        "n_hole": int(C_o.shape[1]),
        "n_particle": int(C_part.shape[1]),
        "n_trans": int(n_trans),
        "cx": float(slater_scale),
        "kernel": "gto_eri",
    }
    if not soc:
        res_s.metadata.update(meta)
        return res_s, meta

    soc_res = solve_soc_si(
        res_s,
        res_t,
        kernel=None,
        mol=mol,
        C_o=C_o,
        C_v=C_part,
        hole_only=True,
        mu_ia=mu_ia,
        origin=origin,
    )
    wrapped = CVSSOCResults(
        omega=soc_res.omega,
        f=soc_res.f,
        mu=soc_res.mu,
        U=soc_res.U,
        singlet_weight=soc_res.singlet_weight,
        triplet_weight=soc_res.triplet_weight,
        singlet=res_s,
        triplet=res_t,
        core=core,
        kernel=None,
        kernel_t=None,
        soc=soc_res,
        Z=None,
        mu_transition=mu_ia,
        metadata={
            **dict(soc_res.meta),
            **meta,
            "soc_mode": "si",
            "n_singlet": int(res_s.omega.size),
            "n_triplet": int(res_t.omega.size),
            "n_si": int(soc_res.omega.size),
            "hole_only": True,
            "particle_ci": True,
        },
    )
    return wrapped, wrapped.metadata
