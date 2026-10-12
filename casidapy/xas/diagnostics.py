"""XAS diagnostics, validation gates, and prototype paths (not production)."""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

from casidapy.xas.chemical_data import DEFAULTS
from casidapy.xas.dipole import (
    RADIAL_N_ANG,
    RADIAL_N_RAD,
    RADIAL_RMAX_BOHR,
    _commit_pruned_grid_virtuals,
    _commit_pruned_gto_virtuals,
    _select_virtuals_by_score,
    absorber_origin_bohr,
    absorber_radial_grid,
    build_core_dipole_grid,
    core_charge_centroid_bohr,
    core_rms_radius_bohr,
    core_virtual_dipole_matrix_gto,
    core_virtual_dipole_matrix_gto_projected,
    core_virtual_dipole_matrix_local,
    core_virtual_dipole_matrix_radial,
    fft_gradient_components,
    trilinear_sample,
    _fragment_dipole_integrals,
    _orthogonalize_core_ao_against_virt_ao,
)
from casidapy.xas.xas_utils import (
    grid_coords_bohr,
    grid_orbital_norms,
    project_grid_orbitals_to_fragment_ao,
)

logger = logging.getLogger(__name__)

_HA_TO_EV = DEFAULTS["ha_to_ev"]


def compare_cvs_excitation_energies(
    results_a,
    results_b,
    *,
    label_a: str = "grid",
    label_b: str = "projected",
    n_compare: Optional[int] = None,
) -> dict:
    """Compare Casida ``omega`` vectors from two CVS-TDA runs (Ha).

    States are compared in ascending-energy order (no cross-matching).
    """
    omega_a = np.asarray(results_a.omega, dtype=float).ravel()
    omega_b = np.asarray(results_b.omega, dtype=float).ravel()
    n = min(len(omega_a), len(omega_b))
    if n_compare is not None:
        n = min(n, int(n_compare))
    if n == 0:
        return {
            "label_a": label_a,
            "label_b": label_b,
            "n": 0,
            "max_abs_ha": 0.0,
            "rms_ha": 0.0,
            "mean_ha": 0.0,
            "max_abs_ev": 0.0,
            "rms_ev": 0.0,
        }
    d = omega_a[:n] - omega_b[:n]
    ha_to_ev = _HA_TO_EV
    return {
        "label_a": label_a,
        "label_b": label_b,
        "n": int(n),
        "max_abs_ha": float(np.max(np.abs(d))),
        "rms_ha": float(np.sqrt(np.mean(d * d))),
        "mean_ha": float(np.mean(d)),
        "max_abs_ev": float(np.max(np.abs(d)) * ha_to_ev),
        "rms_ev": float(np.sqrt(np.mean(d * d)) * ha_to_ev),
        "delta_ha": d,
        "omega_a_ha": omega_a[:n],
        "omega_b_ha": omega_b[:n],
    }


def spillage_gate(
    spillage_summary: dict,
    *,
    max_mean: float = 0.1,
    max_p95: float = 0.2,
    max_large_frac: float = 0.05,
) -> dict:
    """Decision gate for step 1 (fragment AO adequacy after pruning).

    Returns ``passed`` and human-readable ``reason``.
    """
    n = max(int(spillage_summary.get("n", 0)), 1)
    n_large = int(spillage_summary.get("n_large_0p1", 0))
    frac_large = n_large / n
    mean_s = float(spillage_summary.get("mean", 1.0))
    p95_s = float(spillage_summary.get("p95", 1.0))
    passed = (
        mean_s <= max_mean and p95_s <= max_p95 and frac_large <= max_large_frac
    )
    if passed:
        reason = (
            f"spillage OK: mean={mean_s:.4f}, p95={p95_s:.4f}, "
            f"frac(S>0.1)={frac_large:.3f}"
        )
    else:
        reason = (
            f"spillage high: mean={mean_s:.4f} (limit {max_mean}), "
            f"p95={p95_s:.4f} (limit {max_p95}), "
            f"frac(S>0.1)={frac_large:.3f} (limit {max_large_frac}) — "
            "consider diffuse augmentation or Wannier intermediate"
        )
    return {
        "passed": bool(passed),
        "reason": reason,
        "mean": mean_s,
        "p95": p95_s,
        "frac_large_0p1": frac_large,
    }


def projected_k_gate(
    energy_compare: dict,
    *,
    max_rms_ev: float = 0.5,
    max_abs_ev: float = 2.0,
) -> dict:
    """Decision gate for step 3 (open-boundary GTO K vs grid K on benign case)."""
    rms = float(energy_compare.get("rms_ev", np.inf))
    mx = float(energy_compare.get("max_abs_ev", np.inf))
    passed = rms <= max_rms_ev and mx <= max_abs_ev
    if passed:
        reason = f"energies agree: rms={rms:.3f} eV, max|Δ|={mx:.3f} eV"
    else:
        reason = (
            f"energy mismatch: rms={rms:.3f} eV (limit {max_rms_ev}), "
            f"max|Δ|={mx:.3f} eV (limit {max_abs_ev}) — periodic response may matter"
        )
    return {"passed": bool(passed), "reason": reason, "rms_ev": rms, "max_abs_ev": mx}


def _matrix_delta_stats(
    M: np.ndarray,
    Mref: np.ndarray,
    *,
    ref_mask: Optional[np.ndarray] = None,
) -> dict:
    """Summary of ``M - Mref`` (Frobenius, max, RMS)."""
    d = np.asarray(M, dtype=float) - np.asarray(Mref, dtype=float)
    ref = np.asarray(Mref, dtype=float)
    if ref_mask is not None:
        d = d[ref_mask]
        ref = ref[ref_mask]
    ref_norm = max(float(np.linalg.norm(ref)), 1e-30)
    d_norm = float(np.linalg.norm(d))
    n = max(int(d.size), 1)
    return {
        "frobenius": d_norm,
        "rel_frobenius": d_norm / ref_norm,
        "max_abs": float(np.max(np.abs(d))) if d.size else 0.0,
        "rms": float(np.sqrt(np.mean(d * d))) if d.size else 0.0,
        "mean_abs": float(np.mean(np.abs(d))) if d.size else 0.0,
    }


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    if x.size == 0:
        return float("nan")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-30 or sy < 1e-30:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def extract_kernel_coupling_matrix(kernel) -> np.ndarray:
    """Dense eh-coupling ``K`` for the active CVS space (after ``setup``)."""
    if not getattr(kernel, "_ready", False):
        kernel.setup(tda=True)
    n = int(kernel.n_trans)
    cached = getattr(kernel, "_K", None)
    if cached is not None:
        return np.asarray(cached, dtype=float)
    return np.asarray(kernel.dense_K_rows(list(range(n))), dtype=float)


def extract_kernel_energy_gaps(kernel) -> np.ndarray:
    """Diagonal TDA gaps ``Δε`` (Hartree) for the active space."""
    if hasattr(kernel, "diagonal_dE"):
        return np.asarray(kernel.diagonal_dE(), dtype=float).ravel()
    dE = getattr(kernel, "_dE", None)
    if dE is None:
        raise RuntimeError("kernel has no energy gaps; call setup() first")
    return np.asarray(dE, dtype=float).ravel()


def compare_cvs_coupling_matrices(
    K_a: np.ndarray,
    K_b: np.ndarray,
    *,
    dE_a: Optional[Sequence[float]] = None,
    dE_b: Optional[Sequence[float]] = None,
    mu_a: Optional[np.ndarray] = None,
    mu_b: Optional[np.ndarray] = None,
    label_a: str = "grid",
    label_b: str = "projected",
) -> dict:
    """Compare dense Casida coupling blocks between two CVS kernels.

    Reports Frobenius / max norms for ``K``, its diagonal / off-diagonal parts,
  optional ``A = K + diag(Δε)``, optional transition dipoles ``μ`` (n_trans×3),
    and Pearson correlation of flattened ``K``.
    """
    Ka = np.asarray(K_a, dtype=float)
    Kb = np.asarray(K_b, dtype=float)
    if Ka.shape != Kb.shape:
        raise ValueError(f"K shapes differ: {Ka.shape} vs {Kb.shape}")
    if Ka.ndim != 2 or Ka.shape[0] != Ka.shape[1]:
        raise ValueError(f"K must be square (n_trans, n_trans), got {Ka.shape}")

    n = int(Ka.shape[0])
    eye = np.eye(n, dtype=bool)
    offdiag = ~eye

    Ka_diag = np.diag(Ka)
    Kb_diag = np.diag(Kb)
    Ka_off = Ka.copy()
    Kb_off = Kb.copy()
    Ka_off[eye] = 0.0
    Kb_off[eye] = 0.0

    row_delta = np.linalg.norm(Ka - Kb, axis=1)
    row_ref = np.maximum(np.linalg.norm(Ka, axis=1), 1e-30)
    row_rel = row_delta / row_ref

    out: dict[str, Any] = {
        "label_a": label_a,
        "label_b": label_b,
        "n_trans": n,
        "K": _matrix_delta_stats(Ka, Kb),
        "K_corr": _pearson_corr(Ka, Kb),
        "K_diag": _matrix_delta_stats(Ka_diag, Kb_diag),
        "K_offdiag": _matrix_delta_stats(Ka_off, Kb_off, ref_mask=offdiag),
        "K_row_rel_frobenius": {
            "mean": float(np.mean(row_rel)),
            "max": float(np.max(row_rel)),
            "p95": float(np.percentile(row_rel, 95)),
        },
    }

    if dE_a is not None and dE_b is not None:
        dEa = np.asarray(dE_a, dtype=float).ravel()
        dEb = np.asarray(dE_b, dtype=float).ravel()
        if dEa.shape != (n,) or dEb.shape != (n,):
            raise ValueError(
                f"dE must have length {n}, got {dEa.shape} and {dEb.shape}"
            )
        out["dE"] = _matrix_delta_stats(dEa, dEb)
        out["dE_max_abs_ev"] = float(out["dE"]["max_abs"] * _HA_TO_EV)
        Aa = Ka + np.diag(dEa)
        Ab = Kb + np.diag(dEb)
        out["A"] = _matrix_delta_stats(Aa, Ab)
        out["A_corr"] = _pearson_corr(Aa, Ab)
        eig_a = np.sort(np.linalg.eigvalsh(Aa))
        eig_b = np.sort(np.linalg.eigvalsh(Ab))
        deig = eig_a - eig_b
        out["A_eig"] = {
            "max_abs_ha": float(np.max(np.abs(deig))),
            "rms_ha": float(np.sqrt(np.mean(deig * deig))),
            "max_abs_ev": float(np.max(np.abs(deig)) * _HA_TO_EV),
            "rms_ev": float(np.sqrt(np.mean(deig * deig)) * _HA_TO_EV),
        }

    if mu_a is not None and mu_b is not None:
        Ma = np.asarray(mu_a, dtype=float)
        Mb = np.asarray(mu_b, dtype=float)
        if Ma.shape != Mb.shape:
            raise ValueError(f"mu shapes differ: {Ma.shape} vs {Mb.shape}")
        out["mu"] = _matrix_delta_stats(Ma, Mb)
        out["mu_corr"] = _pearson_corr(Ma, Mb)
        mag_a = np.linalg.norm(Ma, axis=-1)
        mag_b = np.linalg.norm(Mb, axis=-1)
        out["mu_mag"] = _matrix_delta_stats(mag_a, mag_b)

    return out


def compare_cvs_tda_decomposition(
    omega_a: Sequence[float],
    omega_b: Sequence[float],
    K_a: np.ndarray,
    K_b: np.ndarray,
    dE_a: Sequence[float],
    dE_b: Sequence[float],
    *,
    n_compare: Optional[int] = None,
    label_a: str = "grid",
    label_b: str = "projected",
) -> dict:
    """Separate shared ``Δε`` from coupling when comparing two CVS-TDA paths.

    TDA uses ``A = K + diag(Δε)``. Identical ``Δε`` with different ``K`` can
    still yield similar ``ω`` when ``‖K‖ ≪ ‖Δε‖`` (agreement inherited from the
    diagonal, not earned from matching response).

    Reports:

    - ``Δε`` identity between paths
    - ``K`` eigenvalue differences (coupling spectrum)
    - ``ω`` differences (total excitation)
    - ``coupling_shift`` = ``ω − sort(Δε)`` with ascending IP pairing (weak-mixing
      diagnostic; not a substitute for analyzing ``Z``)

    Use with :func:`coupling_k_gate` to require coupling agreement, not only ``ω``.
    """
    dEa = np.asarray(dE_a, dtype=float).ravel()
    dEb = np.asarray(dE_b, dtype=float).ravel()
    if dEa.shape != dEb.shape:
        raise ValueError(f"dE shapes differ: {dEa.shape} vs {dEb.shape}")
    n = int(dEa.size)
    if n == 0:
        return {"n": 0}

    Ka = np.asarray(K_a, dtype=float)
    Kb = np.asarray(K_b, dtype=float)
    if Ka.shape != (n, n) or Kb.shape != (n, n):
        raise ValueError(f"K must be ({n}, {n}), got {Ka.shape} and {Kb.shape}")

    oa = np.asarray(omega_a, dtype=float).ravel()
    ob = np.asarray(omega_b, dtype=float).ravel()
    n_cmp = min(len(oa), len(ob), n)
    if n_compare is not None:
        n_cmp = min(n_cmp, int(n_compare))

    dE_stats = _matrix_delta_stats(dEa, dEb)
    K_stats = _matrix_delta_stats(Ka, Kb)

    eig_Ka = np.sort(np.linalg.eigvalsh(Ka))
    eig_Kb = np.sort(np.linalg.eigvalsh(Kb))
    dK_eig = eig_Ka[:n_cmp] - eig_Kb[:n_cmp]

    ip_a = np.sort(dEa)
    ip_b = np.sort(dEb)
    d_omega = oa[:n_cmp] - ob[:n_cmp]
    shift_a = oa[:n_cmp] - ip_a[:n_cmp]
    shift_b = ob[:n_cmp] - ip_b[:n_cmp]
    d_shift = shift_a - shift_b

    de_norm = max(float(np.linalg.norm(dEa)), 1e-30)
    k_norm_a = float(np.linalg.norm(Ka))
    k_norm_b = float(np.linalg.norm(Kb))

    Aa_eig = np.sort(np.linalg.eigvalsh(Ka + np.diag(dEa)))

    def _ev_stats(x: np.ndarray) -> dict:
        x = np.asarray(x, dtype=float).ravel()
        if x.size == 0:
            return {"max_abs_ha": 0.0, "rms_ha": 0.0, "max_abs_ev": 0.0, "rms_ev": 0.0}
        return {
            "max_abs_ha": float(np.max(np.abs(x))),
            "rms_ha": float(np.sqrt(np.mean(x * x))),
            "max_abs_ev": float(np.max(np.abs(x)) * _HA_TO_EV),
            "rms_ev": float(np.sqrt(np.mean(x * x)) * _HA_TO_EV),
        }

    omega_stats = _ev_stats(d_omega)
    K_eig_stats = _ev_stats(dK_eig)
    shift_pair_stats = _ev_stats(d_shift)

    inherited = bool(
        dE_stats["max_abs"] <= 1e-10
        and K_stats["rel_frobenius"] > 0.1
        and omega_stats["rms_ev"] < 2.0
        and (k_norm_a / de_norm) < 0.05
        and (k_norm_b / de_norm) < 0.05
    )

    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_trans": n,
        "n_compare": int(n_cmp),
        "dE": dE_stats,
        "dE_max_abs_ev": float(dE_stats["max_abs"] * _HA_TO_EV),
        "K": K_stats,
        "scales": {
            "dE_frobenius_ha": de_norm,
            "K_frobenius_a_ha": k_norm_a,
            "K_frobenius_b_ha": k_norm_b,
            "K_over_dE_a": k_norm_a / de_norm,
            "K_over_dE_b": k_norm_b / de_norm,
        },
        "omega": omega_stats,
        "K_eig": K_eig_stats,
        "coupling_shift": {
            "pairing": "ascending sort(Δε) vs ascending ω (weak-mixing diagnostic)",
            "shift_a": _ev_stats(shift_a),
            "shift_b": _ev_stats(shift_b),
            "delta_shift": shift_pair_stats,
        },
        "inherited_agreement": {
            "likely": inherited,
            "reason": (
                "Δε identical, ‖K‖/‖Δε‖ small, and ‖ΔK‖/‖K‖ large — "
                "ω agreement plausibly inherited from shared diagonal"
                if inherited
                else "ω agreement not explained by Δε dominance alone, or Δε differs"
            ),
        },
        "ip_omega_a_ev": (ip_a[:n_cmp] * _HA_TO_EV).tolist(),
        "ip_omega_b_ev": (ip_b[:n_cmp] * _HA_TO_EV).tolist(),
        "coupling_shift_a_ev": (shift_a * _HA_TO_EV).tolist(),
        "coupling_shift_b_ev": (shift_b * _HA_TO_EV).tolist(),
        "A_eig_a_ev": (Aa_eig[:n_cmp] * _HA_TO_EV).tolist(),
    }


def coupling_k_gate(
    decomposition: dict,
    *,
    max_K_eig_rms_ev: float = 0.5,
    max_shift_pair_rms_ev: float = 0.5,
) -> dict:
    """Gate for earned coupling agreement (not only shared ``Δε``).

    Fails when ``K`` eigenvalues differ substantially even if total ``ω`` agrees.
    """
    K_eig = decomposition.get("K_eig", {})
    shift = decomposition.get("coupling_shift", {}).get("delta_shift", {})
    k_rms = float(K_eig.get("rms_ev", np.inf))
    sh_rms = float(shift.get("rms_ev", np.inf))
    passed = k_rms <= max_K_eig_rms_ev and sh_rms <= max_shift_pair_rms_ev
    if passed:
        reason = (
            f"coupling agrees: Δλ(K) rms={k_rms:.3f} eV, "
            f"Δ(ω−Δε) rms={sh_rms:.3f} eV"
        )
    else:
        reason = (
            f"coupling mismatch: Δλ(K) rms={k_rms:.3f} eV (limit {max_K_eig_rms_ev}), "
            f"Δ(ω−Δε) rms={sh_rms:.3f} eV (limit {max_shift_pair_rms_ev}) — "
            "ω agreement may be inherited from shared Δε"
        )
    return {
        "passed": bool(passed),
        "reason": reason,
        "K_eig_rms_ev": k_rms,
        "shift_pair_rms_ev": sh_rms,
        "inherited_likely": bool(
            decomposition.get("inherited_agreement", {}).get("likely", False)
        ),
    }


# --- overlap-based virtual selection ---

def core_virtual_overlap_matrix_grid(
    core_orbitals,
    virtual_orbitals,
    grid,
    *,
    use_gpu: bool = False,
) -> np.ndarray:
    """|⟨core|virt⟩| overlaps on a DFTpy grid.

    Parameters
    ----------
    core_orbitals, virtual_orbitals
        Sequences of grid orbitals (``DirectField`` or flat arrays).

    Returns
    -------
    ndarray, shape ``(n_core, n_virt)``
    """
    from casidapy.utils.accel import array_module, asnumpy

    dV = float(grid.dV)
    xp = array_module(use_gpu)
    cores = [
        xp.asarray(np.asarray(p, dtype=float).ravel(), dtype=float)
        for p in core_orbitals
    ]
    virts = [
        xp.asarray(np.asarray(p, dtype=float).ravel(), dtype=float)
        for p in virtual_orbitals
    ]
    if not cores or not virts:
        return np.zeros((len(cores), len(virts)), dtype=float)
    C = xp.stack(cores, axis=0)
    V = xp.stack(virts, axis=0)
    out = xp.abs(C @ V.T) * dV
    return asnumpy(out)


def core_virtual_overlap_matrix_gto(
    core_mo_coeff,
    virt_mo_coeff,
    ovlp: np.ndarray,
) -> np.ndarray:
    """|⟨core|virt⟩| in an AO basis (PySCF overlap matrix)."""
    C_c = np.asarray(core_mo_coeff, dtype=float)
    C_v = np.asarray(virt_mo_coeff, dtype=float)
    S = np.asarray(ovlp, dtype=float)
    if C_c.ndim == 1:
        C_c = C_c.reshape(-1, 1)
    if C_v.ndim == 1:
        C_v = C_v.reshape(-1, 1)
    return np.abs(C_c.T @ S @ C_v)


def virtual_overlap_scores(
    overlap_matrix: np.ndarray,
    *,
    reduction: str = "max",
) -> np.ndarray:
    """Per-virtual score from a core×virtual overlap matrix.

    ``reduction='max'`` (default): max |⟨c|v⟩| over cores.
    ``reduction='sum'``: ∑_c |⟨c|v⟩|² .
    """
    ov = np.asarray(overlap_matrix, dtype=float)
    if ov.ndim != 2:
        raise ValueError(f"overlap_matrix must be 2-D, got shape {ov.shape}")
    mode = str(reduction).lower().strip()
    if mode == "max":
        return np.max(ov, axis=0)
    if mode in ("sum", "sumsq", "l2"):
        return np.sqrt(np.sum(ov**2, axis=0))
    raise ValueError(f"unknown reduction={reduction!r}; use 'max' or 'sum'")

def select_virtuals_by_core_overlap(
    unocc_eigs,
    virtual_orbitals,
    core_orbitals,
    *,
    grid=None,
    ovlp: Optional[np.ndarray] = None,
    min_abs_overlap: float = 1e-5,
    n_keep: Optional[int] = None,
    reduction: str = "max",
    use_gpu: bool = False,
    verbose: bool = False,
):
    """Keep virtuals with the largest core overlap (drop box continuum).

    Parameters
    ----------
    unocc_eigs
        Virtual eigenvalues (Hartree), length ``n_virt``.
    virtual_orbitals
        PW: list of grid orbitals. GTO: ``(nao, n_virt)`` MO coefficients.
    core_orbitals
        PW: list of core grid orbitals. GTO: ``(nao, n_core)`` MO coefficients.
    grid
        Required for grid orbitals (PW).
    ovlp
        AO overlap matrix for the GTO path.
    min_abs_overlap
        Drop virtuals with score below this threshold.
    n_keep
        After thresholding, keep at most this many virtuals (largest scores).

    Returns
    -------
    unocc_eigs_keep, virtual_orbitals_keep, overlap_matrix, kept_indices
    """
    eigs = np.asarray(unocc_eigs, dtype=float).ravel()
    n_virt = int(eigs.size)
    if n_virt == 0:
        return eigs, list(virtual_orbitals), np.zeros((0, 0)), np.zeros(0, dtype=int)

    if grid is not None:
        ov = core_virtual_overlap_matrix_grid(
            core_orbitals, virtual_orbitals, grid, use_gpu=use_gpu
        )
        psi_list = list(virtual_orbitals)
    elif ovlp is not None:
        ov = core_virtual_overlap_matrix_gto(core_orbitals, virtual_orbitals, ovlp)
        psi_list = None  # GTO coeffs updated below
    else:
        raise ValueError("select_virtuals_by_core_overlap needs grid= or ovlp=")

    scores = virtual_overlap_scores(ov, reduction=reduction)
    e_keep, psi_keep, idx = _select_virtuals_by_score(
        eigs,
        psi_list if grid is not None else virtual_orbitals,
        scores,
        min_score=float(min_abs_overlap),
        n_keep=n_keep,
        grid_mode=grid is not None,
        label="overlap",
        verbose=verbose,
    )
    return e_keep, psi_keep, ov, idx

def prune_kernel_virtuals_by_core_overlap(
    kernel,
    core_orbitals,
    *,
    min_abs_overlap: float = 1e-5,
    n_keep: Optional[int] = None,
    reduction: str = "max",
    ovlp: Optional[np.ndarray] = None,
    use_gpu: bool = False,
    verbose: bool = False,
):
    """Drop weakly overlapping virtuals from a kernel before CVS inject.

    Works on ``PlaneWaveKernel`` (grid fields) and ``GTOKernel`` (MO coeffs).
    Occupied placeholder / valence slots are unchanged; only virtuals shrink.

    For XAS on plane-wave hosts prefer
    :func:`prune_kernel_virtuals_by_core_dipole`, which ranks virtuals by the
    intensity they actually carry.
    """
    if not getattr(kernel, "_psi_unocc", None) and not hasattr(kernel, "_C_v"):
        raise RuntimeError("kernel has no virtual active space to prune")

    if getattr(kernel, "provides_grid", False):
        e_keep, psi_keep, ov, idx = select_virtuals_by_core_overlap(
            kernel._unocc_e,
            kernel._psi_unocc,
            core_orbitals,
            grid=kernel.grid,
            min_abs_overlap=min_abs_overlap,
            n_keep=n_keep,
            reduction=reduction,
            use_gpu=use_gpu,
            verbose=verbose,
        )
        _commit_pruned_grid_virtuals(kernel, e_keep, psi_keep)
    elif hasattr(kernel, "_C_v"):
        if ovlp is None:
            mol = getattr(kernel, "mol", None) or getattr(kernel.mf, "mol", None)
            if mol is None:
                raise ValueError("GTO overlap pruning requires ovlp= or kernel.mol")
            ovlp = mol.intor("int1e_ovlp")
        e_keep, C_keep, ov, idx = select_virtuals_by_core_overlap(
            kernel._unocc_e,
            kernel._C_v,
            core_orbitals,
            ovlp=ovlp,
            min_abs_overlap=min_abs_overlap,
            n_keep=n_keep,
            reduction=reduction,
            verbose=verbose,
        )
        _commit_pruned_gto_virtuals(kernel, e_keep, C_keep)
    else:
        raise TypeError(
            f"prune_kernel_virtuals_by_core_overlap: unsupported {type(kernel)}"
        )

    kernel._psi_unocc_arr = None
    kernel._psi_unocc_dev = None
    return kernel, ov, idx



# Radial / FFT dipole helpers are defined in ``casidapy.xas.dipole`` and
# re-exported above for existing diagnostics callers.



# --- AO spillage / projector audits ---

def ao_projection_completeness(
    mol,
    grid,
    orbitals,
    *,
    comm=None,
    blksize: int = 16000,
) -> np.ndarray:
    """Fraction of each grid orbital inside the fragment AO span.

    For normalized ``ψ_n`` on the grid,

    ``c_n = ⟨ψ_n|P_AO|ψ_n⟩ = v_n^T S^{-1} v_n / ‖ψ_n‖²``

  where ``v_{μn} = ∫ φ_μ ψ_n`` and ``S`` is the AO overlap matrix.
    """
    virts = list(orbitals)
    if not virts:
        return np.zeros(0, dtype=float)
    V = project_grid_orbitals_to_fragment_ao(
        mol, grid, virts, blksize=blksize, comm=comm, method="qe_grid"
    )
    S = mol.intor("int1e_ovlp")
    norms = grid_orbital_norms(virts, grid)
    n = V.shape[1]
    completeness = np.empty(n, dtype=float)
    for a in range(n):
        v = V[:, a]
        proj = float(v @ np.linalg.solve(S, v))
        completeness[a] = proj / max(norms[a] ** 2, 1e-30)
    return np.clip(completeness, 0.0, 1.0 + 1e-10)


def virtual_ao_spillage(
    mol,
    grid,
    orbitals,
    *,
    comm=None,
    blksize: int = 16000,
) -> np.ndarray:
    """AO spillage ``S_n = 1 − ⟨ψ_n|P_AO|ψ_n⟩`` for grid virtuals."""
    return 1.0 - ao_projection_completeness(
        mol, grid, orbitals, comm=comm, blksize=blksize
    )


def summarize_ao_spillage(spillage: np.ndarray) -> dict:
    """Summary stats for a spillage vector (fraction outside fragment AO span)."""
    s = np.asarray(spillage, dtype=float).ravel()
    if s.size == 0:
        return {
            "n": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "n_large_0p1": 0,
            "n_large_0p3": 0,
        }
    return {
        "n": int(s.size),
        "min": float(np.min(s)),
        "max": float(np.max(s)),
        "mean": float(np.mean(s)),
        "median": float(np.median(s)),
        "p90": float(np.percentile(s, 90)),
        "p95": float(np.percentile(s, 95)),
        "p99": float(np.percentile(s, 99)),
        "n_large_0p1": int(np.sum(s > 0.1)),
        "n_large_0p3": int(np.sum(s > 0.3)),
    }


def audit_virtual_ao_spillage(
    core,
    virtual_orbitals,
    grid,
    *,
    projection_mol=None,
    comm=None,
    blksize: int = 16000,
    verbose: bool = False,
    check_projector_consistency: bool = False,
) -> dict:
    """Measure AO spillage on kept PW virtuals (step-1 decision gate).

    Returns a dict with per-orbital ``spillage``, ``completeness``, and
    ``summary`` statistics. Uses ``core.fragment_mol`` as the projector basis
    unless ``projection_mol`` is set (multi-atom / diffuse projection tests).

    ``check_projector_consistency``
        When True, run :func:`audit_ao_projector_consistency` on the same
        ``mol`` / ``grid`` / orbitals (norms, ``S^{-1}``, grid quadrature).
    """
    mol = projection_mol if projection_mol is not None else core.fragment_mol
    spillage = virtual_ao_spillage(
        mol, grid, virtual_orbitals, comm=comm, blksize=blksize
    )
    completeness = 1.0 - spillage
    summary = summarize_ao_spillage(spillage)
    out = {
        "spillage": spillage,
        "completeness": completeness,
        "summary": summary,
        "n_virt": int(spillage.size),
        "basis": getattr(mol, "basis", None),
        "n_ao": int(mol.nao_nr()),
    }
    if check_projector_consistency:
        out["projector_consistency"] = audit_ao_projector_consistency(
            mol,
            grid,
            virtual_orbitals,
            core=core,
            comm=comm,
            blksize=blksize,
        )
    if verbose:
        logger.info(
            "AO spillage audit: n=%d  max=%.4f  mean=%.4f  p95=%.4f  "
            "n(S>0.1)=%d  n(S>0.3)=%d",
            summary["n"],
            summary["max"],
            summary["mean"],
            summary["p95"],
            summary["n_large_0p1"],
            summary["n_large_0p3"],
        )
        if check_projector_consistency:
            pc = out["projector_consistency"]
            logger.info(
                "projector consistency: passed=%s  max|‖ψ‖_sum−‖ψ‖_int|=%.2e  "
                "max|Δc(S_analytic vs S_grid)|=%.2e  n_clipped=%d",
                pc["passed"],
                pc["max_abs_norm_delta"],
                pc["max_abs_S_analytic_vs_grid_delta"],
                pc["n_completeness_clipped"],
            )
    return out


def audit_ao_projector_consistency(
    mol,
    grid,
    orbitals,
    core=None,
    *,
    comm=None,
    blksize: int = 16000,
    rtol: float = 1e-3,
) -> dict:
    """Verify spillage and dipole projection share one mechanical projector.

    Checks (same ``mol``, full charge ``grid``, identical ``ψ`` arrays):

    1. ``‖ψ‖`` from ``Σ|ψ|² dV`` matches ``DirectField`` integral (QE norm path).
    2. ``V`` from duplicate ``project_grid_orbitals_to_fragment_ao`` calls matches.
    3. Spillage ``v^T S^{-1} v / ‖ψ‖²`` with analytic ``S`` vs the same
       numerator with grid-built ``S_grid = Σ_i φ_μ(r_i) φ_ν(r_i) dV`` (flags
       mixed grid/analytic metric; dipole ``μ`` uses the same ``V`` but no
       ``S^{-1}``).
    4. Dipole path uses the same ``V`` (``core_virtual_dipole_matrix_gto_projected``).

    ``S`` is always ``mol.intor('int1e_ovlp')`` (analytic GTO overlap). There is
    no near-atom quadrature mask and no density-fit metric in this projector.
    """
    from pyscf.dft import numint
    from dftpy.field import DirectField
    from casidapy.utils.accel import block_slices

    virts = list(orbitals)
    n = len(virts)
    if n == 0:
        return {"passed": True, "n_virt": 0}

    V_a = project_grid_orbitals_to_fragment_ao(
        mol, grid, virts, blksize=blksize, comm=comm
    )
    V_b = project_grid_orbitals_to_fragment_ao(
        mol, grid, virts, blksize=blksize, comm=comm
    )
    max_v_delta = float(np.max(np.abs(V_a - V_b))) if n else 0.0

    S = mol.intor("int1e_ovlp")
    norms_sum = grid_orbital_norms(virts, grid)
    norms_int = np.empty(n, dtype=float)
    for a, psi in enumerate(virts):
        field = DirectField(grid=grid, data=np.asarray(psi, dtype=float).ravel())
        norms_int[a] = float(np.sqrt((field.conj() * field).integral().real))

    norm_delta = np.abs(norms_sum - norms_int)
    norms_sq = np.maximum(norms_sum ** 2, 1e-30)

    completeness_formula = np.empty(n, dtype=float)
    completeness_grid_metric = np.empty(n, dtype=float)
    coords = grid_coords_bohr(grid)
    dV = float(grid.dV)
    ngrid = coords.shape[0]
    nao = int(mol.nao_nr())
    S_grid = np.zeros((nao, nao), dtype=float)

    for p0, p1 in block_slices(ngrid, blksize, comm):
        ao = numint.eval_ao(mol, coords[p0:p1])
        S_grid += ao.T @ ao * dV

    for a in range(n):
        v = V_a[:, a]
        completeness_formula[a] = float(v @ np.linalg.solve(S, v)) / norms_sq[a]
        completeness_grid_metric[a] = float(
            v @ np.linalg.solve(S_grid, v)
        ) / norms_sq[a]

    comp_delta = np.abs(completeness_formula - completeness_grid_metric)

    dipole_v_delta = 0.0
    if core is not None:
        dip = core_virtual_dipole_matrix_gto_projected(
            core, virts, grid, blksize=blksize, comm=comm
        )
        S_d = mol.intor("int1e_ovlp")
        C_c = np.asarray(core.mo_coeff, dtype=float)
        if C_c.ndim == 1:
            C_c = C_c.reshape(-1, 1)
        C_orth = _orthogonalize_core_ao_against_virt_ao(C_c, V_a, S_d)
        R0 = absorber_origin_bohr(core)
        dip_ao = _fragment_dipole_integrals(mol, R0)
        dip_check = core_virtual_dipole_matrix_gto(C_orth, V_a, dip_ao)
        dipole_v_delta = float(np.max(np.abs(dip - dip_check)))

    passed = (
        max_v_delta <= rtol
        and float(np.max(norm_delta)) <= rtol
    )

    return {
        "passed": bool(passed),
        "n_virt": int(n),
        "max_abs_V_duplicate_delta": max_v_delta,
        "max_abs_norm_delta": float(np.max(norm_delta)),
        "mean_abs_norm_delta": float(np.mean(norm_delta)),
        "max_abs_S_analytic_vs_grid_delta": float(np.max(comp_delta)),
        "mean_abs_S_analytic_vs_grid_delta": float(np.mean(comp_delta)),
        "n_completeness_clipped": int(np.sum(completeness_formula > 1.0 + 1e-8)),
        "max_abs_dipole_recompute_delta": dipole_v_delta,
        "completeness_analytic_S_mean": float(np.mean(completeness_formula)),
        "completeness_grid_S_mean": float(np.mean(completeness_grid_metric)),
        "rtol": float(rtol),
        "notes": (
            "V uses full-grid sum*dV; spillage uses analytic S^{-1}; dipole uses "
            "same V + analytic int1e_r (no regional mask, no DF metric). "
            "Large S_analytic vs S_grid spread means spillage absolute values "
            "are quadrature-sensitive; dipole/K share the same V regardless."
        ),
    }
def build_gto_kernel_from_projected_pw(
    mf,
    core,
    virtual_orbitals,
    unocc_eigs,
    grid,
    *,
    xc=None,
    projection_mol=None,
    virt_energy_shift_ha: float = 0.0,
    use_gpu: bool = False,
    use_mpi_response: bool = False,
    use_df: bool = True,
    k_cache_max: int = 4096,
    gs_pass: int = 2,
    normalize_virtuals: str = "s_metric",
    virt_norm_max_scale: Optional[float] = 1.0,
    virt_dipole_n_keep: Optional[int] = None,
    virt_min_abs_dipole: float = 0.0,
    comm=None,
    blksize: int = 16000,
    verbose: bool = False,
    projection_method: Optional[str] = None,
    becke_level: int = 4,
    driver=None,
    band_indices: Optional[Sequence[int]] = None,
    virtual_orbitals_for_dipole=None,
):
    """Build a :class:`~casidapy.kernels.gto.GTOKernel` with projected QE virtuals.

    PW grid orbitals are projected onto the Casida AO span (ghost-expanded
    only when ghost ``use_in_dipole=True``; otherwise the absorber fragment).
    Spillage audits still use the full ghost projection mol when attached.
    Core columns are projected out of the virtual AO span with the proper
    ``S``-metric projector ``P = V(VᵀSV)⁺VᵀS``.  Virtual energies are the QE
    band eigenvalues (plus optional rigid shift), not fragment SCF levels.

    ``projection_method``
        ``'becke'`` (default): atom-centered Becke quadrature + PW G-sum for
        ``ψ_a``.  ``'qe_grid'``: legacy uniform QE charge-mesh quadrature.
    ``normalize_virtuals``
        ``'s_metric'`` (default): column-wise ``‖c‖_S → 1``, but never amplify
        incomplete projections — default ``virt_norm_max_scale=1`` only
        *shrinks* columns with ``‖c‖_S > 1`` (ghost / non-ortho blow-up).
        ``'lowdin'``: full Löwdin of the virtual block (mixes bands; use with care).
        ``'none'``: leave ``C = S^{-1}v`` as-is.
    ``virt_norm_max_scale``
        Cap on the S-metric scale factor (default ``1``).  Pass ``None`` to
        allow full unit normalization (amplifies high-spillage virtuals).
    ``virtual_orbitals_for_dipole``
        Optional parallel list of the *unreconstructed* PW virtuals.  When
        GIPAW-augmented orbitals are projected for ``K``, pass the tilde
        fields here so ``paw_gipaw`` ``μ`` does not apply ``Δφ`` twice.
    """
    from casidapy.kernels.gto import GTOKernel
    from casidapy.xas.xas_utils import (
        grid_projection_to_ao_coeffs,
        normalize_ao_coeffs_s_metric,
    )

    mol = projection_mol if projection_mol is not None else core.fragment_mol
    spill_mol = mol

    abs_atom = None
    eidx = (getattr(core, "meta", None) or {}).get("edge_atom_indices")
    if eidx is not None and len(np.asarray(eidx).ravel()):
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
    if verbose:
        meta = (getattr(mol, "meta", None) or {}).get("pw_ao_projection")
        if meta:
            scr = meta.get("shell_screening") or {}
            print(
                f"  [PW→AO] method={meta.get('method')}  "
                f"npts={((meta.get('becke') or {}).get('n_pts'))}  "
                f"psi_pts={scr.get('n_pts_psi')}  "
                f"psi={(meta.get('psi') or {}).get('source')}  "
                f"r_cut=[{scr.get('r_cut_min')},{scr.get('r_cut_max')}]",
                flush=True,
            )
    unocc_e = np.asarray(unocc_eigs, dtype=float).ravel() + float(virt_energy_shift_ha)
    S = mol.intor("int1e_ovlp")
    C_v = grid_projection_to_ao_coeffs(mol, V_ao)
    norm_mode = str(normalize_virtuals).lower().strip()
    virt_norm_meta: dict = {"mode": norm_mode}
    if norm_mode in ("s_metric", "column", "normalize", "unit"):
        C_v, scales = normalize_ao_coeffs_s_metric(
            C_v, S, max_scale=virt_norm_max_scale
        )
        virt_norm_meta["scales_mean"] = float(np.mean(scales)) if scales.size else 0.0
        virt_norm_meta["scales_max"] = float(np.max(scales)) if scales.size else 0.0
    elif norm_mode in ("lowdin", "löwdin", "symmetric"):
        # Löwdin mixes bands: ε' = diag(Wᵀ diag(ε) W) on the kept subspace.
        C_raw = C_v
        e_raw = np.asarray(unocc_e, dtype=float).ravel()
        G = C_raw.T @ S @ C_raw
        G = 0.5 * (G + G.T)
        evals, evecs = np.linalg.eigh(G)
        thr = 1e-8 * max(float(evals.max()) if evals.size else 1.0, 1.0)
        keep = evals > thr
        if not np.any(keep):
            C_v, scales = normalize_ao_coeffs_s_metric(C_raw, S)
            virt_norm_meta["fallback"] = "s_metric"
            virt_norm_meta["scales_mean"] = float(np.mean(scales)) if scales.size else 0.0
        else:
            W = evecs[:, keep] * (1.0 / np.sqrt(evals[keep]))[None, :]
            C_v = C_raw @ W
            unocc_e = np.einsum("ik,i,ij->k", W, e_raw, W)
            virt_norm_meta["n_keep"] = int(W.shape[1])
            virt_norm_meta["n_drop"] = int(e_raw.size - W.shape[1])
            virt_norm_meta["G_eval_min"] = float(evals.min()) if evals.size else 0.0
    elif norm_mode not in ("none", "off", "raw", ""):
        raise ValueError(
            f"normalize_virtuals={normalize_virtuals!r}; use 's_metric', 'lowdin', or 'none'"
        )

    C_c = np.asarray(core.mo_coeff, dtype=float)
    C_c = _orthogonalize_core_ao_against_virt_ao(C_c, C_v, S, n_pass=gs_pass)
    xc_name = xc if xc is not None else getattr(mf, "xc", "pbe")
    mf_proj = mf

    # Placeholder active window from embedded mf (few fragment MOs); real
    # projected QE virtuals are injected via set_virtual_active_space below.
    kernel = GTOKernel(
        mol,
        np.asarray(mf_proj.mo_coeff, dtype=float),
        np.asarray(mf_proj.mo_energy, dtype=float),
        np.asarray(mf_proj.mo_occ, dtype=float),
        n_occ=1,
        n_unocc=1,
        xc=xc_name,
        use_df=use_df,
        mf=mf_proj,
        verbose=verbose,
        use_gpu=use_gpu,
        use_mpi_response=use_mpi_response,
        k_cache_max=int(k_cache_max),
    )
    kernel.set_core_active_space(core.energies, C_c)
    kernel.set_virtual_active_space(unocc_e, C_v)
    pw_virts = list(virtual_orbitals)
    pw_dip = (
        list(virtual_orbitals_for_dipole)
        if virtual_orbitals_for_dipole is not None
        else pw_virts
    )
    if len(pw_dip) != len(pw_virts):
        raise ValueError(
            "virtual_orbitals_for_dipole length "
            f"{len(pw_dip)} != virtual_orbitals {len(pw_virts)}"
        )
    if virt_dipole_n_keep is not None or float(virt_min_abs_dipole) > 0.0:
        from casidapy.xas.dipole import prune_kernel_virtuals_by_core_dipole

        _, _, idx = prune_kernel_virtuals_by_core_dipole(
            kernel,
            C_c,
            min_abs_dipole=float(virt_min_abs_dipole),
            n_keep=virt_dipole_n_keep,
            verbose=verbose,
        )
        if idx is not None:
            sel = [int(i) for i in np.asarray(idx).ravel()]
            pw_virts = [pw_virts[i] for i in sel]
            pw_dip = [pw_dip[i] for i in sel]
    kernel.dipole_origin = tuple(float(x) for x in absorber_origin_bohr(core))
    kernel._xas_pw_virtuals = pw_dip
    kernel._xas_pw_virtuals_k = pw_virts
    kernel._xas_host_grid = grid
    kernel._xas_ae_core_obj = core
    audit = audit_virtual_ao_spillage(
        core,
        virtual_orbitals,
        grid,
        projection_mol=spill_mol,
        comm=comm,
        blksize=blksize,
        verbose=verbose,
    )
    kernel._xas_spillage_audit = audit
    kernel._xas_virt_norm = virt_norm_meta
    if getattr(core, "meta", None) is not None:
        core.meta["gto_projected_coupling"] = {
            "n_ao": int(mol.nao_nr()),
            "n_ao_spill": int(spill_mol.nao_nr()),
            "spillage_mean": float(audit["summary"]["mean"]),
            "spillage_max": float(audit["summary"]["max"]),
            "ghost_projection": spill_mol is not core.fragment_mol,
            "ghost_in_casida": mol is not core.fragment_mol,
            "virt_norm": virt_norm_meta,
            "n_virt": int(kernel.n_unocc),
        }
    return kernel


def oscillator_strengths_at_origin(kernel, results, origin) -> np.ndarray:
    """``f_n`` from stored Casida ``Z`` and ``dipole_matrix(origin)``."""
    Z = np.asarray(results.Z, dtype=float)
    omega = np.asarray(results.omega, dtype=float).ravel()
    mu = kernel.dipole_matrix(origin=origin)
    d = mu.T @ Z
    return (2.0 / 3.0) * omega * np.sum(d * d, axis=0)


def recompute_oscillator_strengths_local_grid(
    results,
    core,
    virtual_orbitals,
    host_grid,
    *,
    verbose: bool = False,
    r_max=None,
    n_rad=None,
    n_ang=None,
    mapping: str = "log",
    psi_mode: str = "spline3",
):
    """Replace ``results.f`` / ``d_mode`` using dense centroid-grid ``μ`` (A/K unchanged).

    Casida eigenvectors ``Z`` stay from the host GTO/PW coupling path; only
    length-gauge intensities are rebuilt from ``⟨c|r−R₀|v⟩`` on the local
    1s-radius sphere.  Host virtuals default to tricubic sampling.
    """
    from casidapy.xas.dipole import (
        LOCAL_N_ANG,
        LOCAL_N_RAD,
        core_dipole_radius_bohr,
        core_virtual_dipole_matrix_local,
        core_charge_centroid_bohr,
    )

    virts = list(virtual_orbitals)
    n_virt = len(virts)
    Z = np.asarray(results.Z, dtype=float)
    omega = np.asarray(results.omega, dtype=float).ravel()
    n_states = int(omega.size)
    if Z.ndim != 2:
        raise ValueError(f"results.Z must be 2-D, got shape {Z.shape}")
    n_trans = int(Z.shape[0])
    n_core = max(n_trans // max(n_virt, 1), 1)
    if n_core * n_virt != n_trans:
        n_core = 1
        if n_virt != n_trans:
            raise ValueError(
                f"local-grid μ: n_trans={n_trans} incompatible with "
                f"n_virt={n_virt} (expected n_core*n_virt)"
            )
    R0 = core_charge_centroid_bohr(core, host_grid)
    r_use = r_max
    if r_use is None:
        r_use = core_dipole_radius_bohr(core, host_grid, R0)
    d_ov = core_virtual_dipole_matrix_local(
        core,
        virts,
        host_grid,
        origin=R0,
        gauge="length",
        r_max=float(r_use),
        n_rad=int(LOCAL_N_RAD if n_rad is None else n_rad),
        n_ang=int(LOCAL_N_ANG if n_ang is None else n_ang),
        mapping=mapping,
        psi_mode=psi_mode,
    )
    mu = np.sqrt(2.0) * np.stack(
        [d_ov[0].ravel(), d_ov[1].ravel(), d_ov[2].ravel()], axis=-1
    )
    if mu.shape[0] != n_trans:
        raise ValueError(
            f"local-grid μ rows {mu.shape[0]} != n_trans {n_trans}"
        )
    d_mode = mu.T @ Z[:, :n_states]
    f = (2.0 / 3.0) * omega * np.sum(d_mode * d_mode, axis=0)
    results.f = np.asarray(f, dtype=float)
    results.d_mode = np.asarray(d_mode, dtype=float)
    results.mu_transition = np.asarray(mu, dtype=float)
    meta = dict(getattr(results, "metadata", None) or {})
    meta["dipole_method"] = "local_grid"
    meta["dipole_r_max_bohr"] = float(r_use)
    meta["dipole_mapping"] = str(mapping)
    meta["dipole_psi_mode"] = str(psi_mode)
    meta["f_sum_local_grid"] = float(np.sum(f))
    results.metadata = meta
    if verbose:
        print(
            f"  [μ local_grid] rebuilt f  r_max={float(r_use):.3f} Bohr  "
            f"map={mapping}  psi={psi_mode}  "
            f"∑f={float(np.sum(f)):.6f}  n_virt={n_virt}  n_states={n_states}",
            flush=True,
        )
    return results


def recompute_oscillator_strengths_pw_grid(
    results,
    core,
    virtual_orbitals,
    host_grid,
    *,
    verbose: bool = False,
):
    """Replace ``results.f`` / ``d_mode`` using FFT-mesh ``⟨1s|r|ψ̃⟩``.

    AE 1s is mapped onto the QE charge grid; virtuals stay as stored PW
    fields.  No GTO projection and no local spherical quadrature.  Casida
    ``Z`` (A/K) is unchanged.
    """
    from casidapy.xas.dipole import (
        absorber_origin_bohr,
        core_virtual_dipole_matrix_pw_overlap,
    )

    virts = list(virtual_orbitals)
    n_virt = len(virts)
    Z = np.asarray(results.Z, dtype=float)
    omega = np.asarray(results.omega, dtype=float).ravel()
    n_states = int(omega.size)
    if Z.ndim != 2:
        raise ValueError(f"results.Z must be 2-D, got shape {Z.shape}")
    n_trans = int(Z.shape[0])
    n_core = max(n_trans // max(n_virt, 1), 1)
    if n_core * n_virt != n_trans:
        n_core = 1
        if n_virt != n_trans:
            raise ValueError(
                f"pw-grid μ: n_trans={n_trans} incompatible with "
                f"n_virt={n_virt} (expected n_core*n_virt)"
            )
    R0 = absorber_origin_bohr(core)
    d_ov = core_virtual_dipole_matrix_pw_overlap(
        core, virts, host_grid, origin=R0
    )
    mu = np.sqrt(2.0) * np.stack(
        [d_ov[0].ravel(), d_ov[1].ravel(), d_ov[2].ravel()], axis=-1
    )
    if mu.shape[0] != n_trans:
        raise ValueError(f"pw-grid μ rows {mu.shape[0]} != n_trans {n_trans}")
    d_mode = mu.T @ Z[:, :n_states]
    f = (2.0 / 3.0) * omega * np.sum(d_mode * d_mode, axis=0)
    results.f = np.asarray(f, dtype=float)
    results.d_mode = np.asarray(d_mode, dtype=float)
    results.mu_transition = np.asarray(mu, dtype=float)
    meta = dict(getattr(results, "metadata", None) or {})
    meta["dipole_method"] = "pw_grid"
    meta["f_sum_pw_grid"] = float(np.sum(f))
    results.metadata = meta
    if verbose:
        print(
            f"  [μ pw_grid] rebuilt f from FFT-mesh overlap  "
            f"∑f={float(np.sum(f)):.6f}  n_virt={n_virt}  n_states={n_states}",
            flush=True,
        )
    return results


def recompute_oscillator_strengths_paw_gipaw(
    results,
    core,
    virtual_orbitals,
    host_grid,
    gipaw_upf,
    *,
    verbose: bool = False,
    r_max=None,
    n_rad=None,
    n_ang=None,
    mapping: str = "log",
    psi_mode: str = "spline3",
    core_wfc: str = "ae",
    rcond: float = 1e-8,
):
    """Replace ``results.f`` using GIPAW-reconstructed ``⟨c|r|ψ_AE⟩``.

    Casida ``Z`` (A/K) is unchanged.  ``gipaw_upf`` must match the host
    absorber pseudopotential.
    """
    from casidapy.xas.dipole import (
        LOCAL_N_ANG,
        LOCAL_N_RAD,
        core_charge_centroid_bohr,
    )
    from casidapy.xas.gipaw import (
        core_virtual_dipole_matrix_paw_gipaw,
        parse_gipaw_upf,
    )

    gipaw = parse_gipaw_upf(gipaw_upf) if not hasattr(gipaw_upf, "valence_orbitals") else gipaw_upf
    virts = list(virtual_orbitals)
    n_virt = len(virts)
    Z = np.asarray(results.Z, dtype=float)
    omega = np.asarray(results.omega, dtype=float).ravel()
    n_states = int(omega.size)
    if Z.ndim != 2:
        raise ValueError(f"results.Z must be 2-D, got shape {Z.shape}")
    n_trans = int(Z.shape[0])
    n_core = max(n_trans // max(n_virt, 1), 1)
    if n_core * n_virt != n_trans:
        n_core = 1
        if n_virt != n_trans:
            raise ValueError(
                f"paw_gipaw μ: n_trans={n_trans} incompatible with "
                f"n_virt={n_virt} (expected n_core*n_virt)"
            )
    R0 = core_charge_centroid_bohr(core, host_grid)
    d_ov = core_virtual_dipole_matrix_paw_gipaw(
        core,
        virts,
        host_grid,
        gipaw,
        origin=R0,
        gauge="length",
        r_max=r_max,
        n_rad=int(LOCAL_N_RAD if n_rad is None else n_rad),
        n_ang=int(LOCAL_N_ANG if n_ang is None else n_ang),
        mapping=mapping,
        psi_mode=psi_mode,
        core_wfc=core_wfc,
        rcond=rcond,
    )
    mu = np.sqrt(2.0) * np.stack(
        [d_ov[0].ravel(), d_ov[1].ravel(), d_ov[2].ravel()], axis=-1
    )
    if mu.shape[0] != n_trans:
        raise ValueError(f"paw_gipaw μ rows {mu.shape[0]} != n_trans {n_trans}")
    d_mode = mu.T @ Z[:, :n_states]
    f = (2.0 / 3.0) * omega * np.sum(d_mode * d_mode, axis=0)
    results.f = np.asarray(f, dtype=float)
    results.d_mode = np.asarray(d_mode, dtype=float)
    results.mu_transition = np.asarray(mu, dtype=float)
    meta = dict(getattr(results, "metadata", None) or {})
    meta["dipole_method"] = "paw_gipaw"
    meta["gipaw_upf"] = str(getattr(gipaw, "path", gipaw_upf))
    meta["gipaw_r_c_bohr"] = float(gipaw.r_c)
    meta["dipole_mapping"] = str(mapping)
    meta["dipole_psi_mode"] = str(psi_mode)
    meta["f_sum_paw_gipaw"] = float(np.sum(f))
    results.metadata = meta
    if verbose:
        print(
            f"  [μ paw_gipaw] rebuilt f  r_c={float(gipaw.r_c):.3f} Bohr  "
            f"map={mapping}  psi={psi_mode}  "
            f"∑f={float(np.sum(f)):.6f}  n_virt={n_virt}  n_states={n_states}",
            flush=True,
        )
    return results


def check_dipole_origin_gauge(
    kernel,
    results,
    *,
    shift_bohr: float = 0.5,
    rtol: float = 0.05,
    strict: bool = False,
    verbose: bool = True,
) -> dict:
    """Warn or raise if ``∑f`` shifts when the length-gauge origin moves.

    Recomputes oscillator strengths from the solved ``Z`` at ``R₀`` and
    ``R₀ + shift·ê_x`` without re-solving CVS-TDA.
    """
    ae = getattr(kernel, "_xas_ae_core", None)
    base = np.asarray(
        ae.get("origin", getattr(kernel, "dipole_origin", (0.0, 0.0, 0.0)))
        if ae is not None
        else getattr(kernel, "dipole_origin", (0.0, 0.0, 0.0)),
        dtype=float,
    ).reshape(3)
    shifted = base + np.array([float(shift_bohr), 0.0, 0.0])

    f0 = oscillator_strengths_at_origin(kernel, results, tuple(base))
    f1 = oscillator_strengths_at_origin(kernel, results, tuple(shifted))
    s0 = float(np.sum(f0))
    s1 = float(np.sum(f1))
    m0 = float(np.max(f0)) if f0.size else 0.0
    m1 = float(np.max(f1)) if f1.size else 0.0
    rel_sum = abs(s1 - s0) / max(abs(s0), 1e-30)
    rel_max = abs(m1 - m0) / max(abs(m0), 1e-30)
    meta = {
        "origin_bohr": tuple(float(x) for x in base),
        "shift_bohr": float(shift_bohr),
        "f_sum_0": s0,
        "f_sum_shift": s1,
        "f_max_0": m0,
        "f_max_shift": m1,
        "rel_delta_f_sum": rel_sum,
        "rel_delta_f_max": rel_max,
        "rtol": float(rtol),
        "passed": rel_sum <= rtol and rel_max <= rtol,
    }
    if not meta["passed"]:
        msg = (
            f"length-gauge origin sensitivity: Δ∑f/∑f={rel_sum:.2%}, "
            f"Δf_max/f_max={rel_max:.2%} for +{shift_bohr} Bohr along x "
            f"(threshold {rtol:.0%})"
        )
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)
    elif verbose:
        logger.info(
            "dipole origin check passed: Δ∑f/∑f=%.2f%%, Δf_max/f_max=%.2f%%",
            100.0 * rel_sum,
            100.0 * rel_max,
        )
    return meta
