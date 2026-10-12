"""Pauli–Fierz (PF) and Jaynes–Cummings (JC) polariton post-processing.

Backend-agnostic cavity-QED post-processing of finished Casida/TDDFT results.
It couples a single cavity mode to the **electronic eigenstates** produced by any
CasidaPy backend (in particular the plane-wave solver) and reports the resulting
polariton branches under two models:

* **PF** — truncated Pauli–Fierz (electronic ⊗ {0,1} photon + optional dipole
  self-energy), via :func:`casidapy.utils.qed.solve_qed_pf_post`.
* **JC/TC** — few-level Jaynes–Cummings (one bright root) / Tavis–Cummings
  (several roots), via :func:`casidapy.utils.qed.solve_qed_levels`.

Both solvers already exist and take plain arrays; this module is the shared core
+ CLI that feeds them Casida outputs, sweeps the coupling, and saves/plots the
branches. It is used by the standalone ``casidapy-qed-post`` entry point and by
the ``--qed-post`` path of ``casidapy-run`` (same numbers from either route).

All internal math is in atomic units (energies in Hartree, dipoles in a.u.,
coupling vector ``lambda`` in a.u.); energies are converted to eV only at the
save/plot boundary and for CLI ``--omega-c``.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional, Sequence

import numpy as np

from casidapy.utils.qed import (
    solve_qed_levels,
    solve_qed_pf_post,
    solve_qed_pf_full,
    solve_qed_tddft_state,
    state_basis_dipole_matrix,
)

HA2EV = 27.211386245988

# Model aliases -> canonical kind used internally.
_MODEL_ALIASES = {
    "pf": "pf",
    "pauli": "pf",
    "pauli-fierz": "pf",
    "jc": "jc",
    "jaynes": "jc",
    "jaynes-cummings": "jc",
    "tc": "tc",
    "tavis": "tc",
    "tavis-cummings": "tc",
    # excited-state-dipole PF (full state-basis dipole matrix) and full QED-TDDFT
    "pff": "pff",
    "pf-full": "pff",
    "full-dipole": "pff",
    "tddft": "tddft",
    "qed-tddft": "tddft",
}
# models that need the Casida eigenvectors Z + MO dipole blocks (not just d_mode)
_MODELS_NEED_BLOCKS = frozenset({"pff"})


def _canon_model(model: str) -> str:
    key = str(model).strip().lower().replace("_", "-")
    if key not in _MODEL_ALIASES:
        raise ValueError(
            f"Unknown qed-post model {model!r}; expected one of "
            f"{sorted(set(_MODEL_ALIASES))}."
        )
    return _MODEL_ALIASES[key]


def _as_state_dipoles(mu_states: np.ndarray) -> np.ndarray:
    """Coerce transition dipoles to ``(n_states, 3)`` (accepts ``(3, n)``)."""
    d = np.asarray(mu_states, dtype=float)
    if d.ndim != 2:
        raise ValueError(f"mu_states must be 2-D, got shape {d.shape}")
    if d.shape[1] == 3:
        return d.copy()
    if d.shape[0] == 3:
        return d.T.copy()
    raise ValueError(f"mu_states shape {d.shape} is not (n,3) or (3,n)")


def make_lambda_vectors(
    lam_vec: Optional[Sequence[float]] = None,
    *,
    sweep: Optional[Sequence[float]] = None,
    pol: Sequence[float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Build the ``(n_lambda, 3)`` set of coupling vectors to scan.

    ``lam_vec`` gives an explicit single vector; ``sweep`` gives a list of scalar
    magnitudes applied along the unit polarization ``pol``. At least one must be
    provided; both may be combined.
    """
    vecs = []
    if lam_vec is not None:
        v = np.asarray(lam_vec, dtype=float).ravel()
        if v.shape != (3,):
            raise ValueError(f"lam_vec must have shape (3,), got {v.shape}")
        vecs.append(v)
    if sweep is not None:
        p = np.asarray(pol, dtype=float).ravel()
        if p.shape != (3,):
            raise ValueError(f"pol must have shape (3,), got {p.shape}")
        norm = np.linalg.norm(p)
        if norm < 1e-30:
            raise ValueError("pol direction must be non-zero")
        p = p / norm
        for mag in np.asarray(sweep, dtype=float).ravel():
            vecs.append(float(mag) * p)
    if not vecs:
        raise ValueError("Provide lam_vec and/or sweep to define coupling vectors.")
    return np.asarray(vecs, dtype=float)


def _resonant_omega_c(omega: np.ndarray, f: Optional[np.ndarray]) -> float:
    """Cavity frequency (Ha) resonant with the brightest excitation."""
    w = np.asarray(omega, dtype=float).ravel()
    bright = w > 1e-8  # skip the (numerical) ground root
    if f is not None:
        fa = np.asarray(f, dtype=float).ravel()
        cand = np.where(bright, fa, -np.inf)
        k = int(np.argmax(cand))
        if np.isfinite(cand[k]) and cand[k] > 0.0:
            return float(w[k])
    # No oscillator strengths: fall back to the lowest non-zero excitation.
    idx = np.where(bright)[0]
    if idx.size == 0:
        raise ValueError("No non-zero excitation energies to set omega_c.")
    return float(w[idx[0]])


def run_qed_post_on_coupled(
    coupled_result: Dict[str, Any],
    *,
    models: Sequence[str] = ("pf", "jc"),
    omega_c: Optional[float] = None,
    lam_vec: Optional[Sequence[float]] = None,
    lam_sweep: Optional[Sequence[float]] = None,
    pol: Sequence[float] = (0.0, 0.0, 1.0),
    nstates: Optional[int] = None,
    include_dse: bool = True,
) -> Dict[str, Any]:
    """Add cavity-QED post-processing to a **subsystem-coupled** LR-TDDFT result.

    Consumes the merged/coupled result dict from
    :mod:`casidapy.subsystem_coupling` (``compute_coupling_block`` and its
    distributed / far-field variants), which now carries ``omega``, ``f`` and
    ``d_mode`` (the coupled state-basis transition dipoles). Runs the same PF /
    JC / TC / full-QED-TDDFT post-processing used for a single-system Casida run.

    Note: the excited-state-dipole model ``"pff"`` needs the full state-basis
    dipole matrix ``D_full``. For a coupled result this is assembled by
    :func:`casidapy.subsystem_coupling.coupled_state_dipole_matrix` (per-fragment
    full dipole matrices rotated into the coupled basis; cross-fragment position
    matrix elements vanish in the Frenkel diabatic basis). When
    ``coupled_result['D_full']`` is present it is passed through and ``pff`` works;
    otherwise only ``pf``/``jc``/``tc``/``tddft`` are available.

    MPI: the distributed subsystem coupling returns the coupled dict on rank 0
    and ``None`` on all other ranks. This function mirrors that — ``None`` in,
    ``None`` out — so it is safe to call collectively; the QED post (pure NumPy,
    no collectives) then runs only on the rank that holds the coupled spectrum.
    """
    if coupled_result is None:
        return None
    if "d_mode" not in coupled_result or np.size(coupled_result["d_mode"]) == 0:
        raise ValueError(
            "coupled_result has no 'd_mode' (coupled transition dipoles). Rebuild "
            "the subsystem coupling with casidapy >= this version, or ensure the "
            "fragments carry 'dip_tran'."
        )
    D_full = coupled_result.get("D_full")
    return run_qed_post(
        np.asarray(coupled_result["omega"], dtype=float),
        np.asarray(coupled_result["d_mode"], dtype=float),
        f=None if coupled_result.get("f") is None else np.asarray(coupled_result["f"], dtype=float),
        models=models, omega_c=omega_c, lam_vec=lam_vec, lam_sweep=lam_sweep,
        pol=pol, nstates=nstates, include_dse=include_dse,
        D_full=None if D_full is None else np.asarray(D_full, dtype=float),
    )


def run_qed_post(
    omega: np.ndarray,
    mu_states: np.ndarray,
    *,
    f: Optional[np.ndarray] = None,
    models: Sequence[str] = ("pf", "jc"),
    omega_c: Optional[float] = None,
    lam_vecs: Optional[np.ndarray] = None,
    lam_vec: Optional[Sequence[float]] = None,
    lam_sweep: Optional[Sequence[float]] = None,
    pol: Sequence[float] = (0.0, 0.0, 1.0),
    nstates: Optional[int] = None,
    include_dse: bool = True,
    Z: Optional[np.ndarray] = None,
    d_oo: Optional[np.ndarray] = None,
    d_vv: Optional[np.ndarray] = None,
    d_ov: Optional[np.ndarray] = None,
    mu0: Sequence[float] = (0.0, 0.0, 0.0),
    D_full: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run PF and/or JC/TC polariton post-processing over a coupling sweep.

    Models: ``pf`` (S₀↔excited PF), ``jc``/``tc`` (Jaynes/Tavis-Cummings),
    ``tddft`` (full QED-TDDFT in the eigenbasis — asymmetric ω² branches), and
    ``pff`` (PF with the **full** state-basis dipole matrix, i.e. excited-state
    dipoles). ``pff`` requires ``Z`` (Casida eigenvectors) and the MO dipole
    blocks ``d_oo/d_vv/d_ov``; ``mu0`` is the ground dipole (default 0 =
    coherent-state frame, origin-independent).

    Parameters
    ----------
    omega, f
        Excitation energies (Hartree) and oscillator strengths from a Casida run.
    mu_states
        S₀→state transition dipoles, ``(n_states, 3)`` or ``(3, n_states)`` (a.u.).
    models
        Any of ``"pf"``, ``"jc"``, ``"tc"``.
    omega_c
        Cavity frequency (Hartree). ``None`` -> auto-resonant with the brightest
        excitation (:func:`_resonant_omega_c`).
    lam_vecs
        Explicit ``(n_lambda, 3)`` coupling vectors. If ``None`` they are built
        from ``lam_vec``/``lam_sweep``/``pol`` via :func:`make_lambda_vectors`.
    nstates
        Cap on electronic roots (PF manifold / TC roots). ``None`` keeps all;
        JC always uses the single brightest root.

    Returns
    -------
    dict
        ``{"omega_c", "models", "lam_vecs", "results": {model: [per-λ dict, ...]}}``
        where each per-λ dict is the raw solver output augmented with
        ``lam_vec``/``lam_mag`` and eV copies (``omega_ev``).
    """
    w = np.asarray(omega, dtype=float).ravel()
    fa = None if f is None else np.asarray(f, dtype=float).ravel()
    mu = _as_state_dipoles(mu_states)
    if mu.shape[0] != w.size:
        raise ValueError(
            f"mu_states has {mu.shape[0]} states but omega has {w.size}"
        )

    kinds = [_canon_model(m) for m in models]
    if omega_c is None:
        omega_c = _resonant_omega_c(w, fa)
    omega_c = float(omega_c)

    if lam_vecs is None:
        lam_vecs = make_lambda_vectors(lam_vec, sweep=lam_sweep, pol=pol)
    else:
        lam_vecs = np.asarray(lam_vecs, dtype=float)
        if lam_vecs.ndim == 1:
            lam_vecs = lam_vecs[None, :]

    # Full state-basis dipole matrix (built once; λ-independent) for the "pff"
    # model. Either supplied prebuilt (e.g. the subsystem-coupled D_full from
    # casidapy.subsystem_coupling.coupled_state_dipole_matrix) or assembled here
    # from the MO dipole blocks of a single system.
    if D_full is not None:
        D_full = np.asarray(D_full, dtype=float)
    if any(k in _MODELS_NEED_BLOCKS for k in kinds) and D_full is None:
        if Z is None or d_oo is None or d_vv is None or d_ov is None:
            raise ValueError(
                "model 'pff' (excited-state-dipole PF) needs either a prebuilt "
                "D_full or Z and d_oo/d_vv/d_ov."
            )
        D_full = state_basis_dipole_matrix(
            np.asarray(Z), np.asarray(d_oo), np.asarray(d_vv), np.asarray(d_ov),
            mu0=mu0, nstates=nstates,
        )

    results: Dict[str, list] = {k: [] for k in kinds}
    for lam in lam_vecs:
        for kind in kinds:
            if kind == "pf":
                out = solve_qed_pf_post(
                    w, mu, lam, omega_c, f=fa, nstates=nstates,
                    include_dse=include_dse,
                )
            elif kind == "jc":
                out = solve_qed_levels(
                    w, mu, lam, omega_c, f=fa, nstates=1, prefer_bright=True,
                )
            elif kind == "tc":
                out = solve_qed_levels(
                    w, mu, lam, omega_c, f=fa, nstates=nstates, prefer_bright=True,
                )
            elif kind == "tddft":
                out = solve_qed_tddft_state(w, mu, lam, omega_c, f=fa)
            elif kind == "pff":
                out = solve_qed_pf_full(w, D_full, lam, omega_c, include_dse=include_dse)
            else:  # pragma: no cover
                raise ValueError(f"unhandled qed-post model {kind!r}")
            out = dict(out)
            out["lam_vec"] = np.asarray(lam, dtype=float)
            out["lam_mag"] = float(np.linalg.norm(lam))
            out["omega_ev"] = np.asarray(out["omega"], dtype=float) * HA2EV
            out["omega_c_ev"] = omega_c * HA2EV
            results[kind].append(out)

    return {
        "omega_c": omega_c,
        "omega_c_ev": omega_c * HA2EV,
        "models": kinds,
        "lam_vecs": lam_vecs,
        "results": results,
    }


def broaden_spectrum(
    energies_ev: np.ndarray,
    weights: np.ndarray,
    grid_ev: np.ndarray,
    gamma_ev: float = 0.05,
) -> np.ndarray:
    """Lorentzian-broadened stick spectrum on ``grid_ev`` (reuses handler kernel)."""
    from casidapy.polariton_handler import lorentzian

    e = np.asarray(energies_ev, dtype=float).ravel()
    wts = np.asarray(weights, dtype=float).ravel()
    g = np.asarray(grid_ev, dtype=float).ravel()
    spec = np.zeros_like(g)
    for e0, wt in zip(e, wts):
        if wt <= 0.0 or e0 <= 0.0:
            continue
        spec += wt * lorentzian(g, e0, gamma_ev)
    return spec


def save_qed_post_results(
    data: Dict[str, Any],
    output_base: str,
    *,
    bare_omega_ev: Optional[np.ndarray] = None,
    bare_f: Optional[np.ndarray] = None,
    gamma_pol_ev: float = 0.05,
    npts: int = 2000,
    plot: bool = True,
) -> None:
    """Persist per-model / per-λ polariton branches (.npy), a summary, and a plot."""
    results = data["results"]
    lam_vecs = np.asarray(data["lam_vecs"], dtype=float)
    omega_c_ev = float(data["omega_c_ev"])

    # Per-model, per-lambda arrays.
    for kind, per_lambda in results.items():
        for il, out in enumerate(per_lambda):
            tag = f"{output_base}_{kind}_lambda{il}"
            np.save(f"{tag}_energies_ev.npy", np.asarray(out["omega"]) * HA2EV)
            np.save(f"{tag}_photon_frac.npy", np.asarray(out["photon_frac"]))
            np.save(f"{tag}_osc_strength.npy", np.asarray(out["f"]))
    np.save(f"{output_base}_lambda_vectors.npy", lam_vecs)

    # Summary text.
    with open(f"{output_base}_qed_post_summary.txt", "w") as fout:
        fout.write("# QED post-processing (Pauli-Fierz / Jaynes-Cummings)\n")
        fout.write(f"# cavity_energy_wc_eV: {omega_c_ev:.6f}\n")
        fout.write(f"# models: {','.join(results.keys())}\n")
        for kind, per_lambda in results.items():
            for il, out in enumerate(per_lambda):
                lam_mag = float(out["lam_mag"])
                e_ev = np.asarray(out["omega"]) * HA2EV
                pf = np.asarray(out["photon_frac"])
                osc = np.asarray(out["f"])
                fout.write(
                    f"#\n# model={out.get('model', kind)}  lambda_idx={il}  "
                    f"|lambda|={lam_mag:.6e}\n"
                )
                fout.write("# branch  energy_eV  photon_frac  osc_strength\n")
                for i, e in enumerate(e_ev):
                    fout.write(f"{i:4d}  {e:10.5f}  {pf[i]:11.6f}  {osc[i]:14.6f}\n")

    if not plot:
        return
    try:
        _plot_qed_post(
            data, output_base, bare_omega_ev=bare_omega_ev, bare_f=bare_f,
            gamma_pol_ev=gamma_pol_ev, npts=npts,
        )
    except ImportError:
        print("    Warning: matplotlib not available, skipping QED post plot")


def _plot_qed_post(
    data: Dict[str, Any],
    output_base: str,
    *,
    bare_omega_ev: Optional[np.ndarray],
    bare_f: Optional[np.ndarray],
    gamma_pol_ev: float,
    npts: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = data["results"]
    omega_c_ev = float(data["omega_c_ev"])
    lam_vecs = np.asarray(data["lam_vecs"], dtype=float)
    lam_mags = np.linalg.norm(lam_vecs, axis=1)

    # Energy window from all branches (+ bare).
    all_e = [omega_c_ev]
    for per_lambda in results.values():
        for out in per_lambda:
            all_e.append(np.max(np.asarray(out["omega"]) * HA2EV))
    if bare_omega_ev is not None and np.size(bare_omega_ev):
        all_e.append(np.max(np.asarray(bare_omega_ev)))
    e_max = float(max(all_e)) * 1.15 + 1.0
    grid = np.linspace(0.0, e_max, int(npts))

    n_models = len(results)
    fig, axes = plt.subplots(n_models, 2, figsize=(12, 4.2 * n_models), squeeze=False)

    for row, (kind, per_lambda) in enumerate(results.items()):
        # Left: broadened spectra at the strongest lambda (+ bare reference).
        ax = axes[row][0]
        if bare_omega_ev is not None and bare_f is not None:
            ax.plot(
                grid,
                broaden_spectrum(bare_omega_ev, bare_f, grid, gamma_pol_ev),
                color="0.5", lw=1.2, label="bare (no cavity)",
            )
        il_last = int(np.argmax(lam_mags)) if lam_mags.size else 0
        out = per_lambda[il_last]
        ax.plot(
            grid,
            broaden_spectrum(np.asarray(out["omega"]) * HA2EV, out["f"], grid, gamma_pol_ev),
            color="C0", lw=1.6,
            label=f"{out.get('model', kind)} |λ|={out['lam_mag']:.3g}",
        )
        ax.axvline(omega_c_ev, color="C3", ls="--", lw=1.0, label=f"ω_c={omega_c_ev:.3f} eV")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Absorption (arb.)")
        ax.set_title(f"{kind.upper()} polariton spectrum")
        ax.set_xlim(0.0, e_max)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Right: branch energies vs |lambda| (bright branches only).
        ax = axes[row][1]
        for out in per_lambda:
            e_ev = np.asarray(out["omega"]) * HA2EV
            osc = np.asarray(out["f"])
            bright = osc > (0.02 * (np.max(osc) if np.max(osc) > 0 else 1.0))
            ax.scatter(
                np.full(np.count_nonzero(bright), out["lam_mag"]),
                e_ev[bright], s=18, color="C0", alpha=0.8,
            )
        ax.axhline(omega_c_ev, color="C3", ls="--", lw=1.0, label=f"ω_c={omega_c_ev:.3f} eV")
        ax.set_xlabel("|λ| (a.u.)")
        ax.set_ylabel("Polariton energy (eV)")
        ax.set_title(f"{kind.upper()} branches vs coupling")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{output_base}_qed_post.png", dpi=150)
    plt.close(fig)
    print(f"    Saved: {output_base}_qed_post.png")


# --------------------------------------------------------------------------- #
# Standalone CLI
# --------------------------------------------------------------------------- #
def _load_inputs(prefix: str):
    """Load ``omega``, ``f``, and S₀→state dipoles from saved Casida artifacts."""
    omega = np.load(f"{prefix}_omega.npy")
    try:
        f = np.load(f"{prefix}_f.npy")
    except FileNotFoundError:
        f = None

    # Prefer d_mode (state-basis dipoles). Fall back to mu_transition + xpy.
    try:
        d_mode = np.load(f"{prefix}_d_mode.npy")
        mu_states = _as_state_dipoles(d_mode)
    except FileNotFoundError:
        try:
            mu_ov = np.load(f"{prefix}_mu_transition.npy")
            xpy = np.load(f"{prefix}_xpy.npy")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Need {prefix}_d_mode.npy, or both {prefix}_mu_transition.npy "
                f"and {prefix}_xpy.npy, for QED post-processing."
            ) from exc
        mu_states = np.asarray(xpy).T @ np.asarray(mu_ov)

    # Optional: Casida eigenvectors + MO dipole blocks (needed for model 'pff').
    blocks = {}
    for key, suffix in (("Z", "_Z"), ("d_oo", "_mo_dip_oo"),
                        ("d_vv", "_mo_dip_vv"), ("d_ov", "_mo_dip_ov")):
        try:
            blocks[key] = np.load(f"{prefix}{suffix}.npy")
        except FileNotFoundError:
            blocks[key] = None
    return omega, f, mu_states, blocks


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pauli-Fierz / Jaynes-Cummings polariton post-processing of "
                    "Casida results (plane-wave or any backend).",
    )
    parser.add_argument("--prefix", required=True,
                        help="Input prefix: reads {prefix}_omega.npy, _f.npy, and "
                             "_d_mode.npy (or _mu_transition.npy + _xpy.npy).")
    parser.add_argument("--model", default="pf,jc",
                        help="Comma-separated models: pf,jc,tc,tddft,pff (default: pf,jc). "
                             "'tddft'=full QED-TDDFT; 'pff'=PF with excited-state dipoles "
                             "(needs {prefix}_Z.npy + _mo_dip_{oo,vv,ov}.npy).")
    parser.add_argument("--mu0", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        metavar=("MX", "MY", "MZ"),
                        help="Ground-state dipole (a.u.) for 'pff' diagonal; default "
                             "0 0 0 = coherent-state frame (origin-independent).")
    parser.add_argument("--omega-c", default="auto",
                        help="Cavity energy in eV, or 'auto' for resonance with the "
                             "brightest excitation (default: auto).")
    parser.add_argument("--lambda", dest="lam", type=float, nargs=3, default=None,
                        metavar=("LX", "LY", "LZ"),
                        help="Explicit coupling vector lambda (a.u.).")
    parser.add_argument("--lambda-sweep", dest="lam_sweep", type=str, default=None,
                        help="Comma-separated |lambda| magnitudes (a.u.) along --pol.")
    parser.add_argument("--pol", type=float, nargs=3, default=[0.0, 0.0, 1.0],
                        metavar=("PX", "PY", "PZ"),
                        help="Polarization direction for --lambda-sweep (default: 0 0 1).")
    parser.add_argument("--nstates", type=int, default=None,
                        help="Cap on electronic roots (PF manifold / TC). Default: all.")
    parser.add_argument("--no-dse", action="store_true",
                        help="Disable the dipole self-energy term in PF.")
    parser.add_argument("--gamma-pol", type=float, default=0.05,
                        help="Lorentzian broadening (eV) for plotted spectra.")
    parser.add_argument("--npts", type=int, default=2000,
                        help="Grid points for broadened spectra.")
    parser.add_argument("--output-base", default=None,
                        help="Output prefix (default: {prefix}_qed).")
    parser.add_argument("--no-plot", action="store_true", help="Skip the plot.")
    args = parser.parse_args(argv)

    omega, f, mu_states, blocks = _load_inputs(args.prefix)

    omega_c = None
    if str(args.omega_c).strip().lower() != "auto":
        omega_c = float(args.omega_c) / HA2EV

    lam_sweep = None
    if args.lam_sweep:
        lam_sweep = [float(x) for x in args.lam_sweep.split(",") if x.strip()]
    if args.lam is None and lam_sweep is None:
        parser.error("provide --lambda and/or --lambda-sweep")

    models = [m for m in args.model.split(",") if m.strip()]
    data = run_qed_post(
        omega, mu_states, f=f, models=models, omega_c=omega_c,
        lam_vec=args.lam, lam_sweep=lam_sweep, pol=args.pol,
        nstates=args.nstates, include_dse=not args.no_dse,
        Z=blocks.get("Z"), d_oo=blocks.get("d_oo"),
        d_vv=blocks.get("d_vv"), d_ov=blocks.get("d_ov"), mu0=args.mu0,
    )

    output_base = args.output_base or f"{args.prefix}_qed"
    save_qed_post_results(
        data, output_base, bare_omega_ev=np.asarray(omega) * HA2EV,
        bare_f=f, gamma_pol_ev=args.gamma_pol, npts=args.npts,
        plot=not args.no_plot,
    )
    print(f"QED post-processing complete. ω_c = {data['omega_c_ev']:.4f} eV; "
          f"models = {','.join(data['models'])}; "
          f"λ points = {len(data['lam_vecs'])}. Prefix: {output_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
