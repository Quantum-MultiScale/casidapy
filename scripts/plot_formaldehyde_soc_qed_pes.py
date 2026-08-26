#!/usr/bin/env python
"""Formaldehyde C=O stretch PES + excitation spectra.

PES (3 panels): bare TDA, SI-SOC, SOC+PF polaritons only.
Spectra (4 panels, separate PNG): TDDFT, TDDFT+SOC, QED-TDDFT, QED-TDDFT+SOC
at the SCF-minimum geometry.

Example
-------
    cd /projectsn/mp1009_1/am4655/qed-tests
    python plot_formaldehyde_soc_qed_pes.py --npoints 20 \\
        --out formaldehyde_pes.png --spectra-out formaldehyde_spectra.png
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

HA_TO_EV = 27.211386245988


def formaldehyde_atom(r_co: float) -> str:
    """H₂CO with C at origin, O along +z at ``r_co`` (Å); H's scaled with C=O."""
    r0 = 1.208
    scale = r_co / r0
    hy, hz = 0.943, -0.587 * scale
    return f"""
C  0.000000  0.000000  0.000000
O  0.000000  0.000000  {r_co:.6f}
H  0.000000  {hy:.6f}  {hz:.6f}
H  0.000000 {-hy:.6f}  {hz:.6f}
"""


def unique_energies(omega: np.ndarray, tol_ev: float = 0.02) -> np.ndarray:
    """Collapse near-degenerate roots (e.g. Cartesian T_x,T_y,T_z) for plotting."""
    w = np.asarray(omega, dtype=float)
    w = w[np.isfinite(w)]
    if w.size == 0:
        return w
    w = np.sort(w)
    out = [w[0]]
    for x in w[1:]:
        if abs(x - out[-1]) * HA_TO_EV > tol_ev:
            out.append(x)
    return np.asarray(out, dtype=float)


def polariton_indices(phot_qed: np.ndarray, omega_qed: np.ndarray, omega_c: float,
                      pf_min: float = 0.08, pf_max: float = 0.92,
                      window_ev: float = 3.0) -> np.ndarray:
    """Indices of PF roots with polariton character (mixed photon weight near ω_c)."""
    mean_pf = np.nanmean(phot_qed, axis=0)
    mean_exc = np.nanmean(omega_qed, axis=0)
    ok = np.isfinite(mean_exc) & np.isfinite(mean_pf) & (mean_exc * HA_TO_EV > 0.05)
    mixed = ok & (mean_pf >= pf_min) & (mean_pf <= pf_max)
    near = ok & (np.abs(mean_exc - omega_c) * HA_TO_EV < window_ev) & (mean_pf >= pf_min)
    idx = np.where(mixed | near)[0]
    if idx.size == 0:
        idx = np.where(ok & (mean_pf >= pf_min))[0]
    return idx[np.argsort(mean_exc[idx])]


def gaussian_spectrum(energies_ev, strengths, grid_ev, sigma_ev=0.12):
    """Unit-normalized Gaussian sticks → continuous spectrum."""
    spec = np.zeros_like(grid_ev, dtype=float)
    e = np.asarray(energies_ev, dtype=float).ravel()
    s = np.asarray(strengths, dtype=float).ravel()
    for ei, si in zip(e, s):
        if not np.isfinite(ei) or not np.isfinite(si) or si <= 0.0:
            continue
        spec += si * np.exp(-0.5 * ((grid_ev - ei) / sigma_ev) ** 2)
    return spec


def plot_spectra_figure(
    path,
    *,
    title,
    omega_c,
    omega_s,
    f_s,
    omega_t,
    omega_soc,
    f_soc,
    omega_qed_tda,
    f_qed_tda,
    phot_qed_tda,
    omega_qed_soc,
    phot_qed_soc,
    sigma_ev=0.12,
):
    """Four-panel Gaussian-broadened spectra PNG (no stick lines)."""
    import matplotlib.pyplot as plt

    def _pack(w, strength):
        w = np.asarray(w, dtype=float).ravel()
        s = np.asarray(strength, dtype=float).ravel()
        m = np.isfinite(w) & (w * HA_TO_EV > 0.05)
        return w[m] * HA_TO_EV, np.clip(s[m], 0.0, None)

    panels = []
    es, fs = _pack(omega_s, f_s)
    et, ft = _pack(omega_t, np.full(np.asarray(omega_t).shape, 0.08))
    panels.append(("TDDFT (TDA)", es, fs, et, ft))
    e_soc, fso = _pack(omega_soc, f_soc)
    panels.append(("TDDFT + SI-SOC", e_soc, fso, None, None))
    eq, fq = _pack(omega_qed_tda, f_qed_tda)
    panels.append(("QED-TDDFT", eq, fq, None, None))
    w_pf = np.asarray(omega_qed_soc, dtype=float).ravel()
    p_pf = np.asarray(phot_qed_soc, dtype=float).ravel()
    m = np.isfinite(w_pf) & (w_pf * HA_TO_EV > 0.05)
    e_pf = w_pf[m] * HA_TO_EV
    s_pf = np.clip(p_pf[m], 0.0, None)
    s_pf = np.where(s_pf > 0.05, s_pf, 0.02)
    panels.append(("QED-TDDFT + SOC", e_pf, s_pf, None, None))

    all_e = np.concatenate([p[1] for p in panels if p[1].size])
    if all_e.size == 0:
        all_e = np.array([omega_c * HA_TO_EV])
    lo = max(0.0, float(np.min(all_e)) - 1.0)
    hi = float(np.max(all_e)) + 1.5
    grid = np.linspace(lo, hi, 1600)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    axes = axes.ravel()
    colors = ["C0", "C2", "C1", "C3"]
    for ax, (label, e_ev, strength, e_t, s_t), col in zip(axes, panels, colors):
        if strength.size and strength.max() > 0:
            strength = strength / strength.max()
        spec = gaussian_spectrum(e_ev, strength, grid, sigma_ev=sigma_ev)
        if spec.max() > 0:
            spec = spec / spec.max()
        ax.fill_between(grid, 0.0, spec, color=col, alpha=0.30, linewidth=0)
        ax.plot(grid, spec, color=col, lw=2.0, label="spectrum")
        if e_t is not None and e_t.size:
            st = np.asarray(s_t, dtype=float)
            if st.max() > 0:
                st = st / st.max()
            spec_t = gaussian_spectrum(e_t, st, grid, sigma_ev=sigma_ev)
            if spec_t.max() > 0:
                spec_t = 0.35 * spec_t / spec_t.max()
            ax.plot(grid, spec_t, color="0.35", lw=1.5, ls="--", label="triplets")
        ax.axvline(omega_c * HA_TO_EV, color="k", ls=":", lw=1.1, alpha=0.85, label=r"$\omega_c$")
        ax.set_title(label)
        ax.set_ylabel("relative intensity")
        ax.set_ylim(0.0, 1.15)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right", frameon=False)

    for ax in axes[2:]:
        ax.set_xlabel("excitation energy (eV)")
    fig.suptitle(title, y=0.98)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.2)
    print(f"Wrote {path}")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--xc", default="pbe")
    p.add_argument("--npoints", type=int, default=7)
    p.add_argument("--r-min", type=float, default=1.10, help="C=O min (Å)")
    p.add_argument("--r-max", type=float, default=1.40, help="C=O max (Å)")
    p.add_argument("--n-singlet", type=int, default=3)
    p.add_argument("--n-triplet", type=int, default=3)
    p.add_argument("--n-plot", type=int, default=4, help="Max surfaces on TDA/SOC panels")
    p.add_argument("--lam", type=float, default=0.05)
    p.add_argument("--pol", default="z", choices=["x", "y", "z"])
    p.add_argument(
        "--omega-c-ev", type=float, default=None,
        help="Fixed cavity ω_c in eV (default: brightest singlet at first geometry)",
    )
    p.add_argument("--no-dse", action="store_true")
    p.add_argument("--out", type=str, default="formaldehyde_pes.png")
    p.add_argument(
        "--spectra-out", type=str, default=None,
        help="Spectra PNG (default: derive from --out)",
    )
    p.add_argument("--savez", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    from pyscf import gto, dft
    from casidapy import (
        extract_gto_kernel,
        run_casida,
        solve_soc_si,
        solve_soc_qed_pf,
        solve_qed_tda,
        QEDOptions,
    )
    import matplotlib.pyplot as plt

    spectra_out = args.spectra_out
    if spectra_out is None:
        root, ext = os.path.splitext(args.out)
        spectra_out = root.replace("_pes", "") + "_spectra" + (ext or ".png")

    pol = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[args.pol]
    rs = np.linspace(args.r_min, args.r_max, args.npoints)

    e_scf = np.zeros(args.npoints)
    n_s, n_t = args.n_singlet, args.n_triplet
    n_el = 1 + n_s + 3 * n_t
    n_qed = 2 * n_el
    n_soc_u = n_s + n_t
    n_soc_full = n_s + 3 * n_t
    n_qed_tda = n_s + 1

    omega_s = np.full((args.npoints, n_s), np.nan)
    f_s = np.full((args.npoints, n_s), np.nan)
    omega_t = np.full((args.npoints, n_t), np.nan)
    omega_soc_u = np.full((args.npoints, n_soc_u), np.nan)
    omega_soc = np.full((args.npoints, n_soc_full), np.nan)
    f_soc = np.full((args.npoints, n_soc_full), np.nan)
    omega_qed = np.full((args.npoints, n_qed), np.nan)
    phot_qed = np.full((args.npoints, n_qed), np.nan)
    omega_qed_tda = np.full((args.npoints, n_qed_tda), np.nan)
    f_qed_tda = np.full((args.npoints, n_qed_tda), np.nan)
    phot_qed_tda = np.full((args.npoints, n_qed_tda), np.nan)

    omega_c = None
    if args.omega_c_ev is not None:
        omega_c = float(args.omega_c_ev) / HA_TO_EV

    t_all = time.time()
    for ig, r in enumerate(rs):
        t0 = time.time()
        mol = gto.M(atom=formaldehyde_atom(r), basis=args.basis, verbose=0)
        mf = dft.RKS(mol)
        mf.xc = args.xc
        mf.grids.level = 1
        e_scf[ig] = mf.kernel()
        if not mf.converged:
            print(f"  warning: SCF not converged at r={r:.3f}")

        ks, opts_s = extract_gto_kernel(
            mf, n_states=n_s, tda=True, use_df=False, spin_state="singlet",
        )
        kt, opts_t = extract_gto_kernel(
            mf, n_states=n_t, tda=True, use_df=False, spin_state="triplet",
        )
        opts_s.solver_method = opts_t.solver_method = "eigsh"
        res_s = run_casida(ks, opts_s)
        res_t = run_casida(kt, opts_t)
        omega_s[ig, : len(res_s.omega)] = res_s.omega
        f_s[ig, : len(res_s.f)] = res_s.f
        omega_t[ig, : len(res_t.omega)] = res_t.omega

        if omega_c is None:
            k_bright = int(np.argmax(np.asarray(res_s.f, dtype=float)))
            omega_c = float(res_s.omega[k_bright])
            print(
                f"Fixing ω_c = {omega_c * HA_TO_EV:.3f} eV "
                f"to brightest singlet S{k_bright+1} "
                f"(f={res_s.f[k_bright]:.3e}) at first geometry"
            )

        soc = solve_soc_si(res_s, res_t, ks, include_ground=False)
        uniq = unique_energies(soc.omega)
        omega_soc_u[ig, : min(n_soc_u, uniq.size)] = uniq[:n_soc_u]
        nsoc = min(n_soc_full, soc.omega.size)
        omega_soc[ig, :nsoc] = soc.omega[:nsoc]
        f_soc[ig, :nsoc] = soc.f[:nsoc]

        qed_opts = QEDOptions(
            lam_scalar=args.lam,
            polarization=pol,
            omega_c=omega_c,
            nstates=n_qed_tda,
            include_dse=not args.no_dse,
            coherent_state=True,
        )
        qed_tda = solve_qed_tda(ks, options=qed_opts)
        nqt = min(n_qed_tda, qed_tda.omega.size)
        omega_qed_tda[ig, :nqt] = qed_tda.omega[:nqt]
        f_qed_tda[ig, :nqt] = qed_tda.f[:nqt]
        phot_qed_tda[ig, :nqt] = qed_tda.photon_frac[:nqt]

        qed = solve_soc_qed_pf(
            soc,
            lam_vec=np.asarray(pol, float) * args.lam,
            omega_c=omega_c,
            nstates=None,
            prefer_bright=False,
            include_ground_slot=True,
            include_dse=not args.no_dse,
        )
        nq = min(n_qed, len(qed["omega"]))
        omega_qed[ig, :nq] = qed["omega"][:nq]
        phot_qed[ig, :nq] = qed["photon_frac"][:nq]

        print(
            f"r={r:.3f} Å  E_SCF={e_scf[ig]:.6f}  "
            f"S1={res_s.omega[0]*HA_TO_EV:.2f} eV  "
            f"T1={res_t.omega[0]*HA_TO_EV:.2f} eV  "
            f"({time.time()-t0:.1f}s)"
        )

    print(f"Scan done in {time.time()-t_all:.1f}s")

    e0 = float(np.nanmin(e_scf))
    e_scf_rel = (e_scf - e0) * HA_TO_EV
    tot_s = e_scf_rel[:, None] + omega_s * HA_TO_EV
    tot_t = e_scf_rel[:, None] + omega_t * HA_TO_EV
    tot_soc = e_scf_rel[:, None] + omega_soc_u * HA_TO_EV
    tot_qed = e_scf_rel[:, None] + omega_qed * HA_TO_EV
    e_photon = e_scf_rel + omega_c * HA_TO_EV

    # ----- PES figure -----
    n_plot = args.n_plot
    ylab = r"$E - E_\mathrm{SCF,min}$ (eV)"
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=False)

    ax = axes[0]
    ax.plot(rs, e_scf_rel, "k--", lw=1.3, label="S0 (SCF)")
    for k in range(min(n_plot, n_s)):
        ax.plot(rs, tot_s[:, k], "-", lw=1.5, marker="o", ms=3.0, label=f"S{k+1}")
    for k in range(min(2, n_t)):
        ax.plot(rs, tot_t[:, k], ":", lw=1.4, marker="s", ms=3.0, alpha=0.9, label=f"T{k+1}")
    ax.set_title("TDDFT (TDA)")
    ax.legend(fontsize=7, loc="best", frameon=False)

    ax = axes[1]
    ax.plot(rs, e_scf_rel, "k--", lw=1.3, label="S0 (SCF)")
    for k in range(min(n_plot, n_soc_u)):
        if np.all(np.isnan(tot_soc[:, k])):
            continue
        ax.plot(rs, tot_soc[:, k], "-", lw=1.5, marker="o", ms=3.0, label=f"SOC{k+1}")
    ax.set_title("TDDFT + SI-SOC")
    ax.legend(fontsize=7, loc="best", frameon=False)

    ax = axes[2]
    ax.plot(rs, e_scf_rel, "k--", lw=1.3, label="S0 (SCF)")
    ax.plot(rs, e_photon, color="0.45", ls=":", lw=1.4, label=r"S0 + $\omega_c$")
    pidx = polariton_indices(phot_qed, omega_qed, omega_c)
    for j, k in enumerate(pidx):
        if k >= n_qed or np.all(np.isnan(tot_qed[:, k])):
            continue
        ax.plot(
            rs, tot_qed[:, k], "-", lw=1.8, marker=None,
            label=f"polariton {j+1}",
        )
    dse_tag = "no DSE" if args.no_dse else "DSE on"
    ax.set_title(f"SOC + PF-QED (λ={args.lam}, {dse_tag})\npolaritons only")
    ax.legend(fontsize=7, loc="best", frameon=False)

    for ax in axes:
        ax.set_xlabel(r"$r(\mathrm{C=O})$ (Å)")
        ax.grid(True, alpha=0.3)

    y0 = min(ax.get_ylim()[0] for ax in axes)
    y1 = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(y0, y1)

    fig.suptitle(
        f"Formaldehyde PES — {args.xc}/{args.basis}, ω_c={omega_c*HA_TO_EV:.2f} eV",
        y=0.98,
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.84, wspace=0.42)
    for ax in axes:
        ax.set_ylabel(ylab)
        ax.tick_params(axis="y", labelleft=True)
    fig.savefig(args.out, dpi=160, pad_inches=0.2)
    print(f"Wrote {args.out}")

    # ----- Spectra at SCF minimum -----
    ig0 = int(np.nanargmin(e_scf))
    plot_spectra_figure(
        spectra_out,
        title=(
            f"Formaldehyde spectra @ r={rs[ig0]:.3f} Å — "
            f"{args.xc}/{args.basis}, ω_c={omega_c*HA_TO_EV:.2f} eV, λ={args.lam}"
        ),
        omega_c=omega_c,
        omega_s=omega_s[ig0],
        f_s=f_s[ig0],
        omega_t=omega_t[ig0],
        omega_soc=omega_soc[ig0],
        f_soc=f_soc[ig0],
        omega_qed_tda=omega_qed_tda[ig0],
        f_qed_tda=f_qed_tda[ig0],
        phot_qed_tda=phot_qed_tda[ig0],
        omega_qed_soc=omega_qed[ig0],
        phot_qed_soc=phot_qed[ig0],
    )

    if args.savez:
        np.savez(
            args.savez,
            r_co=rs,
            e_scf=e_scf,
            omega_s=omega_s,
            f_s=f_s,
            omega_t=omega_t,
            omega_soc_unique=omega_soc_u,
            omega_soc=omega_soc,
            f_soc=f_soc,
            omega_qed=omega_qed,
            photon_frac_qed=phot_qed,
            omega_qed_tda=omega_qed_tda,
            f_qed_tda=f_qed_tda,
            photon_frac_qed_tda=phot_qed_tda,
            omega_c=omega_c,
            lam=args.lam,
            xc=args.xc,
            basis=args.basis,
            model="pauli-fierz",
            include_dse=not args.no_dse,
        )
        print(f"Wrote {args.savez}")


if __name__ == "__main__":
    main()
