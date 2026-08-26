#!/usr/bin/env python
"""Pauli–Fierz QED-TDDFT (TDA) for formaldehyde (H2CO).

Level A: ordinary PySCF RKS/RHF ground state; DSE + coherent-state enter the
response only via ``casidapy.utils.qed``.

Example
-------
    cd /projectsn/mp1009_1/am4655/casidapy
    python scripts/run_qed_formaldehyde.py
    python scripts/run_qed_formaldehyde.py --xc pbe0 --basis 6-31g --lam 0.05 \\
        --tune-cavity --nstates 10 --plot formaldehyde_qed.png
"""
from __future__ import annotations

import argparse
import time

import numpy as np

HA_TO_EV = 27.211386245988

# Experimental-ish gas-phase geometry (Å); C at origin, C=O along z.
FORMALDEHYDE_ATOM = """
C  0.000000  0.000000  0.000000
O  0.000000  0.000000  1.208000
H  0.000000  0.943000 -0.587000
H  0.000000 -0.943000 -0.587000
"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--basis", default="6-31g", help="AO basis (default: 6-31g)")
    p.add_argument("--xc", default="pbe0", help="XC functional or 'hf' (default: pbe0)")
    p.add_argument("--use-df", action="store_true", help="Density-fit the electronic response")
    p.add_argument("--n-occ", type=int, default=None, help="Active occupied count (default: all)")
    p.add_argument("--n-unocc", type=int, default=None, help="Active virtual count (default: all)")
    p.add_argument("--nstates", type=int, default=10, help="Number of QED roots to print/plot")
    p.add_argument("--lam", type=float, default=0.05, help="Coupling strength λ (a.u.)")
    p.add_argument(
        "--polarization",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 1.0),
        metavar=("EX", "EY", "EZ"),
        help="Cavity polarization (default: z / C=O axis)",
    )
    p.add_argument(
        "--omega-c",
        type=float,
        default=None,
        help="Cavity frequency in Hartree (default: with --tune-cavity, else 0.15)",
    )
    p.add_argument(
        "--omega-c-ev",
        type=float,
        default=None,
        help="Cavity frequency in eV (overrides --omega-c)",
    )
    p.add_argument(
        "--tune-cavity",
        action="store_true",
        help="Set ω_c to the brightest bare electronic TDA root",
    )
    p.add_argument("--no-dse", action="store_true", help="Turn off dipole self-energy (diagnostic)")
    p.add_argument("--no-cs", action="store_true", help="Turn off coherent-state shift (diagnostic)")
    p.add_argument("--origin", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"))
    p.add_argument("--compare-pyscf", action="store_true", help="Print bare PySCF TDA roots for reference")
    p.add_argument("--plot", type=str, default=None, help="Save stick spectrum PNG to this path")
    p.add_argument(
        "--lam-scan",
        type=float,
        nargs=3,
        metavar=("START", "STOP", "N"),
        default=None,
        help="Sweep λ from START to STOP with N points (reuses cached K; no new integrals)",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def build_mf(basis: str, xc: str, verbose: bool = False):
    from pyscf import gto, dft, scf

    mol = gto.M(atom=FORMALDEHYDE_ATOM, basis=basis, verbose=4 if verbose else 0)
    if str(xc).lower() in ("hf", "rhf"):
        mf = scf.RHF(mol)
    else:
        mf = dft.RKS(mol)
        mf.xc = xc
    t0 = time.time()
    e = mf.kernel()
    print(f"SCF ({xc}/{basis}) E = {e:.10f} Ha  ({time.time() - t0:.1f}s)")
    return mf


def bare_tda_from_kernel(kernel):
    A = kernel._K + np.diag(kernel.diagonal_dE())
    w, V = np.linalg.eigh(A)
    mu = kernel.dipole_matrix()
    d = V.T @ mu
    f = (2.0 / 3.0) * w * np.einsum("nx,nx->n", d, d)
    return w, f


def main():
    args = parse_args()

    from casidapy.adapter.pyscf import extract_gto_kernel
    from casidapy.utils.qed import QEDOptions, solve_qed_tda, permanent_dipole, scan_qed_lambda

    mf = build_mf(args.basis, args.xc, verbose=args.verbose)

    if args.compare_pyscf:
        from pyscf import tdscf

        td = tdscf.TDA(mf)
        td.nstates = min(args.nstates, 8)
        td.kernel()
        print("\nPySCF bare TDA (reference):")
        for i, e in enumerate(np.asarray(td.e)):
            print(f"  {i:3d}  {e:12.6f} Ha  {e * HA_TO_EV:10.3f} eV")

    t0 = time.time()
    kernel, _ = extract_gto_kernel(
        mf,
        n_occ=args.n_occ,
        n_unocc=args.n_unocc,
        n_states=args.nstates,
        tda=True,
        use_df=args.use_df,
        verbose=args.verbose,
    )
    kernel.setup(tda=True)
    print(
        f"GTOKernel: n_occ={kernel.n_occ}, n_unocc={kernel.n_unocc}, "
        f"n_trans={kernel.n_trans}, K_cached={kernel._K is not None} "
        f"({time.time() - t0:.1f}s)"
    )

    w_bare, f_bare = bare_tda_from_kernel(kernel)
    ib = int(np.argmax(np.clip(f_bare, 0.0, None)))
    print("\nBare electronic TDA (from cached K):")
    n_show = min(8, len(w_bare))
    print(f"  {'#':>3}  {'ω/Ha':>12}  {'ω/eV':>10}  {'f':>10}")
    for i in range(n_show):
        mark = "  <-- brightest" if i == ib else ""
        print(
            f"  {i:3d}  {w_bare[i]:12.6f}  {w_bare[i] * HA_TO_EV:10.3f}  "
            f"{f_bare[i]:10.4f}{mark}"
        )

    if args.omega_c_ev is not None:
        omega_c = float(args.omega_c_ev) / HA_TO_EV
    elif args.tune_cavity:
        omega_c = float(w_bare[ib])
    elif args.omega_c is not None:
        omega_c = float(args.omega_c)
    else:
        omega_c = 0.15

    nstates = min(int(args.nstates), kernel.n_trans + 1)
    opts = QEDOptions(
        lam_scalar=float(args.lam),
        polarization=tuple(args.polarization),
        omega_c=omega_c,
        origin=tuple(args.origin),
        include_dse=not args.no_dse,
        coherent_state=not args.no_cs,
        nstates=nstates,
    )
    lam_vec = opts.lam_vec()

    mu0 = permanent_dipole(kernel, origin=opts.origin, include_nuclear=True)
    mu0_el = permanent_dipole(kernel, origin=opts.origin, include_nuclear=False)
    print(
        f"\nCavity: λ={args.lam:.4f}, e={lam_vec / (args.lam if abs(args.lam) > 1e-15 else 1)}, "
        f"ω_c={omega_c:.6f} Ha ({omega_c * HA_TO_EV:.3f} eV)"
    )
    print(f"  DSE={opts.include_dse}, coherent_state={opts.coherent_state}")
    print(f"  ⟨μ⟩₀ electronic = {mu0_el}  full (el+nuc) = {mu0}")

    t0 = time.time()
    r = solve_qed_tda(kernel, options=opts)
    print(f"QED-TDA diagonalization ({nstates} roots) in {time.time() - t0:.2f}s")

    print("\nQED-TDA polaritons:")
    print(f"  {'#':>3}  {'ω/Ha':>12}  {'ω/eV':>10}  {'f':>10}  {'|m|²':>10}")
    for i in range(r.omega.size):
        print(
            f"  {i:3d}  {r.omega[i]:12.6f}  {r.omega[i] * HA_TO_EV:10.3f}  "
            f"{r.f[i]:10.4f}  {r.photon_frac[i]:10.4f}"
        )

    # Rough LP/UP gap among states nearest ω_c
    near = np.argsort(np.abs(r.omega - omega_c))[:2]
    if len(near) == 2:
        gap = abs(r.omega[near[0]] - r.omega[near[1]]) * HA_TO_EV
        print(
            f"\nNearest-to-cavity pair: states {near[0]}, {near[1]}  "
            f"Δω ≈ {gap:.3f} eV  "
            f"(|m|² = {r.photon_frac[near[0]]:.3f}, {r.photon_frac[near[1]]:.3f})"
        )

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise SystemExit("matplotlib required for --plot") from exc

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax = axes[0]
        ax.vlines(w_bare[:n_show] * HA_TO_EV, 0, np.clip(f_bare[:n_show], 0, None),
                  colors="C0", lw=2, label="bare TDA")
        ax.vlines(r.omega * HA_TO_EV, 0, np.clip(r.f, 0, None),
                  colors="C1", lw=2, alpha=0.8, label="QED-TDA")
        ax.axvline(omega_c * HA_TO_EV, color="k", ls="--", alpha=0.5, label="ω_c")
        ax.set_xlabel("ω (eV)")
        ax.set_ylabel("oscillator strength")
        ax.set_title(f"H₂CO {args.xc}/{args.basis}, λ={args.lam}")
        ax.legend(fontsize=8)

        ax = axes[1]
        sc = ax.scatter(
            r.omega * HA_TO_EV,
            np.clip(r.f, 0, None),
            c=r.photon_frac,
            cmap="viridis",
            s=70,
            edgecolors="k",
            linewidths=0.4,
        )
        plt.colorbar(sc, ax=ax, label="|m|²")
        ax.axvline(omega_c * HA_TO_EV, color="k", ls="--", alpha=0.5)
        ax.set_xlabel("ω (eV)")
        ax.set_ylabel("f")
        ax.set_title("Photonic character")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\nWrote {args.plot}")

    if args.lam_scan is not None:
        lam0, lam1, nlam = args.lam_scan
        nlam = int(nlam)
        lam_grid = np.linspace(float(lam0), float(lam1), nlam)
        print(f"\nλ-scan: {nlam} points in [{lam0}, {lam1}] (cached K reused)")
        t0 = time.time()
        sweep = scan_qed_lambda(
            kernel,
            lam_grid,
            args.polarization,
            omega_c,
            nstates=nstates,
            include_dse=not args.no_dse,
            coherent_state=not args.no_cs,
            track=True,
        )
        print(f"λ-scan done in {time.time() - t0:.2f}s")
        omega_t = sweep["omega_tracked"]
        phot_t = sweep["photon_frac_tracked"]
        print(f"  tracked shape {omega_t.shape}")
        # print photonic fraction of branch nearest ω_c at endpoints
        for j, lam in enumerate((lam_grid[0], lam_grid[-1])):
            idx = 0 if j == 0 else -1
            ib = int(np.argmin(np.abs(omega_t[idx] - omega_c)))
            print(
                f"  λ={lam:.4f}: nearest branch ω={omega_t[idx, ib]*HA_TO_EV:.3f} eV, "
                f"|m|²={phot_t[idx, ib]:.3f}"
            )
        if args.plot:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for s in range(omega_t.shape[1]):
                axes[0].plot(lam_grid, omega_t[:, s] * HA_TO_EV, "-o", ms=3)
                axes[1].plot(lam_grid, phot_t[:, s], "-o", ms=3)
            axes[0].axhline(omega_c * HA_TO_EV, color="k", ls="--", alpha=0.4)
            axes[0].set_xlabel("λ"); axes[0].set_ylabel("ω (eV)")
            axes[0].set_title("H₂CO polariton branches vs λ")
            axes[1].set_xlabel("λ"); axes[1].set_ylabel("|m|²")
            axes[1].set_title("Photonic fraction vs λ")
            axes[1].set_ylim(-0.05, 1.05)
            fig.tight_layout()
            out = args.plot.replace(".png", "_lamscan.png")
            if out == args.plot:
                out = args.plot + "_lamscan.png"
            fig.savefig(out, dpi=150)
            print(f"Wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
