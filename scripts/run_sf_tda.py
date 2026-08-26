#!/usr/bin/env python
"""Collinear SF-TDDFT (TDA, Route A) and optional QED-SF-TDA.

Exchange-only spin-flip on a high-spin UKS/UHF reference (α-occ → β-virt).
Requires a hybrid XC; oscillator strengths are identically zero (dipole-
forbidden). With ``--qed``, builds the QED-SF-TDA matrix (SF singles ⊗ {0,1}
photons, Δd coupling).

Example
-------
    cd /path/to/casidapy
    python scripts/run_sf_tda.py
    python scripts/run_sf_tda.py --molecule twisted-ethylene --basis 6-31g \\
        --xc bhandhlyp --nstates 8 --plot sf_spectrum.png
    python scripts/run_sf_tda.py --qed --lam 0.05 --omega-c-ev 3.0 --nstates 8
"""
from __future__ import annotations

import argparse
import time

import numpy as np

HA_TO_EV = 27.211386245988

# Triplet methylene (spin = 2·Mₛ = 2), roughly experimental C–H lengths.
CH2_ATOM = """
C  0.000000  0.000000  0.000000
H  0.000000  0.000000  1.080000
H  1.000000  0.000000 -0.400000
"""

# Ethylene at 90° torsion (diradicaloid); SF from the triplet reference.
# Planar C=C along x; one CH₂ in xy, the other in xz.
TWISTED_ETHYLENE_ATOM = """
C   0.000000   0.000000   0.000000
C   1.339000   0.000000   0.000000
H  -0.549000   0.951000   0.000000
H  -0.549000  -0.951000   0.000000
H   1.888000   0.000000   0.951000
H   1.888000   0.000000  -0.951000
"""


MOLECULES = {
    "ch2": {
        "atom": CH2_ATOM,
        "spin": 2,
        "charge": 0,
        "label": "CH₂ triplet",
    },
    "twisted-ethylene": {
        "atom": TWISTED_ETHYLENE_ATOM,
        "spin": 2,
        "charge": 0,
        "label": "twisted ethylene (90°, triplet ref.)",
    },
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--molecule",
        choices=sorted(MOLECULES),
        default="ch2",
        help="Preset geometry / multiplicity (default: ch2)",
    )
    p.add_argument("--basis", default="sto-3g", help="AO basis (default: sto-3g)")
    p.add_argument(
        "--xc",
        default="bhandhlyp",
        help="Hybrid XC for SF Route A (default: bhandhlyp)",
    )
    p.add_argument("--use-df", action="store_true", help="Density-fit the SF response")
    p.add_argument("--n-occ", type=int, default=None, help="Active α-occupied count")
    p.add_argument("--n-unocc", type=int, default=None, help="Active β-virtual count")
    p.add_argument("--nstates", type=int, default=6, help="Number of SF / QED-SF roots")
    p.add_argument(
        "--solver",
        choices=("davidson", "eigsh", "lobpcg"),
        default="davidson",
        help="Matrix-free eigensolver for bare SF (default: davidson)",
    )
    p.add_argument(
        "--qed",
        action="store_true",
        help="Run QED-SF-TDA (Δd coupling) instead of bare SF-TDA",
    )
    p.add_argument("--lam", type=float, default=0.05, help="QED λ (a.u.), with --qed")
    p.add_argument(
        "--polarization",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 1.0),
        metavar=("EX", "EY", "EZ"),
        help="Cavity polarization (default: z)",
    )
    p.add_argument(
        "--omega-c",
        type=float,
        default=0.1,
        help="Cavity frequency in Hartree (default: 0.1)",
    )
    p.add_argument(
        "--omega-c-ev",
        type=float,
        default=None,
        help="Cavity frequency in eV (overrides --omega-c)",
    )
    p.add_argument("--plot", type=str, default=None, help="Save stick-spectrum PNG")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def build_mol(name: str, basis: str, verbose: bool = False):
    from pyscf import gto

    spec = MOLECULES[name]
    mol = gto.M(
        atom=spec["atom"],
        basis=basis,
        spin=spec["spin"],
        charge=spec["charge"],
        verbose=4 if verbose else 0,
    )
    return mol, spec


def print_spectrum(
    omega_ha: np.ndarray,
    f: np.ndarray,
    nstates: int,
    photon_frac=None,
) -> None:
    n = min(nstates, len(omega_ha))
    if photon_frac is None:
        print(f"\n{'state':>6}  {'ω (Ha)':>12}  {'ω (eV)':>10}  {'f':>10}")
        print("-" * 44)
        for i in range(n):
            print(
                f"{i + 1:6d}  {omega_ha[i]:12.6f}  {omega_ha[i] * HA_TO_EV:10.4f}  "
                f"{f[i]:10.3e}"
            )
    else:
        print(
            f"\n{'state':>6}  {'ω (Ha)':>12}  {'ω (eV)':>10}  "
            f"{'|m|²':>10}  {'f':>10}"
        )
        print("-" * 56)
        for i in range(n):
            print(
                f"{i + 1:6d}  {omega_ha[i]:12.6f}  {omega_ha[i] * HA_TO_EV:10.4f}  "
                f"{photon_frac[i]:10.4f}  {f[i]:10.3e}"
            )


def maybe_plot(path: str, omega_ha: np.ndarray, label: str, nstates: int) -> None:
    import matplotlib.pyplot as plt

    n = min(nstates, len(omega_ha))
    ev = omega_ha[:n] * HA_TO_EV
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.vlines(ev, 0.0, 1.0, colors="C0", linewidth=1.5)
    ax.set_xlabel("Excitation energy (eV)")
    ax.set_ylabel("relative intensity (arb.)")
    ax.set_title(label)
    ax.set_ylim(0.0, 1.15)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    args = parse_args()

    from casidapy import extract_sf_gto_kernel, run_casida

    mol, spec = build_mol(args.molecule, args.basis, verbose=args.verbose)
    mode = "QED-SF-TDA" if args.qed else "SF-TDA Route A"
    print(
        f"{mode}: {spec['label']}  {args.xc}/{args.basis}  "
        f"spin={mol.spin}  charge={mol.charge}"
    )

    t0 = time.time()
    kernel, opts = extract_sf_gto_kernel(
        mol,
        xc=args.xc,
        n_occ=args.n_occ,
        n_unocc=args.n_unocc,
        n_states=args.nstates,
        use_df=args.use_df,
        verbose=args.verbose,
    )
    print(
        f"Reference UKS done + SF kernel ready  "
        f"(n_trans = {kernel.n_occ}×{kernel.n_unocc} = {kernel.n_trans})  "
        f"[{time.time() - t0:.1f}s]"
    )

    if args.qed:
        from casidapy import QEDOptions, solve_qed_sf_tda

        omega_c = (
            float(args.omega_c_ev) / HA_TO_EV
            if args.omega_c_ev is not None
            else float(args.omega_c)
        )
        qopts = QEDOptions(
            lam_scalar=args.lam,
            polarization=args.polarization,
            omega_c=omega_c,
            nstates=args.nstates,
        )
        t1 = time.time()
        res = solve_qed_sf_tda(kernel, options=qopts)
        print(
            f"QED-SF-TDA finished in {time.time() - t1:.1f}s  "
            f"(λ={args.lam}, ω_c={omega_c * HA_TO_EV:.3f} eV, "
            f"basis dim={res.meta['basis_dim']})"
        )
        print_spectrum(res.omega, res.f, args.nstates, photon_frac=res.photon_frac)
        title = f"QED-SF-TDA — {spec['label']}"
    else:
        opts.solver_method = args.solver
        opts.matrix_free = True
        t1 = time.time()
        res = run_casida(kernel, opts)
        print(f"Casida ({args.solver}) finished in {time.time() - t1:.1f}s")
        print_spectrum(res.omega, res.f, args.nstates)
        if not np.allclose(res.f[: args.nstates], 0.0, atol=1e-10):
            print("Warning: expected f ≈ 0 for collinear SF; check dipole path.")
        else:
            print(
                "\n(oscillator strengths are zero — SF transitions are dipole-forbidden)"
            )
        title = f"SF-TDA — {spec['label']}"

    if args.plot:
        maybe_plot(args.plot, res.omega, title, args.nstates)


if __name__ == "__main__":
    main()
