#!/usr/bin/env python
"""Demo: 1e SI-SOC mixing + few-level QED on formaldehyde or HI.

Shows why SOC matters for cavity response: triplets borrow oscillator
strength and enter the polariton manifold.

Examples
--------
python scripts/run_soc_qed_demo.py
python scripts/run_soc_qed_demo.py --molecule HI --lam 0.08
"""
from __future__ import annotations

import argparse

import numpy as np

HA_TO_EV = 27.211386245988


def _mol(name: str, basis: str):
    from pyscf import gto

    name = name.lower()
    if name in ("h2co", "formaldehyde"):
        atom = """
        C  0.000000  0.000000  0.000000
        O  0.000000  0.000000  1.210000
        H  0.000000  0.940000 -0.580000
        H  0.000000 -0.940000 -0.580000
        """
    elif name == "hi":
        atom = "I 0 0 0; H 0 0 1.609"
    else:
        raise ValueError(f"unknown molecule {name!r} (use h2co or hi)")
    return gto.M(atom=atom, basis=basis, verbose=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--molecule", default="h2co", choices=["h2co", "hi", "formaldehyde"])
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--xc", default="pbe")
    p.add_argument("--n-singlet", type=int, default=3)
    p.add_argument("--n-triplet", type=int, default=3)
    p.add_argument("--lam", type=float, default=0.05)
    p.add_argument("--pol", default="z", choices=["x", "y", "z"])
    p.add_argument("--omega-c", type=float, default=None,
                   help="Cavity frequency in Ha (default: lowest singlet)")
    args = p.parse_args()

    from pyscf import dft
    from casidapy import (
        extract_gto_kernel,
        run_casida,
        solve_soc_si,
        solve_soc_qed_levels,
        solve_qed_tda,
        QEDOptions,
    )

    mol = _mol(args.molecule, args.basis)
    mf = dft.RKS(mol)
    mf.xc = args.xc
    mf.grids.level = 1
    e_scf = mf.kernel()
    print(f"SCF E = {e_scf:.8f} Ha  ({args.xc}/{args.basis})")

    ks, opts_s = extract_gto_kernel(
        mf, n_states=args.n_singlet, tda=True, use_df=False, spin_state="singlet",
    )
    kt, opts_t = extract_gto_kernel(
        mf, n_states=args.n_triplet, tda=True, use_df=False, spin_state="triplet",
    )
    opts_s.solver_method = opts_t.solver_method = "eigsh"
    res_s = run_casida(ks, opts_s)
    res_t = run_casida(kt, opts_t)

    print("\nSinglet TDA (eV):")
    for i, w in enumerate(res_s.omega):
        print(f"  S{i}: {w * HA_TO_EV:8.3f}   f={res_s.f[i]:.4f}")
    print("Triplet TDA (eV):")
    for i, w in enumerate(res_t.omega):
        print(f"  T{i}: {w * HA_TO_EV:8.3f}   f={res_t.f[i]:.4f}")

    soc = solve_soc_si(res_s, res_t, ks, include_ground=False)
    print("\nSI-SOC mixed roots (eV)  [singlet wt | triplet wt | f]:")
    for i, w in enumerate(soc.omega):
        print(
            f"  {i:2d}: {w * HA_TO_EV:8.3f}   "
            f"S={soc.singlet_weight[i]:.3f}  T={soc.triplet_weight[i]:.3f}  "
            f"f={soc.f[i]:.4f}"
        )

    pol = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[args.pol]
    omega_c = float(args.omega_c) if args.omega_c is not None else float(res_s.omega[0])
    print(f"\nCavity: ω_c = {omega_c * HA_TO_EV:.3f} eV, λ={args.lam}, pol={args.pol}")

    # Closed-shell QED-TDA (singlets only, no SOC)
    qed_opts = QEDOptions(
        lam_scalar=args.lam, polarization=pol, omega_c=omega_c, nstates=6,
    )
    qed_s = solve_qed_tda(ks, options=qed_opts)
    print("\nQED-TDA (singlets only, no SOC):")
    for i, w in enumerate(qed_s.omega[:6]):
        print(f"  {i}: {w * HA_TO_EV:8.3f} eV   |m|²={qed_s.photon_frac[i]:.3f}")

    # Few-level TC and truncated PF on SOC-mixed electronic states
    lam_vec = np.asarray(pol, float) * args.lam
    qed_tc = solve_soc_qed_levels(
        soc, lam_vec=lam_vec, omega_c=omega_c, nstates=4,
    )
    print("\nTavis–Cummings on SI-SOC states:")
    for i, w in enumerate(qed_tc["omega"][:8]):
        print(
            f"  {i}: {w * HA_TO_EV:8.3f} eV   |m|²={qed_tc['photon_frac'][i]:.3f}"
        )

    from casidapy import solve_soc_qed_pf
    qed_pf = solve_soc_qed_pf(
        soc, lam_vec=lam_vec, omega_c=omega_c, nstates=4, include_dse=True,
    )
    print("\nPauli–Fierz on SI-SOC ⊗ {0,1} photons:")
    for i, w in enumerate(qed_pf["omega"][:10]):
        print(
            f"  {i}: {w * HA_TO_EV:8.3f} eV   |m|²={qed_pf['photon_frac'][i]:.3f}"
        )
    print(
        "\nCompare spectra: SOC mixes dark triplets into the bright manifold; "
        "PF adds the ½(λ·μ)² DSE and the full electronic⊗photon 0–1 space."
    )


if __name__ == "__main__":
    main()
