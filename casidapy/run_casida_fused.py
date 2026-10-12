"""Fused single-MPI-program driver: QEpy SCF -> Casida (in memory) -> QED post.

This is the in-memory / communicator-only counterpart of the file-based
``generate_inputs_qepy`` + ``run_casida_parallel_generic`` pipeline. Instead of
serializing the ground state (rho/psi/eig/occs) to disk and reloading it, it
keeps the live QEpy ``Driver`` in memory and hands its density/orbitals straight
into the same Casida construction, then runs the PF/JC post-processing — all on
one MPI communicator with no disk round-trip for the GS->Casida handoff.

The Casida construction here mirrors ``run_casida_parallel_generic`` exactly
(same target grid, psi interpolation + normalization, NLCC core in the XC
functional, non-augmented ``CasidaKS_MPI``) so the results match the file-based
path to numerical precision. Only the *source* of the ground state differs
(live driver vs ``.npy``).

Example
-------
    mpirun -n 1 python -m casidapy.run_casida_fused \\
        --geometry licn_dimer.xyz \\
        --pseudo "Li:li_pbe_v1.4.uspp.F.UPF,C:c_pbe_v1.2.uspp.F.UPF,N:n_pbe_v1.2.uspp.F.UPF" \\
        --ecutwfc 30 --nbnd 40 --gamma-only --nosym \\
        --n-occ 12 --n-unocc 25 --n-states 40 --tda --use-uspp \\
        --qed-post pf,jc --qed-omega-c auto --qed-lambda-sweep 0.005,0.01,0.02,0.05 \\
        --output-prefix licnd_fused --output-dir .
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from mpi4py import MPI
from ase.io import read as ase_read

from qepy.driver import Driver
from qepy.io import QEInput

from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.functional import Functional, TotalFunctional
from dftpy.functional import LocalPseudo
from dftpy.ions import Ions
from dftpy.utils import grid_map_data

from casidapy.casida_engine import CasidaKS_MPI, XC_TO_LIBXC_COMPONENTS
from casidapy.adapter.qepy import slice_active_space
from casidapy.utils.casida_utils import normalize_wavefunctions
from casidapy.generate_inputs_qepy import get_atomic_masses, read_z_valence_from_upf
from casidapy.qed_post import run_qed_post, save_qed_post_results

HA2EV = 27.211386245988


def parse_pseudo_map(s):
    d = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        el, fn = pair.split(":", 1)
        d[el.strip()] = fn.strip()
    return d


def build_qe_options(atoms, args, workdir, pmap):
    masses = get_atomic_masses()
    atomic_species = []
    for el in dict.fromkeys(atoms.get_chemical_symbols()):
        atomic_species.append(f"{el} {masses.get(el, 1.0):.6f} {pmap[el]}")
    ecutrho = args.ecutrho if args.ecutrho else 10.0 * args.ecutwfc
    qe = {
        "&control": {"calculation": "'scf'", "prefix": f"'{args.prefix}'",
                     "pseudo_dir": f"'{workdir}/'", "outdir": f"'{args.outdir}'",
                     "verbosity": "'high'"},
        "&system": {"ibrav": 0, "ecutwfc": args.ecutwfc, "ecutrho": ecutrho,
                    "nbnd": args.nbnd, "occupations": "'fixed'"},
        "&electrons": {"conv_thr": args.conv_thr, "mixing_beta": args.mixing_beta,
                       "electron_maxstep": args.electron_maxstep},
        "atomic_species": atomic_species,
    }
    if args.nosym:
        qe["&system"]["nosym"] = True
    qe["k_points gamma"] = []
    qe = QEInput.update_atoms(atoms, qe_options=qe, extrapolation=False)
    return qe


def main(argv=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    p = argparse.ArgumentParser(description="Fused QEpy SCF -> Casida -> QED (in-memory, MPI).")
    # SCF
    p.add_argument("--geometry", required=True)
    p.add_argument("--pseudo", required=True, help="El:file.UPF,... (SCF + core)")
    p.add_argument("--ecutwfc", type=float, default=30.0)
    p.add_argument("--ecutrho", type=float, default=None)
    p.add_argument("--nbnd", type=int, default=60)
    p.add_argument("--gamma-only", action="store_true")
    p.add_argument("--nosym", action="store_true")
    p.add_argument("--conv-thr", type=float, default=1e-6)
    p.add_argument("--mixing-beta", type=float, default=0.3)
    p.add_argument("--electron-maxstep", type=int, default=200)
    p.add_argument("--prefix", default="fused")
    p.add_argument("--outdir", default="./tmp_fused/")
    # Casida
    p.add_argument("--grid-nr", type=int, nargs=3, default=[80, 80, 80])
    p.add_argument("--n-occ", type=int, default=10)
    p.add_argument("--n-unocc", type=int, default=20)
    p.add_argument("--n-states", type=int, default=40)
    p.add_argument("--n-total-occ", type=int, default=None)
    p.add_argument("--tda", action="store_true")
    p.add_argument("--xc", default="PBE")
    p.add_argument("--use-uspp", action="store_true")
    p.add_argument("--uspp-map", default=None, help="El:file.UPF,... (default: --pseudo)")
    # QED
    p.add_argument("--qed-post", default=None, help="pf,jc,tc")
    p.add_argument("--qed-omega-c", default="auto")
    p.add_argument("--qed-lambda", type=float, nargs=3, default=None)
    p.add_argument("--qed-lambda-sweep", default=None)
    p.add_argument("--qed-pol", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    p.add_argument("--qed-nstates", type=int, default=None)
    p.add_argument("--qed-tda-pf", action="store_true")
    p.add_argument("--qed-tda-nstates", type=int, default=None)
    # Output
    p.add_argument("--output-prefix", default="fused")
    p.add_argument("--output-dir", default="./")
    args = p.parse_args(argv)

    workdir = os.path.abspath(os.path.dirname(args.geometry) or ".")
    pmap = parse_pseudo_map(args.pseudo)
    atoms = ase_read(args.geometry)

    # ---- [1] QEpy SCF (live driver, no disk handoff) ----
    if rank == 0:
        print("[1] QEpy SCF ...", flush=True)
    os.makedirs(args.outdir, exist_ok=True)
    qe_options = build_qe_options(atoms, args, workdir, pmap)
    driver = Driver(qe_options=qe_options, atoms=atoms,
                    logfile=os.path.join(args.output_dir, f"{args.output_prefix}_scf.out"))
    driver.scf()
    if rank == 0:
        print("    SCF converged.", flush=True)

    # ---- [2] Extract GS in memory ----
    eigs = np.asarray(driver.get_eigenvalues()) / 2.0          # Ry -> Ha
    if eigs.ndim == 2:
        eigs = eigs[0]
    occs = np.asarray(driver.get_occupation_numbers())
    psi_raw = driver.get_wave_function()
    ions = driver.get_dftpy_ions()
    rho_raw = driver.get_density()
    rho_grid = driver.get_dftpy_grid()
    rho = rho_raw if hasattr(rho_raw, "grid") else DirectField(grid=rho_grid, data=rho_raw)

    n_total_occ = args.n_total_occ if args.n_total_occ is not None else int(np.sum(occs > 0.5))
    if rank == 0:
        gap = (eigs[n_total_occ] - eigs[n_total_occ - 1]) * HA2EV
        print(f"    bands={len(eigs)}  n_occ={n_total_occ}  HOMO-LUMO={gap:.4f} eV  "
              f"electrons={float(np.sum(occs)):.2f}", flush=True)

    # ---- [3] Casida grid + density/orbitals (mirror the file-based path) ----
    grid = DirectGrid(lattice=ions.cell, nr=args.grid_nr)
    rho_ks = grid_map_data(rho, grid=grid)

    # psi: QEpy raw -> field on the QE grid -> interpolate to the Casida grid.
    nr_qe = int(round(psi_raw[0].size ** (1.0 / 3.0)))
    grid_qe = DirectGrid(lattice=ions.cell, nr=[nr_qe, nr_qe, nr_qe])
    psi_qe = [np.asarray(driver.data2field(pk, grid=grid_qe)) for pk in psi_raw]
    if list(psi_qe[0].shape) != list(grid.nr):
        psi_list = [DirectField(grid=grid, data=grid_map_data(
            DirectField(grid=grid_qe, data=pk), grid=grid)) for pk in psi_qe]
    else:
        psi_list = [DirectField(grid=grid, data=pk) for pk in psi_qe]
    vol = ions.cell.volume
    psi_list = [psi / np.sqrt(vol) for psi in psi_list]
    psi_list = normalize_wavefunctions(psi_list, grid)

    # ---- [4] NLCC core (USPP or NC) into the XC functional; non-augmented kernel ----
    core = None
    umap = parse_pseudo_map(args.uspp_map) if args.uspp_map else pmap
    if args.use_uspp:
        from casidapy.utils.uspp import load_uspp_data
        paths = {el: (fn if os.path.isabs(fn) else os.path.join(workdir, fn))
                 for el, fn in umap.items()}
        _beta, _qij, core = load_uspp_data(upf_files=paths, grid=grid, ions=ions)
    else:
        pseudo = LocalPseudo(grid=grid, ions=ions, PP_list=pmap)
        core = pseudo.core_density

    libxc = None
    xc = Functional(type="XC", name=args.xc, core_density=core)
    totalfunctional = TotalFunctional(KE=None, XC=xc)

    # ---- [5] Build/solve Casida (dense, matching the file path) ----
    if rank == 0:
        print("[5] Casida ...", flush=True)
    casida = CasidaKS_MPI(rho_ks, totalfunctional, spin_state="singlet",
                          libxc_xc_components=libxc)
    occ_e, unocc_e, psi_occ, psi_unocc = slice_active_space(
        eigs, psi_list, args.n_occ, args.n_unocc, n_total_occ=n_total_occ, verbose=(rank == 0))
    casida.set_active_orbitals(occ_e, unocc_e, psi_occ, psi_unocc)
    n_trans = casida.n_occ * casida.n_unocc
    n_states = min(args.n_states, n_trans)
    casida.build_matrices(tda=args.tda)
    omega, Z = casida.solve(k=n_states)
    f = casida.oscillator_strengths(k=n_states)
    if rank != 0:
        return 0

    base = os.path.join(args.output_dir, args.output_prefix)
    np.save(f"{base}_omega.npy", np.asarray(omega))
    np.save(f"{base}_f.npy", np.asarray(f))
    np.save(f"{base}_Z.npy", np.asarray(Z))
    if casida.d_mode is not None:
        np.save(f"{base}_d_mode.npy", np.asarray(casida.d_mode))
    print(f"    Casida done: {n_states} states, E0={omega[0]*HA2EV:.4f} eV", flush=True)

    # ---- [6] QED post-processing (in-memory; same core as file path) ----
    if args.qed_post and casida.d_mode is not None:
        omega_c = None
        if str(args.qed_omega_c).strip().lower() != "auto":
            omega_c = float(args.qed_omega_c) / HA2EV
        sweep = [float(x) for x in args.qed_lambda_sweep.split(",")] if args.qed_lambda_sweep else None
        data = run_qed_post(np.asarray(omega), casida.d_mode, f=np.asarray(f),
                            models=[m for m in args.qed_post.split(",") if m.strip()],
                            omega_c=omega_c, lam_vec=args.qed_lambda, lam_sweep=sweep,
                            pol=args.qed_pol, nstates=args.qed_nstates)
        save_qed_post_results(data, base, bare_omega_ev=np.asarray(omega) * HA2EV,
                              bare_f=np.asarray(f), plot=False)
        print(f"    QED post: models={args.qed_post} ω_c={data['omega_c_ev']:.4f} eV", flush=True)

        if args.qed_tda_pf:
            from casidapy.utils.qed import solve_qed_tda
            from casidapy.qed_post import make_lambda_vectors
            kern = getattr(casida, "_kernel", None)
            if kern is not None:
                wc = data["omega_c"]
                lam_vecs = make_lambda_vectors(args.qed_lambda, sweep=sweep, pol=args.qed_pol)
                for il, lam in enumerate(lam_vecs):
                    r = solve_qed_tda(kern, lam_vec=lam, omega_c=wc,
                                      nstates=args.qed_tda_nstates or n_states,
                                      matrix_free=False, comm=None)
                    np.save(f"{base}_incalcpf_lambda{il}_energies_ev.npy", np.asarray(r.omega) * HA2EV)
                    np.save(f"{base}_incalcpf_lambda{il}_osc_strength.npy", np.asarray(r.f))
                    np.save(f"{base}_incalcpf_lambda{il}_photon_frac.npy", np.asarray(r.photon_frac))
                print(f"    in-calc PF: {len(lam_vecs)} λ points", flush=True)

    try:
        driver.stop()
    except Exception:
        pass
    print("Fused run complete (no GS->Casida disk round-trip).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
