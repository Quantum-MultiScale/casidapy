#!/usr/bin/env python3
"""
Generic parallel Casida TDDFT runner for multi-element systems.

This is a generalized variant of the AG-specific runner that accepts a
pseudopotential map and supports dense/matrix-free Casida workflows,
including ultrasoft pseudopotential (USPP) corrections.
"""

import argparse
import os
import sys
import numpy as np
from mpi4py import MPI

from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.functional import Functional, TotalFunctional
from dftpy.functional import LocalPseudo
from dftpy.ions import Ions
from dftpy.formats import io
from dftpy.constants import Units
from dftpy.utils import grid_map_data
import ase.io

from casidapy.casida_engine import CasidaKS_MPI, XC_TO_LIBXC_COMPONENTS
from casidapy.casida_of import build_of_functional_context
from casidapy.qepy_adapter import slice_active_space

from casidapy.uspp import setup_uspp, setup_nc_pseudos, load_uspp_data, parse_upf
HAS_USPP_MODULE = True

try:
    from casidapy.polariton_handler import setup_polariton, save_polariton_results
    HAS_POLARITON_MODULE = True
except ImportError:
    HAS_POLARITON_MODULE = False


def patch_ions_class():
    if not hasattr(Ions, "_orig_init"):
        Ions._orig_init = Ions.__init__

        def _patched_ions_init(self, symbols=None, positions=None, cell=None, units=None, **kwargs):
            units_val = kwargs.pop("units", units)
            if units_val == "ase":
                if positions is not None:
                    positions = np.asarray(positions) / Units.Bohr
                if cell is not None:
                    cell = np.asarray(cell) / Units.Bohr
            return Ions._orig_init(self, symbols=symbols, positions=positions, cell=cell, **kwargs)

        Ions.__init__ = _patched_ions_init


def normalize_wavefunctions(psi_list, grid):
    normalized = []
    for psi in psi_list:
        psi_data = np.asarray(psi)
        idx_max = np.argmax(np.abs(psi_data))
        phase = np.angle(psi_data.flat[idx_max])
        psi_rot = psi_data * np.exp(-1j * phase)
        psi_real = psi_rot.real
        psi_field = DirectField(grid=grid, data=psi_real)
        norm = (psi_field.conj() * psi_field).integral().real
        psi_field = psi_field / np.sqrt(norm)
        normalized.append(psi_field)
    return normalized


def gauss(freq, omega, sigma=0.1):
    return np.exp(-(freq - omega * 27.2114) ** 2 / (2.0 * sigma ** 2))


def smooth_spectrum(f, omega, freq, sigma=0.1):
    s = np.zeros_like(freq)
    for i in range(len(omega)):
        s += gauss(freq, omega[i], sigma) * f[i]
    return s


def parse_pseudo_map(pseudo_map_str):
    """Parse 'Element:file.UPF,Element2:file2.UPF' into a dict."""
    pp = {}
    for pair in pseudo_map_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(f"Invalid pseudo mapping entry '{pair}'. Use Element:file.UPF")
        el, fn = pair.split(":", 1)
        pp[el.strip()] = fn.strip()
    return pp


def parse_input_file(input_file):
    """
    Parse a .in input file and return a dictionary of arguments.
    
    Input file format (similar to Quantum ESPRESSO):
    - Lines starting with # or ! are comments
    - Blank lines are ignored
    - Key-value pairs: key = value
    - Boolean flags: key (no value means True)
    - Arrays: key = val1 val2 val3
    
    Example:
        # Casida TDDFT Input File
        workdir = ./H2O_COMPARE
        atoms = h2o.vasp
        density = rho_scf_h2o.xsf
        psi = psi_h2o.npy
        eigs = eig_h2o.npy
        occs = occs_h2o.npy
        pseudo_map = H:H_ONCV_PBE-1.2.upf,O:O_ONCV_PBE-1.2.upf
        n_occ = 4
        n_unocc = 30
        n_states = 20
        matrix_free
        tda
        solver_method = eigsh
        xc = PBE
        output_prefix = casida_compare
        plot
    """
    args_dict = {}
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # Remove comments
            if '#' in line:
                line = line[:line.index('#')]
            if '!' in line:
                line = line[:line.index('!')]
            
            line = line.strip()
            
            # Skip blank lines
            if not line:
                continue
            
            # Parse key = value or key (boolean flag)
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip().replace('-', '_')
                value = value.strip()
                
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                # Try to convert to appropriate type
                if value.lower() == 'true':
                    args_dict[key] = True
                elif value.lower() == 'false':
                    args_dict[key] = False
                elif value.lower() == 'none':
                    args_dict[key] = None
                else:
                    # Check if it's a space-separated array (for grid_nr, etc.)
                    value_parts = value.split()
                    if len(value_parts) > 1:
                        # Array value
                        try:
                            # Try to convert all parts to numbers
                            if '.' in value:
                                args_dict[key] = [float(v) for v in value_parts]
                            else:
                                args_dict[key] = [int(v) for v in value_parts]
                        except ValueError:
                            # Mixed types, keep as list of strings
                            args_dict[key] = value_parts
                    else:
                        # Single value
                        try:
                            if '.' in value:
                                args_dict[key] = float(value)
                            else:
                                args_dict[key] = int(value)
                        except ValueError:
                            # String value
                            args_dict[key] = value
            else:
                # Boolean flag (no = sign)
                key = line.strip().replace('-', '_')
                args_dict[key] = True
    
    return args_dict


def convert_input_args_to_cli_args(args_dict):
    """
    Convert input file dictionary to list of CLI arguments.
    
    Handles:
    - Converting underscores to hyphens for CLI args
    - Converting boolean True to flags
    - Converting lists/arrays to space-separated values
    """
    cli_args = []
    
    for key, value in args_dict.items():
        # Convert key from input file format to CLI format
        cli_key = key.replace('_', '-')
        
        if value is True:
            # Boolean flag
            cli_args.append(f'--{cli_key}')
        elif value is False or value is None:
            # Skip False/None values
            continue
        elif isinstance(value, (list, tuple)):
            # Array argument
            cli_args.append(f'--{cli_key}')
            cli_args.extend(str(v) for v in value)
        else:
            # Regular argument
            cli_args.append(f'--{cli_key}')
            cli_args.append(str(value))
    
    return cli_args


def main():
    parser = argparse.ArgumentParser(description="Generic parallel Casida TDDFT calculation")
    parser.add_argument("--workdir", type=str, default="./")
    parser.add_argument("--atoms", type=str, required=True)
    parser.add_argument("--density", type=str, required=True)
    parser.add_argument("--psi", type=str, required=True)
    parser.add_argument("--eigs", type=str, required=True)
    parser.add_argument("--occs", type=str, required=True)
    parser.add_argument("--pseudo-map", type=str, default=None,
                        help='Norm-conserving PP map for LocalPseudo/XC, '
                             'e.g. "Ag:Ag_ONCV_PBE-1.2.upf,O:O_ONCV.upf". '
                             'Optional if --use-uspp + --uspp-map provides NLCC.')
    parser.add_argument("--n-occ", type=int, default=10)
    parser.add_argument("--n-unocc", type=int, default=20)
    parser.add_argument("--n-states", type=int, default=50)
    parser.add_argument("--tda", action="store_true")
    parser.add_argument("--matrix-free", action="store_true")
    parser.add_argument("--grid-nr", type=int, nargs=3, default=[80, 80, 80])
    parser.add_argument("--xc", type=str, default="PBE")
    parser.add_argument("--of-context", action="store_true",
                        help="Build an OF-inspired functional context (TFvW + EXT) "
                             "for KS-Casida without switching to full OF-TDDFT.")
    parser.add_argument("--output-prefix", type=str, default="casida_compare")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--solver-tol", type=float, default=1e-8)
    parser.add_argument("--solver-maxiter", type=int, default=200)
    parser.add_argument("--solver-method", type=str, default="lobpcg", choices=["lobpcg", "eigsh"])
    parser.add_argument("--n-total-occ", type=int, default=None)
    # ----- USPP options -----
    parser.add_argument("--use-uspp", action="store_true",
                        help="Enable ultrasoft pseudopotential corrections")
    parser.add_argument("--uspp-map", type=str, default=None,
                        help='USPP file map for augmentation data, '
                             'e.g. "Ag:ag_pbe_v1.4.uspp.F.UPF". '
                             'Only elements with ultrasoft PPs need to be listed.')
    # ----- Singlet / triplet -----
    parser.add_argument("--spin-state", type=str, default="singlet",
                        choices=["singlet", "triplet"],
                        help='Spin state for excitations (default: singlet). '
                             'Triplet drops the Coulomb coupling and uses the '
                             'spin-antisymmetric XC kernel.  Requires pylibxc.')
    # ----- Polariton postprocessing options -----
    parser.add_argument("--polariton", action="store_true",
                        help="Enable polariton postprocessing (cavity-exciton coupling)")
    parser.add_argument("--wc", type=float, default=3.0,
                        help="Cavity photon energy (eV) for polariton postprocessing")
    parser.add_argument("--g0", type=float, default=0.10,
                        help="Global coupling scale (eV) for polariton postprocessing")
    parser.add_argument("--coupling-model", type=str, default="sqrt-f",
                        choices=["sqrt-f", "uniform", "file", "dipole-vac"],
                        help="Coupling model for polariton: sqrt-f (oscillator strength weighted), "
                             "uniform (equal coupling), file (from file), dipole-vac (dipole-based)")
    parser.add_argument("--g-file", type=str, default=None,
                        help="File containing coupling values (for coupling-model=file)")
    parser.add_argument("--mu-units", type=str, default="au",
                        choices=["au", "si"],
                        help="Units for transition dipoles: au (atomic units) or si (SI)")
    parser.add_argument("--mode-volume-m3", type=float, default=None,
                        help="Cavity mode volume in m³ (required for dipole-vac coupling model)")
    parser.add_argument("--eps-r", type=float, default=1.0,
                        help="Relative permittivity for dipole-vac coupling model")
    parser.add_argument("--orient-factor", type=float, default=1.0,
                        help="Dipole-cavity orientation factor (1.0 = parallel, 1/sqrt(3) = isotropic)")
    parser.add_argument("--max-states", type=int, default=None,
                        help="Maximum number of states to include in polariton calculation")
    parser.add_argument("--gamma-exc", type=float, default=0.05,
                        help="Lorentzian broadening for bare excitonic spectrum (eV)")
    parser.add_argument("--gamma-pol", type=float, default=0.05,
                        help="Lorentzian broadening for polaritonic spectrum (eV)")
    parser.add_argument("--npts", type=int, default=2000,
                        help="Number of points for broadened spectrum grid")
    # ----- Input file -----
    parser.add_argument("--input-file", type=str, default=None,
                        help='Input file (.in format) containing all calculation parameters. '
                             'Command-line arguments override input file values.')
    
    # Check for --input-file in sys.argv first and parse it
    input_file = None
    original_argv = sys.argv[:]
    
    if '--input-file' in sys.argv:
        idx = sys.argv.index('--input-file')
        if idx + 1 < len(sys.argv):
            input_file = sys.argv[idx + 1]
            # Remove --input-file and its value from argv
            sys.argv = sys.argv[:idx] + sys.argv[idx+2:]
    
    # If input file is provided, parse it and prepend args to sys.argv
    # (CLI args come after, so they override input file values)
    if input_file:
        input_args = parse_input_file(input_file)
        input_cli_args = convert_input_args_to_cli_args(input_args)
        # Prepend input file args (CLI args will override)
        sys.argv = [sys.argv[0]] + input_cli_args + sys.argv[1:]
    
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Triplet requires pylibxc
    spin_state = args.spin_state
    if spin_state == "triplet":
        try:
            import pylibxc  # noqa: F401
        except ImportError:
            if rank == 0:
                print("FATAL: pylibxc is required for triplet excitations.", flush=True)
                print("       Install with: pip install pylibxc", flush=True)
            comm.Abort(1)

    if rank == 0:
        print("=" * 60)
        print("Generic Parallel Casida TDDFT Calculation")
        print("=" * 60)
        print(f"MPI ranks: {size}")
        print("Backend: CPU (MPI)")
        print(f"USPP enabled: {args.use_uspp}")
        print(f"Matrix-free mode: {args.matrix_free}")
        if args.matrix_free:
            print(f"Solver: {args.solver_method}")
        print(f"OF context: {args.of_context}")
        print(f"Spin state: {spin_state}")
        print(f"Working directory: {args.workdir}")
        print(f"TDA: {args.tda}")
        print(f"n_occ: {args.n_occ}, n_unocc: {args.n_unocc}")
        print("=" * 60)
        sys.stdout.flush()

    os.chdir(args.workdir)
    # After chdir, all relative paths are now relative to workdir.
    # Store the resolved workdir so we don't accidentally double-prefix later.
    resolved_workdir = os.getcwd()
    patch_ions_class()

    # ===== [1] Load atomic structure =====
    if rank == 0:
        print("\n[1] Loading atomic structure...")
        sys.stdout.flush()

    try:
        atoms = ase.io.read(args.atoms)
        ions = Ions.from_ase(atoms)
        grid = DirectGrid(lattice=ions.cell, nr=args.grid_nr)
    except Exception as e:
        print(f"[rank {rank}] FATAL in step [1]: {e}", flush=True)
        import traceback; traceback.print_exc(); sys.stdout.flush()
        comm.Abort(1)

    if rank == 0:
        print(f"    Grid: {grid.nr}")
        print(f"    Cell volume: {ions.cell.volume:.2f} Bohr³")
        sys.stdout.flush()

    # ===== [2] Load density =====
    # Rank 0 loads the (potentially large) XSF file and broadcasts to
    # avoid every rank independently parsing the same file.
    if rank == 0:
        print("\n[2] Loading ground-state density...")
        sys.stdout.flush()

    try:
        if rank == 0:
            rho_ks = io.read_density(args.density, full=True)
            rho_ks = grid_map_data(rho_ks, grid=grid)
            rho_data = np.asarray(rho_ks).copy()
        else:
            rho_data = None
        rho_data = comm.bcast(rho_data, root=0)
        if rank != 0:
            rho_ks = DirectField(grid=grid, data=rho_data)
        del rho_data
    except Exception as e:
        print(f"[rank {rank}] FATAL in step [2] (density loading): {e}", flush=True)
        import traceback; traceback.print_exc(); sys.stdout.flush()
        comm.Abort(1)

    if rank == 0:
        print(f"    Electrons: {rho_ks.integral():.4f}")
        sys.stdout.flush()

    # ===== [3] Setup functionals + USPP =====
    # The XC kernel f_xc needs the NLCC core density if present.
    # When using USPP, we extract it directly from the USPP file,
    # avoiding the need for LocalPseudo + a separate NC PP.
    if rank == 0:
        print("\n[3] Setting up functionals...")
        sys.stdout.flush()

    core = None  # NLCC core density (will be set below)
    beta_projectors = None
    qij_augmentation = None
    use_uspp_flag = False
    pseudo = None

    # ===== Setup USPP (if requested) =====
    if HAS_USPP_MODULE and args.use_uspp:
        beta_projectors, qij_augmentation, core, use_uspp_flag, _ = setup_uspp(
            args, grid, ions, atoms, resolved_workdir, comm=comm
        )
    elif args.use_uspp:
        # Fallback to old code if module not available
        if rank == 0:
            print("    WARNING: unified uspp module setup unavailable, using fallback code", flush=True)
        # ... (keep old code as fallback)
        if args.uspp_map is None:
            if rank == 0:
                print("    FATAL: --use-uspp requires --uspp-map to specify USPP files.", flush=True)
                print('    Example: --uspp-map "Ag:ag_pbe_v1.4.uspp.F.UPF"', flush=True)
            comm.Abort(1)
        uspp_files = parse_pseudo_map(args.uspp_map)
        uspp_paths = {}
        for el, fn in uspp_files.items():
            p = fn if os.path.isabs(fn) else os.path.join(resolved_workdir, fn)
            if not os.path.exists(p):
                if rank == 0:
                    print(f"    FATAL: USPP file not found: {p}", flush=True)
                comm.Abort(1)
            uspp_paths[el] = p
        try:
            beta_projectors, qij_augmentation, uspp_core_density = load_uspp_data(
                upf_files=uspp_paths, grid=grid, ions=ions,
            )
            use_uspp_flag = True
            core = uspp_core_density
        except Exception as e:
            if rank == 0:
                print(f"    ✗ ERROR loading USPP data: {e}", flush=True)
            use_uspp_flag = False

    # ===== Setup NC pseudos (if requested) =====
    if HAS_USPP_MODULE and args.pseudo_map:
        core, pseudo = setup_nc_pseudos(
            args, grid, ions, atoms, resolved_workdir, core, use_uspp_flag, comm=comm
        )
    elif args.pseudo_map:
        # Fallback to old code
        pp_list = parse_pseudo_map(args.pseudo_map)
        elems = set(atoms.get_chemical_symbols())
        if not elems.issubset(set(pp_list.keys())):
            missing = sorted(elems - set(pp_list.keys()))
            if rank == 0:
                print(f"    FATAL: Missing pseudopotentials for elements: {missing}", flush=True)
            comm.Abort(1)
        try:
            pseudo = LocalPseudo(grid=rho_ks.grid, ions=ions, PP_list=pp_list)
            if core is None:
                core = pseudo.core_density
                if rank == 0 and core is not None:
                    print(f"    Core density from LocalPseudo (NC PP)")
        except Exception as e:
            if rank == 0:
                print(f"    WARNING: LocalPseudo failed: {e}", flush=True)
                if not args.use_uspp:
                    print("    FATAL: No USPP path and LocalPseudo failed.", flush=True)
                    comm.Abort(1)
    elif not args.use_uspp:
        if rank == 0:
            print("    FATAL: Must provide --pseudo-map (NC PPs) or --use-uspp with --uspp-map", flush=True)
        comm.Abort(1)

    libxc_xc_components = None
    if spin_state == "triplet":
        xc_upper = args.xc.upper()
        if xc_upper not in XC_TO_LIBXC_COMPONENTS:
            if rank == 0:
                supported = ", ".join(sorted(XC_TO_LIBXC_COMPONENTS.keys()))
                print(f"    FATAL: No libxc component mapping for XC='{args.xc}'.", flush=True)
                print(f"    Supported XC names for triplet: {supported}", flush=True)
                print(f"    Add the XC to XC_TO_LIBXC_COMPONENTS in casida_utils.py if needed.", flush=True)
            comm.Abort(1)
        libxc_xc_components = XC_TO_LIBXC_COMPONENTS[xc_upper]
        if rank == 0:
            print(f"    Triplet XC kernel components: {libxc_xc_components}")

    # Set up XC/functional context
    xc = Functional(type="XC", name=args.xc, core_density=core)
    totalfunctional = TotalFunctional(KE=None, XC=xc)
    if args.of_context:
        if not args.pseudo_map:
            if rank == 0:
                print("    FATAL: --of-context currently requires --pseudo-map.", flush=True)
            comm.Abort(1)
        pp_list = parse_pseudo_map(args.pseudo_map)
        try:
            of_ctx = build_of_functional_context(
                rho_ks=rho_ks,
                ions=ions,
                xc_name=args.xc,
                pp_list=pp_list,
            )
            totalfunctional = of_ctx["totalfunctional"]
            pseudo = of_ctx["pseudo"]
            core = of_ctx["core_density"]
            xc = Functional(type="XC", name=args.xc, core_density=core)
            if rank == 0:
                print("    OF-context mode active:")
                print("      - KEDF: TFvW")
                print(f"      - XC: {args.xc}")
                print("      - EXT: enabled (derived from TFvW+XC+PSEUDO+HARTREE+VW)")
                print(f"      - |v_ext|_2: {of_ctx['v_ext_norm']:.6e}")
                sys.stdout.flush()
        except Exception as e:
            if rank == 0:
                print(f"    FATAL: Failed to build OF context: {e}", flush=True)
            comm.Abort(1)

    if rank == 0:
        print(f"    XC functional: {args.xc}")
        print(f"    Core density (NLCC): {'present' if core is not None else 'None'}")
        sys.stdout.flush()

    # ===== [4] Load wavefunctions =====
    # Rank 0 loads the (potentially huge) psi file, interpolates to the
    # target grid, then broadcasts the small interpolated array.
    # This avoids every rank loading multi-GB files independently.
    if rank == 0:
        print("\n[4] Loading KS orbitals...")
        sys.stdout.flush()

    try:
        if rank == 0:
            eigs = np.load(args.eigs)
            occs = np.load(args.occs)
        else:
            eigs = None
            occs = None
        eigs = comm.bcast(eigs, root=0)
        occs = comm.bcast(occs, root=0)
    except Exception as e:
        print(f"[rank {rank}] FATAL in step [4] (loading eigs/occs): {e}", flush=True)
        import traceback; traceback.print_exc(); sys.stdout.flush()
        comm.Abort(1)

    n_total_occ = args.n_total_occ if args.n_total_occ is not None else int(np.sum(occs > 0.5))

    # Rank 0 loads and (optionally) interpolates wavefunctions, then bcasts
    try:
        if rank == 0:
            psi_ks = np.load(args.psi)
            print(f"    Number of orbitals: {len(psi_ks)}")
            print(f"    Total occupied orbitals (HOMO+1): {n_total_occ}")
            print(f"    HOMO index: {n_total_occ - 1}, LUMO index: {n_total_occ}")
            print(f"    HOMO energy: {eigs[n_total_occ-1] * 27.2114:.4f} eV")
            print(f"    LUMO energy: {eigs[n_total_occ] * 27.2114:.4f} eV")
            print(f"    HOMO-LUMO gap: {(eigs[n_total_occ] - eigs[n_total_occ-1]) * 27.2114:.4f} eV")

            psi_shape = psi_ks[0].shape
            print(f"    Wavefunction grid: {psi_shape}")
            need_interp = (list(psi_shape) != list(grid.nr))
            if need_interp:
                print(f"    Interpolating from {psi_shape} to {grid.nr} on rank 0...")
                sys.stdout.flush()
                grid_psi = DirectGrid(lattice=ions.cell, nr=list(psi_shape))
                psi_interp = np.empty((len(psi_ks), *grid.nr), dtype=np.float64)
                for ib in range(len(psi_ks)):
                    pf = DirectField(grid=grid_psi, data=psi_ks[ib])
                    psi_interp[ib] = np.asarray(grid_map_data(pf, grid=grid))
                del psi_ks  # free the large array
                psi_small = psi_interp
            else:
                psi_small = psi_ks
            print(f"    Broadcasting interpolated psi ({psi_small.nbytes / 1e6:.1f} MB)...", flush=True)
        else:
            psi_small = None

        psi_small = comm.bcast(psi_small, root=0)
    except Exception as e:
        print(f"[rank {rank}] FATAL in step [4] (loading psi): {e}", flush=True)
        import traceback; traceback.print_exc(); sys.stdout.flush()
        comm.Abort(1)

    psi_list = [DirectField(grid=grid, data=psi_small[i])
                for i in range(len(psi_small))]
    del psi_small
    psi_list = [psi / np.sqrt(ions.cell.volume) for psi in psi_list]

    # ===== [5] Normalize and phase-fix wavefunctions =====
    if rank == 0:
        print("\n[5] Normalizing and phase-fixing wavefunctions...")
        sys.stdout.flush()

    psi_list = normalize_wavefunctions(psi_list, grid)

    if rank == 0:
        norms = [(psi.conj() * psi).integral().real for psi in psi_list[:8]]
        print(f"    Norms (first 8): {[f'{n:.6f}' for n in norms]}")
        sys.stdout.flush()

    # ===== [6] Build / solve Casida =====
    casida = CasidaKS_MPI(
        rho_ks,
        totalfunctional,
        spin_state=spin_state,
        libxc_xc_components=libxc_xc_components,
    )

    comm.Barrier()
    t_start = MPI.Wtime()

    # Check available unoccupied orbitals
    n_total_orb = len(psi_list)
    n_available_unocc = n_total_orb - n_total_occ
    if args.n_unocc > n_available_unocc:
        if rank == 0:
            print(f"\n    WARNING: Requested {args.n_unocc} unoccupied orbitals but only {n_available_unocc} available.")
            print(f"             Maximum transitions: {args.n_occ} × {n_available_unocc} = {args.n_occ * n_available_unocc}")
            print(f"             To get more states, generate more unoccupied orbitals in the ground-state calculation.")
        sys.stdout.flush()

    occ_e, unocc_e, psi_occ, psi_unocc = slice_active_space(
        eigs, psi_list, args.n_occ, args.n_unocc,
        n_total_occ=n_total_occ, verbose=(rank == 0),
    )
    casida.set_active_orbitals(occ_e, unocc_e, psi_occ, psi_unocc)
    n_trans_active = casida.n_occ * casida.n_unocc
    n_states_actual = min(args.n_states, n_trans_active)
    if args.n_states > n_trans_active and rank == 0:
        print(f"\n    WARNING: Requested {args.n_states} states but only {n_trans_active} transitions available.")
        print(f"             Reducing to {n_states_actual} states.")
        sys.stdout.flush()

    if args.matrix_free:
        if rank == 0:
            print(f"\n[6] Setting up matrix-free Casida: "
                  f"{casida.n_occ} occ × {casida.n_unocc} unocc = {n_trans_active} transitions")
            print(f"    MPI ranks: {size} (CPU)")
            sys.stdout.flush()

        casida.setup_matrix_free(tda=args.tda)
        comm.Barrier()
        t_build = MPI.Wtime() - t_start

        if rank == 0:
            print(f"    Setup time: {t_build:.2f} s")
            sys.stdout.flush()

        if rank == 0:
            print(f"\n[7] Solving with matrix-free {args.solver_method.upper()}...")
            sys.stdout.flush()

        t_start = MPI.Wtime()
        omega, Z = casida.solve_matrix_free(
            k=n_states_actual,
            tol=args.solver_tol,
            maxiter=args.solver_maxiter,
            verbose=1 if rank == 0 else 0,
            method=args.solver_method,
        )
        t_solve = MPI.Wtime() - t_start
    else:
        if rank == 0:
            print(f"\n[6] Building Casida matrices: "
                  f"{casida.n_occ} occ × {casida.n_unocc} unocc = {n_trans_active} transitions")
            print(f"    MPI ranks: {size} (CPU)")
            sys.stdout.flush()

        casida.build_matrices(tda=args.tda)
        comm.Barrier()
        t_build = MPI.Wtime() - t_start

        if rank == 0:
            print(f"    Matrix build time: {t_build:.2f} s")
            sys.stdout.flush()

        if rank == 0:
            print("\n[7] Solving eigenvalue problem...")
            sys.stdout.flush()

        t_start = MPI.Wtime()
        omega, Z = casida.solve(k=n_states_actual)
        t_solve = MPI.Wtime() - t_start

    if rank == 0:
        print(f"    Solve time: {t_solve:.2f} s")
        print(f"\n    Excitation energies (eV):")
        for i, w in enumerate(omega[:min(n_states_actual, 20)]):
            print(f"      State {i+1}: {w * 27.2114:.4f} eV")
        sys.stdout.flush()

    # ===== [8] Oscillator strengths =====
    if rank == 0:
        print("\n[8] Computing oscillator strengths...")
        sys.stdout.flush()

    f = casida.oscillator_strengths(k=n_states_actual)

    if rank == 0:
        if spin_state == "triplet":
            print(f"    Triplet excitations are spin-forbidden — oscillator strengths are zero.")
        print(f"    Oscillator strengths (first 20):")
        for i, (w, osc) in enumerate(zip(omega[:min(n_states_actual, 20)],
                                          f[:min(n_states_actual, 20)])):
            print(f"      State {i+1}: E = {w * 27.2114:.4f} eV, f = {osc:.6f}")
        sys.stdout.flush()

    # ===== [9] Save results =====
    if rank == 0:
        print("\n[9] Saving results...")
        output_dir = args.output_dir if args.output_dir else args.workdir
        os.makedirs(output_dir, exist_ok=True)
        output_base = os.path.join(output_dir, args.output_prefix)

        np.save(f"{output_base}_omega.npy", omega)
        np.save(f"{output_base}_f.npy", f)
        np.save(f"{output_base}_Z.npy", Z)
        if casida.mu_transition is not None:
            np.save(f"{output_base}_mu_transition.npy", casida.mu_transition)

        with open(f"{output_base}_results.txt", "w") as fout:
            fout.write("# Casida TDDFT Results\n")
            fout.write(f"# Spin state: {spin_state}\n")
            fout.write(f"# TDA: {args.tda}\n")
            fout.write(f"# Matrix-free: {args.matrix_free}\n")
            fout.write(f"# Solver: {args.solver_method if args.matrix_free else 'direct'}\n")
            fout.write(f"# n_total_occ: {n_total_occ} (HOMO index: {n_total_occ-1})\n")
            fout.write(f"# n_occ: {args.n_occ}\n")
            fout.write(f"# n_unocc: {args.n_unocc}\n")
            fout.write(f"# XC: {args.xc}\n")
            fout.write(f"# USPP: {use_uspp_flag}\n")
            fout.write(f"# MPI ranks: {size}\n")
            fout.write("# Backend: CPU (MPI)\n")
            fout.write(f"# Build/Setup time: {t_build:.2f} s\n")
            fout.write(f"# Solve time: {t_solve:.2f} s\n")
            fout.write("#\n")
            fout.write("# State   Energy(eV)   Osc.Strength\n")
            for i, (w, osc) in enumerate(zip(omega, f)):
                fout.write(f"  {i+1:3d}    {w * 27.2114:10.4f}   {osc:12.6f}\n")

        print(f"    Saved: {output_base}_omega.npy")
        print(f"    Saved: {output_base}_f.npy")
        print(f"    Saved: {output_base}_Z.npy")
        if casida.mu_transition is not None:
            print(f"    Saved: {output_base}_mu_transition.npy")
        print(f"    Saved: {output_base}_results.txt")

        # ===== [10] Plot =====
        if args.plot:
            print("\n[10] Generating spectrum plot...")
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                
                # Auto-adjust x-axis range based on data
                omega_ev = omega * 27.2114
                energy_min = np.min(omega_ev)
                energy_max = np.max(omega_ev)
                energy_range = energy_max - energy_min
                
                # Add padding: 5% on each side, but at least 1 eV
                padding = max(energy_range * 0.05, 1.0)
                x_min = max(0.0, energy_min - padding)
                x_max = energy_max + padding
                
                # Create frequency grid for broadened spectrum
                n_points = max(2000, int((x_max - x_min) / 0.01))  # ~0.01 eV resolution
                freq = np.linspace(x_min, x_max, n_points)
                spectrum = smooth_spectrum(f, omega, freq, sigma=args.sigma)
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

                ax1.stem(omega_ev, f, basefmt=" ", markerfmt="ro", linefmt="r-")
                ax1.set_xlabel("Energy (eV)")
                ax1.set_ylabel("Oscillator strength")
                title_parts = [f"Casida TDDFT - {'TDA' if args.tda else 'Full'}"]
                title_parts.append(f"({args.xc})")
                title_parts.append(f"[{spin_state}]")
                if use_uspp_flag:
                    title_parts.append("+ USPP")
                ax1.set_title(" ".join(title_parts))
                ax1.set_xlim(x_min, x_max)
                ax1.grid(True, alpha=0.3)

                ax2.plot(freq, spectrum, "b-", linewidth=1.5)
                ax2.fill_between(freq, spectrum, alpha=0.3)
                ax2.set_xlabel("Energy (eV)")
                ax2.set_ylabel("Absorption (arb. units)")
                ax2.set_title(f"Broadened spectrum (σ = {args.sigma} eV)")
                ax2.set_xlim(x_min, x_max)
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(f"{output_base}_spectrum.png", dpi=150)
                print(f"    Saved: {output_base}_spectrum.png")
            except ImportError:
                print("    Warning: matplotlib not available, skipping plot")

        # ===== [11] Polariton postprocessing (if requested) =====
        if HAS_POLARITON_MODULE and hasattr(args, 'polariton') and args.polariton:
            if rank == 0:
                print("\n[11] Computing polaritonic effects...")
                sys.stdout.flush()
            
            polariton_data = setup_polariton(
                args, omega, f,
                mu_transition=casida.mu_transition if hasattr(casida, 'mu_transition') else None,
                comm=comm
            )
            
            if polariton_data is not None and rank == 0:
                save_polariton_results(polariton_data, output_base, args, comm=comm)
                print(f"    Saved polariton results with prefix: {output_base}_polariton_*")
                sys.stdout.flush()

        print("\n" + "=" * 60)
        print("Calculation complete!")
        print("=" * 60)


if __name__ == "__main__":
    main()
