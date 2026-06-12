#!/usr/bin/env python
"""
General QEpy script to generate input files for Casida TDDFT calculation.

This script runs a DFT SCF calculation using QEpy and exports:
  - rho_scf_*.xsf    : Ground-state electron density
  - psi_*.npy        : KS wavefunctions
  - eig_*.npy        : KS eigenvalues (in Hartree)
  - occs_*.npy       : Occupation numbers (normalized)

Usage:
    # Command-line arguments:
    python generate_inputs_qepy.py --geometry ag4.vasp --pseudo ag_pbe_v1.4.uspp.F.UPF
    python generate_inputs_qepy.py --geometry ag4.vasp --pseudo Ag_ONCV_PBE-1.2.upf --charge 0 --ecutwfc 40
    
    # Using configuration file:
    python generate_inputs_qepy.py --config example_config.json
    python generate_inputs_qepy.py --config example_config.json --ecutwfc 40  # Override ecutwfc from config
    
    # Configuration file format (JSON):
    # {
    #   "geometry": "AG4_PARALLEL/ag4.vasp",
    #   "pseudo": "AG4_PARALLEL/ag_pbe_v1.4.uspp.F.UPF",
    #   "workdir": "AG4_PARALLEL",
    #   "ecutwfc": 30.0,
    #   "nbnd": 70,
    #   ...
    # }

For SLURM submission, use the accompanying run_generate_inputs.sh script.
"""

import os
import sys
import argparse
import json
import numpy as np
from ase.io import read
from collections import Counter
# QEpy imports
from qepy.driver import Driver
from qepy.io import QEInput

# DFTpy imports for grid and density export
from dftpy.grid import DirectGrid
from dftpy.ions import Ions
from dftpy.field import DirectField

def read_z_valence_from_upf(path):
    """Best-effort Z_valence from a UPF file (for nbnd estimation)."""
    import re
    try:
        with open(path, "r", errors="ignore") as f:
            text = f.read(262144)
    except OSError:
        return None
    m = re.search(r"Z valence\s*=\s*([0-9.]+)", text, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"<PP_ZVALE[^>]*>\s*([0-9.]+)", text)
    if m:
        return float(m.group(1))
    return None


def parse_pseudo_list(pseudo_str):
    """
    Parse pseudopotential string into dictionary.
    
    Format: "Element1:file1.UPF,Element2:file2.UPF" or "file.UPF" (single element)
    """
    pseudo_dict = {}
    if ',' in pseudo_str:
        # Multiple pseudopotentials
        for pair in pseudo_str.split(','):
            if ':' in pair:
                elem, fname = pair.split(':')
                pseudo_dict[elem.strip()] = fname.strip()
            else:
                # Assume single file, try to infer element from filename
                fname = pair.strip()
                elem = os.path.basename(fname).split('_')[0].split('.')[0].capitalize()
                pseudo_dict[elem] = fname
    else:
        # Single pseudopotential
        if ':' in pseudo_str:
            elem, fname = pseudo_str.split(':')
            pseudo_dict[elem.strip()] = fname.strip()
        else:
            fname = pseudo_str.strip()
            elem = os.path.basename(fname).split('_')[0].split('.')[0].capitalize()
            pseudo_dict[elem] = fname
    
    return pseudo_dict


def get_atomic_masses():
    """Return dictionary of atomic masses (common elements)."""
    from ase.data import atomic_masses, chemical_symbols
    masses = {}
    for i, symbol in enumerate(chemical_symbols):
        if i > 0:  # Skip index 0
            masses[symbol] = atomic_masses[i]
    return masses


def main():
    parser = argparse.ArgumentParser(
        description='Generate input files for Casida TDDFT using QEpy',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments (but can be provided via config file)
    parser.add_argument('--geometry', type=str, required=False,
                        help='Geometry file (VASP, XYZ, etc.)')
    parser.add_argument('--pseudo', type=str, required=False,
                        help='Pseudopotential file(s). Format: "Element:file.UPF" or "file.UPF" for single element, or "El1:file1.UPF,El2:file2.UPF" for multiple')
    
    # Output options
    parser.add_argument('--output-prefix', type=str, default=None,
                        help='Prefix for output files (default: inferred from geometry filename)')
    parser.add_argument('--workdir', type=str, default='./',
                        help='Working directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as workdir)')
    
    # Calculation parameters
    parser.add_argument('--ecutwfc', type=float, default=30.0,
                        help='Plane-wave cutoff (Ry)')
    parser.add_argument('--ecutrho', type=float, default=None,
                        help='Charge density cutoff (Ry). Default: 10 * ecutwfc')
    parser.add_argument('--nbnd', type=int, default=None,
                        help='Number of bands (default: auto-calculated)')
    parser.add_argument('--charge', type=float, default=0.0,
                        help='Total charge (in units of e)')
    parser.add_argument('--conv-thr', type=float, default=1.0e-6,
                        help='SCF convergence threshold')
    parser.add_argument('--mixing-beta', type=float, default=0.3,
                        help='Mixing parameter for SCF')
    parser.add_argument('--electron-maxstep', type=int, default=200,
                        help='Maximum SCF iterations')
    
    # Grid options
    parser.add_argument('--grid-nr', type=int, nargs=3, default=None,
                        help='Grid dimensions [nx ny nz] (default: auto from QE)')
    
    # K-points
    parser.add_argument('--kpoints', type=int, nargs=3, default=[1, 1, 1],
                        help='K-points grid [kx ky kz]')
    parser.add_argument('--koffset', type=int, nargs=3, default=[0, 0, 0],
                        help='K-points offset [ox oy oz]')
    parser.add_argument('--gamma-only', action='store_true',
                        help='Force K_POINTS gamma instead of automatic grid')
    parser.add_argument('--nosym', action='store_true',
                        help='Set nosym=.true. in &system')
    parser.add_argument('--nosym-evc', action='store_true',
                        help='Set nosym_evc=.true. in &system')
    
    # QE options
    parser.add_argument('--prefix', type=str, default=None,
                        help='QE prefix (default: inferred from geometry filename)')
    parser.add_argument('--outdir', type=str, default='./tmp/',
                        help='QE output directory')
    parser.add_argument('--write-input-only', action='store_true',
                        help='Only write QE input file, do not run calculation')
    parser.add_argument('--config', type=str, default=None,
                        help='Configuration file (JSON format) to read parameters from. Command-line arguments override config file values.')
    
    # First, check if --config is provided in command line (before full parsing)
    # This allows us to load config and use it as defaults
    config_file = None
    if '--config' in sys.argv:
        idx = sys.argv.index('--config')
        if idx + 1 < len(sys.argv):
            config_file = sys.argv[idx + 1]
    
    # Load config file if provided
    config_values = {}
    if config_file:
        config_path = config_file
        if not os.path.isabs(config_path):
            # Try relative to current directory first
            if not os.path.exists(config_path):
                # Try relative to workdir if specified in config or command line
                workdir_from_cli = None
                if '--workdir' in sys.argv:
                    wd_idx = sys.argv.index('--workdir')
                    if wd_idx + 1 < len(sys.argv):
                        workdir_from_cli = sys.argv[wd_idx + 1]
                if workdir_from_cli and os.path.exists(os.path.join(workdir_from_cli, config_path)):
                    config_path = os.path.join(workdir_from_cli, config_path)
        
        if os.path.exists(config_path):
            print(f"Loading configuration from: {config_path}")
            with open(config_path, 'r') as f:
                config_values = json.load(f)
            
            # Apply config values as defaults (before parsing)
            # Convert config keys (kebab-case) to argparse format (snake_case)
            for key, value in config_values.items():
                arg_key = key.replace('-', '_')
                # Set as default if the argument exists
                for action in parser._actions:
                    if action.dest == arg_key:
                        action.default = value
                        print(f"  Set default {key} = {value} from config file")
                        break
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    # Now parse arguments (command-line will override config defaults)
    args = parser.parse_args()
    
    # ========== Validate required arguments ==========
    # Check if geometry and pseudo are provided (either via config or command line)
    if not args.geometry:
        parser.error("--geometry is required (provide via --config file or --geometry argument)")
    
    if not args.pseudo:
        parser.error("--pseudo is required (provide via --config file or --pseudo argument)")
    
    # ========== Setup ==========
    workdir = os.path.abspath(args.workdir)
    os.makedirs(workdir, exist_ok=True)
    original_cwd = os.getcwd()
    os.chdir(workdir)
    
    output_dir = args.output_dir if args.output_dir else workdir
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine output prefix from geometry filename if not provided
    if args.output_prefix is None:
        geom_basename = os.path.splitext(os.path.basename(args.geometry))[0]
        args.output_prefix = geom_basename
    
    if args.prefix is None:
        args.prefix = args.output_prefix
    
    # Output file names
    DENSITY_OUT = os.path.join(output_dir, f"rho_scf_{args.output_prefix}.xsf")
    PSI_OUT = os.path.join(output_dir, f"psi_{args.output_prefix}.npy")
    EIG_OUT = os.path.join(output_dir, f"eig_{args.output_prefix}.npy")
    OCCS_OUT = os.path.join(output_dir, f"occs_{args.output_prefix}.npy")
    QE_INPUT_OUT = os.path.join(workdir, f"{args.prefix}_scf.in")
    QE_OUTPUT_OUT = os.path.join(workdir, f"{args.prefix}_scf.out")
    
    # Set ecutrho if not provided
    if args.ecutrho is None:
        args.ecutrho = 10.0 * args.ecutwfc
    
    print("=" * 70)
    print("QEpy SCF Calculation for Casida TDDFT")
    print("=" * 70)
    print(f"Working directory: {workdir}")
    print(f"Output directory: {output_dir}")
    print(f"Geometry file: {args.geometry}")
    print(f"Pseudopotential(s): {args.pseudo}")
    print(f"Output prefix: {args.output_prefix}")
    print("=" * 70)
    
    # ========== Load structure ==========
    print(f"\n[1] Loading structure from {args.geometry}...")
    # Handle geometry path: if absolute, use as-is; if relative, resolve from workdir
    if os.path.isabs(args.geometry):
        geometry_path = args.geometry
    else:
        # Check if it exists in workdir first
        geometry_path = os.path.join(workdir, args.geometry)
        if not os.path.exists(geometry_path):
            # Try original working directory
            geometry_path = os.path.join(original_cwd, args.geometry)
    
    if not os.path.exists(geometry_path):
        raise FileNotFoundError(f"Geometry file not found: {args.geometry} (tried: {geometry_path})")
    
    atoms = read(geometry_path)
    print(f"    Number of atoms: {len(atoms)}")
    print(f"    Cell: {atoms.cell.lengths()}")
    print(f"    Species: {set(atoms.get_chemical_symbols())}")
    
    # Count atoms by element
    element_counts = Counter(atoms.get_chemical_symbols())
    print(f"    Composition: {dict(element_counts)}")
    
    # ========== Parse pseudopotentials ==========
    print(f"\n[2] Parsing pseudopotential files...")
    pseudo_dict = parse_pseudo_list(args.pseudo)
    print(f"    Pseudopotentials: {pseudo_dict}")
    
    # Verify all elements have pseudopotentials
    elements_needed = set(atoms.get_chemical_symbols())
    elements_provided = set(pseudo_dict.keys())
    if not elements_provided.issuperset(elements_needed):
        missing = elements_needed - elements_provided
        raise ValueError(f"Missing pseudopotentials for elements: {missing}")
    
    # Resolve pseudopotential file paths
    resolved_pseudo_dict = {}
    for elem, pseudo_file in pseudo_dict.items():
        if os.path.isabs(pseudo_file):
            resolved_pseudo_dict[elem] = pseudo_file
        else:
            # Try workdir first
            pseudo_path = os.path.join(workdir, pseudo_file)
            if not os.path.exists(pseudo_path):
                # Try original working directory
                pseudo_path = os.path.join(original_cwd, pseudo_file)
            if not os.path.exists(pseudo_path):
                raise FileNotFoundError(f"Pseudopotential file not found: {pseudo_file} (tried: {pseudo_path})")
            resolved_pseudo_dict[elem] = pseudo_path
    
    # Build atomic_species list
    masses = get_atomic_masses()
    atomic_species = []
    for elem in sorted(elements_needed):
        mass = masses.get(elem, 1.0)  # Default mass if not found
        pseudo_file = os.path.basename(resolved_pseudo_dict[elem])  # Use basename for QE input
        atomic_species.append(f"{elem} {mass:.6f} {pseudo_file}")
    
    # ========== Calculate number of bands ==========
    if args.nbnd is None:
        # Estimate: count valence electrons from pseudopotential files
        # Try to read Z_val from pseudopotential files, otherwise use common values
        valence_electrons = {
            'H': 1, 'He': 2, 'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Ne': 8,
            'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Ar': 8,
            'K': 1, 'Ca': 2, 'Sc': 3, 'Ti': 4, 'V': 5, 'Cr': 6, 'Mn': 7, 'Fe': 8, 'Co': 9, 'Ni': 10,
            'Cu': 11, 'Zn': 12, 'Ga': 3, 'Ge': 4, 'As': 5, 'Se': 6, 'Br': 7, 'Kr': 8,
            'Rb': 1, 'Sr': 2, 'Y': 3, 'Zr': 4, 'Nb': 5, 'Mo': 6, 'Tc': 7, 'Ru': 8, 'Rh': 9, 'Pd': 10,
            'Ag': 19, 'Cd': 12, 'In': 3, 'Sn': 4, 'Sb': 5, 'Te': 6, 'I': 7, 'Xe': 8,
        }
        
        # Try to get Z_val from pseudopotential files
        total_valence = 0
        for elem in atoms.get_chemical_symbols():
            if elem in valence_electrons:
                total_valence += valence_electrons[elem]
            else:
                z_from_upf = None
                if elem in resolved_pseudo_dict:
                    z_from_upf = read_z_valence_from_upf(resolved_pseudo_dict[elem])
                if z_from_upf is not None:
                    total_valence += int(round(z_from_upf))
                else:
                    total_valence += valence_electrons.get(elem, 10)
        
        n_occupied = int(np.ceil(total_valence / 2.0))
        # Need extra bands for unoccupied states (typically 1.5-2x occupied)
        args.nbnd = max(50, int(n_occupied * 2.0))  # At least 50, or 2x occupied
        print(f"    Auto-calculated nbnd: {args.nbnd} (estimated {n_occupied} occupied from {total_valence} valence electrons)")
    
    # ========== Grid dimensions ==========
    if args.grid_nr is None:
        # Will be determined from QE output
        GRID_NR = None
    else:
        GRID_NR = args.grid_nr
    
    # ========== QEpy Setup ==========
    print("\n[3] Setting up QEpy calculation...")
    
    qe_options = {
        '&control': {
            'calculation': "'scf'",
            'prefix': f"'{args.prefix}'",
            'pseudo_dir': f"'{workdir}/'",
            'outdir': f"'{args.outdir}'",
            'verbosity': "'high'",
        },
        '&system': {
            'ibrav': 0,
            'ecutwfc': args.ecutwfc,
            'ecutrho': args.ecutrho,
            'nbnd': args.nbnd,
            'occupations': "'fixed'",
        },
        '&electrons': {
            'conv_thr': args.conv_thr,
            'mixing_beta': args.mixing_beta,
            'electron_maxstep': args.electron_maxstep,
        },
        'atomic_species': atomic_species,
    }

    if args.nosym:
        qe_options['&system']['nosym'] = True
    if args.nosym_evc:
        qe_options['&system']['nosym_evc'] = True

    use_gamma = args.gamma_only or (list(args.kpoints) == [1, 1, 1] and list(args.koffset) == [0, 0, 0])
    if use_gamma:
        qe_options['k_points gamma'] = []
    else:
        qe_options['k_points automatic'] = [f"{' '.join(map(str, args.kpoints))} {' '.join(map(str, args.koffset))}"]
    
    # Add charge if non-zero
    if abs(args.charge) > 1e-6:
        qe_options['&system']['tot_charge'] = args.charge
    
    # Update with atomic positions
    qe_options = QEInput.update_atoms(atoms, qe_options=qe_options, extrapolation=False)
    
    # Write input file
    QEInput().write_qe_input(QE_INPUT_OUT, qe_options=qe_options)
    print(f"    Input file written: {QE_INPUT_OUT}")
    
    if args.write_input_only:
        print("\n[--write-input-only] Skipping calculation.")
        print(f"QE input file saved: {QE_INPUT_OUT}")
        return
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # ========== Run SCF ==========
    print("\n[4] Running SCF calculation...")
    print(f"    ecutwfc = {args.ecutwfc} Ry")
    print(f"    ecutrho = {args.ecutrho} Ry")
    print(f"    nbnd = {args.nbnd}")
    print(f"    charge = {args.charge}")
    print(f"    conv_thr = {args.conv_thr}")
    print("    This may take a while...")
    sys.stdout.flush()  # Ensure output is written
    
    driver = Driver(qe_options=qe_options, atoms=atoms, logfile=QE_OUTPUT_OUT)
    print("    Driver initialized")
    sys.stdout.flush()
    
    try:
        print("    Starting SCF iteration...")
        sys.stdout.flush()
        driver.scf()
        print("    SCF converged!")
        sys.stdout.flush()
    except Exception as e:
        print(f"    Error during SCF: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        try:
            driver.stop()
        except:
            pass
        sys.exit(1)
    
    # ========== Extract Data ==========
    print("\n[5] Extracting eigenvalues...")
    sys.stdout.flush()
    
    try:
        # QEpy returns eigenvalues in Rydberg, convert to Hartree
        eig_ry = driver.get_eigenvalues()
        eig_ha = eig_ry / 2.0  # Ry -> Ha
        print(f"    Number of bands: {len(eig_ha)}")
        sys.stdout.flush()
    except Exception as e:
        print(f"    ERROR: Failed to extract eigenvalues: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        try:
            driver.stop()
        except:
            pass
        sys.exit(1)
    
    # Find HOMO-LUMO gap
    try:
        occs_temp = driver.get_occupation_numbers()
        n_occ = int(np.sum(occs_temp > 0.1))
        if n_occ > 0 and n_occ < len(eig_ha):
            gap_ev = (eig_ha[n_occ] - eig_ha[n_occ-1]) * 27.2114
            print(f"    HOMO-LUMO gap: {gap_ev:.4f} eV (HOMO: band {n_occ}, LUMO: band {n_occ+1})")
        sys.stdout.flush()
    except Exception as e:
        print(f"    Warning: Could not compute HOMO-LUMO gap: {e}")
        sys.stdout.flush()
    
    print("\n[6] Extracting occupation numbers...")
    sys.stdout.flush()
    try:
        occs = driver.get_occupation_numbers()
        # Normalize occupations (QE uses 2.0 for fully occupied)
        occs_norm = occs / np.max(occs) if np.max(occs) > 0 else occs
        print(f"    Total electrons: {np.sum(occs):.2f}")
        sys.stdout.flush()
    except Exception as e:
        print(f"    ERROR: Failed to extract occupation numbers: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        try:
            driver.stop()
        except:
            pass
        sys.exit(1)
    
    print("\n[7] Extracting wavefunctions...")
    sys.stdout.flush()
    
    try:
        psi_raw = driver.get_wave_function()
        print(f"    Raw wavefunction shape: {psi_raw[0].shape}")
        sys.stdout.flush()
    except Exception as e:
        print(f"    ERROR: Failed to extract wavefunctions: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        try:
            driver.stop()
        except:
            pass
        sys.exit(1)
    
    # Get grid info from QE
    ions = driver.get_dftpy_ions()
    
    # Calculate grid dimensions from wave function
    nr_qe = int(round(psi_raw[0].size ** (1/3)))
    print(f"    QE grid size: {nr_qe}³")
    
    # Create grid matching QE dimensions
    grid_qe = DirectGrid(lattice=ions.cell, nr=[nr_qe, nr_qe, nr_qe])
    
    # Determine target grid
    if GRID_NR is None:
        GRID_NR = [nr_qe, nr_qe, nr_qe]
        print(f"    Using QE grid: {GRID_NR}")
    else:
        print(f"    Converting to target grid: {GRID_NR}")
    
    # Convert wavefunctions to DirectField objects
    if nr_qe == GRID_NR[0]:
        # Same grid, direct conversion
        psi_fields = [driver.data2field(ps, grid=grid_qe) for ps in psi_raw]
    else:
        # Need to interpolate to target grid
        target_grid = DirectGrid(lattice=ions.cell, nr=GRID_NR)
        from dftpy.utils import grid_map_data
        psi_temp = [driver.data2field(ps, grid=grid_qe) for ps in psi_raw]
        psi_fields = [grid_map_data(psi, grid=target_grid) for psi in psi_temp]
    
    # Save as numpy array (list of field data)
    psi_array = np.array([np.asarray(psi) for psi in psi_fields])
    print(f"    Final wavefunction array shape: {psi_array.shape}")
    
    print("\n[8] Extracting density...")
    sys.stdout.flush()
    
    try:
        rho = driver.get_density()
        rho_grid = driver.get_dftpy_grid()
        sys.stdout.flush()
    except Exception as e:
        print(f"    ERROR: Failed to extract density: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        try:
            driver.stop()
        except:
            pass
        sys.exit(1)
    
    # Convert to DirectField for export
    rho_field = DirectField(grid=rho_grid, data=rho)
    print(f"    Density grid: {rho_grid.nr}")
    print(f"    Total electrons (from density): {rho_field.integral():.4f}")
    
    # ========== Save Files ==========
    print("\n[9] Saving output files...")
    
    # Save eigenvalues (Hartree)
    np.save(EIG_OUT, eig_ha)
    print(f"    Saved: {EIG_OUT}")
    
    # Save occupations (normalized)
    np.save(OCCS_OUT, occs_norm)
    print(f"    Saved: {OCCS_OUT}")
    
    # Save wavefunctions
    np.save(PSI_OUT, psi_array)
    print(f"    Saved: {PSI_OUT}")
    
    # Save density to XSF format
    ions_dftpy = Ions.from_ase(atoms)
    rho_field.write(DENSITY_OUT, ions=ions_dftpy)
    print(f"    Saved: {DENSITY_OUT}")
    
    # ========== Cleanup ==========
    print("\n[10] Stopping QEpy driver...")
    try:
        driver.stop()
    except Exception as e:
        print(f"    Warning during cleanup: {e}")
        # Continue anyway - files are already saved
    
    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("Output files generated:")
    print(f"  - {DENSITY_OUT} : Ground-state density (XSF format)")
    print(f"  - {PSI_OUT}     : KS wavefunctions ({psi_array.shape})")
    print(f"  - {EIG_OUT}     : Eigenvalues in Hartree ({len(eig_ha)} bands)")
    print(f"  - {OCCS_OUT}    : Normalized occupations")
    print(f"  - {QE_INPUT_OUT} : QE input file")
    print("=" * 70)
    print("\nNext: run Casida with norm-conserving --pseudo-map (see scripts/run_casida.sh).")


if __name__ == "__main__":
    main()
