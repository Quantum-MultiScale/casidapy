# CasidaPy

Linear-response TDDFT (Casida / RPA) for Kohn–Sham and embedded subsystem calculations. CasidaPy provides:

- **Standalone MPI Casida** from exported SCF data (density, orbitals, eigenvalues)
- **QEpy adapter** to build inputs directly from a QEpy driver
- **Subsystem coupling** (Pavanello non-additive kernel) for multi-fragment eDFTpy workflows

The Python package lives in `casidapy/`. Install with pip; entry points are `casidapy-run` and `casidapy-plot`.

## Installation

```bash
cd CasidaPy
pip install -e .

# Optional extras
pip install -e ".[qepy]"      # QEpy integration
pip install -e ".[libxc]"      # triplet excitations (pylibxc)
pip install -e ".[plotting]"   # matplotlib spectrum plots
pip install -e ".[all]"
```

Core dependencies: `numpy`, `scipy`, `mpi4py`, `dftpy`, `ase`.

For the full embedded workflow you also need **eDFTpy** (with `task = casida`) and **QEpy**.

## Usage overview

| Mode | When to use | Entry point |
|------|-------------|-------------|
| **CLI** (`casidapy-run`) | Single system; SCF outputs already on disk | `scripts/run_casida.sh` or `casidapy-run` |
| **Python API** | Custom scripts, notebooks, tests | `run_casida_in_memory`, `CasidaKS_MPI` |
| **eDFTpy embedded** | Multi-fragment embedding + inter-fragment coupling | `python -m edftpy input.ini` |

Step-by-step H₂O tutorials: `tutorials/h2o_tddft_approach1_highlevel.ipynb` (one-call) and `tutorials/h2o_tddft_approach2_stepwise.ipynb` (explicit stages).

---

## 1. Standalone Casida (SCF → Casida)

### Step 1: Generate SCF outputs (QEpy)

Run QEpy SCF **as a single MPI rank** (QE parallelises internally; multi-rank Python launch breaks wavefunction gather for USPP).

```bash
# Via SLURM wrapper
sbatch scripts/run_scf.sh \
  --geometry path/to/system.vasp \
  --pseudo path/to/Element.UPF \
  --workdir ./my_run \
  --output-prefix system

# Or directly
python casidapy/generate_inputs_qepy.py \
  --geometry system.vasp --pseudo Element.UPF --workdir ./my_run
```

Expected files in `workdir`:

| File | Content |
|------|---------|
| `rho_scf_<prefix>.xsf` | Ground-state density |
| `psi_<prefix>.npy` | KS wavefunctions |
| `eig_<prefix>.npy` | Eigenvalues (Hartree) |
| `occs_<prefix>.npy` | Occupation numbers |

### Step 2: Run Casida (MPI)

```bash
sbatch scripts/run_casida.sh \
  --workdir ./my_run \
  --atoms system.vasp \
  --density rho_scf_system.xsf \
  --psi psi_system.npy \
  --eigs eig_system.npy \
  --occs occs_system.npy \
  --pseudo-map "Ag:Element.UPF" \
  --xc PBE \
  --n-occ 10 --n-unocc 20 --n-states 30 \
  --matrix-free --solver-method eigsh \
  --output-prefix casida_system
```

Ultrasoft pseudopotentials:

```bash
sbatch scripts/run_casida.sh \
  ... \
  --use-uspp --uspp-map "Ag:ag_pbe_v1.4.uspp.F.UPF" \
  --xc PBE
```

### Input file (`.in`)

Pass all options via a key–value file; CLI flags override file values.

```bash
casidapy-run --input-file sample_h2o_pbe.in
# or
sbatch scripts/run_casida.sh --input-file sample_h2o_pbe.in
```

Sample files in the repo root: `sample_h2o_pbe.in`, `sample_h2o_pbe0.in`, `sample_ag5_of.in`.

Example snippet:

```ini
workdir = ./my_run
atoms = h2o.vasp
density = rho_scf_h2o.xsf
psi = psi_h2o.npy
eigs = eig_h2o.npy
occs = occs_h2o.npy
pseudo_map = H:H_ONCV_PBE-1.2.upf,O:O_ONCV_PBE-1.2.upf
n_occ = 4
n_unocc = 30
n_states = 20
n_total_occ = 4
matrix_free
solver_method = eigsh
xc = PBE
output_prefix = casida_h2o
plot
```

### Full pipeline (one SLURM job)

```bash
sbatch scripts/run_full_pipeline.sh \
  --scf "--geometry system.vasp --pseudo Element.UPF --workdir ./my_run --output-prefix system" \
  --casida "--workdir ./my_run --atoms system.vasp --density rho_scf_system.xsf --psi psi_system.npy --eigs eig_system.npy --occs occs_system.npy --pseudo-map Ag:Element.UPF --xc PBE --output-prefix casida_system"
```

Use `--skip-scf` or `--skip-casida` to run only one stage.

### Key CLI options

| Option | Description |
|--------|-------------|
| `--n-occ`, `--n-unocc` | Active occupied / unoccupied bands in the Casida window |
| `--n-total-occ` | Total occupied count (HOMO index + 1 = LUMO index). When set, the `n_occ` valence bands immediately below HOMO are used |
| `--n-states` | Number of excitation roots to solve for |
| `--matrix-free` | Avoid building the full transition matrix (recommended for large grids) |
| `--solver-method` | `lobpcg` or `eigsh` (matrix-free only) |
| `--tda` | Tamm–Dancoff approximation (no B matrix) |
| `--of-context` | OF-inspired functional context inside KS-Casida (needs `pseudo_map`) |
| `--spin-state` | `singlet` (default) or `triplet` (needs pylibxc) |
| `--use-uspp` | Ultrasoft augmentation in Casida matrix elements |

---

## 2. Python API

### One-call solver

```python
from casidapy import CasidaInputs, CasidaOptions, run_casida_in_memory
from casidapy.qepy_adapter import slice_active_space

# ... build grid, rho_ks, psi, eigs, occs ...

opts = CasidaOptions(
    n_occ=4, n_unocc=30, n_states=20, n_total_occ=4,
    matrix_free=True, solver_method="eigsh", xc="PBE",
)
inputs = CasidaInputs(atoms=atoms, grid=grid, rho_ks=rho, psi=psi, eigs=eigs, occs=occs)
results = run_casida_in_memory(inputs, opts, comm=comm)

omega_ev = results.omega * 27.2114   # Hartree → eV
f = results.f                        # oscillator strengths
```

### From a QEpy driver

```python
from casidapy.qepy_adapter import extract_casida_inputs_from_qepy_driver, slice_active_space
from casidapy import run_casida_in_memory

inputs, opts = extract_casida_inputs_from_qepy_driver(driver, subcell, use_eDFTpy=False)
opts.n_states = 20
opts.matrix_free = True
results = run_casida_in_memory(inputs, opts, comm=comm)
```

Set `use_eDFTpy=True` when called from an eDFTpy fragment driver (charge grid, MPI-safe density broadcast).

### Stepwise control (`CasidaKS_MPI`)

See `tutorials/h2o_tddft_approach2_stepwise.ipynb` for the explicit sequence:

1. `slice_active_space(eigs, psi, n_occ, n_unocc, n_total_occ=...)`
2. `casida.set_active_orbitals(...)`
3. `casida.setup_matrix_free()` or `casida.build_matrices()`
4. `casida.solve_matrix_free(k=n_states)` or `casida.solve(k=n_states)`
5. `casida.oscillator_strengths(k=n_states)`

### Active-space windowing

`slice_active_space` is the single source of truth for orbital windows:

- **`n_total_occ=None`**: use the lowest `n_occ` bands as occupied; unoccupied from index `n_occ` upward.
- **`n_total_occ` set**: LUMO is band `n_total_occ`. Occupied window is bands `[n_total_occ - n_occ, n_total_occ)` (the top `n_occ` occupied states). Unoccupied from `n_total_occ`.

When using the QEpy adapter without overrides, `n_occ` / `n_unocc` are inferred from occupation numbers (`> 0.01` / `< 0.01`).

---

## 3. eDFTpy embedded subsystem Casida

For multi-fragment embedded DFT, eDFTpy drives SCF embedding, per-fragment Casida (via CasidaPy in each subsystem MPI communicator), and Pavanello coupling on the global grid (rank 0).

```bash
mpirun -n 10 python -m edftpy input.ini
```

Minimal `.ini` structure:

```ini
[JOB]
task = casida

[TD]
number_of_states   = 10
number_of_bands    = 40      # must match QE nbnd
coupling_energy_threshold = 0.001   # Ha; drop weakly coupled fragment states
matrix_free = true
solver_method = eigsh
casida_plot_txt = casida_merged_system.txt

[GSYSTEM]
cell-file = system.xyz
exc-xc = PBE
grid-ecut = 8163.5

[SUB_FRAG_0]
calculator = qe
embed = KE XC
cell-cut = 5.5 5.5 5.5
cell-index = 0:12
basefile = qe_system.in
decompose-method = manual

[SUB_FRAG_1]
calculator = qe
...
```

**MPI layout:** ranks are split across fragments (e.g. `[5, 5]` for two subsystems on 10 ranks). Each fragment runs Casida on its `comm_sub` in parallel; a WORLD barrier precedes gather/coupling on rank 0.

**Outputs:**

- Per-fragment sticks in the log (`Subsystem N Casida: omega`, `f`)
- `casida_merged_*.txt` — coupled + uncoupled merged spectrum
- `casida_uncoupled_*.txt` — fragment-local states only

**Tuning `coupling_energy_threshold`:** threshold in Hartree on fragment excitation energies before coupling. Lower values keep more charge-transfer states in the coupled problem; very large values couple everything (often unphysical for distant fragments).

CasidaPy coupling implementation: `casidapy.subsystem_coupling.run_subsystem_casida`.

---

## 4. Plotting

```bash
# Broadened spectrum from casida output text
casidapy-plot --input casida_h2o.txt --sigma 0.1 --output spectrum.png

# Uncoupled / merged comparison (eDFTpy outputs)
python -m casidapy.plot_uncoupled_spectrum --help
```

---

## 5. Package layout

```
casidapy/
  casida_api.py          # CasidaInputs, CasidaOptions, CasidaResults
  casida_engine.py       # CasidaKS_MPI, run_casida_in_memory
  casida_utils.py        # transitions, normalization, MPI helpers
  qepy_adapter.py        # extract_casida_inputs_from_qepy_driver, slice_active_space
  subsystem_coupling.py  # Pavanello coupling, run_subsystem_casida
  run_casida_parallel_generic.py  # casidapy-run CLI
  generate_inputs_qepy.py         # SCF export for standalone workflow
  plot_casida_spectrum.py         # casidapy-plot
  uspp.py                # ultrasoft projectors and augmentation
  davidson.py            # Davidson diagonalization (large dense matrices)
  polariton_handler.py   # optional polariton extensions
scripts/                 # SLURM wrappers (run_scf.sh, run_casida.sh, run_full_pipeline.sh)
tutorials/               # H₂O Jupyter notebooks
tests/                   # pytest unit tests
```

## Tests

```bash
pip install -e ".[all]"
pytest tests/ -v
```

## License

MIT — see `LICENSE`.
