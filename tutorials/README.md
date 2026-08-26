# CasidaPy tutorials

Demo notebooks grouped by topic. Shared assets stay in this directory:

- `setup_env.sh` — Amarel / `libffi` + `PYTHONPATH` (source before starting the kernel)
- `paths.py` — resolves this root and ONCV UPFs (`O_ONCV_PBE-1.2.upf`, `H_ONCV_PBE-1.2.upf`)
- ONCV pseudopotentials used by H₂O / XAS demos

```bash
source /projectsn/mp1009_1/am4655/casidapy/tutorials/setup_env.sh
```

## Layout

| Folder | Notebooks |
|--------|-----------|
| [`tddft/`](tddft/) | Plane-wave & GTO Casida on H₂O; GTO call patterns |
| [`xas/`](xas/) | CVS / XAS walkthrough; Hirshfeld embedding demo |
| [`qed/`](qed/) | Pauli–Fierz QED-TDA, spin-flip QED, SOC+QED molecules |
| [`soc/`](soc/) | Atomic TDDFT + SOC (C, Si, Ge) |

### TDDFT (`tddft/`)

| Notebook | What it shows |
|----------|----------------|
| [`h2o_tddft_approach1_highlevel.ipynb`](tddft/h2o_tddft_approach1_highlevel.ipynb) | One-shot QEpy → `run_casida_in_memory` |
| [`h2o_tddft_approach2_stepwise.ipynb`](tddft/h2o_tddft_approach2_stepwise.ipynb) | Explicit `CasidaKS_MPI` stages |
| [`h2o_gto_vs_pw.ipynb`](tddft/h2o_gto_vs_pw.ipynb) | Same molecule, GTO vs PW hosts |
| [`gto_kernel_patterns.ipynb`](tddft/gto_kernel_patterns.ipynb) | `extract_gto_kernel` / SF / custom windows |

### XAS & embedding (`xas/`)

| Notebook | What it shows |
|----------|----------------|
| [`xas_cvs_walkthrough.ipynb`](xas/xas_cvs_walkthrough.ipynb) | GTO CVS, inject, QE Hirshfeld reconstruct |
| [`hirshfeld_partition_demo.ipynb`](xas/hirshfeld_partition_demo.ipynb) | Line cuts of `w_A`, `ρ_env`, `V_env` |

Imports (current API)::

```python
from casidapy import xas
from casidapy.embed import build_ae_embedding_potential
from casidapy.xas import run_xas_gto, run_xas_reconstruct, plot_sticks
```

### QED (`qed/`)

| Notebook | What it shows |
|----------|----------------|
| [`qed_tda_pes_gto.ipynb`](qed/qed_tda_pes_gto.ipynb) | Pauli–Fierz QED-TDA + PES (GTO) |
| [`qed_sf_tda_ethylene.ipynb`](qed/qed_sf_tda_ethylene.ipynb) | Spin-flip QED-TDA |
| [`soc_qed_formaldehyde.ipynb`](qed/soc_qed_formaldehyde.ipynb) | TDDFT → SOC → QED on H₂CO |
| [`soc_qed_iodine.ipynb`](qed/soc_qed_iodine.ipynb) | Same stack on I₂ |

```python
from casidapy.utils.qed import QEDOptions, solve_qed_tda
from casidapy.adapter.pyscf import extract_gto_kernel
```

### SOC (`soc/`)

| Notebook | What it shows |
|----------|----------------|
| [`soc_atoms_c_si_ge.ipynb`](soc/soc_atoms_c_si_ge.ipynb) | Group-14 atoms: TDA vs TDA+SOC |

```python
from casidapy.utils.soc import soc_ao_integrals, solve_soc_si
```

## Package map (after refactor)

```
casidapy/
  adapter/   # qepy, pyscf, stddft, mpi_pyscf
  utils/     # casida_utils, davidson, nac, qed, soc, uspp
  embed/     # potential, hirshfeld
  xas/       # cvs, reconstruct, spectrum
  kernels/   # plane_wave, gto
```
