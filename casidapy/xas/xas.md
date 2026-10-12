# CasidaPy XAS module (`casidapy.xas`)

This document describes the four Python modules under `casidapy/casidapy/xas/`:
`__init__.py`, `cvs.py`, `reconstruct.py`, and `spectrum.py`. Together they implement
**core-excited valence-state (CVS) Tamm–Dancoff approximation (TDA)** X-ray absorption
spectroscopy (XAS) on all-Gaussian-type-orbital (GTO) molecules and on periodic
plane-wave (PW) hosts via QEpy embedding reconstruction.

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │           casidapy.xas (__init__)        │
                    │  run_xas_gto / run_xas_reconstruct       │
                    └─────────────────────────────────────────┘
                           │                    │
              Workflow 1   │                    │  Workflow 2
              (all GTO)  │                    │  (QE → AE core → PW)
                           ▼                    ▼
                    ┌──────────┐         ┌──────────────┐
                    │  cvs.py  │◄────────│ reconstruct.py│
                    │ cores,   │ inject  │ V_env AE SCF  │
                    │ inject,  │         └──────────────┘
                    │ CVS-TDA  │                ▲
                    └────┬─────┘                │ build_ae_embedding_potential
                         │                      │ (casidapy.embed)
                         ▼
                    ┌──────────┐
                    │spectrum.py│  sticks, plots, summaries
                    └──────────┘
```

### Three public workflows

| Workflow | Entry point | Host virtuals | Core source |
|----------|-------------|---------------|-------------|
| 1. All-GTO | `run_xas_gto` (`run_cvs_gto_from_mf`) | PySCF KS virtuals from `mf` | `core_from_mf(mf)` |
| 2. QE reconstruct | `run_xas_reconstruct` (`run_reconstruct_cvs_from_driver`) | QE PW bands projected to GTO AO | AE SCF in frozen `V_env`; Casida `K` is GTO-projected |
| 3. Manual inject | `extract_fragment_core` + `inject_core_orbitals` | User-built kernel | Fragment AE SCF on geometry cut |

Under CVS the **occupied** Casida active space is replaced by imported core orbitals only;
**virtual** orbitals remain those of the full-system host (GTO or PW).

### Related packages (not in this directory)

- `casidapy.embed` — builds frozen embedding potential `V_env` (Hirshfeld default).
- `casidapy.adapter.pyscf` / `casidapy.adapter.qepy` — construct `GTOKernel` / `PlaneWaveKernel`.
- `casidapy.casida_engine` — `CasidaKS_MPI.xas(reconstruct=True, ...)` wraps workflow 2.

---

## `__init__.py`

Single import surface re-exporting the public API and two aliases.

| Name | Meaning |
|------|---------|
| `run_xas_gto` | Alias for `run_cvs_gto_from_mf` |
| `run_xas_reconstruct` | Alias for `run_reconstruct_cvs_from_driver` |

All symbols listed in `__all__` are re-exported from `cvs`, `reconstruct`, `spectrum`, and
`casidapy.embed` (`build_ae_embedding_potential`, `read_upf_z_valence`).

---

## `cvs.py`

CVS building blocks: fragment geometry, core selection, PW field mapping, virtual pruning,
core injection, and CVS-TDA drivers. Module constants include `DEFAULT_OXIDATION_STATES`,
`_EDGE_SHELL`, `_SHELL_ALIAS`, and radial-quadrature defaults (`RADIAL_RMAX_BOHR`, etc.).

### Classes

#### `CoreOrbitals` (dataclass)

Container for fragment all-electron (AE) core molecular orbitals.

| Field | Description |
|-------|-------------|
| `energies` | Core MO energies (Ha), shape `(n_core,)`. |
| `mo_coeff` | MO coefficients in fragment AO basis, `(nao, n_core)`; may be complex after SOC. |
| `fragment_mol` | PySCF `Mole` used for the fragment SCF. |
| `atom_indices` | Parent-geometry atom indices that formed the fragment. |
| `edge` | Edge label (`"K"`, `"L"`, `"M"`). |
| `shell` | Resolved shell string (e.g. `"1s"`, `"2p"`). |
| `meta` | Dict of diagnostics (SCF energy, fragment mode, reference shifts, etc.). |

#### `FragmentSpec` (dataclass)

Specification for a geometry cut used in `extract_fragment_core`.

| Field | Description |
|-------|-------------|
| `atom_indices` | Parent indices in the fragment SCF. |
| `charge`, `spin` | PySCF SCF charge and `2S`. |
| `edge_atom_indices` | Which parent atoms contribute imported cores (default: library logic). |
| `mode` | `"user"`, `"neutral_first_shell"`, or `"radius"`. |
| `meta` | Selection diagnostics. |

**`FragmentSpec.user(...)`** — classmethod building an explicit user fragment from indices,
charge, spin, and optional edge atoms.

### Fragment geometry and oxidation

| Function | Description |
|----------|-------------|
| `_as_positions_symbols(mol_or_atoms)` | Extract Cartesian positions (Å) and element symbols from ASE `Atoms` or PySCF `Mole`. |
| `select_atoms_in_radius(mol_or_atoms, center, radius_ang, *, elements=None)` | Return parent atom indices within `radius_ang` of a center (atom index or XYZ). Optional element filter. |
| `formal_oxidation_charge(symbols, oxidation_states=None)` | Sum formal oxidation numbers using `DEFAULT_OXIDATION_STATES` (override per element). Raises if unknown. |
| `resolve_oxidation_state(symbol, oxidation_state=None, oxidation_states=None)` | Formal oxidation for one atom; explicit override wins. |
| `_covalent_radii_ang(symbols)` | Covalent radii (Å) from ASE or internal fallback table. |
| `_bond_neighbors(pos, symbols, *, bond_factor=1.25)` | Undirected neighbor lists from covalent-radii distance cutoffs. |
| `select_neutral_first_shell(mol_or_atoms, center, *, bond_factor, max_atoms, target_charge, oxidation_states, complete_ligands)` | Build a near-neutral fragment: edge + first shell, optionally growing until formal charge matches `target_charge` (e.g. TiO₂ → neutral cluster). Returns `FragmentSpec`. |
| `select_xas_fragment(mol_or_atoms, edge_atom, *, mode, ...)` | High-level fragment picker. Modes: `"neutral_first_shell"`, `"radius"`, or `"user"` (with explicit `atom_indices`). |

### Core orbital selection

| Function | Description |
|----------|-------------|
| `_resolve_shell(edge, shell=None)` | Map edge (`K`/`L`/`M`) or shell alias (`1s`, `2p`, …) to `(label, n, ℓ)`. |
| `_build_fragment_mole(parent_mol, atom_indices, *, basis, charge, spin, verbose)` | Build PySCF `Mole` for a sub-geometry of a parent molecule. |
| `_ao_shell_mask(mol, n, ell)` | Boolean mask of AOs on principal `(n, ℓ)`. |
| `_ao_mask_for_atoms(mol, atom_ids, base_mask)` | Restrict an AO mask to specific atoms in the fragment. |
| `_select_core_mos(mf, edge, shell, edge_atom_indices, n_core_per_atom, soc, ...)` | Internal: pick core MO indices/energies/coefficients from a mean-field object. |
| `core_from_mf(mf, *, edge, shell, edge_atom_indices, n_core_per_atom, soc, atom_indices, verbose)` | **Public:** extract `CoreOrbitals` from a converged PySCF `mf` (RKS/UKS). Handles K/L/M edges, optional spin–orbit coupling (SOC) for L/M. |
| `extract_fragment_core(parent_mol, atom_indices=None, *, fragment, edge, shell, basis, xc, charge, spin, soc, edge_atom_indices, n_core_per_atom, max_cycle, level_shift, use_gpu, verbose)` | Run AE fragment SCF on a geometry cut, then `core_from_mf`. Supports GPU fragment SCF via `gpu4pyscf` when available. |
| `apply_orbital_soc(mf, mo_coeff, mo_energy, shell, *, atom_indices, verbose)` | Diagonalize atomic SOC in the core-shell spin-orbital subspace; return j-adapted complex MOs. |

### PW field mapping and overlap metrics

| Function | Description |
|----------|-------------|
| `_grid_coords_angstrom(grid)` | Flatten DFTpy grid coordinates (Bohr) for `eval_ao`. |
| `core_mos_to_pw_fields(core, grid, *, normalize, use_gpu, comm, blksize)` | Evaluate fragment core MOs on a DFTpy `DirectGrid`; returns list of `DirectField`. MPI block assembly when `comm` is set. |
| `core_virtual_overlap_matrix_grid(core_orbitals, virtual_orbitals, grid, *, use_gpu)` | `\|⟨core\|virt⟩\|` on the charge grid; shape `(n_core, n_virt)`. |
| `core_virtual_overlap_matrix_gto(core_mo_coeff, virt_mo_coeff, ovlp)` | `\|⟨core\|virt⟩\|` in AO basis with overlap matrix `S`. |
| `virtual_overlap_scores(overlap_matrix, *, reduction="max")` | Per-virtual score from overlap matrix (`max` or `sum`/`l2`). |

### Transition dipole helpers

| Function | Description |
|----------|-------------|
| `_grid_charge_centroid(orbitals, grid)` | Charge centroid `⟨r⟩` of `Σ\|ψ\|²` (Bohr). |
| `_lebedev_unit_sphere(n_ang)` | Lebedev angular quadrature on the unit sphere. |
| `absorber_radial_grid(center_bohr, *, r_max, n_rad, n_ang)` | Absorber-centered spherical quadrature points and weights (Bohr). |
| `_field_on_nr(grid, field)` | Reshape field array to `grid.nr`. |
| `trilinear_sample(grid, field, coords_bohr)` | Periodic-boundary trilinear interpolation of a grid field at Bohr coordinates. |
| `fft_gradient_components(grid, field)` | FFT mesh gradient `∇field` (three components). |
| `absorber_origin_bohr(core, fallback=None)` | Absorber nucleus position from `core.fragment_mol` (edge atom index from meta). |
| `core_virtual_dipole_matrix_radial(core, virtual_orbitals, grid, *, origin, gauge, r_max, n_rad, n_ang, gs_pass)` | Fine on-site `⟨core\|r−R₀\|virt⟩` or velocity gauge via radial quadrature + trilinear PW virtual sampling. Gram–Schmidt core against virtuals. |
| `bind_ae_core_dipole(kernel, core, *, method, gauge, r_max, n_rad, n_ang)` | Attach AE dipole evaluator to PW kernel (`kernel._xas_ae_core`) for intensity calculation instead of coarse FFT mesh dipoles. Methods: `gto`, `radial`, `velocity`. |
| `core_virtual_dipole_matrix_grid(core_orbitals, virtual_orbitals, grid, *, origin, use_gpu)` | Legacy length-gauge dipole on the FFT charge grid (coarse; origin-sensitive). |
| `core_virtual_dipole_matrix_gto(core_mo_coeff, virt_mo_coeff, dip_ao)` | GTO dipole matrix `⟨core\|r\|virt⟩` with precomputed `int1e_r` integrals. |
| `project_grid_orbitals_to_fragment_ao(core, virtual_orbitals, grid)` | L²-project PW virtual grid orbitals onto `core.fragment_mol` AO basis (for GTO-projected dipoles). |
| `_orthogonalize_core_ao_against_virt_ao(C_c, C_v, S)` | Gram–Schmidt core AO coefficients against virtual AOs. |
| `_fragment_dipole_integrals(mol, origin_bohr, *, gauge)` | PySCF `int1e_r` or `int1e_ipovlp` dipole integrals in fragment AO basis. |
| `core_virtual_dipole_matrix_gto_projected(core, virtual_orbitals, grid, *, origin, gauge)` | **Default production dipole:** project PW virtuals to fragment AO, then PySCF dipole integrals with AE core GTOs. |
| `oscillator_strengths_at_origin(kernel, results, origin)` | Recompute stick `f` values at a shifted dipole origin (diagnostic). |
| `check_dipole_origin_gauge(kernel, results, *, origin, shift_bohr, rtol, strict)` | Warn or raise if `∑f` / `f_max` change too much when the dipole origin is shifted (FFT gauge check). |
| `virtual_dipole_scores(dipole_matrix, *, reduction="max")` | Per-virtual score from `(3, n_core, n_virt)` dipole tensor. |

### Virtual pruning (before CVS inject)

| Function | Description |
|----------|-------------|
| `_select_virtuals_by_score(eigs, orbitals, scores, *, min_score, n_keep, grid_mode, label, verbose)` | Shared ranking/filtering by score; returns kept eigenvalues, orbitals, indices. |
| `select_virtuals_by_core_dipole(...)` | Rank virtuals by `\|⟨core\|r−R₀\|virt⟩\|` (oscillator-strength weight). `core_dipole` selects metric: `gto` (default), `radial`, `velocity`, or legacy `fft`. |
| `select_virtuals_by_core_overlap(...)` | Rank virtuals by `\|⟨core\|virt⟩\|` (drops weakly coupled continuum in PW cells). |
| `_commit_pruned_grid_virtuals(kernel, e_keep, psi_keep)` | Write pruned PW virtuals back to kernel. |
| `_commit_pruned_gto_virtuals(kernel, e_keep, C_keep)` | Write pruned GTO virtual MO coefficients back to kernel. |
| `_core_dipole_ao(kernel, core_orbitals)` | Build `int1e_r` AO integrals centered on core charge centroid (GTO path). |
| `prune_kernel_virtuals_by_core_dipole(kernel, core_orbitals, ...)` | Kernel wrapper: prune then commit; supports PW (`provides_grid`) and GTO (`_C_v`) backends. |
| `prune_kernel_virtuals_by_core_overlap(kernel, core_orbitals, ...)` | Overlap analogue of the above. |

### Injection and CVS-TDA

| Function | Description |
|----------|-------------|
| `inject_core_orbitals(kernel, core_energies, core_orbitals, *, disable_uspp, orthogonalize_to_virt, use_gpu)` | Replace occupied active space with imported cores. PW: list of `DirectField`; GTO: `(nao, n_core)` or column list. Orthogonalizes cores against virtuals on PW grid. Disables USPP augmentation by default. |
| `_orthogonalize_occ_to_virt(psi_occ, psi_unocc, grid, *, use_gpu, n_pass)` | Gram–Schmidt core grid orbitals against virtuals (two passes default). |
| `run_cvs_tda(kernel, options, *, n_states=None)` | Solve CVS-TDA via `run_casida` with core-only occupied space. |
| `run_cvs_gto_from_mf(mf, *, edge, shell, edge_atom_indices, n_unocc, n_states, soc, use_gpu, use_mpi_response, use_df, verbose)` | **Workflow 1 driver:** `core_from_mf` → `extract_gto_kernel` → `inject_core_orbitals` → `run_cvs_tda`. Returns `(results, core, kernel)`. |
| `build_pw_kernel_from_qepy(driver, *, n_virt, use_gpu, xc, casida_uspp, n_placeholder_occ, n_total_occ, unocc_window_ev, n_virt_active, virt_energy_shift_ha, pack_band_indices, verbose)` | Build `PlaneWaveKernel` with real QE virtual bands. Wrapper around `extract_pw_kernel`. Returns `(kernel, grid)`. |

---

## `reconstruct.py`

AE core reconstruction in a **frozen** QE embedding potential `V_env`, then optional full
PW CVS pipeline via `run_reconstruct_cvs_from_driver`.

### Module constants

| Name | Value | Role |
|------|-------|------|
| `OH_COVALENT_ANG` | 1.20 | O–H covalent cutoff (Å) for water-fragment peeling |
| `_SHELL_NELEC` | dict | Aufbau shell electron counts for PP / AE bookkeeping |

### Internal helpers

| Function | Description |
|----------|-------------|
| `_phase(label, verbose)` | Context manager: print wall-clock timing per reconstruct phase when `verbose=True`. |
| `frozen_shells_from_pp(atom_symbol, z_valence, z_atomic=None)` | Shells replaced by the pseudopotential: `n_frozen = Z − z_valence`. |
| `_nelec_from_shells(shells)` | Total electrons from shell labels. |
| `_shells_for_nelec(n_elec)` | Fill shells 1s→… until `n_elec` electrons. |
| `_resolve_ae_config(sym, z, *, ae_electrons, shells, n_elec, z_valence, oxidation_state, oxidation_states)` | Map `ae_electrons` mode (`atomic` vs `frozen`) to `(mode, ox, shells, n_elec, charge)`. |
| `_resolve_mol_spin(n_elec, spin=None)` | Consistent PySCF `mol.spin = 2S`. |
| `_potential_ao_matrix(mol, v_env, grid, *, use_gpu, comm, grid_stride, blksize)` | Project `V_env` grid potential onto fragment AO basis (MPI block distribution). |
| `_rks_in_frozen_venv(mol, v_env, grid, *, xc, max_cycle, use_gpu, comm, verbose, grid_stride)` | Run RKS with `H += V_env` added in AO representation; optional `gpu4pyscf`. Returns `(mf, used_gpu_scf)`. |
| `_embed_kwargs(...)` | Pack embedding keyword arguments for `build_ae_embedding_potential`. |
| `_core_charge_centroid(kernel, ref_index=0)` | PW grid charge centroid of injected core orbital (dipole origin for legacy `fft` mode). |

### Active-site helpers (periodic / ice XI)

| Function | Description |
|----------|-------------|
| `bonded_hydrogens(atoms, edge_atom, *, max_oh_ang)` | H indices covalently bound to O (`edge_atom`), PBC minimum-image. |
| `resolve_active_atom_indices(atoms, edge_atom, *, active_atom_indices, oh_fragment, max_oh_ang)` | Atoms peeled from `V_env` and used in AE SCF. `oh_fragment=True` adds two H to O. |
| `fragment_positions_angstrom(atoms, atom_indices)` | Unwrap fragment coordinates about first atom (PBC) for multi-atom AE SCF. |

### AE SCF in `V_env`

| Function | Description |
|----------|-------------|
| `scf_ae_core_semicore(v_env, grid, atom_symbol, position_angstrom, *, basis, xc, shells, z_valence, n_elec, ae_electrons, oxidation_state, oxidation_states, spin, max_cycle, use_gpu, comm, verbose, grid_stride)` | **One-atom** AE RKS/UKS with frozen embedding. Default `ae_electrons="atomic"` uses formal oxidation (O²⁻, Ti⁴⁺, …). |
| `scf_ae_fragment(v_env, grid, atoms, atom_indices, *, basis, xc, charge, spin, max_cycle, use_gpu, comm, verbose, grid_stride)` | **Multi-atom** neutral fragment SCF (e.g. H₂O in ice XI). Charge 0; does not apply ionic `ae_electrons` oxidation. |
| `lowest_gto_virtual_ha(mf)` | Lowest unoccupied KS eigenvalue (Ha) from converged `mf`. |
| `virtual_shift_from_gto_mf(mf, driver)` | Rigid PW virtual shift (Ha) aligning lowest GTO virtual to PW CBM/LUMO reference. |
| `apply_core_gauge_shift(core, shift_ha)` | Add rigid shift to all `core.energies`. |
| `align_core_to_reference(core, eps_ref_ha, *, index=0)` | Shift cores so `core.energies[index]` matches reference; records shift in `core.meta`. |

### High-level drivers

| Function | Description |
|----------|-------------|
| `reconstruct_core_from_driver(driver, edge_atom, *, edge, shell, shells, basis, xc, upf_path, verbose, embed_mode, gauge_align, vhxc_scale, r_damp, vloc_source, ae_electrons, oxidation_state, oxidation_states, spin, zero_embedding, core_gauge_shift_ha, eps_core_reference_ha, core_reference_index, use_gpu, comm, grid_stride, active_atom_indices, oh_fragment, max_oh_ang, soc)` | Full embedding pipeline: `build_ae_embedding_potential` → `scf_ae_fragment` or `scf_ae_core_semicore` → `core_from_mf` → optional reference alignment. Returns `(core, mf, v_env, grid)`. |
| `run_reconstruct_cvs_from_driver(driver, edge_atom, *, ...)` | **Workflow 2 end-to-end:** `reconstruct_core_from_driver` → `prepare_pw_cvs_host` (PW virtual select/prune) → `build_gto_kernel_from_projected_pw` → `run_cvs_tda` on GTO `K`. Returns `(results, core, kernel, mf)`. Alias: `run_xas_reconstruct`. |

#### Key `run_reconstruct_cvs_from_driver` parameters

| Parameter | Role |
|-----------|------|
| `n_virt` | Number of QE conduction bands loaded into PW kernel |
| `n_virt_active` | Energy-stride subsample of virtual pool (optional) |
| `virt_window_ev` | Energy window above CBM for band selection |
| `virt_energy_shift_ha` | Rigid PW virtual energy shift for GTO alignment |
| `virt_select` | `"overlap"` or `"dipole"` pruning metric |
| `virt_n_keep` | How many virtuals to keep after pruning |
| `core_dipole` | Intensity evaluator: `gto` (default), `radial`, `velocity`, or legacy `fft` |
| `oh_fragment` | Peel O+2H neutral water fragment instead of bare O²⁻ ion |
| `eps_core_reference_ha` | Pin reconstructed core to gas-phase AE reference |
| `zero_embedding` | Diagnostic: `V_env = 0` (isolated ion) |

---

## `spectrum.py`

Post-processing of `CasidaResults` from CVS-TDA: energies, oscillator strengths, plots.

| Function / constant | Description |
|---------------------|-------------|
| `HA_TO_EV` | Hartree to eV conversion factor (CODATA). |
| `omega_ev(results)` | Excitation energies `ω` in eV from `results.omega` (Ha). |
| `stick_spectrum(results, *, max_states, normalize)` | Return `(omega_ev, f)` arrays; optional truncation and `∑f=1` normalization. |
| `plot_sticks(results, *, ax, label, color, max_states, broaden_sigma_ev, energy_shift_ev, normalize)` | Matplotlib stick plot; optional Gaussian broadening. Returns `(fig, ax)`. |
| `first_peak_shift_ev(reference, results, *, index=0)` | Rigid shift (eV) aligning stick `index` of `results` to `reference`. |
| `_abs_dipole_au(results)` | Length-gauge `\|μ\|` (a.u.) from `d_mode` or inverted from `f = (2/3)ω\|μ\|²`. |
| `oscillator_table(results, *, n_print=12)` | Formatted table of `ω`, `f`, and `\|μ\|` for leading sticks. |
| `brightest_peak_shift_ev(reference, results)` | Rigid shift aligning brightest stick between two spectra. |
| `summarize_xas(results, core=None, *, n_print=8)` | One-line summary + oscillator table for notebooks. |

---

## Choosing atom vs fragment

Fragment helpers in `cvs.py` are **kept** and are the right tool when you want
geometry-cut AE cores (clusters, molecules, or manual sub-cells). The QE
reconstruct path in `reconstruct.py` chooses between a **single embedded ion**
and a **multi-atom fragment** automatically.

### QE embedding (`run_xas_reconstruct`)

| Goal | How | AE SCF used |
|------|-----|-------------|
| Formal ion at absorber (MgO O²⁻, Ti⁴⁺, …) | Default — only `edge_atom` | `scf_ae_core_semicore` + `ae_electrons="atomic"` |
| Neutral water peel (ice XI) | `oh_fragment=True` | `scf_ae_fragment` on O + 2 covalent H |
| Custom multi-atom peel | `active_atom_indices=[i, j, …]` | `scf_ae_fragment` (charge 0) when len > 1 |
| Single atom, explicit index | `active_atom_indices=[edge_atom]` | Same as default ion path |

`V_env` is built for every atom in the active list; core MOs are taken from
`edge_atom` within that fragment.

### Geometry-cut cores (no QE `V_env`)

| Goal | How |
|------|-----|
| Auto neutral first shell (TiO₂ cluster, …) | `frag = select_xas_fragment(atoms, edge_atom, mode="neutral_first_shell")` |
| Ball cut | `select_xas_fragment(..., mode="radius", radius_ang=3.0)` |
| User indices / charge / spin | `FragmentSpec.user([...], charge=0, spin=0)` or pass `atom_indices=` |
| Run fragment SCF + pick cores | `core = extract_fragment_core(mol, fragment=frag, edge="K")` |
| Inject into host | `inject_core_orbitals(kernel, core.energies, core_fields_or_coeffs)` |

Use **reconstruct** when the solid is periodic and embedding from QE matters.
Use **fragment helpers** when you have a finite cluster/molecule or want to
compare fragment cuts without rebuilding `V_env`.

---

## Recent cleanup notes

- **Production reconstruct path** always uses **projected-GTO Casida `K`**
  (`build_gto_kernel_from_projected_pw`). PW grid ALDA `K` is no longer an
  option. Dipole virtual pruning still uses the charge grid; intensities use
  GTO-projected dipoles on the same projection Mole.
- **Fragment helpers retained** — `select_xas_fragment`, `extract_fragment_core`,
  `FragmentSpec`, `active_atom_indices`, `oh_fragment`, etc.
- **`select_*` vs `prune_*`**: rank/filter vs commit-to-kernel; both kept for
  adapter code.

---

## Typical usage

```python
# Workflow 1 — molecule in PySCF
from casidapy.xas import run_xas_gto, summarize_xas, plot_sticks
res, core, kernel = run_xas_gto(mf, edge="K", edge_atom_indices=[0])
print(summarize_xas(res, core))

# Workflow 2 — periodic solid (single ion peel, default)
from casidapy.xas import run_xas_reconstruct
res, core, kernel, mf = run_xas_reconstruct(
    driver, edge_atom=0, edge="K",
    n_virt=400, virt_n_keep=200, n_states=150,
)

# Workflow 2b — ice XI water fragment in V_env
res, core, kernel, mf = run_xas_reconstruct(
    driver, edge_atom=o_idx, oh_fragment=True, ...
)

# Workflow 3 — geometry-cut fragment (cluster / molecule)
from casidapy.xas import select_xas_fragment, extract_fragment_core, inject_core_orbitals
frag = select_xas_fragment(atoms, edge_atom=ti_idx, mode="neutral_first_shell")
core = extract_fragment_core(mol, fragment=frag, edge="K")
```

For embedding physics and production batch drivers, see also
`CasidaPy_tests/X-ray/PERIODIC_XAS_PIPELINE.md` and `casidapy/casidapy/embed/`.
