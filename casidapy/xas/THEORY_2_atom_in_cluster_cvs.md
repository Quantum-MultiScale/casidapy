# Theory Part 2 — AE atom-in-cluster GTO core + PW virtuals → CVS-TDA XAS

> **Scope.** This document explains *every* step that turns the frozen embedding
> potential `V_env` (built in `THEORY_1_embedding.md`) into an X-ray absorption
> spectrum. It covers: running the all-electron (AE) Gaussian SCF inside `V_env`,
> selecting and gauge-aligning the core orbitals, loading the periodic plane-wave
> (PW) conduction bands, **projecting those PW virtuals into the Gaussian (GTO)
> basis**, orthogonalizing, building the Casida coupling matrix, solving CVS-TDA,
> and computing transition dipoles / oscillator strengths.
>
> Code: `casidapy/xas/{reconstruct.py, cvs.py, diagnostics.py, dipole.py,
> ghost_diffuse.py, xas_utils.py, spectrum.py}`.
> Production entry point: `run_xas_reconstruct` = `run_reconstruct_cvs_from_driver`
> → `run_reconstruct_cvs_gto_projected_from_driver`.

---

## 0. The hybrid idea in one paragraph

Under the **core–valence separation (CVS)** approximation, an XAS transition is a
one-particle excitation from a single deep **core** orbital to an **unoccupied**
final state. We therefore build a Casida (linear-response TDDFT) problem whose

- **occupied active space** = *only the imported AE core orbital(s)* — computed
  all-electron in Gaussians so the 1s actually exists; and
- **virtual active space** = *the crystal's conduction bands* — taken from the
  periodic PW calculation so the final states have the right solid-state
  character;
- **coupling matrix `K`** = evaluated in the **Gaussian AO basis**, after
  projecting the PW virtuals into that basis.

This "cores from GTO, virtuals from PW, coupling in GTO" split is the entire
trick. The rest is making the two representations talk to each other cleanly
(projection, normalization, orthogonality, gauge).

The top-level orchestration is:

```
run_reconstruct_cvs_gto_projected_from_driver          [diagnostics.py]
  ├─ prepare_pw_cvs_host                                [reconstruct.py]   §1–§4
  │    ├─ reconstruct_core_from_driver                  → AE cores        §1–§3
  │    │    ├─ build_ae_embedding_potential  (THEORY_1)  → V_env
  │    │    ├─ scf_ae_core_semicore / scf_ae_fragment    → mf  (AE SCF)
  │    │    └─ core_from_mf                               → CoreOrbitals
  │    └─ build_pw_kernel_from_qepy                       → PW virtuals    §4
  ├─ build_gto_kernel_from_projected_pw                  → GTOKernel + K   §5–§7
  └─ run_cvs_tda                                          → CasidaResults  §8
```

---

## 1. Running the all-electron SCF inside `V_env`

### The embedded Hamiltonian (`_rks_in_frozen_venv`)

The AE fragment is an ordinary PySCF `RKS`/`UKS` object, except the frozen
embedding potential is **added to the core Hamiltonian** as a static external
one-body operator:

```python
h0   = scf.hf.get_hcore(mol)          # T̂ + bare −Z/r of the fragment atoms
v_ao = _potential_ao_matrix(mol, v_env, grid, …)
mf.get_hcore = lambda *a, **k: h0 + v_ao
```

- `h0` uses the **bare** nuclear attraction of the fragment atoms — this is a true
  all-electron calculation, no PP on the absorber. Its 1s is physical.
- `v_ao` is `V_env` represented in the fragment AO basis.

### Projecting `V_env` onto AOs (`_potential_ao_matrix`)

`V_env` lives on the QE real-space charge grid. We need its matrix elements

```
V_μν = ∫ φ_μ(r) V_env(r) φ_ν(r) dr  ≈  Σ_grid  φ_μ(r_i) V_env(r_i) φ_ν(r_i) dV
```

Implementation: evaluate the Gaussian AOs on the grid points with
`pyscf.dft.numint.eval_ao` in blocks (`blksize`), form `(aoᵀ·v)·ao` per block,
sum. The charge grid is Fortran-ordered (QE/DFTpy); coordinates and `V_env`
must be `ravel(order="F")` together — C-order pairing is wrong whenever
`nx, ny, nz` are not equal. Details:
- **MPI**: grid blocks are split across ranks (`block_slices(…, comm)`) and the
  AO matrix is `Allreduce`d.
- **GPU**: the block GEMMs run in CuPy when `use_gpu=True`.
- **`grid_stride`**: optional subsampling of grid points (with a `stride³` volume
  rescale) — only safe when `V_env` is smooth on the stride scale (large vacuum
  cells). Default `stride=1` = full grid.

### Which AE SCF is run

`reconstruct_core_from_driver` picks the backend from the active-atom list
(`resolve_active_atom_indices`):

| Active list | Routine | Electron model |
|-------------|---------|----------------|
| one atom (default) | `scf_ae_core_semicore` | **Formal ion** — see `ae_electrons` below |
| several atoms (`active_atom_indices=[…]` or `oh_fragment=True`) | `scf_ae_fragment` | **neutral** multi-atom peel (charge 0), positions unwrapped by minimum-image (`fragment_positions_angstrom`) |

### `ae_electrons` — how many electrons the AE atom carries (`_resolve_ae_config`)

- **`"atomic"` (default):** the atom is a **formal ion** matching its oxidation
  state (O→O²⁻, Ti→Ti⁴⁺, …), resolved via `resolve_oxidation_state`. Charge =
  oxidation state; electron count and occupied shells follow. This mirrors the
  ionic picture of the solid and gives a closed-shell reference.
- **`"frozen"` (legacy):** only the electrons the PP removed
  (`frozen_shells_from_pp`: `n_frozen = Z − z_valence`, filled 1s→…). Charge =
  `Z − n_elec`.

Shell/electron bookkeeping uses the Aufbau table `_SHELL_NELEC`.
`_resolve_mol_spin` sets PySCF `mol.spin = 2S = Nα − Nβ`.

### Occupation smearing (`_apply_ae_smearing`)

The AE atom sits in the crystal's field, near a Fermi level with possibly (near-)
degenerate frontier states. To get a **well-defined chemical potential** and avoid
SCF sloshing, Gaussian occupation smearing is applied by default
(`σ = 0.005 Ha ≈ QE degauss 0.01 Ry`, `ae_smearing_*` in `chemical_data.json`).
Pass `smearing_sigma_ha=0` to disable. This matters for the Fermi gauge (§3).

---

## 2. Selecting the core orbitals (`core_from_mf` → `_select_core_mos`)

From the converged AE mean-field we must pick the *edge-localized occupied MOs*
(e.g. the O 1s), not just the lowest MO by energy.

1. **Edge/shell resolution** (`_resolve_shell`): edge label → `(n, ℓ)`
   (K→1s, L→2p, M→3d…), or an explicit shell alias like `"2p"`.
2. **Which atoms contribute cores** (`edge_atom_indices`): default = heavy atoms
   (Z≥3) for a K-edge; all atoms for L/M.
3. **AO shell mask** (`_ao_shell_mask`, `_ao_mask_for_atoms`): boolean mask over
   AOs whose label matches the target `nℓ` on the edge atoms (e.g. all `1s` AOs on
   oxygen).
4. **Localization ranking** (`_select_core_mos`): for each occupied MO compute the
   weight it places on the masked shell AOs, `Σ_{μ∈mask} c_{μi}²`; take the MOs
   with the largest weight, then order them by energy and keep the expected count
   (`(2ℓ+1)·n_edge`, or `n_core_per_atom·n_edge`). This robustly finds "the 1s"
   even if it is not literally MO #0.

The result is a **`CoreOrbitals`** dataclass:
`energies (Ha)`, `mo_coeff (nao×n_core, fragment AO basis)`, `fragment_mol`,
`atom_indices`, `edge`, `shell`, `meta`.

### GPU / cupy host safety

`core_from_mf` copies `mo_energy/coeff/occ` off the GPU (`_host`) and, for UKS,
uses the α channel. So the selection logic is always plain NumPy.

### Optional spin–orbit coupling (`apply_orbital_soc`, L/M edges)

For L/M edges the `2p`/`3d` core is spin-split (e.g. L₂/L₃). `apply_orbital_soc`
diagonalizes the **Breit–Pauli one-electron SOC** operator in the *core spatial-MO
spin-orbital subspace*:
- Build `h_so` AO integrals (`casidapy.utils.soc.soc_ao_integrals`, `(3,nao,nao)`
  Hermitian).
- Assemble the `2n_sp × 2n_sp` spin-orbital Hamiltonian: diagonal orbital
  energies + `S_z` (`±½ h_z`) + `S_±` (`(h_x ∓ i h_y)/2`) blocks.
- Diagonalize → real j-split energies + **complex** MO coefficients
  `(nao, 2·n_spatial)`.
This is a *subspace* SOC (only the core shell is rotated), sufficient for edge
splitting. s-shells (K-edge) have no first-order SOC and are skipped.

---

## 3. Gauge-aligning the core energies (the "gauge stuff", part 2)

Recall from `THEORY_1 §6` that the embedding fixed the zero of `V_env`. Two
further, code-side alignments finish the job on the *core eigenvalues* — because
the absolute edge energy is the deep `ε_core`, and it is only defined relative to
the same zero as the virtuals.

### (a) Fermi gauge completion (`align_core_to_fragment_fermi_zero`)

When `gauge_align='fermi'` (crystal default):
- The AE atom's **residual chemical potential** `μ_atom` is measured
  (`ae_fragment_chemical_potential_ha`): with smearing it is the optimized
  smearing μ (`pyscf.scf.addons._smearing_optimize`); without smearing it is the
  mid-gap `(ε_HOMO+ε_LUMO)/2`.
- We shift **all** core energies by `−μ_atom` (and mirror the shift onto
  `mf.mo_energy`). Rationale: after `V_env ← V_env − E_F`, the host Fermi level is
  the zero, but the finite AE atom does not reproduce the host mean field exactly,
  leaving a *constant* well error `μ_atom`. Subtracting it is **exact** for
  eigenvalues (a uniform potential shift). This levels the atom to the PW Fermi
  frame without touching orbitals.
- The **same shift is later applied to the PW virtual energies** (§4) so the
  excitation energies `ω = ε_virt − ε_core` are gauge-invariant.

`apply_core_gauge_shift(core, s)` is the primitive (adds `s` to every
`core.energies`, records cumulative shift in `meta`).

### (b) Reference alignment (`align_core_to_reference`)

Optionally pin the reconstructed core to a known **gas-phase AE reference**
(`eps_core_reference_ha`): shift so `core.energies[index] == eps_ref`. This
removes the residual embedding shift entirely and reproduces a full-AE molecular
edge for benchmarking. Mutually informative with the Fermi gauge (you use one or
the other as the absolute anchor).

The precedence in `reconstruct_core_from_driver`: Fermi-gauge shift → optional
reference shift → optional manual `core_gauge_shift_ha`.

---

## 4. Loading and shifting the PW virtual bands

### The PW kernel (`build_pw_kernel_from_qepy` → `extract_pw_kernel`)

A `PlaneWaveKernel` is built holding **real QE conduction bands** as the virtual
active space, with a small placeholder occupied window (thrown away when cores are
injected). Knobs:
- `n_virt` — number of conduction bands loaded.
- `unocc_window_ev` / `n_virt_active` (`unocc_subsample`) — energy-window or
  energy-strided subsample of a large band pool.
- `virt_energy_shift_ha` — rigid shift applied to the virtual eigenvalues.
- `pack_band_indices` — explicit band selection.

The kernel exposes the virtuals both as grid fields (`_psi_unocc`) and eigenvalues
(`_unocc_e`), and carries the DFTpy `grid`.

### Choosing the virtual shift (`resolve_pw_virt_shift_ha`)

Returns `(shift_ha, mode)`:
- **`fermi`** (auto when `gauge_align='fermi'` and no core reference):
  `ε_virt ← ε_virt − E_F`. This is the **virtual half** of the two-step Fermi
  gauge from §3 — both cores and virtuals are referenced to the host Fermi level,
  so `ω` is invariant.
- **`lumo_peel`** (`align_lumo_peel=True`): align QE `ε_virt` to the AE fragment's
  LUMO (`virtual_shift_from_gto_mf`) — legacy bridge to full-AE molecular runs.
- **`manual`**: use `virt_energy_shift_ha` directly.
- **`none`**: no shift.

`resolve_virt_energy_shift_ha` enforces that at most one of
`align_lumo_mf / align_cbm / align_fermi` is active.

### Optional dipole prune on the grid (`prune_kernel_virtuals_by_core_dipole`)

A PW box has a huge continuum of conduction bands, most carrying **zero** XAS
intensity (they have no dipole overlap with the tight core). Before the expensive
projection/coupling we can keep only the brightest `virt_n_keep` virtuals, ranked
by `|⟨core|r−R₀|virt⟩|` (the oscillator-strength weight), scored via
GTO-projected dipoles on the charge grid (§9). This is a *selection*, not a
physics approximation of the kept states.

---

## 5. Projecting PW virtuals into the Gaussian basis (the heart of it)

Now the two representations meet. Each PW virtual `ψ_a(r)` (a grid field) is
expanded in the fragment/projection GTO AO basis.

### The projection integral (`project_grid_orbitals_to_fragment_ao`)

```
V[μ,a] = ∫ φ_μ(r) ψ_a(r) dr  ≈  Σ_grid φ_μ(r_i) ψ_a(r_i) dV
```

(blocked `eval_ao`, MPI/GPU as before.) This gives the *overlap of the PW virtual
with each AO*, not yet the MO coefficients.

### From projections to AO coefficients (`grid_projection_to_ao_coeffs`)

Crucial subtlety. If `ψ = Σ_ν c_ν φ_ν`, then `V[μ] = ∫φ_μ ψ = Σ_ν S_{μν} c_ν`,
i.e. `V = S·c`. Therefore the **correct AO coefficients are**

```
c = S⁻¹ V           (S = mol.intor("int1e_ovlp"))
```

**Using the raw `V` as if it were `c` is wrong** whenever the basis is
non-orthogonal (always, and especially for diffuse/ghost bases) — it inflates
`|μ|` in the dipole. So we always solve `S c = V`.

### Completeness / spillage

The projection is exact only if the AO basis spans `ψ_a`. The fraction that lives
*inside* the AO span is the **completeness**

```
c_a = ⟨ψ_a|P_AO|ψ_a⟩ = V_aᵀ S⁻¹ V_a / ‖ψ_a‖²        (ao_projection_completeness)
spillage S_a = 1 − c_a                               (virtual_ao_spillage)
```

A crystal conduction state is delocalized; a single-atom GTO basis captures only a
patch of it near the absorber. High spillage means the AO basis is too small to
represent the final states — hence the **ghost diffuse peel** (§6) and the
spillage **audit/gate** (`audit_virtual_ao_spillage`, `spillage_gate`,
`summarize_ao_spillage`). For XAS this is acceptable because the transition
*dipole* only samples `ψ_a` where the core lives (near the nucleus), which the AO
basis does represent well — but the audit tells you when to distrust it.

### Virtual normalization (`normalize_ao_coeffs_s_metric`, `build_gto_kernel_from_projected_pw`)

Projected virtuals `c = S⁻¹V` have S-norm `‖c‖_S = √completeness ≤ 1`. Leaving them
under-normalized would systematically *suppress* dipoles; ghost peels can also give
erratic column norms. So each virtual column is rescaled to `‖c‖_S = 1`
**without mixing bands** (`normalize_virtuals='s_metric'`, the default). A pure
`s_metric` rescale keeps the 1-to-1 attachment of PW band energies to columns
(Löwdin would mix bands and require rotating the eigenvalues — available as
`normalize_virtuals='lowdin'`, used with care). *In the dipole path
(`core_virtual_dipole_matrix_gto_projected`) the rescale is cap-only
(`max_scale=1.0`): shrink `‖c‖_S>1`, never amplify high-spillage columns.*

---

## 6. Ghost diffuse peel — improving AO completeness (`ghost_diffuse.py`)

**Problem.** A minimal absorber-only GTO basis has high spillage: it cannot follow
the conduction state onto neighbouring sites. **Solution.** Keep the AE SCF peel
compact (absorber only — cheap, well-conditioned), but build a *separate, larger
**projection** basis* by placing **diffuse-only "ghost" centres** (PySCF
zero-charge ghost atoms) on neighbour sites. The AE core still comes from the
small SCF; the PW virtuals are projected onto the ghost-expanded span.

Controlled by **`GhostDiffusePeelOptions`**:
- `scf_peel` — atoms in the AE SCF (default `atom` = absorber only).
- `ghost_peel` — neighbour shell that receives ghost centres (default
  `first_shell`).
- `ghost_placement` — `atoms` (on neighbour nuclei), `bonds` (at covalent bond
  midpoints, `peel_bond_midpoints_ang`), or `atoms+bonds`.
- `ghost_on_absorber` — also add a diffuse ladder on the absorber itself.
- Diffuse exponents: geometric **ζ-ladders** (`_zeta_ladder`, `diffuse_ladder_block`,
  `ghost_center_basis`) per angular momentum up to `ghost_lmax`, controlled by
  `zeta_min_{s,p,d}`, `zeta_ratio`, `ladder_depth`.
- `use_in_dipole` — whether the dipole/intensities also use the ghost-expanded
  basis (default True) or stay on the absorber SCF fragment only.

Construction: `build_ghost_diffuse_projection_mol` places the absorber (full SCF
basis, absorber-first) then the ghost centres, and attaches the resulting `Mole`
to `core.projection_mol` (`attach_ghost_projection_mol`). Because the absorber AOs
come first, the fragment core MO coefficients embed by **zero-padding** into the
larger basis (`core_mo_coeff_on_projection_mol`,
`embed_mo_coeff_on_projection_mol`) — the core has zero amplitude on ghost AOs.

`projection_mol_for_core` (spillage/coupling) and `projection_mol_for_dipole`
(intensities) pick the right basis depending on `use_in_dipole`.

---

## 7. Orthogonality and the Casida coupling matrix `K`

### Projecting cores out of the virtual span (`_orthogonalize_core_ao_against_virt_ao`)

The imported core must be orthogonal to the virtual manifold (an excitation goes
core→*unoccupied*). Because the projected virtuals `C_v = S⁻¹V` are **not**
S-orthonormal, the naive Gram–Schmidt `C ← C − V(VᵀSC)` is wrong. The correct
projector onto `span(V)` in the S-metric is

```
P = V (Vᵀ S V)⁺ (Vᵀ S)          (pseudoinverse for near-dependent virtual blocks)
C_core ← (1 − P) C_core         (n_pass iterations), then S-normalize
```

`lstsq` is always used because projected PW virtuals are frequently near-linearly-
dependent in the AO basis. (The grid-side analogue for the PW inject path is
`_orthogonalize_occ_to_virt` in `cvs.py`, a two-pass grid Gram–Schmidt.)

### Building the GTO kernel (`build_gto_kernel_from_projected_pw`)

Assembles a **`GTOKernel`** whose:
- occupied space = projected AE cores `C_c` (`set_core_active_space`), energies
  `core.energies`;
- virtual space = projected, normalized PW virtuals `C_v`
  (`set_virtual_active_space`), energies = **QE band eigenvalues** (+ shift), *not*
  fragment SCF virtual levels;
- mean-field for the response = `mf_on_projection_mol` — the AE `mf` re-expressed
  on the (possibly ghost-expanded) projection `Mole`, with fragment MOs
  zero-embedded, so `gen_response` sees a consistent AO dimension.

The Casida coupling

```
K_{ia,jb} = ⟨ia| (Coulomb + f_xc) |jb⟩
```

is then evaluated by `mf.gen_response` **in the projection AO basis** — i.e. the
electron–hole coupling uses proper Gaussian two-electron integrals + the XC
kernel, not a coarse grid ALDA. This is why the coupling is done in GTO even
though the virtuals come from PW.

A rich `_xas_spillage_audit` and `virt_norm` metadata are attached; an optional
second AO-metric dipole prune can run here too.

---

## 8. Solving CVS-TDA (`run_cvs_tda`)

The excitation problem in the **Tamm–Dancoff approximation (TDA)** is the
Hermitian eigenproblem

```
A Z = ω Z ,    A_{ia,jb} = δ_{ij}δ_{ab} (ε_a − ε_i) + K_{ia,jb}
```

with the occupied index `i` restricted to the core(s) — that restriction *is* CVS
(it decouples the deep core excitations from valence excitations, which are
energetically far away and would otherwise dominate the low eigenvalues). TDA
drops the `B` (de-excitation) block; for deep-core XAS the `A`-only approximation
is standard and well justified.

`run_cvs_tda` builds `CasidaOptions` (`tda=True`, `matrix_free=True`, Davidson
solver) and calls `run_casida`. It requires `kernel.n_occ ≥ 1` (cores injected)
and clamps `n_states` to the number of transitions. Output: **`CasidaResults`**
with `omega` (Ha), oscillator strengths `f`, and eigenvectors `Z`.

`Δε = ε_a − ε_core` dominates the excitation energy (it is huge and negative core
→ positive virtual), while `K` provides the (comparatively small) electron–hole
attraction/relaxation shift. The `diagnostics.py` decomposition tools
(`compare_cvs_tda_decomposition`, `coupling_k_gate`) exist precisely to check that
agreement between two methods is *earned by matching `K`*, not merely inherited
from a shared diagonal `Δε`.

---

## 9. Transition dipoles, gauge origin, and oscillator strengths

### Length-gauge dipole (`core_virtual_dipole_matrix_gto_projected`)

The transition dipole between a core and a virtual is
`⟨core| r − R₀ |virt⟩`. In production it is computed in the projection GTO basis:
1. project PW virtuals → AO coeffs `C_v = S⁻¹V`, cap-normalized (§5);
2. orthogonalize the core against them (§7);
3. `dip_ao = mol.intor("int1e_r")` centred at origin `R₀`
   (`_fragment_dipole_integrals`, via `mol.with_common_orig`);
4. contract `μ_{cv} = C_cᵀ · dip_ao · C_v`
   (`core_virtual_dipole_matrix_gto`).

Doing the dipole in Gaussians (rather than on the coarse FFT mesh) is what makes
the intensity **near-absorber-accurate**: the tight 1s is sampled by analytic
Gaussian integrals, not by a mesh that badly under-resolves it.

### Origin choice (`absorber_origin_bohr`)

`R₀` = the **absorber nucleus** (the edge atom's coordinate from
`fragment_mol.atom_coords()`). For an *exactly* orthogonal core/virtual pair the
length-gauge dipole is origin-independent; on a finite basis a residual
`⟨core|virt⟩ ≠ 0` makes it origin-*dependent*. Centring on the nucleus (where the
core density lives) minimizes `|r − R₀|` there and thus the residual error. This
also sets `kernel.dipole_origin`.

### Oscillator strength

```
f_n = (2/3) ω_n Σ_α |d_{n,α}|²,      d_n = μᵀ Z_n
```

(`oscillator_strengths_at_origin`). Recovering `|μ|` from `f` uses
`|μ| = √(3f / 2ω)` (`spectrum._abs_dipole_au`).

### Gauge-origin sanity check (`check_dipole_origin_gauge`)

Because the length gauge is only origin-invariant for perfectly orthogonal
states, `dipole_origin_check=True` recomputes `∑f` and `f_max` from the solved `Z`
at `R₀` and at `R₀ + shift·x̂` (no re-solve). If they move by more than `rtol` it
warns (or raises with `strict=True`): a large sensitivity flags residual
core–virtual non-orthogonality / insufficient basis.

---

## 10. From `CasidaResults` to a spectrum (`spectrum.py`)

- `omega_ev` — `ω·HA_TO_EV`.
- `stick_spectrum` — `(ω_eV, f)`, optional truncation / `∑f=1` normalization.
- `plot_sticks` — stick plot + optional Gaussian broadening.
- `first_peak_shift_ev` / `brightest_peak_shift_ev` — rigid alignment of one
  spectrum onto a reference (compare GTO vs PW hosts, or shifted gauges).
- `oscillator_table`, `summarize_xas` — notebook summaries (edge, `ε_core`,
  `∑f`, leading `ω`, `f`, `|μ|`).

---

## 11. Complete approximation & assumption ledger

| # | Approximation | Where | Consequence / mitigation |
|---|---------------|-------|--------------------------|
| 1 | Frozen environment density `ρ_env` (not relaxed to the core hole/atom) | THEORY_1 §1,§3 | Environment polarization by the excitation ignored. Local, mostly-electrostatic effect for deep cores. |
| 2 | Non-additive **kinetic** (Pauli) energy neglected | THEORY_1 §1 | No `v_T^nad`; environment Pauli wall only via neighbour PPs + Hirshfeld `ρ_env→0` on core. |
| 3 | Subsystem-additive **XC** (`v_xc[ρ_env]` + `v_xc[ρ_frag]`) | THEORY_1 §1,§4 | Cross XC term dropped; Hartree is exact-additive. |
| 4 | AE fragment vs **pseudized** neighbours | THEORY_1 §1 | Absorber core is real; neighbour cores are PP. Consistent for a core-localized probe. |
| 5 | NLCC zeroed when evaluating `v_xc[ρ_env]` | THEORY_1 §4 | Avoids double-counting model core density. |
| 6 | **CVS**: occupied space = core only | §0,§8 | Decouples core from valence excitations (valid: huge energy separation). |
| 7 | **TDA**: drop `B` block | §8 | Standard for deep-core XAS. |
| 8 | Formal-oxidation ion (`ae_electrons="atomic"`) | §1 | Closed-shell ionic reference; override with `oxidation_state(s)` or `frozen`. |
| 9 | Occupation smearing on the AE atom (`σ≈0.005 Ha`) | §1 | Defines μ for the Fermi gauge; slight occupation softening. |
| 10 | Gauge zero of `V_env` fixed by convention (fermi/hirshfeld/vacuum/mean) | THEORY_1 §6, §3 | Sets absolute edge energy; the largest single lever. Two-step Fermi protocol keeps `ω` invariant. |
| 11 | PW virtuals **projected** into a finite GTO/ghost basis | §5,§6 | Spillage; audited by `audit_virtual_ao_spillage` + `spillage_gate`; mitigated by ghost diffuse peel. |
| 12 | Virtual columns S-normalized (no band mixing) | §5 | Consistent intensity scale; Löwdin variant available. |
| 13 | Length-gauge dipole centred on the absorber nucleus | §9 | Origin-independent only for orthogonal states; residual checked by `check_dipole_origin_gauge`. |
| 14 | Subspace (core-shell) SOC for L/M | §2 | First-order edge splitting; s-edges have none. |
| 15 | Optional grid **dipole prune** of PW virtuals | §4 | Selection of bright states only; discards zero-intensity continuum. |

---

## 12. Cross-reference: key classes & functions

**Embedding (THEORY_1):** `build_ae_embedding_potential`, `build_hirshfeld_embedding`,
`HirshfeldPartition`, `ionic_vloc_residual`, `vloc_atom_from_qepy`,
`hirshfeld_weights_and_partition`, `hartree_potential_ha`, `xc_potential_ha`,
`shift_v_env_gauge_{at,hirshfeld,min_density}`.

**AE SCF & cores:** `reconstruct_core_from_driver`, `_rks_in_frozen_venv`,
`_potential_ao_matrix`, `scf_ae_core_semicore`, `scf_ae_fragment`,
`_resolve_ae_config`, `core_from_mf`, `_select_core_mos`, `apply_orbital_soc`,
`CoreOrbitals`.

**Gauge (core side):** `align_core_to_fragment_fermi_zero`,
`ae_fragment_chemical_potential_ha`, `apply_core_gauge_shift`,
`align_core_to_reference`, `resolve_pw_virt_shift_ha`,
`resolve_virt_energy_shift_ha`.

**PW virtuals:** `build_pw_kernel_from_qepy`, `PlaneWaveKernel`,
`prune_kernel_virtuals_by_core_dipole`.

**Projection & completeness:** `project_grid_orbitals_to_fragment_ao`,
`grid_projection_to_ao_coeffs`, `normalize_ao_coeffs_s_metric`,
`ao_projection_completeness`, `virtual_ao_spillage`, `audit_virtual_ao_spillage`.

**Ghost peel:** `GhostDiffusePeelOptions`, `build_ghost_diffuse_projection_mol`,
`attach_ghost_projection_mol`, `core_mo_coeff_on_projection_mol`,
`projection_mol_for_{core,dipole}`, `ghost_center_basis`, `_zeta_ladder`.

**Coupling & solve:** `_orthogonalize_core_ao_against_virt_ao`,
`build_gto_kernel_from_projected_pw`, `mf_on_projection_mol`, `GTOKernel`,
`run_cvs_tda`, `CasidaResults`.

**Dipole & spectrum:** `core_virtual_dipole_matrix_gto_projected`,
`core_virtual_dipole_matrix_gto`, `_fragment_dipole_integrals`,
`absorber_origin_bohr`, `oscillator_strengths_at_origin`,
`check_dipole_origin_gauge`, `omega_ev`, `plot_sticks`, `summarize_xas`.

**Validation (diagnostics.py):** `spillage_gate`, `projected_k_gate`,
`compare_cvs_tda_decomposition`, `coupling_k_gate`,
`compare_cvs_excitation_energies`.
