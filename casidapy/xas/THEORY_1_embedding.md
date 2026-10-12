# Theory Part 1 — Building the frozen environment potential `V_env`

> **Scope.** This document explains *every* physical and numerical step that
> `casidapy.embed` performs to turn a converged periodic plane-wave (PW) DFT
> calculation into a **frozen embedding potential** `V_env(r)` on the QE charge
> grid. That potential is what lets us later run an *all-electron (AE) Gaussian
> (GTO)* SCF for a single absorber atom (or a small covalent fragment) *inside*
> the crystal — see companion doc `THEORY_2_atom_in_cluster_cvs.md`.
>
> Code: `casidapy/embed/potential.py` and `casidapy/embed/hirshfeld.py`.
> Entry point: `build_ae_embedding_potential` → (Hirshfeld default) →
> `build_hirshfeld_embedding`.

---

## 0. Why we need an embedding potential at all

X-ray absorption (XAS) probes a **deep core orbital** (e.g. O 1s, Ti 1s). Two
irreconcilable facts drive the whole design:

1. **Plane-wave DFT cannot represent the core.** QE uses pseudopotentials (PPs):
   the 1s is frozen into the PP and simply does not exist as a KS orbital. You
   cannot compute a core→conduction transition from PW bands alone.

2. **All-electron Gaussian DFT can represent the core trivially** (a `def2-tzvp`
   basis has real 1s functions), **but cannot cheaply represent the extended
   crystalline conduction manifold** — the XAS final states, which are delocalized
   Bloch-like states of the solid.

The resolution used here is a **subsystem split**:

- The **absorber atom** (or a tiny covalent fragment, e.g. O+2H in ice) is treated
  **all-electron in a GTO basis**, so its core exists.
- The **rest of the crystal** ("the environment") is kept **frozen** at the PW-DFT
  level and enters the atom's Hamiltonian only through a **static external
  potential** `V_env(r)`.

`V_env` is the object this document builds. Physically it must reproduce, at the
absorber site, the electrostatic + exchange–correlation field that the *rest of
the solid* exerts — so that the AE atom's core level sits at the correct absolute
energy (chemical shift) and its orbitals are polarized correctly.

---

## 1. The embedding model and its central approximations

The frozen-environment potential is

```
V_env(r) = V_ionic(r) + v_H[ρ_env](r) + v_xc[ρ_env](r)          (Hirshfeld mode)
```

and the AE fragment then solves, in PySCF,

```
( T̂ + v_nuc^frag + v_H[ρ_frag] + v_xc[ρ_frag] + V_env ) ψ_i = ε_i ψ_i
```

where `T̂` is the kinetic operator, `v_nuc^frag = −Σ_{A∈frag} Z_A/|r−R_A|` is the
**bare** nuclear attraction of the fragment atoms (all-electron, no PP), and
`ρ_frag` is the fragment's own density (relaxed self-consistently by PySCF).

### The approximations, stated plainly

This is a **density (frozen-environment) embedding**, *not* a projector/DFT-in-DFT
embedding. Concretely:

| Term | How it is treated | Approximation |
|------|-------------------|---------------|
| Hartree (Coulomb) | `v_H[ρ_env]` added exactly; `v_H` is **linear** in ρ | **Exact-additive** — no cross error. `v_H[ρ_frag+ρ_env] = v_H[ρ_frag] + v_H[ρ_env]`. |
| Exchange–correlation | `v_xc[ρ_env]` added; PySCF separately adds `v_xc[ρ_frag]` | **Non-additive XC neglected**: the cross term `v_xc[ρ_frag+ρ_env] − v_xc[ρ_frag] − v_xc[ρ_env]` is dropped (subsystem-additive XC). |
| Kinetic / Pauli | **no** non-additive kinetic potential `v_T^nad` is added | **Biggest approximation.** Environment Pauli repulsion on the fragment is represented only crudely (see below). No OFDFT kinetic embedding, no level-shift projector. |
| Environment density `ρ_env` | **frozen** at the converged host density; never relaxed to the presence of the AE fragment | Non-self-consistent (polarization of the environment by the excited atom is ignored). |
| Environment electrons | pseudized (QE valence density only; neighbour cores live inside their PPs) | The fragment atom is AE; its neighbours are **not**. "AE atom in a pseudized bath." |

**How Pauli exclusion is (approximately) enforced without a kinetic term.** Two
mechanisms stand in for the missing `v_T^nad`:

1. The neighbouring atoms' **local pseudopotential** stays in `V_ionic` (their
   repulsive cores are baked into the PP), so the AE fragment orbitals feel a
   wall where neighbour cores sit.
2. Near the absorber nucleus, the Hirshfeld weight drives `ρ_env → 0` (see §3),
   so `v_H[ρ_env] + v_xc[ρ_env] → 0` on the core — the environment does **not**
   push spurious H/XC onto the tightly bound 1s.

There is **no explicit orthogonalization** of the AE fragment orbitals against
environment orbitals. This is acceptable for XAS because the quantity of interest
is the *core level shift* and the *local* polarization, both dominated by
electrostatics, not by fine Pauli details in the bonding region.

### Units

QE returns **Rydberg**; everything in `casidapy.embed` is converted to and
returned in **Hartree**. The single converter is `ry_to_ha(arr) = 0.5·arr`.

---

## 2. Step 1 — Ionic peel: remove the absorber's own local PP

**Goal.** Build `V_ionic` = the *local pseudopotential of every atom except the
active site*. Since the AE fragment supplies its own **bare** `−Z/r`, we must
strip the fragment atom's PP contribution out of the host `vltot` to avoid
double counting the nucleus.

```
V_loc(active) = Σ_{A∈active}  (local PP of only atom A)
V_ionic       = vltot − V_loc(active)
```

### The structure-factor trick (`vloc_atom_from_qepy`)

QE builds the total local potential `vltot` in reciprocal space as

```
V_loc(G) = Σ_species  v_loc^species(G) · S_species(G)
```

where `S_species(G) = Σ_{atoms of species} exp(−iG·τ)` is the **structure
factor**. To isolate one atom `A`:

1. Read the FFT G-vectors and convert QE's packed `nl` map to Miller indices
   (`_miller_from_fft`), then to Cartesian G via the reciprocal lattice `bg`.
2. Build a structure factor that is **nonzero only for atom A**:
   `strf_a[:, species(A)] = exp(−iG·τ_A)` (a single-atom phase), everything else
   zero. The species column is QE ``ityp`` (ATOMIC_SPECIES order), not
   first appearance in the atom list — those differ when species are
   sorted alphabetically (``O`` then ``Ti``) but the xyz lists Ti first.
3. Temporarily install `strf_a`, call QE's `qepy_setlocal()` to regenerate the
   local potential from *that* structure factor, and read it back with
   `get_local_pp`. This is exactly the machinery that produced `vltot`, so it is
   internally consistent.
4. **Always restore** the original structure factor in a `finally` block and
   re-run `setlocal` (the driver is a shared, mutable QE state).

**Vacuum null test (why this design).** Because `V_loc(A)` is generated by the
*same* routine as `vltot`, for a **lone atom in vacuum** `vltot − V_loc(A)` is
*exactly zero* (to machine precision), not just approximately. This is the
correctness anchor for the peel. A real-space UPF interpolation (the `"upf"`
diagnostic source, `vloc_atom_field_upf`) does **not** pass this null test
because of interpolation error — it exists only for cross-checking.

### Home-cell peel (periodic crystals)

`setlocal` of one atom is the **A sublattice**: the absorber plus its copies at
supercell lattice vectors. PySCF supplies a **one-center** `−Z/r`. Subtracting
the sublattice therefore removes the PP local at every image of A without
putting AE nuclei there — a periodic array of monopoles of charge `Z_val`.
The 1s feels that Madelung well, so raw `ω₀` jumps by tens of eV between
crystals and supercells (MgO 2×2×2 vs rutile 2×2×3 O K differed by ~53 eV
after the mean of `V_env` was removed).

Production `ae_host` uses `calc_type`:

| `calc_type` | `vloc_source` | QE `assume_isolated` | `gauge_align` |
|-------------|---------------|----------------------|---------------|
| `crystal` | `home` (home-cell peel) | off | `none` |
| `molecule` (alias `cluster`) | `qepy` (vacuum-null `setlocal`) | `mt` | `none` |

`run_ae_host_pw_grid_cvs(calc_type=...)` applies this bundle. Omitted `calc_type`
is inferred from the QE driver (`assume_isolated` → molecule, else crystal).

```
φ_images     = ⟨ setlocal_A − V_UPF^A ⟩_{r_c < r < ~0.4 L}
V_loc^A_home = setlocal_A − φ_images
V_ionic      = vltot − V_loc^A_home
             = (vltot − setlocal_A) + φ_images
```

`φ_images` is taken in the Coulomb shell outside the local-PP cutoff so the
QE Fourier well (which matches `vltot` on-site, vacuum-null quality) is kept,
and only the slowly varying image field is restored. Over the 1s (~0.2 bohr)
that field is a constant to ~0.1 eV. Clusters with `assume_isolated` keep
`"qepy"` so the vacuum null is unchanged.

### `ionic_vloc_residual`

Loops over the peel list (`atom_indices`, default `[edge_atom]`), sums each
atom's `V_loc`, and returns `(V_loc(first atom), V_ionic = vltot − ΣV_loc)`.
`vloc_source="qepy"` is the cluster / vacuum-null peel; `"home"` is
production for periodic crystals; `"upf"` is diagnostic; `"auto"` selects
`home` unless QE ran with `assume_isolated`.

### MPI subtlety (`gathered_qe_array`)

QEpy sizes a `gather=True` buffer only on the root rank; other ranks get a
1-element stub that silently becomes `np.zeros(1)`. Embedding is **replicated**
work (every rank must hold the whole grid), so `gathered_qe_array` broadcasts the
root buffer to all ranks in Fortran order. Without this the ranks desync at the
next collective.

---

## 3. Step 2 — Hirshfeld (stockholder) partition of the density

**Goal.** Split the converged host valence density `ρ_tot` into an *active-site*
part and an *environment* part, so that the environment H/XC never lands on the
absorber core.

### Free-atom reference densities (`read_upf_rhoatom`)

For each atom we need a neutral free-atom reference density `ρ̃_B(r)`. This is the
UPF `PP_RHOATOM` table — the pseudo-valence atomic density, stored as `4πr²ρ`, so
we divide by `4πr²` to get `ρ(r)` (careful at `r=0`). `upf_rhoatom_on_grid`
interpolates the radial `ρ̃_B` onto the 3D grid at atom B's minimum-image
position, **zero outside the UPF table** (atomic densities are localized —
`outside="zero"` in `project_radial_to_grid`).

### The stockholder weight (`hirshfeld_weights_and_partition`)

```
w_A(r) = Σ_{A∈active} ρ̃_A(r)  /  Σ_B ρ̃_B(r)        # "active share" of the point
ρ_active(r) = w_A(r) · ρ_tot(r)
ρ_env(r)    = max( ρ_tot(r) − ρ_active(r), 0 )
```

Interpretation:
- **Near the absorber nucleus** the absorber's `ρ̃_A` dominates the sum, so
  `w_A → 1`, hence `ρ_env → 0`. No environment charge on the core. ✔
- **Far from the active site** `ρ̃_A → 0`, so `w_A → 0` and `ρ_env → ρ_tot`
  (full environment). ✔

`ρ_tot` is the **QE valence density** (`total_density_on_grid` via
`get_density`) — pseudized, no core electrons. The clamp `max(…,0)` guards
against tiny negative round-off.

The result is packaged in the frozen dataclass **`HirshfeldPartition`**
(`rho_tot, rho_tilde_a, rho_tilde_sum, w_a, rho_active, rho_env`);
`.as_tuple()` preserves a legacy unpack order.

**Parallelism.** The atom loop is round-robin distributed
(`distributed_indices`) and the per-atom densities are summed with an MPI
`Allreduce` (`mpi_allreduce_sum`). `use_gpu=True` does the weight arithmetic in
CuPy. Every rank must be able to see the QEpy driver.

`active_atoms` is the peel list — a single ion `[edge_atom]` (MgO O²⁻), or a
covalent fragment `[O, H, H]` for a water molecule in ice.

---

## 4. Step 3 — Environmental Hartree and XC potentials

**Goal.** Turn `ρ_env` into the potential it generates, `v_H[ρ_env] + v_xc[ρ_env]`,
using **QE's own kernels** (so the functional exactly matches the host).

### Hartree (`hartree_potential_ha`)

Calls QE's `v_h_of_rho_r(rhor, v)` on `ρ_env` and converts Ry→Ha. Because Hartree
is linear this is exact and needs no trickery.

### XC (`xc_potential_ha`) and the temporary-density context manager

XC is **nonlinear**, so to get `v_xc[ρ_env]` we must make QE evaluate its XC
routine *with `ρ_env` installed as the current density*. That is what
`_temporary_valence_density` does:

1. Save the current QE density (`of_r`, `of_g`) and, if `zero_nlcc=True`, the
   nonlinear-core-correction arrays (`rho_core`, `rhog_core`).
2. Install `ρ_env` as the valence density; **zero the NLCC** so `v_xc` sees only
   the trial density and does **not** re-add model core corrections that belong
   to the frozen host (these would otherwise be double-counted).
3. Call QE `v_xc(...)`, convert Ry→Ha.
4. **Restore everything** in a `finally` block (shared mutable QE state again).

**Why zero NLCC.** NLCC adds a fixed model core density into XC for the *full*
system. When we re-evaluate XC on the *environment-only* density we must not also
add the core correction, or the environment XC is inflated. This is a subtle but
important consistency choice.

---

## 5. Assembly (`build_hirshfeld_embedding`)

Putting §2–§4 together on the QE charge grid:

```
V_env = V_ionic + v_H[ρ_env] + v_xc[ρ_env]      (Hartree, DirectField on the grid)
```

Steps in the function:
1. Resolve absorber symbol, `Z`, UPF path, `z_valence`, nuclear position
   (`resolve_upf_path`, `read_upf_local_potential`).
2. `ionic_vloc_residual` → `V_ionic` (§2).
3. `hirshfeld_weights_and_partition` → `ρ_env` (§3).
4. `hartree_potential_ha`, `xc_potential_ha` → environment H/XC (§4).
5. Sum into `v_env_arr`, wrap as a DFTpy `DirectField`.
6. **Gauge shift** (§6).
7. Emit a rich `meta` dict — notably the **decomposition at the absorber nucleus**
   (`v_ionic_at_nuc_ha`, `v_h_at_nuc_ha`, `v_xc_at_nuc_ha`, `v_env_at_nuc_ha`),
   because *that value is what rigidly shifts the deep 1s* and therefore sets the
   XAS absolute edge position.

Returns `(v_env, grid, meta)`.

---

## 6. Gauge — the constant-offset freedom (the "gauge stuff")

### Why there is a gauge at all

A KS eigenvalue is only defined up to the **zero of the potential**. In a periodic
PW calculation the potential's average (the `G=0` component) is arbitrary /
conventionally fixed; there is no vacuum. When we drop an AE atom into `V_env`,
**any constant added to `V_env` shifts every core eigenvalue rigidly by that
constant.** Since the XAS edge energy is a *deep* eigenvalue, choosing this
constant correctly is essential — it is the single largest lever on the absolute
transition energy.

So we must **fix the zero of `V_env`** to a physically meaningful reference. That
choice is the *gauge*.

### The gauge modes

All implemented as "subtract a constant so that `V_env = 0` at/near a chosen
reference region." Core routines: `shift_v_env_gauge_at`,
`shift_v_env_gauge_hirshfeld`, `shift_v_env_gauge_min_density`.

| `gauge_align` | Reference zero | When to use | Routine |
|---------------|----------------|-------------|---------|
| `hirshfeld` (default for clusters/molecules; `far_field`/`vacuum` are promoted to this) | Mean of `V_env` where `ρ̃_A/ρ_tot < tol` — i.e. the **environment interior**, just outside the active basin | Molecular / cluster cells with a real vacuum | `shift_v_env_gauge_hirshfeld` |
| `true_vacuum` / `min_density` | Mean of `V_env` over the lowest-density `frac` (default 0.5%) of grid points — **genuine deep vacuum** | Finite AE cluster fidelity; matches the vacuum zero of an isolated cluster | `shift_v_env_gauge_min_density` |
| `fermi` / `ef` | Subtract the host QE Fermi energy: `V_env ← V_env − E_F` | **Periodic crystals** (no vacuum exists). Puts the host Fermi level at 0 | `shift_v_env_gauge_at(mode="fermi", fermi_ha=…)` |
| `mean` | Subtract `⟨V_env⟩` over the whole cell | Crystals, crude mean-crystal zero | `shift_v_env_gauge_at(mode="mean")` |
| `far_field` | Value at the grid point **farthest** from the absorber | Fallback; only meaningful with vacuum | `shift_v_env_gauge_at(mode="far_field")` |

`shift_v_env_gauge_hirshfeld` has a safety fallback: if fewer than 8 points fall
below `tol` (dense crystal, no low-weight region), it reverts to the far-field
point relative to the `ρ̃_A` peak.

### The crystal gauge is a **two-step** protocol

For a periodic solid there is no vacuum, so `gauge_align='fermi'` is used. It is
*half* of a two-part alignment (the other half lives in the XAS code, see
`THEORY_2`):

1. **Here (embedding):** `V_env ← V_env − E_F(QE)`. Host Fermi level becomes the
   energy zero.
2. **After the AE SCF (XAS side):** the AE fragment's residual chemical potential
   `μ_atom` is shifted to 0 as well (`align_core_to_fragment_fermi_zero`), and the
   **same shift is mirrored onto the PW virtual energies** so the CVS excitation
   energies `ω = ε_virt − ε_core` stay invariant. Step 1 alone leaves a constant
   well error `μ_atom ≠ 0` because the finite AE atom does not perfectly reproduce
   the host's mean field; step 2 removes it exactly (a constant potential shift is
   exact for eigenvalues).

The `meta["fermi_energy_ha"]` and `meta["gauge_shift_ha"]` record what was done.

---

## 7. Legacy / diagnostic embedding modes

The default is `hirshfeld`. `build_ae_embedding_potential` also dispatches
(`_normalize_embed_mode` + `_EMBED_MODE_ALIASES`) to several **non-production**
modes via `_build_legacy_v_env`, kept for ablation studies:

| `embed_mode` | `V_env` formula | Purpose |
|--------------|-----------------|---------|
| `zero` | `0` | Isolated ion / null diagnostic (no embedding at all) |
| `loc_only` | `V_ionic` only | Electrostatic-only embedding (no H/XC) |
| `full` | `v_eff[ρ_tot] − V_loc(A)` | Uses the full host effective potential (over-counts H/XC on the core) |
| `scale_vhxc` | `V_ionic + s·v_Hxc[ρ_tot]` | Geometric-free H/XC scaling. `s = clip(1 − α·z_val/Z, s_min, s_max)` via `auto_vhxc_scale`/`resolve_vhxc_scale` |
| `damp_vhxc` | `V_ionic + v_Hxc[ρ_tot]·fade(r)` | Radial fade `1 − exp(−(r/r_damp)²)` (`vhxc_radial_fade`) kills H/XC near the nucleus geometrically instead of by density partition |

`scale_vhxc` and `damp_vhxc` are the *predecessors* of the Hirshfeld idea: they
suppress H/XC near the core with an *ad hoc* radius/scale instead of the
principled stockholder weight. The Hirshfeld partition replaced them because
`w_A(r)` is parameter-free and density-adaptive.

---

## 8. Self-consistent peel rebuild (optional)

`rebuild_v_env_from_active_density` recomputes `V_env` when the AE fragment's own
density `ρ_active` is available (a self-consistent peel loop): it recomputes
`ρ_env = ρ_tot − ρ_active` with the **AE fragment density** substituted for the
Hirshfeld estimate, keeping `V_ionic` and `ρ_tot` frozen. `mix_v_env_fields`
does linear mixing `V ← (1−mix)·V_old + mix·V_new` for stability. This is an
advanced/optional path; the default single-shot build uses the Hirshfeld
`ρ_active` directly.

---

## 9. Summary of the data flow

```
QE driver (converged PW-DFT, pseudized)
   │
   ├─ vltot  ──────────────┐
   │                       ├─►  V_ionic = vltot − ΣV_loc(active)     [§2]
   ├─ setlocal(strf_A) ────┘         (structure-factor peel; vacuum-null exact)
   │
   ├─ get_density → ρ_tot ─┐
   ├─ UPF PP_RHOATOM ──────┤─►  w_A = Σρ̃_A/Σρ̃_B ; ρ_env=(1−w_A)ρ_tot  [§3]
   │                       │
   │                       ├─►  v_H[ρ_env]   (QE v_h_of_rho_r)         [§4]
   │                       └─►  v_xc[ρ_env]  (QE v_xc, NLCC zeroed)    [§4]
   │
   └────────────────────────►  V_env = V_ionic + v_H + v_xc            [§5]
                                     │
                                     └─► gauge shift (fermi/hirshfeld/…) [§6]
                                             │
                                             ▼
                              V_env(r), grid, meta   →  hand to AE GTO SCF
                                                        (THEORY_2)
```

**Continue in `THEORY_2_atom_in_cluster_cvs.md`:** how `V_env` enters the
all-electron Gaussian SCF, how the core orbitals are selected and gauge-aligned,
how PW conduction bands are projected into the Gaussian basis, and how CVS-TDA
produces the spectrum.
