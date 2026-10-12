"""AE core in the QE host + PW Casida (GIPAW μ/K).

**Default (``ae_ref='pp_core'``)**
    Restore only the orbitals missing from the absorber UPF
    (``n_core = (Z − z_valence)/2`` — 1s for B/C/N/O; 1s2s2p for Ti
    semicore).  Closed-shell core-only RKS guess → SCF of that core in
    ``V_env`` with ``v_H[ρ_core] + v_xc[ρ_core+ρ_tot] + hyb K[ρ_core]``.

**Legacy (removal candidates)**
    ``ae_ref='atom'`` / fragment refs (``bn_fragment``, ``ch4_fragment``, …)
    plus a single oneshot diagonalization.  Kept for bisecting; prefer
    ``pp_core``.

Production Hamiltonian::

    H = T − Z/r + v_H[ρ_core^AE] + v_H[ρ_tot^QE] + v_xc^DFA[ρ_core^AE + ρ_tot^QE]
        + hyb K[ρ_core] + (vltot − v_loc^A) + V_NL^neigh

Entry point: :func:`run_ae_host_pw_grid_cvs`.
"""
from __future__ import annotations

import logging
import time
import warnings
from contextlib import contextmanager
from typing import Optional

import numpy as np

from casidapy.xas.chemical_data import DEFAULTS

logger = logging.getLogger(__name__)

HA_TO_EV = DEFAULTS["ha_to_ev"]


def _strip_qe_species_label(sym: str) -> str:
    """``Mn1`` / ``Cr2`` → ``Mn`` / ``Cr`` (QE AFM multi-type labels)."""
    letters = "".join(c for c in str(sym) if c.isalpha())
    return letters.capitalize() if letters else str(sym)


def patch_qepy_ase_atoms_afm_labels():
    """Make ``Driver.get_ase_atoms`` strip AFM labels ``Mn1``→``Mn`` once."""
    try:
        from qepy.driver import Driver
    except ImportError:
        return
    if getattr(Driver.get_ase_atoms, "_casidapy_afm_ok", False):
        return

    @classmethod
    def _get_ase_atoms_afm_ok(cls):
        from ase.atoms import Atoms

        bohr_a = float(cls.qepy_modules.constants.BOHR_RADIUS_SI) * 1.0e10
        symbols = [_strip_qe_species_label(s) for s in cls.get_ions_symbols()]
        positions = np.asarray(cls.get_ions_positions(), dtype=float) * bohr_a
        lattice = np.asarray(cls.get_ions_lattice(), dtype=float) * bohr_a
        return Atoms(symbols=symbols, positions=positions, cell=lattice)

    _get_ase_atoms_afm_ok._casidapy_afm_ok = True  # type: ignore[attr-defined]
    Driver.get_ase_atoms = _get_ase_atoms_afm_ok


def ase_atoms_from_driver(driver, atoms=None):
    """ASE atoms from QEpy, tolerant of AFM species labels like ``Mn1``.

    QE ``atomic_species`` may use ``Mn1``/``Mn2`` for opposite
    ``starting_magnetization``.  QEpy's ``get_ase_atoms`` then fails in ASE
    ``symbols2numbers``.  Prefer the pre-SCF ASE ``atoms`` when provided;
    otherwise strip trailing non-letters from QE ion symbols.
    """
    if atoms is not None:
        return atoms
    patch_qepy_ase_atoms_afm_labels()
    try:
        return driver.get_ase_atoms()
    except KeyError:
        patch_qepy_ase_atoms_afm_labels()
        return driver.get_ase_atoms()


# Legacy ae_ref values kept for bisect / removal. Prefer ``pp_core``.
_LEGACY_AE_REFS = frozenset({
    "atom",
    "bn_fragment",
    "ch4_fragment",
    "c2h4_fragment",
    "c2h6_fragment",
    "benzene_fragment",
})

_SHELL_NELEC = {
    "1s": 2, "2s": 2, "2p": 6, "3s": 2, "3p": 6, "3d": 10,
    "4s": 2, "4p": 6, "4d": 10, "4f": 14, "5s": 2, "5p": 6,
    "5d": 10, "5f": 14, "6s": 2, "6p": 6,
}

# Neutral-atom Hund 2S = Nα−Nβ (PySCF mol.spin). Closed shells omitted → 0.
_HUND_SPIN_NEUTRAL = {
    1: 1,   # H ²S
    3: 1,   # Li ²S
    5: 1,   # B ²P
    6: 2,   # C ³P
    7: 3,   # N ⁴S
    8: 2,   # O ³P
    9: 1,   # F ²P
    11: 1,  # Na ²S
    13: 1,  # Al ²P
    14: 2,  # Si ³P
    15: 3,  # P ⁴S
    16: 2,  # S ³P
    17: 1,  # Cl ²P
}

_RETIRED_KWARGS = frozenset({
    "frozen_core",
    "embed_mode",
    "ae_host_hxc",
    "ae_dipole_method",
    "gipaw_augment_k",
    "invert_workdir",
    "invert_gauge",
    "scf_peel",
    "ghost_diffuse_peel",
    "local_grid",
    "rho_env",
    "rho_env_g0",
    "host_tot",
    "core_hxc",
    "scf_venv",
    "peel",
    "peel_radius_ang",
    # Failed K-edge experiments (windowed EXX / cx override / ALDA merge).
    "exx_window_ev",
    "casida_exx_cx",
    "particle_replace_ev",
    "particle_ci_merge_gap_ev",
})


def _reject_retired_kwargs(kwargs: dict) -> None:
    bad = sorted(k for k in kwargs if k in _RETIRED_KWARGS)
    extra = sorted(k for k in kwargs if k not in _RETIRED_KWARGS)
    if bad or extra:
        names = ", ".join(bad or extra)
        raise TypeError(
            f"run_ae_host_pw_grid_cvs retired/unknown argument(s): {names}. "
            "Production is frozen-core ae_host + rho_tot + GIPAW μ/K."
        )


# Periodic crystal vs finite molecule/cluster.  ``calc_type`` picks the peel
# (home-cell vs vacuum-null setlocal) and the QE isolation flag together.
_CALC_TYPE_ALIASES = {
    "crystal": "crystal",
    "periodic": "crystal",
    "solid": "crystal",
    "molecule": "molecule",
    "cluster": "molecule",
    "molecular": "molecule",
}

AE_HOST_CALC_DEFAULTS = {
    "crystal": {
        "vloc_source": "home",
        "gauge_align": "none",
        "assume_isolated": None,
    },
    "molecule": {
        "vloc_source": "qepy",
        "gauge_align": "none",
        "assume_isolated": "mt",
    },
}


def resolve_calc_type(calc_type) -> str:
    """Canonical ``crystal`` or ``molecule`` (``cluster`` / ``periodic`` aliases)."""
    if calc_type is None or str(calc_type).strip() == "":
        raise ValueError("calc_type is required; use 'crystal' or 'molecule'")
    key = str(calc_type).lower().strip()
    kind = _CALC_TYPE_ALIASES.get(key)
    if kind is None:
        raise ValueError(
            f"Unknown calc_type={calc_type!r}; use 'crystal' or 'molecule'"
        )
    return kind


def calc_type_from_driver(driver) -> str:
    """``molecule`` if QE ran with ``assume_isolated``, else ``crystal``."""
    from casidapy.embed.potential import qe_driver_is_isolated

    return "molecule" if qe_driver_is_isolated(driver) else "crystal"


def ae_host_params_for_calc_type(calc_type, **overrides):
    """Production ae_host knobs for ``crystal`` vs ``molecule``.

    Returns a dict with ``calc_type``, ``vloc_source``, ``gauge_align``,
    and ``assume_isolated`` (QE ``&system``; ``None`` for periodic).
    Non-``None`` ``overrides`` replace a field.
    """
    kind = resolve_calc_type(calc_type)
    params = {"calc_type": kind, **dict(AE_HOST_CALC_DEFAULTS[kind])}
    for key, val in overrides.items():
        if val is not None:
            if key not in params:
                raise TypeError(f"unknown ae_host calc override {key!r}")
            params[key] = val
    return params


@contextmanager
def _phase(label: str, verbose: bool):
    """Report wall time for one reconstruct phase.

    Large-cluster runs spend hours inside a single phase without emitting
    anything, which makes a stall indistinguishable from slow progress.
    """
    if not verbose:
        yield
        return
    print(f"  [reconstruct] {label} ...", flush=True)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(
            f"  [reconstruct] {label}: {time.perf_counter() - t0:.1f}s",
            flush=True,
        )

def frozen_shells_from_pp(atom_symbol, z_valence, z_atomic=None):
    """Shells replaced by the PP: ``n_frozen = Z − z_valence``, filled 1s → …"""
    from pyscf.data import elements

    z = int(z_atomic if z_atomic is not None else elements.charge(str(atom_symbol).capitalize()))
    n_frozen = int(round(z - float(z_valence)))
    shells, left = [], n_frozen
    for sh, ne in _SHELL_NELEC.items():
        if left <= 0:
            break
        shells.append(sh)
        left -= ne
    return tuple(shells), n_frozen

def _nelec_from_shells(shells):
    return sum(_SHELL_NELEC[str(s).strip().lower()] for s in shells)

def _shells_for_nelec(n_elec):
    shells, left = [], int(n_elec)
    for sh, ne in _SHELL_NELEC.items():
        if left <= 0:
            break
        shells.append(sh)
        left -= ne
    return shells

def _resolve_ae_config(
    sym,
    z,
    *,
    ae_electrons,
    shells,
    n_elec,
    z_valence,
    oxidation_state,
    oxidation_states,
):
    """Map ``ae_electrons`` mode → ``(mode, oxidation, shells, n_elec, charge)``."""
    from casidapy.xas.cvs import resolve_oxidation_state

    mode = str(ae_electrons).lower().strip()
    ox = None
    if mode in ("atomic", "neutral", "full", "all", "oxidation", "ox"):
        # Formal ion (O²⁻, Ti⁴⁺, …): charge = oxidation state
        mode = "atomic"
        ox = resolve_oxidation_state(sym, oxidation_state, oxidation_states)
        if shells is not None:
            shells = [str(s).strip().lower() for s in shells]
            n_elec = int(n_elec) if n_elec is not None else _nelec_from_shells(shells)
        else:
            n_elec = int(n_elec) if n_elec is not None else z - int(ox)
            shells = _shells_for_nelec(n_elec)
        charge = int(ox)
    elif mode in ("frozen", "pp", "core", "semicore"):
        # Only the electrons the PP removed (legacy)
        mode = "frozen"
        if shells is not None:
            shells = [str(s).strip().lower() for s in shells]
            n_elec = int(n_elec) if n_elec is not None else _nelec_from_shells(shells)
        else:
            shells, n_pp = frozen_shells_from_pp(sym, z_valence, z_atomic=z)
            n_elec = int(n_elec) if n_elec is not None else n_pp
        charge = z - n_elec
    else:
        raise ValueError(f"Unknown ae_electrons={ae_electrons!r}; use 'atomic' or 'frozen'")
    return mode, ox, list(shells), int(n_elec), int(charge)

def _resolve_mol_spin(n_elec, spin=None):
    """PySCF ``mol.spin`` = 2S = Nalpha − Nbeta (not 2S+1)."""
    ne = int(n_elec)
    if spin is not None:
        mol_spin = int(spin)
    elif ne % 2 == 1:
        mol_spin = 1
    else:
        mol_spin = 0
    if (ne - mol_spin) % 2 != 0:
        raise ValueError(
            f"Electron count {ne} and spin {mol_spin} are not consistent "
            "(PySCF mol.spin = Nalpha − Nbeta)"
        )
    return mol_spin

def ground_state_spin(z) -> int:
    """Hund ``mol.spin`` for a neutral atom. C is ³P (2), not RKS singlet."""
    return int(_HUND_SPIN_NEUTRAL.get(int(z), 0))

def _total_ao_dm(dm):
    """Spin-sum a PySCF DM to ``(nao, nao)``."""
    arr = np.asarray(dm, float)
    if arr.ndim == 3:
        if arr.shape[0] != 2:
            raise ValueError(f"expected RKS or UKS DM, got shape {arr.shape}")
        return arr[0] + arr[1]
    return arr

def _run_isolated_atom_ks(mol, xc: str):
    """Converged isolated-atom KS. Open-shell atoms are UKS (C ³P), not RKS.

    An unconverged SCF is not an eigenvalue: this raises if DIIS, level
    shift, and Newton all fail.
    """
    from pyscf import dft
    from casidapy.xas.core_hxc import canonical_xc, is_hybrid_xc

    xc = canonical_xc(xc)
    open_shell = int(mol.spin) != 0
    mf = dft.UKS(mol) if open_shell else dft.RKS(mol)
    mf.xc = xc
    mf.verbose = 0
    mf.conv_tol = 1.0e-8
    mf.max_cycle = 120
    # DF-J/K would not match the exact K used on the oneshot mol.
    if not is_hybrid_xc(xc):
        try:
            mf = mf.density_fit()
        except Exception:
            pass
    dm0 = mf.get_init_guess()
    mf.kernel(dm0)
    if not mf.converged:
        logger.info(
            "isolated-atom %s spin=%s did not converge; retrying level_shift=0.4",
            getattr(mol, "atom", "?"),
            mol.spin,
        )
        mf.level_shift = 0.4
        mf.max_cycle = max(int(mf.max_cycle), 150)
        mf.kernel(dm0)
        mf.level_shift = 0.0
    if not mf.converged:
        logger.info("isolated-atom SCF still unconverged; retrying newton()")
        try:
            mf_n = mf.newton()
            mf_n.max_cycle = max(int(getattr(mf, "max_cycle", 80)), 50)
            mf_n.verbose = 0
            mf_n.kernel(mf.make_rdm1())
            if mf_n.converged:
                mf = mf_n
        except Exception as exc:
            logger.warning("isolated-atom newton() retry failed: %s", exc)
    if not mf.converged:
        raise RuntimeError(
            f"Isolated-atom {'UKS' if open_shell else 'RKS'} "
            f"(Z-config spin={mol.spin}) did not converge; ε_1s is not a number."
        )
    return mf

def _freeze_core_occupation(mf, n_core_orb: int, *, apply: bool = True):
    """Read the full-atom 1s eigenvalue; optionally freeze ``mo_occ`` to 1s².

    ``eps_1s`` is always the physical SCF 1s of the atom (³P UKS for C).
    ``apply=True`` then overwrites ``mf.mo_occ`` so ``make_rdm1()`` is the
    frozen core density used in ``v_H[ρ_core]``.
    """
    n_core = max(0, int(n_core_orb))
    mo_e = mf.mo_energy
    uks = isinstance(mo_e, (tuple, list)) or (
        getattr(np.asarray(mo_e), "ndim", 1) == 2 and np.asarray(mo_e).shape[0] == 2
    )
    extra = {"isolated_uks": bool(uks)}
    if uks:
        ea = np.asarray(mf.mo_energy[0], float)
        eb = np.asarray(mf.mo_energy[1], float)
        ia = np.argsort(ea)[: max(n_core, 1)]
        ib = np.argsort(eb)[: max(n_core, 1)]
        extra["eps_isolated_1s_alpha_ha"] = float(ea[ia[0]])
        extra["eps_isolated_1s_beta_ha"] = float(eb[ib[0]])
        extra["isolated_magnetization"] = float(
            np.sum(np.asarray(mf.mo_occ[0], float))
            - np.sum(np.asarray(mf.mo_occ[1], float))
        )
        eps_1s = 0.5 * (
            extra["eps_isolated_1s_alpha_ha"] + extra["eps_isolated_1s_beta_ha"]
        )
        if apply:
            occ_a = np.zeros_like(ea, float)
            occ_b = np.zeros_like(eb, float)
            if n_core:
                occ_a[np.argsort(ea)[:n_core]] = 1.0
                occ_b[np.argsort(eb)[:n_core]] = 1.0
            mf.mo_occ = (occ_a, occ_b)
        return eps_1s, extra
    e = np.asarray(mo_e, float)
    extra["isolated_magnetization"] = 0.0
    eps_1s = float(e[0]) if e.size else float("nan")
    if apply:
        occ = np.zeros(int(e.size), float)
        if n_core:
            occ[:n_core] = 2.0
        mf.mo_occ = occ
    return eps_1s, extra


def _min_image_delta_ang(atoms, i: int, j: int) -> np.ndarray:
    """Minimum-image Cartesian displacement ``pos[j] - pos[i]`` (Å)."""
    cell = np.asarray(atoms.get_cell(), float)
    di = np.asarray(atoms.get_positions()[j] - atoms.get_positions()[i], float)
    if atoms.pbc.any() and cell.size:
        frac = np.linalg.solve(cell.T, di)
        frac -= np.round(frac)
        di = cell.T @ frac
    return di


def _nearest_opposite_neighbor(atoms, edge_atom: int) -> int:
    """Index of the nearest atom of the opposite species (B↔N)."""
    syms = list(atoms.get_chemical_symbols())
    ia = int(edge_atom)
    want = {"B": "N", "N": "B"}.get(str(syms[ia]).capitalize())
    if want is None:
        raise ValueError(
            f"bn_fragment AE ref requires B or N absorber; got {syms[ia]!r}"
        )
    best_j, best_d = None, float("inf")
    for j, s in enumerate(syms):
        if j == ia or str(s).capitalize() != want:
            continue
        d = float(np.linalg.norm(_min_image_delta_ang(atoms, ia, j)))
        if d < best_d:
            best_d, best_j = d, j
    if best_j is None:
        raise RuntimeError(f"no {want} neighbor of atom {ia} ({syms[ia]})")
    return int(best_j)


def _mo_mulliken_on_atom(mf, mo_idx: int, atom_idx: int) -> float:
    """Mulliken population of MO ``mo_idx`` on ``atom_idx``."""
    mol = mf.mol
    s = np.asarray(mol.intor("int1e_ovlp"), float)
    c = np.asarray(mf.mo_coeff, float)[:, int(mo_idx)]
    pop = c * (s @ c)
    i0, i1 = (int(x) for x in mol.aoslice_by_atom()[int(atom_idx), 2:4])
    return float(np.sum(pop[i0:i1]))


def _pick_absorber_1s_mo(mf, absorber_idx: int, n_core_orb: int = 1) -> list[int]:
    """Lowest-energy MO(s) dominated by the absorber (BN: N 1s then B 1s)."""
    e = np.asarray(mf.mo_energy, float)
    n_core = max(1, int(n_core_orb))
    # Deep MOs only (Ha); BN 1s sit well below −2 Ha.
    cands = [int(i) for i in np.argsort(e) if float(e[i]) < -2.0]
    if not cands:
        cands = [int(i) for i in np.argsort(e)[: max(4, n_core + 2)]]
    scored = sorted(
        ((-_mo_mulliken_on_atom(mf, i, absorber_idx), float(e[i]), i) for i in cands),
    )
    picked = []
    for _, _, i in scored:
        if i not in picked:
            picked.append(i)
        if len(picked) >= n_core:
            break
    if len(picked) < n_core:
        raise RuntimeError(
            f"could not locate {n_core} absorber-centered core MO(s) "
            f"on atom {absorber_idx}"
        )
    return picked


def _absorber_localized_1s_ao(
    mf,
    absorber_idx: int,
    *,
    n_core_orb: int = 1,
):
    """Absorber AO coefficients for 1s, localized in the deep-MO subspace.

    Identical-atom fragments (C₂H₆, C₆H₆) delocalize canonical 1s MOs. Rotate
    the deep subspace to maximize Mulliken weight on ``absorber_idx``, then
    return the absorber AO block (and ε / Mulliken diagnostics).
    """
    mol = mf.mol
    e = np.asarray(mf.mo_energy, float)
    c = np.asarray(mf.mo_coeff, float)
    s = np.asarray(mol.intor("int1e_ovlp"), float)
    n_core = max(1, int(n_core_orb))
    # Already-localized canonical MOs (CH4, C2H4, BN): keep them.
    mo_list = _pick_absorber_1s_mo(mf, absorber_idx, n_core_orb=n_core)
    pops_can = [_mo_mulliken_on_atom(mf, int(imo), absorber_idx) for imo in mo_list]
    i0, i1 = (int(x) for x in mol.aoslice_by_atom()[int(absorber_idx), 2:4])
    if min(pops_can) > 0.85:
        coeffs = [c[i0:i1, int(imo)].copy() for imo in mo_list]
        eps_list = [float(e[int(imo)]) for imo in mo_list]
        return coeffs, eps_list, pops_can, mo_list

    deep = [int(i) for i in np.argsort(e) if float(e[i]) < -2.0]
    if len(deep) < n_core:
        deep = [int(i) for i in np.argsort(e)[: max(n_core, 4)]]
    c_sub = c[:, deep]
    sc = s @ c_sub
    pop = c_sub[i0:i1, :].T @ sc[i0:i1, :]
    pop = 0.5 * (pop + pop.T)
    evals, evecs = np.linalg.eigh(pop)
    order = np.argsort(evals)[::-1]
    coeffs = []
    eps_list = []
    pop_list = []
    e_deep = e[deep]
    for k in range(n_core):
        u = np.asarray(evecs[:, int(order[k])], float)
        c_loc = c_sub @ u
        coeffs.append(c_loc[i0:i1].copy())
        eps_list.append(float(np.dot(u * e_deep, u)))
        pop_list.append(float(evals[int(order[k])]))
    return coeffs, eps_list, pop_list, list(deep)


def _nearest_neighbors(atoms, edge_atom: int, n: int, *, element=None) -> list[int]:
    """``n`` nearest neighbors of ``edge_atom`` (optional element filter)."""
    syms = list(atoms.get_chemical_symbols())
    ia = int(edge_atom)
    el = None if element is None else str(element).capitalize()
    scored = []
    for j, s in enumerate(syms):
        if j == ia:
            continue
        if el is not None and str(s).capitalize() != el:
            continue
        d = float(np.linalg.norm(_min_image_delta_ang(atoms, ia, j)))
        scored.append((d, j))
    scored.sort()
    if len(scored) < int(n):
        raise RuntimeError(
            f"need {n} neighbors of atom {ia}; found {len(scored)}"
        )
    return [int(j) for _, j in scored[: int(n)]]


def _project_fragment_1s_to_atom(
    mf,
    absorber_idx: int,
    *,
    basis: str,
    sym: str,
    pos_ang,
    n_core_orb: int,
    ae_ref: str,
    fragment_label: str,
    extra_meta=None,
    verbose: bool = False,
):
    """Shared: fragment RKS → absorber 1s DM on a one-center mol."""
    from pyscf import gto
    from pyscf.data import elements as _elements

    c_blocks, eps_list, pops, mo_list = _absorber_localized_1s_ao(
        mf, absorber_idx, n_core_orb=n_core_orb
    )
    eps_1s = float(np.mean(eps_list))
    pos = np.asarray(pos_ang, float).reshape(3)
    sym = str(sym).capitalize()
    z = int(_elements.charge(sym))
    n_core = max(1, int(n_core_orb))
    n_elec = 2 * n_core
    mol_atom = gto.M(
        atom=f"{sym} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}",
        basis=basis,
        charge=int(z - n_elec),
        spin=0,
        unit="Angstrom",
        verbose=0,
        symmetry=False,
    )
    s_atom = np.asarray(mol_atom.intor("int1e_ovlp"), float)
    if len(c_blocks[0]) != int(mol_atom.nao):
        raise RuntimeError(
            f"AO block size mismatch projecting {sym} 1s: "
            f"fragment block={len(c_blocks[0])} vs atom nao={mol_atom.nao}"
        )
    dm = np.zeros((mol_atom.nao, mol_atom.nao), float)
    for c_a in c_blocks:
        c_a = np.asarray(c_a, float).copy()
        nrm = float(c_a @ s_atom @ c_a)
        if nrm < 1e-10:
            raise RuntimeError(f"absorber 1s projection norm {nrm:g} too small")
        c_a /= np.sqrt(nrm)
        dm += 2.0 * np.outer(c_a, c_a)

    meta = {
        "ae_ref": str(ae_ref),
        "fragment": str(fragment_label),
        "fragment_converged": bool(mf.converged),
        "fragment_e_tot_ha": float(mf.e_tot),
        "eps_isolated_1s_ha": eps_1s,
        "fragment_1s_mo": [int(i) for i in mo_list],
        "fragment_1s_mulliken": [float(p) for p in pops],
        "n_core_orb": n_core,
        "n_hxc_occ": n_core,
        "include_valence": False,
        "operator": "hartree",
        "isolated_converged": True,
        "isolated_uks": False,
        "isolated_spin": 0,
        "isolated_magnetization": 0.0,
        "e_tot_isolated_ha": float(mf.e_tot),
    }
    if extra_meta:
        meta.update(extra_meta)
    if verbose:
        print(
            f"  [ae_host] {fragment_label} fragment RKS  "
            f"E={mf.e_tot:.6f} Ha  ε_1s({sym})={eps_1s:.4f} Ha  "
            f"Mulliken={pops[0]:.3f}",
            flush=True,
        )
    return mol_atom, dm, meta


def _run_bn_fragment_core_ref(
    atoms,
    edge_atom: int,
    *,
    basis: str,
    xc: str,
    n_core_orb: int,
    verbose: bool = False,
):
    """Closed-shell BN RKS → absorber 1s on a one-center mol.

    .. deprecated::
        Legacy ``ae_ref='bn_fragment'``. Prefer ``ae_ref='pp_core'``.
    """
    from pyscf import gto
    from casidapy.xas.core_hxc import canonical_xc

    xc = canonical_xc(xc)
    ia = int(edge_atom)
    ja = _nearest_opposite_neighbor(atoms, ia)
    syms = list(atoms.get_chemical_symbols())
    pos = np.asarray(atoms.get_positions(), float)
    sym_a = str(syms[ia]).capitalize()
    sym_b = str(syms[ja]).capitalize()
    pos_b = pos[ia] + _min_image_delta_ang(atoms, ia, ja)
    mol_frag = gto.M(
        atom=(
            f"{sym_a} {pos[ia, 0]:.10f} {pos[ia, 1]:.10f} {pos[ia, 2]:.10f}; "
            f"{sym_b} {pos_b[0]:.10f} {pos_b[1]:.10f} {pos_b[2]:.10f}"
        ),
        basis=basis,
        charge=0,
        spin=0,
        unit="Angstrom",
        verbose=0,
        symmetry=False,
    )
    mf = _run_isolated_atom_ks(mol_frag, xc)
    bond_ang = float(np.linalg.norm(pos_b - pos[ia]))
    return _project_fragment_1s_to_atom(
        mf,
        0,
        basis=basis,
        sym=sym_a,
        pos_ang=pos[ia],
        n_core_orb=n_core_orb,
        ae_ref="bn_fragment",
        fragment_label=f"{sym_a}{sym_b}",
        extra_meta={
            "fragment_bond_ang": bond_ang,
            "fragment_partner_index": int(ja),
        },
        verbose=verbose,
    )


def _run_ch4_fragment_core_ref(
    atoms,
    edge_atom: int,
    *,
    basis: str,
    xc: str,
    n_core_orb: int,
    verbose: bool = False,
):
    """Closed-shell CH₄ (diamond-like) → C 1s. H along crystal C–C directions."""
    from pyscf import gto
    from casidapy.xas.core_hxc import canonical_xc

    xc = canonical_xc(xc)
    ia = int(edge_atom)
    sym = str(atoms.get_chemical_symbols()[ia]).capitalize()
    if sym != "C":
        raise ValueError(f"ch4_fragment requires C absorber; got {sym}")
    pos = np.asarray(atoms.get_positions()[ia], float)
    nn = _nearest_neighbors(atoms, ia, 4, element="C")
    r_ch = 1.09  # Å
    lines = [f"C {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}"]
    for j in nn:
        u = _min_image_delta_ang(atoms, ia, j)
        nrm = float(np.linalg.norm(u))
        if nrm < 1e-8:
            raise RuntimeError("degenerate C–C vector for CH4 fragment")
        u = u / nrm
        hp = pos + r_ch * u
        lines.append(f"H {hp[0]:.10f} {hp[1]:.10f} {hp[2]:.10f}")
    mol_frag = gto.M(
        atom="; ".join(lines),
        basis=basis,
        charge=0,
        spin=0,
        unit="Angstrom",
        verbose=0,
        symmetry=False,
    )
    mf = _run_isolated_atom_ks(mol_frag, xc)
    return _project_fragment_1s_to_atom(
        mf,
        0,
        basis=basis,
        sym="C",
        pos_ang=pos,
        n_core_orb=n_core_orb,
        ae_ref="ch4_fragment",
        fragment_label="CH4",
        extra_meta={"fragment_r_ch_ang": r_ch, "fragment_nn": list(nn)},
        verbose=verbose,
    )


def _run_c2h4_fragment_core_ref(
    atoms,
    edge_atom: int,
    *,
    basis: str,
    xc: str,
    n_core_orb: int,
    verbose: bool = False,
):
    """Closed-shell C₂H₄ (graphene-like sp²) → C 1s.

    C–C along the nearest crystal neighbor; two in-plane H on the absorber
    along the other two neighbor directions; partner C capped with 2 H.
    """
    from pyscf import gto
    from casidapy.xas.core_hxc import canonical_xc

    xc = canonical_xc(xc)
    ia = int(edge_atom)
    sym = str(atoms.get_chemical_symbols()[ia]).capitalize()
    if sym != "C":
        raise ValueError(f"c2h4_fragment requires C absorber; got {sym}")
    pos = np.asarray(atoms.get_positions()[ia], float)
    nn = _nearest_neighbors(atoms, ia, 3, element="C")
    r_ch = 1.08
    # Partner carbon along first NN at crystal bond length.
    u0 = _min_image_delta_ang(atoms, ia, nn[0])
    r_cc = float(np.linalg.norm(u0))
    u0 = u0 / r_cc
    pos_c2 = pos + r_cc * u0
    lines = [
        f"C {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}",
        f"C {pos_c2[0]:.10f} {pos_c2[1]:.10f} {pos_c2[2]:.10f}",
    ]
    # Absorber H along remaining two NN directions.
    for j in nn[1:]:
        u = _min_image_delta_ang(atoms, ia, j)
        u = u / float(np.linalg.norm(u))
        hp = pos + r_ch * u
        lines.append(f"H {hp[0]:.10f} {hp[1]:.10f} {hp[2]:.10f}")
    # Partner H: reflect absorber lateral directions about the C–C axis.
    for j in nn[1:]:
        u = _min_image_delta_ang(atoms, ia, j)
        u = u / float(np.linalg.norm(u))
        # Component perpendicular to C–C, mirrored for the partner end.
        u_perp = u - np.dot(u, u0) * u0
        nrm = float(np.linalg.norm(u_perp))
        if nrm < 1e-8:
            # Fallback orthogonal in-plane vector.
            tmp = np.array([0.0, 0.0, 1.0]) if abs(u0[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            u_perp = np.cross(u0, tmp)
            u_perp /= float(np.linalg.norm(u_perp))
        else:
            u_perp /= nrm
        hp = pos_c2 + r_ch * u_perp
        lines.append(f"H {hp[0]:.10f} {hp[1]:.10f} {hp[2]:.10f}")
    mol_frag = gto.M(
        atom="; ".join(lines),
        basis=basis,
        charge=0,
        spin=0,
        unit="Angstrom",
        verbose=0,
        symmetry=False,
    )
    # C2H4: 16 electrons — closed shell.
    mf = _run_isolated_atom_ks(mol_frag, xc)
    return _project_fragment_1s_to_atom(
        mf,
        0,
        basis=basis,
        sym="C",
        pos_ang=pos,
        n_core_orb=n_core_orb,
        ae_ref="c2h4_fragment",
        fragment_label="C2H4",
        extra_meta={
            "fragment_r_cc_ang": r_cc,
            "fragment_r_ch_ang": r_ch,
            "fragment_nn": list(nn),
        },
        verbose=verbose,
    )


def _run_c2h6_fragment_core_ref(
    atoms,
    edge_atom: int,
    *,
    basis: str,
    xc: str,
    n_core_orb: int,
    verbose: bool = False,
):
    """Closed-shell C₂H₆ (diamond-like sp³) → C 1s.

    Partner C along the nearest crystal neighbor; three H on each end along
    the remaining tetrahedral crystal directions.
    """
    from pyscf import gto
    from casidapy.xas.core_hxc import canonical_xc

    xc = canonical_xc(xc)
    ia = int(edge_atom)
    sym = str(atoms.get_chemical_symbols()[ia]).capitalize()
    if sym != "C":
        raise ValueError(f"c2h6_fragment requires C absorber; got {sym}")
    pos = np.asarray(atoms.get_positions()[ia], float)
    nn = _nearest_neighbors(atoms, ia, 4, element="C")
    r_ch = 1.09
    u0 = _min_image_delta_ang(atoms, ia, nn[0])
    r_cc = float(np.linalg.norm(u0))
    if r_cc < 1e-8:
        raise RuntimeError("degenerate C–C vector for C2H6 fragment")
    u0 = u0 / r_cc
    pos_c2 = pos + r_cc * u0
    lines = [
        f"C {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}",
        f"C {pos_c2[0]:.10f} {pos_c2[1]:.10f} {pos_c2[2]:.10f}",
    ]
    for j in nn[1:]:
        u = _min_image_delta_ang(atoms, ia, j)
        u = u / float(np.linalg.norm(u))
        hp = pos + r_ch * u
        lines.append(f"H {hp[0]:.10f} {hp[1]:.10f} {hp[2]:.10f}")
    # Partner H along its three crystal neighbors other than the absorber.
    nn2 = _nearest_neighbors(atoms, nn[0], 4, element="C")
    partner_dirs = [j for j in nn2 if j != ia][:3]
    if len(partner_dirs) < 3:
        raise RuntimeError("need 3 partner C neighbors for C2H6 fragment")
    for j in partner_dirs:
        # Direction from partner site in the crystal, applied at pos_c2.
        u = _min_image_delta_ang(atoms, nn[0], j)
        nrm = float(np.linalg.norm(u))
        if nrm < 1e-8:
            raise RuntimeError("degenerate partner C–C vector for C2H6")
        u = u / nrm
        hp = pos_c2 + r_ch * u
        lines.append(f"H {hp[0]:.10f} {hp[1]:.10f} {hp[2]:.10f}")
    mol_frag = gto.M(
        atom="; ".join(lines),
        basis=basis,
        charge=0,
        spin=0,
        unit="Angstrom",
        verbose=0,
        symmetry=False,
    )
    mf = _run_isolated_atom_ks(mol_frag, xc)
    return _project_fragment_1s_to_atom(
        mf,
        0,
        basis=basis,
        sym="C",
        pos_ang=pos,
        n_core_orb=n_core_orb,
        ae_ref="c2h6_fragment",
        fragment_label="C2H6",
        extra_meta={
            "fragment_r_cc_ang": r_cc,
            "fragment_r_ch_ang": r_ch,
            "fragment_nn": list(nn),
            "fragment_partner_nn": list(partner_dirs),
        },
        verbose=verbose,
    )


def _graphene_hexagon_indices(atoms, edge_atom: int) -> list[int]:
    """Six C indices of one hexagon containing ``edge_atom`` (absorber first)."""
    ia = int(edge_atom)
    nn0 = _nearest_neighbors(atoms, ia, 3, element="C")
    # Local plane normal from the three absorber–NN vectors.
    vecs = [_min_image_delta_ang(atoms, ia, j) for j in nn0]
    normal = np.cross(vecs[0], vecs[1])
    nrm = float(np.linalg.norm(normal))
    if nrm < 1e-8:
        normal = np.cross(vecs[0], vecs[2])
        nrm = float(np.linalg.norm(normal))
    if nrm < 1e-8:
        raise RuntimeError("degenerate graphene plane for benzene fragment")
    normal = normal / nrm

    def _next_left(prev: int, cur: int) -> int:
        nns = _nearest_neighbors(atoms, cur, 3, element="C")
        cands = [j for j in nns if j != prev]
        if len(cands) < 1:
            raise RuntimeError(f"no ring continuation from atom {cur}")
        v_in = _min_image_delta_ang(atoms, prev, cur)
        best_j, best_s = None, -float("inf")
        for j in cands:
            v_out = _min_image_delta_ang(atoms, cur, j)
            score = float(np.dot(np.cross(v_in, v_out), normal))
            if score > best_s:
                best_s, best_j = score, j
        return int(best_j)

    # Walk six steps with a consistent in-plane turn; try each first neighbor.
    for first in nn0:
        ring = [ia, int(first)]
        prev, cur = ia, int(first)
        ok = True
        for _ in range(4):
            nxt = _next_left(prev, cur)
            if nxt in ring:
                ok = False
                break
            ring.append(nxt)
            prev, cur = cur, nxt
        if not ok or len(ring) != 6:
            continue
        # Close: next from last should be absorber.
        if _next_left(ring[4], ring[5]) != ia:
            continue
        return ring
    raise RuntimeError(f"could not locate hexagon around atom {ia}")


def _run_benzene_fragment_core_ref(
    atoms,
    edge_atom: int,
    *,
    basis: str,
    xc: str,
    n_core_orb: int,
    verbose: bool = False,
):
    """Closed-shell C₆H₆ (graphene hexagon) → C 1s.

    Six carbons at crystal min-image positions; one outward in-plane H per C.
    """
    from pyscf import gto
    from casidapy.xas.core_hxc import canonical_xc

    xc = canonical_xc(xc)
    ia = int(edge_atom)
    sym = str(atoms.get_chemical_symbols()[ia]).capitalize()
    if sym != "C":
        raise ValueError(f"benzene_fragment requires C absorber; got {sym}")
    ring = _graphene_hexagon_indices(atoms, ia)
    pos_all = np.asarray(atoms.get_positions(), float)
    # Place ring carbons with min-image offsets relative to the absorber.
    c_pos = [np.asarray(pos_all[ia], float)]
    for j in ring[1:]:
        c_pos.append(pos_all[ia] + _min_image_delta_ang(atoms, ia, j))
    c_pos = np.asarray(c_pos, float)
    centroid = np.mean(c_pos, axis=0)
    r_ch = 1.08
    lines = []
    for p in c_pos:
        lines.append(f"C {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}")
    for p in c_pos:
        outward = p - centroid
        nrm = float(np.linalg.norm(outward))
        if nrm < 1e-8:
            raise RuntimeError("degenerate benzene ring centroid")
        hp = p + r_ch * (outward / nrm)
        lines.append(f"H {hp[0]:.10f} {hp[1]:.10f} {hp[2]:.10f}")
    mol_frag = gto.M(
        atom="; ".join(lines),
        basis=basis,
        charge=0,
        spin=0,
        unit="Angstrom",
        verbose=0,
        symmetry=False,
    )
    mf = _run_isolated_atom_ks(mol_frag, xc)
    r_cc = float(np.linalg.norm(c_pos[1] - c_pos[0]))
    return _project_fragment_1s_to_atom(
        mf,
        0,
        basis=basis,
        sym="C",
        pos_ang=c_pos[0],
        n_core_orb=n_core_orb,
        ae_ref="benzene_fragment",
        fragment_label="C6H6",
        extra_meta={
            "fragment_r_cc_ang": r_cc,
            "fragment_r_ch_ang": r_ch,
            "fragment_ring": list(ring),
        },
        verbose=verbose,
    )


def _potential_ao_matrix(
    mol,
    v_field,
    grid,
    *,
    blksize=8000,
    grid_stride: int = 1,
    use_gpu: bool = False,
    comm=None,
):
    """``V_μν = ∫ φ_μ(r) V_env(r) φ_ν(r) dr`` on the QE charge grid.

    Grid blocks can be MPI-distributed (``Allreduce`` of the AO matrix).
    Block GEMMs use CuPy when ``use_gpu=True``.

    ``grid_stride``
        Optional subsampling of grid points (1 = full QE charge grid).
        Values > 1 reduce ``eval_ao`` cost roughly linearly; use only when
        ``V_env`` is smooth on the scale of the stride (typical for Hirshfeld
        embedding on large vacuum cells).
    """
    from pyscf.dft import numint
    from casidapy.utils.accel import (
        array_module,
        asnumpy,
        block_slices,
        mpi_allreduce_sum,
    )

    stride = max(int(grid_stride), 1)
    r = np.asarray(grid.r)
    nr = tuple(int(n) for n in grid.nr)
    # QE / DFTpy charge grids are Fortran-ordered. C-ravel pairs V with the
    # wrong (x,y,z) whenever nx, ny, nz are not equal — cubic MgO accidentally
    # mapped O to another O; rutile 100×100×96 did not, so the 1s saw ~⟨V⟩
    # and ω₀ sat at the isolated-atom edge (~538 eV) instead of ~PBE 480 eV.
    v3 = np.asarray(v_field, float).reshape(nr, order="F")
    if stride > 1:
        sl = (slice(None, None, stride),) * 3
        coords = np.stack(
            [
                np.asarray(r[a], float).reshape(nr, order="F")[sl].ravel(order="F")
                for a in range(3)
            ],
            axis=1,
        )
        v = v3[sl].ravel(order="F") * float(grid.dV) * float(stride**3)
    else:
        coords = np.stack(
            [
                np.asarray(r[0], float).reshape(nr, order="F").ravel(order="F"),
                np.asarray(r[1], float).reshape(nr, order="F").ravel(order="F"),
                np.asarray(r[2], float).reshape(nr, order="F").ravel(order="F"),
            ],
            axis=1,
        )
        v = v3.ravel(order="F") * float(grid.dV)
    nao = mol.nao_nr()
    ngrid = coords.shape[0]
    xp = array_module(use_gpu)
    vmat = xp.zeros((nao, nao), dtype=float)

    for p0, p1 in block_slices(ngrid, blksize, comm):
        ao = numint.eval_ao(mol, coords[p0:p1])  # host (nb, nao)
        ao_x = xp.asarray(ao, dtype=float)
        v_x = xp.asarray(v[p0:p1], dtype=float)
        vmat += (ao_x.T * v_x) @ ao_x

    return mpi_allreduce_sum(comm, asnumpy(vmat))


def _potential_ao_becke_gsum(
    mol,
    v_field,
    grid,
    *,
    blk_pts: int = 64,
    g_cutoff_rel: float = 1e-12,
):
    """``V_μν = ∫ φ_μ V φ_ν`` on the absorber Becke grid; ``V`` by G-sum.

    ``v_field`` is the QE charge-grid field (Fortran layout), typically
    ``V_ionic + v_H[ρ_tot]``.  The trigonometric interpolant is exact at
    arbitrary ``r`` for that band-limited potential — no voxel collocation.
    """
    from pyscf.dft import numint
    from casidapy.xas.core_hxc import build_absorber_core_becke_grid
    from casidapy.xas.pw_eval import fft_coeffs_from_orbital, pw_sum_at_coords

    coords, weights, gmeta = build_absorber_core_becke_grid(mol)
    G, c, fmeta = fft_coeffs_from_orbital(
        v_field, grid, g_cutoff_rel=g_cutoff_rel, order="F"
    )
    v_pts = pw_sum_at_coords(G, c, coords, real=True, blk_pts=int(blk_pts))
    ao = numint.eval_ao(mol, coords)
    w = np.asarray(weights, float) * np.asarray(v_pts, float)
    vmat = (ao.T * w) @ ao
    vmat = 0.5 * (vmat + vmat.T)
    meta = {
        "quadrature": "becke_gsum",
        "n_pts": int(coords.shape[0]),
        "n_G": int(fmeta.get("n_G_keep", G.shape[0])),
        "n_G_full": int(fmeta.get("n_G_full", 0)),
        "g_cutoff_rel": float(g_cutoff_rel),
        "becke": gmeta,
        "fft": fmeta,
    }
    return vmat, meta

def _n_pp_core_orbitals(z_atomic, z_valence) -> int:
    """Spatial orbitals restored by AE but absent from the PP valence density."""
    z = int(z_atomic)
    zv = int(round(float(z_valence)))
    n_core_elec = max(0, z - zv)
    return int(n_core_elec // 2)

def _isolated_ae_core_hxc_ao(
    mol,
    xc: str,
    n_core_orb: int,
    *,
    include_valence: bool = False,
    operator: str = "hxc",
):
    """AO operator from a frozen isolated-atom density.

    Default ``ρ`` is the core only (1s² for C).  ``include_valence=True`` uses
    the full isolated-atom density — diagnostic only; the ρ_tot frozen-core
    path must **not** do this, or valence Hartree is counted twice.

    ``operator='hxc'`` is PySCF ``get_veff`` (Hartree+XC of that density).
    ``operator='hartree'`` is Coulomb ``get_j`` only: XC is evaluated on the
    QE grid from ``ρ_core^AE + ρ_tot^QE``, once.
    """
    n_core = max(0, int(n_core_orb))
    mf0 = _run_isolated_atom_ks(mol, xc)
    eps_1s, occ_meta = _freeze_core_occupation(
        mf0, n_core, apply=not include_valence
    )
    dm_c = _total_ao_dm(mf0.make_rdm1())
    if include_valence:
        n_use = int(round(float(np.trace(dm_c))))
    else:
        n_use = n_core
    op = str(operator or "hxc").lower().strip()
    if op in ("none", "density", "skip"):
        v_ao = np.zeros((int(mol.nao), int(mol.nao)), float)
        op_name = "none"
    elif op in ("hartree", "j", "coulomb"):
        try:
            v_raw = mf0.get_j(dm_c)
        except TypeError:
            v_raw = mf0.get_j(mol, dm_c)
        v_ao = np.asarray(v_raw, float)
        if v_ao.ndim == 3:
            v_ao = 0.5 * (v_ao[0] + v_ao[1])
        op_name = "hartree"
    else:
        v_raw = np.asarray(mf0.get_veff(mol, mf0.make_rdm1()), float)
        if v_raw.ndim == 3:
            v_ao = 0.5 * (v_raw[0] + v_raw[1])
        else:
            v_ao = v_raw
        op_name = "hxc"
    return v_ao, {
        "n_core_orb": n_core,
        "n_hxc_occ": n_use,
        "include_valence": bool(include_valence),
        "operator": op_name,
        "isolated_converged": True,
        "isolated_uks": bool(occ_meta.get("isolated_uks")),
        "isolated_spin": int(mol.spin),
        "eps_isolated_1s_ha": float(eps_1s),
        "e_tot_isolated_ha": float(mf0.e_tot),
        **{k: v for k, v in occ_meta.items() if k != "isolated_uks"},
        "_mf": mf0,
    }

def _oneshot_ae_in_host(
    mol,
    v_env,
    grid,
    *,
    v_hxc_core_ao,
    v_nl_ao=None,
    xc="pbe",
    use_gpu: bool = False,
    comm=None,
    grid_stride: int = 1,
):
    """One diagonalization of ``T − Z/r + v_H[ρ_core] + V_env + V_NL``.

    .. deprecated::
        Legacy path used with ``ae_ref='atom'`` / fragment refs. Prefer
        :func:`_scf_ae_core_in_host` (``ae_ref='pp_core'``).

    ``v_hxc_core_ao`` is Hartree of the isolated AE core plus DFA ``v_xc``
    (and hybrid ``K[ρ_core]`` when ``xc`` is a hybrid) for ``rho_tot``.
    Host Hartree lives in ``V_env``.

    ``V_env`` is integrated on the absorber Becke grid with a G-sum of the
    QE field (not a Riemann sum on charge-FFT voxels).
    """
    from pyscf import dft, scf
    from scipy.linalg import eigh

    h0 = scf.hf.get_hcore(mol)
    v_ao, v_meta = _potential_ao_becke_gsum(mol, v_env, grid)
    if v_nl_ao is not None:
        v_ao = np.asarray(v_ao, float) + np.asarray(v_nl_ao, float)
    h = np.asarray(h0, float) + np.asarray(v_ao, float) + np.asarray(v_hxc_core_ao, float)
    s = np.asarray(mol.intor("int1e_ovlp"), float)
    eps, vec = eigh(h, s)
    n_occ = int(mol.nelectron) // 2
    occ = np.zeros(eps.size, float)
    occ[:n_occ] = 2.0
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.verbose = 0
    mf.mo_energy = eps
    mf.mo_coeff = vec
    mf.mo_occ = occ
    mf.converged = True
    mf.get_hcore = lambda *args, **kwargs: h
    mf._casidapy_v_env_quad = v_meta
    return mf


def _parse_ae_ref(ae_ref) -> str:
    """Normalize ``ae_ref``; default production key is ``pp_core``."""
    key = str(ae_ref or "pp_core").lower().strip()
    if key in ("pp_core", "upf_core", "pp", "core", "pseudo_core"):
        return "pp_core"
    if key in ("atom", "atomic", "hund", "neutral"):
        return "atom"
    if key in ("bn", "bn_fragment", "fragment_bn", "bn_mol"):
        return "bn_fragment"
    if key in ("ch4", "methane", "ch4_fragment", "fragment_ch4"):
        return "ch4_fragment"
    if key in ("c2h4", "ethylene", "c2h4_fragment", "fragment_c2h4"):
        return "c2h4_fragment"
    if key in ("c2h6", "ethane", "c2h6_fragment", "fragment_c2h6"):
        return "c2h6_fragment"
    if key in (
        "benzene",
        "c6h6",
        "benzene_fragment",
        "fragment_benzene",
        "c6h6_fragment",
    ):
        return "benzene_fragment"
    raise ValueError(
        f"ae_ref={ae_ref!r}; use 'pp_core' (default), or legacy "
        f"'atom' / 'bn_fragment' / 'ch4_fragment' / 'c2h4_fragment' / "
        f"'c2h6_fragment' / 'benzene_fragment' (removal candidates)"
    )


def _make_pp_core_mol(sym: str, pos_ang, z: int, z_valence: float, basis: str):
    """One-center mol with only the UPF-missing electrons (closed shell)."""
    from pyscf import gto

    n_core_orb = _n_pp_core_orbitals(z, z_valence)
    if n_core_orb < 1:
        raise ValueError(
            f"UPF z_valence={z_valence} for Z={z} leaves no AE core "
            f"orbitals to restore (all-electron PP?)"
        )
    n_elec = 2 * int(n_core_orb)
    charge = int(z - n_elec)
    pos = np.asarray(pos_ang, float).reshape(3)
    mol = gto.M(
        atom=f"{sym} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}",
        basis=basis,
        charge=int(charge),
        spin=0,
        unit="Angstrom",
        verbose=0,
        symmetry=False,
    )
    shells, n_frozen = frozen_shells_from_pp(sym, z_valence, z_atomic=z)
    if int(n_frozen) != int(n_elec):
        raise RuntimeError(
            f"PP core electron mismatch: shells {shells} → {n_frozen} e, "
            f"expected {n_elec} from Z−z_valence"
        )
    return mol, int(n_core_orb), int(n_elec), int(charge), list(shells)


def _run_pp_core_guess_scf(mol, xc: str, *, verbose: bool = False):
    """Isolated closed-shell RKS on PP-missing electrons only (core guess)."""
    from casidapy.xas.core_hxc import canonical_xc

    xc = canonical_xc(xc)
    mf = _run_isolated_atom_ks(mol, xc)
    e = np.asarray(mf.mo_energy, float)
    eps_1s = float(e[0]) if e.size else float("nan")
    dm = _total_ao_dm(mf.make_rdm1())
    meta = {
        "ae_ref": "pp_core",
        "fragment": "pp_core",
        "fragment_converged": bool(mf.converged),
        "fragment_e_tot_ha": float(mf.e_tot),
        "eps_isolated_1s_ha": eps_1s,
        "n_core_orb": int(mol.nelectron) // 2,
        "n_hxc_occ": int(mol.nelectron) // 2,
        "include_valence": False,
        "operator": "hartree",
        "isolated_converged": bool(mf.converged),
        "isolated_uks": False,
        "isolated_spin": 0,
        "isolated_magnetization": 0.0,
        "e_tot_isolated_ha": float(mf.e_tot),
    }
    if verbose:
        print(
            f"  [ae_host] pp_core guess RKS  "
            f"n_elec={mol.nelectron}  charge={mol.charge}  "
            f"E={mf.e_tot:.6f} Ha  ε_1s={eps_1s:.4f} Ha  "
            f"converged={mf.converged}",
            flush=True,
        )
    return mf, dm, meta


def _scf_ae_core_in_host(
    mol,
    v_env,
    grid,
    driver,
    *,
    dm0,
    xc: str,
    n_core_orb: int,
    upf_path=None,
    v_nl_ao=None,
    verbose: bool = False,
    conv_tol: float = 1.0e-8,
    max_cycle: int = 80,
):
    """SCF the PP-missing core in ``V_env`` (iterates ``ρ_core``, not oneshot).

    ``get_hcore`` = ``T−Z/r + V_env + V_NL``.
    ``get_veff`` = ``v_H[ρ_core] + v_xc[ρ_core+ρ_tot] + hyb K[ρ_core]`` via
    :func:`core_plus_tot_becke_ao` each cycle.
    """
    from pyscf import dft, lib, scf
    from casidapy.xas.core_hxc import canonical_xc, core_plus_tot_becke_ao

    xc = canonical_xc(xc)
    h0 = np.asarray(scf.hf.get_hcore(mol), float)
    v_ext, v_meta = _potential_ao_becke_gsum(mol, v_env, grid)
    v_ext = np.asarray(v_ext, float)
    if v_nl_ao is not None:
        v_ext = v_ext + np.asarray(v_nl_ao, float)
    h_ext = h0 + v_ext

    mf = dft.RKS(mol)
    mf.xc = xc
    mf.verbose = 0
    mf.conv_tol = float(conv_tol)
    mf.max_cycle = int(max_cycle)
    # Density-fit J/K would disagree with Becke/EXX assembly in get_veff.
    mf.get_hcore = lambda *args, **kwargs: h_ext

    becke_holder = {"meta": None}

    def _get_veff(mol_=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if dm is None:
            dm = mf.make_rdm1()
        dm_use = _total_ao_dm(dm)
        v_ao, meta = core_plus_tot_becke_ao(
            mol,
            dm_use,
            driver,
            grid,
            xc=xc,
            n_core_orb=n_core_orb,
            upf_path=upf_path,
        )
        becke_holder["meta"] = meta
        return lib.tag_array(
            np.asarray(v_ao, float),
            ecoul=0.0,
            exc=0.0,
            vj=None,
            vk=None,
        )

    mf.get_veff = _get_veff
    mf.kernel(np.asarray(dm0, float))
    if not mf.converged:
        logger.info("pp_core host SCF not converged; retrying level_shift=0.2")
        mf.level_shift = 0.2
        mf.max_cycle = max(int(mf.max_cycle), 120)
        mf.kernel(mf.make_rdm1())
        mf.level_shift = 0.0
    if not mf.converged:
        try:
            mf_n = mf.newton()
            mf_n.max_cycle = 50
            mf_n.verbose = 0
            mf_n.get_hcore = mf.get_hcore
            mf_n.get_veff = mf.get_veff
            mf_n.kernel(mf.make_rdm1())
            if mf_n.converged:
                mf = mf_n
        except Exception as exc:
            logger.warning("pp_core host newton() retry failed: %s", exc)
    mf._casidapy_v_env_quad = v_meta
    mf._casidapy_becke_meta = becke_holder["meta"]
    if verbose:
        e = np.asarray(mf.mo_energy, float)
        print(
            f"  [ae_host] pp_core host SCF  "
            f"converged={mf.converged}  E={mf.e_tot:.6f} Ha  "
            f"ε_1s={float(e[0]) if e.size else float('nan'):.4f} Ha  "
            f"cycles≤{mf.max_cycle}",
            flush=True,
        )
    return mf


def _physical_cvs_edge(w_ev, f, omega_ip0_ev, *, binding_allow_ev=8.0, window_ev=15.0):
    """First physical CVS root and brightest stick in ``[ω0, ω0+window]``.

    Drops collapsed roots more than ``binding_allow_ev`` below the independent-
    particle onset ``ε_LUMO − ε_1s``.  A few-eV core exciton is kept; a
    ~70 eV kernel ghost is not.
    """
    w = np.asarray(w_ev, float).ravel()
    f = np.asarray(f, float).ravel()
    if w.size == 0:
        nan = float("nan")
        return {
            "omega0_ev": nan,
            "omega_sigma_ev": nan,
            "delta_ev": nan,
            "f0": nan,
            "f_sigma": nan,
            "n_ghost": 0,
            "omega_ip0_ev": float(omega_ip0_ev) if omega_ip0_ev is not None else nan,
        }
    ip0 = float(omega_ip0_ev) if omega_ip0_ev is not None else float("nan")
    phys = np.ones(w.size, dtype=bool)
    n_ghost = 0
    if np.isfinite(ip0):
        phys = w >= (ip0 - float(binding_allow_ev))
        n_ghost = int(w.size - int(np.count_nonzero(phys)))
        if not np.any(phys):
            phys = np.ones(w.size, dtype=bool)
            n_ghost = 0
    i0 = int(np.flatnonzero(phys)[np.argmin(w[phys])])
    edge_ev = float(w[i0])
    window = phys & (w >= edge_ev) & (w <= edge_ev + float(window_ev))
    if np.any(window):
        i_s = int(np.flatnonzero(window)[np.argmax(f[window])])
    else:
        i_s = int(np.argmax(f[phys]))
    sigma_ev = float(w[i_s])
    return {
        "omega0_ev": edge_ev,
        "omega_sigma_ev": sigma_ev,
        "delta_ev": float(sigma_ev - edge_ev),
        "f0": float(f[i0]),
        "f_sigma": float(f[i_s]),
        "n_ghost": n_ghost,
        "omega_ip0_ev": ip0,
    }

def _reconstruct_ae_host_frozen_core(
    driver,
    edge_atom,
    *,
    active=None,
    edge="K",
    basis="aug-cc-pvqz",
    xc="pbe",
    upf_path=None,
    oxidation_state=None,
    oxidation_states=None,
    ae_electrons="atomic",
    ae_ref: str = "pp_core",
    core_localize=None,
    grid_stride=1,
    use_gpu=False,
    comm=None,
    verbose=False,
    calc_type=None,
    atoms=None,
):
    """AE core in the host: default ``pp_core`` SCF; legacy oneshot/fragments.

    Production ``ae_host_hxc='rho_tot'``.

    ``ae_ref``
        ``"pp_core"`` (default) — UPF-missing orbitals only; closed-shell
        guess SCF then host SCF.
        Legacy (removal): ``"atom"``, ``"bn_fragment"``, ``"ch4_fragment"``,
        ``"c2h4_fragment"``, ``"c2h6_fragment"``, ``"benzene_fragment"``.
    """
    from pyscf import gto
    from pyscf.data import elements as _elements
    from casidapy.embed.potential import (
        build_ae_embedding_potential,
        read_upf_z_valence,
        resolve_upf_path,
    )
    from casidapy.xas.cvs import core_from_mf
    from casidapy.xas.core_hxc import canonical_xc, core_plus_tot_becke_ao

    xc_requested = str(xc)
    xc = canonical_xc(xc)
    ae_ref_key = _parse_ae_ref(ae_ref)
    if ae_ref_key in _LEGACY_AE_REFS:
        warnings.warn(
            f"ae_ref={ae_ref_key!r} is a legacy path scheduled for removal; "
            f"prefer ae_ref='pp_core' (UPF-core SCF in the host).",
            DeprecationWarning,
            stacklevel=2,
        )

    atoms = ase_atoms_from_driver(driver, atoms)
    ia = int(edge_atom)
    if active is None:
        active = [ia]
    active = [int(i) for i in active]

    sym = str(atoms.get_chemical_symbols()[ia])
    pos = atoms.get_positions()[ia]
    z = int(_elements.charge(sym))
    upf_for_z = resolve_upf_path(driver, sym, upf_path)
    z_val = float(read_upf_z_valence(upf_for_z))
    n_core_orb = _n_pp_core_orbitals(z, z_val)

    mf_iso = None
    dm_core = None
    mol = None
    n_elec = charge = mol_spin = mode = ox = shells = None
    core_meta: dict = {}
    use_pp_core_scf = ae_ref_key == "pp_core"

    _FRAG_DISPATCH = {
        "bn_fragment": _run_bn_fragment_core_ref,
        "ch4_fragment": _run_ch4_fragment_core_ref,
        "c2h4_fragment": _run_c2h4_fragment_core_ref,
        "c2h6_fragment": _run_c2h6_fragment_core_ref,
        "benzene_fragment": _run_benzene_fragment_core_ref,
    }
    if use_pp_core_scf:
        mol, n_core_orb, n_elec, charge, shells = _make_pp_core_mol(
            sym, pos, z, z_val, basis
        )
        mol_spin = 0
        mode, ox = "pp_core", 0
        mf_iso, dm_core, core_meta = _run_pp_core_guess_scf(
            mol, xc, verbose=verbose
        )
    elif ae_ref_key in _FRAG_DISPATCH:
        mol, dm_core, core_meta = _FRAG_DISPATCH[ae_ref_key](
            atoms,
            ia,
            basis=basis,
            xc=xc,
            n_core_orb=n_core_orb,
            verbose=verbose,
        )
        n_elec = 2 * int(n_core_orb)
        charge = int(z - n_elec)
        mol_spin = 0
        mode, ox, shells = "atomic", 0, _shells_for_nelec(n_elec)
    else:
        spin_iso = ground_state_spin(z)
        mol_core = gto.M(
            atom=f"{sym} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}",
            basis=basis,
            charge=0,
            spin=int(spin_iso),
            symmetry=False,
            unit="Angstrom",
            verbose=0,
        )
        _v_dummy, core_meta = _isolated_ae_core_hxc_ao(
            mol_core,
            xc,
            n_core_orb,
            include_valence=False,
            operator="hartree",
        )
        mf_iso = core_meta.pop("_mf", None)
        if verbose:
            print(
                f"  [ae_host] isolated atom "
                f"{'UKS' if core_meta.get('isolated_uks') else 'RKS'} "
                f"spin={core_meta.get('isolated_spin')}  "
                f"converged={core_meta.get('isolated_converged')}  "
                f"M={core_meta.get('isolated_magnetization')}  "
                f"ε_1s={core_meta.get('eps_isolated_1s_ha')} Ha",
                flush=True,
            )

    calc = ae_host_params_for_calc_type(
        calc_type if calc_type is not None else calc_type_from_driver(driver)
    )
    v_env, grid, meta = build_ae_embedding_potential(
        driver,
        ia,
        upf_path=upf_path,
        verbose=verbose,
        embed_mode="ae_host",
        gauge_align=calc["gauge_align"],
        vloc_source=calc["vloc_source"],
        active_atoms=list(active),
        ae_host_hxc="rho_tot",
        oxidation_state=oxidation_state,
        oxidation_states=oxidation_states,
        kedf="none",
        use_gpu=use_gpu,
        comm=comm,
    )
    meta["calc_type"] = calc["calc_type"]
    meta["assume_isolated"] = calc["assume_isolated"]
    meta["ae_ref"] = ae_ref_key
    z_val = float(meta.get("z_valence") or z_val)

    if ae_ref_key == "atom":
        mode, ox, shells, n_elec, charge = _resolve_ae_config(
            sym,
            z,
            ae_electrons=ae_electrons,
            shells=None,
            n_elec=None,
            z_valence=z_val,
            oxidation_state=oxidation_state,
            oxidation_states=oxidation_states,
        )
        spin_hint = ground_state_spin(z) if int(charge) == 0 else None
        mol_spin = _resolve_mol_spin(n_elec, spin=spin_hint)
        mol = gto.M(
            atom=f"{sym} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}",
            basis=basis,
            charge=int(charge),
            spin=int(mol_spin),
            unit="Angstrom",
            verbose=0,
        )
        if mf_iso is None:
            raise RuntimeError("rho_tot frozen-core needs the isolated-atom mf")
        dm_core = _total_ao_dm(mf_iso.make_rdm1())

    nl_sites = meta.get("ae_host_nl_sites") or []
    v_nl = None
    if nl_sites:
        from casidapy.embed.downfold import nonlocal_pp_ao_matrix_from_sites

        sites = [(s["pos_bohr"], s["upf_path"]) for s in nl_sites]
        v_nl, nl_meta = nonlocal_pp_ao_matrix_from_sites(
            mol, sites, becke_level=4, verbose=verbose
        )
        meta["ae_host_v_nl"] = {
            "n_sites": int(nl_meta.get("n_sites", 0)),
            "frobenius_ha": float(np.linalg.norm(v_nl)),
        }

    if use_pp_core_scf:
        mf = _scf_ae_core_in_host(
            mol,
            v_env,
            grid,
            driver,
            dm0=dm_core,
            xc=xc,
            n_core_orb=n_core_orb,
            upf_path=upf_path or upf_for_z,
            v_nl_ao=v_nl,
            verbose=verbose,
        )
        becke_meta = getattr(mf, "_casidapy_becke_meta", None) or {}
        if not becke_meta:
            _dm_f = _total_ao_dm(mf.make_rdm1())
            _, becke_meta = core_plus_tot_becke_ao(
                mol,
                _dm_f,
                driver,
                grid,
                xc=xc,
                n_core_orb=n_core_orb,
                upf_path=upf_path or upf_for_z,
            )
        core_meta["operator"] = becke_meta.get("assemble") or "radial_hartree+becke_xc"
        core_meta["becke"] = becke_meta
        core_meta["hybrid"] = becke_meta.get("hybrid")
        core_meta["host_scf"] = True
        core_meta["host_scf_converged"] = bool(mf.converged)
        meta["ae_host_becke"] = becke_meta
        meta["xc"] = str(xc)
        meta["xc_requested"] = xc_requested
        meta["host_scf"] = True
        if verbose and becke_meta:
            hyb = becke_meta.get("hybrid") or {}
            exx_s = ""
            if abs(float(hyb.get("hybrid_coeff") or 0.0)) > 1e-14 or abs(
                float(hyb.get("rsh_alpha") or 0.0)
            ) > 1e-14:
                exx_s = (
                    f"  EXX hyb={hyb.get('hybrid_coeff')}  "
                    f"ω={hyb.get('rsh_omega')}  α={hyb.get('rsh_alpha')}  "
                    f"||K||_F={hyb.get('exx_frobenius_ha'):.4f} Ha  "
                    f"scope={hyb.get('exx_scope')}"
                )
            print(
                f"  [ae_host] Becke ∫ρ_core={becke_meta['rho_core_nelec']:.6f} e  "
                f"(target {becke_meta['rho_core_nelec_target']:.3f}, "
                f"err={becke_meta['rho_core_nelec_err']:+.2e})  "
                f"v_H[core](0)={becke_meta['v_h_core_r0_ha']:+.4f} Ha  "
                f"r∞·v_H={becke_meta['v_h_core_rmax_times_r']:+.4f}  "
                f"nlcc={becke_meta['has_nlcc']}  xc={xc}  ae_ref={ae_ref_key}{exx_s}",
                flush=True,
            )
    else:
        v_becke, becke_meta = core_plus_tot_becke_ao(
            mol,
            dm_core,
            driver,
            grid,
            xc=xc,
            n_core_orb=n_core_orb,
            upf_path=upf_path or upf_for_z,
        )
        v_hxc_core = v_becke
        core_meta["operator"] = becke_meta.get("assemble") or "radial_hartree+becke_xc"
        core_meta["becke"] = becke_meta
        core_meta["hybrid"] = becke_meta.get("hybrid")
        core_meta["host_scf"] = False
        meta["ae_host_becke"] = becke_meta
        meta["xc"] = str(xc)
        meta["xc_requested"] = xc_requested
        meta["host_scf"] = False
        if verbose:
            hyb = (becke_meta.get("hybrid") or {})
            exx_s = ""
            if abs(float(hyb.get("hybrid_coeff") or 0.0)) > 1e-14 or abs(
                float(hyb.get("rsh_alpha") or 0.0)
            ) > 1e-14:
                exx_s = (
                    f"  EXX hyb={hyb.get('hybrid_coeff')}  "
                    f"ω={hyb.get('rsh_omega')}  α={hyb.get('rsh_alpha')}  "
                    f"||K||_F={hyb.get('exx_frobenius_ha'):.4f} Ha  "
                    f"scope={hyb.get('exx_scope')}"
                )
            print(
                f"  [ae_host] Becke ∫ρ_core={becke_meta['rho_core_nelec']:.6f} e  "
                f"(target {becke_meta['rho_core_nelec_target']:.3f}, "
                f"err={becke_meta['rho_core_nelec_err']:+.2e})  "
                f"v_H[core](0)={becke_meta['v_h_core_r0_ha']:+.4f} Ha  "
                f"r∞·v_H={becke_meta['v_h_core_rmax_times_r']:+.4f}  "
                f"nlcc={becke_meta['has_nlcc']}  xc={xc}  ae_ref={ae_ref_key}{exx_s}",
                flush=True,
            )
        mf = _oneshot_ae_in_host(
            mol,
            v_env,
            grid,
            v_hxc_core_ao=v_hxc_core,
            v_nl_ao=v_nl,
            xc=xc,
            use_gpu=use_gpu,
            comm=comm,
            grid_stride=grid_stride,
        )
    v_quad = getattr(mf, "_casidapy_v_env_quad", None)
    if v_quad:
        meta["v_env_quadrature"] = v_quad
        if verbose:
            print(
                f"  [ae_host] V_env quadrature=becke_gsum  "
                f"n_pts={v_quad.get('n_pts')}  n_G={v_quad.get('n_G')}",
                flush=True,
            )
    mf._casidapy_reconstruct = {
        "shells": list(shells or []),
        "charge": int(charge),
        "n_elec": int(n_elec),
        "ae_electrons": mode,
        "ae_ref": ae_ref_key,
        "oxidation_state": None if ox is None else int(ox),
        "spin": int(mol_spin),
        "z_valence": float(z_val),
        "z_atomic": z,
        "basis": basis,
        "xc": xc,
        "frozen_core": True,
        "host_scf": bool(use_pp_core_scf),
        "ae_host_hxc": meta.get("ae_host_hxc") or "rho_tot",
        "core_hxc": "core",
        **{
            f"core_{k}": v
            for k, v in core_meta.items()
            if not str(k).startswith("_")
        },
    }
    core = core_from_mf(
        mf,
        edge=edge,
        edge_atom_indices=[0],
        core_localize=core_localize,
        verbose=verbose,
    )
    if getattr(core, "meta", None) is None:
        core.meta = {}
    core.meta["reconstruct_meta"] = dict(meta)
    core.meta["reconstruct_meta"]["method"] = (
        "ae_host_pp_core_scf" if use_pp_core_scf else "ae_host_frozen_core"
    )
    core.meta["reconstruct_meta"]["ae_host_hxc"] = meta.get("ae_host_hxc") or "rho_tot"
    core.meta["reconstruct_meta"]["core_hxc"] = "core"
    core.meta["reconstruct_meta"]["frozen_core"] = True
    core.meta["reconstruct_meta"]["host_scf"] = bool(use_pp_core_scf)
    core.meta["reconstruct_meta"]["ae_ref"] = ae_ref_key
    core.meta["reconstruct_meta"]["ae_core_hxc"] = core_meta
    if verbose:
        rho_meta = meta.get("ae_host_rho_tot") or {}
        becke = core_meta.get("becke") or {}
        vh_nuc = rho_meta.get("v_h_tot_nucleus_ha")
        vhs = (
            f"  v_H[tot](nuc)={float(vh_nuc):+.3f} Ha"
            if vh_nuc is not None
            else ""
        )
        qcs = (
            f"  ∫ρ_core={float(becke['rho_core_nelec']):.4f}"
            if becke.get("rho_core_nelec") is not None
            else ""
        )
        kind = meta.get("vltot_kind") or (rho_meta.get("vltot") or {}).get("kind")
        kinds = f"  vltot={kind}" if kind else ""
        homes = (
            f"  vloc={meta.get('vloc_source')}"
            if meta.get("vloc_source")
            else ""
        )
        print(
            f"  [ae_host] frozen-core "
            f"{'host-SCF' if use_pp_core_scf else 'oneshot'}: "
            f"n_core_orb={core_meta.get('n_core_orb', n_core_orb)}  "
            f"calc={meta.get('calc_type')}  ae_ref={ae_ref_key}  "
            f"core_hxc=core/{core_meta.get('operator')}  "
            f"n_hxc_occ={core_meta.get('n_hxc_occ')}  "
            f"n_elec={n_elec}  eps_1s={float(np.min(core.energies)):.4f} Ha  "
            f"hxc={meta.get('ae_host_hxc')}{kinds}{homes}{vhs}{qcs}",
            flush=True,
        )
    return core, mf, v_env, grid, core.meta["reconstruct_meta"]


def run_ae_host_pw_grid_cvs(
    driver,
    edge_atom,
    *,
    edge: str = "K",
    basis: str = "aug-cc-pvdz",
    xc: str = "pbe",
    pw_xc: str = "PBE",
    n_virt: int = 60,
    n_states: int = 40,
    oxidation_states=None,
    oxidation_state=None,
    ae_electrons: str = "atomic",
    ae_ref: str = "pp_core",
    core_localize: Optional[str] = "proj",
    grid_stride: int = 2,
    upf_path=None,
    use_gpu: bool = False,
    comm=None,
    verbose: bool = True,
    orthogonalize_to_virt: bool = True,
    gipaw_upf,
    gipaw_core_wfc: str = "ae",
    dipole_mapping: str = "log",
    dipole_psi_mode: str = "spline3",
    calc_type=None,
    soc=False,
    casida_exx: str = "off",
    weight_virtuals: bool = False,
    virt_weight_min: float = 0.0,
    virt_capture: float = 0.8,
    virt_window_ev: float = 0.0,
    particle_ci: bool = False,
    slater_scale: float = 1.0,
    particle_ci_min_complete: Optional[float] = None,
    particle_window_ev: Optional[float] = None,
    atoms=None,
    **retired,
):
    """AE core + QE virtuals + GIPAW μ/K + grid ALDA ``K``.

    Default ``ae_ref='pp_core'``: closed-shell SCF of UPF-missing orbitals
    in ``V_env``.  Legacy fragment/atom oneshot refs remain available.
    ``ω = ε_a^QE − ε_core^AE``.
    Hartree of ``ρ_tot`` lives on the FFT; ``v_H[ρ_core]`` and
    ``v_xc^DFA[ρ_core + ρ_tot]`` are Becke.  Hybrids add exact exchange of
    the core DM.  Dipoles and kernel restoration inside ``r_c`` are GIPAW.

    ``atoms``
        Optional pre-SCF ASE structure.  Required for QE AFM runs that
        label species ``Mn1``/``Cr1`` (ASE cannot parse those symbols from
        ``driver.get_ase_atoms()``).

    ``pw_xc`` is the Casida ALDA ``f_xc``.  A hybrid name uses the DFA
    remainder only.  Hybrid ``xc`` adds core Fock on the oneshot.
    ``casida_exx="gto"`` adds fragment ``(ij|ab)`` of GIPAW-projected
    virtuals (functional ``c_x``).  ``weight_virtuals`` prunes the TDA
    virtual pool then runs ordinary CVS-TDA / SI-SOC: L/M ranks by absorber
    ``nd`` GTO weight; K/1s ranks by core→virt dipole weight
    (oscillator-strength capture).  ``virt_capture`` is the cumulative
    fraction kept; ``virt_window_ev`` (optional, eV above CBM) restricts
    the eligible pool before capture.  ``particle_ci`` SVD-projects onto
    the absorber particle shell (``nd`` for L/M, ``np`` for K) and runs
    TDA ``A`` with GTO ``(ia|jb)`` / ``(ij|ab)`` at ``slater_scale``
    (default 1).  ``particle_window_ev`` (K default 8 eV) limits the SVD
    pool to near-CBM virtuals.

    ``soc``
        ``False`` / ``"none"`` — singlet CVS-TDA (K-edge default).
        ``"si"`` — spatial 2p/3d + singlet/triplet TDA coupled by hole-only
        SI-SOC (L/M-edge).  ``True`` / ``"orbital"`` is not used here.

    ``calc_type``
        ``"crystal"`` — home-cell peel, no ``assume_isolated``.
        ``"molecule"`` (alias ``cluster``) — vacuum-null ``setlocal`` peel,
        QE Martyna–Tuckerman.  Omitted: inferred from the QE driver.

    Returns ``(res, core, kernel, mf, diag)``.
    """
    _reject_retired_kwargs(retired)
    if not gipaw_upf:
        raise ValueError(
            "run_ae_host_pw_grid_cvs requires gipaw_upf= (same GIPAW UPF "
            "as the host SCF). ONCV cannot reconstruct."
        )
    from casidapy.casida_api import CasidaOptions
    from casidapy.xas.cvs import (
        _orthogonalize_occ_to_virt,
        build_pw_kernel_from_qepy,
        clone_pw_kernel_triplet,
        core_mos_to_pw_fields,
        inject_core_orbitals,
        parse_soc_mode,
        run_cvs_si_soc,
        run_cvs_tda,
        CVSSOCResults,
    )
    from casidapy.xas.dipole import (
        absorber_origin_bohr,
        bind_ae_core_dipole,
        core_virtual_dipole_matrix_grid,
        dipole_vector_magnitude,
        pseudo_dipole_scale_report,
    )
    from casidapy.xas.gipaw import (
        apply_gipaw_kernel_virtuals,
        core_virtual_dipole_matrix_paw_gipaw,
    )

    atoms = ase_atoms_from_driver(driver, atoms)
    ia = int(edge_atom)
    calc = ae_host_params_for_calc_type(
        calc_type if calc_type is not None else calc_type_from_driver(driver)
    )
    edge_key = str(edge).upper().strip()
    soc_mode = parse_soc_mode(soc)
    if soc_mode == "orbital":
        raise ValueError(
            "run_ae_host_pw_grid_cvs does not apply orbital SOC on the core. "
            "For L/M-edge pass soc='si' (spatial 2p/3d + singlet/triplet SI-SOC)."
        )
    if soc_mode == "si" and edge_key in ("K", "1S"):
        logger.warning(
            "soc='si' on a K-edge 1s: hole SOC is first-order zero; "
            "mixing is particle-only and hole_only drops it."
        )
    if verbose:
        print(
            f"  [pw-grid] ae_host frozen-core  calc={calc['calc_type']}  "
            f"vloc={calc['vloc_source']}  gauge={calc['gauge_align']}  "
            f"hxc=rho_tot  + QE virtuals  n_virt={n_virt}  GIPAW μ/K"
            f"  edge={edge_key}  soc={soc_mode}",
            flush=True,
        )
    core, mf, _v_env, _grid_ae, fc_meta = _reconstruct_ae_host_frozen_core(
        driver,
        ia,
        active=[ia],
        edge=edge,
        basis=basis,
        xc=xc,
        upf_path=upf_path,
        oxidation_state=oxidation_state,
        oxidation_states=oxidation_states,
        ae_electrons=ae_electrons,
        ae_ref=ae_ref,
        core_localize=core_localize,
        grid_stride=int(grid_stride),
        use_gpu=use_gpu,
        comm=comm,
        verbose=verbose,
        calc_type=calc["calc_type"],
        atoms=atoms,
    )

    from casidapy.xas.core_hxc import casida_alda_xc, casida_exx_kernel, parse_casida_exx_mode

    pw_xc_alda, casida_xc_meta = casida_alda_xc(pw_xc)
    casida_exx_mode = parse_casida_exx_mode(casida_exx)
    if casida_exx_mode == "fft":
        raise ValueError(
            "casida_exx='fft' collocates the core product on the QE mesh "
            "(the abandoned 1s artifact). Use 'gto' for fragment (ij|ab) "
            "of GIPAW-projected virtuals, or 'off'."
        )
    weight_virtuals = bool(weight_virtuals)
    particle_ci = bool(particle_ci)
    if particle_ci and weight_virtuals:
        logger.warning(
            "particle_ci SVD-projects the full virtual pool; "
            "weight_virtuals prune is ignored."
        )
        weight_virtuals = False
    virt_window_ev = float(virt_window_ev or 0.0)
    if particle_ci and casida_exx_mode == "gto":
        logger.warning(
            "particle_ci replaces band-TDA EXX; casida_exx=gto is ignored "
            "for the exciton CIS (slater_scale sets c_x)."
        )
    exx_meta = casida_exx_kernel(xc, pw_xc, mode=casida_exx_mode)
    casida_xc_meta.update(exx_meta)
    if verbose and casida_xc_meta.get("casida_fxc") == "dfa_remainder":
        print(
            f"  [pw-grid] Casida f_xc={pw_xc_alda} is the DFA remainder of "
            f"{pw_xc}",
            flush=True,
        )
    if verbose and abs(float(exx_meta.get("hybrid_coeff") or 0.0)) > 1e-14:
        if casida_exx_mode == "gto":
            print(
                f"  [pw-grid] hybrid oneshot Fock + GTO (ij|ab)  "
                f"c_x={exx_meta['hybrid_coeff']:.4f}  xc={exx_meta['exx_xc']}",
                flush=True,
            )
        else:
            print(
                f"  [pw-grid] hybrid 1s Fock only (+U-like)  "
                f"c_x={exx_meta['hybrid_coeff']:.4f}  xc={exx_meta['exx_xc']}  "
                f"Casida A stays ALDA (no (ij|ab))",
                flush=True,
            )

    kernel, grid = build_pw_kernel_from_qepy(
        driver,
        n_virt=int(n_virt),
        use_gpu=use_gpu,
        xc=pw_xc_alda,
        casida_uspp=False,
        virt_energy_shift_ha=0.0,
        verbose=verbose,
    )
    core_fields = core_mos_to_pw_fields(core, grid, use_gpu=use_gpu, comm=comm)
    inject_core_orbitals(
        kernel,
        core.energies,
        core_fields,
        disable_uspp=True,
        orthogonalize_to_virt=bool(orthogonalize_to_virt),
        use_gpu=use_gpu,
    )
    R0 = absorber_origin_bohr(core)
    gipaw_k_meta = apply_gipaw_kernel_virtuals(
        kernel,
        gipaw_upf,
        R0,
        psi_mode=dipole_psi_mode,
        mapping=dipole_mapping,
    )
    if orthogonalize_to_virt:
        tilde = getattr(kernel, "_psi_unocc_tilde", None)
        psi_occ = _orthogonalize_occ_to_virt(
            list(kernel._psi_occ),
            list(kernel._psi_unocc),
            kernel.grid,
            use_gpu=use_gpu,
        )
        kernel.set_active_orbitals(
            kernel._occ_e, kernel._unocc_e, psi_occ, kernel._psi_unocc
        )
        if tilde is not None:
            kernel._psi_unocc_tilde = tilde
    if verbose:
        print(
            f"  [pw-grid] GIPAW-augmented K  n_inside={gipaw_k_meta['n_inside']}"
            f"  mean|c|={gipaw_k_meta['mean_abs_c']:.4f}",
            flush=True,
        )

    C_v_gto = None
    proj_meta = None
    weight_meta = None
    need_gto = casida_exx_mode == "gto" or bool(weight_virtuals) or bool(particle_ci)
    if need_gto:
        from casidapy.xas.projected_gto import (
            hybrid_mo_ijab,
            project_kernel_virtuals_to_gto,
            select_indices_by_score_capture,
            select_virtuals_by_shell_weight,
            subset_pw_kernel_virtuals,
        )

        abs_atom = 0 if core.fragment_mol.natm == 1 else 0
        C_v_gto, proj_meta = project_kernel_virtuals_to_gto(
            core,
            kernel,
            absorber_atom=abs_atom,
        )
        if verbose:
            print(
                f"  [pw-grid] GTO C_v  completeness mean="
                f"{proj_meta['completeness_mean']:.3f}  "
                f"max={proj_meta['completeness_max']:.3f}  "
                f"(not S-normalized)",
                flush=True,
            )
        if weight_virtuals:
            e_v = np.asarray(kernel._unocc_e, float).ravel()
            energy_mask = None
            if virt_window_ev > 0.0 and e_v.size:
                e_rel = (e_v - float(np.min(e_v))) * HA_TO_EV
                energy_mask = e_rel <= virt_window_ev
                if not np.any(energy_mask):
                    raise RuntimeError(
                        f"virt_window_ev={virt_window_ev} eV leaves no "
                        "conduction bands above CBM."
                    )
            if edge_key in ("K", "1S"):
                from casidapy.xas.dipole import (
                    core_virtual_dipole_matrix_gto_projected,
                    virtual_dipole_scores,
                )

                dip = core_virtual_dipole_matrix_gto_projected(
                    core,
                    kernel._psi_unocc,
                    kernel.grid,
                    origin=R0,
                    projection_method="qe_grid",
                )
                scores = virtual_dipole_scores(dip, reduction="max")
                keep, weight_meta = select_indices_by_score_capture(
                    scores,
                    min_capture=float(virt_capture),
                    min_weight=float(virt_weight_min),
                    n_min=8,
                    mask=energy_mask,
                    label="dipole",
                )
                weight_meta["mode"] = "k_dipole"
                weight_meta["virt_window_ev"] = float(virt_window_ev)
            else:
                from casidapy.xas.chemical_data import valence_particle_nl
                from pyscf.data import elements as _elements

                sym = str(core.fragment_mol.atom_symbol(abs_atom)).strip().capitalize()
                z_abs = int(_elements.charge(sym))
                n_p, l_p = valence_particle_nl(z_abs, edge_key)
                keep, _w, weight_meta = select_virtuals_by_shell_weight(
                    core.fragment_mol,
                    C_v_gto,
                    n=int(n_p),
                    ell=int(l_p),
                    absorber_atom=abs_atom,
                    min_weight=float(virt_weight_min),
                    min_capture=float(virt_capture),
                )
                if energy_mask is not None:
                    keep = keep[energy_mask[keep]]
                    if keep.size == 0:
                        raise RuntimeError(
                            f"virt_window_ev={virt_window_ev} eV removed all "
                            f"{weight_meta.get('shell', 'shell')}-weighted CBs."
                        )
                    weight_meta["n_keep"] = int(keep.size)
                    weight_meta["idx"] = [int(i) for i in keep]
                weight_meta["mode"] = "lm_shell"
                weight_meta["virt_window_ev"] = float(virt_window_ev)
            subset_pw_kernel_virtuals(kernel, keep)
            C_v_gto = np.asarray(C_v_gto, float)[:, keep]
            if verbose:
                top = weight_meta.get("weights_top") or []
                top_s = " ".join(f"{x:.3f}" for x in top[:8])
                tops = top or weight_meta.get("weights_keep") or [0.0]
                wmax = float(weight_meta.get("weight_max") or max(tops))
                n_pool = weight_meta.get("n_virt") or weight_meta.get("n_eligible") or "?"
                print(
                    f"  [pw-grid] TDA virtuals  "
                    f"{weight_meta.get('label') or weight_meta.get('shell')}  "
                    f"mode={weight_meta.get('mode')}  "
                    f"capture={weight_meta['captured']:.3f}  "
                    f"keep={weight_meta['n_keep']}/{n_pool}  "
                    f"window={virt_window_ev:g} eV  "
                    f"max={wmax:.3f}  top={top_s}",
                    flush=True,
                )
        if casida_exx_mode == "gto" and not particle_ci:
            if not bool(exx_meta.get("casida_exx_kernel")):
                raise ValueError(
                    "casida_exx='gto' needs a hybrid xc (or pw_xc) so c_x > 0."
                )
            ijab = hybrid_mo_ijab(
                core.fragment_mol,
                np.asarray(core.mo_coeff, dtype=float),
                C_v_gto,
                hybrid_coeff=float(exx_meta["hybrid_coeff"]),
                rsh_omega=float(exx_meta.get("rsh_omega") or 0.0),
                rsh_alpha=float(exx_meta.get("rsh_alpha") or 0.0),
            )
            n_o, n_v = int(kernel.n_occ), int(kernel.n_unocc)
            if ijab.shape != (n_o, n_o, n_v, n_v):
                raise RuntimeError(
                    f"GTO (ij|ab) shape {ijab.shape} != "
                    f"({n_o},{n_o},{n_v},{n_v}); core vs kernel window mismatch."
                )
            kernel.set_gto_ijab(ijab)
            ijab_f = float(np.linalg.norm(ijab))
            exx_meta["gto_ijab_frobenius"] = ijab_f
            if verbose:
                print(
                    f"  [pw-grid] GTO (ij|ab)  shape={tuple(ijab.shape)}  "
                    f"c_x={exx_meta['hybrid_coeff']:.4f}  "
                    f"||ijab||_F={ijab_f:.4e}",
                    flush=True,
                )
    bind_ae_core_dipole(
        kernel,
        core,
        method="paw_gipaw",
        gipaw_upf=gipaw_upf,
        core_wfc=gipaw_core_wfc,
        mapping=dipole_mapping,
        psi_mode=dipole_psi_mode,
    )
    opts = CasidaOptions(
        n_occ=int(kernel.n_occ),
        n_unocc=int(kernel.n_unocc),
        n_states=int(n_states),
        basis="pw",
        tda=True,
        matrix_free=True,
        xc=pw_xc_alda,
        use_gpu=bool(use_gpu),
        spin_state="singlet",
    )
    kernel_t = None
    particle_meta = None
    if particle_ci:
        from casidapy.xas.projected_gto import run_particle_ci
        from pyscf.data import elements as _elements

        if C_v_gto is None:
            raise RuntimeError("particle_ci needs GTO-projected virtuals.")
        sym = str(core.fragment_mol.atom_symbol(0)).strip().capitalize()
        z_abs = int(_elements.charge(sym))
        ox = oxidation_state
        if ox is None and oxidation_states:
            ox = dict(oxidation_states).get(sym) or dict(oxidation_states).get(
                str(sym).capitalize()
            )
        svd_min = (
            float(particle_ci_min_complete)
            if particle_ci_min_complete is not None
            else 0.2
        )
        k_edge = edge_key in ("K", "1S")
        pci_win = particle_window_ev
        if pci_win is None:
            pci_win = 8.0 if k_edge else 0.0
        pci_win = float(pci_win or 0.0)
        if verbose:
            print(
                f"  [pw-grid] particle-manifold TDA  SVD onto absorber shell  "
                f"slater_scale={float(slater_scale):.3f}  "
                f"min_complete={svd_min:.2f}  "
                f"particle_window={pci_win:g} eV  soc={soc_mode}",
                flush=True,
            )
        res, particle_meta = run_particle_ci(
            core.fragment_mol,
            np.asarray(core.mo_coeff, dtype=float),
            C_v_gto,
            np.asarray(core.energies, dtype=float),
            np.asarray(kernel._unocc_e, dtype=float),
            edge=edge_key,
            absorber_atom=0,
            origin=R0,
            oxidation=ox,
            z=z_abs,
            slater_scale=float(slater_scale),
            min_complete=svd_min,
            particle_window_ev=pci_win,
            soc=(soc_mode == "si") and (not k_edge),
            n_states=int(n_states),
            core=core,
        )
        kernel_t = None
    elif soc_mode == "si":
        if verbose:
            print(
                f"  [pw-grid] SI-SOC L/M  n_core={kernel.n_occ}  "
                "singlet+triplet TDA, hole-only h_oo (2p/3d)",
                flush=True,
            )
        kernel_t = clone_pw_kernel_triplet(kernel, xc=pw_xc_alda)
        soc_res, res_s, res_t = run_cvs_si_soc(
            kernel,
            kernel_t,
            opts,
            n_states=int(n_states),
            hole_only=True,
            mol=core.fragment_mol,
            C_o=np.asarray(core.mo_coeff, dtype=float),
        )
        res = CVSSOCResults(
            omega=soc_res.omega,
            f=soc_res.f,
            mu=soc_res.mu,
            U=soc_res.U,
            singlet_weight=soc_res.singlet_weight,
            triplet_weight=soc_res.triplet_weight,
            singlet=res_s,
            triplet=res_t,
            core=core,
            kernel=kernel,
            kernel_t=kernel_t,
            soc=soc_res,
            metadata={
                **dict(soc_res.meta),
                "soc_mode": "si",
                "edge": edge_key,
                "shell": core.shell,
                "n_core": int(core.energies.size),
                "n_singlet": int(np.asarray(res_s.omega).size),
                "n_triplet": int(np.asarray(res_t.omega).size),
                "n_si": int(soc_res.omega.size),
                "hole_only": True,
            },
        )
    else:
        res = run_cvs_tda(kernel, opts, n_states=int(n_states))

    unocc_e = np.asarray(kernel._unocc_e, float).ravel()
    eps_1s = float(np.min(np.asarray(core.energies, float)))
    omega_ip_ev = (unocc_e - eps_1s) * HA_TO_EV
    virt_tilde = getattr(kernel, "_psi_unocc_tilde", None) or kernel._psi_unocc
    mu_gipaw = core_virtual_dipole_matrix_paw_gipaw(
        core,
        list(virt_tilde),
        grid,
        gipaw_upf,
        core_wfc=gipaw_core_wfc,
        mapping=dipole_mapping,
        psi_mode=dipole_psi_mode,
    )
    mu_grid = core_virtual_dipole_matrix_grid(
        core_fields, list(kernel._psi_unocc), grid
    )
    scale = pseudo_dipole_scale_report(mu_gipaw, omega_ip_ev)
    mag_l = dipole_vector_magnitude(mu_gipaw)[0]
    mag_g = dipole_vector_magnitude(mu_grid)[0]
    n_mu = int(min(mag_l.size, mag_g.size, omega_ip_ev.size))
    ratio = mag_l[:n_mu] / np.maximum(mag_g[:n_mu], 1e-30)
    in_win = omega_ip_ev[:n_mu] <= (float(omega_ip_ev[0]) + 30.0)
    scale["grid_over_gipaw_cv"] = (
        float(np.std(ratio[in_win]) / np.mean(ratio[in_win]))
        if np.any(in_win) and float(np.mean(ratio[in_win])) > 0
        else float("nan")
    )
    w_cvs = np.asarray(res.omega, float) * HA_TO_EV
    f_cvs = np.asarray(res.f, float)
    omega_ip0 = float(omega_ip_ev[0]) if omega_ip_ev.size else float("nan")
    sig = _physical_cvs_edge(w_cvs, f_cvs, omega_ip0)
    edge_ev = float(sig["omega0_ev"])
    diag = {
        "method": "pw_grid_cvs",
        "embed_mode": "ae_host",
        "frozen_core": True,
        "calc_type": (fc_meta or {}).get("calc_type") or calc["calc_type"],
        "gauge_align": (fc_meta or {}).get("gauge_align") or calc["gauge_align"],
        "assume_isolated": (fc_meta or {}).get("assume_isolated", calc["assume_isolated"]),
        "virt_energy_shift_ha": 0.0,
        "eps_1s_ha": eps_1s,
        "n_virt": int(kernel.n_unocc),
        "n_virt_pool": int(n_virt),
        "nao_core": int(core.fragment_mol.nao),
        "scale": scale,
        "sigma_star": {
            **sig,
            "published_delta_ev": 3.5,
        },
        "ae_host_hxc": "rho_tot",
        "core_hxc": "core",
        "xc": str(xc),
        "pw_xc": str(pw_xc),
        "casida_exx": casida_exx_mode,
        "casida_xc": casida_xc_meta,
        "hybrid": ((fc_meta or {}).get("ae_host_becke") or {}).get("hybrid"),
        "vltot_kind": (fc_meta or {}).get("vltot_kind"),
        "vloc_source": (fc_meta or {}).get("vloc_source"),
        "home_peel_image_ha": (fc_meta or {}).get("home_peel_image_ha"),
        "ae_host_rho_tot": (fc_meta or {}).get("ae_host_rho_tot"),
        "ae_host_becke": (fc_meta or {}).get("ae_host_becke"),
        "v_env_quadrature": (fc_meta or {}).get("v_env_quadrature"),
        "dipole_method": "paw_gipaw",
        "gipaw_augment_k": True,
        "gipaw_k": {k: v for k, v in gipaw_k_meta.items() if k != "tilde"},
        "soc_mode": soc_mode,
        "n_core": int(np.asarray(core.energies).size),
        "shell": core.shell,
        "weight_virtuals": bool(weight_virtuals),
        "virt_weight_min": float(virt_weight_min),
        "virt_capture": float(virt_capture),
        "virt_window_ev": float(virt_window_ev),
        "particle_ci": bool(particle_ci),
        "slater_scale": float(slater_scale),
        "particle_window_ev": float(
            ((particle_meta or {}).get("svd") or {}).get("particle_window_ev")
            if particle_meta is not None
            else (particle_window_ev or 0.0)
            or 0.0
        ),
        "gto_projection": proj_meta,
        "gto_ijab_frobenius": exx_meta.get("gto_ijab_frobenius"),
    }
    if weight_meta is not None:
        diag["weight_virtuals_meta"] = dict(weight_meta)
    if exx_meta.get("gto_ijab_frobenius") is not None:
        casida_xc_meta["gto_ijab_frobenius"] = float(exx_meta["gto_ijab_frobenius"])
        diag["casida_xc"] = dict(casida_xc_meta)
    if particle_meta is not None:
        diag["particle_ci_meta"] = {
            k: v for k, v in particle_meta.items() if k not in ("U",)
        }
    if soc_mode == "si" and hasattr(res, "singlet"):
        diag["soc"] = {
            "hole_only": True,
            "n_singlet": int(np.asarray(res.singlet.omega).size),
            "n_triplet": int(np.asarray(res.triplet.omega).size),
            "n_si": int(np.asarray(res.omega).size),
            "weight_virtuals": bool(weight_virtuals),
            "particle_ci": bool(particle_ci),
        }
    if getattr(core, "meta", None) is None:
        core.meta = {}
    core.meta["pw_grid_cvs"] = diag
    if verbose:
        print(
            f"  [pw-grid] ω0={edge_ev:.2f} eV  σ*−edge={diag['sigma_star']['delta_ev']:+.2f} eV"
            f"  ghosts={diag['sigma_star'].get('n_ghost', 0)}"
            f"  |μ| flat={scale.get('flat')}  drift={scale.get('rel_drift_over_window')}",
            flush=True,
        )
    return res, core, kernel, mf, diag
