"""CVS / XAS building blocks: pick cores → inject into a host → run TDA.

Prefer the facade :mod:`casidapy.xas` (``run_xas_gto``, ``run_ae_host_pw_grid_cvs``).

Dipole pruning: :mod:`casidapy.xas.dipole`. Validation / prototypes: :mod:`casidapy.xas.diagnostics`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from casidapy.xas.chemical_data import (
    COVALENT_RADII_ANG as _COVALENT_RADII_ANG,
    DEFAULT_OXIDATION_STATES,
    DEFAULTS,
    EDGE_SHELL as _EDGE_SHELL,
    SHELL_ALIAS as _SHELL_ALIAS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Data containers
# ---------------------------------------------------------------------------
@dataclass
class CoreOrbitals:
    """Core MOs from a fragment AE SCF (optionally SOC-adapted)."""

    energies: np.ndarray
    """Orbital energies (Ha), shape ``(n_core,)``."""
    mo_coeff: np.ndarray
    """MO coefficients in the **fragment** AO basis, shape ``(nao, n_core)``.
    May be complex after orbital SOC."""
    fragment_mol: Any
    """PySCF ``Mole`` used for the fragment SCF."""
    atom_indices: np.ndarray
    """Atom indices in the **parent** geometry that formed the fragment."""
    edge: str = "K"
    shell: str = "1s"
    meta: Dict[str, Any] = field(default_factory=dict)


def parse_soc_mode(soc: Union[bool, str, None]) -> str:
    """Normalize CVS SOC flags to ``\"none\"``, ``\"orbital\"``, or ``\"si\"``.

    * ``none`` — spatial cores, no SOC.
    * ``orbital`` — Breit–Pauli in the core subspace *before* injection
      (j-adapted orbital energies).  Not SOC-coupled TDDFT.
    * ``si`` — spatial cores; singlet + triplet CVS-TDA coupled by
      :func:`~casidapy.utils.soc.solve_soc_si` (final-state L₂/L₃ mixing).
    """
    if soc is None or soc is False:
        return "none"
    if soc is True:
        return "orbital"
    key = str(soc).lower().strip().replace("-", "_")
    aliases = {
        "none": "none",
        "off": "none",
        "false": "none",
        "0": "none",
        "orbital": "orbital",
        "bp": "orbital",
        "jadapted": "orbital",
        "j_adapted": "orbital",
        "true": "orbital",
        "1": "orbital",
        "si": "si",
        "final": "si",
        "state": "si",
        "tddft": "si",
        "state_interaction": "si",
        "si_soc": "si",
        "soc_si": "si",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown soc={soc!r}; expected False/'none', True/'orbital', "
            "or 'si' (final-state singlet–triplet TDDFT coupling)."
        )
    return aliases[key]


@dataclass
class CVSSOCResults:
    """SOC-coupled CVS-TDDFT: SI eigenpairs plus the parent singlet/triplet TDA.

    Duck-types :class:`~casidapy.casida_api.CasidaResults` for
    :mod:`casidapy.xas.spectrum` (``omega``, ``f``, ``d_mode``).
    """

    omega: np.ndarray
    f: np.ndarray
    mu: np.ndarray
    U: np.ndarray
    singlet_weight: np.ndarray
    triplet_weight: np.ndarray
    singlet: Any
    triplet: Any
    core: CoreOrbitals
    kernel: Any
    kernel_t: Any
    soc: Any
    Z: Optional[np.ndarray] = None
    mu_transition: Optional[np.ndarray] = None
    rho_transition: Optional[np.ndarray] = None
    xpy: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def d_mode(self) -> np.ndarray:
        return np.asarray(self.mu, dtype=float).T

    @property
    def meta(self) -> Dict[str, Any]:
        return self.metadata


@dataclass
class FragmentSpec:
    """Geometry cut for an AE core SCF (auto or user-defined).

    Pass to :func:`extract_fragment_core` via ``fragment=...``, or use
    ``atom_indices`` / ``charge`` / ``spin`` directly.
    """

    atom_indices: np.ndarray
    """Parent atom indices in the fragment."""
    charge: int = 0
    """SCF charge (usually formal oxidation sum)."""
    spin: int = 0
    """``2S`` as in PySCF."""
    edge_atom_indices: Optional[np.ndarray] = None
    """Parent indices whose cores are imported (default: ``None`` → library default)."""
    mode: str = "user"
    """``\"user\"``, ``\"neutral_first_shell\"``, or ``\"radius\"``."""
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def user(
        cls,
        atom_indices: Sequence[int],
        *,
        charge: int = 0,
        spin: int = 0,
        edge_atom_indices: Optional[Sequence[int]] = None,
        **meta: Any,
    ) -> "FragmentSpec":
        """Explicit user fragment (indices into the parent geometry)."""
        edge = (
            None
            if edge_atom_indices is None
            else np.asarray(edge_atom_indices, dtype=int)
        )
        return cls(
            atom_indices=np.asarray(atom_indices, dtype=int),
            charge=int(charge),
            spin=int(spin),
            edge_atom_indices=edge,
            mode="user",
            meta=dict(meta),
        )


# ---------------------------------------------------------------------------
# 2. Fragment geometry / oxidation
# ---------------------------------------------------------------------------

def _as_positions_symbols(mol_or_atoms):
    """Return (positions_A Å, symbols, parent_handle)."""
    # ASE Atoms
    if hasattr(mol_or_atoms, "get_positions") and hasattr(mol_or_atoms, "get_chemical_symbols"):
        pos = np.asarray(mol_or_atoms.get_positions(), dtype=float)
        sym = list(mol_or_atoms.get_chemical_symbols())
        return pos, sym, mol_or_atoms
    # PySCF Mole
    if hasattr(mol_or_atoms, "atom_coords") and hasattr(mol_or_atoms, "atom_symbol"):
        # PySCF stores Bohr by default in atom_coords()
        pos = np.asarray(mol_or_atoms.atom_coords(unit="Angstrom"), dtype=float)
        natm = mol_or_atoms.natm
        sym = [mol_or_atoms.atom_symbol(i) for i in range(natm)]
        return pos, sym, mol_or_atoms
    raise TypeError(
        "Expected ASE Atoms or PySCF Mole, got "
        f"{type(mol_or_atoms)}"
    )


def select_atoms_in_radius(
    mol_or_atoms,
    center: Union[int, Sequence[float]],
    radius_ang: float,
    *,
    elements: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Return parent atom indices within ``radius_ang`` of ``center``.

    Parameters
    ----------
    center
        Parent atom index, or Cartesian ``(x,y,z)`` in Angstrom.
    elements
        If given, keep only those element symbols (e.g. ``(\"Ag\",\"Cu\")``).
    """
    pos, sym, parent = _as_positions_symbols(mol_or_atoms)
    r_cut = float(radius_ang)
    use_mic = (
        isinstance(center, (int, np.integer))
        and hasattr(parent, "get_distance")
        and hasattr(parent, "pbc")
        and any(parent.pbc)
    )
    if use_mic:
        edge = int(center)
        idx = np.array(
            [
                i
                for i in range(len(sym))
                if float(parent.get_distance(edge, i, mic=True)) <= r_cut
            ],
            dtype=int,
        )
    else:
        if isinstance(center, (int, np.integer)):
            c = pos[int(center)]
        else:
            c = np.asarray(center, dtype=float).reshape(3)
        d = np.linalg.norm(pos - c[None, :], axis=1)
        mask = d <= r_cut
        if elements is not None:
            allowed = {e.strip().capitalize() for e in elements}
            mask &= np.array([s.capitalize() in allowed for s in sym], dtype=bool)
        idx = np.where(mask)[0]
    if elements is not None and use_mic:
        allowed = {e.strip().capitalize() for e in elements}
        idx = np.array(
            [i for i in idx if sym[int(i)].capitalize() in allowed], dtype=int
        )
    if idx.size == 0:
        raise ValueError(
            f"No atoms within {radius_ang} Å of center={center}"
            + (f" with elements={elements}" if elements else "")
        )
    return idx.astype(int)


def formal_oxidation_charge(
    symbols: Sequence[str],
    oxidation_states: Optional[Dict[str, int]] = None,
) -> int:
    """Sum of formal oxidation numbers for a list of element symbols.

    Unknown elements contribute 0 unless listed in ``oxidation_states``.
    """
    table = dict(DEFAULT_OXIDATION_STATES)
    if oxidation_states:
        table.update({str(k).strip().capitalize(): int(v) for k, v in oxidation_states.items()})
    q = 0
    missing = []
    for s in symbols:
        key = str(s).strip().capitalize()
        if key not in table:
            missing.append(key)
            continue
        q += int(table[key])
    if missing:
        uniq = sorted(set(missing))
        raise ValueError(
            f"No formal oxidation state for {uniq}; pass oxidation_states={{...}} "
            f"(defaults cover common main-group / TM oxides)."
        )
    return int(q)


def resolve_oxidation_state(
    symbol: str,
    oxidation_state: Optional[int] = None,
    oxidation_states: Optional[Dict[str, int]] = None,
) -> int:
    """Formal oxidation number for one atom (e.g. O → −2, Ti → +4).

    Override with ``oxidation_state`` or a per-element ``oxidation_states`` map.
    """
    if oxidation_state is not None:
        return int(oxidation_state)
    table = dict(DEFAULT_OXIDATION_STATES)
    if oxidation_states:
        table.update({str(k).strip().capitalize(): int(v) for k, v in oxidation_states.items()})
    key = str(symbol).strip().capitalize()
    if key not in table:
        raise ValueError(
            f"No default oxidation state for {key!r}; pass oxidation_state=... "
            f"or oxidation_states={{'{key}': ...}}."
        )
    return int(table[key])


def _covalent_radii_ang(symbols: Sequence[str]) -> np.ndarray:
    try:
        from ase.data import atomic_numbers, covalent_radii

        out = []
        for s in symbols:
            z = atomic_numbers[str(s).strip().capitalize()]
            out.append(float(covalent_radii[z]))
        return np.asarray(out, dtype=float)
    except Exception:
        out = []
        for s in symbols:
            key = str(s).strip().capitalize()
            if key not in _COVALENT_RADII_ANG:
                raise ValueError(
                    f"No covalent radius for {key!r}; install ASE or extend "
                    "chemical_data.json covalent_radii_ang."
                )
            out.append(_COVALENT_RADII_ANG[key])
        return np.asarray(out, dtype=float)


def _bond_neighbors(
    pos: np.ndarray,
    symbols: Sequence[str],
    *,
    bond_factor: float = DEFAULTS["bond_factor"],
) -> List[List[int]]:
    """Undirected bonded-neighbor lists from covalent-radii cutoffs."""
    n = len(symbols)
    radii = _covalent_radii_ang(symbols)
    neigh: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            cutoff = float(bond_factor) * (radii[i] + radii[j])
            if np.linalg.norm(pos[i] - pos[j]) <= cutoff:
                neigh[i].append(j)
                neigh[j].append(i)
    return neigh


def select_neutral_first_shell(
    mol_or_atoms,
    center: int,
    *,
    bond_factor: float = DEFAULTS["bond_factor"],
    max_atoms: int = DEFAULTS["max_fragment_atoms"],
    target_charge: int = 0,
    oxidation_states: Optional[Dict[str, int]] = None,
    complete_ligands: bool = True,
) -> FragmentSpec:
    """Build a near-neutral fragment: edge atom + first shell (+ ligands).

    1. Take the edge atom and atoms bonded to it (covalent-radii first shell).
    2. If ``complete_ligands`` and the formal charge is not ``target_charge``,
       greedily add atoms bonded to the current cut that reduce
       ``|formal − target|`` (e.g. add O around Ti in TiO₂ until q=0).

    The returned ``charge`` is the formal oxidation sum of the selected atoms
    (use as the PySCF SCF charge for an ionic closed-shell picture).
    """
    pos, sym, _ = _as_positions_symbols(mol_or_atoms)
    c = int(center)
    if c < 0 or c >= len(sym):
        raise ValueError(f"center={center} out of range for natm={len(sym)}")

    neigh = _bond_neighbors(pos, sym, bond_factor=bond_factor)
    first = sorted({c, *neigh[c]})
    if len(first) < 2 and len(sym) > 1:
        # Loosen once if the first shell is empty (soft geometry / radii).
        neigh = _bond_neighbors(
            pos, sym, bond_factor=float(bond_factor) * DEFAULTS["bond_factor_ligand_growth"]
        )
        first = sorted({c, *neigh[c]})

    chosen: set = set(first)

    def _q(idxs) -> int:
        return formal_oxidation_charge(
            [sym[i] for i in idxs], oxidation_states=oxidation_states
        )

    if complete_ligands:
        while len(chosen) < int(max_atoms):
            q_now = _q(chosen)
            if q_now == int(target_charge):
                break
            cands = []
            for i in list(chosen):
                for j in neigh[i]:
                    if j not in chosen:
                        cands.append(j)
            if not cands:
                break
            best = None
            q_cur_dev = abs(q_now - int(target_charge))
            for j in set(cands):
                q_new = _q(chosen | {j})
                dev = abs(q_new - int(target_charge))
                if dev > q_cur_dev:
                    continue
                d = float(np.linalg.norm(pos[j] - pos[c]))
                key = (dev, d, j)
                if best is None or key < best[0]:
                    best = (key, j)
            if best is None:
                break
            chosen.add(best[1])

    idx = np.asarray(sorted(chosen), dtype=int)
    d = np.linalg.norm(pos[idx] - pos[c], axis=1)
    idx = idx[np.argsort(d)]
    formal = _q(idx)
    return FragmentSpec(
        atom_indices=idx,
        charge=int(formal),
        spin=0,
        edge_atom_indices=np.asarray([c], dtype=int),
        mode="neutral_first_shell",
        meta={
            "center": c,
            "first_shell": list(first),
            "formal_charge": int(formal),
            "target_charge": int(target_charge),
            "neutral": formal == int(target_charge),
            "bond_factor": float(bond_factor),
            "max_atoms": int(max_atoms),
            "complete_ligands": bool(complete_ligands),
            "symbols": [sym[i] for i in idx],
            "n_atoms": int(idx.size),
        },
    )


def select_xas_fragment(
    mol_or_atoms,
    edge_atom: int,
    *,
    mode: str = "neutral_first_shell",
    atom_indices: Optional[Sequence[int]] = None,
    charge: Optional[int] = None,
    spin: int = 0,
    radius_ang: float = DEFAULTS["fragment_radius_ang"],
    bond_factor: float = DEFAULTS["bond_factor"],
    max_atoms: int = DEFAULTS["max_fragment_atoms"],
    target_charge: int = 0,
    oxidation_states: Optional[Dict[str, int]] = None,
    complete_ligands: bool = True,
    elements: Optional[Sequence[str]] = None,
) -> FragmentSpec:
    """Choose a fragment for core XAS (auto neutral shell, radius, or user).

    Parameters
    ----------
    mode
        - ``\"neutral_first_shell\"`` (default): first shell + ligands → formal
          charge near ``target_charge`` (usually 0).
        - ``\"radius\"``: atoms within ``radius_ang`` of the edge atom.
        - ``\"user\"``: require ``atom_indices`` (or pass them with any mode).
    atom_indices
        If given, always treated as a user-defined fragment (overrides auto).
    charge
        SCF charge override. Default: formal oxidation sum of the cut.
    """
    pos, sym, _ = _as_positions_symbols(mol_or_atoms)
    edge = int(edge_atom)

    if atom_indices is not None or str(mode).lower().strip() == "user":
        if atom_indices is None:
            raise ValueError("mode='user' requires atom_indices=[...]")
        idx = np.asarray([int(i) for i in atom_indices], dtype=int)
        if edge not in set(idx.tolist()):
            raise ValueError(
                f"edge_atom={edge} must be included in user atom_indices={idx.tolist()}"
            )
        formal = formal_oxidation_charge(
            [sym[i] for i in idx], oxidation_states=oxidation_states
        )
        chg = int(formal if charge is None else charge)
        return FragmentSpec(
            atom_indices=idx,
            charge=chg,
            spin=int(spin),
            edge_atom_indices=np.asarray([edge], dtype=int),
            mode="user",
            meta={
                "formal_charge": int(formal),
                "symbols": [sym[i] for i in idx],
                "n_atoms": int(idx.size),
            },
        )

    mode_l = str(mode).lower().strip()
    if mode_l == "radius":
        idx = select_atoms_in_radius(
            mol_or_atoms, center=edge, radius_ang=radius_ang, elements=elements
        )
        # Keep edge first by distance sort
        d = np.linalg.norm(pos[idx] - pos[edge], axis=1)
        idx = idx[np.argsort(d)]
        if int(max_atoms) > 0 and idx.size > int(max_atoms):
            idx = idx[: int(max_atoms)]
            if edge not in set(idx.tolist()):
                idx = np.unique(np.concatenate([[edge], idx]))[: int(max_atoms)]
        formal = formal_oxidation_charge(
            [sym[i] for i in idx], oxidation_states=oxidation_states
        )
        chg = int(formal if charge is None else charge)
        return FragmentSpec(
            atom_indices=idx.astype(int),
            charge=chg,
            spin=int(spin),
            edge_atom_indices=np.asarray([edge], dtype=int),
            mode="radius",
            meta={
                "radius_ang": float(radius_ang),
                "formal_charge": int(formal),
                "symbols": [sym[i] for i in idx],
                "n_atoms": int(idx.size),
            },
        )

    if mode_l in ("neutral_first_shell", "neutral", "first_shell"):
        frag = select_neutral_first_shell(
            mol_or_atoms,
            edge,
            bond_factor=bond_factor,
            max_atoms=max_atoms,
            target_charge=target_charge,
            oxidation_states=oxidation_states,
            complete_ligands=complete_ligands,
        )
        frag.spin = int(spin)
        if charge is not None:
            frag.charge = int(charge)
        return frag

    raise ValueError(
        f"Unknown fragment mode={mode!r}; use 'neutral_first_shell', 'radius', or 'user'"
    )


def _resolve_shell(edge: str, shell: Optional[str]) -> Tuple[str, int, int]:
    edge_u = str(edge).strip().upper()
    if shell is not None:
        key = str(shell).strip().lower()
        if key not in _SHELL_ALIAS:
            raise ValueError(f"Unknown shell {shell!r}; known {list(_SHELL_ALIAS)}")
        n, ell = _SHELL_ALIAS[key]
        return key, n, ell
    if edge_u not in _EDGE_SHELL:
        raise ValueError(f"edge must be K/L/M, got {edge!r}")
    n, ell = _EDGE_SHELL[edge_u]
    label = {0: "s", 1: "p", 2: "d", 3: "f"}[ell]
    return f"{n}{label}", n, ell


def _build_fragment_mole(
    parent_mol,
    atom_indices: Sequence[int],
    *,
    basis: str,
    charge: int = 0,
    spin: int = 0,
    verbose: int = 0,
):
    from pyscf import gto

    pos, sym, parent = _as_positions_symbols(parent_mol)
    atom_indices = [int(i) for i in atom_indices]
    atom_str = []
    for i in atom_indices:
        atom_str.append([sym[i], pos[i].tolist()])

    # Prefer copying basis from parent for selected atoms when parent is Mole
    basis_spec: Any = basis
    if hasattr(parent, "basis") and basis is None:
        basis_spec = parent.basis

    frag = gto.M(
        atom=atom_str,
        basis=basis_spec if basis_spec is not None else "sto-3g",
        charge=int(charge),
        spin=int(spin),
        unit="Angstrom",
        verbose=verbose,
    )
    return frag


def _ao_shell_mask(mol, n: int, ell: int) -> np.ndarray:
    """Boolean mask over AOs whose label matches principal n and angular l."""
    labels = mol.ao_labels()
    # labels like '0 Ag 1s' or '0 Ag 2px'
    mask = np.zeros(mol.nao, dtype=bool)
    lchar = {0: "s", 1: "p", 2: "d", 3: "f"}[ell]
    needle = f"{n}{lchar}"
    for i, lab in enumerate(labels):
        # lab may be str or tuple
        s = " ".join(lab) if isinstance(lab, (tuple, list)) else str(lab)
        # match '1s', '2p', '2px', '3dxy', ...
        parts = s.replace("_", " ").split()
        for p in parts:
            p_low = p.lower()
            if p_low.startswith(needle):
                mask[i] = True
                break
    return mask


def _ao_mask_for_atoms(mol, atom_ids: Sequence[int], base_mask: np.ndarray) -> np.ndarray:
    """Restrict an AO mask to AOs belonging to the given fragment atom indices."""
    atom_ids = {int(i) for i in atom_ids}
    out = np.zeros(mol.nao, dtype=bool)
    labels = mol.ao_labels()
    for i, lab in enumerate(labels):
        if not base_mask[i]:
            continue
        s = " ".join(lab) if isinstance(lab, (tuple, list)) else str(lab)
        try:
            iatm = int(str(s).split()[0])
        except Exception:
            continue
        if iatm in atom_ids:
            out[i] = True
    return out


def _mo_mulliken_on_aos(
    mo_c: np.ndarray,
    S: np.ndarray,
    ao_mask: np.ndarray,
) -> np.ndarray:
    """Mulliken population of each MO column on the selected AO subset.

    ``pop_i = Σ_{μ ∈ mask} C_{μi} (S C)_{μi}``.  For near-degenerate same-
    element cores (e.g. anthracene C 1s), this is far more reliable than
    ranking by raw ``Σ|C_μ|²`` and then taking the lowest eigenvalue.
    """
    C = np.asarray(mo_c, dtype=float)
    if C.ndim == 1:
        C = C.reshape(-1, 1)
    SC = np.asarray(S, dtype=float) @ C
    mask = np.asarray(ao_mask, dtype=bool)
    return np.sum(C[mask, :] * SC[mask, :], axis=0)


def _core_atom_sets(
    mol, ell: int, edge_frag_atoms: Optional[Sequence[int]] = None
) -> Tuple[List[int], List[int]]:
    """``(edge_ids, manifold_ids)``: requested absorbers and their same-Z shell block.

    The manifold is the full set of same-element atoms, because their core MOs
    are near-degenerate and mix with each other even when only one absorber is
    requested.
    """
    natm = mol.natm
    if edge_frag_atoms is not None:
        edge_ids = [int(i) for i in edge_frag_atoms]
        for i in edge_ids:
            if i < 0 or i >= natm:
                raise ValueError(f"edge fragment atom {i} out of range for natm={natm}")
        zs = {int(mol.atom_charge(i)) for i in edge_ids}
        manifold_ids = [i for i in range(natm) if int(mol.atom_charge(i)) in zs]
        if ell == 0:
            manifold_ids = [
                i for i in manifold_ids if int(mol.atom_charge(i)) >= 3
            ] or manifold_ids
    elif ell == 0:
        edge_ids = [i for i in range(natm) if mol.atom_charge(i) >= 3] or list(
            range(natm)
        )
        manifold_ids = list(edge_ids)
    else:
        edge_ids = list(range(natm))
        manifold_ids = list(edge_ids)
    return edge_ids, manifold_ids


def _mo_fock_expectation(mo_c: np.ndarray, mo_e: np.ndarray, S: np.ndarray, c: np.ndarray) -> float:
    """``<phi|F|phi>`` for an orbital that is not a Fock eigenvector.

    Uses the canonical spectrum, ``F = S C diag(e) C^T S``, so no integrals are
    rebuilt.  Exact for any ``c`` spanned by the MO set.
    """
    p = np.asarray(mo_c, float).T @ (np.asarray(S, float) @ np.asarray(c, float).ravel())
    return float(np.sum(np.asarray(mo_e, float).ravel() * p**2))


def _lowdin_orthonormalize(C: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Symmetric orthonormalization of AO-coefficient columns in the S metric."""
    C = np.asarray(C, float)
    M = C.T @ np.asarray(S, float) @ C
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, 1e-14)
    return C @ (V @ np.diag(w**-0.5) @ V.T)


def _localize_core_mos(
    mf,
    *,
    n: int,
    ell: int,
    n_per_atom: Optional[int] = None,
    edge_frag_atoms: Optional[Sequence[int]] = None,
    method: str = "proj",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Absorber-localized core MOs, for symmetry-equivalent absorbers.

    Canonical core MOs of equivalent atoms are fixed symmetry combinations: in
    anthracene each C 1s MO carries only ~0.5 e on the requested carbon and
    ~0.45 e on its partner.  Such an orbital cannot be compared with — nor used
    as the ``<c|r|v>`` reference for — a single-site (embedded or fragment)
    core, which caps the achievable 1s overlap near ``sqrt(0.5)``.

    This rotates the near-degenerate same-element shell manifold onto the
    individual absorbers:

    ``method='proj'`` (default)
        Maximize the projection onto each atom's ``nl`` AOs (SVD; deterministic).
    ``method='boys'``
        Boys localization of the manifold, then per-atom Mulliken assignment.

    Localized orbitals are not Fock eigenvectors, so energies are returned as
    ``<phi|F|phi>``.  The manifold is near-degenerate, so this stays within
    ~1e-4 Ha of the canonical eigenvalue.

    Returns ``(energies, mo_coeff_columns, selection_meta)``.
    """
    mol = mf.mol
    mo_e = np.asarray(mf.mo_energy, dtype=float).ravel()
    mo_c = np.asarray(mf.mo_coeff, dtype=float)
    mo_occ = np.asarray(mf.mo_occ, dtype=float).ravel()
    occ_idx = np.where(mo_occ > 1e-6)[0]
    if occ_idx.size == 0:
        raise RuntimeError("Fragment SCF has no occupied orbitals.")

    mode = str(method).lower().strip()
    if mode in ("projection", "svd", "atom"):
        mode = "proj"
    if mode not in ("proj", "boys"):
        raise ValueError(f"core_localize must be 'proj' or 'boys' (got {method!r})")

    shell_ao_all = _ao_shell_mask(mol, n, ell)
    if not np.any(shell_ao_all):
        raise RuntimeError(
            f"No AOs matching n={n}, l={ell} in basis; check AE basis / edge."
        )
    edge_ids, manifold_ids = _core_atom_sets(mol, ell, edge_frag_atoms)
    per_atom = max(1, int(n_per_atom) if n_per_atom is not None else (2 * ell + 1))

    S = np.asarray(mol.intor("int1e_ovlp"), dtype=float)
    C_occ = mo_c[:, occ_idx]
    shell_man = _ao_mask_for_atoms(mol, manifold_ids, shell_ao_all)
    if not np.any(shell_man):
        raise RuntimeError(
            f"No n={n}, l={ell} AOs on manifold atoms {manifold_ids}; check basis."
        )

    # The element's shell block: occupied MOs with the largest population on
    # the manifold's nl AOs.
    pops_man = _mo_mulliken_on_aos(C_occ, S, shell_man)
    n_man = int(min(per_atom * len(manifold_ids), occ_idx.size))
    man_pos = np.argsort(-pops_man)[:n_man]
    C_man = C_occ[:, man_pos]
    man_pop_min = float(np.min(pops_man[man_pos]))

    localizer = mode
    if mode == "boys":
        try:
            from pyscf import lo

            loc = lo.Boys(mol, C_man)
            loc.max_cycle = 200
            C_loc = np.asarray(loc.kernel(), dtype=float)
        except Exception:  # fall back to the deterministic projection
            localizer = "proj_fallback"
            C_loc = None
    else:
        C_loc = None

    if C_loc is None:
        cols = []
        for ia in manifold_ids:
            idx = np.where(_ao_mask_for_atoms(mol, [ia], shell_ao_all))[0]
            if idx.size == 0:
                continue
            # Manifold-space vectors maximizing weight on this atom's nl AOs.
            A = C_man.T @ S[:, idx]
            u, _, _ = np.linalg.svd(A, full_matrices=False)
            cols.append(C_man @ u[:, : min(per_atom, u.shape[1])])
        if not cols:
            raise RuntimeError(
                f"No n={n}, l={ell} AOs on manifold atoms {manifold_ids}; check basis."
            )
        C_loc = _lowdin_orthonormalize(np.concatenate(cols, axis=1), S)

    # Per-absorber assignment, as in _select_core_mos but on localized columns.
    remaining = set(range(C_loc.shape[1]))
    chosen: List[int] = []
    per_atom_pops: List[Dict[str, Any]] = []
    expect = int(min(per_atom * len(edge_ids), C_loc.shape[1]))
    for ia in edge_ids:
        atom_shell = _ao_mask_for_atoms(mol, [ia], shell_ao_all)
        if not np.any(atom_shell):
            raise RuntimeError(f"No n={n}, l={ell} AOs on edge atom {ia}; check basis.")
        pops = _mo_mulliken_on_aos(C_loc, S, atom_shell)
        picks, pick_pops = [], []
        for _ in range(per_atom):
            if not remaining:
                break
            best = max(remaining, key=lambda j: float(pops[j]))
            remaining.remove(best)
            picks.append(best)
            pick_pops.append(float(pops[best]))
            chosen.append(best)
        per_atom_pops.append(
            {"atom": int(ia), "mulliken_pops": pick_pops, "columns": picks}
        )
        if len(chosen) >= expect:
            break

    chosen = chosen[:expect]
    energies = np.asarray(
        [_mo_fock_expectation(mo_c, mo_e, S, C_loc[:, j]) for j in chosen], dtype=float
    )
    order = np.argsort(energies)
    chosen = [chosen[i] for i in order]
    energies = energies[order]
    coeff = C_loc[:, chosen].copy()

    meta = {
        "method": f"localized_{localizer}",
        "localize": mode,
        "n": int(n),
        "ell": int(ell),
        "edge_atoms": [int(i) for i in edge_ids],
        "manifold_atoms": [int(i) for i in manifold_ids],
        "n_per_atom": int(per_atom),
        "n_manifold": int(C_loc.shape[1]),
        "manifold_pop_min": man_pop_min,
        "per_atom": per_atom_pops,
        "absorber_pop": float(per_atom_pops[0]["mulliken_pops"][0])
        if per_atom_pops and per_atom_pops[0]["mulliken_pops"]
        else float("nan"),
    }
    return energies, coeff, meta


def _select_core_mos(
    mf,
    *,
    n: int,
    ell: int,
    n_per_atom: Optional[int] = None,
    edge_frag_atoms: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Pick occupied MOs localized on the target nl shell.

    For each absorber, choose the unused occupied MO with the largest
    S-metric Mulliken population on that atom's ``nl`` AOs.  Candidates are
    restricted to the deepest shell-weighted occupied block so valence MOs
    are not selected.

    When several same-element cores are symmetry-delocalized with equal
    absorber populations, any of those MOs is acceptable (degenerate).
    The failure mode this fixes is *mis-assignment* among inequivalent or
    weakly mixed cores (e.g. anthracene C6 vs C8), where energy-sorting the
    ``|c|²`` shortlist picks a neighbor's 1s.

    Parameters
    ----------
    edge_frag_atoms
        Fragment-local atom indices that contribute core orbitals. If ``None``,
        defaults to all heavy atoms (Z≥3) for s-shells, else all atoms.

    Returns ``(energies, mo_coeff_columns, selection_meta)``.
    """
    mol = mf.mol
    mo_e = np.asarray(mf.mo_energy, dtype=float).ravel()
    mo_c = np.asarray(mf.mo_coeff, dtype=float)
    mo_occ = np.asarray(mf.mo_occ, dtype=float).ravel()
    occ_idx = np.where(mo_occ > 1e-6)[0]
    if occ_idx.size == 0:
        raise RuntimeError("Fragment SCF has no occupied orbitals.")

    shell_ao_all = _ao_shell_mask(mol, n, ell)
    if not np.any(shell_ao_all):
        raise RuntimeError(
            f"No AOs matching n={n}, l={ell} in basis; check AE basis / edge."
        )

    # Include the full same-Z shell block in the candidate pool even when only
    # one absorber is requested.
    edge_ids, manifold_ids = _core_atom_sets(mol, ell, edge_frag_atoms)

    per_atom = int(n_per_atom) if n_per_atom is not None else (2 * ell + 1)
    per_atom = max(1, per_atom)
    n_man = len(manifold_ids)
    expect_man = min(per_atom * n_man, occ_idx.size)
    expect = min(per_atom * len(edge_ids), occ_idx.size)

    S = np.asarray(mol.intor("int1e_ovlp"), dtype=float)
    C_occ = mo_c[:, occ_idx]
    shell_man = _ao_mask_for_atoms(mol, manifold_ids, shell_ao_all)
    if not np.any(shell_man):
        raise RuntimeError(
            f"No n={n}, l={ell} AOs on manifold atoms {manifold_ids}; check basis."
        )

    # Candidate pool: shell-weighted occupied MOs, then the deepest expect_man.
    w_man = _mo_mulliken_on_aos(C_occ, S, shell_man)
    pool_n = min(occ_idx.size, max(expect_man * 2, expect_man))
    pool = np.argsort(-w_man)[:pool_n]
    pool = pool[np.argsort(mo_e[occ_idx[pool]])]
    cand_pos = np.asarray(pool[:expect_man], dtype=int)
    if cand_pos.size < expect_man:
        have = set(int(j) for j in cand_pos)
        for j in np.argsort(mo_e[occ_idx]):
            if int(j) in have:
                continue
            cand_pos = np.append(cand_pos, int(j))
            if cand_pos.size >= expect_man:
                break

    C_cand = C_occ[:, cand_pos]
    e_cand = mo_e[occ_idx[cand_pos]]
    remaining = set(range(cand_pos.size))
    chosen_cols: List[int] = []
    per_atom_pops: List[Dict[str, Any]] = []

    for ia in edge_ids:
        atom_shell = _ao_mask_for_atoms(mol, [ia], shell_ao_all)
        if not np.any(atom_shell):
            raise RuntimeError(
                f"No n={n}, l={ell} AOs on edge atom {ia}; check basis."
            )
        pops = _mo_mulliken_on_aos(C_cand, S, atom_shell)
        atom_picks: List[int] = []
        atom_pick_pops: List[float] = []
        for _ in range(per_atom):
            if not remaining:
                break
            best = max(remaining, key=lambda j: float(pops[j]))
            best_pop = float(pops[best])
            remaining.remove(best)
            atom_picks.append(best)
            atom_pick_pops.append(best_pop)
            chosen_cols.append(best)
        per_atom_pops.append(
            {
                "atom": int(ia),
                "mo_indices": [int(occ_idx[cand_pos[j]]) for j in atom_picks],
                "mulliken_pops": atom_pick_pops,
            }
        )
        if len(chosen_cols) >= expect:
            break

    if len(chosen_cols) < expect:
        leftovers = sorted(remaining, key=lambda j: float(e_cand[j]))
        for j in leftovers:
            if len(chosen_cols) >= expect:
                break
            chosen_cols.append(int(j))
            remaining.discard(j)

    chosen_cols = chosen_cols[:expect]
    order = np.argsort(e_cand[chosen_cols])
    chosen_cols = [chosen_cols[i] for i in order]
    energies = np.asarray([e_cand[j] for j in chosen_cols], dtype=float)
    coeff = C_cand[:, chosen_cols].copy()

    meta = {
        "method": "mulliken_max_per_atom",
        "n": int(n),
        "ell": int(ell),
        "edge_atoms": [int(i) for i in edge_ids],
        "manifold_atoms": [int(i) for i in manifold_ids],
        "n_per_atom": int(per_atom),
        "per_atom": per_atom_pops,
        "n_candidates": int(cand_pos.size),
        "chosen_mo_indices": [int(occ_idx[cand_pos[j]]) for j in chosen_cols],
    }
    return energies, coeff, meta



# ---------------------------------------------------------------------------
# 3. Core MO selection from an SCF
# ---------------------------------------------------------------------------

def core_from_mf(
    mf,
    *,
    edge: str = "K",
    shell: Optional[str] = None,
    edge_atom_indices: Optional[Sequence[int]] = None,
    n_core_per_atom: Optional[int] = None,
    soc: Union[bool, str] = False,
    atom_indices: Optional[Sequence[int]] = None,
    core_localize: Optional[str] = None,
    verbose: bool = False,
) -> CoreOrbitals:
    """Select core MOs from a **user-supplied, converged** PySCF mean-field.

    The caller owns the SCF (CPU PySCF, gpu4pyscf, DF, X2C, custom guess, …).
    This only picks the edge-localized occupied MOs and optionally applies
    orbital SOC.  ``soc="si"`` keeps spatial cores (SI-SOC is applied after
    singlet+triplet TDA, not here).

    Parameters
    ----------
    mf
        Converged RKS/RHF (or UKS alpha channel). Must expose ``mol``,
        ``mo_energy``, ``mo_coeff``, ``mo_occ``, and ``converged``.
    edge_atom_indices
        Atom indices **in ``mf.mol``** whose core orbitals are imported.
        Default: heavy atoms (Z≥3) for K-edge; all atoms for L/M.
    atom_indices
        Optional parent-geometry index map stored on the result (defaults to
        ``range(mol.natm)``).
    core_localize
        ``None`` (default) keeps the canonical MOs.  ``'proj'`` / ``'boys'``
        first rotate the same-element shell manifold onto individual absorbers
        (see :func:`_localize_core_mos`).  Required whenever the absorber is
        symmetry-equivalent to other atoms — otherwise the canonical core is a
        multi-centre combination and both the 1s overlap and ``<c|r|v>`` are
        referenced to the wrong initial state.
    """
    if not getattr(mf, "converged", False):
        raise RuntimeError("core_from_mf requires a converged mean-field (mf.converged).")

    mol = mf.mol
    shell_label, n, ell = _resolve_shell(edge, shell)

    # Host arrays (gpu4pyscf may leave cupy on the mf)
    def _host(a):
        try:
            import cupy as cp

            if isinstance(a, cp.ndarray):
                return cp.asnumpy(a)
        except Exception:
            pass
        return np.asarray(a)

    mo_e = _host(mf.mo_energy)
    mo_c = _host(mf.mo_coeff)
    mo_occ = _host(mf.mo_occ)

    class _HostMF:
        pass

    h = _HostMF()
    h.mol = mol
    if mo_c.ndim == 3:
        h.mo_energy = np.asarray(mo_e[0], dtype=float)
        h.mo_coeff = np.asarray(mo_c[0], dtype=float)
        h.mo_occ = np.asarray(mo_occ[0], dtype=float)
    else:
        h.mo_energy = np.asarray(mo_e, dtype=float).ravel()
        h.mo_coeff = np.asarray(mo_c, dtype=float)
        h.mo_occ = np.asarray(mo_occ, dtype=float).ravel()

    edge_frag: Optional[List[int]] = None
    edge_parent: Optional[List[int]] = None
    if edge_atom_indices is not None:
        edge_frag = [int(i) for i in edge_atom_indices]
        edge_parent = list(edge_frag)
        for i in edge_frag:
            if i < 0 or i >= mol.natm:
                raise ValueError(
                    f"edge_atom_indices entry {i} out of range for natm={mol.natm}"
                )

    localize = str(core_localize or "none").lower().strip()
    if localize in ("none", "off", "", "false", "canonical"):
        energies, coeff, sel_meta = _select_core_mos(
            h,
            n=n,
            ell=ell,
            n_per_atom=n_core_per_atom,
            edge_frag_atoms=edge_frag,
        )
    else:
        energies, coeff, sel_meta = _localize_core_mos(
            h,
            n=n,
            ell=ell,
            n_per_atom=n_core_per_atom,
            edge_frag_atoms=edge_frag,
            method=localize,
        )
        if verbose:
            print(
                f"core_from_mf: localized ({sel_meta['method']}) "
                f"absorber pop={sel_meta['absorber_pop']:.4f}  "
                f"eps={energies[0]:.6f} Ha",
                flush=True,
            )

    soc_mode = parse_soc_mode(soc)
    meta: Dict[str, Any] = {
        "xc": getattr(mf, "xc", None),
        "basis": getattr(mol, "basis", None),
        "soc": False,
        "soc_mode": soc_mode if soc_mode != "si" else "none",
        "n": n,
        "ell": ell,
        "from_user_mf": True,
        "edge_atom_indices": edge_parent,
        "e_tot": float(getattr(mf, "e_tot", np.nan)),
        "core_selection": sel_meta,
    }
    if soc_mode == "si":
        meta["soc_deferred"] = "si"

    if soc_mode == "orbital":
        if ell == 0:
            if verbose:
                print("SOC requested for s-shell; skipping (no first-order SOC).")
        else:
            energies, coeff = apply_orbital_soc(
                energies, coeff, mol, method="bp-subspace"
            )
            meta["soc"] = True
            meta["soc_mode"] = "orbital"

    if atom_indices is None:
        atom_indices = list(range(mol.natm))

    return CoreOrbitals(
        energies=np.asarray(energies),
        mo_coeff=np.asarray(coeff),
        fragment_mol=mol,
        atom_indices=np.asarray(atom_indices, dtype=int),
        edge=str(edge).upper(),
        shell=shell_label,
        meta=meta,
    )


def extract_fragment_core(
    parent_mol,
    atom_indices: Optional[Sequence[int]] = None,
    *,
    fragment: Optional[FragmentSpec] = None,
    edge: str = "K",
    shell: Optional[str] = None,
    basis: str = "def2-tzvp",
    xc: str = "pbe",
    charge: Optional[int] = None,
    spin: Optional[int] = None,
    soc: Union[bool, str] = False,
    edge_atom_indices: Optional[Sequence[int]] = None,
    n_core_per_atom: Optional[int] = None,
    max_cycle: int = 80,
    level_shift: float = 0.0,
    use_gpu: bool = False,
    verbose: bool = False,
) -> CoreOrbitals:
    """AE SCF on a geometry cut; return core MOs for the requested edge.

    Parameters
    ----------
    atom_indices
        Parent atom indices that define the **fragment SCF** (radius cut).
        Ignored when ``fragment`` is given (unless you also pass indices to
        override).
    fragment
        Optional :class:`FragmentSpec` from :func:`select_xas_fragment` or
        :meth:`FragmentSpec.user`. Supplies indices, charge, spin, and default
        edge atoms.
    edge_atom_indices
        Parent atom indices whose core orbitals are **imported** (must be a
        subset of ``atom_indices``). Example: fragment = metal + ligands,
        ``edge_atom_indices=[metal]`` for a single K-edge. Default: from
        ``fragment`` if present, else all heavy atoms (Z≥3) for K-edge.
    n_core_per_atom
        Override spatial MOs per edge atom (default ``2ℓ+1``).
    max_cycle
        PySCF SCF iteration limit.
    level_shift
        Optional level shift (Ha) to stabilize difficult cluster SCFs.
    use_gpu
        If True, run the fragment SCF with ``gpu4pyscf`` (falls back to CPU
        with a warning if unavailable).
    soc
        If True (L/M), diagonalize one-electron SOC in the core-shell
        spin-orbital subspace and return j-adapted complex MOs.
    """
    import warnings

    from pyscf import dft

    if fragment is not None:
        if atom_indices is None:
            atom_indices = fragment.atom_indices
        if charge is None:
            charge = int(fragment.charge)
        if spin is None:
            spin = int(fragment.spin)
        if edge_atom_indices is None and fragment.edge_atom_indices is not None:
            edge_atom_indices = fragment.edge_atom_indices

    if atom_indices is None:
        raise ValueError(
            "Provide atom_indices=... or fragment=FragmentSpec / select_xas_fragment(...)"
        )
    if charge is None:
        charge = 0
    if spin is None:
        spin = 0

    shell_label, n, ell = _resolve_shell(edge, shell)
    atom_indices = [int(i) for i in atom_indices]
    parent_to_frag = {p: j for j, p in enumerate(atom_indices)}

    edge_frag: Optional[List[int]] = None
    edge_parent: Optional[List[int]] = None
    if edge_atom_indices is not None:
        edge_parent = [int(i) for i in edge_atom_indices]
        missing = [i for i in edge_parent if i not in parent_to_frag]
        if missing:
            raise ValueError(
                f"edge_atom_indices {missing} are not in fragment atom_indices "
                f"{atom_indices}. Edge atoms must be a subset of the fragment."
            )
        edge_frag = [parent_to_frag[i] for i in edge_parent]

    frag = _build_fragment_mole(
        parent_mol,
        atom_indices,
        basis=basis,
        charge=int(charge),
        spin=int(spin),
        verbose=4 if verbose else 0,
    )

    used_gpu = False
    mf = None
    if use_gpu and spin != 0:
        warnings.warn(
            "use_gpu=True with spin≠0 is not supported for fragment SCF; "
            "falling back to CPU UKS."
        )
    elif use_gpu:
        try:
            import cupy  # noqa: F401
            from gpu4pyscf.dft import rks as gpu_rks

            mf = gpu_rks.RKS(frag)
            mf.xc = xc
            mf.verbose = 4 if verbose else 0
            mf.max_cycle = int(max_cycle)
            if level_shift:
                mf.level_shift = float(level_shift)
            e_tot = mf.kernel()
            used_gpu = True

            import cupy as cp

            def _host(a):
                if isinstance(a, cp.ndarray):
                    return cp.asnumpy(a)
                return np.asarray(a)

            class _HostMF:
                pass

            h = _HostMF()
            h.mol = frag
            h.mo_energy = _host(mf.mo_energy)
            h.mo_coeff = _host(mf.mo_coeff)
            h.mo_occ = _host(mf.mo_occ)
            h.converged = bool(mf.converged)
            mf_host = h
        except Exception as exc:
            warnings.warn(
                f"GPU fragment SCF unavailable ({type(exc).__name__}: {exc}); "
                "falling back to CPU PySCF."
            )
            mf = None
            used_gpu = False

    if mf is None:
        if spin == 0:
            mf = dft.RKS(frag)
        else:
            mf = dft.UKS(frag)
        mf.xc = xc
        mf.verbose = 4 if verbose else 0
        mf.max_cycle = int(max_cycle)
        if level_shift:
            mf.level_shift = float(level_shift)
        e_tot = mf.kernel()
        mf_host = mf

    if not getattr(mf_host, "converged", False):
        raise RuntimeError(f"Fragment SCF did not converge (E={e_tot})")

    # Attach e_tot for meta; core_from_mf reads mf.e_tot when present
    if not hasattr(mf_host, "e_tot"):
        mf_host.e_tot = float(e_tot)
    else:
        try:
            mf_host.e_tot = float(e_tot)
        except Exception:
            pass

    core = core_from_mf(
        mf_host,
        edge=edge,
        shell=shell,
        edge_atom_indices=edge_frag,
        n_core_per_atom=n_core_per_atom,
        soc=soc,
        atom_indices=atom_indices,
        verbose=verbose,
    )
    core.meta["e_tot"] = float(e_tot)
    core.meta["xc"] = xc
    core.meta["basis"] = basis
    core.meta["use_gpu_scf"] = bool(used_gpu)
    core.meta["from_user_mf"] = False
    core.meta["scf_charge"] = int(charge)
    core.meta["scf_spin"] = int(spin)
    core.meta["edge_atom_indices"] = (
        list(edge_parent) if edge_parent is not None else None
    )
    if fragment is not None:
        core.meta["fragment_mode"] = fragment.mode
        core.meta["fragment"] = dict(fragment.meta)
    # fragment_mol should be the built frag (same as mf.mol)
    core.fragment_mol = frag
    return core


# ---------------------------------------------------------------------------
# 4. Optional orbital SOC
# ---------------------------------------------------------------------------

def apply_orbital_soc(
    energies: np.ndarray,
    mo_coeff: np.ndarray,
    mol,
    *,
    method: str = "bp-subspace",
) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonalize BP SOC in the core spatial-MO subspace (spin-orbitals).

    Returns complex MO coefficients in the fragment AO basis with shape
    ``(nao, 2*n_spatial)`` and real SOC-split energies.
    """
    if method != "bp-subspace":
        raise ValueError(f"Unknown SOC method {method!r}")

    from casidapy.utils.soc import soc_ao_integrals

    C = np.asarray(mo_coeff, dtype=float)
    eps = np.asarray(energies, dtype=float).ravel()
    n_sp = C.shape[1]
    nao = C.shape[0]
    hso = soc_ao_integrals(mol)  # (3, nao, nao) complex hermitian

    # Spin-orbital order: [0..n_sp) = α, [n_sp..2n_sp) = β
    n_so = 2 * n_sp
    H = np.zeros((n_so, n_so), dtype=complex)
    # Orbital energy diagonal
    for i in range(n_sp):
        H[i, i] = eps[i]
        H[n_sp + i, n_sp + i] = eps[i]

    # h_mo[u] = C† hso[u] C
    h_mo = np.einsum("xuv,ui,vj->xij", hso, C, C, optimize=True)

    # S_z: αα += +1/2 h_z, ββ += -1/2 h_z
    H[:n_sp, :n_sp] += 0.5 * h_mo[2]
    H[n_sp:, n_sp:] += -0.5 * h_mo[2]
    # S_+ / S_- : αβ ← (h_x - i h_y)/2, βα ← (h_x + i h_y)/2
    h_pm = 0.5 * (h_mo[0] - 1j * h_mo[1])
    H[:n_sp, n_sp:] += h_pm
    H[n_sp:, :n_sp] += h_pm.conj().T

    H = 0.5 * (H + H.conj().T)
    evals, evecs = np.linalg.eigh(H)

    # Map spinor eigenvectors → complex AO MOs (α and β channels summed for
    # a single spatial-like column used by real PW kernels after phase align).
    # For each eigenstate j: ψ_j(r) = Σ_i [U_{iα,j} φ_i + U_{iβ,j} φ_i]
    C_out = np.zeros((nao, n_so), dtype=complex)
    for j in range(n_so):
        u_a = evecs[:n_sp, j]
        u_b = evecs[n_sp:, j]
        C_out[:, j] = C @ (u_a + u_b)

    return evals.real.copy(), C_out


# ---------------------------------------------------------------------------
# 5. Inject cores into a PW / GTO host
# ---------------------------------------------------------------------------

def core_mos_to_pw_fields(
    core: CoreOrbitals,
    grid,
    *,
    normalize: bool = True,
    use_gpu: bool = False,
    comm=None,
    blksize: int = 16000,
):
    """Evaluate fragment core MOs on a DFTpy ``DirectGrid``.

    Returns a list of ``DirectField`` (real after phase alignment).

    ``use_gpu``
        CuPy for ``AO @ C`` contractions.
    ``comm``
        MPI communicator: grid blocks are distributed, then ``Allreduce``
        assembles each MO.
    """
    from pyscf.dft import numint
    from dftpy.field import DirectField
    from casidapy.utils.casida_utils import normalize_wavefunctions
    from casidapy.utils.accel import (
        array_module,
        asnumpy,
        block_slices,
        mpi_allreduce_sum,
    )

    mol = core.fragment_mol
    C = np.asarray(core.mo_coeff)
    if C.ndim != 2:
        raise ValueError(f"mo_coeff must be 2-D, got shape {C.shape}")
    complex_c = np.iscomplexobj(C)
    from casidapy.xas.xas_utils import grid_coords_bohr

    coords = grid_coords_bohr(grid)
    ngrid = coords.shape[0]
    n_core = C.shape[1]
    xp = array_module(use_gpu)
    C_x = xp.asarray(C, dtype=complex if complex_c else float)
    psi_sum = [np.zeros(ngrid, dtype=complex if complex_c else float) for _ in range(n_core)]

    for p0, p1 in block_slices(ngrid, blksize, comm):
        ao = numint.eval_ao(mol, coords[p0:p1])
        block = asnumpy(xp.asarray(ao, dtype=float) @ C_x)
        for j in range(n_core):
            psi_sum[j][p0:p1] = block[:, j]

    fields = []
    for j in range(n_core):
        psi = mpi_allreduce_sum(comm, psi_sum[j])
        if np.iscomplexobj(psi):
            idx = int(np.argmax(np.abs(psi)))
            phase = np.angle(psi.flat[idx])
            psi = (psi * np.exp(-1j * phase)).real
        else:
            psi = np.asarray(psi, dtype=float)
        fields.append(DirectField(grid=grid, rank=1, griddata_3d=psi.reshape(grid.nr)))

    if normalize:
        fields = normalize_wavefunctions(fields, grid)
    return fields

def inject_core_orbitals(
    kernel,
    core_energies,
    core_orbitals,
    *,
    disable_uspp: bool = True,
    orthogonalize_to_virt: bool = True,
    use_gpu: bool = False,
):
    """Replace kernel occupied active space with imported core orbitals.

    Virtuals are left unchanged. For ``PlaneWaveKernel``, ``core_orbitals``
    are ``DirectField`` instances on ``kernel.grid``. For ``GTOKernel``,
    pass MO coefficients ``(nao, n_core)``. Clears setup caches.
    """
    energies = np.asarray(core_energies, dtype=float).ravel()

    # GTO (MO) path: MO coefficient matrix (nao, n_core). Grid backends (PW)
    # take DirectField orbitals below instead.
    if hasattr(kernel, "set_core_active_space"):
        C = np.asarray(core_orbitals)
        if C.ndim == 2 and not getattr(kernel, "provides_grid", False):
            return kernel.set_core_active_space(energies, C)

    if len(core_orbitals) != energies.size:
        raise ValueError(
            f"n_energies={energies.size} != n_orbitals={len(core_orbitals)}"
        )
    if not getattr(kernel, "_psi_unocc", None):
        raise RuntimeError(
            "Kernel has no virtuals; call set_active_orbitals (valence) first "
            "or set virtuals before inject_core_orbitals."
        )

    if getattr(kernel, "provides_grid", False):
        psi_occ = list(core_orbitals)
        psi_unocc = list(kernel._psi_unocc)
        if orthogonalize_to_virt:
            gpu = bool(use_gpu or getattr(kernel, "use_gpu", False))
            psi_occ = _orthogonalize_occ_to_virt(
                psi_occ, psi_unocc, kernel.grid, use_gpu=gpu
            )
        if disable_uspp and getattr(kernel, "use_uspp", False):
            kernel.use_uspp = False
            kernel.beta_projectors = None
            kernel.qij_augmentation = None
        kernel.set_active_orbitals(energies, kernel._unocc_e, psi_occ, psi_unocc)
        # Drop GPU / FFT caches tied to old orbitals
        kernel._psi_occ_arr = None
        kernel._psi_unocc_arr = None
        kernel._psi_occ_dev = None
        kernel._psi_unocc_dev = None
        kernel._ready = False
        kernel._dE = None
        return kernel

    # GTO: list/tuple of columns → stack
    if hasattr(kernel, "set_core_active_space"):
        C = np.column_stack([np.asarray(c).ravel() for c in core_orbitals])
        return kernel.set_core_active_space(energies, C)

    raise TypeError(
        f"inject_core_orbitals: unsupported kernel type {type(kernel)}"
    )


def _orthogonalize_occ_to_virt(
    psi_occ, psi_unocc, grid, *, use_gpu: bool = False, n_pass: int = 2
):
    """Gram–Schmidt core orbitals against virtuals on the grid.

    Two passes (default) drive ``⟨core|virt⟩`` closer to zero than a single
    subtract — the residual is what makes the length-gauge dipole
    origin-dependent on a coarse FFT mesh.
    """
    from dftpy.field import DirectField
    from casidapy.utils.casida_utils import normalize_wavefunctions
    from casidapy.utils.accel import array_module, asnumpy

    dV = float(grid.dV)
    xp = array_module(use_gpu)
    if not psi_unocc:
        return normalize_wavefunctions(list(psi_occ), grid)
    U = xp.stack(
        [
            xp.asarray(np.asarray(p, dtype=float).ravel(), dtype=float)
            for p in psi_unocc
        ],
        axis=0,
    )
    O = xp.stack(
        [
            xp.asarray(np.asarray(psi, dtype=float).ravel(), dtype=float)
            for psi in psi_occ
        ],
        axis=0,
    )
    for _ in range(max(int(n_pass), 1)):
        # ⟨o|u⟩ = (O · U) dV;  O ← O − ⟨o|u⟩ U  (no extra dV).
        overlaps = (O @ U.T) * dV
        O = O - overlaps @ U
    norms = xp.sqrt(xp.sum(O * O, axis=1) * dV)
    if bool(asnumpy(xp.any(norms < 1e-12))):
        raise RuntimeError(
            "Core orbital vanished after orthogonalization to virtuals; "
            "check grid coverage / fragment placement."
        )
    O = O / norms[:, None]
    shape = grid.nr
    out = [
        DirectField(
            grid=grid,
            rank=1,
            griddata_3d=asnumpy(O[i]).reshape(shape),
        )
        for i in range(O.shape[0])
    ]
    return normalize_wavefunctions(out, grid)


# ---------------------------------------------------------------------------
# 6. CVS-TDA drivers
# ---------------------------------------------------------------------------

def run_cvs_tda(kernel, options, *, n_states: Optional[int] = None):
    """Solve CVS-TDA on a kernel whose occupied space is core-only."""
    from casidapy.casida_api import CasidaOptions
    from casidapy.casida_engine import run_casida

    if getattr(kernel, "n_occ", 0) < 1:
        raise RuntimeError("CVS-TDA requires injected core occupied orbitals.")
    n_st = int(n_states) if n_states is not None else int(options.n_states)
    n_st = min(n_st, int(kernel.n_trans))
    opts = CasidaOptions(
        n_occ=int(kernel.n_occ),
        n_unocc=int(kernel.n_unocc),
        n_states=n_st,
        basis=getattr(options, "basis", "pw"),
        tda=True,
        matrix_free=True,
        solver_method=getattr(options, "solver_method", "davidson"),
        solver_tol=getattr(options, "solver_tol", 1e-6),
        solver_maxiter=getattr(options, "solver_maxiter", 80),
        use_gpu=bool(getattr(kernel, "use_gpu", False)),
        xc=getattr(options, "xc", "PBE"),
        use_uspp=bool(getattr(kernel, "use_uspp", False)),
        spin_state=getattr(
            options, "spin_state", getattr(kernel, "spin_state", "singlet")
        ),
    )
    if not getattr(kernel, "_ready", False):
        kernel.setup(tda=True)
    return run_casida(kernel, opts)


def _extract_gto_cvs_kernel(
    mf,
    *,
    n_unocc: int,
    n_states: int,
    spin_state: str,
    use_df: bool,
    use_gpu: bool,
    use_mpi_response: bool,
    verbose: bool,
):
    from casidapy.adapter.pyscf import extract_gto_kernel

    return extract_gto_kernel(
        mf,
        n_occ=1,
        n_unocc=int(n_unocc),
        n_states=int(n_states),
        use_df=use_df and not use_mpi_response,
        use_gpu=use_gpu,
        use_mpi_response=use_mpi_response,
        verbose=verbose,
        k_cache_max=0,
        spin_state=spin_state,
    )


def run_cvs_si_soc(
    kernel_s,
    kernel_t,
    options,
    *,
    n_states: Optional[int] = None,
    n_states_triplet: Optional[int] = None,
    mol=None,
    C_o=None,
    C_v=None,
    hole_only: bool = False,
    include_ground: bool = False,
    scale: Optional[float] = None,
):
    """Singlet + triplet CVS-TDA coupled by one-electron SI-SOC.

    ``kernel_s`` must be the singlet kernel (dipoles).  ``kernel_t`` is the
    triplet-adapted ``f_xc`` kernel on the same core/virtual space.
    Hole-only L-edge (PW): pass fragment ``mol`` + 2p ``C_o`` and
    ``hole_only=True`` (``h_vv = 0``).
    """
    from casidapy.casida_api import CasidaOptions
    from casidapy.utils.soc import solve_soc_si

    res_s = run_cvs_tda(kernel_s, options, n_states=n_states)
    n_t = int(n_states_triplet) if n_states_triplet is not None else (
        int(n_states) if n_states is not None else int(options.n_states)
    )
    opts_t = CasidaOptions(
        n_occ=int(kernel_t.n_occ),
        n_unocc=int(kernel_t.n_unocc),
        n_states=n_t,
        basis=getattr(options, "basis", "gto"),
        tda=True,
        matrix_free=True,
        solver_method=getattr(options, "solver_method", "davidson"),
        solver_tol=getattr(options, "solver_tol", 1e-6),
        solver_maxiter=getattr(options, "solver_maxiter", 80),
        use_gpu=bool(getattr(kernel_t, "use_gpu", False)),
        xc=getattr(options, "xc", "PBE"),
        use_uspp=bool(getattr(kernel_t, "use_uspp", False)),
        spin_state=getattr(kernel_t, "spin_state", "triplet"),
    )
    res_t = run_cvs_tda(kernel_t, opts_t, n_states=n_t)
    soc = solve_soc_si(
        res_s,
        res_t,
        kernel_s,
        mol=mol,
        C_o=C_o,
        C_v=C_v,
        hole_only=bool(hole_only),
        include_ground=bool(include_ground),
        scale=scale,
    )
    return soc, res_s, res_t


def run_cvs_gto_from_mf(
    mf,
    *,
    edge: str = "K",
    shell: Optional[str] = None,
    edge_atom_indices: Optional[Sequence[int]] = None,
    n_unocc: int = 50,
    n_states: int = 50,
    n_states_triplet: Optional[int] = None,
    soc: Union[bool, str] = False,
    hole_only: bool = False,
    use_gpu: bool = False,
    use_mpi_response: bool = False,
    use_df: bool = True,
    core_localize: Optional[str] = None,
    verbose: bool = False,
):
    """CVS-TDA entirely in GTO: cores from ``mf``, **real virtuals** from ``mf``.

    Typical molecule workflow (C₆₀, …)::

        mf = dft.RKS(mol).density_fit()
        mf.xc = "pbe0"
        mf.kernel()
        res, core, kernel = run_cvs_gto_from_mf(
            mf, edge="K", edge_atom_indices=[0], n_unocc=100, n_states=100
        )

    L-edge / M-edge **SOC-coupled TDDFT** (spatial 2p/3d, not orbital SOC)::

        res, core, kernel = run_cvs_gto_from_mf(
            mf, edge="L", edge_atom_indices=[0], soc="si",
            n_unocc=40, n_states=40,
        )
        # res.omega / res.f / res.d_mode are the mixed SI-SOC spectrum
        # res.singlet, res.triplet are the uncoupled CVS-TDA parents

    ``soc``
        ``False`` / ``"none"`` — spatial cores, singlet TDA.
        ``True`` / ``"orbital"`` — Breit–Pauli in the core subspace before
        injection (j-adapted energies only; not L₂/L₃ TDDFT).
        ``"si"`` — singlet + triplet TDA coupled by SI-SOC.
    ``hole_only``
        With ``soc="si"``, drop particle SOC (``h_vv = 0``).  Default is
        full one-electron SI (hole + particle).  PW L-edge uses hole-only.
    ``use_gpu``
        CuPy GTO kernel (+ gpu4pyscf response when available).
    ``use_mpi_response``
        Promote ``mf`` to ``mpi4pyscf`` for MPI ``get_jk`` response.
    """
    soc_mode = parse_soc_mode(soc)
    core_soc = True if soc_mode == "orbital" else False
    core = core_from_mf(
        mf,
        edge=edge,
        shell=shell,
        edge_atom_indices=edge_atom_indices,
        soc=core_soc,
        core_localize=core_localize,
        verbose=verbose,
    )
    kw = dict(
        n_unocc=int(n_unocc),
        n_states=int(n_states),
        use_df=use_df,
        use_gpu=use_gpu,
        use_mpi_response=use_mpi_response,
        verbose=verbose,
    )
    if soc_mode != "si":
        kernel, opts = _extract_gto_cvs_kernel(mf, spin_state="singlet", **kw)
        inject_core_orbitals(kernel, core.energies, core.mo_coeff)
        return run_cvs_tda(kernel, opts, n_states=n_states), core, kernel

    kernel_s, opts_s = _extract_gto_cvs_kernel(mf, spin_state="singlet", **kw)
    kernel_t, _opts_t = _extract_gto_cvs_kernel(mf, spin_state="triplet", **kw)
    inject_core_orbitals(kernel_s, core.energies, core.mo_coeff)
    inject_core_orbitals(kernel_t, core.energies, core.mo_coeff)
    soc_res, res_s, res_t = run_cvs_si_soc(
        kernel_s,
        kernel_t,
        opts_s,
        n_states=n_states,
        n_states_triplet=n_states_triplet,
        hole_only=bool(hole_only),
    )
    n_s = int(np.asarray(res_s.omega).size)
    n_t = int(np.asarray(res_t.omega).size)
    out = CVSSOCResults(
        omega=soc_res.omega,
        f=soc_res.f,
        mu=soc_res.mu,
        U=soc_res.U,
        singlet_weight=soc_res.singlet_weight,
        triplet_weight=soc_res.triplet_weight,
        singlet=res_s,
        triplet=res_t,
        core=core,
        kernel=kernel_s,
        kernel_t=kernel_t,
        soc=soc_res,
        metadata={
            **dict(soc_res.meta),
            "soc_mode": "si",
            "edge": str(edge).upper(),
            "shell": core.shell,
            "n_core": int(core.energies.size),
            "n_singlet": n_s,
            "n_triplet": n_t,
            "n_si": int(soc_res.omega.size),
        },
    )
    return out, core, kernel_s


def build_pw_kernel_from_qepy(
    driver,
    *,
    n_virt: int,
    use_gpu: bool = False,
    xc: str = "PBE",
    casida_uspp: bool = False,
    n_placeholder_occ: int = 1,
    n_total_occ: Optional[int] = None,
    unocc_window_ev: Optional[float] = None,
    n_virt_active: Optional[int] = None,
    virt_energy_shift_ha: float = 0.0,
    pack_band_indices: Optional[Sequence[int]] = None,
    spin_state: str = "singlet",
    verbose: bool = False,
):
    """Build a ``PlaneWaveKernel`` with **real QE virtuals** for CVS inject.

    Thin wrapper around :func:`casidapy.adapter.qepy.extract_pw_kernel`.
    Returns ``(kernel, grid)`` for back-compat with inject helpers.

    ``n_virt_active``
        Optional energy-strided subsample size for large ``n_virt`` pools.
        Leave ``None`` (default) to use every selected virtual.
    """
    from casidapy.adapter.qepy import extract_pw_kernel

    kernel, _opts = extract_pw_kernel(
        driver,
        n_unocc=int(n_virt),
        n_total_occ=n_total_occ,
        n_placeholder_occ=int(n_placeholder_occ),
        n_states=max(int(n_virt_active or n_virt), 1),
        xc=xc,
        use_gpu=bool(use_gpu),
        use_uspp=bool(casida_uspp),
        tda=True,
        matrix_free=True,
        unocc_window_ev=None if pack_band_indices is not None else unocc_window_ev,
        unocc_subsample=None if pack_band_indices is not None else n_virt_active,
        virt_energy_shift_ha=float(virt_energy_shift_ha),
        pack_band_indices=pack_band_indices,
        spin_state=spin_state,
        verbose=verbose,
    )
    return kernel, kernel.grid


def clone_pw_kernel_triplet(kernel_s, *, xc=None):
    """Triplet ALDA ``f_xc`` kernel sharing the singlet core/virtual orbitals.

    Used for PW L-edge SI-SOC: same 2p + QE virtuals, Hartree dropped in ``K``.
    GIPAW-reconstructed virtuals and dipole origin are copied; dipoles stay
    on the singlet kernel.
    """
    from dftpy.functional.xc import XC
    from casidapy.kernels.plane_wave import PlaneWaveKernel
    from casidapy.utils.casida_utils import XC_TO_LIBXC_COMPONENTS

    xc_name = str(
        xc
        or getattr(kernel_s, "xc", None)
        or getattr(getattr(kernel_s, "functional", None), "name", None)
        or "PBE"
    ).upper()
    if xc_name not in XC_TO_LIBXC_COMPONENTS:
        if "PBE" in xc_name:
            xc_name = "PBE"
        elif xc_name in ("LDA", "PZ", "PW"):
            xc_name = "LDA"
        elif "BLYP" in xc_name or "B3LYP" in xc_name or "BHANDH" in xc_name:
            xc_name = "BLYP"
    if xc_name not in XC_TO_LIBXC_COMPONENTS:
        raise ValueError(
            f"No libxc triplet map for {xc_name}; cannot clone PW triplet kernel."
        )
    kernel_t = PlaneWaveKernel(
        kernel_s.rho,
        XC(xc=xc_name),
        use_gpu=bool(getattr(kernel_s, "use_gpu", False)),
        use_uspp=False,
        spin_state="triplet",
        libxc_xc_components=XC_TO_LIBXC_COMPONENTS[xc_name],
        verbose=bool(getattr(kernel_s, "verbose", False)),
    )
    kernel_t.spin_state = "triplet"
    kernel_t.set_active_orbitals(
        kernel_s._occ_e,
        kernel_s._unocc_e,
        list(kernel_s._psi_occ),
        list(kernel_s._psi_unocc),
    )
    tilde = getattr(kernel_s, "_psi_unocc_tilde", None)
    if tilde is not None:
        kernel_t._psi_unocc_tilde = tilde
    kernel_t.dipole_origin = getattr(kernel_s, "dipole_origin", (0.0, 0.0, 0.0))
    ijab = getattr(kernel_s, "_ijab_gto", None)
    if ijab is not None:
        kernel_t.set_gto_ijab(ijab)
    return kernel_t
