"""Element / shell reference tables for XAS (JSON-backed)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

_DATA_PATH = Path(__file__).with_name("chemical_data.json")

ShellQuantumNumbers = Tuple[int, int]


@lru_cache(maxsize=1)
def _load_raw() -> dict:
    with _DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _shell_map(section: str) -> Dict[str, ShellQuantumNumbers]:
    raw = _load_raw()[section]
    return {str(k): (int(v[0]), int(v[1])) for k, v in raw.items()}


@lru_cache(maxsize=1)
def edge_shell() -> Dict[str, ShellQuantumNumbers]:
    """X-ray edge label → ``(n, ℓ)`` for default core shell (K→1s, L→2p, …)."""
    return _shell_map("edge_shell")


@lru_cache(maxsize=1)
def shell_alias() -> Dict[str, ShellQuantumNumbers]:
    """Subshell string → ``(n, ℓ)`` (e.g. ``\"2p\"`` → ``(2, 1)``)."""
    return _shell_map("shell_alias")


@lru_cache(maxsize=1)
def default_oxidation_states() -> Dict[str, int]:
    """Formal oxidation numbers for neutral-fragment automation."""
    raw = _load_raw()["oxidation_states"]
    return {str(k).strip().capitalize(): int(v) for k, v in raw.items()}


@lru_cache(maxsize=1)
def covalent_radii_ang() -> Dict[str, float]:
    """Fallback covalent radii (Å) when ASE is unavailable."""
    raw = _load_raw()["covalent_radii_ang"]
    return {str(k).strip().capitalize(): float(v) for k, v in raw.items()}


@lru_cache(maxsize=1)
def first_shell_peel_radii() -> Dict[str, float]:
    """Heuristic first-neighbour peel radii (Å) for periodic XAS hosts."""
    raw = _load_raw()["first_shell_peel_radius_ang"]
    return {str(k).strip().capitalize(): float(v) for k, v in raw.items()}


@lru_cache(maxsize=1)
def second_shell_peel_radii() -> Dict[str, float]:
    """Heuristic second-neighbour peel radii (Å) for periodic XAS hosts."""
    raw = _load_raw()["second_shell_peel_radius_ang"]
    return {str(k).strip().capitalize(): float(v) for k, v in raw.items()}


@lru_cache(maxsize=1)
def defaults() -> Dict[str, Any]:
    """Scalar defaults shared across XAS modules (units, cutoffs, quadrature)."""
    raw = dict(_load_raw()["defaults"])
    out: Dict[str, Any] = {
        "ha_to_ev": float(raw["ha_to_ev"]),
        "oh_covalent_ang": float(raw["oh_covalent_ang"]),
        "bond_factor": float(raw["bond_factor"]),
        "bond_factor_ligand_growth": float(raw["bond_factor_ligand_growth"]),
        "max_fragment_atoms": int(raw["max_fragment_atoms"]),
        "fragment_radius_ang": float(raw["fragment_radius_ang"]),
        "first_shell_peel_radius_ang": float(raw["first_shell_peel_radius_ang"]),
        "second_shell_peel_radius_ang": float(raw["second_shell_peel_radius_ang"]),
        "diffuse_augment_zeta_s": float(raw["diffuse_augment_zeta_s"]),
        "diffuse_augment_zeta_p": float(raw["diffuse_augment_zeta_p"]),
        "ghost_diffuse_zeta_min_s": float(raw["ghost_diffuse_zeta_min_s"]),
        "ghost_diffuse_zeta_min_p": float(raw["ghost_diffuse_zeta_min_p"]),
        "ghost_diffuse_zeta_min_d": float(raw["ghost_diffuse_zeta_min_d"]),
        "ghost_diffuse_zeta_ratio": float(raw["ghost_diffuse_zeta_ratio"]),
        "radial_rmax_bohr": float(raw["radial_rmax_bohr"]),
        "radial_n_rad": int(raw["radial_n_rad"]),
        "radial_n_ang": int(raw["radial_n_ang"]),
        "ae_smearing_sigma_ha": float(raw["ae_smearing_sigma_ha"]),
        "ae_smearing_method": str(raw["ae_smearing_method"]),
    }
    return out


def first_shell_peel_radius_ang(symbol: str) -> float:
    """Peel radius for ``first_shell`` mode; falls back to ``defaults``."""
    sym = str(symbol).strip().capitalize()
    return first_shell_peel_radii().get(
        sym, defaults()["first_shell_peel_radius_ang"]
    )


def second_shell_peel_radius_ang(symbol: str) -> float:
    """Peel radius for ``second_shell`` mode; falls back to ``defaults``."""
    sym = str(symbol).strip().capitalize()
    return second_shell_peel_radii().get(
        sym, defaults()["second_shell_peel_radius_ang"]
    )


# Noble-gas Z for valence counting (He, Ne, Ar, Kr, Xe, Rn).
_NOBLE_Z = (2, 10, 18, 36, 54, 86)


def noble_gas_z(z: int) -> int:
    """Largest noble-gas Z strictly below ``z`` (0 for H/He)."""
    z = int(z)
    prev = 0
    for n in _NOBLE_Z:
        if z <= n:
            return prev
        prev = n
    return int(_NOBLE_Z[-1])


def valence_electron_count(z: int) -> int:
    """Electrons past the previous noble gas."""
    z = int(z)
    return z - noble_gas_z(z)


def d_electron_count(z: int, oxidation: int) -> int:
    """Ionic d^n for a transition metal: valence − oxidation, clamped to [0, 10].

    Ti⁴⁺ (Z=22, ox=4) → 0; Fe³⁺ (Z=26, ox=3) → 5; Zn²⁺ (Z=30, ox=2) → 10.
    """
    n = valence_electron_count(z) - int(oxidation)
    return int(max(0, min(10, n)))


def valence_particle_nl(z: int, edge: str) -> Tuple[int, int]:
    """Particle ``(n, ℓ)`` for white-line virtuals (dipole-allowed shell).

    K-edge: 1s → valence p (2p/3p/4p by period) — LiF / molecular K white line.
    L-edge: 2p → valence d (3d/4d/5d by period).
    M-edge: 3d hole → 4f (lanthanides) or next p (4p/5p/6p).
    """
    z = int(z)
    edge_u = str(edge).strip().upper()
    if edge_u in ("K", "1S"):
        if z <= 10:
            return 2, 1
        if z <= 18:
            return 3, 1
        if z <= 36:
            return 4, 1
        return 5, 1
    if edge_u in ("L", "2P"):
        if z <= 30:
            return 3, 2
        if z <= 48:
            return 4, 2
        return 5, 2
    if edge_u in ("M", "3D"):
        if 57 <= z <= 71:
            return 4, 3
        if z <= 30:
            return 4, 1
        if z <= 48:
            return 5, 1
        return 6, 1
    raise ValueError(f"particle shell needs edge K/L/M, got {edge!r}")


# Module-level aliases (loaded once) for simple imports.
EDGE_SHELL = edge_shell()
SHELL_ALIAS = shell_alias()
DEFAULT_OXIDATION_STATES = default_oxidation_states()
COVALENT_RADII_ANG = covalent_radii_ang()
DEFAULTS = defaults()
