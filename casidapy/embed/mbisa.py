"""Minimal-Basis Iterative Stockholder (MBIS) atomic charges / densities.

MBIS (Verstraelen et al., JCTC 2016) represents each atom by a sum of
normalized *s*-type Slater functions and iterates stockholder weights until
the shell populations ``N_k`` and exponents ``α_k`` are self-consistent with
the partitioned density::

    ρ̃_A(r) = Σ_{k∈A} N_k (α_k³ / 8π) exp(−α_k |r − R_A|)
    w_A     = ρ̃_A / Σ_B ρ̃_B
    ρ_A     = w_A · ρ_tot
    N_k     ← ∫ ρ_A (ρ̃_k / ρ̃_A)
    α_k     ← 3 N_k / ∫ ρ_A (ρ̃_k / ρ̃_A) r

This module is grid-native (no QE driver required).  Use
:func:`mbis_partition` for charges / weights, or
:func:`mbis_hirshfeld_partition` to obtain a
:class:`~casidapy.embed.hirshfeld.HirshfeldPartition`-compatible active/env split.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from casidapy.embed.hirshfeld import HirshfeldPartition

logger = logging.getLogger(__name__)

DEFAULT_MBIS_MAX_ITER = 100
DEFAULT_MBIS_TOL = 1.0e-8
DEFAULT_MBIS_MIX = 0.5
# Soft floor so exponents cannot collapse on coarse grids.
_ALPHA_MIN = 0.05
_ALPHA_MAX = 50.0
_N_MIN = 1.0e-12


def n_mbis_shells(atomic_number: int) -> int:
    """Number of s-type Slater shells in the MBIS minimal basis for ``Z``."""
    z = int(atomic_number)
    if z <= 0:
        raise ValueError(f"atomic_number must be > 0, got {atomic_number}")
    if z <= 2:
        return 1
    if z <= 10:
        return 2
    if z <= 18:
        return 3
    if z <= 36:
        return 4
    if z <= 54:
        return 5
    return 6


def _initial_shell_populations(z: int, n_shell: int) -> np.ndarray:
    """Neutral-atom shell fill (core 2e⁻, valence remainder on outer shell)."""
    pop = np.zeros(n_shell, dtype=float)
    remaining = float(z)
    for k in range(n_shell - 1):
        take = min(2.0, remaining)
        pop[k] = take
        remaining -= take
    pop[-1] = max(remaining, 0.0)
    if float(pop.sum()) < 1.0e-12:
        pop[-1] = max(float(z), 0.0)
    return pop


def _initial_exponents(z: int, n_shell: int) -> np.ndarray:
    """Slater-rule-ish α guesses (larger α = tighter core)."""
    # Rough: innermost ~ Z, then geometrically softer.
    zf = max(float(z), 1.0)
    alpha = np.empty(n_shell, dtype=float)
    for k in range(n_shell):
        alpha[k] = zf / (1.5 ** k)
    return np.clip(alpha, _ALPHA_MIN, _ALPHA_MAX)


def slater_s_density(r: np.ndarray, N: float, alpha: float) -> np.ndarray:
    """Normalized s-Slater density ``N (α³/8π) e^{−α r}`` (∫ = N)."""
    a = float(alpha)
    n = float(N)
    if n <= 0.0 or a <= 0.0:
        return np.zeros_like(r, dtype=float)
    pre = n * (a ** 3) / (8.0 * np.pi)
    return pre * np.exp(-a * np.asarray(r, dtype=float))


@dataclass
class MBISAtomModel:
    """Per-atom MBIS parameters."""

    atomic_number: int
    populations: np.ndarray  # (n_shell,)
    exponents: np.ndarray  # (n_shell,)

    @property
    def n_shell(self) -> int:
        return int(self.populations.size)

    @property
    def population(self) -> float:
        return float(np.sum(self.populations))

    @property
    def charge(self) -> float:
        return float(self.atomic_number) - self.population


@dataclass
class MBISResult:
    """Converged MBIS stockholder partition on a flat grid."""

    models: list
    weights: np.ndarray  # (n_atom, n_grid)
    rho_atoms: np.ndarray  # (n_atom, n_grid)
    rho_tilde: np.ndarray  # (n_atom, n_grid)
    populations: np.ndarray  # (n_atom,)
    charges: np.ndarray  # (n_atom,)
    n_iter: int
    converged: bool
    history: list = field(default_factory=list)

    def active_env_partition(
        self,
        rho_tot: np.ndarray,
        active_atoms: Sequence[int],
    ) -> HirshfeldPartition:
        """Collapse MBIS weights into a Hirshfeld-like active / env split."""
        rho = np.asarray(rho_tot, dtype=float).ravel()
        active = [int(i) for i in active_atoms]
        ra = np.sum(self.rho_tilde[active], axis=0)
        rs = np.sum(self.rho_tilde, axis=0)
        w_a = np.sum(self.weights[active], axis=0)
        rho_active = w_a * rho
        return HirshfeldPartition(
            rho_tot=rho,
            rho_tilde_a=ra,
            rho_tilde_sum=rs,
            w_a=w_a,
            rho_active=rho_active,
            rho_env=np.maximum(rho - rho_active, 0.0),
        )


def _grid_distances(
    positions: np.ndarray,
    grid_coords: np.ndarray,
    cell: Optional[np.ndarray] = None,
) -> np.ndarray:
    """``(n_atom, n_grid)`` distances; MIC when ``cell`` is given."""
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    coords = np.asarray(grid_coords, dtype=float).reshape(-1, 3)
    n_atom = pos.shape[0]
    n_grid = coords.shape[0]
    dist = np.empty((n_atom, n_grid), dtype=float)
    if cell is None:
        for ia in range(n_atom):
            d = coords - pos[ia]
            dist[ia] = np.linalg.norm(d, axis=1)
        return dist

    from casidapy.utils.uspp import _minimum_image_displacement

    cell = np.asarray(cell, dtype=float).reshape(3, 3)
    # uspp helper expects (3, n) stacked cartesian arrays.
    xyz = coords.T.copy()
    for ia in range(n_atom):
        dx, dy, dz = _minimum_image_displacement(pos[ia], xyz, cell)
        dist[ia] = np.sqrt(dx * dx + dy * dy + dz * dz)
    return dist


def _promolecule_from_models(
    models: Sequence[MBISAtomModel],
    distances: np.ndarray,
) -> tuple[np.ndarray, list[list[np.ndarray]]]:
    """Return ``ρ̃_A`` and per-atom list of shell densities ``ρ̃_{Ak}``."""
    n_atom = len(models)
    n_grid = distances.shape[1]
    rho_tilde = np.zeros((n_atom, n_grid), dtype=float)
    shells: list[list[np.ndarray]] = []
    for ia, model in enumerate(models):
        shell_list: list[np.ndarray] = []
        r = distances[ia]
        for N, alpha in zip(model.populations, model.exponents):
            rho_k = slater_s_density(r, float(N), float(alpha))
            shell_list.append(rho_k)
            rho_tilde[ia] += rho_k
        shells.append(shell_list)
    return rho_tilde, shells


def mbis_partition(
    rho_tot: np.ndarray,
    positions: np.ndarray,
    atomic_numbers: Sequence[int],
    grid_coords: np.ndarray,
    dV: float,
    *,
    cell: Optional[np.ndarray] = None,
    max_iter: int = DEFAULT_MBIS_MAX_ITER,
    tol: float = DEFAULT_MBIS_TOL,
    mix: float = DEFAULT_MBIS_MIX,
    eps: float = 1.0e-30,
) -> MBISResult:
    """Run MBIS on a molecular density sampled on ``grid_coords``.

    Parameters
    ----------
    rho_tot
        Density (e⁻/Bohr³), shape ``(n_grid,)``.
    positions
        Atomic positions in Bohr, ``(n_atom, 3)``.
    atomic_numbers
        Nuclear charges ``Z_A`` (use valence ``Z`` for PP valence densities).
    grid_coords
        Cartesian grid points in Bohr, ``(n_grid, 3)``.
    dV
        Volume element (Bohr³).
    cell
        Optional ``(3, 3)`` lattice for minimum-image distances.
    """
    rho = np.asarray(rho_tot, dtype=float).ravel()
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    z_list = [int(z) for z in atomic_numbers]
    n_atom = pos.shape[0]
    if len(z_list) != n_atom:
        raise ValueError(
            f"atomic_numbers length {len(z_list)} != n_atom {n_atom}"
        )
    coords = np.asarray(grid_coords, dtype=float).reshape(-1, 3)
    if coords.shape[0] != rho.size:
        raise ValueError(
            f"grid_coords rows {coords.shape[0]} != rho_tot size {rho.size}"
        )
    dV = float(dV)
    if dV <= 0.0:
        raise ValueError(f"dV must be > 0, got {dV}")

    models = [
        MBISAtomModel(
            atomic_number=z,
            populations=_initial_shell_populations(z, n_mbis_shells(z)),
            exponents=_initial_exponents(z, n_mbis_shells(z)),
        )
        for z in z_list
    ]
    distances = _grid_distances(pos, coords, cell=cell)
    mix = float(np.clip(mix, 0.0, 1.0))
    tol = float(tol)
    max_iter = max(1, int(max_iter))

    history: list[dict] = []
    converged = False
    rho_tilde = np.zeros((n_atom, rho.size), dtype=float)
    w = np.zeros_like(rho_tilde)
    rho_atoms = np.zeros_like(rho_tilde)
    pop = np.zeros(n_atom, dtype=float)

    for it in range(max_iter):
        rho_tilde, shells = _promolecule_from_models(models, distances)
        denom = np.maximum(rho_tilde.sum(axis=0), float(eps))
        w = rho_tilde / denom[None, :]
        rho_atoms = w * rho[None, :]
        pop = np.array([float(np.sum(rho_atoms[ia]) * dV) for ia in range(n_atom)])

        # Parameter updates (Verstraelen MBIS).
        delta = 0.0
        for ia, model in enumerate(models):
            rho_a = rho_atoms[ia]
            rho_ta = np.maximum(rho_tilde[ia], float(eps))
            r = distances[ia]
            new_N = np.empty(model.n_shell, dtype=float)
            new_a = np.empty(model.n_shell, dtype=float)
            for k, rho_k in enumerate(shells[ia]):
                share = rho_k / rho_ta
                n_k = float(np.sum(rho_a * share) * dV)
                n_k = max(n_k, _N_MIN)
                r_mom = float(np.sum(rho_a * share * r) * dV)
                if r_mom < 1.0e-30:
                    a_k = float(model.exponents[k])
                else:
                    a_k = 3.0 * n_k / r_mom
                a_k = float(np.clip(a_k, _ALPHA_MIN, _ALPHA_MAX))
                new_N[k] = n_k
                new_a[k] = a_k
            delta = max(
                delta,
                float(np.max(np.abs(new_N - model.populations))),
                float(np.max(np.abs(new_a - model.exponents))),
            )
            if mix < 1.0:
                new_N = (1.0 - mix) * model.populations + mix * new_N
                new_a = (1.0 - mix) * model.exponents + mix * new_a
            model.populations = new_N
            model.exponents = new_a

        charges = np.array([m.charge for m in models], dtype=float)
        history.append(
            {
                "iter": it,
                "populations": pop.copy(),
                "charges": charges.copy(),
                "delta": delta,
            }
        )
        if delta < tol:
            converged = True
            break

    # Final pass with last parameters.
    rho_tilde, _ = _promolecule_from_models(models, distances)
    denom = np.maximum(rho_tilde.sum(axis=0), float(eps))
    w = rho_tilde / denom[None, :]
    rho_atoms = w * rho[None, :]
    pop = np.array([float(np.sum(rho_atoms[ia]) * dV) for ia in range(n_atom)])
    charges = np.array([float(z_list[ia]) - pop[ia] for ia in range(n_atom)])

    if not converged:
        logger.warning(
            "[MBIS] not converged in %d iters (last Δ=%.3e, tol=%.3e)",
            max_iter,
            history[-1]["delta"] if history else float("nan"),
            tol,
        )
    return MBISResult(
        models=models,
        weights=w,
        rho_atoms=rho_atoms,
        rho_tilde=rho_tilde,
        populations=pop,
        charges=charges,
        n_iter=len(history),
        converged=converged,
        history=history,
    )


def mbis_hirshfeld_partition(
    rho_tot: np.ndarray,
    positions: np.ndarray,
    atomic_numbers: Sequence[int],
    grid_coords: np.ndarray,
    dV: float,
    *,
    active_atoms: Optional[Sequence[int]] = None,
    cell: Optional[np.ndarray] = None,
    max_iter: int = DEFAULT_MBIS_MAX_ITER,
    tol: float = DEFAULT_MBIS_TOL,
    mix: float = DEFAULT_MBIS_MIX,
) -> tuple[HirshfeldPartition, MBISResult]:
    """MBIS partition collapsed to active-site ``ρ_active`` / ``ρ_env``."""
    result = mbis_partition(
        rho_tot,
        positions,
        atomic_numbers,
        grid_coords,
        dV,
        cell=cell,
        max_iter=max_iter,
        tol=tol,
        mix=mix,
    )
    n_atom = len(result.models)
    active = (
        tuple(range(n_atom))
        if active_atoms is None
        else tuple(int(i) for i in active_atoms)
    )
    part = result.active_env_partition(rho_tot, active)
    return part, result


def mbis_from_dftpy_grid(
    rho_tot: np.ndarray,
    grid,
    positions: np.ndarray,
    atomic_numbers: Sequence[int],
    *,
    cell: Optional[np.ndarray] = None,
    active_atoms: Optional[Sequence[int]] = None,
    max_iter: int = DEFAULT_MBIS_MAX_ITER,
    tol: float = DEFAULT_MBIS_TOL,
    mix: float = DEFAULT_MBIS_MIX,
) -> tuple[HirshfeldPartition, MBISResult]:
    """Convenience wrapper: build ``(n_grid, 3)`` coords from a DFTpy grid."""
    # DFTpy grids store Fortran-ordered flat densities in this codebase.
    nr = tuple(int(x) for x in grid.nr)
    rho = np.asarray(rho_tot, dtype=float).ravel()
    if rho.size != int(np.prod(nr)):
        raise ValueError(f"rho_tot size {rho.size} != prod(nr)={np.prod(nr)}")
    x = np.asarray(grid.r[0], float).reshape(nr, order="F").ravel(order="F")
    y = np.asarray(grid.r[1], float).reshape(nr, order="F").ravel(order="F")
    z = np.asarray(grid.r[2], float).reshape(nr, order="F").ravel(order="F")
    coords = np.column_stack([x, y, z])
    cell_use = cell
    if cell_use is None and getattr(grid, "lattice", None) is not None:
        cell_use = np.asarray(grid.lattice, dtype=float)
    return mbis_hirshfeld_partition(
        rho,
        positions,
        atomic_numbers,
        coords,
        float(grid.dV),
        active_atoms=active_atoms,
        cell=cell_use,
        max_iter=max_iter,
        tol=tol,
        mix=mix,
    )
