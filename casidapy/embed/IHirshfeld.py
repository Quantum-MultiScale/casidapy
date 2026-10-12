"""Iterative Hirshfeld (Hirshfeld-I) density partition.

Standard stockholder weights use a *frozen* promolecule ``ρ̃_A``.  Hirshfeld-I
(Bultinck et al.) updates those references until the atomic populations are
self-consistent with the references::

    w_A^{(k)}(r) = ρ̃_A^{(k)}(r) / Σ_B ρ̃_B^{(k)}(r)
    ρ_A^{(k)}    = w_A^{(k)} · ρ_tot
    N_A^{(k)}    = ∫ ρ_A^{(k)}
    ρ̃_A^{(k+1)} ← reference density carrying population N_A^{(k)}

This module is grid-native and independent of the production one-shot
:mod:`casidapy.embed.hirshfeld` path.  Default reference updates use
**shape-preserving charge scaling** of the seed promolecule (frozen radial
shape, rescaled norm).  Pass ``ref_updater`` for true charge-dependent atomic
DFT rebuilds.

Typical use::

    from casidapy.embed.IHirshfeld import iterative_hirshfeld_partition
    part = iterative_hirshfeld_partition(rho_tot, rho_tilde0, dV)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from casidapy.embed.hirshfeld import HirshfeldPartition

logger = logging.getLogger(__name__)

DEFAULT_IH_MAX_ITER = 50
DEFAULT_IH_TOL = 1.0e-6
DEFAULT_IH_MIX = 0.5


RefUpdater = Callable[
    [np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
]
"""``(rho_tilde, populations, charges) -> rho_tilde_new`` with shapes
``(n_atom, n_grid)``, ``(n_atom,)``, ``(n_atom,)``."""


@dataclass
class IterativeHirshfeldResult:
    """Converged (or last) Hirshfeld-I partition."""

    partition: HirshfeldPartition
    populations: np.ndarray
    charges: np.ndarray
    rho_tilde: np.ndarray  # (n_atom, n_grid) final references
    weights: np.ndarray  # (n_atom, n_grid)
    n_iter: int
    converged: bool
    history: list = field(default_factory=list)
    active_atoms: tuple = ()

    @property
    def rho_active(self) -> np.ndarray:
        return self.partition.rho_active

    @property
    def rho_env(self) -> np.ndarray:
        return self.partition.rho_env

    @property
    def w_a(self) -> np.ndarray:
        return self.partition.w_a


def _as_atom_grid(rho_tilde0: np.ndarray) -> np.ndarray:
    arr = np.asarray(rho_tilde0, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"rho_tilde0 must be (n_grid,) or (n_atom, n_grid), got {arr.shape}")
    return arr


def stockholder_weights(rho_tilde: np.ndarray, *, eps: float = 1.0e-30) -> np.ndarray:
    """``w_A = ρ̃_A / Σ_B ρ̃_B`` for ``rho_tilde`` shaped ``(n_atom, n_grid)``."""
    rt = np.asarray(rho_tilde, dtype=float)
    if rt.ndim != 2:
        raise ValueError(f"expected (n_atom, n_grid), got {rt.shape}")
    denom = np.maximum(rt.sum(axis=0), float(eps))
    return rt / denom[None, :]


def scale_promolecule_to_populations(
    rho_tilde0: np.ndarray,
    populations: np.ndarray,
    *,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Rescale each seed ``ρ̃_A`` so ``∫ρ̃_A = N_A`` (shape frozen)."""
    rt0 = _as_atom_grid(rho_tilde0)
    pop = np.asarray(populations, dtype=float).ravel()
    if pop.size != rt0.shape[0]:
        raise ValueError(
            f"populations length {pop.size} != n_atom {rt0.shape[0]}"
        )
    out = np.empty_like(rt0)
    for ia in range(rt0.shape[0]):
        n0 = float(np.sum(rt0[ia]))
        if n0 < float(eps) or pop[ia] < 0.0:
            out[ia] = 0.0
        else:
            out[ia] = rt0[ia] * (float(pop[ia]) / n0)
    return out


def make_charge_scale_updater(rho_tilde0: np.ndarray) -> RefUpdater:
    """IH updater: always rescale the *seed* promolecule to current ``N_A``."""
    seed = _as_atom_grid(rho_tilde0).copy()

    def _update(rho_tilde, populations, charges):
        _ = rho_tilde, charges
        return scale_promolecule_to_populations(seed, populations)

    return _update


def iterative_hirshfeld_partition(
    rho_tot: np.ndarray,
    rho_tilde0: np.ndarray,
    dV: float,
    *,
    nuclear_charges: Optional[Sequence[float]] = None,
    active_atoms: Optional[Sequence[int]] = None,
    max_iter: int = DEFAULT_IH_MAX_ITER,
    tol: float = DEFAULT_IH_TOL,
    mix: float = DEFAULT_IH_MIX,
    ref_updater: Optional[RefUpdater] = None,
    eps: float = 1.0e-30,
) -> IterativeHirshfeldResult:
    """Run Hirshfeld-I on a flat charge grid.

    Parameters
    ----------
    rho_tot
        Molecular density on the grid (e⁻/Bohr³), shape ``(n_grid,)``.
    rho_tilde0
        Seed promolecular atomic densities ``(n_atom, n_grid)`` or a single
        ``(n_grid,)`` atom.
    dV
        Grid volume element (Bohr³).
    nuclear_charges
        ``Z_A`` (or valence ``Z_val`` for PP densities) used for
        ``q_A = Z_A − N_A``.  Defaults to the seed populations
        (so initial charges are zero).
    active_atoms
        Indices summed into the active-site weight / ``ρ_active`` (default: all).
    max_iter, tol
        Stop when ``max|N_A^{(k)} − N_A^{(k−1)}| < tol``.
    mix
        ``ρ̃ ← (1−mix) ρ̃_old + mix ρ̃_new`` (``mix=1``: undamped).
    ref_updater
        Maps ``(ρ̃, N, q) → ρ̃_new``.  Default: shape-preserving scale of the seed.
    """
    rho = np.asarray(rho_tot, dtype=float).ravel()
    rt = _as_atom_grid(rho_tilde0).copy()
    n_atom, n_grid = rt.shape
    if rho.size != n_grid:
        raise ValueError(f"rho_tot size {rho.size} != n_grid {n_grid}")
    dV = float(dV)
    if dV <= 0.0:
        raise ValueError(f"dV must be > 0, got {dV}")

    if nuclear_charges is None:
        z = np.array([float(np.sum(rt[ia]) * dV) for ia in range(n_atom)])
    else:
        z = np.asarray(nuclear_charges, dtype=float).ravel()
        if z.size != n_atom:
            raise ValueError(f"nuclear_charges length {z.size} != n_atom {n_atom}")

    if active_atoms is None:
        active = tuple(range(n_atom))
    else:
        active = tuple(int(i) for i in active_atoms)
        for ia in active:
            if not (0 <= ia < n_atom):
                raise ValueError(f"active atom {ia} out of range [0, {n_atom})")

    updater = ref_updater if ref_updater is not None else make_charge_scale_updater(rt)
    mix = float(np.clip(mix, 0.0, 1.0))
    tol = float(tol)
    max_iter = max(1, int(max_iter))

    pop_prev = None
    history: list[dict] = []
    converged = False
    w = stockholder_weights(rt, eps=eps)
    pop = np.zeros(n_atom, dtype=float)

    for it in range(max_iter):
        w = stockholder_weights(rt, eps=eps)
        rho_atoms = w * rho[None, :]
        pop = np.array([float(np.sum(rho_atoms[ia]) * dV) for ia in range(n_atom)])
        charges = z - pop
        delta = (
            float("inf")
            if pop_prev is None
            else float(np.max(np.abs(pop - pop_prev)))
        )
        history.append(
            {
                "iter": it,
                "populations": pop.copy(),
                "charges": charges.copy(),
                "delta_pop": delta,
            }
        )
        if pop_prev is not None and delta < tol:
            converged = True
            break

        rt_new = np.asarray(updater(rt, pop, charges), dtype=float)
        if rt_new.shape != rt.shape:
            raise ValueError(
                f"ref_updater returned shape {rt_new.shape}, expected {rt.shape}"
            )
        rt_new = np.maximum(rt_new, 0.0)
        if mix < 1.0:
            rt = (1.0 - mix) * rt + mix * rt_new
        else:
            rt = rt_new
        pop_prev = pop.copy()

    # Final weights / partition from last ``rt`` (one more stockholder pass if
    # we exited by max_iter after an update, recompute; if converged, already ok).
    w = stockholder_weights(rt, eps=eps)
    rho_atoms = w * rho[None, :]
    pop = np.array([float(np.sum(rho_atoms[ia]) * dV) for ia in range(n_atom)])
    charges = z - pop

    rho_tilde_a = np.sum(rt[list(active), :], axis=0)
    rho_tilde_sum = np.sum(rt, axis=0)
    w_a = np.sum(w[list(active), :], axis=0)
    rho_active = w_a * rho
    rho_env = np.maximum(rho - rho_active, 0.0)
    part = HirshfeldPartition(
        rho_tot=rho,
        rho_tilde_a=rho_tilde_a,
        rho_tilde_sum=rho_tilde_sum,
        w_a=w_a,
        rho_active=rho_active,
        rho_env=rho_env,
    )
    if not converged:
        logger.warning(
            "[IHirshfeld] not converged in %d iters (last ΔN=%.3e, tol=%.3e)",
            max_iter,
            history[-1]["delta_pop"] if history else float("nan"),
            tol,
        )
    return IterativeHirshfeldResult(
        partition=part,
        populations=pop,
        charges=charges,
        rho_tilde=rt,
        weights=w,
        n_iter=len(history),
        converged=converged,
        history=history,
        active_atoms=active,
    )


def iterative_hirshfeld_from_driver(
    driver,
    edge_atom,
    grid=None,
    *,
    active_atoms=None,
    upf_map=None,
    hirshfeld_reference: str = "neutral",
    oxidation_state=None,
    oxidation_states=None,
    formal_ion_basis: str = "aug-cc-pvdz",
    formal_ion_xc: str = "pbe",
    formal_ion_valence_only: bool = True,
    use_valence_charges: bool = True,
    max_iter: int = DEFAULT_IH_MAX_ITER,
    tol: float = DEFAULT_IH_TOL,
    mix: float = DEFAULT_IH_MIX,
    use_gpu: bool = False,
    comm=None,
) -> IterativeHirshfeldResult:
    """Hirshfeld-I starting from the same seed refs as one-shot Hirshfeld.

    Builds neutral / formal-ion UPF-style ``ρ̃_A`` on the QE charge grid, then
    iterates with charge-scaled updates.  ``nuclear_charges`` default to UPF
    ``z_valence`` when ``use_valence_charges`` (matches valence ``ρ_tot``).
    """
    from casidapy.adapter.qepy import build_uspp_map_from_driver
    from casidapy.embed.hirshfeld import (
        _normalize_hirshfeld_reference,
        total_density_on_grid,
        upf_rhoatom_on_grid,
    )
    from casidapy.embed.potential import (
        ensure_charge_grid,
        read_upf_local_potential,
        resolve_upf_path,
    )
    from casidapy.utils.accel import distributed_indices, mpi_allreduce_sum

    _ = use_gpu  # partition arithmetic is NumPy here; kept for API symmetry
    grid = grid if grid is not None else ensure_charge_grid(driver)
    edge_atom = int(edge_atom)
    symbols = driver.get_ase_atoms().get_chemical_symbols()
    n_at = len(symbols)
    active = [int(i) for i in (active_atoms if active_atoms is not None else [edge_atom])]
    if not active:
        active = [edge_atom]

    if upf_map is None:
        upf_map = build_uspp_map_from_driver(driver)

    ref_mode = _normalize_hirshfeld_reference(hirshfeld_reference)
    ox_resolver = None
    if ref_mode == "formal_ion":
        from casidapy.embed.formal_ion_density import formal_ion_rhoatom_on_grid
        from casidapy.xas.cvs import resolve_oxidation_state

        def ox_resolver(ia: int) -> int:
            sym_i = str(symbols[ia]).strip().capitalize()
            ox_edge = oxidation_state if ia == edge_atom else None
            return resolve_oxidation_state(sym_i, ox_edge, oxidation_states)

    rho_tot = total_density_on_grid(driver, grid=grid)
    n_grid = int(rho_tot.size)
    rt_loc = np.zeros((n_at, n_grid), dtype=float)
    z_loc = np.zeros(n_at, dtype=float)

    for ia in distributed_indices(n_at, comm):
        sym_raw = symbols[ia]
        sym = str(sym_raw).strip().capitalize()
        path = None
        if upf_map:
            path = upf_map.get(sym) or upf_map.get(sym_raw)
        path = resolve_upf_path(driver, sym, path)
        _, _, z_val = read_upf_local_potential(path)
        if use_valence_charges:
            z_loc[ia] = float(z_val)
        else:
            from pyscf.data import elements

            z_loc[ia] = float(elements.charge(sym))
        if ref_mode == "formal_ion":
            rt_loc[ia] = formal_ion_rhoatom_on_grid(
                driver,
                ia,
                grid,
                ox_resolver(ia),
                upf_path=path,
                basis=formal_ion_basis,
                xc=formal_ion_xc,
                valence_only=bool(formal_ion_valence_only),
            )
        else:
            rt_loc[ia] = upf_rhoatom_on_grid(driver, ia, grid, upf_path=path)

    rt = mpi_allreduce_sum(comm, rt_loc)
    z = mpi_allreduce_sum(comm, z_loc)
    return iterative_hirshfeld_partition(
        rho_tot,
        rt,
        float(grid.dV),
        nuclear_charges=z,
        active_atoms=active,
        max_iter=max_iter,
        tol=tol,
        mix=mix,
    )
