"""
Shared datatypes for the STDDFT Casida driver: options, SCF/KS inputs, and returned spectra.

``CasidaOptions`` controls the active orbital window (via ``n_occ``, ``n_unocc``, ``n_total_occ``),
solver path (dense vs matrix-free), and XC / spin settings consumed by ``casida_engine``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np


ArrayLike = Union[np.ndarray, Sequence[float]]


@dataclass
class CasidaOptions:
    """User knobs for ``run_casida_in_memory`` / ``CasidaKS_MPI``."""

    # Active space: n_occ × n_unocc transitions; n_states caps how many roots are requested.
    n_occ: int
    n_unocc: int
    n_states: int = 50
    # If set: HOMO count (LUMO index). Occupied window is the n_occ bands below this index;
    # if None, the lowest n_occ bands are used (see qepy_adapter.slice_active_space).
    n_total_occ: Optional[int] = None

    # Kernel backend: "pw" = plane-wave / FFT grid (QE, DFTpy); "gto" = molecular AO/MO.
    basis: str = "pw"

    tda: bool = False
    matrix_free: bool = False
    solver_method: str = "lobpcg"  # matrix-free iterative diagonalization (lobpcg or eigsh)
    solver_tol: float = 1e-8
    solver_maxiter: int = 200
    use_gpu: bool = False  # PW: CuPy FFT Hartree; GTO: CuPy matvecs (+ gpu4pyscf response)

    xc: str = "PBE"
    spin_state: str = "singlet"  # or "triplet" (requires libxc component map in the engine)

    # When True, rebuild TotalFunctional from an orbital-free-style context (needs pseudo_map).
    of_context: bool = False
    pseudo_map: Optional[Dict[str, str]] = None

    # USPP: when True, USPP augmentation is applied in the Casida matrix elements.
    use_uspp: bool = False
    uspp_map: Optional[Dict[str, str]] = None

    # When True, store transition densities as class variables for eDFTpy coupling.
    use_eDFTpy: bool = False

    tag: Optional[str] = None  # optional label for logging or downstream bookkeeping


@dataclass
class CasidaInputs:
    """Ground-state objects on the DFTpy grid passed into ``run_casida_in_memory``."""

    atoms: Any  # ase.Atoms
    grid: Any  # DirectGrid
    rho_ks: Any  # converged GS density field
    psi: Any  # ndarray (n_band, nx, ny, nz) or list of DirectField
    eigs: np.ndarray
    occs: np.ndarray

    # USPP data (optional, loaded externally via casidapy.utils.uspp.load_uspp_data)
    beta_projectors: Optional[Any] = None
    qij_augmentation: Optional[Any] = None


@dataclass
class CasidaResults:
    """Casida output: transition energies, normalized oscillator strengths, and eigenvectors."""

    omega: np.ndarray  # excitation energies (Hartree)
    f: np.ndarray  # oscillator strengths (sum normalized to 1 when non-trivial)
    Z: np.ndarray  # Casida eigenvectors in the transition basis
    mu_transition: Optional[np.ndarray]  # dipole matrix in the active space, if computed
    rho_transition: Optional[np.ndarray]  # (n_states, nx, ny, nz) state-basis grids, if use_eDFTpy
    d_mode: Optional[np.ndarray]  # state-basis transition dipoles (3, n_states): mu.T @ (X+Y)
    xpy: Optional[np.ndarray]    # physical Casida amplitudes (n_trans, n_states): Z/sqrt(dE) for RPA, Z for TDA
    metadata: Dict[str, Any]  # timings, n_total_occ, n_trans, flags, etc.