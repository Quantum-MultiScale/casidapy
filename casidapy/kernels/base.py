"""Kernel backend contract for CasidaPy (plane-wave or GTO).

Two layers are provided:

* :class:`KernelBackend` — a ``runtime_checkable`` structural ``Protocol``
  describing the minimal surface the Casida algebra relies on. Kept for
  back-compat / duck-typed checks.
* :class:`CasidaKernel` — an abstract base class both concrete kernels inherit.
  It supplies the *shared* pieces that were previously duplicated in
  ``GTOKernel`` and ``PlaneWaveKernel`` (transition counts, ``diagonal_dE``,
  the singlet/triplet ``dipole_matrix`` policy) plus **capability flags** the
  engine dispatches on instead of ``isinstance`` / class-name checks.

Concrete kernels implement the backend-specific numerics (``setup``,
``apply_K``, ``dense_K_rows``) and one transition-dipole hook
(``_transition_dipole_blocks``); everything else is inherited.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from casidapy.utils.casida_utils import build_energy_differences


@runtime_checkable
class KernelBackend(Protocol):
    """Basis-specific Casida coupling kernel ``K``.

    The shared Casida algebra (TDA/RPA eigensolve, oscillator strengths) calls
    ``apply_K`` and ``dipole_matrix`` without knowing whether the underlying
    representation is a plane-wave FFT grid or a GTO AO/MO basis.
    """

    @property
    def n_occ(self) -> int: ...

    @property
    def n_unocc(self) -> int: ...

    @property
    def n_trans(self) -> int: ...

    def diagonal_dE(self) -> np.ndarray:
        """Orbital energy differences ``Δε_ia`` (length ``n_trans``)."""
        ...

    def apply_K(self, v: np.ndarray) -> np.ndarray:
        """Apply the eh-coupling block ``K`` to ``v``.

        ``v`` may be ``(n_trans,)`` or a LOBPCG block ``(n_trans, k)``.
        Backends may also expose ``apply_K_matmat`` for an explicit batch path.
        """
        ...

    def dipole_matrix(self) -> np.ndarray:
        """Transition dipoles ``μ_ia`` with shape ``(n_trans, 3)``."""
        ...

    def setup(self, tda: bool = False) -> None:
        """Prepare caches for matrix-free (or dense) evaluation."""
        ...


class CasidaKernel(ABC):
    """Abstract base for Casida ``K`` backends with shared bookkeeping.

    Subclasses populate the active-space state contract
    (``_occ_e``, ``_unocc_e``, ``_n_occ``, ``_n_unocc``, ``_n_trans``,
    ``_dE``) in their ``set_active_orbitals`` / ``set_core_active_space``
    method and implement the abstract numeric hooks. The properties and the
    ``dipole_matrix`` template below are then inherited unchanged by both the
    plane-wave and GTO kernels.

    Capability flags (class attributes; override per backend) let the engine
    branch on *behavior* rather than concrete type:

    ``provides_grid``
        Backend carries a real-space density/FFT grid (``rho``/``grid``) —
        ``True`` for the plane-wave kernel, ``False`` for GTO.
    ``distributes_over_comm``
        Backend participates in MPI ``COMM_WORLD`` collectives — ``True`` for
        the plane-wave kernel; GTO parallelism is delegated to PySCF/mpi4pyscf.
    ``hybrid``
        Set to ``True`` once a hybrid-exchange coupling is active (GTO updates
        this in ``setup``); gates matrix-free RPA.
    """

    # --- capability flags (subclasses override) ---
    provides_grid: bool = False
    distributes_over_comm: bool = False
    #: Hybrid exact-exchange coupling active (A-B ≠ diag(Δε)); set in setup().
    hybrid: bool = False
    #: Spin manifold; a triplet/spin-flip block is electric-dipole forbidden.
    triplet: bool = False

    @property
    def n_occ(self) -> int:
        return self._n_occ

    @property
    def n_unocc(self) -> int:
        return self._n_unocc

    @property
    def n_trans(self) -> int:
        return self._n_trans

    @property
    def supports_matrix_free_rpa(self) -> bool:
        """Full (non-TDA) RPA is available matrix-free unless hybrid.

        For hybrids ``A - B`` is not ``diag(Δε)``, so the fast C-chain does not
        apply and the engine must build dense ``A``/``B`` instead.
        """
        return not bool(getattr(self, "hybrid", False))

    def diagonal_dE(self) -> np.ndarray:
        """Cached orbital-energy differences ``Δε_ia`` (length ``n_trans``)."""
        if getattr(self, "_dE", None) is None:
            if getattr(self, "_occ_e", None) is None:
                raise RuntimeError(
                    "Active space not set; call set_active_orbitals() / "
                    "set_core_active_space() before diagonal_dE()."
                )
            self._dE = build_energy_differences(self._occ_e, self._unocc_e)
        return self._dE

    def dipole_matrix(
        self, origin: Sequence[float] = (0.0, 0.0, 0.0)
    ) -> np.ndarray:
        """Singlet transition dipoles ``μ_ia`` with shape ``(n_trans, 3)``.

        Shared policy: triplet and spin-flip blocks are electric-dipole
        forbidden (return zeros); the singlet spin-adaptation factor ``√2`` is
        applied here so backends only supply the raw ``⟨i|r|a⟩`` blocks via
        :meth:`_transition_dipole_blocks`.
        """
        if bool(getattr(self, "triplet", False)) or bool(
            getattr(self, "_spin_flip", False)
        ):
            return np.zeros((self._n_trans, 3), dtype=float)
        return np.sqrt(2.0) * self._transition_dipole_blocks(origin)

    # --- backend-specific numerics ---
    @abstractmethod
    def setup(self, tda: bool = False) -> None:
        """Prepare caches for matrix-free (or dense) evaluation."""

    @abstractmethod
    def apply_K(self, v: np.ndarray) -> np.ndarray:
        """Apply the eh-coupling block ``K`` to ``v`` (``(n_trans,)`` or block)."""

    @abstractmethod
    def dense_K_rows(
        self, row_indices: Sequence[int], verbose: bool = False
    ) -> np.ndarray:
        """Explicit rows ``K[ia, :]`` of the dense coupling (MPI dense path)."""

    @abstractmethod
    def _transition_dipole_blocks(
        self, origin: Sequence[float]
    ) -> np.ndarray:
        """Raw ``⟨i|r|a⟩`` transition dipoles ``(n_trans, 3)``, **pre-√2**.

        The singlet ``√2`` factor and triplet/spin-flip zeroing are applied by
        :meth:`dipole_matrix`; backends only evaluate the position integrals.
        """
