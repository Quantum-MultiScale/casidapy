"""Kernel backend protocol for CasidaPy (plane-wave or GTO)."""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np


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
