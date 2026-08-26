"""Lightweight CuPy / MPI helpers for embed and XAS paths.

Design
------
- ``use_gpu=True`` selects CuPy when installed; otherwise falls back to NumPy
  with a one-time warning (never hard-fails).
- ``comm=None`` is fully serial. Pass an ``mpi4py`` communicator to distribute
  independent work (Hirshfeld atom loop, AO-grid blocks) and ``Allreduce``.
- QE driver / ``setlocal`` calls stay on the host; only array math moves to GPU.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_GPU_WARNED = False


def cupy_available() -> bool:
    try:
        import cupy  # noqa: F401

        return True
    except Exception:
        return False


def array_module(use_gpu: bool = False):
    """Return ``cupy`` if requested and importable, else ``numpy``."""
    global _GPU_WARNED
    if not use_gpu:
        return np
    try:
        import cupy as cp

        return cp
    except Exception as exc:
        if not _GPU_WARNED:
            warnings.warn(
                f"use_gpu=True but CuPy unavailable ({exc}); using NumPy.",
                RuntimeWarning,
                stacklevel=2,
            )
            _GPU_WARNED = True
        return np


def asnumpy(a) -> np.ndarray:
    """Host ``ndarray`` from NumPy / CuPy / objects with ``.get()``."""
    if a is None:
        return a
    if isinstance(a, np.ndarray):
        return np.asarray(a)
    get = getattr(a, "get", None)
    if callable(get):
        try:
            return np.asarray(get())
        except Exception:
            pass
    try:
        import cupy as cp

        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:
        pass
    return np.asarray(a)


def asarray(a, xp=None, dtype=float):
    """``xp.asarray`` with ``xp`` defaulting to NumPy."""
    if xp is None:
        xp = np
    return xp.asarray(a, dtype=dtype)


def resolve_mpi_comm(comm=None, *, allow_world: bool = False):
    """Return ``comm``, or ``COMM_WORLD`` if ``allow_world`` and MPI is up.

    Default is conservative: do **not** touch ``COMM_WORLD`` unless asked
    (login-node Open MPI / mpi4pyscf worker pools).
    """
    if comm is not None:
        return comm
    if not allow_world:
        return None
    try:
        from mpi4py import MPI

        if MPI.Is_initialized() and MPI.COMM_WORLD.Get_size() > 1:
            return MPI.COMM_WORLD
    except Exception:
        pass
    return None


def mpi_rank_size(comm) -> Tuple[int, int]:
    if comm is None:
        return 0, 1
    from casidapy.utils.casida_utils import mpi_comm_rank, mpi_comm_size

    return int(mpi_comm_rank(comm)), int(mpi_comm_size(comm))


def mpi_is_root(comm) -> bool:
    return mpi_rank_size(comm)[0] == 0


def mpi_barrier(comm) -> None:
    if comm is not None and hasattr(comm, "Barrier"):
        comm.Barrier()


def mpi_bcast(comm, obj, root: int = 0):
    if comm is None:
        return obj
    return comm.bcast(obj, root=root)


def mpi_allreduce_sum(comm, arr: np.ndarray) -> np.ndarray:
    """In-place-safe SUM allreduce; returns a new contiguous float array."""
    out = np.ascontiguousarray(arr, dtype=float)
    if comm is None:
        return out
    from mpi4py import MPI

    buf = np.empty_like(out)
    comm.Allreduce(out, buf, op=MPI.SUM)
    return buf


def distributed_indices(n: int, comm) -> range:
    """Round-robin index ownership for ``0 … n-1`` under ``comm``."""
    rank, size = mpi_rank_size(comm)
    if size <= 1:
        return range(n)
    return range(rank, n, size)


def block_slices(n: int, blksize: int, comm=None) -> Sequence[Tuple[int, int]]:
    """``(p0, p1)`` slices of ``[0, n)``, optionally owned by this MPI rank."""
    blksize = max(int(blksize), 1)
    slices = [(p0, min(p0 + blksize, n)) for p0 in range(0, n, blksize)]
    if comm is None:
        return slices
    rank, size = mpi_rank_size(comm)
    if size <= 1:
        return slices
    return [s for i, s in enumerate(slices) if i % size == rank]
