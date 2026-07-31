"""Optional mpi4pyscf integration for MPI-parallel ``gen_response`` / ``get_jk``.

Architecture (mpi4pyscf master/worker pool)
------------------------------------------
Importing ``mpi4pyscf`` parks all non-master ranks in a worker pool. Only the
master continues the Python script (Davidson, plotting, …). Collective
``get_jk`` / ``get_veff`` called from the master's ``gen_response`` wake the
workers — this is **not** SPMD Davidson across ranks.

Use with::

    from casidapy.mpi_pyscf import enable_mpi4pyscf, make_mpi_rks

    enable_mpi4pyscf()          # must be early; workers never return
    mf = make_mpi_rks(mol, xc="wb97xd")
    mf.kernel()
    # pass mf into extract_gto_kernel(..., use_mpi_response=True)

Density fitting bypasses mpi4pyscf's direct ``get_jk``, so DF is disabled
when MPI response is requested.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple


_MPI4PYSCF_ENABLED = False


def mpi4pyscf_available() -> bool:
    try:
        import mpi4pyscf  # noqa: F401
        return True
    except ImportError:
        return False


def enable_mpi4pyscf() -> Tuple[bool, int, int]:
    """Import mpi4pyscf (parks non-master ranks) and return master info.

    Prefer calling this (or bare ``import mpi4pyscf``) **before** importing
    the rest of casidapy so workers park early.

    Returns
    -------
    is_master, rank, size
        Non-master processes never return (they exit inside the mpi4pyscf pool).
    """
    global _MPI4PYSCF_ENABLED
    import mpi4pyscf  # noqa: F401  — side effect: suspend slaves
    from mpi4pyscf.tools import mpi as mpi_tools

    _MPI4PYSCF_ENABLED = True
    rank = int(mpi_tools.rank)
    size = int(mpi_tools.pool.size)
    return bool(mpi_tools.pool.is_master()), rank, size


def is_mpi4pyscf_enabled() -> bool:
    """True when mpi4pyscf has been imported (workers parked / master-only)."""
    global _MPI4PYSCF_ENABLED
    if _MPI4PYSCF_ENABLED:
        return True
    import sys
    if "mpi4pyscf" in sys.modules:
        _MPI4PYSCF_ENABLED = True
        return True
    return False


def spmd_comm_or_none(comm=None):
    """Return ``comm`` for SPMD Casida/QED, or ``None`` under mpi4pyscf.

    mpi4pyscf parks non-master ranks; using ``MPI.COMM_WORLD`` for SPMD
    gathers / barriers would hang. Davidson still uses the MPI JK pool.
    """
    if is_mpi4pyscf_enabled():
        return None
    return comm



def make_mpi_rks(
    mol,
    xc: str = "pbe",
    *,
    mo_coeff=None,
    mo_energy=None,
    mo_occ=None,
    grids_level: int = 3,
    verbose: int = 0,
):
    """Build an ``mpi4pyscf.dft.RKS`` mean-field (MPI ``get_jk`` / ``get_veff``).

    Do **not** call ``density_fit()`` on this object if you want MPI JK;
    DF replaces the MPI direct ``get_jk``.
    """
    if not _MPI4PYSCF_ENABLED:
        enable_mpi4pyscf()
    from mpi4pyscf import dft as mpi_dft

    mf = mpi_dft.RKS(mol)
    mf.xc = xc
    mf.verbose = verbose
    if hasattr(mf, "grids") and mf.grids is not None:
        mf.grids.level = grids_level
    if mo_coeff is not None:
        mf.mo_coeff = mo_coeff
    if mo_energy is not None:
        mf.mo_energy = mo_energy
    if mo_occ is not None:
        mf.mo_occ = mo_occ
    return mf


def promote_mf_to_mpi(mf, *, grids_level: Optional[int] = None) -> Any:
    """Copy a converged serial RKS/RHF onto an mpi4pyscf mean-field for MPI response."""
    if not _MPI4PYSCF_ENABLED:
        enable_mpi4pyscf()
    mol = mf.mol
    is_ks = hasattr(mf, "xc") and getattr(mf, "xc", None) is not None
    if is_ks:
        xc = mf.xc
        level = grids_level
        if level is None and getattr(mf, "grids", None) is not None:
            level = getattr(mf.grids, "level", 3)
        if level is None:
            level = 3
        mpi_mf = make_mpi_rks(
            mol,
            xc=xc,
            mo_coeff=mf.mo_coeff,
            mo_energy=mf.mo_energy,
            mo_occ=mf.mo_occ,
            grids_level=int(level),
            verbose=getattr(mf, "verbose", 0),
        )
    else:
        # mpi4pyscf.RHF is not a pyscf.scf.hf.RHF subclass, so
        # ``gen_response`` refuses it. Use MPI RKS with xc='HF' (pure HF
        # exchange, CIS / TDHF-equivalent response) instead.
        mpi_mf = make_mpi_rks(
            mol,
            xc="HF",
            mo_coeff=mf.mo_coeff,
            mo_energy=mf.mo_energy,
            mo_occ=mf.mo_occ,
            grids_level=int(grids_level if grids_level is not None else 1),
            verbose=getattr(mf, "verbose", 0),
        )
    for attr in ("converged", "e_tot", "conv_tol", "max_cycle"):
        if hasattr(mf, attr):
            try:
                setattr(mpi_mf, attr, getattr(mf, attr))
            except Exception:
                pass
    return mpi_mf
