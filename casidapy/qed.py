"""Pauli–Fierz QED-TDDFT (TDA) on top of the GTO electronic kernel.

Length gauge, dipole approximation, single cavity mode, coherent-state
(relaxed-dipole) reference, TDA only. The electronic coupling ``K`` is
consumed as-is from :class:`~casidapy.kernels.gto.GTOKernel` — this module
does not modify ``apply_K`` or the Casida algebra.

**MPI (closed-shell dense build):** ``build_qed_tda_matrix(..., comm=comm)``
distributes electronic ``K`` / DSE rows over ranks (same round-robin scheme as
``CasidaKS_MPI.build_matrices``). Dipole blocks are replicated (cheap); the
full ``(n_trans+1)`` matrix is assembled on root and buffer-broadcast. Use
this for large active spaces where building ``K`` dominates. Pass the same
``comm`` to ``solve_qed_tda``.

**Closed-shell** path (``solve_qed_tda``): electronic singles ⊕ one photonic
state, size ``(n_trans+1)``. Dipole self-energy (DSE) and coherent-state
shifts are included by default.

**Spin-flip** path (``solve_qed_sf_tda``): QED-SF-TDA / QED-SF-CIS-style
Hamiltonian on the collinear SF manifold (α-occ → β-virt). Basis is SF
singles with 0 and 1 cavity photons (size ``2 n_trans``). Light-matter
coupling uses the one-body dipole *difference* ``Δd`` between SF
configurations (not the spin-forbidden ``⟨α|r|β⟩`` transition dipole).
DSE / coherent-state are off by default (bilinear / JC-like form matching
common QED-SF-CIS presentations).

This is the physical (DSE-inclusive for closed-shell) route. The
phenomenological Tavis–Cummings post-processing in
:mod:`casidapy.polariton_handler` is a separate, cheaper path.

Ground state: ordinary converged PySCF RKS/RHF or UKS (Level A — cavity
terms enter the response only). Full QED-SCF is out of scope.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg


def _comm_rank(comm) -> int:
    if comm is None:
        return 0
    if hasattr(comm, "Get_rank"):
        return int(comm.Get_rank())
    return int(comm.rank)


def _comm_size(comm) -> int:
    if comm is None:
        return 1
    if hasattr(comm, "Get_size"):
        return int(comm.Get_size())
    return int(comm.size)

# ---------------------------------------------------------------------------
# Convention constants (single edit site if DePrince/Foley vs Flick/Rubio
# placement of closed-shell singlet factors differs). Unverified until a
# published QED-TDA number is matched — see module tests / README.
# ---------------------------------------------------------------------------
SQRT2 = np.sqrt(2.0)
DSE_DIRECT_FACTOR = 2.0

# permanent_dipole electronic part = Tr(D, r). Nuclear term uses
# NUCLEAR_DIPOLE_SIGN (physical a.u. dipole is usually el + (−1)*nuc).
# For the coherent-state oo/vv shift we use the *electronic* permanent
# dipole only — including nuclei makes ⟨μ⟩₀ gauge-invariant and then the
# diagonal shift no longer cancels the origin drift of Q_oo/Q_vv.
NUCLEAR_DIPOLE_SIGN = -1


@dataclass
class QEDOptions:
    """Physical cavity inputs for Pauli–Fierz TDA-QED (not CasidaOptions)."""

    lam_scalar: float
    polarization: Sequence[float]
    omega_c: float
    origin: Sequence[float] = (0.0, 0.0, 0.0)
    include_dse: bool = True
    coherent_state: bool = True
    # Nuclear dipole is for reporting / photonic displacement diagnostics.
    # The CS shift inside dipole_blocks uses electronic ⟨μ⟩₀ only (see there).
    include_nuclear: bool = False
    nstates: Optional[int] = None

    def lam_vec(self) -> np.ndarray:
        e = np.asarray(self.polarization, dtype=float).ravel()
        nrm = float(np.linalg.norm(e))
        if nrm < 1e-15:
            raise ValueError("polarization must be a non-zero 3-vector")
        return float(self.lam_scalar) * (e / nrm)


@dataclass
class QEDResults:
    """Eigenpairs of the dense TDA-QED matrix.

    Closed-shell: ``X`` is ``(n_trans, nstates)`` electronic amplitudes and
    ``m`` is the photonic amplitude. Spin-flip: ``X`` is ``(2 n_trans,
    nstates)`` with the 0-photon block in ``X[:n]`` and the 1-photon SF block
    in ``X[n:]``; ``m`` is set to zeros (photon weight is ``photon_frac``).
    """

    omega: np.ndarray
    X: np.ndarray
    m: np.ndarray
    f: np.ndarray
    photon_frac: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)


def permanent_dipole(
    kernel,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    include_nuclear: bool = True,
    nuclear_sign: Optional[float] = None,
) -> np.ndarray:
    """Ground-state permanent dipole ``⟨μ⟩₀`` in the same sign convention as ``q_pq``.

    Electronic part is ``Tr(D, r)`` matching ``mol.intor('int1e_r')`` (same as
    :meth:`GTOKernel.dipole_matrix`). Nuclear contribution uses
    ``NUCLEAR_DIPOLE_SIGN`` (or ``nuclear_sign`` override) so the origin-
    invariance test can resolve the relative sign empirically.
    """
    mol = kernel.mol
    orig = np.asarray(origin, dtype=float).ravel()
    mf = getattr(kernel, "_mf", None)
    if mf is not None and hasattr(mf, "make_rdm1"):
        dm = np.asarray(mf.make_rdm1(), dtype=float)
    else:
        # Closed-shell: D = C diag(n) Cᵀ
        dm = (
            kernel.mo_coeff
            * np.asarray(kernel.mo_occ, dtype=float)[None, :]
        ) @ kernel.mo_coeff.T

    with mol.with_common_orig(tuple(orig)):
        dip_ao = mol.intor("int1e_r", comp=3)  # (3, nao, nao)

    # Unrestricted: make_rdm1 may return (2, nao, nao)
    dm = np.asarray(dm, dtype=float)
    if dm.ndim == 3:
        dm = dm[0] + dm[1]

    el = np.einsum("xpq,qp->x", dip_ao, dm)
    if not include_nuclear:
        return np.asarray(el, dtype=float)

    sign = NUCLEAR_DIPOLE_SIGN if nuclear_sign is None else float(nuclear_sign)
    charges = mol.atom_charges().astype(float)
    coords = mol.atom_coords() - orig
    nuc = np.einsum("A,Ax->x", charges, coords)
    return np.asarray(el + sign * nuc, dtype=float)


def dipole_blocks(
    kernel,
    lam_vec: Sequence[float],
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    include_nuclear: bool = False,
    coherent_state: bool = True,
    nuclear_sign: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """λ-contracted dipole blocks ``(q_ia, Q_oo, Q_vv)``, coherent-state shifted.

    Flattening of ``q_ia`` is ``ia = i * n_unocc + a`` (matches GTOKernel).

    The coherent-state diagonal shift uses the electronic permanent dipole
    ``Tr(D, r)`` divided by ``N_e``. Nuclei are omitted here on purpose: a
    gauge-invariant full ⟨μ⟩₀ would not cancel the origin drift of the oo/vv
    diagonals (verified by the origin-invariance tests).
    """
    if getattr(kernel, "_spin_flip", False):
        raise ValueError(
            "dipole_blocks is for closed-shell QED; use "
            "sf_dipole_difference_matrix for QED-SF-TDA."
        )
    mol, C_o, C_v = kernel.mol, kernel._C_o, kernel._C_v
    lam = np.asarray(lam_vec, dtype=float).ravel()
    if lam.shape != (3,):
        raise ValueError(f"lam_vec must have shape (3,), got {lam.shape}")
    orig = tuple(np.asarray(origin, dtype=float).ravel())

    with mol.with_common_orig(orig):
        dip_ao = mol.intor("int1e_r", comp=3)

    # MO blocks then contract with λ → scalars per orbital pair
    q_ov = np.einsum("x,xpq,pi,qa->ia", lam, dip_ao, C_o, C_v, optimize=True)
    Q_oo = np.einsum("x,xpq,pi,qj->ij", lam, dip_ao, C_o, C_o, optimize=True)
    Q_vv = np.einsum("x,xpq,pa,qb->ab", lam, dip_ao, C_v, C_v, optimize=True)

    if coherent_state:
        # Always electronic for the CS orbital shift (ignore include_nuclear).
        mu0_el = permanent_dipole(
            kernel,
            origin=orig,
            include_nuclear=False,
            nuclear_sign=nuclear_sign,
        )
        nelec = float(getattr(mol, "nelectron", 0) or 0)
        if nelec < 1e-12:
            nelec = float(np.sum(np.asarray(kernel.mo_occ, dtype=float)))
        shift = float(np.dot(lam, mu0_el)) / nelec
        Q_oo = Q_oo - shift * np.eye(Q_oo.shape[0])
        Q_vv = Q_vv - shift * np.eye(Q_vv.shape[0])

    return q_ov.ravel(), Q_oo, Q_vv


def dse_exchange_matrix(Q_oo: np.ndarray, Q_vv: np.ndarray) -> np.ndarray:
    """DSE exchange block ``A_exch[ia,jb] = Q_oo[i,j] * Q_vv[a,b]``.

    Explicit ``einsum`` (not ``kron``) so the ``ia = i*n_v + a`` layout is
    unambiguous.
    """
    n_o, n_v = Q_oo.shape[0], Q_vv.shape[0]
    return np.einsum("ij,ab->iajb", Q_oo, Q_vv, optimize=True).reshape(
        n_o * n_v, n_o * n_v
    )


def dse_exchange_rows(
    Q_oo: np.ndarray,
    Q_vv: np.ndarray,
    row_indices: Sequence[int],
) -> np.ndarray:
    """Selected rows of the DSE exchange block (MPI-friendly).

    ``out[k, :]`` is row ``row_indices[k]`` of ``dse_exchange_matrix(Q_oo, Q_vv)``.
    """
    n_o, n_v = Q_oo.shape[0], Q_vv.shape[0]
    rows = np.asarray(row_indices, dtype=int).ravel()
    if rows.size == 0:
        return np.empty((0, n_o * n_v), dtype=float)
    i_idx = rows // n_v
    a_idx = rows % n_v
    # (nloc, n_o, n_v) → (nloc, n_trans)
    return (Q_oo[i_idx][:, :, None] * Q_vv[a_idx][:, None, :]).reshape(
        rows.size, n_o * n_v
    )


def _mpi_round_robin_rows(n: int, rank: int, size: int) -> List[int]:
    """Global row indices owned by ``rank`` (``ia % size == rank``)."""
    return [ia for ia in range(n) if ia % size == rank]


def _mpi_gatherv_matrix_rows(
    comm,
    local_rows: Sequence[int],
    local_data: np.ndarray,
    n_rows: int,
    n_cols: int,
    root: int = 0,
) -> Optional[np.ndarray]:
    """Gather distributed ``(n_local, n_cols)`` rows onto ``root`` as ``(n_rows, n_cols)``.

    Uses ``Gatherv`` on a contiguous float64 buffer (avoids pickle's ~2 GiB
    limit). Row index lists are small and gathered with lowercase ``gather``.
    """
    from mpi4py import MPI as _MPI

    rank = _comm_rank(comm)
    size = _comm_size(comm)
    local_rows = list(local_rows)
    n_local = len(local_rows)
    send = np.ascontiguousarray(local_data, dtype=np.float64).reshape(n_local, n_cols)

    all_rows = comm.gather(local_rows, root=root)
    counts = np.asarray(comm.gather(n_local * n_cols, root=root), dtype="i")

    if rank == root:
        recv = np.empty(int(n_rows) * int(n_cols), dtype=np.float64)
        displs = np.zeros(size, dtype="i")
        if size > 1:
            np.cumsum(counts[:-1], out=displs[1:])
        comm.Gatherv(
            send.ravel(),
            [recv, counts, displs, _MPI.DOUBLE],
            root=root,
        )
        full = np.empty((n_rows, n_cols), dtype=np.float64)
        offset = 0
        for rows, n_el in zip(all_rows, counts):
            n_r = len(rows)
            block = recv[offset:offset + int(n_el)].reshape(n_r, n_cols)
            for i, gr in enumerate(rows):
                full[gr, :] = block[i]
            offset += int(n_el)
        return full

    comm.Gatherv(send.ravel(), None, root=root)
    return None


def _mpi_bcast_array(comm, arr: Optional[np.ndarray], root: int = 0) -> np.ndarray:
    """Broadcast a contiguous ndarray via uppercase ``Bcast`` (no pickle)."""
    rank = _comm_rank(comm)
    if rank == root:
        arr = np.ascontiguousarray(arr)
        meta = (arr.shape, arr.dtype.str)
    else:
        meta = None
    shape, dtype_str = comm.bcast(meta, root=root)
    dtype = np.dtype(dtype_str)
    if rank == root:
        out = arr
    else:
        out = np.empty(shape, dtype=dtype)
    # Row-wise Bcast keeps each message well under MPI count limits.
    if out.ndim == 0:
        comm.Bcast(out, root=root)
    elif out.ndim == 1:
        comm.Bcast(out, root=root)
    else:
        for i in range(out.shape[0]):
            comm.Bcast(out[i], root=root)
    return out


def qed_electronic_A_rows(
    kernel,
    row_indices: Sequence[int],
    q: np.ndarray,
    Q_oo: np.ndarray,
    Q_vv: np.ndarray,
    *,
    include_dse: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """Electronic block rows ``A[ia,:]`` including optional DSE (no photon).

    ``A = K + diag(Δε) + 2 q⊗q − Q_oo⊗Q_vv`` (DSE terms optional). Used by the
    MPI builder so each rank only evaluates its owned ``K`` rows.
    """
    rows = list(row_indices)
    n = kernel.n_trans
    if not rows:
        return np.empty((0, n), dtype=float)

    K_rows = kernel.dense_K_rows(rows, verbose=verbose)
    A = np.array(K_rows, dtype=float, copy=True)
    dE = kernel.diagonal_dE()
    for k, ia in enumerate(rows):
        A[k, ia] += dE[ia]

    if include_dse:
        q = np.asarray(q, dtype=float).ravel()
        A += DSE_DIRECT_FACTOR * np.outer(q[rows], q)
        A -= dse_exchange_rows(Q_oo, Q_vv, rows)
    return A


def build_qed_tda_matrix(
    kernel,
    lam_vec: Sequence[float],
    omega_c: float,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    *,
    include_dse: bool = True,
    coherent_state: bool = True,
    include_nuclear: bool = False,
    nuclear_sign: Optional[float] = None,
    comm=None,
    verbose: bool = False,
) -> np.ndarray:
    """Dense closed-shell TDA-QED matrix ``M`` of shape ``(n_trans+1, n_trans+1)``.

    Parameters
    ----------
    comm
        Optional MPI communicator. When ``size > 1``, electronic ``K``/DSE rows
        are distributed round-robin across ranks (same ownership as
        ``CasidaKS_MPI``). The full matrix is assembled on rank 0 and
        buffer-broadcast so every rank returns an identical ``M``.
        ``comm=None`` (default) is a pure serial build.
    verbose
        Rank-0 progress prints for the dense ``K`` row build.
    """
    if getattr(kernel, "_spin_flip", False):
        raise ValueError(
            "Closed-shell build_qed_tda_matrix does not accept a spin-flip "
            "kernel; use build_qed_sf_tda_matrix / solve_qed_sf_tda."
        )
    if not getattr(kernel, "_ready", False):
        kernel.setup(tda=True)
    if not getattr(kernel, "tda", True):
        raise NotImplementedError("QED-TDDFT in this module is TDA-only.")

    n = kernel.n_trans
    omega_c = float(omega_c)
    if omega_c <= 0.0:
        raise ValueError(f"omega_c must be positive, got {omega_c}")

    size = _comm_size(comm)
    rank = _comm_rank(comm)

    q, Q_oo, Q_vv = dipole_blocks(
        kernel,
        lam_vec,
        origin=origin,
        include_nuclear=include_nuclear,
        coherent_state=coherent_state,
        nuclear_sign=nuclear_sign,
    )
    g = np.sqrt(omega_c / 2.0) * SQRT2 * np.asarray(q, dtype=float).ravel()

    # --- serial / single-rank fast path (reuse cached dense K when present) ---
    if size == 1:
        K = (
            kernel._K
            if getattr(kernel, "_K", None) is not None
            else kernel.dense_K_rows(range(n), verbose=verbose)
        )
        dE = kernel.diagonal_dE()
        A = np.array(K, dtype=float, copy=True)
        A[np.diag_indices(n)] += dE
        if include_dse:
            A += DSE_DIRECT_FACTOR * np.outer(q, q)
            A -= dse_exchange_matrix(Q_oo, Q_vv)
        M = np.zeros((n + 1, n + 1), dtype=float)
        M[:n, :n] = A
        M[:n, n] = g
        M[n, :n] = g
        M[n, n] = omega_c
        return M

    # --- MPI: distribute electronic rows; assemble + broadcast M ---
    if verbose and rank == 0:
        print(
            f"Building QED-TDA matrix: n_trans={n}, MPI ranks={size}",
            flush=True,
        )

    # If a full K cache already exists on every rank, skip the expensive
    # redistributed response and assemble only on root.
    has_K = getattr(kernel, "_K", None) is not None
    if comm is not None and hasattr(comm, "allreduce"):
        from mpi4py import MPI as _MPI
        has_K = bool(comm.allreduce(bool(has_K), op=_MPI.LAND))

    if has_K:
        if rank == 0:
            dE = kernel.diagonal_dE()
            A = np.array(kernel._K, dtype=float, copy=True)
            A[np.diag_indices(n)] += dE
            if include_dse:
                A += DSE_DIRECT_FACTOR * np.outer(q, q)
                A -= dse_exchange_matrix(Q_oo, Q_vv)
            M = np.zeros((n + 1, n + 1), dtype=float)
            M[:n, :n] = A
            M[:n, n] = g
            M[n, :n] = g
            M[n, n] = omega_c
        else:
            M = None
        return _mpi_bcast_array(comm, M, root=0)

    local_rows = _mpi_round_robin_rows(n, rank, size)
    A_local = qed_electronic_A_rows(
        kernel,
        local_rows,
        q,
        Q_oo,
        Q_vv,
        include_dse=include_dse,
        verbose=verbose and rank == 0,
    )
    A_full = _mpi_gatherv_matrix_rows(comm, local_rows, A_local, n, n, root=0)

    if rank == 0:
        M = np.zeros((n + 1, n + 1), dtype=float)
        M[:n, :n] = A_full
        M[:n, n] = g
        M[n, :n] = g
        M[n, n] = omega_c
        if verbose:
            print(f"QED-TDA matrix complete. M shape: {M.shape}", flush=True)
    else:
        M = None
    return _mpi_bcast_array(comm, M, root=0)


def sf_dipole_difference_matrix(
    kernel,
    lam_vec: Sequence[float],
    origin: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """λ-contracted SF dipole-difference matrix ``Δd`` (n_trans × n_trans).

    For spin-flip configurations ``Φ_{iα}^{aβ}``, ``Φ_{jα}^{bβ}`` the one-body
    dipole matrix element (Slater–Condon) is::

        ⟨Φ_ia| d |Φ_jb⟩ = δ_ij d_{aβ,bβ} − δ_ab d_{iα,jα}

    Contracting with ``λ`` gives
    ``Δ = I_oo ⊗ Q_vv − Q_oo ⊗ I_vv`` with ``Q_oo = λ·d^{αα}``,
    ``Q_vv = λ·d^{ββ}`` on the active α-occ / β-virt MOs.

    This is the light-matter coupling block of QED-SF-CIS/TDA (not the
    spin-forbidden ``⟨α|r|β⟩`` transition dipole).
    """
    if not getattr(kernel, "_spin_flip", False):
        raise ValueError(
            "sf_dipole_difference_matrix requires a spin_flip GTOKernel "
            "(GTOKernel.build_spin_flip / extract_sf_gto_kernel)."
        )
    lam = np.asarray(lam_vec, dtype=float).ravel()
    if lam.shape != (3,):
        raise ValueError(f"lam_vec must have shape (3,), got {lam.shape}")
    orig = tuple(np.asarray(origin, dtype=float).ravel())
    C_o, C_v = kernel._C_o, kernel._C_v  # α-occ, β-virt
    n_o, n_v = kernel.n_occ, kernel.n_unocc

    with kernel.mol.with_common_orig(orig):
        dip_ao = kernel.mol.intor("int1e_r", comp=3)

    # Same-spin MO dipoles (α on occupied, β on virtual)
    Q_oo = np.einsum("x,xpq,pi,qj->ij", lam, dip_ao, C_o, C_o, optimize=True)
    Q_vv = np.einsum("x,xpq,pa,qb->ab", lam, dip_ao, C_v, C_v, optimize=True)

    # Δ[ia,jb] = δ_ij Q_vv[a,b] − δ_ab Q_oo[i,j]
    eye_o = np.eye(n_o)
    eye_v = np.eye(n_v)
    delta = np.kron(eye_o, Q_vv) - np.kron(Q_oo, eye_v)
    return np.asarray(delta, dtype=float)


def _sf_electronic_A(kernel) -> np.ndarray:
    """Dense SF-TDA electronic matrix ``A = K + diag(dE)``."""
    if not getattr(kernel, "_ready", False):
        kernel.setup(tda=True)
    n = kernel.n_trans
    K = kernel._K if getattr(kernel, "_K", None) is not None else kernel.dense_K_rows(
        range(n)
    )
    A = np.array(K, dtype=float, copy=True)
    A[np.diag_indices(n)] += kernel.diagonal_dE()
    return A


def build_qed_sf_tda_matrix(
    kernel,
    lam_vec: Sequence[float],
    omega_c: float,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Dense QED-SF-TDA matrix of shape ``(2 n_trans, 2 n_trans)``.

    Basis: SF singles with 0 photons |Φ_ia⟩ and SF singles with 1 photon
    |Φ_ia, 1⟩. With electronic SF-TDA matrix ``A'`` and dipole-difference
    ``Δ = λ·Δd``::

        M = [[ A' ,  g ],
             [ g  ,  A' + ω_c I ]]

    where ``g = √(ω_c/2) Δ`` (no closed-shell ``√2``; SF is single-determinant
    spin-flip). Matches the QED-SF-CIS block structure (α→β / β→α manifolds
    solved separately; this code uses the kernel's α→β manifold).
    """
    if not getattr(kernel, "_spin_flip", False):
        raise ValueError(
            "build_qed_sf_tda_matrix requires a spin_flip GTOKernel."
        )
    if not getattr(kernel, "_ready", False):
        kernel.setup(tda=True)
    if not getattr(kernel, "tda", True):
        raise NotImplementedError("QED-SF-TDDFT is TDA-only.")

    omega_c = float(omega_c)
    if omega_c <= 0.0:
        raise ValueError(f"omega_c must be positive, got {omega_c}")

    A = _sf_electronic_A(kernel)
    n = A.shape[0]
    delta = sf_dipole_difference_matrix(kernel, lam_vec, origin=origin)
    g = np.sqrt(omega_c / 2.0) * delta

    M = np.zeros((2 * n, 2 * n), dtype=float)
    M[:n, :n] = A
    M[n:, n:] = A + omega_c * np.eye(n)
    M[:n, n:] = g
    M[n:, :n] = g.T
    # Numerical symmetrization (Δ should already be symmetric for real orbs)
    M = 0.5 * (M + M.T)
    return M


def solve_qed_tda(
    kernel,
    lam_vec: Optional[Sequence[float]] = None,
    omega_c: Optional[float] = None,
    nstates: Optional[int] = None,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    *,
    include_dse: bool = True,
    coherent_state: bool = True,
    include_nuclear: bool = False,
    nuclear_sign: Optional[float] = None,
    options: Optional[QEDOptions] = None,
    comm=None,
    verbose: bool = False,
) -> QEDResults:
    """Diagonalize the dense closed-shell TDA-QED matrix.

    Pass ``comm`` (mpi4py communicator) to build ``M`` with MPI-distributed
    electronic rows — see :func:`build_qed_tda_matrix`. Diagonalization is
    currently replicated on every rank after the matrix broadcast.
    """
    if getattr(kernel, "_spin_flip", False):
        raise ValueError(
            "solve_qed_tda is closed-shell only; use solve_qed_sf_tda for "
            "spin-flip kernels."
        )
    if options is not None:
        lam_vec = options.lam_vec()
        omega_c = options.omega_c
        origin = options.origin
        include_dse = options.include_dse
        coherent_state = options.coherent_state
        include_nuclear = options.include_nuclear
        if nstates is None:
            nstates = options.nstates
    if lam_vec is None or omega_c is None:
        raise ValueError("lam_vec and omega_c are required (or pass options=QEDOptions(...))")

    M = build_qed_tda_matrix(
        kernel,
        lam_vec,
        omega_c,
        origin=origin,
        include_dse=include_dse,
        coherent_state=coherent_state,
        include_nuclear=include_nuclear,
        nuclear_sign=nuclear_sign,
        comm=comm,
        verbose=verbose,
    )
    w, V = scipy.linalg.eigh(M)
    n = kernel.n_trans
    X, m = V[:n, :], V[n, :]

    mu = kernel.dipole_matrix()  # (n_trans, 3) unprojected
    d = X.T @ mu  # (n+1, 3)
    f = (2.0 / 3.0) * w * np.einsum("nx,nx->n", d, d)

    if nstates is not None:
        nstates = min(int(nstates), w.shape[0])
        sl = slice(0, nstates)
        w, X, m, f = w[sl], X[:, sl], m[sl], f[sl]

    return QEDResults(
        omega=w,
        X=X,
        m=m,
        f=f,
        photon_frac=np.abs(m) ** 2,
        meta={
            "omega_c": float(omega_c),
            "lam": np.asarray(lam_vec, dtype=float).tolist(),
            "n_trans": n,
            "tda": True,
            "spin_flip": False,
            "include_dse": bool(include_dse),
            "coherent_state": bool(coherent_state),
            "origin": list(np.asarray(origin, dtype=float).ravel()),
            "nuclear_dipole_sign": (
                NUCLEAR_DIPOLE_SIGN if nuclear_sign is None else float(nuclear_sign)
            ),
            "mpi_size": _comm_size(comm),
        },
    )


def solve_qed_sf_tda(
    kernel,
    lam_vec: Optional[Sequence[float]] = None,
    omega_c: Optional[float] = None,
    nstates: Optional[int] = None,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    *,
    options: Optional[QEDOptions] = None,
) -> QEDResults:
    """Diagonalize the dense QED-SF-TDA matrix (SF singles ⊗ {0,1} photons).

    Returns
    -------
    QEDResults
        ``X`` has shape ``(2 n_trans, nstates)``: rows ``[:n]`` are 0-photon SF
        amplitudes, ``[n:]`` are 1-photon SF amplitudes. ``photon_frac`` is the
        weight on the 1-photon block. Oscillator strengths are zero (SF
        dipole-forbidden from the high-spin reference in this model).
    """
    if options is not None:
        lam_vec = options.lam_vec()
        omega_c = options.omega_c
        origin = options.origin
        if nstates is None:
            nstates = options.nstates
    if lam_vec is None or omega_c is None:
        raise ValueError(
            "lam_vec and omega_c are required (or pass options=QEDOptions(...))"
        )
    if not getattr(kernel, "_spin_flip", False):
        raise ValueError(
            "solve_qed_sf_tda requires a spin_flip GTOKernel; "
            "for closed-shell QED use solve_qed_tda."
        )

    M = build_qed_sf_tda_matrix(kernel, lam_vec, omega_c, origin=origin)
    w, V = scipy.linalg.eigh(M)
    n = kernel.n_trans
    photon_frac = np.sum(V[n:, :] ** 2, axis=0)
    f = np.zeros_like(w)

    if nstates is not None:
        nstates = min(int(nstates), w.shape[0])
        sl = slice(0, nstates)
        w = w[sl]
        V = V[:, sl]
        photon_frac = photon_frac[sl]
        f = f[sl]

    return QEDResults(
        omega=w,
        X=V,
        m=np.zeros(w.shape[0], dtype=float),
        f=f,
        photon_frac=photon_frac,
        meta={
            "omega_c": float(omega_c),
            "lam": np.asarray(lam_vec, dtype=float).tolist(),
            "n_trans": n,
            "basis_dim": 2 * n,
            "tda": True,
            "spin_flip": True,
            "include_dse": False,
            "coherent_state": False,
            "origin": list(np.asarray(origin, dtype=float).ravel()),
        },
    )


def _state_overlap(X_ref: np.ndarray, m_ref: np.ndarray, X: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Overlap matrix between two QED eigenbases.

    Closed-shell: ``S[a,b] = X_ref[:,a]·X[:,b] + m_ref[a]*m[b]``.
    Spin-flip (``m`` unused / zero): ``S = X_ref.T @ X`` on the full ``2 n`` space.
    """
    return X_ref.T @ X + np.outer(m_ref, m)


def _match_states(cost_or_neg_overlap: np.ndarray, *, maximize: bool = True) -> np.ndarray:
    """Hungarian assignment.

    If ``maximize`` is True, ``cost_or_neg_overlap`` is treated as a score to
    *maximize* (e.g. |overlap|). If False, it is a cost to *minimize*
    (e.g. energy distance).

    Returns ``perm`` such that branch ``a`` ↔ column ``perm[a]``.
    """
    from scipy.optimize import linear_sum_assignment

    mat = -cost_or_neg_overlap if maximize else cost_or_neg_overlap
    row, col = linear_sum_assignment(mat)
    perm = np.empty(cost_or_neg_overlap.shape[0], dtype=int)
    perm[row] = col
    return perm


def _energy_match_cost(
    omega_prev: np.ndarray,
    phot_prev: np.ndarray,
    omega_new: np.ndarray,
    phot_new: np.ndarray,
    photon_weight: float,
) -> np.ndarray:
    """Cost matrix for adiabatic / geometry-scan tracking."""
    dw = np.abs(omega_prev[:, None] - omega_new[None, :])
    dp = np.abs(phot_prev[:, None] - phot_new[None, :])
    return dw + float(photon_weight) * dp


def track_states(
    results_list: Sequence[QEDResults],
    *,
    method: str = "overlap",
    photon_weight: float = 0.05,
    overlap_floor: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Branch tracking along a λ- or geometry scan.

    Parameters
    ----------
    method
        ``\"overlap\"`` — match by eigenvector overlap (correct for **λ-scans
        at fixed geometry**, where the MO / config basis is shared).
        ``\"energy\"`` — match by |Δω| (+ optional photon-fraction penalty).
        Use this for **geometry / torsion PES**: raw ``X`` overlaps are
        meaningless across geometries because the SF config basis rides on
        geometry-dependent MOs.
        ``\"auto\"`` — try overlap; if the worst assigned |⟨ψ|ψ'⟩| falls below
        ``overlap_floor``, fall back to energy matching for that step (and
        warn once).
    photon_weight
        Extra cost weight on |Δ photon_frac| for ``method=\"energy\"`` (Ha
        units on the energy term). Helps keep electronic vs photonic
        character from swapping when two roots are close in energy.
    overlap_floor
        Minimum accepted assigned |overlap| for ``method=\"auto\"``.

    Returns
    -------
    omega_tracked : (n_pts, n_states)
    photon_tracked : (n_pts, n_states)
    """
    if not results_list:
        raise ValueError("results_list is empty")
    method = str(method).lower().strip()
    if method not in ("overlap", "energy", "auto"):
        raise ValueError(
            f"method must be 'overlap', 'energy', or 'auto', got {method!r}"
        )

    n_pts = len(results_list)
    n_states = results_list[0].omega.shape[0]
    omega_t = np.empty((n_pts, n_states), dtype=float)
    phot_t = np.empty((n_pts, n_states), dtype=float)

    omega_t[0] = results_list[0].omega
    phot_t[0] = results_list[0].photon_frac
    X_prev = np.array(results_list[0].X, dtype=float, copy=True)
    m_prev = np.array(results_list[0].m, dtype=float, copy=True)
    warned_fallback = False

    for i in range(1, n_pts):
        r = results_list[i]
        if r.omega.shape[0] != n_states:
            raise ValueError(
                f"Inconsistent nstates along scan: point 0 has {n_states}, "
                f"point {i} has {r.omega.shape[0]}"
            )
        if r.X.shape[0] != X_prev.shape[0]:
            raise ValueError(
                f"Inconsistent eigenvector length along scan: point 0 has "
                f"{X_prev.shape[0]}, point {i} has {r.X.shape[0]}"
            )

        use_energy = method == "energy"
        if method in ("overlap", "auto"):
            S = _state_overlap(X_prev, m_prev, r.X, r.m)
            Sab = np.abs(S)
            perm_ov = _match_states(Sab, maximize=True)
            worst = float(np.min(Sab[np.arange(n_states), perm_ov]))
            if method == "auto" and worst < float(overlap_floor):
                use_energy = True
                if not warned_fallback:
                    import warnings

                    warnings.warn(
                        "track_states(method='auto'): low eigenvector "
                        f"overlap (min assigned |S|={worst:.3f} < "
                        f"{overlap_floor}); falling back to energy "
                        "matching. For geometry scans prefer "
                        "method='energy'.",
                        UserWarning,
                        stacklevel=2,
                    )
                    warned_fallback = True
            else:
                perm = perm_ov

        if use_energy:
            cost = _energy_match_cost(
                omega_t[i - 1],
                phot_t[i - 1],
                r.omega,
                r.photon_frac,
                photon_weight,
            )
            perm = _match_states(cost, maximize=False)

        X_new = r.X[:, perm]
        m_new = r.m[perm]
        # Align eigenvector phase to previous point (overlap → positive)
        signs = np.sign(np.einsum("ik,ik->k", X_prev, X_new) + m_prev * m_new)
        signs[signs == 0.0] = 1.0
        X_new = X_new * signs
        m_new = m_new * signs

        omega_t[i] = r.omega[perm]
        phot_t[i] = r.photon_frac[perm]
        X_prev, m_prev = X_new, m_new

    return omega_t, phot_t


def scan_qed_tda(
    kernels: Sequence[Any],
    lam_vec: Sequence[float],
    omega_c: float,
    *,
    nstates: Optional[int] = None,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    include_dse: bool = True,
    coherent_state: bool = True,
    track: bool = True,
) -> Dict[str, Any]:
    """Solve TDA-QED at each geometry/kernel and optionally track polariton branches.

    Parameters
    ----------
    kernels
        Sequence of ready (or setup-able) :class:`GTOKernel` instances, one per
        geometry along a coordinate.
    """
    results = [
        solve_qed_tda(
            k,
            lam_vec,
            omega_c,
            nstates=nstates,
            origin=origin,
            include_dse=include_dse,
            coherent_state=coherent_state,
        )
        for k in kernels
    ]
    out: Dict[str, Any] = {"results": results}
    if track and results:
        omega_t, phot_t = track_states(results)
        out["omega_tracked"] = omega_t
        out["photon_frac_tracked"] = phot_t
    return out


def scan_qed_lambda(
    kernel,
    lam_scalars: Sequence[float],
    polarization: Sequence[float],
    omega_c: float,
    *,
    nstates: Optional[int] = None,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    include_dse: bool = True,
    coherent_state: bool = True,
    track: bool = True,
) -> Dict[str, Any]:
    """λ-sweep at **fixed geometry** — electronic ``K`` built once, then reused.

    Each cavity strength only re-assembles the ``(n_trans+1)`` QED matrix
    (dipole blocks scale with λ; no new AO integrals). Prefer this over
    rebuilding the GTOKernel per λ. For independent geometry points, run
    separate processes / a job array instead of intra-solve MPI.
    """
    if not getattr(kernel, "_ready", False):
        kernel.setup(tda=True)

    e = np.asarray(polarization, dtype=float).ravel()
    nrm = float(np.linalg.norm(e))
    if nrm < 1e-15:
        raise ValueError("polarization must be a non-zero 3-vector")
    e = e / nrm

    results = [
        solve_qed_tda(
            kernel,
            float(lam) * e,
            omega_c,
            nstates=nstates,
            origin=origin,
            include_dse=include_dse,
            coherent_state=coherent_state,
        )
        for lam in lam_scalars
    ]
    out: Dict[str, Any] = {
        "results": results,
        "lam_scalars": np.asarray(lam_scalars, dtype=float),
        "polarization": e,
        "omega_c": float(omega_c),
    }
    if track and results:
        omega_t, phot_t = track_states(results)
        out["omega_tracked"] = omega_t
        out["photon_frac_tracked"] = phot_t
    return out


def scan_qed_sf_lambda(
    kernel,
    lam_scalars: Sequence[float],
    polarization: Sequence[float],
    omega_c: float,
    *,
    nstates: Optional[int] = None,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    track: bool = True,
) -> Dict[str, Any]:
    """λ-sweep for QED-SF-TDA at fixed geometry (SF ``K`` / ``A'`` built once)."""
    if not getattr(kernel, "_spin_flip", False):
        raise ValueError("scan_qed_sf_lambda requires a spin_flip GTOKernel.")
    if not getattr(kernel, "_ready", False):
        kernel.setup(tda=True)

    e = np.asarray(polarization, dtype=float).ravel()
    nrm = float(np.linalg.norm(e))
    if nrm < 1e-15:
        raise ValueError("polarization must be a non-zero 3-vector")
    e = e / nrm

    results = [
        solve_qed_sf_tda(
            kernel,
            float(lam) * e,
            omega_c,
            nstates=nstates,
            origin=origin,
        )
        for lam in lam_scalars
    ]
    out: Dict[str, Any] = {
        "results": results,
        "lam_scalars": np.asarray(lam_scalars, dtype=float),
        "polarization": e,
        "omega_c": float(omega_c),
        "spin_flip": True,
    }
    if track and results:
        omega_t, phot_t = track_states(results)
        out["omega_tracked"] = omega_t
        out["photon_frac_tracked"] = phot_t
    return out
