"""Pauli–Fierz QED-TDDFT (TDA) on top of the GTO electronic kernel.

Length gauge, dipole approximation, single cavity mode, coherent-state
(relaxed-dipole) reference, TDA only. The electronic coupling ``K`` is
consumed as-is from :class:`~casidapy.kernels.gto.GTOKernel` — this module
does not modify ``apply_K`` or the Casida algebra.

**Closed-shell solve (default matrix-free):** :func:`solve_qed_tda` applies
``M v`` via ``apply_K`` + DSE + photon coupling and diagonalizes with
Davidson/LOBPCG (no full ``K`` cache). Pass ``matrix_free=False`` for the
legacy dense ``build_qed_tda_matrix`` + ``eigh`` path.

**MPI dense build (optional):** ``build_qed_tda_matrix(..., comm=comm)`` can
still distribute electronic ``K`` / DSE rows when assembling a full matrix.
Under ``--mpi-response`` / mpi4pyscf, matrix-free matvecs use MPI ``get_jk``
inside ``apply_K`` with workers parked (same model as Casida).

**Closed-shell** Hilbert space: electronic singles ⊕ one photonic state,
size ``(n_trans+1)``. Dipole self-energy (DSE) and coherent-state shifts are
included by default.

**Spin-flip** path (``solve_qed_sf_tda``): QED-SF-TDA / QED-SF-CIS-style
Hamiltonian on the collinear SF manifold (α-occ → β-virt). Basis is SF
singles with 0 and 1 cavity photons (size ``2 n_trans``). Light-matter
coupling uses the one-body dipole *difference* ``Δd`` between SF
configurations (not the spin-forbidden ``⟨α|r|β⟩`` transition dipole).
DSE / coherent-state are off by default (bilinear / JC-like form matching
common QED-SF-CIS presentations). Dense diagonalization only for SF.

This is the physical (DSE-inclusive for closed-shell) route in the
**transition** basis. For a cheaper post-processing route that couples the
cavity **after** electronic TDDFT diagonalization (JC / TC / truncated PF on
the eigenstates), use :func:`solve_qed_post`. The phenomenological Tavis–
Cummings path in :mod:`casidapy.polariton_handler` is a separate, even
simpler model.

Ground state: ordinary converged PySCF RKS/RHF or UKS (Level A — cavity
terms enter the response only). Full QED-SCF is out of scope.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg
from scipy.sparse.linalg import LinearOperator


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
# Closed-shell singlet convention (must stay consistent across this module):
#   μ_ia = √2 ⟨i|r|a⟩   (via GTOKernel.dipole_matrix)
#   q_ia = λ · ⟨i|r|a⟩  (raw spatial; dipole_blocks — no √2)
#   g    = √(ω/2) · √2 · q = √(ω/2) · (λ · μ)   ← matches post-process
#   DSE direct = 2 q⊗q = (√2 q)⊗(√2 q)
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
    matrix_free: bool = True
    solver_method: str = "davidson"  # davidson | lobpcg (matrix-free only)

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


def dse_exchange_matvec(
    Q_oo: np.ndarray,
    Q_vv: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Apply DSE exchange ``(Q_oo ⊗ Q_vv) v`` without forming the full matrix.

    For ``v`` shaped ``(n_o*n_v,)`` or ``(n_o*n_v, k)`` with ``ia = i*n_v + a``::

        (A_exch v)_{ia} = Σ_{jb} Q_oo[i,j] Q_vv[a,b] v_{jb}
                        = (Q_oo @ V @ Q_vv.T)_{ia} ,   V = reshape(v)
    """
    Q_oo = np.asarray(Q_oo, dtype=float)
    Q_vv = np.asarray(Q_vv, dtype=float)
    n_o, n_v = Q_oo.shape[0], Q_vv.shape[0]
    n = n_o * n_v
    v_arr = np.asarray(v, dtype=float)
    if v_arr.ndim == 1:
        if v_arr.size != n:
            raise ValueError(f"v length {v_arr.size} != n_o*n_v={n}")
        V = v_arr.reshape(n_o, n_v)
        return (Q_oo @ V @ Q_vv.T).ravel()
    if v_arr.ndim != 2 or v_arr.shape[0] != n:
        raise ValueError(f"v shape {v_arr.shape} incompatible with n_trans={n}")
    V = v_arr.reshape(n_o, n_v, -1)
    out = np.einsum("ij,jbk,ab->iak", Q_oo, V, Q_vv, optimize=True)
    return out.reshape(n, -1)


def qed_tda_apply(
    kernel,
    v: np.ndarray,
    *,
    q: np.ndarray,
    Q_oo: np.ndarray,
    Q_vv: np.ndarray,
    g: np.ndarray,
    omega_c: float,
    dE: np.ndarray,
    include_dse: bool = True,
) -> np.ndarray:
    """Matrix-free ``M @ v`` for closed-shell Pauli–Fierz TDA.

    ``v`` is ``(n+1,)`` or ``(n+1, k)`` with electronic amplitudes in ``[:n]``
    and the photonic amplitude in ``[n]``. Matches :func:`build_qed_tda_matrix`::

        y_e = K v_e + Δε ⊙ v_e + DSE[v_e] + g m
        y_m = g · v_e + ω_c m
    """
    if not getattr(kernel, "_ready", False):
        kernel.setup(tda=True)
    n = int(kernel.n_trans)
    q = np.asarray(q, dtype=float).ravel()
    g = np.asarray(g, dtype=float).ravel()
    dE = np.asarray(dE, dtype=float).ravel()
    if q.size != n or g.size != n or dE.size != n:
        raise ValueError(
            f"q/g/dE length must be n_trans={n}, got "
            f"{q.size}, {g.size}, {dE.size}"
        )
    omega_c = float(omega_c)

    v_arr = np.asarray(v, dtype=float)
    squeeze = False
    if v_arr.ndim == 1:
        if v_arr.size != n + 1:
            raise ValueError(f"v length {v_arr.size} != n_trans+1={n + 1}")
        v_arr = v_arr.reshape(n + 1, 1)
        squeeze = True
    elif v_arr.ndim != 2 or v_arr.shape[0] != n + 1:
        raise ValueError(f"v shape {v_arr.shape} incompatible with n_trans+1={n + 1}")

    ve = v_arr[:n, :]
    m = v_arr[n, :]

    if hasattr(kernel, "apply_K_matmat"):
        Kve = kernel.apply_K_matmat(ve)
    else:
        Kve = np.column_stack(
            [kernel.apply_K(ve[:, j]) for j in range(ve.shape[1])]
        )

    ye = Kve + dE[:, None] * ve
    if include_dse:
        # 2 q (q·v) − (Q_oo ⊗ Q_vv) v
        ye = ye + DSE_DIRECT_FACTOR * np.outer(q, q @ ve)
        ye = ye - dse_exchange_matvec(Q_oo, Q_vv, ve)
    ye = ye + np.outer(g, m)

    ym = g @ ve + omega_c * m
    out = np.empty((n + 1, v_arr.shape[1]), dtype=float)
    out[:n, :] = ye
    out[n, :] = ym
    if squeeze:
        return out.ravel()
    return out


def _qed_tda_initial_guess(dE: np.ndarray, omega_c: float, nroots: int) -> np.ndarray:
    """Unit vectors on lowest Δε singles plus a photonic seed."""
    from casidapy.casida_utils import build_initial_guess

    dE = np.asarray(dE, dtype=float).ravel()
    n = dE.size
    nroots = int(nroots)
    X0 = np.zeros((n + 1, nroots), dtype=float)
    n_el = min(nroots, n)
    if n_el > 0:
        X0[:n, :n_el] = build_initial_guess(dE, n_el)
    if nroots > n_el:
        X0[n, n_el:] = 1.0
    elif nroots >= 2:
        # Ensure the cavity mode is represented in the trial space.
        X0[:, -1] = 0.0
        X0[n, -1] = 1.0
    elif nroots == 1:
        # Mix a little photon into the lowest electronic guess when ω_c is near.
        X0[n, 0] = 0.25
        nrm = np.linalg.norm(X0[:, 0])
        if nrm > 1e-15:
            X0[:, 0] /= nrm
    # Tiny photonic bleed on remaining electronic seeds near resonance.
    if nroots >= 2 and omega_c > 0.0:
        for j in range(nroots - 1):
            if abs(dE[np.argmax(np.abs(X0[:n, j]))] - omega_c) < 0.05:
                X0[n, j] = 0.1
                X0[:, j] /= np.linalg.norm(X0[:, j]) + 1e-30
    return X0


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
    # ``gather`` returns ``None`` on non-root — do not cast it to an array.
    counts_raw = comm.gather(n_local * n_cols, root=root)

    if rank == root:
        counts = np.asarray(counts_raw, dtype="i")
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
        if verbose and rank == 0:
            print(
                f"  using cached dense K ({n}×{n}) — no MPI row rebuild",
                flush=True,
            )
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

    if verbose and rank == 0:
        n_local = (n + size - 1) // size
        print(
            f"  distributing ~{n_local} K rows / rank (hybrid response) …",
            flush=True,
        )

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
    matrix_free: Optional[bool] = None,
    solver_method: Optional[str] = None,
    options: Optional[QEDOptions] = None,
    comm=None,
    verbose: bool = False,
) -> QEDResults:
    """Diagonalize closed-shell TDA-QED (matrix-free by default).

    Default path applies ``M v`` via :func:`qed_tda_apply` (``apply_K`` + DSE +
    photon coupling) and solves with Davidson/LOBPCG — no full ``K`` cache.

    Pass ``matrix_free=False`` for the legacy dense
    :func:`build_qed_tda_matrix` + ``eigh`` path (also used when ``comm`` has
    ``size > 1`` for an explicit distributed dense build).

    Under mpi4pyscf / ``--mpi-response``, leave ``comm=None`` so workers stay
    parked; matrix-free ``apply_K`` uses the MPI JK pool.
    """
    if getattr(kernel, "_spin_flip", False):
        raise ValueError(
            "solve_qed_tda is closed-shell only; use solve_qed_sf_tda for "
            "spin-flip kernels."
        )
    solver_method_eff = "davidson"
    matrix_free_eff = True
    if options is not None:
        lam_vec = options.lam_vec()
        omega_c = options.omega_c
        origin = options.origin
        include_dse = options.include_dse
        coherent_state = options.coherent_state
        include_nuclear = options.include_nuclear
        if nstates is None:
            nstates = options.nstates
        matrix_free_eff = bool(options.matrix_free)
        solver_method_eff = str(options.solver_method)
    if matrix_free is not None:
        matrix_free_eff = bool(matrix_free)
    if solver_method is not None:
        solver_method_eff = str(solver_method)
    if lam_vec is None or omega_c is None:
        raise ValueError("lam_vec and omega_c are required (or pass options=QEDOptions(...))")

    if not getattr(kernel, "_ready", False):
        kernel.setup(tda=True)

    n = int(kernel.n_trans)
    omega_c = float(omega_c)

    # Explicit multi-rank dense assembly still uses the dense path.
    if _comm_size(comm) > 1:
        matrix_free_eff = False

    if not matrix_free_eff:
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
        X, m = V[:n, :], V[n, :]
        solver_method_eff = "eigh"
    else:
        from casidapy.davidson import solve_davidson, solve_lobpcg

        q, Q_oo, Q_vv = dipole_blocks(
            kernel,
            lam_vec,
            origin=origin,
            include_nuclear=include_nuclear,
            coherent_state=coherent_state,
            nuclear_sign=nuclear_sign,
        )
        g = np.sqrt(omega_c / 2.0) * SQRT2 * np.asarray(q, dtype=float).ravel()
        dE = kernel.diagonal_dE()
        diag = np.concatenate([dE, [omega_c]])

        def _apply(V):
            return qed_tda_apply(
                kernel,
                V,
                q=q,
                Q_oo=Q_oo,
                Q_vv=Q_vv,
                g=g,
                omega_c=omega_c,
                dE=dE,
                include_dse=include_dse,
            )

        A_op = LinearOperator(
            shape=(n + 1, n + 1),
            matvec=lambda x: _apply(x),
            matmat=lambda V: _apply(V),
            dtype=np.float64,
        )

        nroots = int(nstates) if nstates is not None else min(20, n + 1)
        nroots = max(1, min(nroots, n + 1))
        X0 = _qed_tda_initial_guess(dE, omega_c, nroots)
        verb = 1 if verbose else 0
        method = solver_method_eff.strip().lower()
        if verbose:
            print(
                f"\nSolving QED-TDA matrix-free using {method.upper()}\n"
                f"  Matrix size: {n + 1} x {n + 1}  (n_trans={n} + 1 photon)\n"
                f"  Seeking {nroots} eigenvalues",
                flush=True,
            )
        if method == "lobpcg":
            w, V = solve_lobpcg(
                A_op,
                nroots=nroots,
                X0=X0,
                diagonal=diag,
                tol=1e-8,
                maxiter=200,
                largest=False,
                verbose=verb,
            )
        elif method in ("davidson", "eigsh"):
            # eigsh routed to Davidson for a batched matmat path.
            w, V = solve_davidson(
                A_op,
                nroots=nroots,
                X0=X0,
                diagonal=diag,
                tol=1e-8,
                maxiter=200,
                largest=False,
                verbose=verb,
            )
            solver_method_eff = "davidson" if method == "eigsh" else method
        else:
            raise ValueError(
                f"Unknown QED solver_method {solver_method_eff!r}; "
                "use 'davidson' or 'lobpcg' (or matrix_free=False)."
            )
        X, m = V[:n, :], V[n, :]
        nstates = nroots  # already truncated

    mu = kernel.dipole_matrix()  # (n_trans, 3)
    d = X.T @ mu  # (nstates_or_all, 3)
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
            "matrix_free": bool(matrix_free_eff),
            "solver_method": str(solver_method_eff),
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


# ---------------------------------------------------------------------------
# Post-processing QED on electronic TDDFT eigenstates (JC / TC / truncated PF)
# ---------------------------------------------------------------------------


def tddft_state_dipoles(res) -> np.ndarray:
    """S₀→state transition dipoles ``(n_states, 3)`` from a Casida result."""
    d_mode = getattr(res, "d_mode", None)
    if d_mode is not None:
        d = np.asarray(d_mode, dtype=float)
        if d.ndim != 2:
            raise ValueError(f"d_mode must be 2-D, got shape {d.shape}")
        if d.shape[0] == 3:
            return d.T.copy()
        if d.shape[1] == 3:
            return d.copy()
        raise ValueError(f"d_mode shape {d.shape} is not (3,n) or (n,3)")

    mu_ov = getattr(res, "mu_transition", None)
    xpy = getattr(res, "xpy", None)
    if xpy is None:
        xpy = getattr(res, "Z", None)
    if mu_ov is not None and xpy is not None:
        mu = np.asarray(mu_ov, dtype=float)
        X = np.asarray(xpy, dtype=float)
        if mu.ndim != 2 or mu.shape[1] != 3:
            raise ValueError(f"mu_transition must be (n_trans, 3), got {mu.shape}")
        if X.ndim != 2 or X.shape[0] != mu.shape[0]:
            raise ValueError(
                f"xpy/Z shape {X.shape} incompatible with mu_transition {mu.shape}"
            )
        return X.T @ mu

    raise ValueError(
        "CasidaResults need d_mode or (mu_transition + xpy/Z) for QED post-processing"
    )


def _select_tddft_indices(
    omega: np.ndarray,
    f: Optional[np.ndarray],
    mu: np.ndarray,
    *,
    nstates: Optional[int],
    skip_ground: bool,
    prefer_bright: bool,
    lam: np.ndarray,
    ground_tol: float = 1e-8,
) -> np.ndarray:
    """Indices into TDDFT roots for the few-level / PF electronic manifold."""
    w = np.asarray(omega, dtype=float).ravel()
    n = w.size
    mask = np.ones(n, dtype=bool)
    if skip_ground:
        mask &= w > float(ground_tol)
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError("No TDDFT roots left after skip_ground filtering.")
    if nstates is not None and int(nstates) < idx.size:
        n_keep = int(nstates)
        if prefer_bright:
            strength = np.abs(mu[idx] @ lam)
            if f is not None and float(np.max(strength)) < 1e-16:
                strength = np.clip(np.asarray(f, dtype=float).ravel()[idx], 0.0, None)
            order = np.argsort(-strength)[:n_keep]
            idx = idx[np.sort(order)]
        else:
            idx = idx[:n_keep]
    return idx


def solve_qed_levels(
    omega: np.ndarray,
    mu: np.ndarray,
    lam_vec: Sequence[float],
    omega_c: float,
    *,
    f: Optional[np.ndarray] = None,
    nstates: Optional[int] = None,
    skip_ground: bool = True,
    prefer_bright: bool = True,
) -> Dict[str, Any]:
    """Few-level Jaynes–Cummings / Tavis–Cummings on TDDFT eigenstates.

    Basis (size ``n + 1``)::

        {|S_k, 0⟩ for selected roots k}  ∪  {|S₀, 1⟩}

    with couplings ``g_k = √(ω_c/2) (λ · μ_k)``. Probe oscillator strengths
    use electronic dipoles from ``|S₀,0⟩`` (outside the TC space). Do **not**
    use ``photon_frac`` as spectrum intensity.
    """
    omega_c = float(omega_c)
    if omega_c <= 0.0:
        raise ValueError(f"omega_c must be positive, got {omega_c}")
    lam = np.asarray(lam_vec, dtype=float).ravel()
    if lam.shape != (3,):
        raise ValueError(f"lam_vec must have shape (3,), got {lam.shape}")
    mu_arr = np.asarray(mu, dtype=float)
    if mu_arr.ndim != 2 or mu_arr.shape[1] != 3:
        raise ValueError(f"mu must have shape (n, 3), got {mu_arr.shape}")

    idx = _select_tddft_indices(
        omega,
        f,
        mu_arr,
        nstates=nstates,
        skip_ground=skip_ground,
        prefer_bright=prefer_bright,
        lam=lam,
    )
    e = np.asarray(omega, dtype=float).ravel()[idx]
    mu_exc = mu_arr[idx]
    g = np.sqrt(omega_c / 2.0) * (mu_exc @ lam)
    n = e.size
    M = np.zeros((n + 1, n + 1), dtype=float)
    M[:n, :n] = np.diag(e)
    M[n, n] = omega_c
    M[:n, n] = g
    M[n, :n] = g
    w, V = scipy.linalg.eigh(M)
    photon_frac = V[n, :] ** 2
    mu_pol = np.einsum("ka,kx->ax", V[:n, :], mu_exc, optimize=True)
    f_out = np.zeros(w.size, dtype=float)
    for a, wa in enumerate(w):
        if wa <= 1e-12:
            continue
        f_out[a] = (2.0 / 3.0) * float(wa) * float(np.dot(mu_pol[a], mu_pol[a]))

    return {
        "omega": w,
        "photon_frac": photon_frac,
        "f": f_out,
        "mu": mu_pol,
        "electronic_omega": e,
        "g": g,
        "omega_c": omega_c,
        "lam": lam,
        "V": V,
        "M": M,
        "tddft_indices": idx,
        "model": "jaynes-cummings" if n == 1 else "tavis-cummings",
        "postprocess": True,
    }


def solve_qed_pf_post(
    omega: np.ndarray,
    mu: np.ndarray,
    lam_vec: Sequence[float],
    omega_c: float,
    *,
    f: Optional[np.ndarray] = None,
    nstates: Optional[int] = None,
    skip_ground: bool = True,
    prefer_bright: bool = False,
    include_ground_slot: bool = True,
    include_dse: bool = True,
    mu0_lam: float = 0.0,
) -> Dict[str, Any]:
    """Truncated Pauli–Fierz on the TDDFT electronic eigenbasis.

    Electronic basis (length ``N``): ``{|S₀⟩, |S_k⟩}`` when
    ``include_ground_slot`` (default), with ``E₀ = 0`` and selected TDDFT
    roots. Dipole matrix is TDA-like (only S₀↔excited)::

        (λ·μ)_{0k} = λ · μ_k ,   (λ·μ)_{kj} = 0 for excited–excited

    Light–matter space: electronic ⊗ {0,1} photons (size ``2N``)::

        H = H_el + ω_c a†a + √(ω_c/2) (λ·μ)(a+a†) + ½ (λ·μ)²

    Returned ``omega`` is relative to the PF ground (``w - w[0]``). Probe
    ``f`` uses ``μ ⊗ I``; do **not** use ``photon_frac`` as intensity.
    """
    from casidapy.soc import build_soc_qed_pf_matrix

    omega_c = float(omega_c)
    if omega_c <= 0.0:
        raise ValueError(f"omega_c must be positive, got {omega_c}")
    lam = np.asarray(lam_vec, dtype=float).ravel()
    if lam.shape != (3,):
        raise ValueError(f"lam_vec must have shape (3,), got {lam.shape}")
    mu_arr = np.asarray(mu, dtype=float)
    if mu_arr.ndim != 2 or mu_arr.shape[1] != 3:
        raise ValueError(f"mu must have shape (n, 3), got {mu_arr.shape}")

    idx = _select_tddft_indices(
        omega,
        f,
        mu_arr,
        nstates=nstates,
        skip_ground=skip_ground,
        prefer_bright=prefer_bright,
        lam=lam,
    )
    e_exc = np.asarray(omega, dtype=float).ravel()[idx]
    mu_exc = mu_arr[idx]
    d_ov = mu_exc @ lam

    if include_ground_slot:
        e = np.concatenate([[0.0], e_exc])
        n = e.size
        d = np.zeros((n, n), dtype=float)
        d[0, 0] = float(mu0_lam)
        d[0, 1:] = d_ov
        d[1:, 0] = d_ov
    else:
        e = e_exc
        n = e.size
        d = np.zeros((n, n), dtype=float)

    M = build_soc_qed_pf_matrix(e, d, omega_c, include_dse=include_dse)
    w_abs, V = scipy.linalg.eigh(M)
    w = w_abs - w_abs[0]
    photon_frac = np.sum(V[n:, :] ** 2, axis=0)

    mu_el = np.zeros((n, n, 3), dtype=float)
    if include_ground_slot:
        mu_el[0, 1:, :] = mu_exc
        mu_el[1:, 0, :] = mu_exc
    mu_pol = (
        np.einsum("i,ijx,jk->kx", V[:n, 0], mu_el, V[:n, :], optimize=True)
        + np.einsum("i,ijx,jk->kx", V[n:, 0], mu_el, V[n:, :], optimize=True)
    )
    f_out = np.zeros(w.size, dtype=float)
    for k, wk in enumerate(w):
        if wk <= 1e-12:
            continue
        f_out[k] = (2.0 / 3.0) * float(wk) * float(np.dot(mu_pol[k], mu_pol[k]))

    return {
        "omega": w,
        "omega_absolute": w_abs,
        "photon_frac": photon_frac,
        "f": f_out,
        "mu": mu_pol,
        "electronic_omega": e,
        "d_mat": d,
        "omega_c": omega_c,
        "lam": lam,
        "V": V,
        "M": M,
        "tddft_indices": idx,
        "include_dse": bool(include_dse),
        "include_ground_slot": bool(include_ground_slot),
        "model": "pauli-fierz",
        "postprocess": True,
    }


def solve_qed_post(
    res,
    lam_vec: Sequence[float],
    omega_c: float,
    *,
    model: str = "pf",
    mu: Optional[np.ndarray] = None,
    nstates: Optional[int] = None,
    skip_ground: bool = True,
    prefer_bright: Optional[bool] = None,
    include_ground_slot: bool = True,
    include_dse: bool = True,
    mu0_lam: float = 0.0,
) -> Dict[str, Any]:
    """Post-process QED on finished TDDFT / Casida results.

    Couples a single cavity mode to the **electronic eigenstates** after
    :func:`~casidapy.casida_engine.run_casida` — no rebuild of the Casida
    matrix. Intended as a cheap alternative to :func:`solve_qed_tda`
    (which builds PF in the transition basis).

    Parameters
    ----------
    res
        :class:`~casidapy.casida_api.CasidaResults` (needs ``omega`` and
        dipoles via ``d_mode`` or ``mu_transition``+``xpy``).
    model
        ``\"jc\"`` — one bright root + ``|S₀,1⟩`` (Jaynes–Cummings).
        ``\"tc\"`` — many roots + ``|S₀,1⟩`` (Tavis–Cummings).
        ``\"pf\"`` — truncated Pauli–Fierz (electronic ⊗ {0,1} + optional DSE;
        **default**).
    mu
        Optional ``(n, 3)`` S₀→state dipoles; default from ``res``.
    nstates
        Cap on electronic roots in the manifold (after ``skip_ground``).
        ``None`` keeps all.
    """
    key = str(model).strip().lower().replace("_", "-")
    aliases = {
        "jc": "jc",
        "jaynes-cummings": "jc",
        "jaynes": "jc",
        "tc": "tc",
        "tavis-cummings": "tc",
        "tavis": "tc",
        "pf": "pf",
        "pauli-fierz": "pf",
        "pauli": "pf",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown qed-post model {model!r}; expected one of "
            f"{sorted(set(aliases))}."
        )
    kind = aliases[key]

    omega = np.asarray(res.omega, dtype=float).ravel()
    f_arr = None if getattr(res, "f", None) is None else np.asarray(res.f, dtype=float).ravel()
    mu_arr = tddft_state_dipoles(res) if mu is None else np.asarray(mu, dtype=float)
    if mu_arr.shape[0] != omega.size:
        raise ValueError(
            f"mu rows ({mu_arr.shape[0]}) != number of TDDFT roots ({omega.size})"
        )

    if kind == "jc":
        return solve_qed_levels(
            omega,
            mu_arr,
            lam_vec=lam_vec,
            omega_c=omega_c,
            f=f_arr,
            nstates=1 if nstates is None else nstates,
            skip_ground=skip_ground,
            prefer_bright=True if prefer_bright is None else prefer_bright,
        )
    if kind == "tc":
        return solve_qed_levels(
            omega,
            mu_arr,
            lam_vec=lam_vec,
            omega_c=omega_c,
            f=f_arr,
            nstates=nstates,
            skip_ground=skip_ground,
            prefer_bright=True if prefer_bright is None else prefer_bright,
        )
    return solve_qed_pf_post(
        omega,
        mu_arr,
        lam_vec=lam_vec,
        omega_c=omega_c,
        f=f_arr,
        nstates=nstates,
        skip_ground=skip_ground,
        prefer_bright=False if prefer_bright is None else prefer_bright,
        include_ground_slot=include_ground_slot,
        include_dse=include_dse,
        mu0_lam=mu0_lam,
    )
