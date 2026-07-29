#Preparing for Casida calculation, defining the inputs and outputs
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from mpi4py import MPI

from dftpy.functional import Functional, LocalPseudo, TotalFunctional
from dftpy.ions import Ions
from dftpy.time_data import timer

from scipy.linalg import eigh as scipy_eigh
from scipy.sparse.linalg import LinearOperator

from casidapy.casida_api import CasidaInputs, CasidaOptions, CasidaResults
from casidapy.casida_of import build_of_functional_context
from casidapy.casida_utils import (
    XC_TO_LIBXC_COMPONENTS,
    build_initial_guess,
    mpi_comm_rank,
    mpi_comm_size,
    normalize_wavefunctions,
    psi_to_fields,
)
from casidapy.davidson import davidson, solve_davidson, solve_eigsh, solve_lobpcg
from casidapy.kernels.plane_wave import PlaneWaveKernel
from casidapy.qepy_adapter import slice_active_space


class CasidaKS_MPI:
    """Casida equation solver with MPI parallelization (CPU).

    Plane-wave / real-space kernel work is delegated to
    :class:`~casidapy.kernels.plane_wave.PlaneWaveKernel`.  The public API
    (``set_active_orbitals``, ``setup_matrix_free``, ``solve_*``, …) is unchanged.
    """

    def __init__(self, rho_ks, xc_functional, polarized=False,
                 rho_cutoff=1e-3, fxc_max=20.0, use_gpu=False,
                 spin_state="singlet", libxc_xc_components=None, comm=None,
                 use_uspp=False, beta_projectors=None, qij_augmentation=None,
                 use_eDFTpy=False, kernel=None):
        if comm is None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm
        self.rank = mpi_comm_rank(self.comm)
        self.size = mpi_comm_size(self.comm)

        if kernel is not None:
            self._kernel = kernel
        else:
            self._kernel = PlaneWaveKernel(
                rho_ks,
                xc_functional,
                polarized=polarized,
                rho_cutoff=rho_cutoff,
                fxc_max=fxc_max,
                spin_state=spin_state,
                libxc_xc_components=libxc_xc_components,
                use_uspp=use_uspp,
                beta_projectors=beta_projectors,
                qij_augmentation=qij_augmentation,
                use_eDFTpy=use_eDFTpy,
                verbose=(self.rank == 0),
                use_gpu=use_gpu,
            )

        # Mirror kernel attributes for backward-compatible access
        self.rho = getattr(self._kernel, "rho", None)
        self.grid = getattr(self._kernel, "grid", None)
        self._grid_shape = getattr(self._kernel, "_grid_shape", None)
        self.N = float(self.rho.integral()) if self.rho is not None else 0.0
        self.polarized = bool(getattr(self._kernel, "polarized", polarized))
        self.use_uspp = bool(getattr(self._kernel, "use_uspp", False))
        self.beta_projectors = getattr(self._kernel, "beta_projectors", None)
        self.qij_augmentation = getattr(self._kernel, "qij_augmentation", None)
        self.use_eDFTpy = bool(getattr(self._kernel, "use_eDFTpy", False))
        self.transition_densities = None
        self.hartree = getattr(self._kernel, "hartree", None)
        self.functional = getattr(self._kernel, "functional", xc_functional)
        self.triplet = bool(getattr(self._kernel, "triplet", False))
        self.fkxc_arr = getattr(self._kernel, "fkxc_arr", None)
        self.fkxc = self.fkxc_arr
        self._fkxc_ndarray = getattr(self._kernel, "_fkxc_ndarray", None)

        self.A = None
        self.B = None
        self.C = None
        self.omega = None
        self.f = None
        self.xpy = None
        self.Z = None
        self.mu_transition = None
        self.d_mode = None
        self.tda = False
        self.n_occ = 0
        self.n_unocc = 0

    def set_active_orbitals(self, occ_eigs, unocc_eigs, psi_occ, psi_unocc):
        """Store pre-sliced active occupied/unoccupied orbital sets.

        Use ``qepy_adapter.slice_active_space`` to obtain the four arguments
        from full KS arrays before calling this method.
        """
        self._kernel.set_active_orbitals(occ_eigs, unocc_eigs, psi_occ, psi_unocc)
        self._occ_e = self._kernel._occ_e
        self._unocc_e = self._kernel._unocc_e
        self._psi_occ = self._kernel._psi_occ
        self._psi_unocc = self._kernel._psi_unocc
        self._n_occ = self._kernel.n_occ
        self._n_unocc = self._kernel.n_unocc
        self.n_occ = self._n_occ
        self.n_unocc = self._n_unocc
        self._proj_occ = getattr(self._kernel, "_proj_occ", None)
        self._proj_unocc = getattr(self._kernel, "_proj_unocc", None)

    def transition_orbital(self, psi_i, psi_a, proj_i=None, proj_a=None):
        """Transition density ψ_i* ψ_a + USPP augmentation if enabled."""
        return self._kernel.transition_orbital(psi_i, psi_a, proj_i=proj_i, proj_a=proj_a)

    def _hartree_potential(self, rho):
        """Compute Hartree potential from density."""
        return self._kernel._hartree_potential(rho)

    def _dipole_matrix(self, psi_occ=None, psi_unocc=None):
        """Transition dipole matrix elements (delegates to plane-wave kernel)."""
        return self._kernel.dipole_matrix()

    def oscillator_strengths(
        self,
        k=None,
        use_davidson=None,
        davidson_tol=1e-8,
        verbose=1,
    ):
        """
        Oscillator strengths f_n = (2/3) omega_n |d_n|^2, d = mu @ Z (TDA) or mu @ xpy (RPA).
        Uses the same active occupied / unoccupied sets as the prior ``setup_matrix_free`` /
        ``build_matrices`` call.
        """
        if self.omega is None:
            self.solve(
                k=k, use_davidson=use_davidson, davidson_tol=davidson_tol, verbose=verbose
            )
        elif (not self.tda) and (self.xpy is None):
            self.solve(
                k=k, use_davidson=use_davidson, davidson_tol=davidson_tol, verbose=verbose
            )

        if self._kernel.n_occ == 0:
            raise RuntimeError("Call build_matrices(...) or setup_matrix_free(...) first.")

        if self.triplet:
            n_states = len(self.omega)
            self.f = np.zeros(n_states, dtype=np.float64)
            self.mu_transition = None
            self.d_mode = None
            return self.f

        mu = self._kernel.dipole_matrix()
        self.mu_transition = mu

        if self.tda:
            d = mu.T @ self.Z
        else:
            d = mu.T @ self.xpy
        self.d_mode = d

        d2 = np.sum(d * d, axis=0)
        f = (2.0 / 3.0) * self.omega * d2
        self.f = f
        return f

    def collapse_transition_densities_to_state_basis(self, k: int) -> Optional[np.ndarray]:
        """ρ_k(r) = Σ_ia amp_ia,k φ_ia(r); amp = Z (TDA) or xpy (RPA)."""
        if not getattr(self, "use_eDFTpy", False):
            return None
        if self.tda:
            amp = np.real(np.asarray(self.Z[:, :k], dtype=float))
        else:
            if self.xpy is None:
                raise RuntimeError(
                    "RPA requires xpy before collapsing transition densities",
                )
            amp = np.real(np.asarray(self.xpy[:, :k], dtype=float))
        # Sync caches from kernel after setup_matrix_free
        self._psi_occ_arr = getattr(self._kernel, "_psi_occ_arr", None)
        self._psi_unocc_arr = getattr(self._kernel, "_psi_unocc_arr", None)
        self._grid_shape = getattr(self._kernel, "_grid_shape", self._grid_shape)
        return self._kernel.collapse_transition_densities_to_state_basis(amp)

    # --- Matrix-free Casida (LOBPCG / eigsh on matvec) ---

    def setup_matrix_free(self, tda: bool = False):
        self._kernel.setup(tda=tda)
        if (
            not tda
            and getattr(self._kernel, "hybrid", False)
            and type(self._kernel).__name__ == "GTOKernel"
        ):
            raise NotImplementedError(
                "Hybrid full TDDFT is not available matrix-free (A-B ≠ diag(dE)). "
                "Use run_casida(), which builds dense A/B via PySCF get_ab(), "
                "or set tda=True."
            )
        self.tda = tda
        self._n_trans = self._kernel.n_trans
        self._n_occ = self._kernel.n_occ
        self._n_unocc = self._kernel.n_unocc
        self._dE = self._kernel.diagonal_dE()
        self._sqrt_dE = np.sqrt(self._dE)
        self._dV = getattr(self._kernel, "_dV", None)
        self._grid_shape = getattr(self._kernel, "_grid_shape", None)
        self._psi_occ_arr = getattr(self._kernel, "_psi_occ_arr", None)
        self._psi_unocc_arr = getattr(self._kernel, "_psi_unocc_arr", None)
        self._proj_occ = getattr(self._kernel, "_proj_occ", None)
        self._proj_unocc = getattr(self._kernel, "_proj_unocc", None)
        self._matrix_free = True

        if self.rank == 0:
            n_grid = int(np.prod(self._grid_shape)) if self._grid_shape is not None else 0
            wfn_mem = (self._n_occ + self._n_unocc) * n_grid * 8 / 1e9 if n_grid else 0.0
            matrix_mem = self._n_trans ** 2 * 8 / 1e9
            print("Matrix-free mode initialized:")
            print(f"  Backend: {type(self._kernel).__name__}")
            print(f"  Transitions: {self._n_occ} occ × {self._n_unocc} unocc = {self._n_trans}")
            if self._grid_shape is not None:
                print(f"  Grid: {self._grid_shape} = {n_grid:,} points")
                print(f"  Wavefunction memory: {wfn_mem:.2f} GB")
            print(f"  (Full matrix would be: {matrix_mem:.2f} GB)")
            backend = "GPU (CuPy)" if getattr(self._kernel, "use_gpu", False) else "CPU"
            print(f"  TDA: {tda}, {backend}")

    def _apply_K_matvec(self, v):
        """Apply eh coupling ``K`` via the active kernel backend.

        Accepts a vector ``(n_trans,)`` or a block ``(n_trans, k)``.  When the
        kernel exposes ``apply_K_matmat`` (GTO), a block uses one batched
        response call instead of ``k`` serial ones — important for LOBPCG.
        """
        v = np.asarray(v, dtype=float)
        if v.ndim == 2 and hasattr(self._kernel, "apply_K_matmat"):
            return self._kernel.apply_K_matmat(v)
        if v.ndim == 2:
            # Fallback: column loop (PlaneWaveKernel / older backends)
            return np.column_stack(
                [self._kernel.apply_K(v[:, j]) for j in range(v.shape[1])]
            )
        return self._kernel.apply_K(v)

    def _apply_C_matvec(self, v):
        """
        Matrix-free multiply by the ``C`` operator used in **full** Casida (non-TDA).

        With diagonal KS energy gaps ``Δε`` and ``S = diag(√Δε)``, this implements
        ``C v = S ( 2 K S v + Δε ⊙ S v )``, i.e. the similarity-transformed RPA chain so the
        eigenproblem matches ``solve_matrix_free`` / dense ``C`` for the same physics.
        """
        v = np.asarray(v, dtype=float)
        input_shape = v.shape
        if v.ndim == 1:
            w1 = self._sqrt_dE * v
            Kw1 = self._apply_K_matvec(w1)
            w2 = 2.0 * Kw1 + self._dE * w1
            return (self._sqrt_dE * w2).reshape(input_shape)

        # Block: (n_trans, k)
        w1 = self._sqrt_dE[:, None] * v
        Kw1 = self._apply_K_matvec(w1)
        w2 = 2.0 * Kw1 + self._dE[:, None] * w1
        return self._sqrt_dE[:, None] * w2

    def _apply_A_matvec(self, v):
        """
        Matrix-free multiply by ``A = K + diag(Δε)`` for **TDA** (solve ``A X = ω X``).
        """
        v = np.asarray(v, dtype=float)
        if v.ndim == 1:
            return self._apply_K_matvec(v) + self._dE * v
        return self._apply_K_matvec(v) + self._dE[:, None] * v

    def _get_linear_operator(self):
        """SciPy ``LinearOperator`` for LOBPCG/eigsh: TDA uses ``A``, full Casida uses ``C``.

        ``matmat`` is set explicitly so LOBPCG's block updates hit one batched
        kernel call (GTO ``apply_K_matmat``) instead of SciPy's default
        column-by-column ``matvec`` loop.
        """
        Ntrans = self._n_trans
        if self.tda:
            matvec = self._apply_A_matvec
        else:
            matvec = self._apply_C_matvec
        return LinearOperator(
            shape=(Ntrans, Ntrans),
            matvec=matvec,
            matmat=matvec,  # same callable handles (n,) and (n, k)
            dtype=np.float64,
        )

    def solve_matrix_free(self, k, tol=1e-8, maxiter=200, verbose=1, method="lobpcg"):
        if not hasattr(self, "_matrix_free") or not self._matrix_free:
            raise RuntimeError("Call setup_matrix_free() first.")

        Ntrans = self._n_trans
        if self.rank == 0 and verbose:
            print(f"\nSolving {'TDA' if self.tda else 'full Casida'} using {method.upper()}")
            print(f"  Matrix size: {Ntrans} x {Ntrans}")
            print(f"  Seeking {k} eigenvalues, tolerance: {tol}")

        A_op = self._get_linear_operator()
        diagonal = self._dE if self.tda else self._dE ** 2

        X0 = build_initial_guess(self._dE, k)

        if method.lower() == "davidson":
            eigenvalues, eigenvectors = solve_davidson(
                A_op, nroots=k, X0=X0, diagonal=diagonal, tol=tol, maxiter=maxiter,
                largest=False, verbose=verbose if self.rank == 0 else 0
            )
        elif method.lower() == "lobpcg":
            eigenvalues, eigenvectors = solve_lobpcg(
                A_op, nroots=k, X0=X0, diagonal=diagonal, tol=tol, maxiter=maxiter,
                largest=False, verbose=verbose if self.rank == 0 else 0
            )
        elif method.lower() == "eigsh":
            eigenvalues, eigenvectors = solve_eigsh(
                A_op, nroots=k, X0=X0, tol=tol, maxiter=maxiter,
                largest=False, verbose=verbose if self.rank == 0 else 0
            )
        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'davidson', 'lobpcg', or 'eigsh'."
            )

        if self.tda:
            omega = eigenvalues
        else:
            if np.any(eigenvalues < -1e-6):
                import warnings
                warnings.warn(
                    f"Found {np.sum(eigenvalues < -1e-6)} negative eigenvalues. "
                    "This may indicate RPA instability."
                )
            eigenvalues = np.maximum(eigenvalues, 0.0)
            omega = np.sqrt(eigenvalues)

        if not self.tda:
            # RPA excitation vector (X+Y), the quantity the transition dipole and
            # transition density couple to:
            #     (X+Y)_n = omega_n^{-1/2} (A-B)^{1/2} Z_n = omega_n^{-1/2} sqrt(dE) Z_n
            # (for pure DFT, A-B = diag(dE)). The previous code used
            # (A-B)^{-1/2} Z = inv_sqrt_dE * Z, which is proportional to (X-Y), not
            # (X+Y): it weights by 1/sqrt(dE) instead of sqrt(dE), over-weighting
            # small-gap transitions and inflating oscillator strengths (and the
            # amplitude-basis transition densities used in subsystem coupling) by
            # ~1/omega. This form reduces correctly to the TDA result as K->0.
            omega_safe = np.sqrt(np.maximum(omega, 1e-30))
            self.xpy = (self._sqrt_dE[:, None] * eigenvectors) / omega_safe[None, :]

        self.omega = omega
        self.Z = eigenvectors

        if self.rank == 0 and verbose:
            print("\nExcitation energies (eV):")
            for i, w in enumerate(omega[:min(k, 10)]):
                print(f"  State {i+1}: {w * 27.2114:.4f} eV")

        return omega, eigenvectors

    # --- Dense-matrix Casida ---

    @timer("Casida Matrix (MPI)")
    def build_matrices(self, tda: bool = False):
        """Dense Casida build: MPI row distribution + A/B/C assembly.

        The basis-specific K rows (transition densities, Hartree + f_xc,
        USPP augmentation) are computed by the kernel backend.
        """
        if self.polarized:
            raise NotImplementedError("Spin-polarized Casida not implemented.")
        if not hasattr(self, "_occ_e") or not hasattr(self, "_psi_occ"):
            raise RuntimeError("Call set_active_orbitals() first.")

        n_o = self._kernel.n_occ
        n_u = self._kernel.n_unocc
        Ntrans = self._kernel.n_trans

        if self.rank == 0:
            print(f"Building Casida matrices: {n_o} occ x {n_u} unocc = {Ntrans} transitions")
            print(f"MPI ranks: {self.size} (CPU only)")

        dE = self._kernel.diagonal_dE()
        if np.any(dE <= 0.0):
            raise ValueError("Found non-positive excitation energies. Check orbital ordering.")

        local_rows = [ia for ia in range(Ntrans) if ia % self.size == self.rank]
        n_local = len(local_rows)
        K_local = self._kernel.dense_K_rows(local_rows, verbose=(self.rank == 0))
        if self.use_eDFTpy:
            self.transition_densities = self._kernel.transition_densities

        self.comm.gather(n_local, root=0)
        if self.rank == 0:
            K_full = np.zeros((Ntrans, Ntrans), dtype=float)
        else:
            K_full = None

        K_gathered = self.comm.gather(K_local, root=0)
        all_rows = self.comm.gather(local_rows, root=0)

        if self.rank == 0:
            for rows, K_part in zip(all_rows, K_gathered):
                for local_idx, global_row in enumerate(rows):
                    K_full[global_row, :] = K_part[local_idx, :]

        if self.rank == 0:
            K_sl = K_full
            A = K_sl + np.diag(dE)
            B = K_sl.copy()

            self.A = np.asarray(A)
            self.B = np.asarray(B)
            self._n_occ = n_o
            self._n_unocc = n_u
            self._n_trans = Ntrans
            self.tda = tda

            if tda:
                self.C = self.A.copy()
            else:
                AmB = self.A - self.B
                ApB = self.A + self.B
                s, U = scipy_eigh(AmB)
                threshold = 1e-10
                s_clip = np.maximum(s, threshold)
                sqrt_s = np.sqrt(s_clip)
                sqrt_AmB = (U @ np.diag(sqrt_s)) @ U.T
                self.C = sqrt_AmB @ ApB @ sqrt_AmB
                self._AmB_evals = s
                self._AmB_evecs = U

            print(f"Matrix construction complete. C shape: {self.C.shape}")

        self.A = self.comm.bcast(self.A if self.rank == 0 else None, root=0)
        self.B = self.comm.bcast(self.B if self.rank == 0 else None, root=0)
        self.C = self.comm.bcast(self.C if self.rank == 0 else None, root=0)
        self.tda = self.comm.bcast(self.tda if self.rank == 0 else None, root=0)
        self._n_occ = self.comm.bcast(self._n_occ if self.rank == 0 else None, root=0)
        self._n_unocc = self.comm.bcast(self._n_unocc if self.rank == 0 else None, root=0)
        self._n_trans = self.comm.bcast(self._n_trans if self.rank == 0 else None, root=0)

        if not self.tda:
            self._AmB_evals = self.comm.bcast(self._AmB_evals if self.rank == 0 else None, root=0)
            self._AmB_evecs = self.comm.bcast(self._AmB_evecs if self.rank == 0 else None, root=0)

        return self.A, self.B, self.C

    def solve(self, k=None, use_davidson=None, davidson_tol=1e-8, davidson_maxiter=200,
              verbose=1, matrix_free=False):
        if matrix_free or (hasattr(self, "_matrix_free") and self._matrix_free and self.C is None):
            if not hasattr(self, "_matrix_free") or not self._matrix_free:
                raise RuntimeError("Call setup_matrix_free() first for matrix-free mode.")
            if k is None:
                raise ValueError("Matrix-free mode requires k to be specified.")
            return self.solve_matrix_free(k=k, tol=davidson_tol, maxiter=davidson_maxiter, verbose=verbose)

        if self.C is None:
            raise RuntimeError("Run build_matrices(...) first.")

        n = self.C.shape[0]
        if use_davidson is None:
            use_davidson = (n > 500 and k is not None and k < n // 2)

        if use_davidson and k is None:
            if self.rank == 0:
                print("Warning: Davidson requires k to be specified. Using full diagonalization.")
            use_davidson = False

        if use_davidson:
            if self.rank == 0:
                print(f"Using Davidson solver for {k} eigenvalues (matrix size: {n}x{n})")
            w2, Z = davidson(
                self.C,
                nroots=k,
                tol=davidson_tol,
                maxiter=davidson_maxiter,
                verbose=verbose if self.rank == 0 else 0,
            )
        else:
            if self.rank == 0 and n > 500:
                print(f"Using full diagonalization (matrix size: {n}x{n})")

            w2, Z = np.linalg.eigh(self.C)

            if k is not None:
                w2 = w2[:k]
                Z = Z[:, :k]

        if self.tda:
            omega = w2
        else:
            if np.any(w2 < -1e-6):
                import warnings
                warnings.warn(
                    f"Found {np.sum(w2 < -1e-6)} negative eigenvalues in C matrix. "
                    "This may indicate RPA instability."
                )
            w2 = np.maximum(w2, 0.0)
            omega = np.sqrt(w2)

        if not self.tda:
            # (X+Y)_n = omega_n^{-1/2} (A-B)^{1/2} Z_n  (see solve_matrix_free note).
            # Build (A-B)^{1/2} from the stored A-B eigendecomposition, then scale
            # each column by omega_n^{-1/2}. The previous code used (A-B)^{-1/2},
            # which is proportional to (X-Y) and inflated intensities by ~1/omega.
            s = self._AmB_evals
            U = self._AmB_evecs
            sqrt_s = np.sqrt(np.maximum(s, 0.0))
            sqrt_AmB = (U * sqrt_s) @ U.T
            omega_safe = np.sqrt(np.maximum(omega, 1e-30))
            self.xpy = (sqrt_AmB @ Z) / omega_safe[None, :]

        self.omega = omega
        self.Z = Z
        return omega, Z

    def __call__(self, k=None):
        if self.rank != 0:
            raise RuntimeError("Call solve/oscillator_strengths from rank 0 only.")
        omega, Z = self.solve(k=k)
        f = self.oscillator_strengths(k=k)
        return omega, f, Z


def run_casida(
    kernel,
    options: CasidaOptions,
    comm=None,
) -> CasidaResults:
    """Solve Casida using a pre-built kernel backend (PW or GTO).

    Typical GTO usage::

        from casidapy.pyscf_adapter import extract_gto_kernel
        from casidapy.casida_engine import run_casida

        kernel, opts = extract_gto_kernel(mf, n_states=10, tda=True)
        results = run_casida(kernel, opts)
    """
    from casidapy.kernels.gto import GTOKernel
    from casidapy.kernels.plane_wave import PlaneWaveKernel

    comm = MPI.COMM_WORLD if comm is None else comm
    rank = mpi_comm_rank(comm)

    if not hasattr(kernel, "apply_K"):
        raise TypeError("kernel must implement apply_K (KernelBackend protocol)")

    # Algebra layer: reuse CasidaKS_MPI with an injected kernel.
    # For GTO, pass a dummy rho/xc only if needed — inject kernel directly.
    if isinstance(kernel, PlaneWaveKernel):
        casida = CasidaKS_MPI(
            kernel.rho,
            kernel.functional,
            comm=comm,
            kernel=kernel,
            use_eDFTpy=kernel.use_eDFTpy,
            use_uspp=kernel.use_uspp,
        )
        # orbitals already on kernel
        casida.set_active_orbitals(
            kernel._occ_e, kernel._unocc_e, kernel._psi_occ, kernel._psi_unocc
        )
    elif isinstance(kernel, GTOKernel):
        # Minimal shell: attach GTO kernel without PW fxc construction
        casida = object.__new__(CasidaKS_MPI)
        casida.comm = comm
        casida.rank = rank
        casida.size = mpi_comm_size(comm)
        casida._kernel = kernel
        casida.rho = None
        casida.grid = None
        casida._grid_shape = None
        casida.N = 0.0
        casida.polarized = False
        casida.use_uspp = False
        casida.beta_projectors = None
        casida.qij_augmentation = None
        casida.use_eDFTpy = False
        casida.transition_densities = None
        casida.hartree = None
        casida.functional = None
        casida.triplet = False
        casida.fkxc_arr = None
        casida.fkxc = None
        casida._fkxc_ndarray = None
        casida.A = casida.B = casida.C = None
        casida.omega = casida.f = casida.xpy = casida.Z = None
        casida.mu_transition = casida.d_mode = None
        casida.tda = False
        casida.n_occ = kernel.n_occ
        casida.n_unocc = kernel.n_unocc
        casida._n_occ = kernel.n_occ
        casida._n_unocc = kernel.n_unocc
        casida._occ_e = kernel._occ_e
        casida._unocc_e = kernel._unocc_e
        casida._psi_occ = []  # unused for GTO dipole path
        casida._psi_unocc = []
    else:
        raise TypeError(f"Unsupported kernel type: {type(kernel)}")

    n_trans = kernel.n_trans
    n_states_actual = min(options.n_states, n_trans)

    t0 = MPI.Wtime()

    # Hybrid full TDDFT: A-B ≠ diag(dE) → dense A/B via PySCF get_ab.
    # Pure-functional RPA stays matrix-free (fast C-chain).
    hybrid_rpa = False
    if isinstance(kernel, GTOKernel) and not options.tda:
        kernel.setup(tda=False)
        hybrid_rpa = bool(getattr(kernel, "hybrid", False))

    if hybrid_rpa:
        if rank == 0:
            print(
                "GTO hybrid full TDDFT: dense A/B from PySCF get_ab() "
                f"(n_trans={n_trans})"
            )
        A, B = kernel.build_AB()
        casida.A = np.asarray(A, dtype=float)
        casida.B = np.asarray(B, dtype=float)
        casida.tda = False
        casida._n_trans = n_trans
        casida._n_occ = kernel.n_occ
        casida._n_unocc = kernel.n_unocc
        casida._dE = kernel.diagonal_dE()
        casida._sqrt_dE = np.sqrt(casida._dE)
        AmB = casida.A - casida.B
        ApB = casida.A + casida.B
        s, U = scipy_eigh(AmB)
        threshold = 1e-10
        s_clip = np.maximum(s, threshold)
        sqrt_AmB = (U * np.sqrt(s_clip)) @ U.T
        casida.C = sqrt_AmB @ ApB @ sqrt_AmB
        casida._AmB_evals = s
        casida._AmB_evecs = U
        casida._matrix_free = False
        t_build = MPI.Wtime() - t0
        t1 = MPI.Wtime()
        omega, Z = casida.solve(
            k=n_states_actual,
            use_davidson=False,
            verbose=1 if rank == 0 else 0,
        )
        t_solve = MPI.Wtime() - t1
        options.matrix_free = False
    else:
        if not options.matrix_free and isinstance(kernel, GTOKernel):
            # Pure GTO TDA/RPA: matrix-free is the supported fast path
            if rank == 0:
                print("GTO backend: forcing matrix_free=True")
            options.matrix_free = True

        if options.matrix_free:
            casida.setup_matrix_free(tda=options.tda)
            t_build = MPI.Wtime() - t0
            t1 = MPI.Wtime()
            omega, Z = casida.solve_matrix_free(
                k=n_states_actual,
                tol=options.solver_tol,
                maxiter=options.solver_maxiter,
                verbose=1 if rank == 0 else 0,
                method=options.solver_method,
            )
            t_solve = MPI.Wtime() - t1
        else:
            casida.build_matrices(tda=options.tda)
            t_build = MPI.Wtime() - t0
            t1 = MPI.Wtime()
            omega, Z = casida.solve(k=n_states_actual)
            t_solve = MPI.Wtime() - t1

    f = casida.oscillator_strengths(k=n_states_actual)
    rho_state = None
    if isinstance(kernel, PlaneWaveKernel) and options.use_eDFTpy:
        rho_state = casida.collapse_transition_densities_to_state_basis(n_states_actual)

    return CasidaResults(
        omega=np.asarray(omega),
        f=np.asarray(f),
        Z=np.asarray(Z),
        mu_transition=None if casida.mu_transition is None else np.asarray(casida.mu_transition),
        rho_transition=rho_state,
        d_mode=None if casida.d_mode is None else np.asarray(casida.d_mode),
        xpy=np.asarray(casida.xpy) if casida.xpy is not None else np.asarray(casida.Z),
        metadata={
            "basis": getattr(options, "basis", "pw"),
            "kernel": type(kernel).__name__,
            "n_trans": n_trans,
            "n_states": n_states_actual,
            "tda": options.tda,
            "matrix_free": options.matrix_free,
            "hybrid_rpa_dense": hybrid_rpa,
            "t_build": t_build,
            "t_solve": t_solve,
        },
    )


def run_casida_in_memory(inputs: CasidaInputs, options: CasidaOptions, comm=None) -> CasidaResults:
    """Plane-wave Casida from :class:`CasidaInputs` (QE / DFTpy grid path).

    For GTO / PySCF ground states use
    :func:`run_casida` with a :class:`~casidapy.kernels.gto.GTOKernel`.
    """
    basis = getattr(options, "basis", "pw") or "pw"
    if str(basis).lower() == "gto":
        raise ValueError(
            "options.basis='gto' requires run_casida(GTOKernel, options). "
            "Use casidapy.pyscf_adapter.extract_gto_kernel(mf)."
        )

    return _run_casida_plane_wave(inputs, options, comm=comm)


def _run_casida_plane_wave(inputs: CasidaInputs, options: CasidaOptions, comm=None) -> CasidaResults:
    comm = MPI.COMM_WORLD if comm is None else comm
    rank = comm.Get_rank()

    atoms = inputs.atoms
    ions = Ions.from_ase(atoms)
    grid = inputs.grid
    rho_ks = inputs.rho_ks
    eigs = np.asarray(inputs.eigs).copy()
    occs = np.asarray(inputs.occs).copy()
    psi_list = inputs.psi
    # Pseudo / core density
    core = None
    if options.pseudo_map:
        missing = set(atoms.get_chemical_symbols()) - set(options.pseudo_map.keys())
        if missing:
            raise ValueError(f"Missing pseudo_map entries for elements: {sorted(missing)}")
        pseudo = LocalPseudo(grid=rho_ks.grid, ions=ions, PP_list=options.pseudo_map)
        core = pseudo.core_density

    libxc_xc_components = None
    if options.spin_state == "triplet":
        xc_up = options.xc.upper()
        if xc_up not in XC_TO_LIBXC_COMPONENTS:
            raise ValueError(f"No XC->libxc mapping for triplet: {options.xc}")
        libxc_xc_components = XC_TO_LIBXC_COMPONENTS[xc_up]

    xc = Functional(type="XC", name=options.xc, core_density=core)
    totalfunctional = TotalFunctional(KE=None, XC=xc)

    if options.of_context:
        if not options.pseudo_map:
            raise ValueError("of_context=True requires pseudo_map.")
        of_ctx = build_of_functional_context(
            rho_ks=rho_ks,
            ions=ions,
            xc_name=options.xc,
            pp_list=options.pseudo_map,
        )
        totalfunctional = of_ctx["totalfunctional"]
        if rank == 0:
            print(f"Orbital Free enabled, |v_ext|_2={of_ctx['v_ext_norm']:.6e}", flush=True)

        # Call the hamiltonian from DFTpy and diagonalize it 
        from dftpy.td import hamiltonian
        Hamiltonian = hamiltonian(potential= totalfunctional(rho_ks).potential)
        eigs, psi_list = Hamiltonian.diagonalize()
        
    psi_list = psi_to_fields(psi_list, grid=grid, ions=ions, comm=comm)
    psi_list = normalize_wavefunctions(psi_list, grid)

    n_total_occ = options.n_total_occ if options.n_total_occ is not None else int(np.sum(occs > 0.01))
    print('N total occ',n_total_occ)
    #n_total_occ = 4
    casida = CasidaKS_MPI(
        rho_ks,
        totalfunctional,
        spin_state=options.spin_state,
        libxc_xc_components=libxc_xc_components,
        use_eDFTpy=options.use_eDFTpy,
        use_uspp=options.use_uspp,
        beta_projectors=inputs.beta_projectors,
        qij_augmentation=inputs.qij_augmentation,
        use_gpu=options.use_gpu,
        comm=comm,
    )
    if eigs.ndim == 2:
        eigs = eigs[0]
    occ_e, unocc_e, psi_occ, psi_unocc = slice_active_space(
        eigs, psi_list, options.n_occ, options.n_unocc,
        n_total_occ=n_total_occ, verbose=(rank == 0),
    )
    print(
    f"rank {rank}: len(eigs)={len(eigs)}, len(occs)={len(occs)}, "
    f"sum(occs>0.5)={int(np.sum(occs > 0.5))}, "
    f"n_total_occ(will use)={options.n_total_occ if options.n_total_occ is not None else int(np.sum(occs > 0.5))}, "
    f"options n_occ={options.n_occ} n_unocc={options.n_unocc}",
    flush=True)
    casida.set_active_orbitals(occ_e, unocc_e, psi_occ, psi_unocc)
    print(
        f"rank {rank}: len(psi_all)={len(psi_list)}, "
        f"n_o={len(psi_occ)}, n_u={len(psi_unocc)}, "
        f"Ntrans={len(psi_occ) * len(psi_unocc)}",
        flush=True,
    )
    n_trans = casida.n_occ * casida.n_unocc
    n_states_actual = min(options.n_states, n_trans)

    t0 = MPI.Wtime()
    if options.matrix_free:
        casida.setup_matrix_free(tda=options.tda)
        t_build = MPI.Wtime() - t0
        t1 = MPI.Wtime()
        omega, Z = casida.solve_matrix_free(
            k=n_states_actual,
            tol=options.solver_tol,
            maxiter=options.solver_maxiter,
            verbose=1 if rank == 0 else 0,
            method=options.solver_method,
        )
        t_solve = MPI.Wtime() - t1
    else:
        casida.build_matrices(tda=options.tda)
        t_build = MPI.Wtime() - t0
        t1 = MPI.Wtime()
        omega, Z = casida.solve(k=n_states_actual)
        t_solve = MPI.Wtime() - t1

    f = casida.oscillator_strengths(k=n_states_actual)
    rho_state = casida.collapse_transition_densities_to_state_basis(n_states_actual)
    return CasidaResults(
        omega=np.asarray(omega),
        f=np.asarray(f),
        Z=np.asarray(Z),
        mu_transition=None if casida.mu_transition is None else np.asarray(casida.mu_transition),
        rho_transition=rho_state,
        d_mode=None if casida.d_mode is None else np.asarray(casida.d_mode),
        xpy=np.asarray(casida.xpy) if casida.xpy is not None else np.asarray(casida.Z),
        metadata={
            "n_total_occ": n_total_occ,
            "n_trans": n_trans,
            "n_states_actual": n_states_actual,
            "matrix_free": options.matrix_free,
            "tda": options.tda,
            "xc": options.xc,
            "of_context": options.of_context,
            "rho_basis": "amplitude_xpy",
            "timings_s": {"build_or_setup": float(t_build), "solve": float(t_solve)},
        },
    )
