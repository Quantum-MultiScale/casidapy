#Preparing for Casida calculation, defining the inputs and outputs
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from mpi4py import MPI

from dftpy.field import DirectField
from dftpy.functional import Functional, LocalPseudo, TotalFunctional
from dftpy.ions import Ions
from dftpy.time_data import timer

from scipy.linalg import eigh as scipy_eigh
from scipy.sparse.linalg import LinearOperator

from casidapy.casida_api import CasidaInputs, CasidaOptions, CasidaResults
from casidapy.casida_of import build_of_functional_context
from casidapy.casida_utils import (
    _as_scalar_field_on_grid,
    _compute_fxc_triplet_pylibxc,
    XC_TO_LIBXC_COMPONENTS,
    build_energy_differences,
    build_initial_guess,
    filtered,
    mpi_comm_rank,
    mpi_comm_size,
    normalize_wavefunctions,
    psi_to_fields,
    proj_overlap,
    rho_aug,
    augmentation_integral,
)
from casidapy.davidson import davidson, solve_eigsh, solve_lobpcg
from casidapy.qepy_adapter import slice_active_space


class CasidaKS_MPI:
    """Casida equation solver with MPI parallelization (CPU only)."""

    def __init__(self, rho_ks, xc_functional, polarized=False,
                 rho_cutoff=1e-3, fxc_max=20.0, use_gpu=False,
                 spin_state="singlet", libxc_xc_components=None, comm=None,
                 use_uspp=False, beta_projectors=None, qij_augmentation=None,
                 use_eDFTpy=False):
        self.rho = rho_ks
        self.grid = rho_ks.grid
        self._grid_shape = self.grid.nr
        self.N = float(rho_ks.integral())
        self.polarized = bool(polarized)

        self.use_uspp = bool(use_uspp) and beta_projectors is not None
        self.beta_projectors = beta_projectors
        self.qij_augmentation = qij_augmentation

        self.use_eDFTpy = bool(use_eDFTpy)
        self.transition_densities = None
        if comm is None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm
        self.rank = mpi_comm_rank(self.comm)
        self.size = mpi_comm_size(self.comm)

        self.hartree = Functional(type="HARTREE")
        self.functional = xc_functional

        spin_state = spin_state.lower()
        if spin_state not in ("singlet", "triplet"):
            raise ValueError(f"spin_state must be 'singlet' or 'triplet', got '{spin_state}'")
        self.triplet = (spin_state == "triplet")

        if self.triplet:
            if libxc_xc_components is None:
                raise ValueError(
                    "libxc_xc_components must be specified for triplet calculations."
                )
            component_names = list(libxc_xc_components)
            f_xc_T = _compute_fxc_triplet_pylibxc(rho_ks, component_names, self.grid)
            fxc_arr = filtered(f_xc_T, fxc_max, rho_ks, rho_cutoff)
            self.fkxc_arr = DirectField(self.grid, rank=1, griddata_3d=fxc_arr)
        else:
            self.fkxc_raw = xc_functional(rho_ks, calcType=["V2"]).v2rho2
            fkxc = _as_scalar_field_on_grid(self.grid, self.fkxc_raw)
            fxc_arr = filtered(fkxc, fxc_max, rho_ks, rho_cutoff)
            self.fkxc_arr = DirectField(self.grid, rank=1, griddata_3d=fxc_arr)

        self.fkxc = self.fkxc_arr
        self._fkxc_ndarray = np.asarray(fxc_arr)

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

    def set_active_orbitals(self, occ_eigs, unocc_eigs, psi_occ, psi_unocc):
        """Store pre-sliced active occupied/unoccupied orbital sets.

        Use ``qepy_adapter.slice_active_space`` to obtain the four arguments
        from full KS arrays before calling this method.
        """
        self._occ_e = np.asarray(occ_eigs, dtype=float).copy()
        self._unocc_e = np.asarray(unocc_eigs, dtype=float).copy()
        self._psi_occ = list(psi_occ)
        self._psi_unocc = list(psi_unocc)
        self._n_occ = len(self._psi_occ)
        self._n_unocc = len(self._psi_unocc)
        self.n_occ = self._n_occ
        self.n_unocc = self._n_unocc

        self._proj_occ = None
        self._proj_unocc = None
        if self.use_uspp:
            self._proj_occ = proj_overlap(self.beta_projectors, self.grid, self._psi_occ)
            self._proj_unocc = proj_overlap(self.beta_projectors, self.grid, self._psi_unocc)

    def transition_orbital(self, psi_i, psi_a, proj_i=None, proj_a=None):
        """Transition density ψ_i* ψ_a + USPP augmentation if enabled."""
        phi = psi_i.conj() * psi_a
        if self.use_uspp and proj_i is not None and proj_a is not None:
            phi = phi + rho_aug(self.grid, self.qij_augmentation, proj_i, proj_a)
        return phi

    def _hartree_potential(self, rho):
        """Compute Hartree potential from density."""
        if np.iscomplexobj(np.asarray(rho)):
            rho = rho.real
        return self.hartree(rho, calcType=["V"]).potential

    def _kernel_element_cpu(self, phi_ia, phi_jb, vh_ia,
                            proj_i=None, proj_a=None, proj_j=None, proj_b=None):
        """Compute kernel matrix element K[ia,jb] on CPU with optional USPP augmentation."""
        xc_part = (phi_ia.conj() * self.fkxc_arr * phi_jb).integral()
        if vh_ia is not None:
            hartree_part = (phi_jb.conj() * vh_ia).integral()
            k_val = hartree_part + xc_part
        else:
            k_val = xc_part

        if self.use_uspp and proj_j is not None and proj_b is not None:
            dV = float(self.grid.dV)
            response = np.asarray(self.fkxc_arr) * np.asarray(phi_ia)
            if vh_ia is not None:
                response = np.asarray(vh_ia) + response
            aug_corr = augmentation_integral(
                response, self.qij_augmentation, proj_j, proj_b, dV
            )
            k_val = float(k_val) + aug_corr

        return float(k_val)

    def _dipole_matrix(self, psi_occ, psi_unocc):
        """Transition dipole matrix elements mu_ia^alpha with optional USPP augmentation."""
        n_o = len(psi_occ)
        n_u = len(psi_unocc)
        dV = float(self.grid.dV)

        # μ[ia, α] = dV Σ_r ψ_i(r) r_α(r) ψ_a(r)
        # BLAS:  mu_α = (psi_o * r_α) @ psi_u.T * dV  (n_o, n_u)
        psi_o = np.stack([np.asarray(p).ravel() for p in psi_occ])   # (n_o, n_grid)
        psi_u = np.stack([np.asarray(p).ravel() for p in psi_unocc]) # (n_u, n_grid)
        rx = np.asarray(self.grid.r[0]).ravel()
        ry = np.asarray(self.grid.r[1]).ravel()
        rz = np.asarray(self.grid.r[2]).ravel()

        mu_x = (psi_o * rx) @ psi_u.T * dV  # (n_o, n_u)
        mu_y = (psi_o * ry) @ psi_u.T * dV
        mu_z = (psi_o * rz) @ psi_u.T * dV
        mu = np.stack([mu_x.ravel(), mu_y.ravel(), mu_z.ravel()], axis=-1)  # (n_trans, 3)

        if self.use_uspp and self._proj_occ is not None:
            for i in range(n_o):
                for a in range(n_u):
                    ia = i * n_u + a
                    p_i = self._proj_occ[:, i]
                    p_a = self._proj_unocc[:, a]
                    rho_a = rho_aug(self.grid, self.qij_augmentation, p_i, p_a)
                    rho_arr = np.asarray(rho_a).ravel()
                    mu[ia, 0] += float(np.dot(rx, rho_arr).real * dV)
                    mu[ia, 1] += float(np.dot(ry, rho_arr).real * dV)
                    mu[ia, 2] += float(np.dot(rz, rho_arr).real * dV)

        return mu

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

        if not hasattr(self, "_psi_occ") or not hasattr(self, "_psi_unocc"):
            raise RuntimeError("Call build_matrices(...) or setup_matrix_free(...) first.")
        psi_occ = self._psi_occ
        psi_unocc = self._psi_unocc

        if self.triplet:
            n_states = len(self.omega)
            self.f = np.zeros(n_states, dtype=np.float64)
            self.mu_transition = None
            self.d_mode = None
            return self.f

        mu = self._dipole_matrix(psi_occ, psi_unocc)
        self.mu_transition = mu

        if self.tda:
            d = mu.T @ self.Z
        else:
            d = mu.T @ self.xpy
        self.d_mode = d

        d2 = np.sum(d * d, axis=0)
        f = (2.0 / 3.0) * self.omega * d2
        s = float(np.sum(f))
        #if s > 0.0:
        #    f = f / s
        self.f = f
        return f

    def collapse_transition_densities_to_state_basis(self, k: int) -> Optional[np.ndarray]:
        """ρ_k(r) = Σ_ia amp_ia,k φ_ia(r); amp = Z (TDA) or xpy (RPA).

        Non-USPP path: vectorized loop over occupied orbitals — one DGEMM per i,
        so only n_occ iterations instead of n_occ * n_unocc.  USPP path falls back
        to the original element-wise loop (augmentation charge per pair).
        """
        if not getattr(self, "use_eDFTpy", False):
            return None
        if not hasattr(self, "_psi_occ"):
            return None
        if self.tda:
            amp = np.real(np.asarray(self.Z[:, :k], dtype=float))
        else:
            if self.xpy is None:
                raise RuntimeError(
                    "RPA requires xpy before collapsing transition densities",
                )
            amp = np.real(np.asarray(self.xpy[:, :k], dtype=float))
        n_trans, n_states = amp.shape
        n_unocc = self._n_unocc

        use_fast = (
            hasattr(self, "_psi_occ_arr")
            and not (self.use_uspp and self._proj_occ is not None)
        )
        if use_fast:
            # φ_k(r) = Σ_i ψ_i(r) · [Σ_a amp[i,a,k] ψ_a(r)]
            # loop over i only; inner sum is a DGEMM.
            n_flat = self._psi_occ_arr.shape[1]
            amp_3d = amp.reshape(self._n_occ, n_unocc, n_states)  # (n_occ, n_unocc, n_states)
            phi_flat = np.zeros((n_states, n_flat), dtype=np.float64)
            for i in range(self._n_occ):
                # c_i[k, r] = Σ_a amp[i,a,k] * ψ_a(r)  →  DGEMM (n_states, n_flat)
                c_i = amp_3d[i].T @ self._psi_unocc_arr
                phi_flat += c_i * self._psi_occ_arr[i]
            return phi_flat.reshape((n_states, *self._grid_shape))

        # Fallback: original per-(i,a) loop; handles USPP augmentation.
        phi_states = None
        for i in range(self._n_occ):
            for a in range(n_unocc):
                ia = i * n_unocc + a
                row = amp[ia]
                if not np.any(row):
                    continue
                p_i = self._proj_occ[:, i] if (self.use_uspp and self._proj_occ is not None) else None
                p_a = self._proj_unocc[:, a] if (self.use_uspp and self._proj_unocc is not None) else None
                phi_ia = np.real(
                    np.asarray(
                        self.transition_orbital(
                            self._psi_occ[i], self._psi_unocc[a], proj_i=p_i, proj_a=p_a,
                        )
                    )
                ).astype(float)
                if phi_states is None:
                    phi_states = [np.zeros_like(phi_ia) for _ in range(n_states)]
                for state_k in range(n_states):
                    c = row[state_k]
                    if c != 0.0:
                        phi_states[state_k] += c * phi_ia
        if phi_states is None:
            return None
        return np.stack(phi_states, axis=0)  # (n_states, nx, ny, nz)

    # --- Matrix-free Casida (LOBPCG / eigsh on matvec) ---

    def setup_matrix_free(self, tda: bool = False):
        if self.polarized:
            raise NotImplementedError("Spin-polarized Casida not implemented.")
        if not hasattr(self, "_occ_e") or not hasattr(self, "_psi_occ"):
            raise RuntimeError("Call set_active_orbitals() first.")

        self._n_trans = self._n_occ * self._n_unocc
        self.tda = tda

        self._dE = build_energy_differences(self._occ_e, self._unocc_e)

        if np.any(self._dE <= 0.0):
            raise ValueError("Found non-positive excitation energies. Check orbital ordering.")

        self._sqrt_dE = np.sqrt(self._dE)
        self._dV = float(self.grid.dV)
        self._grid_shape = self.grid.nr

        # Keep the USPP projector overlaps computed in set_active_orbitals.
        # Previously these were reset to None here, which silently disabled the
        # USPP augmentation branches in the matvec (they are gated on
        # ``_proj_occ is not None``). Recompute defensively if missing.
        if self.use_uspp and getattr(self, "_proj_occ", None) is None:
            self._proj_occ = proj_overlap(self.beta_projectors, self.grid, self._psi_occ)
            self._proj_unocc = proj_overlap(self.beta_projectors, self.grid, self._psi_unocc)

        # Pre-cache flat orbital arrays so the hot matvec loop avoids repeated
        # np.asarray() calls and can use BLAS DGEMM instead of Python loops.
        n_flat = int(np.prod(self._grid_shape))
        self._psi_occ_arr = np.empty((self._n_occ, n_flat), dtype=np.float64)
        self._psi_unocc_arr = np.empty((self._n_unocc, n_flat), dtype=np.float64)
        for _i, _p in enumerate(self._psi_occ):
            self._psi_occ_arr[_i] = np.asarray(_p).ravel()
        for _a, _p in enumerate(self._psi_unocc):
            self._psi_unocc_arr[_a] = np.asarray(_p).ravel()

        self._matrix_free = True

        # NOTE: transition densities are NOT materialized here. The matvec
        # (`_apply_K_matvec`) recomputes φ_ia on the fly from the active
        # orbitals, and `collapse_transition_densities_to_state_basis` does the
        # same at the end. Storing all Ntrans full-grid arrays (replicated on
        # every rank) was the dominant memory cost and is unnecessary.

        if self.rank == 0:
            n_grid = np.prod(self._grid_shape)
            wfn_mem = (self._n_occ + self._n_unocc) * n_grid * 8 / 1e9
            matrix_mem = self._n_trans ** 2 * 8 / 1e9
            print("Matrix-free mode initialized:")
            print(f"  Transitions: {self._n_occ} occ × {self._n_unocc} unocc = {self._n_trans}")
            print(f"  Grid: {self._grid_shape} = {n_grid:,} points")
            print(f"  Wavefunction memory: {wfn_mem:.2f} GB")
            print(f"  (Full matrix would be: {matrix_mem:.2f} GB)")
            print(f"  TDA: {tda}, CPU only")

    def _apply_K_matvec(self, v):
        """
        Apply the eh coupling block ``K`` to a transition vector ``v`` (length Ntrans) without
        forming ``K`` explicitly.

        Indexing: ``ia = i * n_unocc + a`` for occupied ``i`` and unoccupied ``a``.  ``K`` is the
        Hartree + adiabatic ``f_xc`` (singlet) or ``f_xc`` only (triplet) kernel between
        transition densities ``φ_i^* φ_a``.

        Both the rho_v accumulation and the Kv projection are done as BLAS DGEMMs using
        pre-cached flat orbital arrays (_psi_occ_arr, _psi_unocc_arr) set up in
        setup_matrix_free().  Python loops are kept only for the USPP augmentation path.
        """
        Ntrans = self._n_trans
        n_occ = self._n_occ
        n_unocc = self._n_unocc
        v_arr = np.asarray(v, dtype=np.float64)

        # ρ_v(r) = Σ_{jb} v[jb] ψ_j(r) ψ_b(r)
        # BLAS path:  V = v.reshape(n_occ, n_unocc)
        #             w = V @ psi_unocc_arr      (n_occ, n_grid) DGEMM
        #             rho_v = (psi_occ_arr * w).sum(axis=0)     (n_grid,)
        V = v_arr.reshape(n_occ, n_unocc)
        w = V @ self._psi_unocc_arr                       # (n_occ, n_grid)
        rho_v_flat = (self._psi_occ_arr * w).sum(axis=0)  # (n_grid,)
        rho_v = rho_v_flat.reshape(self._grid_shape)

        if self.use_uspp and self._proj_occ is not None:
            for j in range(n_occ):
                for b in range(n_unocc):
                    jb = j * n_unocc + b
                    if abs(v_arr[jb]) < 1e-30:
                        continue
                    p_j = self._proj_occ[:, j]
                    p_b = self._proj_unocc[:, b]
                    rho_v += np.asarray(rho_aug(
                        self.grid, self.qij_augmentation, p_j, p_b
                    )) * v_arr[jb]

        fkxc_arr = np.asarray(self._fkxc_ndarray)

        if self.triplet:
            response_flat = (fkxc_arr * rho_v).ravel()
        else:
            rho_v_field = DirectField(self.grid, rank=1, griddata_3d=rho_v)
            vh_v = self._hartree_potential(rho_v_field)
            response_flat = (np.asarray(vh_v) + fkxc_arr * rho_v).ravel()

        # Kv[i,a] = dV Σ_r ψ_i(r) ψ_a(r) response(r), one DGEMM.
        #
        # IMPORTANT: this matvec must be collective-free. The matrix-free
        # eigensolver (eigsh/lobpcg) runs independently and redundantly on every
        # rank, and ARPACK's reverse-communication calls the matvec a
        # *rank-dependent* number of times (different random initial guesses,
        # non-bit-identical FP). Any MPI collective here (Allreduce) therefore
        # deadlocks the moment two ranks' iteration counts diverge — which is
        # exactly what hung the multi-rank runs. Each rank computes the full Kv
        # itself; the cost is dominated by the per-rank Hartree FFT above, so
        # distributing this trivial DGEMM bought nothing anyway.
        Kv = (
            (self._psi_occ_arr * response_flat) @ self._psi_unocc_arr.T * self._dV
        ).ravel()

        if self.use_uspp and self._proj_occ is not None:
            for i in range(n_occ):
                for a in range(n_unocc):
                    ia = i * n_unocc + a
                    p_i = self._proj_occ[:, i]
                    p_a = self._proj_unocc[:, a]
                    Kv[ia] += augmentation_integral(
                        response_flat.reshape(self._grid_shape),
                        self.qij_augmentation, p_i, p_a, self._dV
                    )

        return Kv

    def _apply_C_matvec(self, v):
        """
        Matrix-free multiply by the ``C`` operator used in **full** Casida (non-TDA).

        With diagonal KS energy gaps ``Δε`` and ``S = diag(√Δε)``, this implements
        ``C v = S ( 2 K S v + Δε ⊙ S v )``, i.e. the similarity-transformed RPA chain so the
        eigenproblem matches ``solve_matrix_free`` / dense ``C`` for the same physics.
        """
        v = np.asarray(v)
        input_shape = v.shape
        v_flat = v.flatten()

        w1 = self._sqrt_dE * v_flat
        Kw1 = self._apply_K_matvec(w1)
        w2 = 2.0 * Kw1 + self._dE * w1
        result = self._sqrt_dE * w2
        return result.reshape(input_shape)

    def _apply_A_matvec(self, v):
        """
        Matrix-free multiply by ``A = K + diag(Δε)`` for **TDA** (solve ``A X = ω X``).
        """
        v = np.asarray(v)
        input_shape = v.shape
        v_flat = v.flatten()
        Kv = self._apply_K_matvec(v_flat)
        result = Kv + self._dE * v_flat
        return result.reshape(input_shape)

    def _get_linear_operator(self):
        """SciPy ``LinearOperator`` for LOBPCG/eigsh: TDA uses ``A``, full Casida uses ``C``."""
        Ntrans = self._n_trans
        if self.tda:
            matvec = self._apply_A_matvec
        else:
            matvec = self._apply_C_matvec
        return LinearOperator(shape=(Ntrans, Ntrans), matvec=matvec, dtype=np.float64)

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

        if method.lower() == "lobpcg":
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
            raise ValueError(f"Unknown method: {method}. Use 'lobpcg' or 'eigsh'.")

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
            inv_sqrt_dE = 1.0 / self._sqrt_dE
            self.xpy = inv_sqrt_dE[:, None] * eigenvectors

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
        if self.polarized:
            raise NotImplementedError("Spin-polarized Casida not implemented.")
        if not hasattr(self, "_occ_e") or not hasattr(self, "_psi_occ"):
            raise RuntimeError("Call set_active_orbitals() first.")

        occ_e = self._occ_e
        unocc_e = self._unocc_e
        psi_occ = self._psi_occ
        psi_unocc = self._psi_unocc

        n_o = len(psi_occ)
        n_u = len(psi_unocc)
        Ntrans = n_o * n_u

        if self.rank == 0:
            print(f"Building Casida matrices: {n_o} occ x {n_u} unocc = {Ntrans} transitions")
            print(f"MPI ranks: {self.size} (CPU only)")

        dE = np.zeros(Ntrans, dtype=float)
        for i in range(n_o):
            for a in range(n_u):
                dE[i * n_u + a] = unocc_e[a] - occ_e[i]

        if np.any(dE <= 0.0):
            raise ValueError("Found non-positive excitation energies. Check orbital ordering.")

        phi_list = []
        proj_map = []
        for i in range(n_o):
            for a in range(n_u):
                p_i = self._proj_occ[:, i] if self.use_uspp else None
                p_a = self._proj_unocc[:, a] if self.use_uspp else None
                phi_ia = self.transition_orbital(psi_occ[i], psi_unocc[a], proj_i=p_i, proj_a=p_a)
                phi_list.append(phi_ia)
                proj_map.append((p_i, p_a))

        if self.use_eDFTpy:
            self.transition_densities = phi_list

        # Pre-flatten all transition densities for vectorized row construction (non-USPP).
        phi_flat = np.stack([np.asarray(f).ravel() for f in phi_list])  # (Ntrans, n_grid)
        fkxc_flat = np.asarray(self.fkxc_arr).ravel()                   # (n_grid,)
        dV = float(self.grid.dV)

        local_rows = [ia for ia in range(Ntrans) if ia % self.size == self.rank]
        n_local = len(local_rows)
        K_local = np.zeros((n_local, Ntrans), dtype=float)

        for local_idx, ia in enumerate(local_rows):
            if self.rank == 0 and local_idx % 10 == 0:
                print(f"  Rank 0: row {local_idx}/{n_local}")

            if self.triplet:
                vh_ia = None
            else:
                vh_ia = self._hartree_potential(phi_list[ia])

            if not self.use_uspp:
                # Vectorized: replace inner jb loop with two DGEMV operations.
                # XC row: K_xc[jb] = dV Σ_r phi_ia[r] fkxc[r] phi_jb[r]
                xc_row = (phi_flat[ia] * fkxc_flat) @ phi_flat.T * dV     # (Ntrans,)
                if vh_ia is not None:
                    hartree_row = phi_flat @ np.asarray(vh_ia).ravel() * dV  # (Ntrans,)
                    K_local[local_idx, :] = hartree_row + xc_row
                else:
                    K_local[local_idx, :] = xc_row
            else:
                for jb in range(Ntrans):
                    pj_i, pj_a = proj_map[jb]
                    K_local[local_idx, jb] = self._kernel_element_cpu(
                        phi_list[ia], phi_list[jb], vh_ia,
                        proj_i=proj_map[ia][0], proj_a=proj_map[ia][1],
                        proj_j=pj_i, proj_b=pj_a,
                    )

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
            s = self._AmB_evals
            U = self._AmB_evecs
            s_clip = np.where(s > 1e-14, s, np.inf)
            inv_sqrt_s = 1.0 / np.sqrt(s_clip)
            inv_sqrt_s = np.where(np.isfinite(inv_sqrt_s), inv_sqrt_s, 0.0)
            inv_sqrt_AmB = (U * inv_sqrt_s) @ U.T
            self.xpy = inv_sqrt_AmB @ Z

        self.omega = omega
        self.Z = Z
        return omega, Z

    def __call__(self, k=None):
        if self.rank != 0:
            raise RuntimeError("Call solve/oscillator_strengths from rank 0 only.")
        omega, Z = self.solve(k=k)
        f = self.oscillator_strengths(k=k)
        return omega, f, Z


def run_casida_in_memory(inputs: CasidaInputs, options: CasidaOptions, comm=None) -> CasidaResults:
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
