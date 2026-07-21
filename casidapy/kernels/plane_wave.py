"""Plane-wave / real-space FFT-grid Casida kernel (QE / DFTpy path)."""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np
from dftpy.field import DirectField
from dftpy.functional import Functional

from casidapy.casida_utils import (
    _as_scalar_field_on_grid,
    _compute_fxc_triplet_pylibxc,
    augmentation_integral,
    build_energy_differences,
    filtered,
    proj_overlap,
    rho_aug,
)


class PlaneWaveKernel:
    """Hartree + local ``f_xc`` kernel on a uniform real-space grid.

    Transition densities are ``φ_ia(r) = ψ_i*(r) ψ_a(r)`` (+ USPP augmentation).
    Matrix-free ``apply_K`` uses BLAS DGEMMs and DFTpy FFT Hartree.
    """

    def __init__(
        self,
        rho_ks,
        xc_functional,
        *,
        polarized: bool = False,
        rho_cutoff: float = 1e-3,
        fxc_max: float = 20.0,
        spin_state: str = "singlet",
        libxc_xc_components=None,
        use_uspp: bool = False,
        beta_projectors=None,
        qij_augmentation=None,
        use_eDFTpy: bool = False,
        verbose: bool = False,
    ):
        self.rho = rho_ks
        self.grid = rho_ks.grid
        self._grid_shape = self.grid.nr
        self.polarized = bool(polarized)
        self.verbose = verbose

        self.use_uspp = bool(use_uspp) and beta_projectors is not None
        self.beta_projectors = beta_projectors
        self.qij_augmentation = qij_augmentation
        self.use_eDFTpy = bool(use_eDFTpy)

        self.hartree = Functional(type="HARTREE")
        self.functional = xc_functional

        spin_state = spin_state.lower()
        if spin_state not in ("singlet", "triplet"):
            raise ValueError(f"spin_state must be 'singlet' or 'triplet', got '{spin_state}'")
        self.triplet = spin_state == "triplet"

        if self.triplet:
            if libxc_xc_components is None:
                raise ValueError(
                    "libxc_xc_components must be specified for triplet calculations."
                )
            f_xc_T = _compute_fxc_triplet_pylibxc(
                rho_ks, list(libxc_xc_components), self.grid
            )
            fxc_arr = filtered(f_xc_T, fxc_max, rho_ks, rho_cutoff)
            self.fkxc_arr = DirectField(self.grid, rank=1, griddata_3d=fxc_arr)
        else:
            self.fkxc_raw = xc_functional(rho_ks, calcType=["V2"]).v2rho2
            fkxc = _as_scalar_field_on_grid(self.grid, self.fkxc_raw)
            fxc_arr = filtered(fkxc, fxc_max, rho_ks, rho_cutoff)
            self.fkxc_arr = DirectField(self.grid, rank=1, griddata_3d=fxc_arr)

        self.fkxc = self.fkxc_arr
        self._fkxc_ndarray = np.asarray(fxc_arr)

        self._occ_e = None
        self._unocc_e = None
        self.transition_densities = None
        self._psi_occ: List = []
        self._psi_unocc: List = []
        self._n_occ = 0
        self._n_unocc = 0
        self._n_trans = 0
        self._dE = None
        self._dV = float(self.grid.dV)
        self._psi_occ_arr = None
        self._psi_unocc_arr = None
        self._proj_occ = None
        self._proj_unocc = None
        self._ready = False
        self.tda = False

    @property
    def n_occ(self) -> int:
        return self._n_occ

    @property
    def n_unocc(self) -> int:
        return self._n_unocc

    @property
    def n_trans(self) -> int:
        return self._n_trans

    def set_active_orbitals(self, occ_eigs, unocc_eigs, psi_occ, psi_unocc):
        self._occ_e = np.asarray(occ_eigs, dtype=float).copy()
        self._unocc_e = np.asarray(unocc_eigs, dtype=float).copy()
        self._psi_occ = list(psi_occ)
        self._psi_unocc = list(psi_unocc)
        self._n_occ = len(self._psi_occ)
        self._n_unocc = len(self._psi_unocc)
        self._n_trans = self._n_occ * self._n_unocc
        self._dE = None
        self._ready = False

        self._proj_occ = None
        self._proj_unocc = None
        if self.use_uspp:
            self._proj_occ = proj_overlap(
                self.beta_projectors, self.grid, self._psi_occ
            )
            self._proj_unocc = proj_overlap(
                self.beta_projectors, self.grid, self._psi_unocc
            )

    def diagonal_dE(self) -> np.ndarray:
        if self._dE is None:
            if self._occ_e is None:
                raise RuntimeError("Call set_active_orbitals() first.")
            self._dE = build_energy_differences(self._occ_e, self._unocc_e)
        return self._dE

    def transition_orbital(self, psi_i, psi_a, proj_i=None, proj_a=None):
        phi = psi_i.conj() * psi_a
        if self.use_uspp and proj_i is not None and proj_a is not None:
            phi = phi + rho_aug(self.grid, self.qij_augmentation, proj_i, proj_a)
        return phi

    def _hartree_potential(self, rho):
        if np.iscomplexobj(np.asarray(rho)):
            rho = rho.real
        return self.hartree(rho, calcType=["V"]).potential

    def setup(self, tda: bool = False) -> None:
        if self.polarized:
            raise NotImplementedError("Spin-polarized Casida not implemented.")
        if not self._psi_occ:
            raise RuntimeError("Call set_active_orbitals() first.")

        self.tda = tda
        self._n_trans = self._n_occ * self._n_unocc
        self._dE = build_energy_differences(self._occ_e, self._unocc_e)
        if np.any(self._dE <= 0.0):
            raise ValueError(
                "Found non-positive excitation energies. Check orbital ordering."
            )

        self._dV = float(self.grid.dV)
        self._grid_shape = self.grid.nr

        if self.use_uspp and self._proj_occ is None:
            self._proj_occ = proj_overlap(
                self.beta_projectors, self.grid, self._psi_occ
            )
            self._proj_unocc = proj_overlap(
                self.beta_projectors, self.grid, self._psi_unocc
            )

        n_flat = int(np.prod(self._grid_shape))
        self._psi_occ_arr = np.empty((self._n_occ, n_flat), dtype=np.float64)
        self._psi_unocc_arr = np.empty((self._n_unocc, n_flat), dtype=np.float64)
        for i, p in enumerate(self._psi_occ):
            self._psi_occ_arr[i] = np.asarray(p).ravel()
        for a, p in enumerate(self._psi_unocc):
            self._psi_unocc_arr[a] = np.asarray(p).ravel()

        self._ready = True
        if self.verbose:
            n_grid = n_flat
            wfn_mem = (self._n_occ + self._n_unocc) * n_grid * 8 / 1e9
            print("PlaneWaveKernel matrix-free setup:")
            print(
                f"  Transitions: {self._n_occ} occ × {self._n_unocc} unocc "
                f"= {self._n_trans}"
            )
            print(f"  Grid: {self._grid_shape} = {n_grid:,} points")
            print(f"  Wavefunction memory: {wfn_mem:.2f} GB")
            print(f"  TDA: {tda}")

    def apply_K(self, v: np.ndarray) -> np.ndarray:
        if not self._ready:
            raise RuntimeError("Call setup() before apply_K().")

        n_occ = self._n_occ
        n_unocc = self._n_unocc
        v_arr = np.asarray(v, dtype=np.float64)

        V = v_arr.reshape(n_occ, n_unocc)
        w = V @ self._psi_unocc_arr
        rho_v_flat = (self._psi_occ_arr * w).sum(axis=0)
        rho_v = rho_v_flat.reshape(self._grid_shape)

        if self.use_uspp and self._proj_occ is not None:
            for j in range(n_occ):
                for b in range(n_unocc):
                    jb = j * n_unocc + b
                    if abs(v_arr[jb]) < 1e-30:
                        continue
                    p_j = self._proj_occ[:, j]
                    p_b = self._proj_unocc[:, b]
                    rho_v = rho_v + np.asarray(
                        rho_aug(self.grid, self.qij_augmentation, p_j, p_b)
                    ) * v_arr[jb]

        fkxc_arr = np.asarray(self._fkxc_ndarray)
        if self.triplet:
            response_flat = (fkxc_arr * rho_v).ravel()
        else:
            rho_v_field = DirectField(self.grid, rank=1, griddata_3d=rho_v)
            vh_v = self._hartree_potential(rho_v_field)
            response_flat = (np.asarray(vh_v) + fkxc_arr * rho_v).ravel()

        # Collective-free: each MPI rank owns a full matvec (see casida_engine notes).
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
                        self.qij_augmentation,
                        p_i,
                        p_a,
                        self._dV,
                    )
        return Kv

    def _dense_element(self, phi_ia, phi_jb, vh_ia,
                       proj_j=None, proj_b=None):
        """Dense K[ia,jb] element with optional USPP augmentation."""
        xc_part = (phi_ia.conj() * self.fkxc_arr * phi_jb).integral()
        if vh_ia is not None:
            k_val = (phi_jb.conj() * vh_ia).integral() + xc_part
        else:
            k_val = xc_part

        if self.use_uspp and proj_j is not None and proj_b is not None:
            response = np.asarray(self.fkxc_arr) * np.asarray(phi_ia)
            if vh_ia is not None:
                response = np.asarray(vh_ia) + response
            k_val = float(k_val) + augmentation_integral(
                response, self.qij_augmentation, proj_j, proj_b, self._dV
            )
        return float(k_val)

    def dense_K_rows(self, row_indices: Sequence[int], verbose: bool = False) -> np.ndarray:
        """Rows ``K[ia, :]`` of the dense coupling matrix for the active space.

        Builds all transition densities once, then evaluates the requested rows
        (Hartree + f_xc; triplet drops Hartree; USPP adds augmentation).  The
        caller (``CasidaKS_MPI.build_matrices``) handles MPI row distribution
        and assembly of A/B/C.  Transition densities are cached on
        ``self.transition_densities`` when ``use_eDFTpy`` is set.
        """
        if not self._psi_occ:
            raise RuntimeError("Call set_active_orbitals() first.")

        n_o = self._n_occ
        n_u = self._n_unocc
        n_trans = self._n_trans

        phi_list = []
        proj_map = []
        for i in range(n_o):
            for a in range(n_u):
                p_i = self._proj_occ[:, i] if self.use_uspp else None
                p_a = self._proj_unocc[:, a] if self.use_uspp else None
                phi_list.append(
                    self.transition_orbital(
                        self._psi_occ[i], self._psi_unocc[a], proj_i=p_i, proj_a=p_a
                    )
                )
                proj_map.append((p_i, p_a))

        if self.use_eDFTpy:
            self.transition_densities = phi_list

        # Pre-flatten all transition densities for vectorized row construction (non-USPP).
        phi_flat = np.stack([np.asarray(f).ravel() for f in phi_list])  # (n_trans, n_grid)
        fkxc_flat = np.asarray(self.fkxc_arr).ravel()                   # (n_grid,)
        dV = float(self.grid.dV)
        self._dV = dV

        n_rows = len(row_indices)
        K_rows = np.zeros((n_rows, n_trans), dtype=float)
        for local_idx, ia in enumerate(row_indices):
            if verbose and local_idx % 10 == 0:
                print(f"  Rank 0: row {local_idx}/{n_rows}")

            if self.triplet:
                vh_ia = None
            else:
                vh_ia = self._hartree_potential(phi_list[ia])

            if not self.use_uspp:
                # Vectorized: replace inner jb loop with two DGEMV operations.
                # XC row: K_xc[jb] = dV Σ_r phi_ia[r] fkxc[r] phi_jb[r]
                xc_row = (phi_flat[ia] * fkxc_flat) @ phi_flat.T * dV
                if vh_ia is not None:
                    K_rows[local_idx, :] = phi_flat @ np.asarray(vh_ia).ravel() * dV + xc_row
                else:
                    K_rows[local_idx, :] = xc_row
            else:
                for jb in range(n_trans):
                    pj_i, pj_a = proj_map[jb]
                    K_rows[local_idx, jb] = self._dense_element(
                        phi_list[ia], phi_list[jb], vh_ia,
                        proj_j=pj_i, proj_b=pj_a,
                    )
        return K_rows

    def dipole_matrix(self) -> np.ndarray:
        if not self._psi_occ:
            raise RuntimeError("Call set_active_orbitals() first.")

        n_o = self._n_occ
        n_u = self._n_unocc
        dV = float(self.grid.dV)
        psi_o = np.stack([np.asarray(p).ravel() for p in self._psi_occ])
        psi_u = np.stack([np.asarray(p).ravel() for p in self._psi_unocc])
        rx = np.asarray(self.grid.r[0]).ravel()
        ry = np.asarray(self.grid.r[1]).ravel()
        rz = np.asarray(self.grid.r[2]).ravel()

        mu_x = (psi_o * rx) @ psi_u.T * dV
        mu_y = (psi_o * ry) @ psi_u.T * dV
        mu_z = (psi_o * rz) @ psi_u.T * dV
        mu = np.stack([mu_x.ravel(), mu_y.ravel(), mu_z.ravel()], axis=-1)

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

    def collapse_transition_densities_to_state_basis(
        self, amp: np.ndarray
    ) -> Optional[np.ndarray]:
        """ρ_k(r) = Σ_ia amp_ia,k φ_ia(r). ``amp`` shape ``(n_trans, n_states)``."""
        if not self.use_eDFTpy or not self._psi_occ:
            return None

        n_unocc = self._n_unocc
        n_trans, n_states = amp.shape
        use_fast = self._psi_occ_arr is not None and not (
            self.use_uspp and self._proj_occ is not None
        )
        if use_fast:
            n_flat = self._psi_occ_arr.shape[1]
            amp_3d = amp.reshape(self._n_occ, n_unocc, n_states)
            phi_flat = np.zeros((n_states, n_flat), dtype=np.float64)
            for i in range(self._n_occ):
                c_i = amp_3d[i].T @ self._psi_unocc_arr
                phi_flat += c_i * self._psi_occ_arr[i]
            return phi_flat.reshape((n_states, *self._grid_shape))

        phi_states = None
        for i in range(self._n_occ):
            for a in range(n_unocc):
                ia = i * n_unocc + a
                row = amp[ia]
                if not np.any(row):
                    continue
                p_i = (
                    self._proj_occ[:, i]
                    if (self.use_uspp and self._proj_occ is not None)
                    else None
                )
                p_a = (
                    self._proj_unocc[:, a]
                    if (self.use_uspp and self._proj_unocc is not None)
                    else None
                )
                phi_ia = np.real(
                    np.asarray(
                        self.transition_orbital(
                            self._psi_occ[i],
                            self._psi_unocc[a],
                            proj_i=p_i,
                            proj_a=p_a,
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
        return np.stack(phi_states, axis=0)
