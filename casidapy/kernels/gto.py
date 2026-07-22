"""GTO / AO-MO Casida kernel (molecular TDDFT path).

RKS + adiabatic XC response, delegated to PySCF ``mf.gen_response`` so the
coupling matches PySCF TDDFT exactly: Coulomb, exact exchange for (range-
separated) hybrids, and the singlet-adapted f_xc are all included. Density
fitting is used when ``use_df`` is set (via ``mf.density_fit()``).

Hybrids are supported in TDA only: the matrix-free RPA chain assumes
``A - B = diag(dE)``, which does not hold once exact exchange enters ``B``.

With ``use_gpu=True`` (requires CuPy):

* Active MO blocks and a cached dense ``K`` are uploaded once; cached
  ``apply_K`` becomes a device DGEMM.
* On-the-fly matvecs do AO↔MO contractions on the GPU.
* If gpu4pyscf is available, ``mf`` is moved to the device so
  ``gen_response`` (J + f_xc + hybrid K) also runs on GPU; otherwise the
  response stays on CPU and only our contractions / cached matvecs use CuPy.
"""
from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence

import numpy as np

from casidapy.casida_utils import build_energy_differences


class GTOKernel:
    """Casida ``K`` in a GTO MO basis using PySCF integral infrastructure.

    Parameters
    ----------
    mol : pyscf.gto.Mole
    mo_coeff, mo_energy, mo_occ : ndarray
        Full MO set from a converged RKS/RHF calculation.
    n_occ, n_unocc : int
        Active-space sizes (after optional windowing).
    occ_indices, virt_indices : optional index arrays into the MO set.
        If omitted, the lowest ``n_occ`` occupied and first ``n_unocc``
        virtuals (by occupation) are used.
    xc : str
        XC functional name understood by PySCF (e.g. ``"pbe"``, ``"pbe0"``,
        ``"b3lyp"``). Hybrids and range-separated hybrids are supported
        (TDA only); plain RHF ground states give CIS.
    use_df : bool
        Use density fitting for the response (``mf.density_fit()``) when the
        mean-field object is not already density-fitted.
    k_cache_max : int
        When ``n_trans <= k_cache_max``, the full coupling matrix ``K`` is
        precomputed in ``setup()`` with a few batched ``gen_response`` calls
        (one AO-integral pass per chunk instead of one per solver iteration);
        ``apply_K`` then reduces to a DGEMM. Memory: ``n_trans**2 * 8`` bytes
        (4096 -> 134 MB). Set to 0 to always evaluate matvecs on the fly.
    use_gpu : bool
        Accelerate MO contractions and cached ``K @ v`` with CuPy. When
        gpu4pyscf is installed, also run ``gen_response`` on the GPU.
    """

    def __init__(
        self,
        mol,
        mo_coeff: np.ndarray,
        mo_energy: np.ndarray,
        mo_occ: np.ndarray,
        *,
        n_occ: Optional[int] = None,
        n_unocc: Optional[int] = None,
        occ_indices: Optional[np.ndarray] = None,
        virt_indices: Optional[np.ndarray] = None,
        xc: str = "pbe",
        use_df: bool = True,
        mf=None,
        verbose: bool = False,
        k_cache_max: int = 4096,
        use_gpu: bool = False,
    ):
        try:
            from pyscf import dft, scf  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "GTOKernel requires PySCF. Install with: pip install pyscf"
            ) from exc

        self.mol = mol
        self.mo_coeff = np.asarray(mo_coeff, dtype=float)
        self.mo_energy = np.asarray(mo_energy, dtype=float)
        self.mo_occ = np.asarray(mo_occ, dtype=float)
        self.xc = xc
        self.use_df = use_df
        self.verbose = verbose
        self._mf = mf

        occ_mask = self.mo_occ > 1e-6
        virt_mask = self.mo_occ < 1e-6
        if occ_indices is None:
            occ_all = np.where(occ_mask)[0]
            n_o = n_occ if n_occ is not None else len(occ_all)
            if n_o > len(occ_all):
                raise ValueError(f"n_occ={n_o} exceeds occupied count {len(occ_all)}")
            # top of valence: last n_o occupied
            occ_indices = occ_all[-n_o:]
        if virt_indices is None:
            virt_all = np.where(virt_mask)[0]
            n_u = n_unocc if n_unocc is not None else len(virt_all)
            if n_u > len(virt_all):
                raise ValueError(
                    f"n_unocc={n_u} exceeds virtual count {len(virt_all)}"
                )
            virt_indices = virt_all[:n_u]

        self._occ_idx = np.asarray(occ_indices, dtype=int)
        self._virt_idx = np.asarray(virt_indices, dtype=int)
        self._n_occ = len(self._occ_idx)
        self._n_unocc = len(self._virt_idx)
        self._n_trans = self._n_occ * self._n_unocc

        self._C_o = self.mo_coeff[:, self._occ_idx]
        self._C_v = self.mo_coeff[:, self._virt_idx]
        self._occ_e = self.mo_energy[self._occ_idx]
        self._unocc_e = self.mo_energy[self._virt_idx]
        self._dE = None
        self._ready = False
        self.tda = True  # GTO path defaults to TDA; RPA uses same K

        self._vresp = None
        self.hybrid = False
        self.k_cache_max = int(k_cache_max)
        self._K = None  # cached dense coupling matrix (n_trans, n_trans)

        self.use_gpu = bool(use_gpu)
        self._cp = None
        self._gpu_response = False
        self._C_o_dev = None
        self._C_v_dev = None
        self._K_dev = None
        if self.use_gpu:
            try:
                import cupy
            except ImportError as exc:
                raise ImportError(
                    "use_gpu=True requires CuPy (pip install cupy-cuda11x or "
                    "cupy-cuda12x matching your CUDA toolkit)."
                ) from exc
            self._cp = cupy

    @property
    def n_occ(self) -> int:
        return self._n_occ

    @property
    def n_unocc(self) -> int:
        return self._n_unocc

    @property
    def n_trans(self) -> int:
        return self._n_trans

    def diagonal_dE(self) -> np.ndarray:
        if self._dE is None:
            self._dE = build_energy_differences(self._occ_e, self._unocc_e)
        return self._dE

    def _as_numpy(self, arr) -> np.ndarray:
        """Host ndarray from numpy or cupy array."""
        if self._cp is not None and isinstance(arr, self._cp.ndarray):
            return self._cp.asnumpy(arr)
        return np.asarray(arr)

    def _move_mf_to_gpu(self, mf):
        """Return a GPU mean-field (gpu4pyscf) or ``None`` if unavailable."""
        # Prefer the recursive to_gpu() hook (PySCF >= 2.x + gpu4pyscf).
        to_gpu = getattr(mf, "to_gpu", None)
        if callable(to_gpu):
            try:
                return to_gpu()
            except Exception as exc:
                if self.verbose:
                    warnings.warn(
                        f"mf.to_gpu() failed ({exc}); trying gpu4pyscf.dft.rks"
                    )

        try:
            from gpu4pyscf.dft import rks as gpu_rks
        except ImportError:
            return None

        try:
            gmf = gpu_rks.RKS(self.mol)
            gmf.xc = getattr(mf, "xc", self.xc)
            # MO data: gpu4pyscf expects cupy arrays on the GPU object.
            cp = self._cp
            gmf.mo_coeff = cp.asarray(self.mo_coeff)
            gmf.mo_energy = cp.asarray(self.mo_energy)
            gmf.mo_occ = cp.asarray(self.mo_occ)
            if self.use_df:
                gmf = gmf.density_fit()
            if getattr(gmf, "grids", None) is not None:
                gmf.grids.build()
            return gmf
        except Exception as exc:
            if self.verbose:
                warnings.warn(f"gpu4pyscf RKS setup failed ({exc})")
            return None

    def setup(self, tda: bool = True) -> None:
        from pyscf import dft

        self.tda = tda
        self._dE = build_energy_differences(self._occ_e, self._unocc_e)
        if np.any(self._dE <= 0.0):
            raise ValueError(
                "Found non-positive excitation energies. Check orbital ordering."
            )

        # Build (or reuse) an RKS handle; its gen_response provides the full
        # singlet TDDFT coupling: J + singlet f_xc + exact exchange for hybrids.
        if self._mf is not None:
            mf = self._mf
        else:
            mf = dft.RKS(self.mol)
            mf.xc = self.xc
            mf.mo_coeff = self.mo_coeff
            mf.mo_energy = self.mo_energy
            mf.mo_occ = self.mo_occ

        is_ks = hasattr(mf, "xc")
        if is_ks and (getattr(mf, "grids", None) is None or mf.grids.coords is None):
            mf.grids.build()

        if self.use_df and getattr(mf, "with_df", None) is None:
            mf = mf.density_fit()

        if is_ks:
            ni = getattr(mf, "_numint", dft.numint.NumInt())
            self.hybrid = bool(ni.libxc.is_hybrid_xc(mf.xc))
        else:
            self.hybrid = True  # plain RHF: response includes full exact exchange
        if self.hybrid and not tda:
            raise NotImplementedError(
                "Hybrid XC in the GTO backend requires TDA: the matrix-free RPA "
                "chain assumes A - B = diag(dE), which exact exchange breaks. "
                "Set tda=True (CasidaOptions.tda)."
            )

        self._gpu_response = False
        if self.use_gpu:
            gmf = self._move_mf_to_gpu(mf)
            if gmf is not None and hasattr(gmf, "gen_response"):
                try:
                    self._vresp = gmf.gen_response(singlet=True, hermi=0)
                    self._mf = gmf
                    self._gpu_response = True
                except Exception as exc:
                    warnings.warn(
                        f"gpu4pyscf gen_response unavailable ({exc}); "
                        "using CPU response with GPU contractions / cached K."
                    )
                    self._vresp = mf.gen_response(singlet=True, hermi=0)
            else:
                warnings.warn(
                    "gpu4pyscf not available or mf.to_gpu() failed; "
                    "using CPU response with GPU contractions / cached K. "
                    "Install with: pip install gpu4pyscf"
                )
                self._vresp = mf.gen_response(singlet=True, hermi=0)
            self._C_o_dev = self._cp.asarray(self._C_o)
            self._C_v_dev = self._cp.asarray(self._C_v)
        else:
            self._vresp = mf.gen_response(singlet=True, hermi=0)

        self._ready = True
        if self.verbose:
            print("GTOKernel setup:")
            print(
                f"  Transitions: {self._n_occ} occ × {self._n_unocc} unocc "
                f"= {self._n_trans}"
            )
            print(
                f"  XC: {getattr(mf, 'xc', 'HF')}, hybrid: {self.hybrid}, "
                f"DF: {getattr(mf, 'with_df', None) is not None}, TDA: {tda}"
            )
            if self.use_gpu:
                print(
                    f"  GPU: CuPy contractions"
                    f"{' + gpu4pyscf response' if self._gpu_response else ' (CPU response)'}"
                )

        # Precompute dense K with batched response calls: a handful of
        # AO-integral passes up front instead of one per solver iteration.
        if self._K is None and 0 < self._n_trans <= self.k_cache_max:
            import time

            t0 = time.time()
            self._K = self.dense_K_rows(range(self._n_trans))
            if self.use_gpu:
                self._K_dev = self._cp.asarray(self._K)
            if self.verbose:
                mem = self._n_trans ** 2 * 8 / 1e6
                print(
                    f"  Cached dense K ({self._n_trans}x{self._n_trans}, "
                    f"{mem:.0f} MB) in {time.time() - t0:.1f}s; "
                    "matvecs are now DGEMMs"
                    + (" on GPU." if self.use_gpu else ".")
                )

    def _pair_dms(self, indices) -> np.ndarray:
        """Stack of AO transition DMs for the given ia indices.

        PySCF TDA convention (tdscf.rhf vind): dm[p,q] = 2 C_v[p,a] C_o[q,i];
        the factor 2 is the closed-shell double occupancy.
        """
        nao = self._C_o.shape[0]
        dms = np.empty((len(indices), nao, nao), dtype=float)
        for x, ia in enumerate(indices):
            i, a = divmod(int(ia), self._n_unocc)
            dms[x] = 2.0 * np.outer(self._C_v[:, a], self._C_o[:, i])
        return dms

    def apply_K_batch(self, indices) -> np.ndarray:
        """K columns for unit vectors e_ia, ia in ``indices`` — one response call."""
        dms = self._pair_dms(indices)
        if self._gpu_response:
            dms = self._cp.asarray(dms)
        v_ao = self._vresp(dms)
        v_ao = self._as_numpy(v_ao)
        v_ao = v_ao.reshape(len(indices), *v_ao.shape[-2:])
        # Prefer GPU MO back-contraction when CuPy is active.
        if self.use_gpu and self._C_o_dev is not None:
            cp = self._cp
            C_o = self._C_o_dev
            C_v = self._C_v_dev
            out = cp.empty((len(indices), self._n_trans), dtype=float)
            v_dev = cp.asarray(v_ao)
            for x in range(len(indices)):
                out[x] = (C_o.T @ v_dev[x].T @ C_v).ravel()
            return cp.asnumpy(out)

        out = np.empty((len(indices), self._n_trans), dtype=float)
        for x in range(len(indices)):
            out[x] = (self._C_o.T @ v_ao[x].T @ self._C_v).ravel()
        return out

    def _transition_dms_from_block(self, V: np.ndarray) -> np.ndarray:
        """Stack of AO transition DMs for columns of ``V`` ``(n_trans, k)``.

        ``dm[j] = 2 C_v @ V[:,j].reshape(n_occ, n_unocc).T @ C_o.T``
        (PySCF closed-shell TDA convention).
        """
        n_o, n_v = self._n_occ, self._n_unocc
        k = V.shape[1]
        W = V.reshape(n_o, n_v, k)  # W[i,a,j]
        # tmp[a,j,q] = Σ_i W[i,a,j] C_o[q,i]
        tmp = np.einsum("iaj,qi->ajq", W, self._C_o, optimize=True)
        # dm[j,p,q] = 2 Σ_a C_v[p,a] tmp[a,j,q]
        return 2.0 * np.einsum("pa,ajq->jpq", self._C_v, tmp, optimize=True)

    def apply_K_matmat(self, V: np.ndarray) -> np.ndarray:
        """Apply ``K`` to a block ``V`` of shape ``(n_trans, k)``.

        One batched ``gen_response`` call for the whole block (LOBPCG's natural
        width). With a cached dense ``K`` this is a single DGEMM.
        """
        if not self._ready:
            raise RuntimeError("Call setup() before apply_K_matmat().")

        V = np.asarray(V, dtype=float)
        if V.ndim == 1:
            return self.apply_K(V)
        if V.shape[0] != self._n_trans:
            raise ValueError(
                f"V has shape {V.shape}, expected ({self._n_trans}, k)"
            )
        k = V.shape[1]
        if k == 0:
            return np.zeros((self._n_trans, 0), dtype=float)
        if k == 1:
            return self.apply_K(V[:, 0]).reshape(self._n_trans, 1)

        if self._K is not None:
            if self.use_gpu and self._K_dev is not None:
                cp = self._cp
                return cp.asnumpy(self._K_dev @ cp.asarray(V))
            return self._K @ V

        dms = self._transition_dms_from_block(V)
        if self._gpu_response:
            dms = self._cp.asarray(dms)
        v_ao = self._vresp(dms)
        v_ao = self._as_numpy(v_ao).reshape(k, *v_ao.shape[-2:])

        out = np.empty((self._n_trans, k), dtype=float)
        if self.use_gpu and self._C_o_dev is not None:
            cp = self._cp
            v_dev = cp.asarray(v_ao)
            C_o, C_v = self._C_o_dev, self._C_v_dev
            for j in range(k):
                out[:, j] = cp.asnumpy((C_o.T @ v_dev[j].T @ C_v).ravel())
            return out

        for j in range(k):
            out[:, j] = (self._C_o.T @ v_ao[j].T @ self._C_v).ravel()
        return out

    def dense_K_rows(self, row_indices: Sequence[int], verbose: bool = False) -> np.ndarray:
        """Rows ``K[ia, :]`` of the dense coupling matrix.

        For real orbitals K is symmetric, so rows are computed as columns via
        batched ``gen_response`` calls (chunked to bound the DM-stack memory).
        This is also the hook for the engine's MPI dense path
        (``CasidaKS_MPI.build_matrices``), mirroring ``PlaneWaveKernel``.
        """
        if not self._ready:
            self.setup(tda=self.tda)

        row_indices = list(row_indices)
        nao = self._C_o.shape[0]
        # ~200 MB cap for the AO DM stack per response call
        chunk = max(1, int(2e8 / (nao * nao * 8)))
        K_rows = np.empty((len(row_indices), self._n_trans), dtype=float)
        for start in range(0, len(row_indices), chunk):
            block = row_indices[start:start + chunk]
            if verbose:
                print(f"  GTO dense K: rows {start}..{start + len(block)}/{len(row_indices)}")
            K_rows[start:start + len(block)] = self.apply_K_batch(block)
        return K_rows

    def _apply_K_gpu_cached(self, v: np.ndarray) -> np.ndarray:
        cp = self._cp
        return cp.asnumpy(self._K_dev @ cp.asarray(np.asarray(v, dtype=float).ravel()))

    def _apply_K_gpu_onthefly(self, v: np.ndarray) -> np.ndarray:
        """MO contractions on GPU; response on GPU if ``_gpu_response`` else CPU."""
        cp = self._cp
        V = cp.asarray(np.asarray(v, dtype=float).ravel()).reshape(
            self._n_occ, self._n_unocc
        )
        # dm1 = 2 C_v @ V.T @ C_o.T
        dm1 = 2.0 * (self._C_v_dev @ V.T @ self._C_o_dev.T)

        if self._gpu_response:
            v_ao = self._vresp(dm1)
            if not isinstance(v_ao, cp.ndarray):
                v_ao = cp.asarray(v_ao)
        else:
            v_ao = cp.asarray(self._vresp(cp.asnumpy(dm1)))

        Kv = self._C_o_dev.T @ v_ao.T @ self._C_v_dev
        return cp.asnumpy(Kv.ravel())

    def apply_K(self, v: np.ndarray) -> np.ndarray:
        if not self._ready:
            raise RuntimeError("Call setup() before apply_K().")

        v_arr = np.asarray(v, dtype=float)
        if v_arr.ndim == 2:
            return self.apply_K_matmat(v_arr)

        if self._K is not None:
            if self.use_gpu and self._K_dev is not None:
                return self._apply_K_gpu_cached(v_arr)
            return self._K @ v_arr.ravel()

        if self.use_gpu and self._C_o_dev is not None:
            return self._apply_K_gpu_onthefly(v_arr)

        V = v_arr.reshape(self._n_occ, self._n_unocc)
        # gen_response yields J + singlet f_xc (+ hybrid exact exchange), so K
        # reproduces the off-diagonal of the PySCF singlet A matrix exactly.
        dm1 = 2.0 * (self._C_v @ V.T @ self._C_o.T)
        v_ao = self._as_numpy(self._vresp(dm1))
        # (K v)_ov = Σ_pq v_ao[p,q] C_o[q,o] C_v[p,v]
        Kv = self._C_o.T @ v_ao.T @ self._C_v
        return Kv.ravel()

    def dipole_matrix(self) -> np.ndarray:
        # μ_AO: (3, nao, nao); then μ_ia = C_o† μ C_v
        with self.mol.with_common_orig((0.0, 0.0, 0.0)):
            dip_ao = self.mol.intor("int1e_r", comp=3)
        mu = np.empty((self._n_trans, 3), dtype=float)
        for alpha in range(3):
            mua = self._C_o.T @ dip_ao[alpha] @ self._C_v
            mu[:, alpha] = mua.ravel()
        return mu
