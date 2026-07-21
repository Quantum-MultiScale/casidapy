"""GTO / AO-MO Casida kernel (molecular TDDFT path).

RKS + adiabatic XC response, delegated to PySCF ``mf.gen_response`` so the
coupling matches PySCF TDDFT exactly: Coulomb, exact exchange for (range-
separated) hybrids, and the singlet-adapted f_xc are all included. Density
fitting is used when ``use_df`` is set (via ``mf.density_fit()``).

Hybrids are supported in TDA only: the matrix-free RPA chain assumes
``A - B = diag(dE)``, which does not hold once exact exchange enters ``B``.
"""
from __future__ import annotations

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

        # Precompute dense K with batched response calls: a handful of
        # AO-integral passes up front instead of one per solver iteration.
        if self._K is None and 0 < self._n_trans <= self.k_cache_max:
            import time

            t0 = time.time()
            self._K = self.dense_K_rows(range(self._n_trans))
            if self.verbose:
                mem = self._n_trans ** 2 * 8 / 1e6
                print(
                    f"  Cached dense K ({self._n_trans}x{self._n_trans}, "
                    f"{mem:.0f} MB) in {time.time() - t0:.1f}s; "
                    "matvecs are now DGEMMs."
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
        v_ao = self._vresp(self._pair_dms(indices))
        v_ao = v_ao.reshape(len(indices), *v_ao.shape[-2:])
        out = np.empty((len(indices), self._n_trans), dtype=float)
        for x in range(len(indices)):
            out[x] = (self._C_o.T @ v_ao[x].T @ self._C_v).ravel()
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

    def apply_K(self, v: np.ndarray) -> np.ndarray:
        if not self._ready:
            raise RuntimeError("Call setup() before apply_K().")

        v_arr = np.asarray(v, dtype=float)
        if self._K is not None:
            return self._K @ v_arr.ravel()

        V = v_arr.reshape(self._n_occ, self._n_unocc)
        # gen_response yields J + singlet f_xc (+ hybrid exact exchange), so K
        # reproduces the off-diagonal of the PySCF singlet A matrix exactly.
        dm1 = 2.0 * (self._C_v @ V.T @ self._C_o.T)
        v_ao = self._vresp(dm1)
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
