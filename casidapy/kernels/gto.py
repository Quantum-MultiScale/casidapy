"""GTO / AO-MO Casida kernel (molecular TDDFT path).

RKS + adiabatic XC response, delegated to PySCF ``mf.gen_response`` so the
coupling matches PySCF TDDFT exactly: Coulomb, exact exchange for (range-
separated) hybrids, and the spin-adapted f_xc (singlet or triplet via
``spin_state``) are all included. Density fitting is used when ``use_df``
is set (via ``mf.density_fit()``).

Pure functionals support TDA and full TDDFT/RPA (matrix-free Casida ``C``
chain with ``A - B = diag(dE)``). Hybrids support TDA matrix-free and full
TDDFT via dense ``A,B`` from PySCF ``get_ab()`` (exact exchange makes
``A - B`` non-diagonal, so the cheap matrix-free RPA chain does not apply).

**Spin-flip TDDFT** (``spin_flip=True``, built via :meth:`GTOKernel.build_spin_flip`)
runs the collinear Mₛ = −1 manifold (α-occupied → β-virtual) on a high-spin
unrestricted (UKS/UHF) reference. In the spin-flip block the Hartree term and
the spin-diagonal ``f_xc`` vanish; only the exact-exchange kernel survives, so
the coupling is ``K = -c_x · get_k(dm_αβ)`` on the off-spin transition density.
This is exchange-only ("Route A") collinear SF-TDDFT: it requires a hybrid
(pure functionals give zero coupling) and is intrinsically TDA. The transverse
DFT kernel (non-collinear ``f_xc``, "Route B") is gated behind ``sf_xc=True``
and not yet implemented.

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
        ``"b3lyp"``).         Hybrids and range-separated hybrids are supported (TDA matrix-free
        or full TDDFT via dense ``build_AB``); plain RHF gives CIS / TDHF.
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
    use_mpi_response : bool
        Promote the mean-field to ``mpi4pyscf.dft.RKS`` so Davidson matvecs
        call MPI-parallel ``get_jk`` inside ``gen_response``. Density fitting
        is disabled (DF bypasses mpi4pyscf JK). Requires ``import mpi4pyscf``
        (or ``enable_mpi4pyscf()``) **before** other work so non-master ranks
        park in the worker pool. Incompatible with ``use_gpu``.
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
        use_mpi_response: bool = False,
        spin_flip: bool = False,
        sf_xc: bool = False,
        spin_state: str = "singlet",
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
        self.use_mpi_response = bool(use_mpi_response)
        if self.use_mpi_response and use_gpu:
            raise ValueError(
                "use_mpi_response and use_gpu cannot be combined; "
                "pick MPI JK (mpi4pyscf) or GPU response (gpu4pyscf)."
            )
        if self.use_mpi_response and use_df:
            warnings.warn(
                "use_mpi_response=True disables density fitting for the "
                "response (mpi4pyscf MPI get_jk is direct / outcore)."
            )
            use_df = False
        self.use_df = use_df
        self.verbose = verbose
        self._mf = mf
        self._spin_flip = bool(spin_flip)
        self._sf_xc = bool(sf_xc)
        spin_state = str(spin_state).lower().strip()
        if spin_state not in ("singlet", "triplet"):
            raise ValueError(
                f"spin_state must be 'singlet' or 'triplet', got {spin_state!r}"
            )
        if self._spin_flip and spin_state != "singlet":
            raise ValueError(
                "spin_flip mode is a separate Ms=−1 manifold; do not set "
                "spin_state='triplet' on an SF kernel."
            )
        self.spin_state = spin_state
        self.triplet = spin_state == "triplet"

        if self._spin_flip:
            # Unrestricted high-spin reference: MO arrays are (alpha, beta).
            # Manifold is alpha-occupied -> beta-virtual (Ms = -1).
            if occ_indices is None or virt_indices is None:
                raise ValueError(
                    "spin_flip mode requires explicit occ_indices (alpha-occ) "
                    "and virt_indices (beta-virt); use GTOKernel.build_spin_flip()."
                )
            if self.mo_coeff.ndim != 3 or self.mo_coeff.shape[0] != 2:
                raise ValueError(
                    "spin_flip mode expects mo_coeff of shape (2, nao, nmo)."
                )
            self._occ_idx = np.asarray(occ_indices, dtype=int)
            self._virt_idx = np.asarray(virt_indices, dtype=int)
            Ca, Cb = self.mo_coeff[0], self.mo_coeff[1]
            ea, eb = self.mo_energy[0], self.mo_energy[1]
            self._C_o = Ca[:, self._occ_idx]      # alpha occupied
            self._C_v = Cb[:, self._virt_idx]     # beta virtual
            self._occ_e = ea[self._occ_idx]
            self._unocc_e = eb[self._virt_idx]
        else:
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
            self._C_o = self.mo_coeff[:, self._occ_idx]
            self._C_v = self.mo_coeff[:, self._virt_idx]
            self._occ_e = self.mo_energy[self._occ_idx]
            self._unocc_e = self.mo_energy[self._virt_idx]

        self._n_occ = len(self._occ_idx)
        self._n_unocc = len(self._virt_idx)
        self._n_trans = self._n_occ * self._n_unocc
        # Closed-shell transition DMs carry a factor 2 (double occupancy); the
        # single-spin spin-flip transition DM does not.
        self._dm_factor = 1.0 if self._spin_flip else 2.0
        self._dE = None
        self._ready = False
        self.tda = True  # GTO path defaults to TDA; RPA uses same K

        self._vresp = None
        self._rsh = None   # (omega, alpha, hyb) exact-exchange split, SF only
        self._cx = None    # global-hybrid exchange fraction, SF only
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

    @classmethod
    def build_spin_flip(
        cls,
        mol,
        *,
        xc: str = "bhandhlyp",
        spin: Optional[int] = None,
        charge: Optional[int] = None,
        n_occ: Optional[int] = None,
        n_unocc: Optional[int] = None,
        use_df: bool = True,
        sf_xc: bool = False,
        mf=None,
        conv_tol: float = 1e-9,
        verbose: bool = False,
        use_gpu: bool = False,
    ) -> "GTOKernel":
        """Build a collinear SF-TDDFT kernel from a high-spin UKS/UHF reference.

        Converges (or accepts) a spin-unrestricted high-spin reference and sets
        up the Mₛ = −1 spin-flip manifold (α-occupied → β-virtual). Requires a
        hybrid ``xc`` for nonzero coupling (exchange-only, Route A). Pass an
        already-converged unrestricted ``mf`` to skip the internal SCF.

        Parameters
        ----------
        mol : pyscf.gto.Mole
            Its ``spin`` (= 2·Mₛ) and ``charge`` are used unless overridden.
        spin, charge : optional
            Override ``mol.spin`` / ``mol.charge`` for the reference (a fresh
            copy is built so ``mol`` is left untouched).
        n_occ, n_unocc : optional active-window sizes (top α-occ / first β-virt).
        sf_xc : bool
            Reserved for the non-collinear transverse XC kernel (Route B);
            ``True`` raises ``NotImplementedError`` in ``setup``.
        """
        from pyscf import dft

        if mf is None:
            m = mol
            if spin is not None or charge is not None:
                m = mol.copy()
                if spin is not None:
                    m.spin = int(spin)
                if charge is not None:
                    m.charge = int(charge)
                m.build(False, False)
            if m.spin == 0:
                raise ValueError(
                    "SF-TDDFT needs a high-spin (open-shell) reference; set "
                    "spin>=2 (spin = 2·Mₛ)."
                )
            mf = dft.UKS(m)
            mf.xc = xc
            if use_df:
                mf = mf.density_fit()
            mf.conv_tol = conv_tol
            mf.kernel()
            if not getattr(mf, "converged", True):
                warnings.warn("SF-TDDFT reference UKS SCF did not converge.")

        mo_c = np.asarray(mf.mo_coeff, dtype=float)
        mo_e = np.asarray(mf.mo_energy, dtype=float)
        mo_o = np.asarray(mf.mo_occ, dtype=float)
        if mo_c.ndim != 3 or mo_c.shape[0] != 2:
            raise ValueError(
                "build_spin_flip requires an unrestricted (UKS/UHF) reference "
                "with (2, nao, nmo) MO coefficients."
            )

        occ_a = np.where(mo_o[0] > 1e-6)[0]   # alpha occupied
        vir_b = np.where(mo_o[1] < 1e-6)[0]   # beta virtual
        if n_occ is not None:
            occ_a = occ_a[-int(n_occ):]
        if n_unocc is not None:
            vir_b = vir_b[:int(n_unocc)]

        return cls(
            mf.mol,
            mo_c,
            mo_e,
            mo_o,
            occ_indices=occ_a,
            virt_indices=vir_b,
            xc=getattr(mf, "xc", xc),
            use_df=use_df,
            mf=mf,
            verbose=verbose,
            use_gpu=use_gpu,
            spin_flip=True,
            sf_xc=sf_xc,
        )

    @property
    def n_occ(self) -> int:
        return self._n_occ

    @property
    def n_unocc(self) -> int:
        return self._n_unocc

    @property
    def n_trans(self) -> int:
        return self._n_trans

    @property
    def mf(self):
        """The underlying PySCF mean-field object (RKS/RHF or UKS/UHF)."""
        return self._mf

    @property
    def occ_indices(self) -> np.ndarray:
        """Active-space occupied MO indices into the full MO set."""
        return self._occ_idx

    @property
    def virt_indices(self) -> np.ndarray:
        """Active-space virtual MO indices into the full MO set."""
        return self._virt_idx

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

    def _ensure_mpi_mf(self, mf):
        """Return an mpi4pyscf RKS with MO data for MPI ``get_jk`` response."""
        mod = type(mf).__module__
        if isinstance(mod, str) and mod.startswith("mpi4pyscf"):
            # Drop DF wrapper if somehow present — it bypasses MPI JK.
            if getattr(mf, "with_df", None) is not None:
                warnings.warn(
                    "mpi4pyscf mf has density fitting; promoting a direct "
                    "MPI RKS so gen_response uses MPI get_jk."
                )
            else:
                return mf
        from casidapy.mpi_pyscf import promote_mf_to_mpi

        return promote_mf_to_mpi(mf)

    def setup(self, tda: bool = True) -> None:
        from pyscf import dft

        self.tda = tda
        self._dE = build_energy_differences(self._occ_e, self._unocc_e)
        if np.any(self._dE <= 0.0):
            raise ValueError(
                "Found non-positive excitation energies. Check orbital ordering."
            )

        if self._spin_flip:
            self._setup_spin_flip(tda)
            return

        # Build (or reuse) an RKS handle; its gen_response provides the full
        # spin-adapted TDDFT coupling (singlet or triplet f_xc + J + hybrid K).
        if self._mf is not None:
            mf = self._mf
        else:
            mf = dft.RKS(self.mol)
            mf.xc = self.xc
            mf.mo_coeff = self.mo_coeff
            mf.mo_energy = self.mo_energy
            mf.mo_occ = self.mo_occ

        if self.use_mpi_response:
            mf = self._ensure_mpi_mf(mf)

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

        singlet = not self.triplet

        # Hybrid full TDDFT uses dense A/B (build_AB); skip gen_response setup.
        if self.hybrid and not tda:
            self._mf = mf
            self._vresp = None
            self._K = None
            self._K_dev = None
            self._gpu_response = False
            self._ready = True
            if self.verbose:
                print("GTOKernel setup:")
                print(
                    f"  Transitions: {self._n_occ} occ × {self._n_unocc} unocc "
                    f"= {self._n_trans}"
                )
                print(
                    f"  XC: {getattr(mf, 'xc', 'HF')}, hybrid: True, "
                    f"spin: {self.spin_state}, "
                    f"DF: {getattr(mf, 'with_df', None) is not None}, "
                    f"TDA: False (dense A/B full TDDFT)"
                )
            return

        self._gpu_response = False
        if self.use_gpu:
            gmf = self._move_mf_to_gpu(mf)
            if gmf is not None and hasattr(gmf, "gen_response"):
                try:
                    self._vresp = gmf.gen_response(singlet=singlet, hermi=0)
                    self._mf = gmf
                    self._gpu_response = True
                except Exception as exc:
                    warnings.warn(
                        f"gpu4pyscf gen_response unavailable ({exc}); "
                        "using CPU response with GPU contractions / cached K."
                    )
                    self._vresp = mf.gen_response(singlet=singlet, hermi=0)
            else:
                warnings.warn(
                    "gpu4pyscf not available or mf.to_gpu() failed; "
                    "using CPU response with GPU contractions / cached K. "
                    "Install with: pip install gpu4pyscf"
                )
                self._vresp = mf.gen_response(singlet=singlet, hermi=0)
            self._C_o_dev = self._cp.asarray(self._C_o)
            self._C_v_dev = self._cp.asarray(self._C_v)
        else:
            self._vresp = mf.gen_response(singlet=singlet, hermi=0)
            self._mf = mf

        self._ready = True
        if self.verbose:
            mpi_tag = ""
            if self.use_mpi_response:
                try:
                    from mpi4pyscf.tools import mpi as _mpi
                    mpi_tag = f", MPI JK pool size={_mpi.pool.size}"
                except Exception:
                    mpi_tag = ", MPI JK"
            print("GTOKernel setup:")
            print(
                f"  Transitions: {self._n_occ} occ × {self._n_unocc} unocc "
                f"= {self._n_trans}"
            )
            print(
                f"  XC: {getattr(mf, 'xc', 'HF')}, hybrid: {self.hybrid}, "
                f"spin: {self.spin_state}, "
                f"DF: {getattr(mf, 'with_df', None) is not None}, TDA: {tda}"
                f"{mpi_tag}"
            )
            if self.use_gpu:
                print(
                    f"  GPU: CuPy contractions"
                    f"{' + gpu4pyscf response' if self._gpu_response else ' (CPU response)'}"
                )

        self._maybe_cache_dense_K()

    def _maybe_cache_dense_K(self) -> None:
        """Precompute dense K with batched response calls: a handful of
        AO-integral passes up front instead of one per solver iteration."""
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

    def _setup_spin_flip(self, tda: bool) -> None:
        """Collinear exchange-only (Route A) SF-TDDFT setup.

        Extracts the exact-exchange split from the hybrid and prepares the
        ``-c_x·get_k`` coupling. No ``gen_response`` is built: Hartree and the
        spin-diagonal f_xc do not contribute to the spin-flip block.
        """
        from pyscf import dft

        if not tda:
            raise NotImplementedError(
                "SF-TDDFT is TDA-only in this backend (the RPA A-B=diag(dE) "
                "factorization does not apply to the spin-flip block)."
            )
        if self._sf_xc:
            raise NotImplementedError(
                "sf_xc=True (non-collinear transverse XC kernel, Route B) is "
                "not implemented; use sf_xc=False for exchange-only SF-TDDFT."
            )
        mf = self._mf
        if mf is None:
            raise ValueError(
                "spin_flip mode requires an unrestricted reference mf "
                "(use GTOKernel.build_spin_flip())."
            )

        if hasattr(mf, "xc"):
            ni = getattr(mf, "_numint", dft.numint.NumInt())
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=self.mol.spin)
        else:
            omega, alpha, hyb = 0.0, 1.0, 1.0  # plain UHF: full exact exchange
        self._rsh = (float(omega), float(alpha), float(hyb))
        self._cx = float(hyb)
        self.hybrid = True
        if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
            raise ValueError(
                "Collinear exchange-only SF-TDDFT needs a hybrid functional: a "
                "pure functional gives zero spin-flip coupling. Use e.g. "
                "xc='bhandhlyp' (or sf_xc=True once the transverse kernel lands)."
            )

        self._vresp = None  # SF coupling is get_k-based, not gen_response
        if self.use_gpu:
            # Contractions/cached K on GPU; the get_k response stays on CPU.
            self._C_o_dev = self._cp.asarray(self._C_o)
            self._C_v_dev = self._cp.asarray(self._C_v)

        self._ready = True
        if self.verbose:
            print("GTOKernel setup (spin-flip, collinear exchange-only):")
            print(
                f"  SF transitions: {self._n_occ} α-occ × {self._n_unocc} β-virt "
                f"= {self._n_trans}"
            )
            print(
                f"  XC: {getattr(mf, 'xc', 'HF')}, exact-exchange split "
                f"(omega, alpha, hyb)={self._rsh}, TDA: {tda}"
            )
        self._maybe_cache_dense_K()

    def _sf_exchange(self, dm):
        """Exact-exchange response ``c_x·K[dm]`` on an off-spin transition DM.

        Handles global hybrids and range-separated hybrids using the PySCF
        ``(omega, alpha, hyb)`` convention: ``hyb·K + (alpha-hyb)·K(omega)``.
        Accepts a single ``(nao, nao)`` DM or a stack ``(k, nao, nao)``.
        """
        omega, alpha, hyb = self._rsh
        mf = self._mf
        k = hyb * mf.get_k(self.mol, dm, hermi=0)
        if abs(omega) > 1e-12 and abs(alpha - hyb) > 1e-12:
            k = k + (alpha - hyb) * mf.get_k(self.mol, dm, hermi=0, omega=omega)
        return k

    def _response(self, dm):
        """Dispatch the AO response: singlet gen_response (RKS) or, for the
        spin-flip block, ``-c_x·get_k`` (Hartree and spin-diagonal f_xc vanish)."""
        if self._spin_flip:
            return -self._sf_exchange(dm)
        return self._vresp(dm)

    def _pair_dms(self, indices) -> np.ndarray:
        """Stack of AO transition DMs for the given ia indices.

        PySCF TDA convention (tdscf.rhf vind): dm[p,q] = 2 C_v[p,a] C_o[q,i];
        the factor 2 is the closed-shell double occupancy. The spin-flip block
        uses a single-spin off-spin DM (``_dm_factor == 1``, C_v=β, C_o=α).
        """
        nao = self._C_o.shape[0]
        dms = np.empty((len(indices), nao, nao), dtype=float)
        for x, ia in enumerate(indices):
            i, a = divmod(int(ia), self._n_unocc)
            dms[x] = self._dm_factor * np.outer(self._C_v[:, a], self._C_o[:, i])
        return dms

    def apply_K_batch(self, indices) -> np.ndarray:
        """K columns for unit vectors e_ia, ia in ``indices`` — one response call."""
        dms = self._pair_dms(indices)
        if self._gpu_response:
            dms = self._cp.asarray(dms)
        v_ao = self._response(dms)
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
        # dm[j,p,q] = f Σ_a C_v[p,a] tmp[a,j,q]  (f = 2 closed-shell, 1 spin-flip)
        return self._dm_factor * np.einsum("pa,ajq->jpq", self._C_v, tmp, optimize=True)

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
        v_ao = self._response(dms)
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

        if self.use_gpu and self._C_o_dev is not None and not self._spin_flip:
            return self._apply_K_gpu_onthefly(v_arr)

        V = v_arr.reshape(self._n_occ, self._n_unocc)
        # RKS: gen_response yields J + singlet f_xc (+ hybrid exact exchange), so
        # K reproduces the off-diagonal of the PySCF singlet A matrix exactly.
        # SF: _response returns -c_x·get_k on the off-spin (β-virt × α-occ) DM.
        dm1 = self._dm_factor * (self._C_v @ V.T @ self._C_o.T)
        v_ao = self._as_numpy(self._response(dm1))
        # (K v)_ov = Σ_pq v_ao[p,q] C_o[q,o] C_v[p,v]
        Kv = self._C_o.T @ v_ao.T @ self._C_v
        return Kv.ravel()

    def build_AB(self) -> tuple:
        """Active-space Casida ``A``, ``B`` matrices (n_trans × n_trans).

        Uses PySCF ``TDDFT.get_ab()`` so hybrid exact exchange enters ``A`` and
        ``B`` correctly (``A - B`` is *not* ``diag(dE)``). Orbital-energy
        gaps are already included on the diagonal of ``A``.

        Intended for full TDDFT/RPA with hybrids (and as a dense reference for
        pure functionals). Spin-flip remains TDA-only.
        """
        if self._spin_flip:
            raise NotImplementedError(
                "SF-TDDFT is TDA-only; build_AB is for spin-conserving TDDFT."
            )
        if not self._ready:
            self.setup(tda=self.tda)
        if self._mf is None:
            raise RuntimeError("build_AB requires a mean-field object (mf).")

        from pyscf import tdscf

        td = tdscf.TDDFT(self._mf)
        # PySCF selects singlet vs triplet kernels via the ``singlet`` attribute
        # (``get_ab`` does not accept a ``singlet=`` keyword on current PySCF).
        td.singlet = not self.triplet
        A4, B4 = td.get_ab()
        A4 = np.asarray(A4, dtype=float)
        B4 = np.asarray(B4, dtype=float)
        if A4.ndim != 4 or B4.ndim != 4:
            raise RuntimeError(
                f"Unexpected get_ab shapes: A={A4.shape}, B={B4.shape} "
                "(expected nocc × nvir × nocc × nvir)."
            )

        mo_occ = np.asarray(self.mo_occ, dtype=float)
        if mo_occ.ndim != 1:
            raise ValueError(
                "build_AB expects a restricted (RKS/RHF) reference with 1-D mo_occ."
            )
        occ_all = np.where(mo_occ > 1e-6)[0]
        virt_all = np.where(mo_occ < 1e-6)[0]
        occ_map = {int(m): i for i, m in enumerate(occ_all)}
        virt_map = {int(m): a for a, m in enumerate(virt_all)}
        try:
            o_loc = np.array([occ_map[int(i)] for i in self._occ_idx], dtype=int)
            v_loc = np.array([virt_map[int(a)] for a in self._virt_idx], dtype=int)
        except KeyError as exc:
            raise ValueError(
                "Active MO index is not in the occupied/virtual set used by get_ab."
            ) from exc

        A_act = A4[np.ix_(o_loc, v_loc, o_loc, v_loc)]
        B_act = B4[np.ix_(o_loc, v_loc, o_loc, v_loc)]
        ntr = self._n_trans
        return A_act.reshape(ntr, ntr), B_act.reshape(ntr, ntr)

    def dipole_matrix(
        self, origin: Sequence[float] = (0.0, 0.0, 0.0)
    ) -> np.ndarray:
        if self._spin_flip:
            # Spin-flip transitions are dipole-forbidden through the (spin-
            # conserving) electric-dipole operator: ⟨α|β⟩ = 0. Oscillator
            # strengths require a spin-orbit treatment not modeled here.
            return np.zeros((self._n_trans, 3), dtype=float)
        if self.triplet:
            # ⟨S0|μ|T⟩ = 0 without SOC (spin forbidden).
            return np.zeros((self._n_trans, 3), dtype=float)
        # Singlet TDA/RPA: ⟨S₀|r|S⟩ = √2 Σ_ia X_ia ⟨i|r|a⟩ (spin-adapted).
        # Matches PySCF oscillator strengths and the √2 in QED-TDA couplings.
        orig = tuple(np.asarray(origin, dtype=float).ravel())
        with self.mol.with_common_orig(orig):
            dip_ao = self.mol.intor("int1e_r", comp=3)
        mu = np.empty((self._n_trans, 3), dtype=float)
        sqrt2 = np.sqrt(2.0)
        for alpha in range(3):
            mua = self._C_o.T @ dip_ao[alpha] @ self._C_v
            mu[:, alpha] = sqrt2 * mua.ravel()
        return mu
