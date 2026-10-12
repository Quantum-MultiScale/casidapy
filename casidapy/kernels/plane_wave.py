"""Plane-wave / real-space FFT-grid Casida kernel (QE / DFTpy path)."""
from __future__ import annotations

import warnings
from typing import Any, List, Optional, Sequence

import numpy as np
from dftpy.field import DirectField
from dftpy.functional import Functional

from casidapy.kernels.base import CasidaKernel
from casidapy.utils.casida_utils import (
    _as_scalar_field_on_grid,
    _compute_fxc_triplet_pylibxc,
    augmentation_integral,
    build_energy_differences,
    filtered,
    grid_slab_bounds,
    proj_overlap,
    rho_aug,
    total_density_for_alda,
)


def center_of_positive_charge(ions, upf_by_symbol=None) -> np.ndarray:
    """Center of positive charge, in Bohr, on the same frame as ``ions.positions``.

    The weight of each ion is the pseudopotential valence charge when a UPF
    map is available. That is the positive charge balanced by the KS density.
    A nonzero ``ions.charges`` entry is the next choice, then the atomic number.
    """
    positions = np.asarray(ions.positions, dtype=float)
    symbols = list(ions.get_chemical_symbols())
    weights = np.zeros(len(symbols), dtype=float)
    upf_by_symbol = dict(upf_by_symbol or {})
    zval_cache: dict = {}
    for i, symbol in enumerate(symbols):
        path = upf_by_symbol.get(symbol)
        if path:
            if symbol not in zval_cache:
                from casidapy.embed.potential import read_upf_z_valence

                zval_cache[symbol] = float(read_upf_z_valence(path))
            weights[i] = zval_cache[symbol]
            continue
        charge = 0.0
        try:
            charge = float(np.asarray(ions.charges)[i])
        except Exception:
            charge = 0.0
        if abs(charge) > 1e-8:
            weights[i] = charge
        else:
            weights[i] = float(np.asarray(ions.numbers)[i])
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("center of positive charge: ion charges sum to zero")
    return (positions * weights[:, None]).sum(axis=0) / total


def _broadcast_scalar_grid(comm, build):
    """Evaluate a full-grid scalar on rank 0 and share that array.

    The PBE kernel is identical on every rank, and building it FFTs the whole
    charge grid. Only one rank does that work.
    """
    import ctypes
    import gc

    rank = 0
    size = 1
    if comm is not None:
        rank = int(comm.Get_rank())
        size = int(comm.Get_size())
    if rank == 0:
        values = np.ascontiguousarray(build(), dtype=np.float64)
        shape = values.shape
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass
    else:
        values = None
        shape = None
    if size > 1:
        shape = comm.bcast(shape, root=0)
        if rank != 0:
            values = np.empty(shape, dtype=np.float64)
        comm.Bcast(values, root=0)
    return values


def _row_block(rows) -> np.ndarray:
    """``(n_rows, n)`` float64 array of 1-D orbital slabs, without a copy if possible.

    The QEpy adapter packs every band into one 2-D array and hands out its
    rows. When the rows are consecutive rows of one C-contiguous base, return
    a view of that base; stacking would add a second copy of every orbital.
    """
    arrs = [np.asarray(r) for r in rows]
    if not arrs:
        raise ValueError("no orbital rows")
    a0 = arrs[0]
    base = a0.base
    if (
        isinstance(base, np.ndarray)
        and base.ndim == 2
        and base.dtype == np.float64
        and base.flags["C_CONTIGUOUS"]
        and all(
            a.base is base and a.ndim == 1 and a.size == base.shape[1] for a in arrs
        )
    ):
        stride = base.strides[0]
        addr0 = base.__array_interface__["data"][0]
        off = a0.__array_interface__["data"][0] - addr0
        if off % stride == 0:
            r0 = off // stride
            if r0 + len(arrs) <= base.shape[0] and all(
                a.__array_interface__["data"][0] == addr0 + (r0 + k) * stride
                for k, a in enumerate(arrs)
            ):
                return base[r0 : r0 + len(arrs)]
    out = np.empty((len(arrs), a0.size), dtype=np.float64)
    for k, a in enumerate(arrs):
        out[k] = np.asarray(a, dtype=np.float64).ravel()
    return out


class PlaneWaveKernel(CasidaKernel):
    """Hartree + local ``f_xc`` (+ optional hybrid ``(ij|ab)``) on a PW grid.

    Transition densities are ``φ_ia(r) = ψ_i*(r) ψ_a(r)`` (+ USPP augmentation).
    Matrix-free ``apply_K`` uses BLAS DGEMMs and DFTpy FFT Hartree.

    Closed-shell **singlet** ``K`` is Hartree + ALDA ``f_xc``; **triplet** is
    ``f_xc`` only.  Hartree of a core–virtual density is repulsive (blueshifts
    ``ω``).  Near-nucleus ALDA ``f_xc`` is clipped (``fxc_max``) and is a
    known limitation for core excitations, not something this kernel fixes.

    Hybrid TDA ``A`` may add CIS ``−c_x (ij|ab)`` via :meth:`set_gto_ijab`
    (fragment GTO ERIs of projected virtuals; singlet and triplet) or
    :meth:`set_hybrid_exx` (FFT collocation, tests only).  Production default
    is off: hybrid ``xc`` is frozen-core Fock on the oneshot (``+U``-like).
    ``casida_exx=gto`` is the honest kernel.  Exciton Hartree ``(ia|jb)``
    is unscaled.

    With ``use_gpu=True`` (requires CuPy) the orbitals, f_xc, and the
    reciprocal-space Coulomb kernel are uploaded to the device once in
    ``setup()``; each ``apply_K`` then runs entirely on the GPU (DGEMMs,
    elementwise grid ops, and a cuFFT spectral Hartree solve) — only the
    small transition vector crosses the PCIe bus per matvec.
    """

    # Carries a real-space grid and participates in MPI COMM_WORLD collectives.
    provides_grid = True
    distributes_over_comm = True

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
        use_gpu: bool = False,
        comm=None,
        hartree_kernel: str = "full",
        hartree_screen_alpha: float = 0.1,
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

        self.use_gpu = bool(use_gpu)
        self._cp = None
        if self.use_gpu and self.use_uspp:
            # USPP augmentation is per-pair Python loops; not ported to GPU.
            warnings.warn(
                "use_gpu=True is not supported with USPP augmentation yet; "
                "falling back to the CPU path."
            )
            self.use_gpu = False
        if self.use_gpu:
            try:
                import cupy
            except ImportError as exc:
                raise ImportError(
                    "use_gpu=True requires CuPy (pip install cupy-cuda11x or "
                    "cupy-cuda12x matching your CUDA toolkit)."
                ) from exc
            self._cp = cupy

        mode = str(hartree_kernel or "full").strip().lower()
        if mode not in ("full", "mt_sr", "none"):
            raise ValueError(
                f"hartree_kernel must be 'full', 'mt_sr', or 'none', got {hartree_kernel!r}"
            )
        self.hartree_kernel = mode
        self.hartree_screen_alpha = float(hartree_screen_alpha)
        # DFTpy Hartree is only used for mode='full' (legacy / validation).
        # mt_sr / none go through the spectral coulG path exclusively.
        self.hartree = Functional(type="HARTREE") if mode == "full" else None
        self.functional = xc_functional
        if self.verbose and mode != "full":
            print(
                f"PlaneWaveKernel Hartree: mode={mode}"
                + (
                    f", screen_alpha={self.hartree_screen_alpha:.4f} a0^-1 (erfc SR)"
                    if mode == "mt_sr"
                    else ""
                ),
                flush=True,
            )

        spin_state = spin_state.lower()
        if spin_state not in ("singlet", "triplet"):
            raise ValueError(f"spin_state must be 'singlet' or 'triplet', got '{spin_state}'")
        self.triplet = spin_state == "triplet"

        # Singlet ALDA needs a scalar ρ / f_xc.  Collinear nspin=2 GS densities
        # are rank-2; evaluate (and threshold) on the total density.
        rho_alda = total_density_for_alda(rho_ks)

        if self.triplet and libxc_xc_components is None:
            raise ValueError(
                "libxc_xc_components must be specified for triplet calculations."
            )

        def _build_fxc():
            if self.triplet:
                f_xc_T = _compute_fxc_triplet_pylibxc(
                    rho_alda, list(libxc_xc_components), self.grid
                )
                return filtered(f_xc_T, fxc_max, rho_alda, rho_cutoff)
            raw = xc_functional(rho_alda, calcType=["V2"]).v2rho2
            fkxc = _as_scalar_field_on_grid(self.grid, raw)
            del raw
            return filtered(fkxc, fxc_max, rho_alda, rho_cutoff)

        fxc_arr = _broadcast_scalar_grid(comm, _build_fxc)
        self.fkxc_arr = DirectField(self.grid, rank=1, griddata_3d=fxc_arr)

        self.fkxc = self.fkxc_arr
        self._fkxc_ndarray = np.asarray(fxc_arr)

        self._occ_e = None
        self._unocc_e = None
        # Origin (Bohr, grid frame) for the length-gauge position operator.
        # ``run_casida_in_memory`` moves this to the center of positive charge.
        # Core-level XAS replaces it with the absorbing-atom position.
        self.dipole_origin = (0.0, 0.0, 0.0)
        self.transition_densities = None
        self._psi_occ: List = []
        self._psi_unocc: List = []
        # GIPAW: reconstructed unocc live in ``_psi_unocc`` for ``K``;
        # ``paw_gipaw`` μ samples ``_psi_unocc_tilde`` so Δφ is not applied twice.
        self._psi_unocc_tilde = None
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

        # MPI communicator for slab-decomposed apply_K
        self._comm = comm
        self._mpi_size = 1
        self._mpi_rank = 0
        if comm is not None:
            self._mpi_size = int(comm.Get_size())
            self._mpi_rank = int(comm.Get_rank())

        # Slab decomposition caches (populated in setup())
        self._slab_p0 = 0
        self._slab_p1 = 0
        self._psi_occ_slab = None
        self._psi_unocc_slab = None

        # Spectral Hartree kernel (4π/G²) and GPU-resident caches
        self._coulG = None
        self._psi_occ_dev = None
        self._psi_unocc_dev = None
        self._fkxc_dev = None
        self._coulG_dev = None

        # CIS −c_x (ij|ab) in TDA A (spatial MOs; occupancy is not in ρ_i).
        self.hybrid_coeff = 0.0
        self.rsh_omega = 0.0
        self.rsh_alpha = 0.0
        self._v_exx_ij = None
        self._v_exx_dev = None
        self._ijab_gto = None
        self._ijab_gto_dev = None

    def set_active_orbitals(self, occ_eigs, unocc_eigs, psi_occ, psi_unocc):
        self._occ_e = np.asarray(occ_eigs, dtype=float).copy()
        self._unocc_e = np.asarray(unocc_eigs, dtype=float).copy()
        self._psi_occ = list(psi_occ)
        self._psi_unocc = list(psi_unocc)
        self._psi_unocc_tilde = None
        self._n_occ = len(self._psi_occ)
        self._n_unocc = len(self._psi_unocc)
        self._n_trans = self._n_occ * self._n_unocc
        self._dE = None
        self._ready = False
        self._v_exx_ij = None
        self._v_exx_dev = None
        # Drop GTO (ij|ab) — orbital window changed. Caller must re-set.
        self._ijab_gto = None
        self._ijab_gto_dev = None

        self._proj_occ = None
        self._proj_unocc = None
        if self.use_uspp:
            self._proj_occ = self._proj_overlap_maybe_slab(self._psi_occ)
            self._proj_unocc = self._proj_overlap_maybe_slab(self._psi_unocc)

    def _orbitals_are_slabs(self, psi_list) -> bool:
        """True if orbitals are flat slabs (1-D, length < full grid)."""
        if not psi_list:
            return False
        sample = np.asarray(psi_list[0]).ravel()
        n_flat = int(np.prod(self.grid.nr))
        return sample.ndim == 1 and sample.size < n_flat

    def _proj_overlap_maybe_slab(self, psi_list):
        """⟨β|ψ⟩ with Allreduce when orbitals are grid slabs."""
        from casidapy.utils.uspp import SparseGridField

        if not self._orbitals_are_slabs(psi_list):
            return proj_overlap(self.beta_projectors, self.grid, psi_list)

        from mpi4py import MPI

        n_flat = int(np.prod(self.grid.nr))
        p0, p1 = grid_slab_bounds(n_flat, self._mpi_rank, self._mpi_size)
        dV = float(self.grid.dV)
        n_p = len(self.beta_projectors)
        n_s = len(psi_list)
        local = np.zeros((n_p, n_s), dtype=complex)
        for n in range(n_p):
            b = self.beta_projectors[n]
            if isinstance(b, SparseGridField):
                mask = (b.indices >= p0) & (b.indices < p1)
                if not np.any(mask):
                    continue
                b_idx = b.indices[mask] - p0
                b_val = b.values[mask]
                for k in range(n_s):
                    p = np.asarray(psi_list[k], dtype=np.float64).ravel()
                    local[n, k] = np.vdot(b_val, p[b_idx]) * dV
            else:
                b_arr = np.asarray(b).ravel(order="C")[p0:p1]
                for k in range(n_s):
                    p = np.asarray(psi_list[k]).ravel()
                    local[n, k] = np.vdot(b_arr, p) * dV
        if self._comm is not None and self._mpi_size > 1:
            out = np.empty_like(local)
            self._comm.Allreduce(local, out, op=MPI.SUM)
            return out
        return local

    def transition_orbital(self, psi_i, psi_a, proj_i=None, proj_a=None):
        phi = psi_i.conj() * psi_a
        if self.use_uspp and proj_i is not None and proj_a is not None:
            phi = phi + rho_aug(self.grid, self.qij_augmentation, proj_i, proj_a)
        return phi

    def _hartree_potential(self, rho):
        if self.hartree_kernel == "none":
            if isinstance(rho, DirectField):
                return DirectField(grid=rho.grid, rank=1)
            out = DirectField(grid=self.grid, rank=1)
            out[:] = 0.0
            return out
        if np.iscomplexobj(np.asarray(rho)):
            rho = rho.real
        # Always go through the spectral kernel so mt_sr / full share one path.
        # Wrap ndarray → DirectField when callers pass raw grids.
        if not isinstance(rho, DirectField):
            fld = DirectField(grid=self.grid, rank=1)
            fld[:] = np.asarray(rho, dtype=float).reshape(fld.shape)
            rho = fld
        vh = self._hartree_spectral(np.asarray(rho), xp=np)
        out = DirectField(grid=rho.grid, rank=1)
        out[:] = np.asarray(vh).reshape(out.shape)
        return out

    def _reciprocal_coulomb_kernel(self) -> np.ndarray:
        """Coulomb kernel on the rfftn half-grid (Hartree a.u.), G=0 zeroed.

        * ``full``  — periodic ``4π/G²``
        * ``mt_sr`` — Martyna–Tuckerman short-range ``4π/G² (1−exp(−G²/4α²))``
          (erfc(α r)/r), matching subsystem coupling's SR piece so fragment
          Casida + coupling LR reconstruct the MT split without double-counting
          full Hartree inside the fragment TDDFT matrix.
        * ``none``  — zero kernel

        G=0 is omitted on purpose: for a core–virtual transition density the
        Coulomb kernel is singular there, and keeping it would be the spurious
        interaction with periodic images of the core exciton.  This is not
        QE's Hartree; both the spectral path and DFTpy's Hartree functional
        drop G=0.  Do not restore it for XAS.

        Built from ``grid.lattice`` on the rfft half-grid. DFTpy's
        ``get_reciprocal().gg`` is the full G mesh (several GB on a charge
        grid) and is not the half-grid this transform uses.
        The kernel is defined so that ``irfftn(coulG * rfftn(rho))`` equals the
        periodic Hartree potential: the dV factor of the forward transform and
        the 1/(N dV) of the inverse cancel for the numpy/cupy FFT convention.
        """
        if self.hartree_kernel == "none":
            nr = tuple(int(n) for n in self._grid_shape)
            half_shape = (nr[0], nr[1], nr[2] // 2 + 1)
            return np.zeros(half_shape, dtype=float)

        nr = tuple(int(n) for n in self._grid_shape)
        lattice = np.asarray(self.grid.lattice, dtype=float)
        b = 2.0 * np.pi * np.linalg.inv(lattice).T  # rows = reciprocal vectors
        fx = np.fft.fftfreq(nr[0]) * nr[0]
        fy = np.fft.fftfreq(nr[1]) * nr[1]
        fz = np.fft.rfftfreq(nr[2]) * nr[2]
        mx, my, mz = np.meshgrid(fx, fy, fz, indexing="ij")
        gx = mx * b[0, 0] + my * b[1, 0] + mz * b[2, 0]
        gy = mx * b[0, 1] + my * b[1, 1] + mz * b[2, 1]
        gz = mx * b[0, 2] + my * b[1, 2] + mz * b[2, 2]
        gg = gx * gx + gy * gy + gz * gz

        coulG = np.zeros_like(gg)
        mask = gg > 1e-12
        coulG[mask] = 4.0 * np.pi / gg[mask]
        if self.hartree_kernel == "mt_sr":
            alpha = float(self.hartree_screen_alpha)
            if alpha <= 0.0:
                raise ValueError("hartree_screen_alpha must be > 0 for hartree_kernel='mt_sr'")
            # erfc(α r)/r ↔ 4π/G² (1 − exp(−G²/(4 α²))); same α as coupling_screen_alpha.
            coulG[mask] *= 1.0 - np.exp(-gg[mask] / (4.0 * alpha * alpha))
        return coulG

    def _ensure_coulG(self) -> np.ndarray:
        if self._coulG is None:
            self._coulG = self._reciprocal_coulomb_kernel()
        return self._coulG

    def _hartree_spectral(self, rho, xp=np, coulG=None):
        """Periodic Hartree potential via a direct 4π/G² spectral solve.

        Works with either numpy or cupy as ``xp``; ``rho`` must be a real 3-D
        array on the same module. Used by the GPU path (numpy version is kept
        for validation against DFTpy's Hartree functional).
        """
        if coulG is None:
            coulG = self._ensure_coulG()
        nr = tuple(int(n) for n in self._grid_shape)
        rho_G = xp.fft.rfftn(rho)
        return xp.fft.irfftn(coulG * rho_G, s=nr, axes=(0, 1, 2))

    def set_hybrid_exx(
        self,
        hybrid_coeff: float = 0.0,
        rsh_omega: float = 0.0,
        rsh_alpha: float = 0.0,
    ) -> None:
        """Enable CIS ``−c_x (ij|ab)`` in singlet TDA ``A``.

        ``hybrid_coeff`` is the full-range HF fraction; range-separated
        functionals add ``(rsh_alpha − hybrid_coeff)`` times the erf(ωr)/r
        kernel (PySCF ``get_ab``).  Does not scale the exciton Hartree
        ``(ia|jb)``.  Triplet kernels ignore the request.
        """
        self.hybrid_coeff = float(hybrid_coeff or 0.0)
        self.rsh_omega = float(rsh_omega or 0.0)
        self.rsh_alpha = float(rsh_alpha or 0.0)
        self._v_exx_ij = None
        self._v_exx_dev = None
        self._ijab_gto = None
        self._ijab_gto_dev = None
        self._ready = False
        self.hybrid = self.exx_ijab_active
        if self.triplet and (
            abs(self.hybrid_coeff) > 1e-14
            or abs(self.rsh_omega) > 1e-14
        ):
            warnings.warn(
                "FFT TDA (ij|ab) is singlet-only; "
                "ignoring set_hybrid_exx on a triplet kernel. "
                "Use set_gto_ijab for SI-SOC hybrids.",
                stacklevel=2,
            )

    def set_gto_ijab(self, ijab: np.ndarray) -> None:
        """CIS ``−(ij|ab)`` from fragment GTO ERIs of projected virtuals.

        ``ijab`` is already scaled by ``c_x`` (and RSH).  Shape
        ``(n_occ, n_occ, n_unocc, n_unocc)``.  Allowed on triplet kernels
        (same exchange, no Hartree).  Does not collocate on the FFT.
        """
        arr = np.asarray(ijab, dtype=np.float64)
        if arr.ndim != 4:
            raise ValueError(f"GTO (ij|ab) must be 4-D, got shape {arr.shape}")
        self._ijab_gto = arr
        self._ijab_gto_dev = None
        self._v_exx_ij = None
        self._v_exx_dev = None
        self._ready = False
        self.hybrid = True

    @property
    def exx_ijab_active(self) -> bool:
        if self._ijab_gto is not None:
            return True
        if self.triplet:
            return False
        hyb = float(self.hybrid_coeff)
        omega = float(self.rsh_omega)
        alpha = float(self.rsh_alpha)
        return abs(hyb) > 1e-14 or (
            abs(omega) > 1e-14 and abs(alpha - hyb) > 1e-14
        )

    def _coulomb_kernel_lr(self, omega: float) -> np.ndarray:
        """4π/G² exp(−G²/4ω²): erf(ωr)/r on the rfftn half-grid; G=0 zero."""
        coulG = self._ensure_coulG()
        if abs(float(omega)) < 1e-14:
            return coulG
        gg = np.zeros_like(coulG)
        mask = coulG > 0.0
        gg[mask] = 4.0 * np.pi / coulG[mask]
        out = np.zeros_like(coulG)
        out[mask] = coulG[mask] * np.exp(
            -gg[mask] / (4.0 * float(omega) * float(omega))
        )
        return out

    def _ensure_orbital_arrays(self) -> None:
        if not self._psi_occ:
            raise RuntimeError("Call set_active_orbitals() first.")
        n_flat = int(np.prod(self._grid_shape))
        if (
            self._psi_occ_arr is not None
            and self._psi_occ_arr.shape == (self._n_occ, n_flat)
            and self._psi_unocc_arr is not None
            and self._psi_unocc_arr.shape == (self._n_unocc, n_flat)
        ):
            return
        self._psi_occ_arr = np.empty((self._n_occ, n_flat), dtype=np.float64)
        self._psi_unocc_arr = np.empty((self._n_unocc, n_flat), dtype=np.float64)
        for i, p in enumerate(self._psi_occ):
            self._psi_occ_arr[i] = np.asarray(p).ravel()
        for a, p in enumerate(self._psi_unocc):
            self._psi_unocc_arr[a] = np.asarray(p).ravel()

    def _ensure_exx_potentials(self) -> None:
        """v_{ij}(r) = c_x v_H[φ_i φ_j] + (α−c_x) v_H^ω[φ_i φ_j]; G=0 dropped.

        ``φ_i φ_j`` is the orbital product (CIS).  G=0 drop matches the
        exciton Hartree convention and is the wrong Coulomb for a charged
        1s — this is the ``casida_exx=fft`` artifact baseline, not
        production.
        """
        if self._ijab_gto is not None:
            self._v_exx_ij = None
            return
        if not self.exx_ijab_active:
            self._v_exx_ij = None
            return
        self._ensure_orbital_arrays()
        n_occ = self._n_occ
        n_flat = int(np.prod(self._grid_shape))
        if self._v_exx_ij is not None and self._v_exx_ij.shape == (
            n_occ,
            n_occ,
            n_flat,
        ):
            return
        nr = tuple(int(n) for n in self._grid_shape)
        hyb = float(self.hybrid_coeff)
        omega = float(self.rsh_omega)
        alpha = float(self.rsh_alpha)
        coulG = self._ensure_coulG()
        coulG_lr = None
        if abs(omega) > 1e-14 and abs(alpha - hyb) > 1e-14:
            coulG_lr = self._coulomb_kernel_lr(omega)
        v_ij = np.empty((n_occ, n_occ, n_flat), dtype=np.float64)
        for i in range(n_occ):
            for j in range(n_occ):
                rho = (self._psi_occ_arr[i] * self._psi_occ_arr[j]).reshape(nr)
                v = hyb * self._hartree_spectral(rho, xp=np, coulG=coulG)
                if coulG_lr is not None:
                    v = v + (alpha - hyb) * self._hartree_spectral(
                        rho, xp=np, coulG=coulG_lr
                    )
                v_ij[i, j] = np.asarray(v, dtype=np.float64).ravel()
        self._v_exx_ij = v_ij

    def _exx_ijab_matvec(self, V, psi_unocc, *, xp=np, v_exx=None):
        """(K_x V)_{ia} = Σ_{jb} (ij|ab)_{c_x} V_{jb}; subtract this from Kv.

        ``V`` is ``(n_occ, n_unocc)``.  ``(ij|ab)_{c_x}`` is already folded
        into ``v_exx[i,j]`` as a local potential.
        """
        if v_exx is None:
            v_exx = self._v_exx_ij
        dV = self._dV
        w = V @ psi_unocc
        resp = xp.einsum("ijg,jg->ig", v_exx, w)
        return (resp @ psi_unocc.T) * dV

    def _add_exx_dense_row(self, K_row, ia: int, dV: float) -> None:
        n_u = self._n_unocc
        i = int(ia) // n_u
        a = int(ia) % n_u
        psi_u = self._psi_unocc_arr
        v_exx = self._v_exx_ij
        for j in range(self._n_occ):
            vec = psi_u[a] * v_exx[i, j]
            sl = slice(j * n_u, (j + 1) * n_u)
            K_row[sl] -= (psi_u @ vec) * dV

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
            self._proj_occ = self._proj_overlap_maybe_slab(self._psi_occ)
            self._proj_unocc = self._proj_overlap_maybe_slab(self._psi_unocc)

        n_flat = int(np.prod(self._grid_shape))
        P = self._mpi_size
        r = self._mpi_rank
        self._slab_p0, self._slab_p1 = grid_slab_bounds(n_flat, r, P)
        self._slab_counts = None
        self._qij_slab = None
        slab_len = self._slab_p1 - self._slab_p0
        pre_slabbed = self._orbitals_are_slabs(self._psi_occ)

        if pre_slabbed:
            # Adapter already scattered flat slabs — no full orbital replica.
            for p in self._psi_occ:
                if np.asarray(p).size != slab_len:
                    raise ValueError(
                        f"pre-scattered occ slab size {np.asarray(p).size} "
                        f"!= expected {slab_len} for rank {r}/{P}"
                    )
            self._psi_occ_slab = _row_block(self._psi_occ)
            self._psi_unocc_slab = _row_block(self._psi_unocc)
            if self.verbose:
                shared = all(
                    np.shares_memory(s, np.asarray(rows[0]))
                    for s, rows in (
                        (self._psi_occ_slab, self._psi_occ),
                        (self._psi_unocc_slab, self._psi_unocc),
                    )
                )
                print(
                    "  orbital slabs: "
                    + ("shared with adapter (no copy)" if shared else "copied"),
                    flush=True,
                )
            self._psi_occ_arr = None
            self._psi_unocc_arr = None
            # Drop field/list references so GC can reclaim any leftover full copies.
            self._psi_occ = []
            self._psi_unocc = []
        else:
            self._psi_occ_arr = np.empty((self._n_occ, n_flat), dtype=np.float64)
            self._psi_unocc_arr = np.empty((self._n_unocc, n_flat), dtype=np.float64)
            for i, p in enumerate(self._psi_occ):
                self._psi_occ_arr[i] = np.asarray(p).ravel()
            for a, p in enumerate(self._psi_unocc):
                self._psi_unocc_arr[a] = np.asarray(p).ravel()

            if P > 1 and not self.use_gpu:
                self._psi_occ_slab = np.ascontiguousarray(
                    self._psi_occ_arr[:, self._slab_p0 : self._slab_p1]
                )
                self._psi_unocc_slab = np.ascontiguousarray(
                    self._psi_unocc_arr[:, self._slab_p0 : self._slab_p1]
                )
                # Free full replicas — matvec uses slabs only.
                self._psi_occ_arr = None
                self._psi_unocc_arr = None
                self._psi_occ = []
                self._psi_unocc = []
            else:
                self._psi_occ_slab = None
                self._psi_unocc_slab = None

        self._ensure_exx_potentials()

        if self.use_gpu:
            if self._psi_occ_arr is None:
                raise RuntimeError(
                    "GPU matrix-free path needs full orbital arrays; "
                    "do not use pre-scattered slabs with use_gpu=True."
                )
            cp = self._cp
            self._psi_occ_dev = cp.asarray(self._psi_occ_arr)
            self._psi_unocc_dev = cp.asarray(self._psi_unocc_arr)
            self._fkxc_dev = cp.asarray(self._fkxc_ndarray)
            self._coulG_dev = cp.asarray(self._ensure_coulG())
            if self.exx_ijab_active and self._v_exx_ij is not None:
                self._v_exx_dev = cp.asarray(self._v_exx_ij)
            else:
                self._v_exx_dev = None
            if self._ijab_gto is not None:
                self._ijab_gto_dev = cp.asarray(self._ijab_gto)
            else:
                self._ijab_gto_dev = None

        self._ready = True
        if self.verbose:
            n_grid = n_flat
            if self._psi_occ_slab is not None:
                wfn_mem = (
                    (self._n_occ + self._n_unocc) * slab_len * 8 / 1e9
                )
                wfn_note = f"{wfn_mem:.2f} GB/rank (slab)"
            else:
                wfn_mem = (self._n_occ + self._n_unocc) * n_grid * 8 / 1e9
                wfn_note = f"{wfn_mem:.2f} GB"
            print("PlaneWaveKernel matrix-free setup:")
            print(
                f"  Transitions: {self._n_occ} occ × {self._n_unocc} unocc "
                f"= {self._n_trans}"
            )
            print(f"  Grid: {self._grid_shape} = {n_grid:,} points")
            print(f"  Wavefunction memory: {wfn_note}")
            print(f"  TDA: {tda}")
            if self.exx_ijab_active:
                if self._ijab_gto is not None:
                    print(
                        f"  EXX (ij|ab): GTO tensor shape="
                        f"{tuple(self._ijab_gto.shape)}  "
                        f"||ijab||_F={float(np.linalg.norm(self._ijab_gto)):.4e}"
                    )
                else:
                    print(
                        f"  EXX (ij|ab): c_x={self.hybrid_coeff:g} "
                        f"omega={self.rsh_omega:g} alpha={self.rsh_alpha:g}"
                    )
            if P > 1:
                print(
                    f"  MPI: {P} ranks, slab {slab_len:,} pts "
                    f"({slab_len / n_grid * 100:.0f}% of grid)"
                )
            if self.use_gpu:
                dev = self._cp.cuda.Device()
                print(
                    f"  GPU: enabled (device {dev.id}, "
                    f"orbitals + f_xc + Coulomb kernel resident on device)"
                )

    def _apply_K_gpu(self, v: np.ndarray) -> np.ndarray:
        """GPU matvec: DGEMMs + elementwise grid ops + cuFFT Hartree on device.

        Only ``v`` (n_trans doubles) is uploaded and ``Kv`` downloaded per call.
        """
        cp = self._cp
        v_dev = cp.asarray(np.asarray(v, dtype=np.float64).ravel())
        V = v_dev.reshape(self._n_occ, self._n_unocc)

        w = V @ self._psi_unocc_dev
        rho_v_flat = (self._psi_occ_dev * w).sum(axis=0)
        rho_v = rho_v_flat.reshape(tuple(int(n) for n in self._grid_shape))

        if self.triplet:
            response_flat = (self._fkxc_dev * rho_v).ravel()
        else:
            vh_v = self._hartree_spectral(rho_v, xp=cp, coulG=self._coulG_dev)
            response_flat = (vh_v + self._fkxc_dev * rho_v).ravel()

        Kv = (self._psi_occ_dev * response_flat) @ self._psi_unocc_dev.T * self._dV
        if self._ijab_gto_dev is not None:
            Kv = Kv - cp.einsum("ijab,jb->ia", self._ijab_gto_dev, V)
        elif self._v_exx_dev is not None:
            Kv = Kv - self._exx_ijab_matvec(
                V, self._psi_unocc_dev, xp=cp, v_exx=self._v_exx_dev
            )
        return cp.asnumpy(Kv.ravel())

    def _uspp_augment_rho_flat(self, rho_flat: np.ndarray, V: np.ndarray) -> None:
        """``rho_flat += Σ_{nm} Q_nm * C_nm`` with ``C = P_o^* V P_u^T`` (in-place).

        Avoids the O(n_trans) full-grid ``rho_aug`` loop that made matrix-free
        Casida appear hung on large USPP systems.
        """
        from casidapy.utils.uspp import SparseGridField

        P_o = np.asarray(self._proj_occ)
        P_u = np.asarray(self._proj_unocc)
        C = np.asarray(P_o.conj() @ V @ P_u.T, dtype=np.complex128).real
        qij = self.qij_augmentation
        n_p = C.shape[0]
        for n in range(n_p):
            for m in range(n_p):
                c = float(C[n, m])
                if abs(c) < 1e-30:
                    continue
                q = qij[n][m]
                if q is None:
                    continue
                if isinstance(q, SparseGridField):
                    rho_flat[q.indices] += c * q.values
                else:
                    rho_flat += c * np.asarray(q, dtype=np.float64).ravel(order="C")

    def _uspp_W(self, resp, p0: int = 0, p1=None) -> np.ndarray:
        """``W_nm = dV Σ Q_nm resp`` on the whole grid or on slab ``[p0, p1)``.

        A slab ``resp`` is indexed from 0. Only augmentation points inside
        the slab contribute, so ranks can sum partial ``W`` with an Allreduce.
        """
        from casidapy.utils.uspp import SparseGridField

        n_p = np.asarray(self._proj_occ).shape[0]
        dV = self._dV
        resp = np.asarray(resp, dtype=np.float64).ravel(order="C")
        W = np.zeros((n_p, n_p), dtype=np.float64)
        qij = self.qij_augmentation
        for n in range(n_p):
            for m in range(n_p):
                q = qij[n][m]
                if q is None:
                    continue
                if isinstance(q, SparseGridField):
                    idx = q.indices
                    if idx.size == 0:
                        continue
                    values = q.values
                    if p1 is not None:
                        mask = (idx >= p0) & (idx < p1)
                        if not np.any(mask):
                            continue
                        W[n, m] = float(np.dot(values[mask], resp[idx[mask] - p0])) * dV
                    else:
                        W[n, m] = float(np.dot(values, resp[idx])) * dV
                else:
                    q_arr = np.asarray(q, dtype=np.float64).ravel(order="C")
                    if p1 is None:
                        W[n, m] = float(np.dot(q_arr, resp)) * dV
                    else:
                        W[n, m] = float(np.dot(q_arr[p0:p1], resp)) * dV
        return W

    def _uspp_augment_Kv(self, Kv: np.ndarray, response_flat: np.ndarray = None, W: np.ndarray = None) -> np.ndarray:
        """``Kv_ia += Σ_{nm} P_o[n,i]^* W_{nm} P_u[m,a]`` with ``W_nm = ∫ Q_nm resp``."""
        if W is None:
            W = self._uspp_W(response_flat)
        P_o = np.asarray(self._proj_occ)
        P_u = np.asarray(self._proj_unocc)
        Kv_mat = Kv.reshape(self._n_occ, self._n_unocc)
        Kv_mat += np.real(P_o.conj().T @ W @ P_u)
        return Kv_mat.ravel()

    def _slab_layout(self):
        """Gatherv/Scatterv counts and displacements of the flat grid slabs."""
        cached = getattr(self, "_slab_counts", None)
        if cached is not None:
            return cached
        n_flat = int(np.prod(self._grid_shape))
        if n_flat >= 2**31:
            raise ValueError(
                f"grid has {n_flat:,} points; slab Gatherv counts are 32-bit"
            )
        size = self._mpi_size
        counts = np.empty(size, dtype=np.int32)
        displs = np.empty(size, dtype=np.int32)
        for r in range(size):
            r0, r1 = grid_slab_bounds(n_flat, r, size)
            counts[r] = r1 - r0
            displs[r] = r0
        self._slab_counts = (counts, displs)
        return self._slab_counts

    def _qij_slab_entries(self):
        """``(n, m, local_idx, values)`` for every ``Q_nm`` point in this slab."""
        cached = getattr(self, "_qij_slab", None)
        if cached is not None:
            return cached
        from casidapy.utils.uspp import SparseGridField

        p0, p1 = self._slab_p0, self._slab_p1
        entries = []
        qij = self.qij_augmentation
        n_p = len(qij)
        for n in range(n_p):
            for m in range(n_p):
                q = qij[n][m]
                if q is None:
                    continue
                if isinstance(q, SparseGridField):
                    idx = q.indices
                    mask = (idx >= p0) & (idx < p1)
                    if not np.any(mask):
                        continue
                    entries.append(
                        (
                            n,
                            m,
                            np.ascontiguousarray(idx[mask] - p0),
                            np.ascontiguousarray(q.values[mask]),
                        )
                    )
                else:
                    q_arr = np.asarray(q, dtype=np.float64).ravel(order="C")[p0:p1]
                    entries.append((n, m, None, q_arr))
        self._qij_slab = entries
        return entries

    def _slab_chunk(self, n_rows: int) -> int:
        """Grid points per chunk so a ``(n_rows, chunk)`` scratch stays ~256 MB."""
        return max(4096, int((256 * 2**20) // (8 * max(1, n_rows))))

    def _hartree_flat(self, rho_flat: np.ndarray) -> np.ndarray:
        """Full-grid Hartree potential of one flat density (FFT owner only)."""
        nr = tuple(int(n) for n in self._grid_shape)
        coulG = self._ensure_coulG()
        rho_G = np.fft.rfftn(rho_flat.reshape(nr))
        del rho_flat
        rho_G *= coulG
        v = np.fft.irfftn(rho_G, s=nr, axes=(0, 1, 2))
        del rho_G
        return np.ascontiguousarray(v, dtype=np.float64).ravel()

    def _apply_K_mpi_block(self, V_block: np.ndarray) -> np.ndarray:
        """Block matvec ``K V`` with orbitals, density, and response on slabs.

        ``V_block`` is ``(n_trans, B)``. Each rank builds the density slabs of
        all ``B`` columns and adds the ultrasoft charge that falls in its slab.
        Only the Hartree solve needs a whole grid: in each round of up to
        ``P`` columns, column ``b`` is gathered onto rank ``b`` for its FFT
        and the potential slabs are scattered back, so every rank transforms
        at most one grid at a time. ``f_xc ρ`` and both augmentation terms
        are local to the slab.
        """
        from mpi4py import MPI

        comm = self._comm
        rank = self._mpi_rank
        P = self._mpi_size
        n_o = self._n_occ
        n_u = self._n_unocc
        n_flat = int(np.prod(self._grid_shape))
        p0, p1 = self._slab_p0, self._slab_p1
        L = p1 - p0
        dV = self._dV
        psi_o = self._psi_occ_slab
        psi_u = self._psi_unocc_slab
        counts, displs = self._slab_layout()

        Vb = np.asarray(V_block, dtype=np.float64).reshape(n_o * n_u, -1)
        B = Vb.shape[1]
        Vt = np.ascontiguousarray(Vb.T).reshape(B, n_o, n_u)

        fxc_slab = np.asarray(self._fkxc_ndarray).ravel()[p0:p1]
        use_aug = bool(self.use_uspp and self._proj_occ is not None)
        if use_aug:
            P_o = np.asarray(self._proj_occ)
            P_u = np.asarray(self._proj_unocc)
            n_p = P_o.shape[0]
            q_entries = self._qij_slab_entries()
            W_loc = np.zeros((B, n_p, n_p), dtype=np.float64)

        Kv_loc = np.zeros((B, n_o, n_u), dtype=np.float64)

        for c0 in range(0, B, P):
            nb = min(P, B - c0)
            V_r = Vt[c0 : c0 + nb]

            # Transition-density slabs ρ_b = Σ_ia V_b[i,a] ψ_i ψ_a.
            rho = np.empty((nb, L), dtype=np.float64)
            V_rows = V_r.reshape(nb * n_o, n_u)
            ch = self._slab_chunk(nb * n_o)
            for s0 in range(0, L, ch):
                s1 = min(L, s0 + ch)
                w = (V_rows @ psi_u[:, s0:s1]).reshape(nb, n_o, s1 - s0)
                rho[:, s0:s1] = np.einsum("ig,big->bg", psi_o[:, s0:s1], w)
                del w

            if use_aug:
                for b in range(nb):
                    C = np.real(P_o.conj() @ V_r[b] @ P_u.T)
                    for n, m, idx, val in q_entries:
                        c = float(C[n, m])
                        if abs(c) < 1e-30:
                            continue
                        if idx is None:
                            rho[b] += c * val
                        else:
                            rho[b, idx] += c * val

            if self.triplet:
                resp = fxc_slab * rho
            else:
                resp = np.empty((nb, L), dtype=np.float64)
                full = None
                for b in range(nb):
                    if rank == b:
                        full = np.empty(n_flat, dtype=np.float64)
                        comm.Gatherv(
                            rho[b], (full, counts, displs, MPI.DOUBLE), root=b
                        )
                    else:
                        comm.Gatherv(rho[b], None, root=b)
                v_full = None
                if full is not None:
                    v_full = self._hartree_flat(full)
                    full = None
                for b in range(nb):
                    if rank == b:
                        comm.Scatterv(
                            (v_full, counts, displs, MPI.DOUBLE), resp[b], root=b
                        )
                    else:
                        comm.Scatterv(None, resp[b], root=b)
                v_full = None
                resp += fxc_slab * rho
            del rho

            if use_aug:
                for b in range(nb):
                    Wb = W_loc[c0 + b]
                    for n, m, idx, val in q_entries:
                        if idx is None:
                            Wb[n, m] += float(np.dot(val, resp[b]))
                        else:
                            Wb[n, m] += float(np.dot(val, resp[b, idx]))

            # Back-projection Σ_r ψ_i ψ_a resp_b on the slab.
            ch = self._slab_chunk(nb * n_o)
            for s0 in range(0, L, ch):
                s1 = min(L, s0 + ch)
                t = resp[:, None, s0:s1] * psi_o[None, :, s0:s1]
                Kv_loc[c0 : c0 + nb] += np.matmul(t, psi_u[:, s0:s1].T)
                del t

            if self._v_exx_ij is not None:
                for b in range(nb):
                    w_b = V_r[b] @ psi_u
                    for i in range(n_o):
                        exx_resp = np.zeros(L, dtype=np.float64)
                        for j in range(n_o):
                            exx_resp += self._v_exx_ij[i, j, p0:p1] * w_b[j]
                        Kv_loc[c0 + b, i] -= (psi_u * exx_resp).sum(axis=1)
            del resp

        Kv_loc *= dV
        Kv = np.empty_like(Kv_loc)
        comm.Allreduce(Kv_loc, Kv, op=MPI.SUM)

        if self._ijab_gto is not None:
            for b in range(B):
                Kv[b] -= np.einsum("ijab,jb->ia", self._ijab_gto, Vt[b])

        if use_aug:
            W_loc *= dV
            W = np.empty_like(W_loc)
            comm.Allreduce(W_loc, W, op=MPI.SUM)
            for b in range(B):
                Kv[b] += np.real(P_o.conj().T @ W[b] @ P_u)

        return np.ascontiguousarray(Kv.reshape(B, n_o * n_u).T)

    def _apply_K_mpi(self, v: np.ndarray) -> np.ndarray:
        """Single-vector MPI matvec (block of one column)."""
        col = np.asarray(v, dtype=np.float64).reshape(-1, 1)
        return self._apply_K_mpi_block(col)[:, 0]

    def apply_K_matmat(self, V: np.ndarray) -> np.ndarray:
        """``K V`` for a block ``(n_trans, B)``; Hartree FFTs spread over ranks."""
        if not self._ready:
            raise RuntimeError("Call setup() before apply_K().")
        V = np.asarray(V, dtype=np.float64)
        if V.ndim == 1:
            return self.apply_K(V)
        if (
            not self.use_gpu
            and self._comm is not None
            and self._mpi_size > 1
            and self._psi_occ_slab is not None
        ):
            return self._apply_K_mpi_block(V)
        return np.column_stack([self.apply_K(V[:, j]) for j in range(V.shape[1])])

    def apply_K(self, v: np.ndarray) -> np.ndarray:
        if not self._ready:
            raise RuntimeError("Call setup() before apply_K().")

        if self.use_gpu:
            return self._apply_K_gpu(v)

        if self._comm is not None and self._mpi_size > 1:
            return self._apply_K_mpi(v)

        n_occ = self._n_occ
        n_unocc = self._n_unocc
        v_arr = np.asarray(v, dtype=np.float64)

        V = v_arr.reshape(n_occ, n_unocc)
        w = V @ self._psi_unocc_arr
        rho_v_flat = (self._psi_occ_arr * w).sum(axis=0)

        if self.use_uspp and self._proj_occ is not None:
            self._uspp_augment_rho_flat(rho_v_flat, V)

        rho_v = rho_v_flat.reshape(self._grid_shape)

        fkxc_arr = np.asarray(self._fkxc_ndarray)
        if self.triplet:
            response_flat = (fkxc_arr * rho_v).ravel()
        else:
            rho_v_field = DirectField(self.grid, rank=1, griddata_3d=rho_v)
            vh_v = self._hartree_potential(rho_v_field)
            response_flat = (np.asarray(vh_v) + fkxc_arr * rho_v).ravel()

        Kv = (
            (self._psi_occ_arr * response_flat) @ self._psi_unocc_arr.T * self._dV
        )
        if self._ijab_gto is not None:
            Kv = Kv - np.einsum("ijab,jb->ia", self._ijab_gto, V)
        elif self._v_exx_ij is not None:
            Kv = Kv - self._exx_ijab_matvec(V, self._psi_unocc_arr)
        Kv = Kv.ravel()

        if self.use_uspp and self._proj_occ is not None:
            Kv = self._uspp_augment_Kv(Kv, response_flat)
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
        if self.exx_ijab_active:
            self._ensure_exx_potentials()

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
            if self._ijab_gto is not None:
                i = int(ia) // n_u
                a = int(ia) % n_u
                K_rows[local_idx] -= self._ijab_gto[i, :, a, :].ravel()
            elif self._v_exx_ij is not None:
                self._add_exx_dense_row(K_rows[local_idx], ia, dV)
        return K_rows

    def dense_K_rows_mpi(
        self,
        row_indices: Sequence[int],
        comm,
        verbose: bool = False,
    ) -> np.ndarray:
        """Like ``dense_K_rows`` but builds transition densities on rank 0 only.

        Rank 0 constructs ``phi_flat`` of shape ``(n_trans, n_grid)`` and
        broadcasts it once.  All ranks then compute their assigned rows using
        the shared array, avoiding redundant ``n_trans`` pointwise orbital
        products on every rank.
        """
        from mpi4py import MPI

        if not self._psi_occ:
            raise RuntimeError("Call set_active_orbitals() first.")

        n_o = self._n_occ
        n_u = self._n_unocc
        n_trans = self._n_trans
        n_grid = int(np.prod(self._grid_shape))
        rank = comm.Get_rank()

        # Rank 0 builds all transition densities; others allocate the buffer.
        phi_list = None
        phi_flat = np.empty((n_trans, n_grid), dtype=np.float64)
        proj_map = []

        if rank == 0:
            phi_list = []
            for i in range(n_o):
                for a in range(n_u):
                    p_i = self._proj_occ[:, i] if self.use_uspp else None
                    p_a = self._proj_unocc[:, a] if self.use_uspp else None
                    phi_list.append(
                        self.transition_orbital(
                            self._psi_occ[i], self._psi_unocc[a],
                            proj_i=p_i, proj_a=p_a,
                        )
                    )
                    proj_map.append((p_i, p_a))
            for idx, f in enumerate(phi_list):
                phi_flat[idx] = np.asarray(f).ravel()
        else:
            for i in range(n_o):
                for a in range(n_u):
                    p_i = self._proj_occ[:, i] if self.use_uspp else None
                    p_a = self._proj_unocc[:, a] if self.use_uspp else None
                    proj_map.append((p_i, p_a))

        comm.Bcast(phi_flat, root=0)

        if self.use_eDFTpy and rank == 0 and phi_list is not None:
            self.transition_densities = phi_list

        fkxc_flat = np.asarray(self.fkxc_arr).ravel()
        dV = float(self.grid.dV)
        self._dV = dV
        if self.exx_ijab_active:
            self._ensure_exx_potentials()

        n_rows = len(row_indices)
        K_rows = np.zeros((n_rows, n_trans), dtype=float)
        for local_idx, ia in enumerate(row_indices):
            if verbose and local_idx % 10 == 0:
                print(f"  Rank {rank}: row {local_idx}/{n_rows}")

            if self.triplet:
                vh_ia = None
            else:
                phi_ia_3d = phi_flat[ia].reshape(self._grid_shape)
                phi_ia_field = DirectField(
                    self.grid, rank=1, griddata_3d=phi_ia_3d
                )
                vh_ia = self._hartree_potential(phi_ia_field)

            if not self.use_uspp:
                xc_row = (phi_flat[ia] * fkxc_flat) @ phi_flat.T * dV
                if vh_ia is not None:
                    K_rows[local_idx, :] = (
                        phi_flat @ np.asarray(vh_ia).ravel() * dV + xc_row
                    )
                else:
                    K_rows[local_idx, :] = xc_row
            else:
                phi_ia_field = DirectField(
                    self.grid, rank=1,
                    griddata_3d=phi_flat[ia].reshape(self._grid_shape),
                )
                for jb in range(n_trans):
                    pj_i, pj_a = proj_map[jb]
                    phi_jb_field = DirectField(
                        self.grid, rank=1,
                        griddata_3d=phi_flat[jb].reshape(self._grid_shape),
                    )
                    K_rows[local_idx, jb] = self._dense_element(
                        phi_ia_field, phi_jb_field, vh_ia,
                        proj_j=pj_i, proj_b=pj_a,
                    )
            if self._ijab_gto is not None:
                i = int(ia) // n_u
                a = int(ia) % n_u
                K_rows[local_idx] -= self._ijab_gto[i, :, a, :].ravel()
            elif self._v_exx_ij is not None:
                self._add_exx_dense_row(K_rows[local_idx], ia, dV)
        return K_rows

    def position_operator(
        self, origin: Sequence[float] = (0.0, 0.0, 0.0)
    ) -> tuple:
        """Cartesian ``r − R₀`` on the FFT grid, each flat ``(n_grid,)``."""
        orig = np.asarray(origin, dtype=float).ravel()
        if orig.shape != (3,):
            raise ValueError(f"origin must have shape (3,), got {orig.shape}")
        rx = np.asarray(self.grid.r[0]).ravel() - orig[0]
        ry = np.asarray(self.grid.r[1]).ravel() - orig[1]
        rz = np.asarray(self.grid.r[2]).ravel() - orig[2]
        return rx, ry, rz

    def _flat_positions(self, flat_idx: np.ndarray, origin) -> np.ndarray:
        """``r − R₀`` at C-order flat grid indices, shape ``(3, n)``.

        Same points as DFTpy ``grid.r`` (``r_k = Σ_j s_j L[j,k]`` with
        ``s = index / nr``), without allocating the three full-grid arrays.
        """
        nx, ny, nz = (int(n) for n in self._grid_shape)
        idx = np.asarray(flat_idx, dtype=np.int64)
        s = np.empty((3, idx.size), dtype=np.float64)
        s[0] = (idx // (ny * nz)) / nx
        s[1] = ((idx // nz) % ny) / ny
        s[2] = (idx % nz) / nz
        r = np.asarray(self.grid.lattice, dtype=np.float64).T @ s
        r -= np.asarray(origin, dtype=np.float64).reshape(3, 1)
        return r

    def _qij_dipole_moments(self, origin, g0: int, g1: int) -> np.ndarray:
        """``M[α,n,m] = ∫ Q_nm (r_α − R₀) dV`` over flat range ``[g0, g1)``."""
        from casidapy.utils.uspp import SparseGridField

        qij = self.qij_augmentation
        n_p = len(qij)
        dV = float(self.grid.dV)
        M = np.zeros((3, n_p, n_p), dtype=np.float64)
        for n in range(n_p):
            for m in range(n_p):
                q = qij[n][m]
                if q is None:
                    continue
                if isinstance(q, SparseGridField):
                    idx = q.indices
                    mask = (idx >= g0) & (idx < g1)
                    if not np.any(mask):
                        continue
                    r = self._flat_positions(idx[mask], origin)
                    M[:, n, m] = (r @ q.values[mask]) * dV
                else:
                    q_arr = np.asarray(q, dtype=np.float64).ravel(order="C")
                    ch = self._slab_chunk(3)
                    for s0 in range(g0, g1, ch):
                        s1 = min(g1, s0 + ch)
                        r = self._flat_positions(np.arange(s0, s1), origin)
                        M[:, n, m] += (r @ q_arr[s0:s1]) * dV
        return M

    def mo_dipole_blocks(
        self,
        origin: Optional[Sequence[float]] = None,
        blocks: Sequence[str] = ("oo", "vv", "ov"),
    ) -> tuple:
        """One-body dipole MO blocks in the active space.

        Returns
        -------
        d_oo : (3, n_occ, n_occ)
        d_vv : (3, n_unocc, n_unocc)
        d_ov : (3, n_occ, n_unocc)
            ``d[α,p,q] = ∫ ψ_p (r_α − R₀) ψ_q dV`` (+ USPP when enabled).
            Blocks not listed in ``blocks`` are returned as ``None``.
        """
        if self._n_occ == 0:
            raise RuntimeError("Call set_active_orbitals() first.")
        if origin is None:
            origin = getattr(self, "dipole_origin", (0.0, 0.0, 0.0))
        want = {str(b).lower() for b in blocks}
        dV = float(self.grid.dV)
        n_o, n_u = self._n_occ, self._n_unocc
        n_flat = int(np.prod(self._grid_shape))
        orig = np.asarray(origin, dtype=float).ravel()
        if orig.shape != (3,):
            raise ValueError(f"origin must have shape (3,), got {orig.shape}")

        distributed = bool(
            self._psi_occ_slab is not None
            and self._comm is not None
            and self._mpi_size > 1
        )
        if self._psi_occ_slab is not None:
            psi_o = self._psi_occ_slab
            psi_u = self._psi_unocc_slab
            g0 = int(self._slab_p0) if distributed else 0
        elif self._psi_occ_arr is not None:
            psi_o = self._psi_occ_arr
            psi_u = self._psi_unocc_arr
            g0 = 0
        else:
            if not self._psi_occ:
                raise RuntimeError("Call set_active_orbitals() / setup() first.")
            psi_o = np.stack([np.asarray(p).ravel() for p in self._psi_occ])
            psi_u = np.stack([np.asarray(p).ravel() for p in self._psi_unocc])
            g0 = 0
        L = psi_o.shape[1]

        d_oo = np.zeros((3, n_o, n_o), dtype=float) if "oo" in want else None
        d_vv = np.zeros((3, n_u, n_u), dtype=float) if "vv" in want else None
        d_ov = np.zeros((3, n_o, n_u), dtype=float) if "ov" in want else None
        ch = self._slab_chunk(max(n_o, n_u))
        for s0 in range(0, L, ch):
            s1 = min(L, s0 + ch)
            r = self._flat_positions(np.arange(g0 + s0, g0 + s1), orig)
            po = psi_o[:, s0:s1]
            pu = psi_u[:, s0:s1]
            for alpha in range(3):
                if d_ov is not None or d_oo is not None:
                    por = po * r[alpha]
                    if d_ov is not None:
                        d_ov[alpha] += por @ pu.T
                    if d_oo is not None:
                        d_oo[alpha] += por @ po.T
                    del por
                if d_vv is not None:
                    d_vv[alpha] += (pu * r[alpha]) @ pu.T
            del r
        for block in (d_oo, d_vv, d_ov):
            if block is not None:
                block *= dV

        use_aug = bool(
            self.use_uspp
            and self.qij_augmentation is not None
            and self._proj_occ is not None
        )
        M = self._qij_dipole_moments(orig, g0, g0 + L) if use_aug else None

        if distributed:
            from mpi4py import MPI

            for block in (d_oo, d_vv, d_ov, M):
                if block is None:
                    continue
                tmp = np.empty_like(block)
                self._comm.Allreduce(block, tmp, op=MPI.SUM)
                block[:] = tmp

        if use_aug:
            P_o = np.asarray(self._proj_occ)
            P_u = np.asarray(self._proj_unocc)
            for alpha in range(3):
                if d_oo is not None:
                    a = np.real(P_o.conj().T @ M[alpha] @ P_o)
                    d_oo[alpha] += np.triu(a) + np.triu(a, 1).T
                if d_vv is not None:
                    a = np.real(P_u.conj().T @ M[alpha] @ P_u)
                    d_vv[alpha] += np.triu(a) + np.triu(a, 1).T
                if d_ov is not None:
                    d_ov[alpha] += np.real(P_o.conj().T @ M[alpha] @ P_u)
        return d_oo, d_vv, d_ov

    def permanent_dipole_el(
        self, origin: Sequence[float] = (0.0, 0.0, 0.0)
    ) -> np.ndarray:
        """Electronic permanent dipole ``∫ (r − R₀) ρ_KS(r) dV`` (a.u.).

        Uses the full KS density on the grid (not only the active MO window),
        matching the GTO ``Tr(D, r)`` coherent-state convention.
        """
        rxyz = self.position_operator(origin)
        dV = float(self.grid.dV)
        rho = np.asarray(self.rho).ravel()
        return np.array(
            [float(np.dot(r, rho) * dV) for r in rxyz], dtype=float
        )

    @property
    def nelectron(self) -> float:
        """Electron count from the KS density integral."""
        return float(np.asarray(self.rho).integral())

    def _transition_dipole_blocks(
        self, origin: Sequence[float] = (0.0, 0.0, 0.0)
    ) -> np.ndarray:
        """Raw ``⟨i|r|a⟩`` blocks ``(n_trans, 3)``; √2 / triplet handled by base.

        When an AE core was bound (``bind_ae_core_dipole``) the occupied–
        virtual block is the on-site radial (or velocity) integral, not the
        coarse FFT length-gauge sum.
        """
        if self._n_occ == 0:
            raise RuntimeError("Call set_active_orbitals() first.")
        ae = getattr(self, "_xas_ae_core", None)
        if ae is not None:
            from casidapy.xas.dipole import _normalize_dipole_method

            method = _normalize_dipole_method(ae.get("method", "gto"))
            gauge = str(ae.get("gauge", "length"))
            origin_ae = ae.get("origin", origin)
            if self._psi_occ_slab is not None and not self._psi_unocc:
                raise RuntimeError(
                    "AE-core dipoles need full virtual orbitals; "
                    "incompatible with pre-scattered slab-only storage."
                )
            if method in ("gto", "ae_gto", "fragment"):
                from casidapy.xas.dipole import core_virtual_dipole_matrix_gto_projected

                d_ov = core_virtual_dipole_matrix_gto_projected(
                    ae["core"],
                    self._psi_unocc,
                    self.grid,
                    origin=origin_ae,
                    gauge=gauge,
                )
            elif method in ("local_grid",):
                from casidapy.xas.dipole import (
                    LOCAL_N_ANG,
                    LOCAL_N_RAD,
                    core_virtual_dipole_matrix_local,
                )

                d_ov = core_virtual_dipole_matrix_local(
                    ae["core"],
                    self._psi_unocc,
                    self.grid,
                    origin=origin_ae,
                    gauge=gauge,
                    r_max=ae.get("r_max"),
                    n_rad=int(ae.get("n_rad") or LOCAL_N_RAD),
                    n_ang=int(ae.get("n_ang") or LOCAL_N_ANG),
                    rms_k=float(ae.get("rms_k", 10.0)),
                    r_min=float(ae.get("r_min", 0.25)),
                    mapping=str(ae.get("mapping") or "log"),
                    psi_mode=str(ae.get("psi_mode") or "spline3"),
                )
            elif method in ("pw_grid",):
                from casidapy.xas.dipole import core_virtual_dipole_matrix_pw_overlap

                d_ov = core_virtual_dipole_matrix_pw_overlap(
                    ae["core"],
                    self._psi_unocc,
                    self.grid,
                    origin=origin_ae,
                )
            elif method in ("paw_gipaw",):
                from casidapy.xas.dipole import LOCAL_N_ANG, LOCAL_N_RAD
                from casidapy.xas.gipaw import core_virtual_dipole_matrix_paw_gipaw

                gipaw = ae.get("gipaw") or ae.get("gipaw_upf")
                if gipaw is None:
                    raise ValueError(
                        "paw_gipaw dipoles need gipaw_upf on bind_ae_core_dipole"
                    )
                virts_mu = getattr(self, "_psi_unocc_tilde", None) or self._psi_unocc
                d_ov = core_virtual_dipole_matrix_paw_gipaw(
                    ae["core"],
                    virts_mu,
                    self.grid,
                    gipaw,
                    origin=origin_ae,
                    gauge=gauge,
                    r_max=ae.get("r_max"),
                    n_rad=int(ae.get("n_rad") or LOCAL_N_RAD),
                    n_ang=int(ae.get("n_ang") or LOCAL_N_ANG),
                    mapping=str(ae.get("mapping") or "log"),
                    psi_mode=str(ae.get("psi_mode") or "spline3"),
                    core_wfc=str(ae.get("core_wfc") or "ae"),
                    rcond=float(ae.get("gipaw_rcond", 1e-8)),
                )
            else:
                from casidapy.xas.dipole import core_virtual_dipole_matrix_radial

                d_ov = core_virtual_dipole_matrix_radial(
                    ae["core"],
                    self._psi_unocc,
                    self.grid,
                    origin=origin_ae,
                    gauge=gauge,
                    r_max=ae.get("r_max", 2.5),
                    n_rad=ae.get("n_rad", 64),
                    n_ang=ae.get("n_ang", 146),
                )
            if gauge in ("velocity", "grad", "momentum"):
                de = (
                    np.asarray(self._unocc_e, float)[None, :]
                    - np.asarray(self._occ_e, float)[:, None]
                )
                d_ov = d_ov / np.maximum(de[None, :, :], 1e-8)
            return np.stack(
                [d_ov[0].ravel(), d_ov[1].ravel(), d_ov[2].ravel()], axis=-1
            )
        _, _, d_ov = self.mo_dipole_blocks(origin=origin, blocks=("ov",))
        # d_ov: (3, n_o, n_u) → (n_trans, 3)
        return np.stack(
            [d_ov[0].ravel(), d_ov[1].ravel(), d_ov[2].ravel()], axis=-1
        )

    def collapse_transition_densities_to_state_basis(
        self, amp: np.ndarray
    ) -> Optional[np.ndarray]:
        """ρ_k(r) = Σ_ia amp_ia,k φ_ia(r). ``amp`` shape ``(n_trans, n_states)``.

        After MPI slab setup, ``_psi_occ`` is cleared; collapse must use
        ``_psi_*_arr`` or ``_psi_*_slab``.  USPP adds ``rho_aug`` on the full
        grid (projectors stay resident).
        """
        if not self.use_eDFTpy:
            return None
        if (
            self._psi_occ_arr is None
            and self._psi_occ_slab is None
            and not self._psi_occ
        ):
            return None

        n_unocc = self._n_unocc
        n_trans, n_states = amp.shape
        amp_3d = amp.reshape(self._n_occ, n_unocc, n_states)
        n_flat = int(np.prod(self._grid_shape))
        use_uspp_aug = bool(
            self.use_uspp
            and self.qij_augmentation is not None
            and self._proj_occ is not None
        )

        # --- Product φ_ia = ψ_i ψ_a on full arrays or slabs ---
        if self._psi_occ_arr is not None:
            phi_flat = np.zeros((n_states, n_flat), dtype=np.float64)
            for i in range(self._n_occ):
                c_i = amp_3d[i].T @ self._psi_unocc_arr
                phi_flat += c_i * self._psi_occ_arr[i]
        elif self._psi_occ_slab is not None:
            from mpi4py import MPI

            p0, p1 = int(self._slab_p0), int(self._slab_p1)
            slab_len = p1 - p0
            phi_slab = np.zeros((n_states, slab_len), dtype=np.float64)
            for i in range(self._n_occ):
                c_i = amp_3d[i].T @ self._psi_unocc_slab
                phi_slab += c_i * self._psi_occ_slab[i]
            phi_flat = np.empty((n_states, n_flat), dtype=np.float64)
            size = max(int(self._mpi_size), 1)
            if size <= 1 or self._comm is None:
                phi_flat[:, p0:p1] = phi_slab
            else:
                counts = np.empty(size, dtype=np.int32)
                displs = np.empty(size, dtype=np.int32)
                for r in range(size):
                    r0, r1 = grid_slab_bounds(n_flat, r, size)
                    counts[r] = r1 - r0
                    displs[r] = r0
                for k in range(n_states):
                    self._comm.Allgatherv(
                        np.ascontiguousarray(phi_slab[k], dtype=np.float64),
                        [phi_flat[k], counts, displs, MPI.DOUBLE],
                    )
        else:
            # Legacy list-of-fields path (serial / no slab free).
            phi_flat = np.zeros((n_states, n_flat), dtype=np.float64)
            for i in range(self._n_occ):
                psi_i = np.asarray(self._psi_occ[i], dtype=np.float64).ravel()
                for a in range(n_unocc):
                    ia = i * n_unocc + a
                    row = amp[ia]
                    if not np.any(row):
                        continue
                    psi_a = np.asarray(self._psi_unocc[a], dtype=np.float64).ravel()
                    phi_ia = psi_i * psi_a
                    for state_k in range(n_states):
                        c = row[state_k]
                        if c != 0.0:
                            phi_flat[state_k] += c * phi_ia

        # --- USPP augmentation on full grid (projectors are global) ---
        if use_uspp_aug:
            for i in range(self._n_occ):
                for a in range(n_unocc):
                    ia = i * n_unocc + a
                    row = amp[ia]
                    if not np.any(row):
                        continue
                    rho_a = np.asarray(
                        rho_aug(
                            self.grid,
                            self.qij_augmentation,
                            self._proj_occ[:, i],
                            self._proj_unocc[:, a],
                        )
                    ).ravel()
                    for state_k in range(n_states):
                        c = float(row[state_k])
                        if c != 0.0:
                            phi_flat[state_k] += c * rho_a

        return phi_flat.reshape((n_states, *self._grid_shape))
