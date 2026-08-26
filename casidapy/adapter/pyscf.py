"""Build CasidaPy GTO kernels from a converged PySCF mean-field object.

This module owns SCF-engine glue (active-space windowing, GPU/MPI mean-field
promotion, spin-flip UKS bootstrap). :class:`~casidapy.kernels.gto.GTOKernel`
only stores MOs and evaluates the Casida coupling.
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np

from casidapy.casida_api import CasidaOptions
from casidapy.kernels.gto import GTOKernel


def _as_numpy(a) -> np.ndarray:
    """Host float array; pull CuPy / gpu4pyscf buffers with ``.get()`` / ``asnumpy``."""
    if a is None:
        return a
    if hasattr(a, "get") and not isinstance(a, np.ndarray):
        try:
            return np.asarray(a.get(), dtype=float)
        except Exception:
            pass
    try:
        import cupy as cp

        if isinstance(a, cp.ndarray):
            return np.asarray(cp.asnumpy(a), dtype=float)
    except Exception:
        pass
    return np.asarray(a, dtype=float)


def active_indices_from_mf(
    mf,
    *,
    n_occ: Optional[int] = None,
    n_unocc: Optional[int] = None,
    n_total_occ: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return ``(occ_indices, virt_indices, n_total_occ)`` for a closed-shell ``mf``.

    Windowing matches ``slice_active_space`` for energy-ordered MOs:

    - occupied = ``[n_total_occ - n_occ, n_total_occ)``
    - virtual  = ``[n_total_occ, n_total_occ + n_unocc)``
    """
    mo_e = _as_numpy(mf.mo_energy)
    mo_occ = _as_numpy(mf.mo_occ)
    n_occ_tot = int(np.sum(mo_occ > 1e-6))
    n_virt_tot = len(mo_e) - n_occ_tot
    if n_total_occ is None:
        n_total_occ = n_occ_tot
    if n_occ is None:
        n_occ = n_total_occ
    if n_unocc is None:
        n_unocc = n_virt_tot

    i1 = int(n_total_occ)
    i0 = max(0, i1 - int(n_occ))
    occ_indices = np.arange(i0, i1)
    u1 = min(len(mo_e), i1 + int(n_unocc))
    virt_indices = np.arange(i1, u1)
    if len(occ_indices) == 0 or len(virt_indices) == 0:
        raise ValueError(
            f"Empty active space: occ={occ_indices}, virt={virt_indices}, "
            f"n_orb={len(mo_e)}"
        )
    return occ_indices, virt_indices, int(n_total_occ)


def ensure_mpi_mf(mf):
    """Return an mpi4pyscf RKS with MO data for MPI ``get_jk`` response."""
    mod = type(mf).__module__
    if isinstance(mod, str) and mod.startswith("mpi4pyscf"):
        if getattr(mf, "with_df", None) is not None:
            warnings.warn(
                "mpi4pyscf mf has density fitting; promoting a direct "
                "MPI RKS so gen_response uses MPI get_jk."
            )
        else:
            return mf
    from casidapy.adapter.mpi_pyscf import promote_mf_to_mpi

    return promote_mf_to_mpi(mf)


def move_mf_to_gpu(
    mf,
    *,
    mol,
    mo_coeff,
    mo_energy,
    mo_occ,
    xc: str,
    use_df: bool = True,
    cp=None,
    verbose: bool = False,
):
    """Return a GPU mean-field (gpu4pyscf) or ``None`` if unavailable."""
    to_gpu = getattr(mf, "to_gpu", None)
    if callable(to_gpu):
        try:
            return to_gpu()
        except Exception as exc:
            if verbose:
                warnings.warn(
                    f"mf.to_gpu() failed ({exc}); trying gpu4pyscf.dft.rks"
                )

    try:
        from gpu4pyscf.dft import rks as gpu_rks
    except ImportError:
        return None

    if cp is None:
        try:
            import cupy as cp
        except ImportError:
            return None

    try:
        gmf = gpu_rks.RKS(mol)
        gmf.xc = getattr(mf, "xc", xc)
        gmf.mo_coeff = cp.asarray(mo_coeff)
        gmf.mo_energy = cp.asarray(mo_energy)
        gmf.mo_occ = cp.asarray(mo_occ)
        if use_df:
            gmf = gmf.density_fit()
        if getattr(gmf, "grids", None) is not None:
            gmf.grids.build()
        return gmf
    except Exception as exc:
        if verbose:
            warnings.warn(f"gpu4pyscf RKS setup failed ({exc})")
        return None


def scf_highspin_uks(
    mol,
    *,
    xc: str = "bhandhlyp",
    spin: Optional[int] = None,
    charge: Optional[int] = None,
    use_df: bool = True,
    conv_tol: float = 1e-9,
):
    """Converge a high-spin UKS reference for collinear SF-TDDFT (Route A)."""
    from pyscf import dft

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
    return mf


def sf_indices_from_mf(
    mf,
    *,
    n_occ: Optional[int] = None,
    n_unocc: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """α-occ / β-virt indices and host MO arrays for spin-flip.

    Returns ``(occ_a, vir_b, mo_c, mo_e, mo_o)``.
    """
    mo_c = _as_numpy(mf.mo_coeff)
    mo_e = _as_numpy(mf.mo_energy)
    mo_o = _as_numpy(mf.mo_occ)
    if mo_c.ndim != 3 or mo_c.shape[0] != 2:
        raise ValueError(
            "spin-flip requires an unrestricted (UKS/UHF) reference "
            "with (2, nao, nmo) MO coefficients."
        )
    occ_a = np.where(mo_o[0] > 1e-6)[0]
    vir_b = np.where(mo_o[1] < 1e-6)[0]
    if n_occ is not None:
        occ_a = occ_a[-int(n_occ) :]
    if n_unocc is not None:
        vir_b = vir_b[: int(n_unocc)]
    return occ_a, vir_b, mo_c, mo_e, mo_o


def build_spin_flip_kernel(
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
) -> GTOKernel:
    """Build a collinear SF-TDDFT :class:`GTOKernel` (α-occ → β-virt).

    Exchange-only Route A: needs a hybrid ``xc`` for nonzero coupling; TDA-only.
    Pass a converged unrestricted ``mf`` to skip the internal SCF.
    """
    if mf is None:
        mf = scf_highspin_uks(
            mol,
            xc=xc,
            spin=spin,
            charge=charge,
            use_df=use_df,
            conv_tol=conv_tol,
        )
    occ_a, vir_b, mo_c, mo_e, mo_o = sf_indices_from_mf(
        mf, n_occ=n_occ, n_unocc=n_unocc
    )
    return GTOKernel(
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


def extract_gto_kernel(
    mf,
    *,
    n_occ: Optional[int] = None,
    n_unocc: Optional[int] = None,
    n_total_occ: Optional[int] = None,
    n_states: int = 20,
    xc: Optional[str] = None,
    use_df: bool = True,
    tda: bool = True,
    verbose: bool = False,
    use_gpu: bool = False,
    use_mpi_response: bool = False,
    spin_state: str = "singlet",
    k_cache_max: int = 4096,
) -> Tuple[GTOKernel, CasidaOptions]:
    """Create a :class:`GTOKernel` and matching :class:`CasidaOptions` from ``mf``.

    Active-space selection matches ``slice_active_space`` semantics for
    energy-ordered MOs. Defaults use the full occupied / virtual spaces.

    ``spin_state`` is ``\"singlet\"`` (default) or ``\"triplet\"`` (closed-shell
    RKS/RHF reference with triplet-adapted response).

    ``use_gpu=True`` enables CuPy MO contractions / cached ``K @ v`` and, when
    gpu4pyscf is installed, a GPU ``gen_response``.

    ``use_mpi_response=True`` promotes ``mf`` to ``mpi4pyscf.dft.RKS`` so
    Davidson matvecs use MPI-parallel ``get_jk`` inside ``gen_response``.
    Forces ``use_df=False`` for the response.

    ``k_cache_max``: if ``n_trans <= k_cache_max``, ``setup()`` builds a dense
    ``K`` for DGEMM matvecs. Use ``0`` for large active spaces.
    """
    mo_e = _as_numpy(mf.mo_energy)
    mo_c = _as_numpy(mf.mo_coeff)
    mo_occ = _as_numpy(mf.mo_occ)

    occ_indices, virt_indices, n_total_occ = active_indices_from_mf(
        mf, n_occ=n_occ, n_unocc=n_unocc, n_total_occ=n_total_occ
    )

    if use_mpi_response and use_df:
        use_df = False

    xc_name = xc if xc is not None else getattr(mf, "xc", "pbe")
    kernel = GTOKernel(
        mf.mol,
        mo_c,
        mo_e,
        mo_occ,
        occ_indices=occ_indices,
        virt_indices=virt_indices,
        xc=xc_name,
        use_df=use_df,
        mf=mf,
        verbose=verbose,
        use_gpu=use_gpu,
        use_mpi_response=use_mpi_response,
        spin_state=spin_state,
        k_cache_max=int(k_cache_max),
    )
    opts = CasidaOptions(
        n_occ=kernel.n_occ,
        n_unocc=kernel.n_unocc,
        n_states=n_states,
        n_total_occ=n_total_occ,
        tda=tda,
        matrix_free=True,
        solver_method="davidson",
        use_gpu=use_gpu,
        xc=str(xc_name),
        basis="gto",
        use_uspp=False,
        use_eDFTpy=False,
        spin_state=str(spin_state).lower().strip(),
    )
    return kernel, opts


def extract_sf_gto_kernel(
    mol,
    *,
    xc: str = "bhandhlyp",
    spin: Optional[int] = None,
    charge: Optional[int] = None,
    n_occ: Optional[int] = None,
    n_unocc: Optional[int] = None,
    n_states: int = 20,
    use_df: bool = True,
    sf_xc: bool = False,
    mf=None,
    verbose: bool = False,
    use_gpu: bool = False,
) -> Tuple[GTOKernel, CasidaOptions]:
    """Build a collinear SF-TDDFT :class:`GTOKernel` (+ options) from ``mol``.

    Converges a high-spin unrestricted (UKS/UHF) reference — Mₛ = spin/2 — and
    sets up the α-occupied → β-virtual spin-flip manifold. Exchange-only (Route
    A): needs a hybrid ``xc`` and is TDA-only. Pass a converged unrestricted
    ``mf`` to skip the internal SCF.
    """
    kernel = build_spin_flip_kernel(
        mol,
        xc=xc,
        spin=spin,
        charge=charge,
        n_occ=n_occ,
        n_unocc=n_unocc,
        use_df=use_df,
        sf_xc=sf_xc,
        mf=mf,
        verbose=verbose,
        use_gpu=use_gpu,
    )
    opts = CasidaOptions(
        n_occ=kernel.n_occ,
        n_unocc=kernel.n_unocc,
        n_states=n_states,
        n_total_occ=kernel.n_occ,
        tda=True,  # SF-TDDFT is TDA-only in this backend
        matrix_free=True,
        solver_method="davidson",
        use_gpu=use_gpu,
        xc=str(xc),
        basis="gto",
        use_uspp=False,
        use_eDFTpy=False,
    )
    return kernel, opts
