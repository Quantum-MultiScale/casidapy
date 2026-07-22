"""Build CasidaPy GTO kernels from a converged PySCF mean-field object."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from casidapy.casida_api import CasidaOptions
from casidapy.kernels.gto import GTOKernel


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
) -> Tuple[GTOKernel, CasidaOptions]:
    """Create a :class:`GTOKernel` and matching :class:`CasidaOptions` from ``mf``.

    Active-space selection matches ``slice_active_space`` semantics for
    energy-ordered MOs:

    - ``n_total_occ`` = LUMO index (number of occupied orbitals in the full set)
    - occupied window = ``[n_total_occ - n_occ, n_total_occ)``
    - virtual window  = ``[n_total_occ, n_total_occ + n_unocc)``

    Defaults use the full occupied / virtual spaces.

    ``use_gpu=True`` enables CuPy MO contractions / cached ``K @ v`` and, when
    gpu4pyscf is installed, a GPU ``gen_response``.
    """
    mo_e = np.asarray(mf.mo_energy, dtype=float)
    mo_c = np.asarray(mf.mo_coeff, dtype=float)
    mo_occ = np.asarray(mf.mo_occ, dtype=float)

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
    )
    opts = CasidaOptions(
        n_occ=kernel.n_occ,
        n_unocc=kernel.n_unocc,
        n_states=n_states,
        n_total_occ=n_total_occ,
        tda=tda,
        matrix_free=True,
        solver_method="eigsh",
        use_gpu=use_gpu,
        xc=str(xc_name),
        basis="gto",
        use_uspp=False,
        use_eDFTpy=False,
    )
    return kernel, opts
