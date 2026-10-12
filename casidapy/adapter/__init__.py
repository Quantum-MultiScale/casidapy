"""External-code adapters (QE / PySCF / STDDFT / mpi4pyscf).

Prefer submodule imports for heavy bridges::

    from casidapy.adapter.qepy import extract_pw_kernel
    from casidapy.adapter.pyscf import extract_gto_kernel
    from casidapy.adapter.stddft import STDDFTBridge
"""
from casidapy.adapter.qepy import (
    build_uspp_map_from_driver,
    extract_casida_inputs_from_qepy_driver,
    extract_pw_kernel,
    plan_pw_band_indices,
    preselect_virtual_bands_on_smooth_grid,
    qe_charge_grid,
    qe_smooth_grid,
    slice_active_space,
    slice_active_space_by_indices,
    subsample_virtual_indices_by_energy,
    subsample_virtuals_by_energy,
)
from casidapy.adapter.pyscf import (
    build_spin_flip_kernel,
    extract_gto_kernel,
    extract_sf_gto_kernel,
)

__all__ = [
    "STDDFTBridge",
    "build_spin_flip_kernel",
    "build_uspp_map_from_driver",
    "extract_casida_inputs_from_qepy_driver",
    "extract_gto_kernel",
    "extract_pw_kernel",
    "extract_sf_gto_kernel",
    "extract_gto_kernel_from_fchk",
    "load_mos_from_fchk",
    "read_fchk",
    "write_fchk_from_mf",
    "plan_pw_band_indices",
    "preselect_virtual_bands_on_smooth_grid",
    "qe_charge_grid",
    "qe_smooth_grid",
    "slice_active_space",
    "slice_active_space_by_indices",
    "subsample_virtual_indices_by_energy",
    "subsample_virtuals_by_energy",
]


def __getattr__(name):
    # Lazy: stddft imports casida_engine (avoid circular import at package load).
    if name == "STDDFTBridge":
        from casidapy.adapter.stddft import STDDFTBridge

        return STDDFTBridge
    if name in {
        "extract_gto_kernel_from_fchk",
        "load_mos_from_fchk",
        "read_fchk",
        "write_fchk_from_mf",
    }:
        from casidapy.adapter import qchem as _qchem

        return getattr(_qchem, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
