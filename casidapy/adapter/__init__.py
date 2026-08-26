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
    slice_active_space,
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
    "slice_active_space",
    "subsample_virtuals_by_energy",
]


def __getattr__(name):
    # Lazy: stddft imports casida_engine (avoid circular import at package load).
    if name == "STDDFTBridge":
        from casidapy.adapter.stddft import STDDFTBridge

        return STDDFTBridge
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
