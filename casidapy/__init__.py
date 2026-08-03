"""CasidaPy: Linear-response TDDFT (Casida) solver for KS and OF-DFT."""

from casidapy.casida_api import CasidaInputs, CasidaOptions, CasidaResults
from casidapy.casida_engine import (
    run_casida_in_memory,
    run_casida,
    CasidaKS_MPI,
)
from casidapy.kernels import KernelBackend, PlaneWaveKernel, GTOKernel
from casidapy.qepy_adapter import slice_active_space, build_uspp_map_from_driver
from casidapy.pyscf_adapter import extract_gto_kernel, extract_sf_gto_kernel
from casidapy.qed import (
    QEDOptions,
    QEDResults,
    build_qed_tda_matrix,
    build_qed_sf_tda_matrix,
    solve_qed_tda,
    solve_qed_sf_tda,
    solve_qed_post,
    solve_qed_levels,
    solve_qed_pf_post,
    tddft_state_dipoles,
    scan_qed_tda,
    scan_qed_lambda,
    scan_qed_sf_lambda,
    dse_exchange_rows,
    dse_exchange_matvec,
    qed_electronic_A_rows,
    qed_tda_apply,
)
from casidapy.soc import (
    SOCResults,
    soc_ao_integrals,
    solve_soc_si,
    solve_soc_qed,
    solve_soc_qed_levels,
    solve_soc_qed_pf,
    build_soc_qed_pf_matrix,
)
from casidapy.stddft_bridge import STDDFTBridge
from casidapy.subsystem_coupling import (
    compute_nadd_kernel,
    compute_coupling_block,
    assemble_coupled_casida,
    coupled_oscillator_strengths,
    run_subsystem_casida,
)
from casidapy.uspp import (
    load_uspp_data,
    parse_upf,
    setup_uspp,
    setup_nc_pseudos,
    normalize_uspp_wavefunctions,
)

__version__ = "0.1.0"

__all__ = [
    "CasidaInputs",
    "CasidaOptions",
    "CasidaResults",
    "CasidaKS_MPI",
    "KernelBackend",
    "PlaneWaveKernel",
    "GTOKernel",
    "STDDFTBridge",
    "run_casida_in_memory",
    "run_casida",
    "slice_active_space",
    "build_uspp_map_from_driver",
    "extract_gto_kernel",
    "extract_sf_gto_kernel",
    "QEDOptions",
    "QEDResults",
    "build_qed_tda_matrix",
    "build_qed_sf_tda_matrix",
    "solve_qed_tda",
    "solve_qed_sf_tda",
    "solve_qed_post",
    "solve_qed_levels",
    "solve_qed_pf_post",
    "tddft_state_dipoles",
    "scan_qed_tda",
    "scan_qed_lambda",
    "scan_qed_sf_lambda",
    "dse_exchange_rows",
    "dse_exchange_matvec",
    "qed_electronic_A_rows",
    "qed_tda_apply",
    "SOCResults",
    "soc_ao_integrals",
    "solve_soc_si",
    "solve_soc_qed",
    "solve_soc_qed_levels",
    "solve_soc_qed_pf",
    "build_soc_qed_pf_matrix",
    "compute_nadd_kernel",
    "compute_coupling_block",
    "assemble_coupled_casida",
    "coupled_oscillator_strengths",
    "run_subsystem_casida",
    "load_uspp_data",
    "parse_upf",
    "setup_uspp",
    "setup_nc_pseudos",
    "normalize_uspp_wavefunctions",
]
