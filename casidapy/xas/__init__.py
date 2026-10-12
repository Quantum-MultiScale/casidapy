"""X-ray absorption / CVS — production import surface.

Two workflows
-------------
1. **All-GTO molecule** (you own the SCF)::

       from casidapy.xas import run_xas_gto
       res, core, kernel = run_xas_gto(mf, edge="K", edge_atom_indices=[0])

       # L-edge SOC-coupled TDDFT (spatial 2p + singlet/triplet SI-SOC)
       res, core, kernel = run_xas_gto(
           mf, edge="L", edge_atom_indices=[0], soc="si",
       )

2. **QE host + frozen-core AE 1s + PW Casida (GIPAW μ/K)**::

       from casidapy.xas import run_ae_host_pw_grid_cvs
       res, core, kernel, mf, diag = run_ae_host_pw_grid_cvs(
           driver, edge_atom=0, edge="K", gipaw_upf=path,
           calc_type="crystal",  # or "molecule"
       )

3. **Fragment core → inject yourself**::

       from casidapy.xas import select_xas_fragment, extract_fragment_core, inject_core_orbitals
"""
from casidapy.embed import build_ae_embedding_potential, read_upf_z_valence
from casidapy.xas.cvs import (
    CVSSOCResults,
    CoreOrbitals,
    FragmentSpec,
    apply_orbital_soc,
    build_pw_kernel_from_qepy,
    core_from_mf,
    core_mos_to_pw_fields,
    extract_fragment_core,
    formal_oxidation_charge,
    inject_core_orbitals,
    parse_soc_mode,
    resolve_oxidation_state,
    run_cvs_gto_from_mf,
    run_cvs_si_soc,
    run_cvs_tda,
    clone_pw_kernel_triplet,
    select_atoms_in_radius,
    select_neutral_first_shell,
    select_xas_fragment,
)
from casidapy.xas.dipole import (
    bind_ae_core_dipole,
    core_charge_centroid_bohr,
    core_dipole_radius_bohr,
    core_enclosed_radius_bohr,
    core_rms_radius_bohr,
    prune_kernel_virtuals_by_core_dipole,
    pseudo_dipole_scale_report,
)
from casidapy.xas.gipaw import (
    apply_gipaw_kernel_virtuals,
    core_virtual_dipole_matrix_paw_gipaw,
    parse_gipaw_upf,
    reconstruct_pw_virtuals,
)
from casidapy.xas.projected_gto import (
    dense_cvs_A,
    gto_dipole_ia,
    hybrid_mo_ijab,
    mo_ijab,
    project_kernel_virtuals_to_gto,
    run_particle_ci,
    select_virtuals_by_shell_weight,
    subset_pw_kernel_virtuals,
    svd_particle_orbitals,
    virtual_shell_weights,
)
from casidapy.xas.reconstruct import (
    AE_HOST_CALC_DEFAULTS,
    ae_host_params_for_calc_type,
    calc_type_from_driver,
    frozen_shells_from_pp,
    ground_state_spin,
    resolve_calc_type,
    run_ae_host_pw_grid_cvs,
)
from casidapy.xas.spectrum import (
    HA_TO_EV,
    brightest_peak_shift_ev,
    first_peak_shift_ev,
    omega_ev,
    oscillator_table,
    plot_sticks,
    stick_spectrum,
    stick_cartesian,
    sum_d2,
    summarize_xas,
)

run_xas_gto = run_cvs_gto_from_mf

__all__ = [
    "HA_TO_EV",
    "CoreOrbitals",
    "CVSSOCResults",
    "FragmentSpec",
    "run_xas_gto",
    "run_cvs_gto_from_mf",
    "run_cvs_si_soc",
    "parse_soc_mode",
    "run_ae_host_pw_grid_cvs",
    "AE_HOST_CALC_DEFAULTS",
    "ae_host_params_for_calc_type",
    "calc_type_from_driver",
    "resolve_calc_type",
    "run_cvs_tda",
    "clone_pw_kernel_triplet",
    "core_from_mf",
    "extract_fragment_core",
    "inject_core_orbitals",
    "prune_kernel_virtuals_by_core_dipole",
    "bind_ae_core_dipole",
    "core_charge_centroid_bohr",
    "core_rms_radius_bohr",
    "core_enclosed_radius_bohr",
    "core_dipole_radius_bohr",
    "core_virtual_dipole_matrix_paw_gipaw",
    "parse_gipaw_upf",
    "reconstruct_pw_virtuals",
    "apply_gipaw_kernel_virtuals",
    "core_mos_to_pw_fields",
    "build_pw_kernel_from_qepy",
    "select_atoms_in_radius",
    "select_neutral_first_shell",
    "select_xas_fragment",
    "formal_oxidation_charge",
    "resolve_oxidation_state",
    "apply_orbital_soc",
    "pseudo_dipole_scale_report",
    "frozen_shells_from_pp",
    "ground_state_spin",
    "build_ae_embedding_potential",
    "read_upf_z_valence",
    "run_particle_ci",
    "select_virtuals_by_shell_weight",
    "subset_pw_kernel_virtuals",
    "hybrid_mo_ijab",
    "project_kernel_virtuals_to_gto",
    "svd_particle_orbitals",
    "virtual_shell_weights",
    "omega_ev",
    "stick_spectrum",
    "stick_cartesian",
    "plot_sticks",
    "first_peak_shift_ev",
    "brightest_peak_shift_ev",
    "oscillator_table",
    "summarize_xas",
    "sum_d2",
]
