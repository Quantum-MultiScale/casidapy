"""X-ray absorption / CVS — one public import surface.

Three workflows
---------------
1. **All-GTO molecule** (you own the SCF)::

       from casidapy.xas import run_xas_gto
       res, core, kernel = run_xas_gto(mf, edge="K", edge_atom_indices=[0])

2. **QE embedding → AE core → PW virtuals**::

       from casidapy.xas import run_xas_reconstruct
       res, core, kernel, mf = run_xas_reconstruct(
           driver, edge_atom=0, edge="K", use_gpu=True, comm=comm
       )

3. **Fragment core → inject yourself**::

       from casidapy.xas import select_xas_fragment, extract_fragment_core, inject_core_orbitals

CuPy / MPI
----------
Pass ``use_gpu=True`` and/or ``comm=<mpi4py communicator>`` on
``run_xas_reconstruct``, ``build_ae_embedding_potential``,
``core_mos_to_pw_fields``, and ``CasidaKS_MPI.xas(...)``. ``comm=None`` is
serial (no auto ``COMM_WORLD``). GTO-only XAS also accepts
``use_mpi_response=True`` for mpi4pyscf response kernels.

Large virtual pools
-------------------
For continuum-dense QE cells, set a large ``n_virt`` (QE ``nbnd``) and
optionally ``n_virt_active=N`` to energy-stride down to ``N`` Casida
virtuals while keeping the full energy span::

    run_xas_reconstruct(driver, 0, n_virt=800, n_virt_active=200, n_states=150)

Leave ``n_virt_active=None`` (default) to use every selected virtual.

Modules
-------
- :mod:`casidapy.xas.cvs` — cores, fragments, inject, CVS-TDA
- :mod:`casidapy.xas.reconstruct` — frozen ``V_env`` AE reconstruction
- :mod:`casidapy.xas.spectrum` — sticks / plots
- :mod:`casidapy.embed` — ``V_env`` builders (Hirshfeld default)
- :mod:`casidapy.utils.accel` — shared CuPy / MPI helpers

``CasidaKS_MPI.xas(reconstruct=True, ...)`` is the OO form of (2).
"""
from casidapy.embed import build_ae_embedding_potential, read_upf_z_valence
from casidapy.xas.cvs import (
    CoreOrbitals,
    FragmentSpec,
    apply_orbital_soc,
    build_pw_kernel_from_qepy,
    core_from_mf,
    core_mos_to_pw_fields,
    extract_fragment_core,
    formal_oxidation_charge,
    inject_core_orbitals,
    resolve_oxidation_state,
    run_cvs_gto_from_mf,
    run_cvs_tda,
    select_atoms_in_radius,
    select_neutral_first_shell,
    select_xas_fragment,
)
from casidapy.xas.reconstruct import (
    align_core_to_reference,
    apply_core_gauge_shift,
    frozen_shells_from_pp,
    reconstruct_core_from_driver,
    run_reconstruct_cvs_from_driver,
    scf_ae_core_semicore,
)
from casidapy.xas.spectrum import (
    HA_TO_EV,
    omega_ev,
    plot_sticks,
    stick_spectrum,
    summarize_xas,
)

run_xas_gto = run_cvs_gto_from_mf
run_xas_reconstruct = run_reconstruct_cvs_from_driver

__all__ = [
    "HA_TO_EV",
    "CoreOrbitals",
    "FragmentSpec",
    "run_xas_gto",
    "run_xas_reconstruct",
    "run_cvs_gto_from_mf",
    "run_reconstruct_cvs_from_driver",
    "run_cvs_tda",
    "core_from_mf",
    "extract_fragment_core",
    "inject_core_orbitals",
    "core_mos_to_pw_fields",
    "build_pw_kernel_from_qepy",
    "select_atoms_in_radius",
    "select_neutral_first_shell",
    "select_xas_fragment",
    "formal_oxidation_charge",
    "resolve_oxidation_state",
    "apply_orbital_soc",
    "reconstruct_core_from_driver",
    "scf_ae_core_semicore",
    "frozen_shells_from_pp",
    "apply_core_gauge_shift",
    "align_core_to_reference",
    "build_ae_embedding_potential",
    "read_upf_z_valence",
    "omega_ev",
    "stick_spectrum",
    "plot_sticks",
    "summarize_xas",
]
