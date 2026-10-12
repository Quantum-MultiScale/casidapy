"""Embedding potentials for AE core reconstruction.

Reading order
-------------
1. :mod:`casidapy.embed.potential` — ``V_ionic = vltot − V_loc(A)``, mode dispatch
2. :mod:`casidapy.embed.hirshfeld` — ``ρ_env`` partition → ``v_H/v_xc[ρ_env]``
3. :mod:`casidapy.embed.IHirshfeld` — iterative Hirshfeld (Hirshfeld-I)
4. :mod:`casidapy.embed.mbisa` — minimal-basis iterative stockholder (MBIS)

Typical call::

    from casidapy.embed import build_ae_embedding_potential
    v_env, grid, meta = build_ae_embedding_potential(
        driver, edge_atom=0, use_gpu=True, comm=comm
    )

``use_gpu`` uses CuPy for Hirshfeld weights / ``V_env`` assembly when
available. ``comm`` distributes the atom loop (round-robin) and
``Allreduce``s densities; leave ``None`` for serial.
"""
from casidapy.embed.potential import (
    DEFAULT_EMBED_MODE,
    DEFAULT_GAUGE_ALIGN,
    DEFAULT_R_DAMP,
    DEFAULT_VLOC_SOURCE,
    auto_vhxc_scale,
    ae_host_rho_tot_hxc_ha,
    build_ae_embedding_potential,
    classify_qe_local_potential,
    classify_vltot_kind,
    density_functional_potential_ha,
    effective_potential_ha,
    ensure_charge_grid,
    ionic_vloc_residual,
    local_pp_total_ha,
    neighbor_vloc_from_residual,
    normalize_ae_host_hxc,
    project_radial_to_grid,
    project_radial_vloc_to_grid,
    read_upf_local_potential,
    read_upf_z_valence,
    resolve_upf_path,
    resolve_vhxc_scale,
    ry_to_ha,
    shift_v_env_gauge_at,
    upf_vloc_lattice_sum_at_point,
    vhxc_radial_fade,
    vloc_atom_field_upf,
    vloc_atom_from_qepy,
    home_cell_image_shift_ha,
    qe_driver_is_isolated,
    resolve_vloc_source,
    upf_coulomb_match_radius,
)
from casidapy.embed.hirshfeld import (
    DEFAULT_HIRSHFELD_GAUGE_TOL,
    DEFAULT_HIRSHFELD_REFERENCE,
    DEFAULT_KEDF,
    HirshfeldPartition,
    build_hirshfeld_embedding,
    build_pot_hirshfeld_embedding,
    hartree_potential_ha,
    hirshfeld_weights_and_partition,
    kedf_potential_ha,
    mix_v_env_fields,
    nonadditive_kinetic_potential_ha,
    project_radial_density_to_grid,
    read_upf_rhoatom,
    rebuild_v_env_from_active_density,
    shift_v_env_gauge_hirshfeld,
    total_density_on_grid,
    upf_rhoatom_on_grid,
    xc_potential_ha,
)
from casidapy.embed.invert import (
    build_invert_embedding,
    invert_xc_potential_qepy,
    revive_host_driver,
)
from casidapy.embed.formal_ion_density import (
    clear_formal_ion_rhoatom_cache,
    formal_ion_rhoatom,
)
from casidapy.embed.IHirshfeld import (
    IterativeHirshfeldResult,
    iterative_hirshfeld_from_driver,
    iterative_hirshfeld_partition,
    make_charge_scale_updater,
    scale_promolecule_to_populations,
    stockholder_weights,
)
from casidapy.embed.mbisa import (
    MBISAtomModel,
    MBISResult,
    mbis_from_dftpy_grid,
    mbis_hirshfeld_partition,
    mbis_partition,
    n_mbis_shells,
    slater_s_density,
)

__all__ = [
    "DEFAULT_EMBED_MODE",
    "DEFAULT_GAUGE_ALIGN",
    "DEFAULT_HIRSHFELD_GAUGE_TOL",
    "DEFAULT_HIRSHFELD_REFERENCE",
    "DEFAULT_KEDF",
    "DEFAULT_R_DAMP",
    "DEFAULT_VLOC_SOURCE",
    "HirshfeldPartition",
    "IterativeHirshfeldResult",
    "MBISAtomModel",
    "MBISResult",
    "auto_vhxc_scale",
    "ae_host_rho_tot_hxc_ha",
    "build_ae_embedding_potential",
    "build_hirshfeld_embedding",
    "build_invert_embedding",
    "build_pot_hirshfeld_embedding",
    "classify_qe_local_potential",
    "classify_vltot_kind",
    "clear_formal_ion_rhoatom_cache",
    "density_functional_potential_ha",
    "effective_potential_ha",
    "ensure_charge_grid",
    "formal_ion_rhoatom",
    "hartree_potential_ha",
    "hirshfeld_weights_and_partition",
    "invert_xc_potential_qepy",
    "ionic_vloc_residual",
    "iterative_hirshfeld_from_driver",
    "iterative_hirshfeld_partition",
    "kedf_potential_ha",
    "local_pp_total_ha",
    "make_charge_scale_updater",
    "mbis_from_dftpy_grid",
    "mbis_hirshfeld_partition",
    "mbis_partition",
    "mix_v_env_fields",
    "n_mbis_shells",
    "neighbor_vloc_from_residual",
    "nonadditive_kinetic_potential_ha",
    "normalize_ae_host_hxc",
    "project_radial_density_to_grid",
    "project_radial_to_grid",
    "project_radial_vloc_to_grid",
    "read_upf_local_potential",
    "read_upf_rhoatom",
    "read_upf_z_valence",
    "rebuild_v_env_from_active_density",
    "resolve_upf_path",
    "resolve_vhxc_scale",
    "revive_host_driver",
    "ry_to_ha",
    "scale_promolecule_to_populations",
    "shift_v_env_gauge_at",
    "shift_v_env_gauge_hirshfeld",
    "slater_s_density",
    "stockholder_weights",
    "total_density_on_grid",
    "upf_rhoatom_on_grid",
    "upf_vloc_lattice_sum_at_point",
    "vhxc_radial_fade",
    "vloc_atom_field_upf",
    "vloc_atom_from_qepy",
    "home_cell_image_shift_ha",
    "qe_driver_is_isolated",
    "resolve_vloc_source",
    "upf_coulomb_match_radius",
    "xc_potential_ha",
]
