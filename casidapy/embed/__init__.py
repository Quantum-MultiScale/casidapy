"""Embedding potentials for AE core reconstruction.

Reading order
-------------
1. :mod:`casidapy.embed.potential` — ``V_ionic = vltot − V_loc(A)``, mode dispatch
2. :mod:`casidapy.embed.hirshfeld` — ``ρ_env`` partition → ``v_H/v_xc[ρ_env]``

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
    build_ae_embedding_potential,
    density_functional_potential_ha,
    effective_potential_ha,
    ensure_charge_grid,
    ionic_vloc_residual,
    local_pp_total_ha,
    neighbor_vloc_from_residual,
    project_radial_to_grid,
    project_radial_vloc_to_grid,
    read_upf_local_potential,
    read_upf_z_valence,
    resolve_upf_path,
    resolve_vhxc_scale,
    ry_to_ha,
    shift_v_env_gauge_at,
    vhxc_radial_fade,
    vloc_atom_field_upf,
    vloc_atom_from_qepy,
)
from casidapy.embed.hirshfeld import (
    DEFAULT_HIRSHFELD_GAUGE_TOL,
    HirshfeldPartition,
    build_hirshfeld_embedding,
    hartree_potential_ha,
    hirshfeld_weights_and_partition,
    project_radial_density_to_grid,
    read_upf_rhoatom,
    shift_v_env_gauge_hirshfeld,
    total_density_on_grid,
    upf_rhoatom_on_grid,
    xc_potential_ha,
)

__all__ = [
    "DEFAULT_EMBED_MODE",
    "DEFAULT_GAUGE_ALIGN",
    "DEFAULT_HIRSHFELD_GAUGE_TOL",
    "DEFAULT_R_DAMP",
    "DEFAULT_VLOC_SOURCE",
    "HirshfeldPartition",
    "auto_vhxc_scale",
    "build_ae_embedding_potential",
    "build_hirshfeld_embedding",
    "density_functional_potential_ha",
    "effective_potential_ha",
    "ensure_charge_grid",
    "hartree_potential_ha",
    "hirshfeld_weights_and_partition",
    "ionic_vloc_residual",
    "local_pp_total_ha",
    "neighbor_vloc_from_residual",
    "project_radial_density_to_grid",
    "project_radial_to_grid",
    "project_radial_vloc_to_grid",
    "read_upf_local_potential",
    "read_upf_rhoatom",
    "read_upf_z_valence",
    "resolve_upf_path",
    "resolve_vhxc_scale",
    "ry_to_ha",
    "shift_v_env_gauge_at",
    "shift_v_env_gauge_hirshfeld",
    "total_density_on_grid",
    "upf_rhoatom_on_grid",
    "vhxc_radial_fade",
    "vloc_atom_field_upf",
    "vloc_atom_from_qepy",
    "xc_potential_ha",
]
