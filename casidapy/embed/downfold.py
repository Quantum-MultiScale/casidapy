"""Host pseudopotential operators expressed in a local GTO basis.

Motivation
----------
The PW→AO route projects host *eigenstates* one at a time, which is why it
carries a per-state spillage (0.72-0.93 in production) and needs per-band
normalization, ghost reach-extension, and orthogonalization against a
separately-defined core.  The alternative is to express the host's *operator*
in the local basis once and diagonalize there, so the resulting states are
native to the basis and mutually orthogonal by construction.

Two constructions exist and they are not equivalent:

*spectral*  ``H_eff = V diag(ε) Vᵀ`` from the bands, with metric
    ``S_eff = V Vᵀ``.  Note the metric is the band-space projector, **not** the
    AO overlap; using ``S`` instead silently treats a rank-deficient projection
    as complete and manufactures intensity.  Spillage does not vanish here, it
    becomes the rank deficiency of ``S_eff``.

*operator*  ``H_eff = <φ| T + V_loc + V_NL |φ>`` with the ordinary AO overlap.
    No bands, no spillage, full-rank metric.  This module builds the
    pseudopotential pieces of that construction.

For a norm-conserving PP the local channel alone is **not** a usable
Hamiltonian: it is strongly attractive, and the repulsive nonlocal projectors
are what enforce orthogonality to the core states the PP removed.  Omitting
``V_NL`` in a flexible basis does not produce a small error, it produces
spurious deeply-bound core-region ghost states.

Units: UPF stores ``D_ij`` and ``v_loc`` in Rydberg; everything returned here
is Hartree.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

RY_TO_HA = 0.5


def _pp_dict(pp: Any) -> Dict[str, Any]:
    """Accept a parsed UPF dict or a path to one."""
    if isinstance(pp, dict):
        return pp
    from casidapy.utils.uspp import parse_upf

    return parse_upf(str(pp))


def beta_radial_cutoff(pp: Dict[str, Any], *, eps: float = 1e-10) -> float:
    """Outermost radius where any beta projector is still nonzero (Bohr)."""
    r = np.asarray(pp["r"], float)
    r_cut = 0.0
    for b in pp.get("beta_r") or []:
        nz = np.nonzero(np.abs(np.asarray(b, float)) > eps)[0]
        if nz.size:
            r_cut = max(r_cut, float(r[nz[-1]]))
    return r_cut


def beta_functions_on_points(
    pp: Dict[str, Any],
    coords_bohr: np.ndarray,
    center_bohr: Sequence[float],
    *,
    eps: float = 1e-10,
):
    """Evaluate every ``(channel, m)`` projector ``β_i(r) Y_{l_i m}(r̂)``.

    Returns ``(values, channels)`` where ``values`` has shape
    ``(n_proj_lm, n_pts)`` and ``channels`` is a list of ``(i, l, m)`` with
    ``i`` the radial channel index into ``pp['beta_l']``.
    """
    from scipy.interpolate import interp1d

    from casidapy.utils.uspp import _ylm_on_grid

    coords = np.asarray(coords_bohr, float).reshape(-1, 3)
    R0 = np.asarray(center_bohr, float).reshape(3)
    d = coords - R0.reshape(1, 3)
    dist = np.sqrt(np.sum(d**2, axis=1))

    r = np.asarray(pp["r"], float)
    beta_l = list(pp.get("beta_l") or [])
    beta_r = list(pp.get("beta_r") or [])

    vals, channels = [], []
    for i, (l, b) in enumerate(zip(beta_l, beta_r)):
        b = np.asarray(b, float)
        nz = np.nonzero(np.abs(b) > eps)[0]
        r_cut = float(r[nz[-1]]) if nz.size else 0.0
        # Projectors are strictly zero beyond r_cut; a cubic spline inside,
        # hard zero outside (extrapolating a PP projector is meaningless).
        f = interp1d(r, b, kind="cubic", bounds_error=False, fill_value=0.0)
        rad = np.where(dist <= r_cut, f(np.minimum(dist, r[-1])), 0.0)
        for m in range(-l, l + 1):
            y = _ylm_on_grid(l, m, d[:, 0], d[:, 1], d[:, 2], dist)
            vals.append(rad * y)
            channels.append((i, l, m))
    if not vals:
        return np.zeros((0, coords.shape[0])), []
    return np.asarray(vals, float), channels


def nonlocal_pp_ao_matrix_from_sites(
    mol,
    sites,
    *,
    becke_level: int = 4,
    eps: float = 1e-10,
    verbose: bool = False,
):
    """``V_NL`` from PP sites that need not be atoms of ``mol``.

    ``sites``
        Iterable of ``(center_bohr, pp)`` where ``pp`` is a parsed UPF dict or
        path.  Used for the AE-absorber-in-host construction: the absorber
        (and optionally its peel) live in ``mol`` as all-electron atoms, while
        the environment keeps its nonlocal projectors centered on the host
        nuclear positions.
    """
    from pyscf.dft import gen_grid, numint

    grids = gen_grid.Grids(mol)
    grids.level = int(becke_level)
    grids.build()
    coords = np.asarray(grids.coords, float)
    weights = np.asarray(grids.weights, float)

    nao = int(mol.nao_nr())
    V = np.zeros((nao, nao), float)
    meta = {"n_sites": 0, "becke_level": int(becke_level), "sites": []}

    for i_site, (center, pp_in) in enumerate(sites or []):
        pp = _pp_dict(pp_in)
        dij = pp.get("dij_ry")
        if dij is None:
            raise ValueError(
                f"site {i_site}: UPF has no PP_DIJ, so V_NL cannot be built."
            )
        dij = np.asarray(dij, float)
        beta_l = list(pp.get("beta_l") or [])
        if not beta_l:
            continue
        r_cut = beta_radial_cutoff(pp, eps=eps)
        if r_cut <= 0.0:
            continue
        R0 = np.asarray(center, float).reshape(3)
        dist = np.linalg.norm(coords - R0.reshape(1, 3), axis=1)
        sel = np.flatnonzero(dist <= r_cut)
        if sel.size == 0:
            continue

        b_vals, channels = beta_functions_on_points(pp, coords[sel], R0, eps=eps)
        if not channels:
            continue
        ao = numint.eval_ao(mol, coords[sel])
        B = (b_vals * weights[sel][None, :]) @ ao

        n_lm = len(channels)
        v_site = np.zeros((nao, nao), float)
        for a in range(n_lm):
            i_a, l_a, m_a = channels[a]
            for b in range(n_lm):
                i_b, l_b, m_b = channels[b]
                if l_a != l_b or m_a != m_b:
                    continue
                d = float(dij[i_a, i_b])
                if d == 0.0:
                    continue
                v_site += d * np.outer(B[a], B[b])
        v_site *= RY_TO_HA
        V += v_site
        meta["n_sites"] += 1
        meta["sites"].append(
            {
                "center_bohr": [float(x) for x in R0],
                "n_channels": len(beta_l),
                "n_proj_lm": n_lm,
                "r_cut_bohr": float(r_cut),
                "n_pts": int(sel.size),
                "frobenius_ha": float(np.linalg.norm(v_site)),
            }
        )
        if verbose:
            print(
                f"  [V_NL] env site {i_site}: {len(beta_l)} channels, "
                f"r_cut={r_cut:.2f} bohr, {sel.size} pts, "
                f"||V||_F={np.linalg.norm(v_site):.4f} Ha",
                flush=True,
            )

    V = 0.5 * (V + V.T)
    return V, meta


def nonlocal_pp_ao_matrix(
    mol,
    pp_by_atom: Dict[int, Any],
    *,
    becke_level: int = 4,
    eps: float = 1e-10,
    verbose: bool = False,
):
    """``V_NL[μν] = Σ_ij D_ij Σ_m <φ_μ|β_i m><β_j m|φ_ν>`` in Hartree.

    ``pp_by_atom``
        Maps atom index in ``mol`` to a parsed UPF dict (or a path).  Atoms
        absent from the map contribute nothing — pass only the atoms that
        should keep their pseudopotential (e.g. everything except an
        all-electron absorber).

    ``D_ij`` couples only channels of equal ``l``; cross-``l`` entries are
    dropped rather than trusted, since the UPF stores a full matrix whose
    off-``l`` blocks are not meaningful.

    Integration uses the molecular Becke grid restricted to each atom's
    projector cutoff, so tight GTO primitives are resolved (the QE charge mesh
    cannot resolve them) while cost stays local.
    """
    from pyscf.dft import gen_grid, numint

    grids = gen_grid.Grids(mol)
    grids.level = int(becke_level)
    grids.build()
    coords = np.asarray(grids.coords, float)
    weights = np.asarray(grids.weights, float)

    nao = int(mol.nao_nr())
    V = np.zeros((nao, nao), float)
    R = np.asarray(mol.atom_coords(), float)
    meta = {"n_atoms_with_pp": 0, "becke_level": int(becke_level), "atoms": {}}

    for ia, pp_in in (pp_by_atom or {}).items():
        ia = int(ia)
        pp = _pp_dict(pp_in)
        dij = pp.get("dij_ry")
        if dij is None:
            raise ValueError(
                f"atom {ia}: UPF has no PP_DIJ, so V_NL cannot be built. "
                "A UPF v2 file is required (the old Vanderbilt reader does "
                "not expose D_ij)."
            )
        dij = np.asarray(dij, float)
        beta_l = list(pp.get("beta_l") or [])
        if not beta_l:
            continue

        r_cut = beta_radial_cutoff(pp, eps=eps)
        if r_cut <= 0.0:
            continue
        dist = np.linalg.norm(coords - R[ia].reshape(1, 3), axis=1)
        sel = np.flatnonzero(dist <= r_cut)
        if sel.size == 0:
            continue

        b_vals, channels = beta_functions_on_points(
            pp, coords[sel], R[ia], eps=eps
        )
        if not channels:
            continue
        ao = numint.eval_ao(mol, coords[sel])
        # <phi_mu | beta_k> on the quadrature
        B = (b_vals * weights[sel][None, :]) @ ao  # (n_lm, nao)

        n_lm = len(channels)
        v_atom = np.zeros((nao, nao), float)
        for a in range(n_lm):
            i_a, l_a, m_a = channels[a]
            for b in range(n_lm):
                i_b, l_b, m_b = channels[b]
                # Block-diagonal in (l, m): D couples radial channels only.
                if l_a != l_b or m_a != m_b:
                    continue
                d = float(dij[i_a, i_b])
                if d == 0.0:
                    continue
                v_atom += d * np.outer(B[a], B[b])
        v_atom *= RY_TO_HA
        V += v_atom
        meta["n_atoms_with_pp"] += 1
        meta["atoms"][ia] = {
            "symbol": mol.atom_symbol(ia),
            "n_channels": len(beta_l),
            "n_proj_lm": n_lm,
            "r_cut_bohr": float(r_cut),
            "n_pts": int(sel.size),
            "frobenius_ha": float(np.linalg.norm(v_atom)),
        }
        if verbose:
            print(
                f"  [V_NL] atom {ia} {mol.atom_symbol(ia)}: "
                f"{len(beta_l)} channels -> {n_lm} (l,m) projectors, "
                f"r_cut={r_cut:.2f} bohr, {sel.size} pts, "
                f"||V||_F={np.linalg.norm(v_atom):.4f} Ha",
                flush=True,
            )

    # Symmetric by construction; enforce against quadrature round-off.
    V = 0.5 * (V + V.T)
    return V, meta


def atom_local_pp_grid_ha(
    pp: Any,
    grid,
    atom_pos_bohr: Sequence[float],
    cell_bohr,
):
    """One atom's local PP channel ``v_loc(|r-R|)`` on the charge grid (Hartree).

    This is the piece that can be removed from the host's total potential
    *exactly*, because it is an analytically known atom-centered function read
    straight from the UPF rather than something inferred by density
    partitioning or recovered by inversion.  Outside the radial mesh it is
    continued as the ``-Z_val/r`` Coulomb tail (verified against the UPF tail
    to 5 decimals for the production ONCV files).

    Note what this does **not** do: the Hartree and XC parts of the host
    potential are nonlinear in the density, so no single atom's share of them
    is defined without a density partition.  Removing ``v_loc`` alone converts
    a PP atom into a bare-valence one, not into an all-electron one.
    """
    from casidapy.embed.potential import project_radial_to_grid

    pp = _pp_dict(pp)
    v_ry = pp.get("vloc_rad_ry")
    if v_ry is None:
        raise ValueError(
            "UPF has no PP_LOCAL (or an old-format file was used), so the "
            "atom's local channel cannot be subtracted."
        )
    v_ha = RY_TO_HA * np.asarray(v_ry, float)
    return project_radial_to_grid(
        np.asarray(pp["r"], float),
        v_ha,
        grid,
        atom_pos_bohr,
        cell_bohr,
        outside="coulomb",
    )


def kinetic_ao_matrix(mol):
    """``T[μν] = <φ_μ|-½∇²|φ_ν>`` (analytic, Hartree)."""
    return np.asarray(mol.intor("int1e_kin"), float)


def host_local_potential_ao_matrix(
    mol,
    v_grid_ha,
    grid,
    *,
    grid_stride: int = 1,
    comm=None,
):
    """``<φ_μ|V(r)|φ_ν>`` for a host potential already on the charge grid (Ha)."""
    from casidapy.xas.reconstruct import _potential_ao_matrix

    return _potential_ao_matrix(
        mol, v_grid_ha, grid, grid_stride=int(grid_stride), comm=comm
    )
