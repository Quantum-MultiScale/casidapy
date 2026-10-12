"""Grid quadrature, fragment peel/basis helpers, and XAS alignment utilities."""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from casidapy.xas.chemical_data import (
    DEFAULTS,
    first_shell_peel_radius_ang,
    first_shell_peel_radius_ang as default_first_shell_radius_ang,
    second_shell_peel_radius_ang as default_second_shell_radius_ang,
)

_BASIS_TYPE = Union[str, dict]

# Default PW→AO projector: atom-centered Becke grid + PW G-sum (not QE mesh).
DEFAULT_PW_AO_PROJECTOR = "becke"


def grid_coords_bohr(grid) -> np.ndarray:
    """Cartesian grid points in Bohr, shape ``(n_grid, 3)``."""
    r = np.asarray(grid.r)
    if r.ndim != 4 or r.shape[0] != 3:
        raise ValueError(f"Unexpected grid.r shape {r.shape}")
    return np.stack([r[0].ravel(), r[1].ravel(), r[2].ravel()], axis=1)


def grid_orbital_norms(orbitals, grid) -> np.ndarray:
    """Grid L² norms ``‖ψ‖`` for orbitals on a DFTpy grid."""
    dV = float(grid.dV)
    return np.array(
        [
            float(np.sqrt(np.sum(np.asarray(p, dtype=float).ravel() ** 2) * dV))
            for p in orbitals
        ],
        dtype=float,
    )


def project_grid_orbitals_to_fragment_ao(
    mol,
    grid,
    orbitals,
    *,
    blksize: int = 16000,
    comm=None,
    method: Optional[str] = None,
    becke_level: int = 4,
    absorber_atom: Optional[int] = None,
    absorber_ao_only: bool = False,
    driver=None,
    band_indices: Optional[Sequence[int]] = None,
    g_cutoff_rel: float = 1e-10,
    ecut_ha: Optional[float] = 15.0,
    shell_cutoff_eps: float = 1e-12,
    psi_blk_pts: int = 256,
):
    """Project PW / grid orbitals onto fragment GTO AOs: ``V[μ,a] = ∫ φ_μ ψ_a dV``.

    ``method``
        ``'becke'`` (default): PySCF Becke/Treutler–Ahlrichs grid for ``φ_μ``
        via ``eval_ao`` (analytic, so the AO side is exact at every point);
        ``ψ_a`` by tricubic B-spline of the stored QE orbital.  Per-shell
        Schwarz radii keep tight primitives local.  Resolves tight absorber
        primitives that a valence ``ecutwfc`` mesh cannot.
        ``'qe_grid'``: legacy uniform QE charge-grid quadrature.

    ``CASIDAPY_PW_AO_PSI`` selects the ``ψ`` reconstruction: ``spline3``
    (default), ``trilinear`` (pre-2026 default, ~1-9% low on core dipoles),
    ``fft`` (exact band-limited G-sum, expensive), ``evc`` (QE coefficients).
    """
    mode = str(method or DEFAULT_PW_AO_PROJECTOR).lower().strip()
    if mode in ("qe_grid", "qe", "fft", "charge_grid", "legacy"):
        return _project_on_qe_charge_grid(
            mol, grid, orbitals, blksize=blksize, comm=comm
        )
    if mode in ("becke", "atomic", "pyscf", "gsum", "pw_sum"):
        return project_pw_orbitals_to_fragment_ao_becke(
            mol,
            grid,
            orbitals,
            becke_level=becke_level,
            absorber_atom=absorber_atom,
            absorber_ao_only=absorber_ao_only,
            driver=driver,
            band_indices=band_indices,
            g_cutoff_rel=g_cutoff_rel,
            ecut_ha=ecut_ha,
            shell_cutoff_eps=shell_cutoff_eps,
            blksize=blksize,
            psi_blk_pts=psi_blk_pts,
            comm=comm,
        )
    raise ValueError(
        f"unknown PW→AO projection method={method!r}; use 'becke' or 'qe_grid'"
    )


def _project_on_qe_charge_grid(
    mol,
    grid,
    orbitals,
    *,
    blksize: int = 16000,
    comm=None,
):
    """Legacy ``V[μ,a]`` on the uniform QE charge mesh."""
    from pyscf.dft import numint
    from casidapy.utils.accel import block_slices

    coords = grid_coords_bohr(grid)
    dV = float(grid.dV)
    nao = int(mol.nao_nr())
    virts = list(orbitals)
    n_virt = len(virts)
    if n_virt == 0:
        return np.zeros((nao, 0), dtype=float)
    flat = [np.asarray(v, dtype=float).ravel() for v in virts]
    ngrid = coords.shape[0]
    V = np.zeros((nao, n_virt), dtype=float)
    for p0, p1 in block_slices(ngrid, blksize, comm):
        ao = numint.eval_ao(mol, coords[p0:p1])
        w = dV
        for a in range(n_virt):
            V[:, a] += ao.T @ (flat[a][p0:p1] * w)
    return V


def build_becke_projection_grid(
    mol,
    *,
    level: int = 4,
    absorber_atom: Optional[int] = None,
):
    """PySCF Becke/Treutler–Ahlrichs grid for PW→AO projection.

    ``absorber_atom``
        If set, keep only quadrature points whose nearest nucleus is that
        atom (absorber-local support).  Enough for single-center absorber
        AOs; full molecular grids are used when ``None``.
    """
    from pyscf.dft import gen_grid

    grids = gen_grid.Grids(mol)
    grids.level = int(level)
    grids.build()
    coords = np.asarray(grids.coords, dtype=float)
    weights = np.asarray(grids.weights, dtype=float)
    meta = {
        "n_pts": int(coords.shape[0]),
        "level": int(level),
        "absorber_atom": None,
    }
    if absorber_atom is None:
        return coords, weights, meta

    ia = int(absorber_atom)
    if ia < 0 or ia >= mol.natm:
        raise ValueError(f"absorber_atom={ia} out of range for natm={mol.natm}")
    R = np.asarray(mol.atom_coords(), dtype=float)  # (natm, 3)
    # Nearest-atom mask (cheap absorber-local cut of the Becke cloud).
    d2 = np.sum((coords[:, None, :] - R[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(d2, axis=1)
    keep = nearest == ia
    coords = coords[keep]
    weights = weights[keep]
    meta.update(
        {
            "n_pts": int(coords.shape[0]),
            "absorber_atom": ia,
            "n_pts_full": int(keep.size),
        }
    )
    return coords, weights, meta


def shell_projection_radii(
    mol,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-shell Schwarz-style cutoffs ``r_cut = sqrt(ln(1/ε) / α_min)``.

    Uses each shell's **most diffuse** primitive exponent so a contracted shell
    is never truncated inside the support of any of its primitives.  Segmented
    core shells (large ``α_min``) get small radii; diffuse / ghost /
    general-contracted Dunning shells that still contain a small exponent keep
    long reach.  That is intentional: cutting on ``α_max`` would silently drop
    the diffuse end of a GCGTO core shell.

    Returns ``(r_cut, alpha_min, atom_of_shell)`` length ``mol.nbas``.
    """
    eps = float(eps)
    if not (0.0 < eps < 1.0):
        raise ValueError(f"shell cutoff eps must be in (0,1), got {eps}")
    ln_inv = float(np.log(1.0 / eps))
    nbas = int(mol.nbas)
    r_cut = np.empty(nbas, dtype=float)
    alpha_min = np.empty(nbas, dtype=float)
    atom_of = np.empty(nbas, dtype=int)
    # Cap absurdly diffuse shells so a single near-zero exponent cannot force
    # the entire molecular Becke cloud into the ψ evaluation set.
    r_cap = 50.0
    for ib in range(nbas):
        exps = np.asarray(mol.bas_exp(ib), dtype=float).ravel()
        a_min = float(np.min(exps)) if exps.size else 0.0
        alpha_min[ib] = a_min
        atom_of[ib] = int(mol.bas_atom(ib))
        if a_min <= 0.0:
            r_cut[ib] = r_cap
        else:
            r_cut[ib] = min(r_cap, float(np.sqrt(ln_inv / a_min)))
    return r_cut, alpha_min, atom_of


def project_pw_orbitals_to_fragment_ao_becke(
    mol,
    grid,
    orbitals,
    *,
    becke_level: int = 4,
    absorber_atom: Optional[int] = None,
    absorber_ao_only: bool = False,
    driver=None,
    band_indices: Optional[Sequence[int]] = None,
    g_cutoff_rel: float = 1e-10,
    ecut_ha: Optional[float] = 15.0,
    shell_cutoff_eps: float = 1e-12,
    blksize: int = 8000,
    psi_blk_pts: int = 256,
    comm=None,
):
    """``V[μ,a]`` on a Becke grid; AOs via ``eval_ao``, PW via sampled/G-sum ψ.

    Default ``ψ`` evaluation is a **tricubic B-spline** of the stored QE-mesh
    orbital onto Becke points.  The mesh oversamples the ``ecutwfc`` band limit
    (``h ≈ 0.2`` bohr vs Nyquist ``≈ 0.57``), so ``ψ`` is fully determined by
    the samples and only the reconstruction order matters: against an exact
    G-sum on a band-limited test field, tricubic reproduces core dipoles to
    ~1e-4 where trilinear (``CASIDAPY_PW_AO_PSI=trilinear``) is 1-9% low,
    worst for tight cores.  Opt in to the exact G-sum with
    ``CASIDAPY_PW_AO_PSI=fft``, or ``=evc`` for QE coefficients.

    Each GTO shell only accumulates from Becke points inside its own
    Schwarz-style radius ``r_cut(μ)=√(ln(1/ε)/α_min(μ))`` so tight core
    primitives stay local while diffuse / ghost shells keep long reach.
    ``ψ_a`` is evaluated once on the union of those points (cached).
    """
    import os

    from pyscf.dft import numint
    from casidapy.xas.pw_eval import (
        orbitals_at_coords_pw_sum,
        orbitals_at_coords_qe_evc,
        orbitals_at_coords_spline,
        orbitals_at_coords_trilinear,
    )

    virts = list(orbitals)
    n_virt = len(virts)
    nao = int(mol.nao_nr())
    if n_virt == 0:
        return np.zeros((nao, 0), dtype=float)

    # Atom peels (natm==1): absorber-local quadrature is exact for all AOs.
    # Multi-atom fragments need the full molecular Becke cloud unless the
    # caller explicitly requests absorber-only AO rows.
    abs_atom = absorber_atom
    if abs_atom is None and mol.natm == 1:
        abs_atom = 0
    grid_atom = abs_atom if (mol.natm == 1 or absorber_ao_only) else None

    coords, weights, grid_meta = build_becke_projection_grid(
        mol, level=becke_level, absorber_atom=grid_atom
    )
    n_pts = int(coords.shape[0])
    if n_pts == 0:
        raise RuntimeError("Becke projection grid is empty")
    w = np.asarray(weights, dtype=float).ravel()

    r_cut, alpha_min, atom_of = shell_projection_radii(
        mol, eps=float(shell_cutoff_eps)
    )
    R = np.asarray(mol.atom_coords(), dtype=float)
    # Distance² from each atom to every Becke point.
    d2_atom = [
        np.sum((coords - R[ia][None, :]) ** 2, axis=1) for ia in range(mol.natm)
    ]

    nbas = int(mol.nbas)
    ao_loc = np.asarray(mol.ao_loc_nr(), dtype=int)
    shell_masks: list[np.ndarray] = []
    need = np.zeros(n_pts, dtype=bool)
    n_pts_per_shell = np.zeros(nbas, dtype=int)
    for ib in range(nbas):
        ia = int(atom_of[ib])
        mask = d2_atom[ia] <= (float(r_cut[ib]) ** 2)
        shell_masks.append(mask)
        n_pts_per_shell[ib] = int(np.count_nonzero(mask))
        need |= mask

    idx_need = np.flatnonzero(need)
    if idx_need.size == 0:
        # Pathological: all shells screened out — fall back to full grid.
        idx_need = np.arange(n_pts, dtype=int)
        need[:] = True
        shell_masks = [np.ones(n_pts, dtype=bool) for _ in range(nbas)]
        n_pts_per_shell[:] = n_pts

    coords_need = coords[idx_need]
    # Map full-grid index → compact need index for ψ lookup.
    full_to_need = np.full(n_pts, -1, dtype=int)
    full_to_need[idx_need] = np.arange(idx_need.size, dtype=int)

    psi_mode = str(os.environ.get("CASIDAPY_PW_AO_PSI", "spline3")).lower().strip()
    if psi_mode in ("evc", "qe", "driver"):
        if driver is None or band_indices is None or len(list(band_indices)) != n_virt:
            raise ValueError(
                "CASIDAPY_PW_AO_PSI=evc requires driver= and matching band_indices"
            )
        psi_need, psi_meta = orbitals_at_coords_qe_evc(
            driver,
            band_indices,
            coords_need,
            real=True,
            blk_pts=psi_blk_pts,
        )
    elif psi_mode in ("fft", "fourier", "gsum", "pw_sum"):
        psi_need, psi_metas = orbitals_at_coords_pw_sum(
            virts,
            grid,
            coords_need,
            g_cutoff_rel=g_cutoff_rel,
            ecut_ha=ecut_ha,
            real=True,
            blk_pts=psi_blk_pts,
        )
        psi_meta = {
            "source": "fft_interpolant",
            "n_G_keep_mean": float(
                np.mean([m["n_G_keep"] for m in psi_metas]) if psi_metas else 0.0
            ),
            "ecut_ha": ecut_ha,
        }
    elif psi_mode in ("spline", "spline3", "tricubic", "bspline"):
        psi_need, psi_meta = orbitals_at_coords_spline(virts, grid, coords_need, order=3)
    elif psi_mode in ("trilinear", "linear", "spline1"):
        psi_need, psi_meta = orbitals_at_coords_trilinear(virts, grid, coords_need)
    else:
        raise ValueError(
            f"unknown CASIDAPY_PW_AO_PSI={psi_mode!r}; use 'spline3', "
            "'trilinear', 'fft', or 'evc'"
        )

    V = np.zeros((nao, n_virt), dtype=float)
    # Per-shell AO eval on that shell's local point set only.
    for ib in range(nbas):
        mask = shell_masks[ib]
        if not np.any(mask):
            continue
        pts = np.flatnonzero(mask)
        # Compact ψ rows for these points.
        rows = full_to_need[pts]
        good = rows >= 0
        if not np.any(good):
            continue
        pts = pts[good]
        rows = rows[good]
        ao_blk = numint.eval_ao(mol, coords[pts], shls_slice=(ib, ib + 1))
        # ao_blk shape (n_pts_shell, n_ao_shell)
        p0, p1 = int(ao_loc[ib]), int(ao_loc[ib + 1])
        ww = w[pts]
        V[p0:p1, :] += ao_blk.T @ (psi_need[rows] * ww[:, None])

    if absorber_ao_only and abs_atom is not None:
        ao_slice = mol.aoslice_by_atom()
        p0, p1 = int(ao_slice[int(abs_atom), 2]), int(ao_slice[int(abs_atom), 3])
        mask_ao = np.ones(nao, dtype=bool)
        mask_ao[p0:p1] = False
        V[mask_ao, :] = 0.0

    screen_meta = {
        "eps": float(shell_cutoff_eps),
        "n_pts_becke": int(n_pts),
        "n_pts_psi": int(idx_need.size),
        "frac_pts_kept": float(idx_need.size) / float(max(n_pts, 1)),
        "r_cut_min": float(np.min(r_cut)),
        "r_cut_max": float(np.max(r_cut)),
        "r_cut_median": float(np.median(r_cut)),
        "alpha_min_max": float(np.max(alpha_min)),  # tightest shell
        "alpha_min_min": float(np.min(alpha_min)),  # most diffuse shell
        "n_pts_per_shell_min": int(np.min(n_pts_per_shell)),
        "n_pts_per_shell_max": int(np.max(n_pts_per_shell)),
    }

    if getattr(mol, "meta", None) is None:
        try:
            mol.meta = {}
        except Exception:
            mol.meta = None
    if isinstance(getattr(mol, "meta", None), dict):
        mol.meta["pw_ao_projection"] = {
            "method": "becke",
            "becke": grid_meta,
            "psi": psi_meta if isinstance(psi_meta, dict) else {"info": str(psi_meta)},
            "shell_screening": screen_meta,
            "n_virt": int(n_virt),
            "nao": int(nao),
        }
    _ = blksize  # unused with per-shell loop; kept for API compat
    _ = comm
    return V


def grid_projection_to_ao_coeffs(mol, V_ao: np.ndarray) -> np.ndarray:
    """AO coefficients from grid projections ``v_μ = ∫ φ_μ ψ dV``.

    For ``ψ = Σ_ν c_ν φ_ν`` with overlap ``S``, ``v = S c`` so ``c = S^{-1} v``.
    Using raw ``v`` as MO coefficients in dipole integrals inflates ``|μ|``
    when the projector basis is non-orthogonal (ghost diffuse peel).
    """
    V = np.asarray(V_ao, dtype=float)
    if V.size == 0:
        return V
    S = mol.intor("int1e_ovlp")
    return np.linalg.solve(S, V)


def normalize_ao_coeffs_s_metric(
    C: np.ndarray,
    S: np.ndarray,
    *,
    target_norm: float = 1.0,
    max_scale: Optional[float] = None,
    min_norm: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Column-wise ``S``-normalize AO MO coeffs (no virtual mixing).

    Projected PW virtuals ``C = S^{-1}v`` have ``‖c‖_S = √completeness ≤ 1``.
    Leaving them under-normalized suppresses dipoles; ghost peels can also
    produce irregular column norms.  Scaling each column to ``target_norm``
    restores a consistent intensity scale without Löwdin mixing (which would
    require rotating the attached PW eigenvalues).

    ``max_scale`` caps amplification of near-null projections (optional).
    Returns ``(C_norm, scales)``.
    """
    C = np.asarray(C, dtype=float)
    if C.ndim == 1:
        C = C.reshape(-1, 1)
    S = np.asarray(S, dtype=float)
    # ‖c‖_S² = cᵀ S c
    norms = np.sqrt(np.maximum(np.einsum("pi,pq,qi->i", C, S, C), 0.0))
    scales = np.full(norms.shape, float(target_norm), dtype=float)
    good = norms >= float(min_norm)
    scales[good] = float(target_norm) / norms[good]
    scales[~good] = 0.0
    if max_scale is not None:
        scales = np.minimum(scales, float(max_scale))
    return C * scales[None, :], scales


def grid_charge_centroid(orbitals, grid) -> tuple:
    """Charge centroid ``<r>`` of ``Σ|ψ|²`` over ``orbitals`` (Bohr)."""
    w = None
    for p in orbitals:
        arr = np.asarray(p, dtype=float).ravel()
        w = arr * arr if w is None else w + arr * arr
    if w is None:
        raise ValueError("no orbitals given for centroid")
    dV = float(grid.dV)
    tot = float(w.sum()) * dV
    if not np.isfinite(tot) or tot <= 0.0:
        raise ValueError("orbitals have non-positive norm on the grid")
    return tuple(
        float((np.asarray(grid.r[a]).ravel() * w).sum()) * dV / tot for a in range(3)
    )


def ensure_closed_shell_peel_indices(atoms, indices, *, charge: int = 0) -> list[int]:
    """Return a peel with even electron count suitable for RKS.

    Radius-based first shells on organics often pick an odd number of H
    (e.g. anthracene C + 2C + 1H → 19 e⁻).  Drop surplus hydrogens (never the
    edge atom) until ``ΣZ − charge`` is even, preserving the heavy-atom
    skeleton that carries the near-edge virtuals.
    """
    from pyscf.data import elements

    idx = [int(i) for i in indices]
    if not idx:
        return idx

    def _nelec(ids):
        return sum(int(elements.charge(str(atoms[i].symbol))) for i in ids) - int(charge)

    if _nelec(idx) % 2 == 0:
        return idx
    # Drop H farthest from the edge atom first (last in the radius-ordered list).
    edge = idx[0]
    for h in reversed([i for i in idx[1:] if str(atoms[i].symbol).upper() == "H"]):
        trial = [i for i in idx if i != h]
        if edge in trial and _nelec(trial) % 2 == 0:
            return trial
    return idx


def resolve_peel_atom_indices(
    atoms,
    edge_atom: int,
    *,
    peel: str = "atom",
    peel_radius_ang: Optional[float] = None,
    active_atom_indices: Optional[Sequence[int]] = None,
) -> list[int]:
    """Active-site atom indices for AE peel / AO projection.

    peel
        ``atom`` — edge atom only (production default).
        ``first_shell`` — all atoms within ``peel_radius_ang`` (or element default).
        ``second_shell`` — atoms within the second-shell peel radius (element default).
        ``all`` / ``molecule`` / ``full`` — every atom in the structure (CH4
        identity check; anthracene π* needs at least this, not an atom peel).
    active_atom_indices
        Explicit override; edge atom is always included first.
    """
    edge = int(edge_atom)
    if active_atom_indices is not None:
        active = [edge] + [int(i) for i in active_atom_indices if int(i) != edge]
        return active

    mode = str(peel).lower().strip()
    if mode in ("atom", "single", "one"):
        return [edge]

    if mode in ("all", "molecule", "full", "system"):
        n = len(atoms)
        return [edge] + [int(i) for i in range(n) if int(i) != edge]

    if mode in ("first_shell", "shell", "radius"):
        from casidapy.xas.cvs import select_atoms_in_radius

        sym = atoms.get_chemical_symbols()[edge]
        r_ang = float(
            peel_radius_ang
            if peel_radius_ang is not None
            else first_shell_peel_radius_ang(sym)
        )
        idx = select_atoms_in_radius(atoms, edge, r_ang)
        ordered = [edge] + [int(i) for i in idx if int(i) != edge]
        return ordered

    if mode in ("second_shell", "shell2", "two_shell"):
        from casidapy.xas.cvs import select_atoms_in_radius

        sym = atoms.get_chemical_symbols()[edge]
        r_ang = float(
            peel_radius_ang
            if peel_radius_ang is not None
            else default_second_shell_radius_ang(sym)
        )
        idx = select_atoms_in_radius(atoms, edge, r_ang)
        ordered = [edge] + [int(i) for i in idx if int(i) != edge]
        return ordered

    raise ValueError(
        f"unknown peel={peel!r}; use 'atom', 'first_shell', 'second_shell', or 'all'"
    )


def make_fragment_basis(
    basis: _BASIS_TYPE,
    element_symbols: Sequence[str],
    *,
    diffuse_augment: bool = False,
    zeta_s: float = DEFAULTS["diffuse_augment_zeta_s"],
    zeta_p: float = DEFAULTS["diffuse_augment_zeta_p"],
) -> _BASIS_TYPE:
    """Return a PySCF basis spec, optionally with one diffuse s+p per element."""
    if not diffuse_augment:
        return basis

    from pyscf import gto

    base_name = str(basis) if isinstance(basis, str) else None
    if isinstance(basis, dict):
        spec = {str(k): list(v) for k, v in basis.items()}
    else:
        spec = {}

    for el in sorted({str(s).strip().capitalize() for s in element_symbols}):
        if el in spec:
            raw = list(spec[el])
        elif base_name is not None:
            raw = list(gto.basis.load(base_name, el))
        else:
            raise ValueError(
                f"cannot augment basis for {el}: pass a named basis string or "
                "a complete dict"
            )
        raw.append([0, [float(zeta_s), 1.0]])
        raw.append([1, [float(zeta_p), 1.0]])
        spec[el] = raw
    return spec


def peel_summary(atoms, peel_indices: Sequence[int]) -> dict:
    """Small metadata dict for spillage / audit reports."""
    idx = [int(i) for i in peel_indices]
    syms = [atoms.get_chemical_symbols()[i] for i in idx]
    return {
        "n_atoms": len(idx),
        "indices": idx,
        "symbols": syms,
        "formula": "".join(
            f"{s}{syms.count(s)}" for s in sorted(set(syms), key=syms.index)
        ),
    }

