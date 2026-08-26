"""AE core reconstruction in a frozen QE embedding potential.

Pipeline
--------
::

    V_env  = build_ae_embedding_potential(driver, edge_atom)   # embed/
    mf     = scf_ae_core_semicore(V_env, …)                    # one-atom PySCF
    core   = core_from_mf(mf)                                  # pick 1s / …
    # then inject into PW virtuals and run CVS-TDA:
    res, core, kernel, mf = run_reconstruct_cvs_from_driver(...)

Defaults: Hirshfeld ``V_env``, formal oxidation state (``ae_electrons="atomic"``).

Public entry points
-------------------
- :func:`reconstruct_core_from_driver` — embedding + AE SCF → ``CoreOrbitals``
- :func:`run_reconstruct_cvs_from_driver` — above + PW CVS-TDA (alias ``run_xas_reconstruct``)
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from casidapy.embed import (
    DEFAULT_EMBED_MODE,
    DEFAULT_GAUGE_ALIGN,
    DEFAULT_R_DAMP,
    DEFAULT_VLOC_SOURCE,
    build_ae_embedding_potential,
)

logger = logging.getLogger(__name__)

# Aufbau filling order for counting frozen / AE electrons
_SHELL_NELEC = {
    "1s": 2, "2s": 2, "2p": 6, "3s": 2, "3p": 6, "3d": 10,
    "4s": 2, "4p": 6, "4d": 10, "4f": 14, "5s": 2, "5p": 6,
    "5d": 10, "5f": 14, "6s": 2, "6p": 6,
}


# ---------------------------------------------------------------------------
# Shell bookkeeping
# ---------------------------------------------------------------------------

def frozen_shells_from_pp(atom_symbol, z_valence, z_atomic=None):
    """Shells replaced by the PP: ``n_frozen = Z − z_valence``, filled 1s → …"""
    from pyscf.data import elements

    z = int(z_atomic if z_atomic is not None else elements.charge(str(atom_symbol).capitalize()))
    n_frozen = int(round(z - float(z_valence)))
    shells, left = [], n_frozen
    for sh, ne in _SHELL_NELEC.items():
        if left <= 0:
            break
        shells.append(sh)
        left -= ne
    return tuple(shells), n_frozen


def _nelec_from_shells(shells):
    return sum(_SHELL_NELEC[str(s).strip().lower()] for s in shells)


def _shells_for_nelec(n_elec):
    shells, left = [], int(n_elec)
    for sh, ne in _SHELL_NELEC.items():
        if left <= 0:
            break
        shells.append(sh)
        left -= ne
    return shells


def _resolve_ae_config(
    sym,
    z,
    *,
    ae_electrons,
    shells,
    n_elec,
    z_valence,
    oxidation_state,
    oxidation_states,
):
    """Map ``ae_electrons`` mode → ``(mode, oxidation, shells, n_elec, charge)``."""
    from casidapy.xas.cvs import resolve_oxidation_state

    mode = str(ae_electrons).lower().strip()
    ox = None
    if mode in ("atomic", "neutral", "full", "all", "oxidation", "ox"):
        # Formal ion (O²⁻, Ti⁴⁺, …): charge = oxidation state
        mode = "atomic"
        ox = resolve_oxidation_state(sym, oxidation_state, oxidation_states)
        if shells is not None:
            shells = [str(s).strip().lower() for s in shells]
            n_elec = int(n_elec) if n_elec is not None else _nelec_from_shells(shells)
        else:
            n_elec = int(n_elec) if n_elec is not None else z - int(ox)
            shells = _shells_for_nelec(n_elec)
        charge = int(ox)
    elif mode in ("frozen", "pp", "core", "semicore"):
        # Only the electrons the PP removed (legacy)
        mode = "frozen"
        if shells is not None:
            shells = [str(s).strip().lower() for s in shells]
            n_elec = int(n_elec) if n_elec is not None else _nelec_from_shells(shells)
        else:
            shells, n_pp = frozen_shells_from_pp(sym, z_valence, z_atomic=z)
            n_elec = int(n_elec) if n_elec is not None else n_pp
        charge = z - n_elec
    else:
        raise ValueError(f"Unknown ae_electrons={ae_electrons!r}; use 'atomic' or 'frozen'")
    return mode, ox, list(shells), int(n_elec), int(charge)


# ---------------------------------------------------------------------------
# One-atom AE SCF in V_env
# ---------------------------------------------------------------------------

def _potential_ao_matrix(
    mol,
    v_field,
    grid,
    *,
    blksize=8000,
    use_gpu: bool = False,
    comm=None,
):
    """``V_μν = ∫ φ_μ(r) V_env(r) φ_ν(r) dr`` on the QE charge grid.

    Grid blocks can be MPI-distributed (``Allreduce`` of the AO matrix).
    Block GEMMs use CuPy when ``use_gpu=True``.
    """
    from pyscf.dft import numint
    from casidapy.utils.accel import (
        array_module,
        asnumpy,
        block_slices,
        mpi_allreduce_sum,
    )

    r = np.asarray(grid.r)
    coords = np.stack([r[0].ravel(), r[1].ravel(), r[2].ravel()], axis=1)
    v = np.asarray(v_field, float).ravel() * float(grid.dV)
    nao = mol.nao_nr()
    ngrid = coords.shape[0]
    xp = array_module(use_gpu)
    vmat = xp.zeros((nao, nao), dtype=float)

    for p0, p1 in block_slices(ngrid, blksize, comm):
        ao = numint.eval_ao(mol, coords[p0:p1])  # host (nb, nao)
        ao_x = xp.asarray(ao, dtype=float)
        v_x = xp.asarray(v[p0:p1], dtype=float)
        vmat += (ao_x.T * v_x) @ ao_x

    return mpi_allreduce_sum(comm, asnumpy(vmat))


def scf_ae_core_semicore(
    v_env,
    grid,
    atom_symbol,
    position_angstrom,
    *,
    basis="def2-tzvp",
    xc="pbe",
    shells=None,
    z_valence=None,
    n_elec=None,
    ae_electrons="atomic",
    oxidation_state=None,
    oxidation_states=None,
    spin=None,
    max_cycle=80,
    use_gpu: bool = False,
    comm=None,
    verbose=False,
):
    """One-atom AE RKS with frozen ``v_env`` added to the core Hamiltonian.

    ``ae_electrons="atomic"`` (default): formal oxidation state (O²⁻, Ti⁴⁺, …).
    ``ae_electrons="frozen"``: PP-removed electron count only (legacy).

    ``use_gpu``
        CuPy for the ``V_env`` AO projection; optional ``gpu4pyscf`` RKS.
    ``comm``
        MPI communicator for distributing AO-grid blocks of ``V_env``.
    """
    import warnings

    from pyscf import gto, dft, scf
    from pyscf.data import elements
    from casidapy.utils.accel import mpi_is_root

    sym = str(atom_symbol).capitalize()
    z = int(elements.charge(sym))
    mode, ox, shells, n_elec, charge = _resolve_ae_config(
        sym,
        z,
        ae_electrons=ae_electrons,
        shells=shells,
        n_elec=n_elec,
        z_valence=z_valence,
        oxidation_state=oxidation_state,
        oxidation_states=oxidation_states,
    )
    if spin not in (None, 0):
        logger.warning("scf_ae_core_semicore uses RKS (spin=0); ignoring spin=%s", spin)

    pos = [float(x) for x in position_angstrom]
    mol = gto.M(
        atom=f"{sym} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}",
        basis=basis,
        charge=int(charge),
        spin=0,
        unit="Angstrom",
        verbose=4 if verbose else 0,
    )

    # H = T − Z/r + V_env  (PySCF hcore already has T − Z/r)
    h0 = scf.hf.get_hcore(mol)
    v_ao = _potential_ao_matrix(
        mol, v_env, grid, use_gpu=use_gpu, comm=comm
    )

    used_gpu_scf = False
    mf = None
    if use_gpu:
        try:
            import cupy  # noqa: F401
            from gpu4pyscf.dft import rks as gpu_rks

            mf = gpu_rks.RKS(mol)
            used_gpu_scf = True
        except Exception as exc:
            warnings.warn(
                f"gpu4pyscf unavailable for AE core SCF ({exc}); using CPU RKS.",
                RuntimeWarning,
                stacklevel=2,
            )
    if mf is None:
        mf = dft.RKS(mol)

    mf.xc = xc
    mf.max_cycle = int(max_cycle)
    mf.verbose = 4 if verbose else 0
    mf.get_hcore = lambda *args, **kwargs: h0 + v_ao

    # Only root needs to run SCF when MPI-splitting AO work; other ranks
    # still built the same V_ao via Allreduce.
    if mpi_is_root(comm) or comm is None:
        mf.kernel()
        converged = bool(mf.converged)
        e_tot = float(mf.e_tot) if converged else None
        mo_energy = np.asarray(mf.mo_energy, float) if converged else None
        mo_coeff = np.asarray(mf.mo_coeff, float) if converged else None
        mo_occ = np.asarray(mf.mo_occ, float) if converged else None
    else:
        converged = False
        e_tot = None
        mo_energy = mo_coeff = mo_occ = None

    if comm is not None:
        from casidapy.utils.accel import mpi_bcast

        converged = bool(mpi_bcast(comm, converged, root=0))
        e_tot = mpi_bcast(comm, e_tot, root=0)
        mo_energy = mpi_bcast(comm, mo_energy, root=0)
        mo_coeff = mpi_bcast(comm, mo_coeff, root=0)
        mo_occ = mpi_bcast(comm, mo_occ, root=0)
        if not mpi_is_root(comm) and converged:
            # Rebuild a lightweight host mf with broadcast MO data
            mf = dft.RKS(mol)
            mf.xc = xc
            mf.converged = True
            mf.e_tot = e_tot
            mf.mo_energy = mo_energy
            mf.mo_coeff = mo_coeff
            mf.mo_occ = mo_occ
            mf.get_hcore = lambda *args, **kwargs: h0 + v_ao

    if not converged:
        raise RuntimeError(f"AE core SCF did not converge for {sym}")

    mf._casidapy_reconstruct = {
        "shells": list(shells),
        "charge": int(charge),
        "n_elec": int(n_elec),
        "ae_electrons": mode,
        "oxidation_state": None if ox is None else int(ox),
        "spin": 0,
        "z_valence": None if z_valence is None else float(z_valence),
        "z_atomic": z,
        "basis": basis,
        "xc": xc,
        "use_gpu_scf": bool(used_gpu_scf),
        "use_gpu_vao": bool(use_gpu),
        "mpi": comm is not None,
    }
    if verbose and mpi_is_root(comm):
        logger.info(
            "[reconstruct-scf] %s mode=%s ox=%s n_elec=%d charge=%d shells=%s "
            "gpu_scf=%s mpi=%s",
            sym,
            mode,
            ox,
            n_elec,
            charge,
            shells,
            used_gpu_scf,
            comm is not None,
        )
    return mf


# ---------------------------------------------------------------------------
# Core eigenvalue gauge (align to a reference / rigid shift)
# ---------------------------------------------------------------------------

def apply_core_gauge_shift(core, shift_ha):
    """Add ``shift_ha`` (Hartree) to every stored core eigenvalue."""
    from dataclasses import replace

    s = float(shift_ha)
    if abs(s) < 1e-15:
        return core
    e = np.asarray(core.energies, float) + s
    meta = dict(getattr(core, "meta", {}) or {})
    meta["core_gauge_shift_ha"] = float(meta.get("core_gauge_shift_ha", 0.0) + s)
    return replace(core, energies=e, meta=meta)


def align_core_to_reference(core, eps_ref_ha, *, index=0):
    """Shift so ``core.energies[index] == eps_ref_ha``; return ``(core, shift)``."""
    eps = np.asarray(core.energies, float)
    shift = float(eps_ref_ha) - float(eps[int(index)])
    core = apply_core_gauge_shift(core, shift)
    if getattr(core, "meta", None) is not None:
        core.meta["core_reference_ha"] = float(eps_ref_ha)
    return core, shift


# ---------------------------------------------------------------------------
# Driver-level orchestration
# ---------------------------------------------------------------------------

def _embed_kwargs(
    *,
    upf_path=None,
    verbose=False,
    embed_mode=DEFAULT_EMBED_MODE,
    gauge_align=DEFAULT_GAUGE_ALIGN,
    vhxc_scale="auto",
    vhxc_scale_alpha=None,
    vhxc_scale_min=None,
    vhxc_scale_max=None,
    r_damp=DEFAULT_R_DAMP,
    vloc_source=DEFAULT_VLOC_SOURCE,
    zero_embedding=False,
    use_gpu: bool = False,
    comm=None,
) -> dict[str, Any]:
    """Keyword dict forwarded to :func:`build_ae_embedding_potential`."""
    kw: dict[str, Any] = dict(
        upf_path=upf_path,
        verbose=verbose,
        embed_mode="zero" if zero_embedding else embed_mode,
        gauge_align=gauge_align,
        vhxc_scale=vhxc_scale,
        r_damp=float(r_damp),
        vloc_source=vloc_source,
        use_gpu=bool(use_gpu),
        comm=comm,
    )
    if vhxc_scale_alpha is not None:
        kw["vhxc_scale_alpha"] = float(vhxc_scale_alpha)
    if vhxc_scale_min is not None:
        kw["vhxc_scale_min"] = float(vhxc_scale_min)
    if vhxc_scale_max is not None:
        kw["vhxc_scale_max"] = float(vhxc_scale_max)
    return kw


def reconstruct_core_from_driver(
    driver,
    edge_atom,
    *,
    edge="K",
    shell=None,
    shells=None,
    basis="def2-tzvp",
    xc="pbe",
    upf_path=None,
    verbose=False,
    embed_mode=DEFAULT_EMBED_MODE,
    gauge_align=DEFAULT_GAUGE_ALIGN,
    vhxc_scale="auto",
    vhxc_scale_alpha=None,
    vhxc_scale_min=None,
    vhxc_scale_max=None,
    r_damp=DEFAULT_R_DAMP,
    vloc_source=DEFAULT_VLOC_SOURCE,
    ae_electrons="atomic",
    oxidation_state=None,
    oxidation_states=None,
    spin=None,
    zero_embedding=False,
    core_gauge_shift_ha=0.0,
    eps_core_reference_ha=None,
    core_reference_index=0,
    use_gpu: bool = False,
    comm=None,
):
    """QE embedding → one-atom AE SCF → :class:`~casidapy.xas.CoreOrbitals`.

    ``zero_embedding=True``
        Diagnostic with ``V_env = 0`` (isolated ion).
    ``eps_core_reference_ha``
        Align core levels to a full-AE GTO reference (removes rigid embed shift).
    ``core_gauge_shift_ha``
        Extra rigid shift after reconstruction.
    ``use_gpu`` / ``comm``
        CuPy for Hirshfeld / ``V_env`` AO projection; MPI atom/grid distribution.

    Returns ``(core, mf, v_env, grid)``.
    """
    from casidapy.xas.cvs import core_from_mf

    # 1) Frozen embedding potential
    v_env, grid, meta = build_ae_embedding_potential(
        driver,
        edge_atom,
        **_embed_kwargs(
            upf_path=upf_path,
            verbose=verbose,
            embed_mode=embed_mode,
            gauge_align=gauge_align,
            vhxc_scale=vhxc_scale,
            vhxc_scale_alpha=vhxc_scale_alpha,
            vhxc_scale_min=vhxc_scale_min,
            vhxc_scale_max=vhxc_scale_max,
            r_damp=r_damp,
            vloc_source=vloc_source,
            zero_embedding=zero_embedding,
            use_gpu=use_gpu,
            comm=comm,
        ),
    )

    # 2) One-atom AE SCF at the edge nuclear position
    atoms = driver.get_ase_atoms()
    ia = int(edge_atom)
    sym = atoms.get_chemical_symbols()[ia]
    mf = scf_ae_core_semicore(
        v_env,
        grid,
        sym,
        atoms.get_positions()[ia],
        basis=basis,
        xc=xc,
        shells=shells,
        z_valence=float(meta["z_valence"]),
        ae_electrons=ae_electrons,
        oxidation_state=oxidation_state,
        oxidation_states=oxidation_states,
        spin=spin,
        use_gpu=use_gpu,
        comm=comm,
        verbose=verbose,
    )

    # 3) Select core MOs; optional reference / gauge alignment
    core = core_from_mf(mf, edge=edge, shell=shell, edge_atom_indices=[0], verbose=verbose)
    ref_shift = 0.0
    if eps_core_reference_ha is not None:
        core, ref_shift = align_core_to_reference(
            core, eps_core_reference_ha, index=core_reference_index
        )
    if core_gauge_shift_ha:
        core = apply_core_gauge_shift(core, core_gauge_shift_ha)

    rec = mf._casidapy_reconstruct
    meta = dict(meta)
    meta.update(
        ae_electrons=rec["ae_electrons"],
        oxidation_state=rec.get("oxidation_state"),
        scf_charge=int(rec["charge"]),
        scf_spin=int(rec["spin"]),
        n_elec=int(rec["n_elec"]),
        core_reference_shift_ha=float(ref_shift),
        core_gauge_shift_ha=float(core_gauge_shift_ha),
        use_gpu=bool(use_gpu),
        mpi=comm is not None,
    )
    core.meta.update(
        reconstruct=True,
        reconstruct_meta=meta,
        reconstruct_shells=list(rec["shells"]),
        z_valence=float(meta["z_valence"]),
        n_frozen=int(rec["n_elec"]),
        ae_electrons=rec["ae_electrons"],
    )
    return core, mf, v_env, grid


def _core_charge_centroid(kernel, ref_index=0):
    """Charge centroid ``∫ r |ψ_core|² dV`` of an injected core orbital (Bohr).

    Computed in the kernel's own grid frame (``kernel.grid.r``), so it is
    directly usable as the length-gauge dipole origin with no unit/frame
    conversion. The cluster is centered in vacuum, so no minimum-image wrap is
    needed.
    """
    psi_occ = getattr(kernel, "_psi_occ", None)
    if not psi_occ:
        raise RuntimeError("kernel has no injected core orbitals")
    idx = int(ref_index) if 0 <= int(ref_index) < len(psi_occ) else 0
    psi = np.asarray(psi_occ[idx]).ravel()
    w = psi * psi
    dV = float(kernel.grid.dV)
    tot = float(w.sum()) * dV
    if not np.isfinite(tot) or tot <= 0.0:
        raise ValueError("core orbital has non-positive norm on the grid")
    r = kernel.grid.r
    return tuple(
        float((np.asarray(r[a]).ravel() * w).sum()) * dV / tot for a in range(3)
    )


def run_reconstruct_cvs_from_driver(
    driver,
    edge_atom,
    *,
    edge="K",
    shell=None,
    shells=None,
    basis="def2-tzvp",
    xc="pbe",
    pw_xc="PBE",
    n_virt=40,
    n_states=40,
    virt_window_ev=None,
    n_virt_active=None,
    use_gpu=False,
    use_gpu_pw=None,
    comm=None,
    upf_path=None,
    verbose=False,
    embed_mode=DEFAULT_EMBED_MODE,
    gauge_align=DEFAULT_GAUGE_ALIGN,
    vhxc_scale="auto",
    vhxc_scale_alpha=None,
    vhxc_scale_min=None,
    vhxc_scale_max=None,
    r_damp=DEFAULT_R_DAMP,
    vloc_source=DEFAULT_VLOC_SOURCE,
    ae_electrons="atomic",
    oxidation_state=None,
    oxidation_states=None,
    spin=None,
    zero_embedding=False,
    core_gauge_shift_ha=0.0,
    eps_core_reference_ha=None,
    core_reference_index=0,
):
    """Reconstruct AE core → inject into PW virtuals → CVS-TDA.

    ``use_gpu``
        CuPy for Hirshfeld / ``V_env`` AO work / (optionally) PW kernel.
    ``use_gpu_pw``
        CuPy for the PW Casida kernel. Defaults to ``use_gpu``. Set ``False``
        when the virtual manifold is too large for GPU VRAM while still using
        GPU for embedding / AO work.
    ``n_virt_active``
        Optional. Energy-stride the QE virtual pool (``n_virt`` bands) down to
        this many active virtuals for Casida. Opt-in for large continuum
        subspaces; ``None`` keeps every selected virtual.
    ``comm``
        MPI communicator for Hirshfeld atom loop and AO-grid blocks.

    Returns ``(results, core, kernel, mf)``. Exposed as ``run_xas_reconstruct``.
    """
    from casidapy.casida_api import CasidaOptions
    from casidapy.xas.cvs import (
        build_pw_kernel_from_qepy,
        core_mos_to_pw_fields,
        inject_core_orbitals,
        run_cvs_tda,
    )

    if use_gpu_pw is None:
        use_gpu_pw = bool(use_gpu)

    core, mf, _, _ = reconstruct_core_from_driver(
        driver,
        edge_atom,
        edge=edge,
        shell=shell,
        shells=shells,
        basis=basis,
        xc=xc,
        upf_path=upf_path,
        verbose=verbose,
        embed_mode=embed_mode,
        gauge_align=gauge_align,
        vhxc_scale=vhxc_scale,
        vhxc_scale_alpha=vhxc_scale_alpha,
        vhxc_scale_min=vhxc_scale_min,
        vhxc_scale_max=vhxc_scale_max,
        r_damp=r_damp,
        vloc_source=vloc_source,
        ae_electrons=ae_electrons,
        oxidation_state=oxidation_state,
        oxidation_states=oxidation_states,
        spin=spin,
        zero_embedding=zero_embedding,
        core_gauge_shift_ha=core_gauge_shift_ha,
        eps_core_reference_ha=eps_core_reference_ha,
        core_reference_index=core_reference_index,
        use_gpu=use_gpu,
        comm=comm,
    )
    kernel, grid = build_pw_kernel_from_qepy(
        driver,
        n_virt=n_virt,
        use_gpu=bool(use_gpu_pw),
        xc=pw_xc,
        casida_uspp=False,
        unocc_window_ev=virt_window_ev,
        n_virt_active=n_virt_active,
        verbose=verbose,
    )
    inject_core_orbitals(
        kernel,
        core.energies,
        core_mos_to_pw_fields(core, grid, use_gpu=use_gpu, comm=comm),
        use_gpu=use_gpu,
    )
    # Center the length-gauge dipole origin on the core orbital so ``(r - R0)`` is
    # small where the tight core density lives. Origin choice is formally
    # irrelevant for orthogonal core→virtual pairs, but on a grid with large ω and
    # |r| ~ box size it stabilizes the transition dipoles against orthogonality
    # residuals. Falls back silently to the box corner on any failure.
    try:
        kernel.dipole_origin = _core_charge_centroid(kernel, core_reference_index)
        if verbose:
            logger.info("dipole origin set to core centroid %s (Bohr)", kernel.dipole_origin)
    except Exception:  # pragma: no cover - diagnostic only
        logger.warning("could not center dipole origin on core; using box corner", exc_info=True)
    opts = CasidaOptions(
        n_occ=kernel.n_occ,
        n_unocc=kernel.n_unocc,
        n_states=min(n_states, kernel.n_trans),
        basis="pw",
        tda=True,
        matrix_free=True,
        solver_method="davidson",
        solver_tol=1e-6,
        solver_maxiter=200,
        xc=pw_xc,
        use_gpu=bool(use_gpu_pw),
    )
    return run_cvs_tda(kernel, opts), core, kernel, mf
