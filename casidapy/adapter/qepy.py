from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from dftpy.field import DirectField
from dftpy.grid import DirectGrid
from dftpy.utils import grid_map_data

from casidapy.casida_api import CasidaInputs, CasidaOptions
from casidapy.utils.casida_utils import (
    _bcast_psi_stack,
    grid_slab_bounds,
    mpi_comm_rank,
    mpi_comm_size,
    scatter_real_band,
)


def _casida_mpi_comm(driver):
    """MPI communicator for Casida packing / density broadcast.

    QEpy ``Driver`` is often constructed with ``comm=None`` even under
    ``mpirun``. In that case ``mpi_comm_size(None)==1`` and every rank would
    incorrectly take the rank-0 pack path. Fall back to ``MPI.COMM_WORLD``.
    """
    comm = getattr(driver, "comm", None)
    if comm is not None and mpi_comm_size(comm) > 1:
        return comm
    try:
        from mpi4py import MPI

        world = MPI.COMM_WORLD
        if world.Get_size() > 1:
            return world
    except Exception:
        pass
    return comm


def _qe_density_root_rank(driver, comm) -> int:
    """Rank that owns the full gathered density (QE ``dfftp`` root)."""
    try:
        mype = int(driver.embed.dfftp.mype)
    except Exception:
        mype = -1
    # Each rank reports its world rank if it is the QE FFT root; else -1.
    mine = mpi_comm_rank(comm) if mype == 0 else -1
    if hasattr(comm, "allreduce"):
        try:
            from mpi4py import MPI

            return int(comm.allreduce(mine, op=MPI.MAX))
        except Exception:
            pass
    return 0 if mine >= 0 else 0

_HA_TO_EV = 27.211386245988


def subsample_virtuals_by_energy(
    unocc_eigs: np.ndarray,
    psi_unocc: Sequence,
    n_keep: int,
) -> Tuple[np.ndarray, list, np.ndarray]:
    """Energy-strided subset of a virtual manifold (opt-in for large pools).

    Picks ``n_keep`` bands whose eigenvalues are nearest to a uniform grid
    spanning ``[ε_lo, ε_hi]``. Always covers the full energy span of the pool.
    No-op when ``n_keep >= len(unocc_eigs)``.

    Returns ``(eigs_keep, psi_keep, indices)`` with ``indices`` into the input
    arrays (ascending energy order of the kept set).
    """
    e = np.asarray(unocc_eigs, dtype=float).ravel()
    n = int(e.size)
    if n == 0:
        return e.copy(), list(psi_unocc), np.zeros(0, dtype=int)
    n_keep = int(n_keep)
    if n_keep <= 0:
        raise ValueError(f"n_keep must be > 0, got {n_keep}")
    if n_keep >= n:
        return e.copy(), list(psi_unocc), np.arange(n, dtype=int)

    order = np.argsort(e, kind="mergesort")
    e_sorted = e[order]
    targets = np.linspace(float(e_sorted[0]), float(e_sorted[-1]), n_keep)
    used = np.zeros(n, dtype=bool)
    chosen_sorted: list[int] = []
    for t in targets:
        dists = np.abs(e_sorted - t)
        dists[used] = np.inf
        best = int(np.argmin(dists))
        used[best] = True
        chosen_sorted.append(best)

    # Restore input-array indices; sort by energy for a stable active space.
    idx = np.array(sorted(int(order[j]) for j in chosen_sorted), dtype=int)
    if idx.size < n_keep:
        remaining = [int(i) for i in order if i not in set(idx.tolist())]
        idx = np.array(sorted(idx.tolist() + remaining[: n_keep - idx.size]), dtype=int)

    e_keep = e[idx].copy()
    psi_keep = [psi_unocc[i] for i in idx]
    return e_keep, psi_keep, idx


def subsample_virtual_indices_by_energy(
    unocc_eigs: np.ndarray,
    n_keep: int,
) -> np.ndarray:
    """Energy-strided virtual indices without loading orbitals."""
    e = np.asarray(unocc_eigs, dtype=float).ravel()
    n = int(e.size)
    if n == 0:
        return np.zeros(0, dtype=int)
    n_keep = int(n_keep)
    if n_keep <= 0:
        raise ValueError(f"n_keep must be > 0, got {n_keep}")
    if n_keep >= n:
        return np.arange(n, dtype=int)
    _, _, idx = subsample_virtuals_by_energy(e, list(range(n)), n_keep)
    return np.asarray(idx, dtype=int)


def plan_pw_band_indices(
    eig_ha: np.ndarray,
    *,
    n_total_occ: int,
    n_occ: int,
    n_unocc: int,
    n_orb: Optional[int] = None,
    unocc_window_ev: Optional[float] = None,
    unocc_subsample: Optional[int] = None,
    virt_global_indices: Optional[Sequence[int]] = None,
    verbose: bool = False,
) -> Tuple[list[int], list[int], dict]:
    """Plan which QE bands to load before charge-grid mapping.

    Applies the same occupied/virtual window, optional energy window, and
    optional energy stride as :func:`slice_active_space`, but returns global
    band indices only (no orbitals). Use this to subsample or core-preselect
    virtuals before the expensive smooth→charge interpolation.

    Returns
    -------
    occ_indices, virt_indices, meta
        ``meta`` includes ``virt_eigs`` for the planned virtual pool.
    """
    n_total_occ = int(n_total_occ)
    n_occ = int(n_occ)
    n_unocc = int(n_unocc)
    eigs_arr = np.asarray(eig_ha, dtype=float).ravel()
    if n_orb is None:
        n_orb = int(eigs_arr.size)
    if eigs_arr.size != n_orb:
        raise ValueError(f"len(eig_ha)={eigs_arr.size} != n_orb={n_orb}")

    i0 = max(0, n_total_occ - n_occ)
    if i0 >= n_total_occ:
        raise ValueError(
            f"invalid occupied window: [{i0}, {n_total_occ}) with n_orb={n_orb}"
        )

    n_u_avail = n_orb - n_total_occ
    if n_u_avail < 0:
        raise ValueError(f"n_total_occ={n_total_occ} exceeds n_orb={n_orb}")
    nu = min(n_unocc, n_u_avail)
    if nu < n_unocc and verbose:
        print(
            f"  WARNING: requested {n_unocc} unoccupied orbitals but only "
            f"{n_u_avail} available; reducing to {nu}."
        )
    u1 = n_total_occ + nu

    occ_indices = list(range(i0, n_total_occ))
    if virt_global_indices is not None:
        virt_indices = [int(i) for i in virt_global_indices]
        bad = [i for i in virt_indices if i < n_total_occ or i >= u1]
        if bad:
            raise ValueError(
                f"virt_global_indices {bad} outside virtual pool "
                f"[{n_total_occ}, {u1})"
            )
    else:
        virt_indices = list(range(n_total_occ, u1))

    virt_eigs = eigs_arr[virt_indices].copy()

    if unocc_window_ev is not None and virt_eigs.size:
        window_ha = float(unocc_window_ev) / _HA_TO_EV
        keep = virt_eigs <= float(virt_eigs[0]) + window_ha
        n_keep = int(np.count_nonzero(keep))
        if n_keep < virt_eigs.size and verbose:
            print(
                f"  virtual energy window: keeping {n_keep}/{virt_eigs.size} "
                f"virtuals within {unocc_window_ev:.2f} eV of the lowest virtual."
            )
        virt_indices = [virt_indices[i] for i, k in enumerate(keep) if k]
        virt_eigs = virt_eigs[keep]

    if unocc_subsample is not None and virt_eigs.size:
        n_sub = int(unocc_subsample)
        if n_sub <= 0:
            raise ValueError(f"unocc_subsample must be > 0, got {n_sub}")
        n_pool = int(virt_eigs.size)
        if n_sub < n_pool:
            span_before = float(virt_eigs[-1] - virt_eigs[0]) * _HA_TO_EV
            idx_local = subsample_virtual_indices_by_energy(virt_eigs, n_sub)
            virt_indices = [virt_indices[int(i)] for i in idx_local]
            virt_eigs = virt_eigs[idx_local]
            if verbose:
                span_after = float(virt_eigs[-1] - virt_eigs[0]) * _HA_TO_EV
                print(
                    f"  virtual subsample: {n_sub}/{n_pool} bands by energy stride "
                    f"(span {span_before:.2f} → {span_after:.2f} eV)"
                )

    if verbose and virt_eigs.size:
        span_ev = float(virt_eigs[-1] - virt_eigs[0]) * _HA_TO_EV
        print(
            f"  planned virtual pool: {virt_eigs.size} bands span {span_ev:.2f} eV "
            f"(ε_lo={virt_eigs[0]:.4f} Ha, ε_hi={virt_eigs[-1]:.4f} Ha)"
        )

    meta = {
        "virt_eigs": virt_eigs,
        "n_total_occ": n_total_occ,
        "n_occ": n_occ,
        "n_unocc": nu,
    }
    return occ_indices, virt_indices, meta


def slice_active_space(
    eigs: np.ndarray,
    psi_all: Sequence,
    n_occ: int,
    n_unocc: Optional[int] = None,
    n_total_occ: Optional[int] = None,
    unocc_window_ev: Optional[float] = None,
    unocc_subsample: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """Slice full KS arrays into active occupied/unoccupied sets.

    Returns (occ_eigs, unocc_eigs, psi_occ, psi_unocc).
    This is the single source of truth for active-space windowing,
    used by both the QEpy adapter path and the generic CLI runner.

    ``unocc_window_ev`` optionally restricts the virtual manifold (after the
    ``n_unocc`` count slice) to bands within that many eV of the lowest virtual.
    A diagnostic on the actual virtual energy span is printed when ``verbose``;
    this surfaces the common failure mode of a fixed band count spanning only a
    few eV in a large vacuum box (far too narrow for a core-level XAS range).

    ``unocc_subsample`` (opt-in, for large virtual pools)
        After the contiguous ``n_unocc`` / window selection, keep only this many
        virtuals by uniform energy striding across ``[ε_lo, ε_hi]``. Use when
        QE ``nbnd`` is large (wide continuum coverage) but Casida cannot afford
        every band. Ignored when ``None`` or when the pool is already ≤ this
        size — default paths are unchanged.
    """
    if n_occ <= 0:
        raise ValueError(f"n_occ must be > 0, got {n_occ}")
    if n_unocc is not None and n_unocc <= 0:
        raise ValueError(f"n_unocc must be > 0 when given, got {n_unocc}")

    n_orb = len(psi_all)
    eigs_arr = np.asarray(eigs, dtype=float)
    if eigs_arr.shape[0] != n_orb:
        raise ValueError(f"len(eigs)={eigs_arr.shape[0]} != len(psi_all)={n_orb}")

    if n_total_occ is None:
        n_total_occ = n_occ
        i0, i1 = 0, n_occ
    else:
        i1 = n_total_occ
        i0 = max(0, n_total_occ - n_occ)

    if i1 > n_orb or i0 >= i1:
        raise ValueError(f"invalid occupied window: [{i0}, {i1}) with n_orb={n_orb}")

    n_u_avail = n_orb - n_total_occ
    if n_u_avail < 0:
        raise ValueError(f"n_total_occ={n_total_occ} exceeds n_orb={n_orb}")

    if n_unocc is None:
        u1 = n_orb
    else:
        nu = n_unocc
        if nu > n_u_avail:
            if verbose:
                print(
                    f"  WARNING: requested {nu} unoccupied orbitals but only {n_u_avail} available."
                )
                print(f"           reducing n_unocc from {nu} to {n_u_avail}.")
            nu = n_u_avail
        u1 = n_total_occ + nu

    occ_eigs = eigs_arr[i0:i1].copy()
    unocc_eigs = eigs_arr[n_total_occ:u1].copy()
    psi_occ = list(psi_all[i0:i1])
    # psi_all may contain None outside the loaded window; only index live slots
    psi_unocc = []
    for i in range(n_total_occ, u1):
        p = psi_all[i]
        if p is None:
            raise ValueError(
                f"psi_all[{i}] is None while slicing virtuals "
                f"[{n_total_occ}, {u1}); load that band window first."
            )
        psi_unocc.append(p)

    # Optional energy-window restriction of the virtual manifold. Eigenvalues in a
    # contiguous QE slice are ascending, so this keeps a low-energy prefix.
    if unocc_window_ev is not None and unocc_eigs.size:
        window_ha = float(unocc_window_ev) / _HA_TO_EV
        keep = unocc_eigs <= float(unocc_eigs[0]) + window_ha
        n_keep = int(np.count_nonzero(keep))
        if n_keep < unocc_eigs.size and verbose:
            print(
                f"  virtual energy window: keeping {n_keep}/{unocc_eigs.size} "
                f"virtuals within {unocc_window_ev:.2f} eV of the lowest virtual."
            )
        unocc_eigs = unocc_eigs[keep]
        psi_unocc = [p for p, k in zip(psi_unocc, keep) if k]

    # Opt-in energy stride for large continuum pools (XAS / big vacuum cells).
    if unocc_subsample is not None and unocc_eigs.size:
        n_sub = int(unocc_subsample)
        if n_sub <= 0:
            raise ValueError(f"unocc_subsample must be > 0, got {n_sub}")
        n_pool = int(unocc_eigs.size)
        if n_sub < n_pool:
            span_before = float(unocc_eigs[-1] - unocc_eigs[0]) * _HA_TO_EV
            unocc_eigs, psi_unocc, _idx = subsample_virtuals_by_energy(
                unocc_eigs, psi_unocc, n_sub
            )
            if verbose:
                span_after = float(unocc_eigs[-1] - unocc_eigs[0]) * _HA_TO_EV
                print(
                    f"  virtual subsample: {n_sub}/{n_pool} bands by energy stride "
                    f"(span {span_before:.2f} → {span_after:.2f} eV)"
                )

    # Diagnostic: how wide is the virtual manifold we actually kept? A narrow span
    # relative to the intended spectral range means nbnd is too small (dense
    # vacuum/box states in a large cell), which compresses the XAS envelope.
    if verbose and unocc_eigs.size:
        span_ev = float(unocc_eigs[-1] - unocc_eigs[0]) * _HA_TO_EV
        print(
            f"  virtual manifold: {unocc_eigs.size} bands span {span_ev:.2f} eV "
            f"(ε_lo={unocc_eigs[0]:.4f} Ha, ε_hi={unocc_eigs[-1]:.4f} Ha)"
        )
        if unocc_window_ev is not None and span_ev + 1e-6 < float(unocc_window_ev):
            print(
                f"  WARNING: available conduction bands span only {span_ev:.2f} eV "
                f"(< requested window {unocc_window_ev:.2f} eV). Increase nbnd in the "
                f"QE SCF to cover a wider XAS range."
            )

    return occ_eigs, unocc_eigs, psi_occ, psi_unocc


def slice_active_space_by_indices(
    eigs: np.ndarray,
    psi_all: Sequence,
    occ_indices: Sequence[int],
    virt_indices: Sequence[int],
    *,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """Build active occupied/virtual sets from explicit global band indices."""
    occ_indices = [int(i) for i in occ_indices]
    virt_indices = [int(i) for i in virt_indices]
    eigs_arr = np.asarray(eigs, dtype=float)
    occ_eigs = eigs_arr[occ_indices].copy()
    unocc_eigs = eigs_arr[virt_indices].copy()
    psi_occ, psi_unocc = [], []
    for i in occ_indices:
        p = psi_all[i]
        if p is None:
            raise ValueError(f"psi_all[{i}] is None (occupied band not loaded)")
        psi_occ.append(p)
    for i in virt_indices:
        p = psi_all[i]
        if p is None:
            raise ValueError(f"psi_all[{i}] is None (virtual band not loaded)")
        psi_unocc.append(p)
    if verbose and unocc_eigs.size:
        span_ev = float(unocc_eigs[-1] - unocc_eigs[0]) * _HA_TO_EV
        print(
            f"  virtual manifold: {unocc_eigs.size} bands span {span_ev:.2f} eV "
            f"(ε_lo={unocc_eigs[0]:.4f} Ha, ε_hi={unocc_eigs[-1]:.4f} Ha)"
        )
    return occ_eigs, unocc_eigs, psi_occ, psi_unocc


def _parse_uspp_map_from_inputfile(inputfile: str) -> Dict[str, str]:
    """Parse pseudo_dir + ATOMIC_SPECIES from a QE pw.x input file."""
    pseudo_dir = "./"
    species_lines = []
    in_atomic_species = False

    with open(inputfile, "r") as f:
        for line in f:
            stripped = line.strip()

            if "pseudo_dir" in stripped.lower():
                parts = stripped.split("=", 1)
                if len(parts) == 2:
                    val = parts[1].strip().rstrip(",").strip("'\"")
                    pseudo_dir = val

            if stripped.upper().startswith("ATOMIC_SPECIES"):
                in_atomic_species = True
                continue

            if in_atomic_species:
                if (stripped == "" or stripped.startswith("&") or
                    stripped.upper().startswith(
                        ("ATOMIC_POSITIONS", "K_POINTS", "CELL_PARAMETERS"))):
                    in_atomic_species = False
                    continue
                species_lines.append(stripped)

    input_dir = os.path.dirname(os.path.abspath(inputfile))
    if not os.path.isabs(pseudo_dir):
        pseudo_dir = os.path.join(input_dir, pseudo_dir)

    uspp_map = {}
    for sl in species_lines:
        tokens = sl.split()
        if len(tokens) >= 3:
            uspp_map[tokens[0]] = os.path.join(pseudo_dir, tokens[2])

    return uspp_map


def build_uspp_map_from_driver(driver) -> Dict[str, str]:
    """Extract element -> UPF path mapping from a QEpy driver.

    Tries driver.qe_options first (the dict passed at init). If that
    yields nothing, falls back to parsing driver.inputfile on disk.

    Parameters
    ----------
    driver : qepy.driver.Driver
        Initialized QEpy driver.

    Returns
    -------
    uspp_map : dict
        Mapping of element symbol -> path to UPF file.
        Example: {"Al": "./Al_ONCV_PBE-1.2.upf"}
    """
    qe_options = getattr(driver, "qe_options", None)
    if qe_options:
        pseudo_dir = "./"
        control = qe_options.get("&control", {})
        if "pseudo_dir" in control:
            pseudo_dir = control["pseudo_dir"].strip("'\"")

        species_lines = qe_options.get("atomic_species", [])

        uspp_map = {}
        for line in species_lines:
            # QEpy/ASE may store atomic_species as strings OR as
            # [symbol, mass, upf] lists (see run_full_lr_tddft.py).
            if isinstance(line, (list, tuple)):
                tokens = [str(x) for x in line]
            else:
                tokens = str(line).split()
            if len(tokens) >= 3:
                uspp_map[tokens[0]] = os.path.join(pseudo_dir, tokens[2])

        if uspp_map:
            return uspp_map

    inputfile = getattr(driver, "inputfile", None)
    if inputfile and os.path.isfile(inputfile):
        return _parse_uspp_map_from_inputfile(inputfile)

    return {}


def _qe_wavefunction_grid(driver, ions) -> DirectGrid:
    """Full smooth FFT grid for KS orbitals (serial DirectGrid, no MPI domain decomp).

    Casida replicates orbitals on every rank; passing ``comm=driver.comm`` would
    domain-decompose the grid and deadlocks if only one rank builds it.
    """
    nrs = np.zeros(3, dtype=np.int32)
    driver.qepy_pw.qepy_mod.qepy_get_grid_smooth(nrs, gather=True)
    return DirectGrid(lattice=ions.cell, nr=list(nrs))


def _serial_charge_grid(driver, ions, target_grid=None) -> DirectGrid:
    """Full QE charge grid as a serial DirectGrid (no MPI domain decomp)."""
    if target_grid is not None:
        return target_grid
    nr_rho = list(driver.get_number_of_grid_points(gather=True))
    return DirectGrid(lattice=ions.cell, nr=nr_rho)


def qe_smooth_grid(driver, atoms=None) -> DirectGrid:
    """QE smooth (FFT) grid for KS orbitals.

    ``atoms`` is accepted for call-site symmetry with the other grid helpers and
    is unused; the grid is derived from the driver's own ions.
    """
    return _qe_wavefunction_grid(driver, driver.get_dftpy_ions())


def qe_charge_grid(driver, atoms=None) -> DirectGrid:
    """QE charge (density) grid as a serial DirectGrid (no MPI domain decomp).

    ``atoms`` is accepted for call-site symmetry and is unused.
    """
    return _serial_charge_grid(driver, driver.get_dftpy_ions())


def load_qe_bands_on_grid(
    driver,
    band_indices: Sequence[int],
    grid: DirectGrid,
    *,
    map_to_charge: bool = False,
    grid_charge: Optional[DirectGrid] = None,
) -> list:
    """Load QE bands onto ``grid`` without building the full band stack.

  When ``map_to_charge=True``, interpolate from the smooth FFT grid to the
  charge grid (same as :func:`_pack_on_density_grid` per band).
    """
    psi_raw = driver.get_wave_function()
    ions = driver.get_dftpy_ions()
    grid_wf = _qe_wavefunction_grid(driver, ions)
    if grid_charge is None and map_to_charge:
        grid_charge = qe_charge_grid(driver)
    out = []
    for ib in band_indices:
        ps = psi_raw[int(ib)]
        pf = driver.data2field(data=ps, grid=grid_wf)
        if not hasattr(pf, "grid"):
            pf = DirectField(grid=grid_wf, data=np.asarray(pf).ravel())
        if map_to_charge:
            if grid_charge is None:
                raise ValueError("grid_charge required when map_to_charge=True")
            if list(grid_wf.nr) != list(grid_charge.nr):
                pf = DirectField(grid=grid_charge, data=grid_map_data(pf, grid=grid_charge))
            else:
                pf = DirectField(grid=grid_charge, data=np.asarray(pf))
        elif list(grid_wf.nr) != list(grid.nr):
            pf = DirectField(grid=grid, data=grid_map_data(pf, grid=grid))
        else:
            pf = DirectField(grid=grid, data=np.asarray(pf))
        out.append(pf)
    return out


def preselect_virtual_bands_on_smooth_grid(
    driver,
    core_orbitals,
    virt_global_indices: Sequence[int],
    virt_eigs: np.ndarray,
    *,
    select_mode: str = "dipole",
    n_keep: Optional[int] = None,
    min_abs_overlap: float = 0.0,
    min_abs_dipole: float = 0.0,
    origin=None,
    use_gpu: bool = False,
    verbose: bool = False,
) -> Tuple[list[int], np.ndarray]:
    """Rank virtuals on the QE smooth grid before charge-grid mapping.

    Returns ``(kept_global_indices, local_kept_indices)``.

    .. warning::
       Not suitable for reconstructed **AE core** orbitals. A 1s cusp is only a
       few tenths of a Bohr wide, so it is badly under-resolved on the smooth
       (``ecutwfc``) grid; ``|<core|r-R0|virt>|`` then collapses by ~2 orders of
       magnitude and ranks the wrong bands. XAS core-virtual pruning must run on
       the dense charge (``ecutrho``) grid — see
       :func:`casidapy.xas.prune_kernel_virtuals_by_core_dipole`. This helper is
       only for smooth valence-like occupied states.
    """
    from casidapy.xas.dipole import select_virtuals_by_core_dipole
    from casidapy.xas.diagnostics import select_virtuals_by_core_overlap

    virt_global_indices = [int(i) for i in virt_global_indices]
    eigs = np.asarray(virt_eigs, dtype=float).ravel()
    if eigs.size != len(virt_global_indices):
        raise ValueError(
            f"len(virt_eigs)={eigs.size} != len(virt_global_indices)="
            f"{len(virt_global_indices)}"
        )
    if eigs.size == 0:
        return [], np.zeros(0, dtype=int)

    from casidapy.utils.casida_utils import normalize_wavefunctions

    grid_smooth = qe_smooth_grid(driver)
    virtual_orbitals = load_qe_bands_on_grid(
        driver, virt_global_indices, grid_smooth, map_to_charge=False
    )
    virtual_orbitals = normalize_wavefunctions(virtual_orbitals, grid_smooth)
    mode = str(select_mode).lower().strip()
    if mode == "dipole":
        if n_keep is None and min_abs_dipole <= 0.0:
            return virt_global_indices, np.arange(eigs.size, dtype=int)
        _, _, _, idx_local = select_virtuals_by_core_dipole(
            eigs,
            virtual_orbitals,
            core_orbitals,
            grid=grid_smooth,
            origin=origin,
            min_abs_dipole=float(min_abs_dipole),
            n_keep=n_keep,
            use_gpu=use_gpu,
            verbose=verbose,
        )
    elif mode == "overlap":
        if n_keep is None and min_abs_overlap <= 0.0:
            return virt_global_indices, np.arange(eigs.size, dtype=int)
        _, _, _, idx_local = select_virtuals_by_core_overlap(
            eigs,
            virtual_orbitals,
            core_orbitals,
            grid=grid_smooth,
            min_abs_overlap=float(min_abs_overlap),
            n_keep=n_keep,
            use_gpu=use_gpu,
            verbose=verbose,
        )
    else:
        raise ValueError(f"select_mode must be 'dipole' or 'overlap', got {select_mode!r}")

    idx_local = np.asarray(idx_local, dtype=int)
    kept_global = [virt_global_indices[int(i)] for i in idx_local]
    if verbose:
        print(
            f"  smooth-grid {mode} preselect: kept {len(kept_global)}/{eigs.size} "
            f"virtuals before charge-grid map"
        )
    return kept_global, idx_local


def _pack_on_density_grid(
    driver,
    atoms,
    ions,
    psi_raw,
    target_grid=None,
    band_indices: Optional[Sequence[int]] = None,
    rho_data=None,
    *,
    grid_wf: Optional[DirectGrid] = None,
    grid_rho: Optional[DirectGrid] = None,
    progress: bool = False,
):
    """Place rho and psi on the QE charge (density) grid.

    ``band_indices``
        If given, only map these QE band indices onto the charge grid.
    ``rho_data``
        Optional pre-fetched / broadcast density (full charge-grid array).
    ``grid_wf`` / ``grid_rho``
        Optional pre-built **serial** grids. When packing on a single MPI rank
        only, callers must query QE grid sizes on *all* ranks first, then pass
        these in so this function never issues QE collectives alone.
    """
    if grid_rho is None:
        grid_rho = _serial_charge_grid(driver, ions, target_grid=target_grid)
    elif target_grid is not None and list(target_grid.nr) != list(grid_rho.nr):
        print(
            f"WARNING: target_grid.nr={target_grid.nr} != QE charge grid {list(grid_rho.nr)}; "
            f"using QE charge grid for Casida."
        )

    if grid_wf is None:
        grid_wf = _qe_wavefunction_grid(driver, ions)

    if list(grid_wf.nr) != list(grid_rho.nr):
        print(
            f"Mapping wavefunctions from smooth grid {list(grid_wf.nr)} "
            f"to charge grid {list(grid_rho.nr)}",
            flush=True,
        )

    indices = (
        list(range(len(psi_raw))) if band_indices is None else list(band_indices)
    )
    remap = list(grid_wf.nr) != list(grid_rho.nr)
    psi_array = None
    expect = int(np.prod(grid_wf.nr))
    for k, ib in enumerate(indices):
        raw = np.asarray(psi_raw[ib]).ravel()
        if raw.size != expect:
            raise ValueError(
                f"Wavefunction band {ib} has size {raw.size}, expected {expect} "
                f"on smooth grid {list(grid_wf.nr)}. get_wave_function(gather=True) "
                "must return the full smooth grid on every MPI rank (rebuild QEpy "
                "with mp_gather_complex_smooth broadcast)."
            )
        # Pure DFTpy path — avoid driver.data2field (may touch QE MPI state).
        pf = DirectField(grid=grid_wf, data=raw)
        if remap:
            pf = DirectField(grid=grid_rho, data=grid_map_data(pf, grid=grid_rho))
        block = np.asarray(pf)
        if psi_array is None:
            psi_array = np.empty((len(indices),) + block.shape, dtype=block.dtype)
        psi_array[k] = block
        if progress and ((k + 1) % 5 == 0 or k + 1 == len(indices)):
            print(f"  mapped band {k+1}/{len(indices)} (qe index {ib})", flush=True)
    if psi_array is None:
        psi_array = np.zeros((0,))

    if rho_data is None:
        rho_data = driver.get_density()
    rho_np = np.asarray(rho_data)
    npts = int(np.prod(grid_rho.nr))
    if rho_np.size < npts:
        raise ValueError(
            f"density has size {rho_np.size}, expected >= {npts} "
            f"on charge grid {list(grid_rho.nr)}. Gather on the QE FFT root and "
            "broadcast before packing."
        )
    # Total density for ALDA (sum spins if needed).
    if rho_np.ndim == 2 and rho_np.shape[-1] > 1:
        total = np.sum(rho_np, axis=-1).ravel(order="F")
    else:
        total = rho_np.ravel(order="F")[:npts]
    rho_field = DirectField(grid=grid_rho, data=np.ascontiguousarray(total, dtype=float))
    return grid_rho, rho_field, psi_array


def _real_charge_orbital(raw, grid_wf, grid_rho) -> np.ndarray:
    """Map one smooth-grid orbital onto the charge grid and keep the real part.

    Phase alignment matches the previous full-stack path: the global phase is
    taken from the largest complex sample, then the real part is stored.
    """
    pf = DirectField(grid=grid_wf, data=np.asarray(raw).ravel())
    if list(grid_wf.nr) != list(grid_rho.nr):
        pf = DirectField(grid=grid_rho, data=grid_map_data(pf, grid=grid_rho))
    flat = np.ascontiguousarray(np.asarray(pf)).ravel()
    if np.iscomplexobj(flat):
        idx = int(np.argmax(np.abs(flat)))
        phase = np.angle(flat[idx])
        flat = np.real(flat * np.exp(-1j * phase))
    return np.ascontiguousarray(flat, dtype=np.float64)


def extract_casida_inputs_from_qepy_driver(
    driver,
    atoms,
    target_grid=None,
    use_eDFTpy: bool = False,
    load_uspp: bool = True,
    pack_band_range: Optional[Tuple[int, int]] = None,
    pack_band_indices: Optional[Sequence[int]] = None,
) -> Tuple[CasidaInputs, CasidaOptions]:
    """Extract Casida inputs from a QEpy driver object.

    Automatically detects ultrasoft pseudopotentials from driver.qe_options
    and loads beta projectors / Q_ij augmentation when USPP files are found
    (unless ``load_uspp=False``).

    Parameters
    ----------
    use_eDFTpy : bool
        If False (default), build rho/psi on the QE **charge (density) grid**
        from the driver (USPP-safe). If True, keep the legacy path that uses
        ``target_grid`` / ``atoms.grid`` and ``atoms.density``.
    target_grid : DirectGrid or None
        Optional grid override (density-grid path only).
        Legacy path (``use_eDFTpy=True``): pass ``atoms.grid`` as third argument.
    load_uspp : bool
        If False, skip building beta / Q_ij projectors (saves a lot of RAM on
        large grids). SCF may still have used USPP; only Casida augmentation
        is omitted.
    pack_band_range
        Optional ``(i0, i1)`` slice of QE bands to map onto the charge grid.
        Skips expensive smooth→charge interpolation for unused conduction bands.
    pack_band_indices
        Optional explicit list of global QE band indices to map (overrides
        ``pack_band_range`` when given).
    """
    eig_ry = driver.get_eigenvalues()
    eig_ha = np.asarray(eig_ry) / 2.0

    occs_norm = np.asarray(driver.get_occupation_numbers())
    ions = driver.get_dftpy_ions()

    if use_eDFTpy:
        # All-band gather is only for the eDFTpy path. The density-grid path
        # below fetches one band at a time so every rank does not hold the
        # full smooth-grid stack.
        psi_raw = driver.get_wave_function()
        # eDFTpy subsystem coupling path.
        rank = mpi_comm_rank(driver.comm)
        size = mpi_comm_size(driver.comm)
        if rank == 0:
            print(occs_norm)

        # Raw QEpy driver: its getters gather to the full grid (gather=True) and
        # its data2field takes ``data`` as the first argument (the eDFTpy
        # DriverKS wrapper instead takes ``grid`` first).
        qe = driver.engine.driver

        # Global smooth (wavefunction) and dense (charge) grids. Build FULL,
        # replicated serial grids (no comm) so every rank holds the whole field:
        # CasidaPy parallelizes over transition pairs, not over a
        # domain-decomposed spatial grid. Passing comm=driver.comm would
        # decompose the grid and break consistency with the full field data
        # (it only worked by accident at comm.size == 1).
        nr_smooth = np.zeros(3, dtype=np.int32)
        qe.qepy_pw.qepy_mod.qepy_get_grid_smooth(nr_smooth, gather=True)
        grid_wf = DirectGrid(lattice=ions.cell, nr=list(nr_smooth))
        grid_casida = DirectGrid(
            lattice=ions.cell, nr=list(qe.get_number_of_grid_points(gather=True))
        )

        # Orbitals: psi_raw came from the collective, gathered get_wave_function
        # on every rank, but only rank 0 assembles the stack (mapped onto the
        # charge grid); psi_to_fields broadcasts it. Non-root ranks pass
        # psi=None, avoiding reliance on every rank returning the full array.
        if rank == 0:
            psi_blocks = []
            for ps in psi_raw:
                pf = qe.data2field(data=ps, grid=grid_wf)
                if not hasattr(pf, "grid"):
                    pf = DirectField(grid=grid_wf, data=np.asarray(pf).ravel())
                if list(grid_wf.nr) != list(grid_casida.nr):
                    pf = DirectField(
                        grid=grid_casida, data=grid_map_data(pf, grid=grid_casida)
                    )
                psi_blocks.append(np.asarray(pf))
            psi_array = np.array(psi_blocks)
        else:
            psi_array = None

        # Ground-state density: get_density(gather=True) is collective but fills
        # the full array only on rank 0 (non-root ranks get a tiny (1, nspin)
        # placeholder). Broadcast the full density so every rank can build the
        # DirectField; CasidaKS needs rho_ks.grid and the XC kernel on all ranks.
        rho_root = qe.get_density(gather=True)
        if size > 1:
            shape = driver.comm.bcast(np.shape(rho_root) if rank == 0 else None, root=0)
            if rank == 0:
                rho_buf = np.ascontiguousarray(rho_root, dtype=np.float64)
            else:
                rho_buf = np.empty(shape, dtype=np.float64)
            driver.comm.Bcast(rho_buf, root=0)
        else:
            rho_buf = np.ascontiguousarray(rho_root, dtype=np.float64)
        rho_field = qe.data2field(data=rho_buf, grid=grid_casida)
    else:
        # Density-grid (USPP-safe) path. Use a real MPI communicator (COMM_WORLD
        # if Driver was built with comm=None). Gather density on the QE FFT
        # root, broadcast to all ranks, then map orbitals on world rank 0 only.
        comm = _casida_mpi_comm(driver)
        rank = mpi_comm_rank(comm)
        size = mpi_comm_size(comm)
        n_orb = int(eig_ha.size)
        if pack_band_indices is not None:
            band_indices = sorted({int(i) for i in pack_band_indices})
            if not band_indices or band_indices[0] < 0 or band_indices[-1] >= n_orb:
                raise ValueError(
                    f"pack_band_indices invalid for n_bands={n_orb}: {band_indices[:5]}..."
                )
        elif pack_band_range is not None:
            i0_pack, i1_pack = (int(pack_band_range[0]), int(pack_band_range[1]))
            if not (0 <= i0_pack < i1_pack <= n_orb):
                raise ValueError(
                    f"pack_band_range=({i0_pack}, {i1_pack}) invalid for n_bands={n_orb}"
                )
            band_indices = list(range(i0_pack, i1_pack))
        else:
            band_indices = list(range(n_orb))

        # Collective gather: full array only on QE dfftp root (size-1 elsewhere).
        try:
            rho_raw = driver.get_density(gather=True)
        except TypeError:
            rho_raw = driver.get_density()

        dens_root = _qe_density_root_rank(driver, comm) if size > 1 else 0
        if size > 1:
            if rank == dens_root:
                rho_buf = np.ascontiguousarray(rho_raw, dtype=np.float64)
                rho_shape = rho_buf.shape
            else:
                rho_shape = None
                rho_buf = None
            rho_shape = comm.bcast(rho_shape, root=dens_root)
            if rank != dens_root:
                rho_buf = np.empty(rho_shape, dtype=np.float64)
            # Uppercase Bcast for large arrays (avoid pickle 2 GiB limit).
            if hasattr(comm, "Bcast"):
                comm.Bcast(rho_buf, root=dens_root)
            else:
                rho_buf = comm.bcast(rho_buf, root=dens_root)
            rho_raw = rho_buf
            if rank == 0:
                print(
                    f"[casida-pack] density broadcast from QE root={dens_root} "
                    f"shape={rho_shape}  world_size={size}",
                    flush=True,
                )

        # QE grid queries are MPI collectives — every rank must enter them.
        # Rank-0-only pack previously deadlocked here after density Bcast.
        if rank == 0:
            print(
                f"[casida-pack] querying QE grids on all {size} ranks "
                f"before remapping {len(band_indices)} bands...",
                flush=True,
            )
        nrs_smooth = np.zeros(3, dtype=np.int32)
        driver.qepy_pw.qepy_mod.qepy_get_grid_smooth(nrs_smooth, gather=True)
        nr_rho = list(driver.get_number_of_grid_points(gather=True))
        grid_wf = DirectGrid(lattice=ions.cell, nr=list(nrs_smooth))
        grid_rho_serial = (
            target_grid
            if target_grid is not None and list(target_grid.nr) == list(nr_rho)
            else DirectGrid(lattice=ions.cell, nr=nr_rho)
        )
        if rank == 0:
            print(
                f"[casida-pack] smooth={list(nrs_smooth)} charge={nr_rho} "
                f"— remapping on rank 0",
                flush=True,
            )

        # One band at a time. get_wave_function(band=) is an MPI collective, so
        # every rank calls it, but only rank 0 keeps the charge-grid copy, and
        # that copy is freed before the next band. The previous path gathered
        # every smooth-grid band onto every rank, then allocated a second full
        # charge-grid stack before the scatter.
        n_flat = int(np.prod(nr_rho))
        n_pack = len(band_indices)
        if size > 1:
            p0, p1 = grid_slab_bounds(n_flat, rank, size)
            psi_packed = np.empty((n_pack, p1 - p0), dtype=np.float64)
        else:
            p0, p1 = 0, n_flat
            psi_packed = np.empty((n_pack, n_flat), dtype=np.float64)
        if rank == 0:
            print(
                f"[casida-pack] mapping {n_pack} bands one at a time "
                f"onto charge grid {nr_rho} ({n_flat:,} pts)",
                flush=True,
            )

        for k, ib in enumerate(band_indices):
            wfs = driver.get_wave_function(band=int(ib))
            raw = wfs[0]
            del wfs
            if rank == 0:
                real_band = _real_charge_orbital(raw, grid_wf, grid_rho_serial)
                if real_band.size != n_flat:
                    raise RuntimeError(
                        f"band {ib} mapped to {real_band.size} pts, expected {n_flat}"
                    )
            else:
                real_band = None
            del raw
            if size > 1:
                scatter_real_band(comm, real_band, psi_packed[k], root=0, tag=k)
            else:
                psi_packed[k] = real_band
            del real_band
            if rank == 0 and ((k + 1) % 5 == 0 or k + 1 == n_pack):
                print(
                    f"  mapped band {k + 1}/{n_pack} (qe index {ib})",
                    flush=True,
                )

        rho_np = np.asarray(rho_raw)
        if rho_np.ndim == 2 and rho_np.shape[-1] > 1:
            total = np.sum(rho_np, axis=-1).ravel(order="F")
        else:
            total = rho_np.ravel(order="F")[:n_flat]
        rho_arr = np.ascontiguousarray(total, dtype=np.float64)
        del rho_raw
        lattice = np.ascontiguousarray(
            np.asarray(grid_rho_serial.lattice), dtype=np.float64
        )
        grid_casida = DirectGrid(lattice=lattice, nr=list(nr_rho))
        rho_field = DirectField(grid=grid_casida, data=rho_arr)
        if rank == 0:
            print(
                f"[casida-pack] slab-ready orbitals: each rank holds "
                f"{psi_packed.shape[1]:,}/{n_flat:,} pts "
                f"(rank0 [{p0}:{p1}))",
                flush=True,
            )

        psi_array = [None] * n_orb
        for k, ib in enumerate(band_indices):
            psi_array[ib] = psi_packed[k]

    n_occ = int(np.sum(occs_norm > 0.01))
    n_unocc = int(np.sum(occs_norm < 0.01))

    beta_projectors = None
    qij_augmentation = None
    use_uspp = False
    uspp_map = build_uspp_map_from_driver(driver)

    if load_uspp and uspp_map:
        from casidapy.utils.uspp import load_uspp_data, parse_upf

        has_uspp_species = any(
            os.path.isfile(path) and parse_upf(path).get("is_uspp", False)
            for path in uspp_map.values()
        )
        if has_uspp_species:
            beta_projectors, qij_augmentation, _ = load_uspp_data(
                upf_files=uspp_map, grid=grid_casida, ions=ions
            )
            use_uspp = True

    casida_options = CasidaOptions(
        n_occ=n_occ,
        n_unocc=n_unocc,
        n_states=50,
        n_total_occ=None,
        tda=False,
        matrix_free=False,
        use_uspp=use_uspp,
        uspp_map=uspp_map if use_uspp else None,
        use_eDFTpy=use_eDFTpy,
    )
    casida_inputs = CasidaInputs(
        atoms=atoms,
        grid=grid_casida,
        rho_ks=rho_field,
        psi=psi_array,
        eigs=eig_ha,
        occs=occs_norm,
        beta_projectors=beta_projectors,
        qij_augmentation=qij_augmentation,
    )
    return casida_inputs, casida_options


def lowest_virtual_energy_ha(driver, *, atoms=None) -> float:
    """Lowest QE conduction eigenvalue (Hartree) on the loaded band window."""
    if atoms is None:
        atoms = driver.get_ase_atoms()
    # Read eigenvalues/occupations straight off the driver. Routing this through
    # extract_casida_inputs_from_qepy_driver maps every band onto the charge grid
    # (tens of GB) just to reach ``inputs.occs``, and its ``get_wave_function``
    # call is an MPI-collective gather, so it deadlocks whenever a single rank
    # asks for the shift.
    eigs_ha = np.asarray(driver.get_eigenvalues(), dtype=float).ravel() / 2.0
    occs = np.asarray(driver.get_occupation_numbers(), dtype=float).ravel()
    n_total_occ = _infer_n_total_occ(driver, atoms, eigs_ha, occs)
    if n_total_occ >= eigs_ha.size:
        raise RuntimeError(
            f"no virtual bands in QE eigenvalue array (n_total_occ={n_total_occ}, "
            f"n_bands={eigs_ha.size}); increase nbnd"
        )
    return float(eigs_ha[n_total_occ])


def fermi_energy_ha(driver) -> float:
    """QE Fermi level (Hartree), same energy zero as ``get_eigenvalues()/2``."""
    get_ef = getattr(driver, "get_fermi_level", None)
    if get_ef is None:
        get_ef = getattr(type(driver), "get_fermi_level", None)
    if get_ef is None:
        raise AttributeError(
            "QE driver has no get_fermi_level(); cannot align virtuals to Fermi"
        )
    ef_ry = float(get_ef())
    return ef_ry / 2.0


def virtual_shift_to_fermi_zero_ha(driver) -> float:
    """Rigid PW virtual shift so conduction energies are referenced to Fermi (E_F → 0)."""
    return -fermi_energy_ha(driver)


def vacuum_level_ha(driver, *, frac: float = 0.005) -> float:
    """Mean KS effective potential (Hartree) in the lowest-density cell fraction.

    Uses the same vacuum mask convention as
    :func:`casidapy.embed.hirshfeld.shift_v_env_gauge_min_density` so PW bands
    and ``V_env`` can share a cluster vacuum zero.
    """
    import numpy as np

    from casidapy.embed.potential import effective_potential_ha, ensure_charge_grid

    veff, grid = effective_potential_ha(driver)
    grid = ensure_charge_grid(driver) if grid is None else grid
    rho = np.asarray(driver.get_density(), dtype=float)
    if rho.ndim == 2:
        rho = rho[:, 0]
    nr = tuple(int(x) for x in grid.nr)
    vflat = np.asarray(veff, dtype=float).reshape(nr, order="F").ravel()
    rhoflat = np.asarray(rho, dtype=float).reshape(nr, order="F").ravel()
    if vflat.size != rhoflat.size:
        # Fall back to flat length match (some QE layouts already match ravel).
        vflat = np.asarray(veff, dtype=float).ravel()
        rhoflat = np.asarray(rho, dtype=float).ravel()
        if vflat.size != rhoflat.size:
            raise ValueError(
                f"vacuum_level_ha: veff size {vflat.size} != rho size {rhoflat.size}"
            )
    n = int(vflat.size)
    k = max(1, min(int(float(frac) * n), n - 1))
    idx = np.argpartition(rhoflat, k)[:k]
    return float(np.mean(vflat[idx]))


def virtual_shift_to_vacuum_zero_ha(driver, *, frac: float = 0.005) -> float:
    """Rigid PW virtual shift so vacuum (min-density V_eff) sits at 0 Ha."""
    return -vacuum_level_ha(driver, frac=frac)


def virtual_shift_to_cbm_zero_ha(driver, *, atoms=None) -> float:
    """Rigid PW virtual shift so the lowest conduction band is at 0 Ha."""
    return -lowest_virtual_energy_ha(driver, atoms=atoms)


def _infer_n_total_occ(driver, atoms, eigs, occs) -> int:
    """Guess occupied band count (C USPP → ``2 * n_C``, else occupations / ε<0)."""
    n_c = sum(1 for s in atoms.get_chemical_symbols() if s == "C")
    if n_c > 0:
        return 2 * n_c
    n = int(np.sum(np.asarray(occs, dtype=float) > 0.01))
    if n > 0:
        return n
    return max(int(np.sum(np.asarray(eigs, dtype=float) < 0.0)), 1)


def _normalized_psi_window(inputs: CasidaInputs, i0: int, i1: int):
    """Normalize orbitals ``[i0, i1)`` onto ``inputs.grid`` as DirectFields."""
    return _normalized_psi_by_indices(inputs, range(i0, i1))


def _normalized_psi_by_indices(inputs: CasidaInputs, band_indices: Sequence[int]):
    """Normalize explicit global band indices onto ``inputs.grid``."""
    from casidapy.utils.casida_utils import normalize_wavefunctions

    vol = float(np.linalg.det(np.asarray(inputs.grid.lattice)))
    raw = []
    for i in band_indices:
        pf = DirectField(grid=inputs.grid, data=np.asarray(inputs.psi[i]))
        raw.append(pf / np.sqrt(vol))
    return normalize_wavefunctions(raw, inputs.grid)


def extract_pw_kernel(
    driver,
    *,
    atoms=None,
    n_occ: Optional[int] = None,
    n_unocc: Optional[int] = None,
    n_total_occ: Optional[int] = None,
    n_states: int = 20,
    xc: str = "PBE",
    use_gpu: bool = False,
    use_uspp: bool = False,
    use_eDFTpy: bool = False,
    spin_state: str = "singlet",
    tda: bool = True,
    matrix_free: bool = True,
    n_placeholder_occ: Optional[int] = None,
    unocc_window_ev: Optional[float] = None,
    unocc_subsample: Optional[int] = None,
    virt_energy_shift_ha: float = 0.0,
    pack_band_indices: Optional[Sequence[int]] = None,
    verbose: bool = False,
) -> Tuple["PlaneWaveKernel", CasidaOptions]:
    """Create a :class:`~casidapy.kernels.plane_wave.PlaneWaveKernel` from a QEpy ``driver``.

    Mirrors :func:`casidapy.adapter.pyscf.extract_gto_kernel`::

        kernel, opts = extract_pw_kernel(driver, n_unocc=40, n_states=40)
        results = run_casida(kernel, opts)

    For CVS core inject, pass ``n_placeholder_occ`` (occupied slots to replace)
    and ``n_unocc`` (QE conduction bands). Grid is ``kernel.grid``.

    Parameters
    ----------
    n_placeholder_occ
        If set, only that many valence orbitals are kept as placeholders for
        :func:`~casidapy.xas.inject_core_orbitals` (CVS). Defaults to a
        normal valence×virtual window when omitted.
    unocc_subsample
        Optional. After selecting ``n_unocc`` (and optional ``unocc_window_ev``),
        energy-stride down to this many active virtuals. Intended for large
        continuum pools (wide ``nbnd``) where Casida cannot hold every band.
    virt_energy_shift_ha
        Rigid shift (Hartree) applied to every selected virtual eigenvalue
        before the Casida kernel is built. Use with
        :func:`casidapy.xas.virtual_shift_from_gto_mf` (LUMO align),
        :func:`casidapy.adapter.qepy.virtual_shift_to_fermi_zero_ha` (Fermi
        reference), or :func:`casidapy.xas.resolve_virt_energy_shift_ha`.
        Pair LUMO align with ``eps_core_reference_ha`` for absolute ω scale.
    pack_band_indices
        Optional explicit global QE band indices to map onto the charge grid.
        When set, occupied/virtual planning and subsampling are assumed done.
    """
    import warnings

    from dftpy.functional.xc import XC
    from casidapy.kernels.plane_wave import PlaneWaveKernel

    if atoms is None:
        atoms = driver.get_ase_atoms()

    eig_ha = np.asarray(driver.get_eigenvalues(), dtype=float) / 2.0
    occs_norm = np.asarray(driver.get_occupation_numbers(), dtype=float)
    n_orb = int(eig_ha.size)

    if n_total_occ is None:
        n_total_occ = _infer_n_total_occ(driver, atoms, eig_ha, occs_norm)
    n_total_occ = int(n_total_occ)

    if n_placeholder_occ is not None:
        n_occ = int(n_placeholder_occ) if n_occ is None else int(n_occ)
    elif n_occ is None:
        n_occ = n_total_occ

    if n_unocc is None:
        n_unocc = max(n_orb - n_total_occ, 1)
    n_unocc = int(n_unocc)

    if n_total_occ + n_unocc > n_orb:
        raise RuntimeError(
            f"Need n_total_occ+n_unocc={n_total_occ + n_unocc} bands, "
            f"have {n_orb}. Increase nbnd in the QE SCF."
        )

    if pack_band_indices is not None:
        band_indices = sorted({int(i) for i in pack_band_indices})
        occ_indices = [i for i in band_indices if i < n_total_occ]
        virt_indices = [i for i in band_indices if i >= n_total_occ]
        if not occ_indices:
            i0 = max(0, n_total_occ - int(n_occ))
            occ_indices = list(range(i0, n_total_occ))
            band_indices = sorted(set(occ_indices) | set(virt_indices))
    else:
        occ_indices, virt_indices, _plan = plan_pw_band_indices(
            eig_ha,
            n_total_occ=n_total_occ,
            n_occ=int(n_occ),
            n_unocc=int(n_unocc),
            n_orb=n_orb,
            unocc_window_ev=unocc_window_ev,
            unocc_subsample=unocc_subsample,
            verbose=verbose,
        )
        band_indices = occ_indices + virt_indices

    inputs, _ = extract_casida_inputs_from_qepy_driver(
        driver,
        atoms,
        use_eDFTpy=use_eDFTpy,
        load_uspp=bool(use_uspp),
        pack_band_indices=band_indices,
    )
    eigs = np.asarray(inputs.eigs, dtype=float)
    occs = np.asarray(inputs.occs, dtype=float)
    packed_psi = _normalized_psi_by_indices(inputs, band_indices)
    psi_list: list = [None] * n_orb
    for i, psi in zip(band_indices, packed_psi):
        psi_list[i] = psi

    occ_e, unocc_e, psi_occ, psi_unocc = slice_active_space_by_indices(
        eigs,
        psi_list,
        occ_indices,
        virt_indices,
        verbose=verbose,
    )
    n_unocc = len(psi_unocc)

    vshift = float(virt_energy_shift_ha)
    if abs(vshift) > 0.0:
        unocc_e = np.asarray(unocc_e, dtype=float) + vshift
        if verbose:
            print(
                f"  virtual energy shift: {vshift:.6f} Ha "
                f"({vshift * _HA_TO_EV:.3f} eV) applied to {unocc_e.size} bands"
            )

    want_uspp = bool(use_uspp) and bool(inputs.beta_projectors is not None)
    if use_gpu and want_uspp:
        warnings.warn(
            "Casida USPP + GPU is unsupported; disabling USPP augmentation "
            "so the GPU FFT path can run.",
            stacklevel=2,
        )
        want_uspp = False

    spin_state = str(spin_state).lower().strip()
    libxc_xc_components = None
    if spin_state == "triplet":
        from casidapy.utils.casida_utils import XC_TO_LIBXC_COMPONENTS

        xc_up = str(xc).upper()
        if xc_up not in XC_TO_LIBXC_COMPONENTS:
            raise ValueError(f"No XC→libxc mapping for triplet: {xc}")
        libxc_xc_components = XC_TO_LIBXC_COMPONENTS[xc_up]

    kernel = PlaneWaveKernel(
        inputs.rho_ks,
        XC(xc=xc),
        use_gpu=bool(use_gpu),
        use_uspp=want_uspp,
        beta_projectors=inputs.beta_projectors if want_uspp else None,
        qij_augmentation=inputs.qij_augmentation if want_uspp else None,
        use_eDFTpy=use_eDFTpy,
        spin_state=spin_state,
        libxc_xc_components=libxc_xc_components,
        verbose=verbose,
    )
    kernel.set_active_orbitals(occ_e, unocc_e, psi_occ, psi_unocc)
    # Global QE band indices for G-sum projection (evc) of active virtuals.
    kernel._qe_band_indices_occ = [int(i) for i in occ_indices]
    kernel._qe_band_indices_unocc = [int(i) for i in virt_indices]
    kernel._qe_driver = driver

    opts = CasidaOptions(
        n_occ=kernel.n_occ,
        n_unocc=kernel.n_unocc,
        n_states=int(n_states),
        n_total_occ=int(n_total_occ),
        tda=tda,
        matrix_free=matrix_free,
        solver_method="davidson",
        use_gpu=bool(use_gpu),
        xc=str(xc),
        basis="pw",
        use_uspp=want_uspp,
        uspp_map=None,
        use_eDFTpy=use_eDFTpy,
        spin_state=spin_state,
    )
    return kernel, opts
