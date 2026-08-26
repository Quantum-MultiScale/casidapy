from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from dftpy.field import DirectField
from dftpy.grid import DirectGrid
from dftpy.utils import grid_map_data

from casidapy.casida_api import CasidaInputs, CasidaOptions
from casidapy.utils.casida_utils import mpi_comm_rank, mpi_comm_size

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
            tokens = line.split()
            if len(tokens) >= 3:
                uspp_map[tokens[0]] = os.path.join(pseudo_dir, tokens[2])

        if uspp_map:
            return uspp_map

    inputfile = getattr(driver, "inputfile", None)
    if inputfile and os.path.isfile(inputfile):
        return _parse_uspp_map_from_inputfile(inputfile)

    return {}


def _qe_wavefunction_grid(driver, ions) -> DirectGrid:
    """FFT grid for QE wavefunctions (smooth grid; may differ from charge grid for USPP)."""
    nrs = np.zeros(3, dtype=np.int32)
    driver.qepy_pw.qepy_mod.qepy_get_grid_smooth(nrs)
    return DirectGrid(lattice=ions.cell, nr=list(nrs), comm=driver.comm)


def _pack_on_density_grid(driver, atoms, ions, psi_raw, target_grid=None):
    """Place rho and psi on the QE charge (density) grid via the QEpy driver."""
    nr_rho = list(driver.get_number_of_grid_points())
    grid_rho = driver.get_dftpy_grid(nr=nr_rho)
    if target_grid is not None and list(target_grid.nr) != list(nr_rho):
        print(
            f"WARNING: target_grid.nr={target_grid.nr} != QE charge grid {nr_rho}; "
            f"using QE charge grid for Casida."
        )
    elif target_grid is not None:
        grid_rho = target_grid

    grid_wf = _qe_wavefunction_grid(driver, ions)
    if list(grid_wf.nr) != list(grid_rho.nr):
        print(
            f"Mapping wavefunctions from smooth grid {grid_wf.nr} "
            f"to charge grid {grid_rho.nr}"
        )

    psi_blocks = []
    for ps in psi_raw:
        pf = driver.data2field(data=ps, grid=grid_wf)
        if list(grid_wf.nr) != list(grid_rho.nr):
            pf = DirectField(grid=grid_rho, data=grid_map_data(pf, grid=grid_rho))
        psi_blocks.append(np.asarray(pf))
    psi_array = np.array(psi_blocks)
    rho_field = driver.data2field(driver.get_density(), grid=grid_rho)
    return grid_rho, rho_field, psi_array


def extract_casida_inputs_from_qepy_driver(
    driver,
    atoms,
    target_grid=None,
    use_eDFTpy: bool = False,
    load_uspp: bool = True,
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
    """
    eig_ry = driver.get_eigenvalues()
    eig_ha = np.asarray(eig_ry) / 2.0

    occs_norm = np.asarray(driver.get_occupation_numbers())
    psi_raw = driver.get_wave_function()
    ions = driver.get_dftpy_ions()

    if use_eDFTpy:
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
            lattice=ions.cell, nr=list(qe.get_number_of_grid_points())
        )

        # Orbitals: psi_raw came from the collective, gathered get_wave_function
        # on every rank, but only rank 0 assembles the stack (mapped onto the
        # charge grid); psi_to_fields broadcasts it. Non-root ranks pass
        # psi=None, avoiding reliance on every rank returning the full array.
        if rank == 0:
            psi_blocks = []
            for ps in psi_raw:
                pf = qe.data2field(data=ps, grid=grid_wf)
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
        grid_casida, rho_field, psi_array = _pack_on_density_grid(
            driver, atoms, ions, psi_raw, target_grid=target_grid
        )

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
    from casidapy.utils.casida_utils import normalize_wavefunctions

    vol = float(np.linalg.det(np.asarray(inputs.grid.lattice)))
    raw = []
    for i in range(i0, i1):
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
    """
    import warnings

    from dftpy.functional.xc import XC
    from casidapy.kernels.plane_wave import PlaneWaveKernel

    if atoms is None:
        atoms = driver.get_ase_atoms()

    inputs, _ = extract_casida_inputs_from_qepy_driver(
        driver,
        atoms,
        use_eDFTpy=use_eDFTpy,
        load_uspp=bool(use_uspp),
    )
    eigs = np.asarray(inputs.eigs, dtype=float)
    occs = np.asarray(inputs.occs, dtype=float)
    n_orb = len(inputs.psi)

    if n_total_occ is None:
        n_total_occ = _infer_n_total_occ(driver, atoms, eigs, occs)
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

    i0 = max(0, n_total_occ - int(n_occ))
    i1 = n_total_occ + n_unocc
    psi_window = _normalized_psi_window(inputs, i0, i1)
    psi_list: list = [None] * n_orb
    for j, i in enumerate(range(i0, i1)):
        psi_list[i] = psi_window[j]

    occ_e, unocc_e, psi_occ, psi_unocc = slice_active_space(
        eigs,
        psi_list,
        n_occ=int(n_occ),
        n_unocc=int(n_unocc),
        n_total_occ=int(n_total_occ),
        unocc_window_ev=unocc_window_ev,
        unocc_subsample=unocc_subsample,
        verbose=verbose,
    )
    n_unocc = len(psi_unocc)

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
