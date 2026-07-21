from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from dftpy.field import DirectField
from dftpy.grid import DirectGrid
from dftpy.utils import grid_map_data

from casidapy.casida_api import CasidaInputs, CasidaOptions
from casidapy.casida_utils import mpi_comm_rank, mpi_comm_size


def slice_active_space(
    eigs: np.ndarray,
    psi_all: Sequence,
    n_occ: int,
    n_unocc: Optional[int] = None,
    n_total_occ: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """Slice full KS arrays into active occupied/unoccupied sets.

    Returns (occ_eigs, unocc_eigs, psi_occ, psi_unocc).
    This is the single source of truth for active-space windowing,
    used by both the QEpy adapter path and the generic CLI runner.
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
    psi_unocc = list(psi_all[n_total_occ:u1])
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
) -> Tuple[CasidaInputs, CasidaOptions]:
    """Extract Casida inputs from a QEpy driver object.

    Automatically detects ultrasoft pseudopotentials from driver.qe_options
    and loads beta projectors / Q_ij augmentation when USPP files are found.

    Parameters
    ----------
    use_eDFTpy : bool
        If False (default), build rho/psi on the QE **charge (density) grid**
        from the driver (USPP-safe). If True, keep the legacy path that uses
        ``target_grid`` / ``atoms.grid`` and ``atoms.density``.
    target_grid : DirectGrid or None
        Optional grid override (density-grid path only).
        Legacy path (``use_eDFTpy=True``): pass ``atoms.grid`` as third argument.
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

    if uspp_map:
        from casidapy.uspp import load_uspp_data, parse_upf

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
