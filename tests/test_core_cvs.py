"""Tests for CVS core helpers (radius cut, fragment SCF, PW inject)."""
from __future__ import annotations

import numpy as np
import pytest


def test_select_atoms_in_radius_ase():
    ase = pytest.importorskip("ase")
    from ase import Atoms
    from casidapy.xas import select_atoms_in_radius

    atoms = Atoms(
        "Ag2H",
        positions=[[0, 0, 0], [2.5, 0, 0], [10, 0, 0]],
    )
    idx = select_atoms_in_radius(atoms, center=0, radius_ang=3.0)
    assert set(idx.tolist()) == {0, 1}

    idx_ag = select_atoms_in_radius(
        atoms, center=0, radius_ang=20.0, elements=("Ag",)
    )
    assert set(idx_ag.tolist()) == {0, 1}


def test_neutral_first_shell_h2o():
    pytest.importorskip("ase")
    from ase import Atoms
    from casidapy.xas import (
        FragmentSpec,
        formal_oxidation_charge,
        select_neutral_first_shell,
        select_xas_fragment,
    )

    atoms = Atoms(
        "OH2",
        positions=[[0, 0, 0], [0, 0.76, 0.59], [0, -0.76, 0.59]],
    )
    assert formal_oxidation_charge(["O", "H", "H"]) == 0
    frag = select_neutral_first_shell(atoms, center=0)
    assert frag.mode == "neutral_first_shell"
    assert frag.meta["neutral"] is True
    assert frag.charge == 0
    assert set(frag.atom_indices.tolist()) == {0, 1, 2}

    user = FragmentSpec.user([0, 1], charge=-1, edge_atom_indices=[0])
    assert user.mode == "user"
    assert user.charge == -1

    via = select_xas_fragment(atoms, edge_atom=0, atom_indices=[0, 1, 2], charge=0)
    assert via.mode == "user"
    assert via.charge == 0


def test_neutral_first_shell_tio2_like():
    """O + 3 Ti alone is +10; ligand completion should reach a neutral cut."""
    pytest.importorskip("ase")
    from ase import Atoms
    from casidapy.xas import select_xas_fragment

    # Edge O at origin, 3 Ti along axes, plus enough O to neutralize (Ti3O6).
    # Ti–O ~1.95 Å (within covalent cutoff).
    r = 1.95
    symbols = ["O", "Ti", "Ti", "Ti", "O", "O", "O", "O", "O"]
    pos = [
        [0.0, 0.0, 0.0],
        [r, 0.0, 0.0],
        [-r, 0.0, 0.0],
        [0.0, r, 0.0],
        [2 * r, 0.0, 0.0],
        [-2 * r, 0.0, 0.0],
        [r, r, 0.0],
        [-r, r, 0.0],
        [0.0, 2 * r, 0.0],
    ]
    atoms = Atoms(symbols=symbols, positions=pos)
    frag = select_xas_fragment(atoms, edge_atom=0, mode="neutral_first_shell", max_atoms=12)
    assert frag.meta["neutral"] is True, frag.meta
    assert frag.charge == 0
    assert 0 in frag.atom_indices
    # Must include first-shell Ti and extra O (not O+3Ti alone)
    assert frag.meta["n_atoms"] > 4
    assert frag.meta["symbols"].count("Ti") >= 1
    assert frag.meta["symbols"].count("O") >= 2


def test_core_from_mf_user_supplied():
    """User runs SCF, then core_from_mf / run_cvs_gto_from_mf."""
    pytest.importorskip("pyscf")
    from pyscf import gto, dft
    from casidapy.xas import core_from_mf, run_cvs_gto_from_mf

    mol = gto.M(
        atom="O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59",
        basis="sto-3g",
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = "lda,vwn"
    mf.kernel()
    assert mf.converged

    core = core_from_mf(mf, edge="K", edge_atom_indices=[0])
    assert core.meta.get("from_user_mf") is True
    assert core.energies.size == 1
    assert core.energies[0] < -10.0

    n_virt_avail = int(np.sum(mf.mo_occ < 1e-6))
    n_unocc = min(4, n_virt_avail)
    res, core2, kernel = run_cvs_gto_from_mf(
        mf,
        edge="K",
        edge_atom_indices=[0],
        n_unocc=n_unocc,
        n_states=n_unocc,
        use_df=False,
    )
    assert kernel.n_occ == 1
    assert kernel.n_unocc == n_unocc
    assert res.omega.size == n_unocc
    assert np.all(res.omega > 10.0)  # core→virt, deep gap


def test_extract_fragment_core_k_edge_h2o():
    pytest.importorskip("pyscf")
    from pyscf import gto
    from casidapy.xas import extract_fragment_core, select_atoms_in_radius

    mol = gto.M(
        atom="O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59",
        basis="sto-3g",
        verbose=0,
    )
    idx = select_atoms_in_radius(mol, center=0, radius_ang=2.0)
    assert 0 in idx
    core = extract_fragment_core(
        mol,
        idx,
        edge="K",
        basis="sto-3g",
        xc="lda,vwn",
        edge_atom_indices=[0],  # O only — not H 1s
        verbose=False,
    )
    assert core.shell == "1s"
    assert core.energies.size == 1
    assert core.energies[0] < -10.0  # O 1s deep
    assert core.mo_coeff.shape[1] == 1
    assert core.meta.get("edge_atom_indices") == [0]

    # edge atoms must be subset of fragment
    with pytest.raises(ValueError, match="not in fragment"):
        extract_fragment_core(
            mol, [0], edge="K", basis="sto-3g", xc="lda,vwn", edge_atom_indices=[1]
        )


def test_pw_inject_synthetic_core():
    """Graft a tight Gaussian '1s' onto a small PW kernel and run CVS-TDA."""
    pytest.importorskip("dftpy")
    from dftpy.grid import DirectGrid
    from dftpy.field import DirectField
    from dftpy.functional.xc import XC
    from casidapy.kernels import PlaneWaveKernel
    from casidapy.casida_api import CasidaOptions
    from casidapy.xas import inject_core_orbitals, run_cvs_tda
    from casidapy.utils.casida_utils import normalize_wavefunctions

    lattice = np.eye(3) * 8.0
    grid = DirectGrid(lattice=lattice, nr=[16, 16, 16])
    r = np.asarray(grid.r)
    center = lattice.sum(axis=0) / 2.0
    dr2 = sum((r[k] - center[k]) ** 2 for k in range(3))

    rho = np.exp(-dr2 / 2.0) + 0.05
    rho_field = DirectField(grid, rank=1, griddata_3d=rho)

    # Fake valence virt + a shallow occupied (will be replaced)
    def _orb(scale, poly=1.0):
        arr = poly * np.exp(-dr2 / scale)
        nrm = np.sqrt(np.sum(arr**2) * grid.dV)
        return DirectField(grid, rank=1, griddata_3d=arr / nrm)

    psi_occ_val = [_orb(3.0)]
    psi_virt = [_orb(4.0, r[0] - center[0]), _orb(4.5, r[1] - center[1])]
    psi_occ_val = normalize_wavefunctions(psi_occ_val, grid)
    psi_virt = normalize_wavefunctions(psi_virt, grid)

    kernel = PlaneWaveKernel(rho_field, XC(xc="PBE"), use_gpu=False, verbose=False)
    kernel.set_active_orbitals([-0.2], [0.1, 0.2], psi_occ_val, psi_virt)
    assert kernel.n_trans == 2

    # Core-like Gaussian (tighter)
    core_field = _orb(0.8)
    core_field = normalize_wavefunctions([core_field], grid)[0]
    e_core = np.array([-20.0])
    inject_core_orbitals(kernel, e_core, [core_field])
    assert kernel.n_occ == 1
    assert kernel.n_unocc == 2
    assert kernel.n_trans == 2
    dE = kernel.diagonal_dE()
    assert np.allclose(dE, np.array([0.1 - (-20.0), 0.2 - (-20.0)]))

    opts = CasidaOptions(
        n_occ=1, n_unocc=2, n_states=2, basis="pw", tda=True, matrix_free=True
    )
    res = run_cvs_tda(kernel, opts)
    assert res.omega.size == 2
    assert np.all(res.omega > 0)
    # Core excitations should sit near dE (small K shift)
    assert res.omega[0] > 15.0


def test_orbital_soc_2p_splits():
    pytest.importorskip("pyscf")
    from pyscf import gto, dft
    from casidapy.xas import extract_fragment_core

    # Neon atom: clear 2p shell for L-edge-like SOC test
    mol = gto.M(atom="Ne 0 0 0", basis="sto-3g", verbose=0)
    core = extract_fragment_core(
        mol, [0], edge="L", basis="sto-3g", xc="lda,vwn", soc=True
    )
    assert core.meta.get("soc") is True
    # 3 spatial 2p → 6 spinor states after SOC
    assert core.energies.size == 6
    # Should show some splitting (not all equal)
    assert np.ptp(core.energies) > 1e-6
