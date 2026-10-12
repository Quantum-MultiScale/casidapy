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
    sel = core.meta.get("core_selection") or {}
    assert sel.get("method") == "mulliken_max_per_atom"
    assert sel.get("edge_atoms") == [0]

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


def test_core_from_mf_same_element_k_edge_picks_absorber():
    """Inequivalent C 1s MOs: select by Mulliken pop on the requested atom.

    (Symmetry-degenerate sites with equal populations are interchangeable.)
    """
    pytest.importorskip("pyscf")
    from pyscf import gto, dft
    from casidapy.xas import core_from_mf
    from casidapy.xas.cvs import _ao_mask_for_atoms, _ao_shell_mask, _mo_mulliken_on_aos

    # Acetaldehyde: methyl C vs carbonyl C — chemically distinct 1s.
    mol = gto.M(
        atom=(
            "C  0.0000  0.0000  0.0000;"  # methyl
            "C  1.5070  0.0000  0.0000;"  # carbonyl
            "O  2.1400  1.0700  0.0000;"
            "H -0.3800  1.0200  0.0000;"
            "H -0.3800 -0.5100  0.8840;"
            "H -0.3800 -0.5100 -0.8840;"
            "H  1.9400 -1.0200  0.0000"
        ),
        basis="sto-3g",
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = "lda,vwn"
    mf.kernel()
    assert mf.converged

    S = mol.intor("int1e_ovlp")
    shell = _ao_shell_mask(mol, 1, 0)

    for ia, other in ((0, 1), (1, 0)):
        core = core_from_mf(mf, edge="K", edge_atom_indices=[ia])
        assert core.energies.size == 1
        C = np.asarray(core.mo_coeff, float)
        if C.ndim == 1:
            C = C.reshape(-1, 1)
        pop_abs = float(_mo_mulliken_on_aos(C, S, _ao_mask_for_atoms(mol, [ia], shell))[0])
        pop_other = float(
            _mo_mulliken_on_aos(C, S, _ao_mask_for_atoms(mol, [other], shell))[0]
        )
        assert pop_abs > 0.7, (ia, pop_abs, pop_other)
        assert pop_abs > pop_other + 0.3, (ia, pop_abs, pop_other)
        per = (core.meta.get("core_selection") or {}).get("per_atom") or []
        assert per and int(per[0]["atom"]) == ia
        assert float(per[0]["mulliken_pops"][0]) == pytest.approx(pop_abs, rel=0, abs=1e-8)

    # Default heavy-atom K-edge: one 1s per C and O.
    both = core_from_mf(mf, edge="K")
    assert both.energies.size == 3
    assert set(both.meta["core_selection"]["edge_atoms"]) == {0, 1, 2}


def test_core_localize_recovers_single_site_1s():
    """Symmetry-equivalent absorbers: canonical 1s is shared, localized is not.

    Benzene C 1s MOs are fixed combinations over equivalent carbons, so the
    canonical pick carries well under 1 e on the requested atom.  ``proj`` /
    ``boys`` must recover a single-site core at near-unchanged energy.
    """
    pytest.importorskip("pyscf")
    from pyscf import dft, gto

    from casidapy.xas import core_from_mf
    from casidapy.xas.cvs import (
        _ao_mask_for_atoms,
        _ao_shell_mask,
        _mo_mulliken_on_aos,
    )

    r_cc, r_ch = 1.39, 2.48
    atom = []
    for k in range(6):
        a = 2.0 * np.pi * k / 6.0
        atom.append(f"C {r_cc * np.cos(a):.6f} {r_cc * np.sin(a):.6f} 0.0")
        atom.append(f"H {r_ch * np.cos(a):.6f} {r_ch * np.sin(a):.6f} 0.0")
    mol = gto.M(atom=";".join(atom), basis="sto-3g", verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "lda,vwn"
    mf.kernel()
    assert mf.converged

    S = mol.intor("int1e_ovlp")
    mask = _ao_mask_for_atoms(mol, [0], _ao_shell_mask(mol, 1, 0))

    def _pop(core):
        C = np.asarray(core.mo_coeff, float).reshape(mol.nao, -1)
        return float(_mo_mulliken_on_aos(C[:, :1], S, mask)[0]), C

    canon = core_from_mf(mf, edge="K", edge_atom_indices=[0])
    pop_canon, _ = _pop(canon)
    assert pop_canon < 0.75, pop_canon

    for method in ("proj", "boys"):
        loc = core_from_mf(
            mf, edge="K", edge_atom_indices=[0], core_localize=method
        )
        pop_loc, C_loc = _pop(loc)
        assert loc.energies.size == 1
        assert pop_loc > 0.9, (method, pop_loc)
        # normalized, and still a core orbital at essentially the same energy
        assert float(C_loc[:, 0] @ S @ C_loc[:, 0]) == pytest.approx(1.0, abs=1e-8)
        assert float(loc.energies[0]) == pytest.approx(
            float(canon.energies[0]), abs=5e-3
        )
        assert (loc.meta["core_selection"]["method"]).startswith("localized")

    # Already-localized cores (inequivalent atoms) must be left alone.
    mol2 = gto.M(atom="C 0 0 0; O 0 0 1.128", basis="sto-3g", verbose=0)
    mf2 = dft.RKS(mol2)
    mf2.xc = "lda,vwn"
    mf2.kernel()
    c_canon = core_from_mf(mf2, edge="K", edge_atom_indices=[0])
    c_loc = core_from_mf(mf2, edge="K", edge_atom_indices=[0], core_localize="proj")
    assert float(c_loc.energies[0]) == pytest.approx(
        float(c_canon.energies[0]), abs=1e-5
    )


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


def test_parse_soc_mode():
    from casidapy.xas import parse_soc_mode

    assert parse_soc_mode(False) == "none"
    assert parse_soc_mode(None) == "none"
    assert parse_soc_mode(True) == "orbital"
    assert parse_soc_mode("orbital") == "orbital"
    assert parse_soc_mode("si") == "si"
    assert parse_soc_mode("state-interaction") == "si"
    with pytest.raises(ValueError):
        parse_soc_mode("magic")


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
    # soc="si" must not apply orbital SOC (spatial 2p stay 3)
    core_si = extract_fragment_core(
        mol, [0], edge="L", basis="sto-3g", xc="lda,vwn", soc="si"
    )
    assert core_si.energies.size == 3
    assert core_si.meta.get("soc") is False
    assert core_si.meta.get("soc_deferred") == "si"


def test_l_edge_si_soc_ne():
    """Ne L-edge: spatial 2p (n=3), singlet+triplet TDA, SI-SOC mixing."""
    pytest.importorskip("pyscf")
    from pyscf import gto, dft
    from casidapy.xas import run_cvs_gto_from_mf, stick_cartesian
    from casidapy.utils.soc import soc_ao_integrals, soc_mo_blocks, st_soc_matrix, _reshape_amps

    mol = gto.M(atom="Ne 0 0 0", basis="6-31g", verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "lda,vwn"
    mf.grids.level = 1
    mf.kernel()
    assert mf.converged

    n_virt = int(np.sum(mf.mo_occ < 1e-6))
    n_unocc = min(4, n_virt)
    n_states = min(6, 3 * n_unocc)
    res, core, kernel = run_cvs_gto_from_mf(
        mf,
        edge="L",
        edge_atom_indices=[0],
        n_unocc=n_unocc,
        n_states=n_states,
        soc="si",
        use_df=False,
    )
    assert core.energies.size == 3
    assert core.meta.get("soc") is False
    assert kernel.n_occ == 3
    assert kernel.n_unocc == n_unocc
    assert getattr(res, "metadata", {}).get("soc_mode") == "si"
    n_s = int(res.singlet.omega.size)
    n_t = int(res.triplet.omega.size)
    assert res.omega.size == n_s + 3 * n_t
    assert np.all(res.omega > 0.0)
    assert float(np.sum(res.f)) > 0.0
    # 2p hole SOC is the L2/L3 coupling between singlet and triplet CVS states
    hso = soc_ao_integrals(kernel.mol)
    h_oo, h_vv, _h_ov = soc_mo_blocks(hso, kernel._C_o, kernel._C_v)
    assert np.max(np.abs(h_oo)) > 0.0
    X_s = _reshape_amps(res.singlet.Z, 3, n_unocc)
    X_t = _reshape_amps(res.triplet.Z, 3, n_unocc)
    H_st = st_soc_matrix(X_s, X_t, h_oo, h_vv)
    assert np.max(np.abs(H_st)) > 0.0
    mixed = (res.singlet_weight > 1e-8) & (res.triplet_weight > 1e-8)
    assert np.any(mixed)
    w, f, fx, fy, fz = stick_cartesian(res)
    assert w.size == res.omega.size
    assert res.d_mode.shape == (3, res.omega.size)

    # Hole-only (h_vv = 0) still couples through the 2p hole
    from casidapy.utils.soc import solve_soc_si

    soc_h = solve_soc_si(
        res.singlet, res.triplet, kernel, hole_only=True
    )
    assert soc_h.meta.get("hole_only") is True
    assert np.max(np.abs(st_soc_matrix(
        X_s, X_t, h_oo, np.zeros_like(h_vv)
    ))) > 0.0
    assert soc_h.omega.size == res.omega.size

