"""Tests for AE core reconstruction and potential embedding."""
from __future__ import annotations

import numpy as np
import pytest


def test_accel_helpers_serial():
    import numpy as np
    from casidapy.utils.accel import (
        array_module,
        asnumpy,
        block_slices,
        distributed_indices,
        mpi_allreduce_sum,
    )

    assert array_module(False) is np
    assert list(distributed_indices(4, None)) == [0, 1, 2, 3]
    assert block_slices(7, 3, None) == [(0, 3), (3, 6), (6, 7)]
    assert mpi_allreduce_sum(None, np.arange(3.0)).tolist() == [0.0, 1.0, 2.0]
    assert asnumpy(np.array([1.0])).shape == (1,)


def test_xas_embed_gpu_comm_kwargs():
    import inspect
    from casidapy.embed import build_ae_embedding_potential, build_hirshfeld_embedding
    from casidapy.xas import (
        core_mos_to_pw_fields,
        reconstruct_core_from_driver,
        run_xas_gto,
        run_xas_reconstruct,
        scf_ae_core_semicore,
    )
    from casidapy.casida_engine import CasidaKS_MPI

    for fn in (
        build_ae_embedding_potential,
        build_hirshfeld_embedding,
        reconstruct_core_from_driver,
        run_xas_reconstruct,
        scf_ae_core_semicore,
        core_mos_to_pw_fields,
    ):
        params = inspect.signature(fn).parameters
        assert "use_gpu" in params
        assert "comm" in params
    assert "use_mpi_response" in inspect.signature(run_xas_gto).parameters
    xas_params = inspect.signature(CasidaKS_MPI.xas).parameters
    assert "use_gpu" in xas_params and "comm" in xas_params


def test_read_upf_local_potential_carbon():
    from pathlib import Path
    from casidapy.embed import read_upf_local_potential

    upf = Path("/projectsn/mp1009_1/am4655/benz_fulv/pseudo/c_pbe_v1.2.uspp.F.UPF")
    if not upf.is_file():
        pytest.skip("C UPF not available")
    r, v, z_val = read_upf_local_potential(str(upf))
    assert r.size == v.size
    assert z_val == pytest.approx(4.0)
    assert np.isfinite(v[0])
    assert abs(v[-1] - (-z_val / r[-1])) < 1e-4


def test_frozen_shells_from_pp():
    from casidapy.xas import frozen_shells_from_pp

    sh, n = frozen_shells_from_pp("C", 4.0)
    assert n == 2 and sh == ("1s",)

    sh, n = frozen_shells_from_pp("Si", 4.0)
    assert n == 10 and sh == ("1s", "2s", "2p")

    sh, n = frozen_shells_from_pp("Ti", 12.0)
    assert n == 10 and sh == ("1s", "2s", "2p")


def test_scf_ae_core_semicore_free_carbon_1s():
    """V_env = 0 → isolated C with PP-frozen 1s (z_valence=4)."""
    pytest.importorskip("pyscf")
    pytest.importorskip("dftpy")
    from dftpy.grid import DirectGrid
    from dftpy.field import DirectField
    from casidapy.xas import scf_ae_core_semicore
    from casidapy.xas import core_from_mf

    lattice = np.eye(3) * 16.0  # Bohr
    grid = DirectGrid(lattice=lattice, nr=[24, 24, 24])
    v_env = DirectField(grid=grid, rank=1, griddata_3d=np.zeros(grid.nr))

    from ase.units import Bohr

    center_ang = (0.5 * lattice.sum(axis=0)) * Bohr
    mf = scf_ae_core_semicore(
        v_env,
        grid,
        "C",
        center_ang,
        basis="sto-3g",
        xc="lda,vwn",
        z_valence=4.0,
        verbose=False,
    )
    assert mf.converged
    assert mf.mol.nelectron == 2
    assert mf._casidapy_reconstruct["shells"] == ["1s"]
    core = core_from_mf(mf, edge="K", edge_atom_indices=[0])
    assert core.energies.size == 1
    assert core.energies[0] < -5.0


def test_align_core_to_reference():
    pytest.importorskip("pyscf")
    from casidapy.xas import CoreOrbitals
    from casidapy.xas import align_core_to_reference, apply_core_gauge_shift

    core = CoreOrbitals(
        energies=np.array([-18.0, -17.5]),
        mo_coeff=np.eye(2),
        fragment_mol=None,
        atom_indices=np.array([0]),
    )
    aligned, shift = align_core_to_reference(core, -18.73, index=0)
    assert shift == pytest.approx(-0.73)
    assert aligned.energies[0] == pytest.approx(-18.73)
    bumped = apply_core_gauge_shift(aligned, 0.1)
    assert bumped.energies[0] == pytest.approx(-18.63)


def test_read_upf_rhoatom_oxygen():
    from pathlib import Path
    from casidapy.embed import read_upf_rhoatom

    upf = Path("/projectsn/mp1009_1/am4655/casidapy/tutorials/O_ONCV_PBE-1.2.upf")
    if not upf.is_file():
        pytest.skip("O UPF not available")
    r, rho = read_upf_rhoatom(str(upf))
    assert r.size == rho.size
    assert np.all(rho >= 0.0)
    q = float(np.trapz(4.0 * np.pi * r**2 * rho, r))
    assert q == pytest.approx(6.0, rel=1e-2)


def test_shift_v_env_gauge_at_mean():
    pytest.importorskip("dftpy")
    from dftpy.grid import DirectGrid
    from dftpy.field import DirectField
    from casidapy.embed import shift_v_env_gauge_at

    grid = DirectGrid(lattice=np.eye(3) * 10.0, nr=[8, 8, 8])
    v3 = np.ones(grid.nr, dtype=float) * 2.5
    v = DirectField(grid=grid, rank=1, griddata_3d=v3)
    out, shift = shift_v_env_gauge_at(v, grid, np.zeros(3), mode="mean")
    assert shift == pytest.approx(2.5)
    assert float(np.mean(np.asarray(out))) == pytest.approx(0.0, abs=1e-12)


def test_vloc_atom_from_qepy_sums_to_local_pp():
    """Single-atom V_loc from QE strf+setlocal must sum to get_local_pp()."""
    pytest.importorskip("qepy")
    from pathlib import Path
    import sys

    sys.path.insert(0, "/projectsn/mp1009_1/am4655/CasidaPy_tests/PW")
    from pw_bench_common import run_graphene_scf, DEFAULT_PSEUDO
    from casidapy.embed import vloc_atom_from_qepy

    if not Path(DEFAULT_PSEUDO).is_file():
        pytest.skip("C UPF missing")
    workdir = Path("/scratch/am4655/tmp/test_vloc_atom")
    workdir.mkdir(parents=True, exist_ok=True)
    driver, _ = run_graphene_scf(
        1, 1, ecutwfc=25.0, n_occ=2, n_virt=4, workdir=workdir, pseudo_path=DEFAULT_PSEUDO
    )
    try:
        v_all = np.asarray(driver.get_local_pp(), dtype=float)
        if v_all.ndim == 2:
            v_all = v_all[:, 0]
        n_at = len(driver.get_ions_symbols())
        v_sum = np.zeros_like(v_all)
        for ia in range(n_at):
            v_sum += vloc_atom_from_qepy(driver, ia)
        assert np.max(np.abs(v_sum - v_all)) < 1e-10
        v_all2 = np.asarray(driver.get_local_pp(), dtype=float)
        if v_all2.ndim == 2:
            v_all2 = v_all2[:, 0]
        assert np.allclose(v_all, v_all2)
    finally:
        driver.stop()
