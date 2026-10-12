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

class _RootOnlyDriver:
    """Driver whose ``gather=True`` getters mimic QEpy's ``create_array``.

    Only the root rank receives the full grid; every other rank gets a
    ``(1, nspin)`` placeholder.
    """

    full = np.arange(12.0).reshape(6, 2)

    def __init__(self, rank, size=2):
        self.nproc = size
        self.is_root = rank == 0
        self._rank = rank
        self.comm = self._Comm(rank, size)

    class _Comm:
        def __init__(self, rank, size):
            self.rank = rank
            self.size = size

        def bcast(self, obj, root=0):
            return obj if self.rank == root else _RootOnlyDriver.full.shape

        def Bcast(self, buf, root=0):
            buf[...] = _RootOnlyDriver.full

    def get_density(self, gather=True):
        return self.full if self._rank == 0 else np.zeros((1, 2))

def test_gathered_qe_array_replicates_root_only_buffer():
    """Every rank must end up with the whole grid, not QEpy's 1-element stub."""
    from casidapy.embed.potential import gathered_qe_array

    for rank in (0, 1):
        driver = _RootOnlyDriver(rank)
        out = gathered_qe_array(driver, driver.get_density)
        assert out.shape == _RootOnlyDriver.full.shape, f"rank {rank} kept the stub"
        np.testing.assert_allclose(out, _RootOnlyDriver.full)

def test_gathered_qe_array_serial_is_passthrough():
    """A single-rank driver must not touch MPI at all."""
    from casidapy.embed.potential import gathered_qe_array

    full = np.arange(6.0).reshape(3, 2)

    class _Serial:
        comm = None
        nproc = 1
        is_root = True

        def get_density(self, gather=True):
            return full

    driver = _Serial()
    np.testing.assert_allclose(gathered_qe_array(driver, driver.get_density), full)


def test_xas_embed_gpu_comm_kwargs():
    import inspect
    from casidapy.embed import build_ae_embedding_potential, build_hirshfeld_embedding
    from casidapy.xas import (
        core_mos_to_pw_fields,
        run_ae_host_pw_grid_cvs,
        run_xas_gto,
    )
    from casidapy.casida_engine import CasidaKS_MPI

    for fn in (
        build_ae_embedding_potential,
        build_hirshfeld_embedding,
        run_ae_host_pw_grid_cvs,
        core_mos_to_pw_fields,
    ):
        params = inspect.signature(fn).parameters
        assert "use_gpu" in params
        assert "comm" in params
    assert "use_mpi_response" in inspect.signature(run_xas_gto).parameters
    xas_params = inspect.signature(CasidaKS_MPI.xas).parameters
    assert "use_gpu" in xas_params and "comm" in xas_params
    assert "gipaw_upf" in xas_params
    assert "edge_atom" in xas_params
    assert "embed" not in xas_params
    assert "reconstruct" not in xas_params
    assert "embed_mode" not in xas_params
    assert "active_atoms" in inspect.signature(build_ae_embedding_potential).parameters
    assert "active_atoms" in inspect.signature(build_hirshfeld_embedding).parameters


def test_retired_xas_kwargs_rejected():
    from casidapy.xas.reconstruct import run_ae_host_pw_grid_cvs

    with pytest.raises(TypeError, match="retired"):
        run_ae_host_pw_grid_cvs(
            None, 0, gipaw_upf="/dev/null", embed_mode="invert"
        )
    with pytest.raises(TypeError, match="retired"):
        run_ae_host_pw_grid_cvs(
            None, 0, gipaw_upf="/dev/null", ae_host_hxc="rho_env"
        )


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

def test_resolve_mol_spin():
    from casidapy.xas.reconstruct import _resolve_mol_spin

    assert _resolve_mol_spin(5, spin=1) == 1
    assert _resolve_mol_spin(5) == 1
    assert _resolve_mol_spin(6, spin=2) == 2
    assert _resolve_mol_spin(10) == 0
    with pytest.raises(ValueError, match="not consistent"):
        _resolve_mol_spin(5, spin=0)

def test_first_peak_shift_ev():
    from types import SimpleNamespace
    from casidapy.xas.spectrum import first_peak_shift_ev, HA_TO_EV

    ref = SimpleNamespace(omega=np.array([0.5, 0.6]))
    res = SimpleNamespace(omega=np.array([0.4, 0.55]))
    assert first_peak_shift_ev(ref, res) == pytest.approx(0.1 * HA_TO_EV)

def test_n_pp_core_orbitals_and_mix_venv():
    pytest.importorskip("dftpy")
    from dftpy.grid import DirectGrid
    from dftpy.field import DirectField
    from casidapy.embed import mix_v_env_fields
    from casidapy.xas.reconstruct import _n_pp_core_orbitals

    assert _n_pp_core_orbitals(8, 6) == 1  # O: 1s
    assert _n_pp_core_orbitals(22, 12) == 5  # Ti: 1s2s2p
    assert _n_pp_core_orbitals(6, 4) == 1  # C: 1s

    grid = DirectGrid(lattice=np.eye(3) * 4.0, nr=[4, 4, 4])
    a = DirectField(grid=grid, rank=1, griddata_3d=np.ones(grid.nr) * 2.0)
    b = DirectField(grid=grid, rank=1, griddata_3d=np.ones(grid.nr) * 4.0)
    m = mix_v_env_fields(a, b, 0.5, grid)
    assert float(np.mean(np.asarray(m))) == pytest.approx(3.0)

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

def test_formal_ion_rhoatom_o2minus_and_ti4():
    """Path A: O²⁻ / Ti⁴⁺ radial refs integrate to valence = z_val − ox."""
    pytest.importorskip("pyscf")
    from pathlib import Path
    from casidapy.embed import clear_formal_ion_rhoatom_cache, formal_ion_rhoatom
    from casidapy.embed.potential import read_upf_z_valence

    clear_formal_ion_rhoatom_cache()
    upf_o = Path("/projectsn/mp1009_1/am4655/casidapy/tutorials/O_ONCV_PBE-1.2.upf")
    if not upf_o.is_file():
        pytest.skip("O UPF not available")
    z_o = float(read_upf_z_valence(str(upf_o)))

    r, rho = formal_ion_rhoatom("O", -2, valence_only=True, z_valence=z_o)
    assert r.size == rho.size
    assert np.all(rho >= 0.0)
    q = float(np.trapz(4.0 * np.pi * r**2 * rho, r))
    assert q == pytest.approx(z_o - (-2), rel=5e-2, abs=0.15)

    # Cache hit returns equal arrays
    r2, rho2 = formal_ion_rhoatom("O", -2, valence_only=True, z_valence=z_o)
    assert np.allclose(r, r2) and np.allclose(rho, rho2)

    r_ti, rho_ti = formal_ion_rhoatom(
        "Ti", 4, valence_only=False, basis="cc-pvdz"
    )
    q_ti = float(np.trapz(4.0 * np.pi * r_ti**2 * rho_ti, r_ti))
    assert q_ti == pytest.approx(18.0, rel=5e-2, abs=0.3)  # Ar-like Ti⁴⁺

    # Bare proton reference is empty
    r_h, rho_h = formal_ion_rhoatom("H", 1, valence_only=False)
    assert float(np.sum(rho_h)) == pytest.approx(0.0)

def test_normalize_hirshfeld_reference():
    from casidapy.embed.hirshfeld import _normalize_hirshfeld_reference

    assert _normalize_hirshfeld_reference("neutral") == "neutral"
    assert _normalize_hirshfeld_reference("upf") == "neutral"
    assert _normalize_hirshfeld_reference("formal_ion") == "formal_ion"
    with pytest.raises(ValueError, match="hirshfeld_reference"):
        _normalize_hirshfeld_reference("hirshfeld_i")

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

def test_shift_v_env_gauge_at_fermi():
    pytest.importorskip("dftpy")
    from dftpy.grid import DirectGrid
    from dftpy.field import DirectField
    from casidapy.embed import shift_v_env_gauge_at

    grid = DirectGrid(lattice=np.eye(3) * 10.0, nr=[8, 8, 8])
    v3 = np.ones(grid.nr, dtype=float) * 2.5
    v = DirectField(grid=grid, rank=1, griddata_3d=v3)
    ef = 0.394396  # Ha (~10.7 eV)
    out, shift = shift_v_env_gauge_at(v, grid, np.zeros(3), mode="fermi", fermi_ha=ef)
    assert shift == pytest.approx(ef)
    assert float(np.mean(np.asarray(out))) == pytest.approx(2.5 - ef, abs=1e-12)

def test_shift_v_env_gauge_at_fermi_requires_ef():
    pytest.importorskip("dftpy")
    from dftpy.grid import DirectGrid
    from dftpy.field import DirectField
    from casidapy.embed import shift_v_env_gauge_at

    grid = DirectGrid(lattice=np.eye(3) * 10.0, nr=[4, 4, 4])
    v = DirectField(grid=grid, rank=1, griddata_3d=np.ones(grid.nr))
    with pytest.raises(ValueError, match="fermi_ha"):
        shift_v_env_gauge_at(v, grid, np.zeros(3), mode="fermi")

def test_classify_vltot_kind_ionic_vs_screened():
    """Whole-grid residual, not a vacuum sample, distinguishes vltot from vrs."""
    from casidapy.embed.potential import classify_vltot_kind, normalize_ae_host_hxc

    v_loc = np.array([-4.0, -1.0, -0.2])
    v_hxc = np.array([2.5, 0.4, 0.01])
    v_eff = v_loc + v_hxc
    kind, rms_ionic, rms_screened = classify_vltot_kind(v_loc, v_eff, v_hxc)
    assert kind == "ionic"
    assert rms_ionic < 1e-12
    assert rms_screened > 0.5

    kind2, rms_i2, rms_s2 = classify_vltot_kind(v_eff, v_eff, v_hxc)
    assert kind2 == "screened"
    assert rms_s2 < 1e-12
    assert rms_i2 > 0.5

    assert normalize_ae_host_hxc("core_plus_tot") == "rho_tot"
    assert normalize_ae_host_hxc("rho_tot") == "rho_tot"
    with pytest.raises(ValueError, match="retired"):
        normalize_ae_host_hxc("host_tot")
    with pytest.raises(ValueError, match="retired"):
        normalize_ae_host_hxc("rho_env")

def test_radial_hartree_hydrogenic_1s2():
    """Spherical Poisson: 1s² (Z=1) has v(0)=2 Ha and r v(r) → 2."""
    from casidapy.xas.core_hxc import radial_hartree_spherical

    r = np.concatenate(([0.0], np.geomspace(1e-5, 40.0, 4000)))
    # Two hydrogenic 1s electrons: ρ = 2/π exp(-2r)
    rho = (2.0 / np.pi) * np.exp(-2.0 * r)
    v = radial_hartree_spherical(r, rho)
    assert v[0] == pytest.approx(2.0, rel=2e-3, abs=2e-3)
    assert v[-1] * r[-1] == pytest.approx(2.0, rel=1e-3, abs=1e-3)
    assert v[r > 1.0][0] < v[0]

def test_becke_core_integral_is_two():
    """Gate: ∫ρ_1s² on the unpruned Treutler–Ahlrichs grid is 2.000."""
    pytest.importorskip("pyscf")
    from pyscf import gto, dft
    from casidapy.xas.core_hxc import (
        build_absorber_core_becke_grid,
        core_density_on_becke,
        radial_hartree_from_core_dm,
        DEFAULT_NELEC_TOL,
    )

    mol = gto.M(atom="C 0 0 0", basis="sto-3g", spin=0, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "pbe"
    mf.verbose = 0
    mf.kernel()
    occ = np.zeros_like(mf.mo_occ)
    occ[0] = 2.0
    dm = mf.make_rdm1(mf.mo_coeff, occ)
    coords, weights, meta = build_absorber_core_becke_grid(mol)
    _rho, nelec, _ao = core_density_on_becke(mol, dm, coords, weights, deriv=0)
    assert nelec == pytest.approx(2.0, abs=DEFAULT_NELEC_TOL)
    r, _rho_r, v_h, _c = radial_hartree_from_core_dm(mol, dm)
    assert v_h[-1] * r[-1] == pytest.approx(2.0, rel=2e-3, abs=2e-3)
    assert np.isfinite(v_h[0]) and v_h[0] > 0.0
    assert meta["prune"] is None
    assert meta["n_radial"] >= 200

def test_ground_state_spin_hund():
    from casidapy.xas.reconstruct import ground_state_spin

    assert ground_state_spin(6) == 2  # C ³P
    assert ground_state_spin(7) == 3  # N ⁴S
    assert ground_state_spin(8) == 2  # O ³P
    assert ground_state_spin(10) == 0  # Ne
    assert ground_state_spin(4) == 0

def test_isolated_carbon_uks_triplet_converges():
    """Neutral C is ³P: UKS must converge and ∫ρ_1s² = 2."""
    pytest.importorskip("pyscf")
    from pyscf import gto
    from casidapy.xas.reconstruct import (
        _isolated_ae_core_hxc_ao,
        _total_ao_dm,
        ground_state_spin,
    )
    from casidapy.xas.core_hxc import (
        build_absorber_core_becke_grid,
        core_density_on_becke,
        DEFAULT_NELEC_TOL,
    )

    mol = gto.M(
        atom="C 0 0 0",
        basis="sto-3g",
        spin=ground_state_spin(6),
        symmetry=False,
        verbose=0,
    )
    _v, meta = _isolated_ae_core_hxc_ao(mol, "pbe", 1, operator="none")
    assert meta["isolated_converged"] is True
    assert meta["isolated_uks"] is True
    assert meta["isolated_spin"] == 2
    assert meta["isolated_magnetization"] == pytest.approx(2.0, abs=0.2)
    assert np.isfinite(meta["eps_isolated_1s_ha"])
    mf = meta["_mf"]
    dm = _total_ao_dm(mf.make_rdm1())
    coords, weights, _g = build_absorber_core_becke_grid(mol)
    _rho, nelec, _ao = core_density_on_becke(mol, dm, coords, weights, deriv=0)
    assert nelec == pytest.approx(2.0, abs=DEFAULT_NELEC_TOL)

def test_potential_ao_matrix_fortran_layout_noncubic():
    """A voxel spike must pair with the Fortran-flat coordinate of that voxel.

    Shape (5,6,7) cannot alias C- vs F-ravel. A C-order projection would put
    the spike on the wrong Cartesian point and the AO matrix would vanish.
    """
    from types import SimpleNamespace

    from pyscf import gto
    from pyscf.dft import numint

    from casidapy.xas.reconstruct import _potential_ao_matrix

    nr = (5, 6, 7)
    d = 0.45
    xs = (np.arange(nr[0]) - nr[0] // 2) * d
    ys = (np.arange(nr[1]) - nr[1] // 2) * d
    zs = (np.arange(nr[2]) - nr[2] // 2) * d
    rx, ry, rz = np.meshgrid(xs, ys, zs, indexing="ij")
    rx, ry, rz = (np.asfortranarray(a) for a in (rx, ry, rz))
    dV = float(d**3)
    grid = SimpleNamespace(nr=nr, r=(rx, ry, rz), dV=dV)

    ix, iy, iz = 1, 4, 2
    v3 = np.zeros(nr, order="F")
    v3[ix, iy, iz] = 1.0 / dV
    pos = np.array([rx[ix, iy, iz], ry[ix, iy, iz], rz[ix, iy, iz]], float)
    mol = gto.M(
        atom=f"O {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}",
        basis="sto-3g",
        unit="Bohr",
        spin=2,
        verbose=0,
    )
    v_ao = _potential_ao_matrix(mol, v3, grid)
    ao = numint.eval_ao(mol, pos.reshape(1, 3))[0]
    expected = np.outer(ao, ao)
    assert np.max(np.abs(v_ao - expected)) < 1e-8
    assert float(ao @ v_ao @ ao) > 0.1


def test_qe_field_gsum_matches_fortran_voxels():
    """G-sum of a Fortran-layout field recovers the voxel values off the mesh origin."""
    from types import SimpleNamespace

    from casidapy.xas.pw_eval import fft_coeffs_from_orbital, pw_sum_at_coords

    nr = (8, 9, 10)
    d = 0.35
    xs = np.arange(nr[0]) * d
    ys = np.arange(nr[1]) * d
    zs = np.arange(nr[2]) * d
    rx, ry, rz = np.meshgrid(xs, ys, zs, indexing="ij")
    rx, ry, rz = (np.asfortranarray(a) for a in (rx, ry, rz))
    lattice = np.diag([nr[0] * d, nr[1] * d, nr[2] * d])
    grid = SimpleNamespace(nr=nr, r=(rx, ry, rz), dV=float(d**3), lattice=lattice)
    v3 = np.asfortranarray(np.cos(2.0 * np.pi * rx / (nr[0] * d)))
    G, c, meta = fft_coeffs_from_orbital(v3, grid, g_cutoff_rel=1e-14, order="F")
    assert meta["fft_order"] == "F"
    pts = np.stack(
        [
            [rx[1, 4, 2], ry[1, 4, 2], rz[1, 4, 2]],
            [rx[3, 0, 7], ry[3, 0, 7], rz[3, 0, 7]],
            [rx[6, 8, 5], ry[6, 8, 5], rz[6, 8, 5]],
        ],
        axis=0,
    )
    got = pw_sum_at_coords(G, c, pts)
    want = np.array([v3[1, 4, 2], v3[3, 0, 7], v3[6, 8, 5]], float)
    assert np.max(np.abs(got - want)) < 1e-10


def test_qe_species_index_uses_ityp_not_first_appearance():
    """Rutile-like: Ti atoms first, ATOMIC_SPECIES is O then Ti.

    First-appearance indexing would peel column 1 at oxygen (Ti v_loc).
    QE ``ityp`` peels column 0 (O).
    """
    from casidapy.embed.potential import _qe_species_index, _qe_species_label

    class _Ions:
        def get_nat(self):
            return 4

        def get_array_ityp(self):
            return np.array([2, 2, 1, 1], dtype=int)

        def get_array_atm(self):
            # QEpy: (3, nsp) Fortran bytes → .T.view('S3')[:,0]
            labels = np.array(["O  ", "Ti "], dtype="S3")
            return labels.view("S1").reshape(2, 3).T

    class _Mods:
        ions_base = _Ions()

    class _Driver:
        qepy_modules = _Mods()

        def get_ions_symbols(self):
            return ["Ti", "Ti", "O", "O"]

    driver = _Driver()
    assert _qe_species_index(driver, 0) == 1
    assert _qe_species_index(driver, 1) == 1
    assert _qe_species_index(driver, 2) == 0
    assert _qe_species_index(driver, 3) == 0
    assert _qe_species_label(driver, 0) == "O"
    assert _qe_species_label(driver, 1) == "Ti"
    seen = []
    for s in driver.get_ions_symbols():
        if s not in seen:
            seen.append(s)
    assert seen.index("O") == 1


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
        from casidapy.embed import classify_qe_local_potential, ensure_charge_grid

        grid = ensure_charge_grid(driver)
        cls = classify_qe_local_potential(driver, grid, edge_atom=0)
        assert cls["kind"] == "ionic", cls
        assert cls["rms_v_eff_minus_v_loc_minus_v_hxc_ha"] < 0.05 * max(
            cls["rms_v_eff_minus_v_loc_ha"], 1e-8
        )
        from casidapy.embed.potential import ionic_vloc_residual

        home_meta = {}
        _v_home, v_ionic_home = ionic_vloc_residual(
            driver, 0, grid, vloc_source="home", peel_meta=home_meta
        )
        _v_q, v_ionic_q = ionic_vloc_residual(driver, 0, grid, vloc_source="qepy")
        phi = float(home_meta["home_peel_image_ha"])
        assert home_meta["vloc_source"] == "home"
        assert np.allclose(
            np.asarray(v_ionic_home, float),
            np.asarray(v_ionic_q, float) + phi,
            atol=1e-8,
        )
    finally:
        driver.stop()


def test_ae_host_calc_type_parameter_bundle():
    from casidapy.xas import (
        AE_HOST_CALC_DEFAULTS,
        ae_host_params_for_calc_type,
        calc_type_from_driver,
        resolve_calc_type,
    )

    assert resolve_calc_type("periodic") == "crystal"
    assert resolve_calc_type("cluster") == "molecule"
    cr = ae_host_params_for_calc_type("crystal")
    mol = ae_host_params_for_calc_type("molecule")
    assert cr == {
        "calc_type": "crystal",
        "vloc_source": "home",
        "gauge_align": "none",
        "assume_isolated": None,
    }
    assert mol == {
        "calc_type": "molecule",
        "vloc_source": "qepy",
        "gauge_align": "none",
        "assume_isolated": "mt",
    }
    assert cr["vloc_source"] != mol["vloc_source"]
    over = ae_host_params_for_calc_type("crystal", vloc_source="qepy")
    assert over["vloc_source"] == "qepy"
    assert over["calc_type"] == "crystal"
    assert AE_HOST_CALC_DEFAULTS["crystal"]["vloc_source"] == "home"
    from types import SimpleNamespace

    crystal = SimpleNamespace(qe_options={"&system": {}})
    cluster = SimpleNamespace(qe_options={"&system": {"assume_isolated": "'mt'"}})
    assert calc_type_from_driver(crystal) == "crystal"
    assert calc_type_from_driver(cluster) == "molecule"


def test_resolve_vloc_source_auto_crystal_vs_cluster():
    from types import SimpleNamespace

    from casidapy.embed.potential import (
        qe_driver_is_isolated,
        resolve_vloc_source,
    )

    crystal = SimpleNamespace(qe_options={"&system": {}})
    cluster = SimpleNamespace(
        qe_options={"&system": {"assume_isolated": "'mt'"}}
    )
    assert qe_driver_is_isolated(crystal) is False
    assert qe_driver_is_isolated(cluster) is True
    assert resolve_vloc_source("auto", crystal) == "home"
    assert resolve_vloc_source("auto", cluster) == "qepy"
    assert resolve_vloc_source("auto", None) == "qepy"
    assert resolve_vloc_source("qepy", crystal) == "qepy"
    assert resolve_vloc_source("home_cell", cluster) == "home"
    assert resolve_vloc_source("setlocal") == "qepy"


def test_home_cell_image_shift_uses_coulomb_shell_not_nucleus():
    """A nucleus spike (QE vs UPF well) must not enter φ_images."""
    from casidapy.embed.potential import (
        home_cell_image_shift_ha,
        upf_coulomb_match_radius,
    )

    r = np.linspace(0.0, 6.0, 61)
    z_val = 6.0
    v = -z_val / np.maximum(r, 0.05)
    v[0] = -15.0
    r_c = upf_coulomb_match_radius(r, v, z_val, atol=0.05)
    assert 0.05 < r_c < 1.0

    n = 40
    dist = np.linspace(0.0, 8.0, n)
    delta = np.full(n, 0.25)
    delta[0] = 5.0
    phi, n_shell = home_cell_image_shift_ha(delta, dist, 1.5, 4.0)
    assert n_shell >= 8
    assert phi == pytest.approx(0.25, abs=1e-12)

    tiny = np.array([3.0, 3.1])
    d_tiny = np.array([0.01, 0.02])
    phi_fb, n_fb = home_cell_image_shift_ha(tiny, d_tiny, 1.5, 4.0)
    assert n_fb == 0
    assert phi_fb == pytest.approx(3.0)

