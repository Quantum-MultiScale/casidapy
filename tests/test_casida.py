"""
Tests for CasidaPy core functionality.

Derived from the workflow in scripts/test.ipynb (Al FCC unit cell with QEpy).
Tests here cover the utility and slicing logic without requiring QEpy/QE.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Unit tests for casida_utils
# ---------------------------------------------------------------------------

class TestBuildEnergyDifferences:
    def test_basic(self):
        from casidapy.casida_utils import build_energy_differences

        occ_e = np.array([-0.5, -0.3])
        unocc_e = np.array([0.1, 0.4, 0.7])
        dE = build_energy_differences(occ_e, unocc_e)

        assert dE.shape == (6,)
        # dE[ia] = unocc_e[a] - occ_e[i]
        assert np.isclose(dE[0], 0.1 - (-0.5))  # i=0, a=0
        assert np.isclose(dE[1], 0.4 - (-0.5))  # i=0, a=1
        assert np.isclose(dE[2], 0.7 - (-0.5))  # i=0, a=2
        assert np.isclose(dE[3], 0.1 - (-0.3))  # i=1, a=0
        assert np.isclose(dE[5], 0.7 - (-0.3))  # i=1, a=2

    def test_single_transition(self):
        from casidapy.casida_utils import build_energy_differences

        occ_e = np.array([0.0])
        unocc_e = np.array([1.0])
        dE = build_energy_differences(occ_e, unocc_e)

        assert dE.shape == (1,)
        assert np.isclose(dE[0], 1.0)


class TestBuildInitialGuess:
    def test_shape_and_unit_vectors(self):
        from casidapy.casida_utils import build_initial_guess

        dE = np.array([0.5, 0.1, 0.8, 0.3])
        k = 3
        x0 = build_initial_guess(dE, k)

        assert x0.shape == (4, 3)
        # The dominant entry in each column should be on a sorted-dE index
        sorted_idx = np.argsort(dE)
        for col in range(k):
            dominant = np.argmax(np.abs(x0[:, col]))
            assert dominant == sorted_idx[col]

    def test_k_larger_than_n_trans(self):
        from casidapy.casida_utils import build_initial_guess

        dE = np.array([1.0, 2.0])
        k = 5
        x0 = build_initial_guess(dE, k)
        assert x0.shape == (2, 5)


# ---------------------------------------------------------------------------
# Unit tests for slice_active_space (qepy_adapter)
# ---------------------------------------------------------------------------

class TestSliceActiveSpace:
    def setup_method(self):
        self.n_orb = 10
        self.eigs = np.linspace(-1.0, 1.0, self.n_orb)
        self.psi_all = list(range(self.n_orb))

    def test_no_total_occ(self):
        """When n_total_occ is None, select the first n_occ orbitals."""
        from casidapy.qepy_adapter import slice_active_space

        n_occ = 4
        n_unocc = 3
        occ_e, unocc_e, psi_occ, psi_unocc = slice_active_space(
            self.eigs, self.psi_all, n_occ, n_unocc, n_total_occ=None
        )

        assert len(occ_e) == 4
        assert len(unocc_e) == 3
        assert len(psi_occ) == 4
        assert len(psi_unocc) == 3
        np.testing.assert_array_equal(occ_e, self.eigs[:4])
        np.testing.assert_array_equal(unocc_e, self.eigs[4:7])

    def test_with_total_occ(self):
        """When n_total_occ is set, select top n_occ from the occupied block."""
        from casidapy.qepy_adapter import slice_active_space

        n_occ = 3
        n_unocc = 2
        n_total_occ = 5
        occ_e, unocc_e, psi_occ, psi_unocc = slice_active_space(
            self.eigs, self.psi_all, n_occ, n_unocc, n_total_occ=n_total_occ
        )

        # Occupied window [n_total_occ - n_occ, n_total_occ) = [2, 5)
        assert len(occ_e) == 3
        np.testing.assert_array_equal(occ_e, self.eigs[2:5])
        assert len(unocc_e) == 2
        np.testing.assert_array_equal(unocc_e, self.eigs[5:7])
        assert psi_occ == [2, 3, 4]
        assert psi_unocc == [5, 6]

    def test_n_unocc_none_uses_all(self):
        """When n_unocc is None, use all available unoccupied orbitals."""
        from casidapy.qepy_adapter import slice_active_space

        n_occ = 4
        occ_e, unocc_e, psi_occ, psi_unocc = slice_active_space(
            self.eigs, self.psi_all, n_occ, n_unocc=None, n_total_occ=None
        )

        assert len(occ_e) == 4
        # n_total_occ defaults to n_occ=4, so unoccupied is eigs[4:]
        assert len(unocc_e) == 6
        np.testing.assert_array_equal(unocc_e, self.eigs[4:])

    def test_invalid_n_occ_raises(self):
        from casidapy.qepy_adapter import slice_active_space

        with pytest.raises(ValueError, match="n_occ must be > 0"):
            slice_active_space(self.eigs, self.psi_all, 0)

    def test_mismatched_lengths_raises(self):
        from casidapy.qepy_adapter import slice_active_space

        with pytest.raises(ValueError, match="len\\(eigs\\)"):
            slice_active_space(self.eigs[:5], self.psi_all, 3)


# ---------------------------------------------------------------------------
# Unit tests for CasidaOptions / CasidaInputs dataclasses
# ---------------------------------------------------------------------------

class TestCasidaAPI:
    def test_options_defaults(self):
        from casidapy.casida_api import CasidaOptions

        opts = CasidaOptions(n_occ=5, n_unocc=5)
        assert opts.n_states == 50
        assert opts.tda is False
        assert opts.matrix_free is False
        assert opts.xc == "PBE"
        assert opts.spin_state == "singlet"
        assert opts.basis == "pw"

    def test_inputs_creation(self):
        from casidapy.casida_api import CasidaInputs

        inputs = CasidaInputs(
            atoms=None,
            grid=None,
            rho_ks=None,
            psi=np.zeros((5, 10, 10, 10)),
            eigs=np.zeros(5),
            occs=np.ones(5),
        )
        assert inputs.psi.shape == (5, 10, 10, 10)


# ---------------------------------------------------------------------------
# Kernel backend unit tests
# ---------------------------------------------------------------------------

class TestKernelBackends:
    def test_plane_wave_kernel_import(self):
        from casidapy.kernels import PlaneWaveKernel, GTOKernel, KernelBackend
        assert PlaneWaveKernel is not None
        assert GTOKernel is not None
        assert KernelBackend is not None

    def test_gto_kernel_requires_pyscf_or_runs(self):
        pytest.importorskip("pyscf")
        from pyscf import gto, dft
        from casidapy.pyscf_adapter import extract_gto_kernel
        from casidapy.casida_engine import run_casida

        # Use 6-31G so n_trans > 1; eigsh cannot solve k >= N for LinearOperator.
        mol = gto.M(
            atom="H 0 0 0; H 0 0 0.74",
            basis="6-31g",
            verbose=0,
        )
        mf = dft.RKS(mol)
        mf.xc = "lda,vwn"
        mf.kernel()
        kernel, opts = extract_gto_kernel(
            mf, n_states=2, tda=True, use_df=False, verbose=False
        )
        opts.solver_method = "eigsh"
        opts.solver_maxiter = 100
        results = run_casida(kernel, opts)
        assert results.omega.size >= 1
        assert np.all(np.asarray(results.omega) > 0)
        assert results.metadata.get("kernel") == "GTOKernel"
        assert results.metadata.get("basis") == "gto"

    @pytest.mark.parametrize("xc", ["pbe", "pbe0", "b3lyp"])
    def test_gto_kernel_matches_pyscf_tda(self, xc):
        """GTO TDA must reproduce PySCF TDA, including hybrids."""
        pytest.importorskip("pyscf")
        from pyscf import gto, dft
        from casidapy.pyscf_adapter import extract_gto_kernel
        from casidapy.casida_engine import run_casida

        # Small H2 system: the matrix-free solver calls gen_response per
        # matvec, so larger molecules make this test very slow.
        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", verbose=0)
        mf = dft.RKS(mol)
        mf.xc = xc
        mf.kernel()

        td = mf.TDA()
        td.nstates = 2
        e_ref = td.kernel()[0]

        kernel, opts = extract_gto_kernel(
            mf, n_states=2, tda=True, use_df=False, verbose=False
        )
        opts.solver_method = "eigsh"
        opts.solver_maxiter = 300
        results = run_casida(kernel, opts)

        n = min(len(results.omega), len(e_ref))
        np.testing.assert_allclose(results.omega[:n], e_ref[:n], atol=1e-6)

    def test_gto_kernel_hybrid_rejects_rpa(self):
        """Hybrid XC + full Casida (non-TDA) is unsupported and must raise."""
        pytest.importorskip("pyscf")
        from pyscf import gto, dft
        from casidapy.pyscf_adapter import extract_gto_kernel
        from casidapy.casida_engine import run_casida

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", verbose=0)
        mf = dft.RKS(mol)
        mf.xc = "pbe0"
        mf.kernel()
        kernel, opts = extract_gto_kernel(
            mf, n_states=2, tda=False, use_df=False, verbose=False
        )
        opts.solver_method = "eigsh"
        with pytest.raises(NotImplementedError, match="requires TDA"):
            run_casida(kernel, opts)


# ---------------------------------------------------------------------------
# Integration test (requires QEpy + MPI + pseudopotential)
# ---------------------------------------------------------------------------

qepy_available = False
try:
    from qepy.driver import Driver
    from dftpy.functional.xc import XC
    from dftpy.field import DirectField
    from mpi4py import MPI
    qepy_available = True
except ImportError:
    pass


@pytest.mark.skipif(not qepy_available, reason="QEpy or DFTpy not available")
class TestFullCasidaWorkflow:
    """Full Casida workflow from QEpy SCF (Al FCC), mirrors scripts/test.ipynb."""

    @pytest.fixture(autouse=True)
    def setup_driver(self, tmp_path):
        from pathlib import Path
        from qepy.driver import Driver

        pseudo_dir = Path(__file__).resolve().parent.parent / "scripts"
        qe_options = {
            '&control': {
                'calculation': "'scf'",
                'pseudo_dir': f"'{pseudo_dir}/'",
            },
            '&system': {
                'ibrav': 0,
                'degauss': 0.005,
                'ecutwfc': 30,
                'nat': 1,
                'ntyp': 1,
                'occupations': "'smearing'",
            },
            'atomic_positions crystal': ['Al    0.0  0.0  0.0'],
            'atomic_species': ['Al  26.98 Al_ONCV_PBE-1.2.upf'],
            'k_points automatic': ['2 2 2 1 1 1'],
            'cell_parameters angstrom': [
                '0.     2.025  2.025',
                '2.025  0.     2.025',
                '2.025  2.025  0.   ',
            ],
        }
        self.driver = Driver(qe_options=qe_options, logfile=False)
        self.driver.scf()
        self.atoms = self.driver.get_ase_atoms()

    def test_run_casida_in_memory(self):
        from casidapy.casida_engine import run_casida_in_memory
        from casidapy.qepy_adapter import extract_casida_inputs_from_qepy_driver

        inputs, options = extract_casida_inputs_from_qepy_driver(
            self.driver, self.atoms
        )
        results = run_casida_in_memory(inputs, options)

        assert results.omega is not None
        assert np.isclose(results.omega[0], 0.5026924502910796, rtol=0, atol=1e-8)
        assert len(results.omega) > 0
        assert np.all(results.omega > 0), "All excitation energies should be positive"

    def test_manual_workflow(self):
        from casidapy.casida_engine import CasidaKS_MPI
        from casidapy.casida_utils import normalize_wavefunctions
        from casidapy.qepy_adapter import (
            extract_casida_inputs_from_qepy_driver,
            slice_active_space,
        )
        from dftpy.functional.xc import XC
        from dftpy.field import DirectField

        inputs, _ = extract_casida_inputs_from_qepy_driver(
            self.driver, self.atoms
        )

        xc_func = XC(xc='PBE')
        casida = CasidaKS_MPI(inputs.rho_ks, xc_func)

        ions = self.driver.get_dftpy_ions()
        psi_list = [
            DirectField(grid=inputs.grid, data=inputs.psi[i])
            for i in range(len(inputs.psi))
        ]
        psi_list = [psi / np.sqrt(ions.cell.volume) for psi in psi_list]
        psi_list = normalize_wavefunctions(psi_list, inputs.grid)

        occs = inputs.occs
        n_occ = int(sum(occs))
        n_unocc = int(len(occs) - n_occ)

        occ_e, unocc_e, psi_occ, psi_unocc = slice_active_space(
            inputs.eigs, psi_list, n_occ, n_unocc, n_total_occ=None
        )

        casida.set_active_orbitals(occ_e, unocc_e, psi_occ, psi_unocc)
        casida.build_matrices(tda=False)
        omega, Z = casida.solve(k=n_occ * n_unocc)

        assert len(omega) == n_occ * n_unocc
        assert np.all(omega > 0)
