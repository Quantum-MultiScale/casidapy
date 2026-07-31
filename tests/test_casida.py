"""
Tests for CasidaPy core functionality.

Derived from the workflow in scripts/test.ipynb (Al FCC unit cell with QEpy).
Tests here cover the utility and slicing logic without requiring QEpy/QE.
"""

import warnings

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

    @pytest.mark.parametrize("xc", ["lda,vwn", "pbe"])
    def test_gto_kernel_matches_pyscf_rpa(self, xc):
        """Pure-functional full TDDFT (matrix-free RPA) must match PySCF TDDFT."""
        pytest.importorskip("pyscf")
        from pyscf import gto, dft
        from casidapy.pyscf_adapter import extract_gto_kernel
        from casidapy.casida_engine import run_casida

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", verbose=0)
        mf = dft.RKS(mol)
        mf.xc = xc
        mf.kernel()

        td = mf.TDDFT()
        td.nstates = 2
        e_ref = td.kernel()[0]

        kernel, opts = extract_gto_kernel(
            mf, n_states=2, tda=False, use_df=False, verbose=False
        )
        opts.solver_method = "eigsh"
        opts.solver_maxiter = 300
        results = run_casida(kernel, opts)

        n = min(len(results.omega), len(e_ref))
        np.testing.assert_allclose(results.omega[:n], e_ref[:n], atol=1e-6)
        assert results.xpy is not None
        assert results.xpy.shape[1] == n
        assert results.metadata.get("hybrid_rpa_dense") is False

    @pytest.mark.parametrize("xc", ["pbe0", "b3lyp", "hf"])
    def test_gto_kernel_matches_pyscf_hybrid_rpa(self, xc):
        """Hybrid/HF full TDDFT via dense A/B must match PySCF TDDFT/TDHF."""
        pytest.importorskip("pyscf")
        from pyscf import gto, dft, scf, tdscf
        from casidapy.pyscf_adapter import extract_gto_kernel
        from casidapy.casida_engine import run_casida

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", verbose=0)
        if xc.lower() == "hf":
            mf = scf.RHF(mol)
        else:
            mf = dft.RKS(mol)
            mf.xc = xc
        mf.kernel()

        td = tdscf.TDDFT(mf)  # dispatches to TDHF for RHF
        td.nstates = 2
        e_ref = td.kernel()[0]

        kernel, opts = extract_gto_kernel(
            mf, n_states=2, tda=False, use_df=False, verbose=False,
            xc=("hf" if xc.lower() == "hf" else xc),
        )
        results = run_casida(kernel, opts)

        n = min(len(results.omega), len(e_ref))
        np.testing.assert_allclose(results.omega[:n], e_ref[:n], atol=1e-6)
        assert results.metadata.get("hybrid_rpa_dense") is True
        assert results.xpy is not None

    def test_gto_hybrid_rpa_rejects_matrix_free_direct(self):
        """Direct matrix-free hybrid RPA must still raise (dense path only)."""
        pytest.importorskip("pyscf")
        from pyscf import gto, dft
        from casidapy.pyscf_adapter import extract_gto_kernel
        from casidapy.casida_engine import CasidaKS_MPI

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", verbose=0)
        mf = dft.RKS(mol)
        mf.xc = "pbe0"
        mf.kernel()
        kernel, _ = extract_gto_kernel(
            mf, n_states=2, tda=False, use_df=False, verbose=False
        )
        casida = object.__new__(CasidaKS_MPI)
        casida.comm = __import__("mpi4py").MPI.COMM_WORLD
        casida.rank = 0
        casida.size = 1
        casida._kernel = kernel
        casida._n_occ = kernel.n_occ
        casida._n_unocc = kernel.n_unocc
        with pytest.raises(NotImplementedError, match="not available matrix-free"):
            casida.setup_matrix_free(tda=False)

    def test_gto_apply_K_matmat_matches_columns(self):
        """Batched matmat must match column-wise apply_K (LOBPCG block path)."""
        pytest.importorskip("pyscf")
        from pyscf import gto, dft
        from casidapy.kernels.gto import GTOKernel

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", verbose=0)
        mf = dft.RKS(mol)
        mf.xc = "pbe"
        mf.kernel()
        # Force on-the-fly response so we exercise batched gen_response
        kernel = GTOKernel(
            mol, mf.mo_coeff, mf.mo_energy, mf.mo_occ,
            xc="pbe", use_df=False, mf=mf, k_cache_max=0, verbose=False,
        )
        kernel.setup(tda=True)
        rng = np.random.default_rng(0)
        V = rng.standard_normal((kernel.n_trans, 4))
        KV_block = kernel.apply_K_matmat(V)
        KV_cols = np.column_stack([kernel.apply_K(V[:, j]) for j in range(4)])
        assert KV_block.shape == V.shape
        assert np.allclose(KV_block, KV_cols, atol=1e-9)


class TestSpinFlipGTO:
    """Collinear exchange-only (Route A) SF-TDDFT in the GTO backend."""

    @staticmethod
    def _triplet_mf(xc="bhandhlyp"):
        from pyscf import gto, dft

        mol = gto.M(atom="C 0 0 0; H 0 0 1.08; H 1.0 0 -0.4",
                    basis="sto-3g", spin=2, charge=0, verbose=0)
        mf = dft.UKS(mol)
        mf.xc = xc
        mf.grids.level = 1  # keep the test fast
        mf.kernel()
        return mol, mf

    @staticmethod
    def _explicit_sf_coupling(mf, hyb):
        """Reference: A_coupling[ia,jb] = -hyb (a b | j i), a,b β-virt; i,j α-occ."""
        mol = mf.mol
        Ca, Cb = mf.mo_coeff
        occa, occb = mf.mo_occ
        C_o = Ca[:, occa > 1e-6]      # alpha occupied
        C_v = Cb[:, occb < 1e-6]      # beta virtual
        no, nv = C_o.shape[1], C_v.shape[1]
        eri = mol.intor("int2e")
        M = np.einsum("pqrs,pa,qb,rj,si->abji", eri, C_v, C_v, C_o, C_o,
                      optimize=True)
        ntr = no * nv
        K = np.zeros((ntr, ntr))
        for i in range(no):
            for a in range(nv):
                for j in range(no):
                    for b in range(nv):
                        K[i * nv + a, j * nv + b] = -hyb * M[a, b, j, i]
        return K

    def test_sf_apply_K_matches_explicit_exchange(self):
        """SF matvec must equal an independent 4-index exchange build."""
        pytest.importorskip("pyscf")
        from casidapy.kernels.gto import GTOKernel

        mol, mf = self._triplet_mf("bhandhlyp")
        kernel = GTOKernel.build_spin_flip(mol, xc="bhandhlyp", use_df=False, mf=mf)
        kernel.setup(tda=True)
        ntr = kernel.n_trans

        K = np.column_stack([kernel.apply_K(np.eye(ntr)[:, j]) for j in range(ntr)])
        assert np.allclose(K, K.T, atol=1e-9)  # real orbitals -> symmetric
        _, _, hyb = kernel._rsh
        K_ref = self._explicit_sf_coupling(mf, hyb)
        np.testing.assert_allclose(K, K_ref, atol=1e-9)

        # Positive spin-flip gaps and a sane cached-K / block path.
        assert np.all(kernel.diagonal_dE() > 0)
        np.testing.assert_allclose(kernel._K, K, atol=1e-9)
        V = np.random.default_rng(0).standard_normal((ntr, 3))
        np.testing.assert_allclose(
            kernel.apply_K_matmat(V),
            np.column_stack([kernel.apply_K(V[:, j]) for j in range(3)]),
            atol=1e-9,
        )

    def test_sf_dipole_is_zero(self):
        """Spin-flip transitions are dipole-forbidden (⟨α|β⟩ = 0)."""
        pytest.importorskip("pyscf")
        from casidapy.kernels.gto import GTOKernel

        mol, mf = self._triplet_mf("bhandhlyp")
        kernel = GTOKernel.build_spin_flip(mol, xc="bhandhlyp", use_df=False, mf=mf)
        kernel.setup(tda=True)
        mu = kernel.dipole_matrix()
        assert mu.shape == (kernel.n_trans, 3)
        assert np.count_nonzero(mu) == 0

    def test_sf_pure_functional_rejected(self):
        """A pure functional gives zero exchange coupling -> must raise."""
        pytest.importorskip("pyscf")
        from casidapy.kernels.gto import GTOKernel

        mol, mf = self._triplet_mf("pbe")
        kernel = GTOKernel.build_spin_flip(mol, xc="pbe", use_df=False, mf=mf)
        with pytest.raises(ValueError, match="hybrid"):
            kernel.setup(tda=True)

    def test_sf_rpa_rejected(self):
        """SF-TDDFT is TDA-only; non-TDA must raise."""
        pytest.importorskip("pyscf")
        from casidapy.kernels.gto import GTOKernel

        mol, mf = self._triplet_mf("bhandhlyp")
        kernel = GTOKernel.build_spin_flip(mol, xc="bhandhlyp", use_df=False, mf=mf)
        with pytest.raises(NotImplementedError, match="TDA"):
            kernel.setup(tda=False)

    def test_sf_xc_route_b_not_implemented(self):
        """The transverse-kernel path (Route B) is gated off for now."""
        pytest.importorskip("pyscf")
        from casidapy.kernels.gto import GTOKernel

        mol, mf = self._triplet_mf("bhandhlyp")
        kernel = GTOKernel.build_spin_flip(
            mol, xc="bhandhlyp", use_df=False, mf=mf, sf_xc=True
        )
        with pytest.raises(NotImplementedError, match="sf_xc"):
            kernel.setup(tda=True)


class TestQEDClosedShellMPI:
    """MPI-aware closed-shell QED-TDA matrix build (row distribution)."""

    @staticmethod
    def _h2_kernel(*, k_cache_max=0):
        from pyscf import gto, dft
        from casidapy.kernels.gto import GTOKernel

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", verbose=0)
        mf = dft.RKS(mol)
        mf.xc = "pbe"
        mf.kernel()
        kernel = GTOKernel(
            mol, mf.mo_coeff, mf.mo_energy, mf.mo_occ,
            xc="pbe", use_df=False, mf=mf, k_cache_max=k_cache_max, verbose=False,
        )
        kernel.setup(tda=True)
        return kernel

    def test_dse_exchange_rows_match_full(self):
        from casidapy.qed import dse_exchange_matrix, dse_exchange_rows

        rng = np.random.default_rng(0)
        n_o, n_v = 3, 4
        Q_oo = rng.standard_normal((n_o, n_o))
        Q_oo = 0.5 * (Q_oo + Q_oo.T)
        Q_vv = rng.standard_normal((n_v, n_v))
        Q_vv = 0.5 * (Q_vv + Q_vv.T)
        full = dse_exchange_matrix(Q_oo, Q_vv)
        rows = [0, 2, 5, 11]
        part = dse_exchange_rows(Q_oo, Q_vv, rows)
        np.testing.assert_allclose(part, full[rows], atol=1e-12)

    def test_partitioned_A_rows_match_serial_M(self):
        """Simulated multi-rank row ownership must reproduce serial A = M[:n,:n]."""
        pytest.importorskip("pyscf")
        from casidapy.qed import (
            build_qed_tda_matrix,
            dipole_blocks,
            qed_electronic_A_rows,
            _mpi_round_robin_rows,
        )

        kernel = self._h2_kernel(k_cache_max=0)
        lam = (0.0, 0.0, 0.05)
        omega_c = 0.2
        M = build_qed_tda_matrix(kernel, lam, omega_c, include_dse=True)
        n = kernel.n_trans
        q, Q_oo, Q_vv = dipole_blocks(kernel, lam)

        for size in (2, 3, 4):
            A = np.zeros((n, n), dtype=float)
            for rank in range(size):
                rows = _mpi_round_robin_rows(n, rank, size)
                A_loc = qed_electronic_A_rows(
                    kernel, rows, q, Q_oo, Q_vv, include_dse=True,
                )
                for k, ia in enumerate(rows):
                    A[ia, :] = A_loc[k]
            np.testing.assert_allclose(A, M[:n, :n], atol=1e-9)

    def test_mpi_comm_world_size1_matches_serial(self):
        pytest.importorskip("pyscf")
        pytest.importorskip("mpi4py")
        from mpi4py import MPI
        from casidapy.qed import build_qed_tda_matrix, solve_qed_tda

        kernel = self._h2_kernel(k_cache_max=0)
        lam = (0.0, 0.0, 0.04)
        omega_c = 0.15
        M_serial = build_qed_tda_matrix(kernel, lam, omega_c)
        M_mpi = build_qed_tda_matrix(kernel, lam, omega_c, comm=MPI.COMM_WORLD)
        np.testing.assert_allclose(M_mpi, M_serial, atol=1e-10)

        res = solve_qed_tda(
            kernel, lam_vec=lam, omega_c=omega_c, nstates=4, comm=MPI.COMM_WORLD,
        )
        assert res.omega.shape == (4,)
        assert res.meta["mpi_size"] == MPI.COMM_WORLD.Get_size()
        assert np.all(np.isfinite(res.omega))


class TestQEDSpinFlip:
    """QED-SF-TDA: Δd coupling on the collinear SF manifold."""

    def test_delta_matrix_structure_and_symmetry(self):
        pytest.importorskip("pyscf")
        from casidapy.kernels.gto import GTOKernel
        from casidapy.qed import sf_dipole_difference_matrix

        mol, mf = TestSpinFlipGTO._triplet_mf("bhandhlyp")
        kernel = GTOKernel.build_spin_flip(mol, xc="bhandhlyp", use_df=False, mf=mf)
        kernel.setup(tda=True)
        lam = np.array([0.0, 0.0, 0.05])
        delta = sf_dipole_difference_matrix(kernel, lam)
        n = kernel.n_trans
        assert delta.shape == (n, n)
        assert np.allclose(delta, delta.T, atol=1e-10)

        # Explicit check of a few Slater–Condon cases
        n_o, n_v = kernel.n_occ, kernel.n_unocc
        C_o, C_v = kernel._C_o, kernel._C_v
        with mol.with_common_orig((0.0, 0.0, 0.0)):
            dip_ao = mol.intor("int1e_r", comp=3)
        Q_oo = np.einsum("x,xpq,pi,qj->ij", lam, dip_ao, C_o, C_o)
        Q_vv = np.einsum("x,xpq,pa,qb->ab", lam, dip_ao, C_v, C_v)
        # diagonal ia=ia: Q_vv[a,a] - Q_oo[i,i]
        for i in range(n_o):
            for a in range(n_v):
                ia = i * n_v + a
                assert abs(delta[ia, ia] - (Q_vv[a, a] - Q_oo[i, i])) < 1e-10

    def test_lambda_zero_recovers_electronic_sf_and_shift(self):
        """λ=0 → eigenvalues are {ω_SF} ∪ {ω_SF + ω_c}."""
        pytest.importorskip("pyscf")
        from casidapy.kernels.gto import GTOKernel
        from casidapy.qed import solve_qed_sf_tda

        mol, mf = TestSpinFlipGTO._triplet_mf("bhandhlyp")
        kernel = GTOKernel.build_spin_flip(mol, xc="bhandhlyp", use_df=False, mf=mf)
        kernel.setup(tda=True)
        n = kernel.n_trans
        A = kernel._K + np.diag(kernel.diagonal_dE())
        w_sf = np.sort(np.linalg.eigvalsh(A))

        omega_c = 0.15
        res = solve_qed_sf_tda(
            kernel,
            lam_vec=(0.0, 0.0, 0.0),
            omega_c=omega_c,
            nstates=2 * n,
        )
        expected = np.sort(np.concatenate([w_sf, w_sf + omega_c]))
        np.testing.assert_allclose(res.omega, expected, atol=1e-8)
        # Pure electronic roots (lowest n if ω_c large enough vs gaps): photon ~0
        # With ω_c=0.15, check that photon_frac is ~0 or ~1 for each root
        assert np.all((res.photon_frac < 1e-8) | (res.photon_frac > 1.0 - 1e-8))
        assert res.meta["spin_flip"] is True
        assert res.X.shape == (2 * n, 2 * n)

    def test_qed_sf_smoke_and_guards(self):
        pytest.importorskip("pyscf")
        from casidapy.kernels.gto import GTOKernel
        from casidapy.qed import (
            build_qed_tda_matrix,
            solve_qed_sf_tda,
            solve_qed_tda,
            scan_qed_sf_lambda,
        )

        mol, mf = TestSpinFlipGTO._triplet_mf("bhandhlyp")
        kernel = GTOKernel.build_spin_flip(mol, xc="bhandhlyp", use_df=False, mf=mf)
        kernel.setup(tda=True)

        res = solve_qed_sf_tda(
            kernel,
            lam_vec=(0.0, 0.0, 0.05),
            omega_c=0.1,
            nstates=4,
        )
        assert res.omega.shape == (4,)
        assert np.all(np.isfinite(res.omega))
        assert np.all(res.omega[:-1] <= res.omega[1:] + 1e-12)
        assert np.allclose(res.f, 0.0)
        assert np.all(res.photon_frac >= -1e-12)
        assert np.all(res.photon_frac <= 1.0 + 1e-12)

        with pytest.raises(ValueError, match="spin-flip"):
            solve_qed_tda(kernel, lam_vec=(0, 0, 0.05), omega_c=0.1)
        with pytest.raises(ValueError, match="spin-flip"):
            build_qed_tda_matrix(kernel, (0, 0, 0.05), 0.1)

        scan = scan_qed_sf_lambda(
            kernel,
            lam_scalars=[0.0, 0.03],
            polarization=(0, 0, 1),
            omega_c=0.1,
            nstates=3,
            track=True,
        )
        assert scan["omega_tracked"].shape == (2, 3)


class TestTrackStates:
    """Geometry vs λ tracking without a full SCF."""

    @staticmethod
    def _res(omega, phot, X=None):
        from casidapy.qed import QEDResults

        omega = np.asarray(omega, dtype=float)
        n = omega.shape[0]
        if X is None:
            X = np.eye(n)
        return QEDResults(
            omega=omega,
            X=np.asarray(X, dtype=float),
            m=np.zeros(n),
            f=np.zeros(n),
            photon_frac=np.asarray(phot, dtype=float),
        )

    def test_energy_tracking_follows_adiabatic_order(self):
        from casidapy.qed import track_states

        pts = [
            self._res([1.0, 2.0], [0.0, 1.0]),
            self._res([1.05, 2.1], [0.0, 1.0]),
            self._res([1.9, 2.05], [1.0, 0.0]),
        ]
        omega_t, phot_t = track_states(pts, method="energy", photon_weight=0.0)
        np.testing.assert_allclose(omega_t[:, 0], [1.0, 1.05, 1.9])
        np.testing.assert_allclose(omega_t[:, 1], [2.0, 2.1, 2.05])
        # With photon weight, prefer character continuity across the crossing
        omega_p, phot_p = track_states(pts, method="energy", photon_weight=1.0)
        np.testing.assert_allclose(phot_p[:, 0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(phot_p[:, 1], [1.0, 1.0, 1.0])

    def test_auto_falls_back_on_garbage_overlap(self):
        from casidapy.qed import track_states

        rng = np.random.default_rng(1)
        X1 = rng.normal(size=(2, 2))
        X1, _ = np.linalg.qr(X1)
        pts = [
            self._res([1.0, 2.0], [0.0, 1.0]),
            self._res([1.05, 2.1], [0.0, 1.0], X=X1),
        ]
        with pytest.warns(UserWarning, match="energy"):
            omega_t, _ = track_states(pts, method="auto", overlap_floor=0.99)
        np.testing.assert_allclose(omega_t[:, 0], [1.0, 1.05])


class TestTripletGTO:
    """Closed-shell GTO triplet TDA vs PySCF."""

    def test_triplet_tda_matches_pyscf(self):
        pytest.importorskip("pyscf")
        from pyscf import gto, dft
        from casidapy.pyscf_adapter import extract_gto_kernel
        from casidapy.casida_engine import run_casida

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", verbose=0)
        mf = dft.RKS(mol)
        mf.xc = "pbe"
        mf.kernel()

        td = mf.TDA()
        td.singlet = False
        td.nstates = 2
        e_ref = td.kernel()[0]

        kernel, opts = extract_gto_kernel(
            mf, n_states=2, tda=True, use_df=False, spin_state="triplet",
        )
        opts.solver_method = "eigsh"
        results = run_casida(kernel, opts)
        n = min(len(results.omega), len(e_ref))
        np.testing.assert_allclose(results.omega[:n], e_ref[:n], atol=1e-6)
        assert np.allclose(results.f[:n], 0.0, atol=1e-12)
        assert kernel.triplet is True


class TestSOC:
    """One-electron SI-SOC helpers."""

    def test_soc_ao_hermitian(self):
        pytest.importorskip("pyscf")
        from pyscf import gto
        from casidapy.soc import soc_ao_integrals

        mol = gto.M(atom="C 0 0 0; O 0 0 1.2", basis="sto-3g", verbose=0)
        h = soc_ao_integrals(mol)
        assert h.shape[0] == 3
        for k in range(3):
            assert np.allclose(h[k], h[k].conj().T, atol=1e-10)

    def test_si_recovers_uncoupled_when_h_zero(self):
        from casidapy.soc import build_soc_si_matrix

        omega_s = np.array([0.1, 0.2])
        omega_t = np.array([0.15])
        H_st = np.zeros((3, 2, 1), dtype=complex)
        H, meta = build_soc_si_matrix(omega_s, omega_t, H_st)
        w = np.linalg.eigvalsh(H)
        expected = np.sort([0.1, 0.2, 0.15, 0.15, 0.15])
        np.testing.assert_allclose(w, expected, atol=1e-12)
        assert meta["n_s"] == 2 and meta["n_t"] == 1

    def test_soc_si_h2co_smoke(self):
        """Formaldehyde: SOC mixes S/T; few-level QED sees borrowed dipoles."""
        pytest.importorskip("pyscf")
        from pyscf import gto, dft
        from casidapy.pyscf_adapter import extract_gto_kernel
        from casidapy.casida_engine import run_casida
        from casidapy.soc import solve_soc_si, solve_soc_qed_levels

        mol = gto.M(
            atom="""
            C  0.000000  0.000000  0.000000
            O  0.000000  0.000000  1.210000
            H  0.000000  0.940000 -0.580000
            H  0.000000 -0.940000 -0.580000
            """,
            basis="sto-3g",
            verbose=0,
        )
        mf = dft.RKS(mol)
        mf.xc = "pbe"
        mf.grids.level = 1
        mf.kernel()

        ks, opts_s = extract_gto_kernel(
            mf, n_states=3, tda=True, use_df=False, spin_state="singlet",
        )
        kt, opts_t = extract_gto_kernel(
            mf, n_states=3, tda=True, use_df=False, spin_state="triplet",
        )
        opts_s.solver_method = opts_t.solver_method = "eigsh"
        res_s = run_casida(ks, opts_s)
        res_t = run_casida(kt, opts_t)

        soc = solve_soc_si(res_s, res_t, ks, include_ground=False)
        assert soc.omega.size == 3 + 3 * 3  # S + 3 Cartesian T
        assert np.all(np.isfinite(soc.omega))
        assert np.all(soc.singlet_weight + soc.triplet_weight > 0.99)

        qed = solve_soc_qed_levels(
            soc, lam_vec=(0.0, 0.0, 0.05), omega_c=float(res_s.omega[0]),
            nstates=4,
        )
        assert qed["omega"].shape[0] == 5  # 4 electronic + |S0,1⟩
        assert np.all(np.isfinite(qed["omega"]))
        # Bright root should mix with the cavity (non-trivial photon weight)
        assert float(np.max(qed["photon_frac"])) > 0.01
        assert "f" in qed and qed["f"].shape == qed["omega"].shape
        assert qed["model"] == "tavis-cummings"
        # Pure photonic character should not dominate oscillator strength
        phot_dom = qed["photon_frac"] > 0.9
        if np.any(phot_dom):
            assert float(np.max(qed["f"][phot_dom])) < 1e-8

        from casidapy.soc import (
            solve_soc_qed,
            solve_soc_qed_pf,
            build_soc_qed_pf_matrix,
        )

        jc = solve_soc_qed(
            soc,
            lam_vec=(0.0, 0.0, 0.05),
            omega_c=float(res_s.omega[0]),
            model="jc",
        )
        assert jc["model"] == "jaynes-cummings"
        assert jc["omega"].shape[0] == 2

        tc = solve_soc_qed(
            soc,
            lam_vec=(0.0, 0.0, 0.05),
            omega_c=float(res_s.omega[0]),
            model="tc",
            nstates=4,
            prefer_bright=True,
        )
        assert tc["model"] == "tavis-cummings"
        np.testing.assert_allclose(tc["omega"], qed["omega"], atol=1e-12)

        # λ→0 PF recovers {0, E_k, ω_c, E_k+ω_c}
        e = np.array([0.0, 0.1, 0.2])
        d = np.zeros((3, 3))
        M0 = build_soc_qed_pf_matrix(e, d, omega_c=0.15, include_dse=False)
        w0 = np.linalg.eigvalsh(M0)
        expected = np.sort([0.0, 0.1, 0.2, 0.15, 0.25, 0.35])
        np.testing.assert_allclose(w0, expected, atol=1e-12)

        pf = solve_soc_qed_pf(
            soc,
            lam_vec=(0.0, 0.0, 0.05),
            omega_c=float(res_s.omega[0]),
            nstates=4,
            include_dse=True,
            prefer_bright=True,
        )
        # N = 1 (S0) + 4 SOC → 2N = 10
        assert pf["omega"].shape[0] == 10
        assert pf["model"] == "pauli-fierz"
        assert abs(pf["omega"][0]) < 1e-10
        assert float(np.max(pf["photon_frac"])) > 0.01
        assert np.all(np.isfinite(pf["omega"]))
        assert "f" in pf and pf["f"].shape == pf["omega"].shape
        # Dark / nearly-pure photonic roots must not dominate the spectrum
        assert float(np.max(pf["f"])) > 0.0
        bright_mask = pf["f"] > 0.05 * float(np.max(pf["f"]))
        # At least one bright root; not every root is bright
        assert np.any(bright_mask)
        assert np.count_nonzero(bright_mask) < pf["f"].size

        # λ=0: electronic f matches SI-SOC; pure 1-photon replicas are dark
        pf0 = solve_soc_qed_pf(
            soc,
            lam_vec=(0.0, 0.0, 0.0),
            omega_c=float(res_s.omega[0]),
            nstates=3,
            include_dse=False,
            prefer_bright=False,
        )
        # Highest photon-weight states should carry negligible oscillator strength
        phot_dom = pf0["photon_frac"] > 0.9
        if np.any(phot_dom):
            assert float(np.max(pf0["f"][phot_dom])) < 1e-8

    def test_qed_post_on_tddft_synthetic(self):
        """JC/TC/PF post-processing on fake TDDFT eigenstates."""
        from types import SimpleNamespace
        from casidapy.qed import solve_qed_post

        omega = np.array([0.15, 0.22, 0.30])
        mu = np.array([[0.0, 0.0, 1.0], [0.5, 0.0, 0.0], [0.0, 0.1, 0.0]])
        res = SimpleNamespace(omega=omega, f=np.array([0.5, 0.2, 0.01]), d_mode=mu.T)
        lam = (0.0, 0.0, 0.05)
        wc = 0.15

        jc = solve_qed_post(res, lam, wc, model="jc")
        assert jc["model"] == "jaynes-cummings"
        assert jc["omega"].shape == (2,)
        assert jc["postprocess"] is True

        tc = solve_qed_post(res, lam, wc, model="tc", nstates=2)
        assert tc["model"] == "tavis-cummings"
        assert tc["omega"].shape == (3,)

        pf = solve_qed_post(res, lam, wc, model="pf", nstates=2, include_dse=True)
        assert pf["model"] == "pauli-fierz"
        assert pf["omega"].shape == (6,)  # N=1+2 → 2N
        assert np.all(np.isfinite(pf["f"]))
        # Ground excitation energy is zero by construction
        assert abs(float(pf["omega"][0])) < 1e-12


def _make_synthetic_pw_kernel(use_gpu=False):
    """Small PlaneWaveKernel on a non-cubic DFTpy grid with Gaussian orbitals.

    No QEpy needed; used to validate the spectral Hartree solve and the GPU
    matvec path against the CPU reference.
    """
    from dftpy.grid import DirectGrid
    from dftpy.field import DirectField
    from dftpy.functional.xc import XC
    from casidapy.kernels import PlaneWaveKernel

    lattice = np.array([
        [8.0, 0.4, 0.0],
        [0.0, 9.0, 0.3],
        [0.2, 0.0, 10.0],
    ])
    grid = DirectGrid(lattice=lattice, nr=[18, 20, 24])
    r = np.asarray(grid.r)  # (3, nx, ny, nz)
    center = lattice.sum(axis=0) / 2.0
    dr2 = sum((r[k] - center[k]) ** 2 for k in range(3))

    rho = np.exp(-dr2 / 2.0) + 0.02
    rho_field = DirectField(grid, rank=1, griddata_3d=rho)

    # Smooth, linearly independent "orbitals": Gaussian × low-order polynomials
    def orb(fn):
        arr = fn * np.exp(-dr2 / 3.0)
        arr /= np.sqrt(np.sum(arr**2) * grid.dV)
        return DirectField(grid, rank=1, griddata_3d=arr)

    psi_occ = [orb(1.0), orb(r[0] - center[0])]
    psi_unocc = [orb(r[1] - center[1]), orb(r[2] - center[2]),
                 orb((r[0] - center[0]) * (r[1] - center[1]))]

    kernel = PlaneWaveKernel(rho_field, XC(xc="PBE"), use_gpu=use_gpu)
    kernel.set_active_orbitals(
        [-0.5, -0.4], [0.1, 0.2, 0.3], psi_occ, psi_unocc
    )
    kernel.setup(tda=True)
    return kernel, rho_field


class TestGPUAcceleration:
    """Spectral (4π/G²) Hartree and CuPy matvec parity checks."""

    def test_spectral_hartree_matches_dftpy(self):
        """numpy spectral Hartree must reproduce DFTpy's Hartree functional.

        Validates FFT normalization and reciprocal-lattice conventions on a
        non-orthogonal cell; this is the same code path the GPU uses via CuPy.
        """
        pytest.importorskip("dftpy")
        kernel, rho_field = _make_synthetic_pw_kernel(use_gpu=False)

        vh_ref = np.asarray(kernel._hartree_potential(rho_field))
        vh_spec = kernel._hartree_spectral(np.asarray(rho_field), xp=np)

        assert vh_spec.shape == vh_ref.shape
        # Agreement is limited by FFT round-off differences between DFTpy's
        # backend and numpy (~1e-9 relative); a convention error (transposed
        # lattice, wrong normalization) would show up at O(1).
        scale = np.max(np.abs(vh_ref))
        assert np.max(np.abs(vh_spec - vh_ref)) < 1e-7 * max(scale, 1.0)

    def test_gpu_apply_K_matches_cpu(self):
        """GPU apply_K (CuPy DGEMMs + cuFFT Hartree) must match CPU apply_K."""
        pytest.importorskip("dftpy")
        cupy = pytest.importorskip("cupy")
        try:
            cupy.cuda.Device().compute_capability
        except Exception:
            pytest.skip("CuPy installed but no usable CUDA device")

        kernel_cpu, _ = _make_synthetic_pw_kernel(use_gpu=False)
        kernel_gpu, _ = _make_synthetic_pw_kernel(use_gpu=True)

        rng = np.random.default_rng(42)
        for _ in range(3):
            v = rng.standard_normal(kernel_cpu.n_trans)
            Kv_cpu = kernel_cpu.apply_K(v)
            Kv_gpu = kernel_gpu.apply_K(v)
            # CPU uses DFTpy's Hartree, GPU the spectral solve: agreement is
            # limited by FFT round-off (~1e-9 relative), not by physics.
            assert np.allclose(Kv_gpu, Kv_cpu, rtol=1e-6, atol=1e-8)

    def test_use_gpu_without_cupy_raises(self):
        """use_gpu=True must fail loudly (ImportError) when CuPy is missing."""
        pytest.importorskip("dftpy")
        try:
            import cupy  # noqa: F401
            pytest.skip("CuPy is installed; cannot test the missing-CuPy error")
        except ImportError:
            pass

        with pytest.raises(ImportError, match="requires CuPy"):
            _make_synthetic_pw_kernel(use_gpu=True)


def _h2_rks(xc="pbe"):
    from pyscf import gto, dft

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", verbose=0)
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.kernel()
    return mf


class TestGTOGPUAcceleration:
    """GTO CuPy matvecs (option A) and optional gpu4pyscf response (option B)."""

    def test_gto_use_gpu_without_cupy_raises(self):
        pytest.importorskip("pyscf")
        try:
            import cupy  # noqa: F401
            pytest.skip("CuPy is installed; cannot test the missing-CuPy error")
        except ImportError:
            pass
        from casidapy.pyscf_adapter import extract_gto_kernel

        mf = _h2_rks()
        with pytest.raises(ImportError, match="requires CuPy"):
            extract_gto_kernel(mf, n_states=2, tda=True, use_gpu=True)

    def test_gto_gpu_cached_apply_K_matches_cpu(self):
        """Cached K @ v on GPU must match the CPU DGEMM."""
        pytest.importorskip("pyscf")
        cupy = pytest.importorskip("cupy")
        try:
            cupy.cuda.Device().compute_capability
        except Exception:
            pytest.skip("CuPy installed but no usable CUDA device")

        from casidapy.pyscf_adapter import extract_gto_kernel

        mf = _h2_rks("pbe")
        k_cpu, _ = extract_gto_kernel(
            mf, n_states=2, tda=True, use_df=False, use_gpu=False, verbose=False
        )
        k_gpu, _ = extract_gto_kernel(
            mf, n_states=2, tda=True, use_df=False, use_gpu=True, verbose=False
        )
        k_cpu.setup(tda=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k_gpu.setup(tda=True)

        assert k_cpu._K is not None and k_gpu._K_dev is not None
        rng = np.random.default_rng(0)
        for _ in range(3):
            v = rng.standard_normal(k_cpu.n_trans)
            assert np.allclose(k_gpu.apply_K(v), k_cpu.apply_K(v), atol=1e-10)

    def test_gto_gpu_onthefly_apply_K_matches_cpu(self):
        """On-the-fly path: GPU contractions (+ optional GPU response) vs CPU."""
        pytest.importorskip("pyscf")
        cupy = pytest.importorskip("cupy")
        try:
            cupy.cuda.Device().compute_capability
        except Exception:
            pytest.skip("CuPy installed but no usable CUDA device")

        from casidapy.kernels.gto import GTOKernel

        mf = _h2_rks("pbe0")
        kwargs = dict(
            mol=mf.mol,
            mo_coeff=mf.mo_coeff,
            mo_energy=mf.mo_energy,
            mo_occ=mf.mo_occ,
            xc="pbe0",
            use_df=False,
            mf=mf,
            k_cache_max=0,  # force on-the-fly
            verbose=False,
        )
        k_cpu = GTOKernel(**kwargs, use_gpu=False)
        k_gpu = GTOKernel(**kwargs, use_gpu=True)
        k_cpu.setup(tda=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k_gpu.setup(tda=True)

        rng = np.random.default_rng(1)
        for _ in range(3):
            v = rng.standard_normal(k_cpu.n_trans)
            assert np.allclose(
                k_gpu.apply_K(v), k_cpu.apply_K(v), rtol=1e-6, atol=1e-8
            )

    def test_gto_gpu4pyscf_response_parity(self):
        """When gpu4pyscf is present, GPU response must match CPU gen_response."""
        pytest.importorskip("pyscf")
        pytest.importorskip("gpu4pyscf")
        cupy = pytest.importorskip("cupy")
        try:
            cupy.cuda.Device().compute_capability
        except Exception:
            pytest.skip("CuPy installed but no usable CUDA device")

        from casidapy.kernels.gto import GTOKernel

        mf = _h2_rks("pbe")
        kwargs = dict(
            mol=mf.mol,
            mo_coeff=mf.mo_coeff,
            mo_energy=mf.mo_energy,
            mo_occ=mf.mo_occ,
            xc="pbe",
            use_df=True,
            mf=mf,
            k_cache_max=0,
            verbose=False,
        )
        k_cpu = GTOKernel(**kwargs, use_gpu=False)
        k_gpu = GTOKernel(**kwargs, use_gpu=True)
        k_cpu.setup(tda=True)
        k_gpu.setup(tda=True)
        if not k_gpu._gpu_response:
            pytest.skip("gpu4pyscf imported but gen_response did not activate")

        rng = np.random.default_rng(2)
        v = rng.standard_normal(k_cpu.n_trans)
        assert np.allclose(
            k_gpu.apply_K(v), k_cpu.apply_K(v), rtol=1e-5, atol=1e-7
        )


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
