"""Tests for the CPU-native ground<->excited TDA NAC path (casidapy.nac_gto).

Requires PySCF (skipped otherwise). Uses small H2/6-31G and LiH/STO-3G
RHF/TDA systems, matching the conventions in scripts/test_gto_hyb.py.

These are correctness *sanity checks* (translational invariance, internal
consistency of the returned NACResults fields) rather than a comparison to
an independent reference value -- there is no CPU reference implementation
of TDDFT NAC to compare against. Where gpu4pyscf + a CUDA GPU are available,
cross-check numerically against casidapy.nac.solve_nac(..., method="tda").

Note on translational invariance: only the **ETF-corrected** coupling
(``de_etf``) is guaranteed translationally invariant (sum over atoms = 0);
the raw CIS-force-matrix-element form (``de``) is not, by construction, in
standard TDDFT-NAC theory (Fatehi & Subotnik; Zhang & Herbert) -- so ``de``
is deliberately *not* asserted to vanish here. H2's homonuclear symmetry can
mask a broken atom-loop (e.g. an accidental atom0==atom1 bug still "looks"
translationally invariant after halving); LiH has no such symmetry and is
the primary check.
"""
import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")

from pyscf import gto, scf

from casidapy.pyscf_adapter import extract_gto_kernel
from casidapy.casida_engine import run_casida
from casidapy.nac_gto import solve_nac_cpu


def _build_system(atom, basis, n_states=3):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    kernel, opts = extract_gto_kernel(
        mf, n_states=n_states, tda=True, use_df=False, verbose=False,
    )
    opts.solver_method = "eigsh"
    casida_res = run_casida(kernel, opts)
    return kernel, casida_res


def _h2_system(n_states=3):
    return _build_system("H 0 0 0; H 0 0 0.74", "6-31g", n_states)


def _lih_system(n_states=2):
    return _build_system("Li 0 0 0; H 0 0 1.6", "sto-3g", n_states)


class TestGroundExcitedNACSanity:
    def test_translational_invariance_lih(self):
        """LiH (no point-group symmetry to accidentally mask a bug): sum over
        atoms of the ETF-corrected NACV must vanish."""
        kernel, casida_res = _lih_system()
        res = solve_nac_cpu(kernel, casida_res, states=(0, 1))

        assert np.allclose(res.de_etf.sum(axis=0), 0.0, atol=1e-8)

    def test_translational_invariance_h2(self):
        kernel, casida_res = _h2_system()
        res = solve_nac_cpu(kernel, casida_res, states=(0, 1))

        assert np.allclose(res.de_etf.sum(axis=0), 0.0, atol=1e-8)

    def test_scaled_consistency(self):
        """de_scaled * EI == de and de_etf_scaled * EI == de_etf by
        construction (NACResults' documented scaling relationship)."""
        kernel, casida_res = _lih_system()
        res = solve_nac_cpu(kernel, casida_res, states=(0, 1))
        EI = res.omega[1]

        assert np.allclose(res.de_scaled * EI, res.de, rtol=1e-10)
        assert np.allclose(res.de_etf_scaled * EI, res.de_etf, rtol=1e-10)

    def test_state_order_antisymmetric(self):
        """<0|d/dR|I> = -<I|d/dR|0>: swapping the pair should negate de."""
        kernel, casida_res = _lih_system()
        res_01 = solve_nac_cpu(kernel, casida_res, states=(0, 1))
        res_10 = solve_nac_cpu(kernel, casida_res, states=(1, 0))

        assert np.allclose(res_01.de, -res_10.de, rtol=1e-10)
        assert res_01.states == (0, 1)
        assert res_10.states == (1, 0)

    def test_excited_excited_not_implemented(self):
        kernel, casida_res = _h2_system()
        with pytest.raises(NotImplementedError):
            solve_nac_cpu(kernel, casida_res, states=(1, 2))
