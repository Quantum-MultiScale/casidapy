"""Quick parity check: CasidaPy GTOKernel (TDA) vs PySCF TDA, incl. hybrids.

Small H2 / 6-31G system so it runs in seconds. Usage:

    source /projectsn/mp1009_1/am4655/stddft/bin/activate
    cd /projectsn/mp1009_1/am4655/CasidaPy
    python scripts/test_gto_hybrid.py
"""
import numpy as np
from pyscf import gto, dft, scf

from casidapy.adapter.pyscf import extract_gto_kernel
from casidapy.casida_engine import run_casida

NSTATES = 2
TOL = 1e-6


def compare(xc):
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", verbose=0)
    if xc == "hf":
        mf = scf.RHF(mol)
    else:
        mf = dft.RKS(mol)
        mf.xc = xc
    mf.kernel()

    td = mf.TDA()
    td.nstates = NSTATES
    e_ref = td.kernel()[0]

    kernel, opts = extract_gto_kernel(
        mf, n_states=NSTATES, tda=True, use_df=False, verbose=False
    )
    opts.solver_method = "eigsh"
    opts.solver_maxiter = 300
    res = run_casida(kernel, opts)

    n = min(len(res.omega), len(e_ref))
    diff = np.max(np.abs(res.omega[:n] - e_ref[:n]))
    status = "OK  " if diff < TOL else "FAIL"
    print(f"[{status}] {xc:8s}  pyscf={np.round(e_ref[:n], 8)}  "
          f"casida={np.round(np.asarray(res.omega[:n]), 8)}  maxdiff={diff:.2e}")
    return diff < TOL


def check_hybrid_rpa_rejected():
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "pbe0"
    mf.kernel()
    kernel, opts = extract_gto_kernel(
        mf, n_states=NSTATES, tda=False, use_df=False, verbose=False
    )
    opts.solver_method = "eigsh"
    try:
        run_casida(kernel, opts)
    except NotImplementedError:
        print("[OK  ] pbe0 + full Casida (non-TDA) correctly raises NotImplementedError")
        return True
    print("[FAIL] pbe0 + full Casida (non-TDA) did NOT raise")
    return False


if __name__ == "__main__":
    ok = all([
        compare("lda,vwn"),
        compare("pbe"),
        compare("pbe0"),    # hybrid: 25% exact exchange
        compare("b3lyp"),   # hybrid: 20% exact exchange
        compare("hf"),      # plain RHF ground state -> CIS
        check_hybrid_rpa_rejected(),
    ])
    print("\nALL PASSED" if ok else "\nSOME TESTS FAILED")
    raise SystemExit(0 if ok else 1)
