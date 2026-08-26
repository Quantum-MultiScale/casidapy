#!/usr/bin/env python
"""GTO GPU spike: CPU vs CuPy(+gpu4pyscf) apply_K / TDA roots.

Run on a GPU node after:
  pip install cupy-cuda12x   # or cupy-cuda11x
  pip install gpu4pyscf      # optional but recommended for response

  python scripts/test_gto_gpu.py
"""
from __future__ import annotations

import time
import warnings

import numpy as np


def main():
    from pyscf import gto, dft
    from casidapy.adapter.pyscf import extract_gto_kernel
    from casidapy.casida_engine import run_casida

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "pbe0"
    mf.kernel()

    k_cpu, opts = extract_gto_kernel(
        mf, n_states=3, tda=True, use_df=False, use_gpu=False
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        k_gpu, opts_gpu = extract_gto_kernel(
            mf, n_states=3, tda=True, use_df=False, use_gpu=True, verbose=True
        )
        for msg in w:
            print(f"  warn: {msg.message}")

    t0 = time.time()
    k_cpu.setup(tda=True)
    t_cpu_setup = time.time() - t0

    t0 = time.time()
    k_gpu.setup(tda=True)
    t_gpu_setup = time.time() - t0

    print(f"setup CPU {t_cpu_setup:.3f}s  GPU {t_gpu_setup:.3f}s")
    print(f"gpu_response={k_gpu._gpu_response}  cached_K={k_gpu._K is not None}")

    rng = np.random.default_rng(0)
    v = rng.standard_normal(k_cpu.n_trans)
    Kv_c = k_cpu.apply_K(v)
    Kv_g = k_gpu.apply_K(v)
    print(f"apply_K max|diff| = {np.max(np.abs(Kv_c - Kv_g)):.3e}")

    # Force on-the-fly for a second check
    from casidapy.kernels.gto import GTOKernel

    otf = GTOKernel(
        mf.mol, mf.mo_coeff, mf.mo_energy, mf.mo_occ,
        xc="pbe0", use_df=False, mf=mf, k_cache_max=0, use_gpu=True,
    )
    otf.setup(tda=True)
    Kv_otf = otf.apply_K(v)
    print(f"on-the-fly max|diff| vs CPU = {np.max(np.abs(Kv_c - Kv_otf)):.3e}")

    opts.solver_method = "eigsh"
    opts_gpu.solver_method = "eigsh"
    r_cpu = run_casida(k_cpu, opts)
    r_gpu = run_casida(k_gpu, opts_gpu)
    print("omega CPU:", r_cpu.omega)
    print("omega GPU:", r_gpu.omega)
    print("omega max|diff|:", np.max(np.abs(r_cpu.omega - r_gpu.omega)))


if __name__ == "__main__":
    main()
