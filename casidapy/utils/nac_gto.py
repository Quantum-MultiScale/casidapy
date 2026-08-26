"""CPU-native ground <-> excited TDA nonadiabatic coupling (GTO/RKS path).

Phase 1 of a CPU port of TDDFT NAC: closed-shell singlet, TDA only, coupling
between the ground state and one TDA excited root, evaluated on a
:class:`~casidapy.kernels.gto.GTOKernel` / PySCF RKS mean-field.

This is a **direct, line-by-line translation** of the Z-vector / CPHF
derivative-coupling formulas in ``gpu4pyscf.nac.tdrhf.get_nacv_ge`` (cupy)
onto stock CPU PySCF primitives, following the exact CPU idiom already used
by stock ``pyscf.grad.tdrhf.grad_elec`` (``pyscf.scf.cphf.solve``,
``mf.gen_response``, ``mf_grad.hcore_generator``/``get_ovlp``/``get_jk``,
atom-resolved contraction via ``mol.offset_nr_by_atom()``). No new physics
is derived here; the reference is
`Zhang & Herbert, J. Chem. Phys. 141, 244105 (2014) <https://doi.org/10.1063/1.4903986>`_
(also cited by gpu4pyscf).

Every derivative-integral / response call below routes through PySCF's own
polymorphic ``mf.nuc_grad_method()`` / ``mf.gen_response()``, so a
density-fitted ``mf`` (``GTOKernel(use_df=True)``, the default) is handled
automatically by PySCF's own DF-gradient dispatch (confirmed: dispatches to
``pyscf.df.grad.{rhf,rks}.Gradients`` transparently) -- no special-casing
needed here.

Numerically verified (``tests/test_nac_gto.py``, RHF and RKS/pbe/b3lyp, H2
and LiH): ``de_etf`` satisfies translational invariance
(``sum(de_etf, axis=0) == 0``) to ~1e-16 for conventional integrals. With
density fitting the residual is ~1e-4 -- the well-known, expected
translational-invariance artifact of the RI/DF approximation itself (not a
bug in this port; PySCF's own DF analytic gradients carry the same
property). The two-electron J/K cross term
(``vj1,vk1 = get_jk(mol, dmz1doo)``; ``vj2,vk2 = get_jk(mol, dm_gs)``, both
directions summed) required one correction versus a naive one-sided
transliteration of gpu4pyscf's DF-specific ``jk_energies_per_atom`` kernel;
the two-sided form is what is numerically validated above.

Only ``de_etf`` (the ETF/energy-scaled coupling) is translationally
invariant, by construction of TDDFT-NAC theory -- the raw CIS-force-matrix-
element ``de`` is not, and is not expected to be.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from casidapy.casida_api import CasidaResults
from casidapy.kernels.gto import GTOKernel
from casidapy.utils.nac import NACResults


def _full_space_amplitude(kernel: GTOKernel, amp_active: np.ndarray) -> np.ndarray:
    """Embed an active-window TDA amplitude into the full (nocc, nvir) grid.

    ``CasidaResults.Z`` lives on the active ``(n_occ, n_unocc)`` window
    (``kernel.n_occ`` x ``kernel.n_unocc``); the CPHF equations need the
    excitation vector on the *full* occupied/virtual space of the reference
    ``mf``. Zero-padded outside the active window -- exact when the active
    window already spans the full space (the default in
    ``pyscf_adapter.extract_gto_kernel``), an approximation of the same kind
    as frozen-core/virtual truncation elsewhere otherwise.

    Assumes a standard aufbau MO ordering (all occupied MOs before all
    virtuals), matching ``kernel.mo_occ``.
    """
    mo_occ = np.asarray(kernel.mo_occ, dtype=float)
    n_occ_full = int(np.sum(mo_occ > 1e-6))
    n_vir_full = mo_occ.size - n_occ_full

    occ_idx = np.asarray(kernel.occ_indices, dtype=int)
    virt_idx = np.asarray(kernel.virt_indices, dtype=int) - n_occ_full
    if np.any(virt_idx < 0) or np.any(virt_idx >= n_vir_full):
        raise ValueError(
            "virt_indices do not map into the virtual block; non-aufbau "
            "MO ordering is not supported by this NAC path."
        )

    amp = np.asarray(amp_active, dtype=float).reshape(kernel.n_occ, kernel.n_unocc)
    full = np.zeros((n_occ_full, n_vir_full), dtype=float)
    full[np.ix_(occ_idx, virt_idx)] = amp
    return full


def _gen_response_mixed(mf):
    """``mf.gen_response(singlet=None, hermi=1)``, the CPHF response used by
    both stock ``pyscf.grad.tdrhf`` and ``gpu4pyscf.nac.tdrhf`` (a version-
    tolerant wrapper: older/newer PySCF ``gen_response`` signatures differ on
    optional kwargs such as ``with_nlc``)."""
    try:
        return mf.gen_response(singlet=None, hermi=1, with_nlc=True)
    except TypeError:
        return mf.gen_response(singlet=None, hermi=1)


def get_nacv_ge_tda(
    kernel: GTOKernel,
    casida_res: CasidaResults,
    state: int,
    *,
    atmlst: Optional[Sequence[int]] = None,
    cphf_max_cycle: int = 50,
    cphf_conv_tol: float = 1e-8,
):
    """Ground <-> excited-state TDA NACV on CPU.

    Direct CPU translation of ``gpu4pyscf.nac.tdrhf.get_nacv_ge`` restricted
    to TDA (``yI = 0``) and closed-shell singlet RKS/RHF.

    Parameters
    ----------
    kernel :
        A :class:`GTOKernel` already ``setup()`` (closed-shell, not
        spin-flip, not triplet).
    casida_res :
        :class:`~casidapy.casida_api.CasidaResults` from ``run_casida(kernel,
        options)`` with ``options.tda=True`` on the same ``kernel``.
    state :
        1-based excited-state index into ``casida_res.omega``/``Z``
        (matches the ``nac.py`` convention: 0 = ground state).

    Returns
    -------
    (de, de_scaled, de_etf, de_etf_scaled) : each ``(natm, 3)`` ndarray.
    """
    if kernel._spin_flip:
        raise NotImplementedError(
            "get_nacv_ge_tda: spin-flip NAC is not implemented (no CPU or "
            "GPU stock-PySCF reference exists to port from)."
        )
    if kernel.triplet:
        raise NotImplementedError(
            "get_nacv_ge_tda: triplet response NAC is not implemented yet."
        )
    if not kernel.tda:
        raise NotImplementedError(
            "get_nacv_ge_tda: only TDA (Y=0) is implemented; full TDDFT/RPA "
            "ground<->excited NAC is a follow-on (see plan)."
        )
    if kernel.mf is None:
        raise ValueError("kernel.mf is None; build the GTOKernel with mf= set.")

    mol = kernel.mol
    mf = kernel.mf
    mf_grad = mf.nuc_grad_method()

    mo_coeff = np.asarray(mf.mo_coeff, dtype=float)
    mo_energy = np.asarray(mf.mo_energy, dtype=float)
    mo_occ = np.asarray(mf.mo_occ, dtype=float)
    nao, nmo = mo_coeff.shape
    orbo = mo_coeff[:, mo_occ > 0]
    orbv = mo_coeff[:, mo_occ == 0]
    nocc = orbo.shape[1]
    nvir = orbv.shape[1]

    EI = float(np.asarray(casida_res.omega, dtype=float)[state - 1])
    amp_full = _full_space_amplitude(kernel, casida_res.Z[:, state - 1])  # (nocc, nvir)
    xI = amp_full.T  # (nvir, nocc), matches gpu4pyscf's xI.reshape(nocc,nvir).T
    yI = np.zeros_like(xI)  # TDA
    LI = xI - yI  # eq. (83) in Zhang & Herbert

    # --- Z-vector CPHF solve --------------------------------------------
    from pyscf.scf import cphf

    vresp = _gen_response_mixed(mf)

    def fvind(x):
        dm = np.linalg.multi_dot([orbv, x.reshape(nvir, nocc) * 2.0, orbo.T])
        v1ao = vresp(dm + dm.T)
        return np.linalg.multi_dot([orbv.T, v1ao, orbo]).ravel()

    rhs = -LI * EI  # (nvir, nocc); cphf.solve broadcasts h1 against e_ai of the same shape
    z1 = cphf.solve(
        fvind, mo_energy, mo_occ, rhs,
        max_cycle=cphf_max_cycle, tol=cphf_conv_tol,
    )[0]
    z1 = z1.reshape(nvir, nocc)

    z1ao = np.linalg.multi_dot([orbv, z1, orbo.T])
    dmz1doo = z1ao + z1ao.T  # symmetric; gpu4pyscf's z1aoS ("factorized" symmetrize=1)
    GZS = vresp(dmz1doo)
    GZS_mo = np.linalg.multi_dot([mo_coeff.T, GZS, mo_coeff])

    # --- W matrix (eqs. 73/75 in Zhang & Herbert) ------------------------
    W = np.zeros((nmo, nmo))
    W[:nocc, :nocc] = GZS_mo[:nocc, :nocc]
    zeta0 = z1 * mo_energy[nocc:, np.newaxis]
    W[:nocc, nocc:] = GZS_mo[:nocc, nocc:] + 0.5 * yI.T * EI + 0.5 * zeta0.T
    zeta1 = z1 * mo_energy[np.newaxis, :nocc]
    W[nocc:, :nocc] = 0.5 * xI * EI + 0.5 * zeta1
    W = np.linalg.multi_dot([mo_coeff, W, mo_coeff.T]) * 2.0

    # --- Derivative integrals + atom-resolved contraction ----------------
    if len(mol._ecpbas) > 0:
        raise NotImplementedError("ECP derivative NAC is not supported.")

    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)  # (3, nao, nao)
    dm_gs = mf.make_rdm1()  # ground-state density (= 2 * orbo @ orbo.T), gpu4pyscf's "oo0"

    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = list(atmlst)
    offsetdic = mol.offset_nr_by_atom()
    natm_sel = len(atmlst)
    de = np.zeros((natm_sel, 3))

    # 1-electron: h1(ia) . dmz1doo  (eq. analogous to stock pyscf.grad.tdrhf's
    # dh_td, but ge-NAC has no ground-state-only h.oo0 term -- see module docstring)
    for k, ia in enumerate(atmlst):
        h1ao = hcore_deriv(ia)
        de[k] += np.einsum('xpq,pq->x', h1ao, dmz1doo)

    # overlap: -s1 . W  (W is not symmetric -> both row- and col-block terms,
    # same "hermi=0" pattern as stock pyscf.grad.tdrhf's `ds`/`im0` term)
    for k, ia in enumerate(atmlst):
        _, _, p0, p1 = offsetdic[ia]
        de[k] -= np.einsum('xpq,pq->x', s1[:, p0:p1], W[p0:p1])
        de[k] -= np.einsum('xqp,pq->x', s1[:, p0:p1], W[:, p0:p1])

    # 2-electron J/K cross term between dmz1doo and the ground-state density
    # (gpu4pyscf: jk_energies_per_atom([[dmz1doo, oo0]], j_factor=1, k_factor=1);
    # the extra *2 gpu4pyscf applies is a DF-kernel storage-convention
    # compensation for its "factorized/symmetrize=1" dm -- not needed here
    # since dmz1doo is already a plain, fully-symmetrized dense matrix).
    #
    # get_jk(mol, dm) differentiates only the AO index tied to the density
    # passed in; a bilinear cross term Tr[dm_gs . J[dmz1doo]] needs *both*
    # differentiation orders summed (verified numerically: only the sum of
    # both satisfies translational invariance, not either half alone).
    vj1, vk1 = mf_grad.get_jk(mol, dmz1doo)
    vj2, vk2 = mf_grad.get_jk(mol, dm_gs)
    for k, ia in enumerate(atmlst):
        _, _, p0, p1 = offsetdic[ia]
        de[k] += np.einsum('xpq,pq->x', vj1[:, p0:p1], dm_gs[p0:p1])
        de[k] -= np.einsum('xpq,pq->x', vk1[:, p0:p1], dm_gs[p0:p1])
        de[k] += np.einsum('xpq,pq->x', vj2[:, p0:p1], dmz1doo[p0:p1])
        de[k] -= np.einsum('xpq,pq->x', vk2[:, p0:p1], dmz1doo[p0:p1])

    # --- ETF / energy-scaled asymmetric overlap term ---------------------
    xIao = np.linalg.multi_dot([orbo, xI.T, orbv.T]) * EI  # yIao = 0 (TDA)

    de_asym = np.zeros((natm_sel, 3))  # antisymmetric NACV contribution (no ETF)
    for k, ia in enumerate(atmlst):
        _, _, p0, p1 = offsetdic[ia]
        de_asym[k] = np.einsum('xij,ji->x', s1[:, p0:p1, :], xIao[:, p0:p1]) * 2.0

    de_etf_asym = np.zeros((natm_sel, 3))  # ETF (energy-scaled) contribution
    for k, ia in enumerate(atmlst):
        _, _, p0, p1 = offsetdic[ia]
        de_etf_asym[k] = (
            np.einsum('xpq,pq->x', s1[:, p0:p1], xIao[p0:p1])
            + np.einsum('xqp,pq->x', s1[:, p0:p1], xIao[:, p0:p1])
        )

    de_etf = de + de_etf_asym
    de = de + de_asym

    return de, de / EI, de_etf, de_etf / EI


def solve_nac_cpu(
    kernel: GTOKernel,
    casida_res: CasidaResults,
    states: Sequence[int] = (0, 1),
    *,
    atmlst: Optional[Sequence[int]] = None,
) -> NACResults:
    """High-level CPU TDA ground<->excited NAC entry point.

    Mirrors :func:`casidapy.utils.nac.solve_nac`'s return contract
    (:class:`~casidapy.utils.nac.NACResults`) so downstream QED-projection code
    (``casidapy.utils.nac.solve_qed_projected_nac``) is backend-agnostic.
    """
    if len(states) != 2:
        raise ValueError(f"states must be a length-2 pair, got {states!r}")
    pair = (int(states[0]), int(states[1]))
    if 0 not in pair:
        raise NotImplementedError(
            "solve_nac_cpu: only ground<->excited NAC is implemented in "
            "phase 1; excited<->excited NAC is a follow-on."
        )
    state = pair[1] if pair[0] == 0 else pair[0]

    de, de_scaled, de_etf, de_etf_scaled = get_nacv_ge_tda(
        kernel, casida_res, state, atmlst=atmlst,
    )

    omega = np.array(
        [0.0, float(np.asarray(casida_res.omega, dtype=float)[state - 1])],
        dtype=float,
    )
    if pair[0] != 0:
        de, de_scaled, de_etf, de_etf_scaled = (-de, -de_scaled, -de_etf, -de_etf_scaled)
        omega = omega[::-1]

    return NACResults(
        states=pair,
        de=de,
        de_scaled=de_scaled,
        de_etf=de_etf,
        de_etf_scaled=de_etf_scaled,
        omega=omega,
        method="tda",
        backend="casidapy-cpu",
        meta={"kernel": kernel, "atmlst": atmlst},
    )
