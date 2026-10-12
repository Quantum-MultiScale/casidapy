"""Plain-PySCF WFT-in-DFT embedding solvers.

Embedded RHF/CASSCF/CCSD excitation energies and transition dipoles for a
fragment sitting in an eDFTpy embedding potential (folded into get_hcore by
``casidapy.pyscf_embed``), plus Frenkel/Forster coupling of two chromophores.

This module is deliberately **qiskit-free** -- it uses PySCF's own CASCI
(``trans_rdm1``) and EOM-CCSD. The qiskit variants (qubit exact-diagonalization,
qEOM/VQE) that consume the same embedded CASSCF live in ``QIS_QE/embed_qiskit.py``.
"""
import numpy as np
from casidapy import pyscf_embed

HARTREE2EV = 27.211386245988


def build_embedded_mf(atom, basis="cc-pVDZ", extemb=None, field=None, ref="farfield",
                      unit="Ry", xc=None, verbose=0, run=True,
                      embed_mode="external", nelec_valence=None):
    """Embedded RHF (or RKS if ``xc``) for ``atom`` at its eDFTpy Cartesian position.

    ``embed_mode``:
      * ``"external"`` (default, physically correct): supply the eDFTpy embedding
        potential as the ONE-ELECTRON external potential, exactly like the bare-QE
        Casida restart -- ``get_hcore = T + v_emb`` with ghost nuclei (no nuclear
        attraction). The fragment is treated **valence-only** (``nelec_valence``,
        the sum of the QE pseudopotential Z_valence), so v_emb supplies the ionic
        and the SCF rebuilds Hartree/exchange on the valence electrons. This avoids
        the double-counting that ``addon`` suffers (all-electron nuclei + v_emb that
        already contains the ionic -> gap doubles, excitations inflate).
      * ``"addon"`` (legacy): all-electron, ``get_hcore = T + V_nuc + v_emb``. Only
        correct if v_emb is a genuine environment-only potential (it usually is not
        for ``embed = ... HARTREE PSEUDO``).
    """
    from pyscf import gto, scf, dft
    mol = gto.M(atom=atom, basis=basis, verbose=verbose)

    if embed_mode == "external":
        if nelec_valence is None:
            raise ValueError("embed_mode='external' requires nelec_valence "
                             "(sum of QE pseudopotential Z_valence over the fragment).")
        mol.nelectron = int(nelec_valence)
        mf = dft.RKS(mol) if xc else scf.RHF(mol)
        if xc:
            mf.xc = xc
        # v_emb is the binding potential here, not an add-on -> no far-field
        # referencing (a constant offset cancels in excitation energies anyway).
        Vemb, grids = pyscf_embed.build_vemb(mol, extemb=extemb, field=field,
                                             unit=unit, ref=None)
        T = mol.intor("int1e_kin")
        hcore = T + Vemb                      # v_emb supplies the ionic (like bare QE)
        mf.get_hcore = lambda *a: hcore
        mf.energy_nuc = lambda *a: 0.0        # ghost nuclei; ion-ion is a const (cancels)
        mf.vemb_m = Vemb
        mf._vemb_grids = grids
        if run:
            mf.run()
        return mf

    # legacy add-on-top mode
    mf = dft.RKS(mol) if xc else scf.RHF(mol)
    if xc:
        mf.xc = xc
    if field is not None:
        pyscf_embed.enable_embedding(mf, field=field, ref=ref, unit=unit)
    elif extemb is not None:
        pyscf_embed.enable_embedding(mf, extemb=extemb, ref=ref, unit=unit)
    if run:
        mf.run()
    return mf


def build_embedded_casscf(mf, ncas=6, nelecas=6, nstates=6):
    """State-averaged CASSCF on an (already embedded) mf -> converged mc.
    v_emb enters automatically through mf.get_hcore. Reused by the qiskit path."""
    from pyscf import mcscf, fci
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.fcisolver = fci.direct_spin0.FCI()
    mc = mc.state_average_([1.0 / nstates] * nstates).run()
    return mc


def active_dipole_integrals(mc):
    """Electronic dipole integrals <p|-r|q> (a.u.) in the active CAS MO basis."""
    mol = mc.mol
    mo = mc.mo_coeff[:, mc.ncore:mc.ncore + mc.ncas]
    return -np.einsum("xuv,up,vq->xpq", mol.intor("int1e_r", comp=3), mo, mo)


def casscf_excitations(mf, ncas=6, nelecas=6, nstates=8, mc=None):
    """Plain-PySCF embedded CASSCF excitation energies + transition-dipole VECTORS
    via CASCI(nroots)+trans_rdm1 on the SA-CASSCF orbitals.

    Returns (states, mc) where states is a list of (omega_Ha, mu_vec[a.u.], s2),
    ascending in energy; mc is the converged CASSCF (reuse for the qiskit path).
    """
    from pyscf import mcscf, fci
    if mc is None:
        mc = build_embedded_casscf(mf, ncas, nelecas, nstates=nstates)
    dip = active_dipole_integrals(mc)
    ci_solver = mcscf.CASCI(mf, ncas, nelecas)
    ci_solver.fcisolver = fci.direct_spin1.FCI()      # full CI vectors -> trans_rdm1
    ci_solver.fcisolver.nroots = nstates
    ck = ci_solver.kernel(mc.mo_coeff)
    e, ci = ck[0], ck[2]
    states = []
    for n in range(1, len(e)):
        tdm = ci_solver.fcisolver.trans_rdm1(ci[0], ci[n], ncas, nelecas)
        mu = np.einsum("pq,xpq->x", tdm, dip).real
        s2 = float(ci_solver.fcisolver.spin_square(ci[n], ncas, nelecas)[0])
        states.append((float(e[n] - e[0]), np.asarray(mu), s2))
    return states, mc


def eomccsd_excitations(mf, nroots=5, kind="singlet", with_dipoles=False):
    """Embedded EOM-EE-CCSD excitation energies (Hartree). v_emb enters via the
    embedded mf's get_hcore -> CCSD reference + integrals.

    ``with_dipoles=True`` also returns CIS-like S0->state transition dipoles
    (a.u.) built from the EOM right-vector singles amplitudes R1,
    ``mu = sqrt(2) * sum_ia R1[i,a] <i|r|a>`` -- an approximate but faithful
    *character* proxy (bright vs dark) so the WFT correction can match states by
    character (Hungarian) instead of energy rank. Returns
    ``(energies, dipoles, e_corr)`` instead of ``(energies, e_corr)``."""
    from pyscf import cc
    mycc = cc.CCSD(mf)
    mycc.verbose = 0
    mycc.kernel()
    eom = mycc.EOMEESinglet() if kind == "singlet" else mycc.EOMEETriplet()
    e, v = eom.kernel(nroots=nroots)
    e = np.atleast_1d(e)
    if not with_dipoles:
        return [float(x) for x in e], float(mycc.e_corr)
    mol = mf.mol
    mo = mf.mo_coeff
    nocc = mol.nelectron // 2
    occ, vir = mo[:, :nocc], mo[:, nocc:]
    r_ao = mol.intor("int1e_r", comp=3)                 # <mu|r|nu> (a.u.)
    dip_ov = np.einsum("xmn,mi,na->xia", r_ao, occ, vir)
    vlist = v if isinstance(v, (list, tuple)) else [v]
    dips = []
    for vi in vlist:
        r1 = eom.vector_to_amplitudes(np.asarray(vi))[0]
        mu = np.sqrt(2.0) * np.einsum("ia,xia->x", r1, dip_ov)
        dips.append(np.asarray(mu, float))
    return [float(x) for x in e], dips, float(mycc.e_corr)


def wft_fragment_states(atom, *, extemb=None, field=None, basis="cc-pVDZ",
                        method="eomccsd", nroots=6, ncas=6, nelecas=6, nstates=8,
                        embed_mode="external", nelec_valence=None):
    """Embedded WFT excitations for ONE fragment, ready for
    ``casidapy.subsystem_coupling.apply_wft_correction`` (returns a list of
    ``(omega_Ha, mu_vec[a.u.], s2)``).

    ``method='eomccsd'``: EOM-EE-CCSD energies only -> ``mu=0, s2=0`` (for the
    active-space-free energy-shift correction; match with ``dipole_weight=0`` and
    use ``energy_only=True``). ``method='casscf'``: SA-CASSCF(ncas,nelecas)
    singlet energies + transition dipoles (for the energy+dipole correction).

    The embedding enters via ``field=`` (in-memory eDFTpy ``embedpot_field``) or
    ``extemb=`` (a .snpy/.pp/.xsf path)."""
    m = method.lower()
    _mf = lambda: build_embedded_mf(atom, basis=basis, extemb=extemb, field=field,
                                    embed_mode=embed_mode, nelec_valence=nelec_valence)
    if m in ("eomccsd", "eom-ccsd", "eom"):
        mf = _mf()
        e, dips, _ = eomccsd_excitations(mf, nroots=nroots, kind="singlet",
                                         with_dipoles=True)
        return [(float(w), np.asarray(mu, float), 0.0) for w, mu in zip(e, dips)]
    if m == "casscf":
        mf = _mf()
        st, _ = casscf_excitations(mf, ncas, nelecas, nstates=nstates)
        return [(w, mu, s2) for w, mu, s2 in st if s2 < 0.5]   # singlets
    raise ValueError(f"unknown WFT method {method!r} (use 'eomccsd' or 'casscf')")


def forster_couple(states_A, states_B, R_vec):
    """Frenkel-exciton coupling of two chromophores by dipole-dipole (Forster).

    states_{A,B}: lists of (omega_Ha, mu_vec[a.u.]). R_vec: A->B vector (Bohr).
    Returns (omega_coupled[Ha], mu_coupled[a.u.], osc) for the exciton manifold.
    """
    R = np.asarray(R_vec, dtype=float)
    r = np.linalg.norm(R)
    n = R / r
    A = list(states_A)
    B = list(states_B)
    nA = len(A)
    E = [w for w, _ in ((s[0], s[1]) for s in A)] + [w for w, _ in ((s[0], s[1]) for s in B)]
    MU = [np.asarray(s[1]) for s in A] + [np.asarray(s[1]) for s in B]
    H = np.diag(E).astype(float)
    for i in range(nA):
        for j in range(len(B)):
            mu1, mu2 = np.asarray(A[i][1]), np.asarray(B[j][1])
            J = (np.dot(mu1, mu2) - 3 * np.dot(mu1, n) * np.dot(mu2, n)) / r ** 3
            H[i, nA + j] = H[nA + j, i] = J
    w, C = np.linalg.eigh(H)
    mu_c = np.array([sum(C[k, m] * MU[k] for k in range(len(MU))) for m in range(len(w))])
    osc = np.array([2.0 / 3.0 * w[m] * mu_c[m] @ mu_c[m] for m in range(len(w))])
    return w, mu_c, osc
