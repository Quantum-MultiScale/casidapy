"""Nonadiabatic couplings via PySCF / gpu4pyscf, plus projected QED NACs.

CPU **PySCF** (``pyscf.nac``) only provides SA-CASSCF / MSPDFT NACs — there is
no TDDFT NAC in stock PySCF.

**TDDFT** NACs (linear-response RHF/RKS and spin-flip UKS) live in
**gpu4pyscf** and use the usual PySCF-style API
(``td.nac_method()`` / ``td.NAC()`` → ``.kernel(states=...)``).

State indexing (gpu4pyscf TDDFT)::

    0 = electronic ground state
    1 = first excited root (``td.e[0]``), …

**Projected QED NACs** (``project_qed_nac`` / ``solve_qed_projected_nac``)
approximate polariton derivative couplings by contracting electronic TDDFT
NACVs with the electronic weights of each polariton. Photon / ∂g/∂R pieces
are neglected. Electronic NACs still need gpu4pyscf (GPU); the QED solve
and projection can run on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

HA_TO_EV = 27.211386245988

States = Tuple[int, int]


@dataclass
class NACResults:
    """NAC vectors between two electronic states.

    Attributes
    ----------
    states :
        Pair ``(I, J)`` using the backend's indexing (gpu4pyscf: ``0`` = GS).
    de :
        Derivative coupling / force-matrix element (natm × 3), backend default.
    de_scaled, de_etf, de_etf_scaled :
        Optional ETF / energy-scaled variants (gpu4pyscf TDDFT).
    omega :
        Excitation energies of the two states in Hartree when available
        (``0.0`` for the ground state). Length-2.
    method, backend :
        Labels such as ``\"tda\"`` / ``\"gpu4pyscf\"``.
    """

    states: States
    de: np.ndarray
    de_scaled: Optional[np.ndarray] = None
    de_etf: Optional[np.ndarray] = None
    de_etf_scaled: Optional[np.ndarray] = None
    omega: Optional[np.ndarray] = None
    method: str = ""
    backend: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def omega_ev(self) -> Optional[np.ndarray]:
        if self.omega is None:
            return None
        return np.asarray(self.omega, dtype=float) * HA_TO_EV


def _as_numpy(a) -> Optional[np.ndarray]:
    if a is None:
        return None
    if hasattr(a, "get"):
        a = a.get()
    return np.asarray(a, dtype=float)


def _require_gpu4pyscf():
    try:
        import gpu4pyscf  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "TDDFT NACs require gpu4pyscf (and a CUDA GPU). "
            "Install with: pip install gpu4pyscf"
        ) from exc


def to_gpu_mf(mf):
    """Return a gpu4pyscf mean-field object (idempotent if already on GPU)."""
    _require_gpu4pyscf()
    if getattr(mf, "device", None) == "gpu":
        return mf
    if hasattr(mf, "to_gpu"):
        return mf.to_gpu()
    raise TypeError(
        f"Cannot move mf of type {type(mf).__name__} to GPU; "
        "pass a PySCF RHF/RKS/UHF/UKS object with .to_gpu()."
    )


def build_tddft(
    mf,
    *,
    method: str = "tda",
    nstates: int = 5,
    singlet: bool = True,
    spin_flip: bool = False,
    collinear: str = "mcol",
    collinear_samples: int = 20,
    extype: int = 1,
    run: bool = True,
):
    """Build and optionally run a gpu4pyscf TDDFT/TDA (or SF) object.

    Parameters
    ----------
    method :
        ``\"tda\"``, ``\"tddft\"`` / ``\"tdhf\"`` for spin-conserving LR;
        with ``spin_flip=True``, ``\"tda\"`` → SF-TDA and ``\"tddft\"`` → SF-TDHF.
    """
    mf_gpu = to_gpu_mf(mf)
    key = str(method).strip().lower().replace("-", "").replace("_", "")
    if key in ("tdhf", "rpa"):
        key = "tddft"

    if spin_flip or key in ("sftda", "sftddft", "sftdhf"):
        from gpu4pyscf.tdscf import uhf as td_uhf

        if key in ("sftddft", "sftdhf", "tddft"):
            td = td_uhf.SpinFlipTDHF(mf_gpu)
            method_label = "sf-tddft"
        else:
            td = td_uhf.SpinFlipTDA(mf_gpu)
            method_label = "sf-tda"
        td.extype = int(extype)
        td.collinear = collinear
        td.collinear_samples = int(collinear_samples)
        td.nstates = int(nstates)
    else:
        if key not in ("tda", "tddft"):
            raise ValueError(
                f"Unknown TDDFT method {method!r}; use 'tda', 'tddft', "
                "or set spin_flip=True."
            )
        if key == "tda":
            td = mf_gpu.TDA()
            method_label = "tda"
        else:
            td = mf_gpu.TDDFT() if hasattr(mf_gpu, "TDDFT") else mf_gpu.TDHF()
            method_label = "tddft"
        td.nstates = int(nstates)
        if hasattr(td, "singlet"):
            td.singlet = bool(singlet)

    td._casidapy_method = method_label
    if run:
        td.kernel()
    return td


def _make_nac_object(td):
    """Return the backend NAC driver for a converged TD object."""
    if hasattr(td, "nac_method"):
        return td.nac_method()
    if hasattr(td, "NAC"):
        return td.NAC()
    raise TypeError(
        f"TD object {type(td).__name__} has no nac_method()/NAC(); "
        "use a gpu4pyscf TDA/TDDFT or SpinFlipTDA/TDHF instance."
    )


def _state_energies(td, states: States) -> np.ndarray:
    """Map (I, J) with 0=GS to excitation energies (Ha); GS → 0."""
    e = np.asarray(getattr(td, "e", []), dtype=float).ravel()
    out = []
    for s in states:
        if s == 0:
            out.append(0.0)
        else:
            idx = int(s) - 1
            if idx < 0 or idx >= e.size:
                raise ValueError(
                    f"State {s} out of range for {e.size} excited roots "
                    f"(gpu4pyscf indexing: 0=GS, 1..{e.size}=excited)."
                )
            out.append(float(e[idx]))
    return np.asarray(out, dtype=float)


def solve_tddft_nac(
    td,
    states: Sequence[int] = (0, 1),
    *,
    atmlst: Optional[Sequence[int]] = None,
    singlet: Optional[bool] = None,
) -> NACResults:
    """Compute TDDFT NACVs for a converged gpu4pyscf TD object.

    Parameters
    ----------
    states :
        ``(I, J)`` with ``0`` = ground state (gpu4pyscf convention).
    """
    _require_gpu4pyscf()
    if len(states) != 2:
        raise ValueError(f"states must be a length-2 pair, got {states!r}")
    pair: States = (int(states[0]), int(states[1]))

    nac = _make_nac_object(td)
    kwargs: Dict[str, Any] = {"states": pair}
    if atmlst is not None:
        kwargs["atmlst"] = list(atmlst)
    if singlet is not None:
        kwargs["singlet"] = bool(singlet)
    nac.kernel(**kwargs)

    method = getattr(td, "_casidapy_method", None) or type(td).__name__
    return NACResults(
        states=pair,
        de=_as_numpy(nac.de),
        de_scaled=_as_numpy(getattr(nac, "de_scaled", None)),
        de_etf=_as_numpy(getattr(nac, "de_etf", None)),
        de_etf_scaled=_as_numpy(getattr(nac, "de_etf_scaled", None)),
        omega=_state_energies(td, pair),
        method=str(method),
        backend="gpu4pyscf",
        meta={
            "td": td,
            "nac": nac,
            "atmlst": getattr(nac, "atmlst", None),
        },
    )


def solve_nac(
    mf,
    *,
    states: Sequence[int] = (0, 1),
    method: str = "tda",
    nstates: int = 5,
    singlet: bool = True,
    spin_flip: bool = False,
    collinear: str = "mcol",
    collinear_samples: int = 20,
    extype: int = 1,
    atmlst: Optional[Sequence[int]] = None,
    td=None,
) -> NACResults:
    """High-level TDDFT NAC: GPU MF → TDDFT/TDA (or SF) → NACV.

    Pass an already-converged ``td`` to skip the TDDFT solve.
    """
    if td is None:
        td = build_tddft(
            mf,
            method=method,
            nstates=nstates,
            singlet=singlet,
            spin_flip=spin_flip,
            collinear=collinear,
            collinear_samples=collinear_samples,
            extype=extype,
            run=True,
        )
    return solve_tddft_nac(td, states, atmlst=atmlst, singlet=singlet if not spin_flip else None)


def solve_sacasscf_nac(
    mc,
    states: Sequence[int] = (0, 1),
    *,
    use_etf: bool = False,
) -> NACResults:
    """Wrap CPU ``pyscf.nac.sacasscf`` (state-averaged CASSCF only).

    State indices are SA-CASSCF state labels (both usually excited / averaged
    roots — see PySCF docs); this is **not** TDDFT.
    """
    try:
        from pyscf import nac as pyscf_nac
    except ImportError as exc:
        raise ImportError("solve_sacasscf_nac requires pyscf") from exc

    if len(states) != 2:
        raise ValueError(f"states must be a length-2 pair, got {states!r}")
    pair: States = (int(states[0]), int(states[1]))

    if hasattr(mc, "nac_method"):
        nac = mc.nac_method()
    else:
        nac = pyscf_nac.sacasscf.NonAdiabaticCouplings(mc)
    de = nac.kernel(state=pair, use_etf=use_etf)
    return NACResults(
        states=pair,
        de=_as_numpy(de if de is not None else getattr(nac, "de", None)),
        de_scaled=None,
        de_etf=None,
        de_etf_scaled=None,
        omega=None,
        method="sacasscf",
        backend="pyscf",
        meta={"nac": nac, "use_etf": bool(use_etf)},
    )


# ---------------------------------------------------------------------------
# Projected QED / polariton NACs
# ---------------------------------------------------------------------------


def assemble_electronic_nac_tensor(
    nac_pairs: Dict[Tuple[int, int], np.ndarray],
    n_states: int,
    *,
    antisym: bool = True,
) -> np.ndarray:
    """Pack pairwise NACVs into ``d[i, j, atom, xyz]``.

    Parameters
    ----------
    nac_pairs :
        Map ``(i, j) → (natm, 3)`` using gpu4pyscf indexing (``0`` = GS).
        Only ``i < j`` (or any orientation) needs to be supplied when
        ``antisym=True``.
    n_states :
        Tensor dimension (``0 … n_states-1`` inclusive). For ``N`` excited
        roots with GS included use ``n_states = N + 1``.
    """
    if n_states < 2:
        raise ValueError(f"n_states must be >= 2, got {n_states}")
    sample = next(iter(nac_pairs.values()))
    sample = np.asarray(sample, dtype=float)
    if sample.ndim != 2 or sample.shape[1] != 3:
        raise ValueError(f"each NACV must have shape (natm, 3), got {sample.shape}")
    natm = int(sample.shape[0])
    d = np.zeros((n_states, n_states, natm, 3), dtype=float)
    for (i, j), vec in nac_pairs.items():
        ii, jj = int(i), int(j)
        if not (0 <= ii < n_states and 0 <= jj < n_states):
            raise ValueError(
                f"pair {(i, j)} out of range for n_states={n_states}"
            )
        if ii == jj:
            continue
        v = np.asarray(vec, dtype=float)
        if v.shape != (natm, 3):
            raise ValueError(
                f"NACV for {(i, j)} has shape {v.shape}, expected ({natm}, 3)"
            )
        d[ii, jj] = v
        if antisym:
            d[jj, ii] = -v
    return d


def compute_electronic_nac_tensor(
    td,
    n_excited: int,
    *,
    include_ground: bool = True,
    spin_flip: bool = False,
    pairs: Optional[Sequence[Tuple[int, int]]] = None,
    which: str = "de",
    atmlst: Optional[Sequence[int]] = None,
    singlet: Optional[bool] = None,
) -> np.ndarray:
    """Build the electronic NAC tensor by looping gpu4pyscf ``NAC.kernel``.

    Parameters
    ----------
    n_excited :
        Number of TDDFT excited roots (must be ``<= len(td.e)``).
    include_ground :
        If True, indices run ``0=GS, 1..n_excited``; else ``0..n_excited-1``
        label excited states only (``0`` → first excited = gpu4pyscf ``1``).
        Forced to False when ``spin_flip=True`` (gpu4pyscf SF has no
        ground–excited NAC).
    spin_flip :
        Use excited-only indexing for spin-flip TDDFT NACs.
    which :
        Which NAC variant to store: ``de``, ``de_scaled``, ``de_etf``,
        or ``de_etf_scaled``.
    """
    key = str(which).strip().lower()
    allowed = ("de", "de_scaled", "de_etf", "de_etf_scaled")
    if key not in allowed:
        raise ValueError(f"which must be one of {allowed}, got {which!r}")

    n_exc = int(n_excited)
    e = np.asarray(getattr(td, "e", []), dtype=float).ravel()
    if n_exc < 1 or n_exc > e.size:
        raise ValueError(
            f"n_excited={n_exc} invalid for td with {e.size} roots"
        )

    if spin_flip:
        include_ground = False

    if include_ground:
        labels = list(range(0, n_exc + 1))  # 0=GS, 1..n_exc
        n_states = n_exc + 1

        def to_gpu_idx(lab: int) -> int:
            return lab
    else:
        labels = list(range(0, n_exc))
        n_states = n_exc

        def to_gpu_idx(lab: int) -> int:
            return lab + 1  # local 0 → gpu 1

    if pairs is None:
        pair_list: List[Tuple[int, int]] = []
        for i, a in enumerate(labels):
            for b in labels[i + 1 :]:
                pair_list.append((a, b))
    else:
        pair_list = [(int(i), int(j)) for i, j in pairs]

    nac_pairs: Dict[Tuple[int, int], np.ndarray] = {}
    for i_lab, j_lab in pair_list:
        gi, gj = to_gpu_idx(i_lab), to_gpu_idx(j_lab)
        if gi == gj:
            continue
        if spin_flip and (gi == 0 or gj == 0):
            raise ValueError(
                "Spin-flip gpu4pyscf NACs are excited–excited only; "
                f"cannot request pair {(gi, gj)}."
            )
        res = solve_tddft_nac(
            td, (gi, gj), atmlst=atmlst, singlet=None if spin_flip else singlet,
        )
        vec = getattr(res, key)
        if vec is None:
            raise RuntimeError(
                f"NAC result for states {(gi, gj)} has no attribute {key!r}"
            )
        nac_pairs[(i_lab, j_lab)] = vec
    return assemble_electronic_nac_tensor(nac_pairs, n_states, antisym=True)


def project_polariton_nac(
    C: np.ndarray,
    d_el: np.ndarray,
    states: Sequence[int],
) -> np.ndarray:
    """Projected NAC: ``d_IJ = Σ_{a,b} C[a,I] C[b,J] d_el[a,b]``.

    Parameters
    ----------
    C :
        Electronic weights ``(n_el, n_pol)`` for each polariton column.
    d_el :
        Electronic NAC tensor ``(n_el, n_el, natm, 3)``.
    states :
        Polariton pair ``(I, J)`` (0-based into columns of ``C``).
    """
    C_arr = np.asarray(C, dtype=float)
    d = np.asarray(d_el, dtype=float)
    if C_arr.ndim != 2:
        raise ValueError(f"C must be 2-D (n_el, n_pol), got {C_arr.shape}")
    if d.ndim != 4 or d.shape[0] != d.shape[1]:
        raise ValueError(
            f"d_el must have shape (n_el, n_el, natm, 3), got {d.shape}"
        )
    if C_arr.shape[0] != d.shape[0]:
        raise ValueError(
            f"C rows ({C_arr.shape[0]}) != d_el dimension ({d.shape[0]})"
        )
    if len(states) != 2:
        raise ValueError(f"states must be length-2, got {states!r}")
    I, J = int(states[0]), int(states[1])
    n_pol = C_arr.shape[1]
    if not (0 <= I < n_pol and 0 <= J < n_pol):
        raise ValueError(
            f"polariton states {(I, J)} out of range for n_pol={n_pol}"
        )
    if I == J:
        return np.zeros(d.shape[2:], dtype=float)
    # d_IJ[atom,xyz] = Σ_ab C[a,I] C[b,J] d[a,b,atom,xyz]
    cI = C_arr[:, I]
    cJ = C_arr[:, J]
    return np.einsum("a,b,abxy->xy", cI, cJ, d, optimize=True)


def electronic_weights_qed_post(
    qed: Dict[str, Any],
    *,
    n_label: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map post-process QED eigenvectors onto electronic labels ``0=GS, 1…``.

    Returns
    -------
    C : ndarray, shape (n_label, n_pol)
        Electronic weights (0- and 1-photon sectors added for PF).
    labels_used : ndarray
        For each row of the *local* QED electronic basis, the global label
        into ``C`` (informational; ``C`` itself is already global).
    """
    if "V" not in qed:
        raise ValueError("qed dict must contain eigenvector matrix 'V'")
    V = np.asarray(qed["V"], dtype=float)
    idx = np.asarray(qed.get("tddft_indices", np.arange(0)), dtype=int).ravel()
    model = str(qed.get("model", "")).lower()

    n_exc_full = int(idx.max()) + 1 if idx.size else 0
    if n_label is None:
        n_label = n_exc_full + 1  # GS + excited through max index
    n_label = int(n_label)
    n_pol = V.shape[1]
    C = np.zeros((n_label, n_pol), dtype=float)

    is_pf = (
        "pauli-fierz" in model
        or model in ("pf", "pauli")
        or (qed.get("postprocess") and "include_ground_slot" in qed
            and "jaynes" not in model and "tavis" not in model
            and V.shape[0] % 2 == 0 and V.shape[0] // 2 == (
                idx.size + (1 if qed.get("include_ground_slot", True) else 0)
            ))
    )

    if is_pf:
        # PF: electronic ⊗ {0,1}; V[:n] and V[n:] share the same electronic map
        include_gs = bool(qed.get("include_ground_slot", True))
        n = V.shape[0] // 2
        if V.shape[0] != 2 * n:
            raise ValueError(
                f"PF eigenvector length {V.shape[0]} is not even (2 * n_el)"
            )
        if include_gs:
            labels_local = [0] + [int(k) + 1 for k in idx]
        else:
            labels_local = [int(k) + 1 for k in idx]
        if len(labels_local) != n:
            raise ValueError(
                f"PF n_el={n} inconsistent with electronic labels "
                f"({len(labels_local)}; include_ground_slot={include_gs}, "
                f"n_idx={idx.size})"
            )
        for loc, lab in enumerate(labels_local):
            if lab < 0 or lab >= n_label:
                raise ValueError(
                    f"electronic label {lab} out of range for n_label={n_label}"
                )
            C[lab, :] += V[loc, :] + V[loc + n, :]
        return C, np.asarray(labels_local, dtype=int)

    # JC / TC: V[:n] = |S_k,0⟩, V[n] = |S0,1⟩ (photon — no electronic NAC weight)
    n = V.shape[0] - 1
    if n != idx.size:
        raise ValueError(
            f"JC/TC electronic block size {n} != len(tddft_indices)={idx.size}"
        )
    labels_local = [int(k) + 1 for k in idx]
    for loc, lab in enumerate(labels_local):
        if lab < 0 or lab >= n_label:
            raise ValueError(
                f"electronic label {lab} out of range for n_label={n_label}"
            )
        C[lab, :] += V[loc, :]
    # Photon amplitude V[n] omitted from electronic projection by design.
    return C, np.asarray(labels_local, dtype=int)


def electronic_weights_qed_dense(
    qed_res,
    casida_res,
    *,
    n_label: Optional[int] = None,
    spin_flip: Optional[bool] = None,
) -> np.ndarray:
    """Project dense QED-TDA amplitudes onto Casida eigenstates.

    Closed-shell
        ``X`` is ``(n_trans, n_pol)``; ``C[1:n_cas+1] = Z.T @ X`` and
        ``C[0] = 0`` (photonic ``m`` is not mixed into electronic NACs).

    Spin-flip
        ``X`` is ``(2 n_trans, n_pol)``; electronic weights are the sum of
        0- and 1-photon blocks projected on ``Z``::

            C = Z.T @ X[:n] + Z.T @ X[n:]

        with shape ``(n_cas, n_pol)`` (no ground-state row — SF manifold).
    """
    X = np.asarray(qed_res.X, dtype=float)
    Z = np.asarray(casida_res.Z, dtype=float)
    if X.ndim != 2 or Z.ndim != 2:
        raise ValueError("qed_res.X and casida_res.Z must be 2-D")

    meta = getattr(qed_res, "meta", None) or {}
    if spin_flip is None:
        spin_flip = bool(meta.get("spin_flip", False))
        if not spin_flip and X.shape[0] == 2 * Z.shape[0]:
            spin_flip = True

    n_trans = Z.shape[0]
    n_cas = Z.shape[1]
    n_pol = X.shape[1]

    if spin_flip:
        if X.shape[0] != 2 * n_trans:
            raise ValueError(
                f"SF QED X rows ({X.shape[0]}) != 2 * n_trans ({2 * n_trans})"
            )
        C = Z.T @ X[:n_trans] + Z.T @ X[n_trans:]
        if n_label is not None and int(n_label) != n_cas:
            raise ValueError(
                f"SF n_label={n_label} must equal n_casida={n_cas} "
                "(excited-only indexing)."
            )
        return C

    if X.shape[0] != n_trans:
        raise ValueError(
            f"transition length mismatch: X {X.shape[0]} vs Z {n_trans}"
        )
    c_exc = Z.T @ X  # (n_cas, n_pol)
    if n_label is None:
        n_label = n_cas + 1
    n_label = int(n_label)
    if n_label < n_cas + 1:
        raise ValueError(
            f"n_label={n_label} < n_casida+1={n_cas + 1}"
        )
    C = np.zeros((n_label, n_pol), dtype=float)
    C[1 : n_cas + 1, :] = c_exc
    return C


def electronic_weights_qed_sf(
    qed_res,
    casida_res,
    *,
    n_label: Optional[int] = None,
) -> np.ndarray:
    """Electronic weights for QED-SF-TDA (alias of dense SF projection)."""
    return electronic_weights_qed_dense(
        qed_res, casida_res, n_label=n_label, spin_flip=True,
    )


def project_qed_nac(
    C: np.ndarray,
    d_el: np.ndarray,
    states: Sequence[int],
    *,
    omega: Optional[np.ndarray] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> NACResults:
    """Project electronic NACVs onto a polariton pair using weights ``C``."""
    pair: States = (int(states[0]), int(states[1]))
    de = project_polariton_nac(C, d_el, pair)
    om = None
    if omega is not None:
        w = np.asarray(omega, dtype=float).ravel()
        if pair[0] < w.size and pair[1] < w.size:
            om = np.array([w[pair[0]], w[pair[1]]], dtype=float)
    return NACResults(
        states=pair,
        de=de,
        omega=om,
        method="qed-projected",
        backend="projected",
        meta=dict(meta or {}),
    )


def solve_qed_projected_nac(
    qed,
    d_el: np.ndarray,
    states: Sequence[int],
    *,
    casida=None,
    n_label: Optional[int] = None,
    spin_flip: Optional[bool] = None,
) -> NACResults:
    """High-level projected QED NAC from a post-process dict or dense results.

    Parameters
    ----------
    qed :
        Either the dict from :func:`~casidapy.qed.solve_qed_post` /
        ``solve_qed_levels`` / ``solve_qed_pf_post``, or a
        :class:`~casidapy.qed.QEDResults` from :func:`~casidapy.qed.solve_qed_tda`
        or :func:`~casidapy.qed.solve_qed_sf_tda`.
        Dense / SF results also need ``casida``
        (:class:`~casidapy.casida_api.CasidaResults`) with eigenvectors ``Z``.
    d_el :
        Electronic NAC tensor. Closed-shell: ``(n_label, n_label, natm, 3)``
        with ``0`` = GS. Spin-flip: ``(n_cas, n_cas, natm, 3)`` with local
        ``0`` = first SF root (gpu4pyscf state ``1``). Build with
        :func:`compute_electronic_nac_tensor` (``spin_flip=True`` for SF).
    states :
        Polariton pair ``(I, J)`` (0-based into QED eigenpairs).
    spin_flip :
        Force SF projection. Default: read ``qed.meta['spin_flip']`` or
        infer from ``X`` vs ``Z`` shapes.
    """
    d = np.asarray(d_el, dtype=float)
    if n_label is None:
        n_label = int(d.shape[0])

    if isinstance(qed, dict) and "V" in qed:
        if spin_flip:
            raise ValueError(
                "spin_flip=True is for QED-SF-TDA (QEDResults); "
                "post-process dicts are closed-shell JC/TC/PF only."
            )
        C, labels_local = electronic_weights_qed_post(qed, n_label=n_label)
        omega = qed.get("omega")
        meta = {
            "model": qed.get("model"),
            "tddft_indices": qed.get("tddft_indices"),
            "labels_local": labels_local,
            "photon_frac": qed.get("photon_frac"),
            "approximation": "projected_electronic_nac",
            "spin_flip": False,
        }
    else:
        if casida is None:
            raise ValueError(
                "Dense / SF QEDResults require casida=CasidaResults "
                "(with Z eigenvectors) for projection onto electronic NACs."
            )
        meta_q = getattr(qed, "meta", None) or {}
        sf = bool(spin_flip) if spin_flip is not None else bool(
            meta_q.get("spin_flip", False)
        )
        C = electronic_weights_qed_dense(
            qed, casida, n_label=n_label if not sf else d.shape[0], spin_flip=sf,
        )
        omega = getattr(qed, "omega", None)
        meta = {
            "model": "qed-sf-tda" if sf else "qed-tda-dense",
            "photon_frac": getattr(qed, "photon_frac", None),
            "approximation": "projected_electronic_nac",
            "spin_flip": sf,
            "note": (
                "SF: d_el should be excited–excited only; gpu4pyscf SF has "
                "no ground–excited NAC. CasidaPy SF (collinear) vs gpu4pyscf "
                "SF (often multicollinear) electronic manifolds may differ."
                if sf else None
            ),
        }

    if C.shape[0] != d.shape[0]:
        raise ValueError(
            f"weight rows ({C.shape[0]}) != d_el dim ({d.shape[0]}); "
            "pass matching n_label / d_el from the same electronic manifold. "
            "For spin-flip use excited-only d_el (spin_flip=True)."
        )
    return project_qed_nac(C, d, states, omega=omega, meta=meta)

