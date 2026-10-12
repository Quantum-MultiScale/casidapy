"""CasidaPy linear response: PySCF mean-field vs generic MO-based paths.

Two entry points:

* **PySCF** — pass a converged ``mf`` → ``gen_response_mf`` delegates to
  ``mf.gen_response`` (full PySCF TDDFT / CPHF response).
* **Generic** — pass ``mol`` + MO coefficients / occupations (no ``mf``) →
  ``gen_response_mos`` builds ``vind(δρ)`` from AO ``J``/``K`` integrals and
  grid ``f_xc`` on every matvec (CasidaPy-owned assembly).

``make_response`` / ``gen_response`` dispatch on which arguments are given.

Optional acceleration (``use_gpu``, ``use_mpi_response``) mirrors
:class:`~casidapy.kernels.gto.GTOKernel`: GPU uses gpu4pyscf ``gen_response``
when available; MPI promotes the mean-field to mpi4pyscf for parallel ``get_jk``.
The generic MO path bootstraps a temporary ``mf`` when either flag is set.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

Array = np.ndarray
VindFn = Callable[[Array], Array]

_OCC_THRESH = 1e-6


@dataclass
class MOSResponseValidation:
    """Result of :func:`validate_mos_response_inputs`."""

    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    nao: int = 0
    nmo: int = 0
    restricted: bool = True

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise ValueError(
                "Invalid MO / response inputs:\n  - "
                + "\n  - ".join(self.errors)
            )


def validate_mos_response_inputs(
    mol,
    mo_coeff: Array,
    mo_occ: Array,
    *,
    mo_energy: Optional[Array] = None,
    xc: Optional[str] = None,
    hermi: int = 0,
    check_orthonormality: bool = True,
    occ_threshold: float = _OCC_THRESH,
) -> MOSResponseValidation:
    """Check that ``mol`` and MO arrays are usable for the generic response path.

    Returns a :class:`MOSResponseValidation` with ``ok=False`` and populated
    ``errors`` when the inputs are inconsistent. Warnings cover soft issues
    (poor MO orthonormality, unexpected electron count).
    """
    errors: List[str] = []
    warns: List[str] = []

    if mol is None:
        return MOSResponseValidation(False, ["mol is None"])

    if not hasattr(mol, "nao_nr"):
        errors.append("mol must be a built PySCF Mole (missing nao_nr())")
        return MOSResponseValidation(False, errors)

    if not getattr(mol, "_built", True):
        errors.append("mol is not built; call mol.build() first")

    try:
        nao = int(mol.nao_nr())
    except Exception as exc:
        errors.append(f"cannot determine nao from mol: {exc}")
        return MOSResponseValidation(False, errors)

    if nao <= 0:
        errors.append(f"invalid nao={nao}")

    mo_coeff = np.asarray(mo_coeff)
    mo_occ = np.asarray(mo_occ)

    if not np.issubdtype(mo_coeff.dtype, np.floating):
        errors.append(f"mo_coeff must be floating point, got {mo_coeff.dtype}")
    if not np.issubdtype(mo_occ.dtype, np.floating):
        errors.append(f"mo_occ must be floating point, got {mo_occ.dtype}")
    if not np.all(np.isfinite(mo_coeff)):
        errors.append("mo_coeff contains non-finite values")
    if not np.all(np.isfinite(mo_occ)):
        errors.append("mo_occ contains non-finite values")

    if hermi not in (0, 1, 2):
        errors.append(f"hermi must be 0, 1, or 2; got {hermi}")

    restricted = True
    nmo = 0

    if mo_coeff.ndim == 2:
        if mo_coeff.shape[0] != nao:
            errors.append(
                f"mo_coeff shape {mo_coeff.shape} incompatible with nao={nao} "
                "(expected (nao, nmo))"
            )
        nmo = mo_coeff.shape[1] if mo_coeff.ndim == 2 else 0
        if mo_occ.ndim != 1:
            errors.append(
                f"closed-shell mo_occ must be 1-D (nmo,), got shape {mo_occ.shape}"
            )
        elif mo_occ.size != nmo:
            errors.append(
                f"mo_occ length {mo_occ.size} != nmo {nmo}"
            )
    elif mo_coeff.ndim == 3 and mo_coeff.shape[0] == 2:
        restricted = False
        if mo_coeff.shape[1] != nao:
            errors.append(
                f"UKS mo_coeff shape {mo_coeff.shape} incompatible with nao={nao}"
            )
        nmo = mo_coeff.shape[2]
        if mo_occ.ndim != 2 or mo_occ.shape[0] != 2 or mo_occ.shape[1] != nmo:
            errors.append(
                f"UKS mo_occ must have shape (2, nmo); got {mo_occ.shape}"
            )
    else:
        errors.append(
            "mo_coeff must be (nao, nmo) for RHF/RKS or (2, nao, nmo) for UKS"
        )

    if mo_energy is not None:
        mo_energy = np.asarray(mo_energy)
        if not np.all(np.isfinite(mo_energy)):
            errors.append("mo_energy contains non-finite values")
        if restricted:
            if mo_energy.ndim != 1 or mo_energy.size != nmo:
                errors.append(
                    f"mo_energy must be length nmo={nmo}; got shape {mo_energy.shape}"
                )
        else:
            if mo_energy.shape != (2, nmo):
                errors.append(
                    f"UKS mo_energy must have shape (2, nmo); got {mo_energy.shape}"
                )

    if xc is not None and not str(xc).strip():
        errors.append("xc must be a non-empty string when provided")

    if nmo > 0 and not errors:
        if restricted:
            n_occ = int(np.sum(mo_occ > occ_threshold))
            if n_occ == 0:
                errors.append("no occupied MOs (mo_occ > threshold)")
            if np.any(mo_occ < -occ_threshold):
                errors.append("mo_occ has negative entries")
        else:
            n_a = int(np.sum(mo_occ[0] > occ_threshold))
            n_b = int(np.sum(mo_occ[1] > occ_threshold))
            if n_a == 0 and n_b == 0:
                errors.append("no occupied alpha/beta MOs")
            if np.any(mo_occ < -occ_threshold):
                errors.append("mo_occ has negative entries")

        if check_orthonormality and restricted:
            try:
                s = mol.intor("int1e_ovlp")
                sm = mo_coeff.T @ s @ mo_coeff
                eye = np.eye(nmo)
                dev = float(np.max(np.abs(sm - eye)))
                if dev > 1e-4:
                    warns.append(
                        f"MOs are not orthonormal in the AO metric (max |S_mo - I| = {dev:.2e}); "
                        "imported MO ordering/basis may be wrong"
                    )
            except Exception as exc:
                warns.append(f"could not check MO orthonormality: {exc}")

        try:
            nelec_ref = mol.nelec
            if restricted:
                nelec = float(np.sum(mo_occ))
                if abs(nelec - nelec_ref[0] - nelec_ref[1]) > 0.5:
                    warns.append(
                        f"sum(mo_occ)={nelec:.2f} differs from mol.nelec "
                        f"({nelec_ref[0]}+{nelec_ref[1]})"
                    )
            else:
                na = float(np.sum(mo_occ[0]))
                nb = float(np.sum(mo_occ[1]))
                if abs(na - nelec_ref[0]) > 0.5 or abs(nb - nelec_ref[1]) > 0.5:
                    warns.append(
                        f"alpha/beta occupations ({na:.2f}, {nb:.2f}) differ from "
                        f"mol.nelec ({nelec_ref[0]}, {nelec_ref[1]})"
                    )
        except Exception:
            pass

    ok = len(errors) == 0
    return MOSResponseValidation(
        ok=ok,
        errors=errors,
        warnings=warns,
        nao=nao,
        nmo=nmo,
        restricted=restricted,
    )


def check_mos_response_inputs(
    mol,
    mo_coeff: Array,
    mo_occ: Array,
    **kwargs,
) -> MOSResponseValidation:
    """Validate MO inputs; raise :class:`ValueError` if invalid."""
    result = validate_mos_response_inputs(mol, mo_coeff, mo_occ, **kwargs)
    result.raise_if_invalid()
    return result


def validate_response_dm(
    dm: Array,
    mol,
    *,
    hermi: int = 0,
) -> None:
    """Raise :class:`ValueError` if a density matrix has the wrong shape/type."""
    if hermi not in (0, 1, 2):
        raise ValueError(f"hermi must be 0, 1, or 2; got {hermi}")
    nao = int(mol.nao_nr())
    dm = np.asarray(dm)
    if not np.issubdtype(dm.dtype, np.floating):
        raise ValueError(f"dm must be floating point, got {dm.dtype}")
    if dm.ndim == 2:
        if dm.shape != (nao, nao):
            raise ValueError(f"dm shape {dm.shape} != ({nao}, {nao})")
    elif dm.ndim == 3:
        if dm.shape == (2, nao, nao):
            pass
        elif dm.shape[1:] == (nao, nao):
            pass
        else:
            raise ValueError(f"unrecognized dm shape {dm.shape} for nao={nao}")
    else:
        raise ValueError(f"dm must be 2-D or 3-D, got ndim={dm.ndim}")
    if not np.all(np.isfinite(dm)):
        raise ValueError("dm contains non-finite values")


@dataclass
class MOSResponseInputs:
    """Minimal state for the generic (non-``mf``) response solver."""

    mol: object
    mo_coeff: Array
    mo_occ: Array
    mo_energy: Optional[Array] = None
    xc: str = "pbe"
    use_df: bool = False
    max_memory: float = 2000.0


def make_response(
    mf=None,
    *,
    mol=None,
    mo_coeff: Optional[Array] = None,
    mo_occ: Optional[Array] = None,
    mo_energy: Optional[Array] = None,
    xc: Optional[str] = None,
    use_df: bool = False,
    use_gpu: bool = False,
    use_mpi_response: bool = False,
    singlet: Optional[bool] = None,
    hermi: int = 0,
    max_memory: Optional[float] = None,
    with_nlc: bool = True,
    with_j: bool = True,
    validate: bool = True,
    verbose: bool = False,
) -> VindFn:
    """Build ``vind(dm1)`` from either a PySCF ``mf`` or bare MO data.

    Parameters
    ----------
    mf
        Converged PySCF mean-field. When set, the PySCF ``gen_response`` path
        is used and ``mol`` / MO kwargs are ignored.
    mol, mo_coeff, mo_occ
        Generic path: AO basis + MOs (``mo_energy`` optional, stored for
        bookkeeping only). Requires ``xc`` for KS (``"hf"`` / ``"rhf"`` for
        Hartree–Fock).
    use_gpu
        Use gpu4pyscf ``gen_response`` when available (mf path, or bootstrap
        mf from MOs on the generic path).
    use_mpi_response
        Promote ``mf`` to mpi4pyscf for MPI ``get_jk`` (disables ``use_df``).
    validate
        Run :func:`check_mos_response_inputs` on the generic path.
    singlet, hermi, with_nlc, with_j
        Passed through to the selected backend (same meaning as PySCF).
    """
    if use_mpi_response and use_gpu:
        raise ValueError(
            "use_mpi_response and use_gpu cannot be combined; "
            "pick MPI JK (mpi4pyscf) or GPU response (gpu4pyscf)."
        )
    if use_mpi_response and use_df:
        warnings.warn(
            "use_mpi_response=True disables density fitting for the response "
            "(mpi4pyscf MPI get_jk is direct).",
            RuntimeWarning,
            stacklevel=2,
        )
        use_df = False

    if mf is not None:
        return gen_response_mf(
            mf,
            mo_coeff=mo_coeff,
            mo_occ=mo_occ,
            singlet=singlet,
            hermi=hermi,
            max_memory=max_memory,
            with_nlc=with_nlc,
            with_j=with_j,
            use_gpu=use_gpu,
            use_mpi_response=use_mpi_response,
            use_df=use_df,
            verbose=verbose,
        )
    if mol is None or mo_coeff is None or mo_occ is None:
        raise TypeError(
            "make_response: pass a PySCF mf, or (mol, mo_coeff, mo_occ) "
            "for the generic MO solver."
        )
    return gen_response_mos(
        mol,
        mo_coeff,
        mo_occ,
        mo_energy=mo_energy,
        xc=xc or "pbe",
        use_df=use_df,
        use_gpu=use_gpu,
        use_mpi_response=use_mpi_response,
        singlet=singlet,
        hermi=hermi,
        max_memory=max_memory,
        with_nlc=with_nlc,
        with_j=with_j,
        validate=validate,
        verbose=verbose,
    )


def gen_response(
    mf=None,
    mo_coeff: Optional[Array] = None,
    mo_occ: Optional[Array] = None,
    *,
    mol=None,
    mo_energy: Optional[Array] = None,
    xc: Optional[str] = None,
    use_df: bool = False,
    use_gpu: bool = False,
    use_mpi_response: bool = False,
    singlet: Optional[bool] = None,
    hermi: int = 0,
    max_memory: Optional[float] = None,
    with_nlc: bool = True,
    with_j: bool = True,
    validate: bool = True,
    verbose: bool = False,
) -> VindFn:
    """Alias for :func:`make_response` (backward compatible)."""
    return make_response(
        mf,
        mol=mol,
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        xc=xc,
        use_df=use_df,
        use_gpu=use_gpu,
        use_mpi_response=use_mpi_response,
        singlet=singlet,
        hermi=hermi,
        max_memory=max_memory,
        with_nlc=with_nlc,
        with_j=with_j,
        validate=validate,
        verbose=verbose,
    )


def _prepare_mf_for_response(
    mf,
    *,
    use_gpu: bool = False,
    use_mpi_response: bool = False,
    use_df: bool = False,
    mol=None,
    mo_coeff=None,
    mo_energy=None,
    mo_occ=None,
    xc: Optional[str] = None,
    verbose: bool = False,
):
    """Promote *mf* for MPI / GPU response (mirrors GTOKernel.setup)."""
    if use_mpi_response:
        from casidapy.adapter.pyscf import ensure_mpi_mf

        mf = ensure_mpi_mf(mf)
    if hasattr(mf, "xc") and (
        getattr(mf, "grids", None) is None or mf.grids.coords is None
    ):
        mf.grids.build()
    if use_df and getattr(mf, "with_df", None) is None:
        from casidapy.adapter.pyscf import density_fit_mf

        mf = density_fit_mf(mf, verbose=verbose)

    if use_gpu:
        from casidapy.adapter.pyscf import move_mf_to_gpu

        _mol = mol if mol is not None else mf.mol
        _xc = getattr(mf, "xc", xc or "pbe")
        gmf = move_mf_to_gpu(
            mf,
            mol=_mol,
            mo_coeff=mo_coeff if mo_coeff is not None else mf.mo_coeff,
            mo_energy=mo_energy if mo_energy is not None else mf.mo_energy,
            mo_occ=mo_occ if mo_occ is not None else mf.mo_occ,
            xc=_xc,
            use_df=use_df and not use_mpi_response,
            verbose=verbose,
        )
        if gmf is not None:
            return gmf
        warnings.warn(
            "gpu4pyscf not available or mf.to_gpu() failed; using CPU gen_response.",
            RuntimeWarning,
            stacklevel=3,
        )
    return mf


def _mf_supports_singlet_response(mf) -> bool:
    """Closed-shell RKS/RHF ``gen_response`` accepts ``singlet``; ROKS/UKS do not."""
    from pyscf import dft, scf

    if isinstance(
        mf,
        (
            dft.roks.ROKS,
            dft.uks.UKS,
            scf.uhf.UHF,
            scf.rohf.ROHF,
        ),
    ):
        return False
    mo = getattr(mf, "mo_coeff", None)
    if mo is not None and np.asarray(mo).ndim == 3:
        return False
    return True


def mf_uses_rohf_collinear_response(mf) -> bool:
    """True when ``mf.gen_response`` expects ROHF collinear (2, nao, nao) I/O.

    Collinear ROKS/ROHF with 2-D MO coefficients cannot batch transition DMs as
  ``(k, nao, nao)`` and returns spin-resolved potentials.  CVS singlet coupling on
    these references is evaluated via :func:`gen_response_mos` instead.
    """
    from pyscf import dft, scf

    if isinstance(mf, (dft.roks.ROKS, scf.rohf.ROHF)):
        mo = getattr(mf, "mo_coeff", None)
        return mo is not None and np.asarray(mo).ndim == 2
    return False


def gen_response_mf(
    mf,
    mo_coeff: Optional[Array] = None,
    mo_occ: Optional[Array] = None,
    *,
    singlet: Optional[bool] = None,
    hermi: int = 0,
    max_memory: Optional[float] = None,
    with_nlc: bool = True,
    with_j: bool = True,
    use_gpu: bool = False,
    use_mpi_response: bool = False,
    use_df: bool = False,
    verbose: bool = False,
) -> VindFn:
    """PySCF mean-field response — calls ``mf.gen_response`` directly."""
    mf = _prepare_mf_for_response(
        mf,
        use_gpu=use_gpu,
        use_mpi_response=use_mpi_response,
        use_df=use_df,
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        verbose=verbose,
    )
    kwargs = {"hermi": hermi}
    if singlet is not None and _mf_supports_singlet_response(mf):
        kwargs["singlet"] = singlet
    if mo_coeff is not None:
        kwargs["mo_coeff"] = mo_coeff
    if mo_occ is not None:
        kwargs["mo_occ"] = mo_occ
    if max_memory is not None:
        kwargs["max_memory"] = max_memory
    if with_nlc is not None:
        try:
            vind = mf.gen_response(**kwargs, with_nlc=with_nlc)
            return _wrap_vind(vind, mf, hermi)
        except TypeError:
            pass
    try:
        vind = mf.gen_response(**kwargs, with_j=with_j)
    except TypeError:
        kwargs.pop("singlet", None)
        try:
            vind = mf.gen_response(**kwargs, with_j=with_j)
        except TypeError:
            vind = mf.gen_response(**kwargs)
    return _wrap_vind(vind, mf, hermi)


def _bootstrap_mf_from_mos(
    mol,
    mo_coeff: Array,
    mo_occ: Array,
    mo_energy: Optional[Array],
    xc: str,
    *,
    use_df: bool = False,
):
    """Build a minimal PySCF mean-field from bare MO data."""
    from pyscf import dft, scf

    if mo_coeff.ndim == 3:
        mf = dft.UKS(mol)
        mf.xc = xc
    elif _is_hf_xc(xc):
        mf = scf.RHF(mol)
    else:
        mf = dft.RKS(mol)
        mf.xc = xc
    mf.mo_coeff = np.asarray(mo_coeff, dtype=float)
    mf.mo_occ = np.asarray(mo_occ, dtype=float)
    if mo_energy is not None:
        mf.mo_energy = np.asarray(mo_energy, dtype=float)
    if use_df:
        from casidapy.adapter.pyscf import density_fit_mf

        mf = density_fit_mf(mf, mol=mol, xc=xc)
    if hasattr(mf, "grids") and (
        getattr(mf, "grids", None) is None or mf.grids.coords is None
    ):
        mf.grids.build()
    return mf


def _wrap_vind(vind: VindFn, mol_or_mf, hermi: int) -> VindFn:
    """Optional DM shape check before calling the underlying ``vind``."""

    def wrapped(dm: Array) -> Array:
        validate_response_dm(dm, getattr(mol_or_mf, "mol", mol_or_mf), hermi=hermi)
        return vind(dm)

    return wrapped


def _resolve_max_memory(max_memory: Optional[float], default: float = 2000.0) -> float:
    if max_memory is not None:
        return float(max_memory)
    try:
        from pyscf import lib

        mem_now = lib.current_memory()[0]
        return max(default, 4000.0 - mem_now)
    except Exception:
        return default


def _is_hf_xc(xc: str) -> bool:
    return str(xc).lower().strip() in ("hf", "rhf", "uhf")


def _build_jk_driver(mol, use_df: bool):
    """Return ``(get_jk, get_j, get_k)`` callables for AO Coulomb / exchange."""
    from pyscf import scf

    if not use_df:

        def get_jk(dm, hermi=0, with_j=True, with_k=True, omega=None):
            return scf.hf.get_jk(
                mol, dm, hermi=hermi, with_j=with_j, with_k=with_k, omega=omega
            )

        def get_j(dm, hermi=0, omega=None):
            vj, _ = scf.hf.get_jk(
                mol, dm, hermi=hermi, with_j=True, with_k=False, omega=omega
            )
            return vj

        def get_k(dm, hermi=0, omega=None):
            _, vk = scf.hf.get_jk(
                mol, dm, hermi=hermi, with_j=False, with_k=True, omega=omega
            )
            return vk

        return get_jk, get_j, get_k

    from pyscf import df

    if not hasattr(df, "DFJK"):
        raise AttributeError(
            "pyscf.df.DFJK is unavailable in this PySCF build; "
            "set use_df=False for the generic MO response path."
        )
    auxmol = df.make_auxmol(mol)
    jkdf = df.DFJK(mol, auxmol)

    def get_jk(dm, hermi=0, with_j=True, with_k=True, omega=None):
        return jkdf.get_jk(dm, hermi=hermi, with_j=with_j, with_k=with_k, omega=omega)

    def get_j(dm, hermi=0, omega=None):
        return jkdf.get_j(dm, hermi=hermi, omega=omega)

    def get_k(dm, hermi=0, omega=None):
        return jkdf.get_k(dm, hermi=hermi, omega=omega)

    return get_jk, get_j, get_k


def _rks_vind_hybrid_part_mol(
    get_jk,
    get_j,
    get_k,
    dm1: Array,
    hermi: int,
    omega: float,
    alpha: float,
    hyb: float,
    hybrid: bool,
    *,
    triplet: bool = False,
) -> Array:
    if not hybrid:
        if hermi == 2 or triplet:
            return np.zeros_like(dm1)
        return get_j(dm1, hermi=hermi)

    if triplet:
        if omega == 0:
            vk = get_k(dm1, hermi) * hyb
        elif alpha == 0:
            vk = get_k(dm1, hermi, omega=-omega) * hyb
        elif hyb == 0:
            vk = get_k(dm1, hermi, omega=omega) * alpha
        else:
            vk = get_k(dm1, hermi) * hyb
            vk = vk + get_k(dm1, hermi, omega=omega) * (alpha - hyb)
        return -0.5 * vk

    if omega == 0:
        vj, vk = get_jk(dm1, hermi)
        vk = np.asarray(vk) * hyb
    elif alpha == 0:
        vj = get_j(dm1, hermi)
        vk = get_k(dm1, hermi, omega=-omega) * hyb
    elif hyb == 0:
        vj = get_j(dm1, hermi)
        vk = get_k(dm1, hermi, omega=omega) * alpha
    else:
        vj, vk = get_jk(dm1, hermi)
        vk = np.asarray(vk) * hyb
        vk = vk + get_k(dm1, hermi, omega=omega) * (alpha - hyb)
    if hermi != 2:
        return np.asarray(vj) - 0.5 * np.asarray(vk)
    return -0.5 * np.asarray(vk)


def gen_response_mos(
    mol,
    mo_coeff: Array,
    mo_occ: Array,
    *,
    mo_energy: Optional[Array] = None,
    xc: str = "pbe",
    use_df: bool = False,
    use_gpu: bool = False,
    use_mpi_response: bool = False,
    singlet: Optional[bool] = None,
    hermi: int = 0,
    max_memory: Optional[float] = None,
    with_nlc: bool = True,
    with_j: bool = True,
    validate: bool = True,
    verbose: bool = False,
) -> VindFn:
    """Generic MO-based TDDFT response (J/K + f_xc per matvec, no ``mf``).

    Uses PySCF integral engines (``get_jk``, ``NumInt``) but does not require
    a converged SCF object — only ``mol`` and MO coefficients / occupations.
    ``mo_energy`` is accepted for API symmetry but is not used in the matvec.

    When ``use_gpu`` or ``use_mpi_response`` is True, a temporary mean-field
    is bootstrapped and :func:`gen_response_mf` is used instead.
    """
    if validate:
        check_mos_response_inputs(
            mol, mo_coeff, mo_occ, mo_energy=mo_energy, xc=xc, hermi=hermi
        )

    mo_coeff = np.asarray(mo_coeff)
    mo_occ = np.asarray(mo_occ)

    if use_gpu or use_mpi_response:
        mf = _bootstrap_mf_from_mos(
            mol, mo_coeff, mo_occ, mo_energy, xc, use_df=use_df and not use_mpi_response
        )
        return gen_response_mf(
            mf,
            singlet=singlet,
            hermi=hermi,
            max_memory=max_memory,
            with_nlc=with_nlc,
            with_j=with_j,
            use_gpu=use_gpu,
            use_mpi_response=use_mpi_response,
            use_df=use_df,
            verbose=verbose,
        )

    _ = mo_energy  # bookkeeping only

    max_memory = _resolve_max_memory(max_memory)
    get_jk, get_j, get_k = _build_jk_driver(mol, use_df)

    # Open-shell UKS layout: mo_coeff shape (2, nao, nmo)
    if mo_coeff.ndim == 3 and mo_coeff.shape[0] == 2:
        vind = _gen_uks_response_mos(
            mol,
            mo_coeff,
            mo_occ,
            xc=xc,
            use_df=use_df,
            hermi=hermi,
            max_memory=max_memory,
            with_nlc=with_nlc,
            with_j=with_j,
            get_jk=get_jk,
            get_j=get_j,
            get_k=get_k,
        )
        return _wrap_vind(vind, mol, hermi)

    # Closed-shell RHF / RKS
    if _is_hf_xc(xc):

        def vind(dm1: Array) -> Array:
            if (singlet is None or singlet) and hermi != 2:
                vj, vk = get_jk(dm1, hermi=hermi)
                return np.asarray(vj) - 0.5 * np.asarray(vk)
            return -0.5 * get_k(dm1, hermi=hermi)

        return _wrap_vind(vind, mol, hermi)

    from pyscf import dft

    ni = dft.numint.NumInt()
    ni.libxc.test_deriv_order(xc, 2, raise_error=True)
    grids = dft.gen_grid.Grids(mol)
    grids.build()
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(xc, mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(xc)
    spin = 0 if singlet is None else 1
    rho0, vxc, fxc = ni.cache_xc_kernel(mol, grids, xc, mo_coeff, mo_occ, spin)
    dm0 = None

    if singlet is None:

        def vind(dm1: Array) -> Array:
            if hermi == 2:
                v1 = np.zeros_like(dm1)
            else:
                v1 = ni.nr_rks_fxc(
                    mol, grids, xc, dm0, dm1, 0, hermi,
                    rho0, vxc, fxc, max_memory=max_memory,
                )
            v1 = v1 + _rks_vind_hybrid_part_mol(
                get_jk, get_j, get_k, dm1, hermi, omega, alpha, hyb, hybrid,
                triplet=False,
            )
            return v1

    elif singlet:
        fxc_s = fxc * 0.5

        def vind(dm1: Array) -> Array:
            if hermi == 2:
                v1 = np.zeros_like(dm1)
            else:
                v1 = ni.nr_rks_fxc_st(
                    mol, grids, xc, dm0, dm1, hermi, True,
                    rho0, vxc, fxc_s, max_memory=max_memory,
                )
            v1 = v1 + _rks_vind_hybrid_part_mol(
                get_jk, get_j, get_k, dm1, hermi, omega, alpha, hyb, hybrid,
                triplet=False,
            )
            return v1

    else:
        fxc_t = fxc * 0.5

        def vind(dm1: Array) -> Array:
            if hermi == 2:
                v1 = np.zeros_like(dm1)
            else:
                v1 = ni.nr_rks_fxc_st(
                    mol, grids, xc, dm0, dm1, hermi, False,
                    rho0, vxc, fxc_t, max_memory=max_memory,
                )
            v1 = v1 + _rks_vind_hybrid_part_mol(
                get_jk, get_j, get_k, dm1, hermi, omega, alpha, hyb, hybrid,
                triplet=True,
            )
            return v1

    return _wrap_vind(vind, mol, hermi)


def _gen_uks_response_mos(
    mol,
    mo_coeff: Array,
    mo_occ: Array,
    *,
    xc: str,
    use_df: bool,
    hermi: int,
    max_memory: float,
    with_nlc: bool,
    with_j: bool,
    get_jk,
    get_j,
    get_k,
) -> VindFn:
    from pyscf import dft

    if _is_hf_xc(xc):
        if with_j:

            def vind(dm1: Array) -> Array:
                vj, vk = get_jk(dm1, hermi=hermi)
                return vj[0] + vj[1] - vk

        else:

            def vind(dm1: Array) -> Array:
                return -get_k(dm1, hermi=hermi)

        return vind

    ni = dft.numint.NumInt()
    ni.libxc.test_deriv_order(xc, 2, raise_error=True)
    grids = dft.gen_grid.Grids(mol)
    grids.build()
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(xc, mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(xc)
    rho0, vxc, fxc = ni.cache_xc_kernel(mol, grids, xc, mo_coeff, mo_occ, 1)
    dm0 = None

    def vind(dm1: Array) -> Array:
        if hermi == 2:
            v1 = np.zeros_like(dm1)
        else:
            v1 = ni.nr_uks_fxc(
                mol, grids, xc, dm0, dm1, 0, hermi,
                rho0, vxc, fxc, max_memory=max_memory,
            )
        if not hybrid:
            if with_j:
                vj = get_j(dm1, hermi=hermi)
                v1 = v1 + vj[0] + vj[1]
        else:
            if omega == 0:
                vj, vk = get_jk(dm1, hermi, with_j=with_j)
                vk = np.asarray(vk) * hyb
            elif alpha == 0:
                vj = get_j(dm1, hermi) if with_j else None
                vk = get_k(dm1, hermi, omega=-omega) * hyb
            elif hyb == 0:
                vj = get_j(dm1, hermi) if with_j else None
                vk = get_k(dm1, hermi, omega=omega) * alpha
            else:
                vj, vk = get_jk(dm1, hermi, with_j=with_j)
                vk = np.asarray(vk) * hyb
                vk = vk + get_k(dm1, hermi, omega=omega) * (alpha - hyb)
            if with_j:
                v1 = v1 + vj[0] + vj[1] - vk
            else:
                v1 = v1 - vk
        return v1

    return vind
