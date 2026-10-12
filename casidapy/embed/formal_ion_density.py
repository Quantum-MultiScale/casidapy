"""Formal-oxidation-state atomic reference densities for Hirshfeld.

Path A for charged-ion stockholder references: instead of neutral UPF
``PP_RHOATOM``, build a spherically averaged atomic DFT density at the
formal charge from :func:`~casidapy.xas.cvs.resolve_oxidation_state`
(Ti⁴⁺, O²⁻, …), cache it, and project onto the PW grid the same way as
:func:`~casidapy.embed.hirshfeld.upf_rhoatom_on_grid`.

By default the density is **valence-only** (PP cores stripped) so the
promolecule matches the host QE valence ``ρ_tot``. Toggle with
``valence_only=False`` for full AE references.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_FORMAL_ION_BASIS = "aug-cc-pvdz"
DEFAULT_FORMAL_ION_XC = "pbe"
DEFAULT_FORMAL_ION_RMAX = 12.0
DEFAULT_FORMAL_ION_NRAD = 400

# (symbol, ox, basis, xc, valence_only, z_valence_key, n_rad, r_max) → (r, ρ)
_RHOATOM_CACHE: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}


def clear_formal_ion_rhoatom_cache() -> None:
    """Drop in-memory formal-ion radial densities (tests / basis changes)."""
    _RHOATOM_CACHE.clear()


def _cache_key(
    symbol: str,
    oxidation: int,
    *,
    basis: str,
    xc: str,
    valence_only: bool,
    z_valence: Optional[float],
    n_rad: int,
    r_max: float,
) -> tuple:
    zv = None if z_valence is None else round(float(z_valence), 6)
    return (
        str(symbol).strip().capitalize(),
        int(oxidation),
        str(basis).lower().strip(),
        str(xc).lower().strip(),
        bool(valence_only),
        zv,
        int(n_rad),
        round(float(r_max), 6),
    )


def _radial_mesh(n_rad: int, r_max: float) -> np.ndarray:
    n = max(8, int(n_rad))
    r_max = float(r_max)
    # Match UPF-like coverage: dense near nucleus, extends to ~r_max.
    return np.concatenate([[0.0], np.geomspace(1.0e-4, r_max, n - 1)])


def _n_core_orbitals(z_atomic: int, z_valence: float) -> int:
    n_core_elec = max(0, int(z_atomic) - int(round(float(z_valence))))
    return int(n_core_elec // 2)


def _make_atomic_mf(symbol: str, charge: int, *, basis: str, xc: str, omega: float = 0.0):
    """Restricted/unrestricted KS atom at formal charge; optional r² confinement."""
    from pyscf import gto, dft

    sym = str(symbol).strip().capitalize()
    z = int(gto.charge(sym))
    ne = z - int(charge)
    if ne < 0:
        raise ValueError(f"{sym} charge={charge} removes more than Z={z} electrons")
    if ne == 0:
        return None, z, 0

    spin = ne % 2
    mol = gto.M(
        atom=f"{sym} 0 0 0",
        basis=basis,
        charge=int(charge),
        spin=int(spin),
        unit="Angstrom",
        verbose=0,
    )
    mf = dft.RKS(mol) if spin == 0 else dft.UKS(mol)
    mf.xc = xc
    mf.max_cycle = 120
    mf.conv_tol = 1.0e-7
    if abs(float(omega)) > 0.0:
        # Soft Watson sphere: ½ ω² r² — helps unbound anions if bare SCF fails.
        h0 = mf.get_hcore
        r2 = mol.intor_symmetric("int1e_r2")
        w2 = 0.5 * float(omega) ** 2

        def _hcore(*args, **kwargs):
            return h0(*args, **kwargs) + w2 * r2

        mf.get_hcore = _hcore
        logger.info(
            "[formal_ion] %s charge=%+d using confinement ω=%.3f Ha^{1/2}",
            sym,
            charge,
            omega,
        )
    return mf, z, ne


def _density_matrix_valence(mf, n_core_orbitals: int):
    """Drop deepest PP-core spatial orbitals from the atomic DM."""
    n_core = max(0, int(n_core_orbitals))
    mo_coeff = mf.mo_coeff
    mo_occ = np.asarray(mf.mo_occ, dtype=float).copy()
    if mo_occ.ndim == 1:
        if n_core > 0:
            n_core = min(n_core, mo_occ.size)
            mo_occ[:n_core] = 0.0
        return mf.make_rdm1(mo_coeff, mo_occ)
    # UKS: (2, nao)
    if n_core > 0:
        for s in range(mo_occ.shape[0]):
            n_s = min(n_core, mo_occ[s].size)
            mo_occ[s, :n_s] = 0.0
    return mf.make_rdm1(mo_coeff, mo_occ)


def _spherical_average_rho(mf, dm, r: np.ndarray) -> np.ndarray:
    """Angle-averaged ρ(r) in e⁻/Bohr³ on radial mesh ``r`` (Bohr)."""
    from pyscf.dft import numint

    mol = mf.mol
    ni = numint.NumInt()
    # Six cardinal rays — enough for closed-shell ions; averages open shells.
    directions = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    rho_acc = np.zeros(r.size, dtype=float)
    r_eval = np.maximum(r, 1.0e-8)
    for d in directions:
        coords = np.outer(r_eval, d)
        ao = ni.eval_ao(mol, coords)
        rho_acc += np.asarray(ni.eval_rho(mol, ao, dm), dtype=float)
    rho = rho_acc / float(len(directions))
    if r[0] <= 1e-12 and rho.size > 1:
        rho = rho.copy()
        rho[0] = rho[1]
    return np.maximum(rho, 0.0)


def compute_formal_ion_rhoatom(
    symbol: str,
    oxidation_state: int,
    *,
    basis: str = DEFAULT_FORMAL_ION_BASIS,
    xc: str = DEFAULT_FORMAL_ION_XC,
    valence_only: bool = True,
    z_valence: Optional[float] = None,
    n_rad: int = DEFAULT_FORMAL_ION_NRAD,
    r_max: float = DEFAULT_FORMAL_ION_RMAX,
    confine_omega: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Atomic ``(r_bohr, ρ)`` at formal charge (e⁻/Bohr³, not 4πr²ρ).

    ``valence_only``
        If true and ``z_valence`` is set, strip PP-core orbitals so the
        integrated density ≈ ``z_valence − oxidation_state`` (matches QE
        valence tables). If ``z_valence`` is omitted, keep the full AE density.
    """
    sym = str(symbol).strip().capitalize()
    ox = int(oxidation_state)
    r = _radial_mesh(n_rad, r_max)

    mf, z, ne = _make_atomic_mf(sym, ox, basis=basis, xc=xc, omega=confine_omega)
    if mf is None:
        return r, np.zeros_like(r)

    e = mf.kernel()
    if not mf.converged and abs(float(confine_omega)) < 1e-15 and ox < 0:
        # Unbound anion retry with mild confinement.
        logger.warning(
            "[formal_ion] %s%+d SCF did not converge (E=%s); retry ω=0.15",
            sym,
            ox,
            e,
        )
        return compute_formal_ion_rhoatom(
            sym,
            ox,
            basis=basis,
            xc=xc,
            valence_only=valence_only,
            z_valence=z_valence,
            n_rad=n_rad,
            r_max=r_max,
            confine_omega=0.15,
        )
    if not mf.converged:
        logger.warning(
            "[formal_ion] %s%+d SCF not converged (E=%s); using last density",
            sym,
            ox,
            e,
        )

    n_core = 0
    if valence_only and z_valence is not None:
        n_core = _n_core_orbitals(z, float(z_valence))
    dm = _density_matrix_valence(mf, n_core) if n_core else mf.make_rdm1()
    rho = _spherical_average_rho(mf, dm, r)

    # Sanity: integrated charge
    q = float(np.trapz(4.0 * np.pi * r**2 * rho, r))
    if valence_only and z_valence is not None:
        q_exp = float(z_valence) - float(ox)
    else:
        q_exp = float(ne)
    if q_exp > 0.5 and abs(q - q_exp) > 0.25 * max(1.0, abs(q_exp)):
        logger.warning(
            "[formal_ion] %s%+d radial integral %.3f vs expected %.3f "
            "(valence_only=%s z_val=%s)",
            sym,
            ox,
            q,
            q_exp,
            valence_only,
            z_valence,
        )
    return r, rho


def formal_ion_rhoatom(
    symbol: str,
    oxidation_state: int,
    *,
    basis: str = DEFAULT_FORMAL_ION_BASIS,
    xc: str = DEFAULT_FORMAL_ION_XC,
    valence_only: bool = True,
    z_valence: Optional[float] = None,
    n_rad: int = DEFAULT_FORMAL_ION_NRAD,
    r_max: float = DEFAULT_FORMAL_ION_RMAX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cached :func:`compute_formal_ion_rhoatom`."""
    key = _cache_key(
        symbol,
        oxidation_state,
        basis=basis,
        xc=xc,
        valence_only=valence_only,
        z_valence=z_valence,
        n_rad=n_rad,
        r_max=r_max,
    )
    hit = _RHOATOM_CACHE.get(key)
    if hit is not None:
        return hit[0].copy(), hit[1].copy()
    r, rho = compute_formal_ion_rhoatom(
        symbol,
        oxidation_state,
        basis=basis,
        xc=xc,
        valence_only=valence_only,
        z_valence=z_valence,
        n_rad=n_rad,
        r_max=r_max,
    )
    _RHOATOM_CACHE[key] = (r, rho)
    return r.copy(), rho.copy()


def formal_ion_rhoatom_on_grid(
    driver,
    atom_index: int,
    grid,
    oxidation_state: int,
    *,
    upf_path=None,
    basis: str = DEFAULT_FORMAL_ION_BASIS,
    xc: str = DEFAULT_FORMAL_ION_XC,
    valence_only: bool = True,
):
    """Formal-ion ``ρ̃`` for one atom on the QE charge grid."""
    from casidapy.embed.hirshfeld import project_radial_density_to_grid
    from casidapy.embed.potential import read_upf_z_valence, resolve_upf_path

    atom_index = int(atom_index)
    sym = driver.get_ase_atoms().get_chemical_symbols()[atom_index]
    z_valence = None
    if valence_only:
        path = resolve_upf_path(driver, sym, upf_path)
        z_valence = float(read_upf_z_valence(path))
    r_rad, rho_rad = formal_ion_rhoatom(
        sym,
        int(oxidation_state),
        basis=basis,
        xc=xc,
        valence_only=valence_only,
        z_valence=z_valence,
    )
    ions = driver.get_dftpy_ions()
    return project_radial_density_to_grid(
        r_rad, rho_rad, grid, ions.positions[atom_index], ions.cell
    )
