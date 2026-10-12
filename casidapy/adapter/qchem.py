"""Import Q-Chem wavefunctions for CasidaPy (fchk → generic MO response).

Q-Chem writes ``<job>.fchk`` (Gaussian-style formatted checkpoint) when run
through ``$QC/bin/qchem``. The reader here extracts geometry, MO energies,
and MO coefficients; the **basis must be supplied separately** as a PySCF
basis string matching the Q-Chem job (e.g. ``"6-31g"``, ``"def2-svp"``).

Typical workflow::

    # 1) On Amarel (after sourcing setup_qchem_amarel.sh):
    #    qchem h2.in h2.out   → h2.fchk

    from casidapy.adapter.qchem import extract_gto_kernel_from_fchk
    kernel, opts = extract_gto_kernel_from_fchk(
        "h2.fchk", basis="6-31g", xc="hf", n_states=4,
    )
    kernel.setup(tda=True)   # generic-mos response (mf=None)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from casidapy.casida_api import CasidaOptions
from casidapy.kernels.gto import GTOKernel

_FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[Ee][-+]?\d+)?")


_HEADER_RE = re.compile(
    r"^(.+?)\s+(I|R)\s+(?:N=\s*(\d+)|([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?))\s*$"
)


def _parse_fchk_header(line: str) -> Optional[Tuple[str, str, int, Optional[list]]]:
    """Parse one fchk section header (Gaussian or Q-Chem layout)."""
    # PySCF/Q-Chem test writer: 45-char label + dtype + right-aligned count.
    if len(line) >= 47 and line[45] in "IR":
        label = line[:45].strip()
        dtype = line[45]
        rest = line[46:].strip()
        if rest.startswith("N="):
            return label, dtype, int(rest[2:].strip()), None
        if re.fullmatch(r"[-+]?\d+", rest):
            return label, dtype, int(rest), None
        if dtype == "R" and _FLOAT_RE.fullmatch(rest):
            return label, dtype, 1, [float(rest)]

    m = _HEADER_RE.match(line.strip())
    if m is None:
        return None
    label = m.group(1).strip()
    dtype = m.group(2)
    if m.group(3) is not None:
        return label, dtype, int(m.group(3)), None
    val = m.group(4)
    if dtype == "R" and ("." in val or "e" in val.lower()):
        return label, dtype, 1, [float(val)]
    if dtype == "I":
        return label, dtype, 1, [int(val)]
    return label, dtype, int(val), None


def _read_fchk_blocks(fh) -> Dict[str, np.ndarray]:
    """Parse Gaussian/Q-Chem formatted checkpoint into label → array."""
    data: Dict[str, list] = {}

    def _read_values(dtype: str, n: int):
        vals: list = []
        while len(vals) < n:
            line = fh.readline()
            if not line:
                raise EOFError("Unexpected end of fchk while reading data block")
            if dtype == "I":
                vals.extend(int(x) for x in _FLOAT_RE.findall(line))
            else:
                vals.extend(float(x) for x in _FLOAT_RE.findall(line))
        return vals[:n]

    for line in fh:
        parsed = _parse_fchk_header(line)
        if parsed is None:
            continue
        label, dtype, n, inline = parsed
        if inline is not None:
            data[label] = np.asarray(inline)
        else:
            data[label] = np.asarray(_read_values(dtype, n))

    return {k: np.asarray(v) for k, v in data.items()}


def read_fchk(path: Union[str, Path]) -> Dict[str, Any]:
    """Read an fchk file; return raw blocks plus parsed MO arrays."""
    path = Path(path)
    with path.open() as fh:
        blocks = _read_fchk_blocks(fh)

    meta: Dict[str, Any] = {"path": str(path.resolve()), "blocks": blocks}

    if "Atomic numbers" not in blocks:
        raise KeyError(f"{path}: missing 'Atomic numbers' in fchk")
    if "Current cartesian coordinates" not in blocks:
        raise KeyError(f"{path}: missing 'Current cartesian coordinates' in fchk")

    z = blocks["Atomic numbers"].astype(int)
    coords = blocks["Current cartesian coordinates"].astype(float).reshape(-1, 3)
    meta["atomic_numbers"] = z
    meta["coords_bohr"] = coords

    if "Charge" in blocks:
        meta["charge"] = int(blocks["Charge"][0])
    if "Multiplicity" in blocks:
        meta["multiplicity"] = int(blocks["Multiplicity"][0])
    if "Number of alpha electrons" in blocks:
        meta["nalpha"] = int(blocks["Number of alpha electrons"][0])
    if "Number of beta electrons" in blocks:
        meta["nbeta"] = int(blocks["Number of beta electrons"][0])
    if "Number of basis functions" in blocks:
        meta["nao"] = int(blocks["Number of basis functions"][0])
    if "Total Energy" in blocks:
        meta["energy_ha"] = float(blocks["Total Energy"][0])

    # MO keys differ slightly between programs
    e_key = _first_key(blocks, ("Alpha Orbital Energies", "Alpha orbital energies"))
    c_key = _first_key(blocks, ("Alpha MO coefficients", "Alpha MO Coefficients"))
    if e_key is None or c_key is None:
        raise KeyError(f"{path}: fchk has no Alpha MO energies/coefficients")

    mo_energy = blocks[e_key].astype(float)
    nmo = mo_energy.size
    nao = int(meta.get("nao", blocks[c_key].size // nmo))
    mo_coeff = blocks[c_key].astype(float).reshape(nao, nmo, order="F")

    meta["mo_energy"] = mo_energy
    meta["mo_coeff"] = mo_coeff
    meta["restricted"] = True

    e_beta = _first_key(blocks, ("Beta Orbital Energies", "Beta orbital energies"))
    c_beta = _first_key(blocks, ("Beta MO coefficients", "Beta MO Coefficients"))
    if e_beta is not None and c_beta is not None:
        mo_e_beta = blocks[e_beta].astype(float)
        mo_c_beta = blocks[c_beta].astype(float).reshape(nao, nmo, order="F")
        meta["mo_energy"] = np.stack([mo_energy, mo_e_beta], axis=0)
        meta["mo_coeff"] = np.stack([mo_coeff, mo_c_beta], axis=0)
        meta["restricted"] = False

    return meta


def _first_key(blocks: Dict[str, np.ndarray], candidates: Tuple[str, ...]) -> Optional[str]:
    for key in candidates:
        if key in blocks:
            return key
    return None


def _gaussian_shell_label(shell_type: int) -> str:
    if shell_type == 0:
        return "S"
    if shell_type == 1:
        return "P"
    if shell_type == 2:
        return "D"
    if shell_type == 3:
        return "F"
    if shell_type == -1:
        return "SP"
    raise ValueError(f"Unsupported Gaussian shell type {shell_type}")


def build_mol_from_fchk_basis(
    fchk_meta: Dict[str, Any],
    *,
    cart: bool = True,
    charge: Optional[int] = None,
    spin: Optional[int] = None,
):
    """Build a PySCF molecule using the primitive basis stored in an fchk file."""
    from pyscf import gto

    blocks = fchk_meta["blocks"]
    required = (
        "Shell to atom map",
        "Shell types",
        "Number of primitives per shell",
        "Primitive exponents",
        "Contraction coefficients",
    )
    for key in required:
        if key not in blocks:
            raise KeyError(f"fchk missing basis block {key!r}")

    z = fchk_meta["atomic_numbers"]
    coords = fchk_meta["coords_bohr"]
    shell_atom = blocks["Shell to atom map"].astype(int) - 1
    shell_type = blocks["Shell types"].astype(int)
    nprim_shell = blocks["Number of primitives per shell"].astype(int)
    exps = blocks["Primitive exponents"].astype(float)
    coeffs = blocks["Contraction coefficients"].astype(float)
    psp = blocks.get("P(S=P) Contraction coefficients")
    if psp is not None:
        psp = np.asarray(psp, dtype=float)

    basis_by_atom: list[list] = [[] for _ in range(len(z))]
    atom_lines = []
    exp_ptr = 0
    for ia, (zi, coord) in enumerate(zip(z, coords)):
        sym = gto.charge(int(zi))
        atom_lines.append(f"{sym} {coord[0]:.12f} {coord[1]:.12f} {coord[2]:.12f}")

    for iat, stype, nprim in zip(shell_atom, shell_type, nprim_shell):
        es = exps[exp_ptr : exp_ptr + nprim]
        cs = coeffs[exp_ptr : exp_ptr + nprim]
        if stype == -1:
            if psp is None:
                raise KeyError("fchk SP shell requires 'P(S=P) Contraction coefficients'")
            ps = psp[exp_ptr : exp_ptr + nprim]
            entry = ("SP", [(float(e), float(c), float(p)) for e, c, p in zip(es, cs, ps)])
        else:
            label = _gaussian_shell_label(int(stype))
            entry = (label, [(float(e), float(c)) for e, c in zip(es, cs)])
        basis_by_atom[int(iat)].append(entry)
        exp_ptr += int(nprim)

    basis: Dict[str, list] = {}
    for ia, zi in enumerate(z):
        sym = gto.charge(int(zi))
        if sym in basis and basis[sym] != basis_by_atom[ia]:
            raise ValueError(
                f"Atoms with symbol {sym} have different fchk basis shells; "
                "cannot map to a single PySCF basis entry."
            )
        basis[sym] = basis_by_atom[ia]

    kwargs = dict(
        atom="\n".join(atom_lines),
        basis=basis,
        unit="Bohr",
        cart=bool(cart),
    )
    if charge is None:
        charge = int(fchk_meta.get("charge", 0))
    kwargs["charge"] = charge
    if spin is None and "multiplicity" in fchk_meta:
        mult = int(fchk_meta["multiplicity"])
        kwargs["spin"] = max(0, mult - 1)
    elif spin is not None:
        kwargs["spin"] = int(spin)
    return gto.M(**kwargs)


def align_mo_coeff_to_pyscf(mol, fchk_meta: Dict[str, Any], mo_coeff: np.ndarray) -> np.ndarray:
    """Reorder Gaussian fchk AO rows to match a PySCF ``mol`` with the same basis."""
    from pyscf.gto import mole as gto_mole

    blocks = fchk_meta.get("blocks", {})
    cart = bool(mol.cart)
    d_key = "Pure/Cartesian d shells"
    if d_key in blocks:
        cart = bool(int(blocks[d_key][0]))
    mol_fchk = build_mol_from_fchk_basis(fchk_meta, cart=cart, charge=mol.charge, spin=mol.spin)
    if mol_fchk.nao_nr() != mol.nao_nr():
        raise ValueError(
            f"AO count mismatch after fchk basis build: fchk={mol_fchk.nao_nr()} "
            f"pyscf={mol.nao_nr()}"
        )
    ovlp = gto_mole.intor_cross("int1e_ovlp", mol_fchk, mol)
    perm = np.argmax(np.abs(ovlp), axis=1)
    if not np.allclose(np.abs(ovlp[range(len(perm)), perm]), 1.0, atol=1e-5):
        raise ValueError("Could not determine a clean AO permutation from fchk to PySCF")
    out = np.zeros_like(mo_coeff)
    out[perm, :] = mo_coeff
    return out


def build_pyscf_mol_from_fchk(
    fchk_meta: Dict[str, Any],
    *,
    basis: str,
    charge: Optional[int] = None,
    spin: Optional[int] = None,
    cart: Optional[bool] = None,
):
    """Build a PySCF :class:`pyscf.gto.Mole` from fchk geometry + external basis."""
    from pyscf import gto

    z = fchk_meta["atomic_numbers"]
    coords = fchk_meta["coords_bohr"]
    if charge is None:
        charge = int(fchk_meta.get("charge", 0))
    if spin is None and "multiplicity" in fchk_meta:
        mult = int(fchk_meta["multiplicity"])
        spin = max(0, mult - 1)

    atom_lines = []
    for zi, c in zip(z, coords):
        sym = gto.charge(int(zi))
        atom_lines.append(f"{sym} {c[0]:.12f} {c[1]:.12f} {c[2]:.12f}")

    kwargs = dict(atom="\n".join(atom_lines), basis=basis, unit="Bohr", charge=charge)
    if spin is not None:
        kwargs["spin"] = int(spin)
    if cart is not None:
        kwargs["cart"] = bool(cart)
    return gto.M(**kwargs)


def mo_occ_from_fchk_meta(fchk_meta: Dict[str, Any]) -> np.ndarray:
    """Occupation numbers consistent with PySCF RHF/RKS or UKS layouts."""
    mo_e = fchk_meta["mo_energy"]
    if mo_e.ndim == 1:
        nalpha = int(fchk_meta.get("nalpha", 0))
        if nalpha <= 0:
            nalpha = int((np.asarray(mo_e) < 0).sum())
        mo_occ = np.zeros(mo_e.size, dtype=float)
        mo_occ[:nalpha] = 2.0
        return mo_occ

    nalpha = int(fchk_meta.get("nalpha", 0))
    nbeta = int(fchk_meta.get("nbeta", 0))
    mo_occ = np.zeros_like(mo_e, dtype=float)
    mo_occ[0, :nalpha] = 1.0
    mo_occ[1, :nbeta] = 1.0
    return mo_occ


def load_mos_from_fchk(
    path: Union[str, Path],
    *,
    basis: str,
    charge: Optional[int] = None,
    spin: Optional[int] = None,
    cart: Optional[bool] = None,
) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return ``(mol, mo_coeff, mo_energy, mo_occ, meta)`` from an fchk file."""
    meta = read_fchk(path)
    if cart is None:
        blocks = meta.get("blocks", {})
        d_key = "Pure/Cartesian d shells"
        if d_key in blocks:
            # Gaussian: 0 = pure/spherical, 1 = cartesian.
            cart = bool(int(blocks[d_key][0]))
    mol = build_pyscf_mol_from_fchk(
        meta, basis=basis, charge=charge, spin=spin, cart=cart
    )
    mo_coeff = np.asarray(meta["mo_coeff"], dtype=float)
    if mo_coeff.ndim == 2 and "Shell to atom map" in meta.get("blocks", {}):
        mo_coeff = align_mo_coeff_to_pyscf(mol, meta, mo_coeff)
    mo_energy = np.asarray(meta["mo_energy"], dtype=float)
    mo_occ = mo_occ_from_fchk_meta(meta)
    return mol, mo_coeff, mo_energy, mo_occ, meta


def extract_gto_kernel_from_fchk(
    path: Union[str, Path],
    *,
    basis: str,
    xc: str,
    charge: Optional[int] = None,
    spin: Optional[int] = None,
    n_occ: Optional[int] = None,
    n_unocc: Optional[int] = None,
    n_states: int = 20,
    use_df: bool = False,
    tda: bool = True,
    verbose: bool = False,
    spin_state: str = "singlet",
    k_cache_max: int = 4096,
) -> Tuple[GTOKernel, CasidaOptions]:
    """Build a :class:`GTOKernel` with ``mf=None`` (generic MO response path)."""
    mol, mo_c, mo_e, mo_o, meta = load_mos_from_fchk(
        path, basis=basis, charge=charge, spin=spin
    )
    if mo_c.ndim == 3:
        raise NotImplementedError(
            "UKS fchk import for generic response is not wired yet; "
            "use closed-shell RHF/RKS jobs."
        )

    kernel = GTOKernel(
        mol,
        mo_c,
        mo_e,
        mo_o,
        n_occ=n_occ,
        n_unocc=n_unocc,
        xc=xc,
        use_df=use_df,
        mf=None,
        verbose=verbose,
        k_cache_max=k_cache_max,
        spin_state=spin_state,
    )
    opts = CasidaOptions(
        n_occ=kernel.n_occ,
        n_unocc=kernel.n_unocc,
        n_states=n_states,
        basis="gto",
        tda=tda,
        xc=xc,
        spin_state=spin_state,
    )
    opts.metadata = {
        "source": "qchem_fchk",
        "fchk": str(Path(path).resolve()),
        "fchk_energy_ha": meta.get("energy_ha"),
        "basis": basis,
    }
    return kernel, opts


def write_fchk_from_mf(
    mf,
    path: Union[str, Path],
    *,
    energy: Optional[float] = None,
) -> Path:
    """Write a minimal Gaussian-style fchk from a PySCF mean-field (for tests).

    Not a full fchk — only the blocks :func:`read_fchk` needs.
    """
    from pyscf import gto

    path = Path(path)
    mol = mf.mol
    mo_coeff = np.asarray(mf.mo_coeff, dtype=float)
    mo_energy = np.asarray(mf.mo_energy, dtype=float)
    mo_occ = np.asarray(mf.mo_occ, dtype=float)

    if energy is None:
        energy = float(mf.e_tot)

    if mo_coeff.ndim == 2:
        nalpha = int(mol.nelec[0])
        lines = [
            f"Test fchk from PySCF                    SP",
            _fchk_line("Number of atoms", "I", mol.natm, mol.atom_charges().astype(int)),
            _fchk_line("Atomic numbers", "I", mol.natm, mol.atom_charges().astype(int)),
            _fchk_line(
                "Current cartesian coordinates",
                "R",
                mol.natm * 3,
                mol.atom_coords(unit="Bohr").ravel(),
            ),
            _fchk_line("Charge", "I", 1, [mol.charge]),
            _fchk_line("Multiplicity", "I", 1, [mol.spin + 1]),
            _fchk_line("Number of alpha electrons", "I", 1, [nalpha]),
            _fchk_line("Number of basis functions", "I", 1, [mo_coeff.shape[0]]),
            _fchk_line("Total Energy", "R", 1, [energy]),
            _fchk_line("Alpha Orbital Energies", "R", mo_energy.size, mo_energy),
            _fchk_line(
                "Alpha MO coefficients",
                "R",
                mo_coeff.size,
                mo_coeff.ravel(order="F"),
            ),
        ]
    else:
        raise NotImplementedError("write_fchk_from_mf: UKS not implemented")

    path.write_text("\n".join(lines) + "\n")
    return path


def _fchk_line(label: str, dtype: str, n: int, values) -> str:
    header = f"{label:<45}{dtype}{n:>10}"
    vals = np.asarray(values, dtype=float if dtype == "R" else int).ravel()
    chunks = []
    row: list[str] = []
    for i, v in enumerate(vals):
        if dtype == "I":
            row.append(f"{int(v):>12}")
        else:
            row.append(f"{v:16.8E}")
        if len(row) == 5:
            chunks.append("".join(row))
            row = []
    if row:
        chunks.append("".join(row))
    return "\n".join([header] + chunks)
