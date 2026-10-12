"""Fold an eDFTpy embedding potential v_emb(r) into a PySCF calculation.

This ports the spline machinery from JezsMartinez/pyscf@PRG_2026
(get_value_at_points_new / Spline_FFT_to_grid / get_vemb_mu_nu / vemb_mat)
but injects the resulting <mu|v_emb|nu> matrix at **get_hcore**, not get_fock.

Why get_hcore: get_fock only dresses the SCF mean field, leaving the one-body
Hamiltonian that CASSCF/CCSD/EOM-CCSD (and qiskit-nature's PySCF path) build
from get_hcore *bare* -- so the embedding would be missing from the correlated
problem. Injecting at get_hcore makes v_emb propagate automatically to SCF,
CASSCF, CCSD and qiskit-nature with one call, on the same mf object.

Usage (works for plain PySCF and qiskit-pyscf):
    from pyscf import gto, scf
    mol = gto.M(atom=..., basis="cc-pVDZ")          # fragment at its eDFTpy coords
    mf  = scf.RHF(mol)
    enable_embedding(mf, extemb="sub_licn.snpy")     # v_emb now in get_hcore
    mf.kernel()                                       # embedded SCF
    mc = mcscf.CASSCF(mf, ncas, nelecas).run()        # embedded CASSCF (h1 has v_emb)
    ham, npart = tensors.from_pyscf_casci(mc)         # -> QIS_QE
"""
import numpy as np


def _field_via_pp(field, ions):
    """Normalize a dftpy field through a QE .pp round-trip (write_potential +
    re-read). ``io.read(.snpy,'field')`` returns v_emb in a grid convention that
    mis-maps onto the PySCF quadrature grid; the .pp round-trip restores the
    correct field (cf. JezsMartinez/pyscf@PRG_2026 Spline_FFT_to_grid). Uses a
    temp file so nothing persists on disk."""
    import os, tempfile
    from dftpy.formats import io
    with tempfile.TemporaryDirectory() as td:
        pp = os.path.join(td, "vemb.pp")
        io.write_potential(pp, data=field, ions=ions)
        _, field2, _ = io.read(pp, kind="field")
    return field2


def read_field(extemb):
    """Read an eDFTpy/QE embedding potential (.snpy/.xsf/.pp) as a dftpy field.

    For .snpy/.xsf the raw read is grid-mismapped (see :func:`_field_via_pp`), so
    it is round-tripped through QE .pp to match JezsMartinez/pyscf@PRG_2026."""
    from dftpy.formats import io
    ext = extemb.lower().rsplit(".", 1)[-1]
    if ext not in ("snpy", "xsf", "pp"):
        raise ValueError("Only .snpy/.xsf/.pp V_emb formats are supported.")
    ions, field, _ = io.read(extemb, kind="field")
    if ext in ("snpy", "xsf"):
        field = _field_via_pp(field, ions)
    return field


def spline_values_at(field, coords):
    """B-spline a dftpy DirectField (uniform FFT grid) onto arbitrary Cartesian
    `coords` (Bohr), periodic wrap. Ported from JezsMartinez/pyscf
    get_value_at_points_new (orthorhombic lattice)."""
    from scipy import ndimage
    if field.spl_coeffs is None:
        field._calc_spline()
    nr = field.grid.nr
    ll = np.sqrt(np.diag(np.dot(field.grid.lattice, field.grid.lattice.T)))
    p = np.asarray(coords, dtype=np.float64).copy()
    for i in range(3):
        p[:, i] = np.mod(p[:, i] / ll[i] * nr[i], nr[i]) + field.spl_order
    return ndimage.map_coordinates(
        field.spl_coeffs, [p[:, 0], p[:, 1], p[:, 2]], mode="wrap", prefilter=False)


def vemb_ao_matrix(mol, vemb_vals, coords, weights):
    """<mu|v_emb|nu> = sum_g w_g phi_mu(r_g) v_emb(r_g) phi_nu(r_g) on a molecular
    (Becke) grid. Ported from get_vemb_mu_nu (0.5*w + mat+mat.T symmetrization)."""
    from pyscf.dft.numint import eval_ao, _scale_ao, _dot_ao_ao
    from pyscf.dft.gen_grid import BLKSIZE
    ao = eval_ao(mol, coords, deriv=0)
    ngrids, nao = ao.shape
    non0tab = np.ones(((ngrids + BLKSIZE - 1) // BLKSIZE, mol.nbas), dtype=np.uint8)
    aow = _scale_ao(ao, 0.5 * weights * vemb_vals)
    mat = _dot_ao_ao(mol, ao, aow, non0tab, (0, mol.nbas), mol.ao_loc_nr())
    return mat + mat.conj().T


def farfield_reference(field, atom_coords_bohr, r_cut=4.0):
    """Ambient v_emb level far from the fragment: mean of the field over grid
    points farther than `r_cut` (Bohr) from every fragment atom (min-image).
    Subtracting it references the embedded energy to the fragment sitting in the
    surrounding solvent baseline, removing the periodic-Hartree G=0 constant.
    Returned in the field's native units (Ry)."""
    grid = field.grid
    coords = np.asarray(grid.r).reshape(3, -1).T          # (Ngrid, 3) Bohr
    vals = np.asarray(field).ravel()
    L = np.sqrt(np.diag(np.dot(grid.lattice, grid.lattice.T)))
    dmin = np.full(coords.shape[0], np.inf)
    for a in np.asarray(atom_coords_bohr):
        d = coords - a
        d -= L * np.round(d / L)                            # minimum image (orthorhombic)
        dmin = np.minimum(dmin, np.sqrt((d * d).sum(1)))
    mask = dmin > r_cut
    return float(vals[mask].mean()) if mask.any() else float(vals.mean())


def build_vemb(mol, extemb=None, spline_values=None, field=None, grid_level=3, unit="Ry",
               ref=None, r_cut=4.0):
    """Return (<mu|v_emb|nu> AO matrix in Hartree, grids). `unit='Ry'` applies
    the QE Rydberg->Hartree factor. `ref='farfield'` subtracts the ambient level
    far from the fragment (meaningful total energies); `ref=None` leaves the raw
    (gauge-constant) potential; `ref=<float>` subtracts a fixed value (Ry).

    `field` is an in-memory dftpy field (e.g. the eDFTpy ``embedpot_field``): it is
    splined at the PySCF grid directly, no file round-trip. Takes precedence over
    `extemb`/`spline_values`."""
    from pyscf import dft
    grids = dft.gen_grid.Grids(mol)
    grids.level = grid_level
    grids.build()
    ref_val = 0.0
    if field is not None:
        vals = spline_values_at(field, grids.coords)
        if ref == "farfield":
            ref_val = farfield_reference(field, mol.atom_coords(), r_cut=r_cut)
        elif ref is not None:
            ref_val = float(ref)
    elif spline_values is None:
        field = read_field(extemb)
        vals = spline_values_at(field, grids.coords)
        if ref == "farfield":
            ref_val = farfield_reference(field, mol.atom_coords(), r_cut=r_cut)
        elif ref is not None:
            ref_val = float(ref)
    else:
        vals = np.asarray(spline_values, dtype=np.float64)
        if ref is not None and ref != "farfield":
            ref_val = float(ref)
    vals = vals - ref_val
    if str(unit).lower() in ("ry", "rydberg"):
        vals = 0.5 * vals                      # Ry -> Hartree
    mat = vemb_ao_matrix(mol, vals, grids.coords, grids.weights)
    return mat, grids


def enable_embedding(mf, extemb=None, spline_values=None, field=None, grid_level=3, unit="Ry",
                     ref="farfield", r_cut=4.0):
    """Wrap `mf.get_hcore` to add v_emb. Propagates to SCF, CASSCF, CCSD and
    qiskit-nature (all read this mf's get_hcore). Returns the v_emb AO matrix.

    `ref='farfield'` (default) references v_emb to its ambient level far from the
    fragment, so embedded total energies are physically comparable to gas phase;
    excitation energies and transition dipoles are unchanged by this choice.

    Note: the fragment geometry in `mol` must sit at the SAME Cartesian position
    (Bohr) it occupies in the eDFTpy cell, so the grid samples v_emb correctly.
    """
    Vemb, grids = build_vemb(mf.mol, extemb=extemb, spline_values=spline_values, field=field,
                             grid_level=grid_level, unit=unit, ref=ref, r_cut=r_cut)
    base_get_hcore = mf.get_hcore

    def get_hcore(mol=None):
        return np.asarray(base_get_hcore(mol)) + Vemb

    mf.get_hcore = get_hcore
    mf.vemb_m = Vemb            # cached <mu|v_emb|nu>
    mf._vemb_grids = grids
    return Vemb
