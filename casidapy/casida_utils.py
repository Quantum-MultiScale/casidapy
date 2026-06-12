import numpy as np

from dftpy.field import DirectField
from dftpy.functional import Functional
from dftpy.grid import DirectGrid
from dftpy.utils import grid_map_data

# Try to import pylibxc for triplet XC kernel
try:
    import pylibxc
    HAS_PYLIBXC = True
except ImportError:
    HAS_PYLIBXC = False
    pylibxc = None


def mpi_comm_rank(comm) -> int:
    """MPI rank for mpi4py Comm or dftpy-style communicators."""
    if comm is None:
        return 0
    if hasattr(comm, "Get_rank"):
        return int(comm.Get_rank())
    return int(comm.rank)


def mpi_comm_size(comm) -> int:
    if comm is None:
        return 1
    if hasattr(comm, "Get_size"):
        return int(comm.Get_size())
    return int(comm.size)


def mpi_comm_bcast(comm, data, root=0):
    if hasattr(comm, "bcast"):
        return comm.bcast(data, root=root)
    return comm.Bcast(data, root=root)


def _bcast_psi_stack(comm, psi_small, root=0):
    """Broadcast a (n_band, *nr) array without hitting MPI's ~2 GiB pickle limit.

    The lowercase ``comm.bcast`` pickles the whole object and passes its byte
    length as a 32-bit int; multi-GB orbital stacks overflow it (MPI_ERR_ARG).
    Broadcast shape/dtype first, then each band as a contiguous buffer.
    """
    rank = mpi_comm_rank(comm)

    if rank == root:
        arr = np.ascontiguousarray(psi_small)
        meta = (arr.shape, arr.dtype.str)
    else:
        meta = None

    shape, dtype_str = comm.bcast(meta, root=root)  # tiny, safe
    dtype = np.dtype(dtype_str)

    if rank == root:
        out = arr
    else:
        out = np.empty(shape, dtype=dtype)

    n_band = shape[0]
    for ib in range(n_band):
        comm.Bcast(out[ib], root=root)  # uppercase buffer Bcast, per band
    return out


def build_energy_differences(occ_e, unocc_e) -> np.ndarray:
    """Return flattened ΔE[ia] = ε_a - ε_i for the active space."""
    n_occ = len(occ_e)
    n_unocc = len(unocc_e)
    dE = np.zeros(n_occ * n_unocc, dtype=float)
    for i in range(n_occ):
        for a in range(n_unocc):
            ia = i * n_unocc + a
            dE[ia] = unocc_e[a] - occ_e[i]
    return dE


def build_initial_guess(dE: np.ndarray, k: int) -> np.ndarray:
    """Construct deterministic LOBPCG initial guess with a small random perturbation."""
    n_trans = len(dE)
    x0 = np.zeros((n_trans, k))
    sorted_idx = np.argsort(dE)
    for i in range(k):
        if i < n_trans:
            x0[sorted_idx[i], i] = 1.0
    # Deterministic perturbation: every MPI rank must build the *identical* guess
    # so the redundant per-rank eigensolver follows the same iteration path. An
    # unseeded np.random.randn gave each rank a different X0 (now harmless because
    # the matvec is collective-free, but determinism is cheap insurance).
    rng = np.random.default_rng(0)
    x0 += 0.01 * rng.standard_normal((n_trans, k))
    return x0


def field_integral(field_a, field_b, dV, xp=np):
    """Return integral(field_a * field_b) in real space."""
    return float(xp.sum(field_a * field_b).real * dV)


def _as_scalar_field_on_grid(grid, obj):
    """
    Return a DirectField(rank=1) living on 'grid' from various possible inputs.
    Accepts:
      - DirectField
      - ndarray shaped like grid.nr
      - list/tuple like [up, down] where each is ndarray/DirectField
    """
    if isinstance(obj, DirectField):
        return obj

    # Spin-like container: [up, down]
    if isinstance(obj, (list, tuple)) and len(obj) == 2:
        up, dn = obj
        up = up.data if isinstance(up, DirectField) else np.asarray(up)
        dn = dn.data if isinstance(dn, DirectField) else np.asarray(dn)
        data = 0.5 * (up + dn)
        return DirectField(grid, rank=1, griddata_3d=data)

    data = obj.data if isinstance(obj, DirectField) else np.asarray(obj)
    return DirectField(grid, rank=1, griddata_3d=data)


def _to_gpu(arr):
    """NumPy-only: return a contiguous ndarray (GPU support removed)."""
    return np.asarray(arr)


def _to_cpu(arr):
    """NumPy-only: return a ndarray on CPU."""
    return np.asarray(arr)


def normalize_wavefunctions(psi_list, grid):
    """Normalize wavefunctions with phase alignment and real-part projection."""
    normalized = []
    for psi in psi_list:
        psi_data = np.asarray(psi)
        idx_max = np.argmax(np.abs(psi_data))
        phase = np.angle(psi_data.flat[idx_max])
        psi_rot = psi_data * np.exp(-1j * phase)
        psi_real = psi_rot.real
        psi_field = DirectField(grid=grid, data=psi_real)
        norm = (psi_field.conj() * psi_field).integral().real
        psi_field = psi_field / np.sqrt(norm)
        normalized.append(psi_field)
    return normalized

def filtered(f, f_max, rho, cut):
    """
    Zero the kernel where density is below ``cut``, then clip to ``[-f_max, f_max]``.
    """
    f_arr = np.asarray(f).copy()
    rho_arr = np.asarray(rho)
    f_arr = np.where(rho_arr > cut, f_arr, 0.0)
    f_arr = np.clip(f_arr, -f_max, f_max)
    return f_arr

def psi_to_fields(psi, grid, ions, comm):
    """Convert raw psi arrays/fields into DirectField list on target grid."""
    rank = mpi_comm_rank(comm)
    size = mpi_comm_size(comm)

    # Root decides the code path; broadcast it so ranks that received psi=None
    # (orbitals assembled only on rank 0) still enter the broadcast branch
    # instead of falling into list(None).
    is_array = isinstance(psi, np.ndarray)
    if size > 1:
        is_array = comm.bcast(is_array if rank == 0 else None, root=0)

    if is_array:
        psi_small = None
        if rank == 0:
            shape = psi[0].shape
            if list(shape) != list(grid.nr):
                grid_psi = DirectGrid(lattice=ions.cell, nr=list(shape))
                psi_interp = np.empty((len(psi), *grid.nr), dtype=np.float64)
                for ib in range(len(psi)):
                    pf = DirectField(grid=grid_psi, data=psi[ib])
                    psi_interp[ib] = np.asarray(grid_map_data(pf, grid=grid))
                psi_small = psi_interp
            else:
                psi_small = psi

        # Single-rank communicator: rank 0 already holds the data; the
        # pickle-based bcast would overflow MPI's 2 GiB count for large grids.
        if size > 1:
            psi_small = _bcast_psi_stack(comm, psi_small, root=0)

        psi_list = [DirectField(grid=grid, data=psi_small[i]) for i in range(len(psi_small))]
    else:
        psi_list = list(psi)

    psi_list = [p / np.sqrt(ions.cell.volume) for p in psi_list]
    return psi_list


def _compute_fxc_triplet_pylibxc(rho_ks, libxc_func_names, grid):
    """Compute spin-antisymmetric triplet f_xc from spin-polarized LibXC."""
    if not HAS_PYLIBXC:
        raise ImportError(
            "pylibxc is required for triplet XC kernel computation. "
            "Install with: pip install pylibxc"
        )

    rho_arr = np.asarray(rho_ks)
    grid_shape = grid.nr
    n_points = int(np.prod(grid_shape))
    rho_half = np.ascontiguousarray(rho_arr.flatten() / 2.0, dtype=np.float64)

    grad_rho = rho_ks.gradient()
    grad_rho_arr = np.asarray(grad_rho)
    grad_rho_sq = grad_rho_arr[0] ** 2 + grad_rho_arr[1] ** 2 + grad_rho_arr[2] ** 2
    sigma_quarter = np.ascontiguousarray((grad_rho_sq / 4.0).flatten(), dtype=np.float64)

    rho_pol = np.ascontiguousarray(np.column_stack([rho_half, rho_half]), dtype=np.float64)
    sigma_pol = np.ascontiguousarray(
        np.column_stack([sigma_quarter, sigma_quarter, sigma_quarter]), dtype=np.float64
    )
    inp_pol = {"rho": rho_pol, "sigma": sigma_pol}

    f_xc_T_flat = np.zeros(n_points, dtype=np.float64)
    for func_name in libxc_func_names:
        func = pylibxc.LibXCFunctional(func_name, "polarized")
        ret = func.compute(inp_pol, do_fxc=True)
        v2rho2 = np.asarray(ret["v2rho2"])
        f_uu = v2rho2[:, 0]
        f_ud = v2rho2[:, 1]
        f_xc_T_flat += (f_uu - f_ud) / 2.0

    f_xc_T = DirectField(grid, rank=1, griddata_3d=f_xc_T_flat.reshape(grid_shape))
    return f_xc_T


# Mapping from common DFTpy XC names to LibXC component functionals.
XC_TO_LIBXC_COMPONENTS = {
    "PBE": ["gga_x_pbe", "gga_c_pbe"],
    "LDA": ["lda_x", "lda_c_pw"],
    "BLYP": ["gga_x_b88", "gga_c_lyp"],
    "PW91": ["gga_x_pw91", "gga_c_pw91"],
}


# =========================================================================
# USPP augmentation kernels for Casida matrix construction
# =========================================================================

def proj_overlap(beta, grid, psi):
    """
    Compute projector overlaps <beta_n|psi_k> for all projectors and states.

    Returns array of shape (n_projectors, n_states).
    """
    n_p = len(beta)
    n_s = len(psi)
    dV = float(grid.dV)
    out = np.zeros((n_p, n_s), dtype=complex)

    for n in range(n_p):
        b = np.asarray(beta[n])
        for k in range(n_s):
            p = np.asarray(psi[k])
            out[n, k] = np.sum(b.conj() * p) * dV
    return out


def rho_aug(grid, qij, p_i, p_a, threshold=1e-30):
    """
    Return USPP augmentation density:
      rho_aug(r) = sum_nm <psi_i|beta_n> Q_nm(r) <beta_m|psi_a>

    Parameters
    ----------
    grid : DirectGrid
    qij : list of lists of DirectField
        Q_nm augmentation charges
    p_i : array (n_proj,)
        Projector overlaps <beta_n|psi_i>
    p_a : array (n_proj,)
        Projector overlaps <beta_m|psi_a>
    """
    n_p = len(p_i)
    rho = np.zeros(grid.nr, dtype=float)
    for n in range(n_p):
        for m in range(n_p):
            c = (p_i[n].conj() * p_a[m]).real
            if abs(c) > threshold:
                rho += c * np.asarray(qij[n][m])
    return DirectField(grid, rank=1, griddata_3d=rho)


def add_aug_contribution(accum, qij, proj_left, proj_right, scalar=1.0, threshold=1e-30, xp=np):
    """
    Add augmentation contribution into an existing array-like accumulator:
      accum += scalar * sum_nm conj(proj_left[n]) * proj_right[m] * Q_nm
    """
    n_p = len(proj_left)
    for n in range(n_p):
        for m in range(n_p):
            coeff = (xp.conj(proj_left[n]) * proj_right[m]).real
            coeff_val = float(coeff)
            if abs(coeff_val) > threshold:
                accum += scalar * coeff_val * qij[n][m]
    return accum


def augmentation_integral(response_field, qij, proj_left, proj_right, dV, threshold=1e-30, xp=np):
    """
    Compute sum_nm coeff_nm * integral(Q_nm * response_field) for USPP corrections.

    Used in Casida matrix element computation where the response (Hartree/XC potential)
    must include augmentation charge contributions.
    """
    n_p = len(proj_left)
    total = 0.0
    for n in range(n_p):
        for m in range(n_p):
            coeff = (xp.conj(proj_left[n]) * proj_right[m]).real
            coeff_val = float(coeff)
            if abs(coeff_val) > threshold:
                q_int = xp.sum(qij[n][m] * response_field).real * dV
                total += coeff_val * float(q_int)
    return total

