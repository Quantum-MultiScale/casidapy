"""
Subsystem LR-TDDFT coupling for embedded Casida calculations.

Pavanello inter-fragment blocks run on a serial global grid (eDFTpy rank 0 patches
``gsystem.grid`` / ``gsystem.density`` before calling ``run_subsystem_casida``).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import eigh as scipy_eigh

from dftpy.field import DirectField
from dftpy.functional import Functional
from dftpy.grid import DirectGrid
from dftpy.mpi import MP, SerialComm
from dftpy.utils import grid_map_data

# Default range-separation screening parameter (a.u.^-1) for coupling_hartree=
# "rangesep". Fixed (not box-derived) so the erf/erfc split is identical across a
# distance scan. ~1/alpha = 10 a0 (~5 A): the grid SR captures fragment overlap,
# the far-field dipole carries the rest. The exact result is alpha-independent.
_DEFAULT_COUPLING_ALPHA = 0.1


def compute_nadd_kernel(
    rho_total: DirectField,
    rho_subsystems: List[DirectField],  # unused; kept for API compatibility
    xc_functional=None,
    ke_functional=None,
    rho_cutoff: float = 1e-3,
    fxc_max: float = 20.0,
) -> np.ndarray:
    """Non-additive kernel f_xc[rho] + f_T[rho] (Pavanello coupling)."""
    del rho_subsystems  # Pavanello kernel is evaluated at rho_total (or rho_I subcell)
    rho_arr = np.asarray(rho_total)
    f_k = np.zeros(rho_arr.shape, dtype=float)
    if xc_functional is not None:
        f_k = f_k + _extract_v2rho2(xc_functional, rho_total)
    if ke_functional is not None:
        f_k = f_k + _extract_v2rho2(ke_functional, rho_total)
    f_k = np.where(rho_arr > rho_cutoff, f_k, 0.0)
    if fxc_max is None:
        return f_k
    return np.clip(f_k, -fxc_max, fxc_max)

def _functional_on_coupling_grid(functional):
    """Drop NLCC/core on the serial global grid (no subsystem pseudo there)."""
    if functional is None:
        return None
    import copy

    f = copy.copy(functional)
    f.core_density = None
    if hasattr(f, "_core_density"):
        f._core_density = None
    if hasattr(f, "pseudo"):
        f.pseudo = None
    return f


def _extract_v2rho2(functional, rho: DirectField) -> np.ndarray:
    functional = _functional_on_coupling_grid(functional)
    obj = functional(rho, calcType=["V2"])
    if not hasattr(obj, "v2rho2"):
        return np.zeros(np.asarray(rho).shape, dtype=float)
    v2 = obj.v2rho2
    if isinstance(v2, DirectField):
        return np.asarray(v2)
    if isinstance(v2, (list, tuple)) and len(v2) == 2:
        return 0.5 * (np.asarray(v2[0]) + np.asarray(v2[1]))
    return np.asarray(v2)


def _embedding_subgrid(gsystem, isub: int) -> DirectGrid:
    """Embedding subcell on the serial global grid (``graph.sub_shape``)."""
    graph = gsystem.graphtopo.graph
    grid_sub = DirectGrid(
        lattice=gsystem.grid.lattice,
        nr=tuple(int(x) for x in graph.sub_shape[isub]),
        full=gsystem.grid.full,
        mp=MP(comm=SerialComm()),
    )
    grid_sub.shift = np.array(graph.sub_shift[isub], dtype=np.int32)
    return grid_sub


def _charge_subgrid_nr(gsystem, isub: int) -> Optional[Tuple[int, int, int]]:
    """QE charge-grid dimensions for subsystem ``isub``, if known."""
    drivers = gsystem.graphtopo.drivers
    if isub < len(drivers) and drivers[isub] is not None:
        gd = getattr(drivers[isub], "grid_driver", None)
        if gd is not None:
            return tuple(int(x) for x in gd.nrR)
    return None


def _serial_grid_for_nr(gsystem, isub: int, nr: Tuple[int, ...]) -> DirectGrid:
    graph = gsystem.graphtopo.graph
    grid_sub = DirectGrid(
        lattice=gsystem.grid.lattice,
        nr=tuple(int(x) for x in nr),
        full=gsystem.grid.full,
        mp=MP(comm=SerialComm()),
    )
    grid_sub.shift = np.array(graph.sub_shift[isub], dtype=np.int32)
    return grid_sub


def _grid_for_data(gsystem, isub: int, data) -> DirectGrid:
    """Pick a serial subcell grid matching ``data`` (embedding or QE charge grid)."""
    arr = np.asarray(data)
    emb = _embedding_subgrid(gsystem, isub)
    emb_nr = tuple(int(x) for x in emb.nrR)
    if arr.ndim >= 3 and tuple(arr.shape[-3:]) == emb_nr:
        return emb
    if arr.size == int(np.prod(emb_nr)):
        return emb

    chg_nr = _charge_subgrid_nr(gsystem, isub)
    if chg_nr is None and arr.ndim >= 3:
        chg_nr = tuple(int(x) for x in arr.shape[-3:])
    if chg_nr is not None:
        if arr.ndim >= 3 and tuple(arr.shape[-3:]) == chg_nr:
            return _serial_grid_for_nr(gsystem, isub, chg_nr)
        if arr.size == int(np.prod(chg_nr)):
            return _serial_grid_for_nr(gsystem, isub, chg_nr)

    raise ValueError(
        f"subsystem {isub}: cannot place array of size {arr.size} "
        f"on embedding grid {emb_nr}"
        + (f" or charge grid {chg_nr}" if chg_nr else ""),
    )


def _as_subcell_field(data, gsystem, isub: int) -> DirectField:
    arr = np.asarray(data)
    sub_grid = _grid_for_data(gsystem, isub, arr)
    field = DirectField(grid=sub_grid, rank=1)
    field[:] = arr.reshape(field.shape)
    return field


def _to_embedding_subcell(field: DirectField, gsystem, isub: int) -> DirectField:
    """Map a subcell field (e.g. QE charge grid) onto the embedding subcell grid."""
    emb = _embedding_subgrid(gsystem, isub)
    src_nr = tuple(int(x) for x in field.grid.nrR)
    emb_nr = tuple(int(x) for x in emb.nrR)
    if src_nr == emb_nr:
        return field
    mapped = grid_map_data(field, grid=emb)
    out = DirectField(grid=emb, rank=1)
    out[:] = np.asarray(mapped).reshape(out.shape)
    return out


def _embedding_subcell_field(data, gsystem, isub: int) -> DirectField:
    """Build a subcell field on the embedding grid (maps from charge grid if needed)."""
    return _to_embedding_subcell(_as_subcell_field(data, gsystem, isub), gsystem, isub)


def _global_sub_index(gsystem, isub: int):
    """Subsystem slice on the full serial supercell (not MPI region layout)."""
    return gsystem.graphtopo.graph.get_sub_index(isub, in_global=True)


def _sub_to_global(field_sub, gsystem, isub: int) -> DirectField:
    """Map a subcell field onto the (serial) global grid."""
    field_global = DirectField(grid=gsystem.grid, rank=1)
    field_global[:] = 0.0
    index = _global_sub_index(gsystem, isub)
    np.asarray(field_global)[index] = np.asarray(field_sub).reshape(
        np.asarray(field_global)[index].shape,
    )
    return field_global


def _global_to_sub(field_global, gsystem, isub: int) -> DirectField:
    """Restrict a global field to subsystem ``isub``."""
    sub_grid = _embedding_subgrid(gsystem, isub)
    field_sub = DirectField(grid=sub_grid, rank=1)
    index = _global_sub_index(gsystem, isub)
    field_sub[:] = np.asarray(field_global)[index].reshape(field_sub.shape)
    return field_sub

def _subcell_kernel_to_global(
    kernel_sub: np.ndarray,
    gsystem,
    isub: int,
) -> np.ndarray:
    """Embed a subcell-shaped kernel array onto the serial global grid."""
    sub_grid = _embedding_subgrid(gsystem, isub)
    field_sub = DirectField(grid=sub_grid, rank=1)
    field_sub[:] = np.asarray(kernel_sub, dtype=float).reshape(field_sub.shape)
    return np.asarray(_sub_to_global(field_sub, gsystem, isub))


def _diagonal_coupling_kernel(
    f_nadd: np.ndarray,
    rho_I_global: DirectField,
    gsystem,
    isub: int,
    ke_functional=None,
    *,
    rho_cutoff: float = 1e-3,
    fxc_max: float = 20.0,
) -> np.ndarray:
    """Diagonal block kernel: f_nadd(rho_tot) - f_T(rho_I) on subsystem subcell."""
    if ke_functional is None:
        return np.asarray(f_nadd, dtype=float)

    rho_I_sub = _global_to_sub(rho_I_global, gsystem, isub)
    f_T_I_sub = compute_nadd_kernel(
        rho_I_sub,
        [],
        None,
        ke_functional,
        rho_cutoff=rho_cutoff,
        fxc_max=fxc_max,
    )
    f_T_I_global = _subcell_kernel_to_global(f_T_I_sub, gsystem, isub)
    return np.asarray(f_nadd, dtype=float) - f_T_I_global


def _remap_k_coupling_to_positions(
    K_coupling: Dict[Tuple[int, int], np.ndarray],
    active_indices: List[int],
) -> Dict[Tuple[int, int], np.ndarray]:
    """Map K keys from original fragment indices to 0..n_active-1 positions."""
    pos = {frag: k for k, frag in enumerate(active_indices)}
    out: Dict[Tuple[int, int], np.ndarray] = {}
    for (I, J), K_IJ in K_coupling.items():
        if I not in pos or J not in pos:
            continue
        out[(pos[I], pos[J])] = K_IJ
    return out


def _empty_coupled_result() -> Dict[str, Any]:
    return {
        "omega": np.array([]),
        "f": np.array([]),
        "Z": np.zeros((0, 0), dtype=float),
        "K_coupling": {},
        "f_nadd": None,
    }

def compute_coupling_block(
    phi_I: List,
    phi_J: List,
    gsystem,
    isub_I: int,
    isub_J: int,
    f_nadd: np.ndarray,
    hartree_func: Optional[Functional] = None,
    include_hartree: bool = True,
) -> np.ndarray:
    """Inter-fragment Pavanello coupling matrix K_IJ (Eq. 22).

    ``include_hartree`` controls the Coulomb (Hartree) part of the coupling
    kernel. For inter-fragment blocks (I != J) it must be True — that Coulomb
    term is the physical inter-fragment coupling. For the diagonal self-block
    (I == J) it must be False: the intra-fragment Coulomb response is already
    contained in the fragment excitation energy omega_I (the fragment Casida
    solved it), so adding V_H[phi_Ij] here double-counts it and spuriously
    blue-shifts the fragment manifold while hoarding oscillator strength into
    one state. The diagonal block then carries only the non-additive
    XC/kinetic kernel correction passed in ``f_nadd``.
    """
    if phi_I is None or phi_J is None:
        raise ValueError(
            f"Missing rho_transition for fragments {isub_I}/{isub_J}",
        )

    if hartree_func is None:
        hartree_func = Functional(type="HARTREE")

    n_I = len(phi_I)
    n_J = len(phi_J)
    if n_I == 0 or n_J == 0:
        return np.zeros((n_I, n_J), dtype=float)

    K_IJ = np.zeros((n_I, n_J), dtype=float)
    global_grid = gsystem.grid

    phi_ia_fields = [_embedding_subcell_field(phi_I[ia], gsystem, isub_I) for ia in range(n_I)]

    # Pre-flatten phi_ia arrays and record dV once; replaces the inner ia loop
    # with a single DGEMV per jb.
    phi_I_flat = np.stack([np.asarray(f).ravel() for f in phi_ia_fields])  # (n_I, n_sub)
    sub_dV = float(phi_ia_fields[0].grid.dV)

    for jb in range(n_J):
        phi_jb = _embedding_subcell_field(phi_J[jb], gsystem, isub_J)
        phi_jb_global = _sub_to_global(phi_jb, gsystem, isub_J)

        if include_hartree:
            vh_jb = hartree_func(phi_jb_global, calcType=["V"]).potential
            response_global = np.asarray(vh_jb) + f_nadd * np.asarray(phi_jb_global)
        else:
            # Diagonal self-block: Coulomb already in omega_I; keep only f_nadd.
            response_global = f_nadd * np.asarray(phi_jb_global)
        response_field = DirectField(grid=global_grid, rank=1)
        response_field[:] = response_global.reshape(response_field.shape)

        # Sanity: with Hartree on, an all-zero response means the DirectField init
        # silently failed and every K element would be zero. (Skip when Hartree is
        # off — there f_nadd*phi can legitimately be tiny/locally zero.)
        if include_hartree and jb == 0 and np.abs(np.asarray(response_field)).sum() == 0.0:
            raise RuntimeError(
                f"compute_coupling_block: response_field is all zeros for "
                f"fragments ({isub_I},{isub_J}), jb=0. "
                "DirectField initialisation from response_global failed."
            )

        response_on_I = _global_to_sub(response_field, gsystem, isub_I)
        response_arr = np.asarray(response_on_I).ravel()  # (n_sub,)

        # K_IJ[:, jb] = dV * phi_I_flat @ response_arr  (DGEMV replaces ia loop)
        K_IJ[:, jb] = phi_I_flat @ response_arr * sub_dV

    return K_IJ


def _res_array(res: Dict[str, Any], *keys: str) -> np.ndarray:
    for key in keys:
        if key in res and res[key] is not None:
            return np.asarray(res[key])
    raise KeyError(f"fragment result missing any of {keys!r}")


def _res_optional(res: Dict[str, Any], *keys: str, default=None):
    for key in keys:
        if key in res and res[key] is not None:
            return res[key]
    return default


def _fragment_amp_matrix(res: Dict[str, Any]) -> np.ndarray:
    """Casida amplitudes in transition basis: xpy (RPA) or Z (TDA)."""
    Z = _res_array(res, "Z", "eigenvectors")
    xpy = _res_optional(res, "xpy")
    if xpy is not None:
        return np.asarray(xpy, dtype=float)
    return np.asarray(Z, dtype=float)

RHO_BASIS_AMPLITUDE_XPY = "amplitude_xpy"

def _project_k_to_state_basis(
    K_blk: np.ndarray,
    res_I: Dict[str, Any],
    res_J: Dict[str, Any],
    n_I: int,
    n_J: int,
) -> np.ndarray:
    """K is expected in state space when rho_transition is amplitude-basis."""
    K_blk = np.asarray(K_blk, dtype=float)
    if K_blk.shape == (n_I, n_J):
        return K_blk

    rho_basis = res_I.get("rho_basis", RHO_BASIS_AMPLITUDE_XPY)
    if rho_basis == RHO_BASIS_AMPLITUDE_XPY:
        raise ValueError(
            f"K block shape {K_blk.shape} != state shape ({n_I}, {n_J}); "
            "expected amplitude-basis rho_transition from fragment Casida.",
        )

    amp_I = _fragment_amp_matrix(res_I)
    amp_J = _fragment_amp_matrix(res_J)
    if amp_I.shape[1] == n_I and amp_J.shape[1] == n_J:
        if K_blk.shape[0] == amp_I.shape[0] and K_blk.shape[1] == amp_J.shape[0]:
            return np.real(amp_I.T @ K_blk @ amp_J)

    Z_I = _res_array(res_I, "Z", "eigenvectors")
    Z_J = _res_array(res_J, "Z", "eigenvectors")
    return np.real(Z_I.T @ K_blk @ Z_J)

def assemble_coupled_casida(
    fragment_results: List[Dict[str, Any]],
    K_coupling: Dict[Tuple[int, int], np.ndarray],
    tda: bool = False,
    *,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    omega_blocks = [_res_array(res, "omega") for res in fragment_results]
    block_sizes = [len(w) for w in omega_blocks]
    offsets = np.cumsum([0] + block_sizes[:-1]).tolist()
    n_tot = sum(block_sizes)

    if n_tot == 0:
        return np.array([]), np.zeros((0, 0), dtype=float)

    if tda:
        A_coupled = np.zeros((n_tot, n_tot), dtype=float)
        for I, res in enumerate(fragment_results):
            i0, i1 = offsets[I], offsets[I] + block_sizes[I]
            A_coupled[i0:i1, i0:i1] = np.diag(omega_blocks[I])

        for (I, J), K_IJ in K_coupling.items():
            if I == J:
                continue  # diagonal stays diag(omega_I); no self-block (see RPA note)
            i0_I, i0_J = offsets[I], offsets[J]
            n_I, n_J = block_sizes[I], block_sizes[J]
            K_blk = _project_k_to_state_basis(
                K_IJ, fragment_results[I], fragment_results[J], n_I, n_J,
            )
            A_coupled[i0_I:i0_I + n_I, i0_J:i0_J + n_J] += K_blk
            if (J, I) not in K_coupling:
                A_coupled[i0_J:i0_J + n_J, i0_I:i0_I + n_I] += K_blk.T

        omega, Z = scipy_eigh(A_coupled)
        return np.asarray(omega, dtype=float), np.asarray(Z, dtype=float)

    C_coupled = np.zeros((n_tot, n_tot), dtype=float)
    for I, res in enumerate(fragment_results):
        i0, i1 = offsets[I], offsets[I] + block_sizes[I]
        C_coupled[i0:i1, i0:i1] = np.diag(omega_blocks[I] ** 2)

    # Only OFF-diagonal blocks couple fragments. The diagonal stays diag(omega_I**2):
    # the fragment excitations already are eigenstates of their embedded Casida, so a
    # self-block K[I,I] re-integrates the kernel over a fragment's own transitions and
    # double-counts response already in omega_I. Empirically the self-block drives the
    # RPA C matrix indefinite (negative eigenvalues -> spurious omega=0 states) and
    # shifts states below the fragment manifold even at large separation where the
    # inter-fragment coupling is ~0. Skipping it makes the coupled spectrum reduce to
    # the fragment spectrum as K_IJ(I!=J) -> 0, as it must.
    for (I, J), K_IJ in K_coupling.items():
        if I == J:
            continue
        i0_I, i0_J = offsets[I], offsets[J]
        n_I, n_J = block_sizes[I], block_sizes[J]
        sqrt_omega_I = np.sqrt(np.maximum(omega_blocks[I], 1e-30))
        sqrt_omega_J = np.sqrt(np.maximum(omega_blocks[J], 1e-30))
        K_blk = _project_k_to_state_basis(
            K_IJ, fragment_results[I], fragment_results[J], n_I, n_J,
        )
        coupling = 2.0 * (sqrt_omega_I[:, None] * K_blk * sqrt_omega_J[None, :])
        C_coupled[i0_I:i0_I + n_I, i0_J:i0_J + n_J] += coupling
        if (J, I) not in K_coupling:
            C_coupled[i0_J:i0_J + n_J, i0_I:i0_I + n_I] += coupling.T

    w2, Z = scipy_eigh(C_coupled)
    n_neg = int(np.sum(w2 < -1e-8))
    if n_neg and verbose:
        print(
            f"  WARNING: {n_neg} negative RPA C eigenvalues "
            f"(min w^2 = {float(np.min(w2)):.6e}); clamping to zero.",
            flush=True,
        )
    return np.sqrt(np.maximum(w2, 0.0)), np.asarray(Z, dtype=float)

def coupled_oscillator_strengths(
    omega_coupled: np.ndarray,
    Z_coupled: np.ndarray,
    fragment_results: List[Dict[str, Any]],
    *,
    tda: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    mu_blocks = []
    omega_blocks = []
    for frag_idx, res in enumerate(fragment_results):
        om = np.asarray(_res_array(res, "omega"), dtype=float)
        n_states = len(om)
        omega_blocks.append(om)
        mu = _res_optional(res, "dip_tran")
        if mu is None:
            if verbose:
                print(
                    f"  WARNING: fragment {frag_idx} has no dip_tran; "
                    "coupled oscillator strengths set to zero there.",
                    flush=True,
                )
            mu_blocks.append(np.zeros((n_states, 3)))
            continue

        mu = np.asarray(mu)
        if mu.shape[0] != n_states:
            amp = _fragment_amp_matrix(res)
            if amp.shape[0] == mu.shape[0] and amp.shape[1] == n_states:
                mu = np.real(amp.conj().T @ mu)
            else:
                Z = _res_array(res, "Z", "eigenvectors")
                mu = np.real(Z.conj().T @ mu)
        mu_blocks.append(mu)

    if not mu_blocks:
        return np.array([], dtype=float)

    mu_full = np.vstack(mu_blocks)               # (Ntot, 3) fragment-state dipoles d_I
    omega_full = np.concatenate(omega_blocks)    # (Ntot,)  fragment excitation energies
    Z_coupled = np.asarray(Z_coupled, dtype=float)
    f_coupled = np.zeros(len(omega_coupled), dtype=float)

    # Coupled transition dipole and oscillator strength. Both forms reduce to the
    # uncoupled fragment value f_I = (2/3) omega_I |d_I|^2 when Z = I (K -> 0).
    #
    #   TDA : f_n = (2/3) omega_n |Σ_I d_I Z_I^n|^2
    #   RPA : f_n = (2/3)         |Σ_I sqrt(omega_I) d_I Z_I^n|^2
    #
    # The RPA form carries the sqrt(omega_I) weight from the symmetric Casida-C
    # eigenvector normalisation. The previous code instead divided by sqrt(omega_n)
    # and kept an omega_n prefactor, which dropped the per-state sqrt(omega_I) factor
    # and inflated every RPA intensity by ~1/omega_I (3-12x for omega~0.1-0.3 Ha),
    # breaking intensity conservation and letting one mixed state hoard the dipole.
    if tda:
        mu_eff = mu_full
    else:
        mu_eff = mu_full * np.sqrt(np.maximum(omega_full, 0.0))[:, None]

    for n, omega_n in enumerate(omega_coupled):
        if omega_n < 1e-30:
            continue
        d_n = np.real(mu_eff.T @ Z_coupled[:, n])
        prefac = (2.0 / 3.0) * float(omega_n) if tda else (2.0 / 3.0)
        f_coupled[n] = prefac * float(np.dot(d_n, d_n))

    return f_coupled

def _rho_transition_to_stack(rho) -> tuple[np.ndarray, int]:
    if rho is None:
        return np.zeros((0,), dtype=float), 0
    if isinstance(rho, np.ndarray):
        if rho.size == 0:
            return np.zeros((0,), dtype=float), 0
        if rho.ndim >= 4:
            stack = np.asarray(rho, dtype=float)
            return stack, int(stack.shape[0])
        if rho.ndim == 3:
            stack = np.asarray(rho, dtype=float)[np.newaxis, ...]
            return stack, 1
        if rho.dtype == object:
            rho = [rho[i] for i in range(len(rho))]
        else:
            raise ValueError(f"unsupported rho_transition ndarray shape {rho.shape}")
    stack = np.stack([np.asarray(x) for x in rho], axis=0)
    return stack, int(stack.shape[0])

def _load_rho_transition_npz(path: str) -> List:
    with np.load(path, allow_pickle=False) as z:
        stack = np.asarray(z["rho_transition"])
    if stack.size == 0:
        return []
    return [stack[i] for i in range(stack.shape[0])]


def _fragment_transition_densities(
    fragment_results: List[Dict[str, Any]],
    fragment_stream_paths: Optional[List[Optional[str]]],
    frag_idx: int,
):
    if fragment_stream_paths is not None and frag_idx < len(fragment_stream_paths):
        path = fragment_stream_paths[frag_idx]
        if path:
            return _load_rho_transition_npz(path)

    res = fragment_results[frag_idx]
    rho = res.get("rho_transition")
    if rho is None:
        raise ValueError(
            f"Fragment {frag_idx}: rho_transition missing in memory and "
            "no stream path available.",
        )
    rho_stack, n_rho = _rho_transition_to_stack(rho)
    if n_rho == 0:
        return []
    return [rho_stack[i] for i in range(n_rho)]


def run_subsystem_casida(
    gsystem,
    drivers: List,
    fragment_results: List[Dict[str, Any]],
    xc_functional,
    ke_functional=None,
    sub_densities=None,
    tda: bool = False,
    rho_cutoff: float = 1e-3,
    fxc_max: float = 20.0,
    verbose: bool = True,
    fragment_stream_paths: Optional[List[Optional[str]]] = None,
    omega_coupling_max: Optional[float] = None,
    comm=None,
    graphtopo=None,
    fragment_owners: Optional[List[int]] = None,
    distributed: bool = False,
    f_nadd: Optional[np.ndarray] = None,
    hartree_mode: str = "full",
    coupling_alpha: Optional[float] = None,
) -> Dict[str, Any]:
    """Pavanello coupling on a serial ``gsystem`` (grid/density patched by eDFTpy).

    Two backends:

    * **Serial** (default): rank 0 holds every fragment's transition densities
      (in memory or file-streamed via ``fragment_stream_paths``) and builds all
      coupling blocks itself.
    * **Distributed** (``distributed=True``): a *collective* call — every rank
      participates. Each fragment's ``rho_transition`` stays resident on its
      owner rank (``fragment_owners[idx]``); blocks are computed where the data
      already lives and the partner's transition densities are streamed
      state-by-state over ``graphtopo.comm``. No disk, no rank-0 gather of grids.
      Requires ``graphtopo`` and ``fragment_owners``. Returns the coupled result
      dict on rank 0 and ``None`` on all other ranks.
    """
    sel = _select_active_fragments(fragment_results, omega_coupling_max, verbose)
    if not isinstance(sel, tuple):
        return sel
    active_indices, active_results = sel
    n_active = len(active_indices)

    if distributed:
        return _run_subsystem_casida_distributed(
            gsystem,
            drivers,
            active_indices,
            active_results,
            xc_functional,
            ke_functional=ke_functional,
            sub_densities=sub_densities,
            tda=tda,
            rho_cutoff=rho_cutoff,
            fxc_max=fxc_max,
            verbose=verbose,
            graphtopo=graphtopo,
            fragment_owners=fragment_owners,
            f_nadd=f_nadd,
            hartree_mode=hartree_mode,
            coupling_alpha=coupling_alpha,
        )

    if verbose:
        stream = fragment_stream_paths is not None
        print(
            f"Subsystem Casida coupling: {n_active} fragments"
            + (" (file-streamed transition densities)" if stream else ""),
            flush=True,
        )
        for idx, res in zip(active_indices, active_results):
            om = _res_array(res, "omega")
            print(
                f"  Fragment {idx}: {len(om)} states, "
                f"omega {float(np.min(om)):.6f}–{float(np.max(om)):.6f} Ha",
                flush=True,
            )

    rho_total = gsystem.density
    rho_subs_by_idx = {}
    for idx in active_indices:
        rho_I_global = DirectField(grid=gsystem.grid, rank=1)
        rho_I_global[:] = 0.0
        if sub_densities is not None and idx < len(sub_densities):
            subrho = sub_densities[idx]
        else:
            driver = drivers[idx] if idx < len(drivers) else None
            subrho = driver.density if driver is not None else None
        if subrho is not None:
            sub_field = _embedding_subcell_field(subrho, gsystem, idx)
            index = _global_sub_index(gsystem, idx)
            np.asarray(rho_I_global)[index] = np.asarray(sub_field).reshape(
                np.asarray(rho_I_global)[index].shape,
            )
        rho_subs_by_idx[idx] = rho_I_global

    if verbose:
        print(
            "  Computing Pavanello coupling kernel (f_xc + f_T at rho_tot)...",
            flush=True,
        )
    f_nadd = compute_nadd_kernel(
        rho_total,
        list(rho_subs_by_idx.values()),
        xc_functional,
        ke_functional,
        rho_cutoff=rho_cutoff,
        fxc_max=fxc_max,
    )

    if verbose:
        fnadd_arr = np.asarray(f_nadd)
        nonzero_frac = float(np.sum(fnadd_arr != 0.0)) / max(fnadd_arr.size, 1)
        print(
            f"  f_nadd stats: min={float(fnadd_arr.min()):.4e}  "
            f"max={float(fnadd_arr.max()):.4e}  "
            f"nonzero={nonzero_frac:.1%}",
            flush=True,
        )
        if nonzero_frac == 0.0:
            print(
                "  WARNING: f_nadd is entirely zero — all coupling will be pure Hartree. "
                "Check rho_cutoff and that rho_total is non-zero.",
                flush=True,
            )

        # Round-trip check: sub→global→sub must be lossless for the first
        # active fragment's first transition density.
        _rt_idx = active_indices[0]
        _rt_phi_list = _fragment_transition_densities(
            fragment_results, fragment_stream_paths, _rt_idx,
        )
        if _rt_phi_list:
            _rt_phi = _embedding_subcell_field(_rt_phi_list[0], gsystem, _rt_idx)
            _rt_global = _sub_to_global(_rt_phi, gsystem, _rt_idx)
            _rt_back = _global_to_sub(_rt_global, gsystem, _rt_idx)
            _rt_orig = np.asarray(_rt_phi).ravel()
            _rt_recovered = np.asarray(_rt_back).ravel()
            _rt_err = float(np.max(np.abs(_rt_orig - _rt_recovered)))
            print(
                f"  sub→global→sub round-trip max error (frag {_rt_idx}): {_rt_err:.3e}",
                flush=True,
            )
            if _rt_err > 1e-10 * max(float(np.max(np.abs(_rt_orig))), 1e-30):
                print(
                    "  WARNING: round-trip error is large — _global_to_sub index "
                    "may be addressing wrong spatial dimensions. "
                    "Check get_sub_index shape vs DirectField array rank.",
                    flush=True,
                )

    hartree = Functional(type="HARTREE")
    K_coupling: Dict[Tuple[int, int], np.ndarray] = {}

    # Only OFF-diagonal (inter-fragment) blocks are computed. The diagonal
    # self-block K[I,I] is intentionally NOT built: the coupled matrix keeps
    # diag(omega_I) on the diagonal (see assemble_coupled_casida). Re-integrating
    # the kernel over a fragment's own transitions double-counts response already
    # in omega_I and makes the RPA matrix indefinite. Skipping it also removes the
    # diagonal Hartree FFTs entirely.
    for a, I in enumerate(active_indices):
        phi_I = _fragment_transition_densities(
            fragment_results, fragment_stream_paths, I,
        )
        for b in range(a + 1, len(active_indices)):
            J = active_indices[b]
            if verbose:
                print(f"  Computing K[{I},{J}] coupling block...", flush=True)
            phi_J = _fragment_transition_densities(
                fragment_results, fragment_stream_paths, J,
            )
            K_IJ = compute_coupling_block(
                phi_I, phi_J, gsystem, I, J, f_nadd, hartree,
                include_hartree=True,
            )
            K_coupling[(I, J)] = K_IJ
            K_coupling[(J, I)] = K_IJ.T

    if verbose:
        for a, I in enumerate(active_indices):
            for b in range(a + 1, len(active_indices)):
                J = active_indices[b]
                K_IJ = K_coupling.get((I, J))
                K_JI = K_coupling.get((J, I))
                if K_IJ is not None and K_JI is not None:
                    sym_err = float(np.max(np.abs(K_IJ - K_JI.T)))
                    print(
                        f"  K[{I},{J}] symmetry error (||K_IJ - K_JI^T||_max): {sym_err:.3e}",
                        flush=True,
                    )
                    if sym_err > 1e-6 * max(float(np.max(np.abs(K_IJ))), 1e-30):
                        print(
                            f"  WARNING: K[{I},{J}] is not symmetric — "
                            "grid embedding or integral is inconsistent.",
                            flush=True,
                        )
                if K_IJ is not None:
                    om_I = _res_array(active_results[a], "omega")
                    om_J = _res_array(active_results[b], "omega")
                    avg_om = float(0.5 * (np.mean(om_I) + np.mean(om_J)))
                    k_max = float(np.max(np.abs(K_IJ)))
                    print(
                        f"  K[{I},{J}] max|element|={k_max:.4e} Ha  "
                        f"vs avg fragment omega={avg_om:.4e} Ha  "
                        f"ratio={k_max/max(avg_om,1e-30):.3f}",
                        flush=True,
                    )
                    if k_max > avg_om:
                        print(
                            f"  WARNING: K[{I},{J}] coupling exceeds mean fragment "
                            "excitation energy — coupling may be unphysically large.",
                            flush=True,
                        )

    if verbose:
        # ---- Full coupling K-matrix dump (every block, untruncated) ----
        # Prints each K[I,J] block in full and the largest |elements| annotated
        # with their fragment-state energies, so unphysically large couplings can
        # be traced to specific (often diffuse, high-lying) state pairs. Also
        # saves every block + per-fragment omega to coupling_K_matrix.npz.
        omega_by_pos = {
            active_indices[a]: np.asarray(_res_array(active_results[a], "omega"))
            for a in range(len(active_indices))
        }
        print("  === FULL COUPLING K MATRIX (Ha) ===", flush=True)
        for (I, J) in sorted(K_coupling.keys()):
            if J < I:
                continue  # upper triangle (incl. diagonal); lower is the transpose
            K_IJ = np.asarray(K_coupling[(I, J)])
            print(f"  --- K[{I},{J}]  shape={K_IJ.shape} ---", flush=True)
            print(
                np.array2string(
                    K_IJ, max_line_width=100000, threshold=np.inf,
                    precision=4, suppress_small=False,
                ),
                flush=True,
            )
            om_I = omega_by_pos[I]
            om_J = omega_by_pos[J]
            ntop = min(10, K_IJ.size)
            order = np.argsort(np.abs(K_IJ), axis=None)[::-1][:ntop]
            print(
                f"  top {ntop} |K[{I},{J}]| (ia, jb, K[Ha], omega_I[eV], omega_J[eV]):",
                flush=True,
            )
            for fidx in order:
                ia, jb = np.unravel_index(int(fidx), K_IJ.shape)
                print(
                    f"    ({ia:3d},{jb:3d})  K={K_IJ[ia, jb]:+.6e}  "
                    f"omega_I={om_I[ia] * 27.2114:7.3f}  "
                    f"omega_J={om_J[jb] * 27.2114:7.3f}",
                    flush=True,
                )
        try:
            payload = {
                f"K_{I}_{J}": np.asarray(K_coupling[(I, J)])
                for (I, J) in K_coupling.keys()
            }
            for pos, om in omega_by_pos.items():
                payload[f"omega_{pos}"] = om
            np.savez("coupling_K_matrix.npz", **payload)
            print("  Saved full K blocks -> coupling_K_matrix.npz", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  (could not save coupling_K_matrix.npz: {exc})", flush=True)

    K_pos = _remap_k_coupling_to_positions(K_coupling, active_indices)

    if verbose:
        print("  Solving coupled Casida equation...", flush=True)

    omega_coupled, Z_coupled = assemble_coupled_casida(
        active_results,
        K_pos,
        tda=tda,
        verbose=verbose,
    )
    f_coupled = coupled_oscillator_strengths(
        omega_coupled,
        Z_coupled,
        active_results,
        tda=tda,
        verbose=verbose,
    )

    if verbose:
        if omega_coupled.size:
            print(
                f"  Coupled omega range: "
                f"{float(np.min(omega_coupled)):.6f}–{float(np.max(omega_coupled)):.6f} Ha",
                flush=True,
            )
            print(
                f"  Raw sum(f_coupled): {float(np.nansum(f_coupled)):.6g}",
                flush=True,
            )
        print(
            f"  Done. {len(omega_coupled)} coupled excitations computed.",
            flush=True,
        )

    return {
        "omega": omega_coupled,
        "f": f_coupled,
        "Z": Z_coupled,
        "K_coupling": K_coupling,
        "f_nadd": f_nadd,
    }


# ---------------------------------------------------------------------------
# Distributed (in-memory, disk-free) subsystem coupling
# ---------------------------------------------------------------------------
#
# Each fragment's amplitude-basis ``rho_transition`` stack stays resident on its
# owner rank. Coupling blocks are computed where the data already lives; the
# partner fragment's transition densities are streamed one state at a time over
# the world communicator. Peak per-rank memory is therefore one resident
# fragment stack + one streamed grid + the (broadcast) ``f_nadd`` global field —
# independent of fragment count and of every other fragment's state count.


def _select_active_fragments(fragment_results, omega_coupling_max, verbose):
    """Fragments with >=1 state (after the optional ``omega`` cap).

    Returns ``(active_indices, active_results)`` when >=2 fragments couple, or a
    final result dict when fewer remain (0 or 1 active). This is the shared
    front-end for both the serial and distributed backends so they agree exactly
    on which fragments/states enter the coupled problem.
    """
    n_frag = len(fragment_results)
    active_indices = [i for i in range(n_frag) if fragment_results[i] is not None]
    n_active = len(active_indices)

    if n_active < 2:
        if verbose:
            print("Less than 2 active fragments — no coupling to compute.", flush=True)
        if n_active == 1:
            idx = active_indices[0]
            res = fragment_results[idx]
            return {
                "omega": _res_array(res, "omega"),
                "f": _res_optional(res, "f", "os_strength", default=np.array([])),
                "Z": _res_optional(res, "Z", "eigenvectors", default=np.array([[]])),
                "K_coupling": {},
                "f_nadd": None,
            }
        return _empty_coupled_result()

    # Drop fragments with zero states after active-space reduction
    paired: List[Tuple[int, Dict[str, Any]]] = []
    for idx in active_indices:
        res = fragment_results[idx]
        omega = _res_array(res, "omega")
        if omega_coupling_max is not None:
            keep = np.where(omega <= float(omega_coupling_max))[0]
            if keep.size == 0:
                if verbose:
                    print(
                        f"  Fragment {idx}: 0 states with omega <= "
                        f"{float(omega_coupling_max)} Ha — skipped.",
                        flush=True,
                    )
                continue
            if keep.size < omega.size:
                res = {
                    **res,
                    "omega": omega[keep],
                    "f": np.asarray(_res_optional(res, "f", "os_strength", default=[]))[keep]
                    if _res_optional(res, "f", "os_strength") is not None else None,
                    "Z": np.eye(len(keep), dtype=float),
                    "eigenvectors": np.eye(len(keep), dtype=float),
                    "xpy": np.asarray(res["xpy"])[:, keep] if res.get("xpy") is not None else None,
                    "dip_tran": np.asarray(res["dip_tran"])[keep]
                    if res.get("dip_tran") is not None
                    and np.asarray(res["dip_tran"]).shape[0] == omega.size
                    else res.get("dip_tran"),
                    "rho_transition": [
                        res["rho_transition"][int(i)] for i in keep
                    ] if res.get("rho_transition") is not None else None,
                }
                omega = res["omega"]
        if len(omega) == 0:
            if verbose:
                print(f"  Fragment {idx}: 0 states in active space — skipped.", flush=True)
            continue
        paired.append((idx, res))

    if len(paired) < 2:
        if verbose:
            print(
                "Fewer than 2 fragments with excitations — no coupling to compute.",
                flush=True,
            )
        return _empty_coupled_result()

    return [idx for idx, _ in paired], [res for _, res in paired]


def round_robin_rounds(items):
    """Circle-method schedule: list of rounds, each a set of disjoint pairs.

    Every unordered pair of ``items`` appears in exactly one round, and within a
    round no item repeats. Used so that in each communication round every owner
    rank is involved in at most one point-to-point exchange — which keeps the
    streaming deadlock-free (each pair is an independent producer→consumer link).
    """
    items = list(items)
    if len(items) % 2:
        items = items + [None]  # bye
    n = len(items)
    arr = items[:]
    rounds = []
    for _ in range(max(n - 1, 0)):
        pairs = []
        for i in range(n // 2):
            a, b = arr[i], arr[n - 1 - i]
            if a is not None and b is not None:
                pairs.append((a, b))
        rounds.append(pairs)
        arr = [arr[0]] + [arr[-1]] + arr[1:-1]  # rotate, fixing the first slot
    return rounds


def build_block_schedule(active_indices, fragment_owners, n_states_by_frag):
    """Assign each fragment pair to the owner that will *compute* its block.

    The computing owner keeps its own fragment resident (all states) and streams
    the partner state-by-state, so its added cost is the partner's state count
    (one Hartree FFT per streamed state). Greedy LPT balancing of that streamed
    FFT load across owner ranks. Deterministic given identical inputs, so every
    rank derives the same schedule without communication.
    """
    load: Dict[int, int] = {}
    sched: Dict[Tuple[int, int], int] = {}
    pairs = [
        (active_indices[a], active_indices[b])
        for a in range(len(active_indices))
        for b in range(a + 1, len(active_indices))
    ]
    # Heaviest pairs first (longest-processing-time-first), tuple tie-break for
    # a fully deterministic order on every rank.
    pairs.sort(key=lambda p: (-(n_states_by_frag[p[0]] + n_states_by_frag[p[1]]), p))
    for (I, J) in pairs:
        oi, oj = fragment_owners[I], fragment_owners[J]
        cost_i = n_states_by_frag[J]  # owner(I) computes -> streams J
        cost_j = n_states_by_frag[I]  # owner(J) computes -> streams I
        li = load.get(oi, 0) + cost_i
        lj = load.get(oj, 0) + cost_j
        # Prefer the lighter resulting load; tie -> owner(I) for determinism.
        if li <= lj:
            sched[(min(I, J), max(I, J))] = oi
            load[oi] = li
        else:
            sched[(min(I, J), max(I, J))] = oj
            load[oj] = lj
    return sched


def broadcast_f_nadd(f_nadd, comm, root=0):
    """Broadcast the (single) global-grid coupling kernel from ``root`` to all ranks."""
    payload = np.ascontiguousarray(np.asarray(f_nadd, dtype=np.float64)) \
        if comm.rank == root else None
    return comm.bcast(payload, root=root)


def _send_grid(comm, arr, dest, tag):
    """Send one transition-density grid (contiguous float64) to ``dest``."""
    comm.send(np.ascontiguousarray(np.asarray(arr, dtype=np.float64)), dest=dest, tag=tag)


def _recv_grid(comm, source, tag):
    """Receive one streamed transition-density grid from ``source``."""
    return comm.recv(source=source, tag=tag)


def _resident_rho(res_by_frag, idx):
    """Local resident ``rho_transition`` for fragment ``idx`` as a list of grids."""
    res = res_by_frag.get(idx)
    rho = res.get("rho_transition") if res is not None else None
    if rho is None:
        raise ValueError(
            f"Fragment {idx}: resident rho_transition missing on its owner rank "
            "(distributed coupling expects the owner to keep its own densities).",
        )
    stack, n_rho = _rho_transition_to_stack(rho)
    return [stack[i] for i in range(n_rho)]


def _union_index_box(gsystem, isub_a, isub_b, pad=0):
    """Global-index bounding box (shift, shape) covering both subcells + ``pad``.

    Pure index arithmetic on the replicated ``graph.sub_shift``/``sub_shape``,
    clipped to the grid. Assumes neither subcell wraps the periodic boundary (the
    same assumption as ``_global_sub_index``).
    """
    graph = gsystem.graphtopo.graph
    nrR = np.asarray(gsystem.grid.nrR, dtype=int)
    sa = np.asarray(graph.sub_shift[isub_a], dtype=int)
    na = np.asarray(graph.sub_shape[isub_a], dtype=int)
    sb = np.asarray(graph.sub_shift[isub_b], dtype=int)
    nb = np.asarray(graph.sub_shape[isub_b], dtype=int)
    lo = np.maximum(np.minimum(sa, sb) - pad, 0)
    hi = np.minimum(np.maximum(sa + na, sb + nb) + pad, nrR)
    return lo, np.maximum(hi - lo, 0)


def _place_in_box(sub_arr, sub_shift, box_shift, box_shape):
    """Place a subcell array into a ``box_shape`` array at ``sub_shift-box_shift``.

    Clips to the box overlap (robust if a subcell edge falls outside the box).
    """
    out = np.zeros(tuple(int(x) for x in box_shape), dtype=float)
    off = np.asarray(sub_shift, dtype=int) - np.asarray(box_shift, dtype=int)
    src, dst = [None] * 3, [None] * 3
    for i in range(3):
        lo = max(int(off[i]), 0)
        hi = min(int(off[i]) + int(sub_arr.shape[i]), int(box_shape[i]))
        if hi <= lo:
            return out  # no overlap along this axis
        dst[i] = slice(lo, hi)
        src[i] = slice(lo - int(off[i]), hi - int(off[i]))
    out[tuple(dst)] = np.asarray(sub_arr)[tuple(src)]
    return out


def _box_grid(gsystem, box_shift, box_shape):
    """Serial DirectGrid on the index sub-box (physical size = shape * spacing)."""
    nrR = np.asarray(gsystem.grid.nrR, dtype=float)
    lattice = np.asarray(gsystem.grid.lattice, dtype=float)
    scale = (np.asarray(box_shape, dtype=float) / nrR)[:, None]
    return DirectGrid(
        lattice=lattice * scale,
        nr=tuple(int(x) for x in box_shape),
        full=gsystem.grid.full,
        mp=MP(comm=SerialComm()),
    )


def _sr_coupling_block_box(
    phi_res, stream_state, n_stream, gsystem, isub_res, isub_stream, f_nadd, alpha, pad=2,
):
    """Screened (erfc) short-range coupling block on the *union sub-box* of the two
    fragments — the memory-saving variant of the SR term.

    Because the SR Coulomb is screened it decays within ~1/alpha, so it only needs
    a grid spanning where the two densities are close, not the full cell. Returns
    ``K`` (n_res, n_stream); combine with the far-field dipole LR as usual.
    """
    n_res = len(phi_res)
    if n_res == 0 or n_stream == 0:
        return np.zeros((n_res, n_stream), dtype=float)

    box_shift, box_shape = _union_index_box(gsystem, isub_res, isub_stream, pad)
    if np.any(np.asarray(box_shape) <= 0):
        return np.zeros((n_res, n_stream), dtype=float)

    graph = gsystem.graphtopo.graph
    s_res = np.asarray(graph.sub_shift[isub_res], dtype=int)
    s_str = np.asarray(graph.sub_shift[isub_stream], dtype=int)
    box_grid = _box_grid(gsystem, box_shift, box_shape)
    box_dV = float(box_grid.dV)

    bsl = tuple(slice(int(box_shift[i]), int(box_shift[i] + box_shape[i])) for i in range(3))
    f_nadd_box = np.asarray(f_nadd)[bsl]

    res_flat = np.stack([
        _place_in_box(
            np.asarray(_embedding_subcell_field(phi_res[ia], gsystem, isub_res)),
            s_res, box_shift, box_shape,
        ).ravel()
        for ia in range(n_res)
    ])  # (n_res, n_box)

    K = np.zeros((n_res, n_stream), dtype=float)
    for jb in range(n_stream):
        phi_j_box = _place_in_box(
            np.asarray(_embedding_subcell_field(stream_state(jb), gsystem, isub_stream)),
            s_str, box_shift, box_shape,
        )
        fld = DirectField(grid=box_grid, rank=1)
        fld[:] = phi_j_box.reshape(fld.shape)
        v_sr = np.asarray(_screened_coulomb_potential(fld, alpha, "sr"))
        response = v_sr + f_nadd_box * phi_j_box
        K[:, jb] = res_flat @ response.ravel() * box_dV
    return K


def _screened_coulomb_potential(rho, alpha, part="sr"):
    """Range-separated Hartree potential of a field on its own grid.

    ``1/r = erfc(alpha r)/r + erf(alpha r)/r``. In reciprocal space the kernel is
    ``4 pi / G^2`` times ``1`` (full), ``1 - exp(-G^2/4 alpha^2)`` (``part='sr'``,
    the screened/erfc short range) or ``exp(-G^2/4 alpha^2)`` (``part='lr'``, erf).
    Exact for **charge-neutral** densities (transition densities): rho(G=0)=0, so
    dropping the G=0 term (invgg=0 there) is exact, and the screened SR part is
    image-free because it decays within ~1/alpha of the density.
    """
    recip = rho.grid.get_reciprocal()
    invgg = recip.invgg
    rho_g = rho.fft()
    if part == "full":
        ker = 4.0 * np.pi * invgg
    else:
        damp = np.exp(-recip.gg / (4.0 * alpha * alpha))
        if part == "sr":
            ker = 4.0 * np.pi * invgg * (1.0 - damp)
        elif part == "lr":
            ker = 4.0 * np.pi * invgg * damp
        else:
            raise ValueError(f"unknown screened-Coulomb part {part!r}")
    return (rho_g * ker).ifft(force_real=True)


def compute_coupling_block_streamed(
    phi_res,
    stream_state,
    n_stream,
    gsystem,
    isub_res,
    isub_stream,
    f_nadd,
    hartree_func,
    include_hartree: bool = True,
    screen_alpha=None,
    screen_part: str = "sr",
) -> np.ndarray:
    """Coupling block with the resident fragment stacked and the partner streamed.

    ``phi_res`` is the full resident transition-density list; ``stream_state(jb)``
    returns the ``jb``-th grid of the streamed fragment (pulled over MPI, or read
    locally). Returns ``K`` with shape ``(len(phi_res), n_stream)`` — the same
    numbers :func:`compute_coupling_block` produces, just consuming the streamed
    fragment one state at a time so at most one of its grids is ever resident.

    When ``screen_alpha`` is set, the Hartree part uses the range-separated
    (``screen_part``) screened Coulomb instead of the full ``1/r`` — the grid half
    of the erf/erfc split (``sr`` = image-free short range).
    """
    n_res = len(phi_res)
    if n_res == 0 or n_stream == 0:
        return np.zeros((n_res, n_stream), dtype=float)

    global_grid = gsystem.grid
    phi_res_fields = [
        _embedding_subcell_field(phi_res[ia], gsystem, isub_res) for ia in range(n_res)
    ]
    phi_res_flat = np.stack([np.asarray(f).ravel() for f in phi_res_fields])  # (n_res, n_sub)
    sub_dV = float(phi_res_fields[0].grid.dV)

    K = np.zeros((n_res, n_stream), dtype=float)
    for jb in range(n_stream):
        phi_jb = _embedding_subcell_field(stream_state(jb), gsystem, isub_stream)
        phi_jb_global = _sub_to_global(phi_jb, gsystem, isub_stream)
        if include_hartree:
            if screen_alpha is not None:
                vh_jb = _screened_coulomb_potential(phi_jb_global, screen_alpha, screen_part)
            else:
                vh_jb = hartree_func(phi_jb_global, calcType=["V"]).potential
            response_global = np.asarray(vh_jb) + f_nadd * np.asarray(phi_jb_global)
        else:
            response_global = f_nadd * np.asarray(phi_jb_global)
        response_field = DirectField(grid=global_grid, rank=1)
        response_field[:] = response_global.reshape(response_field.shape)
        if (include_hartree and screen_alpha is None and jb == 0
                and np.abs(np.asarray(response_field)).sum() == 0.0):
            # (screened SR can legitimately be ~0 between separated fragments)
            raise RuntimeError(
                f"compute_coupling_block_streamed: response_field all zeros for "
                f"fragments ({isub_res},{isub_stream}), jb=0.",
            )
        response_on_res = _global_to_sub(response_field, gsystem, isub_res)
        response_arr = np.asarray(response_on_res).ravel()
        K[:, jb] = phi_res_flat @ response_arr * sub_dV
    return K


def _compute_f_nadd(
    gsystem, drivers, active_indices, sub_densities,
    xc_functional, ke_functional, rho_cutoff, fxc_max, verbose,
):
    """Pavanello non-additive kernel f_xc + f_T at rho_total (serial-grid, root only)."""
    rho_total = gsystem.density
    rho_subs = []
    for idx in active_indices:
        rho_I_global = DirectField(grid=gsystem.grid, rank=1)
        rho_I_global[:] = 0.0
        subrho = None
        if sub_densities is not None and idx < len(sub_densities):
            subrho = sub_densities[idx]
        elif idx < len(drivers) and drivers[idx] is not None:
            subrho = drivers[idx].density
        if subrho is not None:
            sub_field = _embedding_subcell_field(subrho, gsystem, idx)
            index = _global_sub_index(gsystem, idx)
            np.asarray(rho_I_global)[index] = np.asarray(sub_field).reshape(
                np.asarray(rho_I_global)[index].shape,
            )
        rho_subs.append(rho_I_global)

    if verbose:
        print("  Computing Pavanello coupling kernel (f_xc + f_T at rho_tot)...", flush=True)
    f_nadd = compute_nadd_kernel(
        rho_total, rho_subs, xc_functional, ke_functional,
        rho_cutoff=rho_cutoff, fxc_max=fxc_max,
    )
    fnadd_arr = np.asarray(f_nadd, dtype=float)
    if verbose:
        nz = float(np.sum(fnadd_arr != 0.0)) / max(fnadd_arr.size, 1)
        print(
            f"  f_nadd stats: min={float(fnadd_arr.min()):.4e}  "
            f"max={float(fnadd_arr.max()):.4e}  nonzero={nz:.1%}",
            flush=True,
        )
        if nz == 0.0:
            print(
                "  WARNING: f_nadd is entirely zero — all coupling will be pure "
                "Hartree. Check rho_cutoff and that rho_total is non-zero.",
                flush=True,
            )
    return fnadd_arr


def _run_subsystem_casida_distributed(
    gsystem,
    drivers,
    active_indices,
    active_results,
    xc_functional,
    ke_functional=None,
    sub_densities=None,
    tda=False,
    rho_cutoff=1e-3,
    fxc_max=20.0,
    verbose=True,
    graphtopo=None,
    fragment_owners=None,
    f_nadd=None,
    hartree_mode="full",
    coupling_alpha=None,
):
    """Collective distributed coupling. See :func:`run_subsystem_casida`.

    Every rank calls this. Returns the coupled result dict on world rank 0 and
    ``None`` elsewhere (eDFTpy broadcasts the slim spectrum afterwards).

    ``hartree_mode``:
      * ``"full"``  — grid Hartree on the serial global grid (streams densities).
      * ``"dipole"`` — far-field (Förster) coupling: the erf->1 long-range limit
        of the range-separated Coulomb, i.e. transition-dipole/transition-dipole
        interaction. Needs no grids and no density streaming (uses ``dip_tran``
        metadata already present on every rank), so it is O(1) memory and image
        free. Accurate for well-separated fragments; misses short-range overlap
        and higher multipoles.
      * ``"rangesep"`` — full erf/erfc range separation: screened (erfc) short
        range on the **global** grid (image-free) + erf long range as far-field
        dipole. Image-corrected coupling at ~the same memory as ``full``;
        ``coupling_alpha`` (a.u.^-1) sets the split (exact result is
        alpha-independent).
      * ``"rangesep_box"`` — same physics as ``rangesep`` but the screened SR is
        solved on the **union sub-box** of the two fragments instead of the global
        grid (memory-saving; the SR is localized within ~1/alpha so the small box
        is sufficient). Validate against ``rangesep`` — they should agree when the
        box + pad exceeds the screening length.
    """
    if graphtopo is None or fragment_owners is None:
        raise ValueError(
            "distributed coupling requires graphtopo and fragment_owners.",
        )
    comm = graphtopo.comm
    is_mpi = graphtopo.is_mpi
    my_rank = graphtopo.rank
    root = 0

    res_by_frag = {idx: res for idx, res in zip(active_indices, active_results)}
    n_states_by_frag = {
        idx: len(_res_array(res, "omega")) for idx, res in res_by_frag.items()
    }

    # ---- Förster (far-field, dipole) path: metadata-only, no grids/streaming ----
    if str(hartree_mode).lower() == "dipole":
        if is_mpi and my_rank != root:
            return None
        return _assemble_dipole_coupling(
            gsystem, active_indices, active_results, res_by_frag, tda, verbose,
        )

    # ---- f_nadd: build on root (needs serial gsystem.density), broadcast ----
    if f_nadd is None:
        if (not is_mpi) or my_rank == root:
            f_nadd = _compute_f_nadd(
                gsystem, drivers, active_indices, sub_densities,
                xc_functional, ke_functional, rho_cutoff, fxc_max, verbose,
            )
        if is_mpi:
            f_nadd = broadcast_f_nadd(f_nadd, comm, root=root)
    f_nadd = np.asarray(f_nadd, dtype=float)

    hartree = Functional(type="HARTREE")

    # Range-separated Hartree: the grid loop computes the screened (erfc) short
    # range with 'sr' kernel; the erf long range is added afterwards as far-field
    # dipole coupling (root only). A FIXED alpha (a.u.) is used so the split is
    # identical across a distance scan; the exact result is alpha-independent, and
    # a smaller alpha pushes more of the interaction onto the exact grid SR.
    _mode = str(hartree_mode).lower()
    _rangesep = _mode in ("rangesep", "rangesep_box")
    _rangesep_box = _mode == "rangesep_box"
    _screen_alpha = None
    if _rangesep:
        _screen_alpha = float(coupling_alpha) if coupling_alpha else _DEFAULT_COUPLING_ALPHA
        if verbose and ((not is_mpi) or my_rank == root):
            where = "union sub-box" if _rangesep_box else "global grid"
            print(
                f"  Range-separated coupling Hartree: erfc SR on {where} (alpha="
                f"{_screen_alpha:.3f} a0^-1) + erf far-field dipole LR.",
                flush=True,
            )

    # Per-rank diagnostics: resident footprint (proves the O(1)-streamed invariant
    # — a rank only ever holds its own fragment's stack, never all of them) and a
    # sub->global->sub round-trip on the first resident state (embedding sanity).
    if verbose:
        for f in active_indices:
            if fragment_owners[f] != my_rank:
                continue
            rho = res_by_frag[f].get("rho_transition")
            if rho is None:
                continue
            stack, n_rho = _rho_transition_to_stack(rho)
            if n_rho == 0:
                continue
            print(
                f"  [rank {my_rank}] resident fragment {f}: {n_rho} grids, "
                f"{stack.nbytes / 1e9:.3f} GB",
                flush=True,
            )
            phi0 = _embedding_subcell_field(stack[0], gsystem, f)
            back = _global_to_sub(_sub_to_global(phi0, gsystem, f), gsystem, f)
            orig = np.asarray(phi0).ravel()
            rec = np.asarray(back).ravel()
            err = float(np.max(np.abs(orig - rec)))
            scale = max(float(np.max(np.abs(orig))), 1e-30)
            print(
                f"  [rank {my_rank}] fragment {f} sub->global->sub round-trip "
                f"max error: {err:.3e}",
                flush=True,
            )
            if err > 1e-10 * scale:
                print(
                    f"  [rank {my_rank}] WARNING: large round-trip error for "
                    f"fragment {f} — check get_sub_index vs field shape.",
                    flush=True,
                )

    schedule = build_block_schedule(active_indices, fragment_owners, n_states_by_frag)
    rounds = round_robin_rounds(active_indices)

    if verbose and ((not is_mpi) or my_rank == root):
        print(
            f"Distributed subsystem Casida coupling: {len(active_indices)} fragments, "
            f"{sum(len(r) for r in rounds)} blocks over {len(rounds)} rounds "
            f"(streaming partner densities state-by-state).",
            flush=True,
        )

    K_local: Dict[Tuple[int, int], np.ndarray] = {}
    for rnd in rounds:
        for (A, B) in rnd:
            key = (min(A, B), max(A, B))
            owner = schedule[key]
            resident = A if owner == fragment_owners[A] else B
            producer = B if resident == A else A
            resident_rank = fragment_owners[resident]
            producer_rank = fragment_owners[producer]
            n_stream = n_states_by_frag[producer]

            if my_rank == resident_rank:
                phi_res = _resident_rho(res_by_frag, resident)
                if producer_rank == resident_rank:
                    # Same rank owns both fragments (only in non-MPI / SerialComm):
                    # read the partner locally, no messages.
                    phi_prod = _resident_rho(res_by_frag, producer)

                    def _getter(jb, _p=phi_prod):
                        return _p[jb]
                else:
                    def _getter(jb, _src=producer_rank):
                        return _recv_grid(comm, _src, jb)

                if _rangesep_box:
                    K_rs = _sr_coupling_block_box(
                        phi_res, _getter, n_stream, gsystem,
                        resident, producer, f_nadd, _screen_alpha,
                    )
                else:
                    K_rs = compute_coupling_block_streamed(
                        phi_res, _getter, n_stream, gsystem,
                        resident, producer, f_nadd, hartree, include_hartree=True,
                        screen_alpha=_screen_alpha, screen_part="sr",
                    )
                # Canonical orientation K[(min,max)].
                K_local[key] = K_rs if resident == key[0] else K_rs.T
            elif my_rank == producer_rank:
                phi_prod = _resident_rho(res_by_frag, producer)
                for jb in range(n_stream):
                    _send_grid(comm, phi_prod[jb], resident_rank, jb)
            # ranks owning neither fragment sit this pair out
        if is_mpi:
            comm.Barrier()

    # ---- gather the (small) K blocks to root ----
    if is_mpi:
        gathered = comm.gather(K_local, root=root)
    else:
        gathered = [K_local]
    if is_mpi and my_rank != root:
        return None

    K_coupling: Dict[Tuple[int, int], np.ndarray] = {}
    for d in (gathered or []):
        if d:
            K_coupling.update(d)
    for (I, J), K_IJ in list(K_coupling.items()):
        K_coupling.setdefault((J, I), np.asarray(K_IJ).T)

    # Range separation: add the erf long range as far-field dipole coupling. The
    # grid loop above produced the screened (erfc) short range + f_nadd; here we
    # add the smooth erf tail from the transition dipoles (no grid).
    if _rangesep:
        centers = {i: _fragment_center(gsystem, i) for i in active_indices}
        for (I, J) in [k for k in list(K_coupling.keys()) if k[0] < k[1]]:
            mu_I = _res_optional(res_by_frag[I], "dip_tran")
            mu_J = _res_optional(res_by_frag[J], "dip_tran")
            if mu_I is None or mu_J is None:
                continue
            K_lr = dipole_dipole_block(mu_I, mu_J, centers[I], centers[J])
            K_coupling[(I, J)] = np.asarray(K_coupling[(I, J)]) + K_lr
            K_coupling[(J, I)] = np.asarray(K_coupling[(J, I)]) + K_lr.T

    if verbose:
        for (I, J), K_IJ in sorted(K_coupling.items()):
            if I < J:
                K_IJ = np.asarray(K_IJ)
                k_max = float(np.max(np.abs(K_IJ))) if K_IJ.size else 0.0
                print(f"  K[{I},{J}] shape={K_IJ.shape}  max|K|={k_max:.4e} Ha", flush=True)

    K_pos = _remap_k_coupling_to_positions(K_coupling, active_indices)
    if verbose:
        print("  Solving coupled Casida equation...", flush=True)
    omega_coupled, Z_coupled = assemble_coupled_casida(
        active_results, K_pos, tda=tda, verbose=verbose,
    )
    f_coupled = coupled_oscillator_strengths(
        omega_coupled, Z_coupled, active_results, tda=tda, verbose=verbose,
    )
    if verbose:
        print(
            f"  Done. {len(omega_coupled)} coupled excitations computed (distributed).",
            flush=True,
        )

    return {
        "omega": omega_coupled,
        "f": f_coupled,
        "Z": Z_coupled,
        "K_coupling": K_coupling,
        "f_nadd": f_nadd,
    }


# ---------------------------------------------------------------------------
# Far-field (Förster) coupling: erf->1 long-range limit of range-separated
# Coulomb, i.e. transition-dipole/transition-dipole interaction. No grids.
# ---------------------------------------------------------------------------


def dipole_dipole_block(mu_I, mu_J, R_I, R_J) -> np.ndarray:
    """Transition-dipole coupling block K[m,n] = mu_I[m] . T(R_IJ) . mu_J[n].

    ``mu_I`` (n_I, 3), ``mu_J`` (n_J, 3) are transition dipoles (a.u.); these are
    origin-independent because transition densities are charge-neutral, so only
    the separation ``R_J - R_I`` between fragment centers matters. ``T`` is the
    dipole-dipole tensor ``(I - 3 u u^T)/R^3`` (a.u.), the exact long-range limit
    of the Coulomb coupling between two neutral localized densities (Förster).
    Returns K in Hartree, shape (n_I, n_J).
    """
    mu_I = np.atleast_2d(np.asarray(mu_I, dtype=float))
    mu_J = np.atleast_2d(np.asarray(mu_J, dtype=float))
    d = np.asarray(R_J, dtype=float) - np.asarray(R_I, dtype=float)
    R = float(np.linalg.norm(d))
    if R < 1e-8:
        return np.zeros((mu_I.shape[0], mu_J.shape[0]), dtype=float)
    u = d / R
    T = (np.eye(3) - 3.0 * np.outer(u, u)) / R**3
    return mu_I @ T @ mu_J.T


def _fragment_center(gsystem, isub) -> np.ndarray:
    """Cartesian center (a.u.) of subsystem ``isub``'s embedding subcell.

    Derived from the globally-replicated ``graph.sub_shift``/``sub_shape`` and the
    serial grid geometry, so it is valid on every rank without communication.
    """
    graph = gsystem.graphtopo.graph
    shift = np.asarray(graph.sub_shift[isub], dtype=float)
    shape = np.asarray(graph.sub_shape[isub], dtype=float)
    nrR = np.asarray(gsystem.grid.nrR, dtype=float)
    lattice = np.asarray(gsystem.grid.lattice, dtype=float)  # rows = cell vectors (a.u.)
    frac = (shift + 0.5 * shape) / nrR
    return frac @ lattice


def _assemble_dipole_coupling(
    gsystem, active_indices, active_results, res_by_frag, tda, verbose,
) -> Dict[str, Any]:
    """Build + solve the coupled problem with far-field dipole coupling (root only)."""
    centers = {i: _fragment_center(gsystem, i) for i in active_indices}
    K_coupling: Dict[Tuple[int, int], np.ndarray] = {}
    for a, I in enumerate(active_indices):
        mu_I = _res_optional(res_by_frag[I], "dip_tran")
        if mu_I is None:
            continue
        for J in active_indices[a + 1:]:
            mu_J = _res_optional(res_by_frag[J], "dip_tran")
            if mu_J is None:
                continue
            K_IJ = dipole_dipole_block(mu_I, mu_J, centers[I], centers[J])
            K_coupling[(I, J)] = K_IJ
            K_coupling[(J, I)] = K_IJ.T

    if verbose:
        for (I, J), K_IJ in sorted(K_coupling.items()):
            if I < J:
                R = float(np.linalg.norm(centers[J] - centers[I]))
                k_max = float(np.max(np.abs(K_IJ))) if K_IJ.size else 0.0
                print(
                    f"  [dipole] K[{I},{J}] R={R:.3f} a0  max|K|={k_max:.4e} Ha",
                    flush=True,
                )

    K_pos = _remap_k_coupling_to_positions(K_coupling, active_indices)
    omega_coupled, Z_coupled = assemble_coupled_casida(
        active_results, K_pos, tda=tda, verbose=verbose,
    )
    f_coupled = coupled_oscillator_strengths(
        omega_coupled, Z_coupled, active_results, tda=tda, verbose=verbose,
    )
    if verbose:
        print(
            f"  Done. {len(omega_coupled)} coupled excitations (far-field dipole).",
            flush=True,
        )
    return {
        "omega": omega_coupled,
        "f": f_coupled,
        "Z": Z_coupled,
        "K_coupling": K_coupling,
        "f_nadd": None,
    }
