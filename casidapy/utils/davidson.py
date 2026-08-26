"""
Iterative eigensolvers for large matrices using scipy's LinearOperator.

Provides wrappers around scipy's lobpcg and eigsh for matrix-free eigenvalue
problems, particularly useful for Casida TDDFT calculations.

Supports:
- Matrix-free operations via LinearOperator
- Dense and sparse matrices
- Preconditioning for faster convergence

References:
    scipy.sparse.linalg.lobpcg
    scipy.sparse.linalg.eigsh
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, lobpcg, eigsh
from scipy.sparse import issparse, diags
from typing import Optional, Union, Tuple, Callable
import warnings


def create_linear_operator(
    matvec: Callable,
    n: int,
    dtype: np.dtype = np.float64
) -> LinearOperator:
    """
    Create a scipy LinearOperator from a matrix-vector product function.
    
    Parameters
    ----------
    matvec : callable
        Function that computes matrix @ vector
    n : int
        Size of the square matrix
    dtype : dtype
        Data type of the operator
        
    Returns
    -------
    LinearOperator
        Scipy LinearOperator that can be used with lobpcg, eigsh, etc.
    """
    return LinearOperator(
        shape=(n, n),
        matvec=matvec,
        dtype=dtype
    )


def create_preconditioner(
    diagonal: np.ndarray,
    shift: float = 0.0
) -> LinearOperator:
    """
    Create a diagonal preconditioner M ≈ (D - shift)^(-1).
    
    Parameters
    ----------
    diagonal : array
        Diagonal elements of the matrix
    shift : float
        Shift for the preconditioner (e.g., approximate eigenvalue)
        
    Returns
    -------
    LinearOperator
        Preconditioner as LinearOperator
    """
    n = len(diagonal)
    
    # Compute inverse diagonal with regularization
    denom = diagonal - shift
    # Avoid division by zero
    denom_safe = np.where(np.abs(denom) < 1e-10, 
                          np.sign(denom + 1e-16) * 1e-10, 
                          denom)
    diag_inv = 1.0 / denom_safe
    
    def precond_matvec(v):
        """Apply diagonal preconditioner, handling both 1D and 2D inputs."""
        v = np.asarray(v)
        # Flatten to avoid broadcast issues with (n,) * (n, 1) -> (n, n)
        v_flat = v.flatten()
        result = diag_inv * v_flat
        # Return same shape as input
        return result.reshape(v.shape)
    
    return LinearOperator(
        shape=(n, n),
        matvec=precond_matvec,
        dtype=np.float64
    )


def solve_lobpcg(
    A: Union[np.ndarray, LinearOperator, Callable],
    nroots: int = 1,
    X0: Optional[np.ndarray] = None,
    diagonal: Optional[np.ndarray] = None,
    tol: float = 1e-8,
    maxiter: int = 200,
    largest: bool = False,
    verbose: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find eigenvalues using LOBPCG (Locally Optimal Block Preconditioned 
    Conjugate Gradient).
    
    LOBPCG is efficient for finding a few smallest (or largest) eigenvalues
    of large symmetric matrices, especially with matrix-free operations.
    
    Parameters
    ----------
    A : array, LinearOperator, or callable
        The matrix to diagonalize. Can be:
        - Dense numpy array
        - Sparse matrix
        - LinearOperator
        - Callable function that computes A @ v
    nroots : int
        Number of eigenvalues/eigenvectors to compute
    X0 : array, optional
        Initial guess vectors (n, nroots). If None, random vectors are used.
    diagonal : array, optional
        Diagonal of A for preconditioning. Improves convergence significantly.
    tol : float
        Convergence tolerance
    maxiter : int
        Maximum number of iterations
    largest : bool
        If True, find largest eigenvalues. If False, find smallest.
    verbose : int
        Verbosity level (0=silent, 1=summary, 2+=detailed)
        
    Returns
    -------
    eigenvalues : array
        Computed eigenvalues (sorted)
    eigenvectors : array
        Computed eigenvectors (columns)
    """
    # Convert to LinearOperator if needed
    if callable(A) and not isinstance(A, LinearOperator):
        # Determine size from X0 or by probing
        if X0 is not None:
            n = X0.shape[0]
        elif diagonal is not None:
            n = len(diagonal)
        else:
            # Probe the function to get size
            for test_size in [100, 500, 1000, 5000, 10000]:
                try:
                    result = A(np.ones(test_size))
                    n = len(result)
                    break
                except:
                    continue
            else:
                raise ValueError("Cannot determine matrix size. Provide X0 or diagonal.")
        
        A_op = create_linear_operator(A, n)
    elif isinstance(A, np.ndarray):
        n = A.shape[0]
        A_op = LinearOperator(shape=(n, n), matvec=lambda v: A @ v, dtype=A.dtype)
    elif issparse(A):
        n = A.shape[0]
        A_op = A  # Sparse matrices work directly with lobpcg
    else:
        A_op = A
        n = A.shape[0]
    
    # Initialize guess vectors
    if X0 is None:
        X0 = np.random.randn(n, nroots)
        if diagonal is not None:
            # Better initialization: unit vectors for smallest diagonal elements
            sorted_idx = np.argsort(diagonal) if not largest else np.argsort(-diagonal)
            X0 = np.zeros((n, nroots))
            for i in range(min(nroots, len(sorted_idx))):
                X0[sorted_idx[i], i] = 1.0
            # Add small perturbation
            X0 += 0.01 * np.random.randn(n, nroots)
    
    # Orthonormalize initial guess
    X0, _ = np.linalg.qr(X0)
    
    # Create preconditioner if diagonal is provided
    if diagonal is not None:
        M_op = create_preconditioner(diagonal)
    else:
        M_op = None
    
    if verbose >= 1:
        print(f"LOBPCG solver:")
        print(f"  Matrix size: {n}")
        print(f"  Number of roots: {nroots}")
        print(f"  Tolerance: {tol}")
        print(f"  Max iterations: {maxiter}")
        print(f"  Preconditioner: {'Yes' if M_op else 'No'}")
    
    # Run LOBPCG
    try:
        eigenvalues, eigenvectors = lobpcg(
            A_op,
            X0,
            M=M_op,
            tol=tol,
            maxiter=maxiter,
            largest=largest,
            verbosityLevel=max(0, verbose - 1)
        )
    except Exception as e:
        warnings.warn(f"LOBPCG failed: {e}. Falling back to eigsh.")
        return solve_eigsh(A, nroots, X0, diagonal, tol, maxiter, largest, verbose)
    
    # Sort eigenvalues
    if largest:
        idx = np.argsort(-eigenvalues)
    else:
        idx = np.argsort(eigenvalues)
    
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    if verbose >= 1:
        print(f"  Converged eigenvalues: {eigenvalues}")
    
    return eigenvalues, eigenvectors


def solve_eigsh(
    A: Union[np.ndarray, LinearOperator, Callable],
    nroots: int = 1,
    X0: Optional[np.ndarray] = None,
    diagonal: Optional[np.ndarray] = None,
    tol: float = 1e-8,
    maxiter: int = 200,
    largest: bool = False,
    verbose: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find eigenvalues using eigsh (Implicitly Restarted Lanczos Method).
    
    eigsh is robust and reliable for finding a few eigenvalues of large
    symmetric matrices.
    
    Parameters
    ----------
    A : array, LinearOperator, or callable
        The matrix to diagonalize
    nroots : int
        Number of eigenvalues/eigenvectors to compute
    X0 : array, optional
        Initial guess vector (n,). Only the first column is used.
    diagonal : array, optional
        Not used by eigsh, included for API compatibility
    tol : float
        Convergence tolerance
    maxiter : int
        Maximum number of iterations
    largest : bool
        If True, find largest eigenvalues. If False, find smallest.
    verbose : int
        Verbosity level
        
    Returns
    -------
    eigenvalues : array
        Computed eigenvalues (sorted)
    eigenvectors : array
        Computed eigenvectors (columns)
    """
    # Convert to LinearOperator if needed
    if callable(A) and not isinstance(A, LinearOperator):
        if X0 is not None:
            n = X0.shape[0]
        elif diagonal is not None:
            n = len(diagonal)
        else:
            for test_size in [100, 500, 1000, 5000, 10000]:
                try:
                    result = A(np.ones(test_size))
                    n = len(result)
                    break
                except:
                    continue
            else:
                raise ValueError("Cannot determine matrix size.")
        
        A_op = create_linear_operator(A, n)
    elif isinstance(A, np.ndarray):
        n = A.shape[0]
        A_op = A  # eigsh can use dense arrays directly
    elif issparse(A):
        n = A.shape[0]
        A_op = A
    else:
        A_op = A
        n = A.shape[0]
    
    # Initial vector for eigsh (only uses one vector)
    if X0 is not None:
        v0 = X0[:, 0] if X0.ndim > 1 else X0
    else:
        v0 = None
    
    which = 'LA' if largest else 'SA'  # LA=Largest Algebraic, SA=Smallest Algebraic
    
    # eigsh cannot request k >= N eigenvalues from a LinearOperator
    # For LinearOperators, cap to N-1 to avoid scipy's internal eigh fallback
    if isinstance(A_op, LinearOperator) and nroots >= n:
        nroots_actual = n - 1
        if verbose >= 1:
            print(f"eigsh solver:")
            print(f"  Matrix size: {n}")
            print(f"  WARNING: Requested {nroots} eigenvalues but eigsh cannot handle k >= N for LinearOperator.")
            print(f"           Reducing to {nroots_actual} eigenvalues (use lobpcg for k == N).")
            print(f"  Tolerance: {tol}")
            print(f"  Max iterations: {maxiter}")
    else:
        nroots_actual = nroots
        if verbose >= 1:
            print(f"eigsh solver:")
            print(f"  Matrix size: {n}")
            print(f"  Number of roots: {nroots}")
            print(f"  Tolerance: {tol}")
            print(f"  Max iterations: {maxiter}")
    
    # Run eigsh
    eigenvalues, eigenvectors = eigsh(
        A_op,
        k=nroots_actual,
        which=which,
        tol=tol,
        maxiter=maxiter,
        v0=v0
    )
    
    # Sort eigenvalues
    if largest:
        idx = np.argsort(-eigenvalues)
    else:
        idx = np.argsort(eigenvalues)
    
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    if verbose >= 1:
        print(f"  Converged eigenvalues: {eigenvalues}")
    
    return eigenvalues, eigenvectors


def _orthonormalize_against(basis: np.ndarray, new: np.ndarray,
                            thresh: float = 1e-8) -> np.ndarray:
    """Columns of ``new`` orthonormalized against ``basis`` and each other.

    Uses two-pass (iterated) Gram-Schmidt for numerical stability and drops
    columns that become linearly dependent (norm below ``thresh``). ``basis``
    is assumed to already have orthonormal columns; it may be empty.
    """
    n = new.shape[0]
    kept: list = []
    for j in range(new.shape[1]):
        t = np.array(new[:, j], dtype=float, copy=True)
        for _ in range(2):  # iterated MGS: one pass is not enough near-dependence
            if basis is not None and basis.shape[1]:
                t -= basis @ (basis.T @ t)
            for u in kept:
                t -= u * (u @ t)
        nrm = np.linalg.norm(t)
        if nrm > thresh:
            kept.append(t / nrm)
    if kept:
        return np.column_stack(kept)
    return np.zeros((n, 0), dtype=float)


def solve_davidson(
    A: Union[np.ndarray, LinearOperator, Callable],
    nroots: int = 1,
    X0: Optional[np.ndarray] = None,
    diagonal: Optional[np.ndarray] = None,
    tol: float = 1e-8,
    maxiter: int = 100,
    largest: bool = False,
    max_subspace: Optional[int] = None,
    verbose: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Block Davidson-Liu solver for a few extreme eigenpairs of a symmetric op.

    Purpose-built for Casida/TDDFT: it expands the subspace one **block** per
    iteration (all unconverged roots at once) so a matrix-free backend does a
    single batched matvec per iteration — critical for the GTO
    ``apply_K_matmat`` (one ``gen_response`` per step) and the plane-wave FFT
    path — instead of one matvec per vector as a naive Davidson would.

    Robustness features versus the scipy LOBPCG default:

    * diagonal (Davidson) preconditioner with a safe denominator, matched to
      the operator (``dE`` for TDA ``A``, ``dE**2`` for the RPA ``C``);
    * thick restart: when the subspace hits ``max_subspace`` it collapses to the
      current Ritz vectors (+ unconverged directions), bounding memory and the
      Rayleigh-Ritz cost while preserving convergence;
    * iterated Gram-Schmidt with linear-dependence dropping, so a stagnating
      correction never corrupts the basis;
    * a bullet-proof dense fallback for small problems (``n`` small or
      ``nroots`` close to ``n``), where iterative schemes are fragile.

    Only the ``matvec`` / ``matmat`` interface of ``A`` is used, so the solver
    stays basis-agnostic (shared by the GTO and plane-wave kernels).

    Parameters mirror :func:`solve_lobpcg`. ``largest`` finds the algebraically
    largest eigenvalues (the operator is negated internally).
    """
    # --- Resolve operator and size ---
    if isinstance(A, np.ndarray):
        n = A.shape[0]
        matmat = lambda V: A @ V
    elif isinstance(A, LinearOperator):
        n = A.shape[0]
        matmat = A.matmat
    elif callable(A):
        if X0 is not None:
            n = X0.shape[0]
        elif diagonal is not None:
            n = len(diagonal)
        else:
            raise ValueError("Cannot determine matrix size; provide X0 or diagonal.")
        A_op = create_linear_operator(A, n)
        matmat = A_op.matmat
    else:
        A_op = A
        n = A.shape[0]
        matmat = A_op.matmat

    sign = -1.0 if largest else 1.0

    def apply(V):
        # Always feed a 2-D block so backends hit their batched path.
        W = np.asarray(matmat(np.asarray(V, dtype=float)))
        if W.ndim == 1:
            W = W.reshape(-1, 1)
        return sign * W

    nroots = int(nroots)
    if nroots < 1:
        raise ValueError("nroots must be >= 1")

    # --- Dense fallback: small n or nroots close to n (iterative is fragile) ---
    if n <= max(2 * nroots + 4, 32) or nroots >= n - 1:
        Adense = apply(np.eye(n))
        Adense = 0.5 * (Adense + Adense.T)
        w, V = np.linalg.eigh(Adense)
        w = sign * w
        idx = np.argsort(w)
        w, V = w[idx][:nroots], V[:, idx][:, :nroots]
        if verbose >= 1:
            print(f"Davidson: dense fallback (n={n}, nroots={nroots})")
        return w, V

    if max_subspace is None:
        max_subspace = min(n, max(6 * nroots, nroots + 30))
    max_subspace = min(int(max_subspace), n)

    # --- Preconditioner diagonal (for A - theta I) ---
    if diagonal is not None:
        diag = sign * np.asarray(diagonal, dtype=float).ravel()
    else:
        diag = None

    # --- Initial guess block ---
    if X0 is not None:
        V0 = np.asarray(X0, dtype=float)
        if V0.ndim == 1:
            V0 = V0.reshape(-1, 1)
    elif diag is not None:
        order = np.argsort(diag)[:nroots]
        V0 = np.zeros((n, nroots))
        V0[order, np.arange(nroots)] = 1.0
    else:
        rng = np.random.default_rng(0)
        V0 = rng.standard_normal((n, nroots))
    # Guarantee at least nroots starting directions.
    if V0.shape[1] < nroots:
        pad = np.zeros((n, nroots - V0.shape[1]))
        order = np.argsort(diag)[:nroots] if diag is not None else np.arange(nroots)
        for c, r in enumerate(order[V0.shape[1]:]):
            pad[r, c] = 1.0
        V0 = np.hstack([V0, pad])

    V = _orthonormalize_against(np.zeros((n, 0)), V0)
    W = apply(V)  # A @ V

    if verbose >= 1:
        print("Davidson block solver:")
        print(f"  Matrix size: {n},  roots: {nroots},  max subspace: {max_subspace}")
        print(f"  tol: {tol},  maxiter: {maxiter},  preconditioner: "
              f"{'diagonal' if diag is not None else 'none'}")

    theta = np.zeros(nroots)
    X = V[:, :nroots].copy()
    for it in range(1, maxiter + 1):
        # Rayleigh-Ritz in the current subspace.
        H = V.T @ W
        H = 0.5 * (H + H.T)
        evals, evecs = np.linalg.eigh(H)
        theta = evals[:nroots]
        S = evecs[:, :nroots]
        X = V @ S            # Ritz vectors
        AX = W @ S           # A @ Ritz vectors
        R = AX - X * theta   # residuals (n, nroots)
        rnorm = np.linalg.norm(R, axis=0)
        n_conv = int(np.sum(rnorm < tol))
        if verbose >= 2:
            print(f"  iter {it:3d}: subspace={V.shape[1]:4d}  converged={n_conv}/{nroots}"
                  f"  max|r|={rnorm.max():.2e}")
        if n_conv == nroots:
            if verbose >= 1:
                print(f"  converged in {it} iterations (max|r|={rnorm.max():.2e})")
            break

        # Preconditioned correction vectors for the unconverged roots only.
        unconv = np.where(rnorm >= tol)[0]
        T = np.empty((n, len(unconv)))
        for c, j in enumerate(unconv):
            if diag is not None:
                d = diag - theta[j]
                d = np.where(np.abs(d) < 1e-8, np.sign(d) * 1e-8 + 1e-30, d)
                T[:, c] = R[:, j] / d
            else:
                T[:, c] = R[:, j]

        # Thick restart before the subspace overflows.
        if V.shape[1] + len(unconv) > max_subspace:
            V = _orthonormalize_against(np.zeros((n, 0)), X)
            W = apply(V)

        Tn = _orthonormalize_against(V, T)
        if Tn.shape[1] == 0:
            # No new directions (stagnation). Try a random restart once; else stop.
            if verbose >= 1:
                warnings.warn(
                    f"Davidson stagnated at iter {it} (max|r|={rnorm.max():.2e}); "
                    "returning best Ritz estimates."
                )
            break
        V = np.hstack([V, Tn])
        W = np.hstack([W, apply(Tn)])
    else:
        if verbose >= 1:
            warnings.warn(
                f"Davidson did not fully converge in {maxiter} iterations "
                f"(max|r|={rnorm.max():.2e})."
            )

    order = np.argsort(theta)
    theta = theta[order]
    X = X[:, order]
    if verbose >= 1:
        print(f"  eigenvalues: {theta}")
    return theta, X


def davidson(
    matrix: Union[np.ndarray, LinearOperator, Callable],
    nroots: int = 1,
    guess_vectors: Optional[np.ndarray] = None,
    diagonal: Optional[np.ndarray] = None,
    maxiter: int = 200,
    tol: float = 1e-8,
    maxspace: Optional[int] = None,
    nguess: Optional[int] = None,
    precondition: bool = True,
    verbose: int = 1,
    nthreads: Optional[int] = None,
    use_mpi: bool = False,
    use_gpu: bool = False,
    method: str = 'lobpcg'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for iterative eigenvalue computation.
    
    This is a wrapper that calls either lobpcg or eigsh based on the method
    parameter. Maintains backward compatibility with the old Davidson interface.
    
    Parameters
    ----------
    matrix : array, LinearOperator, or callable
        The matrix to diagonalize
    nroots : int
        Number of eigenvalues/eigenvectors to compute
    guess_vectors : array, optional
        Initial guess vectors
    diagonal : array, optional
        Diagonal of matrix for preconditioning
    maxiter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    maxspace : int, optional
        Ignored (for backward compatibility)
    nguess : int, optional
        Ignored (for backward compatibility)
    precondition : bool
        Whether to use preconditioning (requires diagonal)
    verbose : int
        Verbosity level
    nthreads : int, optional
        Ignored (scipy handles threading internally)
    use_mpi : bool
        Ignored (for backward compatibility)
    use_gpu : bool
        Ignored (scipy doesn't support GPU)
    method : str
        'lobpcg' or 'eigsh'
        
    Returns
    -------
    eigenvalues : array
        Computed eigenvalues
    eigenvectors : array
        Computed eigenvectors
    """
    # Select solver
    if method.lower() == 'lobpcg':
        solver = solve_lobpcg
    elif method.lower() == 'eigsh':
        solver = solve_eigsh
    elif method.lower() == 'davidson':
        solver = solve_davidson
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'davidson', 'lobpcg', or 'eigsh'."
        )
    
    # Use diagonal for preconditioning only if requested
    diag = diagonal if precondition else None
    
    return solver(
        A=matrix,
        nroots=nroots,
        X0=guess_vectors,
        diagonal=diag,
        tol=tol,
        maxiter=maxiter,
        largest=False,
        verbose=verbose
    )


def davidson_gpu(
    matrix: Union[np.ndarray, LinearOperator, Callable],
    nroots: int = 1,
    guess_vectors: Optional[np.ndarray] = None,
    diagonal: Optional[np.ndarray] = None,
    maxiter: int = 200,
    tol: float = 1e-8,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper for backward compatibility. GPU not supported with scipy solvers.
    Falls back to CPU lobpcg.
    """
    warnings.warn("GPU not supported with scipy solvers. Using CPU lobpcg.")
    return davidson(
        matrix=matrix,
        nroots=nroots,
        guess_vectors=guess_vectors,
        diagonal=diagonal,
        maxiter=maxiter,
        tol=tol,
        method='lobpcg',
        **kwargs
    )


def davidson_mpi(
    matrix: Union[np.ndarray, LinearOperator, Callable],
    nroots: int = 1,
    guess_vectors: Optional[np.ndarray] = None,
    diagonal: Optional[np.ndarray] = None,
    maxiter: int = 200,
    tol: float = 1e-8,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper for backward compatibility. MPI not directly supported.
    Use MPI in your matvec function instead.
    """
    return davidson(
        matrix=matrix,
        nroots=nroots,
        guess_vectors=guess_vectors,
        diagonal=diagonal,
        maxiter=maxiter,
        tol=tol,
        method='lobpcg',
        **kwargs
    )


if __name__ == "__main__":
    """Test the eigensolvers."""
    from scipy.sparse import random as sparse_random, diags as sparse_diags
    import time
    
    print("=" * 70)
    print("Iterative Eigensolver Tests (lobpcg / eigsh)")
    print("=" * 70)
    
    # Test 1: Dense matrix
    print("\n" + "=" * 70)
    print("Test 1: Dense Matrix")
    print("-" * 70)
    
    n = 500
    np.random.seed(42)
    H = np.random.rand(n, n)
    H = H + H.T  # Make symmetric
    H += np.diag(np.arange(n) * 0.1)  # Add diagonal dominance
    
    # Exact eigenvalues
    evals_exact = np.linalg.eigvalsh(H)[:5]
    
    # Test lobpcg
    t0 = time.time()
    evals_lobpcg, evecs_lobpcg = solve_lobpcg(H, nroots=5, diagonal=np.diag(H), verbose=1)
    t_lobpcg = time.time() - t0
    print(f"  LOBPCG time: {t_lobpcg:.3f}s")
    print(f"  LOBPCG eigenvalues: {evals_lobpcg}")
    print(f"  Exact eigenvalues:  {evals_exact}")
    print(f"  Error: {np.max(np.abs(evals_lobpcg - evals_exact)):.2e}")
    
    # Test eigsh
    t0 = time.time()
    evals_eigsh, evecs_eigsh = solve_eigsh(H, nroots=5, verbose=1)
    t_eigsh = time.time() - t0
    print(f"  eigsh time: {t_eigsh:.3f}s")
    print(f"  eigsh eigenvalues: {evals_eigsh}")
    print(f"  Error: {np.max(np.abs(evals_eigsh - evals_exact)):.2e}")
    
    # Test 2: Sparse matrix
    print("\n" + "=" * 70)
    print("Test 2: Sparse Matrix")
    print("-" * 70)
    
    n = 2000
    np.random.seed(42)
    H_sparse = sparse_random(n, n, density=0.01, format='csr', random_state=42)
    H_sparse = H_sparse + H_sparse.T
    H_sparse += sparse_diags([np.arange(n) * 0.2], [0], shape=(n, n))
    
    diag = np.asarray(H_sparse.diagonal()).flatten()
    
    t0 = time.time()
    evals, evecs = solve_lobpcg(H_sparse, nroots=5, diagonal=diag, verbose=1)
    print(f"  Time: {time.time() - t0:.3f}s")
    print(f"  Eigenvalues: {evals}")
    
    # Test 3: Matrix-free (callable)
    print("\n" + "=" * 70)
    print("Test 3: Matrix-free (callable)")
    print("-" * 70)
    
    n = 1000
    np.random.seed(42)
    H = np.random.rand(n, n)
    H = H + H.T
    H += np.diag(np.arange(n) * 0.1)
    
    def matvec(v):
        return H @ v
    
    t0 = time.time()
    evals, evecs = davidson(matvec, nroots=5, diagonal=np.diag(H), verbose=1)
    print(f"  Time: {time.time() - t0:.3f}s")
    
    evals_exact = np.linalg.eigvalsh(H)[:5]
    print(f"  Computed: {evals}")
    print(f"  Exact:    {evals_exact}")
    print(f"  Error: {np.max(np.abs(evals - evals_exact)):.2e}")
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
