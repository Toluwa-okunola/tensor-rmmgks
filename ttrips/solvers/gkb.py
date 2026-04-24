"""
trips.solvers.gkb
-----------------
Golub–Kahan–Lanczos bidiagonalization for matrices and third-order tensors.

The Golub–Kahan process reduces a matrix (or tensor operator) A to a
bidiagonal form B = U^T A V and is the inner engine of many iterative
regularization methods (LSQR, LSMR, hybrid Krylov methods, RMMGKS).

Functions
---------
gkb   : Standard matrix GKB with full reorthogonalization and optional
        early stopping via the Discrepancy Principle.
t_gkb : Tensor GKB (Algorithm 3 of Reichel & Ugwu 2022) operating entirely
        in the t-product algebra.

References
----------
Golub, G. H. & Kahan, W. (1965). Calculating the singular values and
    pseudo-inverse of a matrix. SIAM J. Numer. Anal.
Reichel, L. & Ugwu, U. O. (2022). Tensor Krylov subspace methods with an
    application to image processing. ETNA.
"""

import numpy as np
from tqdm import tqdm

from ttrips.tensors.ops import t_prod2, t_transpose, t_normalize, tubal_inner


# ---------------------------------------------------------------------------
# Matrix GKB
# ---------------------------------------------------------------------------

def gkb(A, b: np.ndarray, n_iter: int,
        dp_stop: bool = False, **kwargs) -> tuple:
    """
    Golub–Kahan bidiagonalization with full reorthogonalization.

    Builds orthonormal bases U ∈ R^{rows × (n_iter+1)} and
    V ∈ R^{cols × n_iter} such that

        A V ≈ U S   and   A^T U ≈ V S^T

    where S is a lower-bidiagonal matrix.

    Parameters
    ----------
    A       : array-like or LinearOperator of shape (rows, cols)
              The forward operator.  Must support A @ x and A.T @ x.
    b       : 1-D or 2-D ndarray of shape (rows,) or (rows, 1)
              Right-hand side / data vector.
    n_iter  : int
              Maximum number of bidiagonalization steps.
    dp_stop : bool, optional (default False)
              If True, stop early when the Discrepancy Principle is satisfied:
                  ||A x_k - b|| ≤ eta * delta
              Requires kwargs 'gk_eta' (default 1.001) and 'gk_delta'
              (default 0.001, an estimate of the noise level).
    **kwargs :
        gk_eta   : float — safety factor for the Discrepancy Principle.
        gk_delta : float — noise-level estimate for the Discrepancy Principle.

    Returns
    -------
    U : ndarray of shape (rows, k+1) where k ≤ n_iter
    S : ndarray of shape (k+1, k)   — lower bidiagonal matrix
    V : ndarray of shape (cols, k)
    """
    eta   = kwargs.get('gk_eta',   1.001)
    delta = kwargs.get('gk_delta', 0.001)
    rows, cols = A.shape

    betas  = np.zeros(1)
    alphas = np.zeros(1)
    U = np.zeros((rows, n_iter + 1))
    V = np.zeros((cols, n_iter))
    U[:, 0] = (b / np.linalg.norm(b)).flatten()

    res_norm = np.inf

    for ii in tqdm(range(n_iter), desc='GKB: generating basis'):
        # ── check Discrepancy Principle ──────────────────────────────────
        if dp_stop and res_norm <= eta * delta:
            print("Discrepancy principle satisfied — stopping early.")
            U      = U[:, :ii + 1]
            V      = V[:, :ii]
            alphas = alphas[:ii]
            betas  = betas[:ii]
            break

        # ── right Lanczos step (update V) ────────────────────────────────
        v = A.T @ U[:, ii]
        if ii > 0:
            v -= betas[ii - 1] * V[:, ii - 1]
        # Full reorthogonalization against all previous V columns
        for jj in range(ii):
            v -= np.dot(V[:, jj], v) * V[:, jj]
        alphas[ii] = np.linalg.norm(v)
        V[:, ii]   = v / alphas[ii]

        # ── left Lanczos step (update U) ─────────────────────────────────
        u = A @ V[:, ii] - alphas[ii] * U[:, ii]
        # Full reorthogonalization against all previous U columns
        for jj in range(ii + 1):
            u -= np.dot(U[:, jj], u) * U[:, jj]
        betas[ii]      = np.linalg.norm(u)
        U[:, ii + 1]   = u / betas[ii]

        # ── estimate residual for Discrepancy Principle ──────────────────
        if dp_stop:
            S_k  = (np.pad(np.diag(alphas[:ii + 1]), ((0, 1), (0, 0)))
                    + np.pad(np.diag(betas[:ii + 1]),  ((1, 0), (0, 0))))
            bhat = U[:, :ii + 2].T @ b
            y    = np.linalg.lstsq(S_k, bhat, rcond=None)[0]
            x    = V[:, :ii + 1] @ y
            res_norm = np.linalg.norm(A @ x - b)

        if ii < n_iter - 1:
            alphas = np.append(alphas, 0.)
            betas  = np.append(betas,  0.)

    # Assemble lower bidiagonal S
    k = alphas.shape[0]
    S = np.zeros((k + 1, k))
    S[range(k), range(k)]     = alphas
    S[range(1, k + 1), range(k)] = betas
    return U, S, V


# ---------------------------------------------------------------------------
# Tensor GKB (Reichel & Ugwu 2022, Algorithm 3)
# ---------------------------------------------------------------------------

def t_gkb(A: np.ndarray, b: np.ndarray, k: int, tol: float = 1e-12) -> tuple:
    """
    Tensor Golub–Kahan bidiagonalization (t-product variant).

    Builds orthonormal tensor bases U ∈ R^{l×(k+1)×n} and
    V ∈ R^{m×k×n} together with a bidiagonal tensor
    S ∈ R^{(k+1)×k×n} such that (in the t-product sense)

        A * V ≈ U * S   and   A^T * U ≈ V * S^T.

    Parameters
    ----------
    A   : ndarray of shape (l, m, n) — tensor forward operator.
    b   : ndarray of shape (l, 1, n) — right-hand side lateral slice.
    k   : int — number of bidiagonalization steps.
    tol : float, optional — breakdown tolerance passed to t_normalize.

    Returns
    -------
    U : ndarray of shape (l, k+1, n)
    S : ndarray of shape (k+1, k, n)  — bidiagonal tensor
    V : ndarray of shape (m, k, n)

    Notes
    -----
    Reorthogonalization is not applied here.  For ill-conditioned problems
    consider adding explicit tubal reorthogonalization (commented blocks in
    the source indicate where it would go).
    """
    l, m, n = A.shape
    AT = t_transpose(A)

    # Initialise: normalize b to get the first left basis vector
    u1, z1 = t_normalize(b, tol=tol)
    if np.any(np.abs(np.fft.fft(z1, axis=2)) < tol):
        print("Warning: initial z1 has near-zero Fourier coefficients — "
              "breakdown risk.")

    U      = np.zeros((l, k + 1, n))
    V      = np.zeros((m, k,     n))
    U[:, 0, :] = u1.squeeze()

    # Tubal scalars stored as (1, k, n) tensors; grown incrementally
    betas  = np.zeros((1, 1, n))
    alphas = np.zeros((1, 1, n))

    for ii in range(k):
        # ── right step: v_i = A^T u_i - beta_{i-1} v_{i-1} ─────────────
        v = t_prod2(AT, np.expand_dims(U[:, ii], 1))
        if ii > 0:
            v -= t_prod2(
                np.expand_dims(V[:, ii - 1], 1),
                np.expand_dims(betas[:, ii - 1], 0),
            )
        v, alpha = t_normalize(v, tol=tol)
        V[:, ii, :]      = v.squeeze()
        alphas[:, ii, :] = alpha

        # ── left step: u_{i+1} = A v_i - alpha_i u_i ───────────────────
        u = (t_prod2(A, np.expand_dims(V[:, ii], 1))
             - t_prod2(np.expand_dims(U[:, ii], 1),
                       np.expand_dims(alphas[:, ii], 0)))
        u, beta = t_normalize(u, tol=tol)
        U[:, ii + 1, :] = u.squeeze()
        betas[:, ii, :] = beta

        if ii < k - 1:
            alphas = np.append(alphas, np.zeros((1, 1, n)), axis=1)
            betas  = np.append(betas,  np.zeros((1, 1, n)), axis=1)

    # Assemble lower-bidiagonal tensor S of shape (k+1, k, n)
    S = np.zeros((k + 1, k, n))
    S[range(alphas.shape[1]), range(alphas.shape[1])] = alphas
    S[range(1, alphas.shape[1] + 1), range(alphas.shape[1])] = betas
    return U, S, V
