"""
trips.solvers.rmmgks
--------------------
Tensor Restarted Majorization–Minimization Generalized Krylov Subspace (tRMMGKS)
solver for linear inverse problems with ℓp/ℓq regularization.

Solves the problem

    min_{x}  Φ_p(A ∗ x − b)  +  λ Φ_q(L ∗ x)

where Φ_p and Φ_q are (smoothed) ℓp and ℓq norms, A and L are third-order
tensor operators, and ∗ denotes the t-product.  The algorithm alternates
between:

  1. Majorization–Minimization (MM) weight updates that locally approximate
     the non-quadratic penalties with weighted ℓ2 terms.
  2. A projected Krylov step that expands the current search subspace by one
     direction derived from the weighted gradient.
  3. Subspace recycling that compresses the accumulated Krylov basis back to
     a prescribed maximum dimension.

Functions
---------
tRMMGKS                       : Main solver entry point.
smoothed_holder_weights_tensor : MM weight computation for the smoothed ℓp norm.
_estimate_lambda_tensor        : Regularization parameter selection (GCV or fixed).
_recycle_subspace_tensor       : SVD-based subspace compression and recycling.

References
----------
Lanza, A., Morigi, S., Reichel, L., & Sgallari, F. (2015). A generalized
    Krylov subspace method for ℓp-ℓq minimization. SIAM J. Sci. Comput.
Reichel, L. & Ugwu, U. O. (2022). Tensor Krylov subspace methods with an
    application to image processing. ETNA.
"""

import numpy as np
import scipy.linalg as la
from joblib import Parallel, delayed

from ttrips.tensors.ops import t_prod2, t_transpose, t_normalize, tQR, tubal_inner
from ttrips.solvers.gkb import t_gkb
from ttrips.solvers.regparam import *

# ---------------------------------------------------------------------------
# MM weight computation
# ---------------------------------------------------------------------------

def smoothed_holder_weights_tensor(u_tensor: np.ndarray,
                                   epsilon: float = 0.1,
                                   p: float = 1) -> np.ndarray:
    """
    Compute MM (majorization–minimization) weights for the smoothed ℓp norm.

    The smoothed Hölder norm penalty  Φ_p(u) = Σ (u_i² + ε²)^{p/2}  is
    locally majorized by a weighted ℓ2 term with weights

        w_i = (u_i² + ε²)^{p/2 − 1}.

    Parameters
    ----------
    u_tensor : ndarray of shape (s, 1, n)
        Residual or regularization-term values at the current iterate.
    epsilon  : float, optional (default 0.1)
        Smoothing parameter.  Larger values give a smoother (more ℓ2-like)
        penalty; smaller values approach the true ℓp norm.
    p        : float, optional (default 1)
        Exponent of the Hölder norm (p=1 → ℓ1, p=2 → ℓ2).

    Returns
    -------
    w : ndarray of shape (s, 1, n)  — entry-wise weights.
    """
    u_sq = u_tensor[:, 0, :]                        # (s, n)
    w_sq = (u_sq ** 2 + epsilon ** 2) ** (p / 2 - 1)
    return w_sq[:, np.newaxis, :]                    # (s, 1, n)


# ---------------------------------------------------------------------------
# Regularization parameter estimation
# ---------------------------------------------------------------------------

def _estimate_lambda_tensor(Q_A_hat: np.ndarray,
                             R_A_hat: np.ndarray,
                             R_L_hat: np.ndarray,
                             b_w_hat: np.ndarray,
                             regparam,
                             **kwargs) -> float:
    """
    Estimate a scalar regularization parameter λ for the reduced system.

    Currently supports:
      - A fixed scalar passed directly as `regparam`.
      - 'gcv' : Generalized Cross-Validation applied face-by-face in the
                Fourier domain; the median across faces is returned.
                (Placeholder: replace the inner loop body with a call to your
                generalized_crossvalidation_2 routine for a full GCV estimate.)

    Parameters
    ----------
    Q_A_hat : ndarray of shape (min(l,k), k, n) — Fourier-domain Q factor of
              the weighted fidelity basis.
    R_A_hat : ndarray of shape (min(l,k), k, n) — corresponding R factor.
    R_L_hat : ndarray of shape (min(s,k), k, n) — R factor of weighted reg basis.
    b_w_hat : ndarray of shape (l, 1, n)        — Fourier-domain weighted RHS.
    regparam : float or 'gcv'
        If a number, it is returned directly.  If 'gcv', GCV is performed.

    Returns
    -------
    lam : float — scalar regularization parameter.
    """
    if not isinstance(regparam, str):
        return float(regparam)

    n = R_A_hat.shape[2]
    lambdas = []
    for i in range(n):
        # TODO: replace with generalized_crossvalidation_2(Q_A_hat[:,:,i],
        #       R_A_hat[:,:,i], R_L_hat[:,:,i], b_w_hat[:,0,i])
        lambdas.append(1e-3)
    return float(np.median(lambdas))


# ---------------------------------------------------------------------------
# Subspace recycling
# ---------------------------------------------------------------------------

def _recycle_subspace_tensor(V_cols, AV_cols, LV_cols,
                              R_A_hat, R_L_hat,
                              lambdah, kmin, n,
                              A_tens, L_tens, x):
    """
    Compress the accumulated Krylov subspace via SVD-based recycling.

    Forms the stacked system [R_A; √λ R_L] and retains the `kmin-1` right
    singular vectors corresponding to the smallest singular values (the
    directions most relevant to the regularized problem).  The current
    solution x, projected orthogonal to the recycled basis, is appended as an
    additional direction to ensure the next iteration can still improve on the
    current solution.

    Parameters
    ----------
    V_cols   : list of lateral slices (m, 1, n) — current Krylov basis.
    AV_cols  : list of lateral slices (l, 1, n) — A applied to each basis vector.
    LV_cols  : list of lateral slices (s, 1, n) — L applied to each basis vector.
    R_A_hat  : ndarray (min(l,k), k, n)         — Fourier R factor for fidelity.
    R_L_hat  : ndarray (min(s,k), k, n)         — Fourier R factor for regularization.
    lambdah  : float                            — current regularization parameter.
    kmin     : int                              — target subspace dimension after recycling.
    n        : int                              — number of frontal slices.
    A_tens   : ndarray (l, m, n)
    L_tens   : ndarray (s, m, n)
    x        : ndarray (m, 1, n)                — current iterate.

    Returns
    -------
    new_V, new_AV, new_LV : lists of lateral slices for the recycled subspace.
    """
    k_cur = len(V_cols)
    V_tens = np.concatenate(V_cols, axis=1)

    # Stack [R_A; sqrt(λ) R_L] and take right singular vectors face-by-face
    
    W_hat = np.zeros((k_cur, k_cur, n), dtype=complex)
    for i in range(n):
        M = np.vstack([R_A_hat[:, :, i],
                       np.sqrt(lambdah) * R_L_hat[:, :, i]])
        _, _, Vt = la.svd(M)
        W_hat[:, :, i] = Vt.conj().T          # right singular vectors as columns

    W = np.fft.ifft(W_hat, axis=2).real       # (k_cur, k_cur, n)
    # keep first kmin-1 columns
    W_keep = W[:, :kmin-1, :]                 # (k_cur, kmin-1, n)

    V_tilde = t_prod2(V_tens, W_keep)         # (m, kmin-1, n)  — recycled basis
    
    # orthonormalize via tQR
    V_tilde, _ = tQR(V_tilde)

    # rebuild column lists
    x_new =  x - t_prod2(V_tilde, t_prod2(t_transpose(V_tilde), x))
    x_new,_ = t_normalize(x_new, tol=1e-12)
    #V_tilde = np.concatenate([V_tilde[:, j:j+1, :] for j in range(kmin-1)] + [x_new], axis=1)
    #print("V_tilde shape:", V_tilde.shape)
    #V_tilde, _ = tQR(V_tilde) 
    
    new_V  = [V_tilde[:, j:j+1, :] for j in range(kmin-1)] + [x_new]
    new_AV = [t_prod2(A_tens, v)   for v in new_V]
    new_LV = [t_prod2(L_tens, v)   for v in new_V]

    return new_V, new_AV, new_LV


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def tRMMGKS(A_tens: np.ndarray,
            b_tens: np.ndarray,
            L_tens: np.ndarray,
            qnorm: float = 1,
            pnorm: float = 2,
            projection_dim: int = 3,
            n_iter: int = 10,
            epsilon: float = 0.1,
            regparam = 'gcv',
            x_true: np.ndarray = None,
            kmin: int = 3,
            l_max: int = 5,
            reorthogonalize: bool = True,
            tol: float = 1e-12,
            parallel: bool = False,
            **kwargs):
    """
    Tensor RMMGKS: Restarted MM Generalized Krylov Subspace solver.

    Minimizes the functional

        J(x) = ||W_f^{1/2} (A ∗ x − b)||² + λ ||W_r^{1/2} (L ∗ x)||²

    with MM-updated diagonal weight tensors W_f (fidelity) and W_r
    (regularization), corresponding to a smoothed ℓ^pnorm / ℓ^qnorm problem.

    Parameters
    ----------
    A_tens         : ndarray (l, m, n) — forward operator tensor.
    b_tens         : ndarray (l, 1, n) — right-hand side (observed data).
    L_tens         : ndarray (s, m, n) — regularization operator tensor
                     (e.g. first-difference tensor from trips.tensor.derivatives).
    qnorm          : float (default 1) — exponent for the regularization penalty.
    pnorm          : float (default 2) — exponent for the fidelity penalty.
    projection_dim : int   (default 3) — initial Krylov subspace size (t_gkb steps).
    n_iter         : int   (default 10) — number of outer MM / Krylov iterations.
    epsilon        : float (default 0.1) — smoothing parameter for the Hölder weights.
    regparam       : float or 'gcv' (default 'gcv') — regularization parameter or
                     selection strategy.  Pass a positive float to fix λ.
    x_true         : ndarray (m, 1, n) or None — ground-truth image for error tracking.
    kmin           : int (default 3) — subspace dimension after recycling.
    l_max          : int (default 5) — maximum subspace dimension before recycling.
    reorthogonalize: bool (default True) — reserved for future use.
    tol            : float (default 1e-12) — breakdown tolerance in t_normalize.
    parallel       : bool (default False) — if True, solve Fourier-slice systems in
                     parallel using joblib.
    **kwargs       : passed through to _estimate_lambda_tensor.

    Returns
    -------
    x    : ndarray of shape (m, 1, n) — reconstructed solution.
    info : dict with keys
        'xHistory'        : list of iterates (one per outer iteration).
        'regParam'        : final regularization parameter.
        'regParam_history': list of λ values at each iteration.
        'relError'        : list of relative errors ||x_k − x_true|| / ||x_true||
                            (only present when x_true is not None).
    """
    l, m, n = A_tens.shape
    AT = t_transpose(A_tens)
    LT = t_transpose(L_tens)

    # ── initialise subspace from tGKB ──────────────────────────────
    U_tens, B_bar , V_tens= t_gkb(A_tens, b_tens, projection_dim)
    # V_tens : (m, projection_dim, n) — initial Krylov basis
    # make sure columns are stored as a list of lateral slices for easy appending
    V_cols = [V_tens[:, j:j+1, :] for j in range(V_tens.shape[1])]

    # precompute AV and LV in terms of the current basis
    AV_cols = [t_prod2(A_tens, v) for v in V_cols]  # each (l,1,n)
    LV_cols = [t_prod2(L_tens, v) for v in V_cols]  # each (s,1,n)

    # initialise x̃ as A^T b (lateral slice, shape (m,1,n))
    x = t_prod2(AT, b_tens)
    x_orig = x.copy()

    # x = x_orig = V_tens[:, 0:1, :] 
    # x_orig = V_tens[:, 0:1, :] 

    # After computing x = t_prod2(AT, b_tens)
    # print("x_orig init norm:", la.norm(x_orig))
    # print("A*x_orig - b norm:", la.norm(t_prod2(A_tens, x_orig) - b_tens))
    # print("b norm:", la.norm(b_tens))

    x_history = []
    lambda_history = []
    lambdah = 0.0
    s = L_tens.shape[0]
    for ii in range(n_iter):
        # ── MM weights ──────────────────────────────────────────────
        # residual weights (fidelity)
        v_res  = t_prod2(A_tens, x_orig) - b_tens        # (l,1,n)
        wf     = smoothed_holder_weights_tensor(v_res, epsilon=epsilon, p=pnorm)
        # regularization weights  ← this is where ℓ1 lives
        u_reg  = t_prod2(L_tens, x_orig)                 # (s,1,n)
        wr     = smoothed_holder_weights_tensor(u_reg, epsilon=epsilon, p=qnorm)

        # ── gradient for new search direction ───────────────────────
        # weighted fidelity gradient: A^T [wf ⊙ (Ax - b)]
        ra = t_prod2(AT, wf * v_res)                     # (m,1,n)
        # weighted regularization gradient: L^T [wr ⊙ (Lx)]
        rb = t_prod2(LT, wr * u_reg)                     # (m,1,n)
        r  = ra + lambdah * rb

        # project out current subspace (double reorthogonalization)
        for vj in V_cols:
            r = r - t_prod2(vj, tubal_inner(vj, r))
        for vj in V_cols:
            r = r - t_prod2(vj, tubal_inner(vj, r))

        # normalize and append new direction
        vn, _ = t_normalize(r, tol=tol)
        V_cols.append(vn)
        AV_cols.append(t_prod2(A_tens, vn))
        LV_cols.append(t_prod2(L_tens, vn))

        # ── build weighted small system ──────────────────────────────
        # Stack basis into tensor, apply weights, QR in Fourier domain.
        # For each frontal slice i:
        #   AA_hat[:,:,i] = diag(wf[:,0,i]) @ AV_hat[:,:,i]
        #   LL_hat[:,:,i] = diag(wr[:,0,i]) @ LV_hat[:,:,i]
        # then QR each.

        #print(len(V_cols), len(AV_cols), len(LV_cols))

        k_cur = len(V_cols)
        AV_tens = np.concatenate(AV_cols, axis=1)   # (l, k_cur, n)
        LV_tens = np.concatenate(LV_cols, axis=1)   # (s, k_cur, n)

        # weight application: broadcast entry-wise
        # wf is (l,1,n), AV_tens is (l,k_cur,n)  → multiply along axis 0 & 2
        AV_w = AV_tens * wf          # (l, k_cur, n)  weighted fidelity basis
        LV_w = LV_tens  * wr       # (s, k_cur, n)  weighted reg basis

        # QR in Fourier domain (same structure as tQR but rectangular)
        AV_w_hat = np.fft.fft(AV_w, axis=2)
        LV_w_hat = np.fft.fft(LV_w, axis=2)  
        b_w_hat  = np.fft.fft(wf * b_tens, axis=2)   # (l,1,n) weighted rhs

        qc_A = min(l, k_cur)
        qc_L = min(s, k_cur)
        Q_A_hat = np.zeros((l,    min(k_cur,l),  n), dtype=complex)
        R_A_hat = np.zeros((min(k_cur,l), k_cur, n), dtype=complex)
        Q_L_hat = np.zeros((s,    min(k_cur,s),  n), dtype=complex)
        R_L_hat = np.zeros((min(k_cur,s), k_cur, n), dtype=complex)

        # Q_A_hat = np.zeros_like(AV_w_hat)
        # R_A_hat = np.zeros((qc_A, k_cur, n), dtype=complex)
        # Q_L_hat = np.zeros_like(LV_w_hat)
        # R_L_hat = np.zeros((k_cur, k_cur, n), dtype=complex)     
        #print(Q_A_hat.shape, R_A_hat.shape, Q_L_hat.shape, R_L_hat.shape)

        for i in range(n):
            #print(AV_w_hat[:, :, i].shape, LV_w_hat[:, :, i].shape, Q_A_hat[:, :, i].shape, R_A_hat[:, :, i].shape, Q_L_hat[:, :, i].shape, R_L_hat[:, :, i].shape)
            Q_A_hat[:, :, i], R_A_hat[:, :, i] = la.qr(AV_w_hat[:, :, i],
                                                         mode='economic')
            Q_L_hat[:, :, i], R_L_hat[:, :, i] = la.qr(LV_w_hat[:, :, i],
                                                         mode='economic')

        # ── regularization parameter (per-slice GCV or fixed) ───────
        # simplest working option: use a scalar lambda estimated face-by-face
        # and average, matching what your matrix code does via generalized_crossvalidation_2
        #lambdah = _estimate_lambda_tensor(Q_A_hat, R_A_hat, R_L_hat,
                                          #b_w_hat, regparam, **kwargs)
        #lambda_history.append(lambdah)

        # ── solve reduced system face-by-face ───────────────────────
        # min_{y} ||R_A y - Q_A^H (wf*b)||² + λ ||R_L y||²
        # → [R_A ; sqrt(λ) R_L] y = [Q_A^H (wf*b) ; 0]
        y_hat = np.zeros((k_cur, 1, n), dtype=complex)
        #b_proj_hat = np.einsum('ijn,jln->iln', t_transpose(np.conj(Q_A_hat)), b_w_hat)  #np.einsum('ijn,iln->jln', np.conj(Q_A_hat),
                               #b_w_hat)           # Q_A^H * (wf*b), shape (k,1,n)
        b_proj_hat = np.zeros((R_A_hat.shape[0], 1, n), dtype=complex)

        for i in range(n):
            b_proj_hat[:, 0, i] = Q_A_hat[:, :, i].conj().T @ b_w_hat[:, 0, i]
        V_tens_cur = np.concatenate(V_cols, axis=1)
        def solve_slice(i):
            RA = R_A_hat[:, :, i]
            RL = R_L_hat[:, :, i]
            bp = b_proj_hat[:, 0:1, i]
            x_true_hat = np.fft.fft(x_true, axis = 2)[:,:,i]
            V_hat = np.fft.fft(V_tens_cur, axis = 2)[:,:,i]
            if regparam == "gridsearch":
                lambdah = estimate_lambda(RA, RL, bp, regparam,x_true_hat,V_hat,**kwargs)
            else:
                lambdah = regparam
            lhs = np.vstack([R_A_hat[:, :, i], np.sqrt(lambdah) * R_L_hat[:, :, i]])
            rhs = np.vstack([b_proj_hat[:, 0:1, i], np.zeros((R_L_hat.shape[0], 1), dtype=complex)])
            return la.lstsq(lhs, rhs)[0], lambdah
        if parallel:
            results = Parallel(n_jobs=-1, prefer="threads")(delayed(solve_slice)(i) for i in range(n))
        else:
            results = [solve_slice(i) for i in range(n)]

        y_hat   = np.stack([r[0] for r in results], axis=-1)
        lambdas = [r[1] for r in results]
        #lambdah = float(np.mean(lambdas))
        lambda_history.append(lambdas)
        #lambda_history.append(lambdas)
        #y_hat = np.stack(results, axis=-1)  # → (rows, 1, n)
        #print("y_hat conj symmetric:", np.allclose(y_hat[:, :, 1:], np.conj(y_hat[:, :, -1:0:-1]), atol=1e-10))
        #print("max imaginary part of y after ifft:", np.max(np.abs(np.fft.ifft(y_hat, axis=2).imag)))
        y = np.fft.ifft(y_hat, axis=2).real       # (k_cur, 1, n)
        #print(y_hat,y)
        # ── reconstruct x̃ = V * y  (in t-product sense) ────────────
           # (m, k_cur, n)
        x = t_prod2(V_tens_cur, y)                    # (m, 1, n)
        x_orig = x.copy()
        x_history.append(x)

        # ── recycle / truncate subspace (mirrors your SVD recycling) ─
        if len(V_cols) >= l_max:
            V_cols, AV_cols, LV_cols = _recycle_subspace_tensor(
                V_cols, AV_cols, LV_cols, R_A_hat, R_L_hat,
                lambdah, kmin, n, A_tens, L_tens,x_orig)

    info = {
        'xHistory': x_history,
        'regParam': lambdah,
        'regParam_history': lambda_history,
    }
    if x_true is not None:
        info['relError'] = [la.norm(x - x_true) / la.norm(x_true)
                            for x in x_history]

    return x, info