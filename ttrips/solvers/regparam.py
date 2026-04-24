from joblib import Parallel, delayed
import numpy as np
import scipy.linalg as la
import scipy.optimize as op
from pylops import Identity, LinearOperator
import matplotlib.pyplot as plt

def estimate_lambda(R_A, R_L, b_proj, regparam,x_true,V, **kwargs):
    if not isinstance(regparam, str):
        return float(regparam)

    lam_min = kwargs.get('lambda_min', 1e-6)
    lam_max = kwargs.get('lambda_max', 1e2)
    n_lam   = kwargs.get('n_lambda',   40)
    grid    = np.logspace(np.log10(lam_min), np.log10(lam_max), n_lam)

    def score(lam):
        lhs = np.vstack([R_A, np.sqrt(lam) * R_L])
        rhs = np.vstack([b_proj, np.zeros((R_L.shape[0], 1), dtype=complex)])
        y, *_ = la.lstsq(lhs, rhs)
        x = V@y
        return np.linalg.norm(x - x_true) / np.linalg.norm(x_true) #la.norm(lhs@y - rhs)**2 #(np.linalg.norm(R_A @ y - b_proj)**2
                #+ lam * np.linalg.norm(R_L @ y)**2)

    scores = Parallel(n_jobs=-1, prefer="threads")(
        delayed(score)(lam) for lam in grid
    )
    print(scores)
    return float(grid[np.argmin(scores)])


def gcv_numerator_2(reg_param, Q_A, R_A, R_L, b,**kwargs):
    variant = kwargs['variant'] if ('variant' in kwargs) else 'standard'

    # the observation term:

    R_A_2 = R_A.T @ R_A

    R_A_2 = R_A_2.todense() if isinstance(R_A_2, LinearOperator) else R_A_2

    # The regularizer term:

    R_L_2 = (R_L.T @ R_L)
    
    R_L_2 = R_L_2.todense() if isinstance(R_L_2, LinearOperator) else R_L_2

    # the inverse term:

    inverted = np.linalg.lstsq(( R_A_2 + reg_param * R_L_2), (R_A.T @ Q_A.T @ b) ,rcond=None)[0]  # la.solve( ( R_A_2 + reg_param * R_L_2), (R_A.T @ Q_A.T @ b) )

    if variant == 'modified':
        return ((np.linalg.norm( R_A @ inverted - Q_A.T @ b ))**2 + np.linalg.norm(b - Q_A@(Q_A.T@b))**2)
    else:
        return (np.linalg.norm( R_A @ inverted - Q_A.T @ b ))**2

        # return np.sqrt((np.linalg.norm( R_A @ inverted - Q_A.T @ b ))**2 + np.linalg.norm(b - Q_A@(Q_A.T@b))**2)

def gcv_denominator_2(reg_param, R_A, R_L, b, **kwargs):

    variant = kwargs['variant'] if ('variant' in kwargs) else 'standard'
    # print(variant)
    # the observation term:

    R_A_2 = R_A.T @ R_A

    R_A_2 = R_A_2.todense() if isinstance(R_A_2, LinearOperator) else R_A_2

    # The regularizer term:

    R_L_2 = (R_L.T @ R_L)

    R_L_2 = R_L_2.todense() if isinstance(R_L_2, LinearOperator) else R_L_2

    inverted = np.linalg.lstsq(( R_A_2 + reg_param * R_L_2), R_A.T,rcond=None)[0]  #la.solve( ( R_A_2 + reg_param * R_L_2), R_A.T )

    if variant == 'modified':
       m = kwargs['fullsize']
       trace_term = (m - R_A.shape[1]) - np.trace(R_A @ inverted) # b.size - np.trace(R_A @ inverted) # this is defined with respect to the projected quantities 
    else:
        # in this way works even if we revert to the fully projected pb (call with Q_A.T@b)
        # trace_term = b.size - np.trace(R_A @ inverted) # this is defined with respect to the projected quantities
        trace_term = R_A.shape[0]- np.trace(R_A @ inverted)
    return trace_term**2

def generalized_crossvalidation_2(Q_A, R_A, R_L, b, **kwargs):

    if 'tol' in kwargs:
        tol = kwargs['tol']
    else:
        tol = 10**(-12)

    # function to minimize
    gcv_func = lambda reg_param: gcv_numerator_2(reg_param, Q_A, R_A, R_L, b) / gcv_denominator_2(reg_param, R_A, R_L, b, **kwargs)
    lambdah = op.fminbound(func = gcv_func, x1 = 1e-9, x2 = 100, args=(), xtol=1e-12, maxfun=1000, full_output=0, disp=0)
    
    return lambdah

def lcurve(R_A, R_L, b_proj, AV,LV,b, **kwargs):
    """
    L-curve parameter selection for the reduced system
        min_y ||R_A y - b_proj||² + λ ||R_L y||²

    Finds λ at maximum curvature of the log-log plot of
    ||R_A y - b_proj|| vs ||R_L y||.
    """
    lam_min = kwargs.get('lambda_min', 1e-9)
    lam_max = kwargs.get('lambda_max', 1e2)
    n_lam   = kwargs.get('n_lambda',   200)
    grid    = np.logspace(np.log10(lam_min), np.log10(lam_max), n_lam)

    rho = []   # log residual norms
    eta = []   # log regularization norms

    for lam in grid:
        lhs = np.vstack([R_A, np.sqrt(lam) * R_L])
        rhs = np.vstack([b_proj, np.zeros((R_L.shape[0], 1))])
        y, *_ = la.lstsq(lhs, rhs)
        rho.append(np.log(np.linalg.norm(AV @ y - b) + 1e-30))
        eta.append(np.log(np.linalg.norm(LV @ y) + 1e-30))

    rho = np.array(rho)
    eta = np.array(eta)

    # curvature of the parametric curve (rho(t), eta(t))
    # using finite differences
    drho  = np.gradient(rho)
    deta  = np.gradient(eta)
    d2rho = np.gradient(drho)
    d2eta = np.gradient(deta)

    curvature = np.abs(drho * d2eta - deta * d2rho) / (drho**2 + deta**2)**1.5

    best_idx  = np.argmax(curvature)
    best_lam  = float(grid[best_idx])

    if kwargs.get('plot', False):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(rho, eta, 'b-')
        axes[0].scatter(rho[best_idx], eta[best_idx], color='r', zorder=5,
                        label=f'λ = {best_lam:.2e}')
        axes[0].set_xlabel('log ||R_A y - b_proj||')
        axes[0].set_ylabel('log ||R_L y||')
        axes[0].set_title('L-curve')
        axes[0].legend()

        axes[1].semilogy(np.log10(grid), curvature)
        axes[1].axvline(np.log10(best_lam), color='r', linestyle='--',
                        label=f'λ = {best_lam:.2e}')
        axes[1].set_xlabel('log10(lambda)')
        axes[1].set_ylabel('curvature')
        axes[1].set_title('L-curve curvature')
        axes[1].legend()
        plt.tight_layout()
        plt.show()

    return best_lam