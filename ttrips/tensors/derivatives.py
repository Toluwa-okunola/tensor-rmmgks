"""
trips.tensor.derivatives
------------------------
Finite-difference operators used as regularization operators in inverse
problems.  All operators enforce periodic (circular) boundary conditions so
that the resulting matrices are circulant and compatible with the FFT-based
t-product convolution framework.

Functions
---------
gen_first_derivative_circular            : 1-D forward-difference operator (periodic BC).
gen_first_derivative_operator_2D_circular: 2-D gradient operator (stacked x and y
                                           differences) as a sparse matrix.
gen_first_derivative_operator_circ       : Alias / alternative 1-D forward-difference
                                           construction returning a CSR sparse matrix.
Lx_tensor                                : 3-D tensor encoding the x-direction
                                           difference operator for use with t_prod2.
Ly_tensor                                : 3-D tensor encoding the y-direction
                                           difference operator for use with t_prod2.
t_vstack                                 : Stack two third-order tensors vertically
                                           (along the first axis).
"""

import numpy as np
import scipy.sparse as sparse
import scipy


# ---------------------------------------------------------------------------
# 1-D circular (periodic) first-derivative operators
# ---------------------------------------------------------------------------

def gen_first_derivative_circular(n: int):
    """
    Build the n×n forward-difference matrix with wrap-around (periodic) BC.

    Entry (i, i) = -1 and (i, (i+1) % n) = +1, so the operator computes
    D[i] = x[i+1] - x[i] with x[n] identified with x[0].

    Parameters
    ----------
    n : int — number of grid points.

    Returns
    -------
    D : sparse CSR matrix of shape (n, n).
    """
    e = np.ones(n)
    D = scipy.sparse.diags([-e, e], [0, 1], shape=(n, n)).tolil()
    D[-1, 0] = 1      # wrap-around: last row connects back to column 0
    D[-1, -1] = -1    # restore the diagonal entry that was overwritten
    return D.tocsr()


def gen_first_derivative_operator_circ(n: int):
    """
    Alternative construction of the 1-D circular forward-difference operator.

    Builds the same operator as gen_first_derivative_circular but via
    off-diagonal construction with an explicit wrap-around correction.

    Parameters
    ----------
    n : int

    Returns
    -------
    Lx : sparse CSR matrix of shape (n, n).
    """
    D = scipy.sparse.diags(diagonals=np.ones(n - 1), offsets=1,
                           shape=(n, n)).tocsr()
    L = -(sparse.identity(n) - D).tocsr()
    L = L.tolil()
    L[-1, 0]  =  1
    L[-1, -1] = -1
    return L.tocsr()


# ---------------------------------------------------------------------------
# 2-D circular first-derivative operators (sparse matrix form)
# ---------------------------------------------------------------------------

def gen_first_derivative_operator_2D_circular(nx: int, ny: int):
    """
    Build the 2-D gradient operator with periodic boundary conditions.

    The result is a (2·nx·ny) × (nx·ny) sparse matrix formed by stacking:
      - IDx = D_x ⊗ I_y   : differences along the x-direction
      - DyI = I_x ⊗ D_y   : differences along the y-direction

    where D_x and D_y are the 1-D circular forward-difference matrices of
    sizes nx and ny respectively.

    Parameters
    ----------
    nx, ny : int — number of grid points in each direction.

    Returns
    -------
    L : sparse CSR matrix of shape (2*nx*ny, nx*ny).
    """
    D_x = gen_first_derivative_circular(nx)
    D_y = gen_first_derivative_circular(ny)
    IDx = sparse.kron(D_x, sparse.identity(ny))
    DyI = sparse.kron(sparse.identity(nx), D_y)
    return sparse.vstack((IDx, DyI))


# ---------------------------------------------------------------------------
# Tensor-form derivative operators (for use with t_prod2)
# ---------------------------------------------------------------------------

def Lx_tensor(nx: int, ny: int) -> np.ndarray:
    """
    Build the x-direction difference operator as a third-order tensor.

    The returned tensor L ∈ R^{ny×ny×nx} encodes the 1-D circular
    forward-difference matrix D_x in its first frontal slice; all other
    slices are zero.  When contracted via t_prod2, this applies D_x to the
    x-direction of an image stored as a lateral-slice tensor.

    Parameters
    ----------
    nx, ny : int

    Returns
    -------
    L : ndarray of shape (ny, ny, nx)
    """
    D_x = gen_first_derivative_operator_circ(ny)
    L = np.zeros((ny, ny, nx))
    L[:, :, 0] = D_x.toarray()
    return L


def Ly_tensor(nx: int, ny: int) -> np.ndarray:
    """
    Build the y-direction difference operator as a third-order tensor.

    The returned tensor L ∈ R^{ny×ny×nx} encodes a forward difference in the
    y-direction (i.e., between frontal slices).  Slice 0 is -I and slice -1
    is +I, implementing  x[..., k+1] - x[..., k]  with periodic wrap-around.

    Parameters
    ----------
    nx, ny : int

    Returns
    -------
    L : ndarray of shape (ny, ny, nx)
    """
    I = sparse.identity(ny).toarray()
    L = np.zeros((ny, ny, nx))
    L[:, :, 0]  = -I
    L[:, :, -1] =  I
    return L


# ---------------------------------------------------------------------------
# Tensor stacking
# ---------------------------------------------------------------------------

def t_vstack(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Stack two third-order tensors vertically along the first (row) axis.

    Analogous to numpy.vstack but for tensors: the result C satisfies
    C[:m1, :, :] = A and C[m1:, :, :] = B.

    Parameters
    ----------
    A : ndarray of shape (m1, n, p)
    B : ndarray of shape (m2, n, p)

    Returns
    -------
    C : ndarray of shape (m1+m2, n, p)

    Raises
    ------
    AssertionError if the second and third dimensions do not match.
    """
    assert A.shape[1] == B.shape[1] and A.shape[2] == B.shape[2], (
        "t_vstack: tensors must agree on axes 1 and 2."
    )
    m1, n, p = A.shape
    m2, _, _ = B.shape
    C = np.zeros((m1 + m2, n, p))
    C[:m1] = A
    C[m1:] = B
    return C
