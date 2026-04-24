"""
trips.tensor.build
------------------
Utility functions for constructing structured matrices and third-order tensors
that commonly arise in image deblurring and other inverse problems.

Functions
---------
circulant            : Build a circulant matrix from a 1-D vector.
build_tensor_A       : Build the block-circulant-with-circulant-blocks (BCCB)
                       convolution tensor for a given PSF kernel and image size n.
build_circular_H     : Build the full N²×N² circular convolution matrix for a
                       2-D image given a PSF kernel (slower; use build_tensor_A
                       for the tensor representation instead).
tensorize            : Convert an (m, n) matrix to a (m, 1, n) lateral slice tensor.
matrixize            : Inverse of tensorize — collapse a (m, 1, n) tensor to (m, n).
unfold               : Mode-1 unfolding of a third-order tensor (n3·n1, n2).
bcirc               : Construct the block-circulant matrix associated with a tensor.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Circulant building blocks
# ---------------------------------------------------------------------------

def circulant(v: np.ndarray) -> np.ndarray:
    """
    Build an n×n circulant matrix whose first row is v.

    Each row i is a cyclic right-shift of v by i positions, so
    C[i, j] = v[(j - i) % n].

    Parameters
    ----------
    v : 1-D ndarray of length n

    Returns
    -------
    C : ndarray of shape (n, n)
    """
    n = len(v)
    C = np.zeros((n, n))
    for i in range(n):
        C[i] = np.roll(v, i)
    return C


def build_tensor_A(kernel: np.ndarray, n: int) -> np.ndarray:
    """
    Build the BCCB convolution tensor A ∈ R^{n×n×n} for a 2-D PSF kernel
    and a square image of side n (periodic / wrap-around boundary conditions).

    The PSF is zero-padded to size n×n and then centered so that the central
    pixel of the kernel maps to index (0, 0) mod n.  Each frontal slice
    A[:, :, k] is the circulant matrix corresponding to row k of the padded PSF.

    Parameters
    ----------
    kernel : ndarray of shape (kH, kW)
        Point-spread function (PSF).  Typically kH, kW ≪ n.
    n : int
        Side length of the (square) image and of the resulting tensor.

    Returns
    -------
    A : ndarray of shape (n, n, n)
    """
    Kpad = np.zeros((n, n))
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    for i in range(kh):
        for j in range(kw):
            ii = (i - pad_h) % n
            jj = (j - pad_w) % n
            Kpad[ii, jj] = kernel[i, j]

    A = np.zeros((n, n, n))
    for k in range(n):
        A[:, :, k] = circulant(Kpad[k, :])
    return A


def build_circular_H(kernel: np.ndarray, img_shape: tuple) -> np.ndarray:
    """
    Build the (m·n)×(m·n) circular convolution matrix H for a 2-D image.

    This constructs H explicitly as a dense matrix using row-major (C-order)
    flattening and wrap-around (periodic) boundary conditions.  For large
    images prefer the tensor representation from build_tensor_A together with
    the FFT-based t-product, which avoids forming H explicitly.

    Parameters
    ----------
    kernel    : ndarray of shape (kH, kW) — PSF kernel.
    img_shape : tuple (m, n)              — image height and width.

    Returns
    -------
    H : ndarray of shape (m*n, m*n)
    """
    m, n = img_shape
    N = m * n
    H = np.zeros((N, N))
    kH, kW = kernel.shape
    pad_h, pad_w = kH // 2, kW // 2

    def idx(i, j):
        return i * n + j

    for i in range(m):
        for j in range(n):
            row = idx(i, j)
            for ki in range(kH):
                for kj in range(kW):
                    ii = (i + ki - pad_h) % m
                    jj = (j + kj - pad_w) % n
                    H[row, idx(ii, jj)] += kernel[ki, kj]
    return H


# ---------------------------------------------------------------------------
# Tensor ↔ matrix conversions
# ---------------------------------------------------------------------------

def tensorize(X: np.ndarray) -> np.ndarray:
    """
    Convert a matrix X ∈ R^{m×n} to a lateral-slice tensor of shape (m, 1, n).

    Each column j of X becomes the single column of frontal slice j of the
    output tensor.  This is the natural embedding used when the columns of X
    represent the tube fibres of a lateral slice in the t-product framework.

    Parameters
    ----------
    X : ndarray of shape (m, n)

    Returns
    -------
    T : ndarray of shape (m, 1, n)
    """
    m, n = X.shape
    T = np.zeros((m, 1, n))
    for i in range(m):
        T[:, 0, i] = X[i, :]
    return T


def matrixize(T: np.ndarray) -> np.ndarray:
    """
    Inverse of tensorize: collapse a (m, 1, n) tensor to a (m, n) matrix.

    Parameters
    ----------
    T : ndarray of shape (m, 1, n)

    Returns
    -------
    X : ndarray of shape (m, n)
    """
    m, _, n = T.shape
    X = np.zeros((m, n))
    for i in range(m):
        X[i, :] = T[:, 0, i]
    return X


# ---------------------------------------------------------------------------
# Unfolding and block-circulant matrix
# ---------------------------------------------------------------------------

def unfold(X: np.ndarray) -> np.ndarray:
    """
    Mode-1 unfolding of a third-order tensor X ∈ R^{n1×n2×n3}.

    Arranges the tensor as a (n3·n1) × n2 matrix by stacking the n3 frontal
    slices row-wise after transposing the axis ordering to (n3, n1, n2).

    Parameters
    ----------
    X : ndarray of shape (n1, n2, n3)

    Returns
    -------
    Xu : ndarray of shape (n3*n1, n2)
    """
    n1, n2, n3 = X.shape
    return X.transpose(2, 0, 1).reshape(n3 * n1, n2)


def bcirc(X: np.ndarray) -> np.ndarray:
    """
    Construct the block-circulant matrix associated with tensor X ∈ R^{n1×n2×n3}.

    The (i, j) block of the result is X[:, :, (i-j) % n3], giving an
    (n1·n3) × (n2·n3) block matrix.  The t-product A * B equals the matrix
    product bcirc(A) @ unfold(B), reshaped appropriately.

    Parameters
    ----------
    X : ndarray of shape (n1, n2, n3)

    Returns
    -------
    BC : ndarray of shape (n1*n3, n2*n3)
    """
    n1, n2, n3 = X.shape
    blocks = [[None] * n3 for _ in range(n3)]
    for i in range(n3):
        for j in range(n3):
            blocks[i][j] = X[:, :, (i - j) % n3]
    return np.block(blocks)
