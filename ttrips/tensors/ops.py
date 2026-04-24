"""
trips.tensor.ops
----------------
Core linear-algebraic operations for the t-product framework.

All tensors follow the (l, m, n) convention:
  - n  : number of frontal slices (the "tube" dimension, transformed by FFT)
  - l  : number of rows in each frontal slice
  - m  : number of columns in each frontal slice

The t-product A * B between A ∈ R^{l×m×n} and B ∈ R^{m×p×n} is defined by
taking the FFT along the third axis, multiplying corresponding frontal slices
as ordinary matrices, then inverting the FFT.

References
----------
Kilmer, M. E. & Martin, C. D. (2011). Factorization strategies for third-order tensors.
Reichel, L. & Ugwu, U. O. (2022). Tensor Krylov subspace methods with an application to
    image processing. ETNA.
"""

import numpy as np
import scipy.linalg as la


# ---------------------------------------------------------------------------
# t-product and transpose
# ---------------------------------------------------------------------------

def t_prod2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute the t-product C = A * B for third-order tensors.

    Parameters
    ----------
    A : ndarray of shape (l, m, n)
    B : ndarray of shape (m, p, n)

    Returns
    -------
    C : ndarray of shape (l, p, n)
        Real part of the inverse FFT of the face-wise matrix products in the
        Fourier domain.

    Notes
    -----
    The FFT is applied along axis=2 (the tube dimension).  Each frontal slice
    A_hat[:, :, i] is multiplied with B_hat[:, :, i] as ordinary matrices.
    """
    A_hat = np.fft.fft(A, axis=2)
    B_hat = np.fft.fft(B, axis=2)
    l, m, n = A.shape
    m2, p, n2 = B.shape
    assert m == m2 and n == n2, (
        f"Dimension mismatch: A is ({l},{m},{n}), B is ({m2},{p},{n2})"
    )
    C_hat = np.zeros((l, p, n), dtype=complex)
    for i in range(n):
        C_hat[:, :, i] = A_hat[:, :, i] @ B_hat[:, :, i]
    return np.fft.ifft(C_hat, axis=2).real


def t_transpose(A: np.ndarray) -> np.ndarray:
    """
    Compute the tensor transpose A^T for A ∈ R^{l×m×n}.

    The tensor transpose transposes every frontal slice and then reverses the
    order of slices 1..n-1 (leaving slice 0 in place), so that the t-product
    identity (A * B)^T = B^T * A^T holds.

    Parameters
    ----------
    A : ndarray of shape (l, m, n)

    Returns
    -------
    AT : ndarray of shape (m, l, n)
    """
    AT = np.transpose(A, (1, 0, 2)).copy()
    AT[:, :, 1:] = AT[:, :, -1:0:-1].copy()
    return AT


# ---------------------------------------------------------------------------
# Normalization (Algorithm 1, Reichel & Ugwu 2022)
# ---------------------------------------------------------------------------

def t_normalize(v: np.ndarray, tol: float = 1e-12):
    """
    Normalize a lateral slice v ∈ R^{m×1×n} with respect to the t-product norm.

    Implements Algorithm 1 of Reichel & Ugwu (2022): normalization is
    performed face-by-face in the Fourier domain, so the returned v_hat
    satisfies  v = v_hat * a  in the t-product sense and  ||v_hat||_F = 1.

    Parameters
    ----------
    v   : ndarray of shape (m, 1, n)
    tol : float, optional
        Faces whose Fourier-domain norm is below this threshold are replaced
        with a random unit vector and their corresponding scalar is set to 0.

    Returns
    -------
    vn : ndarray of shape (m, 1, n)
        Normalized lateral slice.
    a  : ndarray of shape (1, 1, n)
        Tubal scalar such that  v ≈ vn * a  (t-product).
    """
    m, _, n = v.shape
    v_hat  = np.fft.fft(v, axis=2)
    a_hat  = np.zeros((1, 1, n), dtype=complex)
    vn_hat = np.zeros_like(v_hat)

    for j in range(n):
        face = v_hat[:, 0, j]
        nrm  = la.norm(face)
        if nrm > tol:
            a_hat[0, 0, j]  = nrm
            vn_hat[:, 0, j] = face / nrm
        else:
            # Replace near-zero face with a random unit vector (breakdown
            # safeguard; sets corresponding scalar entry to 0).
            r = np.random.randn(m) + 1j * np.random.randn(m)
            vn_hat[:, 0, j] = r / la.norm(r)
            a_hat[0, 0, j]  = 0.0

    vn = np.fft.ifft(vn_hat, axis=2).real
    a  = np.fft.ifft(a_hat,  axis=2).real
    return vn, a


# ---------------------------------------------------------------------------
# tQR decomposition (Algorithm 2, Reichel & Ugwu 2022)
# ---------------------------------------------------------------------------

def tQR(A: np.ndarray):
    """
    t-product QR decomposition  A = Q * R  for A ∈ R^{l×m×n},  l ≥ m.

    Implements Algorithm 2 of Reichel & Ugwu (2022): an independent economy
    QR factorization is applied to each frontal slice of FFT(A), and the
    result is transformed back via IFFT.

    Parameters
    ----------
    A : ndarray of shape (l, m, n),  l ≥ m

    Returns
    -------
    Q : ndarray of shape (l, m, n)  — f-orthogonal (Q^T * Q = I in t-product sense)
    R : ndarray of shape (m, m, n)  — f-upper triangular
    """
    l, m, n = A.shape
    A_hat = np.fft.fft(A, axis=2)
    Q_hat = np.zeros_like(A_hat)
    R_hat = np.zeros((m, m, n), dtype=complex)

    for i in range(n):
        Q_hat[:, :, i], R_hat[:, :, i] = la.qr(A_hat[:, :, i], mode='economic')

    Q = np.fft.ifft(Q_hat, axis=2).real
    R = np.fft.ifft(R_hat, axis=2).real
    return Q, R

# ---------------------------------------------------------------------------
# tSVD 
# ---------------------------------------------------------------------------

def t_svd(A):
    D = np.fft.fft(A, axis=2)
    l, m, n = D.shape

    U = np.zeros((l, l, n), dtype=complex)
    S = np.zeros((l, m, n), dtype=complex)
    V = np.zeros((m, m, n), dtype=complex)

    for i in range(n):
        Ui, Si, Vhi = la.svd(D[:, :, i], full_matrices=True)
        U[:, :, i] = Ui
        S_slice = np.zeros((l, m), dtype=complex)
        np.fill_diagonal(S_slice, Si)
        S[:, :, i] = S_slice
        V[:, :, i] = Vhi          # scipy returns Vᵀ, store it directly

    U = np.fft.ifft(U, axis=2).real
    S = np.fft.ifft(S, axis=2).real
    V = np.fft.ifft(V, axis=2).real

    return U, S, V


# ---------------------------------------------------------------------------
# Tubal inner product
# ---------------------------------------------------------------------------

def tubal_inner(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute the tubal inner product  <u, v> = u^T * v  for lateral slices.

    The inner product is computed face-by-face in the Fourier domain as a
    conjugate dot product, then transformed back.

    Parameters
    ----------
    u, v : ndarray of shape (m, 1, n)

    Returns
    -------
    inner : ndarray of shape (1, 1, n)
        Tubal scalar satisfying  <u, v> = IFFT( conj(FFT(u)) · FFT(v) ).
    """
    u_hat = np.fft.fft(u, axis=2)
    v_hat = np.fft.fft(v, axis=2)
    inner_hat = np.sum(np.conj(u_hat) * v_hat, axis=0, keepdims=True)
    return np.fft.ifft(inner_hat, axis=2).real
