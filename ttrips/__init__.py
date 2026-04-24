"""
trips — Tensor Regularization Iterative Projection Solvers

Package layout
--------------
trips.tensor.ops         : Core t-product algebra (t_prod2, t_transpose, t_normalize, tQR)
trips.tensor.build       : Matrix/tensor construction helpers (circulant, bcirc, build_tensor_A, ...)
trips.tensor.derivatives : Finite-difference / regularization operators for 1-D and 2-D problems
trips.solvers.gkb        : Golub–Kahan–Lanczos bidiagonalization (matrix and tensor variants)
trips.solvers.rmmgks     : Tensor RMMGKS solver with restarting and MM weighting
"""

from ttrips.tensors import ops, build, derivatives
from ttrips.solvers import gkb, rmmgks, trmmgks
