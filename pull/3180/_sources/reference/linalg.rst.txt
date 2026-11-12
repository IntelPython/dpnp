.. _routines.linalg:

.. currentmodule:: dpnp

Linear algebra (:mod:`dpnp.linalg`)
===================================

.. hint:: `NumPy API Reference: Linear algebra (numpy.linalg) <https://numpy.org/doc/stable/reference/routines.linalg.html>`_

Matrix and vector products
--------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dot
   linalg.multi_dot
   vdot
   vecdot
   linalg.vecdot (Array API compatible)
   inner
   outer
   linalg.outer
   matmul
   linalg.matmul (Array API compatible)
   matvec
   vecmat
   tensordot
   linalg.tensordot (Array API compatible)
   einsum
   einsum_path
   linalg.matrix_power
   kron
   linalg.cross (Array API compatible)

Decompositions
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   linalg.cholesky
   linalg.qr
   linalg.lu_factor
   linalg.svd
   linalg.svdvals

Matrix eigenvalues
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   linalg.eig
   linalg.eigh
   linalg.eigvals
   linalg.eigvalsh

Norms and other numbers
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   linalg.norm
   linalg.matrix_norm (Array API compatible)
   linalg.vector_norm (Array API compatible)
   linalg.cond
   linalg.det
   linalg.matrix_rank
   linalg.slogdet
   trace
   linalg.trace (Array API compatible)

Solving linear equations
--------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   linalg.solve
   linalg.tensorsolve
   linalg.lstsq
   linalg.lu_solve
   linalg.inv
   linalg.pinv
   linalg.tensorinv

Other matrix operations
-----------------------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   diagonal
   linalg.diagonal (Array API compatible)
   linalg.matrix_transpose (Array API compatible)

Exceptions
----------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   linalg.LinAlgError

Linear algebra on several matrices at once
------------------------------------------

Several of the linear algebra routines listed above are able to compute results
for several matrices at once, if they are stacked into the same array.

This is indicated in the documentation via input parameter specifications such
as ``a : (..., M, M) {dpnp.ndarray, usm_ndarray}``. This means that if for
instance given an input array ``a.shape == (N, M, M)``, it is interpreted as a
"stack" of N matrices, each of size M-by-M. Similar specification applies to
return values, for instance the determinant has ``det : (...)`` and will in
this case return an array of shape ``det(a).shape == (N,)``. This generalizes
to linear algebra operations on higher-dimensional arrays: the last 1 or 2
dimensions of a multidimensional array are interpreted as vectors or matrices,
as appropriate for each operation.
