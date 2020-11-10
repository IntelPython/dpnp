Linear Algebra
==============

.. https://docs.scipy.org/doc/numpy/reference/routines.linalg.html

Matrix and vector products
--------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   
   dpnp.cross
   dpnp.dot
   dpnp.vdot
   dpnp.inner
   dpnp.outer
   dpnp.matmul
   dpnp.tensordot
   dpnp.einsum
   dpnp.linalg.matrix_power
   dpnp.kron
   
   dpnpx.scipy.linalg.kron

Decompositions
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.linalg.cholesky
   dpnp.linalg.qr
   dpnp.linalg.svd

Matrix eigenvalues
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.linalg.eigh
   dpnp.linalg.eigvalsh

Norms etc.
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.linalg.det
   dpnp.linalg.norm
   dpnp.linalg.matrix_rank
   dpnp.linalg.slogdet
   dpnp.trace


Solving linear equations
--------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.linalg.solve
   dpnp.linalg.tensorsolve
   dpnp.linalg.lstsq
   dpnp.linalg.inv
   dpnp.linalg.pinv
   dpnp.linalg.tensorinv

   dpnpx.scipy.linalg.lu_factor
   dpnpx.scipy.linalg.lu_solve
   dpnpx.scipy.linalg.solve_triangular

Special Matrices
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.tri
   dpnp.tril
   dpnp.triu

   dpnpx.scipy.linalg.tri
   dpnpx.scipy.linalg.tril
   dpnpx.scipy.linalg.triu
   dpnpx.scipy.linalg.toeplitz
   dpnpx.scipy.linalg.circulant
   dpnpx.scipy.linalg.hankel
   dpnpx.scipy.linalg.hadamard
   dpnpx.scipy.linalg.leslie
   dpnpx.scipy.linalg.block_diag
   dpnpx.scipy.linalg.companion
   dpnpx.scipy.linalg.helmert
   dpnpx.scipy.linalg.hilbert
   dpnpx.scipy.linalg.dft
   dpnpx.scipy.linalg.fiedler
   dpnpx.scipy.linalg.fiedler_companion
   dpnpx.scipy.linalg.convolution_matrix
