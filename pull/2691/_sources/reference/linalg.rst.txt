.. _routines.linalg:

.. py:module:: dpnp.linalg

Linear algebra
==============

.. https://numpy.org/doc/stable/reference/routines.linalg.html

Matrix and vector products
--------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.dot
   dpnp.linalg.multi_dot
   dpnp.vdot
   dpnp.vecdot
   dpnp.linalg.vecdot (Array API compatible)
   dpnp.inner
   dpnp.outer
   dpnp.matmul
   dpnp.linalg.matmul (Array API compatible)
   dpnp.matvec
   dpnp.vecmat
   dpnp.tensordot
   dpnp.linalg.tensordot (Array API compatible)
   dpnp.einsum
   dpnp.einsum_path
   dpnp.linalg.matrix_power
   dpnp.kron
   dpnp.linalg.cross (Array API compatible)

Decompositions
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.linalg.cholesky
   dpnp.linalg.outer
   dpnp.linalg.qr
   dpnp.linalg.lu_factor
   dpnp.linalg.svd
   dpnp.linalg.svdvals

Matrix eigenvalues
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.linalg.eig
   dpnp.linalg.eigh
   dpnp.linalg.eigvals
   dpnp.linalg.eigvalsh

Norms and other numbers
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.linalg.norm
   dpnp.linalg.matrix_norm (Array API compatible)
   dpnp.linalg.vector_norm (Array API compatible)
   dpnp.linalg.cond
   dpnp.linalg.det
   dpnp.linalg.matrix_rank
   dpnp.linalg.slogdet
   dpnp.trace
   dpnp.linalg.trace (Array API compatible)

Solving linear equations
--------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.linalg.solve
   dpnp.linalg.tensorsolve
   dpnp.linalg.lstsq
   dpnp.linalg.lu_solve
   dpnp.linalg.inv
   dpnp.linalg.pinv
   dpnp.linalg.tensorinv

Other matrix operations
-----------------------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.diagonal
   dpnp.linalg.diagonal (Array API compatible)
   dpnp.linalg.matrix_transpose (Array API compatible)

Exceptions
----------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.linalg.linAlgError
