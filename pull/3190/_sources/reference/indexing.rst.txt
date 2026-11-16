.. _routines.indexing:
.. _arrays.indexing:

.. currentmodule:: dpnp

Indexing routines
=================

.. hint:: `NumPy API Reference: Indexing routines <https://numpy.org/doc/stable/reference/routines.indexing.html>`_

Generating index arrays
-----------------------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   c_
   r_
   s_
   nonzero
   where
   indices
   ix_
   ogrid
   ravel_multi_index
   unravel_index
   diag_indices
   diag_indices_from
   mask_indices
   tril_indices
   tril_indices_from
   triu_indices
   triu_indices_from


Indexing-like operations
------------------------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   take
   take_along_axis
   choose
   compress
   diag
   diagonal
   select


Inserting data into arrays
--------------------------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   place
   put
   put_along_axis
   putmask
   fill_diagonal


Iterating over arrays
---------------------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   nditer
   ndenumerate
   ndindex
   nested_iters
   flatiter
   iterable
