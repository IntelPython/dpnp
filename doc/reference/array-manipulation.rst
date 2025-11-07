.. currentmodule:: dpnp

Array manipulation routines
===========================

.. hint:: `NumPy API Reference: Array manipulation routines <https://numpy.org/doc/stable/reference/routines.array-manipulation.html>`_

Basic operations
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   copyto
   ndim
   shape
   size


Changing array shape
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   reshape
   ravel
   ndarray.flat
   ndarray.flatten


Transpose-like operations
-------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   moveaxis
   rollaxis
   swapaxes
   ndarray.T
   transpose
   permute_dims
   matrix_transpose (Array API compatible)


Changing number of dimensions
-----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   atleast_1d
   atleast_2d
   atleast_3d
   broadcast
   broadcast_to
   broadcast_arrays
   expand_dims
   squeeze


Changing kind of array
----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   asarray
   asanyarray
   asnumpy
   asfortranarray
   ascontiguousarray
   asarray_chkfinite
   require


Joining arrays
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   concatenate
   concat
   stack
   block
   vstack
   hstack
   dstack
   column_stack
   row_stack


Splitting arrays
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   split
   array_split
   dsplit
   hsplit
   vsplit
   unstack


Tiling arrays
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   tile
   repeat


Adding and removing elements
----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   delete
   insert
   append
   resize
   trim_zeros
   pad


Rearranging elements
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   flip
   fliplr
   flipud
   roll
   rot90
