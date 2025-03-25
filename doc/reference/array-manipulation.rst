Array manipulation routines
===========================

.. https://numpy.org/doc/stable/reference/routines.array-manipulation.html

Basic operations
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.copyto
   dpnp.ndim
   dpnp.shape
   dpnp.size


Changing array shape
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.reshape
   dpnp.ravel
   dpnp.ndarray.flat
   dpnp.ndarray.flatten


Transpose-like operations
-------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.moveaxis
   dpnp.rollaxis
   dpnp.swapaxes
   dpnp.ndarray.T
   dpnp.transpose
   dpnp.permute_dims
   dpnp.matrix_transpose (Array API compatible)


Changing number of dimensions
-----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.atleast_1d
   dpnp.atleast_2d
   dpnp.atleast_3d
   dpnp.broadcast
   dpnp.broadcast_to
   dpnp.broadcast_arrays
   dpnp.expand_dims
   dpnp.squeeze


Changing kind of array
----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.asarray
   dpnp.asanyarray
   dpnp.asnumpy
   dpnp.asfarray
   dpnp.asfortranarray
   dpnp.ascontiguousarray
   dpnp.asarray_chkfinite
   dpnp.require


Joining arrays
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.concatenate
   dpnp.concat
   dpnp.stack
   dpnp.block
   dpnp.vstack
   dpnp.hstack
   dpnp.dstack
   dpnp.column_stack
   dpnp.row_stack


Splitting arrays
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.split
   dpnp.array_split
   dpnp.dsplit
   dpnp.hsplit
   dpnp.vsplit
   dpnp.unstack


Tiling arrays
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.tile
   dpnp.repeat


Adding and removing elements
----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.delete
   dpnp.insert
   dpnp.append
   dpnp.resize
   dpnp.trim_zeros
   dpnp.pad


Rearranging elements
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.flip
   dpnp.fliplr
   dpnp.flipud
   dpnp.roll
   dpnp.rot90
