Array Manipulation Routines
===========================

.. https://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html

Basic operations
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.copyto


Changing array shape
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.reshape
   dpnp.ravel


Transpose-like operations
-------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.moveaxis
   dpnp.roll
   dpnp.rollaxis
   dpnp.swapaxes
   dpnp.transpose

.. seealso::
   :attr:`dpnp.dparray.T`

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
   dpnp.asnumpy
   dpnp.asanyarray
   dpnp.asfarray
   dpnp.asfortranarray
   dpnp.ascontiguousarray
   dpnp.require


Joining arrays
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.concatenate
   dpnp.stack
   dpnp.column_stack
   dpnp.dstack
   dpnp.hstack
   dpnp.vstack


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

   dpnp.unique
   dpnp.trim_zeros


Rearranging elements
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.flip
   dpnp.fliplr
   dpnp.flipud
   dpnp.reshape
   dpnp.roll
   dpnp.rot90
