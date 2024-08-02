.. _ufunc:

Universal Functions (ufunc)
===========================

.. https://docs.scipy.org/doc/numpy/reference/ufuncs.html

A universal function (or ufunc for short) is a function that operates on ndarrays in an element-by-element fashion, \
supporting array broadcasting, type casting, and several other standard features. That is, a ufunc is a “vectorized” \
wrapper for a function that takes a fixed number of specific inputs and produces a fixed number of specific outputs. \
For full documentation refer to :obj:`numpy.ufunc`.

ufuncs
------
.. autosummary::
   :toctree: generated/

   dpnp.ufunc

Attributes
~~~~~~~~~~

There are some informational attributes that universal functions
possess. None of the attributes can be set.

============  =================================================================
**__doc__**   A docstring for each ufunc. The first part of the docstring is
              dynamically generated from the number of outputs, the name, and
              the number of inputs. The second part of the docstring is
              provided at creation time and stored with the ufunc.

**__name__**  The name of the ufunc.
============  =================================================================

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ufunc.nin
   dpnp.ufunc.nout
   dpnp.ufunc.nargs
   dpnp.ufunc.types
   dpnp.ufunc.ntypes

Methods
~~~~~~~
.. autosummary::
   :toctree: generated/

   dpnp.ufunc.outer

Available ufuncs
----------------

Math operations
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.add
   dpnp.subtract
   dpnp.multiply
   dpnp.divide
   dpnp.logaddexp
   dpnp.logaddexp2
   dpnp.true_divide
   dpnp.floor_divide
   dpnp.negative
   dpnp.power
   dpnp.remainder
   dpnp.mod
   dpnp.fmod
   dpnp.abs
   dpnp.absolute
   dpnp.fabs
   dpnp.rint
   dpnp.sign
   dpnp.exp
   dpnp.exp2
   dpnp.log
   dpnp.log2
   dpnp.log10
   dpnp.expm1
   dpnp.log1p
   dpnp.proj
   dpnp.sqrt
   dpnp.cbrt
   dpnp.square
   dpnp.reciprocal
   dpnp.rsqrt
   dpnp.gcd
   dpnp.lcm


Trigonometric functions
~~~~~~~~~~~~~~~~~~~~~~~
All trigonometric functions use radians when an angle is called for.
The ratio of degrees to radians is :math:`180^{\circ}/\pi.`

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.sin
   dpnp.cos
   dpnp.tan
   dpnp.arcsin
   dpnp.arccos
   dpnp.arctan
   dpnp.arctan2
   dpnp.hypot
   dpnp.sinh
   dpnp.cosh
   dpnp.tanh
   dpnp.arcsinh
   dpnp.arccosh
   dpnp.arctanh
   dpnp.degrees
   dpnp.radians
   dpnp.deg2rad
   dpnp.rad2deg


Bit-twiddling functions
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.bitwise_and
   dpnp.bitwise_or
   dpnp.bitwise_xor
   dpnp.invert
   dpnp.left_shift
   dpnp.right_shift


Comparison functions
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.greater
   dpnp.greater_equal
   dpnp.less
   dpnp.less_equal
   dpnp.not_equal
   dpnp.equal

   dpnp.logical_and
   dpnp.logical_or
   dpnp.logical_xor
   dpnp.logical_not

   dpnp.maximum
   dpnp.minimum
   dpnp.fmax
   dpnp.fmin


Floating functions
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.isfinite
   dpnp.isinf
   dpnp.isnan
   dpnp.isnat
   dpnp.fabs
   dpnp.signbit
   dpnp.copysign
   dpnp.nextafter
   dpnp.spacing
   dpnp.modf
   dpnp.ldexp
   dpnp.frexp
   dpnp.fmod
   dpnp.floor
   dpnp.ceil
   dpnp.trunc
