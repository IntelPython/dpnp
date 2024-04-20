.. _ufunc:

Universal Functions (ufunc)
===========================

.. https://docs.scipy.org/doc/numpy/reference/ufuncs.html

DPNP provides universal functions (a.k.a. ufuncs) to support various elementwise operations.

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
