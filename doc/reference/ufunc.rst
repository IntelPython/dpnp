.. _ufunc:

Universal Functions (ufunc)
===========================

.. https://docs.scipy.org/doc/numpy/reference/ufuncs.html

DPNP provides universal functions (a.k.a. ufuncs) to support various element-wise operations.

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
   dpnp.matmul
   dpnp.divide
   dpnp.logaddexp
   dpnp.logaddexp2
   dpnp.true_divide
   dpnp.floor_divide
   dpnp.negative
   dpnp.positive
   dpnp.power
   dpnp.float_power
   dpnp.remainder
   dpnp.mod
   dpnp.fmod
   dpnp.divmod
   dpnp.absolute
   dpnp.fabs
   dpnp.rint
   dpnp.sign
   dpnp.heaviside
   dpnp.conj
   dpnp.conjugate
   dpnp.exp
   dpnp.exp2
   dpnp.log
   dpnp.log2
   dpnp.log10
   dpnp.expm1
   dpnp.log1p
   dpnp.proj
   dpnp.sqrt
   dpnp.square
   dpnp.cbrt
   dpnp.reciprocal
   dpnp.rsqrt
   dpnp.gcd
   dpnp.lcm

.. tip::

   The optional output arguments can be used to help you save memory
   for large calculations. If your arrays are large, complicated
   expressions can take longer than absolutely necessary due to the
   creation and (later) destruction of temporary calculation
   spaces. For example, the expression ``G = A * B + C`` is equivalent to
   ``T1 = A * B; G = T1 + C; del T1``. It will be more quickly executed
   as ``G = A * B; add(G, C, G)`` which is the same as
   ``G = A * B; G += C``.


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
