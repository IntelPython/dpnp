.. _ufunc:

.. currentmodule:: dpnp

Universal functions
===================

.. hint:: `NumPy API Reference: Universal functions (numpy.ufunc) <https://numpy.org/doc/stable/reference/ufuncs.html>`_

DPNP provides universal functions (a.k.a. ufuncs) to support various element-wise operations.

Available ufuncs
----------------

Math operations
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   add
   subtract
   multiply
   matmul
   divide
   logaddexp
   logaddexp2
   true_divide
   floor_divide
   negative
   positive
   power
   pow
   float_power
   remainder
   mod
   fmod
   divmod
   absolute
   fabs
   rint
   sign
   heaviside
   conj
   conjugate
   exp
   exp2
   log
   log2
   log10
   expm1
   log1p
   proj
   sqrt
   square
   cbrt
   reciprocal
   rsqrt
   gcd
   lcm

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

   sin
   cos
   tan
   arcsin
   asin
   arccos
   acos
   arctan
   atan
   arctan2
   atan2
   hypot
   sinh
   cosh
   tanh
   arcsinh
   asinh
   arccosh
   acosh
   arctanh
   atanh
   degrees
   radians
   deg2rad
   rad2deg


Bit-twiddling functions
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   bitwise_and
   bitwise_not
   bitwise_or
   bitwise_xor
   invert
   bitwise_invert
   left_shift
   bitwise_left_shift
   right_shift
   bitwise_right_shift
   bitwise_count


Comparison functions
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   greater
   greater_equal
   less
   less_equal
   not_equal
   equal

   logical_and
   logical_or
   logical_xor
   logical_not

   maximum
   minimum
   fmax
   fmin


Floating functions
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   isfinite
   isinf
   isnan
   fabs
   signbit
   copysign
   nextafter
   spacing
   modf
   ldexp
   frexp
   fmod
   floor
   ceil
   trunc
