# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2025, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

"""
Interface of the Trigonometric part of the DPNP

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""

# pylint: disable=protected-access
# pylint: disable=no-name-in-module


import dpctl.tensor as dpt
import dpctl.tensor._tensor_elementwise_impl as ti
import dpctl.tensor._type_utils as dtu

import dpnp
import dpnp.backend.extensions.ufunc._ufunc_impl as ufi

from .dpnp_algo.dpnp_elementwise_common import DPNPBinaryFunc, DPNPUnaryFunc
from .dpnp_utils.dpnp_utils_reduction import dpnp_wrap_reduction_call

__all__ = [
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
    "asin",
    "asinh",
    "acos",
    "acosh",
    "atan",
    "atan2",
    "atanh",
    "cbrt",
    "cos",
    "cosh",
    "cumlogsumexp",
    "deg2rad",
    "degrees",
    "exp",
    "exp2",
    "expm1",
    "hypot",
    "log",
    "log10",
    "log1p",
    "log2",
    "logaddexp",
    "logaddexp2",
    "logsumexp",
    "rad2deg",
    "radians",
    "reciprocal",
    "reduce_hypot",
    "rsqrt",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "tan",
    "tanh",
    "unwrap",
]


def _get_accumulation_res_dt(a, dtype):
    """Get a dtype used by dpctl for result array in accumulation function."""

    if dtype is None:
        return dtu._default_accumulation_dtype_fp_types(a.dtype, a.sycl_queue)

    dtype = dpnp.dtype(dtype)
    return dtu._to_device_supported_dtype(dtype, a.sycl_device)


_ACOS_DOCSTRING = r"""
Computes inverse cosine for each element :math:`x_i` for input array `x`.

The inverse of :obj:`dpnp.cos` so that, if :math:`y = cos(x)`, then
:math:`x = acos(y)`. Note that :obj:`dpnp.arccos` is an alias of
:obj:`dpnp.acos`.

For full documentation refer to :obj:`numpy.acos`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise inverse cosine, in radians and in the
    closed interval :math:`[0, \pi]`. The data type of the returned array is
    determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.cos` : Trigonometric cosine, element-wise.
:obj:`dpnp.atan` : Trigonometric inverse tangent, element-wise.
:obj:`dpnp.asin` : Trigonometric inverse sine, element-wise.
:obj:`dpnp.acosh` : Hyperbolic inverse cosine, element-wise.

Notes
-----
:obj:`dpnp.acos` is a multivalued function: for each `x` there are infinitely
many numbers `z` such that :math:`cos(z) = x`. The convention is to return the
angle `z` whose the real part lies in the interval :math:`[0, \pi]`.

For real-valued floating-point input data types, :obj:`dpnp.acos` always
returns real output. For each value that cannot be expressed as a real number
or infinity, it yields ``NaN``.

For complex floating-point input data types, :obj:`dpnp.acos` is a complex
analytic function that has, by convention, the branch cuts
:math:`(-\infty, -1)` and :math:`(1, \infty)` and is continuous from above on
the former and from below on the latter.

The inverse cosine is also known as :math:`cos^{-1}`.

Examples
--------
>>> import dpnp as np
>>> x = np.array([1, -1])
>>> np.acos(x)
array([0.0,  3.14159265])

"""

acos = DPNPUnaryFunc(
    "acos",
    ti._acos_result_type,
    ti._acos,
    _ACOS_DOCSTRING,
    mkl_fn_to_call="_mkl_acos_to_call",
    mkl_impl_fn="_acos",
)

arccos = acos  # arccos is an alias for acos


_ACOSH_DOCSTRING = r"""
Computes inverse hyperbolic cosine for each element :math:`x_i` for input array
`x`.

The inverse of :obj:`dpnp.cosh` so that, if :math:`y = cosh(x)`, then
:math:`x = acosh(y)`. Note that :obj:`dpnp.arccosh` is an alias of
:obj:`dpnp.acosh`.

For full documentation refer to :obj:`numpy.acosh`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise inverse hyperbolic cosine, in radians
    and in the half-closed interval :math:`[0, \infty)`. The data type of the
    returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.cosh` : Hyperbolic cosine, element-wise.
:obj:`dpnp.asinh` : Hyperbolic inverse sine, element-wise.
:obj:`dpnp.sinh` : Hyperbolic sine, element-wise.
:obj:`dpnp.atanh` : Hyperbolic inverse tangent, element-wise.
:obj:`dpnp.tanh` : Hyperbolic tangent, element-wise.
:obj:`dpnp.acos` : Trigonometric inverse cosine, element-wise.

Notes
-----
:obj:`dpnp.acosh` is a multivalued function: for each `x` there are infinitely
many numbers `z` such that :math:`cosh(z) = x`. The convention is to return the
angle `z` whose the real part lies in the interval :math:`[0, \infty)` and the
imaginary part in the interval :math:`[-\pi, \pi]`.

For real-valued floating-point input data types, :obj:`dpnp.acosh` always
returns real output. For each value that cannot be expressed as a real number
or infinity, it yields ``NaN``.

For complex floating-point input data types, :obj:`dpnp.acosh` is a complex
analytic function that has, by convention, the branch cuts :math:`(-\infty, 1)`
and is continuous from above on it.

The inverse hyperbolic cosine is also known as :math:`cosh^{-1}`.

Examples
--------
>>> import dpnp as np
>>> x = np.array([1.0, np.e, 10.0])
>>> np.acosh(x)
array([0.0, 1.65745445, 2.99322285])

"""

acosh = DPNPUnaryFunc(
    "acosh",
    ti._acosh_result_type,
    ti._acosh,
    _ACOSH_DOCSTRING,
    mkl_fn_to_call="_mkl_acosh_to_call",
    mkl_impl_fn="_acosh",
)

arccosh = acosh  # arccosh is an alias for acosh


_ASIN_DOCSTRING = r"""
Computes inverse sine for each element :math:`x_i` for input array `x`.

The inverse of :obj:`dpnp.sin`, so that if :math:`y = sin(x)` then
:math:`x = asin(y)`. Note that :obj:`dpnp.arcsin` is an alias of :obj:`dpnp.asin`.

For full documentation refer to :obj:`numpy.asin`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise inverse sine, in radians and in the
    closed interval :math:`[-\pi/2, \pi/2]`. The data type of the returned
    array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.sin` : Trigonometric sine, element-wise.
:obj:`dpnp.cos` : Trigonometric cosine, element-wise.
:obj:`dpnp.acos` : Trigonometric inverse cosine, element-wise.
:obj:`dpnp.tan` : Trigonometric tangent, element-wise.
:obj:`dpnp.atan` : Trigonometric inverse tangent, element-wise.
:obj:`dpnp.atan2` : Element-wise arc tangent of :math:`\frac{x1}{x2}`
    choosing the quadrant correctly.
:obj:`dpnp.asinh` : Hyperbolic inverse sine, element-wise.

Notes
-----
:obj:`dpnp.asin` is a multivalued function: for each `x` there are infinitely
many numbers `z` such that :math:`sin(z) = x`. The convention is to return the
angle `z` whose the real part lies in the interval :math:`[-\pi/2, \pi/2]`.

For real-valued floating-point input data types, :obj:`dpnp.asin` always
returns real output. For each value that cannot be expressed as a real number
or infinity, it yields ``NaN``.

For complex floating-point input data types, :obj:`dpnp.asin` is a complex
analytic function that has, by convention, the branch cuts
:math:`(-\infty, -1)` and :math:`(1, \infty)` and is continuous from above on
the former and from below on the latter.

The inverse sine is also known as :math:`sin^{-1}`.

Examples
--------
>>> import dpnp as np
>>> x = np.array([0, 1, -1])
>>> np.asin(x)
array([0.0, 1.5707963267948966, -1.5707963267948966])

"""

asin = DPNPUnaryFunc(
    "asin",
    ti._asin_result_type,
    ti._asin,
    _ASIN_DOCSTRING,
    mkl_fn_to_call="_mkl_asin_to_call",
    mkl_impl_fn="_asin",
)

arcsin = asin  # arcsin is an alias for asin


_ASINH_DOCSTRING = r"""
Computes inverse hyperbolic sine for each element :math:`x_i` for input array
`x`.

The inverse of :obj:`dpnp.sinh`, so that if :math:`y = sinh(x)` then
:math:`x = asinh(y)`. Note that :obj:`dpnp.arcsinh` is an alias of
:obj:`dpnp.asinh`.

For full documentation refer to :obj:`numpy.asinh`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise inverse hyperbolic sine, in radians.
    The data type of the returned array is determined by the Type Promotion
    Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.sinh` : Hyperbolic sine, element-wise.
:obj:`dpnp.atanh` : Hyperbolic inverse tangent, element-wise.
:obj:`dpnp.acosh` : Hyperbolic inverse cosine, element-wise.
:obj:`dpnp.asin` : Trigonometric inverse sine, element-wise.

Notes
-----
:obj:`dpnp.asinh` is a multivalued function: for each `x` there are infinitely
many numbers `z` such that :math:`sin(z) = x`. The convention is to return the
angle `z` whose the imaginary part lies in the interval :math:`[-\pi/2, \pi/2]`.

For real-valued floating-point input data types, :obj:`dpnp.asinh` always
returns real output. For each value that cannot be expressed as a real number
or infinity, it yields ``NaN``.

For complex floating-point input data types, :obj:`dpnp.asinh` is a complex
analytic function that has, by convention, the branch cuts
:math:`(-\infty j, -j)` and :math:`(j, \infty j)` and is continuous from the left
on the former and from the right on the latter.

The inverse hyperbolic sine is also known as :math:`sinh^{-1}`.

Examples
--------
>>> import dpnp as np
>>> x = np.array([np.e, 10.0])
>>> np.asinh(x)
array([1.72538256, 2.99822295])

"""

asinh = DPNPUnaryFunc(
    "asinh",
    ti._asinh_result_type,
    ti._asinh,
    _ASINH_DOCSTRING,
    mkl_fn_to_call="_mkl_asinh_to_call",
    mkl_impl_fn="_asinh",
)

arcsinh = asinh  # arcsinh is an alias for asinh


_ATAN_DOCSTRING = r"""
Computes inverse tangent for each element :math:`x_i` for input array `x`.

The inverse of :obj:`dpnp.tan`, so that if :math:`y = tan(x)` then
:math:`x = atan(y)`. Note that :obj:`dpnp.arctan` is an alias of
:obj:`dpnp.atan`.

For full documentation refer to :obj:`numpy.atan`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise inverse tangent, in radians and in the
    closed interval :math:`[-\pi/2, \pi/2]`. The data type of the returned
    array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.atan2` : Element-wise arc tangent of :math:`\frac{x1}{x2}`
    choosing the quadrant correctly.
:obj:`dpnp.angle` : Argument of complex values.
:obj:`dpnp.tan` : Trigonometric tangent, element-wise.
:obj:`dpnp.asin` : Trigonometric inverse sine, element-wise.
:obj:`dpnp.acos` : Trigonometric inverse cosine, element-wise.
:obj:`dpnp.atanh` : Inverse hyperbolic tangent, element-wise.

Notes
-----
:obj:`dpnp.atan` is a multivalued function: for each `x` there are infinitely
many numbers `z` such that :math:`tan(z) = x`. The convention is to return the
angle `z` whose the real part lies in the interval :math:`[-\pi/2, \pi/2]`.

For real-valued floating-point input data types, :obj:`dpnp.atan` always
returns real output. For each value that cannot be expressed as a real number
or infinity, it yields ``NaN``.

For complex floating-point input data types, :obj:`dpnp.atan` is a complex
analytic function that has, by convention, the branch cuts
:math:`(-\infty j, -j)` and :math:`(j, \infty j)` and is continuous from the right
on the former and from the left on the latter.

The inverse tangent is also known as :math:`tan^{-1}`.

Examples
--------
>>> import dpnp as np
>>> x = np.array([0, 1])
>>> np.atan(x)
array([0.0, 0.78539816])

"""

atan = DPNPUnaryFunc(
    "atan",
    ti._atan_result_type,
    ti._atan,
    _ATAN_DOCSTRING,
    mkl_fn_to_call="_mkl_atan_to_call",
    mkl_impl_fn="_atan",
)

arctan = atan  # arctan is an alias for atan


_ATAN2_DOCSTRING = r"""
Calculates the inverse tangent of the quotient :math:`\frac{x1_i}{x2_i}` for
each element :math:`x1_i` of the input array `x1` with the respective element
:math:`x2_i` of the input array `x2`.

Note that :obj:`dpnp.arctan2` is an alias of :obj:`dpnp.atan2`.
This function is not defined for complex-valued arguments; for the so-called
argument of complex values, use :obj:`dpnp.angle`.

For full documentation refer to :obj:`numpy.atan2`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have a real-valued floating-point data type.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to have a real-valued floating-point data
    type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the inverse tangent of the quotient
    :math:`\frac{x1}{x2}`, in radians. The returned array must have a
    real-valued floating-point data type determined by Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.atan` : Trigonometric inverse tangent, element-wise.
:obj:`dpnp.tan` : Compute tangent element-wise.
:obj:`dpnp.angle` : Return the angle of the complex argument.
:obj:`dpnp.asin` : Trigonometric inverse sine, element-wise.
:obj:`dpnp.acos` : Trigonometric inverse cosine, element-wise.
:obj:`dpnp.atanh` : Inverse hyperbolic tangent, element-wise.

Notes
-----
At least one of `x1` or `x2` must be an array.

If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
(which becomes the shape of the output).

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([1., -1.])
>>> x2 = np.array([0., 0.])
>>> np.atan2(x1, x2)
array([1.57079633, -1.57079633])

>>> x1 = np.array([0., 0., np.inf])
>>> x2 = np.array([+0., -0., np.inf])
>>> np.atan2(x1, x2)
array([0.0 , 3.14159265, 0.78539816])

>>> x1 = np.array([-1, +1, +1, -1])
>>> x2 = np.array([-1, -1, +1, +1])
>>> np.atan2(x1, x2) * 180 / np.pi
array([-135.,  -45.,   45.,  135.])

"""

atan2 = DPNPBinaryFunc(
    "atan2",
    ti._atan2_result_type,
    ti._atan2,
    _ATAN2_DOCSTRING,
    mkl_fn_to_call="_mkl_atan2_to_call",
    mkl_impl_fn="_atan2",
)

arctan2 = atan2  # arctan2 is an alias for atan2


_ATANH_DOCSTRING = r"""
Computes hyperbolic inverse tangent for each element :math:`x_i` for input
array `x`.

The inverse of :obj:`dpnp.tanh`, so that if :math:`y = tanh(x)` then
:math:`x = atanh(y)`. Note that :obj:`dpnp.arctanh` is an alias of
:obj:`dpnp.atanh`.

For full documentation refer to :obj:`numpy.atanh`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise inverse hyperbolic tangent, in
    radians. The data type of the returned array is determined by the Type
    Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.tanh` : Hyperbolic tangent, element-wise.
:obj:`dpnp.asinh` : Hyperbolic inverse sine, element-wise.
:obj:`dpnp.acosh` : Hyperbolic inverse cosine, element-wise.
:obj:`dpnp.atan` : Trigonometric inverse tangent, element-wise.

Notes
-----
:obj:`dpnp.atanh` is a multivalued function: for each `x` there are infinitely
many numbers `z` such that :math:`tanh(z) = x`. The convention is to return the
angle `z` whose the imaginary part lies in the interval :math:`[-\pi/2, \pi/2]`.

For real-valued floating-point input data types, :obj:`dpnp.atanh` always
returns real output. For each value that cannot be expressed as a real number
or infinity, it yields ``NaN``.

For complex floating-point input data types, :obj:`dpnp.atanh` is a complex
analytic function that has, by convention, the branch cuts
:math:`(-\infty, -1]` and :math:`[1, \infty)` and is continuous from above on
the former and from below on the latter.

The inverse hyperbolic tangent is also known as :math:`tanh^{-1}`.

Examples
--------
>>> import dpnp as np
>>> x = np.array([0, -0.5])
>>> np.atanh(x)
array([0.0, -0.54930614])

"""

atanh = DPNPUnaryFunc(
    "atanh",
    ti._atanh_result_type,
    ti._atanh,
    _ATANH_DOCSTRING,
    mkl_fn_to_call="_mkl_atanh_to_call",
    mkl_impl_fn="_atanh",
)

arctanh = atanh  # arctanh is an alias for atanh


_CBRT_DOCSTRING = r"""
Computes the cube-root for each element :math:`x_i` for input array `x`.

For full documentation refer to :obj:`numpy.cbrt`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a real-valued floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise cube-root. The data type of the
    returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.sqrt` : Calculate :math:`\sqrt{x}`, element-wise.

Notes
-----
This function is equivalent to :math:`\sqrt[3]{x}`, element-wise.

Examples
--------
>>> import dpnp as np
>>> x = np.array([1, 8, 27])
>>> np.cbrt(x)
array([1., 2., 3.])

"""

cbrt = DPNPUnaryFunc(
    "cbrt",
    ti._cbrt_result_type,
    ti._cbrt,
    _CBRT_DOCSTRING,
    mkl_fn_to_call="_mkl_cbrt_to_call",
    mkl_impl_fn="_cbrt",
)


_COS_DOCSTRING = """
Computes the cosine for each element :math:`x_i` for input array `x`.

For full documentation refer to :obj:`numpy.cos`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise cosine, in radians. The data type of
    the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.acos` : Trigonometric inverse cosine, element-wise.
:obj:`dpnp.sin` : Trigonometric sine, element-wise.
:obj:`dpnp.tan` : Trigonometric tangent, element-wise.
:obj:`dpnp.cosh` : Hyperbolic cosine, element-wise.

Examples
--------
>>> import dpnp as np
>>> x = np.array([0, np.pi/2, np.pi])
>>> np.cos(x)
array([ 1.000000e+00, -4.371139e-08, -1.000000e+00])

"""

cos = DPNPUnaryFunc(
    "cos",
    ti._cos_result_type,
    ti._cos,
    _COS_DOCSTRING,
    mkl_fn_to_call="_mkl_cos_to_call",
    mkl_impl_fn="_cos",
)


_COSH_DOCSTRING = r"""
Computes the hyperbolic cosine for each element :math:`x_i` for input array `x`.

The mathematical definition of the hyperbolic cosine is

.. math:: \operatorname{cosh}(x) = \frac{e^x + e^{-x}}{2}

For full documentation refer to :obj:`numpy.cosh`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise hyperbolic cosine. The data type of
    the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.acosh` : Hyperbolic inverse cosine, element-wise.
:obj:`dpnp.sinh` : Hyperbolic sine, element-wise.
:obj:`dpnp.tanh` : Hyperbolic tangent, element-wise.
:obj:`dpnp.cos` : Trigonometric cosine, element-wise.


Examples
--------
>>> import dpnp as np
>>> x = np.array([0, np.pi/2, np.pi])
>>> np.cosh(x)
array([1.0, 2.5091786, 11.591953])

"""

cosh = DPNPUnaryFunc(
    "cosh",
    ti._cosh_result_type,
    ti._cosh,
    _COSH_DOCSTRING,
    mkl_fn_to_call="_mkl_cosh_to_call",
    mkl_impl_fn="_cosh",
)


def cumlogsumexp(
    x, /, *, axis=None, dtype=None, include_initial=False, out=None
):
    """
    Calculates the cumulative logarithm of the sum of elements in the input
    array `x`.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        Input array, expected to have a boolean or real-valued floating-point
        data type.
    axis : {None, int}, optional
        Axis or axes along which values must be computed. If a tuple of unique
        integers, values are computed over multiple axes. If ``None``, the
        result is computed over the entire array.

        Default: ``None``.
    dtype : {None, str, dtype object}, optional
        Data type of the returned array. If ``None``, the default data type is
        inferred from the "kind" of the input array data type.

        - If `x` has a real-valued floating-point data type, the returned array
          will have the same data type as `x`.
        - If `x` has a boolean or integral data type, the returned array will
          have the default floating point data type for the device where input
          array `x` is allocated.
        - If `x` has a complex-valued floating-point data type, an error is
          raised.

        If the data type (either specified or resolved) differs from the data
        type of `x`, the input array elements are cast to the specified data
        type before computing the result.

        Default: ``None``.
    include_initial : {None, bool}, optional
        A boolean indicating whether to include the initial value (negative
        infinity) as the first value along the provided axis in the output.
        With ``include_initial=True`` the shape of the output is different than
        the shape of the input.

        Default: ``False``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        The array into which the result is written. The data type of `out` must
        match the expected shape and the expected data type of the result or
        (if provided) `dtype`. If ``None`` then a new array is returned.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        An array containing the results. If the result was computed over the
        entire array, a zero-dimensional array is returned. The returned array
        has the data type as described in the `dtype` parameter description
        above.

    See Also
    --------
    :obj:`dpnp.logsumexp` : Logarithm of the sum of elements of the inputs,
                            element-wise.

    Note
    ----
    This function is equivalent of `numpy.logaddexp.accumulate`.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.ones(10)
    >>> np.cumlogsumexp(a)
    array([1.        , 1.69314718, 2.09861229, 2.38629436, 2.60943791,
           2.79175947, 2.94591015, 3.07944154, 3.19722458, 3.30258509])

    """

    dpnp.check_supported_arrays_type(x)
    if x.ndim > 1 and axis is None:
        usm_x = dpnp.ravel(x).get_array()
    else:
        usm_x = dpnp.get_usm_ndarray(x)

    return dpnp_wrap_reduction_call(
        usm_x,
        out,
        dpt.cumulative_logsumexp,
        _get_accumulation_res_dt(x, dtype),
        axis=axis,
        dtype=dtype,
        include_initial=include_initial,
    )


_DEG2RAD_DOCSTRING = r"""
Convert angles from degrees to radians for each element :math:`x_i` for input
array `x`.

For full documentation refer to :obj:`numpy.deg2rad`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a real-valued floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise angle in radians.
    The data type of the returned array is determined by the Type Promotion
    Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.rad2deg` : Convert angles from radians to degrees.
:obj:`dpnp.unwrap` : Remove large jumps in angle by wrapping.
:obj:`dpnp.radians` : Equivalent function.

Notes
-----
The mathematical definition is

.. math:: \operatorname{deg2rad}(x) = \frac{x * \pi}{180}

Examples
--------
>>> import dpnp as np
>>> x = np.array(180)
>>> np.deg2rad(x)
array(3.14159265)

"""

deg2rad = DPNPUnaryFunc(
    "deg2rad",
    ufi._radians_result_type,
    ufi._radians,
    _DEG2RAD_DOCSTRING,
)


_DEGREES_DOCSTRING = r"""
Convert angles from radian to degrees for each element :math:`x_i` for input
array `x`.

For full documentation refer to :obj:`numpy.degrees`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a real-valued floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise angle in degrees.
    The data type of the returned array is determined by the Type Promotion
    Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.rad2deg` : Equivalent function.

Notes
-----
The mathematical definition is

.. math:: \operatorname{degrees}(x) = \frac{180 * x}{\pi}

Examples
--------
>>> import dpnp as np
>>> rad = np.arange(12.) * np.pi/6

Convert a radian array to degrees:

>>> np.degrees(rad)
array([  0.,  30.,  60.,  90., 120., 150., 180., 210., 240., 270., 300.,
       330.])

>>> out = np.zeros_like(rad)
>>> r = np.degrees(rad, out)
>>> np.all(r == out)
array(True)

"""

degrees = DPNPUnaryFunc(
    "degrees",
    ufi._degrees_result_type,
    ufi._degrees,
    _DEGREES_DOCSTRING,
)


_EXP_DOCSTRING = """
Computes the exponential for each element :math:`x_i` of input array `x`.

For full documentation refer to :obj:`numpy.exp`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise exponential of `x`.
    The data type of the returned array is determined by the Type Promotion
    Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.expm1` : Calculate :math:`e^x - 1`, element-wise.
:obj:`dpnp.exp2` : Calculate :math:`2^x`, element-wise.

Examples
--------
>>> import dpnp as np
>>> x = np.arange(3.)
>>> np.exp(x)
array([1.0, 2.718281828, 7.389056099])

"""

exp = DPNPUnaryFunc(
    "exp",
    ti._exp_result_type,
    ti._exp,
    _EXP_DOCSTRING,
    mkl_fn_to_call="_mkl_exp_to_call",
    mkl_impl_fn="_exp",
)


_EXP2_DOCSTRING = """
Computes the base-2 exponential for each element :math:`x_i` for input array
`x`.

For full documentation refer to :obj:`numpy.exp2`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise base-2 exponentials.
    The data type of the returned array is determined by the Type Promotion
    Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.exp` : Calculate :math:`e^x`, element-wise.
:obj:`dpnp.expm1` : Calculate :math:`e^x - 1`, element-wise.
:obj:`dpnp.power` : Calculate :math:`{x1}^{x2}`, element-wise.

Examples
--------
>>> import dpnp as np
>>> x = np.arange(3.)
>>> np.exp2(x)
array([1., 2., 4.])

"""

exp2 = DPNPUnaryFunc(
    "exp2",
    ti._exp2_result_type,
    ti._exp2,
    _EXP2_DOCSTRING,
    mkl_fn_to_call="_mkl_exp2_to_call",
    mkl_impl_fn="_exp2",
)


_EXPM1_DOCSTRING = r"""
Computes the exponential minus ``1`` for each element :math:`x_i` of input
array `x`.

For full documentation refer to :obj:`numpy.expm1`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing containing the evaluated result for each element in `x`.
    The data type of the returned array is determined by the Type Promotion
    Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.exp` : Calculate :math:`e^x`, element-wise.
:obj:`dpnp.exp2` : Calculate :math:`2^x`, element-wise.
:obj:`dpnp.log1p` : Calculate :math:`\log(1 + x)`, element-wise,
    the inverse of :obj:`dpnp.expm1`.

Notes
-----
This function provides greater precision than :math:`e^x - 1` for small values
of `x`.

Examples
--------
>>> import dpnp as np
>>> x = np.arange(3.)
>>> np.expm1(x)
array([0.0, 1.718281828, 6.389056099])

>>> np.expm1(np.array(1e-10))
array(1.00000000005e-10)

>>> np.exp(np.array(1e-10)) - 1
array(1.000000082740371e-10)

"""

expm1 = DPNPUnaryFunc(
    "expm1",
    ti._expm1_result_type,
    ti._expm1,
    _EXPM1_DOCSTRING,
    mkl_fn_to_call="_mkl_expm1_to_call",
    mkl_impl_fn="_expm1",
)


_HYPOT_DOCSTRING = r"""
Computes the square root of the sum of squares for each element :math:`x1_i` of
the input array `x1` with the respective element :math:`x2_i` of the input
array `x2`.

For full documentation refer to :obj:`numpy.hypot`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have a real-valued floating-point data type.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to have a real-valued floating-point data
    type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise hypotenuse. The data type of the
    returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.reduce_hypot` : The square root of the sum of squares of elements
    in the input array.

Notes
-----
At least one of `x1` or `x2` must be an array.

If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
(which becomes the shape of the output).

This function is equivalent to :math:`\sqrt{x1^2 + x2^2}`, element-wise.

Examples
--------
>>> import dpnp as np
>>> x1 = 3 * np.ones((3, 3))
>>> x2 = 4 * np.ones((3, 3))
>>> np.hypot(x1, x2)
array([[5., 5., 5.],
       [5., 5., 5.],
       [5., 5., 5.]])

Example showing broadcast of scalar argument:

>>> np.hypot(x1, 4)
array([[ 5.,  5.,  5.],
       [ 5.,  5.,  5.],
       [ 5.,  5.,  5.]])

"""

hypot = DPNPBinaryFunc(
    "hypot",
    ti._hypot_result_type,
    ti._hypot,
    _HYPOT_DOCSTRING,
    mkl_fn_to_call="_mkl_hypot_to_call",
    mkl_impl_fn="_hypot",
)


_LOG_DOCSTRING = r"""
Computes the natural logarithm for each element :math:`x_i` of input array `x`.

For full documentation refer to :obj:`numpy.log`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise natural logarithm values. The data
    type of the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.log10` : Calculate :math:`\log_{10}(x)`, element-wise.
:obj:`dpnp.log2` : Calculate :math:`\log_2(x)`, element-wise.
:obj:`dpnp.log1p` : Calculate :math:`\log(1 + x)`, element-wise.

Notes
-----
:obj:`dpnp.log` is a multivalued function: for each `x` there are infinitely
many numbers `z` such that :math:`e^z = x`. The convention is to return the `z`
whose the imaginary part lies in the interval :math:`[-\pi, \pi]`.

For real-valued floating-point input data types, :obj:`dpnp.log` always returns
real output. For each value that cannot be expressed as a real number or
nfinity, it yields ``NaN``.

For complex floating-point input data types, :obj:`dpnp.log` is a complex
analytic function that has, by convention, the branch cuts
:math:`(-\infty, 0)` and is continuous from above on it.

In the cases where the input has a negative real part and a very small negative
complex part (approaching 0), the result is so close to :math:`-\pi` that it
evaluates to exactly :math:`-\pi`.

Examples
--------
>>> import dpnp as np
>>> x = np.array([1, np.e, np.e**2, 0])
>>> np.log(x)
array([  0.,   1.,   2., -inf])

"""

log = DPNPUnaryFunc(
    "log",
    ti._log_result_type,
    ti._log,
    _LOG_DOCSTRING,
    mkl_fn_to_call="_mkl_ln_to_call",
    mkl_impl_fn="_ln",
)


_LOG10_DOCSTRING = r"""
Computes the base-10 logarithm for each element :math:`x_i` of input array `x`.

For full documentation refer to :obj:`numpy.log10`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise base-10 logarithm of `x`. The data
    type of the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.log` : Calculate :math:`\log(x)`, element-wise.
:obj:`dpnp.log2` : Calculate :math:`\log_2(x)`, element-wise.
:obj:`dpnp.log1p` : Calculate :math:`\log(1 + x)`, element-wise.

Notes
-----
:obj:`dpnp.log10` is a multivalued function: for each `x` there are infinitely
many numbers `z` such that :math:`10^z = x`. The convention is to return the `z`
whose the imaginary part lies in the interval :math:`[-\pi, \pi]`.

For real-valued floating-point input data types, :obj:`dpnp.log10` always
returns real output. For each value that cannot be expressed as a real number
or nfinity, it yields ``NaN``.

For complex floating-point input data types, :obj:`dpnp.log10` is a complex
analytic function that has, by convention, the branch cuts
:math:`(-\infty, 0)` and is continuous from above on it.

In the cases where the input has a negative real part and a very small negative
complex part (approaching 0), the result is so close to :math:`-\pi` that it
evaluates to exactly :math:`-\pi`.

Examples
--------
>>> import dpnp as np
>>> x = np.arange(3.)
>>> np.log10(x)
array([-inf, 0.0, 0.30102999566])

>>> np.log10(np.array([1e-15, -3.]))
array([-15.,  nan])

"""

log10 = DPNPUnaryFunc(
    "log10",
    ti._log10_result_type,
    ti._log10,
    _LOG10_DOCSTRING,
    mkl_fn_to_call="_mkl_log10_to_call",
    mkl_impl_fn="_log10",
)


_LOG1P_DOCSTRING = r"""
Computes the natural logarithm of (1 + `x`) for each element :math:`x_i` of
input array `x`.

For full documentation refer to :obj:`numpy.log1p`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise :math:`\log(1 + x)` results. The data
    type of the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.expm1` : Calculate :math:`e^x - 1`, element-wise,
    the inverse of :obj:`dpnp.log1p`.
:obj:`dpnp.log` : Calculate :math:`\log(x)`, element-wise.
:obj:`dpnp.log10` : Calculate :math:`\log_{10}(x)`, element-wise.
:obj:`dpnp.log2` : Calculate :math:`\log_2(x)`, element-wise.

Notes
-----
For real-valued floating-point input data types, :obj:`dpnp.log1p` provides
greater precision than :math:`\log(1 + x)` for `x` so small that
:math:`1 + x == 1`.

:obj:`dpnp.log1p` is a multivalued function: for each `x` there are infinitely
many numbers `z` such that :math:`e^z = 1 + x`. The convention is to return the
`z` whose the imaginary part lies in the interval :math:`[-\pi, \pi]`.

For real-valued floating-point input data types, :obj:`dpnp.log1p` always
returns real output. For each value that cannot be expressed as a real number
or nfinity, it yields ``NaN``.

For complex floating-point input data types, :obj:`dpnp.log1p` is a complex
analytic function that has, by convention, the branch cuts
:math:`(-\infty, 0)` and is continuous from above on it.

Examples
--------
>>> import dpnp as np
>>> x = np.arange(3.)
>>> np.log1p(x)
array([0.0, 0.69314718, 1.09861229])

>>> np.log1p(array(1e-99))
array(1e-99)

>>> np.log(array(1 + 1e-99))
array(0.0)

"""

log1p = DPNPUnaryFunc(
    "log1p",
    ti._log1p_result_type,
    ti._log1p,
    _LOG1P_DOCSTRING,
    mkl_fn_to_call="_mkl_log1p_to_call",
    mkl_impl_fn="_log1p",
)


_LOG2_DOCSTRING = r"""
Computes the base-2 logarithm for each element :math:`x_i` of input array `x`.

For full documentation refer to :obj:`numpy.log2`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise base-2 logarithm of `x`. The data type
    of the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.log` : Calculate :math:`\log(x)`, element-wise.
:obj:`dpnp.log10` : Calculate :math:`\log_{10}(x)`, element-wise.
:obj:`dpnp.log1p` : Calculate :math:`\log(1 + x)`, element-wise.

Notes
-----
:obj:`dpnp.log2` is a multivalued function: for each `x` there are infinitely
many numbers `z` such that :math:`2^z = x`. The convention is to return the `z`
whose the imaginary part lies in the interval :math:`[-\pi, \pi]`.

For real-valued floating-point input data types, :obj:`dpnp.log2` always
returns real output. For each value that cannot be expressed as a real number
or nfinity, it yields ``NaN``.

For complex floating-point input data types, :obj:`dpnp.log2` is a complex
analytic function that has, by convention, the branch cuts
:math:`(-\infty, 0)` and is continuous from above on it.

In the cases where the input has a negative real part and a very small negative
complex part (approaching 0), the result is so close to :math:`-\pi` that it
evaluates to exactly :math:`-\pi`.

Examples
--------
>>> import dpnp as np
>>> x = np.array([0, 1, 2, 2**4])
>>> np.log2(x)
array([-inf, 0.0, 1.0, 4.0])

>>> xi = np.array([0+1.j, 1, 2+0.j, 4.j])
>>> np.log2(xi)
array([ 0.+2.26618007j,  0.+0.j        ,  1.+0.j        ,  2.+2.26618007j])

"""

log2 = DPNPUnaryFunc(
    "log2",
    ti._log2_result_type,
    ti._log2,
    _LOG2_DOCSTRING,
    mkl_fn_to_call="_mkl_log2_to_call",
    mkl_impl_fn="_log2",
)


_LOGADDEXP_DOCSTRING = r"""
Calculates the natural logarithm of the sum of exponentiations
:math:`\log(e^{x1} + e^{x2})` for each element :math:`x1_i` of the input array
`x1` with the respective element :math:`x2_i` of the input array `x2`.

For full documentation refer to :obj:`numpy.logaddexp`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have a real-valued floating-point
    data type.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to have a real-valued floating-point data
    type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise results. The data type of the returned
    array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.log` : Calculate :math:`\log(x)`, element-wise.
:obj:`dpnp.exp` : Calculate :math:`e^x`, element-wise.
:obj:`dpnp.logaddexp2`: Calculate :math:`\log_2(2^{x1} + 2^{x2})`, element-wise.
:obj:`dpnp.logsumexp` : Logarithm of the sum of exponentials of elements in the
    input array.

Notes
-----
At least one of `x1` or `x2` must be an array.

If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
(which becomes the shape of the output).

This function is useful in statistics where the calculated probabilities of
events may be so small as to exceed the range of normal floating-point numbers.
In such cases the natural logarithm of the calculated probability is stored.
This function allows adding probabilities stored in such a fashion.

Examples
--------
>>> import dpnp as np
>>> prob1 = np.log(np.array(1e-50))
>>> prob2 = np.log(np.array(2.5e-50))
>>> prob12 = np.logaddexp(prob1, prob2)
>>> prob12
array(-113.87649168)
>>> np.exp(prob12)
array(3.5e-50)

"""

logaddexp = DPNPBinaryFunc(
    "logaddexp",
    ti._logaddexp_result_type,
    ti._logaddexp,
    _LOGADDEXP_DOCSTRING,
)


_LOGADDEXP2_DOCSTRING = r"""
Calculates the base-2 logarithm of the sum of exponentiations
:math:`\log_2(2^{x1} + 2^{x2})` for each element :math:`x1_i` of the input
array `x1` with the respective element :math:`x2_i` of the input array `x2`.

For full documentation refer to :obj:`numpy.logaddexp2`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have a real-valued floating-point
    data type.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to have a real-valued floating-point data
    type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise results. The data type
    of the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.logaddexp`: Calculate :math:`\log(e^{x1} + e^{x2})`, element-wise.
:obj:`dpnp.logsumexp` : Logarithm of the sum of exponentials of elements in the
    input array.

Notes
-----
At least one of `x1` or `x2` must be an array.

If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
(which becomes the shape of the output).

This function is useful in machine learning when the calculated probabilities
of events may be so small as to exceed the range of normal floating-point
numbers. In such cases the base-2 logarithm of the calculated probability can
be used instead. This function allows adding probabilities stored in such a
fashion.

Examples
--------
>>> import dpnp as np
>>> prob1 = np.log2(np.array(1e-50))
>>> prob2 = np.log2(np.array(2.5e-50))
>>> prob12 = np.logaddexp2(prob1, prob2)
>>> prob1, prob2, prob12
(array(-166.09640474), array(-164.77447665), array(-164.28904982))
>>> 2**prob12
array(3.5e-50)

"""

logaddexp2 = DPNPBinaryFunc(
    "logaddexp2",
    ufi._logaddexp2_result_type,
    ufi._logaddexp2,
    _LOGADDEXP2_DOCSTRING,
)


def logsumexp(x, /, *, axis=None, dtype=None, keepdims=False, out=None):
    r"""
    Calculates the natural logarithm of the sum of exponentials of elements in
    the input array.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        Input array, expected to have a boolean or real-valued floating-point
        data type.
    axis : {None, int or tuple of ints}, optional
        Axis or axes along which values must be computed. If a tuple of unique
        integers, values are computed over multiple axes. If ``None``, the
        result is computed over the entire array.

        Default: ``None``.
    dtype : {None, str, dtype object}, optional
        Data type of the returned array. If ``None``, the default data type is
        inferred from the "kind" of the input array data type.

        - If `x` has a real-valued floating-point data type, the returned array
          will have the same data type as `x`.
        - If `x` has a boolean or integral data type, the returned array will
          have the default floating point data type for the device where input
          array `x` is allocated.
        - If `x` has a complex-valued floating-point data type, an error is
          raised.

        If the data type (either specified or resolved) differs from the data
        type of `x`, the input array elements are cast to the specified data
        type before computing the result.

        Default: ``None``.
    keepdims : {None, bool}, optional
        If ``True``, the reduced axes (dimensions) are included in the result
        as singleton dimensions, so that the returned array remains compatible
        with the input arrays according to Array Broadcasting rules. Otherwise,
        if ``False``, the reduced axes are not included in the returned array.
        Default: ``False``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        The array into which the result is written. The data type of `out` must
        match the expected shape and the expected data type of the result or
        (if provided) `dtype`. If ``None`` then a new array is returned.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        An array containing the results. If the result was computed over the
        entire array, a zero-dimensional array is returned. The returned array
        has the data type as described in the `dtype` parameter description
        above.

    See Also
    --------
    :obj:`dpnp.log` : Calculate :math:`\log(x)`, element-wise.
    :obj:`dpnp.exp` : Calculate :math:`e^x`, element-wise.
    :obj:`dpnp.logaddexp`: Calculate :math:`\log(e^{x1} + e^{x2})`,
        element-wise.
    :obj:`dpnp.logaddexp2`: Calculate :math:`\log_2(2^{x1} + 2^{x2})`,
        element-wise.
    :obj:`dpnp.cumlogsumexp` : Cumulative the natural logarithm of the sum of
        elements in the input array.

    Note
    ----
    This function is equivalent of `numpy.logaddexp.reduce`.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.ones(10)
    >>> np.logsumexp(a)
    array(3.30258509)
    >>> np.log(np.sum(np.exp(a)))
    array(3.30258509)

    """

    usm_x = dpnp.get_usm_ndarray(x)
    return dpnp_wrap_reduction_call(
        usm_x,
        out,
        dpt.logsumexp,
        _get_accumulation_res_dt(x, dtype),
        axis=axis,
        dtype=dtype,
        keepdims=keepdims,
    )


_RAD2DEG_DOCSTRING = r"""
Convert angles from radians to degrees for each element :math:`x_i` for input
array `x`.

For full documentation refer to :obj:`numpy.rad2deg`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a real-valued floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise angle in degrees.
    The data type of the returned array is determined by the Type Promotion
    Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.deg2rad` : Convert angles from degrees to radians.
:obj:`dpnp.unwrap` : Remove large jumps in angle by wrapping.
:obj:`dpnp.degrees` : Equivalent function.

Notes
-----
The mathematical definition is

.. math:: \operatorname{rad2deg}(x) = \frac{180 * x}{\pi}

Examples
--------
>>> import dpnp as np
>>> x = np.array(np.pi / 2)
>>> np.rad2deg(x)
array(90.)

"""

rad2deg = DPNPUnaryFunc(
    "rad2deg",
    ufi._degrees_result_type,
    ufi._degrees,
    _RAD2DEG_DOCSTRING,
)


_RADIANS_DOCSTRING = r"""
Convert angles from degrees to radians for each element :math:`x_i` for input
array `x`.

For full documentation refer to :obj:`numpy.radians`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a real-valued floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise angle in radians.
    The data type of the returned array is determined by the Type Promotion
    Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.deg2rad` : Equivalent function.

Notes
-----
The mathematical definition is

.. math:: \operatorname{radians}(x) = \frac{x * \pi}{180}

Examples
--------
>>> import dpnp as np
>>> deg = np.arange(12.) * 30.

Convert a degree array to radians:

>>> np.radians(deg)
array([0.        , 0.52359878, 1.04719755, 1.57079633, 2.0943951 ,
       2.61799388, 3.14159265, 3.66519143, 4.1887902 , 4.71238898,
       5.23598776, 5.75958653])

>>> out = np.zeros_like(deg)
>>> ret = np.radians(deg, out)
>>> ret is out
True

"""

radians = DPNPUnaryFunc(
    "radians",
    ufi._radians_result_type,
    ufi._radians,
    _RADIANS_DOCSTRING,
)


_RECIPROCAL_DOCSTRING = r"""
Computes the reciprocal for each element :math:`x_i` for input array `x`.

For full documentation refer to :obj:`numpy.reciprocal`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise reciprocals. The returned array has
    a floating-point data type determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.rsqrt` : Calculate :math:`\frac{1}{\sqrt{x}}`, element-wise.

Notes
-----
This function is equivalent to :math:`\frac{1}{x}`, element-wise.

Examples
--------
>>> import dpnp as np
>>> x = np.array([1, 2., 3.33])
>>> np.reciprocal(x)
array([1.0, 0.5, 0.3003003])

"""

reciprocal = DPNPUnaryFunc(
    "reciprocal",
    ti._reciprocal_result_type,
    ti._reciprocal,
    _RECIPROCAL_DOCSTRING,
    mkl_fn_to_call="_mkl_inv_to_call",
    mkl_impl_fn="_inv",
)


def reduce_hypot(x, /, *, axis=None, dtype=None, keepdims=False, out=None):
    r"""
    Calculates the square root of the sum of squares of elements in
    the input array.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        Input array, expected to have a boolean or real-valued floating-point
        data type.
    axis : {None, int or tuple of ints}, optional
        Axis or axes along which values must be computed. If a tuple of unique
        integers, values are computed over multiple axes. If ``None``, the
        result is computed over the entire array.

        Default: ``None``.
    dtype : {None, str, dtype object}, optional
        Data type of the returned array. If ``None``, the default data type is
        inferred from the "kind" of the input array data type.

        - If `x` has a real-valued floating-point data type, the returned array
          will have the same data type as `x`.
        - If `x` has a boolean or integral data type, the returned array will
          have the default floating point data type for the device where input
          array `x` is allocated.
        - If `x` has a complex-valued floating-point data type, an error is
          raised.

        If the data type (either specified or resolved) differs from the data
        type of `x`, the input array elements are cast to the specified data
        type before computing the result.

        Default: ``None``.
    keepdims : {None, bool}, optional
        If ``True``, the reduced axes (dimensions) are included in the result
        as singleton dimensions, so that the returned array remains compatible
        with the input arrays according to Array Broadcasting rules. Otherwise,
        if ``False``, the reduced axes are not included in the returned array.
        Default: ``False``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        The array into which the result is written. The data type of `out` must
        match the expected shape and the expected data type of the result or
        (if provided) `dtype`. If ``None`` then a new array is returned.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        An array containing the results. If the result was computed over the
        entire array, a zero-dimensional array is returned. The returned array
        has the data type as described in the `dtype` parameter description
        above.

    See Also
    --------
    :obj:`dpnp.hypot` : Calculates :math:`\sqrt{x1^2 + x2^2}`, element-wise.

    Note
    ----
    This function is equivalent of `numpy.hypot.reduce`.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.ones(10)
    >>> np.reduce_hypot(a)
    array(3.16227766)
    >>> np.sqrt(np.sum(np.square(a)))
    array(3.16227766)

    """

    usm_x = dpnp.get_usm_ndarray(x)
    return dpnp_wrap_reduction_call(
        usm_x,
        out,
        dpt.reduce_hypot,
        _get_accumulation_res_dt(x, dtype),
        axis=axis,
        dtype=dtype,
        keepdims=keepdims,
    )


_RSQRT_DOCSTRING = r"""
Computes the reciprocal square-root for each element :math:`x_i` for input
array `x`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a real-valued floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional:
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is `None`.
    Default: ``"K"``

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise reciprocal square-roots.
    The returned array has a floating-point data type determined by
    the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.sqrt` : Calculate :math:`\sqrt{x}`, element-wise.
:obj:`dpnp.reciprocal` : Calculate :math:`\frac{1}{x}`, element-wise.

Notes
-----
This function is equivalent to :math:`\frac{1}{\sqrt{x}}`, element-wise.

Examples
--------
>>> import dpnp as np
>>> x = np.array([1, 8, 27])
>>> np.rsqrt(x)
array([1.        , 0.35355338, 0.19245009])

"""

rsqrt = DPNPUnaryFunc(
    "rsqrt",
    ti._rsqrt_result_type,
    ti._rsqrt,
    _RSQRT_DOCSTRING,
)


_SIN_DOCSTRING = """
Computes the sine for each element :math:`x_i` of input array `x`.

For full documentation refer to :obj:`numpy.sin`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise sine, in radians. The data type of the
    returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.asin` : Trigonometric inverse sine, element-wise.
:obj:`dpnp.cos` : Trigonometric cosine, element-wise.
:obj:`dpnp.tan` : Trigonometric tangent, element-wise.
:obj:`dpnp.sinh` : Hyperbolic sine, element-wise.

Examples
--------
>>> import dpnp as np
>>> x = np.array([0, np.pi/2, np.pi])
>>> np.sin(x)
array([ 0.000000e+00,  1.000000e+00, -8.742278e-08])

"""

sin = DPNPUnaryFunc(
    "sin",
    ti._sin_result_type,
    ti._sin,
    _SIN_DOCSTRING,
    mkl_fn_to_call="_mkl_sin_to_call",
    mkl_impl_fn="_sin",
)


_SINH_DOCSTRING = r"""
Computes the hyperbolic sine for each element :math:`x_i` for input array `x`.

The mathematical definition of the hyperbolic sine is

.. math:: \operatorname{sinh}(x) = \frac{e^x - e^{-x}}{2}

For full documentation refer to :obj:`numpy.sinh`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise hyperbolic sine. The data type of the
    returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.asinh` : Hyperbolic inverse sine, element-wise.
:obj:`dpnp.cosh` : Hyperbolic cosine, element-wise.
:obj:`dpnp.tanh` : Hyperbolic tangent, element-wise.
:obj:`dpnp.sin` : Trigonometric sine, element-wise.

Examples
--------
>>> import dpnp as np
>>> x = np.array([0, np.pi/2, np.pi])
>>> np.sinh(x)
array([0.0, 2.3012989, 11.548739])

"""

sinh = DPNPUnaryFunc(
    "sinh",
    ti._sinh_result_type,
    ti._sinh,
    _SINH_DOCSTRING,
    mkl_fn_to_call="_mkl_sinh_to_call",
    mkl_impl_fn="_sinh",
)


_SQRT_DOCSTRING = r"""
Computes the principal square-root for each element :math:`x_i` of input array
`x`.

For full documentation refer to :obj:`numpy.sqrt`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise principal square-roots of `x`. The
    data type of the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.cbrt` : Calculate :math:`\sqrt[3]{x}`, element-wise.
:obj:`dpnp.rsqrt` : Calculate :math:`\frac{1}{\sqrt{x}}`, element-wise.

Notes
-----
This function is equivalent to :math:`\sqrt{x}`, element-wise.

By convention, the branch cut of the square root is the negative real axis
:math:`(-\infty, 0)`.

The square root is a continuous function from above the branch cut, taking into
account the sign of the imaginary component.

Accordingly, for complex arguments, the function returns the square root in the
range of the right half-plane, including the imaginary axis (i.e., the plane
defined by :math:`[0, +\infty)` along the real axis and
:math:`(-\infty, +\infty)` along the imaginary axis).

Examples
--------
>>> import dpnp as np
>>> x = np.array([1, 4, 9])
>>> np.sqrt(x)
array([1., 2., 3.])

>>> x2 = np.array([4, -1, np.inf])
>>> np.sqrt(x2)
array([ 2., nan, inf])

"""

sqrt = DPNPUnaryFunc(
    "sqrt",
    ti._sqrt_result_type,
    ti._sqrt,
    _SQRT_DOCSTRING,
    mkl_fn_to_call="_mkl_sqrt_to_call",
    mkl_impl_fn="_sqrt",
)


_SQUARE_DOCSTRING = r"""
Squares each element :math:`x_i` of input array `x`.

For full documentation refer to :obj:`numpy.square`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, may have any data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise squares of `x`. The data type of the
    returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.linalg.matrix_power` : Raise a square matrix to the (integer)
    power `n`.
:obj:`dpnp.sqrt` : Calculate :math:`\sqrt{x}`, element-wise.
:obj:`dpnp.power` : Calculate :math:`{x1}^{x2}`, element-wise.

Examples
--------
>>> import dpnp as np
>>> x = np.array([-1j, 1])
>>> np.square(x)
array([-1.+0.j,  1.+0.j])

"""

square = DPNPUnaryFunc(
    "square",
    ti._square_result_type,
    ti._square,
    _SQUARE_DOCSTRING,
    mkl_fn_to_call="_mkl_sqr_to_call",
    mkl_impl_fn="_sqr",
)


_TAN_DOCSTRING = """
Computes the tangent for each element :math:`x_i` for input array `x`.

For full documentation refer to :obj:`numpy.tan`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise tangent, in radians. The data type of
    the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.atan` : Trigonometric inverse tangent, element-wise.
:obj:`dpnp.sin` : Trigonometric sine, element-wise.
:obj:`dpnp.cos` : Trigonometric cosine, element-wise.
:obj:`dpnp.tanh` : Hyperbolic tangent, element-wise.

Examples
--------
>>> import dpnp as np
>>> x = np.array([-np.pi, np.pi/2, np.pi])
>>> np.tan(x)
array([1.22460635e-16, 1.63317787e+16, -1.22460635e-16])

"""

tan = DPNPUnaryFunc(
    "tan",
    ti._tan_result_type,
    ti._tan,
    _TAN_DOCSTRING,
    mkl_fn_to_call="_mkl_tan_to_call",
    mkl_impl_fn="_tan",
)


_TANH_DOCSTRING = r"""
Computes the hyperbolic tangent for each element :math:`x_i` for input array
`x`.

The mathematical definition of the hyperbolic tangent is

.. math::
    \operatorname{tanh}(x) = \frac{\operatorname{sinh}(x)}{\operatorname{cosh}(x)}

For full documentation refer to :obj:`numpy.tanh`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise hyperbolic tangent. The data type of
    the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.atanh` : Hyperbolic inverse tangent, element-wise.
:obj:`dpnp.sinh` : Hyperbolic sine, element-wise.
:obj:`dpnp.cosh` : Hyperbolic cosine, element-wise.
:obj:`dpnp.tan` : Trigonometric tangent, element-wise.

Examples
--------
>>> import dpnp as np
>>> x = np.array([0, -np.pi, np.pi/2, np.pi])
>>> np.tanh(x)
array([0.0, -0.996272, 0.917152, 0.996272])

"""

tanh = DPNPUnaryFunc(
    "tanh",
    ti._tanh_result_type,
    ti._tanh,
    _TANH_DOCSTRING,
    mkl_fn_to_call="_mkl_tanh_to_call",
    mkl_impl_fn="_tanh",
)


def unwrap(p, discont=None, axis=-1, *, period=2 * dpnp.pi):
    r"""
    Unwrap by taking the complement of large deltas with respect to the period.

    This unwraps a signal `p` by changing elements which have an absolute
    difference from their predecessor of more than
    :math:`\max(discont, \frac{period}{2})` to their `period`-complementary
    values.

    For the default case where `period` is :math:`2\pi` and `discont` is
    :math:`\pi`, this unwraps a radian phase `p` such that adjacent differences
    are never greater than :math:`\pi` by adding :math:`2k\pi` for some integer
    :math:`k`.

    For full documentation refer to :obj:`numpy.unwrap`.

    Parameters
    ----------
    p : {dpnp.ndarray, usm_ndarray}
        Input array.
    discont : {float, None}, optional
        Maximum discontinuity between values, default is ``None`` which is an
        alias for :math:`\frac{period}{2}`. Values below
        :math:`\frac{period}{2}` are treated as if they were
        :math:`\frac{period}{2}`. To have an effect different from the default,
        `discont` should be larger than :math:`\frac{period}{2}`.

        Default: ``None``.
    axis : int, optional
        Axis along which unwrap will operate, default is the last axis.

        Default: ``-1``.
    period : float, optional
        Size of the range over which the input wraps.

        Default: ``2 * pi``.

    Returns
    -------
    out : dpnp.ndarray
        Output array.

    See Also
    --------
    :obj:`dpnp.rad2deg` : Convert angles from radians to degrees.
    :obj:`dpnp.deg2rad` : Convert angles from degrees to radians.

    Notes
    -----
    If the discontinuity in `p` is smaller than :math:`\frac{period}{2}`, but
    larger than `discont`, no unwrapping is done because taking the complement
    would only make the discontinuity larger.

    Examples
    --------
    >>> import dpnp as np
    >>> phase = np.linspace(0, np.pi, num=5)
    >>> phase[3:] += np.pi
    >>> phase
    array([0.        , 0.78539816, 1.57079633, 5.49778714, 6.28318531])
    >>> np.unwrap(phase)
    array([ 0.        ,  0.78539816,  1.57079633, -0.78539816,  0.        ])

    >>> phase = np.array([0, 1, 2, -1, 0])
    >>> np.unwrap(phase, period=4)
    array([0, 1, 2, 3, 4])

    >>> phase = np.array([1, 2, 3, 4, 5, 6, 1, 2, 3])
    >>> np.unwrap(phase, period=6)
    array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    >>> phase = np.array([2, 3, 4, 5, 2, 3, 4, 5])
    >>> np.unwrap(phase, period=4)
    array([2, 3, 4, 5, 6, 7, 8, 9])

    >>> phase_deg = np.mod(np.linspace(0 ,720, 19), 360) - 180
    >>> np.unwrap(phase_deg, period=360)
    array([-180., -140., -100.,  -60.,  -20.,   20.,   60.,  100.,  140.,
            180.,  220.,  260.,  300.,  340.,  380.,  420.,  460.,  500.,
            540.])

    """

    dpnp.check_supported_arrays_type(p)

    p_nd = p.ndim
    p_diff = dpnp.diff(p, axis=axis)

    if discont is None:
        discont = period / 2

    # full slices
    slice1 = [slice(None, None)] * p_nd
    slice1[axis] = slice(1, None)
    slice1 = tuple(slice1)

    dt = dpnp.result_type(p_diff, period)
    if dpnp.issubdtype(dt, dpnp.integer):
        interval_high, rem = divmod(period, 2)
        boundary_ambiguous = rem == 0
    else:
        interval_high = period / 2
        boundary_ambiguous = True
    interval_low = -interval_high

    ddmod = p_diff - interval_low
    ddmod = dpnp.remainder(ddmod, period, out=ddmod)
    ddmod += interval_low

    if boundary_ambiguous:
        mask = ddmod == interval_low
        mask &= p_diff > 0
        ddmod = dpnp.where(mask, interval_high, ddmod, out=ddmod)

    ph_correct = dpnp.subtract(ddmod, p_diff, out=ddmod)
    abs_p_diff = dpnp.abs(p_diff, out=p_diff)
    ph_correct = dpnp.where(abs_p_diff < discont, 0, ph_correct, out=ph_correct)

    up = dpnp.astype(p, dt, copy=True)
    up[slice1] = p[slice1]
    up[slice1] += ph_correct.cumsum(axis=axis)
    return up
