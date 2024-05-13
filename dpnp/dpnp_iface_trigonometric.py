# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2024, Intel Corporation
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
# pylint: disable=c-extension-no-member
# pylint: disable=duplicate-code
# pylint: disable=no-name-in-module


import dpctl.tensor as dpt
import dpctl.tensor._tensor_elementwise_impl as ti
import dpctl.tensor._type_utils as dtu
import numpy

import dpnp
import dpnp.backend.extensions.vm._vm_impl as vmi

from .dpnp_algo import (
    dpnp_degrees,
    dpnp_radians,
    dpnp_unwrap,
)
from .dpnp_algo.dpnp_elementwise_common import DPNPBinaryFunc, DPNPUnaryFunc
from .dpnp_utils import call_origin
from .dpnp_utils.dpnp_utils_reduction import dpnp_wrap_reduction_call

__all__ = [
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
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


def _get_accumulation_res_dt(a, dtype, _out):
    """Get a dtype used by dpctl for result array in accumulation function."""

    if dtype is None:
        return dtu._default_accumulation_dtype_fp_types(a.dtype, a.sycl_queue)

    dtype = dpnp.dtype(dtype)
    return dtu._to_device_supported_dtype(dtype, a.sycl_device)


_ACOS_DOCSTRING = """
Computes inverse cosine for each element `x_i` for input array `x`.

For full documentation refer to :obj:`numpy.arccos`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have numeric data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise inverse cosine, in radians
    and in the closed interval `[-pi/2, pi/2]`. The data type
    of the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.cos` : Trigonometric cosine, element-wise.
:obj:`dpnp.arctan` : Trigonometric inverse tangent, element-wise.
:obj:`dpnp.arcsin` : Trigonometric inverse sine, element-wise.
:obj:`dpnp.arccosh` : Hyperbolic inverse cosine, element-wise.

Examples
--------
>>> import dpnp as np
>>> x = np.array([1, -1])
>>> np.arccos(x)
array([0.0,  3.14159265])
"""

arccos = DPNPUnaryFunc(
    "arccos",
    ti._acos_result_type,
    ti._acos,
    _ACOS_DOCSTRING,
    mkl_fn_to_call=vmi._mkl_acos_to_call,
    mkl_impl_fn=vmi._acos,
)


_ACOSH_DOCSTRING = """
Computes inverse hyperbolic cosine for each element `x_i` for input array `x`.

For full documentation refer to :obj:`numpy.arccosh`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have numeric data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise inverse hyperbolic cosine, in
    radians and in the half-closed interval `[0, inf)`. The data type
    of the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.cosh` : Hyperbolic cosine, element-wise.
:obj:`dpnp.arctanh` : Hyperbolic inverse tangent, element-wise.
:obj:`dpnp.arcsinh` : Hyperbolic inverse sine, element-wise.
:obj:`dpnp.arccos` : Trigonometric inverse cosine, element-wise.

Examples
--------
>>> import dpnp as np
>>> x = np.array([1.0, np.e, 10.0])
>>> np.arccosh(x)
array([0.0, 1.65745445, 2.99322285])
"""

arccosh = DPNPUnaryFunc(
    "arccosh",
    ti._acosh_result_type,
    ti._acosh,
    _ACOSH_DOCSTRING,
    mkl_fn_to_call=vmi._mkl_acosh_to_call,
    mkl_impl_fn=vmi._acosh,
)


_ASIN_DOCSTRING = """
Computes inverse sine for each element `x_i` for input array `x`.

For full documentation refer to :obj:`numpy.arcsin`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have numeric data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise inverse sine, in radians
    and in the closed interval `[-pi/2, pi/2]`. The data type
    of the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.sin` : Trigonometric sine, element-wise.
:obj:`dpnp.arctan` : Trigonometric inverse tangent, element-wise.
:obj:`dpnp.arccos` : Trigonometric inverse cosine, element-wise.
:obj:`dpnp.arcsinh` : Hyperbolic inverse sine, element-wise.

Examples
--------
>>> import dpnp as np
>>> x = np.array([0, 1, -1])
>>> np.arcsin(x)
array([0.0, 1.5707963267948966, -1.5707963267948966])
"""

arcsin = DPNPUnaryFunc(
    "arcsin",
    ti._asin_result_type,
    ti._asin,
    _ASIN_DOCSTRING,
    mkl_fn_to_call=vmi._mkl_asin_to_call,
    mkl_impl_fn=vmi._asin,
)


_ASINH_DOCSTRING = """
Computes inverse hyperbolic sine for each element `x_i` for input array `x`.

For full documentation refer to :obj:`numpy.arcsinh`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have numeric data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type..
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise inverse hyperbolic sine.
    The data type of the returned array is determined by
    the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.sinh` : Hyperbolic sine, element-wise.
:obj:`dpnp.arctanh` : Hyperbolic inverse tangent, element-wise.
:obj:`dpnp.arccosh` : Hyperbolic inverse cosine, element-wise.
:obj:`dpnp.arcsin` : Trigonometric inverse sine, element-wise.

Examples
--------
>>> import dpnp as np
>>> x = np.array([np.e, 10.0])
>>> np.arcsinh(x)
array([1.72538256, 2.99822295])
"""

arcsinh = DPNPUnaryFunc(
    "arcsinh",
    ti._asinh_result_type,
    ti._asinh,
    _ASINH_DOCSTRING,
    mkl_fn_to_call=vmi._mkl_asinh_to_call,
    mkl_impl_fn=vmi._asinh,
)


_ATAN_DOCSTRING = """
Computes inverse tangent for each element `x_i` for input array `x`.

For full documentation refer to :obj:`numpy.arctan`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have numeric data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type..
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise inverse tangent, in radians
    and in the closed interval `[-pi/2, pi/2]`. The data type
    of the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.arctan2` : Element-wise arc tangent of `x1/x2` choosing the quadrant correctly.
:obj:`dpnp.angle` : Argument of complex values.
:obj:`dpnp.tan` : Trigonometric tangent, element-wise.
:obj:`dpnp.arcsin` : Trigonometric inverse sine, element-wise.
:obj:`dpnp.arccos` : Trigonometric inverse cosine, element-wise.
:obj:`dpnp.arctanh` : Inverse hyperbolic tangent, element-wise.

Examples
--------
>>> import dpnp as np
>>> x = np.array([0, 1])
>>> np.arctan(x)
array([0.0, 0.78539816])
"""

arctan = DPNPUnaryFunc(
    "arctan",
    ti._atan_result_type,
    ti._atan,
    _ATAN_DOCSTRING,
    mkl_fn_to_call=vmi._mkl_atan_to_call,
    mkl_impl_fn=vmi._atan,
)


_ATAN2_DOCSTRING = """
Calculates the inverse tangent of the quotient `x1_i/x2_i` for each element
`x1_i` of the input array `x1` with the respective element `x2_i` of the
input array `x2`. Each element-wise result is expressed in radians.

For full documentation refer to :obj:`numpy.arctan2`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray}
    First input array, expected to have a real-valued floating-point
    data type.
x2 : {dpnp.ndarray, usm_ndarray}
    Second input array, also expected to have a real-valued
    floating-point data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the inverse tangent of the quotient `x1`/`x2`.
    The returned array must have a real-valued floating-point data type
    determined by Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword arguments `kwargs` are currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.arctan` : Trigonometric inverse tangent, element-wise.
:obj:`dpnp.tan` : Compute tangent element-wise.
:obj:`dpnp.angle` : Return the angle of the complex argument.
:obj:`dpnp.arcsin` : Trigonometric inverse sine, element-wise.
:obj:`dpnp.arccos` : Trigonometric inverse cosine, element-wise.
:obj:`dpnp.arctanh` : Inverse hyperbolic tangent, element-wise.

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([1., -1.])
>>> x2 = np.array([0., 0.])
>>> np.arctan2(x1, x2)
array([1.57079633, -1.57079633])

>>> x1 = np.array([0., 0., np.inf])
>>> x2 = np.array([+0., -0., np.inf])
>>> np.arctan2(x1, x2)
array([0.0 , 3.14159265, 0.78539816])

>>> x1 = np.array([-1, +1, +1, -1])
>>> x2 = np.array([-1, -1, +1, +1])
>>> np.arctan2(x1, x2) * 180 / np.pi
array([-135.,  -45.,   45.,  135.])
"""

arctan2 = DPNPBinaryFunc(
    "arctan2",
    ti._atan2_result_type,
    ti._atan2,
    _ATAN2_DOCSTRING,
    mkl_fn_to_call=vmi._mkl_atan2_to_call,
    mkl_impl_fn=vmi._atan2,
)


_ATANH_DOCSTRING = """
Computes hyperbolic inverse tangent for each element `x_i` for input array `x`.

For full documentation refer to :obj:`numpy.arctanh`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have numeric data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise hyperbolic inverse tangent.
    The data type of the returned array is determined by
    the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.tanh` : Hyperbolic tangent, element-wise.
:obj:`dpnp.arcsinh` : Hyperbolic inverse sine, element-wise.
:obj:`dpnp.arccosh` : Hyperbolic inverse cosine, element-wise.
:obj:`dpnp.arctan` : Trigonometric inverse tangent, element-wise.

Examples
--------
>>> import dpnp as np
>>> x = np.array([0, -0.5])
>>> np.arctanh(x)
array([0.0, -0.54930614])
"""

arctanh = DPNPUnaryFunc(
    "arctanh",
    ti._atanh_result_type,
    ti._atanh,
    _ATANH_DOCSTRING,
    mkl_fn_to_call=vmi._mkl_atanh_to_call,
    mkl_impl_fn=vmi._atanh,
)


_CBRT_DOCSTRING = """
Computes positive cube-root for each element `x_i` for input array `x`.

For full documentation refer to :obj:`numpy.cbrt`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a real-valued data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise positive cube-root.
    The data type of the returned array is determined by
    the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.sqrt` : Return the positive square-root of an array, element-wise.

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
    mkl_fn_to_call=vmi._mkl_cbrt_to_call,
    mkl_impl_fn=vmi._cbrt,
)


_COS_DOCSTRING = """
Computes cosine for each element `x_i` for input array `x`.

For full documentation refer to :obj:`numpy.cos`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have numeric data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise cosine. The data type
    of the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.arccos` : Trigonometric inverse cosine, element-wise.
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
    mkl_fn_to_call=vmi._mkl_cos_to_call,
    mkl_impl_fn=vmi._cos,
)


_COSH_DOCSTRING = """
Computes hyperbolic cosine for each element `x_i` for input array `x`.

For full documentation refer to :obj:`numpy.cosh`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have numeric data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise hyperbolic cosine. The data type
    of the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.arccosh` : Hyperbolic inverse cosine, element-wise.
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
    mkl_fn_to_call=vmi._mkl_cosh_to_call,
    mkl_impl_fn=vmi._cosh,
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
        Input array, expected to have a real-valued data type.
    axis : {None, int}, optional
        Axis or axes along which values must be computed. If a tuple of unique
        integers, values are computed over multiple axes. If ``None``, the
        result is computed over the entire array.
        Default: ``None``.
    dtype : {None, dtype}, optional
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
        A boolean indicating whether to include the initial value (i.e., the
        additive identity, zero) as the first value along the provided axis in
        the output.
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

    Note
    ----
    This function is equivalent of `numpy.logaddexp.accumulate`.

    See Also
    --------
    :obj:`dpnp.logsumexp` : Logarithm of the sum of elements of the inputs,
                            element-wise.

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
        x,
        out,
        dpt.cumulative_logsumexp,
        _get_accumulation_res_dt,
        usm_x,
        axis=axis,
        dtype=dtype,
        include_initial=include_initial,
    )


def deg2rad(x1):
    """
    Convert angles from degrees to radians.

    For full documentation refer to :obj:`numpy.deg2rad`.

    See Also
    --------
    :obj:`dpnp.rad2deg` : Convert angles from radians to degrees.
    :obj:`dpnp.unwrap` : Remove large jumps in angle by wrapping.

    Notes
    -----
    This function works exactly the same as :obj:`dpnp.radians`.

    """

    return radians(x1)


def degrees(x1, **kwargs):
    """
    Convert angles from radians to degrees.

    For full documentation refer to :obj:`numpy.degrees`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    .. seealso:: :obj:`dpnp.rad2deg` convert angles from radians to degrees.

    Examples
    --------
    >>> import dpnp as np
    >>> rad = np.arange(6.) * np.pi/6
    >>> out = np.degrees(rad)
    >>> [i for i in out]
    [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]

    """

    x1_desc = dpnp.get_dpnp_descriptor(
        x1, copy_when_strides=False, copy_when_nondefault_queue=False
    )
    if kwargs:
        pass
    elif x1_desc:
        return dpnp_degrees(x1_desc).get_pyobj()

    return call_origin(numpy.degrees, x1, **kwargs)


_EXP_DOCSTRING = """
Computes the exponent for each element `x_i` of input array `x`.

For full documentation refer to :obj:`numpy.exp`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have numeric data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise exponent of `x`.
    The data type of the returned array is determined by
    the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.expm1` : Calculate ``exp(x) - 1`` for all elements in the array.
:obj:`dpnp.exp2` : Calculate `2**x` for all elements in the array.

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
    mkl_fn_to_call=vmi._mkl_exp_to_call,
    mkl_impl_fn=vmi._exp,
)


_EXP2_DOCSTRING = """
Computes the base-2 exponent for each element `x_i` for input array `x`.

For full documentation refer to :obj:`numpy.exp2`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a floating-point data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise base-2 exponents.
    The data type of the returned array is determined by
    the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.exp` : Calculate exponent for all elements in the array.
:obj:`dpnp.expm1` : ``exp(x) - 1``, the inverse of :obj:`dpnp.log1p`.
:obj:`dpnp.power` : First array elements raised to powers from second array, element-wise.

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
    mkl_fn_to_call=vmi._mkl_exp2_to_call,
    mkl_impl_fn=vmi._exp2,
)


_EXPM1_DOCSTRING = """
Computes the exponent minus 1 for each element `x_i` of input array `x`.

This function calculates `exp(x) - 1.0` more accurately for small values of `x`.

For full documentation refer to :obj:`numpy.expm1`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have numeric data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise `exp(x) - 1` results.
    The data type of the returned array is determined by the Type
    Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.exp` : Calculate exponents for all elements in the array.
:obj:`dpnp.exp2` : Calculate `2**x` for all elements in the array.
:obj:`dpnp.log1p` : Calculate ``log(1 + x)``, the inverse of :obj:`dpnp.expm1`.

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
    mkl_fn_to_call=vmi._mkl_expm1_to_call,
    mkl_impl_fn=vmi._expm1,
)


_HYPOT_DOCSTRING = """
Calculates the hypotenuse for a right triangle with "legs" `x1_i` and `x2_i` of
input arrays `x1` and `x2`.

For full documentation refer to :obj:`numpy.hypot`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray}
    First input array, expected to have a real-valued data type.
x2 : {dpnp.ndarray, usm_ndarray}
    Second input array, also expected to have a real-valued data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise hypotenuse. The data type
    of the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.reduce_hypot` : The square root of the sum of squares of elements in the input array.

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
    mkl_fn_to_call=vmi._mkl_hypot_to_call,
    mkl_impl_fn=vmi._hypot,
)


_LOG_DOCSTRING = """
Computes the natural logarithm for each element `x_i` of input array `x`.

For full documentation refer to :obj:`numpy.log`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have numeric data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise natural logarithm values.
    The data type of the returned array is determined by the Type
    Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.log10` : Return the base 10 logarithm of the input array,
                    element-wise.
:obj:`dpnp.log2` : Base-2 logarithm of x.
:obj:`dpnp.log1p` : Return the natural logarithm of one plus
                    the input array, element-wise.

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
    mkl_fn_to_call=vmi._mkl_ln_to_call,
    mkl_impl_fn=vmi._ln,
)


_LOG10_DOCSTRING = """
Computes the base-10 logarithm for each element `x_i` of input array `x`.

For full documentation refer to :obj:`numpy.log10`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have numeric data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise base-10 logarithm of `x`.
    The data type of the returned array is determined by the
    Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.log` : Natural logarithm, element-wise.
:obj:`dpnp.log2` : Return the base 2 logarithm of the input array, element-wise.
:obj:`dpnp.log1p` : Return the natural logarithm of one plus the input array, element-wise.

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
    mkl_fn_to_call=vmi._mkl_log10_to_call,
    mkl_impl_fn=vmi._log10,
)


_LOG1P_DOCSTRING = """
Computes the natural logarithm of (1 + `x`) for each element `x_i` of input
array `x`.

This function calculates `log(1 + x)` more accurately for small values of `x`.

For full documentation refer to :obj:`numpy.log1p`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have numeric data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise `log(1 + x)` results. The data type
    of the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.expm1` : ``exp(x) - 1``, the inverse of :obj:`dpnp.log1p`.
:obj:`dpnp.log` : Natural logarithm, element-wise.
:obj:`dpnp.log10` : Return the base 10 logarithm of the input array, element-wise.
:obj:`dpnp.log2` : Return the base 2 logarithm of the input array, element-wise.

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
    mkl_fn_to_call=vmi._mkl_log1p_to_call,
    mkl_impl_fn=vmi._log1p,
)


_LOG2_DOCSTRING = """
Computes the base-2 logarithm for each element `x_i` of input array `x`.

For full documentation refer to :obj:`numpy.log2`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have numeric data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise base-2 logarithm of `x`.
    The data type of the returned array is determined by the
    Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.log` : Natural logarithm, element-wise.
:obj:`dpnp.log10` : Return the base 10 logarithm of the input array, element-wise.
:obj:`dpnp.log1p` : Return the natural logarithm of one plus the input array, element-wise.

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
    mkl_fn_to_call=vmi._mkl_log2_to_call,
    mkl_impl_fn=vmi._log2,
)


_LOGADDEXP_DOCSTRING = """
Calculates the natural logarithm of the sum of exponents for each element `x1_i`
of the input array `x1` with the respective element `x2_i` of the input
array `x2`.

This function calculates `log(exp(x1) + exp(x2))` more accurately for small
values of `x`.

For full documentation refer to :obj:`numpy.logaddexp`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray}
    First input array, expected to have a real-valued floating-point
    data type.
x2 : {dpnp.ndarray, usm_ndarray}
    Second input array, also expected to have a real-valued
    floating-point data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise results. The data type
    of the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword arguments `kwargs` are currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.log` : Natural logarithm, element-wise.
:obj:`dpnp.exp` : Exponential, element-wise.
:obj:`dpnp.logsumdexp` : Logarithm of the sum of exponents of elements in the input array.

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


def logsumexp(x, /, *, axis=None, dtype=None, keepdims=False, out=None):
    """
    Calculates the logarithm of the sum of exponents of elements in
    the input array.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        Input array, expected to have a real-valued data type.
    axis : {None, int or tuple of ints}, optional
        Axis or axes along which values must be computed. If a tuple of unique
        integers, values are computed over multiple axes. If ``None``, the
        result is computed over the entire array.
        Default: ``None``.
    dtype : {None, dtype}, optional
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

    Note
    ----
    This function is equivalent of `numpy.logaddexp.reduce`.

    See Also
    --------
    :obj:`dpnp.log` : Natural logarithm, element-wise.
    :obj:`dpnp.exp` : Exponential, element-wise.
    :obj:`dpnp.logaddexp` : Logarithm of the sum of exponents of
                            the inputs, element-wise.

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
        x,
        out,
        dpt.logsumexp,
        _get_accumulation_res_dt,
        usm_x,
        axis=axis,
        dtype=dtype,
        keepdims=keepdims,
    )


_RECIPROCAL_DOCSTRING = """
Computes the reciprocal square-root for each element `x_i` for input array `x`.

For full documentation refer to :obj:`numpy.reciprocal`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a real-valued floating-point data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise reciprocals.
    The returned array has a floating-point data type determined
    by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.rsqrt` : Return the reciprocal square-root of an array, element-wise.

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
)


def reduce_hypot(x, /, *, axis=None, dtype=None, keepdims=False, out=None):
    """
    Calculates the square root of the sum of squares of elements in
    the input array.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        Input array, expected to have a real-valued data type.
    axis : {None, int or tuple of ints}, optional
        Axis or axes along which values must be computed. If a tuple of unique
        integers, values are computed over multiple axes. If ``None``, the
        result is computed over the entire array.
        Default: ``None``.
    dtype : {None, dtype}, optional
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

    Note
    ----
    This function is equivalent of `numpy.hypot.reduce`.

    See Also
    --------
    :obj:`dpnp.hypot` : Given the "legs" of a right triangle, return its
                        hypotenuse.

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
        x,
        out,
        dpt.reduce_hypot,
        _get_accumulation_res_dt,
        usm_x,
        axis=axis,
        dtype=dtype,
        keepdims=keepdims,
    )


_RSQRT_DOCSTRING = """
Computes the reciprocal square-root for each element `x_i` for input array `x`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a real floating-point data type.
out : ({None, dpnp.ndarray, usm_ndarray}, optional):
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : ({'C', 'F', 'A', 'K'}, optional):
    Memory layout of the newly output array, if parameter `out` is `None`.
    Default: "K"

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise reciprocal square-root.
    The returned array has a floating-point data type determined by
    the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.sqrt` : Return the positive square-root of an array, element-wise.
:obj:`dpnp.reciprocal` : Return the reciprocal of an array, element-wise.

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


def rad2deg(x1):
    """
    Convert angles from radians to degrees.

    For full documentation refer to :obj:`numpy.rad2deg`.

    See Also
    --------
    :obj:`dpnp.deg2rad` : Convert angles from degrees to radians.
    :obj:`dpnp.unwrap` : Remove large jumps in angle by wrapping.

    Notes
    -----
    This function works exactly the same as :obj:`dpnp.degrees`.

    """

    return degrees(x1)


def radians(x1, **kwargs):
    """
    Convert angles from degrees to radians.

    For full documentation refer to :obj:`numpy.radians`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    .. seealso:: :obj:`dpnp.deg2rad` equivalent function.

    Examples
    --------
    >>> import dpnp as np
    >>> deg = np.arange(6.) * 30.
    >>> out = np.radians(deg)
    >>> [i for i in out]
    [0.0, 0.52359878, 1.04719755, 1.57079633, 2.0943951, 2.61799388]

    """

    x1_desc = dpnp.get_dpnp_descriptor(
        x1, copy_when_strides=False, copy_when_nondefault_queue=False
    )
    if kwargs:
        pass
    elif x1_desc:
        return dpnp_radians(x1_desc).get_pyobj()

    return call_origin(numpy.radians, x1, **kwargs)


_SIN_DOCSTRING = """
Computes sine for each element `x_i` of input array `x`.

For full documentation refer to :obj:`numpy.sin`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have numeric data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise sine. The data type of the
    returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.arcsin` : Trigonometric inverse sine, element-wise.
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
    mkl_fn_to_call=vmi._mkl_sin_to_call,
    mkl_impl_fn=vmi._sin,
)


_SINH_DOCSTRING = """
Computes hyperbolic sine for each element `x_i` for input array `x`.

For full documentation refer to :obj:`numpy.sinh`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have numeric data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise hyperbolic sine. The data type
    of the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.arcsinh` : Hyperbolic inverse sine, element-wise.
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
    mkl_fn_to_call=vmi._mkl_sinh_to_call,
    mkl_impl_fn=vmi._sinh,
)


_SQRT_DOCSTRING = """
Computes the positive square-root for each element `x_i` of input array `x`.

For full documentation refer to :obj:`numpy.sqrt`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise positive square-roots of `x`. The
    data type of the returned array is determined by the Type Promotion
    Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.cbrt` : Return the cube-root of an array, element-wise.
:obj:`dpnp.rsqrt` : Return the reciprocal square-root of an array, element-wise.

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
    mkl_fn_to_call=vmi._mkl_sqrt_to_call,
    mkl_impl_fn=vmi._sqrt,
)


_SQUARE_DOCSTRING = """
Squares each element `x_i` of input array `x`.

For full documentation refer to :obj:`numpy.square`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise squares of `x`. The data type of
    the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp..linalg.matrix_power` : Raise a square matrix
                                    to the (integer) power `n`.
:obj:`dpnp.sqrt` : Return the positive square-root of an array,
                    element-wise.
:obj:`dpnp.power` : First array elements raised to powers
                    from second array, element-wise.

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
    mkl_fn_to_call=vmi._mkl_sqr_to_call,
    mkl_impl_fn=vmi._sqr,
)


_TAN_DOCSTRING = """
Computes tangent for each element `x_i` for input array `x`.

For full documentation refer to :obj:`numpy.tan`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have numeric data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise tangent. The data type
    of the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.arctan` : Trigonometric inverse tangent, element-wise.
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
    mkl_fn_to_call=vmi._mkl_tan_to_call,
    mkl_impl_fn=vmi._tan,
)


_TANH_DOCSTRING = """
Computes hyperbolic tangent for each element `x_i` for input array `x`.

For full documentation refer to :obj:`numpy.tanh`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have numeric data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise hyperbolic tangent. The data type
    of the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.arctanh` : Hyperbolic inverse tangent, element-wise.
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
    mkl_fn_to_call=vmi._mkl_tanh_to_call,
    mkl_impl_fn=vmi._tanh,
)


def unwrap(x1, **kwargs):
    """
    Unwrap by changing deltas between values to 2*pi complement.

    For full documentation refer to :obj:`numpy.unwrap`.

    Limitations
    -----------
    Input array is supported as :class:`dpnp.ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.rad2deg` : Convert angles from radians to degrees.
    :obj:`dpnp.deg2rad` : Convert angles from degrees to radians.

    Examples
    --------
    >>> import dpnp as np
    >>> phase = np.linspace(0, np.pi, num=5)
    >>> for i in range(3, 5):
    >>>     phase[i] += np.pi
    >>> out = np.unwrap(phase)
    >>> [i for i in out]
    [0.0, 0.78539816, 1.57079633, 5.49778714, 6.28318531]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if kwargs:
        pass
    elif x1_desc:
        return dpnp_unwrap(x1_desc).get_pyobj()

    return call_origin(numpy.unwrap, x1, **kwargs)
