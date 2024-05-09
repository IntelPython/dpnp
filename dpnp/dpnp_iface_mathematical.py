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
Interface of the Mathematical part of the DPNP

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
from dpctl.tensor._type_utils import _acceptance_fn_divide
from numpy.core.numeric import (
    normalize_axis_index,
    normalize_axis_tuple,
)

import dpnp
import dpnp.backend.extensions.vm._vm_impl as vmi

from .backend.extensions.sycl_ext import _sycl_ext_impl
from .dpnp_algo import (
    dpnp_ediff1d,
    dpnp_fabs,
    dpnp_fmax,
    dpnp_fmin,
    dpnp_fmod,
    dpnp_gradient,
    dpnp_modf,
    dpnp_trapz,
)
from .dpnp_algo.dpnp_elementwise_common import (
    DPNPAngle,
    DPNPBinaryFunc,
    DPNPReal,
    DPNPRound,
    DPNPUnaryFunc,
    acceptance_fn_negative,
    acceptance_fn_positive,
    acceptance_fn_sign,
    acceptance_fn_subtract,
)
from .dpnp_array import dpnp_array
from .dpnp_utils import call_origin, get_usm_allocations
from .dpnp_utils.dpnp_utils_linearalgebra import dpnp_cross
from .dpnp_utils.dpnp_utils_reduction import dpnp_wrap_reduction_call

__all__ = [
    "abs",
    "absolute",
    "add",
    "angle",
    "around",
    "ceil",
    "clip",
    "conj",
    "conjugate",
    "convolve",
    "copysign",
    "cross",
    "cumprod",
    "cumsum",
    "diff",
    "divide",
    "ediff1d",
    "fabs",
    "floor",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "gradient",
    "imag",
    "maximum",
    "minimum",
    "mod",
    "modf",
    "multiply",
    "negative",
    "positive",
    "power",
    "prod",
    "proj",
    "real",
    "remainder",
    "rint",
    "round",
    "sign",
    "signbit",
    "subtract",
    "sum",
    "trapz",
    "true_divide",
    "trunc",
]


def _append_to_diff_array(a, axis, combined, values):
    """
    Append `values` to `combined` list based on data of array `a`.

    Scalar value (including case with 0d array) is expanded to an array
    with length=1 in the direction of axis and the shape of the input array `a`
    along all other axes.
    Note, if `values` is a scalar, then it is converted to 0d array allocating
    on the same SYCL queue as the input array `a` and with the same USM type.

    """

    dpnp.check_supported_arrays_type(values, scalar_type=True, all_scalars=True)
    if dpnp.isscalar(values):
        values = dpnp.asarray(
            values, sycl_queue=a.sycl_queue, usm_type=a.usm_type
        )

    if values.ndim == 0:
        shape = list(a.shape)
        shape[axis] = 1
        values = dpnp.broadcast_to(values, tuple(shape))
    combined.append(values)


def _get_reduction_res_dt(a, dtype, _out):
    """Get a data type used by dpctl for result array in reduction function."""

    if dtype is None:
        return dtu._default_accumulation_dtype(a.dtype, a.sycl_queue)

    dtype = dpnp.dtype(dtype)
    return dtu._to_device_supported_dtype(dtype, a.sycl_device)


_ABS_DOCSTRING = """
Calculates the absolute value for each element `x_i` of input array `x`.

For full documentation refer to :obj:`numpy.absolute`.

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
    An array containing the element-wise absolute values.
    For complex input, the absolute value is its magnitude.
    If `x` has a real-valued data type, the returned array has the
    same data type as `x`. If `x` has a complex floating-point data type,
    the returned array has a real-valued floating-point data type whose
    precision matches the precision of `x`.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.fabs` : Calculate the absolute value element-wise excluding complex types.

Notes
-----
``dpnp.abs`` is a shorthand for this function.

Examples
--------
>>> import dpnp as np
>>> a = np.array([-1.2, 1.2])
>>> np.absolute(a)
array([1.2, 1.2])

>>> a = np.array(1.2 + 1j)
>>> np.absolute(a)
array(1.5620499351813308)
"""

absolute = DPNPUnaryFunc(
    "abs",
    ti._abs_result_type,
    ti._abs,
    _ABS_DOCSTRING,
    mkl_fn_to_call=vmi._mkl_abs_to_call,
    mkl_impl_fn=vmi._abs,
)


abs = absolute


_ADD_DOCSTRING = """
Calculates the sum for each element `x1_i` of the input array `x1` with
the respective element `x2_i` of the input array `x2`.

For full documentation refer to :obj:`numpy.add`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray}
    First input array, expected to have numeric data type.
x2 : {dpnp.ndarray, usm_ndarray}
    Second input array, also expected to have numeric data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise sums. The data type of the
    returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

Notes
-----
Equivalent to `x1` + `x2` in terms of array broadcasting.

Examples
--------
>>> import dpnp as np
>>> a = np.array([1, 2, 3])
>>> b = np.array([1, 2, 3])
>>> np.add(a, b)
array([2, 4, 6])

>>> x1 = np.arange(9.0).reshape((3, 3))
>>> x2 = np.arange(3.0)
>>> np.add(x1, x2)
array([[  0.,   2.,   4.],
       [  3.,   5.,   7.],
       [  6.,   8.,  10.]])

The ``+`` operator can be used as a shorthand for ``add`` on
:class:`dpnp.ndarray`.

>>> x1 + x2
array([[  0.,   2.,   4.],
       [  3.,   5.,   7.],
       [  6.,   8.,  10.]])
"""


add = DPNPBinaryFunc(
    "add",
    ti._add_result_type,
    ti._add,
    _ADD_DOCSTRING,
    mkl_fn_to_call=vmi._mkl_add_to_call,
    mkl_impl_fn=vmi._add,
    binary_inplace_fn=ti._add_inplace,
)


_ANGLE_DOCSTRING = """
Computes the phase angle (also called the argument) of each element `x_i` for
input array `x`.

For full documentation refer to :obj:`numpy.angle`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a complex-valued floating-point data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise phase angles.
    The returned array has a floating-point data type determined
    by the Type Promotion Rules.

Notes
-----
Although the angle of the complex number 0 is undefined, `dpnp.angle(0)` returns the value 0.

See Also
--------
:obj:`dpnp.arctan2` : Element-wise arc tangent of `x1/x2` choosing the quadrant correctly.
:obj:`dpnp.arctan` : Trigonometric inverse tangent, element-wise.
:obj:`dpnp.absolute` : Calculate the absolute value element-wise.

Examples
--------
>>> import dpnp as np
>>> a = np.array([1.0, 1.0j, 1+1j])
>>> np.angle(a) # in radians
array([0.        , 1.57079633, 0.78539816]) # may vary

>>> np.angle(a, deg=True) # in degrees
array([ 0., 90., 45.])
"""

angle = DPNPAngle(
    "angle",
    ti._angle_result_type,
    ti._angle,
    _ANGLE_DOCSTRING,
)


def around(x, /, decimals=0, out=None):
    """
    Round an array to the given number of decimals.

    For full documentation refer to :obj:`numpy.around`.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        Input array, expected to have numeric data type.
    decimals : int, optional
        Number of decimal places to round to (default: 0). If decimals is
        negative, it specifies the number of positions to the left of the
        decimal point.
    out : {None, dpnp.ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.


    Returns
    -------
    out : dpnp.ndarray
        The rounded value of elements of the array.

    See Also
    --------
    :obj:`dpnp.round` : Equivalent function; see for details.
    :obj:`dpnp.ndarray.round` : Equivalent function.
    :obj:`dpnp.rint` : Round elements of the array to the nearest integer.
    :obj:`dpnp.ceil` : Compute the ceiling of the input, element-wise.
    :obj:`dpnp.floor` : Return the floor of the input, element-wise.
    :obj:`dpnp.trunc` : Return the truncated value of the input, element-wise.

    Notes
    -----
    This function works the same as :obj:`dpnp.round`.
    """

    return round(x, decimals, out)


_CEIL_DOCSTRING = """
Returns the ceiling for each element `x_i` for input array `x`.

For full documentation refer to :obj:`numpy.ceil`.

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
    An array containing the element-wise ceiling.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.floor` : Return the floor of the input, element-wise.
:obj:`dpnp.trunc` : Return the truncated value of the input, element-wise.

Examples
--------
>>> import dpnp as np
>>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
>>> np.ceil(a)
array([-1.0, -1.0, -0.0, 1.0, 2.0, 2.0, 2.0])
"""

ceil = DPNPUnaryFunc(
    "ceil",
    ti._ceil_result_type,
    ti._ceil,
    _CEIL_DOCSTRING,
    mkl_fn_to_call=vmi._mkl_ceil_to_call,
    mkl_impl_fn=vmi._ceil,
)


def clip(a, a_min, a_max, *, out=None, order="K", **kwargs):
    """
    Clip (limit) the values in an array.

    For full documentation refer to :obj:`numpy.clip`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Array containing elements to clip.
    a_min, a_max : {dpnp.ndarray, usm_ndarray, None}
        Minimum and maximum value. If ``None``, clipping is not performed on
        the corresponding edge. Only one of `a_min` and `a_max` may be
        ``None``. Both are broadcast against `a`.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        The results will be placed in this array. It may be the input array
        for in-place clipping. `out` must be of the right shape to hold the
        output. Its type is preserved.
    order : {"C", "F", "A", "K", None}, optional
        Memory layout of the newly output array, if parameter `out` is `None`.
        If `order` is ``None``, the default value "K" will be used.

    Returns
    -------
    out : dpnp.ndarray
        An array with the elements of `a`, but where values < `a_min` are
        replaced with `a_min`, and those > `a_max` with `a_max`.

    Limitations
    -----------
    Keyword argument `kwargs` is currently unsupported.
    Otherwise ``NotImplementedError`` exception will be raised.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.arange(10)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> np.clip(a, 1, 8)
    array([1, 1, 2, 3, 4, 5, 6, 7, 8, 8])
    >>> np.clip(a, 8, 1)
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    >>> np.clip(a, 3, 6, out=a)
    array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])
    >>> a
    array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])

    >>> a = np.arange(10)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> min = np.array([3, 4, 1, 1, 1, 4, 4, 4, 4, 4])
    >>> np.clip(a, min, 8)
    array([3, 4, 2, 3, 4, 5, 6, 7, 8, 8])

    """

    if kwargs:
        raise NotImplementedError(f"kwargs={kwargs} is currently not supported")
    if a_min is None and a_max is None:
        raise ValueError("One of max or min must be given")

    if order is None:
        order = "K"

    usm_arr = dpnp.get_usm_ndarray(a)
    usm_min = None if a_min is None else dpnp.get_usm_ndarray_or_scalar(a_min)
    usm_max = None if a_max is None else dpnp.get_usm_ndarray_or_scalar(a_max)

    usm_out = None if out is None else dpnp.get_usm_ndarray(out)
    usm_res = dpt.clip(usm_arr, usm_min, usm_max, out=usm_out, order=order)
    if out is not None and isinstance(out, dpnp_array):
        return out
    return dpnp_array._create_from_usm_ndarray(usm_res)


_CONJ_DOCSTRING = """
Computes conjugate of each element `x_i` for input array `x`.

For full documentation refer to :obj:`numpy.conj`.

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
    An array containing the element-wise conjugate values.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

Examples
--------
>>> import dpnp as np
>>> np.conjugate(np.array(1+2j))
(1-2j)

>>> x = np.eye(2) + 1j * np.eye(2)
>>> np.conjugate(x)
array([[ 1.-1.j,  0.-0.j],
       [ 0.-0.j,  1.-1.j]])
"""

conjugate = DPNPUnaryFunc(
    "conj",
    ti._conj_result_type,
    ti._conj,
    _CONJ_DOCSTRING,
    mkl_fn_to_call=vmi._mkl_conj_to_call,
    mkl_impl_fn=vmi._conj,
)

conj = conjugate


def convolve(a, v, mode="full"):
    """
    Returns the discrete, linear convolution of two one-dimensional sequences.

    For full documentation refer to :obj:`numpy.convolve`.

    Examples
    --------
    >>> ca = dpnp.convolve([1, 2, 3], [0, 1, 0.5])
    >>> print(ca)
    [0. , 1. , 2.5, 4. , 1.5]

    """

    return call_origin(numpy.convolve, a=a, v=v, mode=mode)


_COPYSING_DOCSTRING = """
Composes a floating-point value with the magnitude of `x1_i` and the sign of
`x2_i` for each element of input arrays `x1` and `x2`.

For full documentation refer to :obj:`numpy.copysign`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray}
    First input array, expected to have a real floating-point data type.
x2 : {dpnp.ndarray, usm_ndarray}
    Second input array, also expected to have a real floating-point data
    type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
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
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.negative` : Return the numerical negative of each element of `x`.
:obj:`dpnp.positive` : Return the numerical positive of each element of `x`.

Examples
--------
>>> import dpnp as np
>>> np.copysign(np.array(1.3), np.array(-1))
array(-1.3)
>>> 1 / np.copysign(np.array(0), 1)
array(inf)
>>> 1 / np.copysign(np.array(0), -1)
array(-inf)

>>> x = np.array([-1, 0, 1])
>>> np.copysign(x, -1.1)
array([-1., -0., -1.])
>>> np.copysign(x, np.arange(3) - 1)
array([-1., 0., 1.])
"""

copysign = DPNPBinaryFunc(
    "copysign",
    ti._copysign_result_type,
    ti._copysign,
    _COPYSING_DOCSTRING,
)


def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """
    Return the cross product of two (arrays of) vectors.

    For full documentation refer to :obj:`numpy.cross`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        First input array.
    b : {dpnp.ndarray, usm_ndarray}
        Second input array.
    axisa : int, optional
        Axis of `a` that defines the vector(s).  By default, the last axis.
    axisb : int, optional
        Axis of `b` that defines the vector(s).  By default, the last axis.
    axisc : int, optional
        Axis of `c` containing the cross product vector(s).  Ignored if
        both input vectors have dimension 2, as the return is scalar.
        By default, the last axis.
    axis : {int, None}, optional
        If defined, the axis of `a`, `b` and `c` that defines the vector(s)
        and cross product(s).  Overrides `axisa`, `axisb` and `axisc`.

    Returns
    -------
    out : dpnp.ndarray
        Vector cross product(s).

    See Also
    --------
    :obj:`dpnp.inner` : Inner product.
    :obj:`dpnp.outer` : Outer product.

    Examples
    --------
    Vector cross-product.

    >>> import dpnp as np
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([4, 5, 6])
    >>> np.cross(x, y)
    array([-3,  6, -3])

    One vector with dimension 2.

    >>> x = np.array([1, 2])
    >>> y = np.array([4, 5, 6])
    >>> np.cross(x, y)
    array([12, -6, -3])

    Equivalently:

    >>> x = np.array([1, 2, 0])
    >>> y = np.array([4, 5, 6])
    >>> np.cross(x, y)
    array([12, -6, -3])

    Both vectors with dimension 2.

    >>> x = np.array([1, 2])
    >>> y = np.array([4, 5])
    >>> np.cross(x, y)
    array(-3)

    Multiple vector cross-products. Note that the direction of the cross
    product vector is defined by the *right-hand rule*.

    >>> x = np.array([[1, 2, 3], [4, 5, 6]])
    >>> y = np.array([[4, 5, 6], [1, 2, 3]])
    >>> np.cross(x, y)
    array([[-3,  6, -3],
           [ 3, -6,  3]])

    The orientation of `c` can be changed using the `axisc` keyword.

    >>> np.cross(x, y, axisc=0)
    array([[-3,  3],
           [ 6, -6],
           [-3,  3]])

    Change the vector definition of `x` and `y` using `axisa` and `axisb`.

    >>> x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> y = np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]])
    >>> np.cross(x, y)
    array([[ -6,  12,  -6],
           [  0,   0,   0],
           [  6, -12,   6]])
    >>> np.cross(x, y, axisa=0, axisb=0)
    array([[-24,  48, -24],
           [-30,  60, -30],
           [-36,  72, -36]])

    """

    if axis is not None:
        if not isinstance(axis, int):
            raise TypeError(f"axis should be an integer but got, {type(axis)}.")
        axisa, axisb, axisc = (axis,) * 3
    dpnp.check_supported_arrays_type(a, b)
    if a.dtype == dpnp.bool and b.dtype == dpnp.bool:
        raise TypeError(
            "Input arrays with boolean data type are not supported."
        )
    # Check axisa and axisb are within bounds
    axisa = normalize_axis_index(axisa, a.ndim, msg_prefix="axisa")
    axisb = normalize_axis_index(axisb, b.ndim, msg_prefix="axisb")

    # Move working axis to the end of the shape
    a = dpnp.moveaxis(a, axisa, -1)
    b = dpnp.moveaxis(b, axisb, -1)
    if a.shape[-1] not in (2, 3) or b.shape[-1] not in (2, 3):
        raise ValueError(
            "Incompatible vector dimensions for cross product\n"
            "(the dimension of vector used in cross product must be 2 or 3)"
        )

    # Modify the shape of input arrays if necessary
    a_shape = a.shape
    b_shape = b.shape
    # TODO: replace with dpnp.broadcast_shapes once implemented
    res_shape = numpy.broadcast_shapes(a_shape[:-1], b_shape[:-1])
    if a_shape[:-1] != res_shape:
        a = dpnp.broadcast_to(a, res_shape + (a_shape[-1],))
        a_shape = a.shape
    if b_shape[:-1] != res_shape:
        b = dpnp.broadcast_to(b, res_shape + (b_shape[-1],))
        b_shape = b.shape

    if a_shape[-1] == 3 or b_shape[-1] == 3:
        res_shape += (3,)
        # Check axisc is within bounds
        axisc = normalize_axis_index(axisc, len(res_shape), msg_prefix="axisc")
    # Create the output array
    dtype = dpnp.result_type(a, b)
    res_usm_type, exec_q = get_usm_allocations([a, b])
    cp = dpnp.empty(
        res_shape, dtype=dtype, sycl_queue=exec_q, usm_type=res_usm_type
    )

    # recast arrays as dtype
    a = a.astype(dtype, copy=False)
    b = b.astype(dtype, copy=False)

    cp = dpnp_cross(a, b, cp, exec_q)
    if a_shape[-1] == 2 and b_shape[-1] == 2:
        return cp

    return dpnp.moveaxis(cp, -1, axisc)


def cumprod(a, axis=None, dtype=None, out=None):
    """
    Return the cumulative product of elements along a given axis.

    For full documentation refer to :obj:`numpy.cumprod`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int}, optional
        Axis along which the cumulative product is computed. It defaults to
        compute the cumulative product over the flattened array.
        Default: ``None``.
    dtype : {None, dtype}, optional
        Type of the returned array and of the accumulator in which the elements
        are multiplied. If `dtype` is not specified, it defaults to the dtype
        of `a`, unless `a` has an integer dtype with a precision less than that
        of the default platform integer. In that case, the default platform
        integer is used.
        Default: ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. It must have the
        same shape and buffer length as the expected output but the type will
        be cast if necessary.
        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        A new array holding the result is returned unless `out` is specified as
        :class:`dpnp.ndarray`, in which case a reference to `out` is returned.
        The result has the same size as `a`, and the same shape as `a` if `axis`
        is not ``None`` or `a` is a 1-d array.

    See Also
    --------
    :obj:`dpnp.prod` : Product array elements.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 2, 3])
    >>> np.cumprod(a) # intermediate results 1, 1*2
    ...               # total product 1*2*3 = 6
    array([1, 2, 6])
    >>> a = np.array([[1, 2, 3], [4, 5, 6]])
    >>> np.cumprod(a, dtype=np.float32) # specify type of output
    array([  1.,   2.,   6.,  24., 120., 720.], dtype=float32)

    The cumulative product for each column (i.e., over the rows) of `a`:

    >>> np.cumprod(a, axis=0)
    array([[ 1,  2,  3],
           [ 4, 10, 18]])

    The cumulative product for each row (i.e. over the columns) of `a`:

    >>> np.cumprod(a, axis=1)
    array([[  1,   2,   6],
           [  4,  20, 120]])

    """

    dpnp.check_supported_arrays_type(a)
    if a.ndim > 1 and axis is None:
        usm_a = dpnp.ravel(a).get_array()
    else:
        usm_a = dpnp.get_usm_ndarray(a)

    return dpnp_wrap_reduction_call(
        a,
        out,
        dpt.cumulative_prod,
        _get_reduction_res_dt,
        usm_a,
        axis=axis,
        dtype=dtype,
    )


def cumsum(a, axis=None, dtype=None, out=None):
    """
    Return the cumulative sum of the elements along a given axis.

    For full documentation refer to :obj:`numpy.cumsum`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int}, optional
        Axis along which the cumulative sum is computed. It defaults to compute
        the cumulative sum over the flattened array.
        Default: ``None``.
    dtype : {None, dtype}, optional
        Type of the returned array and of the accumulator in which the elements
        are summed. If `dtype` is not specified, it defaults to the dtype of
        `a`, unless `a` has an integer dtype with a precision less than that of
        the default platform integer. In that case, the default platform
        integer is used.
        Default: ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. It must have the
        same shape and buffer length as the expected output but the type will
        be cast if necessary.
        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        A new array holding the result is returned unless `out` is specified as
        :class:`dpnp.ndarray`, in which case a reference to `out` is returned.
        The result has the same size as `a`, and the same shape as `a` if `axis`
        is not ``None`` or `a` is a 1-d array.

    See Also
    --------
    :obj:`dpnp.sum` : Sum array elements.
    :obj:`dpnp.diff` : Calculate the n-th discrete difference along given axis.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[1, 2, 3], [4, 5, 6]])
    >>> a
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> np.cumsum(a)
    array([ 1,  3,  6, 10, 15, 21])
    >>> np.cumsum(a, dtype=float)     # specifies type of output value(s)
    array([ 1.,  3.,  6., 10., 15., 21.])

    >>> np.cumsum(a, axis=0)     # sum over rows for each of the 3 columns
    array([[1, 2, 3],
           [5, 7, 9]])
    >>> np.cumsum(a, axis=1)     # sum over columns for each of the 2 rows
    array([[ 1,  3,  6],
           [ 4,  9, 15]])

    ``cumsum(b)[-1]`` may not be equal to ``sum(b)``

    >>> b = np.array([1, 2e-9, 3e-9] * 10000)
    >>> b.cumsum().dtype == b.sum().dtype == np.float64
    True
    >>> b.cumsum()[-1] == b.sum()
    array(False)

    """

    dpnp.check_supported_arrays_type(a)
    if a.ndim > 1 and axis is None:
        usm_a = dpnp.ravel(a).get_array()
    else:
        usm_a = dpnp.get_usm_ndarray(a)

    return dpnp_wrap_reduction_call(
        a,
        out,
        dpt.cumulative_sum,
        _get_reduction_res_dt,
        usm_a,
        axis=axis,
        dtype=dtype,
    )


def diff(a, n=1, axis=-1, prepend=None, append=None):
    """
    Calculate the n-th discrete difference along the given axis.

    For full documentation refer to :obj:`numpy.diff`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array
    n : {int}, optional
        The number of times the values differ. If ``zero``, the input
        is returned as-is.
    axis : {int}, optional
        The axis along which the difference is taken, default is the
        last axis.
    prepend, append : {None, scalar, dpnp.ndarray, usm_ndarray}, optional
        Values to prepend or append to `a` along axis prior to
        performing the difference. Scalar values are expanded to
        arrays with length 1 in the direction of axis and the shape
        of the input array in along all other axes. Otherwise the
        dimension and shape must match `a` except along axis.

    Returns
    -------
    out : dpnp.ndarray
        The n-th differences. The shape of the output is the same as `a`
        except along `axis` where the dimension is smaller by `n`. The
        type of the output is the same as the type of the difference
        between any two elements of `a`. This is the same as the type of
        `a` in most cases.

    See Also
    --------
    :obj:`dpnp.gradient` : Return the gradient of an N-dimensional array.
    :obj:`dpnp.ediff1d` : Compute the differences between consecutive elements
                          of an array.
    :obj:`dpnp.cumsum` : Return the cumulative sum of the elements along
                         a given axis.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([1, 2, 4, 7, 0])
    >>> np.diff(x)
    array([ 1,  2,  3, -7])
    >>> np.diff(x, n=2)
    array([  1,   1, -10])

    >>> x = np.array([[1, 3, 6, 10], [0, 5, 6, 8]])
    >>> np.diff(x)
    array([[2, 3, 4],
           [5, 1, 2]])
    >>> np.diff(x, axis=0)
    array([[-1,  2,  0, -2]])

    """

    dpnp.check_supported_arrays_type(a)
    if n == 0:
        return a
    if n < 0:
        raise ValueError(f"order must be non-negative but got {n}")

    nd = a.ndim
    if nd == 0:
        raise ValueError("diff requires input that is at least one dimensional")
    axis = normalize_axis_index(axis, nd)

    combined = []
    if prepend is not None:
        _append_to_diff_array(a, axis, combined, prepend)

    combined.append(a)
    if append is not None:
        _append_to_diff_array(a, axis, combined, append)

    if len(combined) > 1:
        a = dpnp.concatenate(combined, axis=axis)

    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)

    op = dpnp.not_equal if a.dtype == numpy.bool_ else dpnp.subtract
    for _ in range(n):
        a = op(a[slice1], a[slice2])
    return a


_DIVIDE_DOCSTRING = """
Calculates the ratio for each element `x1_i` of the input array `x1` with
the respective element `x2_i` of the input array `x2`.

For full documentation refer to :obj:`numpy.divide`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray}
    First input array, expected to have numeric data type.
x2 : {dpnp.ndarray, usm_ndarray}
    Second input array, also expected to have numeric data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the result of element-wise division. The data type
    of the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

Notes
-----
Equivalent to `x1` / `x2` in terms of array-broadcasting.

The ``true_divide(x1, x2)`` function is an alias for
``divide(x1, x2)``.

Examples
--------
>>> import dpnp as np
>>> np.divide(dp.array([1, -2, 6, -9]), np.array([-2, -2, -2, -2]))
array([-0.5,  1. , -3. ,  4.5])

>>> x1 = np.arange(9.0).reshape((3, 3))
>>> x2 = np.arange(3.0)
>>> np.divide(x1, x2)
array([[nan, 1. , 1. ],
       [inf, 4. , 2.5],
       [inf, 7. , 4. ]])

The ``/`` operator can be used as a shorthand for ``divide`` on
:class:`dpnp.ndarray`.

>>> x1 = np.arange(9.0).reshape((3, 3))
>>> x2 = 2 * np.ones(3)
>>> x1/x2
array([[0. , 0.5, 1. ],
       [1.5, 2. , 2.5],
       [3. , 3.5, 4. ]])
"""

divide = DPNPBinaryFunc(
    "divide",
    ti._divide_result_type,
    ti._divide,
    _DIVIDE_DOCSTRING,
    mkl_fn_to_call=vmi._mkl_div_to_call,
    mkl_impl_fn=vmi._div,
    binary_inplace_fn=ti._divide_inplace,
    acceptance_fn=_acceptance_fn_divide,
)


def ediff1d(x1, to_end=None, to_begin=None):
    """
    The differences between consecutive elements of an array.

    For full documentation refer to :obj:`numpy.ediff1d`.

    Limitations
    -----------
    Parameter `x1`is supported as :class:`dpnp.ndarray`.
    Keyword arguments `to_end` and `to_begin` are currently supported only
    with default values `None`.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.diff` : Calculate the n-th discrete difference along the given
                       axis.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 2, 4, 7, 0])
    >>> result = np.ediff1d(a)
    >>> [x for x in result]
    [1, 2, 3, -7]
    >>> b = np.array([[1, 2, 4], [1, 6, 24]])
    >>> result = np.ediff1d(b)
    >>> [x for x in result]
    [1, 2, -3, 5, 18]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if x1_desc:
        if to_begin is not None:
            pass
        elif to_end is not None:
            pass
        else:
            return dpnp_ediff1d(x1_desc).get_pyobj()

    return call_origin(numpy.ediff1d, x1, to_end=to_end, to_begin=to_begin)


def fabs(x1, **kwargs):
    """
    Compute the absolute values element-wise.

    For full documentation refer to :obj:`numpy.fabs`.

    Limitations
    -----------
    Parameter `x1` is supported as :class:`dpnp.ndarray`.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.absolute` : Calculate the absolute value element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> result = np.fabs(np.array([1, -2, 6, -9]))
    >>> [x for x in result]
    [1.0, 2.0, 6.0, 9.0]

    """

    x1_desc = dpnp.get_dpnp_descriptor(
        x1, copy_when_strides=False, copy_when_nondefault_queue=False
    )
    if x1_desc:
        return dpnp_fabs(x1_desc).get_pyobj()

    return call_origin(numpy.fabs, x1, **kwargs)


_FLOOR_DOCSTRING = """
Returns the floor for each element `x_i` for input array `x`.

The floor of `x_i` is the largest integer `n`, such that `n <= x_i`.

For full documentation refer to :obj:`numpy.floor`.

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
    An array containing the element-wise floor.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.ceil` : Compute the ceiling of the input, element-wise.
:obj:`dpnp.trunc` : Return the truncated value of the input, element-wise.

Notes
-----
Some spreadsheet programs calculate the "floor-towards-zero", in other words floor(-2.5) == -2.
DPNP instead uses the definition of floor where floor(-2.5) == -3.

Examples
--------
>>> import dpnp as np
>>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
>>> np.floor(a)
array([-2.0, -2.0, -1.0, 0.0, 1.0, 1.0, 2.0])
"""

floor = DPNPUnaryFunc(
    "floor",
    ti._floor_result_type,
    ti._floor,
    _FLOOR_DOCSTRING,
    mkl_fn_to_call=vmi._mkl_floor_to_call,
    mkl_impl_fn=vmi._floor,
)


_FLOOR_DIVIDE_DOCSTRING = """
Calculates the ratio for each element `x1_i` of the input array `x1` with
the respective element `x2_i` of the input array `x2` to the greatest
integer-value number that is not greater than the division result.

For full documentation refer to :obj:`numpy.floor_divide`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray}
    First input array, expected to have numeric data type.
x2 : {dpnp.ndarray, usm_ndarray}
    Second input array, also expected to have numeric data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the result of element-wise floor of division.
    The data type of the returned array is determined by the Type
    Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.remainder` : Remainder complementary to floor_divide.
:obj:`dpnp.divide` : Standard division.
:obj:`dpnp.floor` : Round a number to the nearest integer toward minus infinity.
:obj:`dpnp.ceil` : Round a number to the nearest integer toward infinity.

Examples
--------
>>> import dpnp as np
>>> np.floor_divide(np.array([1, -1, -2, -9]), -2)
array([-1,  0,  1,  4])

>>> np.floor_divide(np.array([1., 2., 3., 4.]), 2.5)
array([ 0.,  0.,  1.,  1.])

The ``//`` operator can be used as a shorthand for ``floor_divide`` on
:class:`dpnp.ndarray`.

>>> x1 = np.array([1., 2., 3., 4.])
>>> x1 // 2.5
array([0., 0., 1., 1.])
"""

floor_divide = DPNPBinaryFunc(
    "floor_divide",
    ti._floor_divide_result_type,
    ti._floor_divide,
    _FLOOR_DIVIDE_DOCSTRING,
    binary_inplace_fn=ti._floor_divide_inplace,
)


def fmax(x1, x2, /, out=None, *, where=True, dtype=None, subok=True, **kwargs):
    """
    Element-wise maximum of array elements.

    For full documentation refer to :obj:`numpy.fmax`.

    Returns
    -------
    out : dpnp.ndarray
        The maximum of `x1` and `x2`, element-wise, ignoring NaNs.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar,
    :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`, but both `x1`
    and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default
    values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by real-valued data types.

    See Also
    --------
    :obj:`dpnp.maximum` : Element-wise maximum of array elements, propagates
                          NaNs.
    :obj:`dpnp.fmin` : Element-wise minimum of array elements, ignores NaNs.
    :obj:`dpnp.max` : The maximum value of an array along a given axis,
                      propagates NaNs..
    :obj:`dpnp.nanmax` : The maximum value of an array along a given axis,
                         ignores NaNs.
    :obj:`dpnp.minimum` : Element-wise minimum of array elements, propagates
                          NaNs.
    :obj:`dpnp.fmod` : Calculate the element-wise remainder of division.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = np.array([2, 3, 4])
    >>> x2 = np.array([1, 5, 2])
    >>> np.fmax(x1, x2)
    array([2, 5, 4])

    >>> x1 = np.eye(2)
    >>> x2 = np.array([0.5, 2])
    >>> np.fmax(x1, x2) # broadcasting
    array([[1. , 2. ],
           [0.5, 2. ]])

    >>> x1 = np.array([np.nan, 0, np.nan])
    >>> x2 = np.array([0, np.nan, np.nan])
    >>> np.fmax(x1, x2)
    array([ 0.,  0., nan])

    """

    if kwargs:
        pass
    elif where is not True:
        pass
    elif dtype is not None:
        pass
    elif subok is not True:
        pass
    elif dpnp.isscalar(x1) and dpnp.isscalar(x2):
        # at least either x1 or x2 has to be an array
        pass
    else:
        # get USM type and queue to copy scalar from the host memory
        # into a USM allocation
        usm_type, queue = (
            get_usm_allocations([x1, x2])
            if dpnp.isscalar(x1) or dpnp.isscalar(x2)
            else (None, None)
        )

        x1_desc = dpnp.get_dpnp_descriptor(
            x1,
            copy_when_strides=False,
            copy_when_nondefault_queue=False,
            alloc_usm_type=usm_type,
            alloc_queue=queue,
        )
        x2_desc = dpnp.get_dpnp_descriptor(
            x2,
            copy_when_strides=False,
            copy_when_nondefault_queue=False,
            alloc_usm_type=usm_type,
            alloc_queue=queue,
        )
        if x1_desc and x2_desc:
            if out is not None:
                if not dpnp.is_supported_array_type(out):
                    raise TypeError(
                        "return array must be of supported array type"
                    )
                out_desc = (
                    dpnp.get_dpnp_descriptor(
                        out, copy_when_nondefault_queue=False
                    )
                    or None
                )
            else:
                out_desc = None

            return dpnp_fmax(
                x1_desc, x2_desc, dtype=dtype, out=out_desc, where=where
            ).get_pyobj()

    return call_origin(
        numpy.fmax, x1, x2, dtype=dtype, out=out, where=where, **kwargs
    )


def fmin(x1, x2, /, out=None, *, where=True, dtype=None, subok=True, **kwargs):
    """
    Element-wise minimum of array elements.

    For full documentation refer to :obj:`numpy.fmin`.

    Returns
    -------
    out : dpnp.ndarray
        The minimum of `x1` and `x2`, element-wise, ignoring NaNs.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar,
    :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`, but both `x1`
    and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default
    values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by real-valued data types.

    See Also
    --------
    :obj:`dpnp.minimum` : Element-wise minimum of array elements, propagates
                          NaNs.
    :obj:`dpnp.fmax` : Element-wise maximum of array elements, ignores NaNs.
    :obj:`dpnp.min` : The minimum value of an array along a given axis,
                      propagates NaNs.
    :obj:`dpnp.nanmin` : The minimum value of an array along a given axis,
                         ignores NaNs.
    :obj:`dpnp.maximum` : Element-wise maximum of array elements, propagates
                          NaNs.
    :obj:`dpnp.fmod` : Calculate the element-wise remainder of division.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = np.array([2, 3, 4])
    >>> x2 = np.array([1, 5, 2])
    >>> np.fmin(x1, x2)
    array([1, 3, 2])

    >>> x1 = np.eye(2)
    >>> x2 = np.array([0.5, 2])
    >>> np.fmin(x1, x2) # broadcasting
    array([[0.5, 0. ],
           [0. , 1. ]]

    >>> x1 = np.array([np.nan, 0, np.nan])
    >>> x2 = np.array([0, np.nan, np.nan])
    >>> np.fmin(x1, x2)
    array([ 0.,  0., nan])

    """

    if kwargs:
        pass
    elif where is not True:
        pass
    elif dtype is not None:
        pass
    elif subok is not True:
        pass
    elif dpnp.isscalar(x1) and dpnp.isscalar(x2):
        # at least either x1 or x2 has to be an array
        pass
    else:
        # get USM type and queue to copy scalar from the host memory into
        # a USM allocation
        usm_type, queue = (
            get_usm_allocations([x1, x2])
            if dpnp.isscalar(x1) or dpnp.isscalar(x2)
            else (None, None)
        )

        x1_desc = dpnp.get_dpnp_descriptor(
            x1,
            copy_when_strides=False,
            copy_when_nondefault_queue=False,
            alloc_usm_type=usm_type,
            alloc_queue=queue,
        )
        x2_desc = dpnp.get_dpnp_descriptor(
            x2,
            copy_when_strides=False,
            copy_when_nondefault_queue=False,
            alloc_usm_type=usm_type,
            alloc_queue=queue,
        )
        if x1_desc and x2_desc:
            if out is not None:
                if not dpnp.is_supported_array_type(out):
                    raise TypeError(
                        "return array must be of supported array type"
                    )
                out_desc = (
                    dpnp.get_dpnp_descriptor(
                        out, copy_when_nondefault_queue=False
                    )
                    or None
                )
            else:
                out_desc = None

            return dpnp_fmin(
                x1_desc, x2_desc, dtype=dtype, out=out_desc, where=where
            ).get_pyobj()

    return call_origin(
        numpy.fmin, x1, x2, dtype=dtype, out=out, where=where, **kwargs
    )


def fmod(x1, x2, /, out=None, *, where=True, dtype=None, subok=True, **kwargs):
    """
    Returns the element-wise remainder of division.

    For full documentation refer to :obj:`numpy.fmod`.

    Returns
    -------
    out : dpnp.ndarray
        The remainder of the division of `x1` by `x2`.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar,
    :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`, but both `x1`
    and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default
    values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.remainder` : Remainder complementary to floor_divide.
    :obj:`dpnp.divide` : Standard division.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([-3, -2, -1, 1, 2, 3])
    >>> np.fmod(a, 2)
    array([-1,  0, -1,  1,  0,  1])
    >>> np.remainder(a, 2)
    array([1, 0, 1, 1, 0, 1])

    >>> a = np.array([5, 3])
    >>> b = np.array([2, 2.])
    >>> np.fmod(a, b)
    array([1., 1.])

    >>> a = np.arange(-3, 3).reshape(3, 2)
    >>> a
    array([[-3, -2],
           [-1,  0],
           [ 1,  2]])
    >>> b = np.array([2, 2])
    >>> np.fmod(a, b)
    array([[-1,  0],
           [-1,  0],
           [ 1,  0]])

    """

    if kwargs:
        pass
    elif where is not True:
        pass
    elif dtype is not None:
        pass
    elif subok is not True:
        pass
    elif dpnp.isscalar(x1) and dpnp.isscalar(x2):
        # at least either x1 or x2 has to be an array
        pass
    else:
        # get USM type and queue to copy scalar from the host memory into
        # a USM allocation
        usm_type, queue = (
            get_usm_allocations([x1, x2])
            if dpnp.isscalar(x1) or dpnp.isscalar(x2)
            else (None, None)
        )

        x1_desc = dpnp.get_dpnp_descriptor(
            x1,
            copy_when_strides=False,
            copy_when_nondefault_queue=False,
            alloc_usm_type=usm_type,
            alloc_queue=queue,
        )
        x2_desc = dpnp.get_dpnp_descriptor(
            x2,
            copy_when_strides=False,
            copy_when_nondefault_queue=False,
            alloc_usm_type=usm_type,
            alloc_queue=queue,
        )
        if x1_desc and x2_desc:
            if out is not None:
                if not dpnp.is_supported_array_type(out):
                    raise TypeError(
                        "return array must be of supported array type"
                    )
                out_desc = (
                    dpnp.get_dpnp_descriptor(
                        out, copy_when_nondefault_queue=False
                    )
                    or None
                )
            else:
                out_desc = None

            return dpnp_fmod(
                x1_desc, x2_desc, dtype=dtype, out=out_desc, where=where
            ).get_pyobj()

    return call_origin(
        numpy.fmod, x1, x2, dtype=dtype, out=out, where=where, **kwargs
    )


def gradient(x1, *varargs, **kwargs):
    """
    Return the gradient of an array.

    For full documentation refer to :obj:`numpy.gradient`.

    Limitations
    -----------
    Parameter `y1` is supported as :class:`dpnp.ndarray`.
    Argument `varargs[0]` is supported as `int`.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.diff` : Calculate the n-th discrete difference along the given
                       axis.

    Examples
    --------
    >>> import dpnp as np
    >>> y = np.array([1, 2, 4, 7, 11, 16], dtype=float)
    >>> result = np.gradient(y)
    >>> [x for x in result]
    [1.0, 1.5, 2.5, 3.5, 4.5, 5.0]
    >>> result = np.gradient(y, 2)
    >>> [x for x in result]
    [0.5, 0.75, 1.25, 1.75, 2.25, 2.5]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if x1_desc and not kwargs:
        if len(varargs) > 1:
            pass
        elif len(varargs) == 1 and not isinstance(varargs[0], int):
            pass
        else:
            if len(varargs) == 0:
                return dpnp_gradient(x1_desc).get_pyobj()

            return dpnp_gradient(x1_desc, varargs[0]).get_pyobj()

    return call_origin(numpy.gradient, x1, *varargs, **kwargs)


_IMAG_DOCSTRING = """
Computes imaginary part of each element `x_i` for input array `x`.

For full documentation refer to :obj:`numpy.imag`.

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
    An array containing the element-wise imaginary component of input.
    If the input is a real-valued data type, the returned array has
    the same data type. If the input is a complex floating-point
    data type, the returned array has a floating-point data type
    with the same floating-point precision as complex input.

See Also
--------
:obj:`dpnp.real` : Return the real part of the complex argument.
:obj:`dpnp.conj` : Return the complex conjugate, element-wise.
:obj:`dpnp.conjugate` : Return the complex conjugate, element-wise.

Examples
--------
>>> import dpnp as np
>>> a = np.array([1+2j, 3+4j, 5+6j])
>>> a.imag
array([2., 4., 6.])

>>> a.imag = np.array([8, 10, 12])
>>> a
array([1. +8.j, 3.+10.j, 5.+12.j])

>>> np.imag(np.array(1 + 1j))
array(1.)
"""

imag = DPNPUnaryFunc(
    "imag",
    ti._imag_result_type,
    ti._imag,
    _IMAG_DOCSTRING,
)


_MAXIMUM_DOCSTRING = """
Compares two input arrays `x1` and `x2` and returns a new array containing the
element-wise maxima.

For full documentation refer to :obj:`numpy.maximum`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray}
    First input array, expected to have numeric data type.
x2 : {dpnp.ndarray, usm_ndarray}
    Second input array, also expected to have numeric data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise maxima. The data type of
    the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.minimum` : Element-wise minimum of two arrays, propagates NaNs.
:obj:`dpnp.fmax` : Element-wise maximum of two arrays, ignores NaNs.
:obj:`dpnp.max` : The maximum value of an array along a given axis, propagates NaNs.
:obj:`dpnp.nanmax` : The maximum value of an array along a given axis, ignores NaNs.
:obj:`dpnp.fmin` : Element-wise minimum of two arrays, ignores NaNs.
:obj:`dpnp.min` : The minimum value of an array along a given axis, propagates NaNs.
:obj:`dpnp.nanmin` : The minimum value of an array along a given axis, ignores NaNs.

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([2, 3, 4])
>>> x2 = np.array([1, 5, 2])
>>> np.maximum(x1, x2)
array([2, 5, 4])

>>> x1 = np.eye(2)
>>> x2 = np.array([0.5, 2])
>>> np.maximum(x1, x2) # broadcasting
array([[1. , 2. ],
       [0.5, 2. ]])

>>> x1 = np.array([np.nan, 0, np.nan])
>>> x2 = np.array([0, np.nan, np.nan])
>>> np.maximum(x1, x2)
array([nan, nan, nan])

>>> np.maximum(np.array(np.Inf), 1)
array(inf)
"""

maximum = DPNPBinaryFunc(
    "maximum",
    ti._maximum_result_type,
    ti._maximum,
    _MAXIMUM_DOCSTRING,
)


_MINIMUM_DOCSTRING = """
Compares two input arrays `x1` and `x2` and returns a new array containing the
element-wise minima.

For full documentation refer to :obj:`numpy.minimum`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray}
    First input array, expected to have numeric data type.
x2 : {dpnp.ndarray, usm_ndarray}
    Second input array, also expected to have numeric data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise minima. The data type of
    the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.maximum` : Element-wise maximum of two arrays, propagates NaNs.
:obj:`dpnp.fmin` : Element-wise minimum of two arrays, ignores NaNs.
:obj:`dpnp.min` : The minimum value of an array along a given axis, propagates NaNs.
:obj:`dpnp.nanmin` : The minimum value of an array along a given axis, ignores NaNs.
:obj:`dpnp.fmax` : Element-wise maximum of two arrays, ignores NaNs.
:obj:`dpnp.max` : The maximum value of an array along a given axis, propagates NaNs.
:obj:`dpnp.nanmax` : The maximum value of an array along a given axis, ignores NaNs.

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([2, 3, 4])
>>> x2 = np.array([1, 5, 2])
>>> np.minimum(x1, x2)
array([1, 3, 2])

>>> x1 = np.eye(2)
>>> x2 = np.array([0.5, 2])
>>> np.minimum(x1, x2) # broadcasting
array([[0.5, 0. ],
       [0. , 1. ]]

>>> x1 = np.array([np.nan, 0, np.nan])
>>> x2 = np.array([0, np.nan, np.nan])
>>> np.minimum(x1, x2)
array([nan, nan, nan])

>>> np.minimum(np.array(-np.Inf), 1)
array(-inf)
"""

minimum = DPNPBinaryFunc(
    "minimum",
    ti._minimum_result_type,
    ti._minimum,
    _MINIMUM_DOCSTRING,
)


def mod(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    order="K",
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Compute element-wise remainder of division.

    For full documentation refer to :obj:`numpy.mod`.

    Returns
    -------
    out : dpnp.ndarray
        The element-wise remainder of the quotient `floor_divide(x1, x2)`.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar,
    :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`, but both `x1`
    and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default
    values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.fmod` : Calculate the element-wise remainder of division
    :obj:`dpnp.remainder` : Remainder complementary to floor_divide.
    :obj:`dpnp.divide` : Standard division.

    Notes
    -----
    This function works the same as :obj:`dpnp.remainder`.

    """

    return dpnp.remainder(
        x1,
        x2,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def modf(x1, **kwargs):
    """
    Return the fractional and integral parts of an array, element-wise.

    For full documentation refer to :obj:`numpy.modf`.

    Limitations
    -----------
    Parameter `x` is supported as :obj:`dpnp.ndarray`.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 2])
    >>> result = np.modf(a)
    >>> [[x for x in y] for y in result ]
    [[1.0, 2.0], [0.0, 0.0]]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if x1_desc and not kwargs:
        return dpnp_modf(x1_desc)

    return call_origin(numpy.modf, x1, **kwargs)


_MULTIPLY_DOCSTRING = """
Calculates the product for each element `x1_i` of the input array `x1` with the
respective element `x2_i` of the input array `x2`.

For full documentation refer to :obj:`numpy.multiply`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray}
    First input array, expected to have numeric data type.
x2 : {dpnp.ndarray, usm_ndarray}
    Second input array, also expected to have numeric data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise products. The data type of
    the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

Notes
-----
Equivalent to `x1` * `x2` in terms of array broadcasting.

Examples
--------
>>> import dpnp as np
>>> a = np.array([1, 2, 3, 4, 5])
>>> np.multiply(a, a)
array([ 1,  4,  9, 16, 25])]

>>> x1 = np.arange(9.0).reshape((3, 3))
>>> x2 = np.arange(3.0)
>>> np.multiply(x1, x2)
array([[  0.,   1.,   4.],
       [  0.,   4.,  10.],
       [  0.,   7.,  16.]])

The ``*`` operator can be used as a shorthand for ``multiply`` on
:class:`dpnp.ndarray`.

>>> x1 * x2
array([[  0.,   1.,   4.],
       [  0.,   4.,  10.],
       [  0.,   7.,  16.]])
"""

multiply = DPNPBinaryFunc(
    "multiply",
    ti._multiply_result_type,
    ti._multiply,
    _MULTIPLY_DOCSTRING,
    mkl_fn_to_call=vmi._mkl_mul_to_call,
    mkl_impl_fn=vmi._mul,
    binary_inplace_fn=ti._multiply_inplace,
)


_NEGATIVE_DOCSTRING = """
Computes the numerical negative for each element `x_i` of input array `x`.

For full documentation refer to :obj:`numpy.negative`.

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
    An array containing the negative of `x`.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.positive` : Return the numerical positive of each element of `x`.
:obj:`dpnp.copysign` : Change the sign of `x1` to that of `x2`, element-wise.

Examples
--------
>>> import dpnp as np
>>> np.negative(np.array([1, -1]))
array([-1, 1])

The ``-`` operator can be used as a shorthand for ``negative`` on
:class:`dpnp.ndarray`.

>>> x = np.array([1., -1.])
>>> -x
array([-1.,  1.])
"""

negative = DPNPUnaryFunc(
    "negative",
    ti._negative_result_type,
    ti._negative,
    _NEGATIVE_DOCSTRING,
    acceptance_fn=acceptance_fn_negative,
)


_POSITIVE_DOCSTRING = """
Computes the numerical positive for each element `x_i` of input array `x`.

For full documentation refer to :obj:`numpy.positive`.

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
    An array containing the positive of `x`.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.negative` : Return the numerical negative of each element of `x`.
:obj:`dpnp.copysign` : Change the sign of `x1` to that of `x2`, element-wise.

Note
----
Equivalent to `x.copy()`, but only defined for types that support arithmetic.

Examples
--------
>>> import dpnp as np
>>> np.positive(np.array([1., -1.]))
array([ 1., -1.])

The ``+`` operator can be used as a shorthand for ``positive`` on
:class:`dpnp.ndarray`.

>>> x = np.array([1., -1.])
>>> +x
array([ 1., -1.])
"""

positive = DPNPUnaryFunc(
    "positive",
    ti._positive_result_type,
    ti._positive,
    _POSITIVE_DOCSTRING,
    acceptance_fn=acceptance_fn_positive,
)


_POWER_DOCSTRING = """
Calculates `x1_i` raised to `x2_i` for each element `x1_i` of the input array
`x1` with the respective element `x2_i` of the input array `x2`.

For full documentation refer to :obj:`numpy.power`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray}
    First input array, expected to have numeric data type.
x2 : {dpnp.ndarray, usm_ndarray}
    Second input array, also expected to have numeric data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate. Array must have the correct
    shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the bases in `x1` raised to the exponents in `x2`
    element-wise. The data type of the returned array is determined by the
    Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.fmax` : Element-wise maximum of array elements.
:obj:`dpnp.fmin` : Element-wise minimum of array elements.
:obj:`dpnp.fmod` : Calculate the element-wise remainder of division.


Examples
--------
>>> import dpnp as dp
>>> a = dp.arange(6)
>>> dp.power(a, 3)
array([  0,   1,   8,  27,  64, 125])

Raise the bases to different exponents.

>>> b = dp.array([1.0, 2.0, 3.0, 3.0, 2.0, 1.0])
>>> dp.power(a, b)
array([ 0.,  1.,  8., 27., 16.,  5.])

The effect of broadcasting.

>>> c = dp.array([[1, 2, 3, 3, 2, 1], [1, 2, 3, 3, 2, 1]])
>>> dp.power(a, c)
array([[ 0,  1,  8, 27, 16,  5],
        [ 0,  1,  8, 27, 16,  5]])

The ``**`` operator can be used as a shorthand for ``power`` on
:class:`dpnp.ndarray`.

>>> b = dp.array([1, 2, 3, 3, 2, 1])
>>> a = dp.arange(6)
>>> a ** b
array([ 0,  1,  8, 27, 16,  5])

Negative values raised to a non-integral value will result in ``nan``.

>>> d = dp.array([-1.0, -4.0])
>>> dp.power(d, 1.5)
array([nan, nan])
"""

power = DPNPBinaryFunc(
    "power",
    ti._pow_result_type,
    ti._pow,
    _POWER_DOCSTRING,
    mkl_fn_to_call=vmi._mkl_pow_to_call,
    mkl_impl_fn=vmi._pow,
    binary_inplace_fn=ti._pow_inplace,
)


def prod(
    a,
    axis=None,
    dtype=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
):
    """
    Return the product of array elements over a given axis.

    For full documentation refer to :obj:`numpy.prod`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int or tuple of ints}, optional
        Axis or axes along which a product is performed. The default,
        ``axis=None``, will calculate the product of all the elements in the
        input array. If `axis` is negative it counts from the last to the first
        axis.
        If `axis` is a tuple of integers, a product is performed on all of the
        axes specified in the tuple instead of a single axis or all the axes as
        before.
        Default: ``None``.
    dtype : {None, dtype}, optional
        The type of the returned array, as well as of the accumulator in which
        the elements are multiplied. The dtype of `a` is used by default unless
        `a` has an integer dtype of less precision than the default platform
        integer. In that case, if `a` is signed then the platform integer is
        used while if `a` is unsigned then an unsigned integer of the same
        precision as the platform integer is used.
        Default: ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output, but the type of the output
        values will be cast if necessary.
        Default: ``None``.
    keepdims : {None, bool}, optional
        If this is set to ``True``, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result will
        broadcast correctly against the input array.
        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        An array with the same shape as `a`, with the specified axis removed.
        If `a` is a 0-d array, or if `axis` is ``None``, a zero-dimensional
        array is returned. If an output array is specified, a reference to
        `out` is returned.

    Limitations
    -----------
    Parameters `initial` and `where` are only supported with their default
    values.
    Otherwise ``NotImplementedError`` exception will be raised.

    See Also
    --------
    :obj:`dpnp.nanprod` : Return the product of array elements over a given
        axis treating Not a Numbers (NaNs) as ones.

    Examples
    --------
    >>> import dpnp as np
    >>> np.prod(np.array([1, 2]))
    array(2)

    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.prod(a)
    array(24)

    >>> np.prod(a, axis=1)
    array([ 2, 12])
    >>> np.prod(a, axis=0)
    array([3, 8])

    >>> x = np.array([1, 2, 3], dtype=np.int8)
    >>> np.prod(x).dtype == int
    True

    """

    dpnp.check_limitations(initial=initial, where=where)
    usm_a = dpnp.get_usm_ndarray(a)

    return dpnp_wrap_reduction_call(
        a,
        out,
        dpt.prod,
        _get_reduction_res_dt,
        usm_a,
        axis=axis,
        dtype=dtype,
        keepdims=keepdims,
    )


_PROJ_DOCSTRING = """
Computes projection of each element `x_i` for input array `x`.

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
    An array containing the element-wise projection.

Limitations
-----------
Parameters `where' and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.absolute` : Returns the magnitude of a complex number, element-wise.
:obj:`dpnp.conj` : Return the complex conjugate, element-wise.

Examples
--------
>>> import dpnp as np
>>> np.proj(np.array([1, -2.3, 2.1-1.7j]))
array([ 1. +0.j, -2.3+0.j,  2.1-1.7.j])

>>> np.proj(np.array([complex(1,np.inf), complex(1,-np.inf), complex(np.inf,-1),]))
array([inf+0.j, inf-0.j, inf-0.j])
"""

proj = DPNPUnaryFunc(
    "proj",
    ti._proj_result_type,
    ti._proj,
    _PROJ_DOCSTRING,
)


_REAL_DOCSTRING = """
Computes real part of each element `x_i` for input array `x`.

For full documentation refer to :obj:`numpy.real`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have numeric data type.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise real component of input.
    If the input is a real-valued data type, the returned array has
    the same data type. If the input is a complex floating-point
    data type, the returned array has a floating-point data type
    with the same floating-point precision as complex input.
"""

real = DPNPReal(
    "real",
    ti._real_result_type,
    ti._real,
    _REAL_DOCSTRING,
)


_REMAINDER_DOCSTRING = """
Calculates the remainder of division for each element `x1_i` of the input array
`x1` with the respective element `x2_i` of the input array `x2`.

This function is equivalent to the Python modulus operator.

For full documentation refer to :obj:`numpy.remainder`.

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
    An array containing the element-wise remainders. Each remainder has the
    same sign as respective element `x2_i`. The data type of the returned
    array is determined by the Type Promotion Rules.

Limitations
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.fmod` : Calculate the element-wise remainder of division.
:obj:`dpnp.divide` : Standard division.
:obj:`dpnp.floor` : Round a number to the nearest integer toward minus infinity.
:obj:`dpnp.floor_divide` : Compute the largest integer smaller or equal to the division of the inputs.
:obj:`dpnp.mod` : Calculate the element-wise remainder of division.

Examples
--------
>>> import dpnp as np
>>> np.remainder(np.array([4, 7]), np.array([2, 3]))
array([0, 1])

>>> np.remainder(np.arange(7), 5)
array([0, 1, 2, 3, 4, 0, 1])

The ``%`` operator can be used as a shorthand for ``remainder`` on
:class:`dpnp.ndarray`.

>>> x1 = np.arange(7)
>>> x1 % 5
array([0, 1, 2, 3, 4, 0, 1])
"""

remainder = DPNPBinaryFunc(
    "remainder",
    ti._remainder_result_type,
    ti._remainder,
    _REMAINDER_DOCSTRING,
    binary_inplace_fn=ti._remainder_inplace,
)


_RINT_DOCSTRING = """
Rounds each element `x_i` of the input array `x` to
the nearest integer-valued number.

When two integers are equally close to `x_i`, the result is the nearest even
integer to `x_i`.

For full documentation refer to :obj:`numpy.rint`.

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
    An array containing the element-wise rounded values.

Limitations
-----------
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.round` : Evenly round to the given number of decimals.
:obj:`dpnp.ceil` : Compute the ceiling of the input, element-wise.
:obj:`dpnp.floor` : Return the floor of the input, element-wise.
:obj:`dpnp.trunc` : Return the truncated value of the input, element-wise.

Examples
--------
>>> import dpnp as np
>>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
>>> np.rint(a)
array([-2., -2., -0.,  0.,  2.,  2.,  2.])
"""


rint = DPNPUnaryFunc(
    "rint",
    ti._round_result_type,
    ti._round,
    _RINT_DOCSTRING,
    mkl_fn_to_call=vmi._mkl_round_to_call,
    mkl_impl_fn=vmi._round,
)


_ROUND_DOCSTRING = """
Rounds each element `x_i` of the input array `x` to
the nearest integer-valued number.

When two integers are equally close to `x_i`, the result is the nearest even
integer to `x_i`.

For full documentation refer to :obj:`numpy.round`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have numeric data type.
decimals : int, optional
    Number of decimal places to round to (default: 0). If decimals is negative,
    it specifies the number of positions to the left of the decimal point.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise rounded values.

See Also
--------
:obj:`dpnp.around` : Equivalent function; see for details.
:obj:`dpnp.ndarray.round` : Equivalent function.
:obj:`dpnp.rint` : Round elements of the array to the nearest integer.
:obj:`dpnp.ceil` : Compute the ceiling of the input, element-wise.
:obj:`dpnp.floor` : Return the floor of the input, element-wise.
:obj:`dpnp.trunc` : Return the truncated value of the input, element-wise.

Examples
--------
>>> import dpnp as np
>>> np.round(np.array([0.37, 1.64]))
array([0.,  2.])
>>> np.round(np.array([0.37, 1.64]), decimals=1)
array([0.4,  1.6])
>>> np.round(np.array([.5, 1.5, 2.5, 3.5, 4.5])) # rounds to nearest even value
array([0.,  2.,  2.,  4.,  4.])
>>> np.round(np.array([1, 2, 3, 11]), decimals=1) # ndarray of ints is returned
array([ 1,  2,  3, 11])
>>> np.round(np.array([1, 2, 3, 11]), decimals=-1)
array([ 0,  0,  0, 10])
"""

round = DPNPRound(
    "round",
    ti._round_result_type,
    ti._round,
    _ROUND_DOCSTRING,
    mkl_fn_to_call=vmi._mkl_round_to_call,
    mkl_impl_fn=vmi._round,
)


_SIGN_DOCSTRING = """
Computes an indication of the sign of each element `x_i` of input array `x`
using the signum function.

The signum function returns `-1` if `x_i` is less than `0`,
`0` if `x_i` is equal to `0`, and `1` if `x_i` is greater than `0`.

For full documentation refer to :obj:`numpy.sign`.

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
    An array containing the element-wise result of the signum function. The
    data type of the returned array is determined by the Type Promotion
    Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.signbit` : Returns element-wise `True` where signbit is set (less than zero).

Examples
--------
>>> import dpnp as np
>>> np.sign(np.array([-5., 4.5]))
array([-1.0, 1.0])
>>> np.sign(np.array(0))
array(0)
>>> np.sign(np.array(5-2j))
array([1+0j])
"""

sign = DPNPUnaryFunc(
    "sign",
    ti._sign_result_type,
    ti._sign,
    _SIGN_DOCSTRING,
    acceptance_fn=acceptance_fn_sign,
)


_SIGNBIT_DOCSTRING = """
Computes an indication of whether the sign bit of each element `x_i` of
input array `x` is set.

For full documentation refer to :obj:`numpy.signbit`.

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
    An array containing the element-wise signbit results. The returned array
    must have a data type of `bool`.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.sign` : Returns an element-wise indication of the sign of a number.

Examples
--------
>>> import dpnp as np
>>> np.signbit(np.array([-1.2]))
array([True])

>>> np.signbit(np.array([1, -2.3, 2.1]))
array([False,  True, False])
"""

signbit = DPNPUnaryFunc(
    "signbit",
    ti._signbit_result_type,
    ti._signbit,
    _SIGNBIT_DOCSTRING,
)


_SUBTRACT_DOCSTRING = """
Calculates the difference between each element `x1_i` of the input
array `x1` and the respective element `x2_i` of the input array `x2`.

For full documentation refer to :obj:`numpy.subtract`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray}
    First input array, expected to have numeric data type.
x2 : {dpnp.ndarray, usm_ndarray}
    Second input array, also expected to have numeric data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise differences. The data type
    of the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

Notes
-----
Equivalent to `x1` - `x2` in terms of array broadcasting.

Examples
--------
>>> import dpnp as np
>>> np.subtract(dp.array([4, 3]), np.array([2, 7]))
array([ 2, -4])

>>> x1 = np.arange(9.0).reshape((3, 3))
>>> x2 = np.arange(3.0)
>>> np.subtract(x1, x2)
array([[ 0.,  0.,  0.],
       [ 3.,  3.,  3.],
       [ 6.,  6.,  6.]])

The ``-`` operator can be used as a shorthand for ``subtract`` on
:class:`dpnp.ndarray`.

>>> x1 - x2
array([[ 0.,  0.,  0.],
       [ 3.,  3.,  3.],
       [ 6.,  6.,  6.]])
"""

subtract = DPNPBinaryFunc(
    "subtract",
    ti._subtract_result_type,
    ti._subtract,
    _SUBTRACT_DOCSTRING,
    mkl_fn_to_call=vmi._mkl_sub_to_call,
    mkl_impl_fn=vmi._sub,
    binary_inplace_fn=ti._subtract_inplace,
    acceptance_fn=acceptance_fn_subtract,
)


def sum(
    a,
    axis=None,
    dtype=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
):
    """
    Sum of array elements over a given axis.

    For full documentation refer to :obj:`numpy.sum`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int or tuple of ints}, optional
        Axis or axes along which a sum is performed. The default,
        ``axis=None``, will sum all of the elements of the input array. If axis
        is negative it counts from the last to the first axis.
        If `axis` is a tuple of integers, a sum is performed on all of the axes
        specified in the tuple instead of a single axis or all the axes as
        before.
        Default: ``None``.
    dtype : {None, dtype}, optional
        The type of the returned array and of the accumulator in which the
        elements are summed. The dtype of `a` is used by default unless `a` has
        an integer dtype of less precision than the default platform integer.
        In that case, if `a` is signed then the platform integer is used while
        if `a` is unsigned then an unsigned integer of the same precision as
        the platform integer is used.
        Default: ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. It must have the
        same shape as the expected output, but the type of the output values
        will be cast if necessary.
        Default: ``None``.
    keepdims : {None, bool}, optional
        If this is set to ``True``, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result will
        broadcast correctly against the input array.
        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        An array with the same shape as `a`, with the specified axis removed.
        If `a` is a 0-d array, or if `axis` is ``None``, a zero-dimensional
        array is returned. If an output array is specified, a reference to
        `out` is returned.

    Limitations
    -----------
    Parameters `initial` and `where` are only supported with their default
    values.
    Otherwise ``NotImplementedError`` exception will be raised.

    See Also
    --------
    :obj:`dpnp.ndarray.sum` : Equivalent method.
    :obj:`dpnp.cumsum` : Cumulative sum of array elements.
    :obj:`dpnp.trapz` : Integration of array values using the composite
                        trapezoidal rule.
    :obj:`dpnp.mean` : Compute the arithmetic mean.
    :obj:`dpnp.average` : Compute the weighted average.

    Examples
    --------
    >>> import dpnp as np
    >>> np.sum(np.array([0.5, 1.5]))
    array(2.)
    >>> np.sum(np.array([0.5, 0.7, 0.2, 1.5]), dtype=np.int32)
    array(1)
    >>> a = np.array([[0, 1], [0, 5]])
    >>> np.sum(a)
    array(6)
    >>> np.sum(a, axis=0)
    array([0, 6])
    >>> np.sum(a, axis=1)
    array([1, 5])

    """

    dpnp.check_limitations(initial=initial, where=where)

    sycl_sum_call = False
    if len(a.shape) == 2 and a.itemsize == 4:
        c_contiguous_rules = (
            axis == (0,)
            and a.flags.c_contiguous
            and 32 <= a.shape[1] <= 1024
            and a.shape[0] > a.shape[1]
        )
        f_contiguous_rules = (
            axis == (1,)
            and a.flags.f_contiguous
            and 32 <= a.shape[0] <= 1024
            and a.shape[1] > a.shape[0]
        )
        sycl_sum_call = c_contiguous_rules or f_contiguous_rules

    if sycl_sum_call:
        if axis is not None:
            if not isinstance(axis, (tuple, list)):
                axis = (axis,)

            axis = normalize_axis_tuple(axis, a.ndim, "axis")

        input = a
        if axis == (1,):
            input = input.T
        input = dpnp.get_usm_ndarray(input)

        queue = input.sycl_queue
        out_dtype = (
            dtu._default_accumulation_dtype(input.dtype, queue)
            if dtype is None
            else dtype
        )
        output = dpt.empty(input.shape[1], dtype=out_dtype, sycl_queue=queue)

        get_sum = _sycl_ext_impl._get_sum_over_axis_0
        sycl_sum = get_sum(input, output)

        if sycl_sum:
            sycl_sum(input, output, []).wait()
            result = dpnp_array._create_from_usm_ndarray(output)

            if keepdims:
                if axis == (0,):
                    res_sh = (1,) + output.shape
                else:
                    res_sh = output.shape + (1,)
                result = result.reshape(res_sh)

            return result

    usm_a = dpnp.get_usm_ndarray(a)
    return dpnp_wrap_reduction_call(
        a,
        out,
        dpt.sum,
        _get_reduction_res_dt,
        usm_a,
        axis=axis,
        dtype=dtype,
        keepdims=keepdims,
    )


def trapz(y1, x1=None, dx=1.0, axis=-1):
    """
    Integrate along the given axis using the composite trapezoidal rule.

    For full documentation refer to :obj:`numpy.trapz`.

    Limitations
    -----------
    Parameters `y` and `x` are supported as :class:`dpnp.ndarray`.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([4, 6, 8])
    >>> np.trapz(a)
    4.0
    >>> np.trapz(a, x=b)
    8.0
    >>> np.trapz(a, dx=2)
    8.0

    """

    y_desc = dpnp.get_dpnp_descriptor(y1, copy_when_nondefault_queue=False)
    if y_desc:
        if y_desc.ndim > 1:
            pass
        else:
            y_obj = y_desc.get_array()
            if x1 is None:
                x_obj = dpnp.empty(
                    y_desc.shape,
                    dtype=y_desc.dtype,
                    device=y_obj.sycl_device,
                    usm_type=y_obj.usm_type,
                    sycl_queue=y_obj.sycl_queue,
                )
            else:
                x_obj = x1

            x_desc = dpnp.get_dpnp_descriptor(
                x_obj, copy_when_nondefault_queue=False
            )
            # TODO: change to "not x_desc"
            if x_desc:
                pass
            elif y_desc.size != x_desc.size:
                pass
            elif y_desc.shape != x_desc.shape:
                pass
            else:
                return dpnp_trapz(y_desc, x_desc, dx).get_pyobj()

    return call_origin(numpy.trapz, y1, x1, dx, axis)


true_divide = divide


_TRUNC_DOCSTRING = """
Returns the truncated value for each element `x_i` for input array `x`.

The truncated value of the scalar `x` is the nearest integer i which is
closer to zero than `x` is. In short, the fractional part of the
signed number `x` is discarded.

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
    An array containing the result of element-wise division. The data type
    of the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.floor` : Round a number to the nearest integer toward minus infinity.
:obj:`dpnp.ceil` : Round a number to the nearest integer toward infinity.

Examples
--------
>>> import dpnp as np
>>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
>>> np.trunc(a)
array([-1.0, -1.0, -0.0, 0.0, 1.0, 1.0, 2.0])
"""

trunc = DPNPUnaryFunc(
    "trunc",
    ti._trunc_result_type,
    ti._trunc,
    _TRUNC_DOCSTRING,
    mkl_fn_to_call=vmi._mkl_trunc_to_call,
    mkl_impl_fn=vmi._trunc,
)
