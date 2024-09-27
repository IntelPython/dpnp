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
# pylint: disable=duplicate-code
# pylint: disable=no-name-in-module


import dpctl.tensor as dpt
import dpctl.tensor._tensor_elementwise_impl as ti
import dpctl.tensor._type_utils as dtu
import dpctl.utils as dpu
import numpy
from dpctl.tensor._numpy_helper import (
    normalize_axis_index,
    normalize_axis_tuple,
)
from dpctl.tensor._type_utils import _acceptance_fn_divide

import dpnp
import dpnp.backend.extensions.ufunc._ufunc_impl as ufi

from .dpnp_algo import dpnp_modf
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
    "fix",
    "float_power",
    "floor",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "gradient",
    "heaviside",
    "imag",
    "maximum",
    "minimum",
    "mod",
    "modf",
    "multiply",
    "nan_to_num",
    "negative",
    "nextafter",
    "positive",
    "pow",
    "power",
    "prod",
    "proj",
    "real",
    "real_if_close",
    "remainder",
    "rint",
    "round",
    "sign",
    "signbit",
    "subtract",
    "sum",
    "trapezoid",
    "true_divide",
    "trunc",
]


def _get_max_min(dtype):
    """Get the maximum and minimum representable values for an inexact dtype."""

    f = dpnp.finfo(dtype)
    return f.max, f.min


def _get_reduction_res_dt(a, dtype, _out):
    """Get a data type used by dpctl for result array in reduction function."""

    if dtype is None:
        return dtu._default_accumulation_dtype(a.dtype, a.sycl_queue)

    dtype = dpnp.dtype(dtype)
    return dtu._to_device_supported_dtype(dtype, a.sycl_device)


def _gradient_build_dx(f, axes, *varargs):
    """Build an array with distance per each dimension."""

    len_axes = len(axes)
    n = len(varargs)
    if n == 0:
        # no spacing argument - use 1 in all axes
        dx = [1.0] * len_axes
    elif n == 1 and numpy.ndim(varargs[0]) == 0:
        dpnp.check_supported_arrays_type(
            varargs[0], scalar_type=True, all_scalars=True
        )

        # single scalar for all axes
        dx = varargs * len_axes
    elif n == len_axes:
        # scalar or 1d array for each axis
        dx = list(varargs)
        for i, distances in enumerate(dx):
            dpnp.check_supported_arrays_type(
                distances, scalar_type=True, all_scalars=True
            )

            if numpy.ndim(distances) == 0:
                continue
            if distances.ndim != 1:
                raise ValueError("distances must be either scalars or 1d")

            if len(distances) != f.shape[axes[i]]:
                raise ValueError(
                    "when 1d, distances must match "
                    "the length of the corresponding dimension"
                )

            if dpnp.issubdtype(distances.dtype, dpnp.integer):
                # Convert integer types to default float type to avoid modular
                # arithmetic in dpnp.diff(distances).
                distances = distances.astype(dpnp.default_float_type())
            diffx = dpnp.diff(distances)

            # if distances are constant reduce to the scalar case
            # since it brings a consistent speedup
            if (diffx == diffx[0]).all():
                diffx = diffx[0]
            dx[i] = diffx
    else:
        raise TypeError("invalid number of arguments")
    return dx


def _gradient_num_diff_2nd_order_interior(
    f, ax_dx, out, slices, axis, uniform_spacing
):
    """Numerical differentiation: 2nd order interior."""

    slice1, slice2, slice3, slice4 = slices
    ndim = f.ndim

    slice1[axis] = slice(1, -1)
    slice2[axis] = slice(None, -2)
    slice3[axis] = slice(1, -1)
    slice4[axis] = slice(2, None)

    if uniform_spacing:
        out[tuple(slice1)] = (f[tuple(slice4)] - f[tuple(slice2)]) / (
            2.0 * ax_dx
        )
    else:
        dx1 = ax_dx[0:-1]
        dx2 = ax_dx[1:]
        a = -(dx2) / (dx1 * (dx1 + dx2))
        b = (dx2 - dx1) / (dx1 * dx2)
        c = dx1 / (dx2 * (dx1 + dx2))

        # fix the shape for broadcasting
        shape = [1] * ndim
        shape[axis] = -1

        a = a.reshape(shape)
        b = b.reshape(shape)
        c = c.reshape(shape)

        # 1D equivalent -- out[1:-1] = a * f[:-2] + b * f[1:-1] + c * f[2:]
        t1 = a * f[tuple(slice2)]
        t2 = b * f[tuple(slice3)]
        t3 = c * f[tuple(slice4)]
        t4 = t1 + t2 + t3

        out[tuple(slice1)] = t4
        out[tuple(slice1)] = (
            a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
        )


def _gradient_num_diff_edges(
    f, ax_dx, out, slices, axis, uniform_spacing, edge_order
):
    """Numerical differentiation: 1st and 2nd order edges."""

    slice1, slice2, slice3, slice4 = slices

    # Numerical differentiation: 1st order edges
    if edge_order == 1:
        slice1[axis] = 0
        slice2[axis] = 1
        slice3[axis] = 0
        dx_0 = ax_dx if uniform_spacing else ax_dx[0]

        # 1D equivalent -- out[0] = (f[1] - f[0]) / (x[1] - x[0])
        out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_0

        slice1[axis] = -1
        slice2[axis] = -1
        slice3[axis] = -2
        dx_n = ax_dx if uniform_spacing else ax_dx[-1]

        # 1D equivalent -- out[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
        out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_n

    # Numerical differentiation: 2nd order edges
    else:
        slice1[axis] = 0
        slice2[axis] = 0
        slice3[axis] = 1
        slice4[axis] = 2
        if uniform_spacing:
            a = -1.5 / ax_dx
            b = 2.0 / ax_dx
            c = -0.5 / ax_dx
        else:
            dx1 = ax_dx[0]
            dx2 = ax_dx[1]
            a = -(2.0 * dx1 + dx2) / (dx1 * (dx1 + dx2))
            b = (dx1 + dx2) / (dx1 * dx2)
            c = -dx1 / (dx2 * (dx1 + dx2))

        # 1D equivalent -- out[0] = a * f[0] + b * f[1] + c * f[2]
        out[tuple(slice1)] = (
            a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
        )

        slice1[axis] = -1
        slice2[axis] = -3
        slice3[axis] = -2
        slice4[axis] = -1
        if uniform_spacing:
            a = 0.5 / ax_dx
            b = -2.0 / ax_dx
            c = 1.5 / ax_dx
        else:
            dx1 = ax_dx[-2]
            dx2 = ax_dx[-1]
            a = (dx2) / (dx1 * (dx1 + dx2))
            b = -(dx2 + dx1) / (dx1 * dx2)
            c = (2.0 * dx2 + dx1) / (dx2 * (dx1 + dx2))

        # 1D equivalent -- out[-1] = a * f[-3] + b * f[-2] + c * f[-1]
        out[tuple(slice1)] = (
            a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
        )


def _process_ediff1d_args(arg, arg_name, ary_dtype, ary_sycl_queue, usm_type):
    """Process the argument for ediff1d."""
    if not dpnp.is_supported_array_type(arg):
        arg = dpnp.asarray(arg, usm_type=usm_type, sycl_queue=ary_sycl_queue)
    else:
        usm_type = dpu.get_coerced_usm_type([usm_type, arg.usm_type])
        # check that arrays have the same allocation queue
        if dpu.get_execution_queue([ary_sycl_queue, arg.sycl_queue]) is None:
            raise dpu.ExecutionPlacementError(
                f"ary and {arg_name} must be allocated on the same SYCL queue"
            )

    if not dpnp.can_cast(arg, ary_dtype, casting="same_kind"):
        raise TypeError(
            f"dtype of {arg_name} must be compatible "
            "with input ary under the `same_kind` rule."
        )

    if arg.ndim > 1:
        arg = dpnp.ravel(arg)

    return arg, usm_type


_ABS_DOCSTRING = """
Calculates the absolute value for each element `x_i` of input array `x`.

For full documentation refer to :obj:`numpy.absolute`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have numeric data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

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
    mkl_fn_to_call="_mkl_abs_to_call",
    mkl_impl_fn="_abs",
)


abs = absolute


_ADD_DOCSTRING = """
Calculates the sum for each element `x1_i` of the input array `x1` with
the respective element `x2_i` of the input array `x2`.

For full documentation refer to :obj:`numpy.add`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

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
    mkl_fn_to_call="_mkl_add_to_call",
    mkl_impl_fn="_add",
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
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

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
:obj:`dpnp.real` : Return the real part of the complex argument.
:obj:`dpnp.imag` : Return the imaginary part of the complex argument.
:obj:`dpnp.real_if_close` : Return the real part of the input is complex
                            with all imaginary parts close to zero.

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
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.
        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        The rounded value of elements of the array.

    See Also
    --------
    :obj:`dpnp.round` : Equivalent function; see for details.
    :obj:`dpnp.ndarray.round` : Equivalent function.
    :obj:`dpnp.rint` : Round elements of the array to the nearest integer.
    :obj:`dpnp.fix` : Round to nearest integer towards zero, element-wise.
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
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

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
:obj:`dpnp.rint` : Round elements of the array to the nearest integer.
:obj:`dpnp.fix` : Round to nearest integer towards zero, element-wise.

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
    mkl_fn_to_call="_mkl_ceil_to_call",
    mkl_impl_fn="_ceil",
)


def clip(a, /, min=None, max=None, *, out=None, order="K", **kwargs):
    """
    Clip (limit) the values in an array.

    For full documentation refer to :obj:`numpy.clip`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Array containing elements to clip.
    min, max : {dpnp.ndarray, usm_ndarray, None}
        Minimum and maximum value. If ``None``, clipping is not performed on
        the corresponding edge. If both `min` and `max` are ``None``,
        the elements of the returned array stay the same.
        Both are broadcast against `a`.
        Default : ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        The results will be placed in this array. It may be the input array
        for in-place clipping. `out` must be of the right shape to hold the
        output. Its type is preserved.
        Default : ``None``.
    order : {"C", "F", "A", "K", None}, optional
        Memory layout of the newly output array, if parameter `out` is ``None``.
        If `order` is ``None``, the default value ``"K"`` will be used.
        Default: ``"K"``.

    Returns
    -------
    out : dpnp.ndarray
        An array with the elements of `a`, but where values < `min` are
        replaced with `min`, and those > `max` with `max`.

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

    if order is None:
        order = "K"

    usm_arr = dpnp.get_usm_ndarray(a)
    usm_min = None if min is None else dpnp.get_usm_ndarray_or_scalar(min)
    usm_max = None if max is None else dpnp.get_usm_ndarray_or_scalar(max)

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
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

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
    mkl_fn_to_call="_mkl_conj_to_call",
    mkl_impl_fn="_conj",
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
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have a real floating-point data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to have a real floating-point data
    type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
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

    cp = dpnp_cross(a, b, cp)
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
    :obj:`dpnp.trapezoid` : Integration of array values using composite
                            trapezoidal rule.
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

    usm_a = dpnp.get_usm_ndarray(a)
    usm_pre = (
        None if prepend is None else dpnp.get_usm_ndarray_or_scalar(prepend)
    )
    usm_app = None if append is None else dpnp.get_usm_ndarray_or_scalar(append)

    usm_res = dpt.diff(usm_a, axis=axis, n=n, prepend=usm_pre, append=usm_app)
    return dpnp_array._create_from_usm_ndarray(usm_res)


_DIVIDE_DOCSTRING = """
Calculates the ratio for each element `x1_i` of the input array `x1` with
the respective element `x2_i` of the input array `x2`.

For full documentation refer to :obj:`numpy.divide`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

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
    mkl_fn_to_call="_mkl_div_to_call",
    mkl_impl_fn="_div",
    binary_inplace_fn=ti._divide_inplace,
    acceptance_fn=_acceptance_fn_divide,
)


def ediff1d(ary, to_end=None, to_begin=None):
    """
    The differences between consecutive elements of an array.

    For full documentation refer to :obj:`numpy.ediff1d`.

    Parameters
    ----------
    ary : {dpnp.ndarray, usm_ndarray}
        If necessary, will be flattened before the differences are taken.
    to_end : array_like, optional
        Number(s) to append at the end of the returned differences.
        Default: ``None``.
    to_begin : array_like, optional
        Number(s) to prepend at the beginning of the returned differences.
        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        New array consisting differences among succeeding elements.
        Loosely, this is ``ary.flat[1:] - ary.flat[:-1]``.

    See Also
    --------
    :obj:`dpnp.diff` : Calculate the n-th discrete difference along the given
                       axis.
    :obj:`dpnp.gradient` : Return the gradient of an N-dimensional array.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([1, 2, 4, 7, 0])
    >>> np.ediff1d(x)
    array([ 1,  2,  3, -7])

    >>> np.ediff1d(x, to_begin=-99, to_end=np.array([88, 99]))
    array([-99,   1,   2,   3,  -7,  88,  99])

    The returned array is always 1D.

    >>> y = np.array([[1, 2, 4], [1, 6, 24]])
    >>> np.ediff1d(y)
    array([ 1,  2, -3,  5, 18])

    """

    dpnp.check_supported_arrays_type(ary)
    if ary.ndim > 1:
        ary = dpnp.ravel(ary)

    # fast track default case
    if to_begin is None and to_end is None:
        return ary[1:] - ary[:-1]

    ary_dtype = ary.dtype
    ary_sycl_queue = ary.sycl_queue
    usm_type = ary.usm_type

    if to_begin is None:
        l_begin = 0
    else:
        to_begin, usm_type = _process_ediff1d_args(
            to_begin, "to_begin", ary_dtype, ary_sycl_queue, usm_type
        )
        l_begin = to_begin.size

    if to_end is None:
        l_end = 0
    else:
        to_end, usm_type = _process_ediff1d_args(
            to_end, "to_end", ary_dtype, ary_sycl_queue, usm_type
        )
        l_end = to_end.size

    # calculating using in place operation
    l_diff = max(len(ary) - 1, 0)
    result = dpnp.empty_like(
        ary, shape=l_diff + l_begin + l_end, usm_type=usm_type
    )

    if l_begin > 0:
        result[:l_begin] = to_begin
    if l_end > 0:
        result[l_begin + l_diff :] = to_end
    dpnp.subtract(ary[1:], ary[:-1], out=result[l_begin : l_begin + l_diff])

    return result


_FABS_DOCSTRING = """
Compute the absolute values element-wise.

This function returns the absolute values (positive magnitude) of the data in
`x`. Complex values are not handled, use :obj:`dpnp.absolute` to find the
absolute values of complex data.

For full documentation refer to :obj:`numpy.fabs`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    The array of numbers for which the absolute values are required.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    The absolute values of `x`, the returned values are always floats.
    If `x` does not have a floating point data type, the returned array
    will have a data type that depends on the capabilities of the device
    on which the array resides.

See Also
--------
:obj:`dpnp.absolute` : Absolute values including `complex` types.

Examples
--------
>>> import dpnp as np
>>> a = np.array([-1.2, 1.2])
>>> np.fabs(a)
array([1.2, 1.2])
"""

fabs = DPNPUnaryFunc(
    "fabs",
    ufi._fabs_result_type,
    ufi._fabs,
    _FABS_DOCSTRING,
    mkl_fn_to_call="_mkl_abs_to_call",
    mkl_impl_fn="_abs",
)


_FIX_DOCSTRING = """
Round to nearest integer towards zero.

Round an array of floats element-wise to nearest integer towards zero.
The rounded values are returned as floats.

For full documentation refer to :obj:`numpy.fix`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    An array of floats to be rounded.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array with the rounded values and with the same dimensions as the input.
    The returned array will have the default floating point data type for the
    device where `a` is allocated.
    If `out` is ``None`` then a float array is returned with the rounded values.
    Otherwise the result is stored there and the return value `out` is
    a reference to that array.

See Also
--------
:obj:`dpnp.round` : Round to given number of decimals.
:obj:`dpnp.rint` : Round elements of the array to the nearest integer.
:obj:`dpnp.trunc` : Return the truncated value of the input, element-wise.
:obj:`dpnp.floor` : Return the floor of the input, element-wise.
:obj:`dpnp.ceil` : Return the ceiling of the input, element-wise.

Examples
--------
>>> import dpnp as np
>>> np.fix(np.array(3.14))
array(3.)
>>> np.fix(np.array(3))
array(3.)
>>> a = np.array([2.1, 2.9, -2.1, -2.9])
>>> np.fix(a)
array([ 2.,  2., -2., -2.])
"""

fix = DPNPUnaryFunc(
    "fix",
    ufi._fix_result_type,
    ufi._fix,
    _FIX_DOCSTRING,
)


_FLOAT_POWER_DOCSTRING = """
Calculates `x1_i` raised to `x2_i` for each element `x1_i` of the input array
`x1` with the respective element `x2_i` of the input array `x2`.

This differs from the power function in that boolean, integers, and float16 are
promoted to floats with a minimum precision of float32 so that the result is
always inexact. The intent is that the function will return a usable result for
negative powers and seldom overflow for positive powers.

Negative values raised to a non-integral value will return ``NaN``. To get
complex results, cast the input to complex, or specify the ``dtype`` to be one
of complex dtype.

For full documentation refer to :obj:`numpy.float_power`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have floating-point data types.
    Both inputs `x1` and `x2` can not be scalars at the same time.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to floating-point data types.
    Both inputs `x1` and `x2` can not be scalars at the same time.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate. Array must have the correct shape and
    the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the bases in `x1` raised to the exponents in `x2`
    element-wise.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.power` : Power function that preserves type.

Examples
--------
>>> import dpnp as np

Cube each element in an array:

>>> x1 = np.arange(6)
>>> x1
array([0, 1, 2, 3, 4, 5])
>>> np.float_power(x1, 3)
array([  0.,   1.,   8.,  27.,  64., 125.])

Raise the bases to different exponents:

>>> x2 = np.array([1.0, 2.0, 3.0, 3.0, 2.0, 1.0])
>>> np.float_power(x1, x2)
array([ 0.,  1.,  8., 27., 16.,  5.])

The effect of broadcasting:

>>> x2 = np.array([[1, 2, 3, 3, 2, 1], [1, 2, 3, 3, 2, 1]])
>>> x2
array([[1, 2, 3, 3, 2, 1],
       [1, 2, 3, 3, 2, 1]])
>>> np.float_power(x1, x2)
array([[ 0.,  1.,  8., 27., 16.,  5.],
       [ 0.,  1.,  8., 27., 16.,  5.]])

Negative values raised to a non-integral value will result in ``NaN``:

>>> x3 = np.array([-1, -4])
>>> np.float_power(x3, 1.5)
array([nan, nan])

To get complex results, give the argument one of complex dtype, i.e.
``dtype=np.complex64``:

>>> np.float_power(x3, 1.5, dtype=np.complex64)
array([1.1924881e-08-1.j, 9.5399045e-08-8.j], dtype=complex64)
"""

float_power = DPNPBinaryFunc(
    "float_power",
    ufi._float_power_result_type,
    ti._pow,
    _FLOAT_POWER_DOCSTRING,
    mkl_fn_to_call="_mkl_pow_to_call",
    mkl_impl_fn="_pow",
    binary_inplace_fn=ti._pow_inplace,
)


_FLOOR_DOCSTRING = """
Returns the floor for each element `x_i` for input array `x`.

The floor of `x_i` is the largest integer `n`, such that `n <= x_i`.

For full documentation refer to :obj:`numpy.floor`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a real-valued data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

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
:obj:`dpnp.rint` : Round elements of the array to the nearest integer.
:obj:`dpnp.fix` : Round to nearest integer towards zero, element-wise.

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
    mkl_fn_to_call="_mkl_floor_to_call",
    mkl_impl_fn="_floor",
)


_FLOOR_DIVIDE_DOCSTRING = """
Calculates the ratio for each element `x1_i` of the input array `x1` with
the respective element `x2_i` of the input array `x2` to the greatest
integer-value number that is not greater than the division result.

For full documentation refer to :obj:`numpy.floor_divide`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

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


_FMAX_DOCSTRING = """
Compares two input arrays `x1` and `x2` and returns a new array containing the
element-wise maxima.

If one of the elements being compared is a NaN, then the non-NaN element is
returned. If both elements are NaNs then the first is returned. The latter
distinction is important for complex NaNs, which are defined as at least one of
the real or imaginary parts being a NaN. The net effect is that NaNs are
ignored when possible.

For full documentation refer to :obj:`numpy.fmax`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

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
:obj:`dpnp.fmin` : Element-wise minimum of two arrays, ignores NaNs.
:obj:`dpnp.maximum` : Element-wise maximum of two arrays, propagates NaNs.
:obj:`dpnp.max` : The maximum value of an array along a given axis, propagates NaNs.
:obj:`dpnp.nanmax` : The maximum value of an array along a given axis, ignores NaNs.
:obj:`dpnp.minimum` : Element-wise minimum of two arrays, propagates NaNs.
:obj:`dpnp.min` : The minimum value of an array along a given axis, propagates NaNs.
:obj:`dpnp.nanmin` : The minimum value of an array along a given axis, ignores NaNs.

Notes
-----
``fmax(x1, x2)`` is equivalent to ``dpnp.where(x1 >= x2, x1, x2)`` when neither
`x1` nor `x2` are NaNs, but it is faster and does proper broadcasting.

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([2, 3, 4])
>>> x2 = np.array([1, 5, 2])
>>> np.fmax(x1, x2)
array([2, 5, 4])

>>> x1 = np.eye(2)
>>> x2 = np.array([0.5, 2])
>>> np.fmax(x1, x2)
array([[1. , 2. ],
       [0.5, 2. ]])

>>> x1 = np.array([np.nan, 0, np.nan])
>>> x2 = np.array([0, np.nan, np.nan])
>>> np.fmax(x1, x2)
array([ 0.,  0., nan])
"""

fmax = DPNPBinaryFunc(
    "fmax",
    ufi._fmax_result_type,
    ufi._fmax,
    _FMAX_DOCSTRING,
    mkl_fn_to_call="_mkl_fmax_to_call",
    mkl_impl_fn="_fmax",
)


_FMIN_DOCSTRING = """
Compares two input arrays `x1` and `x2` and returns a new array containing the
element-wise minima.

If one of the elements being compared is a NaN, then the non-NaN element is
returned. If both elements are NaNs then the first is returned. The latter
distinction is important for complex NaNs, which are defined as at least one of
the real or imaginary parts being a NaN. The net effect is that NaNs are
ignored when possible.

For full documentation refer to :obj:`numpy.fmin`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

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
:obj:`dpnp.fmax` : Element-wise maximum of two arrays, ignores NaNs.
:obj:`dpnp.minimum` : Element-wise minimum of two arrays, propagates NaNs.
:obj:`dpnp.min` : The minimum value of an array along a given axis, propagates NaNs.
:obj:`dpnp.nanmin` : The minimum value of an array along a given axis, ignores NaNs.
:obj:`dpnp.maximum` : Element-wise maximum of two arrays, propagates NaNs.
:obj:`dpnp.max` : The maximum value of an array along a given axis, propagates NaNs.
:obj:`dpnp.nanmax` : The maximum value of an array along a given axis, ignores NaNs.

Notes
-----
``fmin(x1, x2)`` is equivalent to ``dpnp.where(x1 <= x2, x1, x2)`` when neither
`x1` nor `x2` are NaNs, but it is faster and does proper broadcasting.

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([2, 3, 4])
>>> x2 = np.array([1, 5, 2])
>>> np.fmin(x1, x2)
array([1, 3, 2])

>>> x1 = np.eye(2)
>>> x2 = np.array([0.5, 2])
>>> np.fmin(x1, x2)
array([[0.5, 0. ],
       [0. , 1. ]])

>>> x1 = np.array([np.nan, 0, np.nan])
>>> x2 = np.array([0, np.nan, np.nan])
>>> np.fmin(x1, x2)
array([ 0.,  0., nan])
"""

fmin = DPNPBinaryFunc(
    "fmin",
    ufi._fmin_result_type,
    ufi._fmin,
    _FMIN_DOCSTRING,
    mkl_fn_to_call="_mkl_fmin_to_call",
    mkl_impl_fn="_fmin",
)


_FMOD_DOCSTRING = """
Calculates the remainder of division for each element `x1_i` of the input array
`x1` with the respective element `x2_i` of the input array `x2`.

This function is equivalent to the Matlab(TM) ``rem`` function and should not
be confused with the Python modulus operator ``x1 % x2``.

For full documentation refer to :obj:`numpy.fmod`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have a real-valued data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to have a real-valued data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise remainders. The data type of the
    returned array is determined by the Type Promotion Rules.

Limitations
----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.remainder` : Equivalent to the Python ``%`` operator.
:obj:`dpnp.divide` : Standard division.

Examples
--------
>>> import dpnp as np
>>> a = np.array([-3, -2, -1, 1, 2, 3])
>>> np.fmod(a, 2)
array([-1,  0, -1,  1,  0,  1])
>>> np.remainder(a, 2)
array([1, 0, 1, 1, 0, 1])

>>> np.fmod(np.array([5, 3]), np.array([2, 2.]))
array([1., 1.])
>>> a = np.arange(-3, 3).reshape(3, 2)
>>> a
array([[-3, -2],
       [-1,  0],
       [ 1,  2]])
>>> np.fmod(a, np.array([2, 2]))
array([[-1,  0],
       [-1,  0],
       [ 1,  0]])
"""

fmod = DPNPBinaryFunc(
    "fmod",
    ufi._fmod_result_type,
    ufi._fmod,
    _FMOD_DOCSTRING,
    mkl_fn_to_call="_mkl_fmod_to_call",
    mkl_impl_fn="_fmod",
)


def gradient(f, *varargs, axis=None, edge_order=1):
    """
    Return the gradient of an N-dimensional array.

    The gradient is computed using second order accurate central differences
    in the interior points and either first or second order accurate one-sides
    (forward or backwards) differences at the boundaries.
    The returned gradient hence has the same shape as the input array.

    For full documentation refer to :obj:`numpy.gradient`.

    Parameters
    ----------
    f : {dpnp.ndarray, usm_ndarray}
        An N-dimensional array containing samples of a scalar function.
    varargs : {scalar, list of scalars, list of arrays}, optional
        Spacing between `f` values. Default unitary spacing for all dimensions.
        Spacing can be specified using:

        1. Single scalar to specify a sample distance for all dimensions.
        2. N scalars to specify a constant sample distance for each dimension.
           i.e. `dx`, `dy`, `dz`, ...
        3. N arrays to specify the coordinates of the values along each
           dimension of `f`. The length of the array must match the size of
           the corresponding dimension
        4. Any combination of N scalars/arrays with the meaning of 2. and 3.

        If `axis` is given, the number of `varargs` must equal the number of
        axes.
        Default: ``1``.
    axis : {None, int, tuple of ints}, optional
        Gradient is calculated only along the given axis or axes.
        The default is to calculate the gradient for all the axes of the input
        array. `axis` may be negative, in which case it counts from the last to
        the first axis.
        Default: ``None``.
    edge_order : {1, 2}, optional
        Gradient is calculated using N-th order accurate differences
        at the boundaries.
        Default: ``1``.

    Returns
    -------
    gradient : {dpnp.ndarray, list of ndarray}
        A list of :class:`dpnp.ndarray` (or a single :class:`dpnp.ndarray` if
        there is only one dimension) corresponding to the derivatives of `f`
        with respect to each dimension.
        Each derivative has the same shape as `f`.

    See Also
    --------
    :obj:`dpnp.diff` : Calculate the n-th discrete difference along the given
                       axis.
    :obj:`dpnp.ediff1d` : Calculate the differences between consecutive
                          elements of an array.

    Examples
    --------
    >>> import dpnp as np
    >>> f = np.array([1, 2, 4, 7, 11, 16], dtype=float)
    >>> np.gradient(f)
    array([1. , 1.5, 2.5, 3.5, 4.5, 5. ])
    >>> np.gradient(f, 2)
    array([0.5 , 0.75, 1.25, 1.75, 2.25, 2.5 ])

    Spacing can be also specified with an array that represents the coordinates
    of the values `f` along the dimensions.
    For instance a uniform spacing:

    >>> x = np.arange(f.size)
    >>> np.gradient(f, x)
    array([1. , 1.5, 2.5, 3.5, 4.5, 5. ])

    Or a non uniform one:

    >>> x = np.array([0., 1., 1.5, 3.5, 4., 6.], dtype=float)
    >>> np.gradient(f, x)
    array([1. , 3. , 3.5, 6.7, 6.9, 2.5])

    For two dimensional arrays, the return will be two arrays ordered by
    axis. In this example the first array stands for the gradient in
    rows and the second one in columns direction:

    >>> np.gradient(np.array([[1, 2, 6], [3, 4, 5]], dtype=float))
    (array([[ 2.,  2., -1.],
            [ 2.,  2., -1.]]),
     array([[1. , 2.5, 4. ],
            [1. , 1. , 1. ]]))

    In this example the spacing is also specified:
    uniform for axis=0 and non uniform for axis=1

    >>> dx = 2.
    >>> y = np.array([1., 1.5, 3.5])
    >>> np.gradient(np.array([[1, 2, 6], [3, 4, 5]], dtype=float), dx, y)
    (array([[ 1. ,  1. , -0.5],
            [ 1. ,  1. , -0.5]]),
     array([[2. , 2. , 2. ],
            [2. , 1.7, 0.5]]))

    It is possible to specify how boundaries are treated using `edge_order`

    >>> x = np.array([0, 1, 2, 3, 4])
    >>> f = x**2
    >>> np.gradient(f, edge_order=1)
    array([1., 2., 4., 6., 7.])
    >>> np.gradient(f, edge_order=2)
    array([0., 2., 4., 6., 8.])

    The `axis` keyword can be used to specify a subset of axes of which the
    gradient is calculated

    >>> np.gradient(np.array([[1, 2, 6], [3, 4, 5]], dtype=float), axis=0)
    array([[ 2.,  2., -1.],
           [ 2.,  2., -1.]])

    """

    dpnp.check_supported_arrays_type(f)
    ndim = f.ndim  # number of dimensions

    if axis is None:
        axes = tuple(range(ndim))
    else:
        axes = normalize_axis_tuple(axis, ndim)

    dx = _gradient_build_dx(f, axes, *varargs)
    if edge_order > 2:
        raise ValueError("'edge_order' greater than 2 not supported")

    # Use central differences on interior and one-sided differences on the
    # endpoints. This preserves second order-accuracy over the full domain.
    outvals = []

    # create slice objects --- initially all are [:, :, ..., :]
    slice1 = [slice(None)] * ndim
    slice2 = [slice(None)] * ndim
    slice3 = [slice(None)] * ndim
    slice4 = [slice(None)] * ndim

    otype = f.dtype
    if dpnp.issubdtype(otype, dpnp.inexact):
        pass
    else:
        # All other types convert to floating point.
        # First check if f is a dpnp integer type; if so, convert f to default
        # float type to avoid modular arithmetic when computing changes in f.
        if dpnp.issubdtype(otype, dpnp.integer):
            f = f.astype(dpnp.default_float_type())
        otype = dpnp.default_float_type()

    for axis_, ax_dx in zip(axes, dx):
        if f.shape[axis_] < edge_order + 1:
            raise ValueError(
                "Shape of array too small to calculate a numerical gradient, "
                "at least (edge_order + 1) elements are required."
            )

        # result allocation
        if dpnp.isscalar(ax_dx):
            usm_type = f.usm_type
        else:
            usm_type = dpu.get_coerced_usm_type([f.usm_type, ax_dx.usm_type])
        out = dpnp.empty_like(f, dtype=otype, usm_type=usm_type)

        # spacing for the current axis
        uniform_spacing = numpy.ndim(ax_dx) == 0

        # Numerical differentiation: 2nd order interior
        _gradient_num_diff_2nd_order_interior(
            f,
            ax_dx,
            out,
            (slice1, slice2, slice3, slice4),
            axis_,
            uniform_spacing,
        )

        # Numerical differentiation: 1st and 2nd order edges
        _gradient_num_diff_edges(
            f,
            ax_dx,
            out,
            (slice1, slice2, slice3, slice4),
            axis_,
            uniform_spacing,
            edge_order,
        )

        outvals.append(out)

        # reset the slice object in this dimension to ":"
        slice1[axis_] = slice(None)
        slice2[axis_] = slice(None)
        slice3[axis_] = slice(None)
        slice4[axis_] = slice(None)

    if len(axes) == 1:
        return outvals[0]
    return tuple(outvals)


_HEAVISIDE_DOCSTRING = """
Compute the Heaviside step function.

The Heaviside step function is defined as::

                          0   if x1 < 0
    heaviside(x1, x2) =  x2   if x1 == 0
                          1   if x1 > 0

where `x2` is often taken to be 0.5, but 0 and 1 are also sometimes used.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    Input values.
    Both inputs `x1` and `x2` can not be scalars at the same time.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    The value of the function when `x1` is ``0``.
    Both inputs `x1` and `x2` can not be scalars at the same time.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    The output array, element-wise Heaviside step function of `x1`.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

Examples
--------
>>> import dpnp as np
>>> a = np.array([-1.5, 0, 2.0])
>>> np.heaviside(a, 0.5)
array([0. , 0.5, 1. ])
>>> np.heaviside(a, 1)
array([0., 1., 1.])
"""

heaviside = DPNPBinaryFunc(
    "heaviside",
    ufi._heaviside_result_type,
    ufi._heaviside,
    _HEAVISIDE_DOCSTRING,
)


_IMAG_DOCSTRING = """
Computes imaginary part of each element `x_i` for input array `x`.

For full documentation refer to :obj:`numpy.imag`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have numeric data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

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
:obj:`dpnp.angle` : Return the angle of the complex argument.
:obj:`dpnp.real_if_close` : Return the real part of the input is complex
                            with all imaginary parts close to zero.
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

If one of the elements being compared is a NaN, then that element is returned.
If both elements are NaNs then the first is returned. The latter distinction is
important for complex NaNs, which are defined as at least one of the real or
imaginary parts being a NaN. The net effect is that NaNs are propagated.

For full documentation refer to :obj:`numpy.maximum`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

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

>>> np.maximum(np.array(np.inf), 1)
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

If one of the elements being compared is a NaN, then that element is returned.
If both elements are NaNs then the first is returned. The latter distinction is
important for complex NaNs, which are defined as at least one of the real or
imaginary parts being a NaN. The net effect is that NaNs are propagated.

For full documentation refer to :obj:`numpy.minimum`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

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

>>> np.minimum(np.array(-np.inf), 1)
array(-inf)
"""

minimum = DPNPBinaryFunc(
    "minimum",
    ti._minimum_result_type,
    ti._minimum,
    _MINIMUM_DOCSTRING,
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
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

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
    mkl_fn_to_call="_mkl_mul_to_call",
    mkl_impl_fn="_mul",
    binary_inplace_fn=ti._multiply_inplace,
)


def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    """
    Replace ``NaN`` with zero and infinity with large finite numbers (default
    behavior) or with the numbers defined by the user using the `nan`,
    `posinf` and/or `neginf` keywords.

    If `x` is inexact, ``NaN`` is replaced by zero or by the user defined value
    in `nan` keyword, infinity is replaced by the largest finite floating point
    values representable by ``x.dtype`` or by the user defined value in
    `posinf` keyword and -infinity is replaced by the most negative finite
    floating point values representable by ``x.dtype`` or by the user defined
    value in `neginf` keyword.

    For complex dtypes, the above is applied to each of the real and
    imaginary components of `x` separately.

    If `x` is not inexact, then no replacements are made.

    For full documentation refer to :obj:`numpy.nan_to_num`.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        Input data.
    copy : bool, optional
        Whether to create a copy of `x` (``True``) or to replace values
        in-place (``False``). The in-place operation only occurs if casting to
        an array does not require a copy.
    nan : {int, float, bool}, optional
        Value to be used to fill ``NaN`` values.
        Default: ``0.0``.
    posinf : {int, float, bool, None}, optional
        Value to be used to fill positive infinity values. If no value is
        passed then positive infinity values will be replaced with a very
        large number.
        Default: ``None``.
    neginf : {int, float, bool, None} optional
        Value to be used to fill negative infinity values. If no value is
        passed then negative infinity values will be replaced with a very
        small (or negative) number.
        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        `x`, with the non-finite values replaced. If `copy` is ``False``, this
        may be `x` itself.

    See Also
    --------
    :obj:`dpnp.isinf` : Shows which elements are positive or negative infinity.
    :obj:`dpnp.isneginf` : Shows which elements are negative infinity.
    :obj:`dpnp.isposinf` : Shows which elements are positive infinity.
    :obj:`dpnp.isnan` : Shows which elements are Not a Number (NaN).
    :obj:`dpnp.isfinite` : Shows which elements are finite
                           (not NaN, not infinity)

    Examples
    --------
    >>> import dpnp as np
    >>> np.nan_to_num(np.array(np.inf))
    array(1.79769313e+308)
    >>> np.nan_to_num(np.array(-np.inf))
    array(-1.79769313e+308)
    >>> np.nan_to_num(np.array(np.nan))
    array(0.)
    >>> x = np.array([np.inf, -np.inf, np.nan, -128, 128])
    >>> np.nan_to_num(x)
    array([ 1.79769313e+308, -1.79769313e+308,  0.00000000e+000,
           -1.28000000e+002,  1.28000000e+002])
    >>> np.nan_to_num(x, nan=-9999, posinf=33333333, neginf=33333333)
    array([ 3.3333333e+07,  3.3333333e+07, -9.9990000e+03, -1.2800000e+02,
            1.2800000e+02])
    >>> y = np.array([complex(np.inf, np.nan), np.nan, complex(np.nan, np.inf)])
    >>> np.nan_to_num(y)
    array([1.79769313e+308 +0.00000000e+000j, # may vary
           0.00000000e+000 +0.00000000e+000j,
           0.00000000e+000 +1.79769313e+308j])
    >>> np.nan_to_num(y, nan=111111, posinf=222222)
    array([222222.+111111.j, 111111.     +0.j, 111111.+222222.j])

    """

    dpnp.check_supported_arrays_type(x)

    # Python boolean is a subtype of an integer
    # so additional check for bool is not needed.
    if not isinstance(nan, (int, float)):
        raise TypeError(
            "nan must be a scalar of an integer, float, bool, "
            f"but got {type(nan)}"
        )

    out = dpnp.empty_like(x) if copy else x
    x_type = x.dtype.type

    if not issubclass(x_type, dpnp.inexact):
        return x

    parts = (
        (x.real, x.imag) if issubclass(x_type, dpnp.complexfloating) else (x,)
    )
    parts_out = (
        (out.real, out.imag)
        if issubclass(x_type, dpnp.complexfloating)
        else (out,)
    )
    max_f, min_f = _get_max_min(x.real.dtype)
    if posinf is not None:
        if not isinstance(posinf, (int, float)):
            raise TypeError(
                "posinf must be a scalar of an integer, float, bool, "
                f"or be None, but got {type(posinf)}"
            )
        max_f = posinf
    if neginf is not None:
        if not isinstance(neginf, (int, float)):
            raise TypeError(
                "neginf must be a scalar of an integer, float, bool, "
                f"or be None, but got {type(neginf)}"
            )
        min_f = neginf

    for part, part_out in zip(parts, parts_out):
        nan_mask = dpnp.isnan(part)
        posinf_mask = dpnp.isposinf(part)
        neginf_mask = dpnp.isneginf(part)

        part = dpnp.where(nan_mask, nan, part, out=part_out)
        part = dpnp.where(posinf_mask, max_f, part, out=part_out)
        part = dpnp.where(neginf_mask, min_f, part, out=part_out)

    return out


_NEGATIVE_DOCSTRING = """
Computes the numerical negative for each element `x_i` of input array `x`.

For full documentation refer to :obj:`numpy.negative`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have numeric data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

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


_NEXTAFTER_DOCSTRING = """
Return the next floating-point value after `x1` towards `x2`, element-wise.

For full documentation refer to :obj:`numpy.nextafter`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    Values to find the next representable value of.
    Both inputs `x1` and `x2` can not be scalars at the same time.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    The direction where to look for the next representable value of `x1`.
    Both inputs `x1` and `x2` can not be scalars at the same time.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate. Array must have the correct shape and
    the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    The next representable values of `x1` in the direction of `x2`. The data
    type of the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

Examples
--------
>>> import dpnp as np
>>> a = np.array(1, dtype=np.float32)
>>> eps = np.finfo(a.dtype).eps
>>> np.nextafter(a, 2) == eps + 1
array(True)

>>> a = np.array([1, 2], dtype=np.float32)
>>> b = np.array([2, 1], dtype=np.float32)
>>> c = np.array([eps + 1, 2 - eps])
>>> np.nextafter(a, b) == c
array([ True,  True])
"""

nextafter = DPNPBinaryFunc(
    "nextafter",
    ti._nextafter_result_type,
    ti._nextafter,
    _NEXTAFTER_DOCSTRING,
    mkl_fn_to_call="_mkl_nextafter_to_call",
    mkl_impl_fn="_nextafter",
)


_POSITIVE_DOCSTRING = """
Computes the numerical positive for each element `x_i` of input array `x`.

For full documentation refer to :obj:`numpy.positive`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have numeric data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

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

Note that :obj:`dpnp.pow` is an alias of :obj:`dpnp.power`.

For full documentation refer to :obj:`numpy.power`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate. Array must have the correct shape and
    the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

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
:obj:`dpnp.float_power` : Power function that promotes integers to floats.

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
    mkl_fn_to_call="_mkl_pow_to_call",
    mkl_impl_fn="_pow",
    binary_inplace_fn=ti._pow_inplace,
)

pow = power  # pow is an alias for power


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
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

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

See Also
--------
:obj:`dpnp.real_if_close` : Return the real part of the input is complex
                            with all imaginary parts close to zero.
:obj:`dpnp.imag` : Return the imaginary part of the complex argument.
:obj:`dpnp.angle` : Return the angle of the complex argument.

Examples
--------
>>> import dpnp as np
>>> a = np.array([1+2j, 3+4j, 5+6j])
>>> a.real
array([1., 3., 5.])
>>> a.real = 9
>>> a
array([9.+2.j, 9.+4.j, 9.+6.j])
>>> a.real = np.array([9, 8, 7])
>>> a
array([9.+2.j, 8.+4.j, 7.+6.j])
>>> np.real(np.array(1 + 1j))
array(1.)
"""

real = DPNPReal(
    "real",
    ti._real_result_type,
    ti._real,
    _REAL_DOCSTRING,
)


def real_if_close(a, tol=100):
    """
    If input is complex with all imaginary parts close to zero, return real
    parts.

    "Close to zero" is defined as `tol` * (machine epsilon of the type for `a`).

    For full documentation refer to :obj:`numpy.real_if_close`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    tol : scalar, optional
        Tolerance in machine epsilons for the complex part of the elements in
        the array. If the tolerance is <=1, then the absolute tolerance is used.
        Default: ``100``.

    Returns
    -------
    out : dpnp.ndarray
        If `a` is real, the type of `a` is used for the output. If `a` has
        complex elements, the returned type is float.

    See Also
    --------
    :obj:`dpnp.real` : Return the real part of the complex argument.
    :obj:`dpnp.imag` : Return the imaginary part of the complex argument.
    :obj:`dpnp.angle` : Return the angle of the complex argument.

    Examples
    --------
    >>> import dpnp as np
    >>> np.finfo(np.float64).eps
    2.220446049250313e-16 # may vary

    >>> a = np.array([2.1 + 4e-14j, 5.2 + 3e-15j])
    >>> np.real_if_close(a, tol=1000)
    array([2.1, 5.2])

    >>> a = np.array([2.1 + 4e-13j, 5.2 + 3e-15j])
    >>> np.real_if_close(a, tol=1000)
    array([2.1+4.e-13j, 5.2+3.e-15j])

    """

    dpnp.check_supported_arrays_type(a)

    if not dpnp.issubdtype(a.dtype, dpnp.complexfloating):
        return a

    if not dpnp.isscalar(tol):
        raise TypeError(f"Tolerance must be a scalar, but got {type(tol)}")

    if tol > 1:
        f = dpnp.finfo(a.dtype.type)
        tol = f.eps * tol

    if dpnp.all(dpnp.abs(a.imag) < tol):
        return a.real
    return a


_REMAINDER_DOCSTRING = """
Calculates the remainder of division for each element `x1_i` of the input array
`x1` with the respective element `x2_i` of the input array `x2`.

This function is equivalent to the Python modulus operator.

For full documentation refer to :obj:`numpy.remainder`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have a real-valued data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to have a real-valued data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise remainders. Each remainder has the
    same sign as respective element `x2_i`. The data type of the returned
    array is determined by the Type Promotion Rules.

Limitations
-----------
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

Notes
-----
Returns ``0`` when `x2` is ``0`` and both `x1` and `x2` are (arrays of)
integers.
:obj:`dpnp.mod` is an alias of :obj:`dpnp.remainder`.

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

mod = remainder


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
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

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
:obj:`dpnp.fix` : Round to nearest integer towards zero, element-wise.
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
    mkl_fn_to_call="_mkl_round_to_call",
    mkl_impl_fn="_round",
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
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.

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
:obj:`dpnp.fix` : Round to nearest integer towards zero, element-wise.
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
    mkl_fn_to_call="_mkl_round_to_call",
    mkl_impl_fn="_round",
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
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

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
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

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
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

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
    mkl_fn_to_call="_mkl_sub_to_call",
    mkl_impl_fn="_sub",
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
    :obj:`dpnp.trapezoid` : Integration of array values using the composite
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


def trapezoid(y, x=None, dx=1.0, axis=-1):
    r"""
    Integrate along the given axis using the composite trapezoidal rule.

    If `x` is provided, the integration happens in sequence along its elements -
    they are not sorted.

    Integrate `y` (`x`) along each 1d slice on the given axis, compute
    :math:`\int y(x) dx`.
    When `x` is specified, this integrates along the parametric curve,
    computing :math:`\int_t y(t) dt =
    \int_t y(t) \left.\frac{dx}{dt}\right|_{x=x(t)} dt`.

    For full documentation refer to :obj:`numpy.trapezoid`.

    Parameters
    ----------
    y : {dpnp.ndarray, usm_ndarray}
        Input array to integrate.
    x : {dpnp.ndarray, usm_ndarray, None}, optional
        The sample points corresponding to the `y` values. If `x` is ``None``,
        the sample points are assumed to be evenly spaced `dx` apart.
        Default: ``None``.
    dx : scalar, optional
        The spacing between sample points when `x` is ``None``.
        Default: ``1``.
    axis : int, optional
        The axis along which to integrate.
        Default: ``-1``.

    Returns
    -------
    out : dpnp.ndarray
        Definite integral of `y` = n-dimensional array as approximated along
        a single axis by the trapezoidal rule. The result is an `n`-1
        dimensional array.

    See Also
    --------
    :obj:`dpnp.sum` : Sum of array elements over a given axis.
    :obj:`dpnp.cumsum` : Cumulative sum of the elements along a given axis.

    Examples
    --------
    >>> import dpnp as np

    Use the trapezoidal rule on evenly spaced points:

    >>> y = np.array([1, 2, 3])
    >>> np.trapezoid(y)
    array(4.)

    The spacing between sample points can be selected by either the `x` or `dx`
    arguments:

    >>> y = np.array([1, 2, 3])
    >>> x = np.array([4, 6, 8])
    >>> np.trapezoid(y, x=x)
    array(8.)
    >>> np.trapezoid(y, dx=2)
    array(8.)

    Using a decreasing `x` corresponds to integrating in reverse:

    >>> y = np.array([1, 2, 3])
    >>> x = np.array([8, 6, 4])
    >>> np.trapezoid(y, x=x)
    array(-8.)

    More generally `x` is used to integrate along a parametric curve. We can
    estimate the integral :math:`\int_0^1 x^2 = 1/3` using:

    >>> x = np.linspace(0, 1, num=50)
    >>> y = x**2
    >>> np.trapezoid(y, x)
    array(0.33340275)

    Or estimate the area of a circle, noting we repeat the sample which closes
    the curve:

    >>> theta = np.linspace(0, 2 * np.pi, num=1000, endpoint=True)
    >>> np.trapezoid(np.cos(theta), x=np.sin(theta))
    array(3.14157194)

    :obj:`dpnp.trapezoid` can be applied along a specified axis to do multiple
    computations in one call:

    >>> a = np.arange(6).reshape(2, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.trapezoid(a, axis=0)
    array([1.5, 2.5, 3.5])
    >>> np.trapezoid(a, axis=1)
    array([2., 8.])

    """

    dpnp.check_supported_arrays_type(y)
    nd = y.ndim

    if x is None:
        d = dx
    else:
        dpnp.check_supported_arrays_type(x)
        if x.ndim == 1:
            d = dpnp.diff(x)

            # reshape to correct shape
            shape = [1] * nd
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = dpnp.diff(x, axis=axis)

    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    return (d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0).sum(axis)


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
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

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
:obj:`dpnp.rint` : Round elements of the array to the nearest integer.
:obj:`dpnp.fix` : Round to nearest integer towards zero, element-wise.

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
    mkl_fn_to_call="_mkl_trunc_to_call",
    mkl_impl_fn="_trunc",
)
