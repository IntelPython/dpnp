# cython: language_level=3
# distutils: language = c++
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2023, Intel Corporation
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
Interface of the Logic part of the DPNP

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""


import dpctl.tensor as dpt
import numpy

import dpnp
from dpnp.dpnp_algo import *
from dpnp.dpnp_array import dpnp_array
from dpnp.dpnp_utils import *

from .dpnp_algo.dpnp_elementwise_common import (
    check_nd_call_func,
    dpnp_equal,
    dpnp_greater,
    dpnp_greater_equal,
    dpnp_isfinite,
    dpnp_isinf,
    dpnp_isnan,
    dpnp_less,
    dpnp_less_equal,
    dpnp_logical_and,
    dpnp_logical_not,
    dpnp_logical_or,
    dpnp_logical_xor,
    dpnp_not_equal,
)

__all__ = [
    "all",
    "allclose",
    "any",
    "equal",
    "greater",
    "greater_equal",
    "isclose",
    "isfinite",
    "isinf",
    "isnan",
    "less",
    "less_equal",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "not_equal",
]


def all(x, /, axis=None, out=None, keepdims=False, *, where=True):
    """
    Test whether all array elements along a given axis evaluate to True.

    For full documentation refer to :obj:`numpy.all`.

    Returns
    -------
    out : dpnp.ndarray
        An array with a data type of `bool`
        containing the results of the logical AND reduction.

    Limitations
    -----------
    Parameters `x` is supported either as :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `out` and `where` are supported with default value.
    Input array data types are limited by supported DPNP :ref:`Data types`.
    Otherwise the function will be executed sequentially on CPU.

    See Also
    --------
    :obj:`dpnp.ndarray.all` : equivalent method
    :obj:`dpnp.any` : Test whether any element along a given axis evaluates to True.

    Notes
    -----
    Not a Number (NaN), positive infinity and negative infinity
    evaluate to `True` because these are not equal to zero.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([[True, False], [True, True]])
    >>> np.all(x)
    array(False)

    >>> np.all(x, axis=0)
    array([ True, False])

    >>> x2 = np.array([-1, 4, 5])
    >>> np.all(x2)
    array(True)

    >>> x3 = np.array([1.0, np.nan])
    >>> np.all(x3)
    array(True)

    """

    if dpnp.is_supported_array_type(x):
        if out is not None:
            pass
        elif where is not True:
            pass
        else:
            dpt_array = dpnp.get_usm_ndarray(x)
            return dpnp_array._create_from_usm_ndarray(
                dpt.all(dpt_array, axis=axis, keepdims=keepdims)
            )

    return call_origin(
        numpy.all, x, axis=axis, out=out, keepdims=keepdims, where=where
    )


def allclose(a, b, rtol=1.0e-5, atol=1.0e-8, **kwargs):
    """
    Returns True if two arrays are element-wise equal within a tolerance.

    For full documentation refer to :obj:`numpy.allclose`.

    Returns
    -------
    out : dpnp.ndarray
        A boolean 0-dim array. If its value is ``True``,
        two arrays are element-wise equal within a tolerance.

    Limitations
    -----------
    Parameters `a` and `b` are supported either as :class:`dpnp.ndarray`,
    :class:`dpctl.tensor.usm_ndarray` or scalars, but both `a` and `b`
    can not be scalars at the same time.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the functions will be executed sequentially on CPU.
    Parameters `rtol` and `atol` are supported as scalars. Otherwise
    ``TypeError`` exception will be raised.
    Input array data types are limited by supported integer and
    floating DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.isclose` : Test whether two arrays are element-wise equal.
    :obj:`dpnp.all` : Test whether all elements evaluate to True.
    :obj:`dpnp.any` : Test whether any element evaluates to True.
    :obj:`dpnp.equal` : Return (x1 == x2) element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1e10, 1e-7])
    >>> b = np.array([1.00001e10, 1e-8])
    >>> np.allclose(a, b)
    array([False])

    >>> a = np.array([1.0, np.nan])
    >>> b = np.array([1.0, np.nan])
    >>> np.allclose(a, b)
    array([False])

    >>> a = np.array([1.0, np.inf])
    >>> b = np.array([1.0, np.inf])
    >>> np.allclose(a, b)
    array([ True])

    """

    if dpnp.isscalar(a) and dpnp.isscalar(b):
        # at least one of inputs has to be an array
        pass
    elif not (
        dpnp.is_supported_array_or_scalar(a)
        and dpnp.is_supported_array_or_scalar(b)
    ):
        pass
    elif kwargs:
        pass
    else:
        if not dpnp.isscalar(rtol):
            raise TypeError(
                "An argument `rtol` must be a scalar, but got {}".format(
                    type(rtol)
                )
            )
        elif not dpnp.isscalar(atol):
            raise TypeError(
                "An argument `atol` must be a scalar, but got {}".format(
                    type(atol)
                )
            )

        if dpnp.isscalar(a):
            a = dpnp.full_like(b, fill_value=a)
        elif dpnp.isscalar(b):
            b = dpnp.full_like(a, fill_value=b)
        elif a.shape != b.shape:
            a, b = dpt.broadcast_arrays(a.get_array(), b.get_array())

        a_desc = dpnp.get_dpnp_descriptor(a, copy_when_nondefault_queue=False)
        b_desc = dpnp.get_dpnp_descriptor(b, copy_when_nondefault_queue=False)
        if a_desc and b_desc:
            return dpnp_allclose(a_desc, b_desc, rtol, atol).get_pyobj()

    return call_origin(numpy.allclose, a, b, rtol=rtol, atol=atol, **kwargs)


def any(x, /, axis=None, out=None, keepdims=False, *, where=True):
    """
    Test whether any array element along a given axis evaluates to True.

    For full documentation refer to :obj:`numpy.any`.

    Returns
    -------
    out : dpnp.ndarray
        An array with a data type of `bool`
        containing the results of the logical OR reduction.

    Limitations
    -----------
    Parameters `x` is supported either as :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `out` and `where` are supported with default value.
    Input array data types are limited by supported DPNP :ref:`Data types`.
    Otherwise the function will be executed sequentially on CPU.

    See Also
    --------
    :obj:`dpnp.ndarray.any` : equivalent method
    :obj:`dpnp.all` : Test whether all elements along a given axis evaluate to True.

    Notes
    -----
    Not a Number (NaN), positive infinity and negative infinity evaluate
    to `True` because these are not equal to zero.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([[True, False], [True, True]])
    >>> np.any(x)
    array(True)

    >>> np.any(x, axis=0)
    array([ True,  True])

    >>> x2 = np.array([0, 0, 0])
    >>> np.any(x2)
    array(False)

    >>> x3 = np.array([1.0, np.nan])
    >>> np.any(x3)
    array(True)

    """

    if dpnp.is_supported_array_type(x):
        if out is not None:
            pass
        elif where is not True:
            pass
        else:
            dpt_array = dpnp.get_usm_ndarray(x)
            return dpnp_array._create_from_usm_ndarray(
                dpt.any(dpt_array, axis=axis, keepdims=keepdims)
            )

    return call_origin(
        numpy.any, x, axis=axis, out=out, keepdims=keepdims, where=where
    )


def equal(
    x1,
    x2,
    /,
    out=None,
    *,
    order="K",
    where=True,
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Return the truth value of (x1 == x2) element-wise.

    For full documentation refer to :obj:`numpy.equal`.

    Returns
    -------
    out : dpnp.ndarray
        Output array of bool type, element-wise comparison of `x1` and `x2`.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.not_equal` : Return (x1 != x2) element-wise.
    :obj:`dpnp.greater_equal` : Return the truth value of (x1 >= x2) element-wise.
    :obj:`dpnp.less_equal` : Return the truth value of (x1 =< x2) element-wise.
    :obj:`dpnp.greater` : Return the truth value of (x1 > x2) element-wise.
    :obj:`dpnp.less` : Return the truth value of (x1 < x2) element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = np.array([0, 1, 3])
    >>> x2 = np.arange(3)
    >>> np.equal(x1, x2)
    array([ True,  True, False])

    What is compared are values, not types. So an int (1) and an array of
    length one can evaluate as True:

    >>> np.equal(1, np.ones(1))
    array([ True])

    The ``==`` operator can be used as a shorthand for ``equal`` on
    :class:`dpnp.ndarray`.

    >>> a = np.array([2, 4, 6])
    >>> b = np.array([2, 4, 2])
    >>> a == b
    array([ True,  True, False])

    """

    return check_nd_call_func(
        numpy.equal,
        dpnp_equal,
        x1,
        x2,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def greater(
    x1,
    x2,
    /,
    out=None,
    *,
    order="K",
    where=True,
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Return the truth value of (x1 > x2) element-wise.

    For full documentation refer to :obj:`numpy.greater`.

    Returns
    -------
    out : dpnp.ndarray
        Output array of bool type, element-wise comparison of `x1` and `x2`.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.greater_equal` : Return the truth value of (x1 >= x2) element-wise.
    :obj:`dpnp.less` : Return the truth value of (x1 < x2) element-wise.
    :obj:`dpnp.less_equal` : Return the truth value of (x1 =< x2) element-wise.
    :obj:`dpnp.equal` : Return (x1 == x2) element-wise.
    :obj:`dpnp.not_equal` : Return (x1 != x2) element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = np.array([4, 2])
    >>> x2 = np.array([2, 2])
    >>> np.greater(x1, x2)
    array([ True, False])

    The ``>`` operator can be used as a shorthand for ``greater`` on
    :class:`dpnp.ndarray`.

    >>> a = np.array([4, 2])
    >>> b = np.array([2, 2])
    >>> a > b
    array([ True, False])

    """

    return check_nd_call_func(
        numpy.greater,
        dpnp_greater,
        x1,
        x2,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def greater_equal(
    x1,
    x2,
    /,
    out=None,
    *,
    order="K",
    where=True,
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Return the truth value of (x1 >= x2) element-wise.

    For full documentation refer to :obj:`numpy.greater_equal`.

    Returns
    -------
    out : dpnp.ndarray
        Output array of bool type, element-wise comparison of `x1` and `x2`.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.greater` : Return the truth value of (x1 > x2) element-wise.
    :obj:`dpnp.less` : Return the truth value of (x1 < x2) element-wise.
    :obj:`dpnp.less_equal` : Return the truth value of (x1 =< x2) element-wise.
    :obj:`dpnp.equal` : Return (x1 == x2) element-wise.
    :obj:`dpnp.not_equal` : Return (x1 != x2) element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = np.array([4, 2, 1])
    >>> x2 = np.array([2, 2, 2])
    >>> np.greater_equal(x1, x2)
    array([ True,  True, False])

    The ``>=`` operator can be used as a shorthand for ``greater_equal`` on
    :class:`dpnp.ndarray`.

    >>> a = np.array([4, 2, 1])
    >>> b = np.array([2, 2, 2])
    >>> a >= b
    array([ True,  True, False])

    """

    return check_nd_call_func(
        numpy.greater_equal,
        dpnp_greater_equal,
        x1,
        x2,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def isclose(x1, x2, rtol=1e-05, atol=1e-08, equal_nan=False):
    """
    Returns a boolean array where two arrays are element-wise equal within a tolerance.

    For full documentation refer to :obj:`numpy.isclose`.

    Limitations
    -----------
    `x2` is supported to be integer if `x1` is :class:`dpnp.ndarray` or
    at least either `x1` or `x2` should be as :class:`dpnp.ndarray`.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.allclose` : Returns True if two arrays are element-wise equal within a tolerance.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = np.array([1e10,1e-7])
    >>> x2 = np.array([1.00001e10,1e-8])
    >>> out = np.isclose(x1, x2)
    >>> [i for i in out]
    [True, False]

    """

    # x1_desc = dpnp.get_dpnp_descriptor(x1)
    # x2_desc = dpnp.get_dpnp_descriptor(x2)
    # if x1_desc and x2_desc:
    #     result_obj = dpnp_isclose(x1_desc, x2_desc, rtol, atol, equal_nan).get_pyobj()
    #     return result_obj

    return call_origin(
        numpy.isclose, x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan
    )


def isfinite(
    x, /, out=None, *, where=True, order="K", dtype=None, subok=True, **kwargs
):
    """
    Test element-wise for finiteness (not infinity or not Not a Number).

    For full documentation refer to :obj:`numpy.isfinite`.

    Limitations
    -----------
    Parameters `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.isinf` : Test element-wise for positive or negative infinity.
    :obj:`dpnp.isneginf` : Test element-wise for negative infinity,
                           return result as bool array.
    :obj:`dpnp.isposinf` : Test element-wise for positive infinity,
                           return result as bool array.
    :obj:`dpnp.isnan` : Test element-wise for NaN and
                        return result as a boolean array.

    Notes
    -----
    Not a Number, positive infinity and negative infinity are considered
    to be non-finite.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([-np.inf, 0., np.inf])
    >>> np.isfinite(x)
    array([False,  True, False])

    """

    return check_nd_call_func(
        numpy.isfinite,
        dpnp_isfinite,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def isinf(
    x, /, out=None, *, where=True, order="K", dtype=None, subok=True, **kwargs
):
    """
    Test element-wise for positive or negative infinity.

    For full documentation refer to :obj:`numpy.isinf`.

    Limitations
    -----------
    Parameters `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.isneginf` : Test element-wise for negative infinity,
                           return result as bool array.
    :obj:`dpnp.isposinf` : Test element-wise for positive infinity,
                           return result as bool array.
    :obj:`dpnp.isnan` : Test element-wise for NaN and
                        return result as a boolean array.
    :obj:`dpnp.isfinite` : Test element-wise for finiteness.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([-np.inf, 0., np.inf])
    >>> np.isinf(x)
    array([ True, False,  True])

    """

    return check_nd_call_func(
        numpy.isinf,
        dpnp_isinf,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def isnan(
    x, /, out=None, *, where=True, order="K", dtype=None, subok=True, **kwargs
):
    """
    Test element-wise for NaN and return result as a boolean array.

    For full documentation refer to :obj:`numpy.isnan`.

    Limitations
    -----------
    Parameters `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.isinf` : Test element-wise for positive or negative infinity.
    :obj:`dpnp.isneginf` : Test element-wise for negative infinity,
                           return result as bool array.
    :obj:`dpnp.isposinf` : Test element-wise for positive infinity,
                           return result as bool array.
    :obj:`dpnp.isfinite` : Test element-wise for finiteness.
    :obj:`dpnp.isnat` : Test element-wise for NaT (not a time)
                        and return result as a boolean array.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([np.inf, 0., np.nan])
    >>> np.isnan(x)
    array([False, False,  True])

    """

    return check_nd_call_func(
        numpy.isnan,
        dpnp_isnan,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def less(
    x1,
    x2,
    /,
    out=None,
    *,
    order="K",
    where=True,
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Return the truth value of (x1 < x2) element-wise.

    For full documentation refer to :obj:`numpy.less`.

    Returns
    -------
    out : dpnp.ndarray
        Output array of bool type, element-wise comparison of `x1` and `x2`.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.greater` : Return the truth value of (x1 > x2) element-wise.
    :obj:`dpnp.less_equal` : Return the truth value of (x1 =< x2) element-wise.
    :obj:`dpnp.greater_equal` : Return the truth value of (x1 >= x2) element-wise.
    :obj:`dpnp.equal` : Return (x1 == x2) element-wise.
    :obj:`dpnp.not_equal` : Return (x1 != x2) element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = np.array([1, 2])
    >>> x2 = np.array([2, 2])
    >>> np.less(x1, x2)
    array([ True, False])

    The ``<`` operator can be used as a shorthand for ``less`` on
    :class:`dpnp.ndarray`.

    >>> a = np.array([1, 2])
    >>> b = np.array([2, 2])
    >>> a < b
    array([ True, False])

    """

    return check_nd_call_func(
        numpy.less,
        dpnp_less,
        x1,
        x2,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def less_equal(
    x1,
    x2,
    /,
    out=None,
    *,
    order="K",
    where=True,
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Return the truth value of (x1 <= x2) element-wise.

    For full documentation refer to :obj:`numpy.less_equal`.

    Returns
    -------
    out : dpnp.ndarray
        Output array of bool type, element-wise comparison of `x1` and `x2`.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.greater` : Return the truth value of (x1 > x2) element-wise.
    :obj:`dpnp.less` : Return the truth value of (x1 < x2) element-wise.
    :obj:`dpnp.greater_equal` : Return the truth value of (x1 >= x2) element-wise.
    :obj:`dpnp.equal` : Return (x1 == x2) element-wise.
    :obj:`dpnp.not_equal` : Return (x1 != x2) element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = np.array([4, 2, 1])
    >>> x2 = np.array([2, 2, 2]
    >>> np.less_equal(x1, x2)
    array([False,  True,  True])

    The ``<=`` operator can be used as a shorthand for ``less_equal`` on
    :class:`dpnp.ndarray`.

    >>> a = np.array([4, 2, 1])
    >>> b = np.array([2, 2, 2])
    >>> a <= b
    array([False,  True,  True])

    """

    return check_nd_call_func(
        numpy.less_equal,
        dpnp_less_equal,
        x1,
        x2,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def logical_and(
    x1,
    x2,
    /,
    out=None,
    *,
    order="K",
    where=True,
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Compute the truth value of x1 AND x2 element-wise.

    For full documentation refer to :obj:`numpy.logical_and`.

    Returns
    -------
    out : dpnp.ndarray
        Boolean result of the logical AND operation applied to the elements
        of `x1` and `x2`; the shape is determined by broadcasting.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.logical_or` : Compute the truth value of x1 OR x2 element-wise.
    :obj:`dpnp.logical_not` : Compute the truth value of NOT x element-wise.
    :obj:`dpnp.logical_xor` : Compute the truth value of x1 XOR x2, element-wise.
    :obj:`dpnp.bitwise_and` : Compute the bit-wise AND of two arrays element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = np.array([True, False])
    >>> x2 = np.array([False, False])
    >>> np.logical_and(x1, x2)
    array([False, False])

    >>> x = np.arange(5)
    >>> np.logical_and(x > 1, x < 4)
    array([False, False,  True,  True, False])

    The ``&`` operator can be used as a shorthand for ``logical_and`` on
    boolean :class:`dpnp.ndarray`.

    >>> a = np.array([True, False])
    >>> b = np.array([False, False])
    >>> a & b
    array([False, False])

    """

    return check_nd_call_func(
        numpy.logical_and,
        dpnp_logical_and,
        x1,
        x2,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def logical_not(
    x,
    /,
    out=None,
    *,
    order="K",
    where=True,
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Compute the truth value of NOT x element-wise.

    For full documentation refer to :obj:`numpy.logical_not`.

    Returns
    -------
    out : dpnp.ndarray
        Boolean result with the same shape as `x` of the NOT operation
        on elements of `x`.

    Limitations
    -----------
    Parameters `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.logical_and` : Compute the truth value of x1 AND x2 element-wise.
    :obj:`dpnp.logical_or` : Compute the truth value of x1 OR x2 element-wise.
    :obj:`dpnp.logical_xor` : Compute the truth value of x1 XOR x2, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([True, False, 0, 1])
    >>> np.logical_not(x)
    array([False,  True,  True, False])

    >>> x = np.arange(5)
    >>> np.logical_not(x < 3)
    array([False, False, False,  True,  True])

    """

    return check_nd_call_func(
        numpy.logical_not,
        dpnp_logical_not,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def logical_or(
    x1,
    x2,
    /,
    out=None,
    *,
    order="K",
    where=True,
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Compute the truth value of x1 OR x2 element-wise.

    For full documentation refer to :obj:`numpy.logical_or`.

    Returns
    -------
    out : dpnp.ndarray
        Boolean result of the logical OR operation applied to the elements
        of `x1` and `x2`; the shape is determined by broadcasting.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.logical_and` : Compute the truth value of x1 AND x2 element-wise.
    :obj:`dpnp.logical_not` : Compute the truth value of NOT x element-wise.
    :obj:`dpnp.logical_xor` : Compute the truth value of x1 XOR x2, element-wise.
    :obj:`dpnp.bitwise_or` : Compute the bit-wise OR of two arrays element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = np.array([True, False])
    >>> x2 = np.array([False, False])
    >>> np.logical_or(x1, x2)
    array([ True, False])

    >>> x = np.arange(5)
    >>> np.logical_or(x < 1, x > 3)
    array([ True, False, False, False,  True])

    The ``|`` operator can be used as a shorthand for ``logical_or`` on
    boolean :class:`dpnp.ndarray`.

    >>> a = np.array([True, False])
    >>> b = np.array([False, False])
    >>> a | b
    array([ True, False])

    """

    return check_nd_call_func(
        numpy.logical_or,
        dpnp_logical_or,
        x1,
        x2,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def logical_xor(
    x1,
    x2,
    /,
    out=None,
    *,
    order="K",
    where=True,
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Compute the truth value of x1 XOR x2 element-wise.

    For full documentation refer to :obj:`numpy.logical_xor`.

    Returns
    -------
    out : dpnp.ndarray
        Boolean result of the logical XOR operation applied to the elements
        of `x1` and `x2`; the shape is determined by broadcasting.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.logical_and` : Compute the truth value of x1 AND x2 element-wise.
    :obj:`dpnp.logical_or` : Compute the truth value of x1 OR x2 element-wise.
    :obj:`dpnp.logical_not` : Compute the truth value of NOT x element-wise.
    :obj:`dpnp.bitwise_xor` : Compute the bit-wise XOR of two arrays element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = np.array([True, True, False, False])
    >>> x2 = np.array([True, False, True, False])
    >>> np.logical_xor(x1, x2)
    array([False,  True,  True, False])

    >>> x = np.arange(5)
    >>> np.logical_xor(x < 1, x > 3)
    array([ True, False, False, False,  True])

    Simple example showing support of broadcasting

    >>> np.logical_xor(0, np.eye(2))
    array([[ True, False],
           [False,  True]])

    """

    return check_nd_call_func(
        numpy.logical_xor,
        dpnp_logical_xor,
        x1,
        x2,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def not_equal(
    x1,
    x2,
    /,
    out=None,
    *,
    order="K",
    where=True,
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Return the truth value of (x1 != x2) element-wise.

    For full documentation refer to :obj:`numpy.not_equal`.

    Returns
    -------
    out : dpnp.ndarray
        Output array of bool type, element-wise comparison of `x1` and `x2`.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.equal` : Return (x1 == x2) element-wise.
    :obj:`dpnp.greater` : Return the truth value of (x1 > x2) element-wise.
    :obj:`dpnp.greater_equal` : Return the truth value of (x1 >= x2) element-wise.
    :obj:`dpnp.less` : Return the truth value of (x1 < x2) element-wise.
    :obj:`dpnp.less_equal` : Return the truth value of (x1 =< x2) element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = np.array([1., 2.])
    >>> x2 = np.arange(1., 3.)
    >>> np.not_equal(x1, x2)
    array([False, False])

    The ``!=`` operator can be used as a shorthand for ``not_equal`` on
    :class:`dpnp.ndarray`.

    >>> a = np.array([1., 2.])
    >>> b = np.array([1., 3.])
    >>> a != b
    array([False,  True])

    """

    return check_nd_call_func(
        numpy.not_equal,
        dpnp_not_equal,
        x1,
        x2,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )
