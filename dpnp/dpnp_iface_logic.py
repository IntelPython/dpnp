# cython: language_level=3
# distutils: language = c++
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2020, Intel Corporation
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


import numpy
import dpnp

from dpnp.dpnp_algo import *
from dpnp.dparray import dparray
from dpnp.dpnp_utils import *


__all__ = [
    "all",
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
    "not_equal"
]


def all(in_array1, axis=None, out=None, keepdims=False):
    """
    Test whether all array elements along a given axis evaluate to True.

    For full documentation refer to :obj:`numpy.all`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.
    Parameter ``axis`` is supported only with default value ``None``.
    Parameter ``out`` is supported only with default value ``None``.
    Parameter ``keepdims`` is supported only with default value ``False``.

    See Also
    --------
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
    False
    >>> x2 = np.array([-1, 4, 5])
    >>> np.all(x2)
    True
    >>> x3 = np.array([1.0, np.nan])
    >>> np.all(x3)
    True

    """

    if not use_origin_backend(in_array1):
        if not isinstance(in_array1, dparray):
            pass
        elif axis is not None:
            pass
        elif out is not None:
            pass
        elif keepdims is not False:
            pass
        else:
            result = dpnp_all(in_array1)
            return result[0]

    return call_origin(numpy.all, axis, out, keepdims)


def any(in_array1, axis=None, out=None, keepdims=False):
    """
    Test whether any array element along a given axis evaluates to True.

    For full documentation refer to :obj:`numpy.any`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.
    Parameter ``axis`` is supported only with default value ``None``.
    Parameter ``out`` is supported only with default value ``None``.
    Parameter ``keepdims`` is supported only with default value ``False``.

    See Also
    --------
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
    True
    >>> x2 = np.array([0, 0, 0])
    >>> np.any(x2)
    False
    >>> x3 = np.array([1.0, np.nan])
    >>> np.any(x3)
    True

    """

    if (not use_origin_backend(in_array1)):
        if not isinstance(in_array1, dparray):
            pass
        elif axis is not None:
            pass
        elif out is not None:
            pass
        elif keepdims is not False:
            pass
        else:
            result = dpnp_any(in_array1)
            return result[0]

    return call_origin(numpy.any, axis, out, keepdims)


def equal(x1, x2):
    """
    Return (x1 == x2) element-wise.

    For full documentation refer to :obj:`numpy.equal`.

    Limitations
    -----------
    Parameter ``x1`` is supported as :obj:`dpnp.ndarray`.
    Parameter ``x2`` is supported as either :obj:`dpnp.ndarray` or int.
    Input array data types are limited by supported DPNP :ref:`Data types`.
    Sizes, shapes and data types of input arrays ``x1`` and ``x2`` are supported to be equal.

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
    >>> out = np.equal(x1, x2)
    >>> [i for i in out]
    [True, True, False]

    """
    if not use_origin_backend(x1):
        if not isinstance(x1, dparray):
            pass
        elif isinstance(x2, int):
            return dpnp_equal(x1, x2)
        elif not isinstance(x2, dparray):
            pass
        elif x1.size != x2.size:
            pass
        elif x1.dtype != x2.dtype:
            pass
        elif x1.shape != x2.shape:
            pass
        else:
            return dpnp_equal(x1, x2)

    return call_origin(numpy.equal, x1, x2)


def greater(x1, x2):
    """
    Return (x1 > x2) element-wise.

    For full documentation refer to :obj:`numpy.greater`.

    Limitations
    -----------
    At least either ``x1`` or ``x2`` should be as :obj:`dpnp.ndarray`.
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
    >>> out = np.greater(x1, x2)
    >>> [i for i in out]
    [True, False]

    """

    if not (use_origin_backend(x1)):
        if not isinstance(x1, dparray):
            pass
        else:
            return dpnp_greater(x1, x2)

    return numpy.greater(x1, x2)


def greater_equal(x1, x2):
    """
    Return (x1 >= x2) element-wise.

    For full documentation refer to :obj:`numpy.greater_equal`.

    Limitations
    -----------
    At least either ``x1`` or ``x2`` should be as :obj:`dpnp.ndarray`.
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
    >>> out = np.greater_equal(x1, x2)
    >>> [i for i in out]
    [True, True, False]

    """

    if not (use_origin_backend(x1)):
        if not isinstance(x1, dparray):
            pass
        else:
            return dpnp_greater_equal(x1, x2)

    return numpy.greater_equal(x1, x2)


def isclose(x1, x2, rtol=1e-05, atol=1e-08, equal_nan=False):
    """
    Returns a boolean array where two arrays are element-wise equal within a tolerance.

    For full documentation refer to :obj:`numpy.isclose`.

    Limitations
    -----------
    ``x2`` is supported to be integer if ``x1`` is :obj:`dpnp.ndarray` or
    at least either ``x1`` or ``x2`` should be as :obj:`dpnp.ndarray`.
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

    if (use_origin_backend(x1)):
        return numpy.greater_equal(x1, x2)

    if isinstance(x1, dparray) and isinstance(x2, int):  # hack to satisfy current test system requirements
        return dpnp_isclose(x1, x2, rtol, atol, equal_nan)

    if isinstance(x1, dparray) or isinstance(x2, dparray):
        return dpnp_isclose(x1, x2, rtol, atol, equal_nan)

    return numpy.isclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)


def isfinite(in_array1, out=None, where=True, **kwargs):
    """
    Test element-wise for finiteness (not infinity or not Not a Number).

    For full documentation refer to :obj:`numpy.isfinite`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.
    Parameter ``out`` is supported only with default value ``None``.
    Parameter ``where`` is supported only with default value ``True``.

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
    >>> import numpy
    >>> import dpnp as np
    >>> x = np.array([-numpy.inf, 0., numpy.inf])
    >>> out = np.isfinite(x)
    >>> [i for i in out]
    [False, True, False]

    """

    is_dparray1 = isinstance(in_array1, dparray)

    if (not use_origin_backend(in_array1) and is_dparray1):
        if out is not None:
            checker_throw_value_error("isfinite", "out", type(out), None)
        if where is not True:
            checker_throw_value_error("isfinite", "where", where, True)

        return dpnp_isfinite(in_array1)

    input1 = dpnp.asnumpy(in_array1) if is_dparray1 else in_array1

    return numpy.isfinite(input1, out, where, **kwargs)


def isinf(in_array1, out=None, where=True, **kwargs):
    """
    Test element-wise for positive or negative infinity.

    For full documentation refer to :obj:`numpy.isinf`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.
    Parameter ``out`` is supported only with default value ``None``.
    Parameter ``where`` is supported only with default value ``True``.

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
    >>> import numpy
    >>> import dpnp as np
    >>> x = np.array([-numpy.inf, 0., numpy.inf])
    >>> out = np.isinf(x)
    >>> [i for i in out]
    [True, False, True]

    """

    is_dparray1 = isinstance(in_array1, dparray)

    if (not use_origin_backend(in_array1) and is_dparray1):
        if out is not None:
            checker_throw_value_error("isinf", "out", type(out), None)
        if where is not True:
            checker_throw_value_error("isinf", "where", where, True)

        return dpnp_isinf(in_array1)

    input1 = dpnp.asnumpy(in_array1) if is_dparray1 else in_array1

    return numpy.isinf(input1, out, where, **kwargs)


def isnan(in_array1, out=None, where=True, **kwargs):
    """
    Test element-wise for NaN and return result as a boolean array.

    For full documentation refer to :obj:`numpy.isnan`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.
    Parameter ``out`` is supported only with default value ``None``.
    Parameter ``where`` is supported only with default value ``True``.

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
    >>> import numpy
    >>> import dpnp as np
    >>> x = np.array([numpy.inf, 0., np.nan])
    >>> out = np.isnan(x)
    >>> [i for i in out]
    [False, False, True]

    """

    is_dparray1 = isinstance(in_array1, dparray)

    if (not use_origin_backend(in_array1) and is_dparray1):
        if out is not None:
            checker_throw_value_error("isnan", "out", type(out), None)
        if where is not True:
            checker_throw_value_error("isnan", "where", where, True)

        return dpnp_isnan(in_array1)

    input1 = dpnp.asnumpy(in_array1) if is_dparray1 else in_array1

    return numpy.isnan(input1, out, where=where, **kwargs)


def less(x1, x2):
    """
    Return (x1 < x2) element-wise.

    For full documentation refer to :obj:`numpy.less`.

    Limitations
    -----------
    At least either ``x1`` or ``x2`` should be as :obj:`dpnp.ndarray`.
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
    >>> out = np.less(x1, x2)
    >>> [i for i in out]
    [True, False]

    """

    if not (use_origin_backend(x1)):
        if not isinstance(x1, dparray):
            pass
        else:
            return dpnp_less(x1, x2)

    return numpy.less(x1, x2)


def less_equal(x1, x2):
    """
    Return (x1 <= x2) element-wise.

    For full documentation refer to :obj:`numpy.less_equal`.

    Limitations
    -----------
    At least either ``x1`` or ``x2`` should be as :obj:`dpnp.ndarray`.
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
    >>> x2 = np.array([2, 2, 2])
    >>> out = np.less_equal(x1, x2)
    >>> [i for i in out]
    [False, True, True]

    """

    if not (use_origin_backend(x1)):
        if not isinstance(x1, dparray):
            pass
        else:
            return dpnp_less_equal(x1, x2)

    return numpy.less_equal(x1, x2)


def logical_and(x1, x2, out=None, where=True, **kwargs):
    """
    Compute the truth value of x1 AND x2 element-wise.

    For full documentation refer to :obj:`numpy.logical_and`.

    Limitations
    -----------
    Input arrays are supported as :obj:`dpnp.ndarray`.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.
    Parameter ``out`` is supported only with default value ``None``.
    Parameter ``where`` is supported only with default value ``True``.

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
    >>> out = np.logical_and(x1, x2)
    >>> [i for i in out]
    [False, False]

    """

    is_dparray1 = isinstance(x1, dparray)
    is_dparray2 = isinstance(x2, dparray)

    if (not use_origin_backend(x1) and is_dparray1 and is_dparray2):
        if out is not None:
            checker_throw_value_error("logical_and", "out", type(out), None)
        if where is not True:
            checker_throw_value_error("logical_and", "where", where, True)

        return dpnp_logical_and(x1, x2)

    input1 = dpnp.asnumpy(x1) if is_dparray1 else x1
    input2 = dpnp.asnumpy(x2) if is_dparray1 else x2

    return numpy.logical_and(input1, input2, out, where, **kwargs)


def logical_not(x1, out=None, where=True, **kwargs):
    """
    Compute the truth value of NOT x element-wise.

    For full documentation refer to :obj:`numpy.logical_not`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.
    Parameter ``out`` is supported only with default value ``None``.
    Parameter ``where`` is supported only with default value ``True``.

    See Also
    --------
    :obj:`dpnp.logical_and` : Compute the truth value of x1 AND x2 element-wise.
    :obj:`dpnp.logical_or` : Compute the truth value of x1 OR x2 element-wise.
    :obj:`dpnp.logical_xor` : Compute the truth value of x1 XOR x2, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([True, False, 0, 1])
    >>> out = np.logical_not(x)
    >>> [i for i in out]
    [False, True, True, False]

    """

    is_dparray1 = isinstance(x1, dparray)

    if (not use_origin_backend(x1) and is_dparray1):
        if out is not None:
            checker_throw_value_error("logical_not", "out", type(out), None)
        if where is not True:
            checker_throw_value_error("logical_not", "where", where, True)

        return dpnp_logical_not(x1)

    input1 = dpnp.asnumpy(x1) if is_dparray1 else x1

    return numpy.logical_not(input1, out, where, **kwargs)


def logical_or(x1, x2, out=None, where=True, **kwargs):
    """
    Compute the truth value of x1 OR x2 element-wise.

    For full documentation refer to :obj:`numpy.logical_or`.

    Limitations
    -----------
    Input arrays are supported as :obj:`dpnp.ndarray`.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.
    Parameter ``out`` is supported only with default value ``None``.
    Parameter ``where`` is supported only with default value ``True``.

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
    >>> out = np.logical_or(x1, x2)
    >>> [i for i in out]
    [True, False]

    """

    is_dparray1 = isinstance(x1, dparray)
    is_dparray2 = isinstance(x2, dparray)

    if (not use_origin_backend(x1) and is_dparray1 and is_dparray2):
        if out is not None:
            checker_throw_value_error("logical_or", "out", type(out), None)
        if where is not True:
            checker_throw_value_error("logical_or", "where", where, True)

        return dpnp_logical_or(x1, x2)

    input1 = dpnp.asnumpy(x1) if is_dparray1 else x1
    input2 = dpnp.asnumpy(x2) if is_dparray1 else x2

    return numpy.logical_or(input1, input2, out, where, **kwargs)


def logical_xor(x1, x2, out=None, where=True, **kwargs):
    """
    Compute the truth value of x1 XOR x2, element-wise.

    For full documentation refer to :obj:`numpy.logical_xor`.

    Limitations
    -----------
    Input arrays are supported as :obj:`dpnp.ndarray`.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.
    Parameter ``out`` is supported only with default value ``None``.
    Parameter ``where`` is supported only with default value ``True``.

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
    >>> out = np.logical_xor(x1, x2)
    >>> [i for i in out]
    [False, True, True, False]

    """

    is_dparray1 = isinstance(x1, dparray)
    is_dparray2 = isinstance(x2, dparray)

    if (not use_origin_backend(x1) and is_dparray1 and is_dparray2):
        if out is not None:
            checker_throw_value_error("logical_xor", "out", type(out), None)
        if where is not True:
            checker_throw_value_error("logical_xor", "where", where, True)

        return dpnp_logical_xor(x1, x2)

    input1 = dpnp.asnumpy(x1) if is_dparray1 else x1
    input2 = dpnp.asnumpy(x2) if is_dparray1 else x2

    return numpy.logical_xor(input1, input2, out, where, **kwargs)


def not_equal(x1, x2):
    """
    Return (x1 != x2) element-wise.

    For full documentation refer to :obj:`numpy.not_equal`.

    Limitations
    -----------
    At least either ``x1`` or ``x2`` should be as :obj:`dpnp.ndarray`.
    If either ``x1`` or ``x2`` is scalar then other one should be :obj:`dpnp.ndarray`.
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
    >>> out = np.not_equal(x1, x2)
    >>> [i for i in out]
    [False, False]

    """

    is_x1_dparray = isinstance(x1, dparray)
    is_x2_dparray = isinstance(x2, dparray)

    is_x1_scalar = numpy.isscalar(x1)
    is_x2_scalar = numpy.isscalar(x2)

    if (not use_origin_backend(x1) and (is_x1_dparray or is_x1_scalar)) and \
            (not use_origin_backend(x2) and (is_x2_dparray or is_x2_scalar)) and \
            not(is_x1_scalar and is_x2_scalar):

        if is_x1_scalar:
            result = dpnp_not_equal(x2, x1)
        else:
            result = dpnp_not_equal(x1, x2)

        return result

    return call_origin(numpy.not_equal, x1, x2)
