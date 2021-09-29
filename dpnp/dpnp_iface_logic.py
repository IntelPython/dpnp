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

import dpnp.config as config
from dpnp.dpnp_algo import *
from dpnp.dpnp_utils import *


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
    "not_equal"
]


def all(x1, axis=None, out=None, keepdims=False):
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

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if axis is not None:
            pass
        elif out is not None:
            pass
        elif keepdims is not False:
            pass
        else:
            result_obj = dpnp_all(x1_desc).get_pyobj()
            result = dpnp.convert_single_elem_array_to_scalar(result_obj)

            return result

    return call_origin(numpy.all, x1, axis, out, keepdims)


def allclose(x1, x2, rtol=1.e-5, atol=1.e-8, **kwargs):
    """
    Returns True if two arrays are element-wise equal within a tolerance.

    For full documentation refer to :obj:`numpy.allclose`.

    Limitations
    -----------
    Parameters ``x1`` and ``x2`` are supported as either :obj:`dpnp.ndarray` or scalar.
    Keyword arguments ``kwargs`` are currently unsupported.
    Otherwise the functions will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> np.allclose([1e10,1e-7], [1.00001e10,1e-8])
    >>> False

    """

    rtol_is_scalar = dpnp.isscalar(rtol)
    atol_is_scalar = dpnp.isscalar(atol)
    x1_desc = dpnp.get_dpnp_descriptor(x1)
    x2_desc = dpnp.get_dpnp_descriptor(x2)

    if x1_desc and x2_desc and not kwargs:
        if not rtol_is_scalar or not atol_is_scalar:
            pass
        else:
            result_obj = dpnp_allclose(x1_desc, x2_desc, rtol, atol).get_pyobj()
            result = dpnp.convert_single_elem_array_to_scalar(result_obj)

            return result

    return call_origin(numpy.allclose, x1, x2, rtol=rtol, atol=atol, **kwargs)


def any(x1, axis=None, out=None, keepdims=False):
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

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if axis is not None:
            pass
        elif out is not None:
            pass
        elif keepdims is not False:
            pass
        else:
            result_obj = dpnp_any(x1_desc).get_pyobj()
            result = dpnp.convert_single_elem_array_to_scalar(result_obj)

            return result

    return call_origin(numpy.any, x1, axis, out, keepdims)


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

    # x1_desc = dpnp.get_dpnp_descriptor(x1)
    # x2_desc = dpnp.get_dpnp_descriptor(x2)
    # if x1_desc and x2_desc:
    #     if x1_desc.size != x2_desc.size:
    #         pass
    #     elif x1_desc.dtype != x2_desc.dtype:
    #         pass
    #     elif x1_desc.shape != x2_desc.shape:
    #         pass
    #     else:
    #         return dpnp_equal(x1_desc, x2_desc).get_pyobj()

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

    # x1_desc = dpnp.get_dpnp_descriptor(x1)
    # x2_desc = dpnp.get_dpnp_descriptor(x2)
    # if x1_desc and x2_desc:
    #     if x1_desc.size < 2:
    #         pass
    #     elif x2_desc.size < 2:
    #         pass
    #     else:
    #         return dpnp_greater(x1_desc, x2_desc).get_pyobj()

    return call_origin(numpy.greater, x1, x2)


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

    # x1_desc = dpnp.get_dpnp_descriptor(x1)
    # x2_desc = dpnp.get_dpnp_descriptor(x2)
    # if x1_desc and x2_desc:
    #     if x1_desc.size < 2:
    #         pass
    #     elif x2_desc.size < 2:
    #         pass
    #     else:
    #         return dpnp_greater_equal(x1_desc, x2_desc).get_pyobj()

    return call_origin(numpy.greater_equal, x1, x2)


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

    # x1_desc = dpnp.get_dpnp_descriptor(x1)
    # x2_desc = dpnp.get_dpnp_descriptor(x2)
    # if x1_desc and x2_desc:
    #     result_obj = dpnp_isclose(x1_desc, x2_desc, rtol, atol, equal_nan).get_pyobj()
    #     return result_obj

    return call_origin(numpy.isclose, x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)


def isfinite(x1, out=None, **kwargs):
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

    # x1_desc = dpnp.get_dpnp_descriptor(x1)
    # if x1_desc and kwargs:
    #     if out is not None:
    #         pass
    #     else:
    #         return dpnp_isfinite(x1_desc).get_pyobj()

    return call_origin(numpy.isfinite, x1, out, **kwargs)


def isinf(x1, out=None, **kwargs):
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

    # x1_desc = dpnp.get_dpnp_descriptor(x1)
    # if x1_desc and kwargs:
    #     if out is not None:
    #         pass
    #     else:
    #         return dpnp_isinf(x1_desc).get_pyobj()

    return call_origin(numpy.isinf, x1, out, **kwargs)


def isnan(x1, out=None, **kwargs):
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

    # x1_desc = dpnp.get_dpnp_descriptor(x1)
    # if x1_desc and kwargs:
    #     if out is not None:
    #         pass
    #     else:
    #         return dpnp_isnan(x1_desc).get_pyobj()

    return call_origin(numpy.isnan, x1, out, **kwargs)


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

    # x1_desc = dpnp.get_dpnp_descriptor(x1)
    # x2_desc = dpnp.get_dpnp_descriptor(x2)
    # if x1_desc and x2_desc:
    #     if x1_desc.size < 2:
    #         pass
    #     elif x2_desc.size < 2:
    #         pass
    #     else:
    #         return dpnp_less(x1_desc, x2_desc).get_pyobj()

    return call_origin(numpy.less, x1, x2)


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

    # x1_desc = dpnp.get_dpnp_descriptor(x1)
    # x2_desc = dpnp.get_dpnp_descriptor(x2)
    # if x1_desc and x2_desc:
    #     if x1_desc.size < 2:
    #         pass
    #     elif x2_desc.size < 2:
    #         pass
    #     else:
    #         return dpnp_less_equal(x1_desc, x2_desc).get_pyobj()

    return call_origin(numpy.less_equal, x1, x2)


def logical_and(x1, x2, out=None, **kwargs):
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

    # x1_desc = dpnp.get_dpnp_descriptor(x1)
    # x2_desc = dpnp.get_dpnp_descriptor(x2)
    # if x1_desc and x2_desc and not kwargs:
    #     if out is not None:
    #         pass
    #     else:
    #         return dpnp_logical_and(x1_desc, x2_desc).get_pyobj()

    return call_origin(numpy.logical_and, x1, x2, out, **kwargs)


def logical_not(x1, out=None, **kwargs):
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

    # x1_desc = dpnp.get_dpnp_descriptor(x1)
    # if x1_desc and not kwargs:
    #     if out is not None:
    #         pass
    #     else:
    #         return dpnp_logical_not(x1_desc).get_pyobj()

    return call_origin(numpy.logical_not, x1, out, **kwargs)


def logical_or(x1, x2, out=None, **kwargs):
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

    # x1_desc = dpnp.get_dpnp_descriptor(x1)
    # x2_desc = dpnp.get_dpnp_descriptor(x2)
    # if x1_desc and x2_desc and not kwargs:
    #     if out is not None:
    #         pass
    #     else:
    #         return dpnp_logical_or(x1_desc, x2_desc).get_pyobj()

    return call_origin(numpy.logical_or, x1, x2, out, **kwargs)


def logical_xor(x1, x2, out=None, **kwargs):
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

    # x1_desc = dpnp.get_dpnp_descriptor(x1)
    # x2_desc = dpnp.get_dpnp_descriptor(x2)
    # if x1_desc and x2_desc and not kwargs:
    #     if out is not None:
    #         pass
    #     else:
    #         return dpnp_logical_xor(x1_desc, x2_desc).get_pyobj()

    return call_origin(numpy.logical_xor, x1, x2, out, **kwargs)


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

    # x1_desc = dpnp.get_dpnp_descriptor(x1)
    # x2_desc = dpnp.get_dpnp_descriptor(x2)
    # if x1_desc and x2_desc:
    #     if x1_desc.size < 2:
    #         pass
    #     elif x2_desc.size < 2:
    #         pass
    #     else:
    #         result = dpnp_not_equal(x1_desc, x2_desc).get_pyobj()

    #         return result

    return call_origin(numpy.not_equal, x1, x2)
