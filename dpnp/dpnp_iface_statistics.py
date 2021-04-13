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
Interface of the statistics function of the DPNP

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""


import numpy

from dpnp.dpnp_algo import *
from dpnp.dparray import dparray
from dpnp.dpnp_utils import *
import dpnp


__all__ = [
    'amax',
    'amin',
    'average',
    'correlate',
    'cov',
    'max',
    'mean',
    'median',
    'min',
    'std',
    'var',
]


def amax(input, axis=None, out=None):
    """
    Return the maximum of an array or maximum along an axis.

    For full documentation refer to :obj:`numpy.amax`.

    See Also
    --------
    :obj:`dpnp.amin` : The minimum value of an array along a given axis,
                       propagating any NaNs.
    :obj:`dpnp.nanmax` : The maximum value of an array along a given axis,
                         ignoring any NaNs.
    :obj:`dpnp.maximum` : Element-wise maximum of two arrays,
                          propagating any NaNs.
    :obj:`dpnp.fmax` : Element-wise maximum of two arrays, ignoring any NaNs.
    :obj:`dpnp.argmax` : Return the indices of the maximum values.
    :obj:`dpnp.nanmin` : Return minimum of an array or minimum along an axis,
                         ignoring any NaNs.
    :obj:`dpnp.minimum` : Element-wise minimum of array elements.
    :obj:`dpnp.fmin` : Element-wise minimum of array elements.

    Notes
    -----
    This function works exactly the same as :obj:`dpnp.max`.

    """
    return max(input, axis=axis, out=out)


def amin(input, axis=None, out=None):
    """
    Return the minimum of an array or minimum along an axis.

    For full documentation refer to :obj:`numpy.amin`.

    See Also
    --------
    :obj:`dpnp.amax` : The maximum value of an array along a given axis,
                       propagating any NaNs.
    :obj:`dpnp.nanmin` : Return minimum of an array or minimum along an axis,
                         ignoring any NaNs.
    :obj:`dpnp.minimum` : Element-wise minimum of array elements.
    :obj:`dpnp.fmin` : Element-wise minimum of array elements.
    :obj:`dpnp.argmin` : Return the indices of the minimum values.
    :obj:`dpnp.nanmax` : The maximum value of an array along a given axis,
                         ignoring any NaNs.
    :obj:`dpnp.maximum` : Element-wise maximum of two arrays,
                          propagating any NaNs.
    :obj:`dpnp.fmax` : Element-wise maximum of two arrays, ignoring any NaNs.

    Notes
    -----
    This function works exactly the same as :obj:`dpnp.min`.

    """

    return min(input, axis=axis, out=out)


def average(a, axis=None, weights=None, returned=False):
    """
    Compute the weighted average along the specified axis.

    For full documentation refer to :obj:`numpy.average`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Prameters ``axis`` is supported only with default value ``None``.
    Prameters ``weights`` is supported only with default value ``None``.
    Prameters ``returned`` is supported only with default value ``False``.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.mean` : Compute the arithmetic mean along the specified axis.

    Examples
    --------
    >>> import dpnp as np
    >>> data = np.arange(1, 5)
    >>> [i for i in data]
    [1, 2, 3, 4]
    >>> np.average(data)
    2.5

    """
    if not use_origin_backend(a):
        if not isinstance(a, dparray):
            pass
        elif axis is not None:
            pass
        elif weights is not None:
            pass
        elif returned:
            pass
        else:
            array_avg = dpnp_average(a)

            # scalar returned
            if array_avg.shape == (1,):
                return array_avg.dtype.type(array_avg[0])

            return array_avg

    return call_origin(numpy.average, a, axis, weights, returned)


def correlate(a, v, mode='valid'):
    """
    Cross-correlation of two 1-dimensional sequences.

    For full documentation refer to :obj:`numpy.correlate`.

    Limitations
    -----------
    Input arrays are supported as :obj:`dpnp.ndarray`.
    Size and shape of input arrays are supported to be equal.
    Prameters ``mode`` is supported only with default value ``"valid``.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.convolve` : Discrete, linear convolution of
                           two one-dimensional sequences.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.correlate([1, 2, 3], [0, 1, 0.5])
    >>> [i for i in x]
    [3.5]

    """
    if not use_origin_backend(a):
        if not isinstance(a, dparray):
            pass
        elif not isinstance(v, dparray):
            pass
        elif a.size != v.size or a.size == 0:
            pass
        elif a.shape != v.shape:
            pass
        elif mode != 'valid':
            pass
        else:
            return dpnp_correlate(a, v)

    return call_origin(numpy.correlate, a, v, mode=mode)


def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None):
    """
    Estimate a covariance matrix, given data and weights.

    For full documentation refer to :obj:`numpy.cov`.

    Limitations
    -----------
    Input array ``m`` is supported as :obj:`dpnp.ndarray`.
    Dimension of input array ``m`` is limited by ``m.ndim > 2``.
    Size and shape of input arrays are supported to be equal.
    Prameters ``y`` is supported only with default value ``None``.
    Prameters ``rowvar`` is supported only with default value ``True``.
    Prameters ``bias`` is supported only with default value ``False``.
    Prameters ``ddof`` is supported only with default value ``None``.
    Prameters ``fweights`` is supported only with default value ``None``.
    Prameters ``aweights`` is supported only with default value ``None``.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    .. seealso:: :obj:`dpnp.corrcoef` normalized covariance matrix.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([[0, 2], [1, 1], [2, 0]]).T
    >>> x.shape
    (2, 3)
    >>> [i for i in x]
    [0, 1, 2, 2, 1, 0]
    >>> out = np.cov(x)
    >>> out.shape
    (2, 2)
    >>> [i for i in out]
    [1.0, -1.0, -1.0, 1.0]

    """
    if not use_origin_backend(m):
        if not isinstance(m, dparray):
            pass
        elif m.ndim > 2:
            pass
        elif y is not None:
            pass
        elif not rowvar:
            pass
        elif bias:
            pass
        elif ddof is not None:
            pass
        elif fweights is not None:
            pass
        elif aweights is not None:
            pass
        else:
            return dpnp_cov(m)

    return call_origin(numpy.cov, m, y, rowvar, bias, ddof, fweights, aweights)


def max(input, axis=None, out=None, keepdims=numpy._NoValue, initial=numpy._NoValue, where=numpy._NoValue):
    """
    Return the maximum of an array or maximum along an axis.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Otherwise the function will be executed sequentially on CPU.
    Prameters ``out`` is supported only with default value ``None``.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.arange(4).reshape((2,2))
    >>> a.shape
    (2, 2)
    >>> [i for i in a]
    [0, 1, 2, 3]
    >>> np.max(a)
    3

    """

    if not use_origin_backend(input):
        if not isinstance(input, dparray):
            pass
        elif out is not None:
            pass
        elif keepdims is not numpy._NoValue:
            pass
        elif initial is not numpy._NoValue:
            pass
        elif where is not numpy._NoValue:
            pass
        else:
            result = dpnp_max(input, axis=axis)

            # scalar returned
            if result.shape == (1,):
                return result.dtype.type(result[0])

            return result

    return call_origin(numpy.max, input, axis, out, keepdims, initial, where)


def mean(a, axis=None, **kwargs):
    """
    Compute the arithmetic mean along the specified axis.

    For full documentation refer to :obj:`numpy.mean`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Keyword arguments ``kwargs`` are currently unsupported.
    Size of input array is limited by ``a.size > 0``.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.average` : Weighted average.
    :obj:`dpnp.std` : Compute the standard deviation along the specified axis.
    :obj:`dpnp.var` : Compute the variance along the specified axis.
    :obj:`dpnp.nanmean` : Compute the arithmetic mean along the specified axis,
                          ignoring NaNs.
    :obj:`dpnp.nanstd` : Compute the standard deviation along
                         the specified axis, while ignoring NaNs.
    :obj:`dpnp.nanvar` : Compute the variance along the specified axis,
                         while ignoring NaNs.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.mean(a)
    2.5

    """
    if not use_origin_backend(a) and not kwargs:
        if not isinstance(a, dparray):
            pass
        elif a.size == 0:
            pass
        else:
            result = dpnp_mean(a, axis=axis)

            # scalar returned
            if result.shape == (1,):
                return result.dtype.type(result[0])

            return result

    return call_origin(numpy.mean, a, axis=axis, **kwargs)


def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    """
    Compute the median along the specified axis.

    For full documentation refer to :obj:`numpy.median`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Prameters ``axis`` is supported only with default value ``None``.
    Prameters ``out`` is supported only with default value ``None``.
    Prameters ``overwrite_input`` is supported only with default value ``False``.
    Prameters ``keepdims`` is supported only with default value ``False``.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.mean` : Compute the arithmetic mean along the specified axis.
    :obj:`dpnp.percentile` : Compute the q-th percentile of the data
                             along the specified axis.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[10, 7, 4], [3, 2, 1]])
    >>> np.median(a)
    3.5

    """
    if not use_origin_backend(a):
        if not isinstance(a, dparray):
            pass
        elif axis is not None:
            pass
        elif out is not None:
            pass
        elif overwrite_input:
            pass
        elif keepdims:
            pass
        else:
            result = dpnp_median(a)

            # scalar returned
            if result.shape == (1,):
                return result.dtype.type(result[0])

            return result

    return call_origin(numpy.median, a, axis, out, overwrite_input, keepdims)


def min(input, axis=None, out=None, keepdims=numpy._NoValue, initial=numpy._NoValue, where=numpy._NoValue):
    """
    Return the minimum along a given axis.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Otherwise the function will be executed sequentially on CPU.
    Prameters ``out`` is supported only with default value ``None``.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.arange(4).reshape((2,2))
    >>> a.shape
    (2, 2)
    >>> [i for i in a]
    [0, 1, 2, 3]
    >>> np.min(a)
    0

    """

    if not use_origin_backend(input):
        if not isinstance(input, dparray):
            pass
        elif out is not None:
            pass
        elif keepdims is not numpy._NoValue:
            pass
        elif initial is not numpy._NoValue:
            pass
        elif where is not numpy._NoValue:
            pass
        else:
            result = dpnp_min(input, axis=axis)

            # scalar returned
            if result.shape == (1,):
                return result.dtype.type(result[0])

            return result

    return call_origin(numpy.min, input, axis, out, keepdims, initial, where)


def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=numpy._NoValue):
    """
    Compute the standard deviation along the specified axis.

    For full documentation refer to :obj:`numpy.std`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Size of input array is limited by ``a.size > 0``.
    Prameters ``axis`` is supported only with default value ``None``.
    Prameters ``dtype`` is supported only with default value ``None``.
    Prameters ``out`` is supported only with default value ``None``.
    Prameters ``keepdims`` is supported only with default value ``numpy._NoValue``.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.var` : Compute the variance along the specified axis.
    :obj:`dpnp.mean` : Compute the arithmetic mean along the specified axis.
    :obj:`dpnp.nanmean` : Compute the arithmetic mean along the specified axis,
                          ignoring NaNs.
    :obj:`dpnp.nanstd` : Compute the standard deviation along
                         the specified axis, while ignoring NaNs.
    :obj:`dpnp.nanvar` : Compute the variance along the specified axis,
                         while ignoring NaNs.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.std(a)
    1.118033988749895

    """

    if not use_origin_backend(a):
        if not isinstance(a, dparray):
            pass
        elif a.size == 0:
            pass
        elif axis is not None:
            pass
        elif dtype is not None:
            pass
        elif out is not None:
            pass
        elif keepdims is not numpy._NoValue:
            pass
        else:
            result = dpnp_std(a, ddof)
            if result.shape == (1,):
                return result.dtype.type(result[0])

            return result

    return call_origin(numpy.std, a, axis, dtype, out, ddof, keepdims)


def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=numpy._NoValue):
    """
    Compute the variance along the specified axis.

    For full documentation refer to :obj:`numpy.var`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Size of input array is limited by ``a.size > 0``.
    Prameters ``axis`` is supported only with default value ``None``.
    Prameters ``dtype`` is supported only with default value ``None``.
    Prameters ``out`` is supported only with default value ``None``.
    Prameters ``keepdims`` is supported only with default value ``numpy._NoValue``.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.std` : Compute the standard deviation along the specified axis.
    :obj:`dpnp.mean` : Compute the arithmetic mean along the specified axis.
    :obj:`dpnp.nanmean` : Compute the arithmetic mean along the specified axis,
                          ignoring NaNs.
    :obj:`dpnp.nanstd` : Compute the standard deviation along
                         the specified axis, while ignoring NaNs.
    :obj:`dpnp.nanvar` : Compute the variance along the specified axis,
                         while ignoring NaNs.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.var(a)
    1.25

    """

    if not use_origin_backend(a):
        if not isinstance(a, dparray):
            pass
        elif a.size == 0:
            pass
        elif axis is not None:
            pass
        elif dtype is not None:
            pass
        elif out is not None:
            pass
        elif keepdims is not numpy._NoValue:
            pass
        else:
            result = dpnp_var(a, ddof)
            if result.shape == (1,):
                return result.dtype.type(result[0])

            return result

    return call_origin(numpy.var, a, axis, dtype, out, ddof, keepdims)
