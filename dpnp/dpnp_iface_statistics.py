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
from dpnp.dpnp_utils import *
import dpnp


__all__ = [
    'amax',
    'amin',
    'average',
    'bincount',
    'correlate',
    'cov',
    'histogram',
    'max',
    'mean',
    'median',
    'min',
    'nanvar',
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


def average(x1, axis=None, weights=None, returned=False):
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

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if axis is not None:
            pass
        elif weights is not None:
            pass
        elif returned:
            pass
        else:
            result_obj = dpnp_average(x1_desc)
            result = dpnp.convert_single_elem_array_to_scalar(result_obj)

            return result

    return call_origin(numpy.average, x1, axis, weights, returned)


def bincount(x1, weights=None, minlength=0):
    """
    Count number of occurrences of each value in array of non-negative ints.

    For full documentation refer to :obj:`numpy.bincount`.

    See Also
    --------
    :obj:`dpnp.unique` : Find the unique elements of an array.

    Examples
    --------
    >>> import dpnp as np
    >>> res = np.bincount(np.arange(5))
    >>> print(res)
    [1, 1, 1, 1, 1]

    """

    return call_origin(numpy.bincount, x1, weights=weights, minlength=minlength)


def correlate(x1, x2, mode='valid'):
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

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    x2_desc = dpnp.get_dpnp_descriptor(x2)
    if x1_desc and x2_desc:
        if x1_desc.size != x2_desc.size or x1_desc.size == 0:
            pass
        elif x1_desc.shape != x2_desc.shape:
            pass
        elif mode != 'valid':
            pass
        else:
            return dpnp_correlate(x1_desc, x2_desc).get_pyobj()

    return call_origin(numpy.correlate, x1, x2, mode=mode)


def cov(x1, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None):
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

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if x1_desc.ndim > 2:
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
            if x1_desc.dtype != dpnp.float64:
                x1_desc = dpnp.get_dpnp_descriptor(dpnp.astype(x1, dpnp.float64))

            return dpnp_cov(x1_desc).get_pyobj()

    return call_origin(numpy.cov, x1, y, rowvar, bias, ddof, fweights, aweights)


def histogram(a, bins=10, range=None, normed=None, weights=None, density=None):
    """
    Compute the histogram of a dataset.
    For full documentation refer to :obj:`numpy.histogram`.
    Examples
    --------
    >>> import dpnp
    >>> dpnp.histogram([1, 2, 1], bins=[0, 1, 2, 3])
    (array([0, 2, 1]), array([0, 1, 2, 3]))
    >>> dpnp.histogram(dpnp.arange(4), bins=dpnp.arange(5), density=True)
    (array([0.25, 0.25, 0.25, 0.25]), array([0, 1, 2, 3, 4]))
    >>> dpnp.histogram([[1, 2, 1], [1, 0, 1]], bins=[0,1,2,3])
    (array([1, 4, 1]), array([0, 1, 2, 3]))
    >>> a = dpnp.arange(5)
    >>> hist, bin_edges = dpnp.histogram(a, density=True)
    >>> hist
    array([0.5, 0. , 0.5, 0. , 0. , 0.5, 0. , 0.5, 0. , 0.5])
    >>> hist.sum()
    2.4999999999999996
    >>> res = dpnp.sum(hist * dpnp.diff(bin_edges))
    >>> print(res)
    1.0
    """

    return call_origin(numpy.histogram, a=a, bins=bins, range=range, normed=normed, weights=weights, density=density)


def max(x1, axis=None, out=None, keepdims=False, initial=None, where=True):
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

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        # Negative values in 'shape' are not allowed in input array
        # 306-322 check on negative and duplicate axis
        isaxis = True
        if axis is not None:
            if dpnp.isscalar(axis):
                if axis < 0:
                    isaxis = False
            else:
                for val in axis:
                    if val < 0:
                        isaxis = False
                        break
                if isaxis:
                    for i in range(len(axis)):
                        for j in range(len(axis)):
                            if i != j:
                                if axis[i] == axis[j]:
                                    isaxis = False
                                    break

        if not isaxis:
            pass
        elif out is not None:
            pass
        elif keepdims:
            pass
        elif initial is not None:
            pass
        elif where is not True:
            pass
        else:
            result_obj = dpnp_max(x1_desc, axis).get_pyobj()
            result = dpnp.convert_single_elem_array_to_scalar(result_obj)

            return result

    return call_origin(numpy.max, x1, axis, out, keepdims, initial, where)


def mean(x1, axis=None, **kwargs):
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

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc and not kwargs:
        if x1_desc.size == 0:
            pass
        else:
            result_obj = dpnp_mean(x1_desc, axis)
            result = dpnp.convert_single_elem_array_to_scalar(result_obj)

            return result

    return call_origin(numpy.mean, x1, axis=axis, **kwargs)


def median(x1, axis=None, out=None, overwrite_input=False, keepdims=False):
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

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if axis is not None:
            pass
        elif out is not None:
            pass
        elif overwrite_input:
            pass
        elif keepdims:
            pass
        else:
            result_obj = dpnp_median(x1_desc).get_pyobj()
            result = dpnp.convert_single_elem_array_to_scalar(result_obj)

            return result

    return call_origin(numpy.median, x1, axis, out, overwrite_input, keepdims)


def min(x1, axis=None, out=None, keepdims=False, initial=None, where=True):
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

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if out is not None:
            pass
        elif keepdims:
            pass
        elif initial is not None:
            pass
        elif where is not True:
            pass
        else:
            result_obj = dpnp_min(x1_desc, axis).get_pyobj()
            result = dpnp.convert_single_elem_array_to_scalar(result_obj)

            return result

    return call_origin(numpy.min, x1, axis, out, keepdims, initial, where)


def nanvar(x1, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    """
    Compute the variance along the specified axis, while ignoring NaNs.

    For full documentation refer to :obj:`numpy.nanvar`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Prameters ``axis`` is supported only with default value ``None``.
    Prameters ``dtype`` is supported only with default value ``None``.
    Prameters ``out`` is supported only with default value ``None``.
    Prameters ``keepdims`` is supported only with default value ``numpy._NoValue``.
    Otherwise the function will be executed sequentially on CPU.
    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if x1.size == 0:
            pass
        elif axis is not None:
            pass
        elif dtype is not None:
            pass
        elif out is not None:
            pass
        elif keepdims:
            pass
        else:
            result_obj = dpnp_nanvar(x1_desc, ddof).get_pyobj()
            result = dpnp.convert_single_elem_array_to_scalar(result_obj)

            return result

    return call_origin(numpy.nanvar, x1, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)


def std(x1, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
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

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if x1_desc.size == 0:
            pass
        elif axis is not None:
            pass
        elif dtype is not None:
            pass
        elif out is not None:
            pass
        elif keepdims:
            pass
        else:
            result_obj = dpnp_std(x1_desc, ddof).get_pyobj()
            result = dpnp.convert_single_elem_array_to_scalar(result_obj)

            return result

    return call_origin(numpy.std, x1, axis, dtype, out, ddof, keepdims)


def var(x1, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
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

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if x1_desc.size == 0:
            pass
        elif axis is not None:
            pass
        elif dtype is not None:
            pass
        elif out is not None:
            pass
        elif keepdims:
            pass
        else:
            result_obj = dpnp_var(x1_desc, ddof).get_pyobj()
            result = dpnp.convert_single_elem_array_to_scalar(result_obj)

            return result

    return call_origin(numpy.var, x1, axis, dtype, out, ddof, keepdims)
