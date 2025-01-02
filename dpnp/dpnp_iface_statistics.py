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
Interface of the statistics function of the DPNP

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""

import math

import dpctl.tensor as dpt
import dpctl.utils as dpu
import numpy
from dpctl.tensor._numpy_helper import normalize_axis_index

import dpnp

# pylint: disable=no-name-in-module
import dpnp.backend.extensions.statistics._statistics_impl as statistics_ext
from dpnp.dpnp_utils.dpnp_utils_common import (
    result_type_for_device,
    to_supported_dtypes,
)

from .dpnp_utils import call_origin, get_usm_allocations
from .dpnp_utils.dpnp_utils_reduction import dpnp_wrap_reduction_call
from .dpnp_utils.dpnp_utils_statistics import dpnp_cov, dpnp_median

min_ = min  # pylint: disable=used-before-assignment

__all__ = [
    "amax",
    "amin",
    "average",
    "corrcoef",
    "correlate",
    "cov",
    "max",
    "mean",
    "median",
    "min",
    "ptp",
    "std",
    "var",
]


def _count_reduce_items(arr, axis, where=True):
    """
    Calculates the number of items used in a reduction operation
    along the specified axis or axes.

    Parameters
    ----------
    arr : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int, tuple of ints}, optional
        axis or axes along which the number of items used in a reduction
        operation must be counted. If a tuple of unique integers is given,
        the items are counted over multiple axes. If ``None``, the variance
        is computed over the entire array.
        Default: `None`.

    Returns
    -------
    out : int
        The number of items should be used in a reduction operation.

    Limitations
    -----------
    Parameters `where` is only supported with its default value.

    """
    if where is True:
        # no boolean mask given, calculate items according to axis
        if axis is None:
            axis = tuple(range(arr.ndim))
        elif not isinstance(axis, tuple):
            axis = (axis,)
        items = 1
        for ax in axis:
            items *= arr.shape[normalize_axis_index(ax, arr.ndim)]
        items = dpnp.intp(items)
    else:
        raise NotImplementedError(
            "where keyword argument is only supported with its default value."
        )
    return items


def amax(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    """
    Return the maximum of an array or maximum along an axis.

    `amax` is an alias of :obj:`dpnp.max`.

    See Also
    --------
    :obj:`dpnp.max` : alias of this function
    :obj:`dpnp.ndarray.max` : equivalent method

    """

    return max(
        a, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where
    )


def amin(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    """
    Return the minimum of an array or minimum along an axis.

    `amin` is an alias of :obj:`dpnp.min`.

    See Also
    --------
    :obj:`dpnp.min` : alias of this function
    :obj:`dpnp.ndarray.min` : equivalent method

    """

    return min(
        a, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where
    )


def average(a, axis=None, weights=None, returned=False, *, keepdims=False):
    """
    Compute the weighted average along the specified axis.

    For full documentation refer to :obj:`numpy.average`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int, tuple of ints}, optional
        Axis or axes along which the averages must be computed. If
        a tuple of unique integers, the averages are computed over multiple
        axes. If ``None``, the average is computed over the entire array.
        Default: ``None``.
    weights : {array_like}, optional
        An array of weights associated with the values in `a`. Each value in
        `a` contributes to the average according to its associated weight.
        The weights array can either be 1-D (in which case its length must be
        the size of `a` along the given axis) or of the same shape as `a`.
        If `weights=None`, then all data in `a` are assumed to have a
        weight equal to one.  The 1-D calculation is::

            avg = sum(a * weights) / sum(weights)

        The only constraint on `weights` is that `sum(weights)` must not be 0.
    returned : {bool}, optional
        If ``True``, the tuple (`average`, `sum_of_weights`) is returned,
        otherwise only the average is returned. If `weights=None`,
        `sum_of_weights` is equivalent to the number of elements over which
        the average is taken.
        Default: ``False``.
    keepdims : {None, bool}, optional
        If ``True``, the reduced axes (dimensions) are included in the result
        as singleton dimensions, so that the returned array remains
        compatible with the input array according to Array Broadcasting
        rules. Otherwise, if ``False``, the reduced axes are not included in
        the returned array.
        Default: ``False``.

    Returns
    -------
    out, [sum_of_weights] : dpnp.ndarray, dpnp.ndarray
        Return the average along the specified axis. When `returned` is
        ``True``, return a tuple with the average as the first element and
        the sum of the weights as the second element. `sum_of_weights` is of
        the same type as `out`. The result dtype follows a general pattern.
        If `weights` is ``None``, the result dtype will be that of `a` , or
        default floating point data type for the device where input array `a`
        is allocated. Otherwise, if `weights` is not ``None`` and `a` is
        non-integral, the result type will be the type of lowest precision
        capable of representing values of both `a` and `weights`. If `a`
        happens to be integral, the previous rules still applies but the result
        dtype will at least be default floating point data type for the device
        where input array `a` is allocated.

    See Also
    --------
    :obj:`dpnp.mean` : Compute the arithmetic mean along the specified axis.
    :obj:`dpnp.sum` : Sum of array elements over a given axis.

    Examples
    --------
    >>> import dpnp as np
    >>> data = np.arange(1, 5)
    >>> data
    array([1, 2, 3, 4])
    >>> np.average(data)
    array(2.5)
    >>> np.average(np.arange(1, 11), weights=np.arange(10, 0, -1))
    array(4.0)

    >>> data = np.arange(6).reshape((3, 2))
    >>> data
    array([[0, 1],
        [2, 3],
        [4, 5]])
    >>> np.average(data, axis=1, weights=[1./4, 3./4])
    array([0.75, 2.75, 4.75])
    >>> np.average(data, weights=[1./4, 3./4])
    TypeError: Axis must be specified when shapes of a and weights differ.

    With ``keepdims=True``, the following result has shape (3, 1).

    >>> np.average(data, axis=1, keepdims=True)
    array([[0.5],
        [2.5],
        [4.5]])

    >>> a = np.ones(5, dtype=np.float64)
    >>> w = np.ones(5, dtype=np.complex64)
    >>> avg = np.average(a, weights=w)
    >>> print(avg.dtype)
    complex128

    """

    dpnp.check_supported_arrays_type(a)
    usm_type, exec_q = get_usm_allocations([a, weights])

    if weights is None:
        avg = dpnp.mean(a, axis=axis, keepdims=keepdims)
        scl = dpnp.asanyarray(
            avg.dtype.type(a.size / avg.size),
            usm_type=usm_type,
            sycl_queue=exec_q,
        )
    else:
        if not dpnp.is_supported_array_type(weights):
            weights = dpnp.asarray(
                weights, usm_type=usm_type, sycl_queue=exec_q
            )

        a_dtype = a.dtype
        if not dpnp.issubdtype(a_dtype, dpnp.inexact):
            default_dtype = dpnp.default_float_type(a.device)
            res_dtype = dpnp.result_type(a_dtype, weights.dtype, default_dtype)
        else:
            res_dtype = dpnp.result_type(a_dtype, weights.dtype)

        # Sanity checks
        wgt_shape = weights.shape
        a_shape = a.shape
        if a_shape != wgt_shape:
            if axis is None:
                raise TypeError(
                    "Axis must be specified when shapes of input array and "
                    "weights differ."
                )
            if weights.ndim != 1:
                raise TypeError(
                    "1D weights expected when shapes of input array and "
                    "weights differ."
                )
            if wgt_shape[0] != a_shape[axis]:
                raise ValueError(
                    "Length of weights not compatible with specified axis."
                )

            # setup weights to broadcast along axis
            weights = dpnp.broadcast_to(
                weights, (a.ndim - 1) * (1,) + wgt_shape
            )
            weights = weights.swapaxes(-1, axis)

        scl = weights.sum(axis=axis, dtype=res_dtype, keepdims=keepdims)
        if dpnp.any(scl == 0.0):
            raise ZeroDivisionError("Weights sum to zero, can't be normalized")

        avg = dpnp.multiply(a, weights).sum(
            axis=axis, dtype=res_dtype, keepdims=keepdims
        )
        avg /= scl

    if returned:
        if scl.shape != avg.shape:
            scl = dpnp.broadcast_to(scl, avg.shape).copy()
        return avg, scl
    return avg


def corrcoef(x, y=None, rowvar=True, *, dtype=None):
    """
    Return Pearson product-moment correlation coefficients.

    For full documentation refer to :obj:`numpy.corrcoef`.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `x` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    y : {None, dpnp.ndarray, usm_ndarray}, optional
        An additional set of variables and observations. `y` has the same
        shape as `x`.
        Default: ``None``.
    rowvar : {bool}, optional
        If `rowvar` is ``True``, then each row represents a variable,
        with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
        Default: ``True``.
    dtype : {None, dtype}, optional
        Data-type of the result.
        Default: ``None``.

    Returns
    -------
    R : {dpnp.ndarray}
        The correlation coefficient matrix of the variables.

    See Also
    --------
    :obj:`dpnp.cov` : Covariance matrix.

    Examples
    --------
    In this example we generate two random arrays, ``xarr`` and ``yarr``, and
    compute the row-wise and column-wise Pearson correlation coefficients,
    ``R``. Since `rowvar` is true by default, we first find the row-wise
    Pearson correlation coefficients between the variables of ``xarr``.

    >>> import dpnp as np
    >>> np.random.seed(123)
    >>> xarr = np.random.rand(3, 3).astype(np.float32)
    >>> xarr
    array([[7.2858386e-17, 2.2066992e-02, 3.9520904e-01],
           [4.8012391e-01, 5.9377134e-01, 4.5147297e-01],
           [9.0728188e-01, 9.9387854e-01, 5.8399546e-01]], dtype=float32)
    >>> R1 = np.corrcoef(xarr)
    >>> R1
    array([[ 0.99999994, -0.6173796 , -0.9685411 ],
           [-0.6173796 ,  1.        ,  0.7937219 ],
           [-0.9685411 ,  0.7937219 ,  0.9999999 ]], dtype=float32)

    If we add another set of variables and observations ``yarr``, we can
    compute the row-wise Pearson correlation coefficients between the
    variables in ``xarr`` and ``yarr``.

    >>> yarr = np.random.rand(3, 3).astype(np.float32)
    >>> yarr
    array([[0.17615308, 0.65354985, 0.15716429],
           [0.09373496, 0.2123185 , 0.84086883],
           [0.9011005 , 0.45206687, 0.00225109]], dtype=float32)
    >>> R2 = np.corrcoef(xarr, yarr)
    >>> R2
    array([[ 0.99999994, -0.6173796 , -0.968541  , -0.48613155,  0.9951523 ,
            -0.8900264 ],
           [-0.6173796 ,  1.        ,  0.7937219 ,  0.9875833 , -0.53702235,
             0.19083664],
           [-0.968541  ,  0.7937219 ,  0.9999999 ,  0.6883078 , -0.9393724 ,
             0.74857277],
           [-0.48613152,  0.9875833 ,  0.6883078 ,  0.9999999 , -0.39783284,
             0.0342579 ],
           [ 0.9951523 , -0.53702235, -0.9393725 , -0.39783284,  0.99999994,
            -0.9305482 ],
           [-0.89002645,  0.19083665,  0.7485727 ,  0.0342579 , -0.9305482 ,
             1.        ]], dtype=float32)

    Finally if we use the option ``rowvar=False``, the columns are now
    being treated as the variables and we will find the column-wise Pearson
    correlation coefficients between variables in ``xarr`` and ``yarr``.

    >>> R3 = np.corrcoef(xarr, yarr, rowvar=False)
    >>> R3
    array([[ 1.        ,  0.9724453 , -0.9909503 ,  0.8104691 , -0.46436927,
            -0.1643624 ],
           [ 0.9724453 ,  1.        , -0.9949381 ,  0.6515728 , -0.6580445 ,
             0.07012729],
           [-0.99095035, -0.994938  ,  1.        , -0.72450536,  0.5790461 ,
             0.03047091],
           [ 0.8104691 ,  0.65157276, -0.72450536,  1.        ,  0.14243561,
            -0.71102554],
           [-0.4643693 , -0.6580445 ,  0.57904613,  0.1424356 ,  0.99999994,
            -0.79727215],
           [-0.1643624 ,  0.07012729,  0.03047091, -0.7110255 , -0.7972722 ,
             0.99999994]], dtype=float32)
    """

    out = dpnp.cov(x, y, rowvar, dtype=dtype)
    if out.ndim == 0:
        # scalar covariance
        # nan if incorrect value (nan, inf, 0), 1 otherwise
        return out / out

    d = dpnp.diag(out)

    stddev = dpnp.sqrt(d.real)
    out /= stddev[:, None]
    out /= stddev[None, :]

    # Clip real and imaginary parts to [-1, 1]. This does not guarantee
    # abs(a[i,j]) <= 1 for complex arrays, but is the best we can do without
    # excessive work.
    dpnp.clip(out.real, -1, 1, out=out.real)
    if dpnp.iscomplexobj(out):
        dpnp.clip(out.imag, -1, 1, out=out.imag)

    return out


def _get_padding(a_size, v_size, mode):
    assert v_size <= a_size

    if mode == "valid":
        l_pad, r_pad = 0, 0
    elif mode == "same":
        l_pad = v_size // 2
        r_pad = v_size - l_pad - 1
    elif mode == "full":
        l_pad, r_pad = v_size - 1, v_size - 1
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Only 'valid', 'same', 'full' are supported."
        )

    return l_pad, r_pad


def _choose_conv_method(a, v, rdtype):
    assert a.size >= v.size
    if rdtype == dpnp.bool:
        return "direct"

    if v.size < 10**4 or a.size < 10**4:
        return "direct"

    if dpnp.issubdtype(rdtype, dpnp.integer):
        max_a = int(dpnp.max(dpnp.abs(a)))
        sum_v = int(dpnp.sum(dpnp.abs(v)))
        max_value = int(max_a * sum_v)

        default_float = dpnp.default_float_type(a.sycl_device)
        if max_value > 2 ** numpy.finfo(default_float).nmant - 1:
            return "direct"

    if dpnp.issubdtype(rdtype, dpnp.number):
        return "fft"

    raise ValueError(f"Unsupported dtype: {rdtype}")


def _run_native_sliding_dot_product1d(a, v, l_pad, r_pad, rdtype):
    queue = a.sycl_queue
    device = a.sycl_device

    supported_types = statistics_ext.sliding_dot_product1d_dtypes()
    supported_dtype = to_supported_dtypes(rdtype, supported_types, device)

    if supported_dtype is None:
        raise ValueError(
            f"function does not support input types "
            f"({a.dtype.name}, {v.dtype.name}), "
            "and the inputs could not be coerced to any "
            f"supported types. List of supported types: "
            f"{[st.name for st in supported_types]}"
        )

    a_casted = dpnp.asarray(a, dtype=supported_dtype, order="C")
    v_casted = dpnp.asarray(v, dtype=supported_dtype, order="C")

    usm_type = dpu.get_coerced_usm_type([a_casted.usm_type, v_casted.usm_type])
    out_size = l_pad + r_pad + a_casted.size - v_casted.size + 1
    # out type is the same as input type
    out = dpnp.empty_like(a_casted, shape=out_size, usm_type=usm_type)

    a_usm = dpnp.get_usm_ndarray(a_casted)
    v_usm = dpnp.get_usm_ndarray(v_casted)
    out_usm = dpnp.get_usm_ndarray(out)

    _manager = dpu.SequentialOrderManager[queue]

    mem_ev, corr_ev = statistics_ext.sliding_dot_product1d(
        a_usm,
        v_usm,
        out_usm,
        l_pad,
        r_pad,
        depends=_manager.submitted_events,
    )
    _manager.add_event_pair(mem_ev, corr_ev)

    return out


def _convolve_fft(a, v, l_pad, r_pad, rtype):
    assert a.size >= v.size
    assert l_pad < v.size

    # +1 is needed to avoid circular convolution
    padded_size = a.size + r_pad + 1
    fft_size = 2 ** int(math.ceil(math.log2(padded_size)))

    af = dpnp.fft.fft(a, fft_size)  # pylint: disable=no-member
    vf = dpnp.fft.fft(v, fft_size)  # pylint: disable=no-member

    r = dpnp.fft.ifft(af * vf)  # pylint: disable=no-member
    if dpnp.issubdtype(rtype, dpnp.floating):
        r = r.real
    elif dpnp.issubdtype(rtype, dpnp.integer) or rtype == dpnp.bool:
        r = r.real.round()

    start = v.size - 1 - l_pad
    end = padded_size - 1

    return r[start:end]


def correlate(a, v, mode="valid", method="auto"):
    r"""
    Cross-correlation of two 1-dimensional sequences.

    This function computes the correlation as generally defined in signal
    processing texts [1]_:

    .. math:: c_k = \sum_n a_{n+k} \cdot \overline{v}_n

    with `a` and `v` sequences being zero-padded where necessary and
    :math:`\overline v` denoting complex conjugation.

    For full documentation refer to :obj:`numpy.correlate`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        First input array.
    v : {dpnp.ndarray, usm_ndarray}
        Second input array.
    mode : {"valid", "same", "full"}, optional
        Refer to the :obj:`dpnp.convolve` docstring. Note that the default
        is ``"valid"``, unlike :obj:`dpnp.convolve`, which uses ``"full"``.

        Default: ``"valid"``.
    method : {"auto", "direct", "fft"}, optional
        `"direct"`: The correlation is determined directly from sums.

        `"fft"`: The Fourier Transform is used to perform the calculations.
        This method is faster for long sequences but can have accuracy issues.

        `"auto"`: Automatically chooses direct or Fourier method based on
        an estimate of which is faster.

        Note: Use of the FFT convolution on input containing NAN or INF
        will lead to the entire output being NAN or INF.
        Use method='direct' when your input contains NAN or INF values.

        Default: ``"auto"``.

    Returns
    -------
    out : {dpnp.ndarray}
        Discrete cross-correlation of `a` and `v`.

    Notes
    -----
    The definition of correlation above is not unique and sometimes
    correlation may be defined differently. Another common definition is [1]_:

    .. math:: c'_k = \sum_n a_{n} \cdot \overline{v_{n+k}}

    which is related to :math:`c_k` by :math:`c'_k = c_{-k}`.

    References
    ----------
    .. [1] Wikipedia, "Cross-correlation",
           https://en.wikipedia.org/wiki/Cross-correlation

    See Also
    --------
    :obj:`dpnp.convolve` : Discrete, linear convolution of two one-dimensional
                        sequences.


    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 2, 3], dtype=np.float32)
    >>> v = np.array([0, 1, 0.5], dtype=np.float32)
    >>> np.correlate(a, v)
    array([3.5], dtype=float32)
    >>> np.correlate(a, v, "same")
    array([2. , 3.5, 3. ], dtype=float32)
    >>> np.correlate([a, v, "full")
    array([0.5, 2. , 3.5, 3. , 0. ], dtype=float32)

    Using complex sequences:

    >>> ac = np.array([1+1j, 2, 3-1j], dtype=np.complex64)
    >>> vc = np.array([0, 1, 0.5j], dtype=np.complex64)
    >>> np.correlate(ac, vc, 'full')
    array([0.5-0.5j, 1. +0.j , 1.5-1.5j, 3. -1.j , 0. +0.j ], dtype=complex64)

    Note that you get the time reversed, complex conjugated result
    (:math:`\overline{c_{-k}}`) when the two input sequences `a` and `v` change
    places:

    >>> np.correlate(vc, ac, 'full')
    array([0. +0.j , 3. +1.j , 1.5+1.5j, 1. +0.j , 0.5+0.5j], dtype=complex64)

    """

    dpnp.check_supported_arrays_type(a, v)

    if a.size == 0 or v.size == 0:
        raise ValueError(
            f"Array arguments cannot be empty. "
            f"Received sizes: a.size={a.size}, v.size={v.size}"
        )
    if a.ndim != 1 or v.ndim != 1:
        raise ValueError(
            f"Only 1-dimensional arrays are supported. "
            f"Received shapes: a.shape={a.shape}, v.shape={v.shape}"
        )

    supported_methods = ["auto", "direct", "fft"]
    if method not in supported_methods:
        raise ValueError(
            f"Unknown method: {method}. Supported methods: {supported_methods}"
        )

    device = a.sycl_device
    rdtype = result_type_for_device([a.dtype, v.dtype], device)

    if dpnp.issubdtype(v.dtype, dpnp.complexfloating):
        v = dpnp.conj(v)

    revert = False
    if v.size > a.size:
        revert = True
        a, v = v, a

    l_pad, r_pad = _get_padding(a.size, v.size, mode)

    if method == "auto":
        method = _choose_conv_method(a, v, rdtype)

    if method == "direct":
        r = _run_native_sliding_dot_product1d(a, v, l_pad, r_pad, rdtype)
    elif method == "fft":
        r = _convolve_fft(a, v[::-1], l_pad, r_pad, rdtype)
    else:
        raise ValueError(f"Unknown method: {method}")

    if revert:
        r = r[::-1]

    return dpnp.asarray(r, dtype=rdtype, order="C")


def cov(
    m,
    y=None,
    rowvar=True,
    bias=False,
    ddof=None,
    fweights=None,
    aweights=None,
    *,
    dtype=None,
):
    """
    Estimate a covariance matrix, given data and weights.

    For full documentation refer to :obj:`numpy.cov`.

    Returns
    -------
    out : dpnp.ndarray
        The covariance matrix of the variables.

    Limitations
    -----------
    Input array ``m`` is supported as :obj:`dpnp.ndarray`.
    Dimension of input array ``m`` is limited by ``m.ndim <= 2``.
    Size and shape of input arrays are supported to be equal.
    Parameter `y` is supported only with default value ``None``.
    Parameter `bias` is supported only with default value ``False``.
    Parameter `ddof` is supported only with default value ``None``.
    Parameter `fweights` is supported only with default value ``None``.
    Parameter `aweights` is supported only with default value ``None``.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.corrcoef` : Normalized covariance matrix

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

    if not dpnp.is_supported_array_type(m):
        pass
    elif m.ndim > 2:
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
        return dpnp_cov(m, y=y, rowvar=rowvar, dtype=dtype)

    return call_origin(
        numpy.cov, m, y, rowvar, bias, ddof, fweights, aweights, dtype=dtype
    )


def max(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    """
    Return the maximum of an array or maximum along an axis.

    For full documentation refer to :obj:`numpy.max`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int or tuple of ints}, optional
        Axis or axes along which to operate. By default, flattened input is
        used. If this is a tuple of integers, the minimum is selected over
        multiple axes, instead of a single axis or all the axes as before.
        Default: ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. Must be of the
        same shape and buffer length as the expected output.
        Default: ``None``.
    keepdims : {None, bool}, optional
        If this is set to ``True``, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result will
        broadcast correctly against the input array.
        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        Maximum of `a`. If `axis` is ``None``, the result is a zero-dimensional
        array. If `axis` is an integer, the result is an array of dimension
        ``a.ndim - 1``. If `axis` is a tuple, the result is an array of
        dimension ``a.ndim - len(axis)``.

    Limitations
    -----------.
    Parameters `where`, and `initial` are only supported with their default
    values. Otherwise ``NotImplementedError`` exception will be raised.

    See Also
    --------
    :obj:`dpnp.min` : Return the minimum of an array.
    :obj:`dpnp.maximum` : Element-wise maximum of two arrays, propagates NaNs.
    :obj:`dpnp.fmax` : Element-wise maximum of two arrays, ignores NaNs.
    :obj:`dpnp.amax` : The maximum value of an array along a given axis,
                       propagates NaNs.
    :obj:`dpnp.nanmax` : The maximum value of an array along a given axis,
                         ignores NaNs.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.arange(4).reshape((2,2))
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> np.max(a)
    array(3)

    >>> np.max(a, axis=0)   # Maxima along the first axis
    array([2, 3])
    >>> np.max(a, axis=1)   # Maxima along the second axis
    array([1, 3])

    >>> b = np.arange(5, dtype=float)
    >>> b[2] = np.nan
    >>> np.max(b)
    array(nan)

    """

    dpnp.check_limitations(initial=initial, where=where)
    usm_a = dpnp.get_usm_ndarray(a)

    return dpnp_wrap_reduction_call(
        usm_a,
        out,
        dpt.max,
        a.dtype,
        axis=axis,
        keepdims=keepdims,
    )


def mean(a, /, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    """
    Compute the arithmetic mean along the specified axis.

    For full documentation refer to :obj:`numpy.mean`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int, tuple of ints}, optional
        Axis or axes along which the arithmetic means must be computed. If
        a tuple of unique integers, the means are computed over multiple
        axes. If ``None``, the mean is computed over the entire array.
        Default: ``None``.
    dtype : {None, dtype}, optional
        Type to use in computing the mean. By default, if `a` has a
        floating-point data type, the returned array will have
        the same data type as `a`.
        If `a` has a boolean or integral data type, the returned array
        will have the default floating point data type for the device
        where input array `a` is allocated.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the calculated
        values) will be cast if necessary.
        Default: ``None``.
    keepdims : {None, bool}, optional
        If ``True``, the reduced axes (dimensions) are included in the result
        as singleton dimensions, so that the returned array remains
        compatible with the input array according to Array Broadcasting
        rules. Otherwise, if ``False``, the reduced axes are not included in
        the returned array.
        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        An array containing the arithmetic means along the specified axis(axes).
        If the input is a zero-size array, an array containing NaN values is
        returned.

    Limitations
    -----------
    Parameter `where` is only supported with its default value.
    Otherwise ``NotImplementedError`` exception will be raised.

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
    array(2.5)
    >>> np.mean(a, axis=0)
    array([2., 3.])
    >>> np.mean(a, axis=1)
    array([1.5, 3.5])

    """

    dpnp.check_limitations(where=where)

    usm_a = dpnp.get_usm_ndarray(a)
    usm_res = dpt.mean(usm_a, axis=axis, keepdims=keepdims)
    if dtype is not None:
        usm_res = dpt.astype(usm_res, dtype)

    return dpnp.get_result_array(usm_res, out, casting="unsafe")


def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    """
    Compute the median along the specified axis.

    For full documentation refer to :obj:`numpy.median`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int, tuple or list of ints}, optional
        Axis or axes along which the medians are computed. The default,
        ``axis=None``, will compute the median along a flattened version of
        the array. If a sequence of axes, the array is first flattened along
        the given axes, then the median is computed along the resulting
        flattened axis.
        Default: ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the calculated
        values) will be cast if necessary.
        Default: ``None``.
    overwrite_input : bool, optional
       If ``True``, then allow use of memory of input array `a` for
       calculations. The input array will be modified by the call to
       :obj:`dpnp.median`. This will save memory when you do not need to
       preserve the contents of the input array. Treat the input as undefined,
       but it will probably be fully or partially sorted.
       Default: ``False``.
    keepdims : bool, optional
        If ``True``, the reduced axes (dimensions) are included in the result
        as singleton dimensions, so that the returned array remains
        compatible with the input array according to Array Broadcasting
        rules. Otherwise, if ``False``, the reduced axes are not included in
        the returned array.
        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        A new array holding the result. If `a` has a floating-point data type,
        the returned array will have the same data type as `a`. If `a` has a
        boolean or integral data type, the returned array will have the
        default floating point data type for the device where input array `a`
        is allocated.

    See Also
    --------
    :obj:`dpnp.mean` : Compute the arithmetic mean along the specified axis.
    :obj:`dpnp.percentile` : Compute the q-th percentile of the data
                             along the specified axis.

    Notes
    -----
    Given a vector ``V`` of length ``N``, the median of ``V`` is the
    middle value of a sorted copy of ``V``, ``V_sorted`` - i.e.,
    ``V_sorted[(N-1)/2]``, when ``N`` is odd, and the average of the
    two middle values of ``V_sorted`` when ``N`` is even.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[10, 7, 4], [3, 2, 1]])
    >>> a
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> np.median(a)
    array(3.5)

    >>> np.median(a, axis=0)
    array([6.5, 4.5, 2.5])
    >>> np.median(a, axis=1)
    array([7., 2.])
    >>> np.median(a, axis=(0, 1))
    array(3.5)

    >>> m = np.median(a, axis=0)
    >>> out = np.zeros_like(m)
    >>> np.median(a, axis=0, out=m)
    array([6.5, 4.5, 2.5])
    >>> m
    array([6.5, 4.5, 2.5])

    >>> b = a.copy()
    >>> np.median(b, axis=1, overwrite_input=True)
    array([7., 2.])
    >>> assert not np.all(a==b)
    >>> b = a.copy()
    >>> np.median(b, axis=None, overwrite_input=True)
    array(3.5)
    >>> assert not np.all(a==b)

    """

    dpnp.check_supported_arrays_type(a)
    return dpnp_median(
        a, axis, out, overwrite_input, keepdims, ignore_nan=False
    )


def min(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    """
    Return the minimum of an array or maximum along an axis.

    For full documentation refer to :obj:`numpy.min`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int or tuple of ints}, optional
        Axis or axes along which to operate. By default, flattened input is
        used. If this is a tuple of integers, the minimum is selected over
        multiple axes, instead of a single axis or all the axes as before.
        Default: ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. Must be of the
        same shape and buffer length as the expected output.
        Default: ``None``.
    keepdims : {None, bool}, optional
        If this is set to ``True``, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result will
        broadcast correctly against the input array.
        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        Minimum of `a`. If `axis` is ``None``, the result is a zero-dimensional
        array. If `axis` is an integer, the result is an array of dimension
        ``a.ndim - 1``. If `axis` is a tuple, the result is an array of
        dimension ``a.ndim - len(axis)``.

    Limitations
    -----------
    Parameters `where`, and `initial` are only supported with their default
    values. Otherwise ``NotImplementedError`` exception will be raised.

    See Also
    --------
    :obj:`dpnp.max` : Return the maximum of an array.
    :obj:`dpnp.minimum` : Element-wise minimum of two arrays, propagates NaNs.
    :obj:`dpnp.fmin` : Element-wise minimum of two arrays, ignores NaNs.
    :obj:`dpnp.amin` : The minimum value of an array along a given axis,
                       propagates NaNs.
    :obj:`dpnp.nanmin` : The minimum value of an array along a given axis,
                         ignores NaNs.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.arange(4).reshape((2,2))
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> np.min(a)
    array(0)

    >>> np.min(a, axis=0)   # Minima along the first axis
    array([0, 1])
    >>> np.min(a, axis=1)   # Minima along the second axis
    array([0, 2])

    >>> b = np.arange(5, dtype=float)
    >>> b[2] = np.nan
    >>> np.min(b)
    array(nan)

    """

    dpnp.check_limitations(initial=initial, where=where)
    usm_a = dpnp.get_usm_ndarray(a)

    return dpnp_wrap_reduction_call(
        usm_a,
        out,
        dpt.min,
        a.dtype,
        axis=axis,
        keepdims=keepdims,
    )


def ptp(
    a,
    /,
    axis=None,
    out=None,
    keepdims=False,
):
    """
    Range of values (maximum - minimum) along an axis.

    For full documentation refer to :obj:`numpy.ptp`.

    Returns
    -------
    ptp : dpnp.ndarray
        The range of a given array.

    Limitations
    -----------
    Input array is supported as :class:`dpnp.dpnp.ndarray` or
    :class:`dpctl.tensor.usm_ndarray`.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([[4, 9, 2, 10],[6, 9, 7, 12]])
    >>> np.ptp(x, axis=1)
    array([8, 6])

    >>> np.ptp(x, axis=0)
    array([2, 0, 5, 2])

    >>> np.ptp(x)
    array(10)

    """

    return dpnp.subtract(
        dpnp.max(a, axis=axis, keepdims=keepdims, out=out),
        dpnp.min(a, axis=axis, keepdims=keepdims),
        out=out,
    )


def std(
    a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True
):
    """
    Compute the standard deviation along the specified axis.

    For full documentation refer to :obj:`numpy.std`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int, tuple of ints}, optional
        Axis or axes along which the standard deviations must be computed.
        If a tuple of unique integers is given, the standard deviations
        are computed over multiple axes. If ``None``, the standard deviation
        is computed over the entire array.
        Default: ``None``.
    dtype : {None, dtype}, optional
        Type to use in computing the standard deviation. By default,
        if `a` has a floating-point data type, the returned array
        will have the same data type as `a`.
        If `a` has a boolean or integral data type, the returned array
        will have the default floating point data type for the device
        where input array `a` is allocated.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the calculated
        values) will be cast if necessary.
    ddof : {int, float}, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` corresponds to the total
        number of elements over which the standard deviation is calculated.
        Default: `0.0`.
    keepdims : {None, bool}, optional
        If ``True``, the reduced axes (dimensions) are included in the result
        as singleton dimensions, so that the returned array remains
        compatible with the input array according to Array Broadcasting
        rules. Otherwise, if ``False``, the reduced axes are not included in
        the returned array. Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        An array containing the standard deviations. If the standard
        deviation was computed over the entire array, a zero-dimensional
        array is returned.

    Limitations
    -----------
    Parameters `where` is only supported with its default value.
    Otherwise ``NotImplementedError`` exception will be raised.

    Notes
    -----
    Note that, for complex numbers, the absolute value is taken before squaring,
    so that the result is always real and non-negative.

    See Also
    --------
    :obj:`dpnp.ndarray.std` : corresponding function for ndarrays.
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
    array(1.118033988749895)
    >>> np.std(a, axis=0)
    array([1.,  1.])
    >>> np.std(a, axis=1)
    array([0.5,  0.5])

    """

    dpnp.check_supported_arrays_type(a)
    dpnp.check_limitations(where=where)

    if not isinstance(ddof, (int, float)):
        raise TypeError(
            f"An integer or float is required, but got {type(ddof)}"
        )

    if dpnp.issubdtype(a.dtype, dpnp.complexfloating):
        result = dpnp.var(
            a,
            axis=axis,
            dtype=None,
            out=out,
            ddof=ddof,
            keepdims=keepdims,
            where=where,
        )
        dpnp.sqrt(result, out=result)
    else:
        usm_a = dpnp.get_usm_ndarray(a)
        usm_res = dpt.std(usm_a, axis=axis, correction=ddof, keepdims=keepdims)
        result = dpnp.get_result_array(usm_res, out)

    if dtype is not None and out is None:
        result = result.astype(dtype, casting="same_kind")
    return result


def var(
    a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True
):
    """
    Compute the variance along the specified axis.

    For full documentation refer to :obj:`numpy.var`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int, tuple of ints}, optional
        axis or axes along which the variances must be computed. If a tuple
        of unique integers is given, the variances are computed over multiple
        axes. If ``None``, the variance is computed over the entire array.
        Default: ``None``.
    dtype : {None, dtype}, optional
        Type to use in computing the variance. By default, if `a` has a
        floating-point data type, the returned array will have
        the same data type as `a`.
        If `a` has a boolean or integral data type, the returned array
        will have the default floating point data type for the device
        where input array `a` is allocated.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the calculated
        values) will be cast if necessary.
    ddof : {int, float}, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` corresponds to the total
        number of elements over which the variance is calculated.
        Default: `0.0`.
    keepdims : {None, bool}, optional
        If ``True``, the reduced axes (dimensions) are included in the result
        as singleton dimensions, so that the returned array remains
        compatible with the input array according to Array Broadcasting
        rules. Otherwise, if ``False``, the reduced axes are not included in
        the returned array. Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        An array containing the variances. If the variance was computed
        over the entire array, a zero-dimensional array is returned.

    Limitations
    -----------
    Parameters `where` is only supported with its default value.
    Otherwise ``NotImplementedError`` exception will be raised.

    Notes
    -----
    Note that, for complex numbers, the absolute value is taken before squaring,
    so that the result is always real and non-negative.

    See Also
    --------
    :obj:`dpnp.ndarray.var` : corresponding function for ndarrays.
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
    array(1.25)
    >>> np.var(a, axis=0)
    array([1.,  1.])
    >>> np.var(a, axis=1)
    array([0.25,  0.25])

    """

    dpnp.check_supported_arrays_type(a)
    dpnp.check_limitations(where=where)

    if not isinstance(ddof, (int, float)):
        raise TypeError(
            f"An integer or float is required, but got {type(ddof)}"
        )

    if dpnp.issubdtype(a.dtype, dpnp.complexfloating):
        # Note that if dtype is not of inexact type then arrmean
        # will not be either.
        arrmean = dpnp.mean(
            a, axis=axis, dtype=dtype, keepdims=True, where=where
        )
        x = dpnp.subtract(a, arrmean)
        x = dpnp.multiply(x, x.conj(), out=x).real
        result = dpnp.sum(
            x,
            axis=axis,
            dtype=a.real.dtype,
            out=out,
            keepdims=keepdims,
            where=where,
        )

        cnt = _count_reduce_items(a, axis, where)
        cnt = numpy.max(cnt - ddof, 0).astype(result.dtype, casting="same_kind")
        if not cnt:
            cnt = dpnp.nan

        dpnp.divide(result, cnt, out=result)
    else:
        usm_a = dpnp.get_usm_ndarray(a)
        usm_res = dpt.var(usm_a, axis=axis, correction=ddof, keepdims=keepdims)
        result = dpnp.get_result_array(usm_res, out)

    if out is None and dtype is not None:
        result = result.astype(dtype, casting="same_kind")
    return result
