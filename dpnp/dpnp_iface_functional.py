# *****************************************************************************
# Copyright (c) 2024-2025, Intel Corporation
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
Interface of the functional programming routines part of the DPNP

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""


from dpctl.tensor._numpy_helper import (
    normalize_axis_index,
    normalize_axis_tuple,
)

import dpnp

__all__ = ["apply_along_axis", "apply_over_axes"]


def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Apply a function to 1-D slices along the given axis.

    Execute ``func1d(a, *args, **kwargs)`` where `func1d` operates on
    1-D arrays and `a` is a 1-D slice of `arr` along `axis`.

    This is equivalent to (but faster than) the following use of
    :obj:`dpnp.ndindex` and :obj:`dpnp.s_`, which sets each of
    ``ii``, ``jj``, and ``kk`` to a tuple of indices::

        Ni, Nk = a.shape[:axis], a.shape[axis+1:]
        for ii in ndindex(Ni):
            for kk in ndindex(Nk):
                f = func1d(arr[ii + s_[:,] + kk])
                Nj = f.shape
                for jj in ndindex(Nj):
                    out[ii + jj + kk] = f[jj]

    Equivalently, eliminating the inner loop, this can be expressed as::

        Ni, Nk = a.shape[:axis], a.shape[axis+1:]
        for ii in ndindex(Ni):
            for kk in ndindex(Nk):
                out[ii + s_[...,] + kk] = func1d(arr[ii + s_[:,] + kk])

    For full documentation refer to :obj:`numpy.apply_along_axis`.

    Parameters
    ----------
    func1d : function (M,) -> (Nj...)
        This function should accept 1-D arrays. It is applied to 1-D
        slices of `arr` along the specified axis.
    axis : int
        Axis along which `arr` is sliced.
    arr : {dpnp.ndarray, usm_ndarray} (Ni..., M, Nk...)
        Input array.
    args : any
        Additional arguments to `func1d`.
    kwargs : any
        Additional named arguments to `func1d`.

    Returns
    -------
    out : dpnp.ndarray  (Ni..., Nj..., Nk...)
        The output array. The shape of `out` is identical to the shape of
        `arr`, except along the `axis` dimension. This axis is removed, and
        replaced with new dimensions equal to the shape of the return value
        of `func1d`.

    See Also
    --------
    :obj:`dpnp.apply_over_axes` : Apply a function repeatedly over
                                multiple axes.

    Examples
    --------
    >>> import dpnp as np
    >>> def my_func(a): # Average first and last element of a 1-D array
    ...     return (a[0] + a[-1]) * 0.5
    >>> b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> np.apply_along_axis(my_func, 0, b)
    array([4., 5., 6.])
    >>> np.apply_along_axis(my_func, 1, b)
    array([2., 5., 8.])

    For a function that returns a 1D array, the number of dimensions in
    `out` is the same as `arr`.

    >>> b = np.array([[8, 1, 7], [4, 3, 9], [5, 2, 6]])
    >>> np.apply_along_axis(sorted, 1, b)
    array([[1, 7, 8],
           [3, 4, 9],
           [2, 5, 6]])

    For a function that returns a higher dimensional array, those dimensions
    are inserted in place of the `axis` dimension.

    >>> b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> np.apply_along_axis(np.diag, -1, b)
    array([[[1, 0, 0],
            [0, 2, 0],
            [0, 0, 3]],
           [[4, 0, 0],
            [0, 5, 0],
            [0, 0, 6]],
           [[7, 0, 0],
            [0, 8, 0],
            [0, 0, 9]]])

    """

    dpnp.check_supported_arrays_type(arr)
    nd = arr.ndim
    exec_q = arr.sycl_queue
    usm_type = arr.usm_type
    axis = normalize_axis_index(axis, nd)

    # arr, with the iteration axis at the end
    inarr_view = dpnp.moveaxis(arr, axis, -1)

    # compute indices for the iteration axes, and append a trailing ellipsis to
    # prevent 0d arrays decaying to scalars
    inds = dpnp.ndindex(inarr_view.shape[:-1])
    inds = (ind + (Ellipsis,) for ind in inds)

    # invoke the function on the first item
    try:
        ind0 = next(inds)
    except StopIteration:
        raise ValueError(
            "Cannot apply_along_axis when any iteration dimensions are 0"
        ) from None
    res = dpnp.asanyarray(
        func1d(inarr_view[ind0], *args, **kwargs),
        sycl_queue=exec_q,
        usm_type=usm_type,
    )

    # build a buffer for storing evaluations of func1d.
    # remove the requested axis, and add the new ones on the end.
    # laid out so that each write is contiguous.
    # for a tuple index inds, buff[inds] = func1d(inarr_view[inds])
    buff = dpnp.empty_like(res, shape=inarr_view.shape[:-1] + res.shape)

    # save the first result, then compute and save all remaining results
    buff[ind0] = res
    for ind in inds:
        buff[ind] = dpnp.asanyarray(
            func1d(inarr_view[ind], *args, **kwargs),
            sycl_queue=exec_q,
            usm_type=usm_type,
        )

    # restore the inserted axes back to where they belong
    for _ in range(res.ndim):
        buff = dpnp.moveaxis(buff, -1, axis)

    return buff


def apply_over_axes(func, a, axes):
    """
    Apply a function repeatedly over multiple axes.

    `func` is called as ``res = func(a, axis)``, where `axis` is the first
    element of `axes`. The result `res` of the function call must have
    either the same dimensions as `a` or one less dimension. If `res`
    has one less dimension than `a`, a dimension is inserted before
    `axis`. The call to `func` is then repeated for each axis in `axes`,
    with `res` as the first argument.

    For full documentation refer to :obj:`numpy.apply_over_axes`.

    Parameters
    ----------
    func : function
         This function must take two arguments, ``func(a, axis)``.
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axes : {int, sequence of ints}
        Axes over which `func` is applied.

    Returns
    -------
    out : dpnp.ndarray
        The output array. The number of dimensions is the same as `a`,
        but the shape can be different. This depends on whether `func`
        changes the shape of its output with respect to its input.

    See Also
    --------
    :obj:`dpnp.apply_along_axis` : Apply a function to 1-D slices of an array
                                   along the given axis.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.arange(24).reshape(2, 3, 4)
    >>> a
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])

    Sum over axes 0 and 2. The result has same number of dimensions
    as the original array:

    >>> np.apply_over_axes(np.sum, a, [0, 2])
    array([[[ 60],
            [ 92],
            [124]]])

    Tuple axis arguments to ufuncs are equivalent:

    >>> np.sum(a, axis=(0, 2), keepdims=True)
    array([[[ 60],
            [ 92],
            [124]]])

    """

    dpnp.check_supported_arrays_type(a)
    if isinstance(axes, int):
        axes = (axes,)
    axes = normalize_axis_tuple(axes, a.ndim)

    for axis in axes:
        res = func(a, axis)
        if res.ndim != a.ndim:
            res = dpnp.expand_dims(res, axis)
            if res.ndim != a.ndim:
                raise ValueError(
                    "function is not returning an array of the correct shape"
                )
        a = res
    return res
