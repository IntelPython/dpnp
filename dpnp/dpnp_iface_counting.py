# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2025, Intel Corporation
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
Interface of the counting function of the dpnp

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""

import dpctl.tensor as dpt

import dpnp

__all__ = ["count_nonzero"]


def count_nonzero(a, axis=None, *, keepdims=False, out=None):
    """
    Counts the number of non-zero values in the array `a`.

    For full documentation refer to :obj:`numpy.count_nonzero`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        The array for which to count non-zeros.
    axis : {None, int, tuple}, optional
        Axis or tuple of axes along which to count non-zeros.
        Default value means that non-zeros will be counted along a flattened
        version of `a`.

        Default: ``None``.
    keepdims : {None, bool}, optional
        If this is set to ``True``, the axes that are counted are left in the
        result as dimensions with size one. With this option, the result will
        broadcast correctly against the input array.

        Default: ``False``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        The array into which the result is written. The data type of `out` must
        match the expected shape and the expected data type of the result.
        If ``None`` then a new array is returned.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        Number of non-zero values in the array along a given axis.
        Otherwise, a zero-dimensional array with the total number of non-zero
        values in the array is returned.

    See Also
    --------
    :obj:`dpnp.nonzero` : Return the coordinates of all the non-zero values.

    Examples
    --------
    >>> import dpnp as np
    >>> np.count_nonzero(np.eye(4))
    array(4)
    >>> a = np.array([[0, 1, 7, 0],
                      [3, 0, 2, 19]])
    >>> np.count_nonzero(a)
    array(5)
    >>> np.count_nonzero(a, axis=0)
    array([1, 1, 2, 1])
    >>> np.count_nonzero(a, axis=1)
    array([2, 3])
    >>> np.count_nonzero(a, axis=1, keepdims=True)
    array([[2],
           [3]])

    """

    usm_a = dpnp.get_usm_ndarray(a)
    usm_out = None if out is None else dpnp.get_usm_ndarray(out)

    usm_res = dpt.count_nonzero(
        usm_a, axis=axis, keepdims=keepdims, out=usm_out
    )
    return dpnp.get_result_array(usm_res, out)
