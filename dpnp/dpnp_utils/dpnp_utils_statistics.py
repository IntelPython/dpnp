# *****************************************************************************
# Copyright (c) 2023-2024, Intel Corporation
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


import warnings

from dpctl.tensor._numpy_helper import normalize_axis_tuple

import dpnp
from dpnp.dpnp_utils import get_usm_allocations, map_dtype_to_device

__all__ = ["dpnp_cov", "dpnp_nanmedian"]


def _calc_nanmedian(a, axis=None, out=None, overwrite_input=False):
    """
    Private function that doesn't support extended axis or keepdims.
    These methods are extended to this function using _ureduce
    See nanmedian for parameter usage

    """
    if axis is None or a.ndim == 1:
        part = dpnp.ravel(a)
        if out is None:
            return _nanmedian1d(part, overwrite_input)
        else:
            out[...] = _nanmedian1d(part, overwrite_input)
            return out
    else:
        result = dpnp.apply_along_axis(_nanmedian1d, axis, a, overwrite_input)
        if out is not None:
            out[...] = result
        return result


def _nanmedian1d(arr1d, overwrite_input=False):
    """
    Private function for rank 1 arrays. Compute the median ignoring NaNs.
    See nanmedian for parameter usage
    """
    arr1d_parsed, overwrite_input = _remove_nan_1d(
        arr1d,
        overwrite_input=overwrite_input,
    )

    if arr1d_parsed.size == 0:
        # Ensure that a nan-esque scalar of the appropriate type (and unit)
        # is returned for `complexfloating`
        return arr1d[-1]

    return dpnp.median(arr1d_parsed, overwrite_input=overwrite_input)


def _remove_nan_1d(arr1d, overwrite_input=False):
    """
    Equivalent to arr1d[~arr1d.isnan()], but in a different order

    Presumably faster as it incurs fewer copies

    Parameters
    ----------
    arr1d : ndarray
        Array to remove nans from
    overwrite_input : bool
        True if `arr1d` can be modified in place

    Returns
    -------
    res : ndarray
        Array with nan elements removed
    overwrite_input : bool
        True if `res` can be modified in place, given the constraint on the
        input
    """

    mask = dpnp.isnan(arr1d)

    s = dpnp.nonzero(mask)[0]
    if s.size == arr1d.size:
        warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=6)
        return arr1d[:0], True
    elif s.size == 0:
        return arr1d, overwrite_input
    else:
        if not overwrite_input:
            arr1d = arr1d.copy()
        # select non-nans at end of array
        enonan = arr1d[-s.size :][~mask[-s.size :]]
        # fill nans in beginning of array with non-nans of end
        arr1d[s[: enonan.size]] = enonan

        return arr1d[: -s.size], True


def dpnp_cov(m, y=None, rowvar=True, dtype=None):
    """
    dpnp_cov(m, y=None, rowvar=True, dtype=None)

    Estimate a covariance matrix based on passed data.
    No support for given weights is provided now.

    The implementation is done through existing dpnp and dpctl methods
    instead of separate function call of dpnp backend.

    """

    def _get_2dmin_array(x, dtype):
        """
        Transform an input array to a form required for building a covariance matrix.

        If applicable, it reshapes the input array to have 2 dimensions or greater.
        If applicable, it transposes the input array when 'rowvar' is False.
        It casts to another dtype, if the input array differs from requested one.

        """
        if x.ndim == 0:
            x = x.reshape((1, 1))
        elif x.ndim == 1:
            x = x[dpnp.newaxis, :]

        if not rowvar and x.shape[0] != 1:
            x = x.T

        if x.dtype != dtype:
            x = dpnp.astype(x, dtype)
        return x

    # input arrays must follow CFD paradigm
    _, queue = get_usm_allocations((m,) if y is None else (m, y))

    # calculate a type of result array if not passed explicitly
    if dtype is None:
        dtypes = [m.dtype, dpnp.default_float_type(sycl_queue=queue)]
        if y is not None:
            dtypes.append(y.dtype)
        dtype = dpnp.result_type(*dtypes)
        # TODO: remove when dpctl.result_type() is returned dtype based on fp64
        dtype = map_dtype_to_device(dtype, queue.sycl_device)

    X = _get_2dmin_array(m, dtype)
    if y is not None:
        y = _get_2dmin_array(y, dtype)

        X = dpnp.concatenate((X, y), axis=0)

    avg = X.mean(axis=1)

    fact = X.shape[1] - 1
    X -= avg[:, None]

    c = dpnp.dot(X, X.T.conj())
    c *= 1 / fact if fact != 0 else dpnp.nan

    return dpnp.squeeze(c)


def dpnp_nanmedian(
    a, keepdims=False, axis=None, out=None, overwrite_input=False
):
    """Internal Function."""

    nd = a.ndim
    if axis is not None:
        _axis = normalize_axis_tuple(axis, nd)

        if keepdims:
            if out is not None:
                index_out = tuple(
                    0 if i in _axis else slice(None) for i in range(nd)
                )
                out = out[(Ellipsis,) + index_out]

        if len(_axis) == 1:
            axis = _axis[0]
        else:
            keep = set(range(nd)) - set(_axis)
            nkeep = len(keep)
            # swap axis that should not be reduced to front
            for i, s in enumerate(sorted(keep)):
                a = dpnp.swapaxes(a, i, s)
            # merge reduced axis
            a = a.reshape(a.shape[:nkeep] + (-1,))
            axis = -1
    else:
        if keepdims:
            if out is not None:
                index_out = (0,) * nd
                out = out[(Ellipsis,) + index_out]

    r = _calc_nanmedian(a, axis=axis, out=out, overwrite_input=overwrite_input)

    if out is not None:
        return out

    if keepdims:
        if axis is None:
            index_r = (dpnp.newaxis,) * nd
        else:
            index_r = tuple(
                dpnp.newaxis if i in _axis else slice(None) for i in range(nd)
            )
        r = r[(Ellipsis,) + index_r]

    return r
