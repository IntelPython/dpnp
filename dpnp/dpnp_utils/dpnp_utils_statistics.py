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


import dpnp
from dpnp.dpnp_utils import get_usm_allocations

__all__ = ["dpnp_cov"]


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
        fp64 = queue.sycl_device.has_aspect_fp64
        if not fp64:
            if dtype == dpnp.float64:
                dtype = dpnp.float32
            elif dtype == dpnp.complex128:
                dtype = dpnp.complex64

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
