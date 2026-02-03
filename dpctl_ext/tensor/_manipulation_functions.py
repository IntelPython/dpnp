# *****************************************************************************
# Copyright (c) 2026, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
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

# TODO: revert to `import dpctl.tensor as dpt`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt

from ._numpy_helper import normalize_axis_tuple

__doc__ = (
    "Implementation module for array manipulation "
    "functions in :module:`dpctl.tensor`"
)


def permute_dims(X, /, axes):
    """permute_dims(x, axes)

    Permute the axes (dimensions) of an array; returns the permuted
    array as a view.

    Args:
        x (usm_ndarray): input array.
        axes (Tuple[int, ...]): tuple containing permutation of
           `(0,1,...,N-1)` where `N` is the number of axes (dimensions)
           of `x`.
    Returns:
        usm_ndarray:
            An array with permuted axes.
            The returned array must has the same data type as `x`,
            is created on the same device as `x` and has the same USM allocation
            type as `x`.
    """
    if not isinstance(X, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray type, got {type(X)}.")

    axes = normalize_axis_tuple(axes, X.ndim, "axes")
    if not X.ndim == len(axes):
        raise ValueError(
            "The length of the passed axes does not match "
            "to the number of usm_ndarray dimensions."
        )

    newstrides = tuple(X.strides[i] for i in axes)
    newshape = tuple(X.shape[i] for i in axes)

    return dpt.usm_ndarray(
        shape=newshape,
        dtype=X.dtype,
        buffer=X,
        strides=newstrides,
        offset=X._element_offset,
    )
