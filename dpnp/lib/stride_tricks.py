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

"""Utilities that manipulate strides to achieve desirable effects."""

import dpnp

__all__ = ["as_strided"]


def as_strided(
    x,
    shape=None,
    strides=None,
    subok=False,
    writeable=True,
    *,
    check_bounds=None,
):
    """
    Create a view into the array with the given shape and strides.

    For full documentation refer to :obj:`numpy.lib.stride_tricks.as_strided`.

    Warnings
    --------
    This function has to be used with extreme care, see notes.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        Array to create a new view from.
    shape : {None, sequence of ints}, optional
        The shape of the new array.

        Default: ``x.shape``.
    strides : {None, sequence of ints}, optional
        The strides of the new array, expressed in bytes.

        Default: ``x.strides``.
    writeable : bool, optional
        If set to ``False``, the returned array will always be read-only.
        Otherwise it will be writable if the original array was.

        Default: ``True``.
    check_bounds : {None, bool}, optional
        Ignored as no effect, the underlying USM array cannot be constructed
        over out-of-bounds memory.

        Default: ``None``.

    Returns
    -------
    view : dpnp.ndarray
        A view into the memory of `x` with the requested `shape` and `strides`,
        sharing the same data.

    Limitations
    -----------
    Parameter `subok` is supported with default value.
    Otherwise ``NotImplementedError`` exception will be raised.

    See Also
    --------
    :obj:`dpnp.broadcast_to` : Broadcast an array to a given shape.
    :obj:`dpnp.reshape` : Give a new shape to an array without changing its
        data.

    Notes
    -----
    :obj:`dpnp.lib.stride_tricks.as_strided` creates a view into the array
    given the exact strides and shape. This means it manipulates the internal
    data structure of the array and, if done incorrectly, the array elements
    can point to the wrong data and silently produce incorrect results. It is
    advisable to always use the original ``x.strides`` when calculating new
    strides to avoid reliance on a contiguous memory layout.

    Furthermore, arrays created with this function often contain self
    overlapping memory, so that two elements are identical. Writing to a shared
    element then changes every position that references it, so element-wise
    write operations on such arrays are typically unpredictable. A bulk write
    over an overlapping view is rejected, because it would address more memory
    than the base allocation holds.

    Since writing to these arrays has to be tested and done with great care,
    you may want to use ``writeable=False`` to avoid accidental write
    operations.

    For these reasons it is advisable to avoid
    :obj:`dpnp.lib.stride_tricks.as_strided` when possible.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([1, 2, 3, 4], dtype=np.int32)

    Downsample the array by taking every second element:

    >>> np.lib.stride_tricks.as_strided(x, shape=(2,),
    ...                                 strides=(2 * x.itemsize,))
    array([1, 3], dtype=int32)

    Broadcast the array along a new leading axis using a zero stride:

    >>> np.lib.stride_tricks.as_strided(x, shape=(3, 4),
    ...                                 strides=(0, x.itemsize))
    array([[1, 2, 3, 4],
           [1, 2, 3, 4],
           [1, 2, 3, 4]], dtype=int32)

    Build a self-overlapping sliding-window view, where a single element maps
    onto several positions. Here a length-5 array yields a ``3x3`` window in
    which each value repeats along the anti-diagonals:

    >>> y = np.arange(5, dtype=np.int32)
    >>> np.lib.stride_tricks.as_strided(y, shape=(3, 3),
    ...                                 strides=(y.itemsize, y.itemsize))
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4]], dtype=int32)

    Attempting to create an out-of-bounds view:

    >>> np.lib.stride_tricks.as_strided(y, shape=(10,),
    ...                                 strides=(y.itemsize,))
    Traceback (most recent call last):
    ...
    ValueError: buffer='[0 1 2 3 4]' can not accommodate the requested array.

    """

    dpnp.check_supported_arrays_type(x)
    dpnp.check_limitations(subok=subok)

    shape = x.shape if shape is None else tuple(shape)
    strides = x.strides if strides is None else tuple(strides)

    view = dpnp.ndarray(
        shape,
        dtype=x.dtype,
        buffer=x,
        strides=strides,
    )

    if view.flags.writable and not writeable:
        view.flags.writable = False
    return view
