# *****************************************************************************
# Copyright (c) 2016, Intel Corporation
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

"""Implementation of flatiter."""

import dpnp


class flatiter:
    """
    Flat iterator object to iterate over arrays.

    A flat iterator is returned by :obj:`dpnp.ndarray.flat` for any array. It
    allows iterating over the array as if it were a 1-D array, either in a
    for-loop or by calling its ``next`` method.

    Iteration is done in row-major, C-style order (the last index varying the
    fastest). The iterator can also be indexed using basic slicing or advanced
    indexing.

    For full documentation refer to :obj:`numpy.flatiter`.

    See Also
    --------
    :obj:`dpnp.ndarray.flat` : Return a flat iterator over an array.
    :obj:`dpnp.ndarray.flatten` : Return a flattened copy of an array.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.arange(6).reshape(2, 3)
    >>> for item in x.flat:
    ...     print(item)
    0
    1
    2
    3
    4
    5

    >>> x.flat[2:4]
    array([2, 3])

    """

    def __init__(self, a):
        if not isinstance(a, dpnp.ndarray):
            raise TypeError(
                f"An array must be of type dpnp.ndarray, but got {type(a)}"
            )
        self._arr = a
        self._size = a.size
        self._i = 0

    @staticmethod
    def _unwrap_tuple(key):
        # a flat iterator is 1-D, so a single-element index tuple is equivalent
        # to its element (e.g. `flat[(idx,)]` behaves like `flat[idx]`)
        if isinstance(key, tuple) and len(key) == 1:
            return key[0]
        return key

    @staticmethod
    def _reject_newaxis(key):
        # newaxis (None) is valid for array indexing but not for flat indexing
        if key is None or (
            isinstance(key, tuple) and any(k is None for k in key)
        ):
            raise IndexError(
                "only integers, slices (`:`), ellipsis (`...`) and integer "
                "or boolean arrays are valid indices"
            )

    def _check_bounds(self, key):
        # fancy int indices wrap instead of raising, so check them vs NumPy
        if key is Ellipsis or isinstance(key, (slice, bool, tuple)):
            return

        if isinstance(key, int) or (
            callable(getattr(key, "__index__", None))
            and not hasattr(key, "ndim")
        ):
            return  # scalar int: regular indexing checks it

        try:
            idx = dpnp.asarray(key, sycl_queue=self._arr.sycl_queue)
        except Exception:
            return  # let regular indexing raise

        if idx.dtype.kind not in "iu" or idx.size == 0:
            return

        size = self._size
        hi, lo = int(dpnp.max(idx)), int(dpnp.min(idx))
        if hi >= size:
            raise IndexError(f"index {hi} is out of bounds for size {size}")
        if lo < -size:
            raise IndexError(f"index {lo} is out of bounds for size {size}")

    def _flatten(self):
        # C-order flat view (copy if non-contiguous)
        return dpnp.reshape(self._arr, -1)

    def __getitem__(self, key):
        key = self._unwrap_tuple(key)
        self._reject_newaxis(key)
        self._check_bounds(key)

        # flat always yields a copy, never a view
        return self._flatten()[key].copy()

    def __setitem__(self, key, val):
        key = self._unwrap_tuple(key)
        self._reject_newaxis(key)
        self._check_bounds(key)

        if isinstance(key, tuple) and len(key) == 0:
            # NumPy rejects arr.flat[()] = val
            raise IndexError(
                "Assigning to a flat iterator with a 0-D index is not "
                "supported"
            )

        a = self._arr
        exec_q = a.sycl_queue
        usm_type = a.usm_type

        # resolve key to flat positions, reusing regular indexing to validate
        if isinstance(key, int) and not isinstance(key, bool):
            # fast path for a scalar index: avoid building a full index array
            pos = key + a.size if key < 0 else key
            if not 0 <= pos < a.size:
                raise IndexError(
                    f"index {key} is out of bounds for size {a.size}"
                )
            idx = dpnp.asarray(pos, sycl_queue=exec_q, usm_type=usm_type)
        elif isinstance(key, slice):
            # slice fast path: build only the selected positions
            start, stop, step = key.indices(a.size)
            idx = dpnp.arange(
                start, stop, step, sycl_queue=exec_q, usm_type=usm_type
            )
        else:
            flat_index = dpnp.arange(
                a.size, sycl_queue=exec_q, usm_type=usm_type
            )
            idx = flat_index[key]

        if not dpnp.isscalar(val):
            val = dpnp.asarray(
                val, sycl_queue=exec_q, usm_type=usm_type
            ).ravel()
            n = idx.size
            if 0 < val.size != n:
                # cycles the values over the selection
                val = val[
                    dpnp.arange(n, sycl_queue=exec_q, usm_type=usm_type)
                    % val.size
                ]

        dpnp.put(a, idx, val)

    def __iter__(self):
        return self

    def __next__(self):
        if self._i < self._size:
            val = self.__getitem__(self._i)
            self._i = self._i + 1
            return val
        else:
            raise StopIteration
