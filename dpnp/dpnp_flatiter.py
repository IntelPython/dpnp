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

import numpy

import dpnp
import dpnp.tensor as dpt

from .dpnp_array import dpnp_array


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
        if not isinstance(a, dpnp_array):
            raise TypeError(
                f"An array must be of type dpnp.ndarray, but got {type(a)}"
            )
        self._arr = a
        self._size = a.size
        self._i = 0

    def _validate_key(self, key):
        # Ellipsis/slice/tuple need no validation here
        if key is Ellipsis or isinstance(key, (slice, tuple)):
            return

        # a genuine scalar int (not bool, not an array): bounds-checked later
        if (
            not isinstance(key, bool)
            and callable(getattr(key, "__index__", None))
            and not hasattr(key, "ndim")
        ):
            return

        if isinstance(key, dpnp_array):
            idx = key
        elif isinstance(key, dpt.usm_ndarray):
            idx = dpnp_array._create_from_usm_ndarray(key)
        else:
            try:
                idx = numpy.asarray(key)
            except (TypeError, ValueError):
                return  # let regular indexing raise

        if dpnp.issubdtype(idx.dtype, dpnp.bool):
            if idx.ndim > 1:
                raise IndexError(
                    "too many indices for flat iterator: flat iterator is "
                    f"1-dimensional, but {idx.ndim} were indexed"
                )

            # only a 1-D boolean ndarray mask is valid; reject scalars/lists
            if idx.ndim == 1 and not isinstance(key, list):
                # an empty mask selects nothing; otherwise sizes must match
                if idx.size not in (0, self._size):
                    raise IndexError(
                        "boolean index did not match indexed array along "
                        f"axis 0; size of axis is {self._size} but size of "
                        f"corresponding boolean axis is {idx.size}"
                    )
                return
            raise IndexError("boolean indices for iterators are not supported")

        if not dpnp.issubdtype(idx.dtype, dpnp.integer) or idx.size == 0:
            return

        # fancy int indices wrap instead of raising, so bounds-check
        size = self._size
        hi, lo = int(idx.max()), int(idx.min())
        if hi >= size:
            raise IndexError(f"index {hi} is out of bounds for size {size}")
        if lo < -size:
            raise IndexError(f"index {lo} is out of bounds for size {size}")

    def _normalize_key(self, key):
        # 1-D iterator: unwrap a 1-elem tuple; reject None and longer tuples
        if isinstance(key, tuple) and len(key) == 1:
            key = key[0]
        if key is None or (isinstance(key, tuple) and len(key) > 1):
            raise IndexError(
                "only integers, slices (`:`), ellipsis (`...`) and integer "
                "or boolean arrays are valid indices"
            )
        self._validate_key(key)
        return key

    def _scalar_pos(self, key):
        # normalize a scalar flat index (wrap negatives) and bounds-check it
        pos = key + self._size if key < 0 else key
        if not 0 <= pos < self._size:
            raise IndexError(
                f"index {key} is out of bounds for size {self._size}"
            )
        return pos

    def __getitem__(self, key):
        key = self._normalize_key(key)

        if isinstance(key, int) and not isinstance(key, bool):
            # scalar fast path: index directly instead of flattening the array
            pos = self._scalar_pos(key)
            return self._arr[numpy.unravel_index(pos, self._arr.shape)].copy()

        res = dpnp.reshape(self._arr, -1)[key]
        # advanced indexing is already fresh; copy only a basic-index view,
        # i.e. when res shares the source allocation. Compare the allocation
        # base `usm_data._pointer` (offset-independent), not the array-level
        # `_pointer` which is adjusted to the first element.
        # pylint: disable=protected-access
        src = self._arr.get_array().usm_data._pointer
        if res.get_array().usm_data._pointer == src:
            res = res.copy()
        return res

    def __setitem__(self, key, val):
        key = self._normalize_key(key)

        if isinstance(key, tuple) and len(key) == 0:
            # NumPy rejects arr.flat[()] = val
            raise IndexError(
                "Assigning to a flat iterator with a 0-D index is not "
                "supported"
            )

        a = self._arr
        exec_q = a.sycl_queue
        usm_type = a.usm_type

        # resolve key to flat positions
        if isinstance(key, int) and not isinstance(key, bool):
            # scalar fast path: avoid building a full index array
            pos = self._scalar_pos(key)
            idx = dpnp.asarray(pos, sycl_queue=exec_q, usm_type=usm_type)
        elif isinstance(key, slice):
            # slice fast path: build only the selected positions
            start, stop, step = key.indices(a.size)
            idx = dpnp.arange(
                start, stop, step, sycl_queue=exec_q, usm_type=usm_type
            )
        elif hasattr(key, "dtype") and dpnp.issubdtype(key.dtype, dpnp.bool):
            # boolean mask fast path
            mask = dpnp.asarray(key, sycl_queue=exec_q, usm_type=usm_type)
            idx = dpnp.nonzero(mask)[0]
        else:
            flat_index = dpnp.arange(
                a.size, sycl_queue=exec_q, usm_type=usm_type
            )
            idx = flat_index[key]

        if not dpnp.isscalar(val):
            val = dpnp.asarray(val, sycl_queue=exec_q, usm_type=usm_type)
            if idx.ndim == 0 and val.ndim != 0:
                # a scalar index targets a single item, reject an array value
                raise ValueError("Error setting single item of array.")

            val = val.ravel()
            n = idx.size
            if val.size and val.size != n:
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
