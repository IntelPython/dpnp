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
    """Flat iterator object to iterate over arrays."""

    def __init__(self, a):
        if not isinstance(a, dpnp.ndarray):
            raise TypeError(
                f"An array must be of type dpnp.ndarray, but got {type(a)}"
            )
        self._arr = a
        self._size = a.size
        self._i = 0

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
        self._reject_newaxis(key)
        self._check_bounds(key)

        # flat always yields a copy, never a view
        return self._flatten()[key].copy()

    def __setitem__(self, key, val):
        self._reject_newaxis(key)
        self._check_bounds(key)

        if isinstance(key, tuple) and len(key) == 0:
            # NumPy rejects arr.flat[()] = val
            raise IndexError(
                "Assigning to a flat iterator with a 0-D index is not "
                "supported"
            )

        # resolve key to flat positions, reusing regular indexing to validate
        arr = self._arr
        flat_index = dpnp.arange(
            arr.size, sycl_queue=arr.sycl_queue, usm_type=arr.usm_type
        )
        positions = dpnp.reshape(flat_index[key], -1)
        dpnp.put(arr, positions, val)

    def __iter__(self):
        return self

    def __next__(self):
        if self._i < self._size:
            val = self.__getitem__(self._i)
            self._i = self._i + 1
            return val
        else:
            raise StopIteration
