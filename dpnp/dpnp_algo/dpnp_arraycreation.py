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

import math
import operator

import dpctl.tensor as dpt
import dpctl.utils as dpu
import numpy

import dpnp
from dpnp.dpnp_array import dpnp_array
from dpnp.dpnp_utils import get_usm_allocations, map_dtype_to_device

__all__ = [
    "dpnp_geomspace",
    "dpnp_linspace",
    "dpnp_logspace",
    "dpnp_nd_grid",
]


def _as_usm_ndarray(a, usm_type, sycl_queue):
    """Converts input object to `dpctl.tensor.usm_ndarray`"""

    if isinstance(a, dpnp_array):
        a = a.get_array()
    return dpt.asarray(a, usm_type=usm_type, sycl_queue=sycl_queue)


def _check_has_zero_val(a):
    """Check if any element in input object is equal to zero"""

    if dpnp.isscalar(a):
        if a == 0:
            return True
    elif hasattr(a, "any"):
        if (a == 0).any():
            return True
    elif (numpy.array(a) == 0).any():
        return True
    return False


def _get_usm_allocations(objs, device=None, usm_type=None, sycl_queue=None):
    """
    Get common USM allocations based on a list of input objects and an explicit
    device, a SYCL queue, or a USM type if specified.

    """

    alloc_usm_type, alloc_sycl_queue = get_usm_allocations(objs)

    if sycl_queue is None and device is None:
        sycl_queue = alloc_sycl_queue

    if usm_type is None:
        usm_type = alloc_usm_type or "device"
    return usm_type, dpnp.get_normalized_queue_device(
        sycl_queue=sycl_queue, device=device
    )


def dpnp_geomspace(
    start,
    stop,
    num,
    dtype=None,
    device=None,
    usm_type=None,
    sycl_queue=None,
    endpoint=True,
    axis=0,
):
    usm_type, sycl_queue = _get_usm_allocations(
        [start, stop], device=device, usm_type=usm_type, sycl_queue=sycl_queue
    )

    if _check_has_zero_val(start) or _check_has_zero_val(stop):
        raise ValueError("Geometric sequence cannot include zero")

    start = dpnp.array(start, usm_type=usm_type, sycl_queue=sycl_queue)
    stop = dpnp.array(stop, usm_type=usm_type, sycl_queue=sycl_queue)

    dt = numpy.result_type(start, stop, float(num))
    dt = map_dtype_to_device(dt, sycl_queue.sycl_device)
    if dtype is None:
        dtype = dt

    # promote both arguments to the same dtype
    start = start.astype(dt, copy=False)
    stop = stop.astype(dt, copy=False)

    # Allow negative real values and ensure a consistent result for complex
    # (including avoiding negligible real or imaginary parts in output) by
    # rotating start to positive real, calculating, then undoing rotation.
    out_sign = dpnp.sign(start)
    start = start / out_sign
    stop = stop / out_sign

    log_start = dpnp.log10(start)
    log_stop = dpnp.log10(stop)
    res = dpnp_logspace(
        log_start,
        log_stop,
        num=num,
        endpoint=endpoint,
        base=10.0,
        dtype=dt,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )

    # Make sure the endpoints match the start and stop arguments. This is
    # necessary because np.exp(np.log(x)) is not necessarily equal to x.
    if num > 0:
        res[0] = start
        if num > 1 and endpoint:
            res[-1] = stop

    res *= out_sign

    if axis != 0:
        res = dpnp.moveaxis(res, 0, axis)
    return res.astype(dtype, copy=False)


def dpnp_linspace(
    start,
    stop,
    num,
    dtype=None,
    device=None,
    usm_type=None,
    sycl_queue=None,
    endpoint=True,
    retstep=False,
    axis=0,
):
    usm_type_alloc, sycl_queue_alloc = get_usm_allocations([start, stop])

    if sycl_queue is None and device is None:
        sycl_queue = sycl_queue_alloc
    sycl_queue_normalized = dpnp.get_normalized_queue_device(
        sycl_queue=sycl_queue, device=device
    )

    if usm_type is None:
        _usm_type = "device" if usm_type_alloc is None else usm_type_alloc
    else:
        _usm_type = usm_type

    if not dpnp.isscalar(start):
        start = _as_usm_ndarray(start, _usm_type, sycl_queue_normalized)

    if not dpnp.isscalar(stop):
        stop = _as_usm_ndarray(stop, _usm_type, sycl_queue_normalized)

    dt = numpy.result_type(start, stop, float(num))
    dt = map_dtype_to_device(dt, sycl_queue_normalized.sycl_device)
    if dtype is None:
        dtype = dt

    num = operator.index(num)
    if num < 0:
        raise ValueError(f"Number of samples={num} must be non-negative.")
    step_num = (num - 1) if endpoint else num

    if dpnp.isscalar(start) and dpnp.isscalar(stop):
        # Call linspace() function for scalars.
        usm_res = dpt.linspace(
            start,
            stop,
            num,
            dtype=dt,
            usm_type=_usm_type,
            sycl_queue=sycl_queue_normalized,
            endpoint=endpoint,
        )

        # calculate the used step to return
        if retstep is True:
            if step_num > 0:
                step = (stop - start) / step_num
            else:
                step = dpnp.nan
    else:
        usm_start = dpt.asarray(
            start,
            dtype=dt,
            usm_type=_usm_type,
            sycl_queue=sycl_queue_normalized,
        )
        usm_stop = dpt.asarray(
            stop, dtype=dt, usm_type=_usm_type, sycl_queue=sycl_queue_normalized
        )

        delta = usm_stop - usm_start

        usm_res = dpt.arange(
            0,
            stop=num,
            step=1,
            dtype=dt,
            usm_type=_usm_type,
            sycl_queue=sycl_queue_normalized,
        )
        usm_res = dpt.reshape(usm_res, (-1,) + (1,) * delta.ndim, copy=False)

        if step_num > 0:
            step = delta / step_num

            # Needed a special handling for denormal numbers (when step == 0),
            # see numpy#5437 for more details.
            # Note, dpt.where() is used to avoid a synchronization branch.
            usm_res = dpt.where(
                step == 0, (usm_res / step_num) * delta, usm_res * step
            )
        else:
            step = dpnp.nan
            usm_res = usm_res * delta

        usm_res += usm_start

        if endpoint and num > 1:
            usm_res[-1, ...] = usm_stop

    if axis != 0:
        usm_res = dpt.moveaxis(usm_res, 0, axis)

    if dpnp.issubdtype(dtype, dpnp.integer):
        dpt.floor(usm_res, out=usm_res)

    res = dpt.astype(usm_res, dtype, copy=False)
    res = dpnp_array._create_from_usm_ndarray(res)

    if retstep is True:
        if dpnp.isscalar(step):
            step = dpt.asarray(
                step, usm_type=res.usm_type, sycl_queue=res.sycl_queue
            )
        return res, dpnp_array._create_from_usm_ndarray(step)
    return res


def dpnp_logspace(
    start,
    stop,
    num=50,
    device=None,
    usm_type=None,
    sycl_queue=None,
    endpoint=True,
    base=10.0,
    dtype=None,
    axis=0,
):
    usm_type, sycl_queue = _get_usm_allocations(
        [start, stop, base],
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )

    if not dpnp.isscalar(base):
        base = dpnp.array(base, usm_type=usm_type, sycl_queue=sycl_queue)
        start = dpnp.array(start, usm_type=usm_type, sycl_queue=sycl_queue)
        stop = dpnp.array(stop, usm_type=usm_type, sycl_queue=sycl_queue)

        start, stop, base = dpnp.broadcast_arrays(start, stop, base)
        base = dpnp.expand_dims(base, axis=axis)

    # assume `res` as not a tuple, because retstep is False
    res = dpnp_linspace(
        start,
        stop,
        num=num,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
        endpoint=endpoint,
        axis=axis,
    )

    dpnp.pow(base, res, out=res)
    if dtype is not None:
        res = res.astype(dtype, copy=False)
    return res


class dpnp_nd_grid:
    """
    Construct a multi-dimensional "meshgrid".

    ``grid = dpnp_nd_grid()`` creates an instance which will return a mesh-grid
    when indexed. The dimension and number of the output arrays are equal
    to the number of indexing dimensions.  If the step length is not a
    complex number, then the stop is not inclusive.

    However, if the step length is a complex number (e.g. 5j), then the
    integer part of its magnitude is interpreted as specifying the
    number of points to create between the start and stop values, where
    the stop value is inclusive.

    If instantiated with an argument of ``sparse=True``, the mesh-grid is
    open (or not fleshed out) so that only one-dimension of each returned
    argument is greater than 1.

    Parameters
    ----------
    sparse : {bool}, optional
        Whether the grid is sparse or not. Default is False.

    """

    def __init__(
        self, sparse=False, device=None, usm_type="device", sycl_queue=None
    ):
        dpu.validate_usm_type(usm_type, allow_none=True)
        self.sparse = sparse
        self.usm_type = "device" if usm_type is None else usm_type
        self.sycl_queue_normalized = dpnp.get_normalized_queue_device(
            sycl_queue=sycl_queue, device=device
        )

    def __getitem__(self, key):
        if isinstance(key, slice):
            step = key.step
            stop = key.stop
            start = key.start
            if start is None:
                start = 0
            if isinstance(step, complex):
                step = abs(step)
                length = int(step)
                if step != 1:
                    step = (stop - start) / float(step - 1)
                stop = stop + step
                return (
                    dpnp.arange(
                        0,
                        length,
                        1,
                        dtype=dpnp.default_float_type(),
                        usm_type=self.usm_type,
                        sycl_queue=self.sycl_queue_normalized,
                    )
                    * step
                    + start
                )
            else:
                return dpnp.arange(
                    start,
                    stop,
                    step,
                    usm_type=self.usm_type,
                    sycl_queue=self.sycl_queue_normalized,
                )

        size = []
        dtype = int
        for k in range(len(key)):
            step = key[k].step
            start = key[k].start
            stop = key[k].stop
            if start is None:
                start = 0
            if step is None:
                step = 1
            if isinstance(step, complex):
                size.append(int(abs(step)))
                dtype = dpnp.default_float_type()
            else:
                size.append(
                    int(math.ceil((key[k].stop - start) / (step * 1.0)))
                )
            if (
                isinstance(step, float)
                or isinstance(start, float)
                or isinstance(stop, float)
            ):
                dtype = dpnp.default_float_type()
        if self.sparse:
            nn = [
                dpnp.arange(
                    _x,
                    dtype=_t,
                    usm_type=self.usm_type,
                    sycl_queue=self.sycl_queue_normalized,
                )
                for _x, _t in zip(size, (dtype,) * len(size))
            ]
        else:
            nn = dpnp.indices(
                size,
                dtype,
                usm_type=self.usm_type,
                sycl_queue=self.sycl_queue_normalized,
            )
        for k in range(len(size)):
            step = key[k].step
            start = key[k].start
            stop = key[k].stop
            if start is None:
                start = 0
            if step is None:
                step = 1
            if isinstance(step, complex):
                step = int(abs(step))
                if step != 1:
                    step = (stop - start) / float(step - 1)
            nn[k] = nn[k] * step + start
        if self.sparse:
            slobj = [dpnp.newaxis] * len(size)
            for k in range(len(size)):
                slobj[k] = slice(None, None)
                nn[k] = nn[k][tuple(slobj)]
                slobj[k] = dpnp.newaxis
        return nn
