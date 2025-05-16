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
    if isinstance(a, dpnp_array):
        return a.get_array()
    return dpt.asarray(a, usm_type=usm_type, sycl_queue=sycl_queue)


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

    start = _as_usm_ndarray(start, _usm_type, sycl_queue_normalized)
    stop = _as_usm_ndarray(stop, _usm_type, sycl_queue_normalized)

    dt = numpy.result_type(start, stop, float(num))
    dt = map_dtype_to_device(dt, sycl_queue_normalized.sycl_device)
    if dtype is None:
        dtype = dt

    if dpnp.any(start == 0) or dpnp.any(stop == 0):
        raise ValueError("Geometric sequence cannot include zero")

    out_sign = dpt.ones(
        dpt.broadcast_arrays(start, stop)[0].shape,
        dtype=dt,
        usm_type=_usm_type,
        sycl_queue=sycl_queue_normalized,
    )
    # Avoid negligible real or imaginary parts in output by rotating to
    # positive real, calculating, then undoing rotation
    if dpnp.issubdtype(dt, dpnp.complexfloating):
        all_imag = (start.real == 0.0) & (stop.real == 0.0)
        if dpnp.any(all_imag):
            start[all_imag] = start[all_imag].imag
            stop[all_imag] = stop[all_imag].imag
            out_sign[all_imag] = 1j

    both_negative = (dpt.sign(start) == -1) & (dpt.sign(stop) == -1)
    if dpnp.any(both_negative):
        dpt.negative(start[both_negative], out=start[both_negative])
        dpt.negative(stop[both_negative], out=stop[both_negative])
        dpt.negative(out_sign[both_negative], out=out_sign[both_negative])

    log_start = dpt.log10(start)
    log_stop = dpt.log10(stop)
    res = dpnp_logspace(
        log_start,
        log_stop,
        num=num,
        endpoint=endpoint,
        base=10.0,
        dtype=dtype,
        usm_type=_usm_type,
        sycl_queue=sycl_queue_normalized,
    ).get_array()

    if num > 0:
        res[0] = start
        if num > 1 and endpoint:
            res[-1] = stop

    res = out_sign * res

    if axis != 0:
        res = dpt.moveaxis(res, 0, axis)

    res = dpt.astype(res, dtype, copy=False)
    return dpnp_array._create_from_usm_ndarray(res)


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
        raise ValueError("Number of points must be non-negative")
    step_num = (num - 1) if endpoint else num

    step_nan = False
    if step_num == 0:
        step_nan = True
        step = dpnp.nan

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
        if retstep is True and step_nan is False:
            step = (stop - start) / step_num
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

        usm_res = dpt.arange(
            0,
            stop=num,
            step=1,
            dtype=dt,
            usm_type=_usm_type,
            sycl_queue=sycl_queue_normalized,
        )

        if step_nan is False:
            step = (usm_stop - usm_start) / step_num
            usm_res = dpt.reshape(usm_res, (-1,) + (1,) * step.ndim, copy=False)
            usm_res = usm_res * step
            usm_res += usm_start

        if endpoint and num > 1:
            usm_res[-1] = dpt.full(step.shape, usm_stop)

    if axis != 0:
        usm_res = dpt.moveaxis(usm_res, 0, axis)

    if numpy.issubdtype(dtype, dpnp.integer):
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
    if not dpnp.isscalar(base):
        usm_type_alloc, sycl_queue_alloc = get_usm_allocations(
            [start, stop, base]
        )

        if sycl_queue is None and device is None:
            sycl_queue = sycl_queue_alloc
        sycl_queue = dpnp.get_normalized_queue_device(
            sycl_queue=sycl_queue, device=device
        )

        if usm_type is None:
            usm_type = "device" if usm_type_alloc is None else usm_type_alloc
        else:
            usm_type = usm_type

        start = _as_usm_ndarray(start, usm_type, sycl_queue)
        stop = _as_usm_ndarray(stop, usm_type, sycl_queue)
        base = _as_usm_ndarray(base, usm_type, sycl_queue)

        [start, stop, base] = dpt.broadcast_arrays(start, stop, base)
        base = dpt.expand_dims(base, axis=axis)

    # assume res as not a tuple, because retstep is False
    res = dpnp_linspace(
        start,
        stop,
        num=num,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
        endpoint=endpoint,
        axis=axis,
    ).get_array()

    dpt.pow(base, res, out=res)
    if dtype is not None:
        res = dpt.astype(res, dtype, copy=False)
    return dpnp_array._create_from_usm_ndarray(res)


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
