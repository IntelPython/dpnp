import math
import operator

import dpctl.utils as dpu
import numpy

import dpnp
import dpnp.dpnp_container as dpnp_container
import dpnp.dpnp_utils as utils

__all__ = [
    "dpnp_geomspace",
    "dpnp_linspace",
    "dpnp_logspace",
    "dpnp_nd_grid",
]


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
    usm_type_alloc, sycl_queue_alloc = utils.get_usm_allocations([start, stop])

    if sycl_queue is None and device is None:
        sycl_queue = sycl_queue_alloc
    sycl_queue_normalized = dpnp.get_normalized_queue_device(
        sycl_queue=sycl_queue, device=device
    )

    if usm_type is None:
        _usm_type = "device" if usm_type_alloc is None else usm_type_alloc
    else:
        _usm_type = usm_type

    if not dpnp.is_supported_array_type(start):
        start = dpnp.asarray(
            start, usm_type=_usm_type, sycl_queue=sycl_queue_normalized
        )
    if not dpnp.is_supported_array_type(stop):
        stop = dpnp.asarray(
            stop, usm_type=_usm_type, sycl_queue=sycl_queue_normalized
        )

    dt = numpy.result_type(start, stop, float(num))
    dt = utils.map_dtype_to_device(dt, sycl_queue_normalized.sycl_device)
    if dtype is None:
        dtype = dt

    if dpnp.any(start == 0) or dpnp.any(stop == 0):
        raise ValueError("Geometric sequence cannot include zero")

    out_sign = dpnp.ones(
        dpnp.broadcast_arrays(start, stop)[0].shape,
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

    both_negative = (dpnp.sign(start) == -1) & (dpnp.sign(stop) == -1)
    if dpnp.any(both_negative):
        dpnp.negative(start[both_negative], out=start[both_negative])
        dpnp.negative(stop[both_negative], out=stop[both_negative])
        dpnp.negative(out_sign[both_negative], out=out_sign[both_negative])

    log_start = dpnp.log10(start)
    log_stop = dpnp.log10(stop)
    result = dpnp_logspace(
        log_start,
        log_stop,
        num=num,
        endpoint=endpoint,
        base=10.0,
        dtype=dtype,
        usm_type=_usm_type,
        sycl_queue=sycl_queue_normalized,
    )

    if num > 0:
        result[0] = start
        if num > 1 and endpoint:
            result[-1] = stop

    result = out_sign * result

    if axis != 0:
        result = dpnp.moveaxis(result, 0, axis)

    return result.astype(dtype, copy=False)


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
    usm_type_alloc, sycl_queue_alloc = utils.get_usm_allocations([start, stop])

    if sycl_queue is None and device is None:
        sycl_queue = sycl_queue_alloc
    sycl_queue_normalized = dpnp.get_normalized_queue_device(
        sycl_queue=sycl_queue, device=device
    )

    if usm_type is None:
        _usm_type = "device" if usm_type_alloc is None else usm_type_alloc
    else:
        _usm_type = usm_type

    if not hasattr(start, "dtype") and not dpnp.isscalar(start):
        start = dpnp.asarray(
            start, usm_type=_usm_type, sycl_queue=sycl_queue_normalized
        )
    if not hasattr(stop, "dtype") and not dpnp.isscalar(stop):
        stop = dpnp.asarray(
            stop, usm_type=_usm_type, sycl_queue=sycl_queue_normalized
        )

    dt = numpy.result_type(start, stop, float(num))
    dt = utils.map_dtype_to_device(dt, sycl_queue_normalized.sycl_device)
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
        res = dpnp_container.linspace(
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
        _start = dpnp.asarray(
            start,
            dtype=dt,
            usm_type=_usm_type,
            sycl_queue=sycl_queue_normalized,
        )
        _stop = dpnp.asarray(
            stop, dtype=dt, usm_type=_usm_type, sycl_queue=sycl_queue_normalized
        )

        res = dpnp_container.arange(
            0,
            stop=num,
            step=1,
            dtype=dt,
            usm_type=_usm_type,
            sycl_queue=sycl_queue_normalized,
        )

        if step_nan is False:
            step = (_stop - _start) / step_num
            res = res.reshape((-1,) + (1,) * step.ndim)
            res = res * step + _start

        if endpoint and num > 1:
            res[-1] = dpnp_container.full(step.shape, _stop)

    if axis != 0:
        res = dpnp.moveaxis(res, 0, axis)

    if numpy.issubdtype(dtype, dpnp.integer):
        dpnp.floor(res, out=res)

    res = res.astype(dtype, copy=False)

    if retstep is True:
        if dpnp.isscalar(step):
            step = dpnp.asarray(
                step, usm_type=res.usm_type, sycl_queue=res.sycl_queue
            )
        return (res, step)

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
        usm_type_alloc, sycl_queue_alloc = utils.get_usm_allocations(
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
        start = dpnp.asarray(start, usm_type=usm_type, sycl_queue=sycl_queue)
        stop = dpnp.asarray(stop, usm_type=usm_type, sycl_queue=sycl_queue)
        base = dpnp.asarray(base, usm_type=usm_type, sycl_queue=sycl_queue)
        [start, stop, base] = dpnp.broadcast_arrays(start, stop, base)
        base = dpnp.expand_dims(base, axis=axis)

    res = dpnp_linspace(
        start,
        stop,
        num=num,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
        endpoint=endpoint,
        axis=axis,
    )

    if dtype is None:
        return dpnp.power(base, res)
    return dpnp.power(base, res).astype(dtype, copy=False)


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
