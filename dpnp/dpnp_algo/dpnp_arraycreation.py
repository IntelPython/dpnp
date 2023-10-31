import operator

import numpy

import dpnp
import dpnp.dpnp_container as dpnp_container
import dpnp.dpnp_utils as utils

__all__ = [
    "dpnp_geomspace",
    "dpnp_linspace",
    "dpnp_logspace",
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

    if not hasattr(start, "dtype"):
        start = dpnp.asarray(
            start, usm_type=_usm_type, sycl_queue=sycl_queue_normalized
        )
    if not hasattr(stop, "dtype"):
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
