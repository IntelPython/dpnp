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


def _get_sycl_queue(sycl_queue_alloc, sycl_queue, device):
    if sycl_queue is None and device is None:
        sycl_queue = sycl_queue_alloc
    return dpnp.get_normalized_queue_device(
        sycl_queue=sycl_queue, device=device
    )


def _get_temporary_usm_type(usm_type_alloc, usm_type):
    if usm_type is None:
        usm_type = "device" if usm_type_alloc is None else usm_type_alloc
    return usm_type


def _list_to_array(a, usm_type, sycl_queue, scalar=False):
    if not hasattr(a, "dtype") and (scalar or not dpnp.isscalar(a)):
        return dpnp.asarray(a, usm_type=usm_type, sycl_queue=sycl_queue)
    return a


def _copy_by_usm_type(a, dtype, usm_type, tmp_usm_type, sycl_queue_normalized):
    if usm_type is None:
        usm_type = tmp_usm_type
        if not hasattr(a, "usm_type"):
            res = dpnp.asarray(
                a,
                dtype=dtype,
                usm_type=usm_type,
                sycl_queue=sycl_queue_normalized,
            )
        else:
            res = dpnp.asarray(a, dtype=dtype, sycl_queue=sycl_queue_normalized)
    else:
        res = dpnp.asarray(
            a, dtype=dtype, usm_type=usm_type, sycl_queue=sycl_queue_normalized
        )
    return res


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

    sycl_queue_normalized = _get_sycl_queue(
        sycl_queue_alloc, sycl_queue, device
    )

    _usm_type = _get_temporary_usm_type(usm_type_alloc, usm_type)

    start = _list_to_array(start, _usm_type, sycl_queue_normalized, scalar=True)
    stop = _list_to_array(stop, _usm_type, sycl_queue_normalized, scalar=True)

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
    axis=0,
):
    usm_type_alloc, sycl_queue_alloc = utils.get_usm_allocations([start, stop])

    sycl_queue_normalized = _get_sycl_queue(
        sycl_queue_alloc, sycl_queue, device
    )

    _usm_type = _get_temporary_usm_type(usm_type_alloc, usm_type)

    start = _list_to_array(start, _usm_type, sycl_queue_normalized)
    stop = _list_to_array(stop, _usm_type, sycl_queue_normalized)

    dt = numpy.result_type(start, stop, float(num))
    dt = utils.map_dtype_to_device(dt, sycl_queue_normalized.sycl_device)
    if dtype is None:
        dtype = dt

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
    else:
        num = operator.index(num)
        if num < 0:
            raise ValueError("Number of points must be non-negative")

        _start = _copy_by_usm_type(
            start, dt, usm_type, _usm_type, sycl_queue_normalized
        )
        _stop = _copy_by_usm_type(
            stop, dt, usm_type, _usm_type, sycl_queue_normalized
        )

        _num = (num - 1) if endpoint else num

        step = (_stop - _start) / _num

        res = dpnp_container.arange(
            0,
            stop=num,
            step=1,
            dtype=dt,
            usm_type=_usm_type,
            sycl_queue=sycl_queue_normalized,
        )

        res = res.reshape((-1,) + (1,) * step.ndim)
        res = res * step + _start

        if endpoint and num > 1:
            res[-1] = dpnp_container.full(step.shape, _stop)

    if axis != 0:
        res = dpnp.moveaxis(res, 0, axis)

    if numpy.issubdtype(dtype, dpnp.integer):
        dpnp.floor(res, out=res)

    return res.astype(dtype, copy=False)


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
    usm_type_alloc, sycl_queue_alloc = utils.get_usm_allocations(
        [start, stop, base]
    )

    sycl_queue_normalized = _get_sycl_queue(
        sycl_queue_alloc, sycl_queue, device
    )

    _usm_type = _get_temporary_usm_type(usm_type_alloc, usm_type)

    start = _list_to_array(start, _usm_type, sycl_queue_normalized)
    stop = _list_to_array(stop, _usm_type, sycl_queue_normalized)
    base = _list_to_array(base, _usm_type, sycl_queue_normalized)

    dt = numpy.result_type(start, stop, float(num))
    dt = utils.map_dtype_to_device(dt, sycl_queue_normalized.sycl_device)
    if dtype is None:
        dtype = dt

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

    else:
        _start = _copy_by_usm_type(
            start, dt, usm_type, _usm_type, sycl_queue_normalized
        )
        _stop = _copy_by_usm_type(
            stop, dt, usm_type, _usm_type, sycl_queue_normalized
        )
        res = dpnp_linspace(
            _start,
            _stop,
            num=num,
            usm_type=_usm_type,
            sycl_queue=sycl_queue_normalized,
            endpoint=endpoint,
            axis=axis,
        )

    _base = _copy_by_usm_type(
        base, dt, usm_type, _usm_type, sycl_queue_normalized
    )

    print(res.shape)
    if dtype is None:
        return dpnp.power(_base, res)
    return dpnp.power(_base, res).astype(dtype, copy=False)
