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

import operator
from numbers import Number

import dpctl
import dpctl.memory as dpm
import dpctl.tensor as dpt
import dpctl.utils
import numpy as np
from dpctl.tensor._data_types import _get_dtype
from dpctl.tensor._device import normalize_queue_device
from dpctl.tensor._usmarray import _is_object_with_buffer_protocol

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt_ext
import dpctl_ext.tensor._tensor_impl as ti

__doc__ = "Implementation of creation functions in :module:`dpctl.tensor`"

_empty_tuple = ()
_host_set = frozenset([None])


def _array_info_dispatch(obj):
    if isinstance(obj, dpt.usm_ndarray):
        return obj.shape, obj.dtype, frozenset([obj.sycl_queue])
    if isinstance(obj, np.ndarray):
        return obj.shape, obj.dtype, _host_set
    if isinstance(obj, range):
        return (len(obj),), int, _host_set
    if isinstance(obj, bool):
        return _empty_tuple, bool, _host_set
    if isinstance(obj, float):
        return _empty_tuple, float, _host_set
    if isinstance(obj, int):
        return _empty_tuple, int, _host_set
    if isinstance(obj, complex):
        return _empty_tuple, complex, _host_set
    if isinstance(
        obj,
        (
            list,
            tuple,
        ),
    ):
        return _array_info_sequence(obj)
    if _is_object_with_buffer_protocol(obj):
        np_obj = np.array(obj)
        return np_obj.shape, np_obj.dtype, _host_set
    if hasattr(obj, "__usm_ndarray__"):
        usm_ar = obj.__usm_ndarray__
        if isinstance(usm_ar, dpt.usm_ndarray):
            return usm_ar.shape, usm_ar.dtype, frozenset([usm_ar.sycl_queue])
    if hasattr(obj, "__sycl_usm_array_interface__"):
        usm_ar = _usm_ndarray_from_suai(obj)
        return usm_ar.shape, usm_ar.dtype, frozenset([usm_ar.sycl_queue])


def _array_info_sequence(li):
    if not isinstance(li, (list, tuple, range)):
        raise TypeError(f"Expected list, tuple, or range, got {type(li)}")
    n = len(li)
    dim = None
    dt = None
    device = frozenset()
    for el in li:
        el_dim, el_dt, el_dev = _array_info_dispatch(el)
        if dim is None:
            dim = el_dim
            dt = np.promote_types(el_dt, el_dt)
            device = device.union(el_dev)
        elif el_dim == dim:
            dt = np.promote_types(dt, el_dt)
            device = device.union(el_dev)
        else:
            raise ValueError(f"Inconsistent dimensions, {dim} and {el_dim}")
    if dim is None:
        dim = ()
        dt = float
        device = _host_set
    return (n,) + dim, dt, device


def _cast_fill_val(fill_val, dt):
    """
    Casts the Python scalar `fill_val` to another Python type coercible to the
    requested data type `dt`, if necessary.
    """
    val_type = type(fill_val)
    if val_type in [float, complex] and np.issubdtype(dt, np.integer):
        return int(fill_val.real)
    elif val_type is complex and np.issubdtype(dt, np.floating):
        return fill_val.real
    elif val_type is int and np.issubdtype(dt, np.integer):
        return _to_scalar(fill_val, dt)
    else:
        return fill_val


def _coerce_and_infer_dt(*args, dt, sycl_queue, err_msg, allow_bool=False):
    """Deduce arange type from sequence spec"""
    nd, seq_dt, d = _array_info_sequence(args)
    if d != _host_set or nd != (len(args),):
        raise ValueError(err_msg)
    dt = _get_dtype(dt, sycl_queue, ref_type=seq_dt)
    if np.issubdtype(dt, np.integer):
        return tuple(int(v) for v in args), dt
    if np.issubdtype(dt, np.floating):
        return tuple(float(v) for v in args), dt
    if np.issubdtype(dt, np.complexfloating):
        return tuple(complex(v) for v in args), dt
    if allow_bool and dt.char == "?":
        return tuple(bool(v) for v in args), dt
    raise ValueError(f"Data type {dt} is not supported")


def _ensure_native_dtype_device_support(dtype, dev) -> None:
    """Check that dtype is natively supported by device.

    Arg:
        dtype:
            Elemental data-type
        dev (:class:`dpctl.SyclDevice`):
            The device about which the query is being made.
    Returns:
        None
    Raise:
        ValueError:
            if device does not natively support this `dtype`.
    """
    if dtype in [dpt.float64, dpt.complex128] and not dev.has_aspect_fp64:
        raise ValueError(
            f"Device {dev.name} does not provide native support "
            "for double-precision floating point type."
        )
    if (
        dtype
        in [
            dpt.float16,
        ]
        and not dev.has_aspect_fp16
    ):
        raise ValueError(
            f"Device {dev.name} does not provide native support "
            "for half-precision floating point type."
        )


def _to_scalar(obj, sc_ty):
    """A way to convert object to NumPy scalar type.
    Raises OverflowError if obj can not be represented
    using the requested scalar type.
    """
    zd_arr = np.asarray(obj, dtype=sc_ty)
    return zd_arr[()]


def _usm_ndarray_from_suai(obj):
    sua_iface = obj.__sycl_usm_array_interface__
    membuf = dpm.as_usm_memory(obj)
    ary = dpt.usm_ndarray(
        sua_iface["shape"],
        dtype=sua_iface["typestr"],
        buffer=membuf,
        strides=sua_iface.get("strides", None),
    )
    _data_field = sua_iface["data"]
    if isinstance(_data_field, tuple) and len(_data_field) > 1:
        ro_field = _data_field[1]
    else:
        ro_field = False
    if ro_field:
        ary.flags["W"] = False
    return ary


def eye(
    n_rows,
    n_cols=None,
    /,
    *,
    k=0,
    dtype=None,
    order="C",
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    eye(n_rows, n_cols=None, /, *, k=0, dtype=None, \
        device=None, usm_type="device", sycl_queue=None)

    Creates :class:`dpctl.tensor.usm_ndarray` with ones on the `k`-th
    diagonal.

    Args:
        n_rows (int):
            number of rows in the output array.
        n_cols (int, optional):
            number of columns in the output array. If ``None``,
            ``n_cols = n_rows``. Default: ``None``
        k (int):
            index of the diagonal, with ``0`` as the main diagonal.
            A positive value of ``k`` is a superdiagonal, a negative value
            is a subdiagonal.
            Raises :exc:`TypeError` if ``k`` is not an integer.
            Default: ``0``
        dtype (optional):
            data type of the array. Can be typestring,
            a :class:`numpy.dtype` object, :mod:`numpy` char string, or
            a NumPy scalar type. Default: ``None``
        order ("C" or "F"):
            memory layout for the array. Default: ``"C"``
        device (optional):
            array API concept of device where the output array
            is created. ``device`` can be ``None``, a oneAPI filter selector
            string, an instance of :class:`dpctl.SyclDevice` corresponding to
            a non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a :class:`dpctl.tensor.Device` object
            returned by :attr:`dpctl.tensor.usm_ndarray.device`.
            Default: ``None``
        usm_type (``"device"``, ``"shared"``, ``"host"``, optional):
            The type of SYCL USM allocation for the output array.
            Default: ``"device"``
        sycl_queue (:class:`dpctl.SyclQueue`, optional):
            The SYCL queue to use
            for output array allocation and copying. ``sycl_queue`` and
            ``device`` are complementary arguments, i.e. use one or another.
            If both are specified, a :exc:`TypeError` is raised unless both
            imply the same underlying SYCL queue to be used. If both are
            ``None``, a cached queue targeting default-selected device is
            used for allocation and population. Default: ``None``

    Returns:
        usm_ndarray:
            A diagonal matrix.
    """
    if not isinstance(order, str) or len(order) == 0 or order[0] not in "CcFf":
        raise ValueError(
            "Unrecognized order keyword value, expecting 'F' or 'C'."
        )
    order = order[0].upper()
    n_rows = operator.index(n_rows)
    n_cols = n_rows if n_cols is None else operator.index(n_cols)
    k = operator.index(k)
    if k >= n_cols or -k >= n_rows:
        return dpt.zeros(
            (n_rows, n_cols),
            dtype=dtype,
            order=order,
            device=device,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
        )
    dpctl.utils.validate_usm_type(usm_type, allow_none=False)
    sycl_queue = normalize_queue_device(sycl_queue=sycl_queue, device=device)
    dtype = _get_dtype(dtype, sycl_queue)
    _ensure_native_dtype_device_support(dtype, sycl_queue.sycl_device)
    res = dpt.usm_ndarray(
        (n_rows, n_cols),
        dtype=dtype,
        buffer=usm_type,
        order=order,
        buffer_ctor_kwargs={"queue": sycl_queue},
    )
    if n_rows != 0 and n_cols != 0:
        _manager = dpctl.utils.SequentialOrderManager[sycl_queue]
        hev, eye_ev = ti._eye(k, dst=res, sycl_queue=sycl_queue)
        _manager.add_event_pair(hev, eye_ev)
    return res


def _validate_fill_value(fill_val):
    """Validates that `fill_val` is a numeric or boolean scalar."""
    # TODO: verify if `np.True_` and `np.False_` should be instances of
    # Number in NumPy, like other NumPy scalars and like Python bools
    # check for `np.bool_` separately as NumPy<2 has no `np.bool`
    if not isinstance(fill_val, Number) and not isinstance(fill_val, np.bool_):
        raise TypeError(
            f"array cannot be filled with scalar of type {type(fill_val)}"
        )


def full(
    shape,
    fill_value,
    *,
    dtype=None,
    order="C",
    device=None,
    usm_type=None,
    sycl_queue=None,
):
    """
    Returns a new :class:`dpctl.tensor.usm_ndarray` having a specified
    shape and filled with `fill_value`.

    Args:
        shape (tuple):
            Dimensions of the array to be created.
        fill_value (int,float,complex,usm_ndarray):
            fill value
        dtype (optional): data type of the array. Can be typestring,
            a :class:`numpy.dtype` object, :mod:`numpy` char string,
            or a NumPy scalar type. Default: ``None``
        order ("C", or "F"):
            memory layout for the array. Default: ``"C"``
        device (optional): array API concept of device where the output array
            is created. ``device`` can be ``None``, a oneAPI filter selector
            string, an instance of :class:`dpctl.SyclDevice` corresponding to
            a non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a :class:`dpctl.tensor.Device` object
            returned by :attr:`dpctl.tensor.usm_ndarray.device`.
            Default: ``None``
        usm_type (``"device"``, ``"shared"``, ``"host"``, optional):
            The type of SYCL USM allocation for the output array.
            Default: ``"device"``
        sycl_queue (:class:`dpctl.SyclQueue`, optional):
            The SYCL queue to use
            for output array allocation and copying. ``sycl_queue`` and
            ``device`` are complementary arguments, i.e. use one or another.
            If both are specified, a :exc:`TypeError` is raised unless both
            imply the same underlying SYCL queue to be used. If both are
            ``None``, a cached queue targeting default-selected device is
            used for allocation and population. Default: ``None``

    Returns:
        usm_ndarray:
            New array initialized with given value.
    """
    if not isinstance(order, str) or len(order) == 0 or order[0] not in "CcFf":
        raise ValueError(
            "Unrecognized order keyword value, expecting 'F' or 'C'."
        )
    order = order[0].upper()
    dpctl.utils.validate_usm_type(usm_type, allow_none=True)

    if isinstance(fill_value, (dpt.usm_ndarray, np.ndarray, tuple, list)):
        if (
            isinstance(fill_value, dpt.usm_ndarray)
            and sycl_queue is None
            and device is None
        ):
            sycl_queue = fill_value.sycl_queue
        else:
            sycl_queue = normalize_queue_device(
                sycl_queue=sycl_queue, device=device
            )
        X = dpt.asarray(
            fill_value,
            dtype=dtype,
            order=order,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
        )
        return dpt_ext.copy(dpt.broadcast_to(X, shape), order=order)
    else:
        _validate_fill_value(fill_value)

    sycl_queue = normalize_queue_device(sycl_queue=sycl_queue, device=device)
    usm_type = usm_type if usm_type is not None else "device"
    dtype = _get_dtype(dtype, sycl_queue, ref_type=type(fill_value))
    res = dpt.usm_ndarray(
        shape,
        dtype=dtype,
        buffer=usm_type,
        order=order,
        buffer_ctor_kwargs={"queue": sycl_queue},
    )
    fill_value = _cast_fill_val(fill_value, dtype)

    _manager = dpctl.utils.SequentialOrderManager[sycl_queue]
    # populating new allocation, no dependent events
    hev, full_ev = ti._full_usm_ndarray(fill_value, res, sycl_queue)
    _manager.add_event_pair(hev, full_ev)
    return res


def linspace(
    start,
    stop,
    /,
    num,
    *,
    dtype=None,
    device=None,
    endpoint=True,
    sycl_queue=None,
    usm_type="device",
):
    """
    linspace(start, stop, num, dtype=None, device=None, endpoint=True, \
        sycl_queue=None, usm_type="device")

    Returns :class:`dpctl.tensor.usm_ndarray` array populated with
    evenly spaced numbers of specified interval.

    Args:
        start:
            the start of the interval.
        stop:
            the end of the interval. If the ``endpoint`` is ``False``, the
            function generates ``num+1`` evenly spaced points starting
            with ``start`` and ending with ``stop`` and exclude the
            ``stop`` from the returned array such that the returned array
            consists of evenly spaced numbers over the half-open interval
            ``[start, stop)``. If ``endpoint`` is ``True``, the output
            array consists of evenly spaced numbers over the closed
            interval ``[start, stop]``. Default: ``True``
        num (int):
            number of samples. Must be a non-negative integer; otherwise,
            the function raises ``ValueError`` exception.
        dtype:
            output array data type. Should be a floating data type.
            If ``dtype`` is ``None``, the output array must be the default
            floating point data type for target device.
            Default: ``None``
        device (optional):
            array API concept of device where the output array
            is created. ``device`` can be ``None``, a oneAPI filter selector
            string, an instance of :class:`dpctl.SyclDevice` corresponding to
            a non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a :class:`dpctl.tensor.Device` object
            returned by :attr:`dpctl.tensor.usm_ndarray.device`.
            Default: ``None``
        usm_type (``"device"``, ``"shared"``, ``"host"``, optional):
            The type of SYCL USM allocation for the output array.
            Default: ``"device"``
        sycl_queue (:class:`dpctl.SyclQueue`, optional):
            The SYCL queue to use
            for output array allocation and copying. ``sycl_queue`` and
            ``device`` are complementary arguments, i.e. use one or another.
            If both are specified, a :exc:`TypeError` is raised unless both
            imply the same underlying SYCL queue to be used. If both are
            ``None``, a cached queue targeting default-selected device is
            used for allocation and population. Default: ``None``
        endpoint: boolean indicating whether to include ``stop`` in the
            interval. Default: ``True``

    Returns:
        usm_ndarray:
            Array populated with evenly spaced numbers in the requested
            interval.
    """
    sycl_queue = normalize_queue_device(sycl_queue=sycl_queue, device=device)
    dpctl.utils.validate_usm_type(usm_type, allow_none=False)
    if endpoint not in [True, False]:
        raise TypeError("endpoint keyword argument must be of boolean type")

    num = operator.index(num)
    if num < 0:
        raise ValueError("Number of points must be non-negative")

    _, dt = _coerce_and_infer_dt(
        start,
        stop,
        dt=dtype,
        sycl_queue=sycl_queue,
        err_msg="start and stop must be Python scalars.",
        allow_bool=True,
    )

    int_dt = None
    if np.issubdtype(dt, np.integer):
        if dtype is not None:
            int_dt = dt
        dt = ti.default_device_fp_type(sycl_queue)
        dt = dpt.dtype(dt)
        start = float(start)
        stop = float(stop)

    res = dpt.empty(num, dtype=dt, usm_type=usm_type, sycl_queue=sycl_queue)
    _manager = dpctl.utils.SequentialOrderManager[sycl_queue]
    hev, la_ev = ti._linspace_affine(
        start, stop, dst=res, include_endpoint=endpoint, sycl_queue=sycl_queue
    )
    _manager.add_event_pair(hev, la_ev)

    return res if int_dt is None else dpt.astype(res, int_dt)


def tril(x, /, *, k=0):
    """
    Returns the lower triangular part of a matrix (or a stack of matrices)
    ``x``.

    The lower triangular part of the matrix is defined as the elements on and
    below the specified diagonal ``k``.

    Args:
        x (usm_ndarray):
            Input array
        k (int, optional):
            Specifies the diagonal above which to set
            elements to zero. If ``k = 0``, the diagonal is the main diagonal.
            If ``k < 0``, the diagonal is below the main diagonal.
            If ``k > 0``, the diagonal is above the main diagonal.
            Default: ``0``

    Returns:
        usm_ndarray:
            A lower-triangular array or a stack of lower-triangular arrays.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(
            "Expected argument of type dpctl.tensor.usm_ndarray, "
            f"got {type(x)}."
        )

    k = operator.index(k)

    order = "F" if (x.flags.f_contiguous) else "C"

    shape = x.shape
    nd = x.ndim
    if nd < 2:
        raise ValueError("Array dimensions less than 2.")

    q = x.sycl_queue
    if k >= shape[nd - 1] - 1:
        res = dpt.empty(
            x.shape,
            dtype=x.dtype,
            order=order,
            usm_type=x.usm_type,
            sycl_queue=q,
        )
        _manager = dpctl.utils.SequentialOrderManager[q]
        dep_evs = _manager.submitted_events
        hev, cpy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=x, dst=res, sycl_queue=q, depends=dep_evs
        )
        _manager.add_event_pair(hev, cpy_ev)
    elif k < -shape[nd - 2]:
        res = dpt.zeros(
            x.shape,
            dtype=x.dtype,
            order=order,
            usm_type=x.usm_type,
            sycl_queue=q,
        )
    else:
        res = dpt.empty(
            x.shape,
            dtype=x.dtype,
            order=order,
            usm_type=x.usm_type,
            sycl_queue=q,
        )
        _manager = dpctl.utils.SequentialOrderManager[q]
        dep_evs = _manager.submitted_events
        hev, tril_ev = ti._tril(
            src=x, dst=res, k=k, sycl_queue=q, depends=dep_evs
        )
        _manager.add_event_pair(hev, tril_ev)

    return res


def triu(x, /, *, k=0):
    """
    Returns the upper triangular part of a matrix (or a stack of matrices)
    ``x``.

    The upper triangular part of the matrix is defined as the elements on and
    above the specified diagonal ``k``.

    Args:
        x (usm_ndarray):
            Input array
        k (int, optional):
            Specifies the diagonal below which to set
            elements to zero. If ``k = 0``, the diagonal is the main diagonal.
            If ``k < 0``, the diagonal is below the main diagonal.
            If ``k > 0``, the diagonal is above the main diagonal.
            Default: ``0``

    Returns:
        usm_ndarray:
            An upper-triangular array or a stack of upper-triangular arrays.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(
            "Expected argument of type dpctl.tensor.usm_ndarray, "
            f"got {type(x)}."
        )

    k = operator.index(k)

    order = "F" if (x.flags.f_contiguous) else "C"

    shape = x.shape
    nd = x.ndim
    if nd < 2:
        raise ValueError("Array dimensions less than 2.")

    q = x.sycl_queue
    if k > shape[nd - 1]:
        res = dpt.zeros(
            x.shape,
            dtype=x.dtype,
            order=order,
            usm_type=x.usm_type,
            sycl_queue=q,
        )
    elif k <= -shape[nd - 2] + 1:
        res = dpt.empty(
            x.shape,
            dtype=x.dtype,
            order=order,
            usm_type=x.usm_type,
            sycl_queue=q,
        )
        _manager = dpctl.utils.SequentialOrderManager[q]
        dep_evs = _manager.submitted_events
        hev, cpy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=x, dst=res, sycl_queue=q, depends=dep_evs
        )
        _manager.add_event_pair(hev, cpy_ev)
    else:
        res = dpt.empty(
            x.shape,
            dtype=x.dtype,
            order=order,
            usm_type=x.usm_type,
            sycl_queue=q,
        )
        _manager = dpctl.utils.SequentialOrderManager[q]
        dep_evs = _manager.submitted_events
        hev, triu_ev = ti._triu(
            src=x, dst=res, k=k, sycl_queue=q, depends=dep_evs
        )
        _manager.add_event_pair(hev, triu_ev)

    return res
