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

import builtins
import operator
from numbers import Integral

import dpctl
import dpctl.memory as dpm
import dpctl.tensor as dpt
import dpctl.utils
import numpy as np
from dpctl.tensor._data_types import _get_dtype
from dpctl.tensor._device import normalize_queue_device
from dpctl.tensor._type_utils import _dtype_supported_by_device_impl

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt_ext
import dpctl_ext.tensor._tensor_impl as ti

from ._numpy_helper import normalize_axis_index

__doc__ = (
    "Implementation module for copy- and cast- operations on "
    ":class:`dpctl.tensor.usm_ndarray`."
)

int32_t_max = 1 + np.iinfo(np.int32).max


def _copy_to_numpy(ary):
    if not isinstance(ary, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(ary)}")
    if ary.size == 0:
        # no data needs to be copied for zero sized array
        return np.ndarray(ary.shape, dtype=ary.dtype)
    nb = ary.usm_data.nbytes
    q = ary.sycl_queue
    hh = dpm.MemoryUSMHost(nb, queue=q)
    h = np.ndarray(nb, dtype="u1", buffer=hh).view(ary.dtype)
    itsz = ary.itemsize
    strides_bytes = tuple(si * itsz for si in ary.strides)
    offset = ary._element_offset * itsz
    # ensure that content of ary.usm_data is final
    q.wait()
    hh.copy_from_device(ary.usm_data)
    return np.ndarray(
        ary.shape,
        dtype=ary.dtype,
        buffer=h,
        strides=strides_bytes,
        offset=offset,
    )


def _copy_from_numpy(np_ary, usm_type="device", sycl_queue=None):
    """Copies numpy array `np_ary` into a new usm_ndarray"""
    # This may perform a copy to meet stated requirements
    Xnp = np.require(np_ary, requirements=["A", "E"])
    alloc_q = normalize_queue_device(sycl_queue=sycl_queue, device=None)
    dt = Xnp.dtype
    if dt.char in "dD" and alloc_q.sycl_device.has_aspect_fp64 is False:
        Xusm_dtype = (
            dpt.dtype("float32") if dt.char == "d" else dpt.dtype("complex64")
        )
    else:
        Xusm_dtype = dt
    Xusm = dpt.empty(
        Xnp.shape, dtype=Xusm_dtype, usm_type=usm_type, sycl_queue=sycl_queue
    )
    _copy_from_numpy_into(Xusm, Xnp)
    return Xusm


def _copy_from_numpy_into(dst, np_ary):
    """Copies `np_ary` into `dst` of type :class:`dpctl.tensor.usm_ndarray"""
    if not isinstance(np_ary, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(np_ary)}")
    if not isinstance(dst, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray, got {type(dst)}")
    if np_ary.flags["OWNDATA"]:
        Xnp = np_ary
    else:
        # Determine base of input array
        base = np_ary.base
        while isinstance(base, np.ndarray):
            base = base.base
        if isinstance(base, dpm._memory._Memory):
            # we must perform a copy, since subsequent
            # _copy_numpy_ndarray_into_usm_ndarray is implemented using
            # sycl::buffer, and using USM-pointers with sycl::buffer
            # results is undefined behavior
            Xnp = np_ary.copy()
        else:
            Xnp = np_ary
    src_ary = np.broadcast_to(Xnp, dst.shape)
    copy_q = dst.sycl_queue
    if copy_q.sycl_device.has_aspect_fp64 is False:
        src_ary_dt_c = src_ary.dtype.char
        if src_ary_dt_c == "d":
            src_ary = src_ary.astype(np.float32)
        elif src_ary_dt_c == "D":
            src_ary = src_ary.astype(np.complex64)
    _manager = dpctl.utils.SequentialOrderManager[copy_q]
    dep_ev = _manager.submitted_events
    # synchronizing call
    ti._copy_numpy_ndarray_into_usm_ndarray(
        src=src_ary, dst=dst, sycl_queue=copy_q, depends=dep_ev
    )


def _extract_impl(ary, ary_mask, axis=0):
    """
    Extract elements of ary by applying mask starting from slot
    dimension axis
    """
    if not isinstance(ary, dpt.usm_ndarray):
        raise TypeError(
            f"Expecting type dpctl.tensor.usm_ndarray, got {type(ary)}"
        )
    if isinstance(ary_mask, dpt.usm_ndarray):
        dst_usm_type = dpctl.utils.get_coerced_usm_type(
            (ary.usm_type, ary_mask.usm_type)
        )
        exec_q = dpctl.utils.get_execution_queue(
            (ary.sycl_queue, ary_mask.sycl_queue)
        )
        if exec_q is None:
            raise dpctl.utils.ExecutionPlacementError(
                "arrays have different associated queues. "
                "Use `y.to_device(x.device)` to migrate."
            )
    elif isinstance(ary_mask, np.ndarray):
        dst_usm_type = ary.usm_type
        exec_q = ary.sycl_queue
        ary_mask = dpt.asarray(
            ary_mask, usm_type=dst_usm_type, sycl_queue=exec_q
        )
    else:
        raise TypeError(
            "Expecting type dpctl.tensor.usm_ndarray or numpy.ndarray, got "
            f"{type(ary_mask)}"
        )
    ary_nd = ary.ndim
    pp = normalize_axis_index(operator.index(axis), ary_nd)
    mask_nd = ary_mask.ndim
    if pp < 0 or pp + mask_nd > ary_nd:
        raise ValueError(
            "Parameter p is inconsistent with input array dimensions"
        )
    mask_nelems = ary_mask.size
    cumsum_dt = dpt.int32 if mask_nelems < int32_t_max else dpt.int64
    cumsum = dpt.empty(mask_nelems, dtype=cumsum_dt, device=ary_mask.device)
    exec_q = cumsum.sycl_queue
    _manager = dpctl.utils.SequentialOrderManager[exec_q]
    dep_evs = _manager.submitted_events
    mask_count = ti.mask_positions(
        ary_mask, cumsum, sycl_queue=exec_q, depends=dep_evs
    )
    dst_shape = ary.shape[:pp] + (mask_count,) + ary.shape[pp + mask_nd :]
    dst = dpt.empty(
        dst_shape, dtype=ary.dtype, usm_type=dst_usm_type, device=ary.device
    )
    if dst.size == 0:
        return dst
    hev, ev = ti._extract(
        src=ary,
        cumsum=cumsum,
        axis_start=pp,
        axis_end=pp + mask_nd,
        dst=dst,
        sycl_queue=exec_q,
        depends=dep_evs,
    )
    _manager.add_event_pair(hev, ev)
    return dst


def _get_indices_queue_usm_type(inds, queue, usm_type):
    """
    Utility for validating indices are NumPy ndarray or usm_ndarray of integral
    dtype or Python integers. At least one must be an array.

    For each array, the queue and usm type are appended to `queue_list` and
    `usm_type_list`, respectively.
    """
    queues = [queue]
    usm_types = [usm_type]
    any_array = False
    for ind in inds:
        if isinstance(ind, (np.ndarray, dpt.usm_ndarray)):
            any_array = True
            if ind.dtype.kind not in "ui":
                raise IndexError(
                    "arrays used as indices must be of integer (or boolean) "
                    "type"
                )
            if isinstance(ind, dpt.usm_ndarray):
                queues.append(ind.sycl_queue)
                usm_types.append(ind.usm_type)
        elif not isinstance(ind, Integral):
            raise TypeError(
                "all elements of `ind` expected to be usm_ndarrays, "
                f"NumPy arrays, or integers, found {type(ind)}"
            )
    if not any_array:
        raise TypeError(
            "at least one element of `inds` expected to be an array"
        )
    usm_type = dpctl.utils.get_coerced_usm_type(usm_types)
    q = dpctl.utils.get_execution_queue(queues)
    return q, usm_type


def _nonzero_impl(ary):
    if not isinstance(ary, dpt.usm_ndarray):
        raise TypeError(
            f"Expecting type dpctl.tensor.usm_ndarray, got {type(ary)}"
        )
    exec_q = ary.sycl_queue
    usm_type = ary.usm_type
    mask_nelems = ary.size
    cumsum_dt = dpt.int32 if mask_nelems < int32_t_max else dpt.int64
    cumsum = dpt.empty(
        mask_nelems, dtype=cumsum_dt, sycl_queue=exec_q, order="C"
    )
    _manager = dpctl.utils.SequentialOrderManager[exec_q]
    dep_evs = _manager.submitted_events
    mask_count = ti.mask_positions(
        ary, cumsum, sycl_queue=exec_q, depends=dep_evs
    )
    indexes_dt = ti.default_device_index_type(exec_q.sycl_device)
    indexes = dpt.empty(
        (ary.ndim, mask_count),
        dtype=indexes_dt,
        usm_type=usm_type,
        sycl_queue=exec_q,
        order="C",
    )
    hev, nz_ev = ti._nonzero(cumsum, indexes, ary.shape, exec_q)
    res = tuple(indexes[i, :] for i in range(ary.ndim))
    _manager.add_event_pair(hev, nz_ev)
    return res


def _prepare_indices_arrays(inds, q, usm_type):
    """
    Utility taking a mix of usm_ndarray and possibly Python int scalar indices,
    a queue (assumed to be common to arrays in inds), and a usm type.

    Python scalar integers are promoted to arrays on the provided queue and
    with the provided usm type. All arrays are then promoted to a common
    integral type (if possible) before being broadcast to a common shape.
    """
    # scalar integers -> arrays
    inds = tuple(
        map(
            lambda ind: (
                ind
                if isinstance(ind, dpt.usm_ndarray)
                else dpt.asarray(ind, usm_type=usm_type, sycl_queue=q)
            ),
            inds,
        )
    )

    # promote to a common integral type if possible
    ind_dt = dpt.result_type(*inds)
    if ind_dt.kind not in "ui":
        raise ValueError(
            "cannot safely promote indices to an integer data type"
        )
    inds = tuple(
        map(
            lambda ind: (
                ind if ind.dtype == ind_dt else dpt.astype(ind, ind_dt)
            ),
            inds,
        )
    )

    # broadcast
    inds = dpt.broadcast_arrays(*inds)

    return inds


def _put_multi_index(ary, inds, p, vals, mode=0):
    if not isinstance(ary, dpt.usm_ndarray):
        raise TypeError(
            f"Expecting type dpctl.tensor.usm_ndarray, got {type(ary)}"
        )
    ary_nd = ary.ndim
    p = normalize_axis_index(operator.index(p), ary_nd)
    mode = operator.index(mode)
    if mode not in [0, 1]:
        raise ValueError(
            "Invalid value for mode keyword, only 0 or 1 is supported"
        )
    if not isinstance(inds, (list, tuple)):
        inds = (inds,)

    exec_q, coerced_usm_type = _get_indices_queue_usm_type(
        inds, ary.sycl_queue, ary.usm_type
    )

    if exec_q is not None:
        if not isinstance(vals, dpt.usm_ndarray):
            vals = dpt.asarray(
                vals,
                dtype=ary.dtype,
                usm_type=coerced_usm_type,
                sycl_queue=exec_q,
            )
        else:
            exec_q = dpctl.utils.get_execution_queue((exec_q, vals.sycl_queue))
            coerced_usm_type = dpctl.utils.get_coerced_usm_type(
                (
                    coerced_usm_type,
                    vals.usm_type,
                )
            )
    if exec_q is None:
        raise dpctl.utils.ExecutionPlacementError(
            "Can not automatically determine where to allocate the "
            "result or performance execution. "
            "Use `usm_ndarray.to_device` method to migrate data to "
            "be associated with the same queue."
        )

    inds = _prepare_indices_arrays(inds, exec_q, coerced_usm_type)

    ind0 = inds[0]
    ary_sh = ary.shape
    p_end = p + len(inds)
    if 0 in ary_sh[p:p_end] and ind0.size != 0:
        raise IndexError(
            "cannot put into non-empty indices along an empty axis"
        )
    expected_vals_shape = ary_sh[:p] + ind0.shape + ary_sh[p_end:]
    if vals.dtype == ary.dtype:
        rhs = vals
    else:
        rhs = dpt_ext.astype(vals, ary.dtype)
    rhs = dpt.broadcast_to(rhs, expected_vals_shape)
    _manager = dpctl.utils.SequentialOrderManager[exec_q]
    dep_ev = _manager.submitted_events
    hev, put_ev = ti._put(
        dst=ary,
        ind=inds,
        val=rhs,
        axis_start=p,
        mode=mode,
        sycl_queue=exec_q,
        depends=dep_ev,
    )
    _manager.add_event_pair(hev, put_ev)
    return


def from_numpy(np_ary, /, *, device=None, usm_type="device", sycl_queue=None):
    """
    from_numpy(arg, device=None, usm_type="device", sycl_queue=None)

    Creates :class:`dpctl.tensor.usm_ndarray` from instance of
    :class:`numpy.ndarray`.

    Args:
        arg:
            Input convertible to :class:`numpy.ndarray`
        device (object): array API specification of device where the
            output array is created. Device can be specified by
            a filter selector string, an instance of
            :class:`dpctl.SyclDevice`, an instance of
            :class:`dpctl.SyclQueue`, or an instance of
            :class:`dpctl.tensor.Device`. If the value is ``None``,
            returned array is created on the default-selected device.
            Default: ``None``
        usm_type (str): The requested USM allocation type for the
            output array. Recognized values are ``"device"``,
            ``"shared"``, or ``"host"``
        sycl_queue (:class:`dpctl.SyclQueue`, optional):
            A SYCL queue that determines output array allocation device
            as well as execution placement of data movement operations.
            The ``device`` and ``sycl_queue`` arguments
            are equivalent. Only one of them should be specified. If both
            are provided, they must be consistent and result in using the
            same execution queue. Default: ``None``

    The returned array has the same shape, and the same data type kind.
    If the device does not support the data type of input array, a
    closest support data type of the same kind may be returned, e.g.
    input array of type ``float16`` may be upcast to ``float32`` if the
    target device does not support 16-bit floating point type.
    """
    q = normalize_queue_device(sycl_queue=sycl_queue, device=device)
    return _copy_from_numpy(np_ary, usm_type=usm_type, sycl_queue=q)


def to_numpy(usm_ary, /):
    """
    to_numpy(usm_ary)

    Copies content of :class:`dpctl.tensor.usm_ndarray` instance ``usm_ary``
    into :class:`numpy.ndarray` instance of the same shape and same data type.

    Args:
        usm_ary (usm_ndarray):
            Input array
    Returns:
        :class:`numpy.ndarray`:
            An instance of :class:`numpy.ndarray` populated with content of
            ``usm_ary``
    """
    return _copy_to_numpy(usm_ary)


def asnumpy(usm_ary):
    """
    asnumpy(usm_ary)

    Copies content of :class:`dpctl.tensor.usm_ndarray` instance ``usm_ary``
    into :class:`numpy.ndarray` instance of the same shape and same data
    type.

    Args:
        usm_ary (usm_ndarray):
            Input array
    Returns:
        :class:`numpy.ndarray`:
            An instance of :class:`numpy.ndarray` populated with content
            of ``usm_ary``
    """
    return _copy_to_numpy(usm_ary)


class Dummy:
    """Helper class with specified ``__sycl_usm_array_interface__`` attribute"""

    def __init__(self, iface):
        self.__sycl_usm_array_interface__ = iface


def _copy_overlapping(dst, src):
    """Assumes src and dst have the same shape."""
    q = normalize_queue_device(sycl_queue=dst.sycl_queue)
    tmp = dpt.usm_ndarray(
        src.shape,
        dtype=src.dtype,
        buffer="device",
        order="C",
        buffer_ctor_kwargs={"queue": q},
    )
    _manager = dpctl.utils.SequentialOrderManager[q]
    dep_evs = _manager.submitted_events
    hcp1, cp1 = ti._copy_usm_ndarray_into_usm_ndarray(
        src=src, dst=tmp, sycl_queue=q, depends=dep_evs
    )
    _manager.add_event_pair(hcp1, cp1)
    hcp2, cp2 = ti._copy_usm_ndarray_into_usm_ndarray(
        src=tmp, dst=dst, sycl_queue=q, depends=[cp1]
    )
    _manager.add_event_pair(hcp2, cp2)


def _copy_same_shape(dst, src):
    """Assumes src and dst have the same shape."""
    # check that memory regions do not overlap
    if ti._array_overlap(dst, src):
        if src._pointer == dst._pointer and (
            src is dst
            or (src.strides == dst.strides and src.dtype == dst.dtype)
        ):
            return
        _copy_overlapping(src=src, dst=dst)
        return

    copy_q = dst.sycl_queue
    _manager = dpctl.utils.SequentialOrderManager[copy_q]
    dep_evs = _manager.submitted_events
    hev, cpy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
        src=src, dst=dst, sycl_queue=copy_q, depends=dep_evs
    )
    _manager.add_event_pair(hev, cpy_ev)


if hasattr(np, "broadcast_shapes"):

    def _broadcast_shapes(sh1, sh2):
        return np.broadcast_shapes(sh1, sh2)

else:

    def _broadcast_shapes(sh1, sh2):
        # use arrays with zero strides, whose memory footprint
        # is independent of the number of array elements
        return np.broadcast(
            np.empty(sh1, dtype=[]),
            np.empty(sh2, dtype=[]),
        ).shape


def _broadcast_strides(X_shape, X_strides, res_ndim):
    """
    Broadcasts strides to match the given dimensions;
    returns tuple type strides.
    """
    out_strides = [0] * res_ndim
    X_shape_len = len(X_shape)
    str_dim = -X_shape_len
    for i in range(X_shape_len):
        shape_value = X_shape[i]
        if not shape_value == 1:
            out_strides[str_dim] = X_strides[i]
        str_dim += 1

    return tuple(out_strides)


def _copy_from_usm_ndarray_to_usm_ndarray(dst, src):
    if any(
        not isinstance(arg, dpt.usm_ndarray)
        for arg in (
            dst,
            src,
        )
    ):
        raise TypeError(
            "Both types are expected to be dpctl.tensor.usm_ndarray, "
            f"got {type(dst)} and {type(src)}."
        )

    if dst.ndim == src.ndim and dst.shape == src.shape:
        _copy_same_shape(dst, src)
        return

    try:
        common_shape = _broadcast_shapes(dst.shape, src.shape)
    except ValueError as exc:
        raise ValueError("Shapes of two arrays are not compatible") from exc

    if dst.size < src.size and dst.size < np.prod(common_shape):
        raise ValueError("Destination is smaller ")

    if len(common_shape) > dst.ndim:
        ones_count = len(common_shape) - dst.ndim
        for k in range(ones_count):
            if common_shape[k] != 1:
                raise ValueError
        common_shape = common_shape[ones_count:]

    if src.ndim < len(common_shape):
        new_src_strides = _broadcast_strides(
            src.shape, src.strides, len(common_shape)
        )
        src_same_shape = dpt.usm_ndarray(
            common_shape,
            dtype=src.dtype,
            buffer=src,
            strides=new_src_strides,
            offset=src._element_offset,
        )
    elif src.ndim == len(common_shape):
        new_src_strides = _broadcast_strides(
            src.shape, src.strides, len(common_shape)
        )
        src_same_shape = dpt.usm_ndarray(
            common_shape,
            dtype=src.dtype,
            buffer=src,
            strides=new_src_strides,
            offset=src._element_offset,
        )
    else:
        # since broadcasting succeeded, src.ndim is greater because of
        # leading sequence of ones, so we trim it
        n = len(common_shape)
        new_src_strides = _broadcast_strides(
            src.shape[-n:], src.strides[-n:], n
        )
        src_same_shape = dpt.usm_ndarray(
            common_shape,
            dtype=src.dtype,
            buffer=src.usm_data,
            strides=new_src_strides,
            offset=src._element_offset,
        )

    _copy_same_shape(dst, src_same_shape)


def _make_empty_like_orderK(x, dt, usm_type, dev):
    """
    Returns empty array with shape and strides like `x`, with dtype `dt`,
    USM type `usm_type`, on device `dev`.
    """
    st = list(x.strides)
    perm = sorted(
        range(x.ndim),
        key=lambda d: builtins.abs(st[d]) if x.shape[d] > 1 else 0,
        reverse=True,
    )
    inv_perm = sorted(range(x.ndim), key=lambda i: perm[i])
    sh = x.shape
    sh_sorted = tuple(sh[i] for i in perm)
    R = dpt.empty(sh_sorted, dtype=dt, usm_type=usm_type, device=dev, order="C")
    if min(st) < 0:
        st_sorted = [st[i] for i in perm]
        sl = tuple(
            (
                slice(None, None, -1)
                if st_sorted[i] < 0
                else slice(None, None, None)
            )
            for i in range(x.ndim)
        )
        R = R[sl]
    return dpt.permute_dims(R, inv_perm)


def _empty_like_orderK(x, dt, usm_type=None, dev=None):
    """
    Returns empty array like `x`, using order='K'

    For an array `x` that was obtained by permutation of a contiguous
    array the returned array will have the same shape and the same
    strides as `x`.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray, got {type(x)}")
    if usm_type is None:
        usm_type = x.usm_type
    if dev is None:
        dev = x.device
    fl = x.flags
    if fl["C"] or x.size <= 1:
        return dpt.empty_like(
            x, dtype=dt, usm_type=usm_type, device=dev, order="C"
        )
    elif fl["F"]:
        return dpt.empty_like(
            x, dtype=dt, usm_type=usm_type, device=dev, order="F"
        )
    return _make_empty_like_orderK(x, dt, usm_type, dev)


def _from_numpy_empty_like_orderK(x, dt, usm_type, dev):
    """
    Returns empty usm_ndarray like NumPy array `x`, using order='K'

    For an array `x` that was obtained by permutation of a contiguous
    array the returned array will have the same shape and the same
    strides as `x`.
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(x)}")
    fl = x.flags
    if fl["C"] or x.size <= 1:
        return dpt.empty(
            x.shape, dtype=dt, usm_type=usm_type, device=dev, order="C"
        )
    elif fl["F"]:
        return dpt.empty(
            x.shape, dtype=dt, usm_type=usm_type, device=dev, order="F"
        )
    return _make_empty_like_orderK(x, dt, usm_type, dev)


def _empty_like_pair_orderK(X1, X2, dt, res_shape, usm_type, dev):
    if not isinstance(X1, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray, got {type(X1)}")
    if not isinstance(X2, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray, got {type(X2)}")
    nd1 = X1.ndim
    nd2 = X2.ndim
    if nd1 > nd2 and X1.shape == res_shape:
        return _empty_like_orderK(X1, dt, usm_type, dev)
    elif nd1 < nd2 and X2.shape == res_shape:
        return _empty_like_orderK(X2, dt, usm_type, dev)
    fl1 = X1.flags
    fl2 = X2.flags
    if fl1["C"] or fl2["C"]:
        return dpt.empty(
            res_shape, dtype=dt, usm_type=usm_type, device=dev, order="C"
        )
    if fl1["F"] and fl2["F"]:
        return dpt.empty(
            res_shape, dtype=dt, usm_type=usm_type, device=dev, order="F"
        )
    st1 = list(X1.strides)
    st2 = list(X2.strides)
    max_ndim = max(nd1, nd2)
    st1 += [0] * (max_ndim - len(st1))
    st2 += [0] * (max_ndim - len(st2))
    sh1 = list(X1.shape) + [0] * (max_ndim - nd1)
    sh2 = list(X2.shape) + [0] * (max_ndim - nd2)
    perm = sorted(
        range(max_ndim),
        key=lambda d: (
            builtins.abs(st1[d]) if sh1[d] > 1 else 0,
            builtins.abs(st2[d]) if sh2[d] > 1 else 0,
        ),
        reverse=True,
    )
    inv_perm = sorted(range(max_ndim), key=lambda i: perm[i])
    st1_sorted = [st1[i] for i in perm]
    st2_sorted = [st2[i] for i in perm]
    sh = res_shape
    sh_sorted = tuple(sh[i] for i in perm)
    R = dpt.empty(sh_sorted, dtype=dt, usm_type=usm_type, device=dev, order="C")
    if max(min(st1_sorted), min(st2_sorted)) < 0:
        sl = tuple(
            (
                slice(None, None, -1)
                if (st1_sorted[i] < 0 and st2_sorted[i] < 0)
                else slice(None, None, None)
            )
            for i in range(nd1)
        )
        R = R[sl]
    return dpt.permute_dims(R, inv_perm)


def _empty_like_triple_orderK(X1, X2, X3, dt, res_shape, usm_type, dev):
    if not isinstance(X1, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray, got {type(X1)}")
    if not isinstance(X2, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray, got {type(X2)}")
    if not isinstance(X3, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray, got {type(X3)}")
    nd1 = X1.ndim
    nd2 = X2.ndim
    nd3 = X3.ndim
    if X1.shape == res_shape and X2.shape == res_shape and len(res_shape) > nd3:
        return _empty_like_pair_orderK(X1, X2, dt, res_shape, usm_type, dev)
    elif (
        X2.shape == res_shape and X3.shape == res_shape and len(res_shape) > nd1
    ):
        return _empty_like_pair_orderK(X2, X3, dt, res_shape, usm_type, dev)
    elif (
        X1.shape == res_shape and X3.shape == res_shape and len(res_shape) > nd2
    ):
        return _empty_like_pair_orderK(X1, X3, dt, res_shape, usm_type, dev)
    fl1 = X1.flags
    fl2 = X2.flags
    fl3 = X3.flags
    if fl1["C"] or fl2["C"] or fl3["C"]:
        return dpt.empty(
            res_shape, dtype=dt, usm_type=usm_type, device=dev, order="C"
        )
    if fl1["F"] and fl2["F"] and fl3["F"]:
        return dpt.empty(
            res_shape, dtype=dt, usm_type=usm_type, device=dev, order="F"
        )
    st1 = list(X1.strides)
    st2 = list(X2.strides)
    st3 = list(X3.strides)
    max_ndim = max(nd1, nd2, nd3)
    st1 += [0] * (max_ndim - len(st1))
    st2 += [0] * (max_ndim - len(st2))
    st3 += [0] * (max_ndim - len(st3))
    sh1 = list(X1.shape) + [0] * (max_ndim - nd1)
    sh2 = list(X2.shape) + [0] * (max_ndim - nd2)
    sh3 = list(X3.shape) + [0] * (max_ndim - nd3)
    perm = sorted(
        range(max_ndim),
        key=lambda d: (
            builtins.abs(st1[d]) if sh1[d] > 1 else 0,
            builtins.abs(st2[d]) if sh2[d] > 1 else 0,
            builtins.abs(st3[d]) if sh3[d] > 1 else 0,
        ),
        reverse=True,
    )
    inv_perm = sorted(range(max_ndim), key=lambda i: perm[i])
    st1_sorted = [st1[i] for i in perm]
    st2_sorted = [st2[i] for i in perm]
    st3_sorted = [st3[i] for i in perm]
    sh = res_shape
    sh_sorted = tuple(sh[i] for i in perm)
    R = dpt.empty(sh_sorted, dtype=dt, usm_type=usm_type, device=dev, order="C")
    if max(min(st1_sorted), min(st2_sorted), min(st3_sorted)) < 0:
        sl = tuple(
            (
                slice(None, None, -1)
                if (
                    st1_sorted[i] < 0
                    and st2_sorted[i] < 0
                    and st3_sorted[i] < 0
                )
                else slice(None, None, None)
            )
            for i in range(nd1)
        )
        R = R[sl]
    return dpt.permute_dims(R, inv_perm)


def copy(usm_ary, /, *, order="K"):
    """copy(ary, order="K")

    Creates a copy of given instance of :class:`dpctl.tensor.usm_ndarray`.

    Args:
        ary (usm_ndarray):
            Input array
        order (``"C"``, ``"F"``, ``"A"``, ``"K"``, optional):
            Controls the memory layout of the output array
    Returns:
        usm_ndarray:
            A copy of the input array.

    Memory layout of the copy is controlled by ``order`` keyword,
    following NumPy's conventions. The ``order`` keywords can be
    one of the following:

    .. list-table::

        * - ``"C"``
          - C-contiguous memory layout
        * - ``"F"``
          - Fortran-contiguous memory layout
        * - ``"A"``
          - Fortran-contiguous if the input array is also Fortran-contiguous,
            otherwise C-contiguous
        * - ``"K"``
          - match the layout of ``usm_ary`` as closely as possible.

    """
    if len(order) == 0 or order[0] not in "KkAaCcFf":
        raise ValueError(
            "Unrecognized order keyword value, expecting 'K', 'A', 'F', or 'C'."
        )
    order = order[0].upper()
    if not isinstance(usm_ary, dpt.usm_ndarray):
        raise TypeError(
            f"Expected object of type dpt.usm_ndarray, got {type(usm_ary)}"
        )
    copy_order = "C"
    if order == "C":
        pass
    elif order == "F":
        copy_order = order
    elif order == "A":
        if usm_ary.flags.f_contiguous:
            copy_order = "F"
    elif order == "K":
        if usm_ary.flags.f_contiguous:
            copy_order = "F"
    else:
        raise ValueError(
            "Unrecognized value of the order keyword. "
            "Recognized values are 'A', 'C', 'F', or 'K'"
        )
    if order == "K":
        R = _empty_like_orderK(usm_ary, usm_ary.dtype)
    else:
        R = dpt.usm_ndarray(
            usm_ary.shape,
            dtype=usm_ary.dtype,
            buffer=usm_ary.usm_type,
            order=copy_order,
            buffer_ctor_kwargs={"queue": usm_ary.sycl_queue},
        )
    _copy_same_shape(R, usm_ary)
    return R


def astype(
    usm_ary, newdtype, /, *, order="K", casting="unsafe", copy=True, device=None
):
    """astype(array, new_dtype, order="K", casting="unsafe", \
            copy=True, device=None)

    Returns a copy of the :class:`dpctl.tensor.usm_ndarray`, cast to a
    specified type.

    Args:
        array (usm_ndarray):
            An input array.
        new_dtype (dtype):
            The data type of the resulting array. If `None`, gives default
            floating point type supported by device where the resulting array
            will be located.
        order ({"C", "F", "A", "K"}, optional):
            Controls memory layout of the resulting array if a copy
            is returned.
        casting ({'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional):
            Controls what kind of data casting may occur. Please see
            :meth:`numpy.ndarray.astype` for description of casting modes.
        copy (bool, optional):
            By default, `astype` always returns a newly allocated array.
            If this keyword is set to `False`, a view of the input array
            may be returned when possible.
        device (object): array API specification of device where the
            output array is created. Device can be specified by
            a filter selector string, an instance of
            :class:`dpctl.SyclDevice`, an instance of
            :class:`dpctl.SyclQueue`, or an instance of
            :class:`dpctl.tensor.Device`. If the value is `None`,
            returned array is created on the same device as `array`.
            Default: `None`.

    Returns:
        usm_ndarray:
            An array with requested data type.

    A view can be returned, if possible, when `copy=False` is used.
    """
    if not isinstance(usm_ary, dpt.usm_ndarray):
        return TypeError(
            f"Expected object of type dpt.usm_ndarray, got {type(usm_ary)}"
        )
    if len(order) == 0 or order[0] not in "KkAaCcFf":
        raise ValueError(
            "Unrecognized order keyword value, expecting 'K', 'A', 'F', or 'C'."
        )
    order = order[0].upper()
    ary_dtype = usm_ary.dtype
    if device is not None:
        if not isinstance(device, dpctl.SyclQueue):
            if isinstance(device, dpt.Device):
                device = device.sycl_queue
            else:
                device = dpt.Device.create_device(device).sycl_queue
        d = device.sycl_device
        target_dtype = _get_dtype(newdtype, device)
        if not _dtype_supported_by_device_impl(
            target_dtype, d.has_aspect_fp16, d.has_aspect_fp64
        ):
            raise ValueError(
                f"Requested dtype '{target_dtype}' is not supported by the "
                "target device"
            )
        usm_ary = usm_ary.to_device(device)
    else:
        target_dtype = _get_dtype(newdtype, usm_ary.sycl_queue)

    if not dpt.can_cast(ary_dtype, target_dtype, casting=casting):
        raise TypeError(
            f"Can not cast from {ary_dtype} to {newdtype} "
            f"according to rule {casting}."
        )
    c_contig = usm_ary.flags.c_contiguous
    f_contig = usm_ary.flags.f_contiguous
    needs_copy = copy or not ary_dtype == target_dtype
    if not needs_copy and (order != "K"):
        # ensure that order="F" for C-contig input triggers copy,
        # and order="C" for F-contig input triggers copy too.
        # 1D arrays which are both C- and F- contig should not
        # force copying for neither order="F", nor order="C", see gh-1926
        needs_copy = (
            c_contig and not f_contig and order not in ["A", "C"]
        ) or (not c_contig and f_contig and order not in ["A", "F"])
    if not needs_copy:
        return usm_ary
    copy_order = "C"
    if order == "C":
        pass
    elif order == "F":
        copy_order = order
    elif order == "A":
        if usm_ary.flags.f_contiguous:
            copy_order = "F"
    elif order == "K":
        if usm_ary.flags.f_contiguous:
            copy_order = "F"
    else:
        raise ValueError(
            "Unrecognized value of the order keyword. "
            "Recognized values are 'A', 'C', 'F', or 'K'"
        )
    if order == "K":
        R = _empty_like_orderK(usm_ary, target_dtype)
    else:
        R = dpt.usm_ndarray(
            usm_ary.shape,
            dtype=target_dtype,
            buffer=usm_ary.usm_type,
            order=copy_order,
            buffer_ctor_kwargs={"queue": usm_ary.sycl_queue},
        )
    _copy_from_usm_ndarray_to_usm_ndarray(R, usm_ary)
    return R
