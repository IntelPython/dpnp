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

import dpctl
import dpctl.tensor as dpt
from dpctl.utils import ExecutionPlacementError, SequentialOrderManager

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt_ext
import dpctl_ext.tensor._tensor_impl as ti
from dpctl_ext.tensor._manipulation_functions import _broadcast_shape_impl

from ._copy_utils import _empty_like_orderK, _empty_like_triple_orderK
from ._scalar_utils import (
    _get_dtype,
    _get_queue_usm_type,
    _get_shape,
    _validate_dtype,
)
from ._type_utils import (
    WeakBooleanType,
    WeakComplexType,
    WeakFloatingType,
    WeakIntegralType,
    _all_data_types,
    _can_cast,
    _is_weak_dtype,
    _strong_dtype_num_kind,
    _to_device_supported_dtype,
    _weak_type_num_kind,
)


def _default_dtype_from_weak_type(dt, dev):
    if isinstance(dt, WeakBooleanType):
        return dpt.bool
    if isinstance(dt, WeakIntegralType):
        return dpt.dtype(ti.default_device_int_type(dev))
    if isinstance(dt, WeakFloatingType):
        return dpt.dtype(ti.default_device_fp_type(dev))
    if isinstance(dt, WeakComplexType):
        return dpt.dtype(ti.default_device_complex_type(dev))


def _resolve_two_weak_types(o1_dtype, o2_dtype, dev):
    """Resolves two weak data types per NEP-0050"""
    if _is_weak_dtype(o1_dtype):
        if _is_weak_dtype(o2_dtype):
            return _default_dtype_from_weak_type(
                o1_dtype, dev
            ), _default_dtype_from_weak_type(o2_dtype, dev)
        o1_kind_num = _weak_type_num_kind(o1_dtype)
        o2_kind_num = _strong_dtype_num_kind(o2_dtype)
        if o1_kind_num > o2_kind_num:
            if isinstance(o1_dtype, WeakIntegralType):
                return dpt.dtype(ti.default_device_int_type(dev)), o2_dtype
            if isinstance(o1_dtype, WeakComplexType):
                if o2_dtype is dpt.float16 or o2_dtype is dpt.float32:
                    return dpt.complex64, o2_dtype
                return (
                    _to_device_supported_dtype(dpt.complex128, dev),
                    o2_dtype,
                )
            return _to_device_supported_dtype(dpt.float64, dev), o2_dtype
        else:
            return o2_dtype, o2_dtype
    elif _is_weak_dtype(o2_dtype):
        o1_kind_num = _strong_dtype_num_kind(o1_dtype)
        o2_kind_num = _weak_type_num_kind(o2_dtype)
        if o2_kind_num > o1_kind_num:
            if isinstance(o2_dtype, WeakIntegralType):
                return o1_dtype, dpt.dtype(ti.default_device_int_type(dev))
            if isinstance(o2_dtype, WeakComplexType):
                if o1_dtype is dpt.float16 or o1_dtype is dpt.float32:
                    return o1_dtype, dpt.complex64
                return o1_dtype, _to_device_supported_dtype(dpt.complex128, dev)
            return (
                o1_dtype,
                _to_device_supported_dtype(dpt.float64, dev),
            )
        else:
            return o1_dtype, o1_dtype
    else:
        return o1_dtype, o2_dtype


def _where_result_type(dt1, dt2, dev):
    res_dtype = dpt_ext.result_type(dt1, dt2)
    fp16 = dev.has_aspect_fp16
    fp64 = dev.has_aspect_fp64

    all_dts = _all_data_types(fp16, fp64)
    if res_dtype in all_dts:
        return res_dtype
    else:
        for res_dtype_ in all_dts:
            if _can_cast(dt1, res_dtype_, fp16, fp64) and _can_cast(
                dt2, res_dtype_, fp16, fp64
            ):
                return res_dtype_
        return None


def where(condition, x1, x2, /, *, order="K", out=None):
    """
    Returns :class:`dpctl.tensor.usm_ndarray` with elements chosen
    from ``x1`` or ``x2`` depending on ``condition``.

    Args:
        condition (usm_ndarray): When ``True`` yields from ``x1``,
            and otherwise yields from ``x2``.
            Must be compatible with ``x1`` and ``x2`` according
            to broadcasting rules.
        x1 (Union[usm_ndarray, bool, int, float, complex]):
            Array from which values are chosen when ``condition`` is ``True``.
            Must be compatible with ``condition`` and ``x2`` according
            to broadcasting rules.
        x2 (Union[usm_ndarray, bool, int, float, complex]):
            Array from which values are chosen when ``condition`` is not
            ``True``.
            Must be compatible with ``condition`` and ``x2`` according
            to broadcasting rules.
        order (``"K"``, ``"C"``, ``"F"``, ``"A"``, optional):
            Memory layout of the new output array,
            if parameter ``out`` is ``None``.
            Default: ``"K"``.
        out (Optional[usm_ndarray]):
            the array into which the result is written.
            The data type of `out` must match the expected shape and the
            expected data type of the result.
            If ``None`` then a new array is returned. Default: ``None``.

    Returns:
        usm_ndarray:
            An array with elements from ``x1`` where ``condition`` is ``True``,
            and elements from ``x2`` elsewhere.

    The data type of the returned array is determined by applying
    the Type Promotion Rules to ``x1`` and ``x2``.
    """
    if not isinstance(condition, dpt.usm_ndarray):
        raise TypeError(
            "Expecting dpctl.tensor.usm_ndarray type, " f"got {type(condition)}"
        )
    if order not in ["K", "C", "F", "A"]:
        order = "K"
    q1, condition_usm_type = condition.sycl_queue, condition.usm_type
    q2, x1_usm_type = _get_queue_usm_type(x1)
    q3, x2_usm_type = _get_queue_usm_type(x2)
    if q2 is None and q3 is None:
        exec_q = q1
        out_usm_type = condition_usm_type
    elif q3 is None:
        exec_q = dpctl.utils.get_execution_queue((q1, q2))
        if exec_q is None:
            raise ExecutionPlacementError(
                "Execution placement can not be unambiguously inferred "
                "from input arguments."
            )
        out_usm_type = dpctl.utils.get_coerced_usm_type(
            (
                condition_usm_type,
                x1_usm_type,
            )
        )
    elif q2 is None:
        exec_q = dpctl.utils.get_execution_queue((q1, q3))
        if exec_q is None:
            raise ExecutionPlacementError(
                "Execution placement can not be unambiguously inferred "
                "from input arguments."
            )
        out_usm_type = dpctl.utils.get_coerced_usm_type(
            (
                condition_usm_type,
                x2_usm_type,
            )
        )
    else:
        exec_q = dpctl.utils.get_execution_queue((q1, q2, q3))
        if exec_q is None:
            raise ExecutionPlacementError(
                "Execution placement can not be unambiguously inferred "
                "from input arguments."
            )
        out_usm_type = dpctl.utils.get_coerced_usm_type(
            (
                condition_usm_type,
                x1_usm_type,
                x2_usm_type,
            )
        )
    dpctl.utils.validate_usm_type(out_usm_type, allow_none=False)
    condition_shape = condition.shape
    x1_shape = _get_shape(x1)
    x2_shape = _get_shape(x2)
    if not all(
        isinstance(s, (tuple, list))
        for s in (
            x1_shape,
            x2_shape,
        )
    ):
        raise TypeError(
            "Shape of arguments can not be inferred. "
            "Arguments are expected to be "
            "lists, tuples, or both"
        )
    try:
        res_shape = _broadcast_shape_impl(
            [
                condition_shape,
                x1_shape,
                x2_shape,
            ]
        )
    except ValueError:
        raise ValueError(
            "operands could not be broadcast together with shapes "
            f"{condition_shape}, {x1_shape}, and {x2_shape}"
        )
    sycl_dev = exec_q.sycl_device
    x1_dtype = _get_dtype(x1, sycl_dev)
    x2_dtype = _get_dtype(x2, sycl_dev)
    if not all(_validate_dtype(o) for o in (x1_dtype, x2_dtype)):
        raise ValueError("Operands have unsupported data types")
    x1_dtype, x2_dtype = _resolve_two_weak_types(x1_dtype, x2_dtype, sycl_dev)
    out_dtype = _where_result_type(x1_dtype, x2_dtype, sycl_dev)
    if out_dtype is None:
        raise TypeError(
            "function 'where' does not support input "
            f"types ({x1_dtype}, {x2_dtype}), "
            "and the inputs could not be safely coerced "
            "to any supported types according to the casting rule ''safe''."
        )

    orig_out = out
    if out is not None:
        if not isinstance(out, dpt.usm_ndarray):
            raise TypeError(
                "output array must be of usm_ndarray type, got " f"{type(out)}"
            )

        if not out.flags.writable:
            raise ValueError("provided `out` array is read-only")

        if out.shape != res_shape:
            raise ValueError(
                "The shape of input and output arrays are "
                f"inconsistent. Expected output shape is {res_shape}, "
                f"got {out.shape}"
            )

        if out_dtype != out.dtype:
            raise ValueError(
                f"Output array of type {out_dtype} is needed, "
                f"got {out.dtype}"
            )

        if dpctl.utils.get_execution_queue((exec_q, out.sycl_queue)) is None:
            raise ExecutionPlacementError(
                "Input and output allocation queues are not compatible"
            )

        if ti._array_overlap(condition, out) and not ti._same_logical_tensors(
            condition, out
        ):
            out = dpt.empty_like(out)

        if isinstance(x1, dpt.usm_ndarray):
            if (
                ti._array_overlap(x1, out)
                and not ti._same_logical_tensors(x1, out)
                and x1_dtype == out_dtype
            ):
                out = dpt.empty_like(out)

        if isinstance(x2, dpt.usm_ndarray):
            if (
                ti._array_overlap(x2, out)
                and not ti._same_logical_tensors(x2, out)
                and x2_dtype == out_dtype
            ):
                out = dpt.empty_like(out)

    if order == "A":
        order = (
            "F"
            if all(
                arr.flags.f_contiguous
                for arr in (
                    condition,
                    x1,
                    x2,
                )
            )
            else "C"
        )
    if not isinstance(x1, dpt.usm_ndarray):
        x1 = dpt.asarray(x1, dtype=x1_dtype, sycl_queue=exec_q)
    if not isinstance(x2, dpt.usm_ndarray):
        x2 = dpt.asarray(x2, dtype=x2_dtype, sycl_queue=exec_q)

    if condition.size == 0:
        if out is not None:
            return out
        else:
            if order == "K":
                return _empty_like_triple_orderK(
                    condition,
                    x1,
                    x2,
                    out_dtype,
                    res_shape,
                    out_usm_type,
                    exec_q,
                )
            else:
                return dpt.empty(
                    res_shape,
                    dtype=out_dtype,
                    order=order,
                    usm_type=out_usm_type,
                    sycl_queue=exec_q,
                )

    _manager = SequentialOrderManager[exec_q]
    dep_evs = _manager.submitted_events
    if x1_dtype != out_dtype:
        if order == "K":
            _x1 = _empty_like_orderK(x1, out_dtype)
        else:
            _x1 = dpt.empty_like(x1, dtype=out_dtype, order=order)
        ht_copy1_ev, copy1_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=x1, dst=_x1, sycl_queue=exec_q, depends=dep_evs
        )
        x1 = _x1
        _manager.add_event_pair(ht_copy1_ev, copy1_ev)

    if x2_dtype != out_dtype:
        if order == "K":
            _x2 = _empty_like_orderK(x2, out_dtype)
        else:
            _x2 = dpt.empty_like(x2, dtype=out_dtype, order=order)
        ht_copy2_ev, copy2_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=x2, dst=_x2, sycl_queue=exec_q, depends=dep_evs
        )
        x2 = _x2
        _manager.add_event_pair(ht_copy2_ev, copy2_ev)

    if out is None:
        if order == "K":
            out = _empty_like_triple_orderK(
                condition, x1, x2, out_dtype, res_shape, out_usm_type, exec_q
            )
        else:
            out = dpt.empty(
                res_shape,
                dtype=out_dtype,
                order=order,
                usm_type=out_usm_type,
                sycl_queue=exec_q,
            )

    if condition_shape != res_shape:
        condition = dpt.broadcast_to(condition, res_shape)
    if x1_shape != res_shape:
        x1 = dpt.broadcast_to(x1, res_shape)
    if x2_shape != res_shape:
        x2 = dpt.broadcast_to(x2, res_shape)

    dep_evs = _manager.submitted_events
    hev, where_ev = ti._where(
        condition=condition,
        x1=x1,
        x2=x2,
        dst=out,
        sycl_queue=exec_q,
        depends=dep_evs,
    )
    _manager.add_event_pair(hev, where_ev)
    if not (orig_out is None or orig_out is out):
        # Copy the out data from temporary buffer to original memory
        ht_copy_out_ev, cpy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=out,
            dst=orig_out,
            sycl_queue=exec_q,
            depends=[where_ev],
        )
        _manager.add_event_pair(ht_copy_out_ev, cpy_ev)
        out = orig_out

    return out
