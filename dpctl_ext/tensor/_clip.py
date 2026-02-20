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
import dpctl.tensor._tensor_elementwise_impl as tei
from dpctl.utils import ExecutionPlacementError, SequentialOrderManager

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt_ext
import dpctl_ext.tensor._tensor_impl as ti
from dpctl_ext.tensor._copy_utils import (
    _empty_like_orderK,
    _empty_like_pair_orderK,
    _empty_like_triple_orderK,
)
from dpctl_ext.tensor._manipulation_functions import _broadcast_shape_impl
from dpctl_ext.tensor._type_utils import _can_cast

from ._scalar_utils import (
    _get_dtype,
    _get_queue_usm_type,
    _get_shape,
    _validate_dtype,
)
from ._type_utils import (
    _resolve_one_strong_one_weak_types,
    _resolve_one_strong_two_weak_types,
)


def _check_clip_dtypes(res_dtype, arg1_dtype, arg2_dtype, sycl_dev):
    """
    Checks if both types `arg1_dtype` and `arg2_dtype` can be
    cast to `res_dtype` according to the rule `safe`
    """
    if arg1_dtype == res_dtype and arg2_dtype == res_dtype:
        return None, None, res_dtype

    _fp16 = sycl_dev.has_aspect_fp16
    _fp64 = sycl_dev.has_aspect_fp64
    if _can_cast(arg1_dtype, res_dtype, _fp16, _fp64) and _can_cast(
        arg2_dtype, res_dtype, _fp16, _fp64
    ):
        # prevent unnecessary casting
        ret_buf1_dt = None if res_dtype == arg1_dtype else res_dtype
        ret_buf2_dt = None if res_dtype == arg2_dtype else res_dtype
        return ret_buf1_dt, ret_buf2_dt, res_dtype
    else:
        return None, None, None


def _clip_none(x, val, out, order, _binary_fn):
    q1, x_usm_type = x.sycl_queue, x.usm_type
    q2, val_usm_type = _get_queue_usm_type(val)
    if q2 is None:
        exec_q = q1
        res_usm_type = x_usm_type
    else:
        exec_q = dpctl.utils.get_execution_queue((q1, q2))
        if exec_q is None:
            raise ExecutionPlacementError(
                "Execution placement can not be unambiguously inferred "
                "from input arguments."
            )
        res_usm_type = dpctl.utils.get_coerced_usm_type(
            (
                x_usm_type,
                val_usm_type,
            )
        )
    dpctl.utils.validate_usm_type(res_usm_type, allow_none=False)
    x_shape = x.shape
    val_shape = _get_shape(val)
    if not isinstance(val_shape, (tuple, list)):
        raise TypeError(
            "Shape of arguments can not be inferred. "
            "Arguments are expected to be "
            "lists, tuples, or both"
        )
    try:
        res_shape = _broadcast_shape_impl(
            [
                x_shape,
                val_shape,
            ]
        )
    except ValueError:
        raise ValueError(
            "operands could not be broadcast together with shapes "
            f"{x_shape} and {val_shape}"
        )
    sycl_dev = exec_q.sycl_device
    x_dtype = x.dtype
    val_dtype = _get_dtype(val, sycl_dev)
    if not _validate_dtype(val_dtype):
        raise ValueError("Operands have unsupported data types")

    val_dtype = _resolve_one_strong_one_weak_types(x_dtype, val_dtype, sycl_dev)

    res_dt = x.dtype
    _fp16 = sycl_dev.has_aspect_fp16
    _fp64 = sycl_dev.has_aspect_fp64
    if not _can_cast(val_dtype, res_dt, _fp16, _fp64):
        raise ValueError(
            f"function 'clip' does not support input types "
            f"({x_dtype}, {val_dtype}), "
            "and the inputs could not be safely coerced to any "
            "supported types according to the casting rule ''safe''."
        )

    orig_out = out
    if out is not None:
        if not isinstance(out, dpt.usm_ndarray):
            raise TypeError(
                f"output array must be of usm_ndarray type, got {type(out)}"
            )

        if not out.flags.writable:
            raise ValueError("provided `out` array is read-only")

        if out.shape != res_shape:
            raise ValueError(
                "The shape of input and output arrays are inconsistent. "
                f"Expected output shape is {res_shape}, got {out.shape}"
            )

        if res_dt != out.dtype:
            raise ValueError(
                f"Output array of type {res_dt} is needed, got {out.dtype}"
            )

        if dpctl.utils.get_execution_queue((exec_q, out.sycl_queue)) is None:
            raise ExecutionPlacementError(
                "Input and output allocation queues are not compatible"
            )

        if ti._array_overlap(x, out):
            if not ti._same_logical_tensors(x, out):
                out = dpt.empty_like(out)

        if isinstance(val, dpt.usm_ndarray):
            if (
                ti._array_overlap(val, out)
                and not ti._same_logical_tensors(val, out)
                and val_dtype == res_dt
            ):
                out = dpt.empty_like(out)

    if isinstance(val, dpt.usm_ndarray):
        val_ary = val
    else:
        val_ary = dpt.asarray(val, dtype=val_dtype, sycl_queue=exec_q)

    if order == "A":
        order = (
            "F"
            if all(
                arr.flags.f_contiguous
                for arr in (
                    x,
                    val_ary,
                )
            )
            else "C"
        )
    if val_dtype == res_dt:
        if out is None:
            if order == "K":
                out = _empty_like_pair_orderK(
                    x, val_ary, res_dt, res_shape, res_usm_type, exec_q
                )
            else:
                out = dpt.empty(
                    res_shape,
                    dtype=res_dt,
                    usm_type=res_usm_type,
                    sycl_queue=exec_q,
                    order=order,
                )
        if x_shape != res_shape:
            x = dpt.broadcast_to(x, res_shape)
        if val_ary.shape != res_shape:
            val_ary = dpt.broadcast_to(val_ary, res_shape)
        _manager = SequentialOrderManager[exec_q]
        dep_evs = _manager.submitted_events
        ht_binary_ev, binary_ev = _binary_fn(
            src1=x, src2=val_ary, dst=out, sycl_queue=exec_q, depends=dep_evs
        )
        _manager.add_event_pair(ht_binary_ev, binary_ev)
        if not (orig_out is None or orig_out is out):
            # Copy the out data from temporary buffer to original memory
            ht_copy_out_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                src=out,
                dst=orig_out,
                sycl_queue=exec_q,
                depends=[binary_ev],
            )
            _manager.add_event_pair(ht_copy_out_ev, copy_ev)
            out = orig_out
        return out
    else:
        if order == "K":
            buf = _empty_like_orderK(val_ary, res_dt)
        else:
            buf = dpt.empty_like(val_ary, dtype=res_dt, order=order)
        _manager = SequentialOrderManager[exec_q]
        dep_evs = _manager.submitted_events
        ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=val_ary, dst=buf, sycl_queue=exec_q, depends=dep_evs
        )
        _manager.add_event_pair(ht_copy_ev, copy_ev)
        if out is None:
            if order == "K":
                out = _empty_like_pair_orderK(
                    x, buf, res_dt, res_shape, res_usm_type, exec_q
                )
            else:
                out = dpt.empty(
                    res_shape,
                    dtype=res_dt,
                    usm_type=res_usm_type,
                    sycl_queue=exec_q,
                    order=order,
                )

        if x_shape != res_shape:
            x = dpt.broadcast_to(x, res_shape)
        buf = dpt.broadcast_to(buf, res_shape)
        ht_binary_ev, binary_ev = _binary_fn(
            src1=x,
            src2=buf,
            dst=out,
            sycl_queue=exec_q,
            depends=[copy_ev],
        )
        _manager.add_event_pair(ht_binary_ev, binary_ev)
        if not (orig_out is None or orig_out is out):
            # Copy the out data from temporary buffer to original memory
            ht_copy_out_ev, cpy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                src=out,
                dst=orig_out,
                sycl_queue=exec_q,
                depends=[binary_ev],
            )
            _manager.add_event_pair(ht_copy_out_ev, cpy_ev)
            out = orig_out
        return out


def clip(x, /, min=None, max=None, out=None, order="K"):
    """clip(x, min=None, max=None, out=None, order="K")

    Clips to the range [`min_i`, `max_i`] for each element `x_i`
    in `x`.

    Args:
        x (usm_ndarray): Array containing elements to clip.
            Must be compatible with `min` and `max` according
            to broadcasting rules.
        min ({None, Union[usm_ndarray, bool, int, float, complex]}, optional):
            Array containing minimum values.
            Must be compatible with `x` and `max` according
            to broadcasting rules.
        max ({None, Union[usm_ndarray, bool, int, float, complex]}, optional):
            Array containing maximum values.
            Must be compatible with `x` and `min` according
            to broadcasting rules.
        out ({None, usm_ndarray}, optional):
            Output array to populate.
            Array must have the correct shape and the expected data type.
        order ("C","F","A","K", optional):
            Memory layout of the newly output array, if parameter `out` is
            `None`.
            Default: "K".

    Returns:
        usm_ndarray:
            An array with elements clipped to the range [`min`, `max`].
            The returned array has the same data type as `x`.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(
            "Expected `x` to be of dpctl.tensor.usm_ndarray type, got "
            f"{type(x)}"
        )
    if order not in ["K", "C", "F", "A"]:
        order = "K"
    if x.dtype.kind in "iu":
        if isinstance(min, int) and min <= dpt_ext.iinfo(x.dtype).min:
            min = None
        if isinstance(max, int) and max >= dpt_ext.iinfo(x.dtype).max:
            max = None
    if min is None and max is None:
        exec_q = x.sycl_queue
        orig_out = out
        if out is not None:
            if not isinstance(out, dpt.usm_ndarray):
                raise TypeError(
                    "output array must be of usm_ndarray type, got "
                    f"{type(out)}"
                )

            if not out.flags.writable:
                raise ValueError("provided `out` array is read-only")

            if out.shape != x.shape:
                raise ValueError(
                    "The shape of input and output arrays are "
                    f"inconsistent. Expected output shape is {x.shape}, "
                    f"got {out.shape}"
                )

            if x.dtype != out.dtype:
                raise ValueError(
                    f"Output array of type {x.dtype} is needed, "
                    f"got {out.dtype}"
                )

            if (
                dpctl.utils.get_execution_queue((exec_q, out.sycl_queue))
                is None
            ):
                raise ExecutionPlacementError(
                    "Input and output allocation queues are not compatible"
                )

            if ti._array_overlap(x, out):
                if not ti._same_logical_tensors(x, out):
                    out = dpt.empty_like(out)
                else:
                    return out
        else:
            if order == "K":
                out = _empty_like_orderK(x, x.dtype)
            else:
                out = dpt.empty_like(x, order=order)

        _manager = SequentialOrderManager[exec_q]
        dep_evs = _manager.submitted_events
        ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=x, dst=out, sycl_queue=exec_q, depends=dep_evs
        )
        _manager.add_event_pair(ht_copy_ev, copy_ev)
        if not (orig_out is None or orig_out is out):
            # Copy the out data from temporary buffer to original memory
            ht_copy_out_ev, cpy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                src=out,
                dst=orig_out,
                sycl_queue=exec_q,
                depends=[copy_ev],
            )
            _manager.add_event_pair(ht_copy_ev, cpy_ev)
            out = orig_out
        return out
    elif max is None:
        return _clip_none(x, min, out, order, tei._maximum)
    elif min is None:
        return _clip_none(x, max, out, order, tei._minimum)
    else:
        q1, x_usm_type = x.sycl_queue, x.usm_type
        q2, min_usm_type = _get_queue_usm_type(min)
        q3, max_usm_type = _get_queue_usm_type(max)
        if q2 is None and q3 is None:
            exec_q = q1
            res_usm_type = x_usm_type
        elif q3 is None:
            exec_q = dpctl.utils.get_execution_queue((q1, q2))
            if exec_q is None:
                raise ExecutionPlacementError(
                    "Execution placement can not be unambiguously inferred "
                    "from input arguments."
                )
            res_usm_type = dpctl.utils.get_coerced_usm_type(
                (
                    x_usm_type,
                    min_usm_type,
                )
            )
        elif q2 is None:
            exec_q = dpctl.utils.get_execution_queue((q1, q3))
            if exec_q is None:
                raise ExecutionPlacementError(
                    "Execution placement can not be unambiguously inferred "
                    "from input arguments."
                )
            res_usm_type = dpctl.utils.get_coerced_usm_type(
                (
                    x_usm_type,
                    max_usm_type,
                )
            )
        else:
            exec_q = dpctl.utils.get_execution_queue((q1, q2, q3))
            if exec_q is None:
                raise ExecutionPlacementError(
                    "Execution placement can not be unambiguously inferred "
                    "from input arguments."
                )
            res_usm_type = dpctl.utils.get_coerced_usm_type(
                (
                    x_usm_type,
                    min_usm_type,
                    max_usm_type,
                )
            )
        dpctl.utils.validate_usm_type(res_usm_type, allow_none=False)
        x_shape = x.shape
        min_shape = _get_shape(min)
        max_shape = _get_shape(max)
        if not all(
            isinstance(s, (tuple, list))
            for s in (
                min_shape,
                max_shape,
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
                    x_shape,
                    min_shape,
                    max_shape,
                ]
            )
        except ValueError:
            raise ValueError(
                "operands could not be broadcast together with shapes "
                f"{x_shape}, {min_shape}, and {max_shape}"
            )
        sycl_dev = exec_q.sycl_device
        x_dtype = x.dtype
        min_dtype = _get_dtype(min, sycl_dev)
        max_dtype = _get_dtype(max, sycl_dev)
        if not all(_validate_dtype(o) for o in (min_dtype, max_dtype)):
            raise ValueError("Operands have unsupported data types")

        min_dtype, max_dtype = _resolve_one_strong_two_weak_types(
            x_dtype, min_dtype, max_dtype, sycl_dev
        )

        buf1_dt, buf2_dt, res_dt = _check_clip_dtypes(
            x_dtype,
            min_dtype,
            max_dtype,
            sycl_dev,
        )

        if res_dt is None:
            raise ValueError(
                f"function '{clip}' does not support input types "
                f"({x_dtype}, {min_dtype}, {max_dtype}), "
                "and the inputs could not be safely coerced to any "
                "supported types according to the casting rule ''safe''."
            )

        orig_out = out
        if out is not None:
            if not isinstance(out, dpt.usm_ndarray):
                raise TypeError(
                    "output array must be of usm_ndarray type, got "
                    f"{type(out)}"
                )

            if not out.flags.writable:
                raise ValueError("provided `out` array is read-only")

            if out.shape != res_shape:
                raise ValueError(
                    "The shape of input and output arrays are "
                    f"inconsistent. Expected output shape is {res_shape}, "
                    f"got {out.shape}"
                )

            if res_dt != out.dtype:
                raise ValueError(
                    f"Output array of type {res_dt} is needed, "
                    f"got {out.dtype}"
                )

            if (
                dpctl.utils.get_execution_queue((exec_q, out.sycl_queue))
                is None
            ):
                raise ExecutionPlacementError(
                    "Input and output allocation queues are not compatible"
                )

            if ti._array_overlap(x, out):
                if not ti._same_logical_tensors(x, out):
                    out = dpt.empty_like(out)

            if isinstance(min, dpt.usm_ndarray):
                if (
                    ti._array_overlap(min, out)
                    and not ti._same_logical_tensors(min, out)
                    and buf1_dt is None
                ):
                    out = dpt.empty_like(out)

            if isinstance(max, dpt.usm_ndarray):
                if (
                    ti._array_overlap(max, out)
                    and not ti._same_logical_tensors(max, out)
                    and buf2_dt is None
                ):
                    out = dpt.empty_like(out)

        if isinstance(min, dpt.usm_ndarray):
            a_min = min
        else:
            a_min = dpt.asarray(min, dtype=min_dtype, sycl_queue=exec_q)
        if isinstance(max, dpt.usm_ndarray):
            a_max = max
        else:
            a_max = dpt.asarray(max, dtype=max_dtype, sycl_queue=exec_q)

        if order == "A":
            order = (
                "F"
                if all(
                    arr.flags.f_contiguous
                    for arr in (
                        x,
                        a_min,
                        a_max,
                    )
                )
                else "C"
            )
        if buf1_dt is None and buf2_dt is None:
            if out is None:
                if order == "K":
                    out = _empty_like_triple_orderK(
                        x,
                        a_min,
                        a_max,
                        res_dt,
                        res_shape,
                        res_usm_type,
                        exec_q,
                    )
                else:
                    out = dpt.empty(
                        res_shape,
                        dtype=res_dt,
                        usm_type=res_usm_type,
                        sycl_queue=exec_q,
                        order=order,
                    )
            if x_shape != res_shape:
                x = dpt.broadcast_to(x, res_shape)
            if a_min.shape != res_shape:
                a_min = dpt.broadcast_to(a_min, res_shape)
            if a_max.shape != res_shape:
                a_max = dpt.broadcast_to(a_max, res_shape)
            _manager = SequentialOrderManager[exec_q]
            dep_ev = _manager.submitted_events
            ht_binary_ev, binary_ev = ti._clip(
                src=x,
                min=a_min,
                max=a_max,
                dst=out,
                sycl_queue=exec_q,
                depends=dep_ev,
            )
            _manager.add_event_pair(ht_binary_ev, binary_ev)
            if not (orig_out is None or orig_out is out):
                # Copy the out data from temporary buffer to original memory
                ht_copy_out_ev, cpy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                    src=out,
                    dst=orig_out,
                    sycl_queue=exec_q,
                    depends=[binary_ev],
                )
                _manager.add_event_pair(ht_copy_out_ev, cpy_ev)
                out = orig_out
            return out

        elif buf1_dt is None:
            if order == "K":
                buf2 = _empty_like_orderK(a_max, buf2_dt)
            else:
                buf2 = dpt.empty_like(a_max, dtype=buf2_dt, order=order)
            _manager = SequentialOrderManager[exec_q]
            dep_ev = _manager.submitted_events
            ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                src=a_max, dst=buf2, sycl_queue=exec_q, depends=dep_ev
            )
            _manager.add_event_pair(ht_copy_ev, copy_ev)
            if out is None:
                if order == "K":
                    out = _empty_like_triple_orderK(
                        x,
                        a_min,
                        buf2,
                        res_dt,
                        res_shape,
                        res_usm_type,
                        exec_q,
                    )
                else:
                    out = dpt.empty(
                        res_shape,
                        dtype=res_dt,
                        usm_type=res_usm_type,
                        sycl_queue=exec_q,
                        order=order,
                    )

            x = dpt.broadcast_to(x, res_shape)
            if a_min.shape != res_shape:
                a_min = dpt.broadcast_to(a_min, res_shape)
            buf2 = dpt.broadcast_to(buf2, res_shape)
            ht_binary_ev, binary_ev = ti._clip(
                src=x,
                min=a_min,
                max=buf2,
                dst=out,
                sycl_queue=exec_q,
                depends=[copy_ev],
            )
            _manager.add_event_pair(ht_binary_ev, binary_ev)
            if not (orig_out is None or orig_out is out):
                # Copy the out data from temporary buffer to original memory
                ht_copy_out_ev, cpy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                    src=out,
                    dst=orig_out,
                    sycl_queue=exec_q,
                    depends=[binary_ev],
                )
                _manager.add_event_pair(ht_copy_out_ev, cpy_ev)
                out = orig_out
            return out

        elif buf2_dt is None:
            if order == "K":
                buf1 = _empty_like_orderK(a_min, buf1_dt)
            else:
                buf1 = dpt.empty_like(a_min, dtype=buf1_dt, order=order)
            _manager = SequentialOrderManager[exec_q]
            dep_ev = _manager.submitted_events
            ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                src=a_min, dst=buf1, sycl_queue=exec_q, depends=dep_ev
            )
            _manager.add_event_pair(ht_copy_ev, copy_ev)
            if out is None:
                if order == "K":
                    out = _empty_like_triple_orderK(
                        x,
                        buf1,
                        a_max,
                        res_dt,
                        res_shape,
                        res_usm_type,
                        exec_q,
                    )
                else:
                    out = dpt.empty(
                        res_shape,
                        dtype=res_dt,
                        usm_type=res_usm_type,
                        sycl_queue=exec_q,
                        order=order,
                    )

            x = dpt.broadcast_to(x, res_shape)
            buf1 = dpt.broadcast_to(buf1, res_shape)
            if a_max.shape != res_shape:
                a_max = dpt.broadcast_to(a_max, res_shape)
            ht_binary_ev, binary_ev = ti._clip(
                src=x,
                min=buf1,
                max=a_max,
                dst=out,
                sycl_queue=exec_q,
                depends=[copy_ev],
            )
            _manager.add_event_pair(ht_binary_ev, binary_ev)
            if not (orig_out is None or orig_out is out):
                # Copy the out data from temporary buffer to original memory
                ht_copy_out_ev, cpy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                    src=out,
                    dst=orig_out,
                    sycl_queue=exec_q,
                    depends=[binary_ev],
                )
                _manager.add_event_pair(ht_copy_out_ev, cpy_ev)
                out = orig_out
            return out

        if order == "K":
            if (
                x.flags.c_contiguous
                and a_min.flags.c_contiguous
                and a_max.flags.c_contiguous
            ):
                order = "C"
            elif (
                x.flags.f_contiguous
                and a_min.flags.f_contiguous
                and a_max.flags.f_contiguous
            ):
                order = "F"
        if order == "K":
            buf1 = _empty_like_orderK(a_min, buf1_dt)
        else:
            buf1 = dpt.empty_like(a_min, dtype=buf1_dt, order=order)

        _manager = SequentialOrderManager[exec_q]
        dep_evs = _manager.submitted_events
        ht_copy1_ev, copy1_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=a_min, dst=buf1, sycl_queue=exec_q, depends=dep_evs
        )
        _manager.add_event_pair(ht_copy1_ev, copy1_ev)
        if order == "K":
            buf2 = _empty_like_orderK(a_max, buf2_dt)
        else:
            buf2 = dpt.empty_like(a_max, dtype=buf2_dt, order=order)
        ht_copy2_ev, copy2_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=a_max, dst=buf2, sycl_queue=exec_q, depends=dep_evs
        )
        _manager.add_event_pair(ht_copy2_ev, copy2_ev)
        if out is None:
            if order == "K":
                out = _empty_like_triple_orderK(
                    x, buf1, buf2, res_dt, res_shape, res_usm_type, exec_q
                )
            else:
                out = dpt.empty(
                    res_shape,
                    dtype=res_dt,
                    usm_type=res_usm_type,
                    sycl_queue=exec_q,
                    order=order,
                )

        x = dpt.broadcast_to(x, res_shape)
        buf1 = dpt.broadcast_to(buf1, res_shape)
        buf2 = dpt.broadcast_to(buf2, res_shape)
        ht_, clip_ev = ti._clip(
            src=x,
            min=buf1,
            max=buf2,
            dst=out,
            sycl_queue=exec_q,
            depends=[copy1_ev, copy2_ev],
        )
        _manager.add_event_pair(ht_, clip_ev)
        return out
