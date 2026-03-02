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
import dpctl_ext.tensor._tensor_accumulation_impl as tai
import dpctl_ext.tensor._tensor_impl as ti
from dpctl_ext.tensor._type_utils import (
    _default_accumulation_dtype,
    _default_accumulation_dtype_fp_types,
    _to_device_supported_dtype,
)

from ._numpy_helper import normalize_axis_index


def _accumulate_common(
    x,
    axis,
    dtype,
    include_initial,
    out,
    _accumulate_fn,
    _accumulate_include_initial_fn,
    _dtype_supported,
    _default_accumulation_type_fn,
):
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x)}")
    appended_axis = False
    if x.ndim == 0:
        x = x[dpt.newaxis]
        appended_axis = True
    nd = x.ndim
    if axis is None:
        if nd > 1:
            raise ValueError(
                "`axis` cannot be `None` for array of dimension `{}`".format(nd)
            )
        axis = 0
    else:
        axis = normalize_axis_index(axis, nd, "axis")
    sh = x.shape
    res_sh = (
        sh[:axis] + (sh[axis] + 1,) + sh[axis + 1 :] if include_initial else sh
    )
    a1 = axis + 1
    if a1 == nd:
        perm = list(range(nd))
        arr = x
    else:
        perm = [i for i in range(nd) if i != axis] + [
            axis,
        ]
        arr = dpt_ext.permute_dims(x, perm)
    q = x.sycl_queue
    inp_dt = x.dtype
    res_usm_type = x.usm_type
    if dtype is None:
        res_dt = _default_accumulation_type_fn(inp_dt, q)
    else:
        res_dt = dpt.dtype(dtype)
        res_dt = _to_device_supported_dtype(res_dt, q.sycl_device)

    # checking now avoids unnecessary allocations
    implemented_types = _dtype_supported(inp_dt, res_dt)
    if dtype is None and not implemented_types:
        raise RuntimeError(
            "Automatically determined accumulation data type does not "
            "have direct implementation"
        )
    orig_out = out
    if out is not None:
        if not isinstance(out, dpt.usm_ndarray):
            raise TypeError(
                f"output array must be of usm_ndarray type, got {type(out)}"
            )
        if not out.flags.writable:
            raise ValueError("provided `out` array is read-only")
        out_sh = out.shape
        # append an axis to `out` if scalar
        if appended_axis and not include_initial:
            out = out[dpt.newaxis, ...]
            orig_out = out
            final_res_sh = res_sh[1:]
        else:
            final_res_sh = res_sh
        if not out_sh == final_res_sh:
            raise ValueError(
                "The shape of input and output arrays are inconsistent. "
                f"Expected output shape is {final_res_sh}, got {out_sh}"
            )
        if res_dt != out.dtype:
            raise ValueError(
                f"Output array of type {res_dt} is needed, " f"got {out.dtype}"
            )
        if dpctl.utils.get_execution_queue((q, out.sycl_queue)) is None:
            raise ExecutionPlacementError(
                "Input and output allocation queues are not compatible"
            )
        # permute out array dims if necessary
        if a1 != nd:
            out = dpt_ext.permute_dims(out, perm)
            orig_out = out
        if ti._array_overlap(x, out) and implemented_types:
            out = dpt_ext.empty_like(out)
    else:
        out = dpt_ext.empty(
            res_sh, dtype=res_dt, usm_type=res_usm_type, sycl_queue=q
        )
        if a1 != nd:
            out = dpt_ext.permute_dims(out, perm)

    _manager = SequentialOrderManager[q]
    depends = _manager.submitted_events
    if implemented_types:
        if not include_initial:
            ht_e, acc_ev = _accumulate_fn(
                src=arr,
                trailing_dims_to_accumulate=1,
                dst=out,
                sycl_queue=q,
                depends=depends,
            )
        else:
            ht_e, acc_ev = _accumulate_include_initial_fn(
                src=arr, dst=out, sycl_queue=q, depends=depends
            )
        _manager.add_event_pair(ht_e, acc_ev)
        if not (orig_out is None or out is orig_out):
            # Copy the out data from temporary buffer to original memory
            ht_e_cpy, cpy_e = ti._copy_usm_ndarray_into_usm_ndarray(
                src=out, dst=orig_out, sycl_queue=q, depends=[acc_ev]
            )
            _manager.add_event_pair(ht_e_cpy, cpy_e)
            out = orig_out
    else:
        if _dtype_supported(res_dt, res_dt):
            tmp = dpt_ext.empty(
                arr.shape, dtype=res_dt, usm_type=res_usm_type, sycl_queue=q
            )
            ht_e_cpy, cpy_e = ti._copy_usm_ndarray_into_usm_ndarray(
                src=arr, dst=tmp, sycl_queue=q, depends=depends
            )
            _manager.add_event_pair(ht_e_cpy, cpy_e)
            if not include_initial:
                ht_e, acc_ev = _accumulate_fn(
                    src=tmp,
                    trailing_dims_to_accumulate=1,
                    dst=out,
                    sycl_queue=q,
                    depends=[cpy_e],
                )
            else:
                ht_e, acc_ev = _accumulate_include_initial_fn(
                    src=tmp,
                    dst=out,
                    sycl_queue=q,
                    depends=[cpy_e],
                )
            _manager.add_event_pair(ht_e, acc_ev)
        else:
            buf_dt = _default_accumulation_type_fn(inp_dt, q)
            tmp = dpt_ext.empty(
                arr.shape, dtype=buf_dt, usm_type=res_usm_type, sycl_queue=q
            )
            ht_e_cpy, cpy_e = ti._copy_usm_ndarray_into_usm_ndarray(
                src=arr, dst=tmp, sycl_queue=q, depends=depends
            )
            _manager.add_event_pair(ht_e_cpy, cpy_e)
            tmp_res = dpt_ext.empty(
                res_sh, dtype=buf_dt, usm_type=res_usm_type, sycl_queue=q
            )
            if a1 != nd:
                tmp_res = dpt_ext.permute_dims(tmp_res, perm)
            if not include_initial:
                ht_e, acc_ev = _accumulate_fn(
                    src=tmp,
                    trailing_dims_to_accumulate=1,
                    dst=tmp_res,
                    sycl_queue=q,
                    depends=[cpy_e],
                )
            else:
                ht_e, acc_ev = _accumulate_include_initial_fn(
                    src=tmp,
                    dst=tmp_res,
                    sycl_queue=q,
                    depends=[cpy_e],
                )
            _manager.add_event_pair(ht_e, acc_ev)
            ht_e_cpy2, cpy_e2 = ti._copy_usm_ndarray_into_usm_ndarray(
                src=tmp_res, dst=out, sycl_queue=q, depends=[acc_ev]
            )
            _manager.add_event_pair(ht_e_cpy2, cpy_e2)

    if appended_axis:
        out = dpt_ext.squeeze(out)
    if a1 != nd:
        inv_perm = sorted(range(nd), key=lambda d: perm[d])
        out = dpt_ext.permute_dims(out, inv_perm)

    return out


def cumulative_sum(
    x, /, *, axis=None, dtype=None, include_initial=False, out=None
):
    """
    cumulative_sum(x, /, *, axis=None, dtype=None, include_initial=False,
                   out=None)

    Calculates the cumulative sum of elements in the input array `x`.

    Args:
        x (usm_ndarray):
            input array.
        axis (Optional[int]):
            axis along which cumulative sum must be computed.
            If `None`, the sum is computed over the entire array.
            If `x` is a one-dimensional array, providing an `axis` is optional;
            however, if `x` has more than one dimension, providing an `axis`
            is required.
            Default: `None`.
        dtype (Optional[dtype]):
            data type of the returned array. If `None`, the default data
            type is inferred from the "kind" of the input array data type.

                * If `x` has a real- or complex-valued floating-point data
                  type, the returned array will have the same data type as
                  `x`.
                * If `x` has signed integral data type, the returned array
                  will have the default signed integral type for the device
                  where input array `x` is allocated.
                * If `x` has unsigned integral data type, the returned array
                  will have the default unsigned integral type for the device
                  where input array `x` is allocated.
                * If `x` has a boolean data type, the returned array will
                  have the default signed integral type for the device
                  where input array `x` is allocated.

            If the data type (either specified or resolved) differs from the
            data type of `x`, the input array elements are cast to the
            specified data type before computing the cumulative sum.
            Default: `None`.
        include_initial (bool):
            boolean indicating whether to include the initial value (i.e., the
            additive identity, zero) as the first value along the provided axis
            in the output. Default: `False`.
        out (Optional[usm_ndarray]):
            the array into which the result is written.
            The data type of `out` must match the expected shape and the
            expected data type of the result or (if provided) `dtype`.
            If `None` then a new array is returned. Default: `None`.

    Returns:
        usm_ndarray:
            an array containing cumulative sums. The returned array has the data
            type as described in the `dtype` parameter description above.

            The returned array shape is determined as follows:

                * If `include_initial` is `False`, the returned array will
                  have the same shape as `x`
                * If `include_initial` is `True`, the returned array will
                  have the same shape as `x` except the axis along which the
                  cumulative sum is calculated, which will have size `N+1`

            where `N` is the size of the axis the cumulative sums are computed
            along.
    """
    return _accumulate_common(
        x,
        axis,
        dtype,
        include_initial,
        out,
        tai._cumsum_over_axis,
        tai._cumsum_final_axis_include_initial,
        tai._cumsum_dtype_supported,
        _default_accumulation_dtype,
    )


def cumulative_prod(
    x, /, *, axis=None, dtype=None, include_initial=False, out=None
):
    """
    cumulative_prod(x, /, *, axis=None, dtype=None, include_initial=False,
                   out=None)

    Calculates the cumulative product of elements in the input array `x`.

    Args:
        x (usm_ndarray):
            input array.
        axis (Optional[int]):
            axis along which cumulative product must be computed.
            If `None`, the product is computed over the entire array.
            If `x` is a one-dimensional array, providing an `axis` is optional;
            however, if `x` has more than one dimension, providing an `axis`
            is required.
            Default: `None`.
        dtype (Optional[dtype]):
            data type of the returned array. If `None`, the default data
            type is inferred from the "kind" of the input array data type.

                * If `x` has a real- or complex-valued floating-point data
                  type, the returned array will have the same data type as
                  `x`.
                * If `x` has signed integral data type, the returned array
                  will have the default signed integral type for the device
                  where input array `x` is allocated.
                * If `x` has unsigned integral data type, the returned array
                  will have the default unsigned integral type for the device
                  where input array `x` is allocated.
                * If `x` has a boolean data type, the returned array will
                  have the default signed integral type for the device
                  where input array `x` is allocated.

            If the data type (either specified or resolved) differs from the
            data type of `x`, the input array elements are cast to the
            specified data type before computing the cumulative product.
            Default: `None`.
        include_initial (bool):
            boolean indicating whether to include the initial value (i.e., the
            additive identity, zero) as the first value along the provided
            axis in the output. Default: `False`.
        out (Optional[usm_ndarray]):
            the array into which the result is written.
            The data type of `out` must match the expected shape and the
            expected data type of the result or (if provided) `dtype`.
            If `None` then a new array is returned. Default: `None`.

    Returns:
        usm_ndarray:
            an array containing cumulative products. The returned array has
            the data type as described in the `dtype` parameter description
            above.

            The returned array shape is determined as follows:

                * If `include_initial` is `False`, the returned array will
                  have the same shape as `x`
                * If `include_initial` is `True`, the returned array will
                  have the same shape as `x` except the axis along which the
                  cumulative product is calculated, which will have size `N+1`

            where `N` is the size of the axis the cumulative products are
            computed along.
    """
    return _accumulate_common(
        x,
        axis,
        dtype,
        include_initial,
        out,
        tai._cumprod_over_axis,
        tai._cumprod_final_axis_include_initial,
        tai._cumprod_dtype_supported,
        _default_accumulation_dtype,
    )


def cumulative_logsumexp(
    x, /, *, axis=None, dtype=None, include_initial=False, out=None
):
    """
    cumulative_logsumexp(x, /, *, axis=None, dtype=None, include_initial=False,
                   out=None)

    Calculates the cumulative logsmumexp of elements in the input array `x`.

    Args:
        x (usm_ndarray):
            input array.
        axis (Optional[int]):
            axis along which cumulative logsumexp must be computed.
            If `None`, the logsumexp is computed over the entire array.
            If `x` is a one-dimensional array, providing an `axis` is optional;
            however, if `x` has more than one dimension, providing an `axis`
            is required.
            Default: `None`.
        dtype (Optional[dtype]):
            data type of the returned array. If `None`, the default data
            type is inferred from the "kind" of the input array data type.

                * If `x` has a real- or complex-valued floating-point data
                  type, the returned array will have the same data type as
                  `x`.
                * If `x` has signed integral data type, the returned array
                  will have the default signed integral type for the device
                  where input array `x` is allocated.
                * If `x` has unsigned integral data type, the returned array
                  will have the default unsigned integral type for the device
                  where input array `x` is allocated.
                * If `x` has a boolean data type, the returned array will
                  have the default signed integral type for the device
                  where input array `x` is allocated.

            If the data type (either specified or resolved) differs from the
            data type of `x`, the input array elements are cast to the
            specified data type before computing the cumulative logsumexp.
            Default: `None`.
        include_initial (bool):
            boolean indicating whether to include the initial value (i.e., the
            additive identity, zero) as the first value along the provided axis
            in the output. Default: `False`.
        out (Optional[usm_ndarray]):
            the array into which the result is written.
            The data type of `out` must match the expected shape and the
            expected data type of the result or (if provided) `dtype`.
            If `None` then a new array is returned. Default: `None`.

    Returns:
        usm_ndarray:
            an array containing cumulative logsumexp results. The returned
            array has the data type as described in the `dtype` parameter
            description above.

            The returned array shape is determined as follows:

                * If `include_initial` is `False`, the returned array will
                  have the same shape as `x`
                * If `include_initial` is `True`, the returned array will
                  have the same shape as `x` except the axis along which the
                  cumulative logsumexp is calculated, which will have size
                  `N+1`
    """
    return _accumulate_common(
        x,
        axis,
        dtype,
        include_initial,
        out,
        tai._cumlogsumexp_over_axis,
        tai._cumlogsumexp_final_axis_include_initial,
        tai._cumlogsumexp_dtype_supported,
        _default_accumulation_dtype_fp_types,
    )
