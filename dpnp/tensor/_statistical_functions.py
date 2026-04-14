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

import dpctl.utils as du

import dpnp.tensor as dpt
import dpnp.tensor._tensor_elementwise_impl as tei
import dpnp.tensor._tensor_impl as ti
import dpnp.tensor._tensor_reductions_impl as tri

from ._numpy_helper import normalize_axis_tuple


def _var_impl(x, axis, correction, keepdims):
    nd = x.ndim
    if axis is None:
        axis = tuple(range(nd))
    if not isinstance(axis, (tuple, list)):
        axis = (axis,)
    axis = normalize_axis_tuple(axis, nd, "axis")
    perm = []
    nelems = 1
    for i in range(nd):
        if i not in axis:
            perm.append(i)
        else:
            nelems *= x.shape[i]
    red_nd = len(axis)
    perm = perm + list(axis)
    q = x.sycl_queue
    inp_dt = x.dtype
    res_dt = (
        inp_dt
        if inp_dt.kind == "f"
        else dpt.dtype(ti.default_device_fp_type(q))
    )
    res_usm_type = x.usm_type

    _manager = du.SequentialOrderManager[q]
    dep_evs = _manager.submitted_events
    if inp_dt != res_dt:
        buf = dpt.empty_like(x, dtype=res_dt)
        ht_e_buf, c_e1 = ti._copy_usm_ndarray_into_usm_ndarray(
            src=x, dst=buf, sycl_queue=q, depends=dep_evs
        )
        _manager.add_event_pair(ht_e_buf, c_e1)
    else:
        buf = x
    # calculate mean
    buf2 = dpt.permute_dims(buf, perm)
    res_shape = buf2.shape[: nd - red_nd]
    # use keepdims=True path for later broadcasting
    if red_nd == 0:
        mean_ary = dpt.empty_like(buf)
        dep_evs = _manager.submitted_events
        ht_e1, c_e2 = ti._copy_usm_ndarray_into_usm_ndarray(
            src=buf, dst=mean_ary, sycl_queue=q, depends=dep_evs
        )
        _manager.add_event_pair(ht_e1, c_e2)
    else:
        mean_ary = dpt.empty(
            res_shape,
            dtype=res_dt,
            usm_type=res_usm_type,
            sycl_queue=q,
        )
        dep_evs = _manager.submitted_events
        ht_e1, r_e1 = tri._sum_over_axis(
            src=buf2,
            trailing_dims_to_reduce=red_nd,
            dst=mean_ary,
            sycl_queue=q,
            depends=dep_evs,
        )
        _manager.add_event_pair(ht_e1, r_e1)

        mean_ary_shape = res_shape + (1,) * red_nd
        inv_perm = sorted(range(nd), key=lambda d: perm[d])
        mean_ary = dpt.permute_dims(
            dpt.reshape(mean_ary, mean_ary_shape), inv_perm
        )
    # divide in-place to get mean
    mean_ary_shape = mean_ary.shape

    dep_evs = _manager.submitted_events
    ht_e2, d_e1 = tei._divide_by_scalar(
        src=mean_ary, scalar=nelems, dst=mean_ary, sycl_queue=q, depends=dep_evs
    )
    _manager.add_event_pair(ht_e2, d_e1)

    # subtract mean from original array to get deviations
    dev_ary = dpt.empty_like(buf)
    if mean_ary_shape != buf.shape:
        mean_ary = dpt.broadcast_to(mean_ary, buf.shape)
    ht_e4, su_e = tei._subtract(
        src1=buf, src2=mean_ary, dst=dev_ary, sycl_queue=q, depends=[d_e1]
    )
    _manager.add_event_pair(ht_e4, su_e)
    # square deviations
    ht_e5, sq_e = tei._square(
        src=dev_ary, dst=dev_ary, sycl_queue=q, depends=[su_e]
    )
    _manager.add_event_pair(ht_e5, sq_e)

    # take sum of squared deviations
    dev_ary2 = dpt.permute_dims(dev_ary, perm)
    if red_nd == 0:
        res = dev_ary
    else:
        res = dpt.empty(
            res_shape,
            dtype=res_dt,
            usm_type=res_usm_type,
            sycl_queue=q,
        )
        ht_e6, r_e2 = tri._sum_over_axis(
            src=dev_ary2,
            trailing_dims_to_reduce=red_nd,
            dst=res,
            sycl_queue=q,
            depends=[sq_e],
        )
        _manager.add_event_pair(ht_e6, r_e2)

        if keepdims:
            res_shape = res_shape + (1,) * red_nd
            inv_perm = sorted(range(nd), key=lambda d: perm[d])
            res = dpt.permute_dims(dpt.reshape(res, res_shape), inv_perm)
    res_shape = res.shape
    # when nelems - correction <= 0, yield nans
    div = max(nelems - correction, 0)
    if not div:
        div = dpt.nan
    dep_evs = _manager.submitted_events
    ht_e7, d_e2 = tei._divide_by_scalar(
        src=res, scalar=div, dst=res, sycl_queue=q, depends=dep_evs
    )
    _manager.add_event_pair(ht_e7, d_e2)
    return res, [d_e2]


def mean(x, axis=None, keepdims=False):
    """mean(x, axis=None, keepdims=False)

    Calculates the arithmetic mean of elements in the input array `x`.

    Args:
        x (usm_ndarray):
            input array.
        axis (Optional[int, Tuple[int, ...]]):
            axis or axes along which the arithmetic means must be computed. If
            a tuple of unique integers, the means are computed over multiple
            axes. If `None`, the mean is computed over the entire array.
            Default: `None`.
        keepdims (Optional[bool]):
            if `True`, the reduced axes (dimensions) are included in the result
            as singleton dimensions, so that the returned array remains
            compatible with the input array according to Array Broadcasting
            rules. Otherwise, if `False`, the reduced axes are not included in
            the returned array. Default: `False`.
    Returns:
        usm_ndarray:
            an array containing the arithmetic means. If the mean was computed
            over the entire array, a zero-dimensional array is returned.

            If `x` has a floating-point data type, the returned array will have
            the same data type as `x`.
            If `x` has a boolean or integral data type, the returned array
            will have the default floating point data type for the device
            where input array `x` is allocated.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected dpnp.tensor.usm_ndarray, got {type(x)}")
    nd = x.ndim
    if axis is None:
        axis = tuple(range(nd))
    if not isinstance(axis, (tuple, list)):
        axis = (axis,)
    axis = normalize_axis_tuple(axis, nd, "axis")
    perm = []
    nelems = 1
    for i in range(nd):
        if i not in axis:
            perm.append(i)
        else:
            nelems *= x.shape[i]
    sum_nd = len(axis)
    perm = perm + list(axis)
    arr2 = dpt.permute_dims(x, perm)
    res_shape = arr2.shape[: nd - sum_nd]
    q = x.sycl_queue
    inp_dt = x.dtype
    res_dt = (
        x.dtype
        if x.dtype.kind in "fc"
        else dpt.dtype(ti.default_device_fp_type(q))
    )
    res_usm_type = x.usm_type
    if sum_nd == 0:
        return dpt.astype(x, res_dt, copy=True)

    _manager = du.SequentialOrderManager[q]
    dep_evs = _manager.submitted_events
    if tri._sum_over_axis_dtype_supported(inp_dt, res_dt, res_usm_type, q):
        res = dpt.empty(
            res_shape, dtype=res_dt, usm_type=res_usm_type, sycl_queue=q
        )
        ht_e1, r_e = tri._sum_over_axis(
            src=arr2,
            trailing_dims_to_reduce=sum_nd,
            dst=res,
            sycl_queue=q,
            depends=dep_evs,
        )
        _manager.add_event_pair(ht_e1, r_e)
    else:
        tmp = dpt.empty(
            arr2.shape, dtype=res_dt, usm_type=res_usm_type, sycl_queue=q
        )
        ht_e_cpy, cpy_e = ti._copy_usm_ndarray_into_usm_ndarray(
            src=arr2, dst=tmp, sycl_queue=q, depends=dep_evs
        )
        _manager.add_event_pair(ht_e_cpy, cpy_e)
        res = dpt.empty(
            res_shape, dtype=res_dt, usm_type=res_usm_type, sycl_queue=q
        )
        ht_e_red, r_e = tri._sum_over_axis(
            src=tmp,
            trailing_dims_to_reduce=sum_nd,
            dst=res,
            sycl_queue=q,
            depends=[cpy_e],
        )
        _manager.add_event_pair(ht_e_red, r_e)

    if keepdims:
        res_shape = res_shape + (1,) * sum_nd
        inv_perm = sorted(range(nd), key=lambda d: perm[d])
        res = dpt.permute_dims(dpt.reshape(res, res_shape), inv_perm)

    dep_evs = _manager.submitted_events
    ht_e2, div_e = tei._divide_by_scalar(
        src=res, scalar=nelems, dst=res, sycl_queue=q, depends=dep_evs
    )
    _manager.add_event_pair(ht_e2, div_e)
    return res


def var(x, axis=None, correction=0.0, keepdims=False):
    """var(x, axis=None, correction=0.0, keepdims=False)

    Calculates the variance of elements in the input array `x`.

    Args:
        x (usm_ndarray):
            input array.
        axis (Optional[int, Tuple[int, ...]]):
            axis or axes along which the variances must be computed. If a tuple
            of unique integers, the variances are computed over multiple axes.
            If `None`, the variance is computed over the entire array.
            Default: `None`.
        correction (Optional[float, int]):
            degrees of freedom adjustment. The divisor used in calculating the
            variance is `N - correction`, where `N` corresponds to the total
            number of elements over which the variance is calculated.
            Default: `0.0`.
        keepdims (Optional[bool]):
            if `True`, the reduced axes (dimensions) are included in the result
            as singleton dimensions, so that the returned array remains
            compatible with the input array according to Array Broadcasting
            rules. Otherwise, if `False`, the reduced axes are not included in
            the returned array. Default: `False`.
    Returns:
        usm_ndarray:
            an array containing the variances. If the variance was computed
            over the entire array, a zero-dimensional array is returned.

            If `x` has a real-valued floating-point data type, the returned
            array will have the same data type as `x`.
            If `x` has a boolean or integral data type, the returned array
            will have the default floating point data type for the device
            where input array `x` is allocated.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected dpnp.tensor.usm_ndarray, got {type(x)}")

    if not isinstance(correction, (int, float)):
        raise TypeError(
            "Expected a Python integer or float for `correction`, got"
            f"{type(x)}"
        )

    if x.dtype.kind == "c":
        raise ValueError("`var` does not support complex types")

    res, _ = _var_impl(x, axis, correction, keepdims)
    return res


def std(x, axis=None, correction=0.0, keepdims=False):
    """std(x, axis=None, correction=0.0, keepdims=False)

    Calculates the standard deviation of elements in the input array `x`.

    Args:
        x (usm_ndarray):
            input array.
        axis (Optional[int, Tuple[int, ...]]):
            axis or axes along which the standard deviations must be computed.
            If a tuple of unique integers, the standard deviations are computed
            over multiple axes. If `None`, the standard deviation is computed
            over the entire array. Default: `None`.
        correction (Optional[float, int]):
            degrees of freedom adjustment. The divisor used in calculating the
            standard deviation is `N - correction`, where `N` corresponds to the
            total number of elements over which the standard deviation is
            calculated. Default: `0.0`.
        keepdims (Optional[bool]):
            if `True`, the reduced axes (dimensions) are included in the result
            as singleton dimensions, so that the returned array remains
            compatible with the input array according to Array Broadcasting
            rules. Otherwise, if `False`, the reduced axes are not included in
            the returned array. Default: `False`.
    Returns:
        usm_ndarray:
            an array containing the standard deviations. If the standard
            deviation was computed over the entire array, a zero-dimensional
            array is returned.

            If `x` has a real-valued floating-point data type, the returned
            array will have the same data type as `x`.
            If `x` has a boolean or integral data type, the returned array
            will have the default floating point data type for the device
            where input array `x` is allocated.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected dpnp.tensor.usm_ndarray, got {type(x)}")

    if not isinstance(correction, (int, float)):
        raise TypeError(
            "Expected a Python integer or float for `correction`,"
            f"got {type(x)}"
        )

    if x.dtype.kind == "c":
        raise ValueError("`std` does not support complex types")

    exec_q = x.sycl_queue
    _manager = du.SequentialOrderManager[exec_q]
    res, deps = _var_impl(x, axis, correction, keepdims)
    ht_ev, sqrt_ev = tei._sqrt(
        src=res, dst=res, sycl_queue=exec_q, depends=deps
    )
    _manager.add_event_pair(ht_ev, sqrt_ev)
    return res
