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


import dpctl.tensor as dpt
import dpctl.utils as du

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt_ext
import dpctl_ext.tensor._tensor_impl as ti
import dpctl_ext.tensor._tensor_reductions_impl as tri

from ._numpy_helper import normalize_axis_tuple


def _boolean_reduction(x, axis, keepdims, func):
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x)}")

    nd = x.ndim
    if axis is None:
        red_nd = nd
        # case of a scalar
        if red_nd == 0:
            return dpt_ext.astype(x, dpt.bool)
        x_tmp = x
        res_shape = ()
        perm = list(range(nd))
    else:
        if not isinstance(axis, (tuple, list)):
            axis = (axis,)
        axis = normalize_axis_tuple(axis, nd, "axis")

        red_nd = len(axis)
        # check for axis=()
        if red_nd == 0:
            return dpt_ext.astype(x, dpt.bool)
        perm = [i for i in range(nd) if i not in axis] + list(axis)
        x_tmp = dpt_ext.permute_dims(x, perm)
        res_shape = x_tmp.shape[: nd - red_nd]

    exec_q = x.sycl_queue
    res_usm_type = x.usm_type

    _manager = du.SequentialOrderManager[exec_q]
    dep_evs = _manager.submitted_events
    # always allocate the temporary as
    # int32 and usm-device  to ensure that atomic updates
    # are supported
    res_tmp = dpt_ext.empty(
        res_shape,
        dtype=dpt.int32,
        usm_type="device",
        sycl_queue=exec_q,
    )
    hev0, ev0 = func(
        src=x_tmp,
        trailing_dims_to_reduce=red_nd,
        dst=res_tmp,
        sycl_queue=exec_q,
        depends=dep_evs,
    )
    _manager.add_event_pair(hev0, ev0)

    # copy to boolean result array
    res = dpt_ext.empty(
        res_shape,
        dtype=dpt.bool,
        usm_type=res_usm_type,
        sycl_queue=exec_q,
    )
    hev1, ev1 = ti._copy_usm_ndarray_into_usm_ndarray(
        src=res_tmp, dst=res, sycl_queue=exec_q, depends=[ev0]
    )
    _manager.add_event_pair(hev1, ev1)

    if keepdims:
        res_shape = res_shape + (1,) * red_nd
        inv_perm = sorted(range(nd), key=lambda d: perm[d])
        res = dpt_ext.permute_dims(dpt_ext.reshape(res, res_shape), inv_perm)
    return res


def all(x, /, *, axis=None, keepdims=False):
    """
    all(x, axis=None, keepdims=False)

    Tests whether all input array elements evaluate to True along a given axis.

    Args:
        x (usm_ndarray): Input array.
        axis (Optional[Union[int, Tuple[int,...]]]): Axis (or axes)
            along which to perform a logical AND reduction.
            When `axis` is `None`, a logical AND reduction
            is performed over all dimensions of `x`.
            If `axis` is negative, the axis is counted from
            the last dimension to the first.
            Default: `None`.
        keepdims (bool, optional): If `True`, the reduced axes are included
            in the result as singleton dimensions, and the result is
            broadcastable to the input array shape.
            If `False`, the reduced axes are not included in the result.
            Default: `False`.

    Returns:
        usm_ndarray:
            An array with a data type of `bool`
            containing the results of the logical AND reduction.
    """
    return _boolean_reduction(x, axis, keepdims, tri._all)
