# *****************************************************************************
# Copyright (c) 2024, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
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

"""
Helping functions to implement the FFT interface.

These include assertion functions to validate input array and
functions with the main implementation part to fulfill the interface.
The main computational work is performed by enabling FFT functions
available as a pybind11 extension.

"""

# pylint: disable=protected-access
# pylint: disable=c-extension-no-member
# pylint: disable=no-name-in-module

import dpctl
import dpctl.tensor as dpt
import dpctl.tensor._tensor_impl as ti
import numpy
from dpctl.utils import ExecutionPlacementError
from numpy.core.numeric import normalize_axis_index

import dpnp
import dpnp.backend.extensions.fft._fft_impl as fi
from dpnp.dpnp_utils.dpnp_utils_linearalgebra import (
    _standardize_strides_to_nonzero,
)

from ..dpnp_array import dpnp_array
from ..dpnp_utils import map_dtype_to_device

__all__ = [
    "dpnp_fft",
]


def _check_norm(norm):
    if norm not in (None, "ortho", "forward", "backward"):
        raise ValueError(
            f"Invalid norm value {norm} should be None, "
            '"ortho", "forward", or "backward".'
        )


def _commit_descriptor(a, a_strides, index, axes):
    """Commit the FFT descriptor for the input array."""

    a_shape = a.shape
    shape = a_shape[index:]
    strides = (0,) + a_strides[index:]
    if a.dtype == dpnp.complex64:
        dsc = fi.Complex64Descriptor(shape)
    else:
        dsc = fi.Complex128Descriptor(shape)

    dsc.fwd_strides = strides
    dsc.bwd_strides = dsc.fwd_strides
    dsc.transform_in_place = False
    if axes is not None:  # batch_fft
        dsc.fwd_distance = a_strides[0]
        dsc.bwd_distance = dsc.fwd_distance
        dsc.number_of_transforms = numpy.prod(a_shape[0])
    dsc.commit(a.sycl_queue)

    return dsc


def _compute_result(dsc, a, out, is_forward, a_strides, hev_list, dev_list):
    """Compute the result of the FFT."""

    a_usm = a.get_array()
    if (
        out is not None
        and out.strides == a_strides
        and not ti._array_overlap(a_usm, out.get_array())
    ):
        res_usm = out.get_array()
    else:
        # Result array that is used in OneMKL must have the exact same
        # stride as input array
        res_usm = dpt.usm_ndarray(
            a.shape,
            dtype=a.dtype,
            buffer=a.usm_type,
            strides=a_strides,
            offset=0,
            buffer_ctor_kwargs={"queue": a.sycl_queue},
        )
    fft_event, _ = fi.compute_fft(dsc, a_usm, res_usm, is_forward, dev_list)
    hev_list.append(fft_event)
    dpctl.SyclEvent.wait_for(hev_list)

    res = dpnp_array._create_from_usm_ndarray(res_usm)

    return res


def _copy_array(x, dep_events, host_events):
    """
    Creating a C-contiguous copy of input array if input array has a negative
    stride or it does not have a complex data types.
    """
    dtype = x.dtype
    copy_flag = False
    if numpy.min(x.strides) < 0:
        # negative stride is not allowed in OneMKL FFT
        copy_flag = True
    elif not dpnp.issubdtype(dtype, dpnp.complexfloating):
        # if input is not complex, convert to complex
        copy_flag = True
        if dtype == dpnp.float32:
            dtype = dpnp.complex64
        else:
            dtype = map_dtype_to_device(dpnp.complex128, x.sycl_device)

    if copy_flag:
        x_copy = dpnp.empty_like(x, dtype=dtype, order="C")
        ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=dpnp.get_usm_ndarray(x),
            dst=x_copy.get_array(),
            sycl_queue=x.sycl_queue,
        )
        dep_events.append(copy_ev)
        host_events.append(ht_copy_ev)
        return x_copy
    return x


def _fft(a, norm, out, is_forward, hev_list, dev_list, axes=None):
    """Calculates FFT of the input array along the specified axes."""

    index = 0
    if axes is not None:  # batch_fft
        len_axes = 1 if isinstance(axes, int) else len(axes)
        local_axes = numpy.arange(-len_axes, 0)
        a = dpnp.moveaxis(a, axes, local_axes)
        a_shape_orig = a.shape
        local_shape = (-1,) + a_shape_orig[-len_axes:]
        a = dpnp.reshape(a, local_shape)
        index = 1

    a_strides = _standardize_strides_to_nonzero(a.strides, a.shape)
    dsc = _commit_descriptor(a, a_strides, index, axes)
    res = _compute_result(
        dsc, a, out, is_forward, a_strides, hev_list, dev_list
    )

    scale = numpy.prod(a.shape[index:], dtype=a.real.dtype)
    norm_factor = 1
    if norm == "ortho":
        norm_factor = numpy.sqrt(scale)
    elif norm == "forward" and is_forward:
        norm_factor = scale
    elif norm in [None, "backward"] and not is_forward:
        norm_factor = scale

    res /= norm_factor
    if axes is not None:  # batch_fft
        res = dpnp.reshape(res, a_shape_orig)
        res = dpnp.moveaxis(res, local_axes, axes)

    result = dpnp.get_result_array(res, out=out, casting="same_kind")
    if not (result.flags.c_contiguous or result.flags.f_contiguous):
        result = dpnp.ascontiguousarray(result)
    return result


def _truncate_or_pad(a, shape, axes, copy_ht_ev, copy_dp_ev):
    """Truncating or zero-padding the input array along the specified axes."""

    shape = (shape,) if isinstance(shape, int) else shape
    axes = (axes,) if isinstance(axes, int) else axes

    for s, axis in zip(shape, axes):
        a_shape = list(a.shape)
        index = [slice(None)] * a.ndim
        if s == a_shape[axis]:
            pass
        elif s < a_shape[axis]:
            # truncating
            index[axis] = slice(0, s)
            a = a[tuple(index)]
        else:
            # zero-padding
            exec_q = a.sycl_queue
            index[axis] = slice(0, a_shape[axis])  # orig shape
            a_shape[axis] = s  # modified shape
            order = "F" if a.flags.f_contiguous else "C"
            z = dpnp.zeros(
                a_shape,
                dtype=a.dtype,
                order=order,
                usm_type=a.usm_type,
                sycl_queue=exec_q,
            )
            ht_ev, dp_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                src=a.get_array(),
                dst=z.get_array()[tuple(index)],
                sycl_queue=exec_q,
                depends=copy_dp_ev,
            )
            copy_ht_ev.append(ht_ev)
            copy_dp_ev.append(dp_ev)
            a = z

    return a


def _validate_out_keyword(a, out):
    """Validate out keyword argument."""
    if out is not None:
        dpnp.check_supported_arrays_type(out)
        if (
            dpctl.utils.get_execution_queue((a.sycl_queue, out.sycl_queue))
            is None
        ):
            raise ExecutionPlacementError(
                "Input and output allocation queues are not compatible"
            )

        if out.shape != a.shape:
            raise ValueError("output array has incorrect shape.")

        if not dpnp.issubdtype(out.dtype, dpnp.complexfloating):
            raise TypeError("output array has incorrect data type.")


def dpnp_fft(a, is_forward, n=None, axis=-1, norm=None, out=None):
    """Calculates 1-D FFT of the input array along axis"""

    _check_norm(norm)
    a_ndim = a.ndim
    copy_ht_ev = []
    copy_dp_ev = []
    a = _copy_array(a, copy_ht_ev, copy_dp_ev)

    if a_ndim == 0:
        raise ValueError("Input array must be at least 1D")

    axis = normalize_axis_index(axis, a_ndim)
    if n is None:
        n = a.shape[axis]
    if not isinstance(n, int):
        raise TypeError("`n` should be None or an integer")
    if n < 1:
        raise ValueError(f"Invalid number of FFT data points ({n}) specified")

    a = _truncate_or_pad(a, n, axis, copy_ht_ev, copy_dp_ev)
    _validate_out_keyword(a, out)

    if a.size == 0:
        return a

    if a_ndim == 1:
        return _fft(
            a,
            norm=norm,
            out=out,
            is_forward=is_forward,
            hev_list=copy_ht_ev,
            dev_list=copy_dp_ev,
        )

    return _fft(
        a,
        norm=norm,
        out=out,
        is_forward=is_forward,
        axes=axis,
        hev_list=copy_ht_ev,
        dev_list=copy_dp_ev,
    )
