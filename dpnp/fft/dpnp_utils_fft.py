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

import dpnp
import dpnp.backend.extensions.fft._fft_impl as fi

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


def _fft(a, norm, is_forward, hev_list, dev_list, axes=None):
    """Calculates FFT of the input array along the specified axes."""

    index = 0
    if axes is not None:  # batch_fft
        len_axes = 1 if isinstance(axes, int) else len(axes)
        local_axes = numpy.arange(-len_axes, 0)
        a = dpnp.moveaxis(a, axes, local_axes)
        a_shape_orig = a.shape
        local_shape = (-1,) + a.shape[-len_axes:]
        a = dpnp.reshape(a, local_shape)
        index = 1

    shape = a.shape[index:]
    strides = (0,) + a.strides[index:]

    if a.dtype == dpnp.complex64:
        dsc = fi.Complex64Descriptor(shape)
    else:
        dsc = fi.Complex128Descriptor(shape)

    dsc.fwd_strides = strides
    dsc.bwd_strides = dsc.fwd_strides
    dsc.transform_in_place = False
    if axes is not None:  # batch_fft
        dsc.fwd_distance = a.strides[0]
        dsc.bwd_distance = dsc.fwd_distance
        dsc.number_of_transforms = numpy.prod(a.shape[0])
    dsc.commit(a.sycl_queue)

    res = dpt.usm_ndarray(
        a.shape,
        dtype=a.dtype,
        buffer=a.usm_type,
        strides=a.strides,
        offset=0,
        buffer_ctor_kwargs={"queue": a.sycl_queue},
    )
    fft_event, _ = fi.compute_fft(dsc, a.get_array(), res, is_forward, dev_list)
    hev_list.append(fft_event)
    dpctl.SyclEvent.wait_for(hev_list)

    res = dpnp_array._create_from_usm_ndarray(res)

    scale = numpy.prod(shape, dtype=a.real.dtype)
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
    return res


def _truncate_or_pad(a, shape, axes):
    """Truncating or zero-padding the input array along the specified axes."""

    copy_ht_ev = []
    copy_dp_ev = []
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
            z = dpnp.zeros(
                a_shape,
                dtype=a.dtype,
                usm_type=a.usm_type,
                sycl_queue=exec_q,
            )
            ht_ev, dp_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                src=a.get_array(),
                dst=z.get_array()[tuple(index)],
                sycl_queue=exec_q,
            )
            copy_ht_ev.append(ht_ev)
            copy_dp_ev.append(dp_ev)
            a = z

    return a, copy_ht_ev, copy_dp_ev


def dpnp_fft(a, is_forward, n=None, axis=-1, norm=None):
    """Calculates 1-D FFT of the input array along axis"""

    dpnp.check_supported_arrays_type(a)
    _check_norm(norm)
    if not dpnp.issubdtype(a.dtype, dpnp.complexfloating):
        if a.dtype == dpnp.float32:
            dtype = dpnp.complex64
        else:
            dtype = map_dtype_to_device(dpnp.complex128, a.sycl_device)
        a = dpnp.astype(a, dtype, copy=False)

    if a.ndim == 0:
        raise ValueError("Input array must be at least 1D")
    if not isinstance(axis, int):
        raise TypeError("Axis should be an integer")

    if n is None:
        n = a.shape[axis]
    if not isinstance(n, int):
        raise TypeError("`n` should be None or an integer")
    if n < 1:
        raise ValueError(f"Invalid number of FFT data points ({n}) specified")

    a, copy_ht_ev, copy_dp_ev = _truncate_or_pad(a, n, axis)
    if a.size == 0:
        if a.shape[axis] == 0:
            raise ValueError(
                f"Invalid number of FFT data points ({0}) specified."
            )
        return a

    if a.ndim == 1:
        return _fft(
            a,
            norm=norm,
            is_forward=is_forward,
            hev_list=copy_ht_ev,
            dev_list=copy_dp_ev,
        )

    return _fft(
        a,
        norm=norm,
        is_forward=is_forward,
        axes=axis,
        hev_list=copy_ht_ev,
        dev_list=copy_dp_ev,
    )
