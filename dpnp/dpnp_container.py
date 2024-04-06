# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2024, Intel Corporation
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
Container specific part of the DPNP

Notes
-----
This module contains code and dependency on different containers used in DPNP

"""


import dpctl.tensor as dpt
import dpctl.utils as dpu

import dpnp
from dpnp.dpnp_array import dpnp_array

__all__ = [
    "arange",
    "asarray",
    "copy",
    "empty",
    "eye",
    "full",
    "linspace",
    "ones",
    "tril",
    "triu",
    "zeros",
]


def arange(
    start,
    /,
    stop=None,
    step=1,
    *,
    dtype=None,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """Validate input parameters before passing them into `dpctl.tensor` module"""
    dpu.validate_usm_type(usm_type, allow_none=False)
    sycl_queue_normalized = dpnp.get_normalized_queue_device(
        sycl_queue=sycl_queue, device=device
    )

    array_obj = dpt.arange(
        start,
        stop=stop,
        step=step,
        dtype=dtype,
        usm_type=usm_type,
        sycl_queue=sycl_queue_normalized,
    )

    return dpnp_array(array_obj.shape, buffer=array_obj)


def asarray(
    x1,
    dtype=None,
    copy=None,
    order=None,
    device=None,
    usm_type=None,
    sycl_queue=None,
):
    """Converts `x1` to `dpnp_array`."""
    dpu.validate_usm_type(usm_type, allow_none=True)

    if order is None:
        order = "K"

    """Converts incoming 'x1' object to 'dpnp_array'."""
    if isinstance(x1, (list, tuple, range)):
        array_obj = dpt.asarray(
            x1,
            dtype=dtype,
            copy=copy,
            order=order,
            device=device,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
        )
    else:
        if isinstance(x1, dpnp_array):
            x1_obj = x1.get_array()
        else:
            x1_obj = x1

        sycl_queue_normalized = dpnp.get_normalized_queue_device(
            x1_obj, device=device, sycl_queue=sycl_queue
        )

        array_obj = dpt.asarray(
            x1_obj,
            dtype=dtype,
            copy=copy,
            order=order,
            usm_type=usm_type,
            sycl_queue=sycl_queue_normalized,
        )

        # return x1 if dpctl returns a zero copy of x1_obj
        if array_obj is x1_obj and isinstance(x1, dpnp_array):
            return x1

    return dpnp_array(array_obj.shape, buffer=array_obj, order=order)


def copy(x1, /, *, order="K"):
    """Creates `dpnp_array` as a copy of given instance of input array."""
    if order is None:
        order = "K"

    array_obj = dpt.copy(dpnp.get_usm_ndarray(x1), order=order)
    return dpnp_array(array_obj.shape, buffer=array_obj, order="K")


def empty(
    shape,
    *,
    dtype=None,
    order="C",
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """Validate input parameters before passing them into `dpctl.tensor` module"""
    dpu.validate_usm_type(usm_type, allow_none=False)
    sycl_queue_normalized = dpnp.get_normalized_queue_device(
        sycl_queue=sycl_queue, device=device
    )
    if order is None:
        order = "C"

    """Creates `dpnp_array` from uninitialized USM allocation."""
    array_obj = dpt.empty(
        shape,
        dtype=dtype,
        order=order,
        usm_type=usm_type,
        sycl_queue=sycl_queue_normalized,
    )
    return dpnp_array(array_obj.shape, buffer=array_obj, order=order)


def eye(
    N,
    M=None,
    /,
    *,
    k=0,
    dtype=None,
    order="C",
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """Validate input parameters before passing them into `dpctl.tensor` module"""
    dpu.validate_usm_type(usm_type, allow_none=False)
    sycl_queue_normalized = dpnp.get_normalized_queue_device(
        sycl_queue=sycl_queue, device=device
    )
    if order is None:
        order = "C"

    """Creates `dpnp_array` with ones on the `k`th diagonal."""
    array_obj = dpt.eye(
        N,
        M,
        k=k,
        dtype=dtype,
        order=order,
        usm_type=usm_type,
        sycl_queue=sycl_queue_normalized,
    )
    return dpnp_array(array_obj.shape, buffer=array_obj, order=order)


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
    """Validate input parameters before passing them into `dpctl.tensor` module"""
    dpu.validate_usm_type(usm_type, allow_none=True)

    sycl_queue_normalized = dpnp.get_normalized_queue_device(
        fill_value, sycl_queue=sycl_queue, device=device
    )
    if order is None:
        order = "C"

    if isinstance(fill_value, dpnp_array):
        fill_value = fill_value.get_array()

    """Creates `dpnp_array` having a specified shape, filled with fill_value."""
    array_obj = dpt.full(
        shape,
        fill_value,
        dtype=dtype,
        order=order,
        usm_type=usm_type,
        sycl_queue=sycl_queue_normalized,
    )
    return dpnp_array(array_obj.shape, buffer=array_obj, order=order)


def linspace(
    start,
    stop,
    /,
    num,
    *,
    dtype=None,
    device=None,
    usm_type="device",
    sycl_queue=None,
    endpoint=True,
):
    """Validate input parameters before passing them into `dpctl.tensor` module"""
    dpu.validate_usm_type(usm_type, allow_none=False)
    sycl_queue_normalized = dpnp.get_normalized_queue_device(
        sycl_queue=sycl_queue, device=device
    )

    """Creates `dpnp_array` with evenly spaced numbers of specified interval."""
    array_obj = dpt.linspace(
        start,
        stop,
        num,
        dtype=dtype,
        usm_type=usm_type,
        sycl_queue=sycl_queue_normalized,
        endpoint=endpoint,
    )
    return dpnp_array(array_obj.shape, buffer=array_obj)


def ones(
    shape,
    *,
    dtype=None,
    order="C",
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """Validate input parameters before passing them into `dpctl.tensor` module"""
    dpu.validate_usm_type(usm_type, allow_none=False)
    sycl_queue_normalized = dpnp.get_normalized_queue_device(
        sycl_queue=sycl_queue, device=device
    )
    if order is None:
        order = "C"

    """Creates `dpnp_array` of ones with the given shape, dtype, and order."""
    array_obj = dpt.ones(
        shape,
        dtype=dtype,
        order=order,
        usm_type=usm_type,
        sycl_queue=sycl_queue_normalized,
    )
    return dpnp_array(array_obj.shape, buffer=array_obj, order=order)


def tril(x1, /, *, k=0):
    """Creates `dpnp_array` as lower triangular part of an input array."""
    array_obj = dpt.tril(dpnp.get_usm_ndarray(x1), k=k)
    return dpnp_array(array_obj.shape, buffer=array_obj, order="K")


def triu(x1, /, *, k=0):
    """Creates `dpnp_array` as upper triangular part of an input array."""
    array_obj = dpt.triu(dpnp.get_usm_ndarray(x1), k=k)
    return dpnp_array(array_obj.shape, buffer=array_obj, order="K")


def zeros(
    shape,
    *,
    dtype=None,
    order="C",
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """Validate input parameters before passing them into `dpctl.tensor` module"""
    dpu.validate_usm_type(usm_type, allow_none=False)
    sycl_queue_normalized = dpnp.get_normalized_queue_device(
        sycl_queue=sycl_queue, device=device
    )
    if order is None:
        order = "C"

    """Creates `dpnp_array` of zeros with the given shape, dtype, and order."""
    array_obj = dpt.zeros(
        shape,
        dtype=dtype,
        order=order,
        usm_type=usm_type,
        sycl_queue=sycl_queue_normalized,
    )
    return dpnp_array(array_obj.shape, buffer=array_obj, order=order)
