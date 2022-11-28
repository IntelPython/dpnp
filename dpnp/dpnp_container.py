# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2022, Intel Corporation
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
This module contains code and dependency on diffrent containers used in DPNP

"""


import dpctl.utils as dpu
import dpctl.tensor as dpt

from dpnp.dpnp_array import dpnp_array
import dpnp


__all__ = [
    "arange",
    "asarray",
    "empty",
    "full",
]


def arange(start,
           /,
           stop=None,
           step=1,
           *,
           dtype=None,
           device=None,
           usm_type="device",
           sycl_queue=None):
    """Validate input parameters before passing them into `dpctl.tensor` module"""
    dpu.validate_usm_type(usm_type, allow_none=False)
    sycl_queue_normalized = dpnp.get_normalized_queue_device(sycl_queue=sycl_queue, device=device)

    array_obj = dpt.arange(start,
                           stop=stop,
                           step=step,
                           dtype=dtype,
                           usm_type=usm_type,
                           sycl_queue=sycl_queue_normalized)

    return dpnp_array(array_obj.shape, buffer=array_obj)


def asarray(x1,
            dtype=None,
            copy=False,
            order="C",
            device=None,
            usm_type=None,
            sycl_queue=None):
    """Converts `x1` to `dpnp_array`."""
    if isinstance(x1, dpnp_array):
        x1_obj = x1.get_array()
    else:
        x1_obj = x1

    sycl_queue_normalized = dpnp.get_normalized_queue_device(x1_obj, device=device, sycl_queue=sycl_queue)

    """Converts incoming 'x1' object to 'dpnp_array'."""
    array_obj = dpt.asarray(x1_obj,
                            dtype=dtype,
                            copy=copy,
                            order=order,
                            usm_type=usm_type,
                            sycl_queue=sycl_queue_normalized)
    return dpnp_array(array_obj.shape, buffer=array_obj, order=order)


def empty(shape,
          dtype="f8",
          order="C",
          device=None,
          usm_type="device",
          sycl_queue=None):
    sycl_queue_normalized = dpnp.get_normalized_queue_device(device=device, sycl_queue=sycl_queue)

    """Creates `dpnp_array` from uninitialized USM allocation."""
    array_obj = dpt.empty(shape,
                          dtype=dtype,
                          order=order,
                          usm_type=usm_type,
                          sycl_queue=sycl_queue_normalized)
    return dpnp_array(array_obj.shape, buffer=array_obj, order=order)


def full(shape,
         fill_value,
         *,
         dtype=None,
         order="C",
         device=None,
         usm_type=None,
         sycl_queue=None):
    """Validate input parameters before passing them into `dpctl.tensor` module"""
    dpu.validate_usm_type(usm_type, allow_none=True)
    if sycl_queue is None and device is None:
        sycl_queue_normalized = None
    else:
        sycl_queue_normalized = dpnp.get_normalized_queue_device(sycl_queue=sycl_queue, device=device)

    if isinstance(fill_value, dpnp.ndarray):
        fill_value = fill_value.get_array()

    """Creates `dpnp_array` where every element is equal to fill_value."""
    array_obj = dpt.full(shape,
                         fill_value,
                         dtype=dtype,
                         order=order,
                         usm_type=usm_type,
                         sycl_queue=sycl_queue_normalized)
    return dpnp_array(array_obj.shape, buffer=array_obj, order=order)
