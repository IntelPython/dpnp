# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2020, Intel Corporation
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


import dpnp.config as config
# from dpnp.dparray import dparray
from dpnp.dpnp_array import dpnp_array

import numpy

import dpctl.tensor as dpt
from dpctl.tensor._device import normalize_queue_device


if config.__DPNP_OUTPUT_DPCTL__:
    try:
        """
        Detect DPCtl availability to use data container
        """
        import dpctl.tensor as dpctl

    except ImportError:
        """
        No DPCtl data container available
        """
        config.__DPNP_OUTPUT_DPCTL__ = 0


__all__ = [
    "asarray",
    "empty",
]


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

    sycl_queue_normalized = normalize_queue_device(sycl_queue=sycl_queue, device=device)
    array_obj = dpt.asarray(x1_obj,
                            dtype=dtype,
                            copy=copy,
                            order=order,
                            device=device,
                            usm_type=usm_type,
                            sycl_queue=sycl_queue_normalized)

    return dpnp_array(array_obj.shape, buffer=array_obj, order=order)


def empty(shape,
          dtype="f8",
          order="C",
          device=None,
          usm_type="device",
          sycl_queue=None):
    """Creates `dpnp_array` from uninitialized USM allocation."""
    sycl_queue_normalized = normalize_queue_device(sycl_queue=sycl_queue, device=device)

    array_obj = dpt.empty(shape,
                          dtype=dtype,
                          order=order,
                          device=device,
                          usm_type=usm_type,
                          sycl_queue=sycl_queue_normalized)

    return dpnp_array(array_obj.shape, buffer=array_obj, order=order)
