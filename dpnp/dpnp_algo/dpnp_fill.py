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

import dpctl.tensor as dpt
import numpy as np
from dpctl.tensor._tensor_impl import (
    _copy_usm_ndarray_into_usm_ndarray,
    _full_usm_ndarray,
    _zeros_usm_ndarray,
)
from dpctl.utils import SequentialOrderManager

import dpnp
from dpnp.dpnp_array import dpnp_array


def dpnp_fill(arr, val):
    arr = dpnp.get_usm_ndarray(arr)
    exec_q = arr.sycl_queue

    dpnp.check_supported_arrays_type(val, scalar_type=True, all_scalars=True)
    # if val is an array, process it
    if isinstance(val, (dpnp_array, dpt.usm_ndarray)):
        val = dpnp.get_usm_ndarray(val)
        if val.shape != ():
            raise ValueError("`val` must be a scalar")
        a_val = dpt.asarray(
            val,
            dtype=arr.dtype,
            usm_type=arr.usm_type,
            sycl_queue=exec_q,
        )
        a_val = dpt.broadcast_to(a_val, arr.shape)
        _manager = SequentialOrderManager[exec_q]
        dep_evs = _manager.submitted_events
        h_ev, c_ev = _copy_usm_ndarray_into_usm_ndarray(
            src=a_val, dst=arr, sycl_queue=exec_q, depends=dep_evs
        )
        _manager.add_event_pair(h_ev, c_ev)
        return

    dt = arr.dtype
    val_type = type(val)
    if val_type in [float, complex] and dpnp.issubdtype(dt, dpnp.integer):
        val = int(val.real)
    elif val_type is complex and dpnp.issubdtype(dt, dpnp.floating):
        val = val.real
    elif val_type is int and dpnp.issubdtype(dt, dpnp.integer):
        val = np.asarray(val, dtype=dt)[()]

    _manager = SequentialOrderManager[exec_q]
    dep_evs = _manager.submitted_events
    # can leverage efficient memset when val is 0
    if arr.flags["FORC"] and val == 0:
        h_ev, zeros_ev = _zeros_usm_ndarray(arr, exec_q, depends=dep_evs)
        _manager.add_event_pair(h_ev, zeros_ev)
    else:
        h_ev, fill_ev = _full_usm_ndarray(val, arr, exec_q, depends=dep_evs)
        _manager.add_event_pair(h_ev, fill_ev)
