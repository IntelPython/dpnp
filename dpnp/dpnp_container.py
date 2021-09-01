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
from dpnp.dparray import dparray

import numpy


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
    "create_output_container"
]


# https://github.com/IntelPython/dpctl/blob/3fe25706995e76255a931d8ed87786da69db685c/dpctl/tests/test_usm_ndarray_ctor.py#L157  # noqa
def _to_numpy(usm_ary):
    if type(usm_ary) is dpctl.usm_ndarray:
        usm_buf = usm_ary.usm_data
        s = usm_buf.nbytes
        host_buf = usm_buf.copy_to_host().view(usm_ary.dtype)
        usm_ary_itemsize = usm_ary.itemsize
        R_offset = (
            usm_ary.__sycl_usm_array_interface__["offset"] * usm_ary_itemsize
        )
        R = numpy.ndarray((s,), dtype="u1", buffer=host_buf)
        R = R[R_offset:].view(usm_ary.dtype)
        R_strides = (usm_ary_itemsize * si for si in usm_ary.strides)
        return np_st.as_strided(R, shape=usm_ary.shape, strides=R_strides)
    else:
        raise ValueError(
            "Expected dpctl.tensor.usm_ndarray, got {}".format(type(usm_ary))
        )


def create_output_container(shape, type):
    if config.__DPNP_OUTPUT_NUMPY__:
        """ Create NumPy ndarray """
        # TODO need to use "buffer=" parameter to use SYCL aware memory
        result = numpy.ndarray(shape, dtype=type)
    elif config.__DPNP_OUTPUT_DPCTL__:
        """ Create DPCTL array """
        if config.__DPNP_OUTPUT_DPCTL_DEFAULT_SHARED__:
            """
            From DPCtrl documentation:
            'buffer can be strings ('device'|'shared'|'host' to allocate new memory)'
            """
            result = dpctl.usm_ndarray(shape, dtype=numpy.dtype(type).name, buffer='shared')
        else:
            """
            Can't pass 'None' as buffer= parameter to allow DPCtrl uses it's default
            """
            result = dpctl.usm_ndarray(shape, dtype=numpy.dtype(type).name)
    else:
        """ Create DPNP array """
        result = dparray(shape, dtype=type)

    return result    


def container_copy(dst_obj, src_obj, dst_idx = 0):
    """
    Copy values to `dst` by iterating element by element in `input_obj`
    """

    for elem_value in src_obj:
        if isinstance(elem_value, (list, tuple)):
            dst_idx = container_copy(dst_obj, elem_value, dst_idx)
        elif issubclass(type(elem_value), (numpy.ndarray, dparray)):
            dst_idx = container_copy(dst_obj, elem_value, dst_idx)
        else:
            dst_obj.flat[dst_idx] = elem_value
            dst_idx += 1

    return dst_idx
