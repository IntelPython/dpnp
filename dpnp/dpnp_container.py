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


try:
    """
    Detect DPCtl availability to use data container
    """
    import dpctl.tensor as dpctl

    config.__DPNP_DPCTL_AVAILABLE__ = True

except ImportError:
    """
    No DPCtl data container available
    """
    config.__DPNP_DPCTL_AVAILABLE__ = False

# config.__DPNP_DPCTL_AVAILABLE__ = False


__all__ = [
    "create_output_container"
]


def create_output_container(shape, type):
    if config.__DPNP_OUTPUT_NUMPY__:
        """ Create NumPy ndarray """
        # TODO need to use "buffer=" parameter to use SYCL aware memory
        result = numpy.ndarray(shape, dtype=type)
    elif config.__DPNP_DPCTL_AVAILABLE__:
        """ Create DPCTL array """
        result = dpctl.usm_ndarray(shape, dtype=numpy.dtype(type).name)
    else:
        """ Create DPNP array """
        result = dparray(shape, dtype=type)

    return result    
