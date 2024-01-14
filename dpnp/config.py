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


import os

__DPNP_ORIGIN__ = int(os.getenv("DPNP_ORIGIN", 0))
"""
Explicitly use original host Python NumPy
"""

__DPNP_QUEUE_GPU__ = int(os.getenv("DPNP_QUEUE_GPU", 0))
"""
Explicitly use GPU for SYCL queue
"""

__DPNP_OUTPUT_NUMPY__ = int(os.getenv("DPNP_OUTPUT_NUMPY", 0))
"""
Explicitly use NumPy.ndarray as return type for creation functions
"""

__DPNP_OUTPUT_DPCTL_DEFAULT_SHARED__ = int(
    os.getenv("DPNP_OUTPUT_DPCTL_DEFAULT_SHARED", 0)
)
"""
Explicitly use SYCL shared memory parameter in DPCtl array constructor for creation functions
"""

__DPNP_RAISE_EXCEPION_ON_NUMPY_FALLBACK__ = int(
    os.getenv("DPNP_RAISE_EXCEPION_ON_NUMPY_FALLBACK", 1)
)
"""
Trigger non-implemented exception when DPNP fallbacks on NumPy implementation
"""
