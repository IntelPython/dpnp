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

import sys
import warnings

# import os
# myvar=os.environ

# import pprint
# pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(myvar)

# os.system("ldd ./dpnp/libdpnp_backend_c.so")
# subprocess.call("echo Hello 123", shell=True)

if "numpy" in sys.modules:
    warnings.warn("\nDPNP: Module NumPy found. Please load DPNP module before NumPy.\n")

# from dpnp.dparray import dparray as ndarray
from dpnp.dpnp_array import dpnp_array as ndarray
from dpnp.dpnp_flatiter import flatiter as flatiter

from dpnp.dpnp_iface import *
from dpnp.dpnp_iface import __all__ as _iface__all__
from dpnp.dpnp_iface_types import *
from dpnp.version import __version__


__all__ = _iface__all__


dpnp_queue_initialize()
'''
SYCL queue initialization
'''
