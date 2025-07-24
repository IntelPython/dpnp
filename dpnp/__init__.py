# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2025, Intel Corporation
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
import sys

mypath = os.path.dirname(os.path.realpath(__file__))

# workaround against hanging in OneMKL calls and in DPCTL
os.environ.setdefault("SYCL_QUEUE_THREAD_POOL_SIZE", "6")

import dpctl

dpctlpath = os.path.dirname(dpctl.__file__)

# For Windows OS with Python >= 3.7, it is required to explicitly define a path
# where to search for DLLs towards both DPNP backend and DPCTL Sycl interface,
# otherwise DPNP import will be failing. This is because the libraries
# are not installed under any of default paths where Python is searching.
from platform import system

if system() == "Windows":  # pragma: no cover
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(mypath)
        os.add_dll_directory(dpctlpath)

    os.environ["PATH"] = os.pathsep.join(
        [os.getenv("PATH", ""), mypath, dpctlpath]
    )

    # For virtual environments on Windows, add folder with DPC++ libraries
    # to the DLL search path
    if sys.base_exec_prefix != sys.exec_prefix and os.path.isfile(
        os.path.join(sys.exec_prefix, "pyvenv.cfg")
    ):
        dll_path = os.path.join(sys.exec_prefix, "Library", "bin")
        if os.path.isdir(dll_path):
            os.environ["PATH"] = os.pathsep.join(
                [os.getenv("PATH", ""), dll_path]
            )

# Borrowed from DPCTL
from dpctl.tensor import __array_api_version__, DLDeviceType

from .dpnp_array import dpnp_array as ndarray
from .dpnp_array_api_info import __array_namespace_info__
from .dpnp_flatiter import flatiter as flatiter
from .dpnp_iface_types import *
from .dpnp_iface import *
from .dpnp_iface import __all__ as _iface__all__
from .dpnp_iface_utils import *
from .dpnp_iface_utils import __all__ as _ifaceutils__all__
from ._version import get_versions

__all__ = _iface__all__
__all__ += _ifaceutils__all__


__version__ = get_versions()["version"]
del get_versions
