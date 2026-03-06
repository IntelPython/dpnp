# *****************************************************************************
# Copyright (c) 2026, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
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

from dpctl._sycl_device import SyclDevice

from ._usmarray import DLDeviceType


def dldevice_to_sycl_device(dl_dev: tuple):
    if isinstance(dl_dev, tuple):
        if len(dl_dev) != 2:
            raise ValueError("dldevice tuple must have length 2")
    else:
        raise TypeError(
            f"dl_dev is expected to be a 2-tuple, got " f"{type(dl_dev)}"
        )
    if dl_dev[0] != DLDeviceType.kDLOneAPI:
        raise ValueError("dldevice type must be kDLOneAPI")
    return SyclDevice(str(dl_dev[1]))


def sycl_device_to_dldevice(dev: SyclDevice):
    if not isinstance(dev, SyclDevice):
        raise TypeError(
            "dev is expected to be a SyclDevice, got " f"{type(dev)}"
        )
    return (DLDeviceType.kDLOneAPI, dev.get_device_id())
