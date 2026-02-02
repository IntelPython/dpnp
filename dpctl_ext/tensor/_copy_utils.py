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

import dpctl.memory as dpm
import numpy as np

# TODO: revert to `import dpctl.tensor as dpt`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt

__doc__ = (
    "Implementation module for copy- and cast- operations on "
    ":class:`dpctl.tensor.usm_ndarray`."
)


def _copy_to_numpy(ary):
    if not isinstance(ary, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(ary)}")
    if ary.size == 0:
        # no data needs to be copied for zero sized array
        return np.ndarray(ary.shape, dtype=ary.dtype)
    nb = ary.usm_data.nbytes
    q = ary.sycl_queue
    hh = dpm.MemoryUSMHost(nb, queue=q)
    h = np.ndarray(nb, dtype="u1", buffer=hh).view(ary.dtype)
    itsz = ary.itemsize
    strides_bytes = tuple(si * itsz for si in ary.strides)
    offset = ary._element_offset * itsz
    # ensure that content of ary.usm_data is final
    q.wait()
    hh.copy_from_device(ary.usm_data)
    return np.ndarray(
        ary.shape,
        dtype=ary.dtype,
        buffer=h,
        strides=strides_bytes,
        offset=offset,
    )


def asnumpy(usm_ary):
    """
    asnumpy(usm_ary)

    Copies content of :class:`dpctl.tensor.usm_ndarray` instance ``usm_ary``
    into :class:`numpy.ndarray` instance of the same shape and same data
    type.

    Args:
        usm_ary (usm_ndarray):
            Input array
    Returns:
        :class:`numpy.ndarray`:
            An instance of :class:`numpy.ndarray` populated with content
            of ``usm_ary``
    """
    return _copy_to_numpy(usm_ary)
