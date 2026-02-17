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

import dpctl
import dpctl.memory as dpm
import dpctl.tensor as dpt
import dpctl.utils
import numpy as np
from dpctl.tensor._device import normalize_queue_device

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor._tensor_impl as ti

__doc__ = (
    "Implementation module for copy- and cast- operations on "
    ":class:`dpctl.tensor.usm_ndarray`."
)

int32_t_max = 1 + np.iinfo(np.int32).max


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


def _copy_from_numpy(np_ary, usm_type="device", sycl_queue=None):
    """Copies numpy array `np_ary` into a new usm_ndarray"""
    # This may perform a copy to meet stated requirements
    Xnp = np.require(np_ary, requirements=["A", "E"])
    alloc_q = normalize_queue_device(sycl_queue=sycl_queue, device=None)
    dt = Xnp.dtype
    if dt.char in "dD" and alloc_q.sycl_device.has_aspect_fp64 is False:
        Xusm_dtype = (
            dpt.dtype("float32") if dt.char == "d" else dpt.dtype("complex64")
        )
    else:
        Xusm_dtype = dt
    Xusm = dpt.empty(
        Xnp.shape, dtype=Xusm_dtype, usm_type=usm_type, sycl_queue=sycl_queue
    )
    _copy_from_numpy_into(Xusm, Xnp)
    return Xusm


def _copy_from_numpy_into(dst, np_ary):
    """Copies `np_ary` into `dst` of type :class:`dpctl.tensor.usm_ndarray"""
    if not isinstance(np_ary, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(np_ary)}")
    if not isinstance(dst, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray, got {type(dst)}")
    if np_ary.flags["OWNDATA"]:
        Xnp = np_ary
    else:
        # Determine base of input array
        base = np_ary.base
        while isinstance(base, np.ndarray):
            base = base.base
        if isinstance(base, dpm._memory._Memory):
            # we must perform a copy, since subsequent
            # _copy_numpy_ndarray_into_usm_ndarray is implemented using
            # sycl::buffer, and using USM-pointers with sycl::buffer
            # results is undefined behavior
            Xnp = np_ary.copy()
        else:
            Xnp = np_ary
    src_ary = np.broadcast_to(Xnp, dst.shape)
    copy_q = dst.sycl_queue
    if copy_q.sycl_device.has_aspect_fp64 is False:
        src_ary_dt_c = src_ary.dtype.char
        if src_ary_dt_c == "d":
            src_ary = src_ary.astype(np.float32)
        elif src_ary_dt_c == "D":
            src_ary = src_ary.astype(np.complex64)
    _manager = dpctl.utils.SequentialOrderManager[copy_q]
    dep_ev = _manager.submitted_events
    # synchronizing call
    ti._copy_numpy_ndarray_into_usm_ndarray(
        src=src_ary, dst=dst, sycl_queue=copy_q, depends=dep_ev
    )


def from_numpy(np_ary, /, *, device=None, usm_type="device", sycl_queue=None):
    """
    from_numpy(arg, device=None, usm_type="device", sycl_queue=None)

    Creates :class:`dpctl.tensor.usm_ndarray` from instance of
    :class:`numpy.ndarray`.

    Args:
        arg:
            Input convertible to :class:`numpy.ndarray`
        device (object): array API specification of device where the
            output array is created. Device can be specified by
            a filter selector string, an instance of
            :class:`dpctl.SyclDevice`, an instance of
            :class:`dpctl.SyclQueue`, or an instance of
            :class:`dpctl.tensor.Device`. If the value is ``None``,
            returned array is created on the default-selected device.
            Default: ``None``
        usm_type (str): The requested USM allocation type for the
            output array. Recognized values are ``"device"``,
            ``"shared"``, or ``"host"``
        sycl_queue (:class:`dpctl.SyclQueue`, optional):
            A SYCL queue that determines output array allocation device
            as well as execution placement of data movement operations.
            The ``device`` and ``sycl_queue`` arguments
            are equivalent. Only one of them should be specified. If both
            are provided, they must be consistent and result in using the
            same execution queue. Default: ``None``

    The returned array has the same shape, and the same data type kind.
    If the device does not support the data type of input array, a
    closest support data type of the same kind may be returned, e.g.
    input array of type ``float16`` may be upcast to ``float32`` if the
    target device does not support 16-bit floating point type.
    """
    q = normalize_queue_device(sycl_queue=sycl_queue, device=device)
    return _copy_from_numpy(np_ary, usm_type=usm_type, sycl_queue=q)


def to_numpy(usm_ary, /):
    """
    to_numpy(usm_ary)

    Copies content of :class:`dpctl.tensor.usm_ndarray` instance ``usm_ary``
    into :class:`numpy.ndarray` instance of the same shape and same data type.

    Args:
        usm_ary (usm_ndarray):
            Input array
    Returns:
        :class:`numpy.ndarray`:
            An instance of :class:`numpy.ndarray` populated with content of
            ``usm_ary``
    """
    return _copy_to_numpy(usm_ary)


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
