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

import collections
import ctypes

import dpctl

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt
import dpctl_ext.tensor._dlpack as _dlp
import dpctl_ext.tensor._usmarray as dpt_arr
import numpy as np
import pytest

from .helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)

device_CPU = dpt_arr.DLDeviceType.kDLCPU
device_oneAPI = dpt_arr.DLDeviceType.kDLOneAPI

_usm_types_list = ["shared", "device", "host"]


@pytest.fixture(params=_usm_types_list)
def usm_type(request):
    return request.param


_typestrs_list = [
    "b1",
    "u1",
    "i1",
    "u2",
    "i2",
    "u4",
    "i4",
    "u8",
    "i8",
    "f2",
    "f4",
    "f8",
    "c8",
    "c16",
]


@pytest.fixture(params=_typestrs_list)
def typestr(request):
    return request.param


@pytest.fixture
def all_root_devices():
    """
    Caches root devices. For the sake of speed
    of test suite execution, keep at most two
    devices from each platform
    """
    devs = dpctl.get_devices()
    devs_per_platform = collections.defaultdict(list)
    for dev in devs:
        devs_per_platform[dev.sycl_platform].append(dev)

    pruned = map(lambda li: li[:2], devs_per_platform.values())
    return sum(pruned, start=[])


def test_dlpack_device(usm_type, all_root_devices):
    for sycl_dev in all_root_devices:
        X = dpt.empty((64,), dtype="u1", usm_type=usm_type, device=sycl_dev)
        dev = X.__dlpack_device__()
        assert type(dev) is tuple
        assert len(dev) == 2
        assert dev[0] == device_oneAPI
        assert dev[1] == sycl_dev.get_device_id()


def test_dlpack_exporter(typestr, usm_type, all_root_devices):
    caps_fn = ctypes.pythonapi.PyCapsule_IsValid
    caps_fn.restype = bool
    caps_fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
    for sycl_dev in all_root_devices:
        skip_if_dtype_not_supported(typestr, sycl_dev)
        X = dpt.empty((64,), dtype=typestr, usm_type=usm_type, device=sycl_dev)
        caps = X.__dlpack__()
        assert caps_fn(caps, b"dltensor")
        Y = X[::2]
        caps2 = Y.__dlpack__()
        assert caps_fn(caps2, b"dltensor")


def test_dlpack_exporter_empty(typestr, usm_type):
    caps_fn = ctypes.pythonapi.PyCapsule_IsValid
    caps_fn.restype = bool
    caps_fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
    try:
        sycl_dev = dpctl.select_default_device()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    skip_if_dtype_not_supported(typestr, sycl_dev)
    X = dpt.empty((0,), dtype=typestr, usm_type=usm_type, device=sycl_dev)
    caps = X.__dlpack__()
    assert caps_fn(caps, b"dltensor")
    Y = dpt.empty(
        (
            1,
            0,
        ),
        dtype=typestr,
        usm_type=usm_type,
        device=sycl_dev,
    )
    caps = Y.__dlpack__()
    assert caps_fn(caps, b"dltensor")


def test_dlpack_exporter_stream():
    try:
        q1 = dpctl.SyclQueue()
        q2 = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Could not create default queues")
    X = dpt.empty((64,), dtype="u1", sycl_queue=q1)
    cap1 = X.__dlpack__(stream=q1)
    cap2 = X.__dlpack__(stream=q2)
    assert type(cap1) is type(cap2)


@pytest.mark.parametrize("shape", [tuple(), (2,), (3, 0, 1), (2, 2, 2)])
def test_from_dlpack(shape, typestr, usm_type, all_root_devices):
    for sycl_dev in all_root_devices:
        skip_if_dtype_not_supported(typestr, sycl_dev)
        X = dpt.empty(shape, dtype=typestr, usm_type=usm_type, device=sycl_dev)
        Y = dpt.from_dlpack(X)
        assert X.shape == Y.shape
        assert X.dtype == Y.dtype
        assert X.usm_type == Y.usm_type
        assert X._pointer == Y._pointer
        # we can only expect device to round-trip for USM-device and
        # USM-shared allocations, which are made for specific device
        assert (Y.usm_type == "host") or (X.sycl_device == Y.sycl_device)
        if Y.ndim:
            V = Y[::-1]
            W = dpt.from_dlpack(V)
            assert V.strides == W.strides


@pytest.mark.parametrize("mod", [2, 5])
def test_from_dlpack_strides(mod, typestr, usm_type, all_root_devices):
    for sycl_dev in all_root_devices:
        skip_if_dtype_not_supported(typestr, sycl_dev)
        X0 = dpt.empty(
            3 * mod, dtype=typestr, usm_type=usm_type, device=sycl_dev
        )
        for start in range(mod):
            X = X0[slice(-start - 1, None, -mod)]
            Y = dpt.from_dlpack(X)
            assert X.shape == Y.shape
            assert X.dtype == Y.dtype
            assert X.usm_type == Y.usm_type
            assert X._pointer == Y._pointer
            # we can only expect device to round-trip for USM-device and
            # USM-shared allocations, which are made for specific device
            assert (Y.usm_type == "host") or (X.sycl_device == Y.sycl_device)
            if Y.ndim:
                V = Y[::-1]
                W = dpt.from_dlpack(V)
                assert V.strides == W.strides


def test_from_dlpack_input_validation():
    v = dpt._dlpack.get_build_dlpack_version()
    assert type(v) is tuple
    with pytest.raises(TypeError):
        dpt.from_dlpack(None)

    class DummyWithProperty:
        @property
        def __dlpack__(self):
            return None

    with pytest.raises(TypeError):
        dpt.from_dlpack(DummyWithProperty())

    class DummyWithMethod:
        def __dlpack__(self):
            return None

    with pytest.raises(TypeError):
        dpt.from_dlpack(DummyWithMethod())


def test_from_dlpack_fortran_contig_array_roundtripping():
    """Based on examples from issue gh-1241"""
    n0, n1 = 3, 5
    try:
        ar1d = dpt.arange(n0 * n1, dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No default device available")
    ar2d_c = dpt.reshape(ar1d, (n0, n1), order="C")
    ar2d_f = dpt.asarray(ar2d_c, order="F")
    ar2d_r = dpt.from_dlpack(ar2d_f)

    assert dpt.all(dpt.equal(ar2d_f, ar2d_r))
    assert dpt.all(dpt.equal(ar2d_c, ar2d_r))


def test_dlpack_from_subdevice():
    """
    This test checks that array allocated on a sub-device,
    with memory bound to platform-default SyclContext can be
    exported and imported via DLPack.
    """
    n = 64
    try:
        dev = dpctl.SyclDevice()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No default device available")
    try:
        sdevs = dev.create_sub_devices(partition="next_partitionable")
    except dpctl.SyclSubDeviceCreationError:
        sdevs = None
    try:
        if sdevs is None:
            sdevs = dev.create_sub_devices(partition=[1, 1])
    except dpctl.SyclSubDeviceCreationError:
        pytest.skip("Default device can not be partitioned")
    assert isinstance(sdevs, list) and len(sdevs) > 0
    try:
        ctx = sdevs[0].sycl_platform.default_context
    except dpctl.SyclContextCreationError:
        pytest.skip("Platform's default_context is not available")
    try:
        q = dpctl.SyclQueue(ctx, sdevs[0])
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    ar = dpt.arange(n, dtype=dpt.int32, sycl_queue=q)
    ar2 = dpt.from_dlpack(ar)
    assert ar2.sycl_device == sdevs[0]


def test_legacy_dlpack_capsule():
    try:
        x = dpt.arange(100, dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No default device available")

    legacy_ver = (0, 8)

    cap = x.__dlpack__(max_version=legacy_ver)
    y = _dlp.from_dlpack_capsule(cap)
    del cap
    assert x._pointer == y._pointer

    x = dpt.arange(100, dtype="u4")
    x2 = dpt.reshape(x, (10, 10)).mT
    cap = x2.__dlpack__(max_version=legacy_ver)
    y = _dlp.from_dlpack_capsule(cap)
    del cap
    assert x2._pointer == y._pointer
    del x2

    x = dpt.arange(100, dtype="f4")
    x2 = dpt.asarray(dpt.reshape(x, (10, 10)), order="F")
    cap = x2.__dlpack__(max_version=legacy_ver)
    y = _dlp.from_dlpack_capsule(cap)
    del cap
    assert x2._pointer == y._pointer

    x = dpt.arange(100, dtype="c8")
    x3 = x[::-2]
    cap = x3.__dlpack__(max_version=legacy_ver)
    y = _dlp.from_dlpack_capsule(cap)
    assert x3._pointer == y._pointer
    del x3, y, x
    del cap

    x = dpt.ones(100, dtype="?")
    x4 = x[::-2]
    cap = x4.__dlpack__(max_version=legacy_ver)
    y = _dlp.from_dlpack_capsule(cap)
    assert x4._pointer == y._pointer
    del x4, y, x
    del cap


def test_versioned_dlpack_capsule():
    try:
        x = dpt.arange(100, dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No default device available")

    max_supported_ver = _dlp.get_build_dlpack_version()
    cap = x.__dlpack__(max_version=max_supported_ver)
    y = _dlp.from_dlpack_capsule(cap)
    del cap
    assert x._pointer == y._pointer

    x2 = dpt.asarray(dpt.reshape(x, (10, 10)), order="F")
    cap = x2.__dlpack__(max_version=max_supported_ver)
    y = _dlp.from_dlpack_capsule(cap)
    del cap
    assert x2._pointer == y._pointer
    del x2

    x3 = x[::-2]
    cap = x3.__dlpack__(max_version=max_supported_ver)
    y = _dlp.from_dlpack_capsule(cap)
    assert x3._pointer == y._pointer
    del x3, y, x
    del cap

    # read-only array
    x = dpt.arange(100, dtype="i4")
    x.flags["W"] = False
    cap = x.__dlpack__(max_version=max_supported_ver)
    y = _dlp.from_dlpack_capsule(cap)
    assert x._pointer == y._pointer
    assert not y.flags.writable

    # read-only array, and copy
    cap = x.__dlpack__(max_version=max_supported_ver, copy=True)
    y = _dlp.from_dlpack_capsule(cap)
    assert x._pointer != y._pointer
    assert not y.flags.writable


def test_from_dlpack_kwargs():
    try:
        x = dpt.arange(100, dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No default device available")

    y = dpt.from_dlpack(x, copy=True)
    assert x._pointer != y._pointer

    z = dpt.from_dlpack(x, device=x.sycl_device)
    assert z._pointer == x._pointer


def test_dlpack_deleters():
    try:
        x = dpt.arange(100, dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No default device available")

    legacy_ver = (0, 8)
    cap = x.__dlpack__(max_version=legacy_ver)
    del cap

    max_supported_ver = _dlp.get_build_dlpack_version()
    cap = x.__dlpack__(max_version=max_supported_ver)
    del cap


def test_from_dlpack_device():
    try:
        x = dpt.arange(100, dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No default device available")

    out = dpt.from_dlpack(x, device=x.__dlpack_device__())
    assert x.device == out.device
    assert x._pointer == out._pointer

    out = dpt.from_dlpack(x, device=x.device)
    assert x.device == out.device
    assert x._pointer == out._pointer

    out = dpt.from_dlpack(x, device=x.sycl_device)
    assert x.device == out.device
    assert x._pointer == out._pointer


def test_used_dlpack_capsule():
    try:
        x = dpt.arange(100, dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No default device available")

    legacy_ver = (0, 8)
    cap = x.__dlpack__(max_version=legacy_ver)
    _dlp.from_dlpack_capsule(cap)
    with pytest.raises(
        ValueError,
        match="A DLPack tensor object can not be consumed multiple times",
    ):
        _dlp.from_dlpack_capsule(cap)
    del cap

    max_supported_ver = _dlp.get_build_dlpack_version()
    cap = x.__dlpack__(max_version=max_supported_ver)
    _dlp.from_dlpack_capsule(cap)
    with pytest.raises(
        ValueError,
        match="A DLPack tensor object can not be consumed multiple times",
    ):
        _dlp.from_dlpack_capsule(cap)
    del cap


def test_dlpack_size_0():
    try:
        x = dpt.ones(0, dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No default device available")

    legacy_ver = (0, 8)
    cap = x.__dlpack__(max_version=legacy_ver)
    y = _dlp.from_dlpack_capsule(cap)
    assert y._pointer == x._pointer

    max_supported_ver = _dlp.get_build_dlpack_version()
    cap = x.__dlpack__(max_version=max_supported_ver)
    y = _dlp.from_dlpack_capsule(cap)
    assert y._pointer == x._pointer


def test_dlpack_max_version_validation():
    try:
        x = dpt.ones(100, dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No default device available")

    with pytest.raises(
        TypeError,
        match=r"`__dlpack__` expects `max_version` to be a "
        r"2-tuple of integers `\(major, minor\)`, instead "
        r"got .*",
    ):
        x.__dlpack__(max_version=1)


def test_dlpack_kwargs():
    try:
        q1 = dpctl.SyclQueue()
        q2 = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Could not create default queues")
    x = dpt.arange(100, dtype="i4", sycl_queue=q1)

    legacy_ver = (0, 8)
    cap = x.__dlpack__(stream=q2, max_version=legacy_ver, copy=True)
    # `copy` ignored for legacy path
    y = _dlp.from_dlpack_capsule(cap)
    assert y._pointer == x._pointer
    del x, y
    del cap

    x1 = dpt.arange(100, dtype="i4", sycl_queue=q1)
    max_supported_ver = _dlp.get_build_dlpack_version()
    cap = x1.__dlpack__(stream=q2, max_version=max_supported_ver, copy=False)
    y = _dlp.from_dlpack_capsule(cap)
    assert y._pointer == x1._pointer
    del x1, y
    del cap

    x2 = dpt.arange(100, dtype="i4", sycl_queue=q1)
    cap = x2.__dlpack__(stream=q2, max_version=max_supported_ver, copy=True)
    y = _dlp.from_dlpack_capsule(cap)
    assert y._pointer != x2._pointer
    del x2, y
    del cap


def _is_capsule(o):
    t = type(o)
    return t.__module__ == "builtins" and t.__name__ == "PyCapsule"


def test_dlpack_dl_device():
    try:
        x = dpt.arange(100, dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    max_supported_ver = _dlp.get_build_dlpack_version()
    cap1 = x.__dlpack__(
        dl_device=x.__dlpack_device__(), max_version=max_supported_ver
    )
    assert _is_capsule(cap1)
    cap2 = x.__dlpack__(dl_device=(1, 0), max_version=max_supported_ver)
    assert _is_capsule(cap2)
    cap3 = x.__dlpack__(
        dl_device=(device_CPU, 0),
        max_version=max_supported_ver,
    )
    assert _is_capsule(cap3)
    cap4 = x.__dlpack__(dl_device=("kDLCPU", 0), max_version=max_supported_ver)
    assert _is_capsule(cap4)
    with pytest.raises(TypeError):
        # pass method instead of return of its __call__ invocation
        x.__dlpack__(
            dl_device=x.__dlpack_device__, max_version=max_supported_ver
        )
    with pytest.raises(TypeError):
        # exercise check for length
        x.__dlpack__(dl_device=(3,), max_version=max_supported_ver)


def test_from_dlpack_kdlcpu_interop_numpy():
    """
    Basic test that usm_ndarray can interoperate with NumPy ndarray
    `__dlpack_device__`.
    """
    get_queue_or_skip()

    sh = 5
    dt = dpt.int32

    X = dpt.empty(sh, dtype=dt)
    dl_device_np = np.empty(()).__dlpack_device__()

    Y = dpt.from_dlpack(X, device=dl_device_np)
    assert isinstance(Y, np.ndarray)
    assert X.shape == Y.shape
    assert X.dtype == Y.dtype

    V = dpt.from_dlpack(Y)
    assert isinstance(V, np.ndarray)
    assert Y.shape == V.shape
    assert Y.dtype == V.dtype


@pytest.mark.parametrize("shape", [tuple(), (2,), (3, 0, 1), (2, 2, 2)])
def test_from_dlpack_to_kdlcpu(shape, typestr):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(typestr, q.sycl_device)

    X = dpt.empty(shape, dtype=typestr, sycl_queue=q)
    Y = dpt.from_dlpack(X, device=(device_CPU, 0))
    assert isinstance(Y, np.ndarray)
    assert X.shape == Y.shape
    assert X.dtype == Y.dtype
    # NumPy does not treat size 0 arrays consistently
    # w.r.t. strides, so skip these cases
    if X.ndim and X.size != 0:
        V = Y[::-1]
        W = dpt.from_dlpack(V)
        assert V.strides == W.strides


@pytest.mark.parametrize("mod", [2, 5])
def test_from_dlpack_to_kdlcpu_strides(mod, typestr):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(typestr, q.sycl_device)

    X0 = dpt.empty(3 * mod, dtype=typestr, sycl_queue=q)
    for start in range(mod):
        X = X0[slice(-start - 1, None, -mod)]
        Y = dpt.from_dlpack(X, device=(device_CPU, 0))
        assert X.shape == Y.shape
        assert X.dtype == Y.dtype
        if Y.ndim:
            V = Y[::-1]
            W = dpt.from_dlpack(V)
            assert V.strides == W.strides


def test_dlpack_from_subdevice_to_kdlcpu():
    """
    Check that array allocated on a sub-device can be
    imported via DLPack to kDLCPU device (as a NumPy array).
    """
    n = 64
    try:
        dev = dpctl.SyclDevice()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No default device available")
    try:
        sdevs = dev.create_sub_devices(partition="next_partitionable")
    except dpctl.SyclSubDeviceCreationError:
        sdevs = None
    try:
        if sdevs is None:
            sdevs = dev.create_sub_devices(partition=[1, 1])
    except dpctl.SyclSubDeviceCreationError:
        pytest.skip("Default device can not be partitioned")
    assert isinstance(sdevs, list) and len(sdevs) > 0
    try:
        ctx = sdevs[0].sycl_platform.default_context
    except dpctl.SyclContextCreationError:
        pytest.skip("Platform's default_context is not available")
    try:
        q = dpctl.SyclQueue(ctx, sdevs[0])
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    ar = dpt.arange(n, dtype=dpt.int32, sycl_queue=q)
    ar2 = dpt.from_dlpack(ar, dl_device=(device_CPU, 0))
    assert isinstance(ar2, np.ndarray)


def test_legacy_dlpack_capsule_from_numpy():
    """
    Check that NumPy's exported legacy DLPack capsule
    will interoperate with from_dlpack_capsule,
    especially with zero-copy.
    """
    x = np.arange(100, dtype="i4")
    cap = x.__dlpack__()
    y = _dlp.from_dlpack_capsule(cap)
    del cap
    assert x.ctypes.data == y.ctypes.data

    x = np.arange(100, dtype="u4").reshape((10, 10)).T
    cap = x.__dlpack__()
    y = _dlp.from_dlpack_capsule(cap)
    del cap
    assert x.ctypes.data == y.ctypes.data
    del x

    x = np.arange(100, dtype="f4").reshape((10, 10), order="F")
    cap = x.__dlpack__()
    y = _dlp.from_dlpack_capsule(cap)
    del cap
    assert x.ctypes.data == y.ctypes.data

    x = np.arange(100, dtype="c8")
    x1 = x[::-2]
    cap = x1.__dlpack__()
    y = _dlp.from_dlpack_capsule(cap)
    assert x1.ctypes.data == y.ctypes.data
    del x1, y, x
    del cap

    x = np.ones(100, dtype="?")
    x1 = x[::-2]
    cap = x1.__dlpack__()
    y = _dlp.from_dlpack_capsule(cap)
    assert x1.ctypes.data == y.ctypes.data
    del x1, y, x
    del cap


def test_dlpack_capsule_readonly_array_to_kdlcpu():
    try:
        x = dpt.arange(100, dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No default device available")

    max_supported_ver = _dlp.get_build_dlpack_version()
    # read-only array
    x.flags["W"] = False
    cap = x.__dlpack__(max_version=max_supported_ver, dl_device=(device_CPU, 0))
    y = _dlp.from_dlpack_capsule(cap)
    assert dpt.all(x == dpt.asarray(y))
    assert not y.flags["W"]

    cap1 = _dlp.numpy_to_dlpack_versioned_capsule(y, not y.flags["W"])
    y1 = _dlp.from_dlpack_capsule(cap1)
    assert not y1.flags["W"]


def test_to_dlpack_capsule_c_and_f_contig():
    try:
        x = dpt.asarray(np.random.rand(2, 3))
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No default device available")

    cap = _dlp.to_dlpack_capsule(x)
    y = _dlp.from_dlpack_capsule(cap)
    assert np.allclose(dpt.asnumpy(x), dpt.asnumpy(y))
    assert x.strides == y.strides

    x_f = x.T
    cap = _dlp.to_dlpack_capsule(x_f)
    yf = _dlp.from_dlpack_capsule(cap)
    assert np.allclose(dpt.asnumpy(x_f), dpt.asnumpy(yf))
    assert x_f.strides == yf.strides
    del cap


def test_to_dlpack_versioned_capsule_c_and_f_contig():
    try:
        x = dpt.asarray(np.random.rand(2, 3))
        max_supported_ver = _dlp.get_build_dlpack_version()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No default device available")

    cap = x.__dlpack__(max_version=max_supported_ver)
    y = _dlp.from_dlpack_capsule(cap)
    assert np.allclose(dpt.asnumpy(x), dpt.asnumpy(y))
    assert x.strides == y.strides

    x_f = x.T
    cap = x_f.__dlpack__(max_version=max_supported_ver)
    yf = _dlp.from_dlpack_capsule(cap)
    assert np.allclose(dpt.asnumpy(x_f), dpt.asnumpy(yf))
    assert x_f.strides == yf.strides
    del cap


def test_used_dlpack_capsule_from_numpy():
    get_queue_or_skip()

    x_np = np.arange(100, dtype="i4")

    cap = x_np.__dlpack__()
    _dlp.from_dlpack_capsule(cap)
    with pytest.raises(
        ValueError,
        match="A DLPack tensor object can not be consumed multiple times",
    ):
        _dlp.from_dlpack_capsule(cap)
    del cap

    x = dpt.asarray(x_np)
    max_supported_ver = _dlp.get_build_dlpack_version()
    cap = x.__dlpack__(max_version=max_supported_ver, dl_device=(device_CPU, 0))
    _dlp.from_dlpack_capsule(cap)
    with pytest.raises(
        ValueError,
        match="A DLPack tensor object can not be consumed multiple times",
    ):
        _dlp.from_dlpack_capsule(cap)
    del cap


def test_dlpack_size_0_on_kdlcpu():
    get_queue_or_skip()
    x_np = np.ones(0, dtype="i4")

    cap = x_np.__dlpack__()
    y = _dlp.from_dlpack_capsule(cap)
    assert y.ctypes.data == x_np.ctypes.data


def test_copy_via_host():
    get_queue_or_skip()
    x = dpt.ones(1, dtype="i4")
    x_np = np.ones(1, dtype="i4")
    x_dl_dev = x.__dlpack_device__()
    y = dpt.from_dlpack(x_np, device=x_dl_dev)
    assert isinstance(y, dpt.usm_ndarray)
    assert y.sycl_device == x.sycl_device
    assert y.usm_type == "device"

    with pytest.raises(ValueError):
        # incorrect length of tuple
        dpt.from_dlpack(x_np, device=(1, 0, 0))
    with pytest.raises(ValueError):
        # only kDLCPU and kDLOneAPI are supported
        dpt.from_dlpack(x, device=(2, 0))

    num_devs = dpctl.get_num_devices()
    if num_devs > 1:
        j = [i for i in range(num_devs) if i != x_dl_dev[1]][0]
        z = dpt.from_dlpack(x, device=(x_dl_dev[0], j))
        assert isinstance(z, dpt.usm_ndarray)
        assert z.usm_type == "device"


def test_copy_via_host_gh_1789():
    "Test based on review example from gh-1789"
    get_queue_or_skip()
    x_np = np.ones((10, 10), dtype="i4")
    # strides are no longer multiple of itemsize
    x_np.strides = (x_np.strides[0] - 1, x_np.strides[1])
    with pytest.raises(BufferError):
        dpt.from_dlpack(x_np)
    with pytest.raises(BufferError):
        dpt.from_dlpack(x_np, device=(14, 0))


class LegacyContainer:
    "Helper class implementing legacy `__dlpack__` protocol"

    def __init__(self, array):
        self._array = array

    def __dlpack__(self, stream=None):
        return self._array.__dlpack__(stream=stream)

    def __dlpack_device__(self):
        return self._array.__dlpack_device__()


class Container:
    "Helper class implementing `__dlpack__` protocol version 1.0"

    def __init__(self, array):
        self._array = array

    def __dlpack__(
        self, max_version=None, dl_device=None, copy=None, stream=None
    ):
        return self._array.__dlpack__(
            max_version=max_version,
            dl_device=dl_device,
            copy=copy,
            stream=stream,
        )

    def __dlpack_device__(self):
        return self._array.__dlpack_device__()


def test_generic_container_legacy():
    get_queue_or_skip()
    C = LegacyContainer(dpt.linspace(0, 100, num=20, dtype="int16"))

    X = dpt.from_dlpack(C)
    assert isinstance(X, dpt.usm_ndarray)
    assert X._pointer == C._array._pointer
    assert X.sycl_device == C._array.sycl_device
    assert X.dtype == C._array.dtype

    Y = dpt.from_dlpack(C, device=(dpt.DLDeviceType.kDLCPU, 0))
    assert isinstance(Y, np.ndarray)
    assert Y.dtype == X.dtype

    Z = dpt.from_dlpack(C, device=X.device)
    assert isinstance(Z, dpt.usm_ndarray)
    assert Z._pointer == X._pointer
    assert Z.device == X.device


def test_generic_container_legacy_np():
    get_queue_or_skip()
    C = LegacyContainer(np.linspace(0, 100, num=20, dtype="int16"))

    X = dpt.from_dlpack(C)
    assert isinstance(X, np.ndarray)
    assert X.ctypes.data == C._array.ctypes.data
    assert X.dtype == C._array.dtype

    Y = dpt.from_dlpack(C, device=(dpt.DLDeviceType.kDLCPU, 0))
    assert isinstance(Y, np.ndarray)
    assert Y.dtype == X.dtype

    dev = dpt.Device.create_device()
    Z = dpt.from_dlpack(C, device=dev)
    assert isinstance(Z, dpt.usm_ndarray)
    assert Z.device == dev


def test_generic_container():
    get_queue_or_skip()
    C = Container(dpt.linspace(0, 100, num=20, dtype="int16"))

    X = dpt.from_dlpack(C)
    assert isinstance(X, dpt.usm_ndarray)
    assert X._pointer == C._array._pointer
    assert X.sycl_device == C._array.sycl_device
    assert X.dtype == C._array.dtype

    Y = dpt.from_dlpack(C, device=(dpt.DLDeviceType.kDLCPU, 0))
    assert isinstance(Y, np.ndarray)
    assert Y.dtype == X.dtype

    Z = dpt.from_dlpack(C, device=X.device)
    assert isinstance(Z, dpt.usm_ndarray)
    assert Z._pointer == X._pointer
    assert Z.device == X.device


def test_sycl_device_to_dldevice(all_root_devices):
    for sycl_dev in all_root_devices:
        dev = dpt.sycl_device_to_dldevice(sycl_dev)
        assert type(dev) is tuple
        assert len(dev) == 2
        assert dev[0] == device_oneAPI
        assert dev[1] == sycl_dev.get_device_id()


def test_dldevice_to_sycl_device(all_root_devices):
    for sycl_dev in all_root_devices:
        dldev = dpt.empty(0, device=sycl_dev).__dlpack_device__()
        dev = dpt.dldevice_to_sycl_device(dldev)
        assert type(dev) is dpctl.SyclDevice
        assert dev.get_device_id() == sycl_dev.get_device_id()


def test_dldevice_conversion_arg_validation():
    bad_dldevice_type = (dpt.DLDeviceType.kDLCPU, 0)
    with pytest.raises(ValueError):
        dpt.dldevice_to_sycl_device(bad_dldevice_type)

    bad_dldevice_len = bad_dldevice_type + (0,)
    with pytest.raises(ValueError):
        dpt.dldevice_to_sycl_device(bad_dldevice_len)

    bad_dldevice = {}
    with pytest.raises(TypeError):
        dpt.dldevice_to_sycl_device(bad_dldevice)

    bad_sycldevice = {}
    with pytest.raises(TypeError):
        dpt.sycl_device_to_dldevice(bad_sycldevice)
