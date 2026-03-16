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
import numpy as np
import pytest

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt
from dpnp.tests.tensor.helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)


@pytest.mark.parametrize(
    "src_usm_type, dst_usm_type",
    [
        ("device", "shared"),
        ("device", "host"),
        ("shared", "device"),
        ("shared", "host"),
        ("host", "device"),
        ("host", "shared"),
    ],
)
def test_asarray_change_usm_type(src_usm_type, dst_usm_type):
    try:
        d = dpctl.SyclDevice()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    X = dpt.empty(10, dtype="u1", usm_type=src_usm_type)
    Y = dpt.asarray(X, usm_type=dst_usm_type)
    assert X.shape == Y.shape
    assert X.usm_type == src_usm_type
    assert Y.usm_type == dst_usm_type

    with pytest.raises(ValueError):
        # zero copy is not possible
        dpt.asarray(X, usm_type=dst_usm_type, copy=False)

    Y = dpt.asarray(X, usm_type=dst_usm_type, sycl_queue=X.sycl_queue)
    assert X.shape == Y.shape
    assert Y.usm_type == dst_usm_type

    Y = dpt.asarray(
        X,
        usm_type=dst_usm_type,
        sycl_queue=X.sycl_queue,
        device=d.get_filter_string(),
    )
    assert X.shape == Y.shape
    assert Y.usm_type == dst_usm_type


def test_asarray_from_numpy():
    Xnp = np.arange(10)
    try:
        Y = dpt.asarray(Xnp, usm_type="device")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    assert type(Y) is dpt.usm_ndarray
    assert Y.shape == Xnp.shape
    assert Y.dtype == Xnp.dtype
    # Fortran contiguous case
    Xnp = np.array([[1, 2, 3], [4, 5, 6]], dtype="f4", order="F")
    Y = dpt.asarray(Xnp, usm_type="shared")
    assert type(Y) is dpt.usm_ndarray
    assert Y.shape == Xnp.shape
    assert Y.dtype == Xnp.dtype
    # general strided case
    Xnp = np.array([[1, 2, 3], [4, 5, 6]], dtype="i8")
    Y = dpt.asarray(Xnp[::-1, ::-1], usm_type="host")
    assert type(Y) is dpt.usm_ndarray
    assert Y.shape == Xnp.shape
    assert Y.dtype == Xnp.dtype


def test_asarray_from_sequence():
    X = [1, 2, 3]
    try:
        Y = dpt.asarray(X, usm_type="device")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    assert type(Y) is dpt.usm_ndarray

    X = [(1, 1), (2.0, 2.0 + 1.0j), range(4, 6), np.array([3, 4], dtype="c16")]
    Y = dpt.asarray(X, usm_type="device")
    assert type(Y) is dpt.usm_ndarray
    assert Y.ndim == 2
    assert Y.shape == (len(X), 2)

    X = []
    Y = dpt.asarray(X, usm_type="device")
    assert type(Y) is dpt.usm_ndarray
    assert Y.shape == (0,)

    X = [[], []]
    Y = dpt.asarray(X, usm_type="device")
    assert type(Y) is dpt.usm_ndarray
    assert Y.shape == (2, 0)

    X = [True, False]
    Y = dpt.asarray(X, usm_type="device")
    assert type(Y) is dpt.usm_ndarray
    assert Y.dtype.kind == "b"


def test_asarray_from_object_with_suai():
    """Test that asarray can deal with opaque objects implementing SUAI"""

    class Dummy:
        def __init__(self, obj, iface):
            self.obj = obj
            self.__sycl_usm_array_interface__ = iface

    try:
        X = dpt.empty((2, 3, 4), dtype="f4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    Y = dpt.asarray(Dummy(X, X.__sycl_usm_array_interface__))
    assert Y.shape == X.shape
    assert X.usm_type == Y.usm_type
    assert X.dtype == Y.dtype
    assert X.sycl_device == Y.sycl_device


def test_asarray_input_validation():
    with pytest.raises(TypeError):
        # copy keyword is not of right type
        dpt.asarray([1], copy="invalid")
    with pytest.raises(TypeError):
        # order keyword is not valid
        dpt.asarray([1], order=1)
    with pytest.raises(TypeError):
        # dtype is not valid
        dpt.asarray([1], dtype="invalid")
    with pytest.raises(ValueError):
        # unexpected value of order
        dpt.asarray([1], order="Z")
    with pytest.raises(TypeError):
        # usm_type is of wrong type
        dpt.asarray([1], usm_type=dict())
    with pytest.raises(ValueError):
        # usm_type has wrong value
        dpt.asarray([1], usm_type="mistake")
    try:
        wrong_queue_type = dpctl.SyclContext()
    except dpctl.SyclContextCreationError:
        # use any other type
        wrong_queue_type = Ellipsis
    with pytest.raises(TypeError):
        # sycl_queue type is not right
        dpt.asarray([1], sycl_queue=wrong_queue_type)
    with pytest.raises(ValueError):
        # sequence is not rectangular
        dpt.asarray([[1], 2])
    with pytest.raises(OverflowError):
        # Python int too large for type
        dpt.asarray(-9223372036854775809, dtype="i4")
    with pytest.raises(ValueError):
        # buffer to usm_ndarray requires a copy
        dpt.asarray(memoryview(np.arange(5)), copy=False)
    with pytest.raises(ValueError):
        # Numpy array to usm_ndarray requires a copy
        dpt.asarray(np.arange(5), copy=False)
    with pytest.raises(ValueError):
        # Python sequence to usm_ndarray requires a copy
        dpt.asarray([1, 2, 3], copy=False)
    with pytest.raises(ValueError):
        # Python scalar to usm_ndarray requires a copy
        dpt.asarray(5, copy=False)


def test_asarray_input_validation2():
    d = dpctl.get_devices()
    if len(d) < 2:
        pytest.skip("Not enough SYCL devices available")

    d0, d1 = d[:2]
    try:
        q0 = dpctl.SyclQueue(d0)
    except dpctl.SyclQueueCreationError:
        pytest.skip(f"SyclQueue could not be created for {d0}")
    try:
        q1 = dpctl.SyclQueue(d1)
    except dpctl.SyclQueueCreationError:
        pytest.skip(f"SyclQueue could not be created for {d1}")
    with pytest.raises(TypeError):
        dpt.asarray([1, 2], sycl_queue=q0, device=q1)


def test_asarray_scalars():
    import ctypes

    try:
        Y = dpt.asarray(5)
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    assert Y.dtype == dpt.dtype(int)
    Y = dpt.asarray(5.2)
    if Y.sycl_device.has_aspect_fp64:
        assert Y.dtype == dpt.dtype(float)
    else:
        assert Y.dtype == dpt.dtype(dpt.float32)
    Y = dpt.asarray(np.float32(2.3))
    assert Y.dtype == dpt.dtype(dpt.float32)
    Y = dpt.asarray(1.0j)
    if Y.sycl_device.has_aspect_fp64:
        assert Y.dtype == dpt.dtype(complex)
    else:
        assert Y.dtype == dpt.dtype(dpt.complex64)
    Y = dpt.asarray(ctypes.c_int(8))
    assert Y.dtype == dpt.dtype(ctypes.c_int)


def test_asarray_copy_false():
    q = get_queue_or_skip()
    rng = np.random.default_rng()
    Xnp = rng.integers(low=-255, high=255, size=(10, 4), dtype=np.int64)
    X = dpt.from_numpy(Xnp, usm_type="device", sycl_queue=q)
    Y1 = dpt.asarray(X, copy=False, order="K")
    assert Y1 is X
    Y1c = dpt.asarray(X, copy=True, order="K")
    assert not (Y1c is X)
    Y2 = dpt.asarray(X, copy=False, order="C")
    assert Y2 is X
    Y3 = dpt.asarray(X, copy=False, order="A")
    assert Y3 is X
    with pytest.raises(ValueError):
        Y1 = dpt.asarray(X, copy=False, order="F")
    Xf = dpt.empty(
        X.shape,
        dtype=X.dtype,
        usm_type="device",
        sycl_queue=X.sycl_queue,
        order="F",
    )
    Xf[:] = X
    Y4 = dpt.asarray(Xf, copy=False, order="K")
    assert Y4 is Xf
    Y5 = dpt.asarray(Xf, copy=False, order="F")
    assert Y5 is Xf
    Y6 = dpt.asarray(Xf, copy=False, order="A")
    assert Y6 is Xf
    with pytest.raises(ValueError):
        dpt.asarray(Xf, copy=False, order="C")


def test_asarray_invalid_dtype():
    q = get_queue_or_skip()
    Xnp = np.array([1, 2, 3], dtype=object)
    with pytest.raises(TypeError):
        dpt.asarray(Xnp, sycl_queue=q)


def test_asarray_cross_device():
    q = get_queue_or_skip()
    qprof = dpctl.SyclQueue(property="enable_profiling")
    x = dpt.empty(10, dtype="i8", sycl_queue=q)
    y = dpt.asarray(x, sycl_queue=qprof)
    assert y.sycl_queue == qprof


def test_asarray_seq_of_arrays_simple():
    get_queue_or_skip()
    r = dpt.arange(10)
    m = dpt.asarray(
        [
            r,
        ]
        * 4
    )
    assert m.shape == (4,) + r.shape
    assert m.dtype == r.dtype
    assert m.device == r.device


def test_asarray_seq_of_arrays():
    get_queue_or_skip()
    m = dpt.ones((2, 4), dtype="i4")
    w = dpt.zeros(4)
    v = dpt.full(4, -1)
    ar = dpt.asarray([m, [w, v]])
    assert ar.shape == (2, 2, 4)
    assert ar.device == m.device
    assert ar.device == w.device
    assert ar.device == v.device


def test_asarray_seq_of_array_different_queue():
    get_queue_or_skip()
    m = dpt.ones((2, 4), dtype="i4")
    w = dpt.zeros(4)
    v = dpt.full(4, -1)
    qprof = dpctl.SyclQueue(property="enable_profiling")
    ar = dpt.asarray([m, [w, v]], sycl_queue=qprof)
    assert ar.shape == (2, 2, 4)
    assert ar.sycl_queue == qprof


def test_asarray_seq_of_suai():
    get_queue_or_skip()

    class Dummy:
        def __init__(self, obj, iface):
            self.obj = obj
            self.__sycl_usm_array_interface__ = iface

    o = dpt.empty(0, usm_type="shared")
    d = Dummy(o, o.__sycl_usm_array_interface__)
    x = dpt.asarray(d)
    assert x.shape == (0,)
    assert x.usm_type == o.usm_type
    assert x._pointer == o._pointer
    assert x.sycl_queue == o.sycl_queue

    x = dpt.asarray([d, d])
    assert x.shape == (2, 0)
    assert x.usm_type == o.usm_type
    assert x.sycl_queue == o.sycl_queue


def test_asarray_seq_of_suai_different_queue():
    q = get_queue_or_skip()

    class Dummy:
        def __init__(self, obj, iface):
            self.obj = obj
            self.__sycl_usm_array_interface__ = iface

        @property
        def shape(self):
            return self.__sycl_usm_array_interface__["shape"]

    q2 = dpctl.SyclQueue()
    assert q != q2
    o = dpt.empty((2, 2), usm_type="shared", sycl_queue=q2)
    d = Dummy(o, o.__sycl_usm_array_interface__)

    x = dpt.asarray(d, sycl_queue=q)
    assert x.sycl_queue == q
    assert x.shape == d.shape
    x = dpt.asarray([d], sycl_queue=q)
    assert x.sycl_queue == q
    assert x.shape == (1,) + d.shape
    x = dpt.asarray([d, d], sycl_queue=q)
    assert x.sycl_queue == q
    assert x.shape == (2,) + d.shape


def test_asarray_seq_of_arrays_on_different_queues():
    q = get_queue_or_skip()

    m = dpt.empty((2, 4), dtype="i2", sycl_queue=q)
    q2 = dpctl.SyclQueue()
    w = dpt.empty(4, dtype="i1", sycl_queue=q2)
    q3 = dpctl.SyclQueue()
    py_seq = [
        0,
    ] * w.shape[0]
    res = dpt.asarray([m, [w, py_seq]], sycl_queue=q3)
    assert res.sycl_queue == q3
    assert dpt.isdtype(res.dtype, "integral")

    res = dpt.asarray([m, [w, range(w.shape[0])]], sycl_queue=q3)
    assert res.sycl_queue == q3
    assert dpt.isdtype(res.dtype, "integral")

    res = dpt.asarray([m, [w, w]], sycl_queue=q)
    assert res.sycl_queue == q
    assert dpt.isdtype(res.dtype, "integral")

    res = dpt.asarray([m, [w, dpt.asnumpy(w)]], sycl_queue=q2)
    assert res.sycl_queue == q2
    assert dpt.isdtype(res.dtype, "integral")

    res = dpt.asarray([w, dpt.asnumpy(w)])
    assert res.sycl_queue == w.sycl_queue
    assert dpt.isdtype(res.dtype, "integral")

    with pytest.raises(dpctl.utils.ExecutionPlacementError):
        dpt.asarray([m, [w, py_seq]])


def test_ulonglong_gh_1167():
    get_queue_or_skip()
    x = dpt.asarray(9223372036854775807, dtype="u8")
    assert x.dtype == dpt.uint64
    x = dpt.asarray(9223372036854775808, dtype="u8")
    assert x.dtype == dpt.uint64


def test_orderK_gh_1350():
    get_queue_or_skip()
    a = dpt.empty((2, 3, 4), dtype="u1")
    b = dpt.permute_dims(a, (2, 0, 1))
    c = dpt.asarray(b, copy=True, order="K")

    assert c.shape == b.shape
    assert c.strides == b.strides
    assert c._element_offset == 0
    assert not c._pointer == b._pointer


def _typesafe_arange(n: int, dtype_: dpt.dtype, device: object):
    n_half = n // 2
    if dtype_.kind in "ui":
        ii = dpt.iinfo(dtype_)
        m0 = max(ii.min, -n_half)
        m1 = min(m0 + n, ii.max)
        n_tiles = (n + m1 - m0 - 1) // (m1 - m0)
        res = dpt.arange(m0, m1, dtype=dtype_, device=device)
    elif dtype_.kind == "b":
        n_tiles = (n + 1) // 2
        res = dpt.asarray([False, True], dtype=dtype_, device=device)
    else:
        m0 = -n_half
        m1 = m0 + n
        n_tiles = 1
        res = dpt.linspace(m0, m1, num=n, dtype=dtype_, device=device)
    if n_tiles > 1:
        res = dpt.tile(res, n_tiles)[:n]
    return res


_all_dtypes = [
    "b1",
    "i1",
    "u1",
    "i2",
    "u2",
    "i4",
    "u4",
    "i8",
    "u8",
    "f2",
    "f4",
    "f8",
    "c8",
    "c16",
]


@pytest.mark.parametrize("dt", _all_dtypes)
def test_as_c_contig_rect(dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    dtype_ = dpt.dtype(dt)
    n0, n1, n2 = 6, 35, 37

    arr_flat = _typesafe_arange(n0 * n1 * n2, dtype_, q)
    x = dpt.reshape(arr_flat, (n0, n1, n2)).mT

    y = dpt.asarray(x, order="C")
    assert dpt.all(x == y)

    x2 = x[0]
    y2 = dpt.asarray(x2, order="C")
    assert dpt.all(x2 == y2)

    x3 = dpt.flip(x, axis=1)
    y3 = dpt.asarray(x3, order="C")
    assert dpt.all(x3 == y3)

    x4 = dpt.reshape(arr_flat, (2, 3, n1, n2)).mT
    x5 = x4[:, :2]
    y5 = dpt.asarray(x5, order="C")
    assert dpt.all(x5 == y5)

    x6 = dpt.reshape(arr_flat, (n0, n1, n2), order="F")
    y6 = dpt.asarray(x6, order="C")
    assert dpt.all(x6 == y6)


@pytest.mark.parametrize("dt", _all_dtypes)
def test_as_f_contig_rect(dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    dtype_ = dpt.dtype(dt)
    n0, n1, n2 = 6, 35, 37

    arr_flat = _typesafe_arange(n0 * n1 * n2, dtype_, q)
    x = dpt.reshape(arr_flat, (n0, n1, n2))

    y = dpt.asarray(x, order="F")
    assert dpt.all(x == y)

    x2 = x[0]
    y2 = dpt.asarray(x2, order="F")
    assert dpt.all(x2 == y2)

    x3 = dpt.flip(x, axis=1)
    y3 = dpt.asarray(x3, order="F")
    assert dpt.all(x3 == y3)

    x4 = dpt.reshape(arr_flat, (2, 3, n1, n2))
    x5 = dpt.moveaxis(x4[:, :2], (2, 3), (0, 1))
    y5 = dpt.asarray(x5, order="F")
    assert dpt.all(x5 == y5)


@pytest.mark.parametrize("dt", _all_dtypes)
def test_as_c_contig_square(dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    dtype_ = dpt.dtype(dt)
    n0, n1 = 4, 53

    arr_flat = _typesafe_arange(n0 * n1 * n1, dtype_, q)
    x = dpt.reshape(arr_flat, (n0, n1, n1)).mT

    y = dpt.asarray(x, order="C")
    assert dpt.all(x == y)

    x2 = x[0]
    y2 = dpt.asarray(x2, order="C")
    assert dpt.all(x2 == y2)

    x3 = dpt.flip(x, axis=1)
    y3 = dpt.asarray(x3, order="C")
    assert dpt.all(x3 == y3)


@pytest.mark.parametrize("dt", _all_dtypes)
def test_as_f_contig_square(dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    dtype_ = dpt.dtype(dt)
    n0, n1 = 6, 53

    arr_flat = _typesafe_arange(n0 * n1 * n1, dtype_, q)
    x = dpt.moveaxis(dpt.reshape(arr_flat, (n0, n1, n1)), (1, 2), (0, 1))

    y = dpt.asarray(x, order="F")
    assert dpt.all(x == y)

    x2 = x[..., 0]
    y2 = dpt.asarray(x2, order="F")
    assert dpt.all(x2 == y2)

    x3 = dpt.flip(x, axis=1)
    y3 = dpt.asarray(x3, order="F")
    assert dpt.all(x3 == y3)


class MockArrayWithBothProtocols:
    """
    Object that implements both __sycl_usm_array_interface__
    and __usm_ndarray__ properties.
    """

    def __init__(self, usm_ar):
        if not isinstance(usm_ar, dpt.usm_ndarray):
            raise TypeError
        self._arr = usm_ar

    @property
    def __usm_ndarray__(self):
        return self._arr

    @property
    def __sycl_usm_array_interface__(self):
        return self._arr.__sycl_usm_array_interface__


class MockArrayWithSUAIOnly:
    """
    Object that implements only the
    __sycl_usm_array_interface__ property.
    """

    def __init__(self, usm_ar):
        if not isinstance(usm_ar, dpt.usm_ndarray):
            raise TypeError
        self._arr = usm_ar

    @property
    def __sycl_usm_array_interface__(self):
        return self._arr.__sycl_usm_array_interface__


@pytest.mark.parametrize("usm_type", ["shared", "device", "host"])
def test_asarray_support_for_usm_ndarray_protocol(usm_type):
    get_queue_or_skip()

    x = dpt.arange(256, dtype="i4", usm_type=usm_type)

    o1 = MockArrayWithBothProtocols(x)
    o2 = MockArrayWithSUAIOnly(x)

    y1 = dpt.asarray(o1)
    assert x.sycl_queue == y1.sycl_queue
    assert x.usm_type == y1.usm_type
    assert x.dtype == y1.dtype
    assert y1.usm_data.reference_obj is None
    assert dpt.all(x == y1)

    y2 = dpt.asarray(o2)
    assert x.sycl_queue == y2.sycl_queue
    assert x.usm_type == y2.usm_type
    assert x.dtype == y2.dtype
    assert not (y2.usm_data.reference_obj is None)
    assert dpt.all(x == y2)

    y3 = dpt.asarray([o1, o2])
    assert x.sycl_queue == y3.sycl_queue
    assert x.usm_type == y3.usm_type
    assert x.dtype == y3.dtype
    assert y3.usm_data.reference_obj is None
    assert dpt.all(x[dpt.newaxis, :] == y3)


@pytest.mark.parametrize("dt", [dpt.float16, dpt.float64, dpt.complex128])
def test_asarray_to_device_with_unsupported_dtype(dt):
    aspect = "fp16" if dt == dpt.float16 else "fp64"
    try:
        d0 = dpctl.select_device_with_aspects(aspect)
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No device with aspect for test")
    d1 = None
    for d in dpctl.get_devices():
        if d.default_selector_score < 0:
            pass
        try:
            d1 = dpctl.select_device_with_aspects(
                d.device_type.name, excluded_aspects=[aspect]
            )
        except dpctl.SyclDeviceCreationError:
            pass
    if d1 is None:
        pytest.skip("No device with missing aspect for test")
    x = dpt.ones(10, dtype=dt, device=d0)
    y = dpt.asarray(x, device=d1)
    assert y.sycl_device == d1
