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

import ctypes
import numbers
from math import prod

import dpctl
import dpctl.memory as dpm
import numpy as np
import pytest
from numpy.testing import assert_raises_regex

import dpnp.tensor as dpt
from dpnp.tensor import Device

from .helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)

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


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (4,),
        (0,),
        (0, 1),
        (0, 0),
        (4, 5),
        (2, 5, 2),
        (2, 2, 2, 2, 2, 2, 2, 2),
        5,
        np.int32(7),
    ],
)
@pytest.mark.parametrize("usm_type", ["shared", "host", "device"])
def test_allocate_usm_ndarray(shape, usm_type):
    q = get_queue_or_skip()
    X = dpt.usm_ndarray(
        shape, dtype="i8", buffer=usm_type, buffer_ctor_kwargs={"queue": q}
    )
    Xnp = np.ndarray(shape, dtype="i8")
    assert X.usm_type == usm_type
    assert X.sycl_context == q.sycl_context
    assert X.sycl_device == q.sycl_device
    assert X.size == Xnp.size
    assert X.shape == Xnp.shape
    assert X.shape == X.__sycl_usm_array_interface__["shape"]


def test_usm_ndarray_flags():
    get_queue_or_skip()
    f = dpt.usm_ndarray((5,), dtype="i4").flags
    assert f.fc
    assert f.forc

    f = dpt.usm_ndarray((5, 2), dtype="i4").flags
    assert f.c_contiguous
    assert f.forc

    f = dpt.usm_ndarray((5, 2), dtype="i4", order="F").flags
    assert f.f_contiguous
    assert f.forc
    assert f.fnc

    f = dpt.usm_ndarray((5,), dtype="i4", strides=(1,)).flags
    assert f.fc
    assert f.forc

    f = dpt.usm_ndarray((5, 1, 2), dtype="i4", strides=(2, 0, 1)).flags
    assert f.c_contiguous
    assert f.forc

    f = dpt.usm_ndarray((5, 1, 2), dtype="i4", strides=(1, 0, 5)).flags
    assert f.f_contiguous
    assert f.forc
    assert f.fnc

    f = dpt.usm_ndarray((5, 0, 1), dtype="i4", strides=(1, 0, 1)).flags
    assert f.fc
    assert f.forc
    assert not dpt.usm_ndarray(
        (5, 1, 1), dtype="i4", strides=(2, 0, 1)
    ).flags.forc

    x = dpt.empty(5, dtype="u2")
    assert x.flags.writable is True
    x.flags.writable = False
    assert x.flags.writable is False
    with pytest.raises(ValueError):
        x[:] = 0
    x.flags["W"] = True
    assert x.flags.writable is True
    x.flags["WRITABLE"] = True
    assert x.flags.writable is True
    x[:] = 0

    with pytest.raises(TypeError):
        x.flags.writable = {}
    with pytest.raises(ValueError):
        x.flags["C"] = False


def test_usm_ndarray_flags_bug_gh_1334():
    get_queue_or_skip()
    a = dpt.ones((2, 3), dtype="u4")
    r = dpt.reshape(a, (1, 6, 1))
    assert r.flags["C"] and r.flags["F"]

    a = dpt.ones((2, 3), dtype="u4", order="F")
    r = dpt.reshape(a, (1, 6, 1), order="F")
    assert r.flags["C"] and r.flags["F"]

    a = dpt.ones((2, 3, 4), dtype="i8")
    r = dpt.sum(a, axis=(1, 2), keepdims=True)
    assert r.flags["C"] and r.flags["F"]

    a = dpt.ones((2, 1), dtype="?")
    r = a[:, 1::-1]
    assert r.flags["F"] and r.flags["C"]


def test_usm_ndarray_writable_flag_views():
    get_queue_or_skip()
    a = dpt.arange(10, dtype="f4")
    a.flags["W"] = False

    a.shape = (5, 2)
    assert not a.flags.writable
    assert not a.T.flags.writable
    assert not a.mT.flags.writable
    assert not a.real.flags.writable
    assert not a[0:3].flags.writable

    a = dpt.arange(10, dtype="c8")
    a.flags["W"] = False

    assert not a.real.flags.writable
    assert not a.imag.flags.writable


@pytest.mark.parametrize("dt1", _all_dtypes)
@pytest.mark.parametrize("dt2", _all_dtypes)
def test_usm_ndarray_from_zero_sized_usm_ndarray(dt1, dt2):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt1, q)
    skip_if_dtype_not_supported(dt2, q)

    x1 = dpt.ones((0,), dtype=dt1, sycl_queue=q)
    x2 = dpt.usm_ndarray(x1.shape, dtype=dt2, buffer=x1)
    assert x2.dtype == dt2
    assert x2.sycl_queue == q
    assert x2._pointer == x1._pointer
    assert x2.shape == x1.shape


def test_usm_ndarray_from_usm_ndarray_readonly():
    get_queue_or_skip()

    x1 = dpt.arange(10, dtype="f4")
    x1.flags["W"] = False
    x2 = dpt.usm_ndarray(x1.shape, dtype="f4", buffer=x1)
    assert not x2.flags.writable


@pytest.mark.parametrize(
    "dtype",
    _all_dtypes
    + [
        b"float32",
        dpt.dtype("d"),
        np.half,
    ],
)
def test_dtypes(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)
    Xusm = dpt.usm_ndarray((1,), dtype=dtype)
    assert Xusm.itemsize == dpt.dtype(dtype).itemsize
    expected_fmt = (dpt.dtype(dtype).str)[1:]
    actual_fmt = Xusm.__sycl_usm_array_interface__["typestr"][1:]
    assert expected_fmt == actual_fmt


@pytest.mark.parametrize("usm_type", ["device", "shared", "host"])
@pytest.mark.parametrize("buffer_ctor_kwargs", [dict(), {"queue": None}])
def test_default_dtype(usm_type, buffer_ctor_kwargs):
    q = get_queue_or_skip()
    dev = q.get_sycl_device()
    if buffer_ctor_kwargs:
        buffer_ctor_kwargs["queue"] = q
    Xusm = dpt.usm_ndarray(
        (1,), buffer=usm_type, buffer_ctor_kwargs=buffer_ctor_kwargs
    )
    if dev.has_aspect_fp64:
        expected_dtype = "f8"
    else:
        expected_dtype = "f4"
    assert Xusm.itemsize == dpt.dtype(expected_dtype).itemsize
    expected_fmt = (dpt.dtype(expected_dtype).str)[1:]
    actual_fmt = Xusm.__sycl_usm_array_interface__["typestr"][1:]
    assert expected_fmt == actual_fmt


@pytest.mark.parametrize(
    "dtype",
    [
        "",
        ">f4",
        "invalid",
        123,
        np.dtype(">f4"),
        np.dtype([("a", ">f4"), ("b", "i4")]),
    ],
)
def test_dtypes_invalid(dtype):
    with pytest.raises((TypeError, ValueError)):
        dpt.usm_ndarray((1,), dtype=dtype)


@pytest.mark.parametrize("dt", ["f", "c8"])
def test_properties(dt):
    """
    Test that properties execute
    """
    try:
        X = dpt.usm_ndarray((3, 4, 5), dtype=dt)
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    assert isinstance(X.sycl_queue, dpctl.SyclQueue)
    assert isinstance(X.sycl_device, dpctl.SyclDevice)
    assert isinstance(X.sycl_context, dpctl.SyclContext)
    assert isinstance(X.dtype, dpt.dtype)
    assert isinstance(X.__sycl_usm_array_interface__, dict)
    assert isinstance(X.mT, dpt.usm_ndarray)
    assert isinstance(X.imag, dpt.usm_ndarray)
    assert isinstance(X.real, dpt.usm_ndarray)
    assert isinstance(X.shape, tuple)
    assert isinstance(X.strides, tuple)
    assert X.usm_type in ("shared", "device", "host")
    assert isinstance(X.size, numbers.Integral)
    assert isinstance(X.nbytes, numbers.Integral)
    assert isinstance(X.ndim, numbers.Integral)
    assert isinstance(X._pointer, numbers.Integral)
    assert isinstance(X.device, Device)
    with pytest.raises(ValueError):
        # array-API mandates exception for .ndim != 2
        X.T
    Y = dpt.usm_ndarray((2, 3), dtype=dt)
    assert isinstance(Y.mT, dpt.usm_ndarray)
    V = dpt.usm_ndarray((3,), dtype=dt)
    with pytest.raises(ValueError):
        # array-API mandates exception for .ndim != 2
        V.mT


@pytest.mark.parametrize("shape", [tuple(), (1,), (1, 1), (1, 1, 1)])
@pytest.mark.parametrize("dtype", ["|b1", "|u2", "|f4", "|i8"])
class TestCopyScalar:
    @pytest.mark.parametrize("func", [bool, float, int, complex])
    def test_copy_scalar_with_func(self, func, shape, dtype):
        try:
            X = dpt.usm_ndarray(shape, dtype=dtype)
        except dpctl.SyclDeviceCreationError:
            pytest.skip("No SYCL devices available")
        Y = np.arange(1, X.size + 1, dtype=dtype)
        X.usm_data.copy_from_host(Y.view("|u1"))
        Y = Y.reshape(())
        # Non-0D numeric arrays must not be convertible to Python scalars
        if len(shape) != 0:
            assert_raises_regex(TypeError, "only 0-dimensional arrays", func, X)
        else:
            # 0D arrays are allowed to convert
            assert func(X) == func(Y)

    @pytest.mark.parametrize(
        "method", ["__bool__", "__float__", "__int__", "__complex__"]
    )
    def test_copy_scalar_with_method(self, method, shape, dtype):
        try:
            X = dpt.usm_ndarray(shape, dtype=dtype)
        except dpctl.SyclDeviceCreationError:
            pytest.skip("No SYCL devices available")
        Y = np.arange(1, X.size + 1, dtype=dtype)
        X.usm_data.copy_from_host(Y.view("|u1"))
        Y = Y.reshape(())
        if len(shape) != 0:
            assert_raises_regex(
                TypeError, "only 0-dimensional arrays", getattr(X, method)
            )
        else:
            assert getattr(X, method)() == getattr(Y, method)()


@pytest.mark.parametrize("func", [bool, float, int, complex])
@pytest.mark.parametrize("shape", [(2,), (1, 2), (3, 4, 5), (0,)])
def test_copy_scalar_invalid_shape(func, shape):
    try:
        X = dpt.usm_ndarray(shape, dtype="i8")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    with pytest.raises(ValueError):
        func(X)


def test_index_noninteger():
    import operator

    try:
        X = dpt.usm_ndarray(1, "f4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    with pytest.raises(IndexError):
        operator.index(X)


@pytest.mark.parametrize(
    "ind",
    [
        tuple(),
        (None,),
        (
            None,
            Ellipsis,
            None,
        ),
        (2, 2, None, 3, 4),
        (Ellipsis,),
        (None, slice(0, None, 2), Ellipsis, slice(0, None, 3)),
        (None, slice(1, None, 2), Ellipsis, slice(1, None, 3)),
        (None, slice(None, -1, -2), Ellipsis, slice(2, None, 3)),
        (
            slice(None, None, -1),
            slice(None, None, -1),
            slice(0, None, 3),
            slice(1, None, 2),
        ),
    ],
)
def test_basic_slice(ind):
    try:
        X = dpt.usm_ndarray((2 * 3, 2 * 4, 3 * 5, 2 * 7), dtype="u1")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    Xnp = np.empty(X.shape, dtype=X.dtype)
    S = X[ind]
    Snp = Xnp[ind]
    assert S.shape == Snp.shape
    assert S.strides == Snp.strides
    assert S.dtype == X.dtype


def test_empty_slice():
    # see gh801
    try:
        X = dpt.empty((1, 0, 1), dtype="u1")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    Y = X[:, ::-1, :]
    assert Y.shape == X.shape
    Z = X[:, ::2, :]
    assert Z.shape == X.shape
    X = dpt.empty(0)
    Y = X[::-1]
    assert Y.shape == X.shape
    Z = X[::2]
    assert Z.shape == X.shape
    X = dpt.empty((0, 4), dtype="u1")
    assert X[:, 1].shape == (0,)
    assert X[:, 1:3].shape == (0, 2)


def test_slice_constructor_1d():
    Xh = np.arange(37, dtype="i4")
    try:
        Xusm = dpt.arange(Xh.size, dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    for ind in [
        slice(1, None, 2),
        slice(0, None, 3),
        slice(1, None, 3),
        slice(2, None, 3),
        slice(None, None, -1),
        slice(-2, 2, -2),
        slice(-1, 1, -2),
        slice(None, None, -13),
    ]:
        assert np.array_equal(
            dpt.asnumpy(Xusm[ind]), Xh[ind]
        ), "Failed for {}".format(ind)


def test_slice_constructor_3d():
    Xh = np.ones((37, 24, 35), dtype="i4")
    try:
        Xusm = dpt.ones(Xh.shape, dtype=Xh.dtype)
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    for ind in [
        slice(1, None, 2),
        slice(0, None, 3),
        slice(1, None, 3),
        slice(2, None, 3),
        slice(None, None, -1),
        slice(-2, 2, -2),
        slice(-1, 1, -2),
        slice(None, None, -13),
        (slice(None, None, -2), Ellipsis, None, 15),
    ]:
        assert np.array_equal(
            dpt.to_numpy(Xusm[ind]), Xh[ind]
        ), "Failed for {}".format(ind)


@pytest.mark.parametrize("usm_type", ["device", "shared", "host"])
def test_slice_suai(usm_type):
    Xh = np.arange(0, 10, dtype="u1")
    try:
        Xusm = dpt.arange(0, 10, dtype="u1", usm_type=usm_type)
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    for ind in [slice(2, 3, None), slice(5, 7, None), slice(3, 9, None)]:
        assert np.array_equal(
            dpm.as_usm_memory(Xusm[ind]).copy_to_host(), Xh[ind]
        ), "Failed for {}".format(ind)


def test_slicing_basic():
    try:
        Xusm = dpt.usm_ndarray((10, 5), dtype="c8")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    Xusm[None]
    Xusm[...]
    Xusm[8]
    Xusm[-3]
    with pytest.raises(IndexError):
        Xusm[..., ...]
    with pytest.raises(IndexError):
        Xusm[1, 1, :, 1]
    Xusm[:, -4]
    with pytest.raises(IndexError):
        Xusm[:, -128]
    with pytest.raises(IndexError):
        Xusm[{1, 2, 3, 4, 5, 6, 7}]
    X = dpt.usm_ndarray(10, "u1")
    X.usm_data.copy_from_host(b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09")
    int(
        X[X[2]]
    )  # check that objects with __index__ method can be used as indices
    Xh = dpm.as_usm_memory(X[X[2] : X[5]]).copy_to_host()
    Xnp = np.arange(0, 10, dtype="u1")
    assert np.array_equal(Xh, Xnp[Xnp[2] : Xnp[5]])


def test_slicing_empty():
    try:
        X = dpt.usm_ndarray((0, 10), dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    x = dpt.moveaxis(X, 1, 0)
    # this used to raise ValueError
    y = x[1]
    assert y.ndim == 1
    assert y.shape == (0,)
    assert y.dtype == X.dtype
    assert y.usm_type == X.usm_type
    assert y.sycl_queue == X.sycl_queue
    w = x[1:3]
    assert w.ndim == 2
    assert w.shape == (
        2,
        0,
    )
    assert w.dtype == X.dtype
    assert w.usm_type == X.usm_type
    assert w.sycl_queue == X.sycl_queue


def test_ctor_invalid_shape():
    with pytest.raises(TypeError):
        dpt.usm_ndarray(dict())


def test_ctor_invalid_order():
    get_queue_or_skip()
    with pytest.raises(ValueError):
        dpt.usm_ndarray((5, 5, 3), order="Z")
    with pytest.raises(ValueError):
        dpt.usm_ndarray((10), strides=(1,), order="Z")
    with pytest.raises(ValueError):
        dpt.usm_ndarray((), order="Z")


def test_ctor_buffer_kwarg():
    try:
        dpt.usm_ndarray(10, dtype="i8", buffer=b"device")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    with pytest.raises(ValueError):
        dpt.usm_ndarray(10, buffer="invalid_param")
    Xusm = dpt.usm_ndarray((10, 5), dtype="c8")
    Xusm[...] = 1
    X2 = dpt.usm_ndarray(Xusm.shape, buffer=Xusm, dtype=Xusm.dtype)
    Horig_copy = Xusm.usm_data.copy_to_host()
    H2_copy = X2.usm_data.copy_to_host()
    assert np.array_equal(Horig_copy, H2_copy)
    with pytest.raises(ValueError):
        dpt.usm_ndarray(10, dtype="i4", buffer=dict())
    # use device-specific default fp data type
    X3 = dpt.usm_ndarray(Xusm.shape, buffer=Xusm)
    assert np.array_equal(Horig_copy, X3.usm_data.copy_to_host())


def test_usm_ndarray_props():
    try:
        Xusm = dpt.usm_ndarray((10, 5), dtype="c8", order="F")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    Xusm.ndim
    repr(Xusm)
    Xusm.flags
    Xusm.__sycl_usm_array_interface__
    Xusm.device
    Xusm.strides
    Xusm.real
    Xusm.imag
    try:
        dpctl.SyclQueue("cpu")
    except dpctl.SyclQueueCreationError:
        pytest.skip("Sycl device CPU was not detected")
    Xusm.to_device("cpu")


def test_datapi_device():
    try:
        X = dpt.usm_ndarray(1, dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    dev_t = type(X.device)
    with pytest.raises(TypeError):
        dev_t()
    dev_t.create_device(X.device)
    dev_t.create_device(X.sycl_queue)
    d1 = dev_t.create_device(X.sycl_device)
    d2 = dev_t.create_device(X.sycl_device.filter_string)
    d3 = dev_t.create_device(None)
    assert d1.sycl_queue == d2.sycl_queue
    assert d1.sycl_queue == d3.sycl_queue
    X.device.sycl_context
    X.device.sycl_queue
    X.device.sycl_device
    repr(X.device)
    X.device.print_device_info()


def _pyx_capi_fnptr_to_callable(
    X,
    pyx_capi_name,
    caps_name,
    fn_restype=ctypes.c_void_p,
    fn_argtypes=(ctypes.py_object,),
):
    import sys

    mod = sys.modules[X.__class__.__module__]
    cap = mod.__pyx_capi__.get(pyx_capi_name, None)
    if cap is None:
        raise ValueError(
            "__pyx_capi__ does not export {} capsule".format(pyx_capi_name)
        )
    # construct Python callable to invoke these functions
    cap_ptr_fn = ctypes.pythonapi.PyCapsule_GetPointer
    cap_ptr_fn.restype = ctypes.c_void_p
    cap_ptr_fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
    fn_ptr = cap_ptr_fn(cap, caps_name)
    callable_maker_ptr = ctypes.PYFUNCTYPE(fn_restype, *fn_argtypes)
    return callable_maker_ptr(fn_ptr)


def test_pyx_capi_get_data():
    try:
        X = dpt.usm_ndarray(17, dtype="i8")[1::2]
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    get_data_fn = _pyx_capi_fnptr_to_callable(
        X,
        "UsmNDArray_GetData",
        b"char *(struct PyUSMArrayObject *)",
        fn_restype=ctypes.c_void_p,
        fn_argtypes=(ctypes.py_object,),
    )
    r1 = get_data_fn(X)
    sua_iface = X.__sycl_usm_array_interface__
    assert r1 == sua_iface["data"][0] + sua_iface.get("offset") * X.itemsize


def test_pyx_capi_get_shape():
    try:
        X = dpt.usm_ndarray(17, dtype="u4")[1::2]
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    get_shape_fn = _pyx_capi_fnptr_to_callable(
        X,
        "UsmNDArray_GetShape",
        b"Py_ssize_t *(struct PyUSMArrayObject *)",
        fn_restype=ctypes.c_void_p,
        fn_argtypes=(ctypes.py_object,),
    )
    c_longlong_p = ctypes.POINTER(ctypes.c_longlong)
    shape0 = ctypes.cast(get_shape_fn(X), c_longlong_p).contents.value
    assert shape0 == X.shape[0]


def test_pyx_capi_get_strides():
    try:
        X = dpt.usm_ndarray(17, dtype="f4")[1::2]
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    get_strides_fn = _pyx_capi_fnptr_to_callable(
        X,
        "UsmNDArray_GetStrides",
        b"Py_ssize_t *(struct PyUSMArrayObject *)",
        fn_restype=ctypes.c_void_p,
        fn_argtypes=(ctypes.py_object,),
    )
    c_longlong_p = ctypes.POINTER(ctypes.c_longlong)
    strides0_p = get_strides_fn(X)
    if strides0_p:
        strides0_p = ctypes.cast(strides0_p, c_longlong_p).contents
        strides0_p = strides0_p.value
    assert strides0_p == 0 or strides0_p == X.strides[0]


def test_pyx_capi_get_ndim():
    try:
        X = dpt.usm_ndarray(17, dtype="?")[1::2]
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    get_ndim_fn = _pyx_capi_fnptr_to_callable(
        X,
        "UsmNDArray_GetNDim",
        b"int (struct PyUSMArrayObject *)",
        fn_restype=ctypes.c_int,
        fn_argtypes=(ctypes.py_object,),
    )
    assert get_ndim_fn(X) == X.ndim


def test_pyx_capi_get_typenum():
    try:
        X = dpt.usm_ndarray(17, dtype="c8")[1::2]
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    get_typenum_fn = _pyx_capi_fnptr_to_callable(
        X,
        "UsmNDArray_GetTypenum",
        b"int (struct PyUSMArrayObject *)",
        fn_restype=ctypes.c_int,
        fn_argtypes=(ctypes.py_object,),
    )
    typenum = get_typenum_fn(X)
    assert type(typenum) is int
    assert typenum == X.dtype.num


def test_pyx_capi_get_elemsize():
    try:
        X = dpt.usm_ndarray(17, dtype="u8")[1::2]
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    get_elemsize_fn = _pyx_capi_fnptr_to_callable(
        X,
        "UsmNDArray_GetElementSize",
        b"int (struct PyUSMArrayObject *)",
        fn_restype=ctypes.c_int,
        fn_argtypes=(ctypes.py_object,),
    )
    itemsize = get_elemsize_fn(X)
    assert type(itemsize) is int
    assert itemsize == X.itemsize


def test_pyx_capi_get_flags():
    try:
        X = dpt.usm_ndarray(17, dtype="i8")[1::2]
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    get_flags_fn = _pyx_capi_fnptr_to_callable(
        X,
        "UsmNDArray_GetFlags",
        b"int (struct PyUSMArrayObject *)",
        fn_restype=ctypes.c_int,
        fn_argtypes=(ctypes.py_object,),
    )
    flags = get_flags_fn(X)
    assert type(flags) is int and X.flags == flags


def test_pyx_capi_get_offset():
    try:
        X = dpt.usm_ndarray(17, dtype="u2")[1::2]
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    get_offset_fn = _pyx_capi_fnptr_to_callable(
        X,
        "UsmNDArray_GetOffset",
        b"Py_ssize_t (struct PyUSMArrayObject *)",
        fn_restype=ctypes.c_longlong,
        fn_argtypes=(ctypes.py_object,),
    )
    offset = get_offset_fn(X)
    assert type(offset) is int
    assert offset == X.__sycl_usm_array_interface__["offset"]


def test_pyx_capi_get_usmdata():
    try:
        X = dpt.usm_ndarray(17, dtype="u2")[1::2]
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    get_usmdata_fn = _pyx_capi_fnptr_to_callable(
        X,
        "UsmNDArray_GetUSMData",
        b"PyObject *(struct PyUSMArrayObject *)",
        fn_restype=ctypes.py_object,
        fn_argtypes=(ctypes.py_object,),
    )
    capi_usm_data = get_usmdata_fn(X)
    assert isinstance(capi_usm_data, dpm._memory._Memory)
    assert capi_usm_data.nbytes == X.usm_data.nbytes
    assert capi_usm_data._pointer == X.usm_data._pointer
    assert capi_usm_data.sycl_queue == X.usm_data.sycl_queue


def test_pyx_capi_get_queue_ref():
    try:
        X = dpt.usm_ndarray(17, dtype="i2")[1::2]
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    get_queue_ref_fn = _pyx_capi_fnptr_to_callable(
        X,
        "UsmNDArray_GetQueueRef",
        b"DPCTLSyclQueueRef (struct PyUSMArrayObject *)",
        fn_restype=ctypes.c_void_p,
        fn_argtypes=(ctypes.py_object,),
    )
    queue_ref = get_queue_ref_fn(X)  # address of a copy, should be unequal
    assert queue_ref != X.sycl_queue.addressof_ref()


def test_pyx_capi_make_from_memory():
    q = get_queue_or_skip()
    n0, n1 = 4, 6
    c_tuple = (ctypes.c_ssize_t * 2)(n0, n1)
    mem = dpm.MemoryUSMShared(n0 * n1 * 4, queue=q)
    typenum = dpt.dtype("single").num
    any_usm_ndarray = dpt.empty(tuple(), dtype="i4", sycl_queue=q)
    make_from_memory_fn = _pyx_capi_fnptr_to_callable(
        any_usm_ndarray,
        "UsmNDArray_MakeSimpleFromMemory",
        b"PyObject *(int, Py_ssize_t const *, int, "
        b"struct Py_MemoryObject *, Py_ssize_t, char)",
        fn_restype=ctypes.py_object,
        fn_argtypes=(
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ssize_t),
            ctypes.c_int,
            ctypes.py_object,
            ctypes.c_ssize_t,
            ctypes.c_char,
        ),
    )
    r = make_from_memory_fn(
        ctypes.c_int(2),
        c_tuple,
        ctypes.c_int(typenum),
        mem,
        ctypes.c_ssize_t(0),
        ctypes.c_char(b"C"),
    )
    assert isinstance(r, dpt.usm_ndarray)
    assert r.ndim == 2
    assert r.shape == (n0, n1)
    assert r._pointer == mem._pointer
    assert r.usm_type == "shared"
    assert r.sycl_queue == q
    assert r.flags["C"]
    r2 = make_from_memory_fn(
        ctypes.c_int(2),
        c_tuple,
        ctypes.c_int(typenum),
        mem,
        ctypes.c_ssize_t(0),
        ctypes.c_char(b"F"),
    )
    ptr = mem._pointer
    del mem
    del r
    assert isinstance(r2, dpt.usm_ndarray)
    assert r2._pointer == ptr
    assert r2.usm_type == "shared"
    assert r2.sycl_queue == q
    assert r2.flags["F"]


def test_pyx_capi_set_writable_flag():
    q = get_queue_or_skip()
    usm_ndarray = dpt.empty((4, 5), dtype="i4", sycl_queue=q)
    assert isinstance(usm_ndarray, dpt.usm_ndarray)
    assert usm_ndarray.flags["WRITABLE"] is True
    set_writable = _pyx_capi_fnptr_to_callable(
        usm_ndarray,
        "UsmNDArray_SetWritableFlag",
        b"void (struct PyUSMArrayObject *, int)",
        fn_restype=None,
        fn_argtypes=(ctypes.py_object, ctypes.c_int),
    )
    set_writable(usm_ndarray, ctypes.c_int(0))
    assert isinstance(usm_ndarray, dpt.usm_ndarray)
    assert usm_ndarray.flags["WRITABLE"] is False
    set_writable(usm_ndarray, ctypes.c_int(1))
    assert isinstance(usm_ndarray, dpt.usm_ndarray)
    assert usm_ndarray.flags["WRITABLE"] is True


def test_pyx_capi_make_from_ptr():
    q = get_queue_or_skip()
    usm_ndarray = dpt.empty(tuple(), dtype="i4", sycl_queue=q)
    make_from_ptr = _pyx_capi_fnptr_to_callable(
        usm_ndarray,
        "UsmNDArray_MakeSimpleFromPtr",
        b"PyObject *(size_t, int, DPCTLSyclUSMRef, "
        b"DPCTLSyclQueueRef, PyObject *)",
        fn_restype=ctypes.py_object,
        fn_argtypes=(
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.py_object,
        ),
    )
    nelems = 10
    dt = dpt.int64
    mem = dpm.MemoryUSMDevice(nelems * dt.itemsize, queue=q)
    arr = make_from_ptr(
        ctypes.c_size_t(nelems),
        dt.num,
        mem._pointer,
        mem.sycl_queue.addressof_ref(),
        mem,
    )
    assert isinstance(arr, dpt.usm_ndarray)
    assert arr.shape == (nelems,)
    assert arr.dtype == dt
    assert arr.sycl_queue == q
    assert arr._pointer == mem._pointer
    del mem
    assert isinstance(arr.__repr__(), str)


def test_pyx_capi_make_general():
    q = get_queue_or_skip()
    usm_ndarray = dpt.empty(tuple(), dtype="i4", sycl_queue=q)
    make_from_ptr = _pyx_capi_fnptr_to_callable(
        usm_ndarray,
        "UsmNDArray_MakeFromPtr",
        b"PyObject *(int, Py_ssize_t const *, int, Py_ssize_t const *, "
        b"DPCTLSyclUSMRef, DPCTLSyclQueueRef, Py_ssize_t, PyObject *)",
        fn_restype=ctypes.py_object,
        fn_argtypes=(
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ssize_t),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ssize_t),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_ssize_t,
            ctypes.py_object,
        ),
    )
    # Create array to view into diagonal of a matrix
    n = 5
    mat = dpt.reshape(
        dpt.arange(n * n, dtype="i4", sycl_queue=q),
        (
            n,
            n,
        ),
    )
    c_shape = (ctypes.c_ssize_t * 1)(
        n,
    )
    c_strides = (ctypes.c_ssize_t * 1)(
        n + 1,
    )
    diag = make_from_ptr(
        ctypes.c_int(1),
        c_shape,
        ctypes.c_int(mat.dtype.num),
        c_strides,
        mat._pointer,
        mat.sycl_queue.addressof_ref(),
        ctypes.c_ssize_t(0),
        mat,
    )
    assert isinstance(diag, dpt.usm_ndarray)
    assert diag.shape == (n,)
    assert diag.strides == (n + 1,)
    assert diag.dtype == mat.dtype
    assert diag.sycl_queue == q
    assert diag._pointer == mat._pointer
    del mat
    assert isinstance(diag.__repr__(), str)
    # create 0d scalar
    mat = dpt.reshape(
        dpt.arange(n * n, dtype="i4", sycl_queue=q),
        (
            n,
            n,
        ),
    )
    sc = make_from_ptr(
        ctypes.c_int(0),
        None,  # NULL pointer
        ctypes.c_int(mat.dtype.num),
        None,  # NULL pointer
        mat._pointer,
        mat.sycl_queue.addressof_ref(),
        ctypes.c_ssize_t(0),
        mat,
    )
    assert isinstance(sc, dpt.usm_ndarray)
    assert sc.shape == ()
    assert sc.dtype == mat.dtype
    assert sc.sycl_queue == q
    assert sc._pointer == mat._pointer
    c_shape = (ctypes.c_ssize_t * 2)(0, n)
    c_strides = (ctypes.c_ssize_t * 2)(0, 1)
    zd_arr = make_from_ptr(
        ctypes.c_int(2),
        c_shape,
        ctypes.c_int(mat.dtype.num),
        c_strides,
        mat._pointer,
        mat.sycl_queue.addressof_ref(),
        ctypes.c_ssize_t(0),
        mat,
    )
    assert isinstance(zd_arr, dpt.usm_ndarray)
    assert zd_arr.shape == (
        0,
        n,
    )
    assert zd_arr.strides == (
        0,
        1,
    )
    assert zd_arr.dtype == mat.dtype
    assert zd_arr.sycl_queue == q
    assert zd_arr._pointer == mat._pointer


def test_pyx_capi_make_fns_invalid_typenum():
    q = get_queue_or_skip()
    usm_ndarray = dpt.empty(tuple(), dtype="i4", sycl_queue=q)

    make_simple_from_ptr = _pyx_capi_fnptr_to_callable(
        usm_ndarray,
        "UsmNDArray_MakeSimpleFromPtr",
        b"PyObject *(size_t, int, DPCTLSyclUSMRef, "
        b"DPCTLSyclQueueRef, PyObject *)",
        fn_restype=ctypes.py_object,
        fn_argtypes=(
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.py_object,
        ),
    )

    nelems = 10
    dtype = dpt.int64
    arr = dpt.arange(nelems, dtype=dtype, sycl_queue=q)

    with pytest.raises(ValueError):
        make_simple_from_ptr(
            ctypes.c_size_t(nelems),
            -1,
            arr._pointer,
            arr.sycl_queue.addressof_ref(),
            arr,
        )

    make_from_ptr = _pyx_capi_fnptr_to_callable(
        usm_ndarray,
        "UsmNDArray_MakeFromPtr",
        b"PyObject *(int, Py_ssize_t const *, int, Py_ssize_t const *, "
        b"DPCTLSyclUSMRef, DPCTLSyclQueueRef, Py_ssize_t, PyObject *)",
        fn_restype=ctypes.py_object,
        fn_argtypes=(
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ssize_t),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ssize_t),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_ssize_t,
            ctypes.py_object,
        ),
    )
    c_shape = (ctypes.c_ssize_t * 1)(
        nelems,
    )
    c_strides = (ctypes.c_ssize_t * 1)(
        1,
    )
    with pytest.raises(ValueError):
        make_from_ptr(
            ctypes.c_int(1),
            c_shape,
            -1,
            c_strides,
            arr._pointer,
            arr.sycl_queue.addressof_ref(),
            ctypes.c_ssize_t(0),
            arr,
        )
    del arr


def _pyx_capi_int(X, pyx_capi_name, caps_name=b"int", val_restype=ctypes.c_int):
    import sys

    mod = sys.modules[X.__class__.__module__]
    cap = mod.__pyx_capi__.get(pyx_capi_name, None)
    if cap is None:
        raise ValueError(
            "__pyx_capi__ does not export {} capsule".format(pyx_capi_name)
        )
    # construct Python callable to invoke these functions
    cap_ptr_fn = ctypes.pythonapi.PyCapsule_GetPointer
    cap_ptr_fn.restype = ctypes.c_void_p
    cap_ptr_fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
    cap_ptr = cap_ptr_fn(cap, caps_name)
    val_ptr = ctypes.cast(cap_ptr, ctypes.POINTER(val_restype))
    return val_ptr.contents.value


def test_pyx_capi_check_constants():
    try:
        X = dpt.usm_ndarray(17, dtype="i1")[1::2]
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    cc_flag = _pyx_capi_int(X, "USM_ARRAY_C_CONTIGUOUS")
    assert cc_flag > 0 and 0 == (cc_flag & (cc_flag - 1))
    fc_flag = _pyx_capi_int(X, "USM_ARRAY_F_CONTIGUOUS")
    assert fc_flag > 0 and 0 == (fc_flag & (fc_flag - 1))
    w_flag = _pyx_capi_int(X, "USM_ARRAY_WRITABLE")
    assert w_flag > 0 and 0 == (w_flag & (w_flag - 1))

    bool_typenum = _pyx_capi_int(X, "UAR_BOOL")
    assert bool_typenum == dpt.dtype("bool_").num

    byte_typenum = _pyx_capi_int(X, "UAR_BYTE")
    assert byte_typenum == dpt.dtype(np.byte).num
    ubyte_typenum = _pyx_capi_int(X, "UAR_UBYTE")
    assert ubyte_typenum == dpt.dtype(np.ubyte).num

    short_typenum = _pyx_capi_int(X, "UAR_SHORT")
    assert short_typenum == dpt.dtype(np.short).num
    ushort_typenum = _pyx_capi_int(X, "UAR_USHORT")
    assert ushort_typenum == dpt.dtype(np.ushort).num

    int_typenum = _pyx_capi_int(X, "UAR_INT")
    assert int_typenum == dpt.dtype(np.intc).num
    uint_typenum = _pyx_capi_int(X, "UAR_UINT")
    assert uint_typenum == dpt.dtype(np.uintc).num

    long_typenum = _pyx_capi_int(X, "UAR_LONG")
    assert long_typenum == dpt.dtype("l").num
    ulong_typenum = _pyx_capi_int(X, "UAR_ULONG")
    assert ulong_typenum == dpt.dtype("L").num

    longlong_typenum = _pyx_capi_int(X, "UAR_LONGLONG")
    assert longlong_typenum == dpt.dtype(np.longlong).num
    ulonglong_typenum = _pyx_capi_int(X, "UAR_ULONGLONG")
    assert ulonglong_typenum == dpt.dtype(np.ulonglong).num

    half_typenum = _pyx_capi_int(X, "UAR_HALF")
    assert half_typenum == dpt.dtype(np.half).num
    float_typenum = _pyx_capi_int(X, "UAR_FLOAT")
    assert float_typenum == dpt.dtype(np.single).num
    double_typenum = _pyx_capi_int(X, "UAR_DOUBLE")
    assert double_typenum == dpt.dtype(np.double).num

    cfloat_typenum = _pyx_capi_int(X, "UAR_CFLOAT")
    assert cfloat_typenum == dpt.dtype(np.csingle).num
    cdouble_typenum = _pyx_capi_int(X, "UAR_CDOUBLE")
    assert cdouble_typenum == dpt.dtype(np.cdouble).num


@pytest.mark.parametrize(
    "shape", [tuple(), (1,), (5,), (2, 3), (2, 3, 4), (2, 2, 2, 2, 2)]
)
@pytest.mark.parametrize(
    "dtype",
    _all_dtypes,
)
@pytest.mark.parametrize("usm_type", ["device", "shared", "host"])
def test_tofrom_numpy(shape, dtype, usm_type):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)
    Xusm = dpt.zeros(shape, dtype=dtype, usm_type=usm_type, sycl_queue=q)
    Ynp = np.ones(shape, dtype=dtype)
    Ynp[(0,) * len(shape)] = 0
    ind = (slice(None, None, None),) * Ynp.ndim
    Xusm[ind] = Ynp
    assert np.array_equal(dpt.to_numpy(Xusm), Ynp)


@pytest.mark.parametrize(
    "dtype",
    _all_dtypes,
)
@pytest.mark.parametrize("usm_type", ["device", "shared", "host"])
def test_tofrom_numpy_permuted(dtype, usm_type):
    shape = (3, 5, 7)
    perm = (1, 2, 0)
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)
    Xusm = dpt.permute_dims(
        dpt.zeros(shape, dtype=dtype, usm_type=usm_type, sycl_queue=q), perm
    )
    Ynp = np.transpose(np.ones(shape, dtype=dtype), perm)
    Ynp[:, ::2, ::2] = 0
    ind = (slice(None, None, None),) * Ynp.ndim
    # even though Xusm and Ynp are strided, simple memcpy could be done.
    # This test validates that it is being done correctly
    Xusm[ind] = Ynp
    assert np.array_equal(dpt.to_numpy(Xusm), Ynp)


@pytest.mark.parametrize(
    "dtype",
    _all_dtypes,
)
@pytest.mark.parametrize("src_usm_type", ["device", "shared", "host"])
@pytest.mark.parametrize("dst_usm_type", ["device", "shared", "host"])
def test_setitem_same_dtype(dtype, src_usm_type, dst_usm_type):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)
    shape = (2, 4, 3)
    Xnp = (
        np.random.randint(-10, 10, size=prod(shape))
        .astype(dtype)
        .reshape(shape)
    )
    X = dpt.from_numpy(Xnp, usm_type=src_usm_type)
    Z = dpt.zeros(shape, dtype=dtype, usm_type=dst_usm_type)
    Zusm_0d = dpt.copy(Z[0, 0, 0])
    ind = (-1, -1, -1)
    Xusm_0d = X[ind]
    Zusm_0d[Ellipsis] = Xusm_0d
    assert np.array_equal(dpt.to_numpy(Zusm_0d), Xnp[ind])
    Zusm_1d = dpt.copy(Z[0, 1:3, 0])
    ind = (-1, slice(0, 2, None), -1)
    Xusm_1d = X[ind]
    Zusm_1d[Ellipsis] = Xusm_1d
    assert np.array_equal(dpt.to_numpy(Zusm_1d), Xnp[ind])
    Zusm_2d = dpt.copy(Z[:, 1:3, 0])[::-1]
    Xusm_2d = X[:, 1:4, -1]
    Zusm_2d[:] = Xusm_2d[:, 0:2]
    assert np.array_equal(dpt.to_numpy(Zusm_2d), Xnp[:, 1:3, -1])
    Zusm_3d = dpt.copy(Z)
    Xusm_3d = X
    Zusm_3d[:] = Xusm_3d
    assert np.array_equal(dpt.to_numpy(Zusm_3d), Xnp)
    Zusm_3d[::-1] = Xusm_3d[::-1]
    assert np.array_equal(dpt.to_numpy(Zusm_3d), Xnp)
    Zusm_3d[:] = Xusm_3d[0]
    R1 = dpt.to_numpy(Zusm_3d)
    R2 = np.broadcast_to(Xnp[0], R1.shape)
    assert R1.shape == R2.shape
    assert np.allclose(R1, R2)
    Zusm_empty = Zusm_1d[0:0]
    Zusm_empty[Ellipsis] = Zusm_3d[0, 0, 0:0]


def test_setitem_broadcasting():
    "See gh-1503"
    get_queue_or_skip()
    dst = dpt.ones((2, 3, 4), dtype="u4")
    src = dpt.zeros((3, 1), dtype=dst.dtype)
    dst[...] = src
    expected = np.zeros(dst.shape, dtype=dst.dtype)
    assert np.array_equal(dpt.asnumpy(dst), expected)


def test_setitem_broadcasting_offset():
    get_queue_or_skip()
    dt = dpt.int32
    x = dpt.asarray([[1, 2, 3], [6, 7, 8]], dtype=dt)
    y = dpt.asarray([4, 5], dtype=dt)
    x[0] = y[1]
    expected = dpt.asarray([[5, 5, 5], [6, 7, 8]], dtype=dt)
    assert dpt.all(x == expected)


def test_setitem_broadcasting_empty_dst_validation():
    "Broadcasting rules apply, except exception"
    get_queue_or_skip()
    dst = dpt.ones((2, 0, 5, 4), dtype="i8")
    src = dpt.ones((2, 0, 3, 4), dtype="i8")
    with pytest.raises(ValueError):
        dst[...] = src


def test_setitem_broadcasting_empty_dst_edge_case():
    """RHS is shunken to empty array by
    broadasting rule, hence no exception"""
    get_queue_or_skip()
    dst = dpt.ones(1, dtype="i8")[0:0]
    src = dpt.ones(tuple(), dtype="i8")
    dst[...] = src


def test_setitem_broadcasting_src_ndim_equal_dst_ndim():
    get_queue_or_skip()
    dst = dpt.ones((2, 3, 4), dtype="i4")
    src = dpt.zeros((2, 1, 4), dtype="i4")
    dst[...] = src

    expected = np.zeros(dst.shape, dtype=dst.dtype)
    assert np.array_equal(dpt.asnumpy(dst), expected)


def test_setitem_broadcasting_src_ndim_greater_than_dst_ndim():
    get_queue_or_skip()
    dst = dpt.ones((2, 3, 4), dtype="i4")
    src = dpt.zeros((1, 2, 1, 4), dtype="i4")
    dst[...] = src

    expected = np.zeros(dst.shape, dtype=dst.dtype)
    assert np.array_equal(dpt.asnumpy(dst), expected)


@pytest.mark.parametrize(
    "dtype",
    _all_dtypes,
)
@pytest.mark.parametrize("usm_type", ["device", "shared", "host"])
def test_setitem_scalar(dtype, usm_type):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)
    X = dpt.usm_ndarray((6, 6), dtype=dtype, buffer=usm_type)
    for i in range(X.size):
        X[np.unravel_index(i, X.shape)] = np.asarray(i, dtype=dtype)
    assert np.array_equal(
        dpt.to_numpy(X), np.arange(X.size).astype(dtype).reshape(X.shape)
    )
    Y = dpt.usm_ndarray((2, 3), dtype=dtype, buffer=usm_type)
    for i in range(Y.size):
        Y[np.unravel_index(i, Y.shape)] = i
    assert np.array_equal(
        dpt.to_numpy(Y), np.arange(Y.size).astype(dtype).reshape(Y.shape)
    )


def test_setitem_errors():
    q = get_queue_or_skip()
    X = dpt.empty((4,), dtype="u1", sycl_queue=q)
    Y = dpt.empty((4, 2), dtype="u1", sycl_queue=q)
    with pytest.raises(ValueError):
        X[:] = Y
    with pytest.raises(ValueError):
        X[:] = Y[:, 0:1]
    X[:] = Y[None, :, 0]


@pytest.mark.parametrize("src_dt,dst_dt", [("i4", "i8"), ("f4", "f8")])
def test_setitem_different_dtypes(src_dt, dst_dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dst_dt, q)
    X = dpt.ones(10, dtype=src_dt, sycl_queue=q)
    Y = dpt.zeros(10, dtype=src_dt, sycl_queue=q)
    Z = dpt.empty((20,), dtype=dst_dt, sycl_queue=q)
    Z[::2] = X
    Z[1::2] = Y
    assert np.allclose(dpt.asnumpy(Z), np.tile(np.array([1, 0], Z.dtype), 10))


def test_setitem_wingaps():
    q = get_queue_or_skip()
    if dpt.dtype("intc").itemsize == dpt.dtype("int32").itemsize:
        dpt_dst = dpt.empty(4, dtype="int32", sycl_queue=q)
        np_src = np.arange(4, dtype="intc")
        dpt_dst[:] = np_src  # should not raise exceptions
        assert np.array_equal(dpt.asnumpy(dpt_dst), np_src)
    if dpt.dtype("long").itemsize == dpt.dtype("longlong").itemsize:
        dpt_dst = dpt.empty(4, dtype="longlong", sycl_queue=q)
        np_src = np.arange(4, dtype="long")
        dpt_dst[:] = np_src  # should not raise exceptions
        assert np.array_equal(dpt.asnumpy(dpt_dst), np_src)


def test_shape_setter():
    def cc_strides(sh):
        return np.empty(sh, dtype="u1").strides

    def relaxed_strides_equal(st1, st2, sh):
        eq_ = True
        for s1, s2, d in zip(st1, st2, sh):
            eq_ = eq_ and ((d == 1) or (s1 == s2))
        return eq_

    sh_s = (2 * 3 * 4 * 5,)
    sh_f = (
        2,
        3,
        4,
        5,
    )
    try:
        X = dpt.usm_ndarray(sh_s, dtype="i8")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    X.shape = sh_f
    assert X.shape == sh_f
    assert relaxed_strides_equal(X.strides, cc_strides(sh_f), sh_f)
    assert X.flags.c_contiguous, "reshaped array expected to be C-contiguous"

    sh_s = (
        2,
        12,
        5,
    )
    sh_f = (
        2,
        3,
        4,
        5,
    )
    X = dpt.usm_ndarray(sh_s, dtype="u4", order="C")
    X.shape = sh_f
    assert X.shape == sh_f
    assert relaxed_strides_equal(X.strides, cc_strides(sh_f), sh_f)

    sh_s = (2, 3, 4, 5)
    sh_f = (4, 3, 2, 5)
    X = dpt.usm_ndarray(sh_s, dtype="f4")
    X.shape = sh_f
    assert relaxed_strides_equal(X.strides, cc_strides(sh_f), sh_f)

    sh_s = (2, 3, 4, 5)
    sh_f = (4, 3, 1, 2, 5)
    X = dpt.usm_ndarray(sh_s, dtype="?")
    X.shape = sh_f
    assert relaxed_strides_equal(X.strides, cc_strides(sh_f), sh_f)
    sz = X.size
    X.shape = sz
    assert X.shape == (sz,)
    assert relaxed_strides_equal(X.strides, (1,), (sz,))

    X = dpt.usm_ndarray(sh_s, dtype="u4")
    with pytest.raises(TypeError):
        X.shape = "abcbe"
    X = dpt.usm_ndarray((4, 4), dtype="u1")[::2, ::2]
    with pytest.raises(AttributeError):
        X.shape = (4,)
    X = dpt.usm_ndarray((0,), dtype="i4")
    X.shape = (0,)
    X.shape = (
        2,
        0,
    )
    X.shape = (
        0,
        2,
    )
    X.shape = (
        1,
        0,
        1,
    )


def test_len():
    try:
        X = dpt.usm_ndarray(1, "i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    assert len(X) == 1
    X = dpt.usm_ndarray((2, 1), "i4")
    assert len(X) == 2
    X = dpt.usm_ndarray(tuple(), "i4")
    with pytest.raises(TypeError):
        len(X)


def test_array_namespace():
    try:
        X = dpt.usm_ndarray(1, "i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    X.__array_namespace__()
    X._set_namespace(dpt)
    assert X.__array_namespace__() is dpt
    X.__array_namespace__(api_version=dpt.__array_api_version__)
    assert X.__array_namespace__() is dpt


def test_dlpack():
    try:
        X = dpt.usm_ndarray(1, "i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    X.__dlpack_device__()
    X.__dlpack__(stream=None)


def test_to_device():
    try:
        X = dpt.usm_ndarray(1, "f4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    for dev in dpctl.get_devices():
        if dev.default_selector_score > 0:
            Y = X.to_device(dev)
            assert Y.sycl_device == dev


def test_to_device_stream_validation():
    try:
        X = dpt.usm_ndarray(1, "f4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    # invalid type of stream keyword
    with pytest.raises(TypeError):
        X.to_device(X.sycl_queue, stream=dict())
    # stream is keyword-only arg
    with pytest.raises(TypeError):
        X.to_device(X.sycl_queue, X.sycl_queue)


def test_to_device_stream_use():
    try:
        X = dpt.usm_ndarray(1, "f4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    q1 = dpctl.SyclQueue(
        X.sycl_context, X.sycl_device, property="enable_profiling"
    )
    X.to_device(q1, stream=q1)


def test_to_device_migration():
    q1 = get_queue_or_skip()  # two distinct copies of default-constructed queue
    q2 = get_queue_or_skip()
    X1 = dpt.empty((5,), dtype="i8", sycl_queue=q1)  # X1 is associated with q1
    X2 = X1.to_device(q2)  # X2 is reassociated with q2
    assert X1.sycl_queue == q1
    assert X2.sycl_queue == q2
    assert X1.usm_data._pointer == X2.usm_data._pointer


def test_astype():
    try:
        X = dpt.empty((5, 5), dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    X[:] = np.full((5, 5), 7, dtype="i4")
    Y = dpt.astype(X, "c8", order="C")
    assert np.allclose(dpt.to_numpy(Y), np.full((5, 5), 7, dtype="c8"))
    if Y.sycl_device.has_aspect_fp16:
        Y = dpt.astype(X[::2, ::-1], "f2", order="K")
        assert np.allclose(dpt.to_numpy(Y), np.full(Y.shape, 7, dtype="f2"))
    Y = dpt.astype(X[::2, ::-1], "f4", order="K")
    assert np.allclose(dpt.to_numpy(Y), np.full(Y.shape, 7, dtype="f4"))
    Y = dpt.astype(X[::2, ::-1], "i4", order="K", copy=False)
    assert Y.usm_data is X.usm_data
    Y = dpt.astype(X, None, order="K")
    if X.sycl_queue.sycl_device.has_aspect_fp64:
        assert Y.dtype is dpt.float64
    else:
        assert Y.dtype is dpt.float32


def test_astype_invalid_order():
    try:
        X = dpt.usm_ndarray(5, "i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    with pytest.raises(ValueError):
        dpt.astype(X, "i4", order="WRONG")


def test_astype_device():
    get_queue_or_skip()
    q1 = dpctl.SyclQueue()
    q2 = dpctl.SyclQueue()

    x = dpt.arange(5, dtype="i4", sycl_queue=q1)
    r = dpt.astype(x, "f4")
    assert r.sycl_queue == x.sycl_queue
    assert r.sycl_device == x.sycl_device

    r = dpt.astype(x, "f4", device=q2)
    assert r.sycl_queue == q2


def test_astype_gh_1926():
    get_queue_or_skip()

    x = dpt.ones(64)
    x_ = dpt.astype(x, x.dtype, copy=False, order="C")
    assert x is x_

    x__ = dpt.astype(x, x.dtype, copy=False, order="F")
    assert x is x__


def test_astype_gh_2121():
    get_queue_or_skip()

    x_np = np.asarray([0, 3, 1, 2, 0, 1], dtype="u1").view("?")
    x = dpt.asarray(x_np)
    res = dpt.astype(x, dpt.uint8)
    expected = dpt.asarray([0, 1, 1, 1, 0, 1], dtype="u1")
    assert dpt.all(res == expected)


def test_copy():
    try:
        X = dpt.usm_ndarray((5, 5), "i4")[2:4, 1:4]
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    X[:] = 42
    Yc = dpt.copy(X, order="C")
    Yf = dpt.copy(X, order="F")
    Ya = dpt.copy(X, order="A")
    Yk = dpt.copy(X, order="K")
    assert Yc.usm_data is not X.usm_data
    assert Yf.usm_data is not X.usm_data
    assert Ya.usm_data is not X.usm_data
    assert Yk.usm_data is not X.usm_data
    assert Yc.strides == (3, 1)
    assert Yf.strides == (1, 2)
    assert Ya.strides == (3, 1)
    assert Yk.strides == (3, 1)
    ref = np.full(X.shape, 42, dtype=X.dtype)
    assert np.array_equal(dpt.asnumpy(Yc), ref)
    assert np.array_equal(dpt.asnumpy(Yf), ref)
    assert np.array_equal(dpt.asnumpy(Ya), ref)
    assert np.array_equal(dpt.asnumpy(Yk), ref)


def test_copy_unaligned():
    get_queue_or_skip()

    x = dpt.ones(513, dtype="i4")
    r = dpt.astype(x[1:], "f4")

    assert dpt.all(r == 1)


def test_ctor_invalid():
    try:
        m = dpm.MemoryUSMShared(12)
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    with pytest.raises(ValueError):
        dpt.usm_ndarray((4,), dtype="i4", buffer=m)
    m = dpm.MemoryUSMShared(64)
    with pytest.raises(ValueError):
        dpt.usm_ndarray((4,), dtype="u1", buffer=m, strides={"not": "valid"})


def test_reshape():
    try:
        X = dpt.usm_ndarray((5, 5), "i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    # can be done as views
    Y = dpt.reshape(X, (25,))
    assert Y.shape == (25,)
    Z = X[::2, ::2]
    # requires a copy
    W = dpt.reshape(Z, (Z.size,), order="F")
    assert W.shape == (Z.size,)
    with pytest.raises(TypeError):
        dpt.reshape("invalid")
    with pytest.raises(ValueError):
        dpt.reshape(Z, (2, 2, 2, 2, 2))
    with pytest.raises(ValueError):
        dpt.reshape(Z, Z.shape, order="invalid")
    W = dpt.reshape(Z, (-1,), order="C")
    assert W.shape == (Z.size,)

    X = dpt.usm_ndarray((1,), dtype="i8")
    Y = dpt.reshape(X, X.shape)
    assert Y.flags == X.flags

    A = dpt.usm_ndarray((0,), "i4")
    A1 = dpt.reshape(A, (0,))
    assert A1.shape == (0,)
    requested_shape = (
        2,
        0,
    )
    A2 = dpt.reshape(A, requested_shape)
    assert A2.shape == requested_shape
    requested_shape = (
        0,
        2,
    )
    A3 = dpt.reshape(A, requested_shape)
    assert A3.shape == requested_shape
    requested_shape = (
        1,
        0,
        2,
    )
    A4 = dpt.reshape(A, requested_shape)
    assert A4.shape == requested_shape


def test_reshape_orderF():
    try:
        a = dpt.arange(6 * 3 * 4, dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    b = dpt.reshape(a, (6, 2, 6))
    c = dpt.reshape(b, (9, 8), order="F")
    assert c.flags.f_contiguous
    assert c._pointer != b._pointer
    assert b._pointer == a._pointer

    a_np = np.arange(6 * 3 * 4, dtype="i4")
    b_np = np.reshape(a_np, (6, 2, 6))
    c_np = np.reshape(b_np, (9, 8), order="F")
    assert np.array_equal(c_np, dpt.asnumpy(c))


def test_reshape_noop():
    """Per gh-1664"""
    try:
        a = dpt.ones((2, 1))
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    b = dpt.reshape(a, (2, 1))
    assert b is a


def test_reshape_zero_size():
    try:
        a = dpt.empty((0,))
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    with pytest.raises(ValueError):
        dpt.reshape(a, (-1, 0))


def test_reshape_large_ndim():
    ndim = 32
    idx = tuple(1 if i + 1 < ndim else ndim for i in range(ndim))
    try:
        d = dpt.ones(ndim, dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    d = dpt.reshape(d, idx)
    assert d.shape == idx


def test_reshape_copy_kwrd():
    try:
        X = dpt.usm_ndarray((2, 3), "i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    new_shape = (6,)
    Z = dpt.reshape(X, new_shape, copy=True)
    assert Z.shape == new_shape
    assert Z.usm_data is not X.usm_data
    X = dpt.usm_ndarray((3, 3), "i4")[::2, ::2]
    new_shape = (4,)
    with pytest.raises(ValueError):
        Z = dpt.reshape(X, new_shape, copy=False)
    with pytest.raises(ValueError):
        invalid = Ellipsis
        Z = dpt.reshape(X, new_shape, copy=invalid)


def test_transpose():
    n, m = 2, 3
    try:
        X = dpt.usm_ndarray((n, m), "f4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    Xnp = np.arange(n * m, dtype="f4").reshape((n, m))
    X[:] = Xnp
    assert np.array_equal(dpt.to_numpy(X.T), Xnp.T)
    assert np.array_equal(dpt.to_numpy(X[1:].T), Xnp[1:].T)


def test_real_imag_views():
    n, m = 2, 3
    try:
        X = dpt.usm_ndarray((n, m), "c8")
        X_scalar = dpt.usm_ndarray((), dtype="c8")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    Xnp_r = np.arange(n * m, dtype="f4").reshape((n, m))
    Xnp_i = np.arange(n * m, 2 * n * m, dtype="f4").reshape((n, m))
    Xnp = Xnp_r + 1j * Xnp_i
    X[:] = Xnp
    X_real = X.real
    X_imag = X.imag
    assert np.array_equal(dpt.to_numpy(X_real), Xnp.real)
    assert np.array_equal(dpt.to_numpy(X.imag), Xnp.imag)
    assert not X_real.flags["C"] and not X_real.flags["F"]
    assert not X_imag.flags["C"] and not X_imag.flags["F"]
    assert X_real.strides == X_imag.strides
    assert np.array_equal(dpt.to_numpy(X[1:].real), Xnp[1:].real)
    assert np.array_equal(dpt.to_numpy(X[1:].imag), Xnp[1:].imag)

    X_scalar[...] = complex(n * m, 2 * n * m)
    assert X_scalar.real and X_scalar.imag

    # check that _zero_like works for scalars
    X_scalar = dpt.usm_ndarray((), dtype="f4")
    assert isinstance(X_scalar.imag, dpt.usm_ndarray)
    assert not X_scalar.imag
    assert X_scalar.real.sycl_queue == X_scalar.imag.sycl_queue


def test_real_imag_views_fp16():
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dpt.float16, q)

    X = dpt.usm_ndarray(
        (3, 4), dtype=dpt.float16, buffer_ctor_kwargs={"queue": q}
    )
    assert isinstance(X.real, dpt.usm_ndarray) and isinstance(
        X.imag, dpt.usm_ndarray
    )


@pytest.mark.parametrize(
    "dtype",
    _all_dtypes,
)
def test_zeros(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)
    X = dpt.zeros(10, dtype=dtype, sycl_queue=q)
    assert np.array_equal(dpt.asnumpy(X), np.zeros(10, dtype=dtype))


@pytest.mark.parametrize(
    "dtype",
    _all_dtypes,
)
def test_ones(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)
    X = dpt.ones(10, dtype=dtype, sycl_queue=q)
    assert np.array_equal(dpt.asnumpy(X), np.ones(10, dtype=dtype))


@pytest.mark.parametrize(
    "dtype",
    _all_dtypes,
)
def test_full(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)
    X = dpt.full(10, 4, dtype=dtype, sycl_queue=q)
    assert np.array_equal(dpt.asnumpy(X), np.full(10, 4, dtype=dtype))


def test_full_cmplx128():
    q = get_queue_or_skip()
    dtype = "c16"
    skip_if_dtype_not_supported(dtype, q)
    fill_v = 1 + 1j
    X = dpt.full(tuple(), fill_value=fill_v, dtype=dtype, sycl_queue=q)
    assert np.array_equal(
        dpt.asnumpy(X), np.full(tuple(), fill_value=fill_v, dtype=dtype)
    )
    fill_v = 0 + 1j
    X = dpt.full(tuple(), fill_value=fill_v, dtype=dtype, sycl_queue=q)
    assert np.array_equal(
        dpt.asnumpy(X), np.full(tuple(), fill_value=fill_v, dtype=dtype)
    )
    fill_v = 0 + 0j
    X = dpt.full(tuple(), fill_value=fill_v, dtype=dtype, sycl_queue=q)
    assert np.array_equal(
        dpt.asnumpy(X), np.full(tuple(), fill_value=fill_v, dtype=dtype)
    )


def test_full_dtype_inference():
    try:
        X = dpt.full(10, 4)
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    assert np.issubdtype(X.dtype, np.integer)
    try:
        X = dpt.full(10, True)
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    assert X.dtype is dpt.dtype(np.bool_)
    assert np.issubdtype(dpt.full(10, 12.3).dtype, np.floating)
    try:
        X = dpt.full(10, 0.3 - 2j)
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    cdt = X.dtype
    assert np.issubdtype(cdt, np.complexfloating)

    assert np.issubdtype(dpt.full(10, 12.3, dtype=int).dtype, np.integer)
    assert np.issubdtype(dpt.full(10, 0.3 - 2j, dtype=int).dtype, np.integer)
    rdt = np.finfo(cdt).dtype
    assert np.issubdtype(dpt.full(10, 0.3 - 2j, dtype=rdt).dtype, np.floating)


@pytest.mark.parametrize("dt", ["f2", "f4", "f8"])
def test_full_special_fp(dt):
    """See gh-1314"""
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    ar = dpt.full(10, fill_value=dpt.nan)
    err_msg = f"Failed for fill_value=dpt.nan and dtype {dt}"
    assert dpt.isnan(ar[0]), err_msg

    ar = dpt.full(10, fill_value=dpt.inf)
    err_msg = f"Failed for fill_value=dpt.inf and dtype {dt}"
    assert dpt.isinf(ar[0]) and dpt.greater(ar[0], 0), err_msg

    ar = dpt.full(10, fill_value=-dpt.inf)
    err_msg = f"Failed for fill_value=-dpt.inf and dtype {dt}"
    assert dpt.isinf(ar[0]) and dpt.less(ar[0], 0), err_msg

    ar = dpt.full(10, fill_value=dpt.pi)
    err_msg = f"Failed for fill_value=dpt.pi and dtype {dt}"
    check = abs(float(ar[0]) - dpt.pi) < 16 * dpt.finfo(ar.dtype).eps
    assert check, err_msg


def test_full_fill_array():
    q = get_queue_or_skip()

    Xnp = np.array([1, 2, 3], dtype="i4")
    X = dpt.asarray(Xnp, sycl_queue=q)

    shape = (3, 3)
    Y = dpt.full(shape, X)
    Ynp = np.full(shape, Xnp)

    assert Y.dtype == Ynp.dtype
    assert Y.usm_type == "device"
    assert np.array_equal(dpt.asnumpy(Y), Ynp)


def test_full_compute_follows_data():
    q1 = get_queue_or_skip()
    q2 = get_queue_or_skip()

    X = dpt.arange(10, dtype="i4", sycl_queue=q1, usm_type="shared")
    Y = dpt.full(10, X[3])

    assert Y.dtype == X.dtype
    assert Y.usm_type == X.usm_type
    assert dpctl.utils.get_execution_queue((Y.sycl_queue, X.sycl_queue))
    assert np.array_equal(dpt.asnumpy(Y), np.full(10, 3, dtype="i4"))

    Y = dpt.full(10, X[3], dtype="f4", sycl_queue=q2, usm_type="host")

    assert Y.dtype == dpt.dtype("f4")
    assert Y.usm_type == "host"
    assert dpctl.utils.get_execution_queue((Y.sycl_queue, q2))
    assert np.array_equal(dpt.asnumpy(Y), np.full(10, 3, dtype="f4"))


@pytest.mark.parametrize("order1", ["F", "C"])
@pytest.mark.parametrize("order2", ["F", "C"])
def test_full_order(order1, order2):
    q = get_queue_or_skip()
    Xnp = np.array([1, 2, 3], order=order1)
    Ynp = np.full((3, 3), Xnp, order=order2)
    Y = dpt.full((3, 3), Xnp, order=order2, sycl_queue=q)
    assert Y.flags.c_contiguous == Ynp.flags.c_contiguous
    assert Y.flags.f_contiguous == Ynp.flags.f_contiguous
    assert np.array_equal(dpt.asnumpy(Y), Ynp)


def test_full_strides():
    q = get_queue_or_skip()
    X = dpt.full((3, 3), dpt.arange(3, dtype="i4"), sycl_queue=q)
    Xnp = np.full((3, 3), np.arange(3, dtype="i4"))
    assert X.strides == tuple(el // Xnp.itemsize for el in Xnp.strides)
    assert np.array_equal(dpt.asnumpy(X), Xnp)

    X = dpt.full((3, 3), dpt.arange(6, dtype="i4")[::2], sycl_queue=q)
    Xnp = np.full((3, 3), np.arange(6, dtype="i4")[::2])
    assert X.strides == tuple(el // Xnp.itemsize for el in Xnp.strides)
    assert np.array_equal(dpt.asnumpy(X), Xnp)


@pytest.mark.parametrize("dt", ["i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8"])
def test_full_gh_1230(dt):
    get_queue_or_skip()
    dtype = dpt.dtype(dt)
    dt_maxint = dpt.iinfo(dtype).max

    if (dtype.itemsize < 8) and (np.lib.NumpyVersion(np.__version__) < "2.0.0"):
        try:
            X = dpt.full(1, fill_value=(dt_maxint + 1), dtype=dt)
        except OverflowError:
            pytest.skip("Expected OverflowError raised")
        Y = dpt.full_like(X, fill_value=dpt.iinfo(dt).min)
        assert dpt.all(X == Y)
    else:
        with pytest.raises(OverflowError):
            dpt.full(1, dt_maxint + 1, dtype=dt)


@pytest.mark.parametrize(
    "dt",
    _all_dtypes[1:],
)
def test_arange(dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)
    X = dpt.arange(0, 123, dtype=dt, sycl_queue=q)
    dt = dpt.dtype(dt)
    if np.issubdtype(dt, np.integer):
        assert int(X[47]) == 47
    elif np.issubdtype(dt, np.floating):
        assert float(X[47]) == 47.0
    elif np.issubdtype(dt, np.complexfloating):
        assert complex(X[47]) == 47.0 + 0.0j

    # choose size larger than maximal value that u1/u2 can accommodate
    sz = int(dpt.iinfo(dpt.int8).max)
    X1 = dpt.arange(sz + 1, dtype=dt, sycl_queue=q)
    assert X1.shape == (sz + 1,)

    X2 = dpt.arange(sz, 0, -1, dtype=dt, sycl_queue=q)
    assert X2.shape == (sz,)


def test_arange_fp():
    q = get_queue_or_skip()

    assert dpt.arange(7, 0, -2, dtype="f4", device=q).shape == (4,)
    assert dpt.arange(0, 1, 0.25, dtype="f4", device=q).shape == (4,)

    has_fp64 = q.sycl_device.has_aspect_fp64
    if has_fp64:
        assert dpt.arange(7, 0, -2, dtype="f8", device=q).shape == (4,)
    assert dpt.arange(0, 1, 0.25, dtype="f4", device=q).shape == (4,)

    x = dpt.arange(9.7, stop=10, sycl_queue=q)
    assert x.shape == (1,)
    assert x.dtype == dpt.float64 if has_fp64 else dpt.float32


def test_arange_step_None():
    q = get_queue_or_skip()

    x = dpt.arange(0, stop=10, step=None, dtype="int32", sycl_queue=q)
    assert x.shape == (10,)


def test_arange_bool():
    q = get_queue_or_skip()

    x = dpt.arange(0, stop=2, dtype="bool", sycl_queue=q)
    assert x.shape == (2,)
    assert x.dtype == dpt.bool


def test_arange_mixed_types():
    q = get_queue_or_skip()

    x = dpt.arange(-2.5, stop=200, step=100, dtype="int32", sycl_queue=q)
    assert x.shape[0] == 3
    assert int(x[1]) == 99 + int(x[0])

    x = dpt.arange(+2.5, stop=200, step=100, dtype="int32", device=x.device)
    assert x.shape[0] == 2
    assert int(x[1]) == 100 + int(x[0])

    _stop = np.float32(504)
    x = dpt.arange(0, stop=_stop, step=100, dtype="f4", device=x.device)
    assert x.shape == (6,)

    # ensure length is determined using uncast parameters
    x = dpt.arange(-5, stop=10**2, step=2.7, dtype="int64", device=x.device)
    assert x.shape == (39,)


@pytest.mark.parametrize(
    "dt",
    _all_dtypes,
)
def test_linspace(dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)
    X = dpt.linspace(0, 1, num=2, dtype=dt, sycl_queue=q)
    assert np.allclose(dpt.asnumpy(X), np.linspace(0, 1, num=2, dtype=dt))


def test_linspace_fp():
    q = get_queue_or_skip()
    n = 16
    X = dpt.linspace(0, n - 1, num=n, sycl_queue=q)
    if q.sycl_device.has_aspect_fp64:
        assert X.dtype == dpt.dtype("float64")
    else:
        assert X.dtype == dpt.dtype("float32")
    assert X.shape == (n,)
    assert X.strides == (1,)


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_linspace_fp_max(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)
    n = 16
    dt = dpt.dtype(dtype)
    max_ = dpt.finfo(dt).max
    X = dpt.linspace(max_, max_, endpoint=True, num=n, dtype=dt, sycl_queue=q)
    assert X.shape == (n,)
    assert X.strides == (1,)
    assert np.allclose(
        dpt.asnumpy(X), np.linspace(max_, max_, endpoint=True, num=n, dtype=dt)
    )


def test_linspace_int():
    q = get_queue_or_skip()
    X = dpt.linspace(0.1, 9.1, 11, endpoint=True, dtype=int, sycl_queue=q)
    Xnp = np.linspace(0.1, 9.1, 11, endpoint=True, dtype=int)
    assert np.array_equal(dpt.asnumpy(X), Xnp)


@pytest.mark.parametrize(
    "dt",
    _all_dtypes,
)
@pytest.mark.parametrize(
    "usm_kind",
    [
        "shared",
        "device",
        "host",
    ],
)
def test_empty_like(dt, usm_kind):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    X = dpt.empty((4, 5), dtype=dt, usm_type=usm_kind, sycl_queue=q)
    Y = dpt.empty_like(X)
    assert X.shape == Y.shape
    assert X.dtype == Y.dtype
    assert X.usm_type == Y.usm_type
    assert X.sycl_queue == Y.sycl_queue

    X = dpt.empty(tuple(), dtype=dt, usm_type=usm_kind, sycl_queue=q)
    Y = dpt.empty_like(X)
    assert X.shape == Y.shape
    assert X.dtype == Y.dtype
    assert X.usm_type == Y.usm_type
    assert X.sycl_queue == Y.sycl_queue


def test_empty_unexpected_data_type():
    with pytest.raises(TypeError):
        try:
            dpt.empty(1, dtype=np.object_)
        except dpctl.SyclDeviceCreationError:
            pytest.skip("No SYCL devices available")


@pytest.mark.parametrize(
    "dt",
    _all_dtypes,
)
@pytest.mark.parametrize(
    "usm_kind",
    [
        "shared",
        "device",
        "host",
    ],
)
def test_zeros_like(dt, usm_kind):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    X = dpt.empty((4, 5), dtype=dt, usm_type=usm_kind, sycl_queue=q)
    Y = dpt.zeros_like(X)
    assert X.shape == Y.shape
    assert X.dtype == Y.dtype
    assert X.usm_type == Y.usm_type
    assert X.sycl_queue == Y.sycl_queue
    assert np.allclose(dpt.asnumpy(Y), np.zeros(X.shape, dtype=X.dtype))

    X = dpt.empty(tuple(), dtype=dt, usm_type=usm_kind, sycl_queue=q)
    Y = dpt.zeros_like(X)
    assert X.shape == Y.shape
    assert X.dtype == Y.dtype
    assert X.usm_type == Y.usm_type
    assert X.sycl_queue == Y.sycl_queue
    assert np.array_equal(dpt.asnumpy(Y), np.zeros(X.shape, dtype=X.dtype))


@pytest.mark.parametrize(
    "dt",
    _all_dtypes,
)
@pytest.mark.parametrize(
    "usm_kind",
    [
        "shared",
        "device",
        "host",
    ],
)
def test_ones_like(dt, usm_kind):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    X = dpt.empty((4, 5), dtype=dt, usm_type=usm_kind, sycl_queue=q)
    Y = dpt.ones_like(X)
    assert X.shape == Y.shape
    assert X.dtype == Y.dtype
    assert X.usm_type == Y.usm_type
    assert X.sycl_queue == Y.sycl_queue
    assert np.allclose(dpt.asnumpy(Y), np.ones(X.shape, dtype=X.dtype))

    X = dpt.empty(tuple(), dtype=dt, usm_type=usm_kind, sycl_queue=q)
    Y = dpt.ones_like(X)
    assert X.shape == Y.shape
    assert X.dtype == Y.dtype
    assert X.usm_type == Y.usm_type
    assert X.sycl_queue == Y.sycl_queue
    assert np.array_equal(dpt.asnumpy(Y), np.ones(X.shape, dtype=X.dtype))


@pytest.mark.parametrize(
    "dt",
    _all_dtypes,
)
@pytest.mark.parametrize(
    "usm_kind",
    [
        "shared",
        "device",
        "host",
    ],
)
def test_full_like(dt, usm_kind):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    fill_v = dpt.dtype(dt).type(1)
    X = dpt.empty((4, 5), dtype=dt, usm_type=usm_kind, sycl_queue=q)
    Y = dpt.full_like(X, fill_v)
    assert X.shape == Y.shape
    assert X.dtype == Y.dtype
    assert X.usm_type == Y.usm_type
    assert X.sycl_queue == Y.sycl_queue
    assert np.allclose(dpt.asnumpy(Y), np.ones(X.shape, dtype=X.dtype))

    X = dpt.empty(tuple(), dtype=dt, usm_type=usm_kind, sycl_queue=q)
    Y = dpt.full_like(X, fill_v)
    assert X.shape == Y.shape
    assert X.dtype == Y.dtype
    assert X.usm_type == Y.usm_type
    assert X.sycl_queue == Y.sycl_queue
    assert np.array_equal(dpt.asnumpy(Y), np.ones(X.shape, dtype=X.dtype))


@pytest.mark.parametrize("dtype", _all_dtypes)
@pytest.mark.parametrize("usm_kind", ["shared", "device", "host"])
def test_eye(dtype, usm_kind):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    X = dpt.eye(4, 5, k=1, dtype=dtype, usm_type=usm_kind, sycl_queue=q)
    Xnp = np.eye(4, 5, k=1, dtype=dtype)
    assert X.dtype == Xnp.dtype
    assert np.array_equal(Xnp, dpt.asnumpy(X))


@pytest.mark.parametrize("dtype", _all_dtypes[1:])
def test_tril(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    shape = (2, 3, 4, 5, 5)
    X = dpt.reshape(dpt.arange(prod(shape), dtype=dtype, sycl_queue=q), shape)
    Y = dpt.tril(X)
    Xnp = np.arange(prod(shape), dtype=dtype).reshape(shape)
    Ynp = np.tril(Xnp)
    assert Y.dtype == Ynp.dtype
    assert np.array_equal(Ynp, dpt.asnumpy(Y))


@pytest.mark.parametrize("dtype", _all_dtypes[1:])
def test_triu(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    shape = (4, 5)
    X = dpt.reshape(dpt.arange(prod(shape), dtype=dtype, sycl_queue=q), shape)
    Y = dpt.triu(X, k=1)
    Xnp = np.arange(prod(shape), dtype=dtype).reshape(shape)
    Ynp = np.triu(Xnp, k=1)
    assert Y.dtype == Ynp.dtype
    assert np.array_equal(Ynp, dpt.asnumpy(Y))


@pytest.mark.parametrize("tri_fn", [dpt.tril, dpt.triu])
@pytest.mark.parametrize("usm_type", ["device", "shared", "host"])
def test_tri_usm_type(tri_fn, usm_type):
    q = get_queue_or_skip()
    dtype = dpt.uint16

    shape = (2, 3, 4, 5, 5)
    size = prod(shape)
    X = dpt.reshape(
        dpt.arange(size, dtype=dtype, usm_type=usm_type, sycl_queue=q), shape
    )
    Y = tri_fn(X)  # main execution branch
    assert Y.usm_type == X.usm_type
    assert Y.sycl_queue == q
    Y = tri_fn(X, k=-6)  # special case of Y == X
    assert Y.usm_type == X.usm_type
    assert Y.sycl_queue == q
    Y = tri_fn(X, k=6)  # special case of Y == 0
    assert Y.usm_type == X.usm_type
    assert Y.sycl_queue == q


def test_tril_slice():
    q = get_queue_or_skip()

    shape = (6, 10)
    X = dpt.reshape(dpt.arange(prod(shape), dtype="int", sycl_queue=q), shape)[
        1:, ::-2
    ]
    Y = dpt.tril(X)
    Xnp = np.arange(prod(shape), dtype="int").reshape(shape)[1:, ::-2]
    Ynp = np.tril(Xnp)
    assert Y.dtype == Ynp.dtype
    assert np.array_equal(Ynp, dpt.asnumpy(Y))


def test_triu_permute_dims():
    q = get_queue_or_skip()

    shape = (2, 3, 4, 5)
    X = dpt.permute_dims(
        dpt.reshape(dpt.arange(prod(shape), dtype="int", sycl_queue=q), shape),
        (3, 2, 1, 0),
    )
    Y = dpt.triu(X)
    Xnp = np.transpose(
        np.arange(prod(shape), dtype="int").reshape(shape), (3, 2, 1, 0)
    )
    Ynp = np.triu(Xnp)
    assert Y.dtype == Ynp.dtype
    assert np.array_equal(Ynp, dpt.asnumpy(Y))


def test_tril_broadcast_to():
    q = get_queue_or_skip()

    shape = (5, 5)
    X = dpt.broadcast_to(dpt.ones((1), dtype="int", sycl_queue=q), shape)
    Y = dpt.tril(X)
    Xnp = np.broadcast_to(np.ones((1), dtype="int"), shape)
    Ynp = np.tril(Xnp)
    assert Y.dtype == Ynp.dtype
    assert np.array_equal(Ynp, dpt.asnumpy(Y))


def test_triu_bool():
    q = get_queue_or_skip()

    shape = (4, 5)
    X = dpt.ones((shape), dtype="bool", sycl_queue=q)
    Y = dpt.triu(X)
    Xnp = np.ones((shape), dtype="bool")
    Ynp = np.triu(Xnp)
    assert Y.dtype == Ynp.dtype
    assert np.array_equal(Ynp, dpt.asnumpy(Y))


@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("k", [-10, -2, -1, 3, 4, 10])
def test_triu_order_k(order, k):
    q = get_queue_or_skip()

    shape = (3, 3)
    X = dpt.reshape(
        dpt.arange(prod(shape), dtype="int", sycl_queue=q),
        shape,
        order=order,
    )
    Y = dpt.triu(X, k=k)
    Xnp = np.arange(prod(shape), dtype="int").reshape(shape, order=order)
    Ynp = np.triu(Xnp, k=k)
    assert Y.dtype == Ynp.dtype
    assert X.flags == Y.flags
    assert np.array_equal(Ynp, dpt.asnumpy(Y))


@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("k", [-10, -4, -3, 1, 2, 10])
def test_tril_order_k(order, k):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")
    shape = (3, 3)
    X = dpt.reshape(
        dpt.arange(prod(shape), dtype="int", sycl_queue=q),
        shape,
        order=order,
    )
    Y = dpt.tril(X, k=k)
    Xnp = np.arange(prod(shape), dtype="int").reshape(shape, order=order)
    Ynp = np.tril(Xnp, k=k)
    assert Y.dtype == Ynp.dtype
    assert X.flags == Y.flags
    assert np.array_equal(Ynp, dpt.asnumpy(Y))


def test_meshgrid():
    q = get_queue_or_skip()

    X = dpt.arange(5, sycl_queue=q)
    Y = dpt.arange(3, sycl_queue=q)
    Z = dpt.meshgrid(X, Y)
    Znp = np.meshgrid(dpt.asnumpy(X), dpt.asnumpy(Y))
    n = len(Z)
    assert n == len(Znp)
    for i in range(n):
        assert np.array_equal(dpt.asnumpy(Z[i]), Znp[i])
    assert dpt.meshgrid() == []
    # dimension > 1 must raise ValueError
    with pytest.raises(ValueError):
        dpt.meshgrid(dpt.usm_ndarray((4, 4)))
    # unknown indexing kwarg must raise ValueError
    with pytest.raises(ValueError):
        dpt.meshgrid(X, indexing="ji")
    # input arrays with different data types must raise ValueError
    with pytest.raises(ValueError):
        dpt.meshgrid(X, dpt.asarray(Y, dtype="b1"))


def test_meshgrid2():
    q1 = get_queue_or_skip()
    q2 = get_queue_or_skip()
    q3 = get_queue_or_skip()

    x1 = dpt.arange(0, 2, dtype="int16", sycl_queue=q1)
    x2 = dpt.arange(3, 6, dtype="int16", sycl_queue=q2)
    x3 = dpt.arange(6, 10, dtype="int16", sycl_queue=q3)
    y1, y2, y3 = dpt.meshgrid(x1, x2, x3, indexing="xy")
    z1, z2, z3 = dpt.meshgrid(x1, x2, x3, indexing="ij")
    assert all(
        x.sycl_queue == y.sycl_queue for x, y in zip((x1, x2, x3), (y1, y2, y3))
    )
    assert all(
        x.sycl_queue == z.sycl_queue for x, z in zip((x1, x2, x3), (z1, z2, z3))
    )
    assert y1.shape == y2.shape and y2.shape == y3.shape
    assert z1.shape == z2.shape and z2.shape == z3.shape
    assert y1.shape == (len(x2), len(x1), len(x3))
    assert z1.shape == (len(x1), len(x2), len(x3))


def test_common_arg_validation():
    order = "I"
    # invalid order must raise ValueError
    with pytest.raises(ValueError):
        dpt.empty(10, order=order)
    with pytest.raises(ValueError):
        dpt.zeros(10, order=order)
    with pytest.raises(ValueError):
        dpt.ones(10, order=order)
    with pytest.raises(ValueError):
        dpt.full(10, 1, order=order)
    with pytest.raises(ValueError):
        dpt.eye(10, order=order)
    try:
        X = dpt.empty(10)
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    with pytest.raises(ValueError):
        dpt.empty_like(X, order=order)
    with pytest.raises(ValueError):
        dpt.zeros_like(X, order=order)
    with pytest.raises(ValueError):
        dpt.ones_like(X, order=order)
    with pytest.raises(ValueError):
        dpt.full_like(X, 1, order=order)
    X = {}
    # test for type validation
    with pytest.raises(TypeError):
        dpt.empty_like(X)
    with pytest.raises(TypeError):
        dpt.zeros_like(X)
    with pytest.raises(TypeError):
        dpt.ones_like(X)
    with pytest.raises(TypeError):
        dpt.full_like(X, 1)
    with pytest.raises(TypeError):
        dpt.tril(X)
    with pytest.raises(TypeError):
        dpt.triu(X)
    with pytest.raises(TypeError):
        dpt.meshgrid(X)


def test_flags():
    try:
        x = dpt.empty(tuple(), dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    f = x.flags
    # check comparison with generic types
    assert f != Ellipsis
    f.__repr__()
    assert f.c_contiguous == f["C"]
    assert f.f_contiguous == f["F"]
    assert f.contiguous == f["CONTIGUOUS"]
    assert f.fc == f["FC"]
    assert f.forc == f["FORC"]
    assert f.fnc == f["FNC"]
    assert f.writable == f["W"]


def test_asarray_uint64():
    Xnp = np.ndarray(1, dtype=np.uint64)
    try:
        X = dpt.asarray(Xnp)
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    assert X.dtype == Xnp.dtype


def test_Device():
    try:
        dev = dpctl.select_default_device()
        d1 = dpt.Device.create_device(dev)
        d2 = dpt.Device.create_device(dev)
    except (dpctl.SyclQueueCreationError, dpctl.SyclDeviceCreationError):
        pytest.skip(
            "Could not create default device, or a queue that targets it"
        )
    assert d1 == d2
    dict = {d1: 1}
    assert dict[d2] == 1
    assert d1 == d2.sycl_queue
    assert not d1 == Ellipsis


def test_element_offset():
    n0, n1 = 3, 8
    try:
        x = dpt.empty((n0, n1), dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    assert isinstance(x._element_offset, int)
    assert x._element_offset == 0
    y = x[::-1, ::2]
    assert y._element_offset == (n0 - 1) * n1


def test_byte_bounds():
    n0, n1 = 3, 8
    try:
        x = dpt.empty((n0, n1), dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    assert isinstance(x._byte_bounds, tuple)
    assert len(x._byte_bounds) == 2
    lo, hi = x._byte_bounds
    assert hi - lo == n0 * n1 * x.itemsize
    y = x[::-1, ::2]
    lo, hi = y._byte_bounds
    assert hi - lo == (n0 * n1 - 1) * x.itemsize


def test_gh_1201():
    n = 100
    a = np.flipud(np.arange(n, dtype="i4"))
    try:
        b = dpt.asarray(a)
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    assert (dpt.asnumpy(b) == a).all()
    c = dpt.flip(dpt.empty(a.shape, dtype=a.dtype))
    c[:] = a
    assert (dpt.asnumpy(c) == a).all()


class ObjWithSyclUsmArrayInterface:
    def __init__(self, ary):
        self._array_obj = ary

    @property
    def __sycl_usm_array_interface__(self):
        _suai = self._array_obj.__sycl_usm_array_interface__
        return _suai


@pytest.mark.parametrize("ro_flag", [True, False])
def test_asarray_writable_flag(ro_flag):
    try:
        a = dpt.empty(8)
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")

    a.flags["W"] = not ro_flag
    wrapped = ObjWithSyclUsmArrayInterface(a)

    b = dpt.asarray(wrapped)

    assert b.flags["W"] == (not ro_flag)
    assert b._pointer == a._pointer


def test_getitem_validation():
    """Test based on gh-1785"""
    try:
        a = dpt.empty((2, 2, 2))
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    with pytest.raises(IndexError):
        a[0.0]
    with pytest.raises(IndexError):
        a[1, 0.0, ...]
    with pytest.raises(IndexError):
        a[1, 0.0, dpt.newaxis, 1]
    with pytest.raises(IndexError):
        a[dpt.newaxis, ..., 0.0]
    with pytest.raises(IndexError):
        a[dpt.newaxis, ..., 0.0, dpt.newaxis]
    with pytest.raises(IndexError):
        a[..., 0.0, dpt.newaxis]
    with pytest.raises(IndexError):
        a[:, 0.0, dpt.newaxis]


def test_array_like_ctors_order_K():
    get_queue_or_skip()

    sh = (10, 10)
    x1 = dpt.zeros(sh, dtype="i4", order="C")
    r1 = dpt.full_like(x1, 2, order="K")
    assert dpt.all(r1 == 2)
    assert r1.flags.c_contiguous
    r2 = dpt.empty_like(x1, order="K")
    assert r2.flags.c_contiguous
    r3 = dpt.ones_like(x1, order="K")
    assert dpt.all(r3 == 1)
    assert r3.flags.c_contiguous
    r4 = dpt.zeros_like(x1, order="K")
    assert dpt.all(r4 == 0)
    assert r4.flags.c_contiguous

    x2 = dpt.zeros(sh, dtype="i4", order="F")
    r5 = dpt.full_like(x2, 2, order="K")
    assert dpt.all(r5 == 2)
    assert r5.flags.f_contiguous
    r6 = dpt.empty_like(x2, order="K")
    assert r6.flags.f_contiguous
    r7 = dpt.ones_like(x2, order="K")
    assert dpt.all(r7 == 1)
    assert r7.flags.f_contiguous
    r8 = dpt.zeros_like(x2, order="K")
    assert dpt.all(r8 == 0)
    assert r8.flags.f_contiguous

    x3 = dpt.zeros(sh, dtype="i4", order="C")[::-2, :5]
    st_expected = (-5, 1)
    r9 = dpt.full_like(x3, 2, order="K")
    assert dpt.all(r1 == 2)
    assert r9.strides == st_expected
    assert not r9.flags.forc
    r10 = dpt.empty_like(x3, order="K")
    assert not r10.flags.forc
    assert r10.strides == st_expected
    r11 = dpt.ones_like(x3, order="K")
    assert dpt.all(r11 == 1)
    assert not r11.flags.forc
    assert r11.strides == st_expected
    r12 = dpt.zeros_like(x3, order="K")
    assert dpt.all(r12 == 0)
    assert not r12.flags.forc
    assert r12.strides == st_expected


def test_array_like_ctors_order_A():
    get_queue_or_skip()

    sh = (10, 10)
    x1 = dpt.zeros(sh, dtype="i4", order="C")
    r1 = dpt.full_like(x1, 2, order="A")
    assert dpt.all(r1 == 2)
    assert r1.flags.c_contiguous
    r2 = dpt.empty_like(x1, order="A")
    assert r2.flags.c_contiguous
    r3 = dpt.ones_like(x1, order="A")
    assert dpt.all(r3 == 1)
    assert r3.flags.c_contiguous
    r4 = dpt.zeros_like(x1, order="A")
    assert dpt.all(r4 == 0)
    assert r4.flags.c_contiguous

    x2 = dpt.zeros(sh, dtype="i4", order="F")
    r5 = dpt.full_like(x2, 2, order="A")
    assert dpt.all(r5 == 2)
    assert r5.flags.f_contiguous
    r6 = dpt.empty_like(x2, order="A")
    assert r6.flags.f_contiguous
    r7 = dpt.ones_like(x2, order="A")
    assert dpt.all(r7 == 1)
    assert r7.flags.f_contiguous
    r8 = dpt.zeros_like(x2, order="A")
    assert dpt.all(r8 == 0)
    assert r8.flags.f_contiguous

    x3 = dpt.zeros(sh, dtype="i4", order="C")[::-2, :5]
    r9 = dpt.full_like(x3, 2, order="A")
    assert dpt.all(r1 == 2)
    assert r9.flags.c_contiguous
    r10 = dpt.empty_like(x3, order="A")
    assert r10.flags.c_contiguous
    r11 = dpt.ones_like(x3, order="A")
    assert dpt.all(r11 == 1)
    assert r11.flags.c_contiguous
    r12 = dpt.zeros_like(x3, order="A")
    assert dpt.all(r12 == 0)
    assert r12.flags.c_contiguous


def test_full_like_order_K_array_fill_v():
    get_queue_or_skip()

    x = dpt.zeros((10, 10), dtype="i4")
    fill_v = dpt.asarray(2, dtype="i4")

    r1 = dpt.full_like(x, fill_v, order="K")
    assert dpt.all(r1 == 2)

    # broadcast behavior
    fill_v = dpt.arange(10, dtype="i4")[:, dpt.newaxis]
    r1 = dpt.full_like(x, fill_v, order="K")
    assert dpt.all(r1 == dpt.tile(fill_v, (1, 10)))


def test_full_like_order_K_same_input_output_queues():
    q1 = get_queue_or_skip()
    q2 = get_queue_or_skip()

    x = dpt.zeros((10, 10), dtype="i4", sycl_queue=q1)
    fill_v = dpt.asarray(2, dtype="i4", sycl_queue=q2)

    r = dpt.full_like(x, fill_v, order="K")
    assert r.sycl_queue == x.sycl_queue


def test_asarray_from_numpy_contig():
    get_queue_or_skip()

    i_dt = np.int64
    Xnp = np.arange(32, dtype=i_dt)

    fp_dt = dpt.float32
    # Use contig copy kernel
    Xdpt = dpt.asarray(Xnp, dtype=fp_dt)

    assert dpt.all(Xdpt == dpt.arange(32, dtype=fp_dt))


def test_setitem_from_numpy_contig():
    get_queue_or_skip()

    i_dt = np.int64
    fp_dt = dpt.float32

    Xnp = np.flip(np.arange(32, dtype=i_dt))
    Xdpt = dpt.flip(dpt.empty(Xnp.shape, dtype=fp_dt))
    # Use contig copy kernel, after stride simplification
    Xdpt[:] = Xnp

    expected = dpt.arange(31, stop=-1, step=-1, dtype=fp_dt)
    assert dpt.all(Xdpt == expected)

    Xnp = np.fliplr(np.reshape(np.arange(-10, 10, dtype=i_dt), (4, 5)))
    Xdpt = dpt.flip(dpt.empty(Xnp.shape, dtype=fp_dt), axis=-1)

    # after stride simplification, contig kernel is used
    Xdpt[:] = Xnp

    expected = dpt.reshape(dpt.arange(-10, 10, dtype=fp_dt), (4, 5))
    assert dpt.all(dpt.flip(Xdpt, axis=-1) == expected)


def test_full_functions_raise_type_error():
    get_queue_or_skip()

    with pytest.raises(TypeError):
        dpt.full(1, "0")

    x = dpt.ones(1, dtype="i4")
    with pytest.raises(TypeError):
        dpt.full_like(x, "0")


@pytest.mark.parametrize("dt", _all_dtypes)
def test_setitem_copy_as_contig_alignment(dt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dt, q)

    dtype_ = dpt.dtype(dt)
    n0, n1 = 8, 23

    x = dpt.zeros((n0, n1), dtype=dtype_, sycl_queue=q)

    vals = dpt.ones(n1, dtype=dtype_, sycl_queue=q)[dpt.newaxis, :]
    x[1:, ...] = vals
    assert dpt.all(x[0] == 0)
    assert dpt.all(x[1:, :] == vals)


def test_asarray_property():
    get_queue_or_skip()

    x = dpt.ones(11, dtype="i4")

    with pytest.raises(TypeError):
        np.asarray(x)
