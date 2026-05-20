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

import dpctl
import dpctl.memory as dpm
import numpy as np
import pytest

import dpnp.tensor as dpt

from .helper import get_queue_or_skip


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
    any_usm_ndarray = dpt.empty((), dtype="i4", sycl_queue=q)
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
    usm_ndarray = dpt.empty((), dtype="i4", sycl_queue=q)
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
    usm_ndarray = dpt.empty((), dtype="i4", sycl_queue=q)
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
    usm_ndarray = dpt.empty((), dtype="i4", sycl_queue=q)

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
