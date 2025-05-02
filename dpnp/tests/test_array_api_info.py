import numpy
import pytest
from dpctl import SyclDeviceCreationError, get_devices, select_default_device
from dpctl.tensor._tensor_impl import default_device_complex_type

import dpnp
from dpnp.tests.helper import (
    has_support_aspect64,
    is_win_platform,
    numpy_version,
)

info = dpnp.__array_namespace_info__()
default_device = select_default_device()


def test_capabilities():
    caps = info.capabilities()
    assert caps["boolean indexing"] is True
    assert caps["data-dependent shapes"] is True
    assert caps["max dimensions"] == None


def test_default_device():
    assert info.default_device() == default_device


def test_default_dtypes():
    dtypes = info.default_dtypes()
    assert (
        dtypes["real floating"]
        == dpnp.default_float_type()
        == dpnp.asarray(0.0).dtype
    )
    # TODO: add dpnp.default_complex_type() function
    assert (
        dtypes["complex floating"]
        == default_device_complex_type(default_device)
        == dpnp.asarray(0.0j).dtype
    )
    if not is_win_platform() or numpy_version() >= "2.0.0":
        # numpy changed default integer on Windows since 2.0
        assert dtypes["integral"] == dpnp.intp == dpnp.asarray(0).dtype
        assert (
            dtypes["indexing"] == dpnp.intp == dpnp.argmax(dpnp.zeros(10)).dtype
        )

    with pytest.raises(
        TypeError, match="Unsupported type for device argument:"
    ):
        info.default_dtypes(device=1)


def test_dtypes_all():
    dtypes = info.dtypes()
    assert dtypes == (
        {
            "bool": dpnp.bool_,
            "int8": dpnp.int8,
            "int16": dpnp.int16,
            "int32": dpnp.int32,
            "int64": dpnp.int64,
            "uint8": dpnp.uint8,
            "uint16": dpnp.uint16,
            "uint32": dpnp.uint32,
            "uint64": dpnp.uint64,
            "float32": dpnp.float32,
        }
        | ({"float64": dpnp.float64} if has_support_aspect64() else {})
        | {"complex64": dpnp.complex64}
        | ({"complex128": dpnp.complex128} if has_support_aspect64() else {})
    )


dtype_categories = {
    "bool": {"bool": dpnp.bool_},
    "signed integer": {
        "int8": dpnp.int8,
        "int16": dpnp.int16,
        "int32": dpnp.int32,
        "int64": dpnp.int64,
    },
    "unsigned integer": {
        "uint8": dpnp.uint8,
        "uint16": dpnp.uint16,
        "uint32": dpnp.uint32,
        "uint64": dpnp.uint64,
    },
    "integral": ("signed integer", "unsigned integer"),
    "real floating": {"float32": dpnp.float32}
    | ({"float64": dpnp.float64} if has_support_aspect64() else {}),
    "complex floating": {"complex64": dpnp.complex64}
    | ({"complex128": dpnp.complex128} if has_support_aspect64() else {}),
    "numeric": ("integral", "real floating", "complex floating"),
}


@pytest.mark.parametrize("kind", dtype_categories)
def test_dtypes_kind(kind):
    expected = dtype_categories[kind]
    if isinstance(expected, tuple):
        assert info.dtypes(kind=kind) == info.dtypes(kind=expected)
    else:
        assert info.dtypes(kind=kind) == expected


def test_dtypes_tuple():
    dtypes = info.dtypes(kind=("bool", "integral"))
    assert dtypes == {
        "bool": dpnp.bool_,
        "int8": dpnp.int8,
        "int16": dpnp.int16,
        "int32": dpnp.int32,
        "int64": dpnp.int64,
        "uint8": dpnp.uint8,
        "uint16": dpnp.uint16,
        "uint32": dpnp.uint32,
        "uint64": dpnp.uint64,
    }


def test_dtypes_invalid_kind():
    with pytest.raises(ValueError, match="Unrecognized data type kind"):
        info.dtypes(kind="invalid")


def test_dtypes_invalid_device():
    with pytest.raises(SyclDeviceCreationError, match="Could not create"):
        info.dtypes(device="str")


def test_devices():
    assert info.devices() == get_devices()
