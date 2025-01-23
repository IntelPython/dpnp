import numpy
import pytest
from dpctl import get_devices, select_default_device
from dpctl.tensor._tensor_impl import default_device_complex_type

import dpnp

info = dpnp.__array_namespace_info__()
default_device = select_default_device()


def test_capabilities():
    caps = info.capabilities()
    assert caps["boolean indexing"] is True
    assert caps["data-dependent shapes"] is True
    assert caps["max dimensions"] == 64


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
    assert dtypes["integral"] == dpnp.intp == dpnp.asarray(0).dtype
    assert dtypes["indexing"] == dpnp.intp == dpnp.argmax(dpnp.zeros(10)).dtype

    with pytest.raises(TypeError, match="Expected type"):
        info.default_dtypes(device="gpu")


def test_dtypes_all():
    dtypes = info.dtypes()
    assert dtypes == (
        {
            "bool": dpnp.bool_,
            "int8": numpy.int8,  # TODO: replace with dpnp.int8
            "int16": numpy.int16,  # TODO: replace with dpnp.int16
            "int32": dpnp.int32,
            "int64": dpnp.int64,
            "uint8": numpy.uint8,  # TODO: replace with dpnp.uint8
            "uint16": numpy.uint16,  # TODO: replace with dpnp.uint16
            "uint32": numpy.uint32,  # TODO: replace with dpnp.uint32
            "uint64": numpy.uint64,  # TODO: replace with dpnp.uint64
            "float32": dpnp.float32,
        }
        | ({"float64": dpnp.float64} if default_device.has_aspect_fp64 else {})
        | {"complex64": dpnp.complex64}
        |
        # TODO: update once dpctl-1977 is resolved
        {"complex128": dpnp.complex128}
        # ({"complex128": dpnp.complex128} if default_device.has_aspect_fp64 else {})
    )


dtype_categories = {
    "bool": {"bool": dpnp.bool_},
    "signed integer": {
        "int8": numpy.int8,  # TODO: replace with dpnp.int8
        "int16": numpy.int16,  # TODO: replace with dpnp.int16
        "int32": dpnp.int32,
        "int64": dpnp.int64,
    },
    "unsigned integer": {  # TODO: replace with dpnp dtypes once available
        "uint8": numpy.uint8,
        "uint16": numpy.uint16,
        "uint32": numpy.uint32,
        "uint64": numpy.uint64,
    },
    "integral": ("signed integer", "unsigned integer"),
    "real floating": {"float32": dpnp.float32}
    | ({"float64": dpnp.float64} if default_device.has_aspect_fp64 else {}),
    "complex floating": {"complex64": dpnp.complex64} |
    # TODO: update once dpctl-1977 is resolved
    {"complex128": dpnp.complex128},
    # ({"complex128": dpnp.complex128} if default_device.has_aspect_fp64 else {}),
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
        "int8": numpy.int8,  # TODO: replace with dpnp.int8
        "int16": numpy.int16,  # TODO: replace with dpnp.int16
        "int32": dpnp.int32,
        "int64": dpnp.int64,
        "uint8": numpy.uint8,  # TODO: replace with dpnp.uint8
        "uint16": numpy.uint16,  # TODO: replace with dpnp.uint16
        "uint32": numpy.uint32,  # TODO: replace with dpnp.uint32
        "uint64": numpy.uint64,  # TODO: replace with dpnp.uint64
    }


def test_dtypes_invalid_kind():
    with pytest.raises(ValueError, match="Unrecognized data type kind"):
        info.dtypes(kind="invalid")


@pytest.mark.skip("due to dpctl-1978")
def test_dtypes_invalid_device():
    with pytest.raises(ValueError, match="Device not understood"):
        info.dtypes(device="gpu")


def test_devices():
    assert info.devices() == get_devices()
