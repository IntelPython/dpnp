import unittest

import numpy
import pytest

import dpnp as cupy
from dpnp.tests.helper import has_support_aspect16, has_support_aspect64
from dpnp.tests.third_party.cupy import testing


def _generate_type_routines_input(xp, dtype, obj_type):
    dtype = numpy.dtype(dtype)
    if obj_type == "dtype":
        return dtype
    if obj_type == "specifier":
        return str(dtype)
    if obj_type == "scalar":
        return dtype.type(3)
    if obj_type == "array":
        return xp.zeros(3, dtype=dtype)
    if obj_type == "primitive":
        return type(dtype.type(3).tolist())
    assert False


@testing.parameterize(
    *testing.product(
        {
            "obj_type": ["dtype", "specifier", "scalar", "array", "primitive"],
        }
    )
)
class TestCanCast(unittest.TestCase):

    @testing.for_all_dtypes_combination(names=("from_dtype", "to_dtype"))
    @testing.numpy_cupy_equal()
    def test_can_cast(self, xp, from_dtype, to_dtype):
        if (
            self.obj_type == "scalar"
            and numpy.lib.NumpyVersion(numpy.__version__) < "2.0.0"
        ):
            pytest.skip("to be aligned with NEP-50")

        from_obj = _generate_type_routines_input(xp, from_dtype, self.obj_type)
        ret = xp.can_cast(from_obj, to_dtype)
        assert isinstance(ret, bool)
        return ret


class TestCommonType(unittest.TestCase):

    @testing.numpy_cupy_equal()
    def test_common_type_empty(self, xp):
        ret = xp.common_type()
        assert type(ret) is type
        # NumPy always returns float16 for empty input,
        # but dpnp returns float32 if the device does not support
        # 16-bit precision floating point operations
        if xp is numpy and not has_support_aspect16():
            return xp.float32
        return ret

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_equal()
    def test_common_type_single_argument(self, xp, dtype):
        array = _generate_type_routines_input(xp, dtype, "array")
        ret = xp.common_type(array)
        assert type(ret) is type
        # NumPy promotes integer types to float64,
        # but dpnp may return float32 if the device does not support
        # 64-bit precision floating point operations.
        if xp is numpy and not has_support_aspect64():
            return xp.float32
        return ret

    @testing.for_all_dtypes_combination(
        names=("dtype1", "dtype2"), no_bool=True
    )
    @testing.numpy_cupy_equal()
    def test_common_type_two_arguments(self, xp, dtype1, dtype2):
        array1 = _generate_type_routines_input(xp, dtype1, "array")
        array2 = _generate_type_routines_input(xp, dtype2, "array")
        ret = xp.common_type(array1, array2)
        assert type(ret) is type
        if xp is numpy and not has_support_aspect64():
            return xp.float32
        return ret

    @testing.for_all_dtypes()
    def test_common_type_bool(self, dtype):
        for xp in (numpy, cupy):
            array1 = _generate_type_routines_input(xp, dtype, "array")
            array2 = _generate_type_routines_input(xp, "bool_", "array")
            with pytest.raises(TypeError):
                xp.common_type(array1, array2)


@testing.parameterize(
    *testing.product(
        {
            "obj_type1": ["dtype", "specifier", "scalar", "array", "primitive"],
            "obj_type2": ["dtype", "specifier", "scalar", "array", "primitive"],
        }
    )
)
class TestResultType(unittest.TestCase):

    @testing.for_all_dtypes_combination(names=("dtype1", "dtype2"))
    @testing.numpy_cupy_equal()
    def test_result_type(self, xp, dtype1, dtype2):
        if (
            "scalar" in {self.obj_type1, self.obj_type2}
            and numpy.lib.NumpyVersion(numpy.__version__) < "2.0.0"
        ):
            pytest.skip("to be aligned with NEP-50")

        input1 = _generate_type_routines_input(xp, dtype1, self.obj_type1)
        input2 = _generate_type_routines_input(xp, dtype2, self.obj_type2)

        # dpnp.result_type takes into account device capabilities, when one of
        # the inputs is an array. If dtype is `float32` and the object is
        # primitive, the final dtype is `float` which needs a device with
        # double precision support. So we have to skip the test for such a case
        # on a device that does not support fp64
        flag1 = self.obj_type1 == "array" or self.obj_type2 == "array"
        flag2 = (self.obj_type1 == "primitive" and input1 == float) or (
            self.obj_type2 == "primitive" and input2 == float
        )
        if flag1 and flag2 and not has_support_aspect64():
            pytest.skip("No fp64 support by device.")

        ret = xp.result_type(input1, input2)

        # dpnp.result_type takes into account device capabilities, when one of the inputs
        # is an array.
        # So, we have to modify the results for NumPy to align it with
        # device capabilities.
        flag1 = isinstance(input1, numpy.ndarray)
        flag2 = isinstance(input2, numpy.ndarray)
        if (flag1 or flag2) and not has_support_aspect64():
            if ret == numpy.float64:
                ret = numpy.dtype(numpy.float32)
            elif ret == numpy.complex128:
                ret = numpy.dtype(numpy.complex64)

        assert isinstance(ret, numpy.dtype)
        return ret
