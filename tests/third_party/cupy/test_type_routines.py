import unittest

import numpy
import pytest

import dpnp as cupy
from tests.helper import has_support_aspect64
from tests.third_party.cupy import testing


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
        if self.obj_type == "scalar":
            pytest.skip("to be aligned with NEP-50")

        from_obj = _generate_type_routines_input(xp, from_dtype, self.obj_type)

        ret = xp.can_cast(from_obj, to_dtype)
        assert isinstance(ret, bool)
        return ret


@pytest.mark.skip("dpnp.common_type() is not implemented yet")
class TestCommonType(unittest.TestCase):
    @testing.numpy_cupy_equal()
    def test_common_type_empty(self, xp):
        ret = xp.common_type()
        assert type(ret) == type
        return ret

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_equal()
    def test_common_type_single_argument(self, xp, dtype):
        array = _generate_type_routines_input(xp, dtype, "array")
        ret = xp.common_type(array)
        assert type(ret) == type
        return ret

    @testing.for_all_dtypes_combination(
        names=("dtype1", "dtype2"), no_bool=True
    )
    @testing.numpy_cupy_equal()
    def test_common_type_two_arguments(self, xp, dtype1, dtype2):
        array1 = _generate_type_routines_input(xp, dtype1, "array")
        array2 = _generate_type_routines_input(xp, dtype2, "array")
        ret = xp.common_type(array1, array2)
        assert type(ret) == type
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
        if "scalar" in {self.obj_type1, self.obj_type2}:
            pytest.skip("to be aligned with NEP-50")

        input1 = _generate_type_routines_input(xp, dtype1, self.obj_type1)

        input2 = _generate_type_routines_input(xp, dtype2, self.obj_type2)

        flag1 = isinstance(input1, (numpy.ndarray, cupy.ndarray))
        flag2 = isinstance(input2, (numpy.ndarray, cupy.ndarray))
        dt1 = cupy.dtype(input1) if not flag1 else None
        dt2 = cupy.dtype(input2) if not flag2 else None
        # dpnp takes into account device capabilities only if one of the
        # inputs is an array, for such a case, if the other dtype is not
        # supported by device, dpnp raise ValueError. So, we skip the test.
        if flag1 or flag2:
            if (
                dt1 in [cupy.float64, cupy.complex128]
                or dt2 in [cupy.float64, cupy.complex128]
            ) and not has_support_aspect64():
                pytest.skip("No fp64 support by device.")

        ret = xp.result_type(input1, input2)

        # dpnp takes into account device capabilities if one of the inputs
        # is an array, for such a case, we have to modify the results for
        # NumPy to align it with device capabilities.
        if (flag1 or flag2) and xp == numpy and not has_support_aspect64():
            ret = numpy.dtype(numpy.float32) if ret == numpy.float64 else ret
            ret = (
                numpy.dtype(numpy.complex64) if ret == numpy.complex128 else ret
            )

        assert isinstance(ret, numpy.dtype)
        return ret
