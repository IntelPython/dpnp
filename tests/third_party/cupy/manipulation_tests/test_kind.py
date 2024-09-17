import unittest

import numpy
import pytest

import dpnp as cupy
from tests.helper import has_support_aspect64
from tests.third_party.cupy import testing


class TestKind(unittest.TestCase):
    @testing.for_orders("CFAK")
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_asarray_chkfinite(self, xp, dtype, order):
        a = [0, 4, 0, 5]
        return xp.asarray_chkfinite(a, dtype=dtype, order=order)

    @testing.for_orders("CFAK")
    @testing.for_all_dtypes(no_bool=True)
    def test_asarray_chkfinite_non_finite_vals(self, dtype, order):
        a = [-numpy.inf, 0.0, numpy.inf, numpy.nan]
        for xp in (numpy, cupy):
            if xp.issubdtype(dtype, xp.integer):
                error = OverflowError
            else:
                error = ValueError
            with pytest.raises(error):
                xp.asarray_chkfinite(a, dtype=dtype, order=order)

    @testing.with_requires("numpy<2.0")
    @testing.for_all_dtypes()
    def test_asfarray(self, dtype):
        a = cupy.asarray([1, 2, 3])
        a_gpu = cupy.asfarray(a, dtype)
        a_cpu = numpy.asfarray(a, dtype)
        if (
            has_support_aspect64()
            or cupy.issubdtype(dtype, cupy.complexfloating)
            or cupy.issubdtype(dtype, cupy.floating)
        ):
            assert a_cpu.dtype == a_gpu.dtype
        else:
            assert a_cpu.dtype == cupy.float64
            assert a_gpu.dtype == cupy.float32

    @testing.for_all_dtypes()
    def test_asfortranarray1(self, dtype):
        def func(xp):
            x = xp.zeros((2, 3), dtype=dtype)
            ret = xp.asfortranarray(x)
            assert x.flags.c_contiguous
            assert ret.flags.f_contiguous

        assert func(numpy) == func(cupy)

    @testing.for_all_dtypes()
    def test_asfortranarray2(self, dtype):
        def func(xp):
            x = xp.zeros((2, 3, 4), dtype=dtype)
            ret = xp.asfortranarray(x)
            assert x.flags.c_contiguous
            assert ret.flags.f_contiguous

        assert func(numpy) == func(cupy)

    @testing.for_all_dtypes()
    def test_asfortranarray3(self, dtype):
        def func(xp):
            x = xp.zeros((2, 3, 4), dtype=dtype)
            ret = xp.asfortranarray(xp.asfortranarray(x))
            assert x.flags.c_contiguous
            assert ret.flags.f_contiguous

        assert func(numpy) == func(cupy)

    @testing.for_all_dtypes()
    def test_asfortranarray4(self, dtype):
        def func(xp):
            x = xp.zeros((2, 3), dtype=dtype)
            x = xp.transpose(x, (1, 0))
            ret = xp.asfortranarray(x)
            assert ret.flags.f_contiguous

        assert func(numpy) == func(cupy)

    @testing.for_all_dtypes()
    def test_asfortranarray5(self, dtype):
        def func(xp):
            x = testing.shaped_arange((2, 3), xp, dtype)
            ret = xp.asfortranarray(x)
            assert x.flags.c_contiguous
            assert ret.flags.f_contiguous

        assert func(numpy) == func(cupy)

    @testing.for_all_dtypes()
    def test_require_flag_check(self, dtype):
        possible_flags = [["C_CONTIGUOUS"], ["F_CONTIGUOUS"]]
        x = cupy.zeros((2, 3, 4), dtype=dtype)
        for flags in possible_flags:
            arr = cupy.require(x, dtype, flags)
            for parameter in flags:
                assert arr.flags[parameter]
                assert arr.dtype == dtype

    @pytest.mark.skip("dpnp.require() does not support requirement ['O']")
    @testing.for_all_dtypes()
    def test_require_owndata(self, dtype):
        x = cupy.zeros((2, 3, 4), dtype=dtype)
        arr = x.view()
        arr = cupy.require(arr, dtype, ["O"])
        assert arr.flags["OWNDATA"]

    @testing.for_all_dtypes()
    def test_require_C_and_F_flags(self, dtype):
        x = cupy.zeros((2, 3, 4), dtype=dtype)
        with pytest.raises(ValueError):
            cupy.require(x, dtype, ["C", "F"])

    @testing.for_all_dtypes()
    def test_require_incorrect_requirments(self, dtype):
        x = cupy.zeros((2, 3, 4), dtype=dtype)
        with pytest.raises(ValueError):
            cupy.require(x, dtype, ["O"])

    @testing.for_all_dtypes()
    def test_require_incorrect_dtype(self, dtype):
        x = cupy.zeros((2, 3, 4), dtype=dtype)
        with pytest.raises((ValueError, TypeError)):
            cupy.require(x, "random", "C")

    @testing.for_all_dtypes()
    def test_require_empty_requirements(self, dtype):
        x = cupy.zeros((2, 3, 4), dtype=dtype)
        x = cupy.require(x, dtype, [])
        assert x.flags["C_CONTIGUOUS"]
