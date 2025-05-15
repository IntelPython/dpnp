import numpy
import pytest
from dpctl.tensor._numpy_helper import AxisError
from numpy.testing import assert_array_equal, assert_equal, assert_raises

import dpnp

from .helper import (
    assert_dtype_allclose,
    generate_random_numpy_array,
    get_all_dtypes,
    get_complex_dtypes,
    get_float_dtypes,
)
from .third_party.cupy import testing


class TestArgsort:
    @pytest.mark.parametrize("kind", [None, "stable", "mergesort", "radixsort"])
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_basic(self, kind, dtype):
        a = generate_random_numpy_array(10, dtype)
        ia = dpnp.array(a)

        if dpnp.issubdtype(dtype, dpnp.complexfloating) and kind == "radixsort":
            assert_raises(ValueError, dpnp.argsort, ia, kind=kind)
        else:
            result = dpnp.argsort(ia, kind=kind)
            expected = numpy.argsort(a, kind="stable")
            assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("axis", [None, -2, -1, 0, 1, 2])
    def test_axis(self, axis):
        a = generate_random_numpy_array((3, 4, 3), dtype="i8")
        ia = dpnp.array(a)

        result = dpnp.argsort(ia, axis=axis)
        expected = numpy.argsort(a, axis=axis, kind="stable")
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("axis", [None, -2, -1, 0, 1])
    def test_ndarray_method(self, dtype, axis):
        a = generate_random_numpy_array((6, 2), dtype)
        ia = dpnp.array(a)

        result = ia.argsort(axis=axis)
        expected = a.argsort(axis=axis, kind="stable")
        assert_dtype_allclose(result, expected)

    # this test validates that all different options of kind in dpnp are stable
    @pytest.mark.parametrize("kind", [None, "stable", "mergesort", "radixsort"])
    def test_kind(self, kind):
        a = numpy.repeat(numpy.arange(10), 10)
        ia = dpnp.array(a)

        result = dpnp.argsort(ia, kind=kind)
        expected = numpy.argsort(a, kind="stable")
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("descending", [False, True])
    def test_descending(self, descending):
        a = numpy.repeat(numpy.arange(10), 10)
        ia = dpnp.array(a)

        result = dpnp.argsort(ia, descending=descending)
        if not descending:
            expected = numpy.argsort(a, kind="stable")
        else:
            expected = numpy.flip(numpy.argsort(numpy.flip(a), kind="stable"))
            expected = (a.shape[0] - 1) - expected
        assert_array_equal(result, expected)

        # test ndarray method
        result = ia.argsort(descending=descending)
        if not descending:
            expected = a.argsort(kind="stable")
        else:
            a = numpy.flip(a)
            expected = numpy.flip(a.argsort(kind="stable"))
            expected = (a.shape[0] - 1) - expected
        assert_array_equal(result, expected)

    # `stable` keyword is supported in numpy 2.0 and above
    @testing.with_requires("numpy>=2.0")
    @pytest.mark.parametrize("stable", [None, False, True])
    def test_stable(self, stable):
        a = numpy.repeat(numpy.arange(10), 10)
        ia = dpnp.array(a)

        result = dpnp.argsort(ia, stable=stable)
        expected = numpy.argsort(a, stable=True)
        assert_array_equal(result, expected)

    def test_zero_dim(self):
        a = numpy.array(2.5)
        ia = dpnp.array(a)

        # with default axis=-1
        with pytest.raises(AxisError):
            dpnp.argsort(ia)

        # with axis = None
        result = dpnp.argsort(ia, axis=None)
        expected = numpy.argsort(a, axis=None)
        assert_array_equal(result, expected)


class TestSearchSorted:
    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("dtype", get_float_dtypes(no_float16=False))
    def test_nans_float(self, side, dtype):
        a = numpy.array([0, 1, numpy.nan], dtype=dtype)
        ia = dpnp.array(a)

        result = ia.searchsorted(ia, side=side)
        expected = a.searchsorted(a, side=side)
        assert_equal(result, expected)

        result = dpnp.searchsorted(ia, ia[-1], side=side)
        expected = numpy.searchsorted(a, a[-1], side=side)
        assert_equal(result, expected)

    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    def test_nans_complex(self, side, dtype):
        a = numpy.zeros(9, dtype=dtype)
        a.real += [0, 0, 1, 1, 0, 1, numpy.nan, numpy.nan, numpy.nan]
        a.imag += [0, 1, 0, 1, numpy.nan, numpy.nan, 0, 1, numpy.nan]
        ia = dpnp.array(a)

        result = ia.searchsorted(ia, side=side)
        expected = a.searchsorted(a, side=side)
        assert_equal(result, expected)

    @pytest.mark.parametrize("n", range(3))
    @pytest.mark.parametrize("side", ["left", "right"])
    def test_n_elements(self, n, side):
        a = numpy.ones(n)
        ia = dpnp.array(a)

        v = numpy.array([0, 1, 2])
        iv = dpnp.array(v)

        result = ia.searchsorted(iv, side=side)
        expected = a.searchsorted(v, side=side)
        assert_equal(result, expected)

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_smart_resetting(self, side):
        a = numpy.arange(5)
        ia = dpnp.array(a)

        v = numpy.array([6, 5, 4])
        iv = dpnp.array(v)

        result = ia.searchsorted(iv, side=side)
        expected = a.searchsorted(v, side=side)
        assert_equal(result, expected)

    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
    def test_type_specific(self, side, dtype):
        if dtype == numpy.bool_:
            a = numpy.arange(2, dtype=dtype)
        else:
            a = numpy.arange(0, 5, dtype=dtype)
        ia = dpnp.array(a)

        result = ia.searchsorted(ia, side=side)
        expected = a.searchsorted(a, side=side)
        assert_equal(result, expected)

        e = numpy.ndarray(shape=0, buffer=b"", dtype=dtype)
        dp_e = dpnp.array(e)

        result = dp_e.searchsorted(ia, side=side)
        expected = e.searchsorted(a, side=side)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_sorter(self, dtype):
        a = numpy.random.rand(300).astype(dtype)
        s = a.argsort()
        k = numpy.linspace(0, 1, 20, dtype=dtype)

        ia = dpnp.array(a)
        dp_s = dpnp.array(s)
        dp_k = dpnp.array(k)

        result = ia.searchsorted(dp_k, sorter=dp_s)
        expected = a.searchsorted(k, sorter=s)
        assert_equal(result, expected)

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_sorter_with_side(self, side):
        a = numpy.array([0, 1, 2, 3, 5] * 20)
        s = a.argsort()
        k = [0, 1, 2, 3, 5]

        ia = dpnp.array(a)
        dp_s = dpnp.array(s)
        dp_k = dpnp.array(k)

        result = ia.searchsorted(dp_k, side=side, sorter=dp_s)
        expected = a.searchsorted(k, side=side, sorter=s)
        assert_equal(result, expected)

    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
    def test_sorter_type_specific(self, side, dtype):
        if dtype == numpy.bool_:
            a = numpy.array([1, 0], dtype=dtype)
            # a sorter array to be of a type that is different
            # from np.intp in all platforms
            s = numpy.array([1, 0], dtype=numpy.int16)
        else:
            a = numpy.arange(0, 5, dtype=dtype)
            # a sorter array to be of a type that is different
            # from np.intp in all platforms
            s = numpy.array([4, 2, 3, 0, 1], dtype=numpy.int16)

        ia = dpnp.array(a)
        dp_s = dpnp.array(s)

        result = ia.searchsorted(ia, side, dp_s)
        expected = a.searchsorted(a, side, s)
        assert_equal(result, expected)

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_sorter_non_contiguous(self, side):
        a = numpy.array([3, 4, 1, 2, 0])
        srt = numpy.empty((10,), dtype=numpy.intp)
        srt[1::2] = -1
        srt[::2] = [4, 2, 3, 0, 1]
        s = srt[::2]

        ia = dpnp.array(a)
        dp_s = dpnp.array(s)

        result = ia.searchsorted(ia, side=side, sorter=dp_s)
        expected = a.searchsorted(a, side=side, sorter=s)
        assert_equal(result, expected)

    def test_invalid_sorter(self):
        for xp in [dpnp, numpy]:
            a = xp.array([5, 2, 1, 3, 4])

            assert_raises(
                TypeError,
                ValueError,
                xp.searchsorted,
                a,
                0,
                sorter=xp.array([1.1]),
            )
            assert_raises(
                ValueError, xp.searchsorted, a, 0, sorter=xp.array([1, 2, 3, 4])
            )
            assert_raises(
                ValueError,
                xp.searchsorted,
                a,
                0,
                sorter=xp.array([1, 2, 3, 4, 5, 6]),
            )

    def test_v_scalar(self):
        v = 0
        a = numpy.array([-8, -5, -1, 3, 6, 10])
        ia = dpnp.array(a)

        result = dpnp.searchsorted(ia, v)
        expected = numpy.searchsorted(a, v)
        assert_equal(result, expected)


class TestSort:
    @pytest.mark.parametrize("kind", [None, "stable", "mergesort", "radixsort"])
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_basic(self, kind, dtype):
        a = generate_random_numpy_array(10, dtype)
        ia = dpnp.array(a)

        if dpnp.issubdtype(dtype, dpnp.complexfloating) and kind == "radixsort":
            assert_raises(ValueError, dpnp.argsort, ia, kind=kind)
        else:
            result = dpnp.sort(ia, kind=kind)
            expected = numpy.sort(a)
            assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("axis", [None, -2, -1, 0, 1, 2])
    def test_axis(self, axis):
        a = generate_random_numpy_array((3, 4, 3), dtype="i8")
        ia = dpnp.array(a)

        result = dpnp.sort(ia, axis=axis)
        expected = numpy.sort(a, axis=axis, kind="stable")
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("axis", [-2, -1, 0, 1])
    def test_ndarray_method(self, dtype, axis):
        a = generate_random_numpy_array((6, 2), dtype)
        ia = dpnp.array(a)

        ia.sort(axis=axis)
        a.sort(axis=axis)
        assert_dtype_allclose(ia, a)

    # this test validates that all different options of kind in dpnp are stable
    @pytest.mark.parametrize("kind", [None, "stable", "mergesort", "radixsort"])
    def test_kind(self, kind):
        a = numpy.repeat(numpy.arange(10), 10)
        ia = dpnp.array(a)

        result = dpnp.sort(ia, kind=kind)
        expected = numpy.sort(a, kind="stable")
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("descending", [False, True])
    def test_descending(self, descending):
        a = numpy.repeat(numpy.arange(10), 10)
        ia = dpnp.array(a)

        result = dpnp.sort(ia, descending=descending)
        expected = numpy.sort(a, kind="stable")
        if descending:
            expected = numpy.flip(expected)
        assert_array_equal(result, expected)

        # test ndarray method
        ia.sort(descending=descending)
        a.sort(kind="stable")
        if descending:
            a = numpy.flip(a)
        assert_array_equal(ia, a)

    # `stable` keyword is supported in numpy 2.0 and above
    @testing.with_requires("numpy>=2.0")
    @pytest.mark.parametrize("stable", [None, False, True])
    def test_stable(self, stable):
        a = numpy.repeat(numpy.arange(10), 10)
        ia = dpnp.array(a)

        result = dpnp.sort(ia, stable=stable)
        expected = numpy.sort(a, stable=True)
        assert_array_equal(result, expected)

    def test_ndarray_axis_none(self):
        a = numpy.random.uniform(-10, 10, 12)
        ia = dpnp.array(a).reshape(6, 2)
        with pytest.raises(TypeError):
            ia.sort(axis=None)

    def test_zero_dim(self):
        a = numpy.array(2.5)
        ia = dpnp.array(a)

        # with default axis=-1
        with pytest.raises(AxisError):
            dpnp.sort(ia)

        # with axis = None
        result = dpnp.sort(ia, axis=None)
        expected = numpy.sort(a, axis=None)
        assert_array_equal(result, expected)

    def test_error(self):
        ia = dpnp.arange(10)

        # quicksort is currently not supported
        with pytest.raises(ValueError):
            dpnp.sort(ia, kind="quicksort")

        with pytest.raises(NotImplementedError):
            dpnp.sort(ia, order=["age"])

        # both kind and stable are given
        with pytest.raises(ValueError):
            dpnp.sort(ia, kind="mergesort", stable=True)

        # stable is not valid
        with pytest.raises(ValueError):
            dpnp.sort(ia, stable="invalid")


class TestSortComplex:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_complex=True) + [numpy.int8, numpy.int16]
    )
    def test_real(self, dtype):
        # sort_complex() type casting for real input types
        a = numpy.array([5, 3, 6, 2, 1], dtype=dtype)
        ia = dpnp.array(a)

        result = dpnp.sort_complex(ia)
        expected = numpy.sort_complex(a)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    def test_complex(self, dtype):
        # sort_complex() handling of complex input
        a = numpy.array([2 + 3j, 1 - 2j, 1 - 3j, 2 + 1j], dtype=dtype)
        ia = dpnp.array(a)

        result = dpnp.sort_complex(ia)
        expected = numpy.sort_complex(a)
        assert_equal(result, expected)


@pytest.mark.parametrize("kth", [0, 1])
@pytest.mark.parametrize(
    "dtype",
    get_all_dtypes(
        no_none=True, no_unsigned=True, xfail_dtypes=[dpnp.int8, dpnp.int16]
    ),
)
@pytest.mark.parametrize(
    "array",
    [
        [3, 4, 2, 1],
        [[1, 0], [3, 0]],
        [[3, 2], [1, 6]],
        [[4, 2, 3], [3, 4, 1]],
        [[[1, -3], [3, 0]], [[5, 2], [0, 1]], [[1, 0], [0, 1]]],
        [
            [[[8, 2], [3, 0]], [[5, 2], [0, 1]]],
            [[[1, 3], [3, 1]], [[5, 2], [0, 1]]],
        ],
    ],
    ids=[
        "[3, 4, 2, 1]",
        "[[1, 0], [3, 0]]",
        "[[3, 2], [1, 6]]",
        "[[4, 2, 3], [3, 4, 1]]",
        "[[[1, -3], [3, 0]], [[5, 2], [0, 1]], [[1, 0], [0, 1]]]",
        "[[[[8, 2], [3, 0]], [[5, 2], [0, 1]]], [[[1, 3], [3, 1]], [[5, 2], [0, 1]]]]",
    ],
)
def test_partition(array, dtype, kth):
    a = dpnp.array(array, dtype)
    p = dpnp.partition(a, kth)

    assert (p[..., 0:kth] <= p[..., kth : kth + 1]).all()
    assert (p[..., kth : kth + 1] <= p[..., kth + 1 :]).all()
