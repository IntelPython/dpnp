import dpctl.tensor as dpt
import numpy
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_equal,
    assert_equal,
    assert_raises_regex,
)

import dpnp

from .helper import (
    get_abs_array,
    get_all_dtypes,
    get_complex_dtypes,
    get_float_dtypes,
    has_support_aspect64,
)
from .third_party.cupy import testing


class TestAsType:
    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.parametrize("res_dtype", get_all_dtypes())
    @pytest.mark.parametrize("arr_dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize(
        "arr",
        [[-2, -1, 0, 1, 2], [[-2, -1], [1, 2]], []],
        ids=["1d", "2d", "empty"],
    )
    def test_basic(self, arr, arr_dtype, res_dtype):
        a = get_abs_array(arr, arr_dtype)
        ia = dpnp.array(a)

        expected = a.astype(res_dtype)
        result = ia.astype(res_dtype)
        assert_allclose(result, expected)

    def test_subok_error(self):
        x = dpnp.ones(4)
        with pytest.raises(NotImplementedError):
            x.astype("i4", subok=False)


class TestAttributes:
    def setup_method(self):
        self.one = dpnp.arange(10)
        self.two = dpnp.arange(20).reshape(4, 5)
        self.three = dpnp.arange(60).reshape(2, 5, 6)

    def test_attributes(self):
        assert_equal(self.one.shape, (10,))
        assert_equal(self.two.shape, (4, 5))
        assert_equal(self.three.shape, (2, 5, 6))

        self.three.shape = (10, 3, 2)
        assert_equal(self.three.shape, (10, 3, 2))
        self.three.shape = (2, 5, 6)

        assert_equal(self.one.strides, (self.one.itemsize / self.one.itemsize,))
        num = self.two.itemsize / self.two.itemsize
        assert_equal(self.two.strides, (5 * num, num))
        num = self.three.itemsize / self.three.itemsize
        assert_equal(self.three.strides, (30 * num, 6 * num, num))

        assert_equal(self.one.ndim, 1)
        assert_equal(self.two.ndim, 2)
        assert_equal(self.three.ndim, 3)

        num = self.two.itemsize
        assert_equal(self.two.size, 20)
        assert_equal(self.two.nbytes, 20 * num)
        assert_equal(self.two.itemsize, self.two.dtype.itemsize)


@pytest.mark.parametrize(
    "arr",
    [
        numpy.array([1]),
        dpnp.array([1]),
        [1],
    ],
    ids=["numpy", "dpnp", "list"],
)
def test_create_from_usm_ndarray_error(arr):
    with pytest.raises(TypeError):
        dpnp.ndarray._create_from_usm_ndarray(arr)


@pytest.mark.parametrize("arr_dtype", get_all_dtypes(no_none=True))
@pytest.mark.parametrize(
    "arr",
    [[-2, -1, 0, 1, 2], [[-2, -1], [1, 2]], []],
    ids=["[-2, -1, 0, 1, 2]", "[[-2, -1], [1, 2]]", "[]"],
)
def test_flatten(arr, arr_dtype):
    a = get_abs_array(arr, arr_dtype)
    ia = dpnp.array(a)
    expected = a.flatten()
    result = ia.flatten()
    assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "shape",
    [(), 0, (0,), (2), (5, 2), (5, 0, 2), (5, 3, 2)],
    ids=["()", "0", "(0,)", "(2)", "(5, 2)", "(5, 0, 2)", "(5, 3, 2)"],
)
@pytest.mark.parametrize("order", ["C", "F"])
def test_flags(shape, order):
    usm_array = dpt.usm_ndarray(shape, order=order)
    a = numpy.ndarray(shape, order=order)
    ia = dpnp.ndarray(shape, order=order)
    assert usm_array.flags == ia.flags
    assert a.flags.c_contiguous == ia.flags.c_contiguous
    assert a.flags.f_contiguous == ia.flags.f_contiguous


@pytest.mark.parametrize(
    "dtype",
    [numpy.complex64, numpy.float32, numpy.int64, numpy.int32, numpy.bool_],
)
@pytest.mark.parametrize("strides", [(1, 4), (4, 1)], ids=["(1, 4)", "(4, 1)"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_flags_strides(dtype, order, strides):
    itemsize = numpy.dtype(dtype).itemsize
    numpy_strides = tuple([el * itemsize for el in strides])
    usm_array = dpt.usm_ndarray(
        (4, 4), dtype=dtype, order=order, strides=strides
    )
    a = numpy.ndarray((4, 4), dtype=dtype, order=order, strides=numpy_strides)
    ia = dpnp.ndarray((4, 4), dtype=dtype, order=order, strides=strides)
    assert usm_array.flags == ia.flags
    assert a.flags.c_contiguous == ia.flags.c_contiguous
    assert a.flags.f_contiguous == ia.flags.f_contiguous


def test_flags_writable():
    a = dpnp.arange(10, dtype="f4")
    a.flags["W"] = False

    a.shape = (5, 2)
    assert not a.flags.writable
    assert not a.T.flags.writable
    assert not a.real.flags.writable
    assert not a[0:3].flags.writable

    a = dpnp.arange(10, dtype="c8")
    a.flags["W"] = False

    assert not a.real.flags.writable
    assert not a.imag.flags.writable


class TestArrayNamespace:
    def test_basic(self):
        a = dpnp.arange(2)
        xp = a.__array_namespace__()
        assert xp is dpnp

    @pytest.mark.parametrize("api_version", [None, "2024.12"])
    def test_api_version(self, api_version):
        a = dpnp.arange(2)
        xp = a.__array_namespace__(api_version=api_version)
        assert xp is dpnp

    @pytest.mark.parametrize(
        "api_version", ["2021.12", "2022.12", "2023.12", "2025.12"]
    )
    def test_unsupported_api_version(self, api_version):
        a = dpnp.arange(2)
        assert_raises_regex(
            ValueError,
            "Only 2024.12 is supported",
            a.__array_namespace__,
            api_version=api_version,
        )

    @pytest.mark.parametrize(
        "api_version",
        [
            2023,
            (2022,),
            [
                2021,
            ],
        ],
    )
    def test_wrong_api_version(self, api_version):
        a = dpnp.arange(2)
        assert_raises_regex(
            TypeError,
            "Expected type str",
            a.__array_namespace__,
            api_version=api_version,
        )


class TestArrayUfunc:
    def test_add(self):
        a = numpy.ones(10)
        b = dpnp.ones(10)
        msg = "An array must be any of supported type"

        with assert_raises_regex(TypeError, msg):
            a + b

        with assert_raises_regex(TypeError, msg):
            b + a

    def test_add_inplace(self):
        a = numpy.ones(10)
        b = dpnp.ones(10)
        with assert_raises_regex(
            TypeError, "operand 'dpnp_array' does not support ufuncs"
        ):
            a += b


class TestItem:
    @pytest.mark.parametrize("args", [2, 7, (1, 2), (2, 0)])
    def test_basic(self, args):
        a = numpy.arange(12).reshape(3, 4)
        ia = dpnp.array(a)

        expected = a.item(args)
        result = ia.item(args)
        assert isinstance(result, int)
        assert expected == result

    def test_0D(self):
        a = numpy.array(5)
        ia = dpnp.array(a)

        expected = a.item()
        result = ia.item()
        assert isinstance(result, int)
        assert expected == result

    def test_error(self):
        ia = dpnp.arange(12).reshape(3, 4)
        with pytest.raises(ValueError):
            ia.item()


class TestUsmNdarrayProtocol:
    def test_basic(self):
        a = dpnp.arange(256, dtype=dpnp.int64)
        usm_a = dpt.asarray(a)

        assert a.sycl_queue == usm_a.sycl_queue
        assert a.usm_type == usm_a.usm_type
        assert a.dtype == usm_a.dtype
        assert usm_a.usm_data.reference_obj is None
        assert (a == usm_a).all()


def test_print_dpnp_int():
    result = repr(dpnp.array([1, 0, 2, -3, -1, 2, 21, -9], dtype="i4"))
    expected = "array([ 1,  0,  2, -3, -1,  2, 21, -9], dtype=int32)"
    assert result == expected

    result = str(dpnp.array([1, 0, 2, -3, -1, 2, 21, -9], dtype="i4"))
    expected = "[ 1  0  2 -3 -1  2 21 -9]"
    assert result == expected
    # int32
    result = repr(dpnp.array([1, -1, 21], dtype=dpnp.int32))
    expected = "array([ 1, -1, 21], dtype=int32)"
    assert result == expected

    result = str(dpnp.array([1, -1, 21], dtype=dpnp.int32))
    expected = "[ 1 -1 21]"
    assert result == expected
    # uint8
    result = repr(dpnp.array([1, 0, 3], dtype=numpy.uint8))
    expected = "array([1, 0, 3], dtype=uint8)"
    assert result == expected

    result = str(dpnp.array([1, 0, 3], dtype=numpy.uint8))
    expected = "[1 0 3]"
    assert result == expected


@pytest.mark.parametrize("dtype", get_float_dtypes())
def test_print_dpnp_float(dtype):
    result = repr(dpnp.array([1, -1, 21], dtype=dtype))
    expected = "array([ 1., -1., 21.])"
    if dtype is dpnp.float32:
        expected = expected[:-1] + ", dtype=float32)"

    result = str(dpnp.array([1, -1, 21], dtype=dtype))
    expected = "[ 1. -1. 21.]"
    assert result == expected


@pytest.mark.parametrize("dtype", get_complex_dtypes())
def test_print_dpnp_complex(dtype):
    result = repr(dpnp.array([1, -1, 21], dtype=dtype))
    expected = "array([ 1.+0.j, -1.+0.j, 21.+0.j])"
    if dtype is dpnp.complex64:
        expected = expected[:-1] + ", dtype=complex64)"
    assert result == expected

    result = str(dpnp.array([1, -1, 21], dtype=dtype))
    expected = "[ 1.+0.j -1.+0.j 21.+0.j]"
    assert result == expected


def test_print_dpnp_boolean():
    result = repr(dpnp.array([1, 0, 3], dtype=bool))
    expected = "array([ True, False,  True])"
    assert result == expected

    result = str(dpnp.array([1, 0, 3], dtype=bool))
    expected = "[ True False  True]"
    assert result == expected


@pytest.mark.parametrize("character", [dpnp.nan, dpnp.inf])
def test_print_dpnp_special_character(character):
    result = repr(dpnp.array([1.0, 0.0, character, 3.0]))
    expected = f"array([ 1.,  0., {character},  3.])"
    if not has_support_aspect64():
        expected = expected[:-1] + ", dtype=float32)"
    assert result == expected

    result = str(dpnp.array([1.0, 0.0, character, 3.0]))
    expected = f"[ 1.  0. {character}  3.]"
    assert result == expected


def test_print_dpnp_1d():
    dtype = dpnp.default_float_type()
    result = repr(dpnp.arange(10000, dtype=dtype))
    expected = "array([0.000e+00, 1.000e+00, 2.000e+00, ..., 9.997e+03, 9.998e+03,\n       9.999e+03], shape=(10000,))"
    if not has_support_aspect64():
        expected = expected[:-1] + ", dtype=float32)"
    assert result == expected

    result = str(dpnp.arange(10000, dtype=dtype))
    expected = (
        "[0.000e+00 1.000e+00 2.000e+00 ... 9.997e+03 9.998e+03 9.999e+03]"
    )
    assert result == expected


def test_print_dpnp_2d():
    dtype = dpnp.default_float_type()
    result = repr(dpnp.array([[1, 2], [3, 4]], dtype=dtype))
    expected = "array([[1., 2.],\n       [3., 4.]])"
    if not has_support_aspect64():
        expected = expected[:-1] + ", dtype=float32)"
    assert result == expected

    result = str(dpnp.array([[1, 2], [3, 4]]))
    expected = "[[1 2]\n [3 4]]"
    assert result == expected


def test_print_dpnp_zero_shape():
    result = repr(dpnp.empty(shape=(0, 0)))
    if has_support_aspect64():
        expected = "array([], shape=(0, 0), dtype=float64)"
    else:
        expected = "array([], shape=(0, 0), dtype=float32)"
    assert result == expected

    result = str(dpnp.empty(shape=(0, 0)))
    expected = "[]"
    assert result == expected


# Numpy will raise an error when converting a.ndim > 0 to a scalar
# TODO: Discuss dpnp behavior according to these future changes
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("func", [bool, float, int, complex])
@pytest.mark.parametrize("shape", [tuple(), (1,), (1, 1), (1, 1, 1)])
@pytest.mark.parametrize(
    "dtype", get_all_dtypes(no_float16=False, no_complex=True)
)
def test_scalar_type_casting(func, shape, dtype):
    a = numpy.full(shape, 5, dtype=dtype)
    ia = dpnp.full(shape, 5, dtype=dtype)
    assert func(a) == func(ia)


# Numpy will raise an error when converting a.ndim > 0 to a scalar
# TODO: Discuss dpnp behavior according to these future changes
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "method", ["__bool__", "__float__", "__int__", "__complex__"]
)
@pytest.mark.parametrize("shape", [tuple(), (1,), (1, 1), (1, 1, 1)])
@pytest.mark.parametrize(
    "dtype", get_all_dtypes(no_float16=False, no_complex=True)
)
def test_scalar_type_casting_by_method(method, shape, dtype):
    a = numpy.full(shape, 4.7, dtype=dtype)
    ia = dpnp.full(shape, 4.7, dtype=dtype)
    assert_allclose(getattr(a, method)(), getattr(ia, method)(), rtol=1e-06)


@pytest.mark.parametrize("shape", [(1,), (1, 1), (1, 1, 1)])
@pytest.mark.parametrize("index_dtype", [dpnp.int32, dpnp.int64])
def test_array_as_index(shape, index_dtype):
    ind_arr = dpnp.ones(shape, dtype=index_dtype)
    a = numpy.arange(ind_arr.size + 1)
    assert a[tuple(ind_arr)] == a[1]


# numpy.ndarray.mT is available since numpy >= 2.0
@testing.with_requires("numpy>=2.0")
@pytest.mark.parametrize(
    "shape",
    [(3, 5), (2, 5, 2), (2, 3, 3, 6)],
    ids=["(3, 5)", "(2, 5, 2)", "(2, 3, 3, 6)"],
)
def test_matrix_transpose(shape):
    a = numpy.arange(numpy.prod(shape)).reshape(shape)
    dp_a = dpnp.array(a)

    expected = a.mT
    result = dp_a.mT

    assert_allclose(result, expected)

    # result is a view of dp_a:
    # changing result, modifies dp_a
    first_elem = (0,) * dp_a.ndim

    result[first_elem] = -1.0
    assert dp_a[first_elem] == -1.0


@testing.with_requires("numpy>=2.0")
def test_matrix_transpose_error():
    # 1D array
    dp_a = dpnp.arange(6)
    with pytest.raises(ValueError):
        dp_a.mT


def test_ravel():
    a = dpnp.ones((2, 2))
    b = a.ravel()
    a[0, 0] = 5
    assert_array_equal(a.ravel(), b)


def test_repeat():
    a = numpy.arange(4).repeat(3)
    ia = dpnp.arange(4).repeat(3)
    assert_array_equal(a, ia)


def test_clip():
    a = numpy.arange(10)
    ia = dpnp.arange(10)
    result = dpnp.clip(ia, 3, 7)
    expected = numpy.clip(a, 3, 7)

    assert_array_equal(result, expected)


def test_rmatmul_dpnp_array():
    a = dpnp.ones(10)
    b = dpnp.ones(10)

    class Dummy(dpnp.ndarray):
        def __init__(self, x):
            self._array_obj = x.get_array()

        def __matmul__(self, other):
            return NotImplemented

    d = Dummy(a)

    result = d @ b
    expected = a @ b
    assert (result == expected).all()


def test_rmatmul_numpy_array():
    a = dpnp.ones(10)
    b = numpy.ones(10)

    with pytest.raises(TypeError):
        b @ a
