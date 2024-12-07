import itertools
import warnings

import numpy
import pytest

if numpy.lib.NumpyVersion(numpy.__version__) >= "2.0.0b1":
    from numpy.exceptions import ComplexWarning
else:
    from numpy import ComplexWarning

import dpnp as cupy
from dpnp.tests.helper import has_support_aspect16, has_support_aspect64
from dpnp.tests.third_party.cupy import testing

float_types = [numpy.float16, numpy.float32, numpy.float64]
complex_types = [numpy.complex64, numpy.complex128]
signed_int_types = [numpy.int8, numpy.int16, numpy.int32, numpy.int64]
unsigned_int_types = [numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]
int_types = signed_int_types + unsigned_int_types
all_types = [numpy.bool_] + float_types + int_types + complex_types
negative_types = [numpy.bool_] + float_types + signed_int_types + complex_types
negative_types_wo_fp16 = (
    [numpy.bool_]
    + [numpy.float32, numpy.float64]
    + [numpy.int16, numpy.int32, numpy.int64]
    + complex_types
)
negative_no_complex_types = [numpy.bool_] + float_types + signed_int_types
no_complex_types = [numpy.bool_] + float_types + int_types


@testing.parameterize(
    *(
        testing.product(
            {
                "nargs": [1],
                "name": ["reciprocal", "conj", "conjugate", "angle"],
            }
        )
        + testing.product(
            {
                "nargs": [2],
                "name": [
                    "add",
                    "multiply",
                    "divide",
                    "power",
                    "subtract",
                    "true_divide",
                    "floor_divide",
                    "float_power",
                    "fmod",
                    "remainder",
                ],
            }
        )
    )
)
class TestArithmeticRaisesWithNumpyInput:

    def test_raises_with_numpy_input(self):
        nargs = self.nargs
        name = self.name

        # Check TypeError is raised if numpy.ndarray is given as input
        func = getattr(cupy, name)
        for input_xp_list in itertools.product(*[[numpy, cupy]] * nargs):
            if all(xp is cupy for xp in input_xp_list):
                # We don't test all-cupy-array inputs here
                continue
            arys = [xp.array([2, -3]) for xp in input_xp_list]
            with pytest.raises(TypeError):
                func(*arys)


@testing.parameterize(
    *(
        testing.product(
            {
                "arg1": (
                    [
                        testing.shaped_arange((2, 3), numpy, dtype=d)
                        for d in all_types
                    ]
                    # scalar input is not supported
                    # + [0, 0.0j, 0j, 2, 2.0, 2j, True, False]
                ),
                "name": ["conj", "conjugate", "real", "imag"],
            }
        )
        + testing.product(
            {
                "arg1": (
                    [
                        testing.shaped_arange((2, 3), numpy, dtype=d)
                        for d in all_types
                    ]
                    # scalar input is not supported
                    # + [0, 0.0j, 0j, 2, 2.0, 2j, True, False]
                ),
                "deg": [True, False],
                "name": ["angle"],
            }
        )
        + testing.product(
            {
                "arg1": (
                    [
                        numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                        for d in negative_types_wo_fp16
                    ]
                    # scalar input is not supported
                    # + [0, 0.0j, 0j, 2, 2.0, 2j, -2, -2.0, -2j, True, False]
                ),
                "deg": [True, False],
                "name": ["angle"],
            }
        )
        + testing.product(
            {
                "arg1": (
                    [
                        testing.shaped_arange((2, 3), numpy, dtype=d) + 1
                        for d in all_types
                    ]
                    # scalar input is not supported
                    # + [2, 2.0, 2j, True]
                ),
                "name": ["reciprocal"],
            }
        )
    )
)
class TestArithmeticUnary:

    @testing.numpy_cupy_allclose(atol=1e-5, type_check=has_support_aspect64())
    def test_unary(self, xp):
        arg1 = self.arg1
        if isinstance(arg1, numpy.ndarray):
            arg1 = xp.asarray(arg1)

        if self.name in ("reciprocal") and xp is numpy:
            # In NumPy, for integer arguments with absolute value larger than 1 the result is always zero.
            # We need to convert the input data type to float then compare the output with DPNP.
            if numpy.issubdtype(arg1.dtype, numpy.integer):
                if arg1.dtype.char in "bB":  # int8
                    np_dtype = numpy.float16
                elif arg1.dtype.char in "hH":  # int16
                    np_dtype = numpy.float32
                else:  # int32, int64
                    if has_support_aspect64():
                        np_dtype = numpy.float64
                    else:
                        np_dtype = numpy.float32
                arg1 = xp.asarray(arg1, dtype=np_dtype)

        if self.name in {"angle"}:
            y = getattr(xp, self.name)(arg1, self.deg)
            if isinstance(arg1, cupy.ndarray):
                if arg1.dtype == cupy.bool and has_support_aspect64():
                    # In NumPy, for boolean input the output data type is always default floating data type.
                    # while data type of output in DPNP is determined by Type Promotion Rules.
                    y = y.astype(cupy.float64)
                elif arg1.dtype.char in "bBe" and has_support_aspect16():
                    # In NumPy, for int8, uint8 and float16 inputs the output data type is always float16.
                    # while data type of output in DPNP is float32.
                    y = y.astype(cupy.float16)
        else:
            y = getattr(xp, self.name)(arg1)

        return y


@testing.parameterize(
    *testing.product(
        {
            "shape": [(3, 2), (), (3, 0, 2)],
        }
    )
)
class TestComplex:

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_real_ndarray_nocomplex(self, xp, dtype):
        x = testing.shaped_arange(self.shape, xp, dtype=dtype)
        real = x.real
        assert real is x  # real returns self
        return real

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_real_nocomplex(self, xp, dtype):
        x = testing.shaped_arange(self.shape, xp, dtype=dtype)
        real = xp.real(x)
        assert real is x  # real returns self
        return real

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_imag_ndarray_nocomplex(self, xp, dtype):
        x = testing.shaped_arange(self.shape, xp, dtype=dtype)
        imag = x.imag
        return imag

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_imag_nocomplex(self, xp, dtype):
        x = testing.shaped_arange(self.shape, xp, dtype=dtype)
        imag = xp.imag(x)
        return imag

    @pytest.mark.skip("'dpnp_array' object has no attribute 'base' yet")
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_real_ndarray_complex(self, xp, dtype):
        x = testing.shaped_arange(self.shape, xp, dtype=dtype)
        x_ = x.copy()
        real = x_.real
        # real returns a view
        assert real.base is x_
        x_ += 1 + 1j
        testing.assert_array_equal(real, x.real + 1)
        return real

    @pytest.mark.skip("'dpnp_array' object has no attribute 'base' yet")
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_real_complex(self, xp, dtype):
        x = testing.shaped_arange(self.shape, xp, dtype=dtype)
        x_ = x.copy()
        real = xp.real(x_)
        # real returns a view
        assert real.base is x_
        x_ += 1 + 1j
        testing.assert_array_equal(real, x.real + 1)
        return real

    @pytest.mark.skip("'dpnp_array' object has no attribute 'base' yet")
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_imag_ndarray_complex(self, xp, dtype):
        x = testing.shaped_arange(self.shape, xp, dtype=dtype)
        x_ = x.copy()
        imag = x_.imag
        # imag returns a view
        assert imag.base is x_
        x_ += 1 + 1j
        testing.assert_array_equal(imag, x.imag + 1)
        return imag

    @pytest.mark.skip("'dpnp_array' object has no attribute 'base' yet")
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_imag_complex(self, xp, dtype):
        x = testing.shaped_arange(self.shape, xp, dtype=dtype)
        x_ = x.copy()
        imag = xp.imag(x_)
        # imag returns a view
        assert imag.base is x_
        x_ += 1 + 1j
        testing.assert_array_equal(imag, x.imag + 1)
        return imag


class ArithmeticBinaryBase:

    @testing.numpy_cupy_allclose(rtol=1e-4, type_check=has_support_aspect64())
    def check_binary(self, xp):
        arg1 = self.arg1
        arg2 = self.arg2
        np1 = numpy.asarray(arg1)
        np2 = numpy.asarray(arg2)
        dtype1 = np1.dtype
        dtype2 = np2.dtype

        if xp.isscalar(arg1) and xp.isscalar(arg2):
            pytest.skip("both scalar inputs is not supported")

        if self.name in ("true_divide", "floor_divide", "fmod", "remainder"):
            if dtype1.kind in "u" and xp.isscalar(arg2) and arg2 < 0:
                # TODO: Fix this: array(3, dtype=uint) / -2
                #     numpy => -1.5
                #     cupy => 0.01181102
                pytest.skip("due to dpctl gh-1711")
            if dtype2.kind in "u" and xp.isscalar(arg1) and arg1 < 0:
                # TODO: Fix this: 2 / array(3, dtype=uint)
                #     numpy => -0.666667
                #     cupy => 84.666667
                pytest.skip("due to dpctl gh-1711")

        if self.name == "power" or self.name == "float_power":
            # TODO(niboshi): Fix this: power(0, 1j)
            #     numpy => 1+0j
            #     cupy => 0j
            if dtype2 in complex_types and (np1 == 0).any():
                return xp.array(True)
            # TODO: Fix this: power(0j, 0)
            #     numpy => 1+0j
            #     cupy => nan+nanj
            elif dtype1 in complex_types and (np2 == 0).any():
                return xp.array(True)

        if isinstance(arg1, numpy.ndarray):
            arg1 = xp.asarray(arg1)
        if isinstance(arg2, numpy.ndarray):
            arg2 = xp.asarray(arg2)

        # Subtraction between booleans is not allowed.
        if (
            self.name == "subtract"
            and dtype1 == numpy.bool_
            and dtype2 == numpy.bool_
        ):
            return xp.array(True)

        func = getattr(xp, self.name)
        with numpy.errstate(divide="ignore"):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                if self.use_dtype:
                    if (
                        xp is numpy
                        and self.name == "float_power"
                        and self.dtype == numpy.float32
                    ):
                        # numpy.float_power does not have a loop for float32,
                        # while dpnp.float_power does
                        self.dtype = numpy.float64
                    y = func(arg1, arg2, dtype=self.dtype)
                else:
                    y = func(arg1, arg2)

        # TODO(niboshi): Fix this. If rhs is a Python complex,
        #    numpy returns complex64
        #    cupy returns complex128
        if (
            xp is cupy
            and isinstance(arg2, complex)
            and self.name != "float_power"
        ):
            if dtype1 in (numpy.float16, numpy.float32):
                y = y.astype(numpy.complex64)

        if xp is cupy and self.name == "float_power" and has_support_aspect64():
            # numpy.float_power does not have a loop for float32 and complex64,
            # and will upcast input array to float64 or complex128,
            # while dpnp has to support float32 and complex64 to compatibility
            # with devices without fp64 support
            if y.dtype == cupy.float32:
                y = y.astype(cupy.float64)
            elif y.dtype == cupy.complex64:
                y = y.astype(cupy.complex128)

        # NumPy returns different values (nan/inf) on division by zero
        # depending on the architecture.
        # As it is not possible for CuPy to replicate this behavior, we ignore
        # the difference here.
        if self.name in ("floor_divide", "remainder"):
            if y.dtype in (float_types + complex_types) and (np2 == 0).any():
                y = xp.asarray(y)
                y[y == numpy.inf] = numpy.nan
                y[y == -numpy.inf] = numpy.nan

        return y


@testing.parameterize(
    *(
        testing.product(
            {
                # TODO(unno): boolean subtract causes DeprecationWarning in numpy>=1.13
                "arg1": [
                    testing.shaped_arange((2, 3), numpy, dtype=d)
                    for d in all_types
                ]
                + [0, 0.0, 0j, 2, 2.0, 2j, True, False],
                "arg2": [
                    testing.shaped_reverse_arange((2, 3), numpy, dtype=d)
                    for d in all_types
                ]
                + [0, 0.0, 0j, 2, 2.0, 2j, True, False],
                "name": ["add", "multiply", "power", "subtract", "float_power"],
            }
        )
        + testing.product(
            {
                "arg1": [
                    numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                    for d in negative_types
                ]
                + [0, 0.0, 0j, 2, 2.0, 2j, -2, -2.0, -2j, True, False],
                "arg2": [
                    numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                    for d in negative_types
                ]
                + [0, 0.0, 0j, 2, 2.0, 2j, -2, -2.0, -2j, True, False],
                "name": ["divide", "true_divide", "subtract"],
            }
        )
    )
)
class TestArithmeticBinary(ArithmeticBinaryBase):

    def test_binary(self):
        self.use_dtype = False
        self.check_binary()


@testing.parameterize(
    *(
        testing.product(
            {
                "arg1": [
                    numpy.array([3, 2, 1, 1, 2, 3], dtype=d)
                    for d in unsigned_int_types
                ]
                + [0, 0.0, 2, 2.0, -2, -2.0, True, False],
                "arg2": [
                    numpy.array([3, 2, 1, 1, 2, 3], dtype=d)
                    for d in unsigned_int_types
                ]
                + [0, 0.0, 2, 2.0, -2, -2.0, True, False],
                "name": ["true_divide"],
                "dtype": [cupy.default_float_type()],
                "use_dtype": [True, False],
            }
        )
        + testing.product(
            {
                "arg1": [
                    numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                    for d in signed_int_types
                ]
                + [0, 0.0, 2, 2.0, -2, -2.0, True, False],
                "arg2": [
                    numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                    for d in signed_int_types
                ]
                + [0, 0.0, 2, 2.0, -2, -2.0, True, False],
                "name": ["true_divide"],
                "dtype": [cupy.default_float_type()],
                "use_dtype": [True, False],
            }
        )
        + testing.product(
            {
                "arg1": [
                    numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                    for d in float_types
                ]
                + [0.0, 2.0, -2.0],
                "arg2": [
                    numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                    for d in float_types
                ]
                + [0.0, 2.0, -2.0],
                "name": ["power", "true_divide", "subtract", "float_power"],
                "dtype": [cupy.default_float_type()],
                "use_dtype": [True, False],
            }
        )
        + testing.product(
            {
                "arg1": [
                    numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                    for d in negative_no_complex_types
                ]
                + [0, 0.0, 2, 2.0, -2, -2.0, True, False],
                "arg2": [
                    numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                    for d in negative_no_complex_types
                ]
                + [0, 0.0, 2, 2.0, -2, -2.0, True, False],
                "name": ["floor_divide", "fmod", "remainder"],
                "dtype": [cupy.default_float_type()],
                "use_dtype": [True, False],
            }
        )
    )
)
class TestArithmeticBinary2(ArithmeticBinaryBase):

    def test_binary(self):
        self.check_binary()


@testing.with_requires("numpy>=2.0")
class TestArithmeticBinary3(ArithmeticBinaryBase):

    @pytest.mark.parametrize(
        "arg1",
        [
            testing.shaped_arange((2, 3), numpy, dtype=d)
            for d in no_complex_types
        ]
        + [0, 0.0, 2, 2.0, -2, -2.0, True, False],
    )
    @pytest.mark.parametrize(
        "arg2",
        [
            testing.shaped_reverse_arange((2, 3), numpy, dtype=d)
            for d in no_complex_types
        ]
        + [0, 0.0, 2, 2.0, -2, -2.0, True, False],
    )
    @pytest.mark.parametrize("name", ["floor_divide", "fmod", "remainder"])
    @pytest.mark.parametrize("dtype", [cupy.default_float_type()])
    @pytest.mark.parametrize("use_dtype", [True, False])
    @testing.numpy_cupy_allclose(
        accept_error=OverflowError, type_check=has_support_aspect64()
    )
    def test_both_raise(self, arg1, arg2, name, dtype, use_dtype, xp):
        if xp.isscalar(arg1) and xp.isscalar(arg2):
            pytest.skip("both scalar inputs is not supported")

        func = getattr(xp, name)

        if isinstance(arg1, numpy.ndarray):
            arg1 = xp.asarray(arg1)
        if isinstance(arg2, numpy.ndarray):
            arg2 = xp.asarray(arg2)

        dtype_arg = {"dtype": dtype} if use_dtype else {}
        with numpy.errstate(divide="ignore"):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                y = func(arg1, arg2, **dtype_arg)

        return y


@pytest.mark.skip("'casting' keyword is not supported yet")
class UfuncTestBase:

    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def check_casting_out(self, in0_type, in1_type, out_type, casting, xp):
        a = testing.shaped_arange((2, 3), xp, in0_type)
        b = testing.shaped_arange((2, 3), xp, in1_type)
        c = xp.zeros((2, 3), out_type)
        if casting != "unsafe":
            # may raise TypeError
            return xp.add(a, b, out=c, casting=casting)

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            ret = xp.add(a, b, out=c, casting=casting)
        ws = [w.category for w in ws]
        assert all([w == ComplexWarning for w in ws]), str(ws)
        return ret, xp.array(len(ws))

    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def check_casting_dtype(self, in0_type, in1_type, dtype, casting, xp):
        a = testing.shaped_arange((2, 3), xp, in0_type)
        b = testing.shaped_arange((2, 3), xp, in1_type)
        if casting != "unsafe":
            # may raise TypeError
            return xp.add(a, b, dtype=dtype, casting=casting)

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            ret = xp.add(a, b, dtype=dtype, casting="unsafe")
        ws = [w.category for w in ws]
        assert all([w == ComplexWarning for w in ws]), str(ws)
        return ret, xp.array(len(ws))

    # delete this, once check_casting_dtype passes
    @testing.numpy_cupy_allclose()
    def check_casting_dtype_unsafe_ignore_warnings(
        self, in0_type, in1_type, dtype, xp
    ):
        a = testing.shaped_arange((2, 3), xp, in0_type)
        b = testing.shaped_arange((2, 3), xp, in1_type)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return xp.add(a, b, dtype=dtype, casting="unsafe")


class TestUfunc(UfuncTestBase):

    @pytest.mark.parametrize(
        "casting",
        [
            "no",
            "equiv",
            "safe",
            "same_kind",
            "unsafe",
        ],
    )
    @testing.for_all_dtypes_combination(names=["in_type", "out_type"])
    def test_casting_out_only(self, in_type, out_type, casting):
        self.check_casting_out(in_type, in_type, out_type, casting)

    @pytest.mark.parametrize(
        "casting",
        [
            pytest.param("no", marks=pytest.mark.skip("flaky xfail")),
            pytest.param("equiv", marks=pytest.mark.skip("flaky xfail")),
            "safe",
            "same_kind",
            "unsafe",
        ],
    )
    @testing.for_all_dtypes_combination(
        names=["in0_type", "in1_type", "out_type"], full=False
    )
    def test_casting_in_out(self, in0_type, in1_type, out_type, casting):
        self.check_casting_out(in0_type, in1_type, out_type, casting)

    @pytest.mark.xfail()
    @pytest.mark.parametrize(
        "casting",
        [
            "no",
            "equiv",
        ],
    )
    @pytest.mark.parametrize(
        ("in0_type", "in1_type", "out_type"),
        [
            (numpy.int16, numpy.int32, numpy.int32),
        ],
    )
    def test_casting_in_xfail1(self, in0_type, in1_type, out_type, casting):
        self.check_casting_out(in0_type, in1_type, out_type, casting)

    @pytest.mark.skip("flaky xfail")
    @pytest.mark.parametrize(
        "casting",
        [
            "no",
            "equiv",
            "safe",
            "same_kind",
            "unsafe",
        ],
    )
    @testing.for_all_dtypes_combination(
        names=["in0_type", "in1_type", "dtype"], full=False
    )
    def test_casting_dtype(self, in0_type, in1_type, dtype, casting):
        self.check_casting_dtype(in0_type, in1_type, dtype, casting)

    @pytest.mark.xfail()
    @pytest.mark.parametrize(
        "casting",
        [
            "no",
            "equiv",
        ],
    )
    @pytest.mark.parametrize(
        ("in0_type", "in1_type", "dtype"),
        [
            (numpy.int16, numpy.int32, numpy.int32),
        ],
    )
    def test_casting_dtype_xfail1(self, in0_type, in1_type, dtype, casting):
        self.check_casting_dtype(in0_type, in1_type, dtype, casting)

    @pytest.mark.xfail()
    @pytest.mark.parametrize(
        "casting",
        [
            "no",
            "equiv",
            "safe",
            "same_kind",
        ],
    )
    @pytest.mark.parametrize(
        ("in0_type", "in1_type", "dtype"),
        [
            (numpy.int32, numpy.int32, numpy.bool_),
            (numpy.float64, numpy.float64, numpy.int32),
        ],
    )
    def test_casting_dtype_xfail2(self, in0_type, in1_type, dtype, casting):
        self.check_casting_dtype(in0_type, in1_type, dtype, casting)

    @testing.for_all_dtypes_combination(
        names=["in0_type", "in1_type", "dtype"], full=False
    )
    def test_casting_dtype_unsafe_ignore_warnings(
        self, in0_type, in1_type, dtype
    ):
        self.check_casting_dtype_unsafe_ignore_warnings(
            in0_type, in1_type, dtype
        )


@testing.slow
class TestUfuncSlow(UfuncTestBase):
    @pytest.mark.parametrize(
        "casting",
        [
            pytest.param("no", marks=pytest.mark.xfail()),
            pytest.param("equiv", marks=pytest.mark.xfail()),
            "safe",
            "same_kind",
            "unsafe",
        ],
    )
    @testing.for_all_dtypes_combination(
        names=["in0_type", "in1_type", "out_type"], full=True
    )
    def test_casting_out(self, in0_type, in1_type, out_type, casting):
        self.check_casting_out(in0_type, in1_type, out_type, casting)

    @pytest.mark.xfail()
    @pytest.mark.parametrize(
        "casting",
        [
            "no",
            "equiv",
            "safe",
            "same_kind",
            "unsafe",
        ],
    )
    @testing.for_all_dtypes_combination(
        names=["in0_type", "in1_type", "dtype"], full=True
    )
    def test_casting_dtype(self, in0_type, in1_type, dtype, casting):
        self.check_casting_dtype(in0_type, in1_type, dtype, casting)

    @testing.for_all_dtypes_combination(
        names=["in0_type", "in1_type", "dtype"], full=True
    )
    def test_casting_dtype_unsafe_ignore_warnings(
        self, in0_type, in1_type, dtype
    ):
        self.check_casting_dtype_unsafe_ignore_warnings(
            in0_type, in1_type, dtype
        )


class TestArithmeticModf:

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_modf(self, xp, dtype):
        a = xp.array([-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5], dtype=dtype)
        b, c = xp.modf(a)
        d = xp.empty((2, 7), dtype=dtype)
        d[0] = b
        d[1] = c
        return d


@testing.parameterize(
    *testing.product({"xp": [numpy, cupy], "shape": [(3, 2), (), (3, 0, 2)]})
)
class TestBoolSubtract:

    def test_bool_subtract(self):
        xp = self.xp
        shape = self.shape
        x = testing.shaped_random(shape, xp, dtype=numpy.bool_)
        y = testing.shaped_random(shape, xp, dtype=numpy.bool_)
        with pytest.raises(TypeError):
            xp.subtract(x, y)
