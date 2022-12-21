import itertools
import unittest

import numpy
import pytest

import dpnp as cupy
from tests.third_party.cupy import testing


float_types = [numpy.float32, numpy.float64]
complex_types = []
signed_int_types = [numpy.int32, numpy.int64]
unsigned_int_types = []
int_types = signed_int_types + unsigned_int_types
all_types = float_types + int_types + complex_types
negative_types = (float_types + signed_int_types + complex_types)
negative_no_complex_types = float_types + signed_int_types
no_complex_types = float_types + int_types


@testing.gpu
@testing.parameterize(*(
    testing.product({
        'nargs': [1],
        'name': ['reciprocal', 'angle'],
    }) + testing.product({
        'nargs': [2],
        'name': [
            'add', 'multiply', 'divide', 'power', 'subtract', 'true_divide',
            'floor_divide', 'fmod', 'remainder', 'mod'],
    })
))
class TestArithmeticRaisesWithNumpyInput(unittest.TestCase):

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
            with self.assertRaises(TypeError):
                func(*arys)


@testing.gpu
@testing.parameterize(*(
    testing.product({
        'arg1': ([testing.shaped_arange((2, 3), numpy, dtype=d) + 1
                  for d in all_types
                  ] + [2, 2.0]),
        'name': ['reciprocal'],
    })
))
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestArithmeticUnary(unittest.TestCase):

    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_unary(self, xp):
        arg1 = self.arg1
        if isinstance(arg1, numpy.ndarray):
            arg1 = xp.asarray(arg1)
        y = getattr(xp, self.name)(arg1)

        if self.name in ('real', 'imag'):
            # Some NumPy functions return Python scalars for Python scalar
            # inputs.
            # We need to convert them to arrays to compare with CuPy outputs.
            if xp is numpy and isinstance(arg1, (bool, int, float, complex)):
                y = xp.asarray(y)

            # TODO(niboshi): Fix this
            # numpy.real and numpy.imag return Python int if the input is
            # Python bool. CuPy should return an array of dtype.int32 or
            # dtype.int64 (depending on the platform) in such cases, instead
            # of an array of dtype.bool.
            if xp is cupy and isinstance(arg1, bool):
                y = y.astype(int)

        return y


class ArithmeticBinaryBase:

    @testing.numpy_cupy_allclose(atol=1e-4)
    def check_binary(self, xp):
        arg1 = self.arg1
        arg2 = self.arg2
        np1 = numpy.asarray(arg1)
        np2 = numpy.asarray(arg2)
        dtype1 = np1.dtype
        dtype2 = np2.dtype

        if self.name == 'power':
            # TODO(niboshi): Fix this: power(0, 1j)
            #     numpy => 1+0j
            #     cupy => 0j
            if dtype2 in complex_types and (np1 == 0).any():
                return xp.array(True)

            # TODO(niboshi): Fix this: xp.power(0j, 0)
            #     numpy => 1+0j
            #     cupy => 0j
            c_arg1 = dtype1 in complex_types
            if c_arg1 and (np1 == 0j).any() and (np2 == 0).any():
                return xp.array(True)

        # TODO(niboshi): Fix this: xp.add(0j, xp.array([2.], 'f')).dtype
        #     numpy => complex64
        #     cupy => complex128
        if isinstance(arg1, complex):
            if dtype2 in (numpy.float16, numpy.float32):
                return xp.array(True)

        if isinstance(arg1, numpy.ndarray):
            arg1 = xp.asarray(arg1)
        if isinstance(arg2, numpy.ndarray):
            arg2 = xp.asarray(arg2)

        # Subtraction between booleans is not allowed.
        if (self.name == 'subtract'
                and dtype1 == numpy.bool_
                and dtype2 == numpy.bool_):
            return xp.array(True)

        func = getattr(xp, self.name)
        with testing.NumpyError(divide='ignore'):
            with numpy.warnings.catch_warnings():
                numpy.warnings.filterwarnings('ignore')
                if self.use_dtype:
                    y = func(arg1, arg2, dtype=self.dtype)
                else:
                    y = func(arg1, arg2)

        # TODO(niboshi): Fix this. If rhs is a Python complex,
        #    numpy returns complex64
        #    cupy returns complex128
        if xp is cupy and isinstance(arg2, complex):
            if dtype1 in (numpy.float16, numpy.float32):
                y = y.astype(numpy.complex64)

        # NumPy returns an output array of another type than DPNP when input ones have diffrent types.
        if self.name == 'multiply' and xp is cupy:
            if xp.isscalar(arg1) and xp.isscalar(arg2):
                # If both are scalars, the result will be a scalar, so needs to convert into numpy-scalar.
                y = numpy.asarray(y)
            elif dtype1 != dtype2:
                is_array_arg1 = not xp.isscalar(arg1)
                is_array_arg2 = not xp.isscalar(arg2)

                is_int_float = lambda _x, _y: numpy.issubdtype(_x, numpy.integer) and numpy.issubdtype(_y, numpy.floating)
                is_same_type = lambda _x, _y, _type: numpy.issubdtype(_x, _type) and numpy.issubdtype(_y, _type)

                if is_array_arg1 and is_array_arg2:
                    # If both inputs are arrays where one is of floating type and another - integer,
                    # NumPy will return an output array of always "float64" type,
                    # while DPNP will return the array of a wider type from the input arrays.
                    if is_int_float(dtype1, dtype2) or is_int_float(dtype2, dtype1):
                        y = y.astype(numpy.float64)
                elif is_same_type(dtype1, dtype2, numpy.floating) or is_same_type(dtype1, dtype2, numpy.integer):
                    # If one input is an array and another - scalar,
                    # NumPy will return an output array of the same type as the inpupt array has,
                    # while DPNP will return the array of a wider type from the inputs (considering both array and scalar).
                    if is_array_arg1 and not is_array_arg2:
                        y = y.astype(dtype1)
                    elif is_array_arg2 and not is_array_arg1:
                        y = y.astype(dtype2)

        # NumPy returns different values (nan/inf) on division by zero
        # depending on the architecture.
        # As it is not possible for CuPy to replicate this behavior, we ignore
        # the difference here.
        # if self.name in ('floor_divide', 'remainder', 'mod'):
        #     if y.dtype in (float_types + complex_types) and (np2 == 0).any():
        #         y = xp.asarray(y)
        #         y[y == numpy.inf] = numpy.nan
        #        y[y == -numpy.inf] = numpy.nan

        return y


@testing.gpu
@testing.parameterize(*(
    testing.product({
        # TODO(unno): boolean subtract causes DeprecationWarning in numpy>=1.13
        'arg1': [testing.shaped_arange((2, 3), numpy, dtype=d)
                 for d in all_types
                 ] + [0, 0.0, 2, 2.0],
        'arg2': [testing.shaped_reverse_arange((2, 3), numpy, dtype=d)
                 for d in all_types
                 ] + [0, 0.0, 2, 2.0],
        'name': ['add', 'multiply', 'power', 'subtract'],
    }) + testing.product({
        'arg1': [numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                 for d in negative_types
                 ] + [0, 0.0, 2, 2.0, -2, -2.0],
        'arg2': [numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                 for d in negative_types
                 ] + [0, 0.0, 2, 2.0, -2, -2.0],
        'name': ['divide', 'true_divide', 'subtract'],
    })
))
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestArithmeticBinary(ArithmeticBinaryBase, unittest.TestCase):

    def test_binary(self):
        self.use_dtype = False
        self.check_binary()


@testing.gpu
@testing.parameterize(*(
    testing.product({
        'arg1': [numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                 for d in int_types
                 ] + [0, 0.0, 2, 2.0, -2, -2.0],
        'arg2': [numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                 for d in int_types
                 ] + [0, 0.0, 2, 2.0, -2, -2.0],
        'name': ['true_divide'],
        'dtype': [numpy.float64],
        'use_dtype': [True, False],
    }) + testing.product({
        'arg1': [numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                 for d in float_types] + [0.0, 2.0, -2.0],
        'arg2': [numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                 for d in float_types] + [0.0, 2.0, -2.0],
        'name': ['power', 'true_divide', 'subtract'],
        'dtype': [numpy.float64],
        'use_dtype': [True, False],
    }) + testing.product({
        'arg1': [testing.shaped_arange((2, 3), numpy, dtype=d)
                 for d in no_complex_types
                 ] + [0, 0.0, 2, 2.0, -2, -2.0],
        'arg2': [testing.shaped_reverse_arange((2, 3), numpy, dtype=d)
                 for d in no_complex_types
                 ] + [0, 0.0, 2, 2.0, -2, -2.0],
        'name': ['floor_divide', 'fmod', 'remainder', 'mod'],
        'dtype': [numpy.float64],
        'use_dtype': [True, False],
    }) + testing.product({
        'arg1': [numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                 for d in negative_no_complex_types
                 ] + [0, 0.0, 2, 2.0, -2, -2.0],
        'arg2': [numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                 for d in negative_no_complex_types
                 ] + [0, 0.0, 2, 2.0, -2, -2.0],
        'name': ['floor_divide', 'fmod', 'remainder', 'mod'],
        'dtype': [numpy.float64],
        'use_dtype': [True, False],
    })
))
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestArithmeticBinary2(ArithmeticBinaryBase, unittest.TestCase):

    def test_binary(self):
        if (self.use_dtype and
                numpy.lib.NumpyVersion(numpy.__version__) < '1.10.0'):
            raise unittest.SkipTest('Test for numpy>=1.10')
        self.check_binary()


class TestArithmeticModf(unittest.TestCase):

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_modf(self, xp, dtype):
        a = xp.array([-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5], dtype=dtype)
        b, c = xp.modf(a)
        d = xp.empty((2, 7), dtype=dtype)
        d[0] = b
        d[1] = c
        return d


@testing.parameterize(*testing.product({
    'xp': [numpy, cupy],
    'shape': [(3, 2), (), (3, 0, 2)]
}))
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@testing.gpu
class TestBoolSubtract(unittest.TestCase):

    def test_bool_subtract(self):
        xp = self.xp
        if xp is numpy and not testing.numpy_satisfies('>=1.14.0'):
            raise unittest.SkipTest('NumPy<1.14.0')
        shape = self.shape
        x = testing.shaped_random(shape, xp, dtype=numpy.bool_)
        y = testing.shaped_random(shape, xp, dtype=numpy.bool_)
        with pytest.raises(TypeError):
            xp.subtract(x, y)
