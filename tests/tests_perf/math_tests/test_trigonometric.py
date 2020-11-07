import dpnp
import numpy
import pytest

from tests.tests_perf.data_generator import *
from tests.tests_perf.test_perf_base import DPNPTestPerfBase


class TestDPNPTrigonometric(DPNPTestPerfBase):

    @pytest.mark.parametrize("func_name", ["arccos", "arccosh", "arcsin", "arcsinh", "arctan", "arctanh", "cbrt", "cos", "cosh", "deg2rad", "degrees", "exp", "exp2", "expm1", "log", "log10", "log1p", "log2", "rad2deg", "radians", "reciprocal", "sin", "sinh", "sqrt", "square", "tan", "tanh"])
    @pytest.mark.parametrize("dtype", [numpy.float64, numpy.float32, numpy.int64, numpy.int32])
    @pytest.mark.parametrize("size", [512, 1024, 2048, 4096, 8192, 16384, 32768])
    def test_cos(self, func_name, lib, dtype, size):
        input1 = gen_array_1d(lib, size, dtype=dtype, seed=self.seed)

        self.dpnp_benchmark(func_name, lib, dtype, input1.size, input1)
