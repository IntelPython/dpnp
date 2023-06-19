import numpy
import pytest

from tests.tests_perf.data_generator import *
from tests.tests_perf.test_perf_base import DPNPTestPerfBase


class TestDPNPMathematical(DPNPTestPerfBase):

    @pytest.mark.parametrize('func_name', ['add', 'divide', 'multiply', 'subtract'])
    @pytest.mark.parametrize('dtype', [numpy.float64, numpy.float32, numpy.int64, numpy.int32])
    @pytest.mark.parametrize('size', [512, 1024, 2048, 4096, 8192, 16384, 32768])
    def test_math_2args(self, func_name, lib, dtype, size):
        input1 = gen_array_1d(lib, size, dtype=dtype, seed=self.seed)
        input2 = gen_array_1d(lib, size, dtype=dtype, seed=self.seed)

        self.dpnp_benchmark(func_name, lib, dtype, input1.size, input1, input2)
