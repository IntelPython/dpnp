import dpnp
import numpy
import pytest

from tests.tests_perf.data_generator import gen_array
from tests.tests_perf.test_perf_base import DPNPTestPerfBase


def pytest_generate_tests(metafunc):
    metafunc.parametrize("lib", [numpy, dpnp], ids=["base", "DPNP"])

class TestDPNP(DPNPTestPerfBase):

    @pytest.mark.parametrize("dtype", [numpy.float64, numpy.float32, numpy.int64, numpy.int32])
    @pytest.mark.parametrize("size", [32, 64, 128, 256])
    def test_matmul(self, lib, dtype, size):
        input1 = gen_array(lib, size * size, dtype=dtype, seed=self.seed).reshape((size, size))
        input2 = gen_array(lib, size * size, dtype=dtype, seed=self.seed).reshape((size, size))

        self.dpnp_benchmark("matmul", lib, dtype, input1.size, input1, input2)
