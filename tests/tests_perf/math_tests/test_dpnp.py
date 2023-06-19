import dpnp
import numpy
import pytest

from tests.tests_perf.data_generator import *
from tests.tests_perf.test_perf_base import DPNPTestPerfBase


class TestDPNP(DPNPTestPerfBase):

    @pytest.mark.parametrize('dtype', [numpy.float64, numpy.float32, numpy.int64, numpy.int32])
    @pytest.mark.parametrize('size', [32, 64, 128, 256])  # , 512, 1024, 2048, 4096])
    def test_matmul(self, lib, dtype, size):
        input1 = gen_array_2d(lib, size, size, dtype=dtype, seed=self.seed)
        input2 = gen_array_2d(lib, size, size, dtype=dtype, seed=self.seed)

        self.dpnp_benchmark('matmul', lib, dtype, input1.size, input1, input2)
