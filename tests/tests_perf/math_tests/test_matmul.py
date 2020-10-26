import dpnp
import numpy
import pytest

from tests.tests_perf.data_generator import gen_array
from tests.tests_perf.test_perf_base import TestBase


# pytest tests/tests_perf/math_tests/test_matmul.py::TestMatmul -s
class TestMatmul(TestBase):

    @pytest.mark.parametrize("lib", [numpy, dpnp])
    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64, numpy.int32, numpy.int64])
    @pytest.mark.parametrize("size", [2 ** 18, 2 ** 20, 2 ** 22])
    def test_matmul(self, lib, dtype, size):
        arr1 = gen_array(lib, size, dtype=dtype, seed=self.seed)
        arr2 = gen_array(lib, size, dtype=dtype, seed=self.seed)
        self._test_func("matmul", lib, dtype, size, arr1, arr2, repeat=5, number=100)
