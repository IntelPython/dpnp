import math
import numpy
import pytest

from tests.tests_perf.data_generator import *
from tests.tests_perf.test_perf_base import DPNPTestPerfBase


SEED = 7777777
SL, SH = 10.0, 50.0
KL, KH = 10.0, 50.0
TL, TH = 1.0, 2.0
RISK_FREE = 0.1
VOLATILITY = 0.2


def math_erf(x):
    result = numpy.empty(x.shape, dtype=x.dtype)
    for i in range(result.size):
        result[i] = math.erf(x[i])

    return result


# module 'numpy' has no attribute 'erf'
numpy.erf = math_erf


def gen_data(lib, low, high, size):
    return lib.random.uniform(low, high, size)


def black_scholes_put(lib, S, K, T, r, sigma):
    d1 = (lib.log(S / K) + (r + sigma * sigma / 2.0) * T) / (
        sigma * lib.sqrt(T)
    )
    d2 = d1 - sigma * lib.sqrt(T)

    cdf_d1 = (1 + lib.erf(d1 / lib.sqrt(2))) / 2
    cdf_d2 = (1 + lib.erf(d2 / lib.sqrt(2))) / 2

    bs_call = S * cdf_d1 - K * lib.exp(-r * T) * cdf_d2

    return K * lib.exp(-r * T) - S + bs_call


class TestBlackScholes(DPNPTestPerfBase):
    @pytest.mark.parametrize("dtype", [numpy.float64])
    @pytest.mark.parametrize("size", [1024, 2048, 4096, 8192])
    def test_bs_put(self, lib, dtype, size):
        numpy.random.seed(SEED)
        S = gen_data(lib, SL, SH, size)
        K = gen_data(lib, KL, KH, size)
        T = gen_data(lib, TL, TH, size)

        self.dpnp_benchmark(
            "bs_put",
            lib,
            dtype,
            size,
            lib,
            S,
            K,
            T,
            RISK_FREE,
            VOLATILITY,
            custom_fptr=black_scholes_put,
        )
