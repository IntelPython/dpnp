import numpy
import pytest

import dpnp
from tests.tests_perf.data_generator import *
from tests.tests_perf.test_perf_base import DPNPTestPerfBase


def cos_2_args(input_A, input_B, lib):
    """sin(A + B) = sin A cos B + cos A sin B"""
    sin_A = lib.sin(input_A)
    cos_B = lib.cos(input_B)
    sincosA = sin_A * cos_B
    cos_A = lib.cos(input_A)
    sin_B = lib.sin(input_B)
    sincosB = cos_A * sin_B
    result = sincosA + sincosB

    return result


class TestDPNPTrigonometric(DPNPTestPerfBase):
    @pytest.mark.parametrize(
        "func_name",
        [
            "arccos",
            "arccosh",
            "arcsin",
            "arcsinh",
            "arctan",
            "arctanh",
            "cbrt",
            "cos",
            "cosh",
            "deg2rad",
            "degrees",
            "exp",
            "exp2",
            "expm1",
            "log",
            "log10",
            "log1p",
            "log2",
            "rad2deg",
            "radians",
            "reciprocal",
            "sin",
            "sinh",
            "sqrt",
            "square",
            "tan",
            "tanh",
        ],
    )
    @pytest.mark.parametrize(
        "dtype", [numpy.float64, numpy.float32, numpy.int64, numpy.int32]
    )
    @pytest.mark.parametrize(
        "size", [512, 1024, 2048, 4096, 8192, 16384, 32768]
    )
    def test_trig1(self, func_name, lib, dtype, size):
        input1 = gen_array_1d(lib, size, dtype=dtype, seed=self.seed)

        self.dpnp_benchmark(func_name, lib, dtype, input1.size, input1)

    @pytest.mark.parametrize("dtype", [numpy.float64])
    @pytest.mark.parametrize("size", [16777216])
    def test_app1(self, lib, dtype, size):
        """
        /opt/intel/oneapi/vtune/2021.1-beta10/bin64/vtune -collect gpu-offload
            -knob enable-stack-collection=true -app-working-dir /home/work/dpnp --
            /home/work/anaconda3/bin/python -m pytest
                tests/tests_perf/math_tests/test_trigonometric.py::TestDPNPTrigonometric::test_app1[32768-float64-dpnp]
                -sv
        """
        input1 = gen_array_1d(lib, size, dtype=dtype, seed=self.seed)
        input2 = gen_array_1d(lib, size, dtype=dtype, seed=self.seed)

        self.dpnp_benchmark(
            "cos_2_args",
            lib,
            dtype,
            input1.size,
            input1,
            input2,
            lib,
            custom_fptr=cos_2_args,
        )
