import numpy

import dpnp

from .common import Benchmark


# asv run --python=python --bench Elementwise
# --quick option will run every case once
# but looks like first execution has additional overheads
# (need to be investigated)
class Elementwise(Benchmark):
    executors = {"dpnp": dpnp, "numpy": numpy}
    params = [
        ["dpnp", "numpy"],
        [2**16, 2**20, 2**24],
        ["float64", "float32", "int64", "int32"],
    ]
    param_names = ["executor", "size", "dtype"]

    def setup(self, executor, size, dtype):
        self.np = self.executors[executor]
        dt = getattr(self.np, dtype)
        self.a = self.np.arange(size, dtype=dt)

    def time_arccos(self, *args):
        self.np.arccos(self.a)

    def time_arccosh(self, *args):
        self.np.arccosh(self.a)

    def time_arcsin(self, *args):
        self.np.arcsin(self.a)

    def time_arcsinh(self, *args):
        self.np.arcsinh(self.a)

    def time_arctan(self, *args):
        self.np.arctan(self.a)

    def time_arctanh(self, *args):
        self.np.arctanh(self.a)

    def time_cbrt(self, *args):
        self.np.cbrt(self.a)

    def time_cos(self, *args):
        self.np.cos(self.a)

    def time_cosh(self, *args):
        self.np.cosh(self.a)

    def time_degrees(self, *args):
        self.np.degrees(self.a)

    def time_exp(self, *args):
        self.np.exp(self.a)

    def time_exp2(self, *args):
        self.np.exp2(self.a)

    def time_expm1(self, *args):
        self.np.expm1(self.a)

    def time_log(self, *args):
        self.np.log(self.a)

    def time_log10(self, *args):
        self.np.log10(self.a)

    def time_log1p(self, *args):
        self.np.log1p(self.a)

    def time_log2(self, *args):
        self.np.log2(self.a)

    def time_rad2deg(self, *args):
        self.np.rad2deg(self.a)

    def time_radians(self, *args):
        self.np.radians(self.a)

    def time_reciprocal(self, *args):
        self.np.reciprocal(self.a)

    def time_sin(self, *args):
        self.np.sin(self.a)

    def time_sinh(self, *args):
        self.np.sinh(self.a)

    def time_sqrt(self, *args):
        self.np.sqrt(self.a)

    def time_square(self, *args):
        self.np.square(self.a)

    def time_tan(self, *args):
        self.np.tan(self.a)

    def time_tanh(self, *args):
        self.np.tanh(self.a)
