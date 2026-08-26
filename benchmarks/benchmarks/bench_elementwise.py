# *****************************************************************************
# Copyright (c) 2020, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

"""Benchmarks for elementwise ufuncs, dpnp against NumPy.

The ufunc is a parameter rather than a method per function, following NumPy's
own ``benchmarks/benchmarks/bench_ufunc.py``, so extending the coverage is a
one-line change.
"""

from ._utils import (
    _EXECUTOR_NAMES,
    _EXECUTORS,
    _SIZES_1D,
    default_queue,
    make_synchronizer,
    skip_unsupported_dtype,
)

# Float only. These ufuncs return floats, so an integer input would time the
# int-to-float promotion rather than the kernel, and would not be comparable
# with the float cells. NumPy's own suite restricts them the same way.
_FLOAT_DTYPES = ["float64", "float32"]

# Only one name per ufunc: rad2deg/deg2rad wrap the same backend functions as
# degrees/radians, and abs, true_divide and pow are aliases of absolute, divide
# and power.
_UNARY = [
    "absolute",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "cbrt",
    "ceil",
    "cos",
    "cosh",
    "degrees",
    "exp",
    "exp2",
    "expm1",
    "floor",
    "log",
    "log10",
    "log1p",
    "log2",
    "radians",
    "reciprocal",
    "rint",
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "tan",
    "tanh",
    "trunc",
]

_BINARY = [
    "add",
    "arctan2",
    "divide",
    "hypot",
    "multiply",
    "power",
    "subtract",
]

# Input ranges keeping each ufunc inside its domain, so that none is timed
# entirely on an out-of-domain path.
_RANGES = {
    "arccos": (-1, 1),
    "arccosh": (1, 10),
    "arcsin": (-1, 1),
    "arctanh": (-0.9, 0.9),
    "log": (1, 10),
    "log10": (1, 10),
    "log1p": (1, 10),
    "log2": (1, 10),
    "reciprocal": (1, 10),
    "sqrt": (1, 10),
}
_DEFAULT_RANGE = (-10, 10)

# Positive first operand and a small second one, so divide never sees a zero
# and power stays in range.
_BINARY_RANGES = ((1, 10), (1, 2))


class _Ufunc:
    """Shared setup for a ufunc benchmark.

    Defines no ``time_*`` method, so ASV does not discover it as a benchmark.
    """

    param_names = ["executor", "ufunc", "size", "dtype"]

    def setup(self, executor, ufunc, size, dtype):
        self.np = _EXECUTORS[executor]
        if executor == "dpnp":
            skip_unsupported_dtype(default_queue(), dtype)
        self.sync = make_synchronizer(executor)
        self.fn = getattr(self.np, ufunc)

    def _input(self, size, dtype, bounds):
        lo, hi = bounds
        return self.np.linspace(lo, hi, size, dtype=getattr(self.np, dtype))


# ---------------------------------------------------------------------------
# One input -- transcendental, rounding and sign ufuncs
# ---------------------------------------------------------------------------


class Unary(_Ufunc):
    """Unary ufuncs, e.g. exp, sqrt, floor."""

    params = [_EXECUTOR_NAMES, _UNARY, _SIZES_1D, _FLOAT_DTYPES]

    def setup(self, executor, ufunc, size, dtype):
        super().setup(executor, ufunc, size, dtype)
        bounds = _RANGES.get(ufunc, _DEFAULT_RANGE)
        self.a = self._input(size, dtype, bounds)
        # Warm up, so the first timed call does not pay device setup.
        self.sync(self.fn(self.a))

    def time_unary(self, executor, ufunc, size, dtype):
        self.sync(self.fn(self.a))


# ---------------------------------------------------------------------------
# Two inputs -- arithmetic ufuncs
# ---------------------------------------------------------------------------


class Binary(_Ufunc):
    """Binary ufuncs, e.g. add, power, hypot."""

    params = [_EXECUTOR_NAMES, _BINARY, _SIZES_1D, _FLOAT_DTYPES]

    def setup(self, executor, ufunc, size, dtype):
        super().setup(executor, ufunc, size, dtype)
        first, second = _BINARY_RANGES
        self.a = self._input(size, dtype, first)
        self.b = self._input(size, dtype, second)
        self.sync(self.fn(self.a, self.b))

    def time_binary(self, executor, ufunc, size, dtype):
        self.sync(self.fn(self.a, self.b))
