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

The ufunc is a parameter, as in NumPy's own ``bench_ufunc.py``.
"""

from ._utils import (
    _EXECUTOR_NAMES,
    _EXECUTORS,
    _FLOAT_DTYPES,
    _SIZES_1D,
    default_queue,
    make_synchronizer,
    skip_unsupported_dtype,
)

# One name per ufunc; rad2deg, deg2rad, abs, true_divide and pow are aliases.
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

# Ranges keeping each ufunc inside its domain.
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

# Positive first operand, small second: no divide by zero, no overflow.
_BINARY_RANGES = ((1, 10), (1, 2))


class _Ufunc:
    """Shared setup for a ufunc benchmark.

    Defines no ``time_*``, so ASV does not discover it as a benchmark.
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
        # Warm up.
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
