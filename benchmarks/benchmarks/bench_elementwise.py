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

"""Benchmarks for unary elementwise math functions, dpnp against NumPy."""

from ._utils import (
    _DTYPES,
    _EXECUTOR_NAMES,
    _EXECUTORS,
    _SIZES_1D,
    default_queue,
    make_synchronizer,
    skip_unsupported_dtype,
)


class Elementwise:
    """Unary elementwise ufuncs, dpnp against NumPy."""

    params = [_EXECUTOR_NAMES, _SIZES_1D, _DTYPES]
    param_names = ["executor", "size", "dtype"]

    def setup(self, executor, size, dtype):
        self.np = _EXECUTORS[executor]
        if executor == "dpnp":
            skip_unsupported_dtype(default_queue(), dtype)
        self.sync = make_synchronizer(executor)
        dt = getattr(self.np, dtype)
        self.a = self.np.arange(size, dtype=dt)

    def time_arccos(self, *args):
        self.sync(self.np.arccos(self.a))

    def time_arccosh(self, *args):
        self.sync(self.np.arccosh(self.a))

    def time_arcsin(self, *args):
        self.sync(self.np.arcsin(self.a))

    def time_arcsinh(self, *args):
        self.sync(self.np.arcsinh(self.a))

    def time_arctan(self, *args):
        self.sync(self.np.arctan(self.a))

    def time_arctanh(self, *args):
        self.sync(self.np.arctanh(self.a))

    def time_cbrt(self, *args):
        self.sync(self.np.cbrt(self.a))

    def time_cos(self, *args):
        self.sync(self.np.cos(self.a))

    def time_cosh(self, *args):
        self.sync(self.np.cosh(self.a))

    def time_degrees(self, *args):
        self.sync(self.np.degrees(self.a))

    def time_exp(self, *args):
        self.sync(self.np.exp(self.a))

    def time_exp2(self, *args):
        self.sync(self.np.exp2(self.a))

    def time_expm1(self, *args):
        self.sync(self.np.expm1(self.a))

    def time_log(self, *args):
        self.sync(self.np.log(self.a))

    def time_log10(self, *args):
        self.sync(self.np.log10(self.a))

    def time_log1p(self, *args):
        self.sync(self.np.log1p(self.a))

    def time_log2(self, *args):
        self.sync(self.np.log2(self.a))

    def time_rad2deg(self, *args):
        self.sync(self.np.rad2deg(self.a))

    def time_radians(self, *args):
        self.sync(self.np.radians(self.a))

    def time_reciprocal(self, *args):
        self.sync(self.np.reciprocal(self.a))

    def time_sin(self, *args):
        self.sync(self.np.sin(self.a))

    def time_sinh(self, *args):
        self.sync(self.np.sinh(self.a))

    def time_sqrt(self, *args):
        self.sync(self.np.sqrt(self.a))

    def time_square(self, *args):
        self.sync(self.np.square(self.a))

    def time_tan(self, *args):
        self.sync(self.np.tan(self.a))

    def time_tanh(self, *args):
        self.sync(self.np.tanh(self.a))
