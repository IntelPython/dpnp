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

"""Benchmarks for matrix products, dpnp against NumPy."""

from ._utils import (
    _DTYPES,
    _EXECUTOR_NAMES,
    _EXECUTORS,
    default_queue,
    make_synchronizer,
    skip_unsupported_dtype,
)

# square matrix orders -- local to this suite
_ORDERS = [16, 32, 64, 128, 256, 512, 1024]


# ---------------------------------------------------------------------------
# Square matrix products
# ---------------------------------------------------------------------------


class MatMul:
    """Products of two square matrices -- dot, matmul, inner and einsum."""

    params = [_EXECUTOR_NAMES, _ORDERS, _DTYPES]
    param_names = ["executor", "order", "dtype"]

    def setup(self, executor, order, dtype):
        self.np = _EXECUTORS[executor]
        self.sync = make_synchronizer(executor)
        if executor == "dpnp":
            skip_unsupported_dtype(default_queue(), dtype)
        dt = getattr(self.np, dtype)
        self.a = self.np.arange(order * order, dtype=dt).reshape((order, order))
        self.b = self.np.arange(order * order, dtype=dt).reshape((order, order))

    def time_dot(self, executor, order, dtype):
        self.sync(self.np.dot(self.a, self.b))

    def time_matmul(self, executor, order, dtype):
        self.sync(self.np.matmul(self.a, self.b))

    def time_inner(self, executor, order, dtype):
        self.sync(self.np.inner(self.a, self.b))

    def time_einsum_ij_jk(self, executor, order, dtype):
        self.sync(self.np.einsum("ij,jk", self.a, self.b))
