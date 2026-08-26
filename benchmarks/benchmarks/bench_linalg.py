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

"""Benchmarks for matrix products and decompositions, dpnp against NumPy."""

from ._utils import (
    _EXECUTOR_NAMES,
    _EXECUTORS,
    default_queue,
    make_synchronizer,
    skip_unsupported_dtype,
)

# square matrix orders -- local to this suite
_ORDERS = [16, 32, 64, 128, 256, 512, 1024]

# Integers are native for the matrix products, which do not promote.
_DTYPES = ["float64", "float32", "int64", "int32"]

# LAPACK has no integer path, so an integer input to a decomposition would time
# the promotion instead of the factorization.
_FLOAT_DTYPES = ["float64", "float32"]


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
        # Non-contiguous operand, which reaches a different BLAS path: the
        # transpose is expressed as a flag rather than as a copy.
        self.at = self.a.T
        # Pay the one-time SYCL and oneMKL initialization before timing.
        self.sync(self.np.dot(self.a, self.b))

    def time_dot(self, executor, order, dtype):
        self.sync(self.np.dot(self.a, self.b))

    def time_dot_transposed(self, executor, order, dtype):
        self.sync(self.np.dot(self.a, self.at))

    def time_matmul(self, executor, order, dtype):
        self.sync(self.np.matmul(self.a, self.b))

    def time_matmul_transposed(self, executor, order, dtype):
        self.sync(self.np.matmul(self.a, self.at))

    def time_inner(self, executor, order, dtype):
        self.sync(self.np.inner(self.a, self.b))

    def time_einsum_ij_jk(self, executor, order, dtype):
        self.sync(self.np.einsum("ij,jk", self.a, self.b))


# ---------------------------------------------------------------------------
# LAPACK-backed decompositions and norms
# ---------------------------------------------------------------------------


class Linalg:
    """Square-matrix decompositions -- det, norm, solve and svd."""

    params = [_EXECUTOR_NAMES, _ORDERS, _FLOAT_DTYPES]
    param_names = ["executor", "order", "dtype"]

    def setup(self, executor, order, dtype):
        self.np = _EXECUTORS[executor]
        self.sync = make_synchronizer(executor)
        if executor == "dpnp":
            skip_unsupported_dtype(default_queue(), dtype)
        dt = getattr(self.np, dtype)
        # I + 1/order is diagonally dominant, so it is non-singular with a
        # condition number of 2, and its determinant is 2 at every order rather
        # than overflowing. An arange matrix is rank 2, which would make solve
        # and det meaningless.
        self.a = (
            self.np.eye(order, dtype=dt)
            + self.np.ones((order, order), dtype=dt) / order
        )
        self.b = self.np.ones(order, dtype=dt)
        # norm is the cheapest of these, so it warms up the device without
        # paying for a second factorization.
        self.sync(self.np.linalg.norm(self.a))

    def time_det(self, executor, order, dtype):
        self.sync(self.np.linalg.det(self.a))

    def time_norm(self, executor, order, dtype):
        self.sync(self.np.linalg.norm(self.a))

    def time_solve(self, executor, order, dtype):
        self.sync(self.np.linalg.solve(self.a, self.b))

    def time_svd(self, executor, order, dtype):
        self.sync(self.np.linalg.svd(self.a))
