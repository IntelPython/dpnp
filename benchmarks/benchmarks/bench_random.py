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

"""Benchmarks for random sampling, dpnp.random against numpy.random."""

from functools import partial

from ._utils import (
    _EXECUTOR_NAMES,
    _EXECUTORS,
    _FLOAT_DTYPES,
    _SIZES_1D,
    default_queue,
    make_synchronizer,
    skip_unsupported_dtype,
)

_SEED = 1


# One name per generator; rand, random, ranf, sample and randn are aliases.
class Sample:
    """Random sampling, dpnp against NumPy."""

    params = [_EXECUTOR_NAMES, _SIZES_1D]
    param_names = ["executor", "size"]

    def setup(self, executor, size):
        self.executor = _EXECUTORS[executor]
        if executor == "dpnp":
            # No dtype keyword: without fp64 dpnp returns float32, NumPy f64.
            skip_unsupported_dtype(default_queue(), "float64")
        self.sync = make_synchronizer(executor)
        # Warm up.
        self.sync(self.executor.random.random_sample(size))

    def time_random_sample(self, executor, size):
        np = self.executor
        self.sync(np.random.random_sample(size))

    def time_standard_normal(self, executor, size):
        np = self.executor
        self.sync(np.random.standard_normal(size))


# ---------------------------------------------------------------------------
# Generators taking an explicit dtype
# ---------------------------------------------------------------------------


class TypedSample:
    """Uniform and normal sampling at an explicit dtype.

    dpnp reaches these through ``RandomState`` and NumPy through
    ``default_rng``; the module-level functions in ``Sample`` take no dtype, so
    only this class can compare float32 on a device without fp64.
    """

    params = [_EXECUTOR_NAMES, _SIZES_1D, _FLOAT_DTYPES]
    param_names = ["executor", "size", "dtype"]

    def setup(self, executor, size, dtype):
        mod = _EXECUTORS[executor]
        if executor == "dpnp":
            skip_unsupported_dtype(default_queue(), dtype)
        self.sync = make_synchronizer(executor)
        dt = getattr(mod, dtype)
        if executor == "dpnp":
            rng = mod.random.RandomState(seed=_SEED)
            self._uniform = partial(rng.uniform, 0.0, 1.0, size, dtype=dt)
            self._normal = partial(rng.normal, 0.0, 1.0, size, dtype=dt)
        else:
            rng = mod.random.default_rng(_SEED)
            self._uniform = partial(rng.random, size, dtype=dt)
            self._normal = partial(rng.standard_normal, size, dtype=dt)
        # Warm up.
        self.sync(self._uniform())

    def time_uniform(self, executor, size, dtype):
        self.sync(self._uniform())

    def time_normal(self, executor, size, dtype):
        self.sync(self._normal())
