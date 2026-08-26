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

from ._utils import (
    _EXECUTOR_NAMES,
    _EXECUTORS,
    _SIZES_1D,
    make_synchronizer,
)


class Sample:
    """Random sampling, dpnp against NumPy."""

    params = [_EXECUTOR_NAMES, _SIZES_1D]
    param_names = ["executor", "size"]

    def setup(self, executor, size):
        self.executor = _EXECUTORS[executor]
        self.sync = make_synchronizer(executor)
        # Warm up, so the first timed call does not pay device setup.
        self.sync(self.executor.random.rand(size))

    def time_rand(self, executor, size):
        np = self.executor
        self.sync(np.random.rand(size))

    def time_randn(self, executor, size):
        np = self.executor
        self.sync(np.random.randn(size))

    def time_random_sample(self, executor, size):
        np = self.executor
        self.sync(np.random.random_sample((size,)))
