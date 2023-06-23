import numpy

import dpnp

from .common import Benchmark


# asv run --python=python --quick --bench Sample
class Sample(Benchmark):
    executors = {"dpnp": dpnp, "numpy": numpy}
    params = [["dpnp", "numpy"], [2**16, 2**20, 2**24]]
    param_names = ["executor", "size"]

    def setup(self, executor, size):
        self.executor = self.executors[executor]

    def time_rand(self, executor, size):
        np = self.executor
        np.random.rand(size)

    def time_randn(self, executor, size):
        np = self.executor
        np.random.randn(size)

    def time_random_sample(self, executor, size):
        np = self.executor
        np.random.random_sample((size,))
