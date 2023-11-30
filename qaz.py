import dpnp
import numpy

a = numpy.array(
    [
        [2, 3, 1, 4, 5],
        [5, 6, 7, 8, 9],
        [9, 7, 7, 2, 3],
        [1, 4, 5, 1, 8],
        [8, 9, 8, 5, 3],
    ]
)

a_dp = dpnp.array(a)

print(numpy.linalg.slogdet(a))
print(dpnp.linalg.slogdet(a_dp))
