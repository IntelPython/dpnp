import numpy

import dpnp
import dpnp.linalg
from tests.helper import generate_random_numpy_array

# a = numpy.array([[[1,2],[2,3]],[[2,3],[1,2]]],dtype='f4')
a = numpy.array([[[1,2],[2,3]],[[1,2],[2,3]]],dtype='f4')
# a = numpy.array([[1,2],[2,3]],dtype='f4')
# a = numpy.array([[[2,3],[1,2]]],dtype='f4')
# a = numpy.array([[[1,2],[2,3]]],dtype='f4')
a_dp = dpnp.array(a)

res = numpy.linalg.eigh(a)
res_dp = dpnp.linalg.eigh(a_dp)

print("VALS: ")
print("NUMPY: ")
print(res[0])
print("DPNP: ")
print(res_dp[0])

print("VECS: ")
print("NUMPY: ")
print(res[1])
print("DPNP: ")
print(res_dp[1])


# a = generate_random_numpy_array((10,3,3),dtype=numpy.float32, hermitian=True, seed_value=81)
# a_dp = dpnp.array(a, device='gpu')

# res = numpy.linalg.eigh(a)
# res_dp = dpnp.linalg.eigh(a_dp)

# print("Done")
