import dpnp
import numpy
import dpnp.linalg
from tests.helper import generate_random_numpy_array

a = numpy.array([[[1,2],[2,3]],[[1,2],[2,3]]],dtype='f4')
a_dp = dpnp.array(a)

res = numpy.linalg.eigh(a)
res_dp = dpnp.linalg.eigh(a_dp)

print("NUMPY: ")
print(res)
print("DPNP: ")
print(res_dp)

print("Numpy shapes: ")
print(res[0].shape)
print(res[1].shape)


# a = generate_random_numpy_array((10,3,3),dtype=numpy.float32, hermitian=True, seed_value=81)
# a_dp = dpnp.array(a, device='gpu')

# res = numpy.linalg.eigh(a)
# res_dp = dpnp.linalg.eigh(a_dp)

# print("Done")
