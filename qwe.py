import dpnp
import numpy
from tests.helper import generate_random_numpy_array

a = dpnp.array(generate_random_numpy_array((2,3,3,3,3),dtype='f4',hermitian=False, seed_value=81))
b = dpnp.array(generate_random_numpy_array((2,3,3,3),dtype='f4',hermitian=False, seed_value=76))

res = dpnp.linalg.solve(a,b,new=True)

print(res.shape)
print(res.flags)
