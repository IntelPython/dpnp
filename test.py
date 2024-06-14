import numpy

import dpnp
import dpnp.linalg
from tests.helper import generate_random_numpy_array

# a = numpy.array([[[2,3],[1,2]],[[1,2],[2,3]]],dtype='f4')
# a = numpy.array([[[1,2],[3,5]],[[1,2],[3,5]],[[1,2],[3,5]]], dtype='f4')
# b = numpy.array([[1,2],[1,2],[1,2]], dtype='f4')
# a = numpy.array([[1,2],[2,3]],dtype='f4')
# a = numpy.array([[[2,3],[1,2]]],dtype='f4')
# a = numpy.array([[[1,2],[2,3]]],dtype='f4')
a = generate_random_numpy_array(
    (2, 3, 3), dtype='c8', hermitian=False, seed_value=81
)

b = generate_random_numpy_array(
    (2, 3, 3), dtype='c8', hermitian=False, seed_value=76
)
# a = numpy.array(
#             [
#                 [[1, 0, 3], [0, 5, 0], [7, 0, 9]],
#                 [[3, 0, 3], [0, 7, 0], [7, 0, 11]],
#             ],
#             dtype='f4',
#         )
a_dp = dpnp.array(a)
b_dp = dpnp.array(b)

# a_f = numpy.array(a,order="F")
# a_dp_f = dpnp.array(a_dp,order="F")

# res = numpy.linalg.eigh(a)
# res_dp = dpnp.linalg.eigh(a_dp)
# res = numpy.linalg.eigh(a_f)
# res_dp = dpnp.linalg.eigh(a_dp_f)
res = numpy.linalg.solve(a,b)
res_dp = dpnp.linalg.solve(a_dp,b_dp)
# res = numpy.linalg.eigvalsh(a)
# res_dp = dpnp.linalg.eigvalsh(a_dp)


print("VALS: ")
print("NUMPY: ")
print(res)
print("DPNP: ")
print(res_dp)

# print("BEFORE")
# print(a_dp)

# print("VALS: ")
# print("NUMPY: ")
# print(res[0])
# print("DPNP: ")
# print(res_dp[0])


# print("VECS: ")
# print("NUMPY: ")
# print(res[1])
# print("DPNP: ")
# print(res_dp[1])

# print("AFTER")
# print(a_dp)


# a = generate_random_numpy_array((10,3,3),dtype=numpy.float32, hermitian=True, seed_value=81)
# a_dp = dpnp.array(a, device='gpu')

# res = numpy.linalg.eigh(a)
# res_dp = dpnp.linalg.eigh(a_dp)

# print("Done")
