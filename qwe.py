import numpy
# from scipy.linalg import qr

import dpnp

a = dpnp.array([[1,2],[3,5]])
b = dpnp.array([[1,2],[1,2]])

a_np = numpy.array([[1, 2], [3, 5]])
b_np = numpy.array([[1,2],[1,2]])


res = dpnp.linalg.lstsq(a,b)
res_np = numpy.linalg.lstsq(a_np,b_np)

print("Numpy: ")
print(res_np)
print("DPNP: ")
print(res)
