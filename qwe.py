import dpnp
import numpy
from tests.third_party.cupy import testing

# dtype = 'c8'
# a = numpy.empty((3,0),dtype=dtype)
# b = testing.shaped_random((3,), numpy, dtype=dtype)

# a_dp = dpnp.array(a,device='cpu')
# b_dp = dpnp.array(b, device='cpu')

# # res = numpy.linalg.lstsq(a,b,rcond=-1)
# res_dp = dpnp.linalg.lstsq(a_dp,b_dp, rcond=-1)
# res_np = dpnp.linalg.lstsq(a,b, rcond=-1, numpy=True)

# # print(res)
# print(res_np)
# print(res_dp)

a = numpy.random.randint(-10,10, size=(5,5))
b = numpy.random.randint(-10,10, size=(5,5))

a_dp = dpnp.array(a,device='cpu')
b_dp = dpnp.array(b,device='cpu')

res = dpnp.linalg.lstsq(a_dp,b_dp)

# print(res)
