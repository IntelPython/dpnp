# import dpnp
# from dpnp.linalg.
# import numpy

# a = numpy.arange(1,49,dtype='f4').reshape(2,2,3,4)
# a = numpy.arange(1,5,dtype='f4').reshape(2,2)
# a_dp = dpnp.array(a)
# print(numpy.linalg.svd(a,compute_uv=False))
# print("===================")
# print(dpnp.linalg.svd(a_dp,compute_uv=False))
# print("===================")
# a_dp = dpnp.array(a,device='cpu')
# print(dpnp.linalg.svd(a_dp,full_matrices=False))

# print(numpy.linalg.svd(a,full_matrices=False))
# res_np = numpy.linalg.svd(a)
# print(res_np[0].shape)
# print(res_np[1].shape)
# print(res_np[2].shape)
# print("===================")
# print(dpnp.linalg.svd(a_dp, full_matrices=False))
# res_dp = dpnp.linalg.svd(a_dp)
# print(res_dp[0].shape)
# print(res_dp[1].shape)
# print(res_dp[2].shape)
# print(a_dp)


from dpnp.linalg.dpnp_utils_linalg import _lu_factor
import dpnp
import numpy
import scipy

a = numpy.array([[1, 2], [3, 4]],dtype="f4",order='C')
a_dp = dpnp.array(a)

print(scipy.linalg.lu_factor(a))
print("===========================")
print(_lu_factor(a_dp))
