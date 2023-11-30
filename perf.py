import numpy
import dpnp


# na = numpy.random.randint(-10**4, 10**4, size=(4096,4096))

# a = numpy.array(na, dtype = "int32")

# a_dp = dpnp.array(a,device='cpu')

# def test(xp,dtype,device):
#     na = numpy.random.randint(-10**4, 10**4, size=(4096,4096))

#     a = numpy.array(na, dtype = dtype)
#     a_dp = dpnp.array(a,device=device)

#     if xp == 0:
#         numpy.linalg.slogdet(a)
#     else:
#         dpnp.linalg.slogdet(a_dp)

def init(na, dtype):
    # na = numpy.random.randint(-10**4, 10**4, size=(4096,4096))

    a = numpy.array(na, dtype = dtype)
    a_dp_cpu = dpnp.array(a,device='cpu')
    a_dp_gpu = dpnp.array(a,device='gpu')

    numpy.linalg.slogdet(a)
    dpnp.linalg.slogdet(a_dp_cpu)

    return a,a_dp_cpu,a_dp_gpu
