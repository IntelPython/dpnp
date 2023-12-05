import dpnp
import numpy
from tests.helper import assert_dtype_allclose


dtype='float32'
data = numpy.arange(100, dtype=dtype)
dpnp_data = dpnp.array(data, device='cpu')

np_res = numpy.fft.fft(data)
dpnp_res = dpnp.fft.fft(dpnp_data)

print("numpy: ",np_res.dtype)
print("dpnp: ",dpnp_res.dtype)

assert_dtype_allclose(dpnp_res, np_res)
