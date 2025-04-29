import dpnp as np
import numpy


# x = np.array([2.5,3.0,3.4,3.7], dtype='f8')
# xp = np.array([1, 2, 3], dtype='f8')
# fp = np.array([3 ,2 ,0], dtype='int32')

x = np.asarray([0, 1, 2, 4, 6, 8, 9, 10], dtype='f8')
xp = np.asarray([1, 3, 5, 7, 9], dtype='f8')
fp = np.sin(xp).astype('f8')

left=10
right=20

res = np.interp(x,xp,fp,left,right)

res_np = numpy.interp(np.asnumpy(x), np.asnumpy(xp), np.asnumpy(fp), left, right)

print(res)
print("numpy")
print(res_np)
