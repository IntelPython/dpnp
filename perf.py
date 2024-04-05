import dpnp
import numpy
import cProfile, pstats, io

na = numpy.random.randint(-10**4, 10**4, size=(1024,1024))
nb = numpy.random.randint(-10**4, 10**4, size=(1024,1024))

dtype = 'int32'

a = numpy.array(na, dtype = dtype)
b = numpy.array(na, dtype=dtype)

a_dp = dpnp.array(a, device='gpu')
b_dp = dpnp.array(b, device='gpu')

# dpnp.linalg.lstsq(a_dp,b_dp)
# numpy.linalg.lstsq(a,b)
dpnp.linalg.svd(a_dp)
pr = cProfile.Profile()
pr.enable()
# res = numpy.linalg.lstsq(a,b)
res = dpnp.linalg.svd(a_dp)
pr.disable()
s = io.StringIO()
sortBy = pstats.SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortBy)
ps.print_stats()
print(s.getvalue())
