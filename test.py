import dpnp
import numpy
import cProfile, pstats, io


# na = numpy.random.randint(-10**4, 10**4, size=(4096,4096))
na = numpy.random.randint(-10**4, 10**4, size=(8192,8192))
a = numpy.array(na, dtype = "int32")
a_dp = dpnp.array(a, device='cpu')

a_stride = a[::2,::2]
a_dp_stride = a_dp[::2,::2]
dpnp.linalg.slogdet(a_dp_stride)

pr = cProfile.Profile()
pr.enable()
# res = numpy.linalg.slogdet(a)
res = dpnp.linalg.slogdet(a_dp_stride)
pr.disable()

s = io.StringIO()
sortBy = pstats.SortKey.CUMULATIVE

ps = pstats.Stats(pr, stream=s).sort_stats(sortBy)
ps.print_stats()

print(s.getvalue())
