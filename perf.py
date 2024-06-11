import numpy
import dpnp
from tests.helper import generate_random_numpy_array
import cProfile, pstats, io

a = generate_random_numpy_array((256,512,512),dtype='f4',hermitian=False, seed_value=81)
b = generate_random_numpy_array((256,512,),dtype='f4',hermitian=False, seed_value=76)

a_dp = dpnp.array(a,device='cpu')
b_dp = dpnp.array(b,device='cpu')

cold_run = dpnp.linalg.solve(a_dp, b_dp)
pr = cProfile.Profile()
pr.enable()
# res = numpy.linalg.solve(a, b)
res = dpnp.linalg.solve(a_dp, b_dp)
pr.disable()
s = io.StringIO()
sortBy = pstats.SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortBy)
ps.print_stats()
print(s.getvalue())
