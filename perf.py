import numpy
import dpnp
import dpctl
from dpnp.dpnp_utils import get_usm_allocations

import time
from IPython import get_ipython

ipython = get_ipython()
if ipython is None:
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
    ipython = TerminalInteractiveShell()


dtypes = ['f4', 'f8']

print(dpctl.select_default_device().name)

for dtype in dtypes:
    xp_size = 16_000_000
    fp_size = 2_000_000

    xp_type = dtype
    fp_type = dtype

    numpy.random.seed(81)
    xp = numpy.sort(numpy.random.uniform(0, 1000, size=xp_size).astype(xp_type))
    numpy.random.seed(76)
    fp = numpy.sort(numpy.random.uniform(-100, 100, size=xp_size).astype(fp_type))

    numpy.random.seed(70)
    x = numpy.random.uniform(xp[0], xp[-1], size=fp_size).astype(xp_type)

    x_dp = dpnp.array(x)
    xp_dp = dpnp.array(xp)
    fp_dp = dpnp.array(fp)

    _, exec_q = get_usm_allocations([x_dp, xp_dp, fp_dp])

    _ = dpnp.interp(x_dp, xp_dp, fp_dp)

    print(f"xp_type : {xp_type}")
    print(f"fp_type : {fp_type}")
    print(f"Numpy: ")
    ipython.run_line_magic('timeit', '-n 10 -r 7 numpy.interp(x,xp,fp)')

    time.sleep(1)

    print(f"DPNP: ")
    ipython.run_line_magic('timeit', '-n 10 -r 7 dpnp.interp(x_dp,xp_dp,fp_dp); exec_q.wait()')
