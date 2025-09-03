import dpnp
import numpy as np
from dpnp.tests.helper import generate_random_numpy_array
import time
from IPython import get_ipython

ipython = get_ipython()
if ipython is None:
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
    ipython = TerminalInteractiveShell()


dtypes = ['f4', 'f8', 'c8', 'c16']
n = 256
print(f"size: ({n},{n},{n}) ")
for dtype in dtypes:
    print(f"\n=== dtype: {dtype} ===")
    a = generate_random_numpy_array((n,n,n), dtype=dtype, seed_value=81)

    # dpnp arrays on GPU
    a_dp = dpnp.array(a, device='gpu')
    exec_q = a_dp.sycl_queue

    # Cold run
    _ = dpnp.linalg.slogdet(a_dp)
    exec_q.wait()

    time.sleep(1)
    print("DPNP (GPU, Old):")
    ipython.run_line_magic('timeit', 'dpnp.linalg.slogdet(a_dp); exec_q.wait()')
