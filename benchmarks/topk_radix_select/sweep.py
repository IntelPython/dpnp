"""Time dpnp.tensor.top_k along the last axis.

usage: python sweep.py <device> <dtype> <cases> <ks> [<variant> ...]

  cases    "ROWSxN,ROWSxN", N may be an expression, e.g. "1x2**25"
  ks       "1,10,100", values of k larger than N are skipped
  variant  "label:VAR=val;VAR=val", environment set around each call,
           e.g. "merge:DPNP_TOPK_ALGO=merge", defaults to one plain "t:"

Prints "rows n k <time per variant>" in milliseconds, the minimum over the
timed calls that follow 2 warm-up calls, the variants are interleaved so
drift affects them equally. Floats are standard normal, or uniform in [0, 1)
with SWEEP_UNIFORM=1, integers span the whole type.

The number of timed calls comes from what the warm-ups took: about
SWEEP_BUDGET (default 3) seconds per variant, at most SWEEP_NIT (15) calls
and at least MIN_NIT (5). Short calls therefore get all 15 and a merge sort
of one 2**26-element row, which takes seconds, gets 5.

ROW_SCALE=S multiplies the rows of every case by S, keeping n, to make the
data large enough for a big GPU. The rows are capped so a case holds no more
elements than devinfo.element_cap allows: MAX_ELEMS (default 2**28), the
share MEM_FRACTION (default 0.4) of the device memory that top_k's input,
output and scratch fit in, and the largest int64 index output that one
allocation can hold. A case is never made smaller than given, so that the
suites which choose their row counts on purpose keep them; a case that is
over the cap on its own is skipped with a message instead. The printed rows
are the ones used.

Arrays larger than HOST_CHUNK (default 2**26) elements are filled on the
device in row blocks, so the host copy stays small; smaller ones are
generated in one call as before, which keeps their data unchanged.

The variant environment is read by the extension at call time, so it only
has an effect with the DPNP_TOPK_ALGO switch (algo_switch.py) built in. Pass
it as a variant, not in the environment: every variant's keys, DPNP_TOPK_ALGO
included, are cleared before each call so one variant can't leak into another.
"""

import os
import sys
import time

import numpy as np

import dpnp.tensor as dpt
from devinfo import element_cap

dev, dtype = sys.argv[1], sys.argv[2]
cases = [
    tuple(int(eval(v)) for v in c.split("x")) for c in sys.argv[3].split(",")
]
ks = [int(v) for v in sys.argv[4].split(",")]
variants = []
for v in sys.argv[5:] or ["t:"]:
    label, _, spec = v.partition(":")
    variants.append(
        (label, dict(kv.split("=", 1) for kv in spec.split(";") if kv))
    )
# cleared before every call so one variant's setting can't leak into the next
keys = {k for _, d in variants for k in d} | {"DPNP_TOPK_ALGO"}
n_iter = int(os.environ.get("SWEEP_NIT", 15))
min_iter = int(os.environ.get("MIN_NIT", 5))
budget = float(os.environ.get("SWEEP_BUDGET", 3.0))
uniform = bool(os.environ.get("SWEEP_UNIFORM"))
rng = np.random.default_rng(0)
dt = np.dtype(dtype)
row_scale = int(os.environ.get("ROW_SCALE", 1))
max_elems = int(eval(os.environ.get("MAX_ELEMS", "2**28")))
host_chunk = int(eval(os.environ.get("HOST_CHUNK", "2**26")))
cap = element_cap(dpt.empty(0, device=dev).sycl_device, dt.itemsize, max_elems)
if row_scale > 1:
    cases = [
        (max(rows, min(rows * row_scale, cap // n)), n) for rows, n in cases
    ]
kept = []
for rows, n in cases:
    if rows * n > cap:
        print(
            f"skipping {rows}x{n}: {rows * n} elements is over the cap of "
            f"{cap} for {dt}, see devinfo.py",
            file=sys.stderr,
        )
    else:
        kept.append((rows, n))
cases = kept


def setenv(env):
    for kk in keys:
        os.environ.pop(kk, None)
    os.environ.update(env)


def block(rows, n):
    if dt == np.bool_:
        return rng.integers(0, 2, size=(rows, n)).astype(bool)
    if dt.kind == "f":
        a = rng.random((rows, n)) if uniform else rng.standard_normal((rows, n))
        return a.astype(dt)
    info = np.iinfo(dt)
    return rng.integers(info.min, info.max, size=(rows, n), dtype=dt)


def make(rows, n):
    """The input array. Anything over HOST_CHUNK elements is filled in row
    blocks so the host never holds the whole thing; up to that size it is one
    call, which gives the same data as before the blocks existed."""
    if rows * n <= host_chunk:
        return dpt.asarray(block(rows, n), device=dev)
    xd = dpt.empty((rows, n), dtype=dt, device=dev)
    step = max(1, host_chunk // n)
    for i in range(0, rows, step):
        r = min(step, rows - i)
        xd[i : i + r, :] = block(r, n)
    return xd


print("rows n k " + " ".join(label for label, _ in variants), flush=True)
for rows, n in cases:
    xd = make(rows, n)
    q = xd.sycl_queue
    for k in ks:
        if k > n:
            continue
        slowest = 0.0
        for _, env in variants:
            setenv(env)
            for _ in range(2):
                t0 = time.perf_counter()
                r = dpt.top_k(xd, k, axis=-1)
                q.wait()
                slowest = max(slowest, time.perf_counter() - t0)
                del r
        # the warm-ups say how long a call takes, so spend about SWEEP_BUDGET
        # seconds per variant on it: SWEEP_NIT calls for the short ones, fewer
        # for a merge sort of 2**26 elements, never fewer than MIN_NIT
        iters = int(min(n_iter, max(min_iter, budget / slowest)))
        ts = [[] for _ in variants]
        for _ in range(iters):
            for vi, (_, env) in enumerate(variants):
                setenv(env)
                q.wait()
                t0 = time.perf_counter()
                r = dpt.top_k(xd, k, axis=-1)
                q.wait()
                ts[vi].append(time.perf_counter() - t0)
                del r
        res = [np.min(t) * 1e3 for t in ts]
        print(rows, n, k, " ".join(f"{t:8.3f}" for t in res), flush=True)
    del xd
