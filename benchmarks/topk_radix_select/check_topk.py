"""Check dpnp.tensor.top_k against a stable NumPy reference.

usage: python check_topk.py <device> [quick] [set]

quick drops the largest shapes, set compares the selected index sets only
(needed for radix select, whose output order is unspecified).
"""

import itertools
import sys

import numpy as np

import dpnp.tensor as dpt

dev = sys.argv[1] if len(sys.argv) > 1 else "gpu"
quick = "quick" in sys.argv[2:]
# compare the selected elements only, ignoring their order
as_set = "set" in sys.argv[2:]
rng = np.random.default_rng(1234)


def ref_topk(x, k, largest):
    """values and indices of top k along the last axis, ties by index"""
    if largest:
        n = x.shape[-1]
        o = np.argsort(x[..., ::-1], axis=-1, kind="stable")[..., ::-1]
        o = n - 1 - o
    else:
        o = np.argsort(x, axis=-1, kind="stable")
    o = o[..., :k]
    return np.take_along_axis(x, o, axis=-1), o


def make(dtype, shape, kind):
    dt = np.dtype(dtype)
    size = int(np.prod(shape))
    if dt == np.bool_:
        return rng.integers(0, 2, size=shape).astype(bool)
    if kind == "random":
        if dt.kind == "f":
            return rng.standard_normal(size).reshape(shape).astype(dt)
        info = np.iinfo(dt)
        return rng.integers(info.min, info.max, size=shape, endpoint=True,
                            dtype=dt)
    if kind == "dups":
        return rng.integers(0, 4, size=shape).astype(dt)
    if kind == "const":
        return np.full(shape, 3, dtype=dt)
    if kind == "sorted":
        return np.sort(make(dtype, (size,), "random")).reshape(shape)
    if kind == "special":
        vals = [0.0, -0.0, np.nan, np.inf, -np.inf, 1.0, -1.0, -np.nan]
        return rng.choice(np.array(vals, dtype=dt), size=shape)
    raise ValueError(kind)


dtypes = ["?", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f2", "f4",
          "f8"]
shapes = [(1,), (2,), (7,), (100,), (1000,), (5003,), (70001,), (3, 1),
          (5, 33), (64, 257), (1000, 50), (3, 20000), (2, 300007)]
if not quick:
    shapes += [(1_000_003,), (2, 2_000_000)]
kinds = ["random", "dups", "const", "sorted", "special"]

q = dpt.empty(0, device=dev).sycl_queue
n_fail = 0
n_ok = 0
for dtype, shape, kind in itertools.product(dtypes, shapes, kinds):
    if kind == "special" and np.dtype(dtype).kind != "f":
        continue
    if dtype == "f2" and not q.sycl_device.has_aspect_fp16:
        continue
    if dtype == "f8" and not q.sycl_device.has_aspect_fp64:
        continue
    x = make(dtype, shape, kind)
    n = shape[-1]
    ks = sorted({1, min(n, 2), min(n, 5), max(1, n // 3), n})
    xd = dpt.asarray(x, device=dev)
    for k, largest in itertools.product(ks, [True, False]):
        rv, ri = ref_topk(x, k, largest)
        r = dpt.top_k(xd, k, axis=-1,
                      mode="largest" if largest else "smallest")
        gv = dpt.asnumpy(r.values)
        gi = dpt.asnumpy(r.indices)
        if as_set:
            gi, ri = np.sort(gi, axis=-1), np.sort(ri, axis=-1)
            gv = np.take_along_axis(x.reshape(gi.shape[:-1] + (-1,)), gi, -1)
            rv = np.take_along_axis(x.reshape(ri.shape[:-1] + (-1,)), ri, -1)
        ok = np.array_equal(gi, ri) and np.array_equal(
            gv, rv, equal_nan=(np.dtype(dtype).kind == "f"))
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            if n_fail <= 20:
                bad = np.argwhere(gi != ri)
                print(f"FAIL {dtype} {shape} {kind} k={k} largest={largest}"
                      f" first mismatches at {bad[:3].tolist()}"
                      f" got {gi.ravel()[:8]} expected {ri.ravel()[:8]}")
print(f"{dev}: {n_ok} passed, {n_fail} failed")
sys.exit(1 if n_fail else 0)
