"""Long-row check of dpnp.tensor.top_k: exact selected set, values, signs.

usage: python check_long.py <device>
"""
import sys, numpy as np, dpnp.tensor as dpt
dev = sys.argv[1]; rng = np.random.default_rng(1); fails = 0; total = 0
def ref(x, k, largest):
    xs = -x.astype(np.float64) if largest and x.dtype.kind != 'f' else x
    if x.dtype.kind == 'f':
        v = np.where(x == 0, 0, x)
        key = v if not largest else -v
        key = np.where(np.isnan(v), np.inf, key)  # nan always "largest" -> largest-mode first
        if largest: key = np.where(np.isnan(v), -np.inf, key)
    else:
        key = xs
    return key, np.argsort(key, axis=-1, kind='stable')[..., :k]
for dt in ["i1", "u2", "i4", "f4", "f8", "i8", "f2"]:
    for rows, n in [(1, 1 << 20), (3, 300001), (16, 1 << 17), (1, 1 << 22)]:
        for gen in ["rand", "few", "uniform01", "const"]:
            d = np.dtype(dt)
            if gen == "const": x = np.full((rows, n), 3).astype(d)
            elif gen == "few": x = rng.integers(0, 5, (rows, n)).astype(d)
            elif d.kind == "f":
                x = (rng.random((rows, n)) if gen == "uniform01" else rng.standard_normal((rows, n))).astype(d)
                x[:, ::97] = np.nan; x[:, 5::101] = -0.0; x[:, 7::103] = 0.0
            else:
                info = np.iinfo(d); x = rng.integers(info.min, info.max, (rows, n), dtype=d, endpoint=True)
            xd = dpt.asarray(x, device=dev)
            for k in [1, 7, 100, 3000, n // 3]:
                for largest in [True, False]:
                    total += 1
                    r = dpt.top_k(xd, k, axis=-1, mode="largest" if largest else "smallest")
                    got = dpt.asnumpy(r.indices); key, sel = ref(x, k, largest)
                    exp = sel; got_s = np.argsort(got, axis=-1); exp_s = np.argsort(exp, axis=-1)
                    vals = dpt.asnumpy(r.values); ev = np.take_along_axis(x, got, axis=-1)
                    ok = np.array_equal(np.sort(got, axis=-1), np.sort(exp, axis=-1)) and np.array_equal(vals, ev, equal_nan=d.kind == 'f')
                    if d.kind == 'f': ok = ok and np.array_equal(np.signbit(vals), np.signbit(ev))
                    if not ok:
                        fails += 1
                        if fails < 10: print("FAIL", dt, rows, n, gen, k, largest)
print(f"{dev}: {total - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
