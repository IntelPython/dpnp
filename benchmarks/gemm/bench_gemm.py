"""Benchmark the libtensor GEMM (dpnp.tensor.matmul) across layouts/shapes.

Run one process per dtype (long in-process sweeps skew later timings on
the iGPU):

    for dt in i8 i32 i64 f32 f64 c64 bool; do
        python bench_gemm.py $dt --out results/base
    done

Each case is checked against NumPy before it is timed.
"""

import argparse
import gc
import json
import os
import statistics
import sys
import time

import numpy as np

import dpnp.tensor as dpt

DTYPES = {
    "bool": np.bool_,
    "i8": np.int8,
    "u8": np.uint8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    "u64": np.uint64,
    "f16": np.float16,
    "f32": np.float32,
    "f64": np.float64,
    "c64": np.complex64,
    "c128": np.complex128,
}


def _rand(rng, shape, dt):
    if dt == np.bool_:
        return rng.integers(0, 2, size=shape).astype(dt)
    if np.issubdtype(dt, np.integer):
        return rng.integers(
            -3 if np.issubdtype(dt, np.signedinteger) else 0, 4, size=shape
        ).astype(dt)
    x = rng.standard_normal(shape)
    if np.issubdtype(dt, np.complexfloating):
        x = x + 1j * rng.standard_normal(shape)
    return x.astype(dt)


def _layout(q, a_np, kind):
    """Put `a_np` on device with the requested memory layout.

    kind: "C" (C-contig), "F" (F-contig), "T" (transpose view of a C-contig
    array of the transposed shape), "S" (strided every other column)
    """
    if kind == "C":
        return dpt.asarray(np.ascontiguousarray(a_np), sycl_queue=q)
    if kind == "F":
        return dpt.asarray(np.asfortranarray(a_np), sycl_queue=q, order="F")
    if kind == "T":
        base = dpt.asarray(
            np.ascontiguousarray(np.swapaxes(a_np, -1, -2)), sycl_queue=q
        )
        return dpt.matrix_transpose(base)
    if kind == "S":
        shp = a_np.shape[:-1] + (2 * a_np.shape[-1],)
        big = np.zeros(shp, dtype=a_np.dtype)
        big[..., ::2] = a_np
        return dpt.asarray(big, sycl_queue=q)[..., ::2]
    raise ValueError(kind)


def cases():
    """(name, a_shape, b_shape, a_layout, b_layout)"""
    out = []
    for n in (128, 512, 1024):
        for la, lb in (
            ("C", "C"),
            ("F", "F"),
            ("T", "C"),
            ("C", "T"),
            ("T", "T"),
            ("S", "C"),
        ):
            out.append((f"sq{n}_{la}{lb}", (n, n), (n, n), la, lb))
    # rectangular
    out += [
        ("tall_16384x64x64_CC", (16384, 64), (64, 64), "C", "C"),
        ("wide_64x64x16384_CC", (64, 64), (64, 16384), "C", "C"),
        ("smallk_1024x8x1024_CC", (1024, 8), (8, 1024), "C", "C"),
        ("largek_64x16384x64_CC", (64, 16384), (16384, 64), "C", "C"),
        ("largek_64x16384x64_FF", (64, 16384), (16384, 64), "F", "F"),
        ("tiny_16x16x16_CC", (16, 16), (16, 16), "C", "C"),
    ]
    # gemv / gevm (2D with a unit dim, plus true 1-D)
    for n, k in ((2048, 1024), (256, 8192), (8192, 256)):
        out += [
            (f"gemv_{n}x{k}_C", (n, k), (k,), "C", "C"),
            (f"gemv_{n}x{k}_F", (n, k), (k,), "F", "C"),
            (f"gemv_{n}x{k}_T", (n, k), (k,), "T", "C"),
            (f"gevm_{k}x{n}_C", (k,), (k, n), "C", "C"),
            (f"gevm_{k}x{n}_F", (k,), (k, n), "C", "F"),
            (f"gemv2d_{n}x{k}_C", (n, k), (k, 1), "C", "C"),
        ]
    # syrk-pattern: x.T @ x and x @ x.T (same buffer)
    for k, n in ((2048, 512), (512, 1024), (16384, 64), (64, 1024)):
        out.append((f"syrk_xTx_{k}x{n}", (k, n), None, "C", "syrkT"))
        out.append((f"syrk_xxT_{n}x{k}", (n, k), None, "C", "syrkN"))
    # batched
    for b, n in ((64, 128), (512, 32), (8, 512), (4096, 8)):
        out.append((f"batch{b}_{n}_CC", (b, n, n), (b, n, n), "C", "C"))
        out.append((f"batch{b}_{n}_TC", (b, n, n), (b, n, n), "T", "C"))
    out.append(("batch64_128_bcast", (64, 128, 128), (128, 128), "C", "C"))
    out.append(("batch256_64x1_gemv", (256, 64, 256), (256, 256, 1), "C", "C"))
    return out


def _time(fn, q, reps, min_time=0.2):
    fn()
    q.wait()
    ts = []
    t_start = time.perf_counter()
    while len(ts) < reps or (
        time.perf_counter() - t_start < min_time and len(ts) < 200
    ):
        t0 = time.perf_counter()
        r = fn()
        q.wait()
        ts.append(time.perf_counter() - t0)
        del r
    return statistics.median(ts), min(ts)


def _check(res_np, ref, dt):
    if np.issubdtype(dt, np.inexact):
        tol = {np.float16: 5e-2, np.float32: 1e-4, np.complex64: 1e-4}.get(
            dt, 1e-10
        )
        scale = max(1.0, float(np.max(np.abs(ref))))
        return bool(np.max(np.abs(res_np - ref)) <= tol * scale * 8)
    return bool(np.array_equal(res_np, ref))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dtype", choices=sorted(DTYPES))
    p.add_argument("--out", default=None, help="output directory for json")
    p.add_argument("--filter", default="", help="substring filter on name")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--max-elems", type=int, default=2**21)
    p.add_argument("--api", choices=("dpt", "dpnp"), default="dpt")
    args = p.parse_args()
    dt = DTYPES[args.dtype]

    q = dpt.empty(0).sycl_queue
    rng = np.random.default_rng(0)
    print(
        f"# device: {q.sycl_device.name}  dtype: {args.dtype}  "
        f"api: {args.api}",
        file=sys.stderr,
    )
    if args.api == "dpnp":
        import dpnp

    results = []
    for name, ash, bsh, la, lb in cases():
        if args.filter and args.filter not in name:
            continue
        a_np = _rand(rng, ash, dt)
        if lb in ("syrkT", "syrkN"):
            if a_np.size > args.max_elems:
                continue
            a = _layout(q, a_np, la)
            if lb == "syrkT":
                x1, x2 = dpt.matrix_transpose(a), a
                ref = a_np.T @ a_np
            else:
                x1, x2 = a, dpt.matrix_transpose(a)
                ref = a_np @ a_np.T
        else:
            b_np = _rand(rng, bsh, dt)
            if max(a_np.size, b_np.size) > args.max_elems:
                continue
            x1, x2 = _layout(q, a_np, la), _layout(q, b_np, lb)
            ref = a_np @ b_np
        if ref.size > args.max_elems:
            continue

        if args.api == "dpnp":
            x1, x2 = dpnp.ndarray._create_from_usm_ndarray(
                x1
            ), dpnp.ndarray._create_from_usm_ndarray(x2)

            def fn():
                return dpnp.matmul(x1, x2)

        else:

            def fn():
                return dpt.matmul(x1, x2)

        r = fn()
        if args.api == "dpnp":
            r = r.get_array()
        ok = _check(dpt.asnumpy(r), ref.astype(dpt.asnumpy(r).dtype), dt)
        del r
        med, best = _time(fn, q, args.reps)
        n_ = ref.shape[-2] if ref.ndim >= 2 else 1
        m_ = ref.shape[-1] if ref.ndim >= 1 else 1
        k_ = ash[0] if lb == "syrkT" else ash[-1]
        batch = int(np.prod(ref.shape[:-2])) if ref.ndim > 2 else 1
        if len(ash) == 1:  # gevm
            n_, m_ = 1, ref.shape[-1]
        elif bsh is not None and len(bsh) == 1:  # gemv
            n_, m_ = ref.shape[-1], 1
        flops = 2.0 * batch * n_ * m_ * k_
        rec = dict(
            name=name,
            dtype=args.dtype,
            api=args.api,
            ok=ok,
            median_ms=med * 1e3,
            best_ms=best * 1e3,
            gops=flops / med / 1e9,
        )
        results.append(rec)
        print(
            f"{name:32s} {'ok ' if ok else 'BAD'} {med*1e3:10.3f} ms "
            f"{rec['gops']:9.2f} Gop/s",
            flush=True,
        )
        del x1, x2, a_np, ref
        gc.collect()

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(
            os.path.join(args.out, f"{args.api}_{args.dtype}.json"), "w"
        ) as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
