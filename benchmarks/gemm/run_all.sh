#!/bin/bash
# Usage: run_all.sh <outdir> [api] [extra bench_gemm.py args...]
# One process per dtype.
out=${1:?outdir}; api=${2:-dpt}; shift $(($# < 2 ? $# : 2))
here=$(dirname "$0")
for dt in bool i8 i16 i32 i64 f16 f32 f64 c64 c128; do
    python "$here/bench_gemm.py" $dt --api "$api" --out "$out" "$@" || echo "FAILED $dt"
done
