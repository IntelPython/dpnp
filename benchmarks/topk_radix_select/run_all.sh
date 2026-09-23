#!/bin/bash
# usage: run_all.sh [SUITE ...]     (default: all suites, see --list)
#
# Runs the top_k radix select benchmarks and checks, configured by config.sh
# (override any value from the environment). QUICK=1 runs only the first
# dtype and case of every suite with 3 timed calls, to check the setup.
set -u
here=$(cd "$(dirname "$0")" && pwd)
source "$here/config.sh"

suites_all="check builds kn small long algo algo3 rowsn"
describe() {
  cat <<'EOF'
check   correctness of NEW_ROOT: check_topk.py (set mode) and check_long.py
builds  NEW_ROOT against MASTER/BASE/SORTED/FEAT_ROOT, rows from 64 to 2**25
kn      NEW_ROOT against MASTER_ROOT with k from n/2 to n
small   NEW_ROOT against MASTER_ROOT on rows of 8 to 48 elements, k 1 to n
long    NEW_ROOT against MASTER_ROOT on a few rows of up to 2**28, k near n
algo    radix select vs merge sort on short rows, sets the merge threshold
        (needs ALGO_ROOT)
algo3   radix select vs merge sort vs radix sort, including 1 byte types
        and rows of 64 to 256 (needs ALGO_ROOT)
rowsn   the same three, varying the row count and n independently, to tell
        which of them the 1 byte crossover follows (needs ALGO_ROOT)
EOF
}
if [[ ${1:-} == --list || ${1:-} == -h ]]; then describe; exit 0; fi
suites=${*:-$suites_all}

setup_env || exit 1
# Another sweep on the same device would be mixed into these numbers: a run
# left over from an interrupted driver inflated a whole round here by 3x.
# (dropping this shell and its parent, whose command lines can name sweep.py)
busy=$(pgrep -f "sweep\.py" | grep -cvx -e "$$" -e "$PPID")
if ((busy > 0)) && [[ -z ${FORCE:-} ]]; then
  echo "$busy sweep.py processes are already running, their device time would" >&2
  echo "land in these results. Stop them, or set FORCE=1 to run anyway." >&2
  exit 1
fi
export PYTHON=${PYTHON:-$CONDA_ENV/bin/python}
if [[ ${QUICK:-} ]]; then export SWEEP_NIT=3; fi
mkdir -p "$RESULTS_DIR"

sorting_so() { ls "$1"/dpnp/tensor/_tensor_sorting_impl*.so 2>/dev/null | head -1; }
have_root() { [[ -n $1 && -n $(sorting_so "$1") ]]; }
have_switch() { have_root "$1" && grep -aq DPNP_TOPK_ALGO "$(sorting_so "$1")"; }
first() { if [[ ${QUICK:-} ]]; then echo "${1%%[, ]*}"; else echo "$1"; fi; }

if ! have_root "$NEW_ROOT"; then
  echo "NEW_ROOT=$NEW_ROOT has no dpnp/tensor/_tensor_sorting_impl*.so" >&2
  exit 1
fi
if [[ -n $ALGO_ROOT ]] && ! have_switch "$ALGO_ROOT"; then
  echo "ALGO_ROOT=$ALGO_ROOT is not built with algo_switch.py" >&2
  exit 1
fi

# run_dev DEV CMD... runs CMD for one device, selecting the OpenCL CPU for cpu
run_dev() {
  local dev=$1; shift
  local sel=$GPU_SELECTOR
  [[ $dev == cpu ]] && sel=$CPU_SELECTOR
  if [[ -n $sel ]]; then
    ONEAPI_DEVICE_SELECTOR=$sel "$@"
  else
    "$@"
  fi
}

# device_name DEV describes the device DEV runs on and the sizing. The rows
# each case ended up with are on its own line, devinfo.py explains the caps.
device_name() {
  echo "$(run_dev "$1" env PYTHONPATH="$NEW_ROOT" "$PYTHON" "$here/devinfo.py" \
        "$1" 2>&1 | tail -1), ROW_SCALE=${ROW_SCALE:-1}" \
       "MAX_ELEMS=${MAX_ELEMS:-2**28} MEM_FRACTION=${MEM_FRACTION:-0.4}"
}

# bench SUITE "GPU_DTYPES" "CPU_DTYPES" "GPU_CASES" "CPU_CASES" KS VARIANT...
bench() {
  local suite=$1 gdt=$2 cdt=$3 gcases=$4 ccases=$5 ks=$6; shift 6
  local dev dts cases dt out
  for dev in $DEVICES; do
    if [[ $dev == cpu ]]; then
      dts=${CPU_DTYPES:-$cdt}; cases=$ccases
    else
      dts=${GPU_DTYPES:-$gdt}; cases=$gcases
    fi
    out=$RESULTS_DIR/${suite}_$dev.txt
    echo "# $suite on $(device_name "$dev")" | tee -a "$out"
    for dt in $(first "$dts"); do
      echo "== $suite $dev $dt"
      run_dev "$dev" "$here/compare.sh" "$dev" "$dt" "$(first "$cases")" "$ks" "$@"
    done | tee -a "$out"
  done
}

old_variants() {
  local v
  for v in master:MASTER_ROOT base:BASE_ROOT sorted:SORTED_ROOT feat:FEAT_ROOT; do
    local name=${v#*:} root
    root=${!name}
    if have_root "$root"; then echo "${v%%:*}=$root"; else echo "skipping ${v%%:*}, ${v#*:} not set or not built" >&2; fi
  done
}

suite_check() {
  local dev out=$RESULTS_DIR/check.txt
  for dev in $DEVICES; do
    {
      echo "# check on $(device_name "$dev")"
      echo "== check_topk $dev"
      run_dev "$dev" env PYTHONPATH="$NEW_ROOT" "$PYTHON" "$here/check_topk.py" "$dev" ${QUICK:+quick} set
      echo "== check_long $dev"
      run_dev "$dev" env PYTHONPATH="$NEW_ROOT" "$PYTHON" "$here/check_long.py" "$dev"
    } 2>&1 | tee -a "$out"
  done
}

suite_builds() {
  local olds; mapfile -t olds < <(old_variants)
  if ((${#olds[@]} == 0)); then echo "builds: no other root to compare with" >&2; return; fi
  bench builds "f4 i1 f8 i4" "f4 i1 f8" \
    "65536x64,16384x256,4096x1024,256x16384,16x262144,1x4194304,1x2**25" \
    "65536x64,16384x256,4096x1024,256x16384,1x4194304" \
    "1,10,100,1000" "new=$NEW_ROOT" "${olds[@]}"
}

suite_kn() {
  have_root "$MASTER_ROOT" || { echo "kn: MASTER_ROOT not built" >&2; return; }
  local c="262144x16,65536x64,16384x256,4096x1024,1024x4096"
  bench kn "f4 i1 f8 i4" "f4 i1 f8" "$c" "$c" \
    "auto:{n//2, 3*n//4, 7*n//8, n-1, n}" "new=$NEW_ROOT" "master=$MASTER_ROOT"
}

suite_small() {
  have_root "$MASTER_ROOT" || { echo "small: MASTER_ROOT not built" >&2; return; }
  local c="524288x8,262144x16,174762x24,131072x32,87381x48"
  bench small "i2 f4 i4 f8 i8" "i2 f4 f8" "$c" "$c" \
    "auto:{1, n//4, n//2, 3*n//4, n-1, n}" "new=$NEW_ROOT" "master=$MASTER_ROOT"
}

suite_long() {
  have_root "$MASTER_ROOT" || { echo "long: MASTER_ROOT not built" >&2; return; }
  # a few very long rows with k approaching n, the worst case for a selection
  # algorithm: at k = n nothing can be discarded, so it does a full sort's
  # work while still paying for the digit passes. Sized explicitly, so run
  # this one without ROW_SCALE.
  local gc="1x2**22,1x2**24,1x2**26,1x2**28,4x2**24,16x2**22,64x2**20"
  local cc="1x2**22,1x2**24,4x2**24,16x2**22,64x2**20"
  ( export ROW_SCALE=1
    bench long "f4 i1 f8 i4" "f4 i1 f8" "$gc" "$cc" \
      "auto:{n//2, 3*n//4, n-1, n}" "new=$NEW_ROOT" "master=$MASTER_ROOT" )
}

suite_rowsn() {
  [[ -n $ALGO_ROOT ]] || { echo "rowsn: ALGO_ROOT not set" >&2; return; }
  # For 1 byte keys radix select's cost is about 17 ns per row whatever n is,
  # and radix sort's is a smaller per row cost plus a fixed floor, so which
  # wins should depend on the row count, not on n. Every case so far held 4M
  # elements, where those are the same line. Vary them independently: three
  # row counts at fixed n, and three n at fixed row count.
  local v=("select=$ALGO_ROOT:DPNP_TOPK_ALGO=select" "sort=$ALGO_ROOT:DPNP_TOPK_ALGO=sort"
           "merge=$ALGO_ROOT:DPNP_TOPK_ALGO=merge")
  local c="8192x64,32768x64,131072x64,524288x64,2097152x64"
  c+=",8192x256,32768x256,131072x256,524288x256"
  c+=",65536x16,65536x1024,1048576x16,1048576x256"
  ( export ROW_SCALE=1
    bench rowsn "i1 u1 i2 f4" "i1 u1 f4" "$c" "$c" "auto:{1, n}" "${v[@]}" )
}

suite_algo() {
  [[ -n $ALGO_ROOT ]] || { echo "algo: ALGO_ROOT not set" >&2; return; }
  local c="1048576x4,524288x8,349525x12,262144x16,209715x20,174762x24,131072x32"
  c+=",16384x4,8192x8,4096x16,2048x32"
  bench algo "i1 i2 f2 f4 i4 f8 i8" "i1 i2 f4 f8" "$c" "$c" \
    "auto:{1, n//4, n//2, 3*n//4, n-1, n}" \
    "select=$ALGO_ROOT:DPNP_TOPK_ALGO=select" "merge=$ALGO_ROOT:DPNP_TOPK_ALGO=merge"
}

suite_algo3() {
  [[ -n $ALGO_ROOT ]] || { echo "algo3: ALGO_ROOT not set" >&2; return; }
  local v=("select=$ALGO_ROOT:DPNP_TOPK_ALGO=select" "merge=$ALGO_ROOT:DPNP_TOPK_ALGO=merge"
           "sort=$ALGO_ROOT:DPNP_TOPK_ALGO=sort")
  # master uses radix sort for 1 and 2 byte types, which beats radix select on
  # rows of up to 64 of them on some GPUs, "mid" locates the crossover
  local gdt="i1 u1 b1 i2 f4 i4 f8 i8" cdt="i1 u1 i2 f4 i4 f8"
  local short="1048576x4,524288x8,349525x12,262144x16,174762x24,131072x32,87381x48"
  local mid="65536x64,43690x96,32768x128,21845x192,16384x256"
  local near="65536x64,16384x256,4096x1024,1024x4096"
  bench algo3 "$gdt" "$cdt" "$short" "$short" "auto:{1, n//2, n}" "${v[@]}"
  [[ ${QUICK:-} ]] && return
  bench algo3 "$gdt" "$cdt" "$mid" "$mid" "auto:{1, n//4, n//2, 3*n//4, n}" "${v[@]}"
  bench algo3 "$gdt" "$cdt" "$near" "$near" \
    "auto:{n//2, 3*n//4, n-1, n}" "${v[@]}"
}

for s in $suites; do
  if ! declare -F "suite_$s" >/dev/null; then
    echo "unknown suite $s, one of: $suites_all" >&2; exit 1
  fi
done
echo "results in $RESULTS_DIR"
for s in $suites; do
  rm -f "$RESULTS_DIR/${s}_"*.txt; [[ $s == check ]] && rm -f "$RESULTS_DIR/check.txt"
  "suite_$s"
done
