#!/bin/bash
# usage: compare.sh DEV DTYPE CASES KS VARIANT [VARIANT ...]
#
#   CASES    "ROWSxN,ROWSxN" as in sweep.py
#   KS       "1,10,100", or "auto:EXPR" computed per case from n, e.g.
#            "auto:{1, n//2, n}" (a python set, 0 and k > n are dropped)
#   VARIANT  LABEL=ROOT[:VAR=val;VAR=val], ROOT is put on PYTHONPATH and the
#            optional environment is passed to sweep.py, e.g.
#            new=/path/root  merge=/path/root:DPNP_TOPK_ALGO=merge
#
# Every (case, k, variant) runs in a fresh process, a long in-process sweep
# degrades later timings on some GPUs. Prints the time in ms per variant,
# the ratio of every other variant to the first one (> 1 means the first one
# is faster) and the fastest label. ROW_SCALE and MAX_ELEMS are passed on to
# sweep.py, the rows printed are the scaled ones.
set -u
here=$(cd "$(dirname "$0")" && pwd)
dev=$1; dt=$2; cases=$3; ks=$4; shift 4
labels=(); roots=(); envs=()
for v in "$@"; do
  labels+=("${v%%=*}"); rest=${v#*=}
  roots+=("${rest%%:*}")
  if [[ $rest == *:* ]]; then envs+=("${rest#*:}"); else envs+=(""); fi
done
py=${PYTHON:-python}
# An orphaned loop or sweep keeps the device busy and quietly inflates
# whatever is measured next, so go down with the driver that started this.
driver=$PPID
hdr="rows n k ${labels[*]}"
for ((i = 1; i < ${#labels[@]}; i++)); do hdr+=" ${labels[i]}/${labels[0]}"; done
echo "$hdr best"
for c in ${cases//,/ }; do
  rows=${c%%x*}; n=$($py -c "print(int(${c#*x}))")
  if [[ $ks == auto:* ]]; then
    kk=$($py -c "n=$n; print(','.join(map(str, sorted(k for k in (${ks#auto:}) if 0 < k <= n))))")
  else
    kk=$ks
  fi
  for k in ${kk//,/ }; do
    ((k > n)) && continue
    ts=()
    for ((i = 0; i < ${#labels[@]}; i++)); do
      out=$(mktemp)
      PYTHONPATH=${roots[i]} timeout "${TIMEOUT:-900}" $py "$here/sweep.py" \
            "$dev" "$dt" "$c" "$k" "t:${envs[i]}" >"$out" 2>&1 &
      sweep=$!
      while kill -0 "$sweep" 2>/dev/null; do
        if ! kill -0 "$driver" 2>/dev/null; then
          kill "$sweep" 2>/dev/null  # timeout passes it on to python
          rm -f "$out"
          echo "compare.sh: driver $driver is gone, stopping" >&2
          exit 1
        fi
        sleep 0.25
      done
      line=$(tail -1 "$out")
      skipped=$(grep -c '^skipping ' "$out")
      read -r r _ _ t <<< "$line"
      # a case over the element cap prints only sweep.py's header, so check
      # that the line really starts with a row count
      if [[ $(wc -w <<< "$line") == 4 && $r =~ ^[0-9]+$ ]]; then
        ts+=("$t"); rows=$r  # sweep.py reports the rows it used (ROW_SCALE)
      elif ((skipped > 0)); then
        ts+=("SKIP")
      else
        ts+=("ERR")
        # why it failed exists only in this process's output, so pass it on
        # instead of leaving a bare ERR in the results
        { echo "ERR ${labels[i]} $dt $c k=$k:"; tail -5 "$out"; } >&2
      fi
      rm -f "$out"
    done
    echo "$rows $n $k ${ts[*]}" | awk -v labels="${labels[*]}" '{
      nl = split(labels, lab, " "); line = $1 " " $2 " " $3; best = ""; bt = 0; bad = ""
      for (i = 1; i <= nl; i++) {
        t = $(3 + i); line = line sprintf(" %9s", t)
        if (t !~ /^[0-9]/) { bad = t; continue }   # ERR or SKIP
        if (best == "" || t + 0 < bt) { best = lab[i]; bt = t + 0 }
      }
      for (i = 2; i <= nl; i++) {
        t0 = $4; t = $(3 + i)
        if (t0 !~ /^[0-9]/ || t !~ /^[0-9]/ || t0 + 0 == 0) line = line "      -"
        else line = line sprintf(" %6.2fx", t / t0)
      }
      print line "  " best (bad ? " (" bad ")" : "")
    }'
  done
done
