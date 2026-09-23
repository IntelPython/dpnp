# shellcheck shell=bash
# Sourced by run_all.sh, every value can be overridden from the environment.

# The conda environment dpnp was built in, the active one by default. With
# DPC++ installed from conda it provides the compiler and the SYCL runtime.
: "${CONDA_ENV:=${CONDA_PREFIX:-$HOME/miniforge3/envs/dpnp_dev}}"
# Optional, a oneAPI setvars.sh to source first, for a standalone oneAPI
# install, e.g. /opt/intel/oneapi/setvars.sh. Empty sources nothing.
: "${ONEAPI_SETVARS:=}"

# Build roots, each a directory holding a "dpnp" package that is put on
# PYTHONPATH (see make_root.sh). An empty or missing root skips what needs it,
# set one to empty to skip it.
#   NEW_ROOT     the radix select build being measured
#   MASTER_ROOT  master, merge sort (radix sort for 1 and 2 byte types)
#   BASE_ROOT    any earlier radix select build, e.g. the first commit
#   SORTED_ROOT  a radix select build that still sorted its output
#   FEAT_ROOT    the radix select feature branch
#   ALGO_ROOT    NEW_ROOT built with algo_switch.py applied
: "${NEW_ROOT=$HOME/repos/dpnp-topk}"
: "${MASTER_ROOT=/tmp/topk_work/master_root}"
: "${BASE_ROOT=/tmp/topk_work/base_root}"
: "${SORTED_ROOT=/tmp/topk_work/sorted_root}"
: "${FEAT_ROOT=/tmp/topk_work/feat_root}"
: "${ALGO_ROOT=}"

# Devices to run on, "gpu" and "cpu". Each runs with its ONEAPI_DEVICE_SELECTOR,
# an empty GPU_SELECTOR keeps the default GPU. On a machine with an
# integrated and a discrete GPU pick one, e.g. GPU_SELECTOR=level_zero:1,
# `sycl-ls` lists the devices.
: "${DEVICES:=gpu cpu}"
: "${GPU_SELECTOR:=}"
: "${CPU_SELECTOR:=opencl:cpu}"
# where the result files go
: "${RESULTS_DIR:=$(pwd)/results}"
# per-run timeout in seconds, and timed calls per run (sweep.py)
: "${TIMEOUT:=900}"
: "${SWEEP_NIT:=15}"
export TIMEOUT SWEEP_NIT

# setup_env puts the conda environment first on PATH and LD_LIBRARY_PATH,
# after sourcing ONEAPI_SETVARS if one is set
setup_env() {
  if [[ -n $ONEAPI_SETVARS ]]; then
    if [[ ! -f $ONEAPI_SETVARS ]]; then
      echo "ONEAPI_SETVARS=$ONEAPI_SETVARS not found" >&2; return 1
    fi
    # setvars.sh reads unset variables
    set +u; source "$ONEAPI_SETVARS" --force >/dev/null 2>&1; set -u
  fi
  export LD_LIBRARY_PATH=$CONDA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  export PATH=$CONDA_ENV/bin:$PATH
}
