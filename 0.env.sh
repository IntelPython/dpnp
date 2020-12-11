#!/bin/bash

THEDIR=$(dirname $(readlink -e ${BASH_SOURCE[0]}))

# We can not use common setup script because
# using Intel Python brakes build and run procedure
export ONEAPI_ROOT=/opt/intel/oneapi
. ${ONEAPI_ROOT}/mkl/latest/env/vars.sh
. ${ONEAPI_ROOT}/compiler/latest/env/vars.sh
. ${ONEAPI_ROOT}/tbb/latest/env/vars.sh

# TODO: remove when math lib vars.sh script will work properly
# MKLD-10520
export MKLROOT=${ONEAPI_ROOT}/mkl/latest
export LD_LIBRARY_PATH=${MKLROOT}/lib/intel64:$LD_LIBRARY_PATH

export DPCPPROOT=${ONEAPI_ROOT}/compiler/latest

export PYTHONPATH=$PYTHONPATH:${THEDIR}
