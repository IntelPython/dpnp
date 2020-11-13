#!/bin/bash

THEDIR=$(dirname $(readlink -e ${BASH_SOURCE[0]}))

# We can not use common setup script because
# using Intel Python brakes build and run procedure
export ONEAPI_ROOT=/opt/intel/oneapi
# Temporary use MKLVARS due to MKLD-10520
export MKLVARS=${ONEAPI_ROOT}/mkl/latest/env
. ${MKLVARS}/vars.sh
. ${ONEAPI_ROOT}/compiler/latest/env/vars.sh
. ${ONEAPI_ROOT}/dpl/latest/env/vars.sh
. ${ONEAPI_ROOT}/tbb/latest/env/vars.sh

export DPCPPROOT=${ONEAPI_ROOT}/compiler/latest

export PYTHONPATH=$PYTHONPATH:${THEDIR}
