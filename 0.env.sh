#!/bin/bash

THEDIR=$(dirname $(readlink -e ${BASH_SOURCE[0]}))

# We can not use common setup script because
# using Intel Python brakes build and run procedure
export ONEAPI_ROOT=/opt/intel/oneapi

. ${ONEAPI_ROOT}/compiler/latest/env/vars.sh
. ${ONEAPI_ROOT}/tbb/latest/env/vars.sh

if true
then
    # Temporary use explicit version (arg_verz) due to MKLD-10520
    arg_verz=latest
    . ${ONEAPI_ROOT}/mkl/latest/env/vars.sh
    unset arg_verz
else
    . ${ONEAPI_ROOT}/mkl/latest/env/vars.sh
fi

export DPCPPROOT=${ONEAPI_ROOT}/compiler/latest

export PYTHONPATH=$PYTHONPATH:${THEDIR}
