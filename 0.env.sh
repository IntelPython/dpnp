#!/bin/bash

# We can not use common setup script because
# using Intel Python brakes build and run procedure
export ONEAPI_ROOT=/opt/intel/oneapi
. ${ONEAPI_ROOT}/mkl/latest/env/vars.sh
. ${ONEAPI_ROOT}/compiler/latest/env/vars.sh
. ${ONEAPI_ROOT}/tbb/latest/env/vars.sh
