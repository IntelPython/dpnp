#!/bin/bash

# if ONEAPI_ROOT is specified (use all from it)
if [ -n "${ONEAPI_ROOT}" ]; then
    export DPCPPROOT=${ONEAPI_ROOT}/compiler/latest
    export MKLROOT=${ONEAPI_ROOT}/mkl/latest
    export TBBROOT=${ONEAPI_ROOT}/tbb/latest
    export DPLROOT=${ONEAPI_ROOT}/dpl/latest
fi

# if DPCPPROOT is specified (work with custom DPCPP)
if [ -n "${DPCPPROOT}" ]; then
    . ${DPCPPROOT}/env/vars.sh
fi

# if MKLROOT is specified (work with custom math library)
if [ -n "${MKLROOT}" ]; then
    . ${MKLROOT}/env/vars.sh
fi

# have to activate while SYCL CPU device/driver needs paths
# if TBBROOT is specified
if [ -n "${TBBROOT}" ]; then
    . ${TBBROOT}/env/vars.sh
fi
