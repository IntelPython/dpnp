#!/bin/bash

# if ONEAPI_ROOT is specified (use all from it)
if [ -n "${ONEAPI_ROOT}" ]; then
    export DPCPPROOT=/localdisk/opt/intel/oneapi_b10/compiler/2021.1-beta10
    # TODO uncomment when CI will be changed
    # export MKLROOT=${ONEAPI_ROOT}/mkl/latest
    export TBBROOT=${ONEAPI_ROOT}/tbb/latest
fi

# if DPCPPROOT is specified (work with custom DPCPP)
if [ -n "${DPCPPROOT}" ]; then
    . ${DPCPPROOT}/env/vars.sh
fi

# if MKLROOT is specified (work with custom math library)
if [ -n "${MKLROOT}" ]; then
    . ${MKLROOT}/env/vars.sh
    conda remove mkl --force -y || true
fi

# have to activate while SYCL CPU device/driver needs paths
# if TBBROOT is specified
if [ -n "${TBBROOT}" ]; then
    . ${TBBROOT}/env/vars.sh
fi

$PYTHON setup.py build_clib
$PYTHON setup.py build_ext install
