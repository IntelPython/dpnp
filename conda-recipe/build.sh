#!/bin/bash

# if ONEAPI_ROOT is specified (use all from it)
if [ ! -z "${ONEAPI_ROOT}" ]; then
    # have to activate beta09 because mkl-2021.1b10-intel_2147 package use lib from it
    . ${ONEAPI_ROOT}/compiler/2021.1-beta09/env/vars.sh
    export DPCPPROOT=${ONEAPI_ROOT}/compiler/latest
    # TODO uncomment when CI will be changed
    # export MKLROOT=${ONEAPI_ROOT}/mkl/latest
    export TBBROOT=${ONEAPI_ROOT}/tbb/latest
fi

# if DPCPPROOT is specified (work with custom DPCPP)
if [ ! -z "${DPCPPROOT}" ]; then
    . ${DPCPPROOT}/env/vars.sh
fi

# if MKLROOT is specified (work with custom MKL)
if [ ! -z "${MKLROOT}" ]; then
    . ${MKLROOT}/env/vars.sh
    conda remove mkl --force -y || true
fi

# have to activate while SYCL CPU device/driver needs paths
# if TBBROOT is specified
if [ ! -z "${TBBROOT}" ]; then
    . ${TBBROOT}/env/vars.sh
fi

$PYTHON setup.py build_clib
$PYTHON setup.py build_ext install
