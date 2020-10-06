#!/bin/bash

#. $RECIPE_DIR/activate_env.sh

if [ ! -z "${ONEAPI_ROOT}" ]; then
    . ${ONEAPI_ROOT}/mkl/latest/env/vars.sh
    . ${ONEAPI_ROOT}/compiler/2021.1-beta09/env/vars.sh
    . ${ONEAPI_ROOT}/tbb/latest/env/vars.sh
fi

# if MKLROOT is specified (build with custom MKL)
if [ ! -z "${MKLROOT}" ]; then
    conda remove mkl --force -y || true
fi
