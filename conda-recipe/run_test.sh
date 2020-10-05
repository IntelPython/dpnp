#!/bin/bash

#. $RECIPE_DIR/activate_env.sh

if [ ! -z "${ONEAPI_ROOT}" ]; then
    . ${ONEAPI_ROOT}/compiler/latest/env/vars.sh
fi

# if MKLROOT is specified (build with custom MKL)
if [ ! -z "${MKLROOT}" ]; then
    conda remove mkl --force -y || true
fi
