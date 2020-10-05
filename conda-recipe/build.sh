#!/bin/bash

#. $RECIPE_DIR/activate_env.sh

if [ ! -z "${ONEAPI_ROOT}" ]; then
    . ${ONEAPI_ROOT}/compiler/latest/env/vars.sh
    . ${ONEAPI_ROOT}/mkl/latest/env/vars.sh
    # export MKLROOT=${CONDA_PREFIX}
fi

$PYTHON setup.py build_clib
$PYTHON setup.py build_ext install
