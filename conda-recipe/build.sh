#!/bin/bash

#. $RECIPE_DIR/activate_env.sh

if [ ! -z "${ONEAPI_ROOT}" ]; then
    . ${ONEAPI_ROOT}/mkl/latest/env/vars.sh
    . ${ONEAPI_ROOT}/compiler/2021.1-beta09/env/vars.sh
    . ${ONEAPI_ROOT}/tbb/latest/env/vars.sh
fi

$PYTHON setup.py build_clib
#cp $SRC_DIR/dpnp/libdpnp_backend_c.so $SP_DIR/libdpnp_backend_c.so
$PYTHON setup.py build_ext install
