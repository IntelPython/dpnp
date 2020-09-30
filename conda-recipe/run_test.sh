#!/bin/bash

#. $RECIPE_DIR/activate_env.sh

if [ ! -z "${ONEAPI_ROOT}" ]; then
    . ${ONEAPI_ROOT}/mkl/latest/env/vars.sh
    . ${ONEAPI_ROOT}/compiler/latest/env/vars.sh
    . ${ONEAPI_ROOT}/tbb/latest/env/vars.sh
fi

# if MKLROOT is specified (build with custom MKL)
if [ ! -z "${MKLROOT}" ]; then
    conda remove mkl --force -y || true
fi

set -x

##teamcity[testStarted name='dummyTestName' captureStandardOutput='true']

# python setup.py clean
# python setup.py build_clib

# inplace build
# python setup.py build_ext --inplace

# development build. Root privileges needed
# python setup.py develop

PROJ_ROOT=".."

echo
echo =========env==============
pwd
sudo apt-get install -y clinfo
clinfo
clinfo -l
echo ${SHELL}
python --version
clang++ --version
echo ${LD_LIBRARY_PATH}
echo ${PATH}

# readelf -d ${PROJ_ROOT}/dpnp/libdpnp_backend_c.so
# ldd ${PROJ_ROOT}/dpnp/libdpnp_backend_c.so

# readelf -d ${PROJ_ROOT}/dpnp/dparray.cpython*
# ldd ${PROJ_ROOT}/dpnp/dparray.cpython*

# echo =========example3==============
# clang++ -g -fPIC ${PROJ_ROOT}/examples/example3.cpp -I${PROJ_ROOT}/dpnp -I${PROJ_ROOT}/dpnp/backend -L${PROJ_ROOT}/dpnp -Wl,-rpath='$ORIGIN'/dpnp -ldpnp_backend_c -o example3
# ./example3

# echo
# echo =========example1==============
# python examples/example1.py

python -c "import dpnp"
