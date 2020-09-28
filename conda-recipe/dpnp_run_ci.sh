#!/bin/bash

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
echo ${SHELL}
python --version
clang++ varsion
echo ${LD_LIBRARY_PATH}
echo ${PATH}

readelf -d ${PROJ_ROOT}/dpnp/libdpnp_backend_c.so
ldd ${PROJ_ROOT}/dpnp/libdpnp_backend_c.so

readelf -d ${PROJ_ROOT}/dpnp/dparray.cpython*
ldd ${PROJ_ROOT}/dpnp/dparray.cpython*

echo =========example3==============
clang++ -g -fPIC ${PROJ_ROOT}/examples/example3.cpp -I${PROJ_ROOT}/dpnp -I${PROJ_ROOT}/dpnp/backend -L${PROJ_ROOT}/dpnp -Wl,-rpath='$ORIGIN'/dpnp -ldpnp_backend_c -o example3
./example3

echo
echo =========example1==============
python examples/example1.py

python -c "import dpnp"
