#!/bin/bash
THEDIR=$(dirname $(readlink -e ${BASH_SOURCE[0]}))

. ${THEDIR}/0.env.sh
cd ${THEDIR}

export DPNP_DEBUG=1

python setup.py clean
python setup.py build_clib

# inplace build
python setup.py build_ext --inplace

# development build. Root privileges needed
# python setup.py develop

echo
echo =========example3==============
clang++ -g -fPIC dpnp/backend/examples/example3.cpp -Idpnp -Idpnp/backend/include -Ldpnp -Wl,-rpath='$ORIGIN'/dpnp -ldpnp_backend_c -o example3
./example3

echo =========LD_LIBRARY_PATH==============
echo $LD_LIBRARY_PATH

echo =========ldd example3==============
ldd ./example3
echo =========readelf example3==============
readelf -d ./example3
echo =========ldd dpnp/libdpnp_backend_c.so==============
ldd ./dpnp/libdpnp_backend_c.so
echo =========readelf dpnp/libdpnp_backend_c.so==============
readelf -d ./dpnp/libdpnp_backend_c.so

echo
echo =========example1==============
python examples/example1.py
