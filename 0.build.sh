#!/bin/bash
THEDIR=$(dirname $(readlink -e ${BASH_SOURCE[0]}))

# . ${THEDIR}/0.env.sh
cd ${THEDIR}

# Assign $TMP env variable to a directory where the script locates.
# The env variable is used by compiler as a path to temporary folder,
# where it can store a temporary files generated during compilation and linkage phases.
# By default the compiler uses /tmp folder, but it is limited by the size and
# there might be not enough space to temporary keep all generated data.
export TMP=${THEDIR}


export DPNP_DEBUG=1

python setup.py clean
python setup.py build_clib

# inplace build
CC=icpx python setup.py build_ext --inplace

# development build. Root privileges needed
# python setup.py develop

echo
echo =========example3==============
DPCTL_INCLUDES=$(python -m dpctl --includes)
icpx -fsycl -g -O0 -ggdb3 -fPIC dpnp/backend/examples/example3.cpp $DPCTL_INCLUDES -Idpnp -Idpnp/backend/include -Ldpnp -Wl,-rpath='$ORIGIN'/dpnp -ldpnp_backend_c -o example3
# LD_DEBUG=libs,bindings,symbols ./example3
./example3

# gcc --version
# echo =========LD_LIBRARY_PATH==============
# echo $LD_LIBRARY_PATH

# echo =========ldd example3==============
# ldd ./example3
# echo =========readelf example3==============
# readelf -d ./example3
# echo =========ldd dpnp/libdpnp_backend_c.so==============
# ldd ./dpnp/libdpnp_backend_c.so
# echo =========readelf dpnp/libdpnp_backend_c.so==============
# readelf -d ./dpnp/libdpnp_backend_c.so

# echo ========= libstdc++.so ==============
# ls -l /usr/share/miniconda/envs/dpnp*/lib/libstdc++.so
# strings /usr/share/miniconda/envs/dpnp*/lib/libstdc++.so | grep GLIBCXX | sort -n


echo
echo =========example1==============
# LD_DEBUG=libs,bindings,symbols python examples/example1.py
# LD_DEBUG=libs python examples/example1.py
python examples/example1.py

# echo ========= find /opt ==============
# find /opt -name libstdc++.so*
# echo ========= find anaconda ==============
# find /usr/share/miniconda -name libstdc++.so*
# echo ========= dpkg-query -L libstdc++6 ==============
# dpkg-query -L libstdc++6
# echo ========= ls -l /lib/x86_64-linux-gnu/libstdc* ==============
# ls -l /lib/x86_64-linux-gnu/libstdc*

# gcc --version
# g++ --version
# dpcpp --version

# echo ========= APT ==============
# apt list --installed
# echo ========= conda ==============
# conda list
