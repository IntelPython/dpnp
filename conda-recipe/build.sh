#!/bin/bash

# This is necessary to help DPC++ find Intel libraries such as SVML, IRNG, etc in build prefix
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${BUILD_PREFIX}/lib"

# Intel LLVM must cooperate with compiler and sysroot from conda
echo "--gcc-toolchain=${BUILD_PREFIX} --sysroot=${BUILD_PREFIX}/${HOST}/sysroot -target ${HOST}" > icpx_for_conda.cfg

ICPXCFG="$(pwd)/icpx_for_conda.cfg"
export ICPXCFG

ICXCFG="$(pwd)/icpx_for_conda.cfg"
export ICXCFG

export CMAKE_GENERATOR="Ninja"
export TBB_ROOT_HINT=$PREFIX
export DPL_ROOT_HINT=$PREFIX
export MKL_ROOT_HINT=$PREFIX
SKBUILD_ARGS=(-- "-DCMAKE_C_COMPILER:PATH=icx" "-DCMAKE_CXX_COMPILER:PATH=icpx" "-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON")
SKBUILD_ARGS=("${SKBUILD_ARGS[@]}" "-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON")

# Build wheel package
WHEELS_BUILD_ARGS=("-p" "manylinux_2_28_x86_64")
if [ -n "${WHEELS_OUTPUT_FOLDER}" ]; then
    $PYTHON setup.py install bdist_wheel "${WHEELS_BUILD_ARGS[@]}" "${SKBUILD_ARGS[@]}"
    cp dist/dpnp*.whl "${WHEELS_OUTPUT_FOLDER}"
else
    $PYTHON setup.py install "${SKBUILD_ARGS[@]}"
fi
