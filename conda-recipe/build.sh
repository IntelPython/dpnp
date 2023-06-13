#!/bin/bash

# Workaround to Klocwork overwriting LD_LIBRARY_PATH that was modified
# by DPC++ compiler conda packages. Will need to be added to DPC++ compiler
# activation scripts.
export LDFLAGS="$LDFLAGS -Wl,-rpath,$PREFIX/lib"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PREFIX/compiler/lib/intel64_lin:$PREFIX/compiler/lib:$PREFIX/lib"

# Intel LLVM must cooperate with compiler and sysroot from conda
echo "--gcc-toolchain=${BUILD_PREFIX} --sysroot=${BUILD_PREFIX}/${HOST}/sysroot -target ${HOST}" > icpx_for_conda.cfg
export ICPXCFG="$(pwd)/icpx_for_conda.cfg"
export ICXCFG="$(pwd)/icpx_for_conda.cfg"

export CMAKE_GENERATOR="Ninja"
export TBB_ROOT_HINT=$PREFIX
export DPL_ROOT_HINT=$PREFIX
export MKL_ROOT_HINT=$PREFIX
SKBUILD_ARGS="-- -DDPCTL_MODULE_PATH=$($PYTHON -m dpctl --cmakedir) "
SKBUILD_ARGS="${SKBUILD_ARGS} -DCMAKE_C_COMPILER:PATH=icx -DCMAKE_CXX_COMPILER:PATH=icpx"
SKBUILD_ARGS="${SKBUILD_ARGS} -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON"

# Build wheel package
if [ "$CONDA_PY" == "36" ]; then
    WHEELS_BUILD_ARGS="-p manylinux1_x86_64"
else
    WHEELS_BUILD_ARGS="-p manylinux2014_x86_64"
fi
if [ -n "${WHEELS_OUTPUT_FOLDER}" ]; then
    $PYTHON setup.py install bdist_wheel ${WHEELS_BUILD_ARGS} ${SKBUILD_ARGS}
    cp dist/dpnp*.whl ${WHEELS_OUTPUT_FOLDER}
else
    $PYTHON setup.py install ${SKBUILD_ARGS}
fi
