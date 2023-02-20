#!/bin/bash

# if ONEAPI_ROOT is specified (use all from it)
if [ -n "${ONEAPI_ROOT}" ]; then
    export DPCPPROOT=${ONEAPI_ROOT}/compiler/latest
    export MKLROOT=${ONEAPI_ROOT}/mkl/latest
    export TBBROOT=${ONEAPI_ROOT}/tbb/latest
    export DPLROOT=${ONEAPI_ROOT}/dpl/latest
fi

# if DPCPPROOT is specified (work with custom DPCPP)
if [ -n "${DPCPPROOT}" ]; then
    . ${DPCPPROOT}/env/vars.sh
fi

# if MKLROOT is specified (work with custom math library)
if [ -n "${MKLROOT}" ]; then
    . ${MKLROOT}/env/vars.sh
    # conda remove mkl --force -y || true
fi

# have to activate while SYCL CPU device/driver needs paths
# if TBBROOT is specified
if [ -n "${TBBROOT}" ]; then
    . ${TBBROOT}/env/vars.sh
fi

# Set RPATH for wheels
export CFLAGS="-Wl,-rpath,\$ORIGIN/../dpctl,-rpath,\$ORIGIN $CFLAGS"
export LDFLAGS="-Wl,-rpath,\$ORIGIN/../dpctl,-rpath,\$ORIGIN $LDFLAGS"

# Intel LLVM must cooperate with compiler and sysroot from conda
echo "--gcc-toolchain=${BUILD_PREFIX} --sysroot=${BUILD_PREFIX}/${HOST}/sysroot -target ${HOST}" > icpx_for_conda.cfg
export ICPXCFG="$(pwd)/icpx_for_conda.cfg"
export ICXCFG="$(pwd)/icpx_for_conda.cfg"

$PYTHON setup.py build_clib
$PYTHON setup.py build_ext install

# Build wheel package
if [ "$CONDA_PY" == "36" ]; then
    WHEELS_BUILD_ARGS="-p manylinux1_x86_64"
else
    WHEELS_BUILD_ARGS="-p manylinux2014_x86_64"
fi
if [ -n "${WHEELS_OUTPUT_FOLDER}" ]; then
    $PYTHON setup.py bdist_wheel ${WHEELS_BUILD_ARGS}
    cp dist/dpnp*.whl ${WHEELS_OUTPUT_FOLDER}
fi
