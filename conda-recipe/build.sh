#!/bin/bash

# This is necessary to help DPC++ find Intel libraries such as SVML, IRNG, etc in build prefix
export LIBRARY_PATH="$LIBRARY_PATH:${BUILD_PREFIX}/lib"

# Intel LLVM must cooperate with compiler and sysroot from conda
echo "--gcc-toolchain=${BUILD_PREFIX} --sysroot=${BUILD_PREFIX}/${HOST}/sysroot -target ${HOST}" > icpx_for_conda.cfg

ICPXCFG="$(pwd)/icpx_for_conda.cfg"
export ICPXCFG

ICXCFG="$(pwd)/icpx_for_conda.cfg"
export ICXCFG

read -r GLIBC_MAJOR GLIBC_MINOR <<<"$(conda list '^sysroot_linux-64$' \
    | tail -n 1 | awk '{print $2}' | grep -oP '\d+' | head -n 2 | tr '\n' ' ')"

if [ -e "_skbuild" ]; then
    ${PYTHON} setup.py clean --all
fi

export CC=icx
export CXX=icpx

export CMAKE_GENERATOR=Ninja
# Make CMake verbose
export VERBOSE=1

CMAKE_ARGS="${CMAKE_ARGS} -DDPNP_WITH_REDIST:BOOL=ON"

# -wnx flags mean: --wheel --no-isolation --skip-dependency-check
${PYTHON} -m build -w -n -x

${PYTHON} -m wheel tags --remove --build "$GIT_DESCRIBE_NUMBER" \
    --platform-tag "manylinux_${GLIBC_MAJOR}_${GLIBC_MINOR}_x86_64" \
    dist/dpnp*.whl

${PYTHON} -m pip install dist/dpnp*.whl \
    --no-build-isolation \
    --no-deps \
    --only-binary :all: \
    --no-index \
    --prefix "${PREFIX}" \
    -vv

# Copy wheel package
if [[ -d "${WHEELS_OUTPUT_FOLDER}" ]]; then
    cp dist/dpnp*.whl "${WHEELS_OUTPUT_FOLDER[@]}"
fi
