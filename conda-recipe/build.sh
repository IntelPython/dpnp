#!/bin/bash

# This is necessary to help DPC++ find Intel libraries such as SVML, IRNG, etc in build prefix
export LIBRARY_PATH="$LIBRARY_PATH:${BUILD_PREFIX}/lib"

# Intel LLVM must cooperate with compiler and sysroot from conda
echo "--gcc-toolchain=${BUILD_PREFIX} --sysroot=${BUILD_PREFIX}/${HOST}/sysroot -target ${HOST}" > icpx_for_conda.cfg

ICPXCFG="$(pwd)/icpx_for_conda.cfg"
export ICPXCFG

ICXCFG="$(pwd)/icpx_for_conda.cfg"
export ICXCFG

# Get sysroot glibc version from the build env metadata (no conda CLI needed)
SYSROOT_META=("${BUILD_PREFIX}"/conda-meta/sysroot_linux-64-*.json)
if [[ "${SYSROOT_META[0]##*/}" =~ ^sysroot_linux-64-([0-9]+)\.([0-9]+) ]]; then
    GLIBC_MAJOR="${BASH_REMATCH[1]}"
    GLIBC_MINOR="${BASH_REMATCH[2]}"
else
    echo "Error: unable to detect sysroot_linux-64 version in ${BUILD_PREFIX}" >&2
    exit 1
fi

if [ -e "_skbuild" ]; then
    ${PYTHON} setup.py clean --all
fi

export CC=icx
export CXX=icpx

# conda-forge's gcc/g++ activation injects -fno-merge-constants into CFLAGS/CXXFLAGS
# when CONDA_BUILD==1, while icx/icpx doesn't support it and emit the warning.
# See CMPLRLLVM-19167 / CMPLRLLVM-28729 for more context and PR which added the flag:
# https://github.com/conda-forge/ctng-compiler-activation-feedstock/pull/193
export CFLAGS="${CFLAGS//-fno-merge-constants/}"
export CXXFLAGS="${CXXFLAGS//-fno-merge-constants/}"

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
