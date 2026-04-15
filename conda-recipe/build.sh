#!/bin/bash

# Test reproducer:
echo "building ..."
icpx -fsycl --gcc-install-dir=$BUILD_PREFIX/lib/gcc/x86_64-conda-linux-gnu/14.3.0 --sysroot=$BUILD_PREFIX/x86_64-conda-linux-gnu/sysroot test_minimal.cpp -o test_minimal
echo "build is completed, run now ..."
./test_minimal
echo "run is done"

echo "create tmp folder: $SRC_DIR/tmp"
mkdir -p $SRC_DIR/tmp
echo "run with dump enabled ..."
export SYCL_CACHE_DISABLE=1
export IGC_ShaderDumpEnable=1
export IGC_ShaderDumpEnableAll=1
export IGC_DumpToCustomDir=$SRC_DIR/tmp/
./test_minimal

echo "waiting for .asm files..."
timeout=5
while [ $timeout -gt 0 ]; do
    if find $SRC_DIR/tmp -name "*.asm" -print -quit | grep -q .; then
        echo "found .asm files"
        break
    fi
    sleep 1
    ((timeout--))
done

echo "list files..."
ls -la $SRC_DIR/tmp
echo "print dump:"
find $SRC_DIR/tmp -name "*.asm"
find $SRC_DIR/tmp -name "*.asm" | head -n 1 | xargs -r cat
echo "test is complete"

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
