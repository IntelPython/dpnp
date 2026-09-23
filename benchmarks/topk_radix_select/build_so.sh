#!/bin/bash
# usage: build_so.sh SRC BUILD_DIR
#
# Configures a separate CMake build of the dpnp checkout SRC (e.g. a
# `git worktree` of master or of another branch) and builds only the sorting
# extension, printing the path of the resulting .so for make_root.sh. Takes
# about 5 minutes, don't benchmark while it runs.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
source "$here/config.sh"
src=$(cd "$1" && pwd); build=$2
E=$CONDA_ENV
setup_env
py=$E/bin/python
pyver=$($py -c "import sys, sysconfig; print(f'python{sys.version_info.major}.{sys.version_info.minor}' + sysconfig.get_config_var('ABIFLAGS'))")
site=$($py -c "import sysconfig; print(sysconfig.get_path('purelib'))")
npinc=$($py -c "import numpy; print(numpy.get_include())")
# the dpctl .pxd files have to be visible to Cython
export PYTHONPATH=$site${PYTHONPATH:+:$PYTHONPATH}
if [[ ! -f $build/build.ninja ]]; then
  cmake -S "$src" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_MODULE_PATH="$site/skbuild/resources/cmake" \
    -DDpctl_ROOT="$site/dpctl/resources/cmake" \
    -DPython_EXECUTABLE="$py" -DPython3_EXECUTABLE="$py" -DPYTHON_EXECUTABLE="$py" \
    -DPython_ROOT_DIR="$E" -DPython3_ROOT_DIR="$E" \
    -DPython_INCLUDE_DIR="$E/include/$pyver" -DPython3_INCLUDE_DIR="$E/include/$pyver" \
    -DPYTHON_INCLUDE_DIR="$E/include/$pyver" -DPYTHON_LIBRARY="$E/lib/lib$pyver.so" \
    -DPython_NumPy_INCLUDE_DIRS="$npinc" -DPython3_NumPy_INCLUDE_DIRS="$npinc" \
    -DCYTHON_EXECUTABLE="$E/bin/cython" -DPython_FIND_REGISTRY=NEVER -DSKBUILD=TRUE
fi
ninja -C "$build" _tensor_sorting_impl
find "$build" -name "_tensor_sorting_impl*.so" -newer "$build/build.ninja" | head -1
