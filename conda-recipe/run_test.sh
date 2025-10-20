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
    # shellcheck source=/dev/null
    . "${DPCPPROOT}"/env/vars.sh
fi

# if MKLROOT is specified (work with custom math library)
if [ -n "${MKLROOT}" ]; then
    # shellcheck source=/dev/null
    . "${MKLROOT}"/env/vars.sh
fi

# have to activate while SYCL CPU device/driver needs paths
# if TBBROOT is specified
if [ -n "${TBBROOT}" ]; then
    # shellcheck source=/dev/null
    . "${TBBROOT}"/env/vars.sh
fi

# If PYTHON is not set
# assign it to the Python interpreter from the testing environment
if [ -z "${PYTHON}" ]; then
    PYTHON=$PREFIX/bin/python
fi

set -e

$PYTHON -c "import dpnp; print(dpnp.__version__)"
$PYTHON -m dpctl -f

timeout 10m gdb --batch -ex run -ex 'info sharedlibrary' -ex 'set print elements 1000' -ex thread apply all bt --args "$PYTHON" -m pytest -ra --disable-warnings --pyargs dpnp.tests.test_histogram || true
$PYTHON -m pytest -sv --pyargs dpnp.tests.test_histogram

$PYTHON -m pytest -ra --pyargs dpnp
