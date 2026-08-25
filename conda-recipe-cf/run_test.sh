#!/bin/bash

# Skip tensor tests by default to avoid OOM in conda builds.
# Set SKIP_TENSOR_TESTS=0 to run them on machines with enough memory.
if [ -z "${SKIP_TENSOR_TESTS}" ]; then
    export SKIP_TENSOR_TESTS=1
fi

set -e

$PYTHON -c "import dpnp; print(dpnp.__version__)"
$PYTHON -m dpctl -f
$PYTHON -m pytest -ra --pyargs dpnp
