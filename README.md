[![Build Status](https://dev.azure.com/IntelPython/dpnp/_apis/build/status/IntelPython.dpnp?branchName=master)](https://dev.azure.com/IntelPython/dpnp/_build/latest?definitionId=6&branchName=master)
[![codecov](https://codecov.io/gh/IntelPython/dpnp/branch/master/graph/badge.svg)](https://codecov.io/gh/IntelPython/dpnp)
![Build Sphinx](https://github.com/IntelPython/dpnp/workflows/Build%20Sphinx/badge.svg)

# DPNP: NumPy-like API accelerated with SYCL

Full documentation: https://intelpython.github.io/dpnp/

DPNP C++ backend documentation: https://intelpython.github.io/dpnp/backend_doc/

The project contains:
- Python interface with NumPy-like API
- C++ library with SYCL based kernels

## How to run
By default main CPU SYCL queue is used. To use Intel GPU please use:
```python
DPNP_QUEUE_GPU=1 python examples/example1.py
```

## Build from source:
```
git clone https://github.com/IntelPython/dpnp
cd dpnp
./0.build.sh
```

## Run test
```python
. ./0.env
pytest
# or
pytest tests/test_matmul.py -s -v
# or
python -m unittest tests/test_mixins.py
```

## Run numpy test
```
. ./0.env
python -m tests.third_party.numpy
# or
python -m tests.third_party.numpy core/tests/test_umath.py
# or
python -m tests.third_party.numpy core/tests/test_umath.py::TestHypot::test_simple
```

### Building documentation:
```
Prerequisites:
$ conda install sphinx sphinx_rtd_theme
Building:
1. Install dpnp into your python environment
2. $ cd doc && make html
3. The documentation will be in doc/_build/html
```

## Packaging:
```
. ./0.env
conda-build conda-recipe/
```

## Run benchmark:
```
$ cd benchmarks/

$ asv run --python=python --bench <filename without .py>
example:
$ asv run --python=python --bench bench_elementwise

or

$ asv run --python=python --bench <class>.<bench>
example:
$ asv run --python=python --bench Elementwise.time_square

add --quick option to run every case once
but looks like first execution has additional overheads and takes a lot of time (need to be investigated)
```
