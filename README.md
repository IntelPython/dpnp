[![Build Status](https://dev.azure.com/IntelPython/inp/_apis/build/status/shssf.inumpy?branchName=master)](https://dev.azure.com/IntelPython/inp/_build/latest?definitionId=5&branchName=master)
[![codecov](https://codecov.io/gh/shssf/inumpy/branch/master/graph/badge.svg?token=4RQYO6AFDN)](https://codecov.io/gh/shssf/inumpy)

# DPNP: NumPy-like API accelerated with SYCL

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
git clone <this repo>
cd dpnp
. /opt/intel/oneapi/setvars.sh
# or
. ./0.env.sh

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

run tests from config:
```
python tests/run_pytest.py tests/config/example.yaml tests/config/deselect_failed.yaml -v --tb=no
tests/run_pytest.py - runner
tests/config/example.yaml tests/config/deselect_failed.yaml - one or several configs *.yaml
-v --tb=no - any usual pytest args
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
