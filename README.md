[![Build Status](https://dev.azure.com/IntelPython/dpnp/_apis/build/status/IntelPython.dpnp?branchName=master)](https://dev.azure.com/IntelPython/dpnp/_build/latest?definitionId=6&branchName=master)
[![codecov](https://codecov.io/gh/IntelPython/dpnp/branch/master/graph/badge.svg)](https://codecov.io/gh/IntelPython/dpnp)
[![Build Sphinx](https://github.com/IntelPython/dpnp/workflows/Build%20Sphinx/badge.svg)](https://intelpython.github.io/dpnp)

# DPNP - Data Parallel Extension for NumPy*
[API coverage summary](https://intelpython.github.io/dpnp/reference/comparison.html#summary)

[Full documentation](https://intelpython.github.io/dpnp/)

[DPNP C++ backend documentation](https://intelpython.github.io/dpnp/backend_doc/)

## Build from source:
Ensure you have the following prerequisite packages installed:

- `mkl-devel-dpcpp`
- `dpcpp_linux-64` or `dpcpp_win-64` (depending on your OS)
- `tbb-devel`
- `dpctl`

In addition, you need oneDPL installed on your system. There are two ways to do
so:

1. Install oneAPI and run the oneDPL activation script. E.g., on linux:

   ```bash
   source /opt/intel/oneapi/dpl/latest/env/vars.sh
   ```

2. Clone dpl from https://github.com/oneapi-src/oneDPL and set the `DPL_ROOT`
   environment variable to point to the `include` directory in the repository.

   E.g., on linux

   ```bash
   git clone https://github.com/oneapi-src/oneDPL
   export DPL_ROOT=$(pwd)/oneDPL/include
   ```

After these steps, `dpnp` can be built in debug mode as follows:


```bash
git clone https://github.com/IntelPython/dpnp
cd dpnp
./0.build.sh
```

## Install Wheel Package from Pypi
Install DPNP
```cmd
python -m pip install --index-url https://pypi.anaconda.org/intel/simple --extra-index-url https://pypi.org/simple dpnp
```
Note: DPNP wheel package is placed on Pypi, but some of its dependencies (like Intel numpy) are in Anaconda Cloud.
That is why install command requires additional intel Pypi channel from Anaconda Cloud.

Set path to Performance Libraries in case of using venv or system Python:
```cmd
export LD_LIBRARY_PATH=<path_to_your_env>/lib
```

It is also required to set following environment variables:
```cmd
export OCL_ICD_FILENAMES_RESET=1
export OCL_ICD_FILENAMES=libintelocl.so
```

## Run test
```bash
. ./0.env.sh
pytest
# or
pytest tests/test_matmul.py -s -v
# or
python -m unittest tests/test_mixins.py
```

## Run numpy external test
```bash
. ./0.env.sh
python -m tests.third_party.numpy_ext
# or
python -m tests.third_party.numpy_ext core/tests/test_umath.py
# or
python -m tests.third_party.numpy_ext core/tests/test_umath.py::TestHypot::test_simple
```

### Building documentation:
```bash
Prerequisites:
$ conda install sphinx sphinx_rtd_theme
Building:
1. Install dpnp into your python environment
2. $ cd doc && make html
3. The documentation will be in doc/_build/html
```

## Packaging:
```bash
. ./0.env.sh
conda-build conda-recipe/
```

## Run benchmark:
```bash
cd benchmarks/

asv run --python=python --bench <filename without .py>
# example:
asv run --python=python --bench bench_elementwise

# or

asv run --python=python --bench <class>.<bench>
# example:
asv run --python=python --bench Elementwise.time_square

# add --quick option to run every case once but looks like first execution has additional overheads and takes a lot of time (need to be investigated)
```


## Tests matrix:
| # |Name                                |OS   |distributive|interpreter|python used from|SYCL queue manager|build commands set                                                                                                                              |forced environment                                                                                                       |
|---|------------------------------------|-----|------------|-----------|:--------------:|:----------------:|------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
|1  |Ubuntu 20.04 Python37               |Linux|Ubuntu 20.04|Python 3.7 |  IntelOneAPI   |      local       |export DPNP_DEBUG=1 python setup.py clean python setup.py build_clib python setup.py build_ext --inplace pytest                                 |cmake-3.19.2, valgrind, pytest-valgrind, conda-build, pytest, hypothesis                                                 |
|2  |Ubuntu 20.04 Python38               |Linux|Ubuntu 20.04|Python 3.8 |  IntelOneAPI   |      local       |export DPNP_DEBUG=1 python setup.py clean python setup.py build_clib python setup.py build_ext --inplace pytest                                 |cmake-3.19.2, valgrind, pytest-valgrind, conda-build, pytest, hypothesis                                                 |
|3  |Ubuntu 20.04 Python39               |Linux|Ubuntu 20.04|Python 3.9 |  IntelOneAPI   |      local       |export DPNP_DEBUG=1 python setup.py clean python setup.py build_clib python setup.py build_ext --inplace pytest                                 |cmake-3.19.2, valgrind, pytest-valgrind, conda-build, pytest, hypothesis                                                 |
|4  |Ubuntu 20.04 External Tests Python37|Linux|Ubuntu 20.04|Python 3.7 |  IntelOneAPI   |      local       |export DPNP_DEBUG=1 python setup.py clean python setup.py build_clib python setup.py build_ext --inplace python -m tests_external.numpy.runtests|cmake-3.19.2, valgrind, pytest-valgrind, conda-build, pytest, hypothesis                                                 |
|5  |Ubuntu 20.04 External Tests Python38|Linux|Ubuntu 20.04|Python 3.8 |  IntelOneAPI   |      local       |export DPNP_DEBUG=1 python setup.py clean python setup.py build_clib python setup.py build_ext --inplace python -m tests_external.numpy.runtests|cmake-3.19.2, valgrind, pytest-valgrind, conda-build, pytest, hypothesis                                                 |
|6  |Ubuntu 20.04 External Tests Python39|Linux|Ubuntu 20.04|Python 3.9 |  IntelOneAPI   |      local       |export DPNP_DEBUG=1 python setup.py clean python setup.py build_clib python setup.py build_ext --inplace python -m tests_external.numpy.runtests|cmake-3.19.2, valgrind, pytest-valgrind, conda-build, pytest, hypothesis                                                 |
|7  |Code style                          |Linux|Ubuntu 20.04|Python 3.8 |  IntelOneAPI   |      local       |python ./setup.py style                                                                                                                         |cmake-3.19.2, valgrind, pytest-valgrind, conda-build, pytest, hypothesis, conda-verify, pycodestyle, autopep8, black     |
|8  |Valgrind                            |Linux|Ubuntu 20.04|           |  IntelOneAPI   |      local       |export DPNP_DEBUG=1 python setup.py clean python setup.py build_clib python setup.py build_ext --inplace                                        |cmake-3.19.2, valgrind, pytest-valgrind, conda-build, pytest, hypothesis                                                 |
|9  |Code coverage                       |Linux|Ubuntu 20.04|Python 3.8 |  IntelOneAPI   |      local       |export DPNP_DEBUG=1 python setup.py clean python setup.py build_clib python setup.py build_ext --inplace                                        |cmake-3.19.2, valgrind, pytest-valgrind, conda-build, pytest, hypothesis, conda-verify, pycodestyle, autopep8, pytest-cov|
