[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Pre-commit](https://github.com/IntelPython/dpnp/actions/workflows/pre-commit.yml/badge.svg?branch=master&event=push)](https://github.com/IntelPython/dpnp/actions/workflows/pre-commit.yml)
[![Conda package](https://github.com/IntelPython/dpnp/actions/workflows/conda-package.yml/badge.svg?branch=master&event=push)](https://github.com/IntelPython/dpnp/actions/workflows/conda-package.yml)
[![Coverage Status](https://coveralls.io/repos/github/IntelPython/dpnp/badge.svg?branch=master)](https://coveralls.io/github/IntelPython/dpnp?branch=master)
[![Build Sphinx](https://github.com/IntelPython/dpnp/workflows/Build%20Sphinx/badge.svg)](https://intelpython.github.io/dpnp)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/IntelPython/dpnp/badge)](https://securityscorecards.dev/viewer/?uri=github.com/IntelPython/dpnp)

<img align="left" src="https://spec.oneapi.io/oneapi-logo-white-scaled.jpg" alt="oneAPI logo" width="75"/>

# DPNP - Data Parallel Extension for NumPy*

Data Parallel Extension for NumPy* or `dpnp` is a Python library that
implements a subset of NumPy* that can be executed on any data parallel device.
The subset is a drop-in replacement of core NumPy* functions and numerical data types.

[API coverage summary](https://intelpython.github.io/dpnp/reference/comparison.html#summary)

[Full documentation](https://intelpython.github.io/dpnp/)

`Dpnp` is the core part of a larger family of [data-parallel Python libraries and tools](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html)
to program on XPUs.


# Installing

You can install the library using `conda`, `mamba` or [pip](https://pypi.org/project/dpnp/)
package managers. It is also available as part of the [Intel(R) Distribution for Python](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html)
(IDP).

## Intel(R) Distribution for Python

You can find the most recent release of `dpnp` every quarter as part of the IDP
releases.

To get the library from the latest release, follow the instructions from
[Get Started With Intel® Distribution for Python](https://www.intel.com/content/www/us/en/developer/articles/technical/get-started-with-intel-distribution-for-python.html).

## Conda

To install `dpnp` from the Intel(R) conda channel, use the following command:

```bash
conda install dpnp -c https://software.repos.intel.com/python/conda/ -c conda-forge --override-channels
```

## Pip

The `dpnp` can be installed using `pip` obtaining wheel packages either from
PyPi or from Intel(R) channel. To install `dpnp` wheel package from Intel(R)
channel, run the following command:

```bash
python -m pip install --index-url https://software.repos.intel.com/python/pypi dpnp
```

## Installing the bleeding edge

To try out the latest features, install `dpnp` using our development channel on
Anaconda cloud:

```bash
conda install dpnp -c dppy/label/dev -c https://software.repos.intel.com/python/conda/ -c conda-forge --override-channels
```


# Building

Refer to our [Documentation](https://intelpython.github.io/dpnp/quick_start_guide.html)
for more information on setting up a development environment and building `dpnp`
from the source.


# Running Tests

Tests are located in folder [dpnp/tests](dpnp/tests).

To run the tests, use:
```bash
python -m pytest --pyargs dpnp
```
