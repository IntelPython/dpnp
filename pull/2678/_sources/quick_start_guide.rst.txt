.. _quick_start_guide:
.. include:: ./ext_links.txt

.. |copy| unicode:: U+000A9

.. |trade| unicode:: U+2122

=================
Quick Start Guide
=================

Device Drivers
=================

To start programming data parallel devices beyond CPU, you will need
an appropriate hardware. The Data Parallel Extension for NumPy* works fine
on Intel |copy| laptops with integrated graphics. In majority of cases,
your Windows*-based laptop already has all necessary device drivers installed.
But if you want the most up-to-date driver, you can always
`update it to the latest one <https://www.intel.com/content/www/us/en/download-center/home.html>`_.
Follow device driver installation instructions to complete the step.


Python Interpreter
==================

You will need Python 3.9, 3.10, 3.11, 3.12, 3.13 or 3.14 installed on your system. If you
do not have one yet the easiest way to do that is to install
`Intel Distribution for Python*`_. It installs all essential Python numerical
and machine learning packages optimized for the Intel hardware, including
Data Parallel Extension for NumPy*.
If you have Python installation from another vendor, it is fine too. All you
need is to install Data Parallel Extension for NumPy* manually as shown
in the next installation section.


Installation
============

Install Package from Intel(R) channel
-------------------------------------------

You will need one of the commands below:

* Conda: ``conda install dpnp -c https://software.repos.intel.com/python/conda/ -c conda-forge --override-channels``

* Pip: ``python -m pip install --index-url https://software.repos.intel.com/python/pypi dpnp``

These commands install dpnp package along with its dependencies, including
``dpctl`` package with `Data Parallel Control Library`_ and all required
compiler runtimes and OneMKL.

.. warning::
    Packages from the Intel channel are meant to be used together with dependencies from the **conda-forge** channel, and might not
    work correctly when used in an environment where packages from the ``anaconda`` default channel have been installed. It is
    advisable to use the `miniforge <https://github.com/conda-forge/miniforge>`__ installer for ``conda``/``mamba``, as it comes with
    ``conda-forge`` as the only default channel.

.. note::
   Before installing with conda or pip it is strongly advised to update ``conda`` and ``pip`` to latest versions


Build and Install Conda Package
-------------------------------

Alternatively you can create and activate a local conda build environment:

.. code-block:: bash

    conda create -n build-env conda-build
    conda activate build-env

And to build dpnp package from the sources:

.. code-block:: bash

    conda build conda-recipe -c https://software.repos.intel.com/python/conda/ -c conda-forge --override-channels

Finally, to install the result package:

.. code-block:: bash

    conda install dpnp -c local


Build and Install with scikit-build
-----------------------------------

Another way to build and install dpnp package from the source is to use Python
``setuptools`` and ``scikit-build``. You will need to create a local conda
build environment by command below depending on hosting OS.

On Linux:

.. code-block:: bash

    conda create -n build-env dpctl cython dpcpp_linux-64 mkl-devel-dpcpp tbb-devel        \
          onedpl-devel cmake scikit-build ninja versioneer pytest intel-gpu-ocl-icd-system \
          -c dppy/label/dev -c https://software.repos.intel.com/python/conda/ -c conda-forge --override-channels
    conda activate build-env

On Windows:

.. code-block:: bash

    conda create -n build-env dpctl cython dpcpp_win-64 mkl-devel-dpcpp tbb-devel          \
          onedpl-devel cmake scikit-build ninja versioneer pytest intel-gpu-ocl-icd-system \
          -c dppy/label/dev -c https://software.repos.intel.com/python/conda/ -c conda-forge --override-channels
    conda activate build-env

To build and install the package on Linux OS, run:

.. code-block:: bash

    python setup.py install -- -G Ninja -DCMAKE_C_COMPILER:PATH=icx -DCMAKE_CXX_COMPILER:PATH=icpx

To build and install the package on Windows OS, run:

.. code-block:: bash

    python setup.py install -- -G Ninja -DCMAKE_C_COMPILER:PATH=icx -DCMAKE_CXX_COMPILER:PATH=icx

Alternatively, to develop on Linux OS, you can use the driver script:

.. code-block:: bash

    python scripts/build_locally.py

Building for custom SYCL targets
--------------------------------
Project ``dpnp`` is written using generic SYCL and supports building for multiple SYCL targets,
subject to limitations of `CodePlay <https://codeplay.com/>`_ plugins implementing SYCL
programming model for classes of devices.

Building ``dpnp`` for these targets requires that these CodePlay plugins be installed into DPC++
installation layout of compatible version. The following plugins from CodePlay are supported:

    - `oneAPI for NVIDIA(R) GPUs <codeplay_nv_plugin_>`_
    - `oneAPI for AMD GPUs <codeplay_amd_plugin_>`_

.. _codeplay_nv_plugin: https://developer.codeplay.com/products/oneapi/nvidia/
.. _codeplay_amd_plugin: https://developer.codeplay.com/products/oneapi/amd/

Building ``dpnp`` also requires `building Data Parallel Control Library for custom SYCL targets.
<https://intelpython.github.io/dpctl/latest/beginners_guides/installation.html#building-for-custom-sycl-targets>`_

Builds for CUDA and AMD devices internally use SYCL alias targets that are passed to the compiler.
A full list of available SYCL alias targets is available in the
`DPC++ Compiler User Manual <https://intel.github.io/llvm/UsersManual.html>`_.

CUDA build
~~~~~~~~~~

To build for CUDA devices, use the ``--target-cuda`` argument.

To target a specific architecture (e.g., ``sm_80``):

.. code-block:: bash

    python scripts/build_locally.py --target-cuda=sm_80

To use the default architecture (``sm_50``), run:

.. code-block:: bash

    python scripts/build_locally.py --target-cuda

Note that kernels are built for the default architecture (``sm_50``), allowing them to work on a
wider range of architectures, but limiting the usage of more recent CUDA features.

For reference, compute architecture strings like ``sm_80`` correspond to specific
CUDA Compute Capabilities (e.g., Compute Capability 8.0 corresponds to ``sm_80``).
A complete mapping between NVIDIA GPU models and their respective
Compute Capabilities can be found in the official
`CUDA GPU Compute Capability <https://developer.nvidia.com/cuda-gpus>`_ documentation.

AMD build
~~~~~~~~~

To build for AMD devices, use the ``--target-hip=<arch>`` argument:

.. code-block:: bash

    python scripts/build_locally.py --target-hip=<arch>

Note that the *oneAPI for AMD GPUs* plugin requires the architecture be specified and only
one architecture can be specified at a time.

To determine the architecture code (``<arch>``) for your AMD GPU, run:

.. code-block:: bash

    rocminfo | grep 'Name: *gfx.*'

This will print names like ``gfx90a``, ``gfx1030``, etc.
You can then use one of them as the argument to ``--target-hip``.

For example:

.. code-block:: bash
    python scripts/build_locally.py --target-hip=gfx90a

Multi-target build
~~~~~~~~~~~~~~~~~~

The default ``dpnp`` build from the source enables support of Intel devices only.
Extending the build with a custom SYCL target additionally enables support of CUDA or AMD
device in ``dpnp``. Besides, the support can be also extended to enable both CUDA and AMD
devices at the same time:

.. code-block:: bash

    python scripts/build_locally.py --target-cuda --target-hip=gfx90a


Testing
=======

If you want to execute the scope of Python test suites which are available
by the source, you will need to run a command as below:

.. code-block:: bash

    pytest -s tests

Examples
========

The examples below demonstrates a simple usage of the Data Parallel Extension for NumPy*

.. literalinclude:: ../examples/example_sum.py
  :linenos:
  :language: python
  :lines: 35-
  :caption: How to create an array and to sum the elements

.. literalinclude:: ../examples/example_cfd.py
  :linenos:
  :language: python
  :lines: 34-
  :caption: How to create an array on the specific device type and how the next computations follow it

More examples on how to use ``dpnp`` can be found in ``dpnp/examples``.
