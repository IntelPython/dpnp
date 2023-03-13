#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2020, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

""" NumPy is the fundamental package for array computing with Python.

It provides:

- a powerful N-dimensional array object
- sophisticated (broadcasting) functions
- tools for integrating C/C++ and Fortran code
- useful linear algebra, Fourier transform, and random number capabilities
- and much more

"""

import importlib.machinery as imm  # Python 3 is required
import sys
import os
import dpctl
import numpy

from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options as cython_options

from utils.command_style import source_style
from utils.command_clean import source_clean
from utils.command_build_clib import custom_build_clib, dpnp_backend_c_description, _project_backend_dir, _sdl_cflags, _project_extra_link_args, IS_WIN
from utils.command_build_cmake_clib import custom_build_cmake_clib


"""
Python version check
"""
if sys.version_info[:2] < (3, 6):
    raise RuntimeError("DPNP: Python version >= 3.6 required.")


"""
Get the project version
"""
thefile_path = os.path.abspath(os.path.dirname(__file__))
version_mod = imm.SourceFileLoader('version', os.path.join(thefile_path, 'dpnp', 'version.py')).load_module()
__version__ = version_mod.__version__


"""
Set project auxilary data like readme and licence files
"""
with open('README.md') as f:
    __readme_file__ = f.read()

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: C
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: Implementation :: CPython
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

"""
Extra defined commands for the build system

>$ python ./setup.py --help-commands

>$ python ./setup.py style
>$ python ./setup.py style -a
>$ python ./setup.py clean

TODO: spell check, valgrind, code coverage
"""

# TODO: refactor/fix
# on Win we need a specific build_clib definition to prevent using cmake during build_ext execution
if IS_WIN:
    dpnp_build_commands = {'style': source_style,
                           'build_clib_setuptools': custom_build_clib,
                           'build_clib': custom_build_clib,
                           'clean': source_clean
                           }
else:
    dpnp_build_commands = {'style': source_style,
                           'build_clib_setuptools': custom_build_clib,
                           'build_clib': custom_build_cmake_clib,
                           'clean': source_clean
                           }

if IS_WIN:
    '''
    This variable controls setuptools execution on windows
    to avoid automatically search and confirm workability of the compiler
    If not set, error "Microsoft Visual C++ 14.0 or greater is required." appears
    '''
    os.environ["DISTUTILS_USE_SDK"] = "1"

"""
The project modules description
"""
kwargs_common = {
    "include_dirs": [numpy.get_include(), dpctl.get_include()] + _project_backend_dir,
    "library_dirs": [os.path.dirname(dpctl.get_include()),],
    "libraries": ["DPCTLSyclInterface"],
    "extra_compile_args": _sdl_cflags,
    "extra_link_args": _project_extra_link_args,
    "define_macros": [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    "language": "c++"
}

dpnp_algo = Extension(
    name="dpnp.dpnp_algo.dpnp_algo",
    sources=[os.path.join("dpnp", "dpnp_algo", "dpnp_algo.pyx")],
    depends=[
        os.path.join("dpnp", "dpnp_algo", "dpnp_algo_linearalgebra.pyx"),
        os.path.join("dpnp", "dpnp_algo", "dpnp_algo_manipulation.pyx"),
        os.path.join("dpnp", "dpnp_algo", "dpnp_algo_counting.pyx"),
        os.path.join("dpnp", "dpnp_algo", "dpnp_algo_statistics.pyx"),
        os.path.join("dpnp", "dpnp_algo", "dpnp_algo_trigonometric.pyx"),
        os.path.join("dpnp", "dpnp_algo", "dpnp_algo_sorting.pyx"),
        os.path.join("dpnp", "dpnp_algo", "dpnp_algo_arraycreation.pyx"),
        os.path.join("dpnp", "dpnp_algo", "dpnp_algo_mathematical.pyx"),
        os.path.join("dpnp", "dpnp_algo", "dpnp_algo_searching.pyx"),
        os.path.join("dpnp", "dpnp_algo", "dpnp_algo_indexing.pyx"),
        os.path.join("dpnp", "dpnp_algo", "dpnp_algo_logic.pyx"),
        os.path.join("dpnp", "dpnp_algo", "dpnp_algo_bitwise.pyx"),
        os.path.join("dpnp", "dpnp_algo", "dpnp_algo_special.pyx"),
    ],
    **kwargs_common)

dpnp_dparray = Extension(
    name="dpnp.dparray",
    sources=[os.path.join("dpnp", "dparray.pyx")],
    **kwargs_common)

dpnp_random = Extension(
    name="dpnp.random.dpnp_algo_random",
    sources=[os.path.join("dpnp", "random", "dpnp_algo_random.pyx")],
    **kwargs_common)

dpnp_linalg = Extension(
    name="dpnp.linalg.dpnp_algo_linalg",
    sources=[os.path.join("dpnp", "linalg", "dpnp_algo_linalg.pyx")],
    **kwargs_common)

dpnp_fft = Extension(
    name="dpnp.fft.dpnp_algo_fft",
    sources=[os.path.join("dpnp", "fft", "dpnp_algo_fft.pyx")],
    **kwargs_common)

dpnp_utils = Extension(
    name="dpnp.dpnp_utils.dpnp_algo_utils",
    sources=[os.path.join("dpnp", "dpnp_utils", "dpnp_algo_utils.pyx")],
    **kwargs_common)

cython_options.docstrings = True
cython_options.warning_errors = True

dpnp_cython_mods = cythonize([dpnp_algo, dpnp_dparray, dpnp_random, dpnp_utils, dpnp_linalg, dpnp_fft],
                             compiler_directives={"language_level": sys.version_info[0],
                                                  "warn.unused": False,
                                                  "warn.unused_result": False,
                                                  "warn.maybe_uninitialized": False,
                                                  "warn.undeclared": False,
                                                  "boundscheck": True,
                                                  "linetrace": True
                                                  },
                             gdb_debug=False,
                             build_dir="build_cython",
                             annotate=False,
                             quiet=False)

setup(name="dpnp",
      version=__version__,
      description="NumPy-like API accelerated with SYCL",
      long_description=__readme_file__,
      long_description_content_type="text/markdown",
      author="Intel Corporation",
      maintainer="Intel Corp.",
      maintainer_email="scripting@intel.com",
      url="https://intelpython.github.io/dpnp/",
      download_url="https://github.com/IntelPython/dpnp",
      license='BSD',
      classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
      keywords="sycl numpy python3 intel mkl oneapi gpu dpcpp pstl",
      platforms=["Linux", "Windows"],
      test_suite="pytest",
      python_requires=">=3.6",
      install_requires=["numpy>=1.15"],
      setup_requires=["numpy>=1.15"],
      tests_require=["numpy>=1.15"],
      ext_modules=dpnp_cython_mods,
      cmdclass=dpnp_build_commands,
      packages=['dpnp',
                'dpnp.dpnp_algo',
                'dpnp.dpnp_utils',
                'dpnp.fft',
                'dpnp.linalg',
                'dpnp.random'
                ],
      package_data={'dpnp': ['libdpnp_backend_c.so', 'dpnp_backend_c.lib', 'dpnp_backend_c.dll']},
      include_package_data=True,

      # this is needed for 'build' command to automatically call 'build_clib'
      # it attach the library to all extensions (it is not needed)
      libraries=dpnp_backend_c_description
      )
