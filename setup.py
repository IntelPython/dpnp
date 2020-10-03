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
import numpy

from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options as cython_options

from utils.command_style import source_style
from utils.command_clean import source_clean
from utils.command_build_clib import custom_build_clib


"""
Python version check
"""
if sys.version_info[:2] < (3, 6):
    raise RuntimeError("Intel NumPy: Python version >= 3.5 required.")


"""
Get the project version
"""
thefile_path = os.path.abspath(os.path.dirname(__file__))
version_mod = imm.SourceFileLoader('version', os.path.join(thefile_path, 'dpnp', '_version.py')).load_module()
__version__ = version_mod.__version__


"""
Set project auxilary data like readme and licence files
"""
with open('README.md') as f:
    __readme_file__ = f.read()

with open('LICENSE.txt') as f:
    __license_file__ = f.read()

CLASSIFIERS = """\
Development Status :: 0 - Alpha
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

IS_WIN = False
IS_MAC = False
IS_LIN = False

if 'linux' in sys.platform:
    IS_LIN = True
elif sys.platform == 'darwin':
    IS_MAC = True
elif sys.platform in ['win32', 'cygwin']:
    IS_WIN = True
else:
    raise EnvironmentError("Intel NumPy: " + sys.platform + " not supported")

"""
Set compiler for the project
"""
# default variables (for Linux)
_project_compiler = "clang++"
_project_linker = "clang++"
_project_cmplr_flag_sycl_devel = ["-fsycl-device-code-split=per_kernel"]
_project_cmplr_flag_sycl = ["-fsycl"]
_project_cmplr_flag_compatibility = ["-Wl,--enable-new-dtags", "-fPIC"]
_project_cmplr_flag_lib = []
_project_cmplr_macro = []
_project_force_build = False
_project_sycl_queue_control_macro = [("DPNP_LOCAL_QUEUE", "1")]
_project_rpath = ["$ORIGIN"]
_dpctrl_include = []
_dpctrl_libpath = []
_dpctrl_lib = []


try:
    """
    Detect external SYCL queue handling library
    """
    import dpctrl

    # TODO this will not work with no Conda environment
    _conda_root = os.environ.get('CONDA_PREFIX', "conda_include_error")
    _dpctrl_include += [os.path.join(_conda_root, 'include')]
    _dpctrl_libpath += [os.path.join(_conda_root, 'lib')]
    _dpctrl_lib += ["dpctrlsyclinterface"]
except ImportError:
    """
    Set local SYCL queue handler
    """
    _project_cmplr_macro += _project_sycl_queue_control_macro


# other OS specific
if IS_WIN:
    _project_compiler = "dpcpp-cl"   # "clang-cl"
    _project_linker = "lld-link"  # "dpcpp-cl"
    _project_cmplr_flag_sycl = []
    _project_cmplr_flag_compatibility = []
    _project_cmplr_flag_lib = ['/DLL']
    _project_cmplr_macro = [("_WIN", "1"), ("MKL_ILP64", "1")]
    _project_rpath = []
    # TODO obtain setuptools.compiler.buildline options line and replace /MD with /MT instead adding it
    os.environ["CFLAGS"] = "/MT"


try:
    """
    set environment variables to control setuptools build procedure
    """
    # check if we have preset variables in environment
    os.environ["CC"] == _project_compiler
    os.environ["CXX"] == _project_compiler
    os.environ["LD"] == _project_linker
except KeyError:
    # set variables if not presented in environment
    os.environ["CC"] = _project_compiler
    os.environ["CXX"] = _project_compiler
    os.environ["LD"] = _project_linker


"""
Get the project build type
"""
__dpnp_debug__ = os.environ.get('DPNP_DEBUG', None)
if __dpnp_debug__ is not None:
    _project_cmplr_flag_sycl += _project_cmplr_flag_sycl_devel


"""
Search and set MKL environemnt
"""
_mkl_rpath = []
_cmplr_rpath = []
_omp_rpath = []


_mkl_root = os.environ.get('MKLROOT', None)
if _mkl_root is None:
    raise EnvironmentError("Intel NumPy: Please install Intel OneAPI environment. MKLROOT is empty")
_mkl_include = [os.path.join(_mkl_root, 'include')]
_mkl_libs = ['mkl_rt', 'mkl_sycl', 'mkl_intel_ilp64', 'mkl_tbb_thread', 'mkl_core', 'tbb', 'iomp5']

_mkl_libpath = [os.path.join(_mkl_root, 'lib', 'intel64')]
if IS_LIN:
    _mkl_rpath = _mkl_libpath
elif IS_WIN:
    _mkl_libs = ["mkl_sycl", "mkl_intel_ilp64", "mkl_tbb_thread", "mkl_core", "sycl", "OpenCL", "tbb"]

_cmplr_root = os.environ.get('ONEAPI_ROOT', None)
if _cmplr_root is None:
    raise EnvironmentError("Please install Intel OneAPI environment. ONEAPI_ROOT is empty")

if IS_LIN:
    _cmplr_libpath = [os.path.join(_cmplr_root, 'compiler', 'latest', 'linux', 'lib')]
    _omp_libpath = [os.path.join(_cmplr_root, 'compiler', 'latest', 'linux', 'compiler', 'lib', 'intel64')]
    _cmplr_rpath = _cmplr_libpath
    _omp_rpath = _omp_libpath
elif IS_WIN:
    _cmplr_libpath = [os.path.join(_cmplr_root, 'compiler', 'latest', 'windows', 'lib')]
    _omp_libpath = [os.path.join(_cmplr_root, 'compiler', 'latest', 'windows', 'compiler', 'lib', 'intel64_win')]


"""
Final set of arguments for extentions
"""
_project_extra_link_args = _project_cmplr_flag_compatibility + ["-Wl,-rpath," + x for x in _project_rpath]
_project_dir = os.path.dirname(os.path.abspath(__file__))
_project_main_module_dir = [os.path.join(_project_dir, "dpnp")]
_project_backend_dir = [os.path.join(_project_dir, "dpnp", "backend")]


"""
Extra defined commands for the build system

>$ python ./setup.py --help-commands

>$ python ./setup.py style
>$ python ./setup.py style -a
>$ python ./setup.py clean

TODO: spell check, valgrind, code coverage
"""
dpnp_build_commands = {'style': source_style,
                       'build_clib': custom_build_clib,
                       'clean': source_clean
                       }


"""
The project modules description
"""
dpnp_backend_c = [
    ["dpnp_backend_c",
        {
            "sources": [
                "dpnp/backend/backend_iface_fptr.cpp",
                "dpnp/backend/custom_kernels.cpp",
                "dpnp/backend/custom_kernels_elemwise.cpp",
                "dpnp/backend/custom_kernels_manipulation.cpp",
                "dpnp/backend/custom_kernels_mathematical.cpp",
                "dpnp/backend/custom_kernels_reduction.cpp",
                "dpnp/backend/custom_kernels_searching.cpp",
                "dpnp/backend/custom_kernels_sorting.cpp",
                "dpnp/backend/custom_kernels_statistics.cpp",
                "dpnp/backend/memory_sycl.cpp",
                "dpnp/backend/mkl_wrap_blas1.cpp",
                "dpnp/backend/mkl_wrap_blas3.cpp",
                "dpnp/backend/mkl_wrap_lapack.cpp",
                "dpnp/backend/mkl_wrap_rng.cpp",
                "dpnp/backend/queue_sycl.cpp"
            ],
            "include_dirs": _mkl_include + _project_backend_dir + _dpctrl_include,
            "library_dirs": _mkl_libpath + _omp_libpath + _dpctrl_libpath,
            "runtime_library_dirs": [],  # _project_rpath + _mkl_rpath + _cmplr_rpath + _omp_rpath + _dpctrl_libpath,
            "extra_preargs": _project_cmplr_flag_sycl,
            "extra_link_postargs": _project_cmplr_flag_compatibility + _project_cmplr_flag_lib,
            "libraries": _mkl_libs + _dpctrl_lib,
            "macros": _project_cmplr_macro,
            "force_build": _project_force_build,
            "language": "c++"
        }
     ]
]

dpnp_backend = Extension(
    name="dpnp.backend",
    sources=["dpnp/backend.pyx"],
    libraries=[],
    include_dirs=[numpy.get_include()] + _project_backend_dir,
    extra_compile_args=[],
    extra_link_args=_project_extra_link_args,
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    language="c++"
)

dpnp_dparray = Extension(
    name="dpnp.dparray",
    sources=["dpnp/dparray.pyx"],
    libraries=[],
    include_dirs=[numpy.get_include()] + _project_backend_dir,
    extra_compile_args=[],
    extra_link_args=_project_extra_link_args,
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    language="c++"
)

dpnp_random = Extension(
    name="dpnp.random._random",
    sources=["dpnp/random/_random.pyx"],
    include_dirs=[numpy.get_include()] + _project_backend_dir,
    extra_link_args=_project_extra_link_args,
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    language="c++"
)

dpnp_utils = Extension(
    name="dpnp.dpnp_utils",
    sources=["dpnp/dpnp_utils.pyx"],
    include_dirs=[numpy.get_include()] + _project_backend_dir,
    extra_compile_args=[],
    extra_link_args=_project_extra_link_args,
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    language="c++"
)

dpnp_linalg = Extension(
    name="dpnp.linalg.linalg",
    sources=["dpnp/linalg/linalg.pyx"],
    include_dirs=[numpy.get_include()] + _project_backend_dir,
    extra_link_args=_project_extra_link_args,
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    language="c++"
)

cython_options.docstrings = True
cython_options.embed_pos_in_docstring = True
cython_options.warning_errors = True

dpnp_cython_mods = cythonize([dpnp_backend, dpnp_dparray, dpnp_random, dpnp_utils, dpnp_linalg],
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

setup(name="DPNP",
      version=__version__,
      description="Subclass of numpy.ndarray that uses mkl_malloc",
      long_description=__readme_file__,
      author="Intel Corporation",
      author_email="Intel Corporation",
      maintainer="Intel Corp.",
      maintainer_email="scripting@intel.com",
      url="http://github.com/IntelPython/mkl_array",
      download_url="http://github.com/IntelPython/mkl_array",
      license=__license_file__,
      classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
      keywords="python numeric algebra blas",
      platforms=["Linux", "Windows", "Mac OS-X"],
      test_suite="pytest",
      python_requires=">=3.6",
      install_requires=["numpy>=1.15"],
      setup_requires=["numpy>=1.15"],
      tests_require=["numpy>=1.15"],
      ext_modules=dpnp_cython_mods,
      cmdclass=dpnp_build_commands,
      packages=['dpnp',
                'dpnp.random',
                'dpnp.linalg',
                ],
      package_data={'dpnp': ['libdpnp_backend_c.so']},
      include_package_data=True,

      # this is needed for 'build' command to automatically call 'build_clib'
      # it attach the library to all extensions (it is not needed)
      libraries=dpnp_backend_c
      )
