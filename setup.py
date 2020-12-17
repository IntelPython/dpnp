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
from utils.dpnp_build_utils import find_cmplr, find_dpl, find_mathlib, find_omp


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
    raise EnvironmentError("DPNP: " + sys.platform + " not supported")

"""
Set compiler for the project
"""
# default variables (for Linux)
_project_compiler = "dpcpp"
_project_linker = "dpcpp"
_project_cmplr_flag_sycl_devel = ["-fsycl-device-code-split=per_kernel"]
_project_cmplr_flag_sycl = ["-fsycl"]
_project_cmplr_flag_compatibility = ["-Wl,--enable-new-dtags"]
_project_cmplr_flag_lib = ["-shared"]
_project_cmplr_flag_release_build = ["-O3", "-DNDEBUG", "-fPIC"]
_project_cmplr_flag_debug_build = ["-g", "-O1", "-W", "-Wextra", "-Wshadow", "-Wall", "-Wstrict-prototypes", "-fPIC"]
_project_cmplr_flag_default_build = []
_project_cmplr_macro = []
_project_force_build = False
_project_sycl_queue_control_macro = [("DPNP_LOCAL_QUEUE", "1")]
_project_rpath = ["$ORIGIN"]
_dpctrl_include = []
_dpctrl_libpath = []
_dpctrl_lib = []
_sdl_cflags = ["-fstack-protector-strong",
               "-fPIC", "-D_FORTIFY_SOURCE=2",
               "-Wformat",
               "-Wformat-security",
               "-fno-strict-overflow",
               "-fno-delete-null-pointer-checks"]
_sdl_ldflags = ["-Wl,-z,noexecstack,-z,relro,-z,now"]

# TODO remove when it will be fixed on TBB side. Details:
# In GCC versions 9 and 10 the application that uses Parallel STL algorithms may fail to compile due to incompatible
# interface changes between earlier versions of Intel TBB and oneTBB. Disable support for Parallel STL algorithms
# by defining PSTL_USE_PARALLEL_POLICIES (in GCC 9), _GLIBCXX_USE_TBB_PAR_BACKEND (in GCC 10) macro to zero
# before inclusion of the first standard header file in each translation unit.
_project_cmplr_macro += [("PSTL_USE_PARALLEL_POLICIES", "0"), ("_GLIBCXX_USE_TBB_PAR_BACKEND", "0")]

try:
    """
    Detect external SYCL queue handling library
    """
    import dpctl

    _dpctrl_include += [dpctl.get_include()]
    # _dpctrl_libpath = for package build + for local build
    _dpctrl_libpath = ["$ORIGIN/../dpctl"] + [os.path.join(dpctl.get_include(), '..')]
    _dpctrl_lib = ["DPCTLSyclInterface"]
except ImportError:
    """
    Set local SYCL queue handler
    """
    _project_cmplr_macro += _project_sycl_queue_control_macro


# other OS specific
if IS_WIN:
    _project_compiler = "dpcpp"
    _project_linker = "lld-link"
    _project_cmplr_flag_sycl = []
    _project_cmplr_flag_compatibility = []
    _project_cmplr_flag_lib = ["/DLL"]
    _project_cmplr_macro += [("_WIN", "1")]
    _project_rpath = []
    # TODO this flag creates unexpected behavior during compilation, need to be fixed
    # _sdl_cflags = ["-GS"]
    _sdl_cflags = []
    _sdl_ldflags = ["-NXCompat", "-DynamicBase"]


# try:
#     """
#     set environment variables to control setuptools build procedure
#     """
#     # check if we have preset variables in environment
#     os.environ["CC"] == _project_compiler
#     os.environ["CXX"] == _project_compiler
#     os.environ["LD"] == _project_linker
# except KeyError:
#     # set variables if not presented in environment
#     os.environ["CC"] = _project_compiler
#     os.environ["CXX"] = _project_compiler
#     os.environ["LD"] = _project_linker


"""
Get the project build type
"""
__dpnp_debug__ = os.environ.get('DPNP_DEBUG', None)
if __dpnp_debug__ is not None:
    """
    Debug configuration
    """
    _project_cmplr_flag_sycl += _project_cmplr_flag_sycl_devel
    _project_cmplr_flag_default_build = _project_cmplr_flag_debug_build
else:
    """
    Release configuration
    """
    _project_cmplr_flag_default_build = _project_cmplr_flag_release_build


"""
Search and set math library environemnt
"""
_mathlib_rpath = []
_cmplr_rpath = []
_omp_rpath = []


"""
Get the math library environemnt
"""
_mathlib_include, _mathlib_path = find_mathlib(verbose=True)

_project_cmplr_macro += [("MKL_ILP64", "1")]  # using 64bit integers in MKL interface (long)
_mathlibs = ["mkl_rt", "mkl_sycl", "mkl_intel_ilp64", "mkl_sequential",
             "mkl_core", "sycl", "OpenCL", "pthread", "m", "dl"]

if IS_LIN:
    _mathlib_rpath = _mathlib_path
elif IS_WIN:
    _mathlibs = ["mkl_sycl", "mkl_intel_ilp64", "mkl_tbb_thread", "mkl_core", "sycl", "OpenCL", "tbb"]

"""
Get the compiler environemnt
"""
_cmplr_include, _cmplr_libpath = find_cmplr(verbose=True)
_dpl_include, _ = find_dpl(verbose=True)
_, _omp_libpath = find_omp(verbose=True)

if IS_LIN:
    _cmplr_rpath = _cmplr_libpath
    _omp_rpath = _omp_libpath


"""
Final set of arguments for extentions
"""
_project_extra_link_args = _project_cmplr_flag_compatibility + \
    ["-Wl,-rpath," + x for x in _project_rpath] + _sdl_ldflags
_project_dir = os.path.dirname(os.path.abspath(__file__))
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
                "dpnp/backend/custom_kernels_bitwise.cpp",
                "dpnp/backend/custom_kernels.cpp",
                "dpnp/backend/custom_kernels_elemwise.cpp",
                "dpnp/backend/custom_kernels_linalg.cpp",
                "dpnp/backend/custom_kernels_manipulation.cpp",
                "dpnp/backend/custom_kernels_mathematical.cpp",
                "dpnp/backend/custom_kernels_reduction.cpp",
                "dpnp/backend/custom_kernels_searching.cpp",
                "dpnp/backend/custom_kernels_sorting.cpp",
                "dpnp/backend/custom_kernels_statistics.cpp",
                "dpnp/backend/dpnp_kernels_fft.cpp",
                "dpnp/backend/dpnp_kernels_random.cpp",
                "dpnp/backend/memory_sycl.cpp",
                "dpnp/backend/queue_sycl.cpp"
            ],
            "include_dirs": _cmplr_include + _dpl_include + _mathlib_include + _project_backend_dir + _dpctrl_include,
            "library_dirs": _mathlib_path + _omp_libpath + _dpctrl_libpath,
            "runtime_library_dirs": _project_rpath + _dpctrl_libpath, # + _mathlib_rpath + _cmplr_rpath + _omp_rpath,
            "extra_preargs": _project_cmplr_flag_sycl + _sdl_cflags,
            "extra_link_preargs": _project_cmplr_flag_compatibility + _sdl_ldflags,
            "extra_link_postargs": _project_cmplr_flag_lib,
            "libraries": _mathlibs + _dpctrl_lib,
            "macros": _project_cmplr_macro,
            "force_build": _project_force_build,
            "compiler": [_project_compiler],
            "linker": [_project_linker],
            "default_flags": _project_cmplr_flag_default_build,
            "language": "c++"
        }
     ]
]

dpnp_backend = Extension(
    name="dpnp.backend",
    sources=["dpnp/backend.pyx"],
    libraries=[],
    include_dirs=[numpy.get_include()] + _project_backend_dir,
    extra_compile_args=_sdl_cflags,
    extra_link_args=_project_extra_link_args,
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    language="c++"
)

dpnp_dparray = Extension(
    name="dpnp.dparray",
    sources=["dpnp/dparray.pyx"],
    libraries=[],
    include_dirs=[numpy.get_include()] + _project_backend_dir,
    extra_compile_args=_sdl_cflags,
    extra_link_args=_project_extra_link_args,
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    language="c++"
)

dpnp_random = Extension(
    name="dpnp.random._random",
    sources=["dpnp/random/_random.pyx"],
    include_dirs=[numpy.get_include()] + _project_backend_dir,
    extra_compile_args=_sdl_cflags,
    extra_link_args=_project_extra_link_args,
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    language="c++"
)

dpnp_utils = Extension(
    name="dpnp.dpnp_utils",
    sources=["dpnp/dpnp_utils.pyx"],
    include_dirs=[numpy.get_include()] + _project_backend_dir,
    extra_compile_args=_sdl_cflags,
    extra_link_args=_project_extra_link_args,
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    language="c++"
)

dpnp_linalg = Extension(
    name="dpnp.linalg.linalg",
    sources=["dpnp/linalg/linalg.pyx"],
    include_dirs=[numpy.get_include()] + _project_backend_dir,
    extra_compile_args=_sdl_cflags,
    extra_link_args=_project_extra_link_args,
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    language="c++"
)

dpnp_fft = Extension(
    name="dpnp.fft.dpnp_algo_fft",
    sources=["dpnp/fft/dpnp_algo_fft.pyx"],
    include_dirs=[numpy.get_include()] + _project_backend_dir,
    extra_compile_args=_sdl_cflags,
    extra_link_args=_project_extra_link_args,
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    language="c++"
)

cython_options.docstrings = True
cython_options.warning_errors = True

dpnp_cython_mods = cythonize([dpnp_backend, dpnp_dparray, dpnp_random, dpnp_utils, dpnp_linalg, dpnp_fft],
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
                'dpnp.fft',
                'dpnp.linalg',
                'dpnp.random'
                ],
      package_data={'dpnp': ['libdpnp_backend_c.so']},
      include_package_data=True,

      # this is needed for 'build' command to automatically call 'build_clib'
      # it attach the library to all extensions (it is not needed)
      libraries=dpnp_backend_c
      )
