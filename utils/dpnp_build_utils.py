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

import os
import sys


IS_CONDA_BUILD = os.environ.get("CONDA_BUILD") == "1"


def find_library(var_name, rel_header_paths, rel_lib_paths,
                 rel_include_path="include", rel_libdir_path="lib", verbose=False):
    """
    Find specified libraries/headers in the directory from the environment variable.

    Parameters
    ----------
    var_name : str
        the name of the environment variable
    rel_header_paths : list(str)
        relative paths to required headers
    rel_lib_paths : list(str)
        relative paths to required libraries
    rel_include_path : str
        relative path to the include directory
    rel_libdir_path : str
        relative path to the library directory
    verbose : bool
        to print paths to include and library directories

    Returns
    -------
    tuple(list(str), list(str))
        path to include directory, path to library directory
    """
    root_dir = os.getenv(var_name)
    if root_dir is None:
        return [], []

    include_find = os.path.join(root_dir, rel_include_path)
    libpath_find = os.path.join(root_dir, rel_libdir_path)
    required_headers = [os.path.join(include_find, rel_path) for rel_path in rel_header_paths]
    required_libs = [os.path.join(libpath_find, rel_path) for rel_path in rel_lib_paths]

    for required_file in required_headers + required_libs:
        if not os.path.exists(required_file):
            return [], []

    if verbose:
        msg_template = "DPNP: using ${} based library. include={}, libpath={}"
        print(msg_template.format(var_name, include_find, libpath_find))

    return [include_find], [libpath_find]


def _find_cmplr_in_dpcpp_root(verbose=False):
    """
    Find compiler in dpcpp root using $DPCPPROOT.

    Parameters
    ----------
    verbose : bool
        to print paths to include and library directories

    Returns
    -------
    tuple(list(str), list(str))
        path to include directory, path to library directory
    """
    rel_header_paths = rel_lib_paths = []

    if 'linux' in sys.platform:
        rel_include_path = os.path.join('linux', 'include')
        rel_libdir_path = os.path.join('linux', 'lib')
    elif sys.platform in ['win32', 'cygwin']:
        rel_include_path = os.path.join('windows', 'include')
        rel_libdir_path = os.path.join('windows', 'lib')
    else:
        rel_include_path, rel_libdir_path = 'include', 'lib'

    return find_library("DPCPPROOT", rel_header_paths, rel_lib_paths,
                        rel_include_path=rel_include_path, rel_libdir_path=rel_libdir_path, verbose=verbose)


def find_cmplr(verbose=False):
    """
    Find compiler in environment.

    Parameters
    ----------
    verbose : bool
        to print paths to include and library directories

    Returns
    -------
    tuple(list(str), list(str))
        path to include directory, path to library directory
    """
    cmplr_include, cmplr_libpath = _find_cmplr_in_dpcpp_root(verbose=verbose)

    if not cmplr_include or not cmplr_libpath:
        raise EnvironmentError(f"DPNP: Unable to find compiler. Please install Intel OneAPI environment")

    return cmplr_include, cmplr_libpath


def _find_dpl_in_oneapi_root(verbose=False):
    """
    Find DPL in oneAPI root using $ONEAPI_ROOT.

    Parameters
    ----------
    verbose : bool
        to print paths to include and library directories

    Returns
    -------
    tuple(list(str), list(str))
        path to include directory, path to library directory
    """
    rel_header_paths = rel_lib_paths = []

    if 'linux' in sys.platform:
        rel_include_path = os.path.join('dpl', 'latest', 'linux', 'include')
        rel_libdir_path = os.path.join('dpl', 'latest', 'linux', 'lib')
    elif sys.platform in ['win32', 'cygwin']:
        rel_include_path = os.path.join('dpl', 'latest', 'windows', 'include')
        rel_libdir_path = os.path.join('dpl', 'latest', 'windows', 'lib')
    else:
        rel_include_path, rel_libdir_path = 'include', 'lib'

    return find_library("ONEAPI_ROOT", rel_header_paths, rel_lib_paths,
                        rel_include_path=rel_include_path, rel_libdir_path=rel_libdir_path, verbose=verbose)


def find_dpl(verbose=False):
    """
    Find DPL in environment.

    Parameters
    ----------
    verbose : bool
        to print paths to include and library directories

    Returns
    -------
    tuple(list(str), list(str))
        path to include directory, path to library directory
    """
    dpl_include, dpl_libpath = _find_dpl_in_oneapi_root(verbose=verbose)

    if not dpl_include or not dpl_libpath:
        raise EnvironmentError(f"DPNP: Unable to find DPL. Please install Intel OneAPI environment")

    return dpl_include, dpl_libpath


def _find_mathlib_in_conda_root(verbose=False):
    """
    Find mathlib in conda root using $CONDA_PREFIX or $PREFIX.

    Parameters
    ----------
    verbose : bool
        to print paths to include and library directories

    Returns
    -------
    tuple(list(str), list(str))
        path to include directory, path to library directory
    """
    conda_root_var = "PREFIX" if IS_CONDA_BUILD else "CONDA_PREFIX"
    rel_header_paths = [os.path.join("oneapi", "mkl.h")]
    rel_lib_paths = ["libmkl_sycl.so"]

    return find_library(conda_root_var, rel_header_paths, rel_lib_paths, verbose=verbose)


def _find_mathlib_in_mathlib_root(verbose=False):
    """
    Find mathlib in mathlib root using $MKLROOT.

    Parameters
    ----------
    verbose : bool
        to print paths to include and library directories

    Returns
    -------
    tuple(list(str), list(str))
        path to include directory, path to library directory
    """
    rel_header_paths = ["mkl.h"]
    rel_lib_paths = ["libmkl_sycl.so"]
    rel_libdir_path = os.path.join("lib", "intel64")

    # return find_library("MKLROOT", rel_header_paths, rel_lib_paths, rel_libdir_path=rel_libdir_path, verbose=verbose)
    return (["/opt/intel/oneapi/mkl/latest/include"], ["/opt/intel/oneapi/mkl/latest/lib/intel64"])


def find_mathlib(verbose=False):
    """
    Find mathlib in conda root then in mathlib root.

    Parameters
    ----------
    verbose : bool
        to print paths to include and library directories

    Returns
    -------
    tuple(list(str), list(str))
        path to include directory, path to library directory
    """

    mathlib_include, mathlib_path = _find_mathlib_in_conda_root(verbose=verbose)

    if not mathlib_include or not mathlib_path:
        mathlib_include, mathlib_path = _find_mathlib_in_mathlib_root(verbose=verbose)

    if not mathlib_include or not mathlib_path:
        raise EnvironmentError("DPNP: Unable to find math library")

    return mathlib_include, mathlib_path


def _find_omp_in_dpcpp_root(verbose=False):
    """
    Find omp in dpcpp root using $DPCPPROOT.

    Parameters
    ----------
    verbose : bool
        to print paths to include and library directories

    Returns
    -------
    tuple(list(str), list(str))
        path to include directory, path to library directory
    """
    rel_header_paths = rel_lib_paths = []

    if 'linux' in sys.platform:
        rel_include_path = os.path.join('linux', 'compiler', 'include')
        rel_libdir_path = os.path.join('linux', 'compiler', 'lib', 'intel64')
    elif sys.platform in ['win32', 'cygwin']:
        rel_include_path = os.path.join('windows', 'compiler', 'include')
        rel_libdir_path = os.path.join('windows', 'compiler', 'lib', 'intel64_win')
    else:
        rel_include_path, rel_libdir_path = 'include', 'lib'

    return find_library("DPCPPROOT", rel_header_paths, rel_lib_paths,
                        rel_include_path=rel_include_path, rel_libdir_path=rel_libdir_path, verbose=verbose)


def find_omp(verbose=False):
    """
    Find omp in environment.

    Parameters
    ----------
    verbose : bool
        to print paths to include and library directories

    Returns
    -------
    tuple(list(str), list(str))
        path to include directory, path to library directory
    """
    omp_include, omp_libpath = _find_omp_in_dpcpp_root(verbose=verbose)

    if not omp_include or not omp_libpath:
        raise EnvironmentError(f"DPNP: Unable to find omp. Please install Intel OneAPI environment")

    return omp_include, omp_libpath
