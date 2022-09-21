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


def find_cmplr(verbose=False):
    """
    Find compiler.

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

    # try to find library in specified directory from $DPCPPROOT
    if 'linux' in sys.platform:
        rel_include_path = os.path.join('linux', 'include')
        rel_libdir_path = os.path.join('linux', 'lib')
    elif sys.platform in ['win32', 'cygwin']:
        rel_include_path = os.path.join('windows', 'include')
        rel_libdir_path = os.path.join('windows', 'lib')
    else:
        raise EnvironmentError("DPNP: " + sys.platform + " not supported")

    cmplr_include, cmplr_libpath = find_library("DPCPPROOT", rel_header_paths, rel_lib_paths,
                                                rel_include_path=rel_include_path,
                                                rel_libdir_path=rel_libdir_path,
                                                verbose=verbose)

    # try to find library in specified directory from $ONEAPI_ROOT
    if not cmplr_include or not cmplr_libpath:
        if sys.platform in ['linux']:
            rel_include_path = os.path.join('compiler', 'latest', 'linux', 'include')
            rel_libdir_path = os.path.join('compiler', 'latest', 'linux', 'lib')
        elif sys.platform in ['win32', 'cygwin']:
            rel_include_path = os.path.join('compiler', 'latest', 'windows', 'include')
            rel_libdir_path = os.path.join('compiler', 'latest', 'windows', 'lib')
        else:
            raise EnvironmentError("DPNP: " + sys.platform + " not supported")

        cmplr_include, cmplr_libpath = find_library("ONEAPI_ROOT", rel_header_paths, rel_lib_paths,
                                                    rel_include_path=rel_include_path,
                                                    rel_libdir_path=rel_libdir_path,
                                                    verbose=verbose)

    # try to find in Python environment
    if not cmplr_include or not cmplr_libpath:
        if sys.platform in ['linux']:
            rel_include_path = os.path.join('include')
            rel_libdir_path = os.path.join('lib')
        elif sys.platform in ['win32', 'cygwin']:
            rel_include_path = os.path.join('Library', 'include')
            rel_libdir_path = os.path.join('Library', 'lib')
        else:
            raise EnvironmentError("DPNP: " + sys.platform + " not supported")

        conda_root_var = "PREFIX" if IS_CONDA_BUILD else "CONDA_PREFIX"

        cmplr_include, cmplr_libpath = find_library(conda_root_var, rel_header_paths, rel_lib_paths,
                                                    rel_include_path=rel_include_path,
                                                    rel_libdir_path=rel_libdir_path,
                                                    verbose=verbose)

    if not cmplr_include or not cmplr_libpath:
        raise EnvironmentError("DPNP: Unable to find compiler")

    return cmplr_include, cmplr_libpath


def find_dpl(verbose=False):
    """
    Find DPL.

    Parameters
    ----------
    verbose : bool
        to print paths to include and library directories

    Returns
    -------
    tuple(list(str), list(str))
        path to include directory, path to library directory
    """

    rel_header_paths = [os.path.join("oneapi", "dpl", "algorithm")]
    rel_lib_paths = []
    rel_libdir_path = ""

    # try to find library in specified directory from $DPLROOT like a repository
    rel_include_path = os.path.join('include')

    dpl_include, dpl_libpath = find_library("DPLROOT", rel_header_paths, rel_lib_paths,
                                            rel_include_path=rel_include_path,
                                            rel_libdir_path=rel_libdir_path,
                                            verbose=verbose)

    # try to find library in specified directory from $DPLROOT
    if not dpl_include or not dpl_libpath:
        if 'linux' in sys.platform:
            rel_include_path = os.path.join('linux', 'include')
        elif sys.platform in ['win32', 'cygwin']:
            rel_include_path = os.path.join('windows', 'include')
        else:
            raise EnvironmentError("DPNP: " + sys.platform + " not supported")

        dpl_include, dpl_libpath = find_library("DPLROOT", rel_header_paths, rel_lib_paths,
                                                rel_include_path=rel_include_path,
                                                rel_libdir_path=rel_libdir_path,
                                                verbose=verbose)

    # try to find library in specified directory from $ONEAPI_ROOT
    if not dpl_include or not dpl_libpath:
        if sys.platform in ['linux']:
            rel_include_path = os.path.join('dpl', 'latest', 'linux', 'include')
        elif sys.platform in ['win32', 'cygwin']:
            rel_include_path = os.path.join('dpl', 'latest', 'windows', 'include')
        else:
            raise EnvironmentError("DPNP: " + sys.platform + " not supported")

        dpl_include, dpl_libpath = find_library("ONEAPI_ROOT", rel_header_paths, rel_lib_paths,
                                                rel_include_path=rel_include_path,
                                                rel_libdir_path=rel_libdir_path,
                                                verbose=verbose)

    # try to find in Python environment
    if not dpl_include or not dpl_libpath:
        if sys.platform in ['linux']:
            rel_include_path = os.path.join('include')
        elif sys.platform in ['win32', 'cygwin']:
            rel_include_path = os.path.join('Library', 'include')
        else:
            raise EnvironmentError("DPNP: " + sys.platform + " not supported")

        conda_root_var = "PREFIX" if IS_CONDA_BUILD else "CONDA_PREFIX"

        dpl_include, dpl_libpath = find_library(conda_root_var, rel_header_paths, rel_lib_paths,
                                                rel_include_path=rel_include_path,
                                                rel_libdir_path=rel_libdir_path,
                                                verbose=verbose)

    if not dpl_include or not dpl_libpath:
        raise EnvironmentError("DPNP: Unable to find DPL")

    return dpl_include, dpl_libpath


def find_mathlib(verbose=False):
    """
    Find mathlib.

    Parameters
    ----------
    verbose : bool
        to print paths to include and library directories

    Returns
    -------
    tuple(list(str), list(str))
        path to include directory, path to library directory
    """

    if sys.platform in ['linux']:
        rel_header_paths = [os.path.join("oneapi", "mkl.hpp")]
        rel_lib_paths = ["libmkl_sycl.so"]
    elif sys.platform in ['win32', 'cygwin']:
        rel_header_paths = [os.path.join("oneapi", "mkl.hpp")]
        rel_lib_paths = ["mkl_sycl_dll.lib"]
    else:
        raise EnvironmentError("DPNP: " + sys.platform + " not supported")

    # try to find library in specified directory from $MKLROOT
    if sys.platform in ['linux']:
        rel_include_path = os.path.join('linux', 'include')
        rel_libdir_path = os.path.join('linux', 'lib')
    elif sys.platform in ['win32', 'cygwin']:
        rel_include_path = os.path.join('windows', 'include')
        rel_libdir_path = os.path.join('windows', 'lib')
    else:
        raise EnvironmentError("DPNP: " + sys.platform + " not supported")

    mathlib_include, mathlib_path = find_library("MKLROOT", rel_header_paths, rel_lib_paths,
                                                 rel_include_path=rel_include_path,
                                                 rel_libdir_path=rel_libdir_path,
                                                 verbose=verbose)

    # try to find library in specified directory from $ONEAPI_ROOT
    if not mathlib_include or not mathlib_path:
        if sys.platform in ['linux']:
            rel_include_path = os.path.join('mkl', 'latest', 'linux', 'include')
            rel_libdir_path = os.path.join('mkl', 'latest', 'linux', 'lib')
        elif sys.platform in ['win32', 'cygwin']:
            rel_include_path = os.path.join('mkl', 'latest', 'windows', 'include')
            rel_libdir_path = os.path.join('mkl', 'latest', 'windows', 'lib')
        else:
            raise EnvironmentError("DPNP: " + sys.platform + " not supported")

        mathlib_include, mathlib_path = find_library("ONEAPI_ROOT", rel_header_paths, rel_lib_paths,
                                                     rel_include_path=rel_include_path,
                                                     rel_libdir_path=rel_libdir_path,
                                                     verbose=verbose)

    # try to find in Python environment
    if not mathlib_include or not mathlib_path:
        if sys.platform in ['linux']:
            rel_include_path = os.path.join('include')
            rel_libdir_path = os.path.join('lib')
        elif sys.platform in ['win32', 'cygwin']:
            rel_include_path = os.path.join('Library', 'include')
            rel_libdir_path = os.path.join('Library', 'lib')
        else:
            raise EnvironmentError("DPNP: " + sys.platform + " not supported")

        conda_root_var = "PREFIX" if IS_CONDA_BUILD else "CONDA_PREFIX"

        mathlib_include, mathlib_path = find_library(conda_root_var, rel_header_paths, rel_lib_paths,
                                                     rel_include_path=rel_include_path,
                                                     rel_libdir_path=rel_libdir_path,
                                                     verbose=verbose)

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


def find_python_env(verbose=False):
    """
    Find Python environment.

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

    if sys.platform in ['linux']:
        rel_include_path = os.path.join('include')
        rel_libdir_path = os.path.join('lib')
    elif sys.platform in ['win32', 'cygwin']:
        rel_include_path = os.path.join('Library', 'include')
        rel_libdir_path = os.path.join('Library', 'lib')
    else:
        raise EnvironmentError("DPNP: " + sys.platform + " not supported")

    conda_root_var = "PREFIX" if IS_CONDA_BUILD else "CONDA_PREFIX"

    env_include, env_path = find_library(conda_root_var, rel_header_paths, rel_lib_paths,
                                         rel_include_path=rel_include_path,
                                         rel_libdir_path=rel_libdir_path,
                                         verbose=verbose)

    env_include += [os.path.join(os.getenv(conda_root_var), 'include')]

    if not env_include or not env_path:
        raise EnvironmentError(f"DPNP: Unable to find Python environment paths")

    return env_include, env_path
