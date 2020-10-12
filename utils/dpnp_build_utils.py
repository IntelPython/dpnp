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


IS_CONDA_BUILD = os.environ.get("CONDA_BUILD") == "1"


def _find_mkl_by_var(name, rel_include_path="include", rel_libdir_path="lib",
                     rel_header_path="mkl_blas_sycl.hpp", rel_lib_path="libmkl_sycl.so", verbose=False):
    """
    Find MKL in the directory from the environment variable.
    Parameters
    ----------
    name : str
        the name of the environemnt variable
    rel_include_path : str
        relative path to the include directory
    rel_libdir_path : str
        relative path to the library directory
    rel_header_path : str
        relative path to the header
    rel_lib_path : str
        relative path to the library
    verbose : bool
        to print paths to include and library directories
    Returns
    -------
    tuple(list(str), list(str))
        path to include directory, path to library directory
    """
    root_dir = os.environ.get(name)
    if root_dir is None:
        return [], []

    mkl_include_find = os.path.join(root_dir, rel_include_path)
    mkl_libpath_find = os.path.join(root_dir, rel_libdir_path)
    required_header = os.path.join(mkl_include_find, rel_header_path)
    required_library = os.path.join(mkl_libpath_find, rel_lib_path)

    for required_file in [required_header, required_library]:
        if not os.path.exists(required_file):
            return [], []

    if verbose:
        msg_template = "Intel DPNP: using ${} based math library. include={}, libpath={}"
        print(msg_template.format(name, mkl_include_find, mkl_libpath_find))

    return [mkl_include_find], [mkl_libpath_find]


def _find_mkl_in_conda_root(verbose=False):
    """
    Find MKL in conda root using $CONDA_PREFIX or $PREFIX.
    Parameters
    ----------
    verbose : bool
        to print paths to include and library directories
    Returns
    -------
    tuple(list(str), list(str))
        path to include directory, path to library directory
    """
    _conda_root_var = "PREFIX" if IS_CONDA_BUILD else "CONDA_PREFIX"
    rel_header_path = os.path.join("oneapi", "mkl.hpp")
    return _find_mkl_by_var(_conda_root_var, rel_header_path=rel_header_path, verbose=verbose)


def _find_mkl_in_mkl_root(verbose=False):
    """
    Find MKL in MKL root using $MKLROOT.
    Parameters
    ----------
    verbose : bool
        to print paths to include and library directories
    Returns
    -------
    tuple(list(str), list(str))
        path to include directory, path to library directory
    """
    rel_libdir_path = os.path.join("lib", "intel64")
    return _find_mkl_by_var("MKLROOT", rel_libdir_path=rel_libdir_path, verbose=verbose)


def find_mkl(verbose=False):
    """
    Find MKL in conda root then in MKL root.
    Parameters
    ----------
    verbose : bool
        to print paths to include and library directories
    Returns
    -------
    tuple(list(str), list(str))
        path to include directory, path to library directory
    """

    _mkl_include, _mkl_libpath = _find_mkl_in_conda_root(verbose=verbose)

    if not _mkl_include or not _mkl_libpath:
        _mkl_include, _mkl_libpath = _find_mkl_in_mkl_root(verbose=verbose)

    if not _mkl_include or not _mkl_libpath:
        raise EnvironmentError("Intel DPNP: Unable to find math library")

    return _mkl_include, _mkl_libpath
