# cython: language_level=3
# distutils: language = c++
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

"""
Interface of the DPNP

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""


import os
import numpy
import numpy.lib.stride_tricks as np_st
import dpnp.config as config
import collections

import dpctl

from dpnp.dpnp_algo import *
from dpnp.dpnp_utils import *
from dpnp.fft import *
from dpnp.linalg import *
from dpnp.random import *

__all__ = [
    "array_equal",
    "asnumpy",
    "astype",
    "convert_single_elem_array_to_scalar",
    "dpnp_queue_initialize",
    "dpnp_queue_is_cpu",
    "get_dpnp_descriptor",
    "get_include"
]

from dpnp.dpnp_iface_arraycreation import *
from dpnp.dpnp_iface_bitwise import *
from dpnp.dpnp_iface_counting import *
from dpnp.dpnp_iface_indexing import *
from dpnp.dpnp_iface_libmath import *
from dpnp.dpnp_iface_linearalgebra import *
from dpnp.dpnp_iface_logic import *
from dpnp.dpnp_iface_manipulation import *
from dpnp.dpnp_iface_mathematical import *
from dpnp.dpnp_iface_searching import *
from dpnp.dpnp_iface_sorting import *
from dpnp.dpnp_iface_statistics import *
from dpnp.dpnp_iface_trigonometric import *

from dpnp.dpnp_iface_arraycreation import __all__ as __all__arraycreation
from dpnp.dpnp_iface_bitwise import __all__ as __all__bitwise
from dpnp.dpnp_iface_counting import __all__ as __all__counting
from dpnp.dpnp_iface_indexing import __all__ as __all__indexing
from dpnp.dpnp_iface_libmath import __all__ as __all__libmath
from dpnp.dpnp_iface_linearalgebra import __all__ as __all__linearalgebra
from dpnp.dpnp_iface_logic import __all__ as __all__logic
from dpnp.dpnp_iface_manipulation import __all__ as __all__manipulation
from dpnp.dpnp_iface_mathematical import __all__ as __all__mathematical
from dpnp.dpnp_iface_searching import __all__ as __all__searching
from dpnp.dpnp_iface_sorting import __all__ as __all__sorting
from dpnp.dpnp_iface_statistics import __all__ as __all__statistics
from dpnp.dpnp_iface_trigonometric import __all__ as __all__trigonometric

__all__ += __all__arraycreation
__all__ += __all__bitwise
__all__ += __all__counting
__all__ += __all__indexing
__all__ += __all__libmath
__all__ += __all__linearalgebra
__all__ += __all__logic
__all__ += __all__manipulation
__all__ += __all__mathematical
__all__ += __all__searching
__all__ += __all__sorting
__all__ += __all__statistics
__all__ += __all__trigonometric


def array_equal(a1, a2, equal_nan=False):
    """
    True if two arrays have the same shape and elements, False otherwise.

    For full documentation refer to :obj:`numpy.array_equal`.

    See Also
    --------
    :obj:`dpnp.allclose` : Returns True if two arrays are element-wise equal
                           within a tolerance.
    :obj:`dpnp.array_equiv` : Returns True if input arrays are shape consistent
                              and all elements equal.

    """

    return numpy.array_equal(a1, a2)


def asnumpy(input, order='C'):
    """
    Returns the NumPy array with input data.

    Notes
    -----
    This function works exactly the same as :obj:`numpy.asarray`.

    """

    if isinstance(input, dpctl.tensor.usm_ndarray):
        return dpctl.tensor.to_numpy(input)

    if config.__DPNP_OUTPUT_DPCTL__ and hasattr(input, "__sycl_usm_array_interface__"):
        return dpctl.tensor.to_numpy(input._array_obj)

    return numpy.asarray(input, order=order)


def astype(x1, dtype, order='K', casting='unsafe', subok=True, copy=True):
    """Copy the array with data type casting."""
    if config.__DPNP_OUTPUT_DPCTL__ and hasattr(x1, "__sycl_usm_array_interface__"):
        import dpctl.tensor as dpt
        # TODO: remove check dpctl.tensor has attribute "astype"
        if hasattr(dpt, "astype"):
            # return dpt.astype(x1, dtype, order=order, casting=casting, copy=copy)
            return dpt.astype(x1._array_obj, dtype, order=order, casting=casting, copy=copy)

    x1_desc = get_dpnp_descriptor(x1)
    if not x1_desc:
        pass
    elif order != 'K':
        pass
    elif casting != 'unsafe':
        pass
    elif not subok:
        pass
    elif not copy:
        pass
    elif x1_desc.dtype == numpy.complex128 or dtype == numpy.complex128:
        pass
    elif x1_desc.dtype == numpy.complex64 or dtype == numpy.complex64:
        pass
    else:
        return dpnp_astype(x1_desc, dtype).get_pyobj()

    return call_origin(numpy.ndarray.astype, x1, dtype, order=order, casting=casting, subok=subok, copy=copy)


def convert_single_elem_array_to_scalar(obj, keepdims=False):
    """
    Convert array with single element to scalar
    """

    if (obj.ndim > 0) and (obj.size == 1) and (keepdims is False):
        return obj.dtype.type(obj[0])

    return obj


def get_dpnp_descriptor(ext_obj, copy_when_strides=True):
    """
    Return True:
      never
    Return DPNP internal data discriptor object if:
      1. We can proceed with input data object with DPNP
      2. We want to handle input data object
    Return False if:
      1. We do not want to work with input data object
      2. We can not handle with input data object
    """

    # TODO need to allow "import dpnp" with no build procedure
    # if no_modules_load_doc_build();
    #    return False

    if use_origin_backend():
        return False

    # while dpnp functions have no implementation with strides support
    # we need to create a non-strided copy
    # if function get implementation for strides case
    # then this behavior can be disabled with setting "copy_when_strides"
    if copy_when_strides and getattr(ext_obj, "strides", None) is not None:
        # TODO: replace this workaround when usm_ndarray will provide such functionality
        shape_offsets = tuple(numpy.prod(ext_obj.shape[i+1:], dtype=numpy.int64) for i in range(ext_obj.ndim))
        if ext_obj.strides != shape_offsets:
            ext_obj = array(ext_obj)

    dpnp_desc = dpnp_descriptor(ext_obj)
    if dpnp_desc.is_valid:
        return dpnp_desc

    return False


def get_include():
    """
    Return the directory that contains the DPNP C++ backend \\*.h header files.
    """

    dpnp_path = os.path.join(os.path.dirname(__file__), "backend", "include")

    return dpnp_path
