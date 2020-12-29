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
import collections

from dpnp.dpnp_algo import *
from dpnp.dparray import dparray
from dpnp.dpnp_utils import *
from dpnp.fft import *
from dpnp.linalg import *
from dpnp.random import *

__all__ = [
    "array_equal",
    "asnumpy",
    "dpnp_queue_initialize",
    "dpnp_queue_is_cpu",
    "get_include",
    "matmul"
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

    return numpy.asarray(input, order=order)


def get_include():
    """
    Return the directory that contains the DPNP C++ backend \\*.h header files.
    """

    dpnp_path = os.path.join(os.path.dirname(__file__), "backend", "include")

    return dpnp_path


def matmul(in_array1, in_array2, out=None):
    """
    Matrix product of two arrays.

    For full documentation refer to :obj:`numpy.matmul`.

    Limitations
    -----------
    Input arrays are supported as :obj:`dpnp.ndarray`.
    Otherwise the function will be executed sequentially on CPU.
    Parameter ``out`` is supported only with default value ``None``.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.vdot` : Complex-conjugating dot product.
    :obj:`dpnp.tensordot` : Sum products over arbitrary axes.
    :obj:`dpnp.einsum` : Einstein summation convention.
    :obj:`dpnp.dot` : Alternative matrix product with
                      different broadcasting rules.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.ones([9, 5, 7, 4])
    >>> c = np.ones([9, 5, 4, 3])
    >>> np.matmul(a, c).shape
    (9, 5, 7, 3)
    >>> a = np.array([[1, 0], [0, 1]])
    >>> b = np.array([[4, 1], [2, 2]])
    >>> np.matmul(a, b)
    array([[4, 1],
           [2, 2]])

    """

    is_dparray1 = isinstance(in_array1, dparray)
    is_dparray2 = isinstance(in_array2, dparray)

    if (not use_origin_backend(in_array1) and is_dparray1 and is_dparray2):

        if out is not None:
            checker_throw_value_error("matmul", "out", type(out), None)

        """
        Cost model checks
        """
        cost_size = 4096  # 2D array shape(64, 64)
        if ((in_array1.dtype == numpy.float64) or (in_array1.dtype == numpy.float32)):
            """
            Floating point types are handled via original math library better than SYCL math library
            """
            cost_size = 262144  # 2D array shape(512, 512)

        dparray1_size = in_array1.size
        dparray2_size = in_array2.size

        if (dparray1_size > cost_size) and (dparray2_size > cost_size):
            # print(f"dparray1_size={dparray1_size}")
            return dpnp_matmul(in_array1, in_array2)

    input1 = asnumpy(in_array1) if is_dparray1 else in_array1
    input2 = asnumpy(in_array2) if is_dparray2 else in_array2

    # TODO need to return dparray instead ndarray
    return numpy.matmul(input1, input2, out=out)
