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
Interface of the counting function of the dpnp

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""


from dpnp.dpnp_algo.dpnp_algo import *  # TODO need to investigate why dpnp.dpnp_algo can not be used
from dpnp.dparray import dparray

# full module name because dpnp_iface_counting loaded from cython too early
from dpnp.dpnp_utils.dpnp_algo_utils import *

import dpnp
import numpy

__all__ = [
    'count_nonzero'
]


def count_nonzero(in_array1, axis=None, *, keepdims=False):
    """
    Counts the number of non-zero values in the array ``in_array1``.

    For full documentation refer to :obj:`numpy.count_nonzero`.

    Limitations
    -----------
        Parameter ``in_array1`` is supported as :obj:`dpnp.ndarray`.
        Otherwise the function will be executed sequentially on CPU.
        Parameter ``axis`` is supported only with default value `None`.
        Parameter ``keepdims`` is supported only with default value `False`.

    Examples
    --------
    >>> import dpnp as np
    >>> np.count_nonzero(np.array([1, 0, 3, 0, 5])
    3
    >>> np.count_nonzero(np.array([[1, 0, 3, 0, 5],[0, 9, 0, 7, 0]]))
    5

    """

    is_dparray1 = isinstance(in_array1, dparray)

    if (not use_origin_backend(in_array1) and is_dparray1):
        if axis is not None:
            checker_throw_value_error("count_nonzero", "axis", type(axis), None)
        if keepdims is not False:
            checker_throw_value_error("count_nonzero", "keepdims", keepdims, False)

        result_obj = dpnp_count_nonzero(in_array1)
        result = dpnp.convert_single_elem_array_to_scalar(result_obj)

        return result

    return numpy.count_nonzero(in_array1, axis, keepdims=keepdims)
