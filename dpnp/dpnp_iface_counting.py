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


import dpnp
import numpy

import dpnp.config as config
from dpnp.dpnp_utils import *

from dpnp.dpnp_algo.dpnp_algo import *  # TODO need to investigate why dpnp.dpnp_algo can not be used

__all__ = [
    'count_nonzero'
]


def count_nonzero(x1, axis=None, *, keepdims=False):
    """
    Counts the number of non-zero values in the array ``in_array1``.

    For full documentation refer to :obj:`numpy.count_nonzero`.

    Limitations
    -----------
        Parameter ``x1`` is supported as :obj:`dpnp.ndarray`.
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
    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if axis is not None:
            pass
        elif keepdims is not False:
            pass
        else:
            result_obj = dpnp_count_nonzero(x1_desc).get_pyobj()
            result = dpnp.convert_single_elem_array_to_scalar(result_obj)

            return result

    return call_origin(numpy.count_nonzero, x1, axis, keepdims=keepdims)
