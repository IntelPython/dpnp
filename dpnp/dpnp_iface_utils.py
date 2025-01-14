# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2024-2025, Intel Corporation
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
Interface of the utils function of the DPNP

Notes
-----
This module is a face or public interface file for the library

"""

import dpnp

__all__ = ["byte_bounds"]


def byte_bounds(a):
    """
    Returns a 2-tuple with pointers to the end-points of the array.

    For full documentation refer to :obj:`numpy.lib.array_utils.byte_bounds`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array

    Returns
    -------
    (low, high) : tuple of 2 integers
        The first integer is the first byte of the array, the second integer is
        just past the last byte of the array. If `a` is not contiguous it will
        not use every byte between the (`low`, `high`) values.

    Examples
    --------
    >>> import dpnp as np
    >>> I = np.eye(2, dtype=np.complex64);
    >>> low, high = np.byte_bounds(I)
    >>> high - low == I.size*I.itemsize
    True
    >>> I = np.eye(2);
    >>> low, high = np.byte_bounds(I)
    >>> high - low == I.size*I.itemsize
    True

    """

    # pylint: disable=protected-access
    return dpnp.get_usm_ndarray(a)._byte_bounds
