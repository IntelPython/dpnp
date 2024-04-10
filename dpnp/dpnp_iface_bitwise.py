# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2024, Intel Corporation
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
Interface of the Binary operations of the DPNP

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""

# pylint: disable=protected-access
# pylint: disable=c-extension-no-member


import dpctl.tensor._tensor_elementwise_impl as ti

from dpnp.dpnp_algo.dpnp_elementwise_common import DPNPBinaryFunc, DPNPUnaryFunc

__all__ = [
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "invert",
    "left_shift",
    "right_shift",
]


_BITWISE_AND_DOCSTRING = """
Computes the bitwise AND of the underlying binary representation of each
element `x1_i` of the input array `x1` with the respective element `x2_i`
of the input array `x2`.

For full documentation refer to :obj:`numpy.bitwise_and`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray}
    First input array, expected to have integer or boolean data type.
x2 : {dpnp.ndarray, usm_ndarray}
    Second input array, also expected to have integer or boolean data
    type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise results. The data type
    of the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.logical_and` : Compute the truth value of ``x1`` AND ``x2`` element-wise.
:obj:`dpnp.bitwise_or`: Compute the bit-wise OR of two arrays element-wise.
:obj:`dpnp.bitwise_xor` : Compute the bit-wise XOR of two arrays element-wise.

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([2, 5, 255])
>>> x2 = np.array([3,14,16])
>>> np.bitwise_and(x1, x2)
[2, 4, 16]

>>> a = np.array([True, True])
>>> b = np.array([False, True])
>>> np.bitwise_and(a, b)
array([False,  True])

The ``&`` operator can be used as a shorthand for ``bitwise_and`` on
:class:`dpnp.ndarray`.

>>> x1 & x2
array([ 2,  4, 16])
"""

bitwise_and = DPNPBinaryFunc(
    "bitwise_and",
    ti._bitwise_and_result_type,
    ti._bitwise_and,
    _BITWISE_AND_DOCSTRING,
)


_BITWISE_OR_DOCSTRING = """
Computes the bitwise OR of the underlying binary representation of each
element `x1_i` of the input array `x1` with the respective element `x2_i`
of the input array `x2`.

For full documentation refer to :obj:`numpy.bitwise_or`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray}
    First input array, expected to have integer or boolean data type.
x2 : {dpnp.ndarray, usm_ndarray}
    Second input array, also expected to have integer or boolean data
    type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise results. The data type
    of the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.logical_or` : Compute the truth value of ``x1`` OR ``x2`` element-wise.
:obj:`dpnp.bitwise_and`: Compute the bit-wise AND of two arrays element-wise.
:obj:`dpnp.bitwise_xor` : Compute the bit-wise XOR of two arrays element-wise.

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([2, 5, 255])
>>> x2 = np.array([4])
>>> np.bitwise_or(x1, x2)
array([  6,   5, 255])

The ``|`` operator can be used as a shorthand for ``bitwise_or`` on
:class:`dpnp.ndarray`.

>>> x1 | x2
array([  6,   5, 255])
"""

bitwise_or = DPNPBinaryFunc(
    "bitwise_or",
    ti._bitwise_or_result_type,
    ti._bitwise_or,
    _BITWISE_OR_DOCSTRING,
)


_BITWISE_XOR_DOCSTRING = """
Computes the bitwise XOR of the underlying binary representation of each
element `x1_i` of the input array `x1` with the respective element `x2_i`
of the input array `x2`.

For full documentation refer to :obj:`numpy.bitwise_xor`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray}
    First input array, expected to have integer or boolean data type.
x2 : {dpnp.ndarray, usm_ndarray}
    Second input array, also expected to have integer or boolean data
    type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise results. The data type
    of the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.logical_xor` : Compute the truth value of ``x1`` XOR `x2`, element-wise.
:obj:`dpnp.bitwise_and`: Compute the bit-wise AND of two arrays element-wise.
:obj:`dpnp.bitwise_or` : Compute the bit-wise OR of two arrays element-wise.

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([31, 3])
>>> x2 = np.array([5, 6])
>>> np.bitwise_xor(x1, x2)
array([26,  5])

>>> a = np.array([True, True])
>>> b = np.array([False, True])
>>> np.bitwise_xor(a, b)
array([ True, False])

The ``^`` operator can be used as a shorthand for ``bitwise_xor`` on
:class:`dpnp.ndarray`.

>>> a ^ b
array([ True, False])
"""

bitwise_xor = DPNPBinaryFunc(
    "bitwise_xor",
    ti._bitwise_xor_result_type,
    ti._bitwise_xor,
    _BITWISE_XOR_DOCSTRING,
)


_INVERT_DOCSTRING = """
Inverts (flips) each bit for each element `x_i` of the input array `x`.

For full documentation refer to :obj:`numpy.invert`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have integer or boolean data type.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise results.
    The data type of the returned array is same as the data type of the
    input array.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.bitwise_and`: Compute the bit-wise AND of two arrays element-wise.
:obj:`dpnp.bitwise_or` : Compute the bit-wise OR of two arrays element-wise.
:obj:`dpnp.bitwise_xor` : Compute the bit-wise XOR of two arrays element-wise.
:obj:`dpnp.logical_not` : Compute the truth value of NOT x element-wise.

Examples
--------
>>> import dpnp as np
>>> x = np.array([13])
>>> np.invert(x)
-14

>>> a = np.array([True, False])
>>> np.invert(a)
array([False,  True])

The ``~`` operator can be used as a shorthand for ``invert`` on
:class:`dpnp.ndarray`.

>>> ~a
array([False,  True])
"""

invert = DPNPUnaryFunc(
    "invert",
    ti._bitwise_invert_result_type,
    ti._bitwise_invert,
    _INVERT_DOCSTRING,
)


bitwise_not = invert  # bitwise_not is an alias for invert


_LEFT_SHIFT_DOCSTRING = """
Shifts the bits of each element `x1_i` of the input array x1 to the left by
appending `x2_i` (i.e., the respective element in the input array `x2`) zeros to
the right of `x1_i`.

For full documentation refer to :obj:`numpy.left_shift`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray}
    First input array, expected to have integer data type.
x2 : {dpnp.ndarray, usm_ndarray}
    Second input array, also expected to have integer data type.
    Each element must be greater than or equal to 0.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".
Returns
-------
out : dpnp.ndarray
    An array containing the element-wise results. The data type
    of the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.right_shift` : Shift the bits of an integer to the right.

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([5])
>>> x2 = np.array([1, 2, 3])
>>> np.left_shift(x1, x2)
array([10, 20, 40])

The ``<<`` operator can be used as a shorthand for ``left_shift`` on
:class:`dpnp.ndarray`.

>>> x1 << x2
array([10, 20, 40])
"""

left_shift = DPNPBinaryFunc(
    "left_shift",
    ti._bitwise_left_shift_result_type,
    ti._bitwise_left_shift,
    _LEFT_SHIFT_DOCSTRING,
)


_RIGHT_SHIFT_DOCSTRING = """
Shifts the bits of each element `x1_i` of the input array `x1` to the right
according to the respective element `x2_i` of the input array `x2`.

For full documentation refer to :obj:`numpy.right_shift`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray}
    First input array, expected to have integer data type.
x2 : {dpnp.ndarray, usm_ndarray}
    Second input array, also expected to have integer data type.
    Each element must be greater than or equal to 0.
out : {None, dpnp.ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: "K".

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise results. The data type
    of the returned array is determined by the Type Promotion Rules.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.left_shift` : Shift the bits of an integer to the left.

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([10])
>>> x2 = np.array([1, 2, 3])
>>> np.right_shift(x1, x2)
array([5, 2, 1])

The ``>>`` operator can be used as a shorthand for ``right_shift`` on
:class:`dpnp.ndarray`.

>>> x1 >> x2
array([5, 2, 1])
"""

right_shift = DPNPBinaryFunc(
    "right_shift",
    ti._bitwise_right_shift_result_type,
    ti._bitwise_right_shift,
    _RIGHT_SHIFT_DOCSTRING,
)
