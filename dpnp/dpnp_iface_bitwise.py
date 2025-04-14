# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2025, Intel Corporation
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
Interface of the Bitwise part of the DPNP

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""

# pylint: disable=protected-access
# pylint: disable=no-name-in-module

import dpctl.tensor._tensor_elementwise_impl as ti
import numpy

import dpnp.backend.extensions.ufunc._ufunc_impl as ufi
from dpnp.dpnp_algo.dpnp_elementwise_common import DPNPBinaryFunc, DPNPUnaryFunc

__all__ = [
    "binary_repr",
    "bitwise_and",
    "bitwise_count",
    "bitwise_invert",
    "bitwise_left_shift",
    "bitwise_not",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "invert",
    "left_shift",
    "right_shift",
]


def binary_repr(num, width=None):
    """
    Return the binary representation of the input number as a string.

    For negative numbers, if `width` is not given, a minus sign is added to the
    front. If `width` is given, the two's complement of the number is returned,
    with respect to that width.

    In a two's-complement system negative numbers are represented by the two's
    complement of the absolute value. A N-bit two's-complement system can
    represent every integer in the range :math:`-2^{N-1}` to :math:`+2^{N-1}-1`.

    For full documentation refer to :obj:`numpy.binary_repr`.

    Parameters
    ----------
    num : int
        Only an integer decimal number can be used.
    width : {None, int}, optional
        The length of the returned string if `num` is positive, or the length
        of the two's complement if `num` is negative, provided that `width` is
        at least a sufficient number of bits for `num` to be represented in the
        designated form. If the `width` value is insufficient, an error is
        raised.

        Default: ``None``.

    Returns
    -------
    bin : str
        Binary representation of `num` or two's complement of `num`.

    See Also
    --------
    :obj:`dpnp.base_repr` : Return a string representation of a number in the
                            given base system.
    bin : Python's built-in binary representation generator of an integer.

    Notes
    -----
    :obj:`dpnp.binary_repr` is equivalent to using :obj:`dpnp.base_repr` with
    base 2, but significantly faster.

    Examples
    --------
    >>> import numpy as np
    >>> np.binary_repr(3)
    '11'
    >>> np.binary_repr(-3)
    '-11'
    >>> np.binary_repr(3, width=4)
    '0011'

    The two's complement is returned when the input number is negative and
    `width` is specified:

    >>> np.binary_repr(-3, width=3)
    '101'
    >>> np.binary_repr(-3, width=5)
    '11101'

    """

    return numpy.binary_repr(num, width)


_BITWISE_AND_DOCSTRING = """
Computes the bitwise AND of the underlying binary representation of each
element :math:`x1_i` of the input array `x1` with the respective element
:math:`x2_i` of the input array `x2`.

For full documentation refer to :obj:`numpy.bitwise_and`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have an integer or boolean data type.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to have an integer or boolean data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

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
:obj:`dpnp.binary_repr` : Return the binary representation of the input number
                          as a string.

Notes
-----
At least one of `x1` or `x2` must be an array.

If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
(which becomes the shape of the output).

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([2, 5, 255])
>>> x2 = np.array([3, 14, 16])
>>> np.bitwise_and(x1, x2)
array([ 2,  4, 16])

>>> a = np.array([True, True])
>>> b = np.array([False, True])
>>> np.bitwise_and(a, b)
array([False,  True])

The ``&`` operator can be used as a shorthand for ``bitwise_and`` on
:class:`dpnp.ndarray`.

>>> x1 & x2
array([ 2,  4, 16])

The number 13 is represented by ``00001101``. Likewise, 17 is represented by
``00010001``. The bit-wise AND of 13 and 17 is therefore ``000000001``, or 1:

>>> np.bitwise_and(np.array(13), 17)
array(1)

>>> np.bitwise_and(np.array(14), 13)
array(12)
>>> np.binary_repr(12)
'1100'
>>> np.bitwise_and(np.array([14, 3]), 13)
array([12,  1])

"""

bitwise_and = DPNPBinaryFunc(
    "bitwise_and",
    ti._bitwise_and_result_type,
    ti._bitwise_and,
    _BITWISE_AND_DOCSTRING,
    binary_inplace_fn=ti._bitwise_and_inplace,
)


_BITWISE_COUNT_DOCSTRING = """
Computes the number of 1-bits in the absolute value of `x`.

For full documentation refer to :obj:`numpy.bitwise_count`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have an integer data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    The corresponding number of 1-bits in the input. Returns ``uint8`` for all
    integer types.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Keyword argument `kwargs` is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

Examples
--------
>>> import dpnp as np
>>> a = np.array(1023)
>>> np.bitwise_count(a)
array(10, dtype=uint8)

>>> a = np.array([2**i - 1 for i in range(16)])
>>> np.bitwise_count(a)
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
      dtype=uint8)

"""

bitwise_count = DPNPUnaryFunc(
    "bitwise_count",
    ufi._bitwise_count_result_type,
    ufi._bitwise_count,
    _BITWISE_COUNT_DOCSTRING,
)


_BITWISE_OR_DOCSTRING = """
Computes the bitwise OR of the underlying binary representation of each
element :math:`x1_i` of the input array `x1` with the respective element
:math:`x2_i` of the input array `x2`.

For full documentation refer to :obj:`numpy.bitwise_or`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have an integer or boolean data type.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to have an integer or boolean data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

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
:obj:`dpnp.binary_repr` : Return the binary representation of the input number
                          as a string.

Notes
-----
At least one of `x1` or `x2` must be an array.

If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
(which becomes the shape of the output).

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

The number 13 has the binary representation ``00001101``. Likewise, 16 is
represented by ``00010000``. The bit-wise OR of 13 and 16 is then ``00011101``,
or 29:

>>> np.bitwise_or(np.array(13), 16)
array(29)
>>> np.binary_repr(29)
'11101'

"""

bitwise_or = DPNPBinaryFunc(
    "bitwise_or",
    ti._bitwise_or_result_type,
    ti._bitwise_or,
    _BITWISE_OR_DOCSTRING,
    binary_inplace_fn=ti._bitwise_or_inplace,
)


_BITWISE_XOR_DOCSTRING = """
Computes the bitwise XOR of the underlying binary representation of each
element :math:`x1_i` of the input array `x1` with the respective element
:math:`x2_i` of the input array `x2`.

For full documentation refer to :obj:`numpy.bitwise_xor`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have an integer or boolean data type.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to have an integer or boolean data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

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
:obj:`dpnp.binary_repr` : Return the binary representation of the input number
                          as a string.

Notes
-----
At least one of `x1` or `x2` must be an array.

If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
(which becomes the shape of the output).

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

The number 13 is represented by ``00001101``. Likewise, 17 is represented by
``00010001``. The bit-wise XOR of 13 and 17 is therefore ``00011100``, or 28:

>>> np.bitwise_xor(np.array(13), 17)
array(28)
>>> np.binary_repr(28)
'11100'

"""

bitwise_xor = DPNPBinaryFunc(
    "bitwise_xor",
    ti._bitwise_xor_result_type,
    ti._bitwise_xor,
    _BITWISE_XOR_DOCSTRING,
    binary_inplace_fn=ti._bitwise_xor_inplace,
)


_INVERT_DOCSTRING = """
Inverts (flips) each bit for each element :math:`x_i` of the input array `x`.

Note that :obj:`dpnp.bitwise_invert` is an alias of :obj:`dpnp.invert`.

For full documentation refer to :obj:`numpy.invert`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have an integer or boolean data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

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
:obj:`dpnp.binary_repr` : Return the binary representation of the input number
                          as a string.

Examples
--------
>>> import dpnp as np

The number 13 is represented by ``00001101``. The invert or bit-wise NOT of 13
is then:

>>> x = np.array([13])
>>> np.invert(x)
array([-14])
>>> np.binary_repr(-14, width=8)
'11110010'

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
bitwise_invert = invert  # bitwise_invert is an alias for invert

_LEFT_SHIFT_DOCSTRING = """
Shifts the bits of each element :math:`x1_i` of the input array `x1` to the
left by appending :math:`x2_i` (i.e., the respective element in the input array
`x2`) zeros to the right of :math:`x1_i`.

Note that :obj:`dpnp.bitwise_left_shift` is an alias of :obj:`dpnp.left_shift`.

For full documentation refer to :obj:`numpy.left_shift`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have an integer data type.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to have an integer data type.
    Each element must be greater than or equal to ``0``.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.
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
:obj:`dpnp.binary_repr` : Return the binary representation of the input number
                          as a string.

Notes
-----
At least one of `x1` or `x2` must be an array.

If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
(which becomes the shape of the output).

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

>>> np.binary_repr(5)
'101'
>>> np.left_shift(np.array(5), 2)
array(20)
>>> np.binary_repr(20)
'10100'

"""

left_shift = DPNPBinaryFunc(
    "left_shift",
    ti._bitwise_left_shift_result_type,
    ti._bitwise_left_shift,
    _LEFT_SHIFT_DOCSTRING,
    binary_inplace_fn=ti._bitwise_left_shift_inplace,
)

bitwise_left_shift = left_shift  # bitwise_left_shift is an alias for left_shift


_RIGHT_SHIFT_DOCSTRING = """
Shifts the bits of each element :math:`x1_i` of the input array `x1` to the
right according to the respective element :math:`x2_i` of the input array `x2`.

Note that :obj:`dpnp.bitwise_right_shift` is an alias of :obj:`dpnp.right_shift`.

For full documentation refer to :obj:`numpy.right_shift`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have an integer data type.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to have an integer data type.
    Each element must be greater than or equal to ``0``.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

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
:obj:`dpnp.binary_repr` : Return the binary representation of the input number
                          as a string.

Notes
-----
At least one of `x1` or `x2` must be an array.

If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
(which becomes the shape of the output).

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

>>> np.binary_repr(10)
'1010'
>>> np.right_shift(np.array(10), 1)
array(5)
>>> np.binary_repr(5)
'101'

"""

right_shift = DPNPBinaryFunc(
    "right_shift",
    ti._bitwise_right_shift_result_type,
    ti._bitwise_right_shift,
    _RIGHT_SHIFT_DOCSTRING,
    binary_inplace_fn=ti._bitwise_right_shift_inplace,
)

# bitwise_right_shift is an alias for right_shift
bitwise_right_shift = right_shift
