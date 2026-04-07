# *****************************************************************************
# Copyright (c) 2026, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
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

import dpnp.tensor._tensor_elementwise_impl as ti

from ._elementwise_common import BinaryElementwiseFunc, UnaryElementwiseFunc
from ._type_utils import (
    _acceptance_fn_divide,
    _acceptance_fn_negative,
    _acceptance_fn_reciprocal,
    _acceptance_fn_subtract,
    _resolve_weak_types_all_py_ints,
)

# U01: ==== ABS    (x)
_abs_docstring_ = r"""

Calculates the absolute value for each element `x_i` of input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array. May have any data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array,
        if parameter `out` is ``None``.


        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise absolute values.
        For complex input, the absolute value is its magnitude.
        If `x` has a real-valued data type, the returned array has the
        same data type as `x`. If `x` has a complex floating-point data type,
        the returned array has a real-valued floating-point data type whose
        precision matches the precision of `x`.

"""

abs = UnaryElementwiseFunc("abs", ti._abs_result_type, ti._abs, _abs_docstring_)
del _abs_docstring_

# U02: ==== ACOS   (x)
_acos_docstring = r"""

Computes inverse cosine for each element `x_i` for input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a floating-point data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise inverse cosine, in radians
        and in the closed interval :math:`[0, \pi]`. The data type of the
        returned array is determined by the Type Promotion Rules.

"""

acos = UnaryElementwiseFunc(
    "acos", ti._acos_result_type, ti._acos, _acos_docstring
)
del _acos_docstring

# U03: ===== ACOSH (x)
_acosh_docstring = r"""

Computes inverse hyperbolic cosine for each element `x_i` for input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a floating-point data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise inverse hyperbolic cosine, in
        radians and in the half-closed interval :math:`[0, \infty)`. The data
        type of the returned array is determined by the Type Promotion Rules.

"""

acosh = UnaryElementwiseFunc(
    "acosh", ti._acosh_result_type, ti._acosh, _acosh_docstring
)
del _acosh_docstring

# B01: ===== ADD   (x1, x2)

_add_docstring_ = r"""

Calculates the sum for each element `x1_i` of the input array `x1` with
the respective element `x2_i` of the input array `x2`.

Parameters
----------
    x1 : usm_ndarray
        First input array. May have any data type.
    x2 : usm_ndarray
        Second input array. May have any data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise sums. The data type of the
        returned array is determined by the Type Promotion Rules.

"""

add = BinaryElementwiseFunc(
    "add",
    ti._add_result_type,
    ti._add,
    _add_docstring_,
    binary_inplace_fn=ti._add_inplace,
)
del _add_docstring_

# U04: ===== ASIN  (x)
_asin_docstring = r"""

Computes inverse sine for each element `x_i` for input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a floating-point data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise inverse sine, in radians
        and in the closed interval :math:`[-\pi/2, \pi/2]`. The data type
        of the returned array is determined by the Type Promotion Rules.

"""

asin = UnaryElementwiseFunc(
    "asin", ti._asin_result_type, ti._asin, _asin_docstring
)
del _asin_docstring

# U05: ===== ASINH (x)
_asinh_docstring = r"""

Computes inverse hyperbolic sine for each element `x_i` for input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a floating-point data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise inverse hyperbolic sine, in
        radians. The data type of the returned array is determined by
        the Type Promotion Rules.

"""

asinh = UnaryElementwiseFunc(
    "asinh", ti._asinh_result_type, ti._asinh, _asinh_docstring
)
del _asinh_docstring

# U06: ===== ATAN  (x)
_atan_docstring = r"""

Computes inverse tangent for each element `x_i` for input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a floating-point data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise inverse tangent, in radians
        and in the closed interval :math:`[-\pi/2, \pi/2]`. The data type
        of the returned array is determined by the Type Promotion Rules.

"""

atan = UnaryElementwiseFunc(
    "atan", ti._atan_result_type, ti._atan, _atan_docstring
)
del _atan_docstring

# B02: ===== ATAN2 (x1, x2)
_atan2_docstring_ = r"""
Calculates the inverse tangent of the quotient `x1_i/x2_i` for each element
`x1_i` of the input array `x1` with the respective element `x2_i` of the
input array `x2`. Each element-wise result is expressed in radians.

Parameters
----------
    x1 : usm_ndarray
        First input array, expected to have a real-valued floating-point
        data type.
    x2 : usm_ndarray
        Second input array, also expected to have a real-valued
        floating-point data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the inverse tangent of the quotient `x1`/`x2`.
        The returned array must have a real-valued floating-point data type
        determined by Type Promotion Rules.

"""

atan2 = BinaryElementwiseFunc(
    "atan2", ti._atan2_result_type, ti._atan2, _atan2_docstring_
)
del _atan2_docstring_

# U07: ===== ATANH (x)
_atanh_docstring = r"""

Computes hyperbolic inverse tangent for each element `x_i` for input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a floating-point data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise hyperbolic inverse tangent, in
        radians. The data type of the returned array is determined by
        the Type Promotion Rules.

"""

atanh = UnaryElementwiseFunc(
    "atanh", ti._atanh_result_type, ti._atanh, _atanh_docstring
)
del _atanh_docstring

# B03: ===== BITWISE_AND           (x1, x2)
_bitwise_and_docstring_ = r"""

Computes the bitwise AND of the underlying binary representation of each
element `x1_i` of the input array `x1` with the respective element `x2_i`
of the input array `x2`.

Parameters
----------
    x1 : usm_ndarray
        First input array, expected to have integer or boolean data type.
    x2 : usm_ndarray
        Second input array, also expected to have integer or boolean data
        type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise results. The data type
        of the returned array is determined by the Type Promotion Rules.

"""

bitwise_and = BinaryElementwiseFunc(
    "bitwise_and",
    ti._bitwise_and_result_type,
    ti._bitwise_and,
    _bitwise_and_docstring_,
    binary_inplace_fn=ti._bitwise_and_inplace,
)
del _bitwise_and_docstring_

# B04: ===== BITWISE_LEFT_SHIFT    (x1, x2)
_bitwise_left_shift_docstring_ = r"""

Shifts the bits of each element `x1_i` of the input array x1 to the left by
appending `x2_i` (i.e., the respective element in the input array `x2`) zeros to
the right of `x1_i`.

Parameters
----------
    x1 : usm_ndarray
        First input array, expected to have integer data type.
    x2 : usm_ndarray
        Second input array, also expected to have integer data type.
        Each element must be greater than or equal to 0.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise results. The data type
        of the returned array is determined by the Type Promotion Rules.

"""

bitwise_left_shift = BinaryElementwiseFunc(
    "bitwise_left_shift",
    ti._bitwise_left_shift_result_type,
    ti._bitwise_left_shift,
    _bitwise_left_shift_docstring_,
    binary_inplace_fn=ti._bitwise_left_shift_inplace,
)
del _bitwise_left_shift_docstring_

# U08: ===== BITWISE_INVERT        (x)
_bitwise_invert_docstring = r"""

Inverts (flips) each bit for each element `x_i` of the input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have integer or boolean data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise results.
        The data type of the returned array is same as the data type of the
        input array.

"""

bitwise_invert = UnaryElementwiseFunc(
    "bitwise_invert",
    ti._bitwise_invert_result_type,
    ti._bitwise_invert,
    _bitwise_invert_docstring,
)
del _bitwise_invert_docstring

# B05: ===== BITWISE_OR            (x1, x2)
_bitwise_or_docstring_ = r"""

Computes the bitwise OR of the underlying binary representation of each
element `x1_i` of the input array `x1` with the respective element `x2_i`
of the input array `x2`.

Parameters
----------
    x1 : usm_ndarray
        First input array, expected to have integer or boolean data type.
    x2 : usm_ndarray
        Second input array, also expected to have integer or boolean data
        type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise results. The data type
        of the returned array is determined by the Type Promotion Rules.

"""

bitwise_or = BinaryElementwiseFunc(
    "bitwise_or",
    ti._bitwise_or_result_type,
    ti._bitwise_or,
    _bitwise_or_docstring_,
    binary_inplace_fn=ti._bitwise_or_inplace,
)
del _bitwise_or_docstring_

# B06: ===== BITWISE_RIGHT_SHIFT   (x1, x2)
_bitwise_right_shift_docstring_ = r"""

Shifts the bits of each element `x1_i` of the input array `x1` to the right
according to the respective element `x2_i` of the input array `x2`.

Parameters
----------
    x1 : usm_ndarray
        First input array, expected to have integer data type.
    x2 : usm_ndarray
        Second input array, also expected to have integer data type.
        Each element must be greater than or equal to 0.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise results. The data type
        of the returned array is determined by the Type Promotion Rules.

"""

bitwise_right_shift = BinaryElementwiseFunc(
    "bitwise_right_shift",
    ti._bitwise_right_shift_result_type,
    ti._bitwise_right_shift,
    _bitwise_right_shift_docstring_,
    binary_inplace_fn=ti._bitwise_right_shift_inplace,
)
del _bitwise_right_shift_docstring_


# B07: ===== BITWISE_XOR           (x1, x2)
_bitwise_xor_docstring_ = r"""

Computes the bitwise XOR of the underlying binary representation of each
element `x1_i` of the input array `x1` with the respective element `x2_i`
of the input array `x2`.

Parameters
----------
    x1 : usm_ndarray
        First input array, expected to have integer or boolean data type.
    x2 : usm_ndarray
        Second input array, also expected to have integer or boolean data
        type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise results. The data type
        of the returned array is determined by the Type Promotion Rules.

"""

bitwise_xor = BinaryElementwiseFunc(
    "bitwise_xor",
    ti._bitwise_xor_result_type,
    ti._bitwise_xor,
    _bitwise_xor_docstring_,
    binary_inplace_fn=ti._bitwise_xor_inplace,
)
del _bitwise_xor_docstring_

# U09: ==== CEIL          (x)
_ceil_docstring = r"""

Returns the ceiling for each element `x_i` for input array `x`.

The ceil of `x_i` is the smallest integer `n`, such that `n >= x_i`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a boolean or real-valued data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise ceiling.

"""

ceil = UnaryElementwiseFunc(
    "ceil", ti._ceil_result_type, ti._ceil, _ceil_docstring
)
del _ceil_docstring

# U10: ==== CONJ          (x)
_conj_docstring = r"""

Computes conjugate of each element `x_i` for input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array. May have any data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise conjugate values.

"""

conj = UnaryElementwiseFunc(
    "conj", ti._conj_result_type, ti._conj, _conj_docstring
)
del _conj_docstring

# U11: ==== COS           (x)
_cos_docstring = r"""

Computes cosine for each element `x_i` for input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a floating-point data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise cosine. The data type
        of the returned array is determined by the Type Promotion Rules.

"""

cos = UnaryElementwiseFunc("cos", ti._cos_result_type, ti._cos, _cos_docstring)
del _cos_docstring

# U12: ==== COSH          (x)
_cosh_docstring = r"""

Computes hyperbolic cosine for each element `x_i` for input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a floating-point data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise hyperbolic cosine. The data type
        of the returned array is determined by the Type Promotion Rules.

"""

cosh = UnaryElementwiseFunc(
    "cosh", ti._cosh_result_type, ti._cosh, _cosh_docstring
)
del _cosh_docstring

# B08: ==== DIVIDE        (x1, x2)
_divide_docstring_ = r"""

Calculates the ratio for each element `x1_i` of the input array `x1` with
the respective element `x2_i` of the input array `x2`.

Parameters
----------
    x1 : usm_ndarray
        First input array, expected to have a floating-point data type.
    x2 : usm_ndarray
        Second input array, also expected to have a floating-point data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the result of element-wise division. The data type
        of the returned array is determined by the Type Promotion Rules.

"""

divide = BinaryElementwiseFunc(
    "divide",
    ti._divide_result_type,
    ti._divide,
    _divide_docstring_,
    binary_inplace_fn=ti._divide_inplace,
    acceptance_fn=_acceptance_fn_divide,
    weak_type_resolver=_resolve_weak_types_all_py_ints,
)
del _divide_docstring_

# B09: ==== EQUAL         (x1, x2)
_equal_docstring_ = r"""

Calculates equality test results for each element `x1_i` of the input array `x1`
with the respective element `x2_i` of the input array `x2`.

Parameters
----------
    x1 : usm_ndarray
        First input array. May have any data type.
    x2 : usm_ndarray
        Second input array. May have any data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the result of element-wise equality comparison.
        The returned array has a data type of `bool`.

"""

equal = BinaryElementwiseFunc(
    "equal",
    ti._equal_result_type,
    ti._equal,
    _equal_docstring_,
    weak_type_resolver=_resolve_weak_types_all_py_ints,
)
del _equal_docstring_

# U13: ==== EXP           (x)
_exp_docstring = r"""

Computes the exponential for each element `x_i` of input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a floating-point data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise exponential of `x`.
        The data type of the returned array is determined by
        the Type Promotion Rules.

"""

exp = UnaryElementwiseFunc("exp", ti._exp_result_type, ti._exp, _exp_docstring)
del _exp_docstring

# B10: ==== FLOOR_DIVIDE  (x1, x2)
_floor_divide_docstring_ = r"""

Calculates the ratio for each element `x1_i` of the input array `x1` with
the respective element `x2_i` of the input array `x2` to the greatest
integer-value number that is not greater than the division result.

Parameters
----------
    x1 : usm_ndarray
        First input array, expected to have a real-valued data type.
    x2 : usm_ndarray
        Second input array, also expected to have a real-valued data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the result of element-wise floor of division.
        The data type of the returned array is determined by the Type
        Promotion Rules.

"""

floor_divide = BinaryElementwiseFunc(
    "floor_divide",
    ti._floor_divide_result_type,
    ti._floor_divide,
    _floor_divide_docstring_,
    binary_inplace_fn=ti._floor_divide_inplace,
)
del _floor_divide_docstring_

# B11: ==== GREATER       (x1, x2)
_greater_docstring_ = r"""

Computes the greater-than test results for each element `x1_i` of
the input array `x1` with the respective element `x2_i` of the input array `x2`.

Parameters
----------
    x1 : usm_ndarray
        First input array. May have any data type.
    x2 : usm_ndarray
        Second input array. May have any data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the result of element-wise greater-than comparison.
        The returned array has a data type of `bool`.

"""

greater = BinaryElementwiseFunc(
    "greater",
    ti._greater_result_type,
    ti._greater,
    _greater_docstring_,
    weak_type_resolver=_resolve_weak_types_all_py_ints,
)
del _greater_docstring_

# B12: ==== GREATER_EQUAL (x1, x2)
_greater_equal_docstring_ = r"""

Computes the greater-than or equal-to test results for each element `x1_i` of
the input array `x1` with the respective element `x2_i` of the input array `x2`.

Parameters
----------
    x1 : usm_ndarray
        First input array. May have any data type.
    x2 : usm_ndarray
        Second input array. May have any data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the result of element-wise greater-than or equal-to
        comparison.
        The returned array has a data type of `bool`.

"""

greater_equal = BinaryElementwiseFunc(
    "greater_equal",
    ti._greater_equal_result_type,
    ti._greater_equal,
    _greater_equal_docstring_,
    weak_type_resolver=_resolve_weak_types_all_py_ints,
)
del _greater_equal_docstring_

# U14: ==== EXPM1         (x)
_expm1_docstring = r"""
expm1(x, /, \*, out=None, order='K')

Computes the exponential minus 1 for each element `x_i` of input array `x`.

This function calculates `exp(x) - 1.0` more accurately for small values of `x`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a floating-point data type.
    out : usm_ndarray
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise `exp(x) - 1` results.
        The data type of the returned array is determined by the Type
        Promotion Rules.

"""

expm1 = UnaryElementwiseFunc(
    "expm1", ti._expm1_result_type, ti._expm1, _expm1_docstring
)
del _expm1_docstring

# U15: ==== FLOOR         (x)
_floor_docstring = r"""

Returns the floor for each element `x_i` for input array `x`.

The floor of `x_i` is the largest integer `n`, such that `n <= x_i`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a boolean or real-valued data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise floor.

"""

floor = UnaryElementwiseFunc(
    "floor", ti._floor_result_type, ti._floor, _floor_docstring
)
del _floor_docstring

# U16: ==== IMAG        (x)
_imag_docstring = r"""

Computes imaginary part of each element `x_i` for input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array. May have any data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise imaginary component of input.
        If the input is a real-valued data type, the returned array has
        the same data type. If the input is a complex floating-point
        data type, the returned array has a floating-point data type
        with the same floating-point precision as complex input.

"""

imag = UnaryElementwiseFunc(
    "imag", ti._imag_result_type, ti._imag, _imag_docstring
)
del _imag_docstring

# U17: ==== ISFINITE    (x)
_isfinite_docstring_ = r"""

Test if each element of input array is a finite number.

Parameters
----------
    x : usm_ndarray
        Input array. May have any data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array which is True where `x` is not positive infinity,
        negative infinity, or NaN, False otherwise.
        The data type of the returned array is `bool`.

"""

isfinite = UnaryElementwiseFunc(
    "isfinite", ti._isfinite_result_type, ti._isfinite, _isfinite_docstring_
)
del _isfinite_docstring_

# U18: ==== ISINF       (x)
_isinf_docstring_ = r"""

Test if each element of input array is an infinity.

Parameters
----------
    x : usm_ndarray
        Input array. May have any data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array which is True where `x` is positive or negative infinity,
        False otherwise. The data type of the returned array is `bool`.

"""

isinf = UnaryElementwiseFunc(
    "isinf", ti._isinf_result_type, ti._isinf, _isinf_docstring_
)
del _isinf_docstring_

# U19: ==== ISNAN       (x)
_isnan_docstring_ = r"""

Test if each element of an input array is a NaN.

Parameters
----------
    x : usm_ndarray
        Input array. May have any data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array which is True where x is NaN, False otherwise.
        The data type of the returned array is `bool`.

"""

isnan = UnaryElementwiseFunc(
    "isnan", ti._isnan_result_type, ti._isnan, _isnan_docstring_
)
del _isnan_docstring_

# B13: ==== LESS        (x1, x2)
_less_docstring_ = r"""

Computes the less-than test results for each element `x1_i` of
the input array `x1` with the respective element `x2_i` of the input array `x2`.

Parameters
----------
    x1 : usm_ndarray
        First input array. May have any data type.
    x2 : usm_ndarray
        Second input array. May have any data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the result of element-wise less-than comparison.
        The returned array has a data type of `bool`.

"""

less = BinaryElementwiseFunc(
    "less",
    ti._less_result_type,
    ti._less,
    _less_docstring_,
    weak_type_resolver=_resolve_weak_types_all_py_ints,
)
del _less_docstring_


# B14: ==== LESS_EQUAL  (x1, x2)
_less_equal_docstring_ = r"""

Computes the less-than or equal-to test results for each element `x1_i` of
the input array `x1` with the respective element `x2_i` of the input array `x2`.

Parameters
----------
    x1 : usm_ndarray
        First input array. May have any data type.
    x2 : usm_ndarray
        Second input array. May have any data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the result of element-wise less-than or equal-to
        comparison. The returned array has a data type of `bool`.

"""

less_equal = BinaryElementwiseFunc(
    "less_equal",
    ti._less_equal_result_type,
    ti._less_equal,
    _less_equal_docstring_,
    weak_type_resolver=_resolve_weak_types_all_py_ints,
)
del _less_equal_docstring_

# U20: ==== LOG         (x)
_log_docstring = r"""

Computes the natural logarithm for each element `x_i` of input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a floating-point data type.
    out : usm_ndarray
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise natural logarithm values.
        The data type of the returned array is determined by the Type
        Promotion Rules.

"""

log = UnaryElementwiseFunc("log", ti._log_result_type, ti._log, _log_docstring)
del _log_docstring

# U21: ==== LOG1P       (x)
_log1p_docstring = r"""
log1p(x, /, \*, out=None, order='K')

Computes the natural logarithm of (1 + `x`) for each element `x_i` of input
array `x`.

This function calculates `log(1 + x)` more accurately for small values of `x`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a floating-point data type.
    out : usm_ndarray
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise `log(1 + x)` results. The data type
        of the returned array is determined by the Type Promotion Rules.

"""

log1p = UnaryElementwiseFunc(
    "log1p", ti._log1p_result_type, ti._log1p, _log1p_docstring
)
del _log1p_docstring

# U22: ==== LOG2        (x)
_log2_docstring_ = r"""
log2(x, /, \*, out=None, order='K')

Computes the base-2 logarithm for each element `x_i` of input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a floating-point data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise base-2 logarithm of `x`.
        The data type of the returned array is determined by the
        Type Promotion Rules.

"""

log2 = UnaryElementwiseFunc(
    "log2", ti._log2_result_type, ti._log2, _log2_docstring_
)
del _log2_docstring_

# U23: ==== LOG10       (x)
_log10_docstring_ = r"""
log10(x, /, \*, out=None, order='K')

Computes the base-10 logarithm for each element `x_i` of input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a floating-point data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise base-10 logarithm of `x`.
        The data type of the returned array is determined by the
        Type Promotion Rules.

"""

log10 = UnaryElementwiseFunc(
    "log10", ti._log10_result_type, ti._log10, _log10_docstring_
)
del _log10_docstring_

# B15: ==== LOGADDEXP   (x1, x2)
_logaddexp_docstring_ = r"""

Calculates the natural logarithm of the sum of exponentials for each element
`x1_i` of the input array `x1` with the respective element `x2_i` of the input
array `x2`.

This function calculates `log(exp(x1) + exp(x2))` more accurately for small
values of `x`.

Parameters
----------
    x1 : usm_ndarray
        First input array, expected to have a real-valued floating-point data
        type.
    x2 : usm_ndarray
        Second input array, also expected to have a real-valued floating-point
        data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise results. The data type
        of the returned array is determined by the Type Promotion Rules.

"""

logaddexp = BinaryElementwiseFunc(
    "logaddexp", ti._logaddexp_result_type, ti._logaddexp, _logaddexp_docstring_
)
del _logaddexp_docstring_

# B16: ==== LOGICAL_AND (x1, x2)
_logical_and_docstring_ = r"""

Computes the logical AND for each element `x1_i` of the input array `x1` with
the respective element `x2_i` of the input array `x2`.

Parameters
----------
    x1 : usm_ndarray
        First input array. May have any data type.
    x2 : usm_ndarray
        Second input array. May have any data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise logical AND results.

"""

logical_and = BinaryElementwiseFunc(
    "logical_and",
    ti._logical_and_result_type,
    ti._logical_and,
    _logical_and_docstring_,
)
del _logical_and_docstring_

# B17: ==== LOGICAL_OR  (x1, x2)
_logical_or_docstring_ = r"""

Computes the logical OR for each element `x1_i` of the input array `x1`
with the respective element `x2_i` of the input array `x2`.

Parameters
----------
    x1 : usm_ndarray
        First input array. May have any data type.
    x2 : usm_ndarray
        Second input array. May have any data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise logical OR results.

"""

logical_or = BinaryElementwiseFunc(
    "logical_or",
    ti._logical_or_result_type,
    ti._logical_or,
    _logical_or_docstring_,
)
del _logical_or_docstring_

# B18: ==== LOGICAL_XOR (x1, x2)
_logical_xor_docstring_ = r"""

Computes the logical XOR for each element `x1_i` of the input array `x1`
with the respective element `x2_i` of the input array `x2`.

Parameters
----------
    x1 : usm_ndarray
        First input array. May have any data type.
    x2 : usm_ndarray
        Second input array. May have any data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise logical XOR results.

"""

logical_xor = BinaryElementwiseFunc(
    "logical_xor",
    ti._logical_xor_result_type,
    ti._logical_xor,
    _logical_xor_docstring_,
)
del _logical_xor_docstring_

# U24: ==== LOGICAL_NOT (x)
_logical_not_docstring = r"""

Computes the logical NOT for each element `x_i` of input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array. May have any data type.
    out : usm_ndarray
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise logical NOT results.

"""

logical_not = UnaryElementwiseFunc(
    "logical_not",
    ti._logical_not_result_type,
    ti._logical_not,
    _logical_not_docstring,
)
del _logical_not_docstring

# B26: ==== MAXIMUM    (x1, x2)
_maximum_docstring_ = r"""

Compares two input arrays `x1` and `x2` and returns a new array containing the
element-wise maxima.

Parameters
----------
    x1 : usm_ndarray
        First input array. May have any data type.
    x2 : usm_ndarray
        Second input array. May have any data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise maxima. The data type of
        the returned array is determined by the Type Promotion Rules.

"""

maximum = BinaryElementwiseFunc(
    "maximum",
    ti._maximum_result_type,
    ti._maximum,
    _maximum_docstring_,
)
del _maximum_docstring_

# B27: ==== MINIMUM    (x1, x2)
_minimum_docstring_ = r"""

Compares two input arrays `x1` and `x2` and returns a new array containing the
element-wise minima.

Parameters
----------
    x1 : usm_ndarray
        First input array. May have any data type.
    x2 : usm_ndarray
        Second input array. May have any data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise minima. The data type of
        the returned array is determined by the Type Promotion Rules.

"""

minimum = BinaryElementwiseFunc(
    "minimum",
    ti._minimum_result_type,
    ti._minimum,
    _minimum_docstring_,
)
del _minimum_docstring_

# B19: ==== MULTIPLY    (x1, x2)
_multiply_docstring_ = r"""

Calculates the product for each element `x1_i` of the input array `x1` with the
respective element `x2_i` of the input array `x2`.

Parameters
----------
    x1 : usm_ndarray
        First input array. May have any data type.
    x2 : usm_ndarray
        Second input array. May have any data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise products. The data type of
        the returned array is determined by the Type Promotion Rules.

"""

multiply = BinaryElementwiseFunc(
    "multiply",
    ti._multiply_result_type,
    ti._multiply,
    _multiply_docstring_,
    binary_inplace_fn=ti._multiply_inplace,
)
del _multiply_docstring_

# U25: ==== NEGATIVE    (x)
_negative_docstring_ = r"""

Computes the numerical negative for each element `x_i` of input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a numeric data type.
    out : usm_ndarray
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the negative of `x`.

"""

negative = UnaryElementwiseFunc(
    "negative",
    ti._negative_result_type,
    ti._negative,
    _negative_docstring_,
    acceptance_fn=_acceptance_fn_negative,
)
del _negative_docstring_

# B28: ==== NEXTAFTER    (x1, x2)
_nextafter_docstring_ = r"""

Calculates the next floating-point value after element `x1_i` of the input
array `x1` toward the respective element `x2_i` of the input array `x2`.

Parameters
----------
    x1 : usm_ndarray
        First input array, expected to have a real-valued floating-point data
        type.
    x2 : usm_ndarray
        Second input array, expected to have a real-valued floating-point data
        type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise next representable values of `x1`
        in the direction of `x2`. The data type of the returned array is
        determined by the Type Promotion Rules.

"""

nextafter = BinaryElementwiseFunc(
    "nextafter",
    ti._nextafter_result_type,
    ti._nextafter,
    _nextafter_docstring_,
)
del _nextafter_docstring_

# B20: ==== NOT_EQUAL   (x1, x2)
_not_equal_docstring_ = r"""

Calculates inequality test results for each element `x1_i` of the
input array `x1` with the respective element `x2_i` of the input array `x2`.

Parameters
----------
    x1 : usm_ndarray
        First input array.
    x2 : usm_ndarray
        Second input array.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the result of element-wise inequality comparison.
        The returned array has a data type of `bool`.

"""

not_equal = BinaryElementwiseFunc(
    "not_equal",
    ti._not_equal_result_type,
    ti._not_equal,
    _not_equal_docstring_,
    weak_type_resolver=_resolve_weak_types_all_py_ints,
)
del _not_equal_docstring_

# U26: ==== POSITIVE    (x)
_positive_docstring_ = r"""

Computes the numerical positive for each element `x_i` of input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a numeric data type.
    out : usm_ndarray
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the positive of `x`.

"""

positive = UnaryElementwiseFunc(
    "positive", ti._positive_result_type, ti._positive, _positive_docstring_
)
del _positive_docstring_

# B21: ==== POW         (x1, x2)
_pow_docstring_ = r"""

Calculates `x1_i` raised to `x2_i` for each element `x1_i` of the input array
`x1` with the respective element `x2_i` of the input array `x2`.

Parameters
----------
    x1 : usm_ndarray
        First input array, expected to have a numeric data type.
    x2 : usm_ndarray
        Second input array, also expected to have a numeric data type.
    out : usm_ndarray
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the bases in `x1` raised to the exponents in `x2`
        element-wise. The data type of the returned array is determined by the
        Type Promotion Rules.

"""

pow = BinaryElementwiseFunc(
    "pow",
    ti._pow_result_type,
    ti._pow,
    _pow_docstring_,
    binary_inplace_fn=ti._pow_inplace,
)
del _pow_docstring_

# U27: ==== REAL        (x)
_real_docstring = r"""

Computes real part of each element `x_i` for input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array. May have any data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise real component of input.
        If the input is a real-valued data type, the returned array has
        the same data type. If the input is a complex floating-point
        data type, the returned array has a floating-point data type
        with the same floating-point precision as complex input.

"""

real = UnaryElementwiseFunc(
    "real", ti._real_result_type, ti._real, _real_docstring
)
del _real_docstring

# B22: ==== REMAINDER   (x1, x2)
_remainder_docstring_ = r"""

Calculates the remainder of division for each element `x1_i` of the input array
`x1` with the respective element `x2_i` of the input array `x2`.

This function is equivalent to the Python modulus operator.

Parameters
----------
    x1 : usm_ndarray
        First input array, expected to have a real-valued data type.
    x2 : usm_ndarray
        Second input array, also expected to have a real-valued data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise remainders. Each remainder has the
        same sign as respective element `x2_i`. The data type of the returned
        array is determined by the Type Promotion Rules.

"""

remainder = BinaryElementwiseFunc(
    "remainder",
    ti._remainder_result_type,
    ti._remainder,
    _remainder_docstring_,
    binary_inplace_fn=ti._remainder_inplace,
)
del _remainder_docstring_

# U28: ==== ROUND       (x)
_round_docstring = r"""

Rounds each element `x_i` of the input array `x` to
the nearest integer-valued number.

When two integers are equally close to `x_i`, the result is the nearest even
integer to `x_i`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a numeric data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise rounded values.

"""

round = UnaryElementwiseFunc(
    "round", ti._round_result_type, ti._round, _round_docstring
)
del _round_docstring

# U29: ==== SIGN        (x)
_sign_docstring = r"""

Computes an indication of the sign of each element `x_i` of input array `x`
using the signum function.

The signum function returns `-1` if `x_i` is less than `0`,
`0` if `x_i` is equal to `0`, and `1` if `x_i` is greater than `0`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a numeric data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise result of the signum function. The
        data type of the returned array is determined by the Type Promotion
        Rules.

"""

sign = UnaryElementwiseFunc(
    "sign", ti._sign_result_type, ti._sign, _sign_docstring
)
del _sign_docstring

# U30: ==== SIN         (x)
_sin_docstring = r"""

Computes sine for each element `x_i` of input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a real-valued floating-point data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise sine. The data type of the
        returned array is determined by the Type Promotion Rules.

"""

sin = UnaryElementwiseFunc("sin", ti._sin_result_type, ti._sin, _sin_docstring)
del _sin_docstring

# U31: ==== SINH        (x)
_sinh_docstring = r"""

Computes hyperbolic sine for each element `x_i` for input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a floating-point data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise hyperbolic sine. The data type
        of the returned array is determined by the Type Promotion Rules.

"""

sinh = UnaryElementwiseFunc(
    "sinh", ti._sinh_result_type, ti._sinh, _sinh_docstring
)
del _sinh_docstring

# U32: ==== SQUARE      (x)
_square_docstring_ = r"""

Squares each element `x_i` of input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array. May have any data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise squares of `x`. The data type of
        the returned array is determined by the Type Promotion Rules.

"""

square = UnaryElementwiseFunc(
    "square", ti._square_result_type, ti._square, _square_docstring_
)
del _square_docstring_

# U33: ==== SQRT        (x)
_sqrt_docstring_ = r"""

Computes the positive square-root for each element `x_i` of input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a floating-point data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise positive square-roots of `x`. The
        data type of the returned array is determined by the Type Promotion
        Rules.

"""

sqrt = UnaryElementwiseFunc(
    "sqrt", ti._sqrt_result_type, ti._sqrt, _sqrt_docstring_
)
del _sqrt_docstring_

# B23: ==== SUBTRACT    (x1, x2)
_subtract_docstring_ = r"""

Calculates the difference between each element `x1_i` of the input
array `x1` and the respective element `x2_i` of the input array `x2`.

Parameters
----------
    x1 : usm_ndarray
        First input array, expected to have a numeric data type.
    x2 : usm_ndarray
        Second input array, also expected to have a numeric data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise differences. The data type
        of the returned array is determined by the Type Promotion Rules.

"""
subtract = BinaryElementwiseFunc(
    "subtract",
    ti._subtract_result_type,
    ti._subtract,
    _subtract_docstring_,
    binary_inplace_fn=ti._subtract_inplace,
    acceptance_fn=_acceptance_fn_subtract,
)
del _subtract_docstring_

# U34: ==== TAN         (x)
_tan_docstring = r"""

Computes tangent for each element `x_i` for input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a floating-point data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise tangent. The data type
        of the returned array is determined by the Type Promotion Rules.

"""

tan = UnaryElementwiseFunc("tan", ti._tan_result_type, ti._tan, _tan_docstring)
del _tan_docstring

# U35: ==== TANH        (x)
_tanh_docstring = r"""

Computes hyperbolic tangent for each element `x_i` for input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a floating-point data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise hyperbolic tangent. The data type
        of the returned array is determined by the Type Promotion Rules.

"""

tanh = UnaryElementwiseFunc(
    "tanh", ti._tanh_result_type, ti._tanh, _tanh_docstring
)
del _tanh_docstring

# U36: ==== TRUNC       (x)
_trunc_docstring = r"""

Returns the truncated value for each element `x_i` for input array `x`.

The truncated value of the scalar `x` is the nearest integer i which is
closer to zero than `x` is. In short, the fractional part of the
signed number `x` is discarded.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a boolean or real-valued data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the result of element-wise division. The data type
        of the returned array is determined by the Type Promotion Rules.

"""
trunc = UnaryElementwiseFunc(
    "trunc", ti._trunc_result_type, ti._trunc, _trunc_docstring
)
del _trunc_docstring

# B24: ==== HYPOT        (x1, x2)
_hypot_docstring_ = r"""

Computes the square root of the sum of squares for each element `x1_i` of the
input array `x1` with the respective element `x2_i` of the input array `x2`.

Parameters
----------
    x1 : usm_ndarray
        First input array, expected to have a real-valued floating-point data
        type.
    x2 : usm_ndarray
        Second input array, also expected to have a real-valued floating-point
        data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise hypotenuse. The data type
        of the returned array is determined by the Type Promotion Rules.

"""

hypot = BinaryElementwiseFunc(
    "hypot", ti._hypot_result_type, ti._hypot, _hypot_docstring_
)
del _hypot_docstring_

# U37: ==== CBRT        (x)
_cbrt_docstring_ = r"""

Computes the cube-root for each element `x_i` for input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a real-valued floating-point data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise cube-root.
        The data type of the returned array is determined by
        the Type Promotion Rules.

"""

cbrt = UnaryElementwiseFunc(
    "cbrt", ti._cbrt_result_type, ti._cbrt, _cbrt_docstring_
)
del _cbrt_docstring_

# U38: ==== EXP2        (x)
_exp2_docstring_ = r"""
exp2(x, /, \*, out=None, order='K')

Computes the base-2 exponential for each element `x_i` for input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a floating-point data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise base-2 exponentials.
        The data type of the returned array is determined by
        the Type Promotion Rules.

"""

exp2 = UnaryElementwiseFunc(
    "exp2", ti._exp2_result_type, ti._exp2, _exp2_docstring_
)
del _exp2_docstring_

# B25: ==== COPYSIGN    (x1, x2)
_copysign_docstring_ = r"""

Composes a floating-point value with the magnitude of `x1_i` and the sign of
`x2_i` for each element of input arrays `x1` and `x2`.

Parameters
----------
    x1 : usm_ndarray
        First input array, expected to have a real-valued floating-point data
        type.
    x2 : usm_ndarray
        Second input array, also expected to have a real-valued floating-point
        data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise results. The data type
        of the returned array is determined by the Type Promotion Rules.

"""
copysign = BinaryElementwiseFunc(
    "copysign",
    ti._copysign_result_type,
    ti._copysign,
    _copysign_docstring_,
)
del _copysign_docstring_

# U39: ==== RSQRT        (x)
_rsqrt_docstring_ = r"""

Computes the reciprocal square-root for each element `x_i` for input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a real-valued floating-point data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise reciprocal square-root.
        The returned array has a floating-point data type determined by
        the Type Promotion Rules.

"""

rsqrt = UnaryElementwiseFunc(
    "rsqrt", ti._rsqrt_result_type, ti._rsqrt, _rsqrt_docstring_
)
del _rsqrt_docstring_

# U40: ==== PROJ        (x)
_proj_docstring = r"""

Computes projection of each element `x_i` for input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a complex data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise projection.

"""

proj = UnaryElementwiseFunc(
    "proj", ti._proj_result_type, ti._proj, _proj_docstring
)
del _proj_docstring

# U41: ==== SIGNBIT        (x)
_signbit_docstring = r"""

Computes an indication of whether the sign bit of each element `x_i` of
input array `x` is set.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a real-valued floating-point data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array must have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise signbit results. The returned array
        must have a data type of `bool`.

"""

signbit = UnaryElementwiseFunc(
    "signbit", ti._signbit_result_type, ti._signbit, _signbit_docstring
)
del _signbit_docstring

# U42: ==== RECIPROCAL        (x)
_reciprocal_docstring = r"""

Computes the reciprocal of each element `x_i` for input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a floating-point data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise reciprocals.
        The returned array has a floating-point data type determined
        by the Type Promotion Rules.

"""

reciprocal = UnaryElementwiseFunc(
    "reciprocal",
    ti._reciprocal_result_type,
    ti._reciprocal,
    _reciprocal_docstring,
    acceptance_fn=_acceptance_fn_reciprocal,
)
del _reciprocal_docstring

# U43: ==== ANGLE        (x)
_angle_docstring = r"""

Computes the phase angle (also called the argument) of each element `x_i` for
input array `x`.

Parameters
----------
    x : usm_ndarray
        Input array, expected to have a complex floating-point data type.
    out : {None, usm_ndarray}, optional
        Output array to populate.
        Array have the correct shape and the expected data type.

        Default: ``None``.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the new output array, if parameter
        `out` is ``None``.

        Default: ``"K"``.

Returns
-------
    out : usm_ndarray
        An array containing the element-wise phase angles.
        The returned array has a floating-point data type determined
        by the Type Promotion Rules.

"""

angle = UnaryElementwiseFunc(
    "angle",
    ti._angle_result_type,
    ti._angle,
    _angle_docstring,
)
del _angle_docstring

del ti
