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

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor._tensor_elementwise_impl as ti

from ._elementwise_common import UnaryElementwiseFunc

# U01: ==== ABS    (x)
_abs_docstring_ = r"""
abs(x, /, \*, out=None, order='K')

Calculates the absolute value for each element `x_i` of input array `x`.

Args:
    x (usm_ndarray):
        Input array. May have any data type.
    out (Union[usm_ndarray, None], optional):
        Output array to populate.
        Array must have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the new output array,
        if parameter `out` is ``None``.
        Default: `"K"`.

Returns:
    usm_ndarray:
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
acos(x, /, \*, out=None, order='K')

Computes inverse cosine for each element `x_i` for input array `x`.

Args:
    x (usm_ndarray):
        Input array, expected to have a floating-point data type.
    out (Union[usm_ndarray, None], optional):
        Output array to populate.
        Array must have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the new output array, if parameter
        `out` is ``None``.
        Default: "K".

Returns:
    usm_ndarray:
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
acosh(x, /, \*, out=None, order='K')

Computes inverse hyperbolic cosine for each element `x_i` for input array `x`.

Args:
    x (usm_ndarray):
        Input array, expected to have a floating-point data type.
    out (Union[usm_ndarray, None], optional):
        Output array to populate.
        Array must have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the new output array, if parameter
        `out` is ``None``.
        Default: "K".

Returns:
    usm_ndarray:
        An array containing the element-wise inverse hyperbolic cosine, in
        radians and in the half-closed interval :math:`[0, \infty)`. The data
        type of the returned array is determined by the Type Promotion Rules.
"""

acosh = UnaryElementwiseFunc(
    "acosh", ti._acosh_result_type, ti._acosh, _acosh_docstring
)
del _acosh_docstring

# U04: ===== ASIN  (x)
_asin_docstring = r"""
asin(x, /, \*, out=None, order='K')

Computes inverse sine for each element `x_i` for input array `x`.

Args:
    x (usm_ndarray):
        Input array, expected to have a floating-point data type.
    out (Union[usm_ndarray, None], optional):
        Output array to populate.
        Array must have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the new output array, if parameter
        `out` is ``None``.
        Default: "K".

Returns:
    usm_ndarray:
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
asinh(x, /, \*, out=None, order='K')

Computes inverse hyperbolic sine for each element `x_i` for input array `x`.

Args:
    x (usm_ndarray):
        Input array, expected to have a floating-point data type.
    out (Union[usm_ndarray, None], optional):
        Output array to populate.
        Array must have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the new output array, if parameter
        `out` is ``None``.
        Default: "K".

Returns:
    usm_ndarray:
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
atan(x, /, \*, out=None, order='K')

Computes inverse tangent for each element `x_i` for input array `x`.

Args:
    x (usm_ndarray):
        Input array, expected to have a floating-point data type.
    out (Union[usm_ndarray, None], optional):
        Output array to populate.
        Array must have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the new output array, if parameter
        `out` is ``None``.
        Default: "K".

Returns:
    usm_ndarray:
        An array containing the element-wise inverse tangent, in radians
        and in the closed interval :math:`[-\pi/2, \pi/2]`. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

atan = UnaryElementwiseFunc(
    "atan", ti._atan_result_type, ti._atan, _atan_docstring
)
del _atan_docstring

# U07: ===== ATANH (x)
_atanh_docstring = r"""
atanh(x, /, \*, out=None, order='K')

Computes hyperbolic inverse tangent for each element `x_i` for input array `x`.

Args:
    x (usm_ndarray):
        Input array, expected to have a floating-point data type.
    out (Union[usm_ndarray, None], optional):
        Output array to populate.
        Array must have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the new output array, if parameter
        `out` is ``None``.
        Default: "K".

Returns:
    usm_ndarray:
        An array containing the element-wise hyperbolic inverse tangent, in
        radians. The data type of the returned array is determined by
        the Type Promotion Rules.
"""

atanh = UnaryElementwiseFunc(
    "atanh", ti._atanh_result_type, ti._atanh, _atanh_docstring
)
del _atanh_docstring

# U08: ===== BITWISE_INVERT        (x)
_bitwise_invert_docstring = r"""
bitwise_invert(x, /, \*, out=None, order='K')

Inverts (flips) each bit for each element `x_i` of the input array `x`.

Args:
    x (usm_ndarray):
        Input array, expected to have integer or boolean data type.
    out (Union[usm_ndarray, None], optional):
        Output array to populate.
        Array must have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the new output array, if parameter
        `out` is ``None``.
        Default: "K".

Returns:
    usm_ndarray:
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

# U09: ==== CEIL          (x)
_ceil_docstring = r"""
ceil(x, /, \*, out=None, order='K')

Returns the ceiling for each element `x_i` for input array `x`.

The ceil of `x_i` is the smallest integer `n`, such that `n >= x_i`.

Args:
    x (usm_ndarray):
        Input array, expected to have a boolean or real-valued data type.
    out (Union[usm_ndarray, None], optional):
        Output array to populate.
        Array must have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the new output array, if parameter
        `out` is ``None``.
        Default: "K".

Returns:
    usm_ndarray:
        An array containing the element-wise ceiling.
"""

ceil = UnaryElementwiseFunc(
    "ceil", ti._ceil_result_type, ti._ceil, _ceil_docstring
)
del _ceil_docstring

# U10: ==== CONJ          (x)
_conj_docstring = r"""
conj(x, /, \*, out=None, order='K')

Computes conjugate of each element `x_i` for input array `x`.

Args:
    x (usm_ndarray):
        Input array. May have any data type.
    out (Union[usm_ndarray, None], optional):
        Output array to populate.
        Array must have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the new output array, if parameter
        `out` is ``None``.
        Default: "K".

Returns:
    usm_ndarray:
        An array containing the element-wise conjugate values.
"""

conj = UnaryElementwiseFunc(
    "conj", ti._conj_result_type, ti._conj, _conj_docstring
)
del _conj_docstring

# U11: ==== COS           (x)
_cos_docstring = r"""
cos(x, /, \*, out=None, order='K')

Computes cosine for each element `x_i` for input array `x`.

Args:
    x (usm_ndarray):
        Input array, expected to have a floating-point data type.
    out (Union[usm_ndarray, None], optional):
        Output array to populate.
        Array must have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the new output array, if parameter
        `out` is ``None``.
        Default: "K".

Returns:
    usm_ndarray:
        An array containing the element-wise cosine. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

cos = UnaryElementwiseFunc("cos", ti._cos_result_type, ti._cos, _cos_docstring)
del _cos_docstring

# U12: ==== COSH          (x)
_cosh_docstring = r"""
cosh(x, /, \*, out=None, order='K')

Computes hyperbolic cosine for each element `x_i` for input array `x`.

Args:
    x (usm_ndarray):
        Input array, expected to have a floating-point data type.
    out (Union[usm_ndarray, None], optional):
        Output array to populate.
        Array must have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the new output array, if parameter
        `out` is ``None``.
        Default: "K".

Returns:
    usm_ndarray:
        An array containing the element-wise hyperbolic cosine. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

cosh = UnaryElementwiseFunc(
    "cosh", ti._cosh_result_type, ti._cosh, _cosh_docstring
)
del _cosh_docstring

# U13: ==== EXP           (x)
_exp_docstring = r"""
exp(x, /, \*, out=None, order='K')

Computes the exponential for each element `x_i` of input array `x`.

Args:
    x (usm_ndarray):
        Input array, expected to have a floating-point data type.
    out (Union[usm_ndarray, None], optional):
        Output array to populate.
        Array must have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the new output array, if parameter
        `out` is ``None``.
        Default: "K".

Returns:
    usm_ndarray:
        An array containing the element-wise exponential of `x`.
        The data type of the returned array is determined by
        the Type Promotion Rules.
"""

exp = UnaryElementwiseFunc("exp", ti._exp_result_type, ti._exp, _exp_docstring)
del _exp_docstring

# U14: ==== EXPM1         (x)
_expm1_docstring = r"""
expm1(x, /, \*, out=None, order='K')

Computes the exponential minus 1 for each element `x_i` of input array `x`.

This function calculates `exp(x) - 1.0` more accurately for small values of `x`.

Args:
    x (usm_ndarray):
        Input array, expected to have a floating-point data type.
    out (usm_ndarray):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is ``None``.
        Default: "K".

Returns:
    usm_ndarray:
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
floor(x, /, \*, out=None, order='K')

Returns the floor for each element `x_i` for input array `x`.

The floor of `x_i` is the largest integer `n`, such that `n <= x_i`.

Args:
    x (usm_ndarray):
        Input array, expected to have a boolean or real-valued data type.
    out (Union[usm_ndarray, None], optional):
        Output array to populate.
        Array must have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the new output array, if parameter
        `out` is ``None``.
        Default: "K".

Returns:
    usm_ndarray:
        An array containing the element-wise floor.
"""

floor = UnaryElementwiseFunc(
    "floor", ti._floor_result_type, ti._floor, _floor_docstring
)
del _floor_docstring

# U16: ==== IMAG        (x)
_imag_docstring = r"""
imag(x, /, \*, out=None, order='K')

Computes imaginary part of each element `x_i` for input array `x`.

Args:
    x (usm_ndarray):
        Input array. May have any data type.
    out (Union[usm_ndarray, None], optional):
        Output array to populate.
        Array must have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the new output array, if parameter
        `out` is ``None``.
        Default: "K".

Returns:
    usm_ndarray:
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
isfinite(x, /, \*, out=None, order='K')

Test if each element of input array is a finite number.

Args:
    x (usm_ndarray):
        Input array. May have any data type.
    out (Union[usm_ndarray, None], optional):
        Output array to populate.
        Array must have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the new output array, if parameter
        `out` is ``None``.
        Default: "K".

Returns:
    usm_ndarray:
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
isinf(x, /, \*, out=None, order='K')

Test if each element of input array is an infinity.

Args:
    x (usm_ndarray):
        Input array. May have any data type.
    out (Union[usm_ndarray, None], optional):
        Output array to populate.
        Array must have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the new output array, if parameter
        `out` is ``None``.
        Default: "K".

Returns:
    usm_ndarray:
        An array which is True where `x` is positive or negative infinity,
        False otherwise. The data type of the returned array is `bool`.
"""

isinf = UnaryElementwiseFunc(
    "isinf", ti._isinf_result_type, ti._isinf, _isinf_docstring_
)
del _isinf_docstring_

# U19: ==== ISNAN       (x)
_isnan_docstring_ = r"""
isnan(x, /, \*, out=None, order='K')

Test if each element of an input array is a NaN.

Args:
    x (usm_ndarray):
        Input array. May have any data type.
    out (Union[usm_ndarray, None], optional):
        Output array to populate.
        Array must have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the new output array, if parameter
        `out` is ``None``.
        Default: "K".

Returns:
    usm_ndarray:
        An array which is True where x is NaN, False otherwise.
        The data type of the returned array is `bool`.
"""

isnan = UnaryElementwiseFunc(
    "isnan", ti._isnan_result_type, ti._isnan, _isnan_docstring_
)
del _isnan_docstring_

# U20: ==== LOG         (x)
_log_docstring = r"""
log(x, /, \*, out=None, order='K')

Computes the natural logarithm for each element `x_i` of input array `x`.

Args:
    x (usm_ndarray):
        Input array, expected to have a floating-point data type.
    out (usm_ndarray):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is ``None``.
        Default: "K".

Returns:
    usm_ndarray:
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

Args:
    x (usm_ndarray):
        Input array, expected to have a floating-point data type.
    out (usm_ndarray):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is ``None``.
        Default: "K".

Returns:
    usm_ndarray:
        An array containing the element-wise `log(1 + x)` results. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

log1p = UnaryElementwiseFunc(
    "log1p", ti._log1p_result_type, ti._log1p, _log1p_docstring
)
del _log1p_docstring

# U43: ==== ANGLE        (x)
_angle_docstring = r"""
angle(x, /, \*, out=None, order='K')

Computes the phase angle (also called the argument) of each element `x_i` for
input array `x`.

Args:
    x (usm_ndarray):
        Input array, expected to have a complex floating-point data type.
    out (Union[usm_ndarray, None], optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the new output array, if parameter
        `out` is ``None``.
        Default: "K".

Returns:
    usm_ndarray:
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
