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
