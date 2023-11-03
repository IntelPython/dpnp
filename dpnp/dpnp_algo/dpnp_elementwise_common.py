# cython: language_level=3
# distutils: language = c++
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2023, Intel Corporation
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


import dpctl.tensor as dpt

import dpnp
import dpnp.backend.extensions.vm._vm_impl as vmi
from dpnp.dpnp_array import dpnp_array
from dpnp.dpnp_utils import call_origin

__all__ = [
    "check_nd_call_func",
    "dpnp_abs",
    "dpnp_acos",
    "dpnp_acosh",
    "dpnp_add",
    "dpnp_asin",
    "dpnp_asinh",
    "dpnp_atan",
    "dpnp_atan2",
    "dpnp_atanh",
    "dpnp_bitwise_and",
    "dpnp_bitwise_or",
    "dpnp_bitwise_xor",
    "dpnp_ceil",
    "dpnp_conj",
    "dpnp_cos",
    "dpnp_cosh",
    "dpnp_divide",
    "dpnp_equal",
    "dpnp_exp",
    "dpnp_expm1",
    "dpnp_floor",
    "dpnp_floor_divide",
    "dpnp_greater",
    "dpnp_greater_equal",
    "dpnp_hypot",
    "dpnp_imag",
    "dpnp_invert",
    "dpnp_isfinite",
    "dpnp_isinf",
    "dpnp_isnan",
    "dpnp_left_shift",
    "dpnp_less",
    "dpnp_less_equal",
    "dpnp_log",
    "dpnp_log10",
    "dpnp_log1p",
    "dpnp_log2",
    "dpnp_logaddexp",
    "dpnp_logical_and",
    "dpnp_logical_not",
    "dpnp_logical_or",
    "dpnp_logical_xor",
    "dpnp_maximum",
    "dpnp_minimum",
    "dpnp_multiply",
    "dpnp_negative",
    "dpnp_positive",
    "dpnp_not_equal",
    "dpnp_power",
    "dpnp_proj",
    "dpnp_real",
    "dpnp_remainder",
    "dpnp_right_shift",
    "dpnp_round",
    "dpnp_sign",
    "dpnp_signbit",
    "dpnp_sin",
    "dpnp_sinh",
    "dpnp_sqrt",
    "dpnp_square",
    "dpnp_subtract",
    "dpnp_tan",
    "dpnp_tanh",
    "dpnp_trunc",
]


def check_nd_call_func(
    origin_func,
    dpnp_func,
    *x_args,
    out=None,
    where=True,
    order="K",
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Checks arguments and calls a function.

    Chooses a common internal elementwise function to call in DPNP based on input arguments
    or to fallback on NumPy call if any passed argument is not currently supported.

    """

    args_len = len(x_args)
    if kwargs:
        pass
    elif where is not True:
        pass
    elif dtype is not None:
        pass
    elif subok is not True:
        pass
    elif args_len < 1 or args_len > 2:
        raise ValueError(
            "Unsupported number of input arrays to pass in elementwise function {}".format(
                dpnp_func.__name__
            )
        )
    elif args_len == 1 and dpnp.isscalar(x_args[0]):
        # input has to be an array
        pass
    elif (
        args_len == 2 and dpnp.isscalar(x_args[0]) and dpnp.isscalar(x_args[1])
    ):
        # at least one of input has to be an array
        pass
    else:
        if order in "afkcAFKC":
            order = order.upper()
        elif order is None:
            order = "K"
        else:
            raise ValueError(
                "order must be one of 'C', 'F', 'A', or 'K' (got '{}')".format(
                    order
                )
            )
        return dpnp_func(*x_args, out=out, order=order)
    return call_origin(
        origin_func,
        *x_args,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def _make_unary_func(
    name, dpt_unary_fn, fn_docstring, mkl_fn_to_call=None, mkl_impl_fn=None
):
    impl_fn = dpt_unary_fn.get_implementation_function()
    type_resolver_fn = dpt_unary_fn.get_type_result_resolver_function()

    def _call_func(src, dst, sycl_queue, depends=None):
        """A callback to register in UnaryElementwiseFunc class of dpctl.tensor"""

        if depends is None:
            depends = []

        if mkl_fn_to_call is not None and mkl_fn_to_call(sycl_queue, src, dst):
            # call pybind11 extension for unary function from OneMKL VM
            return mkl_impl_fn(sycl_queue, src, dst, depends)
        return impl_fn(src, dst, sycl_queue, depends)

    func = dpt_unary_fn.__class__(
        name, type_resolver_fn, _call_func, fn_docstring
    )
    return func


def _make_binary_func(
    name, dpt_binary_fn, fn_docstring, mkl_fn_to_call=None, mkl_impl_fn=None
):
    impl_fn = dpt_binary_fn.get_implementation_function()
    type_resolver_fn = dpt_binary_fn.get_type_result_resolver_function()
    fn_inplce = dpt_binary_fn.get_implementation_inplace_function()

    def _call_func(src1, src2, dst, sycl_queue, depends=None):
        """A callback to register in UnaryElementwiseFunc class of dpctl.tensor"""

        if depends is None:
            depends = []

        if mkl_fn_to_call is not None and mkl_fn_to_call(
            sycl_queue, src1, src2, dst
        ):
            # call pybind11 extension for binary function from OneMKL VM
            return mkl_impl_fn(sycl_queue, src1, src2, dst, depends)
        return impl_fn(src1, src2, dst, sycl_queue, depends)

    func = dpt_binary_fn.__class__(
        name, type_resolver_fn, _call_func, fn_docstring, fn_inplce
    )
    return func


_abs_docstring = """
abs(x, out=None, order='K')

Calculates the absolute value for each element `x_i` of input array `x`.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    dpnp.ndarray:
        An array containing the element-wise absolute values.
        For complex input, the absolute value is its magnitude.
        If `x` has a real-valued data type, the returned array has the
        same data type as `x`. If `x` has a complex floating-point data type,
        the returned array has a real-valued floating-point data type whose
        precision matches the precision of `x`.
"""

abs_func = _make_unary_func(
    "abs", dpt.abs, _abs_docstring, vmi._mkl_abs_to_call, vmi._abs
)


def dpnp_abs(x, out=None, order="K"):
    """
    Invokes abs() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for abs() function.

    """
    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = abs_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_acos_docstring = """
acos(x, out=None, order='K')

Computes inverse cosine for each element `x_i` for input array `x`.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    dpnp.ndarray:
        An array containing the element-wise inverse cosine, in radians
        and in the closed interval `[-pi/2, pi/2]`. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

acos_func = _make_unary_func(
    "arccos", dpt.acos, _acos_docstring, vmi._mkl_acos_to_call, vmi._acos
)


def dpnp_acos(x, out=None, order="K"):
    """
    Invokes acos() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for acos() function.

    """
    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = acos_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_acosh_docstring = """
acosh(x, out=None, order='K')

Computes hyperbolic inverse cosine for each element `x_i` for input array `x`.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    dpnp.ndarray:
        An array containing the element-wise inverse hyperbolic cosine.
        The data type of the returned array is determined by
        the Type Promotion Rules.
"""

acosh_func = _make_unary_func(
    "arccosh", dpt.acosh, _acosh_docstring, vmi._mkl_acosh_to_call, vmi._acosh
)


def dpnp_acosh(x, out=None, order="K"):
    """
    Invokes acosh() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for acosh() function.

    """
    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = acosh_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_add_docstring = """
add(x1, x2, out=None, order="K")

Calculates the sum for each element `x1_i` of the input array `x1` with
the respective element `x2_i` of the input array `x2`.

Args:
    x1 (dpnp.ndarray):
        First input array, expected to have numeric data type.
    x2 (dpnp.ndarray):
        Second input array, also expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", None, optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        an array containing the result of element-wise addition. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

add_func = _make_binary_func(
    "add", dpt.add, _add_docstring, vmi._mkl_add_to_call, vmi._add
)


def dpnp_add(x1, x2, out=None, order="K"):
    """
    Invokes add() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for add() function.
    """

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = add_func(
        x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order
    )
    return dpnp_array._create_from_usm_ndarray(res_usm)


_asin_docstring = """
asin(x, out=None, order='K')

Computes inverse sine for each element `x_i` for input array `x`.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    dpnp.ndarray:
        An array containing the element-wise inverse sine, in radians
        and in the closed interval `[-pi/2, pi/2]`. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

asin_func = _make_unary_func(
    "arcsin", dpt.asin, _asin_docstring, vmi._mkl_asin_to_call, vmi._asin
)


def dpnp_asin(x, out=None, order="K"):
    """
    Invokes asin() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for asin() function.

    """
    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = asin_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_asinh_docstring = """
asinh(x, out=None, order='K')

Computes inverse hyperbolic sine for each element `x_i` for input array `x`.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    dpnp.ndarray:
        An array containing the element-wise inverse hyperbolic sine.
        The data type of the returned array is determined by
        the Type Promotion Rules.
"""

asinh_func = _make_unary_func(
    "arcsinh", dpt.asinh, _asinh_docstring, vmi._mkl_asinh_to_call, vmi._asinh
)


def dpnp_asinh(x, out=None, order="K"):
    """
    Invokes asinh() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for asinh() function.

    """
    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = asinh_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_atan_docstring = """
atan(x, out=None, order='K')

Computes inverse tangent for each element `x_i` for input array `x`.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    dpnp.ndarray:
        An array containing the element-wise inverse tangent, in radians
        and in the closed interval `[-pi/2, pi/2]`. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

atan_func = _make_unary_func(
    "arctan", dpt.atan, _atan_docstring, vmi._mkl_atan_to_call, vmi._atan
)


def dpnp_atan(x, out=None, order="K"):
    """
    Invokes atan() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for atan() function.

    """
    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = atan_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_atan2_docstring = """
atan2(x1, x2, out=None, order="K")

Calculates the inverse tangent of the quotient `x1_i/x2_i` for each element
`x1_i` of the input array `x1` with the respective element `x2_i` of the
input array `x2`. Each element-wise result is expressed in radians.

Args:
    x1 (dpnp.ndarray):
        First input array, expected to have a real-valued floating-point
        data type.
    x2 (dpnp.ndarray):
        Second input array, also expected to have a real-valued
        floating-point data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", None, optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        An array containing the inverse tangent of the quotient `x1`/`x2`.
        The returned array must have a real-valued floating-point data type
        determined by Type Promotion Rules.
"""

atan2_func = _make_binary_func(
    "arctan2", dpt.atan2, _atan2_docstring, vmi._mkl_atan2_to_call, vmi._atan2
)


def dpnp_atan2(x1, x2, out=None, order="K"):
    """
    Invokes atan2() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for atan2() function.
    """

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = atan2_func(
        x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order
    )
    return dpnp_array._create_from_usm_ndarray(res_usm)


_atanh_docstring = """
atanh(x, out=None, order='K')

Computes hyperbolic inverse tangent for each element `x_i` for input array `x`.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    dpnp.ndarray:
        An array containing the element-wise hyperbolic inverse tangent.
        The data type of the returned array is determined by
        the Type Promotion Rules.
"""

atanh_func = _make_unary_func(
    "arctanh", dpt.atanh, _atanh_docstring, vmi._mkl_atanh_to_call, vmi._atanh
)


def dpnp_atanh(x, out=None, order="K"):
    """
    Invokes atanh() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for atanh() function.

    """
    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = atanh_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_bitwise_and_docstring = """
bitwise_and(x1, x2, out=None, order='K')

Computes the bitwise AND of the underlying binary representation of each
element `x1_i` of the input array `x1` with the respective element `x2_i`
of the input array `x2`.

Args:
    x1 (dpnp.ndarray):
        First input array, expected to have integer or boolean data type.
    x2 (dpnp.ndarray):
        Second input array, also expected to have integer or boolean data
        type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        An array containing the element-wise results. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

bitwise_and_func = _make_binary_func(
    "bitwise_and", dpt.bitwise_and, _bitwise_and_docstring
)


def dpnp_bitwise_and(x1, x2, out=None, order="K"):
    """Invokes bitwise_and() from dpctl.tensor implementation for bitwise_and() function."""

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = bitwise_and_func(
        x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order
    )
    return dpnp_array._create_from_usm_ndarray(res_usm)


_bitwise_or_docstring = """
bitwise_or(x1, x2, out=None, order='K')

Computes the bitwise OR of the underlying binary representation of each
element `x1_i` of the input array `x1` with the respective element `x2_i`
of the input array `x2`.

Args:
    x1 (dpnp.ndarray):
        First input array, expected to have integer or boolean data type.
    x2 (dpnp.ndarray):
        Second input array, also expected to have integer or boolean data
        type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        An array containing the element-wise results. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

bitwise_or_func = _make_binary_func(
    "bitwise_or", dpt.bitwise_or, _bitwise_or_docstring
)


def dpnp_bitwise_or(x1, x2, out=None, order="K"):
    """Invokes bitwise_or() from dpctl.tensor implementation for bitwise_or() function."""

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = bitwise_or_func(
        x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order
    )
    return dpnp_array._create_from_usm_ndarray(res_usm)


_bitwise_xor_docstring = """
bitwise_xor(x1, x2, out=None, order='K')

Computes the bitwise XOR of the underlying binary representation of each
element `x1_i` of the input array `x1` with the respective element `x2_i`
of the input array `x2`.

Args:
    x1 (dpnp.ndarray):
        First input array, expected to have integer or boolean data type.
    x2 (dpnp.ndarray):
        Second input array, also expected to have integer or boolean data
        type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        An array containing the element-wise results. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

bitwise_xor_func = _make_binary_func(
    "bitwise_xor", dpt.bitwise_xor, _bitwise_xor_docstring
)


def dpnp_bitwise_xor(x1, x2, out=None, order="K"):
    """Invokes bitwise_xor() from dpctl.tensor implementation for bitwise_xor() function."""

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = bitwise_xor_func(
        x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order
    )
    return dpnp_array._create_from_usm_ndarray(res_usm)


_ceil_docstring = """
ceil(x, out=None, order='K')

Returns the ceiling for each element `x_i` for input array `x`.
The ceil of the scalar `x` is the smallest integer `i`, such that `i >= x`.

Args:
    x (dpnp.ndarray):
        Input array, expected to have a real-valued data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    dpnp.ndarray:
        An array containing the element-wise ceiling of input array.
        The returned array has the same data type as `x`.
"""

ceil_func = _make_unary_func(
    "ceil", dpt.ceil, _ceil_docstring, vmi._mkl_ceil_to_call, vmi._ceil
)


def dpnp_ceil(x, out=None, order="K"):
    """
    Invokes ceil() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for ceil() function.
    """
    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = ceil_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_cos_docstring = """
cos(x, out=None, order='K')

Computes cosine for each element `x_i` for input array `x`.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    dpnp.ndarray:
        An array containing the element-wise cosine. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

cos_func = _make_unary_func(
    "cos", dpt.cos, _cos_docstring, vmi._mkl_cos_to_call, vmi._cos
)


def dpnp_cos(x, out=None, order="K"):
    """
    Invokes cos() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for cos() function.

    """
    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = cos_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_cosh_docstring = """
cosh(x, out=None, order='K')

Computes hyperbolic cosine for each element `x_i` for input array `x`.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    dpnp.ndarray:
        An array containing the element-wise hyperbolic cosine. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

cosh_func = _make_unary_func(
    "cosh", dpt.cosh, _cosh_docstring, vmi._mkl_cosh_to_call, vmi._cosh
)


def dpnp_cosh(x, out=None, order="K"):
    """
    Invokes cosh() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for cosh() function.

    """
    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = cosh_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_conj_docstring = """
conj(x, out=None, order='K')

Computes conjugate for each element `x_i` for input array `x`.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    dpnp.ndarray:
        An array containing the element-wise conjugate.
        The returned array has the same data type as `x`.
"""

conj_func = _make_unary_func(
    "conj", dpt.conj, _conj_docstring, vmi._mkl_conj_to_call, vmi._conj
)


def dpnp_conj(x, out=None, order="K"):
    """
    Invokes conj() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for conj() function.
    """
    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = conj_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_divide_docstring = """
divide(x1, x2, out=None, order="K")

Calculates the ratio for each element `x1_i` of the input array `x1` with
the respective element `x2_i` of the input array `x2`.

Args:
    x1 (dpnp.ndarray):
        First input array, expected to have numeric data type.
    x2 (dpnp.ndarray):
        Second input array, also expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", None, optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        an array containing the result of element-wise division. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

divide_func = _make_binary_func(
    "divide", dpt.divide, _divide_docstring, vmi._mkl_div_to_call, vmi._div
)


def dpnp_divide(x1, x2, out=None, order="K"):
    """
    Invokes div() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for divide() function.
    """

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = divide_func(
        x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order
    )
    return dpnp_array._create_from_usm_ndarray(res_usm)


_equal_docstring = """
equal(x1, x2, out=None, order="K")

Calculates equality results for each element `x1_i` of
the input array `x1` the respective element `x2_i` of the input array `x2`.

Args:
    x1 (dpnp.ndarray):
        First input array, expected to have numeric data type.
    x2 (dpnp.ndarray):
        Second input array, also expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", None, optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        an array containing the result of element-wise equality comparison.
        The data type of the returned array is determined by the Type Promotion Rules.
"""

equal_func = _make_binary_func("equal", dpt.equal, _equal_docstring)


def dpnp_equal(x1, x2, out=None, order="K"):
    """Invokes equal() from dpctl.tensor implementation for equal() function."""

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = equal_func(
        x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order
    )
    return dpnp_array._create_from_usm_ndarray(res_usm)


_exp_docstring = """
exp(x, out=None, order='K')

Computes the exponential for each element `x_i` of input array `x`.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    dpnp.ndarray:
        An array containing the element-wise exponential of `x`.
        The data type of the returned array is determined by
        the Type Promotion Rules.
"""

exp_func = _make_unary_func(
    "exp", dpt.exp, _exp_docstring, vmi._mkl_exp_to_call, vmi._exp
)


def dpnp_exp(x, out=None, order="K"):
    """
    Invokes exp() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for exp() function.
    """

    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = exp_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_expm1_docstring = """
expm1(x, out=None, order='K')

Computes the exponential minus 1 for each element `x_i` of input array `x`.

This function calculates `exp(x) - 1.0` more accurately for small values of `x`.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    dpnp.ndarray:
        An array containing the element-wise `exp(x) - 1` results.
        The data type of the returned array is determined by the Type
        Promotion Rules.
"""

expm1_func = _make_unary_func(
    "expm1", dpt.expm1, _expm1_docstring, vmi._mkl_expm1_to_call, vmi._expm1
)


def dpnp_expm1(x, out=None, order="K"):
    """
    Invokes expm1() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for expm1() function.
    """

    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = expm1_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_floor_docstring = """
floor(x, out=None, order='K')

Returns the floor for each element `x_i` for input array `x`.
The floor of the scalar `x` is the largest integer `i`, such that `i <= x`.

Args:
    x (dpnp.ndarray):
        Input array, expected to have a real-valued data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    dpnp.ndarray:
        An array containing the element-wise floor of input array.
        The returned array has the same data type as `x`.
"""

floor_func = _make_unary_func(
    "floor", dpt.floor, _floor_docstring, vmi._mkl_floor_to_call, vmi._floor
)


def dpnp_floor(x, out=None, order="K"):
    """
    Invokes floor() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for floor() function.
    """
    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = floor_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_floor_divide_docstring = """
floor_divide(x1, x2, out=None, order="K")

Calculates the ratio for each element `x1_i` of the input array `x1` with
the respective element `x2_i` of the input array `x2` to the greatest
integer-value number that is not greater than the division result.

Args:
    x1 (dpnp.ndarray):
        First input array, expected to have numeric data type.
    x2 (dpnp.ndarray):
        Second input array, also expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", None, optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        an array containing the result of element-wise floor division.
        The data type of the returned array is determined by the Type
        Promotion Rules
"""

floor_divide_func = _make_binary_func(
    "floor_divide", dpt.floor_divide, _floor_divide_docstring
)


def dpnp_floor_divide(x1, x2, out=None, order="K"):
    """Invokes floor_divide() from dpctl.tensor implementation for floor_divide() function."""

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = floor_divide_func(
        x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order
    )
    return dpnp_array._create_from_usm_ndarray(res_usm)


_greater_docstring = """
greater(x1, x2, out=None, order="K")

Calculates the greater-than results for each element `x1_i` of
the input array `x1` the respective element `x2_i` of the input array `x2`.

Args:
    x1 (dpnp.ndarray):
        First input array, expected to have numeric data type.
    x2 (dpnp.ndarray):
        Second input array, also expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", None, optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        an array containing the result of element-wise greater-than comparison.
        The data type of the returned array is determined by the Type Promotion Rules.
"""

greater_func = _make_binary_func("greater", dpt.greater, _greater_docstring)


def dpnp_greater(x1, x2, out=None, order="K"):
    """Invokes greater() from dpctl.tensor implementation for greater() function."""

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = greater_func(
        x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order
    )
    return dpnp_array._create_from_usm_ndarray(res_usm)


_greater_equal_docstring = """
greater_equal(x1, x2, out=None, order="K")

Calculates the greater-than or equal-to results for each element `x1_i` of
the input array `x1` the respective element `x2_i` of the input array `x2`.

Args:
    x1 (dpnp.ndarray):
        First input array, expected to have numeric data type.
    x2 (dpnp.ndarray):
        Second input array, also expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", None, optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        an array containing the result of element-wise greater-than or equal-to comparison.
        The data type of the returned array is determined by the Type Promotion Rules.
"""

greater_equal_func = _make_binary_func(
    "greater_equal", dpt.greater_equal, _greater_equal_docstring
)


def dpnp_greater_equal(x1, x2, out=None, order="K"):
    """Invokes greater_equal() from dpctl.tensor implementation for greater_equal() function."""

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = greater_equal_func(
        x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order
    )
    return dpnp_array._create_from_usm_ndarray(res_usm)


_hypot_docstring = """
hypot(x1, x2, out=None, order="K")

Calculates the hypotenuse for a right triangle with "legs" `x1_i` and `x2_i` of
input arrays `x1` and `x2`.

Args:
    x1 (dpnp.ndarray):
        First input array, expected to have a real-valued data type.
    x2 (dpnp.ndarray):
        Second input array, also expected to have a real-valued data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", None, optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        An array containing the element-wise hypotenuse. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

hypot_func = _make_binary_func(
    "hypot", dpt.hypot, _hypot_docstring, vmi._mkl_hypot_to_call, vmi._hypot
)


def dpnp_hypot(x1, x2, out=None, order="K"):
    """
    Invokes hypot() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for hypot() function.
    """

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = hypot_func(
        x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order
    )
    return dpnp_array._create_from_usm_ndarray(res_usm)


_imag_docstring = """
imag(x, out=None, order="K")

Computes imaginary part of each element `x_i` for input array `x`.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        An array containing the element-wise imaginary component of input.
        If the input is a real-valued data type, the returned array has
        the same data type. If the input is a complex floating-point
        data type, the returned array has a floating-point data type
        with the same floating-point precision as complex input.
"""

imag_func = _make_unary_func("imag", dpt.imag, _imag_docstring)


def dpnp_imag(x, out=None, order="K"):
    """Invokes imag() from dpctl.tensor implementation for imag() function."""

    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = imag_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_invert_docstring = """
invert(x, out=None, order='K')

Inverts (flips) each bit for each element `x_i` of the input array `x`.

Args:
    x (dpnp.ndarray):
        Input array, expected to have integer or boolean data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array must have the correct shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    dpnp.ndarray:
        An array containing the element-wise results.
        The data type of the returned array is same as the data type of the
        input array.
"""

invert_func = _make_unary_func("invert", dpt.bitwise_invert, _invert_docstring)


def dpnp_invert(x, out=None, order="K"):
    """Invokes bitwise_invert() from dpctl.tensor implementation for invert() function."""

    # dpctl.tensor only works with usm_ndarray or scalar
    x_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = invert_func(x_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_isfinite_docstring = """
isfinite(x, out=None, order="K")

Checks if each element of input array is a finite number.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        An array which is True where `x` is not positive infinity,
        negative infinity, or NaN, False otherwise.
        The data type of the returned array is `bool`.
"""

isfinite_func = _make_unary_func("isfinite", dpt.isfinite, _isfinite_docstring)


def dpnp_isfinite(x, out=None, order="K"):
    """Invokes isfinite() from dpctl.tensor implementation for isfinite() function."""

    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = isfinite_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_isinf_docstring = """
isinf(x, out=None, order="K")

Checks if each element of input array is an infinity.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        An array which is True where `x` is positive or negative infinity,
        False otherwise. The data type of the returned array is `bool`.
"""

isinf_func = _make_unary_func("isinf", dpt.isinf, _isinf_docstring)


def dpnp_isinf(x, out=None, order="K"):
    """Invokes isinf() from dpctl.tensor implementation for isinf() function."""

    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = isinf_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_isnan_docstring = """
isnan(x, out=None, order="K")

Checks if each element of an input array is a NaN.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        An array which is True where x is NaN, False otherwise.
        The data type of the returned array is `bool`.
"""

isnan_func = _make_unary_func("isnan", dpt.isnan, _isnan_docstring)


def dpnp_isnan(x, out=None, order="K"):
    """Invokes isnan() from dpctl.tensor implementation for isnan() function."""

    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = isnan_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_left_shift_docstring = """
left_shift(x1, x2, out=None, order='K')

Shifts the bits of each element `x1_i` of the input array x1 to the left by
appending `x2_i` (i.e., the respective element in the input array `x2`) zeros to
the right of `x1_i`.

Args:
    x1 (dpnp.ndarray):
        First input array, expected to have integer data type.
    x2 (dpnp.ndarray):
        Second input array, also expected to have integer data type.
        Each element must be greater than or equal to 0.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        An array containing the element-wise results. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

left_shift_func = _make_binary_func(
    "left_shift", dpt.bitwise_left_shift, _left_shift_docstring
)


def dpnp_left_shift(x1, x2, out=None, order="K"):
    """Invokes bitwise_left_shift() from dpctl.tensor implementation for left_shift() function."""

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = left_shift_func(
        x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order
    )
    return dpnp_array._create_from_usm_ndarray(res_usm)


_less_docstring = """
less(x1, x2, out=None, order="K")

Calculates the less-than results for each element `x1_i` of
the input array `x1` the respective element `x2_i` of the input array `x2`.

Args:
    x1 (dpnp.ndarray):
        First input array, expected to have numeric data type.
    x2 (dpnp.ndarray):
        Second input array, also expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", None, optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        an array containing the result of element-wise less-than comparison.
        The data type of the returned array is determined by the Type Promotion Rules.
"""

less_func = _make_binary_func("less", dpt.less, _less_docstring)


def dpnp_less(x1, x2, out=None, order="K"):
    """Invokes less() from dpctl.tensor implementation for less() function."""

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = less_func(
        x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order
    )
    return dpnp_array._create_from_usm_ndarray(res_usm)


_less_equal_docstring = """
less_equal(x1, x2, out=None, order="K")

Calculates the less-than or equal-to results for each element `x1_i` of
the input array `x1` the respective element `x2_i` of the input array `x2`.

Args:
    x1 (dpnp.ndarray):
        First input array, expected to have numeric data type.
    x2 (dpnp.ndarray):
        Second input array, also expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", None, optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        An array containing the result of element-wise less-than or equal-to comparison.
        The data type of the returned array is determined by the Type Promotion Rules.
"""

less_equal_func = _make_binary_func(
    "less_equal", dpt.less_equal, _less_equal_docstring
)


def dpnp_less_equal(x1, x2, out=None, order="K"):
    """Invokes less_equal() from dpctl.tensor implementation for less_equal() function."""

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = less_equal_func(
        x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order
    )
    return dpnp_array._create_from_usm_ndarray(res_usm)


_log_docstring = """
log(x, out=None, order='K')

Computes the natural logarithm element-wise.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    dpnp.ndarray:
        An array containing the element-wise natural logarithm values.
        The data type of the returned array is determined by the Type
        Promotion Rules.
"""

log_func = _make_unary_func(
    "log", dpt.log, _log_docstring, vmi._mkl_ln_to_call, vmi._ln
)


def dpnp_log(x, out=None, order="K"):
    """
    Invokes log() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for log() function.
    """

    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = log_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_log10_docstring = """
log10(x, out=None, order='K')

Computes the base-10 logarithm for each element `x_i` of input array `x`.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    dpnp.ndarray:
        An array containing the base-10 logarithm of `x`.
        The data type of the returned array is determined by the
        Type Promotion Rules.
"""

log10_func = _make_unary_func(
    "log10", dpt.log10, _log10_docstring, vmi._mkl_log10_to_call, vmi._log10
)


def dpnp_log10(x, out=None, order="K"):
    """
    Invokes log10() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for log10() function.
    """

    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = log10_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_log1p_docstring = """
log1p(x, out=None, order='K')

Computes an approximation of `log(1+x)` element-wise.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    dpnp.ndarray:
        An array containing the element-wise `log(1+x)` values. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

log1p_func = _make_unary_func(
    "log1p", dpt.log1p, _log1p_docstring, vmi._mkl_log1p_to_call, vmi._log1p
)


def dpnp_log1p(x, out=None, order="K"):
    """
    Invokes log1p() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for log1p() function.
    """

    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = log1p_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_log2_docstring = """
log2(x, out=None, order='K')

Computes the base-2 logarithm for each element `x_i` of input array `x`.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    dpnp.ndarray:
        An array containing the base-2 logarithm of `x`.
        The data type of the returned array is determined by the
        Type Promotion Rules.
"""

log2_func = _make_unary_func(
    "log2", dpt.log2, _log2_docstring, vmi._mkl_log2_to_call, vmi._log2
)


def dpnp_log2(x, out=None, order="K"):
    """
    Invokes log2() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for log2() function.
    """

    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = log2_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_logaddexp_docstring = """
logaddexp(x1, x2, out=None, order="K")

Calculates the natural logarithm of the sum of exponentiations for each element
`x1_i` of the input array `x1` with the respective element `x2_i` of the input
array `x2`.

This function calculates `log(exp(x1) + exp(x2))` more accurately for small
values of `x`.

Args:
    x1 (dpnp.ndarray):
        First input array, expected to have a real-valued floating-point
        data type.
    x2 (dpnp.ndarray):
        Second input array, also expected to have a real-valued
        floating-point data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", None, optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        An array containing the result of element-wise result. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

logaddexp_func = _make_binary_func(
    "logaddexp", dpt.logaddexp, _logaddexp_docstring
)


def dpnp_logaddexp(x1, x2, out=None, order="K"):
    """Invokes logaddexp() from dpctl.tensor implementation for logaddexp() function."""

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = logaddexp_func(
        x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order
    )
    return dpnp_array._create_from_usm_ndarray(res_usm)


_logical_and_docstring = """
logical_and(x1, x2, out=None, order='K')

Computes the logical AND for each element `x1_i` of the input array `x1`
with the respective element `x2_i` of the input array `x2`.

Args:
    x1 (dpnp.ndarray):
        First input array.
    x2 (dpnp.ndarray):
        Second input array.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        An array containing the element-wise logical AND results.
"""

logical_and_func = _make_binary_func(
    "logical_and", dpt.logical_and, _logical_and_docstring
)


def dpnp_logical_and(x1, x2, out=None, order="K"):
    """Invokes logical_and() from dpctl.tensor implementation for logical_and() function."""

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = logical_and_func(
        x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order
    )
    return dpnp_array._create_from_usm_ndarray(res_usm)


_logical_not_docstring = """
logical_not(x, out=None, order='K')

Computes the logical NOT for each element `x_i` of input array `x`.

Args:
    x (dpnp.ndarray):
        Input array.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    dpnp.ndarray:
        An array containing the element-wise logical NOT results.
"""

logical_not_func = _make_unary_func(
    "logical_not", dpt.logical_not, _logical_not_docstring
)


def dpnp_logical_not(x, out=None, order="K"):
    """Invokes logical_not() from dpctl.tensor implementation for logical_not() function."""

    # dpctl.tensor only works with usm_ndarray or scalar
    x_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = logical_not_func(x_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_logical_or_docstring = """
logical_or(x1, x2, out=None, order='K')

Computes the logical OR for each element `x1_i` of the input array `x1`
with the respective element `x2_i` of the input array `x2`.

Args:
    x1 (dpnp.ndarray):
        First input array.
    x2 (dpnp.ndarray):
        Second input array.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        An array containing the element-wise logical OR results.
"""

logical_or_func = _make_binary_func(
    "logical_or", dpt.logical_or, _logical_or_docstring
)


def dpnp_logical_or(x1, x2, out=None, order="K"):
    """Invokes logical_or() from dpctl.tensor implementation for logical_or() function."""

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = logical_or_func(
        x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order
    )
    return dpnp_array._create_from_usm_ndarray(res_usm)


_logical_xor_docstring = """
logical_xor(x1, x2, out=None, order='K')

Computes the logical XOR for each element `x1_i` of the input array `x1`
with the respective element `x2_i` of the input array `x2`.

Args:
    x1 (dpnp.ndarray):
        First input array.
    x2 (dpnp.ndarray):
        Second input array.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        An array containing the element-wise logical XOR results.
"""

logical_xor_func = _make_binary_func(
    "logical_xor", dpt.logical_xor, _logical_xor_docstring
)


def dpnp_logical_xor(x1, x2, out=None, order="K"):
    """Invokes logical_xor() from dpctl.tensor implementation for logical_xor() function."""

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = logical_xor_func(
        x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order
    )
    return dpnp_array._create_from_usm_ndarray(res_usm)


_maximum_docstring = """
maximum(x1, x2, out=None, order='K')

Compares two input arrays `x1` and `x2` and returns
a new array containing the element-wise maxima.

Args:
    x1 (dpnp.ndarray):
        First input array, expected to have numeric data type.
    x2 (dpnp.ndarray):
        Second input array, also expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        An array containing the element-wise maxima. The data type of
        the returned array is determined by the Type Promotion Rules.
"""

maximum_func = _make_binary_func("maximum", dpt.maximum, _maximum_docstring)


def dpnp_maximum(x1, x2, out=None, order="K"):
    """Invokes maximum() from dpctl.tensor implementation for maximum() function."""

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = maximum_func(
        x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order
    )
    return dpnp_array._create_from_usm_ndarray(res_usm)


_minimum_docstring = """
minimum(x1, x2, out=None, order='K')

Compares two input arrays `x1` and `x2` and returns
a new array containing the element-wise minima.

Args:
    x1 (dpnp.ndarray):
        First input array, expected to have numeric data type.
    x2 (dpnp.ndarray):
        Second input array, also expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        An array containing the element-wise minima. The data type of
        the returned array is determined by the Type Promotion Rules.
"""

minimum_func = _make_binary_func("minimum", dpt.minimum, _minimum_docstring)


def dpnp_minimum(x1, x2, out=None, order="K"):
    """Invokes minimum() from dpctl.tensor implementation for minimum() function."""

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = minimum_func(
        x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order
    )
    return dpnp_array._create_from_usm_ndarray(res_usm)


_multiply_docstring = """
multiply(x1, x2, out=None, order="K")

Calculates the product for each element `x1_i` of the input array `x1`
with the respective element `x2_i` of the input array `x2`.

Args:
    x1 (dpnp.ndarray):
        First input array, expected to have numeric data type.
    x2 (dpnp.ndarray):
        Second input array, also expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", None, optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        an array containing the result of element-wise multiplication. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

multiply_func = _make_binary_func(
    "multiply",
    dpt.multiply,
    _multiply_docstring,
    vmi._mkl_mul_to_call,
    vmi._mul,
)


def dpnp_multiply(x1, x2, out=None, order="K"):
    """
    Invokes mul() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for multiply() function.
    """

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = multiply_func(
        x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order
    )
    return dpnp_array._create_from_usm_ndarray(res_usm)


_negative_docstring = """
negative(x, out=None, order="K")

Computes the numerical negative for each element `x_i` of input array `x`.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        An array containing the negative of `x`.
"""

negative_func = _make_unary_func("negative", dpt.negative, _negative_docstring)


def dpnp_negative(x, out=None, order="K"):
    """Invokes negative() from dpctl.tensor implementation for negative() function."""

    # TODO: discuss with dpctl if the check is needed to be moved there
    if not dpnp.isscalar(x) and x.dtype == dpnp.bool:
        raise TypeError(
            "DPNP boolean negative, the `-` operator, is not supported, "
            "use the `~` operator or the logical_not function instead."
        )

    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = negative_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_not_equal_docstring = """
not_equal(x1, x2, out=None, order="K")

Calculates inequality results for each element `x1_i` of
the input array `x1` the respective element `x2_i` of the input array `x2`.

Args:
    x1 (dpnp.ndarray):
        First input array, expected to have numeric data type.
    x2 (dpnp.ndarray):
        Second input array, also expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", None, optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        an array containing the result of element-wise inequality comparison.
        The data type of the returned array is determined by the Type Promotion Rules.
"""

not_equal_func = _make_binary_func(
    "not_equal", dpt.not_equal, _not_equal_docstring
)


def dpnp_not_equal(x1, x2, out=None, order="K"):
    """Invokes not_equal() from dpctl.tensor implementation for not_equal() function."""

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = not_equal_func(
        x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order
    )
    return dpnp_array._create_from_usm_ndarray(res_usm)


_positive_docstring = """
positive(x, out=None, order="K")

Computes the numerical positive for each element `x_i` of input array `x`.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        An array containing the positive of `x`.
"""

positive_func = _make_unary_func("positive", dpt.positive, _positive_docstring)


def dpnp_positive(x, out=None, order="K"):
    """Invokes positive() from dpctl.tensor implementation for positive() function."""

    # TODO: discuss with dpctl if the check is needed to be moved there
    if not dpnp.isscalar(x) and x.dtype == dpnp.bool:
        raise TypeError(
            "DPNP boolean positive, the `+` operator, is not supported."
        )

    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = positive_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_power_docstring = """
power(x1, x2, out=None, order="K")

Calculates `x1_i` raised to `x2_i` for each element `x1_i` of the input array
`x1` with the respective element `x2_i` of the input array `x2`.

Args:
    x1 (dpnp.ndarray):
        First input array, expected to have numeric data type.
    x2 (dpnp.ndarray):
        Second input array, also expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", None, optional):
        Output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        An array containing the result of element-wise of raising each element
        to a specified power.
        The data type of the returned array is determined by the Type Promotion Rules.
"""

power_func = _make_binary_func(
    "power", dpt.pow, _power_docstring, vmi._mkl_pow_to_call, vmi._pow
)


def dpnp_power(x1, x2, out=None, order="K"):
    """
    Invokes pow() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for pow() function.
    """

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = power_func(
        x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order
    )
    return dpnp_array._create_from_usm_ndarray(res_usm)


_proj_docstring = """
proj(x, out=None, order="K")

Computes projection of each element `x_i` for input array `x`.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        An array containing the element-wise projection.
        The returned array has the same data type as `x`.
"""

proj_func = _make_unary_func("proj", dpt.proj, _proj_docstring)


def dpnp_proj(x, out=None, order="K"):
    """Invokes proj() from dpctl.tensor implementation for proj() function."""

    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = proj_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_real_docstring = """
real(x, out=None, order="K")

Computes real part of each element `x_i` for input array `x`.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        An array containing the element-wise real component of input.
        If the input is a real-valued data type, the returned array has
        the same data type. If the input is a complex floating-point
        data type, the returned array has a floating-point data type
        with the same floating-point precision as complex input.
"""

real_func = _make_unary_func("real", dpt.real, _real_docstring)


def dpnp_real(x, out=None, order="K"):
    """Invokes real() from dpctl.tensor implementation for real() function."""

    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = real_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_remainder_docstring = """
remainder(x1, x2, out=None, order='K')

Calculates the remainder of division for each element `x1_i` of the input array
`x1` with the respective element `x2_i` of the input array `x2`.
This function is equivalent to the Python modulus operator.

Args:
    x1 (dpnp.ndarray):
        First input array, expected to have a real-valued data type.
    x2 (dpnp.ndarray):
        Second input array, also expected to have a real-valued data type.
    out ({None, usm_ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        an array containing the element-wise remainders. The data type of
        the returned array is determined by the Type Promotion Rules.
"""

remainder_func = _make_binary_func(
    "remainder", dpt.remainder, _remainder_docstring
)


def dpnp_remainder(x1, x2, out=None, order="K"):
    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = remainder_func(
        x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order
    )
    return dpnp_array._create_from_usm_ndarray(res_usm)


_right_shift_docstring = """
right_shift(x1, x2, out=None, order='K')

Shifts the bits of each element `x1_i` of the input array `x1` to the right
according to the respective element `x2_i` of the input array `x2`.

Args:
    x1 (dpnp.ndarray):
        First input array, expected to have integer data type.
    x2 (dpnp.ndarray):
        Second input array, also expected to have integer data type.
        Each element must be greater than or equal to 0.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        An array containing the element-wise results. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

right_shift_func = _make_binary_func(
    "right_shift", dpt.bitwise_right_shift, _right_shift_docstring
)


def dpnp_right_shift(x1, x2, out=None, order="K"):
    """Invokes bitwise_right_shift() from dpctl.tensor implementation for right_shift() function."""

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = right_shift_func(
        x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order
    )
    return dpnp_array._create_from_usm_ndarray(res_usm)


_round_docstring = """
round(x, out=None, order='K')

Rounds each element `x_i` of the input array `x` to
the nearest integer-valued number.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    dpnp.ndarray:
        An array containing the element-wise rounded value. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

round_func = _make_unary_func(
    "round", dpt.round, _round_docstring, vmi._mkl_round_to_call, vmi._round
)


def dpnp_round(x, out=None, order="K"):
    """
    Invokes round() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for round() function.
    """
    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = round_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_sign_docstring = """
sign(x, out=None, order="K")

Computes an indication of the sign of each element `x_i` of input array `x`
using the signum function.

The signum function returns `-1` if `x_i` is less than `0`,
`0` if `x_i` is equal to `0`, and `1` if `x_i` is greater than `0`.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        An array containing the element-wise results. The data type of the
        returned array is determined by the Type Promotion Rules.
"""

sign_func = _make_unary_func("sign", dpt.sign, _sign_docstring)


def dpnp_sign(x, out=None, order="K"):
    """Invokes sign() from dpctl.tensor implementation for sign() function."""

    # TODO: discuss with dpctl if the check is needed to be moved there
    if not dpnp.isscalar(x) and x.dtype == dpnp.bool:
        raise TypeError("DPNP boolean sign is not supported.")

    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = sign_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_signbit_docstring = """
signbit(x, out=None, order="K")

Computes an indication of whether the sign bit of each element `x_i` of
input array `x` is set.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        An array containing the element-wise results. The returned array
        must have a data type of `bool`.
"""

signbit_func = _make_unary_func("signbit", dpt.signbit, _signbit_docstring)


def dpnp_signbit(x, out=None, order="K"):
    """Invokes signbit() from dpctl.tensor implementation for signbit() function."""

    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = signbit_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_sin_docstring = """
sin(x, out=None, order='K')

Computes sine for each element `x_i` of input array `x`.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    dpnp.ndarray:
        An array containing the element-wise sine. The data type of the
        returned array is determined by the Type Promotion Rules.
"""

sin_func = _make_unary_func(
    "sin", dpt.sin, _sin_docstring, vmi._mkl_sin_to_call, vmi._sin
)


def dpnp_sin(x, out=None, order="K"):
    """
    Invokes sin() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for sin() function.
    """

    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = sin_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_sinh_docstring = """
sinh(x, out=None, order='K')

Computes hyperbolic sine for each element `x_i` for input array `x`.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    dpnp.ndarray:
        An array containing the element-wise hyperbolic sine. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

sinh_func = _make_unary_func(
    "sinh", dpt.sinh, _sinh_docstring, vmi._mkl_sinh_to_call, vmi._sinh
)


def dpnp_sinh(x, out=None, order="K"):
    """
    Invokes sinh() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for sinh() function.

    """
    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = sinh_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_sqrt_docstring = """
sqrt(x, out=None, order='K')

Computes the non-negative square-root for each element `x_i` for input array `x`.

Args:
    x (dpnp.ndarray):
        Input array.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    dpnp.ndarray:
        An array containing the element-wise square-root results.
"""

sqrt_func = _make_unary_func(
    "sqrt", dpt.sqrt, _sqrt_docstring, vmi._mkl_sqrt_to_call, vmi._sqrt
)


def dpnp_sqrt(x, out=None, order="K"):
    """
    Invokes sqrt() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for sqrt() function.
    """

    # dpctl.tensor only works with usm_ndarray or scalar
    x_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = sqrt_func(x_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_square_docstring = """
square(x, out=None, order='K')

Computes `x_i**2` (or `x_i*x_i`) for each element `x_i` of input array `x`.

Args:
    x (dpnp.ndarray):
        Input array.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    dpnp.ndarray:
        An array containing the element-wise square results.
"""

square_func = _make_unary_func(
    "square", dpt.square, _square_docstring, vmi._mkl_sqr_to_call, vmi._sqr
)


def dpnp_square(x, out=None, order="K"):
    """
    Invokes sqr() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for square() function.
    """

    # dpctl.tensor only works with usm_ndarray or scalar
    x_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = square_func(x_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_subtract_docstring = """
subtract(x1, x2, out=None, order="K")

Calculates the difference between each element `x1_i` of the input
array `x1` and the respective element `x2_i` of the input array `x2`.

Args:
    x1 (dpnp.ndarray):
        First input array, expected to have numeric data type.
    x2 (dpnp.ndarray):
        Second input array, also expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", None, optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        an array containing the result of element-wise subtraction. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

subtract_func = _make_binary_func(
    "subtract",
    dpt.subtract,
    _subtract_docstring,
    vmi._mkl_sub_to_call,
    vmi._sub,
)


def dpnp_subtract(x1, x2, out=None, order="K"):
    """
    Invokes sub() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for subtract() function.
    """

    # TODO: discuss with dpctl if the check is needed to be moved there
    if (
        not dpnp.isscalar(x1)
        and not dpnp.isscalar(x2)
        and x1.dtype == x2.dtype == dpnp.bool
    ):
        raise TypeError(
            "DPNP boolean subtract, the `-` operator, is not supported, "
            "use the bitwise_xor, the `^` operator, or the logical_xor function instead."
        )

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = subtract_func(
        x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order
    )
    return dpnp_array._create_from_usm_ndarray(res_usm)


_tan_docstring = """
tan(x, out=None, order='K')

Computes tangent for each element `x_i` for input array `x`.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    dpnp.ndarray:
        An array containing the element-wise tangent. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

tan_func = _make_unary_func(
    "tan", dpt.tan, _tan_docstring, vmi._mkl_tan_to_call, vmi._tan
)


def dpnp_tan(x, out=None, order="K"):
    """
    Invokes tan() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for tan() function.

    """
    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = tan_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_tanh_docstring = """
tanh(x, out=None, order='K')

Computes hyperbolic tangent for each element `x_i` for input array `x`.

Args:
    x (dpnp.ndarray):
        Input array, expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    dpnp.ndarray:
        An array containing the element-wise hyperbolic tangent. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

tanh_func = _make_unary_func(
    "tanh", dpt.tanh, _tanh_docstring, vmi._mkl_tanh_to_call, vmi._tanh
)


def dpnp_tanh(x, out=None, order="K"):
    """
    Invokes tanh() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for tanh() function.

    """
    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = tanh_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_trunc_docstring = """
trunc(x, out=None, order='K')

Returns the truncated value for each element `x_i` for input array `x`.
The truncated value of the scalar `x` is the nearest integer `i` which is
closer to zero than `x` is. In short, the fractional part of the
signed number `x` is discarded.

Args:
    x (dpnp.ndarray):
        Input array, expected to have a real-valued data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate. Array must have the correct
        shape and the expected data type.
    order ("C","F","A","K", optional): memory layout of the new
        output array, if parameter `out` is `None`.
        Default: "K".
Return:
    dpnp.ndarray:
        An array containing the truncated value of each element in `x`. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

trunc_func = _make_unary_func(
    "trunc", dpt.trunc, _trunc_docstring, vmi._mkl_trunc_to_call, vmi._trunc
)


def dpnp_trunc(x, out=None, order="K"):
    """
    Invokes trunc() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for trunc() function.
    """
    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    res_usm = trunc_func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)
