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


import dpctl
import dpctl.tensor as dpt
import dpctl.tensor._tensor_impl as ti
from dpctl.tensor._elementwise_common import (
    BinaryElementwiseFunc,
    UnaryElementwiseFunc,
)

import dpnp
import dpnp.backend.extensions.vm._vm_impl as vmi
from dpnp.dpnp_array import dpnp_array
from dpnp.dpnp_utils import call_origin

__all__ = [
    "check_nd_call_func",
    "dpnp_add",
    "dpnp_divide",
    "dpnp_equal",
    "dpnp_greater",
    "dpnp_greater_equal",
    "dpnp_less",
    "dpnp_less_equal",
    "dpnp_log",
    "dpnp_multiply",
    "dpnp_not_equal",
    "dpnp_subtract",
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
    Checks arguments and calls function with a single input array.

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


_add_docstring_ = """
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
        an array containing the result of element-wise division. The data type
        of the returned array is determined by the Type Promotion Rules.
"""


def dpnp_add(x1, x2, out=None, order="K"):
    """
    Invokes add() from dpctl.tensor implementation for add() function.

    TODO: add a pybind11 extension of add() from OneMKL VM where possible
    and would be performance effective.

    """

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    func = BinaryElementwiseFunc(
        "add", ti._add_result_type, ti._add, _add_docstring_, ti._add_inplace
    )
    res_usm = func(x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_divide_docstring_ = """
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


def dpnp_divide(x1, x2, out=None, order="K"):
    """
    Invokes div() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for divide() function.

    """

    def _call_divide(src1, src2, dst, sycl_queue, depends=None):
        """A callback to register in BinaryElementwiseFunc class of dpctl.tensor"""

        if depends is None:
            depends = []

        if vmi._mkl_div_to_call(sycl_queue, src1, src2, dst):
            # call pybind11 extension for div() function from OneMKL VM
            return vmi._div(sycl_queue, src1, src2, dst, depends)
        return ti._divide(src1, src2, dst, sycl_queue, depends)

    def _call_divide_inplace(lhs, rhs, sycl_queue, depends=None):
        """In place workaround until dpctl.tensor provides the functionality."""

        if depends is None:
            depends = []

        # allocate temporary memory for out array
        out = dpt.empty_like(lhs, dtype=dpnp.result_type(lhs.dtype, rhs.dtype))

        # call a general callback
        div_ht_, div_ev_ = _call_divide(lhs, rhs, out, sycl_queue, depends)

        # store the result into left input array and return events
        cp_ht_, cp_ev_ = ti._copy_usm_ndarray_into_usm_ndarray(
            src=out, dst=lhs, sycl_queue=sycl_queue, depends=[div_ev_]
        )
        dpctl.SyclEvent.wait_for([div_ht_])
        return (cp_ht_, cp_ev_)

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    func = BinaryElementwiseFunc(
        "divide",
        ti._divide_result_type,
        _call_divide,
        _divide_docstring_,
        _call_divide_inplace,
    )
    res_usm = func(x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_equal_docstring_ = """

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


def dpnp_equal(x1, x2, out=None, order="K"):
    """Invokes equal() from dpctl.tensor implementation for equal() function."""

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    func = BinaryElementwiseFunc(
        "equal", ti._equal_result_type, ti._equal, _equal_docstring_
    )
    res_usm = func(x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_greater_docstring_ = """
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


def dpnp_greater(x1, x2, out=None, order="K"):
    """Invokes greater() from dpctl.tensor implementation for greater() function."""

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    func = BinaryElementwiseFunc(
        "greater", ti._greater_result_type, ti._greater, _greater_docstring_
    )
    res_usm = func(x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_greater_equal_docstring_ = """
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


def dpnp_greater_equal(x1, x2, out=None, order="K"):
    """Invokes greater_equal() from dpctl.tensor implementation for greater_equal() function."""

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    func = BinaryElementwiseFunc(
        "greater_equal",
        ti._greater_equal_result_type,
        ti._greater_equal,
        _greater_equal_docstring_,
    )
    res_usm = func(x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_less_docstring_ = """
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


def dpnp_less(x1, x2, out=None, order="K"):
    """Invokes less() from dpctl.tensor implementation for less() function."""

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    func = BinaryElementwiseFunc(
        "less", ti._less_result_type, ti._less, _less_docstring_
    )
    res_usm = func(x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_less_equal_docstring_ = """
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


def dpnp_less_equal(x1, x2, out=None, order="K"):
    """Invokes less_equal() from dpctl.tensor implementation for less_equal() function."""

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    func = BinaryElementwiseFunc(
        "less_equal",
        ti._less_equal_result_type,
        ti._less_equal,
        _less_equal_docstring_,
    )
    res_usm = func(x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order)
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
    usm_ndarray:
        An array containing the element-wise natural logarithm values.
"""


def dpnp_log(x, out=None, order="K"):
    """
    Invokes log() function from pybind11 extension of OneMKL VM if possible.

    Otherwise fully relies on dpctl.tensor implementation for log() function.

    """

    def _call_log(src, dst, sycl_queue, depends=None):
        """A callback to register in UnaryElementwiseFunc class of dpctl.tensor"""

        if depends is None:
            depends = []

        if vmi._mkl_ln_to_call(sycl_queue, src, dst):
            # call pybind11 extension for ln() function from OneMKL VM
            return vmi._ln(sycl_queue, src, dst, depends)
        return ti._log(src, dst, sycl_queue, depends)

    # dpctl.tensor only works with usm_ndarray
    x1_usm = dpnp.get_usm_ndarray(x)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    func = UnaryElementwiseFunc(
        "log", ti._log_result_type, _call_log, _log_docstring
    )
    res_usm = func(x1_usm, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_multiply_docstring_ = """
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
        an array containing the result of element-wise division. The data type
        of the returned array is determined by the Type Promotion Rules.
"""


def dpnp_multiply(x1, x2, out=None, order="K"):
    """
    Invokes multiply() from dpctl.tensor implementation for multiply() function.

    TODO: add a pybind11 extension of mul() from OneMKL VM where possible
    and would be performance effective.

    """

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    func = BinaryElementwiseFunc(
        "multiply",
        ti._multiply_result_type,
        ti._multiply,
        _multiply_docstring_,
        ti._multiply_inplace,
    )
    res_usm = func(x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_not_equal_docstring_ = """
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


def dpnp_not_equal(x1, x2, out=None, order="K"):
    """Invokes not_equal() from dpctl.tensor implementation for not_equal() function."""

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    func = BinaryElementwiseFunc(
        "not_equal",
        ti._not_equal_result_type,
        ti._not_equal,
        _not_equal_docstring_,
    )
    res_usm = func(x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)


_subtract_docstring_ = """
subtract(x1, x2, out=None, order="K")

Calculates the difference bewteen each element `x1_i` of the input
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
        an array containing the result of element-wise division. The data type
        of the returned array is determined by the Type Promotion Rules.
"""


def dpnp_subtract(x1, x2, out=None, order="K"):
    """
    Invokes subtract() from dpctl.tensor implementation for subtract() function.

    TODO: add a pybind11 extension of sub() from OneMKL VM where possible
    and would be performance effective.

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

    func = BinaryElementwiseFunc(
        "subtract",
        ti._subtract_result_type,
        ti._subtract,
        _subtract_docstring_,
        ti._subtract_inplace,
    )
    res_usm = func(x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)
