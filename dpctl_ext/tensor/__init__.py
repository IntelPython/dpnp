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

"""
**Data Parallel Tensor** provides an N-dimensional array container
backed by typed USM allocations and implements operations to
create and manipulate such arrays, as well as perform operations
on arrays in conformance with Python Array API standard.

[ArrayAPI] https://data-apis.org/array-api
"""

# TODO: revert to `from dpctl.tensor._copy_utils import ...`
# when dpnp fully migrates dpctl/tensor
from ._copy_utils import (  # astype,; copy,; from_numpy,; to_numpy,
    asnumpy,
)

# TODO: revert to `from dpctl.tensor._ctors import ...`
from ._ctors import (  # arange,; asarray,; empty_like,; eye,; full,; full_like,; linspace,; meshgrid,; ones,; ones_like,; tril,; triu,; zeros,; zeros_like,
    empty,
)

# TODO: revert to `from dpctl.tensor._data_types import ...`
from ._data_types import (
    bool,
    complex64,
    complex128,
    dtype,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)

# TODO: revert to `from dpctl.tensor._device import ...`
from ._device import Device

# TODO: revert to `from dpctl.tensor._dlpack import ...`
from ._dlpack import from_dlpack

# from dpctl.tensor._indexing_functions import (
#     extract,
#     nonzero,
#     place,
#     put,
#     put_along_axis,
#     take,
#     take_along_axis,
# )
# from dpctl.tensor._linear_algebra_functions import (
#     matmul,
#     matrix_transpose,
#     tensordot,
#     vecdot,
# )
# TODO: revert to `from dpctl.tensor._manipulation_functions import ...`
from ._manipulation_functions import (  # broadcast_arrays,; broadcast_to,; concat,; expand_dims,; flip,; moveaxis,; repeat,; roll,; squeeze,; stack,; swapaxes,; tile,; unstack,
    permute_dims,
)

# TODO: revert to `from dpctl.tensor._print import ...`
from ._print import (
    get_print_options,
    print_options,
    set_print_options,
    usm_ndarray_repr,
    usm_ndarray_str,
)

# TODO: revert to `from dpctl.tensor._usmarray import ...`
from ._usmarray import DLDeviceType, usm_ndarray

# from dpctl.tensor._dldevice_conversions import (
#     dldevice_to_sycl_device,
#     sycl_device_to_dldevice,
# )


# from dpctl.tensor._reshape import reshape
# from dpctl.tensor._search_functions import where
# from dpctl.tensor._statistical_functions import mean, std, var


# from dpctl.tensor._utility_functions import all, any, diff

# from ._accumulation import cumulative_logsumexp, cumulative_prod, cumulative_sum
# from ._array_api import __array_api_version__, __array_namespace_info__
# from ._clip import clip
# from ._constants import e, inf, nan, newaxis, pi
# from ._elementwise_funcs import (
#     abs,
#     acos,
#     acosh,
#     add,
#     angle,
#     asin,
#     asinh,
#     atan,
#     atan2,
#     atanh,
#     bitwise_and,
#     bitwise_invert,
#     bitwise_left_shift,
#     bitwise_or,
#     bitwise_right_shift,
#     bitwise_xor,
#     cbrt,
#     ceil,
#     conj,
#     copysign,
#     cos,
#     cosh,
#     divide,
#     equal,
#     exp,
#     exp2,
#     expm1,
#     floor,
#     floor_divide,
#     greater,
#     greater_equal,
#     hypot,
#     imag,
#     isfinite,
#     isinf,
#     isnan,
#     less,
#     less_equal,
#     log,
#     log1p,
#     log2,
#     log10,
#     logaddexp,
#     logical_and,
#     logical_not,
#     logical_or,
#     logical_xor,
#     maximum,
#     minimum,
#     multiply,
#     negative,
#     nextafter,
#     not_equal,
#     positive,
#     pow,
#     proj,
#     real,
#     reciprocal,
#     remainder,
#     round,
#     rsqrt,
#     sign,
#     signbit,
#     sin,
#     sinh,
#     sqrt,
#     square,
#     subtract,
#     tan,
#     tanh,
#     trunc,
# )
# from ._reduction import (
#     argmax,
#     argmin,
#     count_nonzero,
#     logsumexp,
#     max,
#     min,
#     prod,
#     reduce_hypot,
#     sum,
# )
# from ._searchsorted import searchsorted
# from ._set_functions import (
#     isin,
#     unique_all,
#     unique_counts,
#     unique_inverse,
#     unique_values,
# )
# from ._sorting import argsort, sort, top_k
# from ._testing import allclose
# from ._type_utils import can_cast, finfo, iinfo, isdtype, result_type

__all__ = [
    "Device",
    "usm_ndarray",
    # "arange",
    # "asarray",
    # "astype",
    # "copy",
    "empty",
    # "zeros",
    # "ones",
    # "full",
    # "eye",
    # "linspace",
    # "empty_like",
    # "zeros_like",
    # "ones_like",
    # "full_like",
    # "flip",
    # "reshape",
    # "roll",
    # "concat",
    # "stack",
    # "broadcast_arrays",
    # "broadcast_to",
    # "expand_dims",
    "permute_dims",
    # "squeeze",
    # "take",
    # "put",
    # "extract",
    # "place",
    # "nonzero",
    # "from_numpy",
    # "to_numpy",
    "asnumpy",
    "from_dlpack",
    # "tril",
    # "triu",
    # "where",
    # "matrix_transpose",
    # "all",
    # "any",
    "dtype",
    # "isdtype",
    "bool",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
    # "iinfo",
    # "finfo",
    # "unstack",
    # "moveaxis",
    # "swapaxes",
    # "can_cast",
    # "result_type",
    # "meshgrid",
    "get_print_options",
    "set_print_options",
    "print_options",
    "usm_ndarray_repr",
    "usm_ndarray_str",
    # "newaxis",
    # "e",
    # "pi",
    # "nan",
    # "inf",
    # "abs",
    # "acos",
    # "acosh",
    # "add",
    # "asin",
    # "asinh",
    # "atan",
    # "atan2",
    # "atanh",
    # "bitwise_and",
    # "bitwise_invert",
    # "bitwise_left_shift",
    # "bitwise_or",
    # "bitwise_right_shift",
    # "bitwise_xor",
    # "ceil",
    # "conj",
    # "cos",
    # "cosh",
    # "divide",
    # "equal",
    # "exp",
    # "expm1",
    # "floor",
    # "floor_divide",
    # "greater",
    # "greater_equal",
    # "hypot",
    # "imag",
    # "isfinite",
    # "isinf",
    # "isnan",
    # "less",
    # "less_equal",
    # "log",
    # "logical_and",
    # "logical_not",
    # "logical_or",
    # "logical_xor",
    # "log1p",
    # "log2",
    # "log10",
    # "maximum",
    # "minimum",
    # "multiply",
    # "negative",
    # "not_equal",
    # "positive",
    # "pow",
    # "logaddexp",
    # "proj",
    # "real",
    # "remainder",
    # "round",
    # "sign",
    # "signbit",
    # "sin",
    # "sinh",
    # "sqrt",
    # "square",
    # "subtract",
    # "not_equal",
    # "floor_divide",
    # "sum",
    # "tan",
    # "tanh",
    # "trunc",
    # "allclose",
    # "repeat",
    # "tile",
    # "max",
    # "min",
    # "argmax",
    # "argmin",
    # "prod",
    # "cbrt",
    # "exp2",
    # "copysign",
    # "rsqrt",
    # "clip",
    # "logsumexp",
    # "reduce_hypot",
    # "mean",
    # "std",
    # "var",
    # "__array_api_version__",
    # "__array_namespace_info__",
    # "reciprocal",
    # "angle",
    # "sort",
    # "argsort",
    # "unique_all",
    # "unique_counts",
    # "unique_inverse",
    # "unique_values",
    # "matmul",
    # "tensordot",
    # "vecdot",
    # "searchsorted",
    # "cumulative_logsumexp",
    # "cumulative_prod",
    # "cumulative_sum",
    # "nextafter",
    # "diff",
    # "count_nonzero",
    "DLDeviceType",
    # "take_along_axis",
    # "put_along_axis",
    # "top_k",
    # "dldevice_to_sycl_device",
    # "sycl_device_to_dldevice",
    # "isin",
]
