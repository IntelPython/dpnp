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


from dpctl.tensor._search_functions import where

from dpctl_ext.tensor._copy_utils import (
    asnumpy,
    astype,
    copy,
    from_numpy,
    to_numpy,
)
from dpctl_ext.tensor._ctors import (
    arange,
    asarray,
    empty,
    empty_like,
    eye,
    full,
    full_like,
    linspace,
    meshgrid,
    ones,
    ones_like,
    tril,
    triu,
    zeros,
    zeros_like,
)
from dpctl_ext.tensor._indexing_functions import (
    extract,
    nonzero,
    place,
    put,
    put_along_axis,
    take,
    take_along_axis,
)
from dpctl_ext.tensor._manipulation_functions import (
    broadcast_arrays,
    broadcast_to,
    concat,
    expand_dims,
    flip,
    moveaxis,
    permute_dims,
    repeat,
    roll,
    squeeze,
    stack,
    swapaxes,
    tile,
    unstack,
)
from dpctl_ext.tensor._reshape import reshape
from dpctl_ext.tensor._utility_functions import all, any, diff

from ._accumulation import cumulative_logsumexp, cumulative_prod, cumulative_sum
from ._clip import clip
from ._elementwise_funcs import (
    abs,
    acos,
    acosh,
    angle,
    asin,
    asinh,
    atan,
    atanh,
    bitwise_invert,
    ceil,
    conj,
    cos,
    cosh,
    exp,
    expm1,
    floor,
    imag,
    isfinite,
    isinf,
)
from ._reduction import (
    argmax,
    argmin,
    count_nonzero,
    logsumexp,
    max,
    min,
    prod,
    reduce_hypot,
    sum,
)
from ._searchsorted import searchsorted
from ._set_functions import (
    isin,
    unique_all,
    unique_counts,
    unique_inverse,
    unique_values,
)
from ._sorting import argsort, sort, top_k
from ._type_utils import can_cast, finfo, iinfo, isdtype, result_type

__all__ = [
    "abs",
    "acos",
    "acosh",
    "all",
    "angle",
    "any",
    "arange",
    "argmax",
    "argmin",
    "argsort",
    "asarray",
    "asin",
    "asinh",
    "asnumpy",
    "astype",
    "atan",
    "atanh",
    "bitwise_invert",
    "broadcast_arrays",
    "broadcast_to",
    "can_cast",
    "ceil",
    "concat",
    "conj",
    "copy",
    "cos",
    "cosh",
    "count_nonzero",
    "clip",
    "cumulative_logsumexp",
    "cumulative_prod",
    "cumulative_sum",
    "diff",
    "empty",
    "empty_like",
    "extract",
    "expand_dims",
    "eye",
    "exp",
    "expm1",
    "finfo",
    "flip",
    "floor",
    "from_numpy",
    "full",
    "full_like",
    "iinfo",
    "imag",
    "isfinite",
    "isinf",
    "isdtype",
    "isin",
    "linspace",
    "logsumexp",
    "max",
    "meshgrid",
    "min",
    "moveaxis",
    "permute_dims",
    "nonzero",
    "ones",
    "ones_like",
    "place",
    "prod",
    "put",
    "put_along_axis",
    "reduce_hypot",
    "repeat",
    "reshape",
    "result_type",
    "roll",
    "searchsorted",
    "sort",
    "squeeze",
    "stack",
    "sum",
    "swapaxes",
    "take",
    "take_along_axis",
    "tile",
    "top_k",
    "to_numpy",
    "tril",
    "triu",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
    "unstack",
    "where",
    "zeros",
    "zeros_like",
]
