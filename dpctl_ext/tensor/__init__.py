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
    eye,
    full,
    tril,
    triu,
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
    repeat,
    roll,
)
from dpctl_ext.tensor._reshape import reshape

from ._clip import clip
from ._type_utils import can_cast, finfo, iinfo, isdtype, result_type

__all__ = [
    "asnumpy",
    "astype",
    "can_cast",
    "copy",
    "clip",
    "extract",
    "eye",
    "finfo",
    "from_numpy",
    "full",
    "iinfo",
    "isdtype",
    "nonzero",
    "place",
    "put",
    "put_along_axis",
    "repeat",
    "reshape",
    "result_type",
    "roll",
    "take",
    "take_along_axis",
    "to_numpy",
    "tril",
    "triu",
    "where",
]
