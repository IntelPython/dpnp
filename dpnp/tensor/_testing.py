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

import dpctl.utils as du
import numpy as np

import dpnp.tensor as dpt

from ._manipulation_functions import _broadcast_shape_impl
from ._type_utils import _to_device_supported_dtype


def _allclose_complex_fp(z1, z2, atol, rtol, equal_nan):
    z1r = dpt.real(z1)
    z1i = dpt.imag(z1)
    z2r = dpt.real(z2)
    z2i = dpt.imag(z2)
    if equal_nan:
        check1 = dpt.all(dpt.isnan(z1r) == dpt.isnan(z2r)) and dpt.all(
            dpt.isnan(z1i) == dpt.isnan(z2i)
        )
    else:
        check1 = (
            dpt.logical_not(dpt.any(dpt.isnan(z1r)))
            and dpt.logical_not(dpt.any(dpt.isnan(z1i)))
        ) and (
            dpt.logical_not(dpt.any(dpt.isnan(z2r)))
            and dpt.logical_not(dpt.any(dpt.isnan(z2i)))
        )
    if not check1:
        return check1
    mr = dpt.isinf(z1r)
    mi = dpt.isinf(z1i)
    check2 = dpt.all(mr == dpt.isinf(z2r)) and dpt.all(mi == dpt.isinf(z2i))
    if not check2:
        return check2
    check3 = dpt.all(z1r[mr] == z2r[mr]) and dpt.all(z1i[mi] == z2i[mi])
    if not check3:
        return check3
    mr = dpt.isfinite(z1r)
    mi = dpt.isfinite(z1i)
    mv1 = z1r[mr]
    mv2 = z2r[mr]
    check4 = dpt.all(
        dpt.abs(mv1 - mv2)
        < dpt.maximum(atol, rtol * dpt.maximum(dpt.abs(mv1), dpt.abs(mv2)))
    )
    if not check4:
        return check4
    mv1 = z1i[mi]
    mv2 = z2i[mi]
    check5 = dpt.all(
        dpt.abs(mv1 - mv2)
        <= dpt.maximum(atol, rtol * dpt.maximum(dpt.abs(mv1), dpt.abs(mv2)))
    )
    return check5


def _allclose_real_fp(r1, r2, atol, rtol, equal_nan):
    if equal_nan:
        check1 = dpt.all(dpt.isnan(r1) == dpt.isnan(r2))
    else:
        check1 = dpt.logical_not(dpt.any(dpt.isnan(r1))) and dpt.logical_not(
            dpt.any(dpt.isnan(r2))
        )
    if not check1:
        return check1
    mr = dpt.isinf(r1)
    check2 = dpt.all(mr == dpt.isinf(r2))
    if not check2:
        return check2
    check3 = dpt.all(r1[mr] == r2[mr])
    if not check3:
        return check3
    m = dpt.isfinite(r1)
    mv1 = r1[m]
    mv2 = r2[m]
    check4 = dpt.all(
        dpt.abs(mv1 - mv2)
        <= dpt.maximum(atol, rtol * dpt.maximum(dpt.abs(mv1), dpt.abs(mv2)))
    )
    return check4


def _allclose_others(r1, r2):
    return dpt.all(r1 == r2)


def allclose(a1, a2, atol=1e-8, rtol=1e-5, equal_nan=False):
    """allclose(a1, a2, atol=1e-8, rtol=1e-5, equal_nan=False)

    Returns True if two arrays are element-wise equal within tolerances.

    The testing is based on the following elementwise comparison:

           abs(a - b) <= max(atol, rtol * max(abs(a), abs(b)))
    """
    if not isinstance(a1, dpt.usm_ndarray):
        raise TypeError(
            f"Expected dpctl.tensor.usm_ndarray type, got {type(a1)}."
        )
    if not isinstance(a2, dpt.usm_ndarray):
        raise TypeError(
            f"Expected dpctl.tensor.usm_ndarray type, got {type(a2)}."
        )
    atol = float(atol)
    rtol = float(rtol)
    if atol < 0.0 or rtol < 0.0:
        raise ValueError(
            "Absolute and relative tolerances must be non-negative"
        )
    equal_nan = bool(equal_nan)
    exec_q = du.get_execution_queue(tuple(a.sycl_queue for a in (a1, a2)))
    if exec_q is None:
        raise du.ExecutionPlacementError(
            "Execution placement can not be unambiguously inferred "
            "from input arguments."
        )
    res_sh = _broadcast_shape_impl([a1.shape, a2.shape])
    b1 = a1
    b2 = a2
    if b1.dtype == b2.dtype:
        res_dt = b1.dtype
    else:
        res_dt = np.promote_types(b1.dtype, b2.dtype)
        res_dt = _to_device_supported_dtype(res_dt, exec_q.sycl_device)
        b1 = dpt.astype(b1, res_dt)
        b2 = dpt.astype(b2, res_dt)

    b1 = dpt.broadcast_to(b1, res_sh)
    b2 = dpt.broadcast_to(b2, res_sh)

    k = b1.dtype.kind
    if k == "c":
        return _allclose_complex_fp(b1, b2, atol, rtol, equal_nan)
    elif k == "f":
        return _allclose_real_fp(b1, b2, atol, rtol, equal_nan)
    else:
        return _allclose_others(b1, b2)
