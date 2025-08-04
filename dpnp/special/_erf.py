# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2025, Intel Corporation
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
Interface of the Error functions

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions

"""

import dpnp.backend.extensions.ufunc._ufunc_impl as ufi
from dpnp.dpnp_algo.dpnp_elementwise_common import DPNPUnaryFunc

__all__ = ["erf"]


class DPNPErf(DPNPUnaryFunc):
    """Class that implements a family of erf functions."""

    def __init__(
        self,
        name,
        result_type_resolver_fn,
        unary_dp_impl_fn,
        docs,
    ):
        super().__init__(
            name,
            result_type_resolver_fn,
            unary_dp_impl_fn,
            docs,
        )

    def __call__(self, x, out=None):
        return super().__call__(x, out=out)


_ERF_DOCSTRING = r"""
Returns the error function of complex argument.

It is defined as :math:`\frac{2}{\sqrt{\pi}} \int_{0}^{z} e^{-t^2} \, dt`.

For full documentation refer to :obj:`scipy.special.erf`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array.
out : {dpnp.ndarray, usm_ndarray}, optional
    Optional output array for the function values.

See Also
--------
:obj:`dpnp.special.erfc` : Complementary error function.
:obj:`dpnp.special.erfinv` : Inverse of the error function.
:obj:`dpnp.special.erfcinv` : Inverse of the complementary error function.
:obj:`dpnp.special.erfcx` : Scaled complementary error function.
:obj:`dpnp.special.erfi` : Imaginary error function.

Notes
-----
The cumulative of the unit normal distribution is given by

.. math::
    \Phi(z) = \frac{1}{2} \left[
        1 + \operatorname{erf} \left(
            \frac{z}{\sqrt{2}}
        \right)
    \right]

Examples
--------
>>> import dpnp as np
>>> x = np.linspace(-3, 3, num=5)
>>> np.special.erf(x)
array([[-0.99997791, -0.96610515,  0.        ,  0.96610515,  0.99997791])

"""

erf = DPNPErf(
    "erf",
    ufi._erf_result_type,
    ufi._erf,
    _ERF_DOCSTRING,
    # mkl_fn_to_call="_mkl_erf_to_call",
    # mkl_impl_fn="_erf",
)
