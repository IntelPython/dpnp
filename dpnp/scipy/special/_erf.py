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
Interface of the Error functions

Notes
-----
This module exposes the public interface for ``dpnp.scipy.special``.
it contains:
 - Interface functions
 - documentation for the functions

"""

# pylint: disable=no-name-in-module
# pylint: disable=protected-access

import dpnp.backend.extensions.ufunc._ufunc_impl as ufi
from dpnp.dpnp_algo.dpnp_elementwise_common import DPNPUnaryFunc


# pylint: disable=too-few-public-methods
class DPNPErf(DPNPUnaryFunc):
    """Class that implements a family of erf functions."""

    def __init__(
        self,
        name,
        result_type_resolver_fn,
        unary_dp_impl_fn,
        docs,
        mkl_fn_to_call=None,
        mkl_impl_fn=None,
    ):
        super().__init__(
            name,
            result_type_resolver_fn,
            unary_dp_impl_fn,
            docs,
            mkl_fn_to_call=mkl_fn_to_call,
            mkl_impl_fn=mkl_impl_fn,
        )

    def __call__(self, x, /, out=None):
        return super().__call__(x, out=out)


_ERF_DOCSTRING = r"""
Calculates the Gauss error function of a given input array.

It is defined as :math:`\frac{2}{\sqrt{\pi}} \int_{0}^{z} e^{-t^2} \, dt`.

For full documentation refer to :obj:`scipy.special.erf`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a real-valued floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray, tuple of ndarray}, optional
    Optional output array for the function values.
    A tuple (possible only as a keyword argument) must have length equal to the
    number of outputs.

    Default: ``None``.

Returns
-------
out : dpnp.ndarray
    The values of the error function at the given points `x`.

See Also
--------
:obj:`dpnp.scipy.special.erfc` : Complementary error function.
:obj:`dpnp.scipy.special.erfinv` : Inverse of the error function.
:obj:`dpnp.scipy.special.erfcinv` : Inverse of the complementary error function.
:obj:`dpnp.scipy.special.erfcx` : Scaled complementary error function.
:obj:`dpnp.scipy.special.erfi` : Imaginary error function.

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
>>> np.scipy.special.erf(x)
array([[-0.99997791, -0.96610515,  0.        ,  0.96610515,  0.99997791])

"""

erf = DPNPErf(
    "erf",
    ufi._erf_result_type,
    ufi._erf,
    _ERF_DOCSTRING,
    mkl_fn_to_call="_mkl_erf_to_call",
    mkl_impl_fn="_erf",
)

_ERFC_DOCSTRING = r"""
Calculates the complementary error function of a given input array.

It is defined as :math:`1 - \operatorname{erf}(x)`.

For full documentation refer to :obj:`scipy.special.erfc`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a real-valued floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray, tuple of ndarray}, optional
    Optional output array for the function values.
    A tuple (possible only as a keyword argument) must have length equal to the
    number of outputs.

    Default: ``None``.

Returns
-------
out : dpnp.ndarray
    The values of the complementary error function at the given points `x`.

See Also
--------
:obj:`dpnp.scipy.special.erf` : Gauss error function.
:obj:`dpnp.scipy.special.erfinv` : Inverse of the error function.
:obj:`dpnp.scipy.special.erfcinv` : Inverse of the complementary error function.
:obj:`dpnp.scipy.special.erfcx` : Scaled complementary error function.
:obj:`dpnp.scipy.special.erfi` : Imaginary error function.

Examples
--------
>>> import dpnp as np
>>> x = np.linspace(-3, 3, num=5)
>>> np.scipy.special.erfc(x)
array([[-0.99997791, -0.96610515,  0.        ,  0.96610515,  0.99997791])

"""

erfc = DPNPErf(
    "erfc",
    ufi._erf_result_type,
    ufi._erfc,
    _ERFC_DOCSTRING,
    mkl_fn_to_call="_mkl_erf_to_call",
    mkl_impl_fn="_erfc",
)

_ERFCX_DOCSTRING = r"""
Calculates the scaled complementary error function of a given input array.

It is defined as :math:`\exp(x^2) * \operatorname{erfc}(x)`.

For full documentation refer to :obj:`scipy.special.erfcx`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a real-valued floating-point data type.
out : {None, dpnp.ndarray, usm_ndarray, tuple of ndarray}, optional
    Optional output array for the function values.
    A tuple (possible only as a keyword argument) must have length equal to the
    number of outputs.

    Default: ``None``..

Returns
-------
out : dpnp.ndarray
    The values of the scaled complementary error function at the given points
    `x`.

See Also
--------
:obj:`dpnp.scipy.special.erf` : Gauss error function.
:obj:`dpnp.scipy.special.erfc` : Complementary error function.
:obj:`dpnp.scipy.special.erfinv` : Inverse of the error function.
:obj:`dpnp.scipy.special.erfcinv` : Inverse of the complementary error function.

Examples
--------
>>> import dpnp as np
>>> x = np.linspace(-3, 3, num=4)
>>> np.scipy.special.erfcx(x)
array([1.62059889e+04, 5.00898008e+00, 4.27583576e-01, 1.79001151e-01])

"""

erfcx = DPNPErf(
    "erfcx",
    ufi._erf_result_type,
    ufi._erfcx,
    _ERFCX_DOCSTRING,
    mkl_fn_to_call="_mkl_erf_to_call",
    mkl_impl_fn="_erfcx",
)

_ERFINV_DOCSTRING = r"""
Calculates the inverse of the Gauss error function of a given input array.

It is defined as :math:`\operatorname{erf}^{-1}(x)`.

For full documentation refer to :obj:`scipy.special.erfinv`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a real-valued floating-point data type.
    Domain: [-1, 1].
out : {None, dpnp.ndarray, usm_ndarray, tuple of ndarray}, optional
    Optional output array for the function values.
    A tuple (possible only as a keyword argument) must have length equal to the
    number of outputs.

    Default: ``None``.

Returns
-------
out : dpnp.ndarray
    The values of the inverse of the error function at the given points `x`.

See Also
--------
:obj:`dpnp.scipy.special.erf` : Gauss error function.
:obj:`dpnp.scipy.special.erfc` : Complementary error function.
:obj:`dpnp.scipy.special.erfcx` : Scaled complementary error function.
:obj:`dpnp.scipy.special.erfcinv` : Inverse of the complementary error function.

Examples
--------
>>> import dpnp as np
>>> x = np.linspace(-1.0, 1.0, num=9)
>>> y = np.scipy.special.erfinv(x)
>>> y
array([       -inf, -0.81341985, -0.47693628, -0.22531206,  0.        ,
        0.22531206,  0.47693628,  0.81341985,         inf])

Verify that ``erf(erfinv(x))`` is ``x``:

>>> np.scipy.special.erf(y)
array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ])

"""

erfinv = DPNPErf(
    "erfinv",
    ufi._erf_result_type,
    ufi._erfinv,
    _ERFINV_DOCSTRING,
    mkl_fn_to_call="_mkl_erf_to_call",
    mkl_impl_fn="_erfinv",
)

_ERFCINV_DOCSTRING = r"""
Calculates the inverse of the complementary error function of a given input
array.

It is defined as :math:`\operatorname{erfinv}(1 - x)`.

For full documentation refer to :obj:`scipy.special.erfcinv`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a real-valued floating-point data type.
    Domain: [0, 2].
out : {None, dpnp.ndarray, usm_ndarray, tuple of ndarray}, optional
    Optional output array for the function values.
    A tuple (possible only as a keyword argument) must have length equal to the
    number of outputs.

    Default: ``None``.

Returns
-------
out : dpnp.ndarray
    The values of the inverse of the complementary error function at the given
    points `x`.

See Also
--------
:obj:`dpnp.scipy.special.erf` : Gauss error function.
:obj:`dpnp.scipy.special.erfc` : Complementary error function.
:obj:`dpnp.scipy.special.erfcx` : Scaled complementary error function.
:obj:`dpnp.scipy.special.erfinv` : Inverse of the error function.

Examples
--------
>>> import dpnp as np
>>> x = np.linspace(0.0, 2.0, num=11)
>>> y = np.scipy.special.erfcinv(x)
>>> y
array([        inf,  0.9061938 ,  0.59511608,  0.37080716,  0.17914345,
        0.        , -0.17914345, -0.37080716, -0.59511608, -0.9061938 ,
              -inf])

Verify that ``erfc(erfcinv(x))`` is ``x``:

>>> np.scipy.special.erfc(y)
array([0. , 0.2, 0.4, 0.6, 0.8, 1. , 1.2, 1.4, 1.6, 1.8, 2. ])

"""

erfcinv = DPNPErf(
    "erfcinv",
    ufi._erf_result_type,
    ufi._erfcinv,
    _ERFCINV_DOCSTRING,
    mkl_fn_to_call="_mkl_erf_to_call",
    mkl_impl_fn="_erfcinv",
)
