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
Interface of the window functions of the dpnp

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""

import dpnp

from .dpnp_algo.dpnp_elementwise_common import DPNPAngleHamming

# pylint: disable=invalid-name

__all__ = ["hamming", "hamming_ufunc"]


def hamming(M, device=None, usm_type=None, sycl_queue=None):
    r"""
    Return the Hamming window.

    The Hamming window is a taper formed by using a weighted cosine.

    For full documentation refer to :obj:`numpy.hamming`.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty array
        is returned.
    device : {None, string, SyclDevice, SyclQueue, Device}, optional
        An array API concept of device where the output array is created.
        `device` can be ``None``, a oneAPI filter selector string, an instance
        of :class:`dpctl.SyclDevice` corresponding to a non-partitioned SYCL
        device, an instance of :class:`dpctl.SyclQueue`, or a
        :class:`dpctl.tensor.Device` object returned by
        :attr:`dpnp.ndarray.device`.

        Default: ``None``.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.

        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying. The
        `sycl_queue` can be passed as ``None`` (the default), which means
        to get the SYCL queue from `device` keyword if present or to use
        a default queue.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        The window, with the maximum value normalized to one (the value one
        appears only if the number of samples is odd).

    See Also
    --------
    :obj:`dpnp.bartlett` : Return the Bartlett window.
    :obj:`dpnp.blackman` : Return the Blackman window.
    :obj:`dpnp.hanning` : Return the Hanning window.
    :obj:`dpnp.kaiser` : Return the Kaiser window.

    Notes
    -----
    The Hamming window is defined as

    .. math::  w(n) = 0.54 - 0.46\\cos\\left(\\frac{2\\pi{n}}{M-1}\\right)
               \\qquad 0 \\leq n \\leq M-1

    Examples
    --------
    >>> import dpnp as np
    >>> np.hamming(12)
    array([0.08      , 0.15302337, 0.34890909, 0.60546483, 0.84123594,
           0.98136677, 0.98136677, 0.84123594, 0.60546483, 0.34890909,
           0.15302337, 0.08      ])  # may vary

    Creating the output array on a different device or with a
    specified usm_type:

    >>> x = np.hamming(4) # default case
    >>> x, x.device, x.usm_type
    (array([0.08, 0.77, 0.77, 0.08]), Device(level_zero:gpu:0), 'device')

    >>> y = np.hamming(4, device="cpu")
    >>> y, y.device, y.usm_type
    (array([0.08, 0.77, 0.77, 0.08]), Device(opencl:cpu:0), 'device')

    >>> z = np.hamming(4, usm_type="host")
    >>> z, z.device, z.usm_type
    (array([0.08, 0.77, 0.77, 0.08]), Device(level_zero:gpu:0), 'host')

    """

    if not isinstance(M, (int, float, dpnp.integer, dpnp.floating)):
        raise TypeError("M must be an integer")

    cfd_kwarg = {
        "device": device,
        "usm_type": usm_type,
        "sycl_queue": sycl_queue,
    }
    if M < 1:
        return dpnp.empty(0, **cfd_kwarg)
    if M == 1:
        return dpnp.ones(1, **cfd_kwarg)

    n = dpnp.arange(1 - M, M, 2, **cfd_kwarg)
    out = dpnp.empty_like(n, dtype=dpnp.default_float_type(n.device))

    alpha = dpnp.pi / (M - 1)
    dpnp.multiply(alpha, n, out=out)
    dpnp.cos(out, out=out)
    dpnp.multiply(out, 0.46, out=out)
    dpnp.add(out, 0.54, out=out)
    return out


def hamming_ufunc(M, device=None, usm_type=None, sycl_queue=None):

    if not isinstance(M, (int, float, dpnp.integer, dpnp.floating)):
        raise TypeError("M must be an integer")

    cfd_kwarg = {
        "device": device,
        "usm_type": usm_type,
        "sycl_queue": sycl_queue,
    }
    if M < 1:
        return dpnp.empty(0, **cfd_kwarg)
    if M == 1:
        return dpnp.ones(1, **cfd_kwarg)

    n = dpnp.arange(1 - M, M, 2, **cfd_kwarg)
    out = dpnp.empty_like(n, dtype=dpnp.default_float_type(n.device))

    alpha = dpnp.pi / (M - 1)
    dpnp.multiply(alpha, n, out=out)
    dpnp.cos(out, out=out)
    dpnp.multiply(out, 0.46, out=out)
    dpnp.add(out, 0.54, out=out)
    return out


_ANGLE_DOCSTRING = """
Computes the phase angle (also called the argument) of each element `x_i` for
input array `x`.

For full documentation refer to :obj:`numpy.angle`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have a complex-valued floating-point data type.
deg : bool, optional
    Return angle in degrees if ``True``, radians if ``False``.

    Default: ``False``.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise phase angles.
    The returned array has a floating-point data type determined
    by the Type Promotion Rules.

Notes
-----
Although the angle of the complex number 0 is undefined, `dpnp.angle(0)` returns the value 0.

See Also
--------
:obj:`dpnp.arctan2` : Element-wise arc tangent of `x1/x2` choosing the quadrant correctly.
:obj:`dpnp.arctan` : Trigonometric inverse tangent, element-wise.
:obj:`dpnp.absolute` : Calculate the absolute value element-wise.
:obj:`dpnp.real` : Return the real part of the complex argument.
:obj:`dpnp.imag` : Return the imaginary part of the complex argument.
:obj:`dpnp.real_if_close` : Return the real part of the input is complex
                            with all imaginary parts close to zero.

Examples
--------
>>> import dpnp as np
>>> a = np.array([1.0, 1.0j, 1+1j])
>>> np.angle(a) # in radians
array([0.        , 1.57079633, 0.78539816]) # may vary

>>> np.angle(a, deg=True) # in degrees
array([ 0., 90., 45.])
"""

hamming_ufunc = DPNPHamming(
    "hamming",
    ti._hamming_result_type,
    ti._hamming,
    _HAMMING_DOCSTRING,
)
