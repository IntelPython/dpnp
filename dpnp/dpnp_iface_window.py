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
Interface of the window functions of dpnp

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""

# pylint: disable=no-name-in-module
# pylint: disable=invalid-name
# pylint: disable=protected-access

import dpctl.utils as dpu

import dpnp
import dpnp.backend.extensions.window._window_impl as wi

__all__ = ["bartlett", "blackman", "hamming", "hanning", "kaiser"]


def _call_window_kernel(
    M, _window_kernel, device=None, usm_type=None, sycl_queue=None, beta=None
):

    try:
        M = int(M)
    except Exception as e:
        raise TypeError("M must be an integer") from e

    cfd_kwarg = {
        "device": device,
        "usm_type": usm_type,
        "sycl_queue": sycl_queue,
    }

    if M < 1:
        return dpnp.empty(0, **cfd_kwarg)
    if M == 1:
        return dpnp.ones(1, **cfd_kwarg)

    result = dpnp.empty(M, **cfd_kwarg)
    exec_q = result.sycl_queue
    _manager = dpu.SequentialOrderManager[exec_q]

    # there are no dependent events for window kernels
    if beta is None:
        ht_ev, win_ev = _window_kernel(
            exec_q,
            dpnp.get_usm_ndarray(result),
        )
    else:
        ht_ev, win_ev = _window_kernel(
            exec_q,
            beta,
            dpnp.get_usm_ndarray(result),
        )

    _manager.add_event_pair(ht_ev, win_ev)

    return result


def bartlett(M, *, device=None, usm_type=None, sycl_queue=None):
    r"""
    Return the Bartlett window.

    The Bartlett window is very similar to a triangular window, except that the
    end points are at zero. It is often used in signal processing for tapering
    a signal, without generating too much ripple in the frequency domain.

    For full documentation refer to :obj:`numpy.bartlett`.

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
    out : dpnp.ndarray of shape (M,)
        The triangular window, with the maximum value normalized to one
        (the value one appears only if the number of samples is odd), with the
        first and last samples equal to zero.

    See Also
    --------
    :obj:`dpnp.blackman` : Return the Blackman window.
    :obj:`dpnp.hamming` : Return the Hamming window.
    :obj:`dpnp.hanning` : Return the Hanning window.
    :obj:`dpnp.kaiser` : Return the Kaiser window.

    Notes
    -----
    The Bartlett window is defined as

    .. math::  w(n) = \frac{2}{M-1} \left(\frac{M-1}{2} -
               \left|n - \frac{M-1}{2}\right|\right)
               \qquad 0 \leq n \leq M-1

    Examples
    --------
    >>> import dpnp as np
    >>> np.bartlett(12)
    array([0.        , 0.18181818, 0.36363636, 0.54545455, 0.72727273,
        0.90909091, 0.90909091, 0.72727273, 0.54545455, 0.36363636,
        0.18181818, 0.        ])

    Creating the output array on a different device or with a
    specified usm_type:

    >>> x = np.bartlett(4) # default case
    >>> x, x.device, x.usm_type
    (array([0.        , 0.66666667, 0.66666667, 0.        ]),
     Device(level_zero:gpu:0),
     'device')

    >>> y = np.bartlett(4, device="cpu")
    >>> y, y.device, y.usm_type
    (array([0.        , 0.66666667, 0.66666667, 0.        ]),
     Device(opencl:cpu:0),
     'device')

    >>> z = np.bartlett(4, usm_type="host")
    >>> z, z.device, z.usm_type
    (array([0.        , 0.66666667, 0.66666667, 0.        ]),
     Device(level_zero:gpu:0),
     'host')

    """

    return _call_window_kernel(
        M, wi._bartlett, device=device, usm_type=usm_type, sycl_queue=sycl_queue
    )


def blackman(M, *, device=None, usm_type=None, sycl_queue=None):
    r"""
    Return the Blackman window.

    The Blackman window is a taper formed by using the first three terms of a
    summation of cosines. It was designed to have close to the minimal leakage
    possible. It is close to optimal, only slightly worse than a Kaiser window.

    For full documentation refer to :obj:`numpy.blackman`.

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
    out : dpnp.ndarray of shape (M,)
        The window, with the maximum value normalized to one (the value one
        appears only if the number of samples is odd).

    See Also
    --------
    :obj:`dpnp.bartlett` : Return the Bartlett window.
    :obj:`dpnp.hamming` : Return the Hamming window.
    :obj:`dpnp.hanning` : Return the Hanning window.
    :obj:`dpnp.kaiser` : Return the Kaiser window.

    Notes
    -----
    The Blackman window is defined as

    .. math::  w(n) = 0.42 - 0.5\cos\left(\frac{2\pi{n}}{M-1}\right)
               + 0.08\cos\left(\frac{4\pi{n}}{M-1}\right)
               \qquad 0 \leq n \leq M-1

    Examples
    --------
    >>> import dpnp as np
    >>> np.blackman(12)
    array([-1.38777878e-17,  3.26064346e-02,  1.59903635e-01,  4.14397981e-01,
            7.36045180e-01,  9.67046769e-01,  9.67046769e-01,  7.36045180e-01,
            4.14397981e-01,  1.59903635e-01,  3.26064346e-02, -1.38777878e-17])

    Creating the output array on a different device or with a
    specified usm_type:

    >>> x = np.blackman(3) # default case
    >>> x, x.device, x.usm_type
    (array([-1.38777878e-17,  1.00000000e+00, -1.38777878e-17]),
     Device(level_zero:gpu:0),
     'device')

    >>> y = np.blackman(3, device="cpu")
    >>> y, y.device, y.usm_type
    (array([-1.38777878e-17,  1.00000000e+00, -1.38777878e-17]),
     Device(opencl:cpu:0),
     'device')

    >>> z = np.blackman(3, usm_type="host")
    >>> z, z.device, z.usm_type
    (array([-1.38777878e-17,  1.00000000e+00, -1.38777878e-17]),
     Device(level_zero:gpu:0),
     'host')

    """

    return _call_window_kernel(
        M, wi._blackman, device=device, usm_type=usm_type, sycl_queue=sycl_queue
    )


def hamming(M, *, device=None, usm_type=None, sycl_queue=None):
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
    out : dpnp.ndarray of shape (M,)
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

    .. math::  w(n) = 0.54 - 0.46\cos\left(\frac{2\pi{n}}{M-1}\right)
               \qquad 0 \leq n \leq M-1

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

    return _call_window_kernel(
        M, wi._hamming, device=device, usm_type=usm_type, sycl_queue=sycl_queue
    )


def hanning(M, *, device=None, usm_type=None, sycl_queue=None):
    r"""
    Return the Hanning window.

    The Hanning window is a taper formed by using a weighted cosine.

    For full documentation refer to :obj:`numpy.hanning`.

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
    out : dpnp.ndarray of shape (M,)
        The window, with the maximum value normalized to one (the value one
        appears only if the number of samples is odd).

    See Also
    --------
    :obj:`dpnp.bartlett` : Return the Bartlett window.
    :obj:`dpnp.blackman` : Return the Blackman window.
    :obj:`dpnp.hamming` : Return the Hamming window.
    :obj:`dpnp.kaiser` : Return the Kaiser window.

    Notes
    -----
    The Hanning window is defined as

    .. math::  w(n) = 0.5 - 0.5\cos\left(\frac{2\pi{n}}{M-1}\right)
               \qquad 0 \leq n \leq M-1

    Examples
    --------
    >>> import dpnp as np
    >>> np.hanning(12)
    array([0.        , 0.07937323, 0.29229249, 0.57115742, 0.82743037,
           0.97974649, 0.97974649, 0.82743037, 0.57115742, 0.29229249,
           0.07937323, 0.        ])

    Creating the output array on a different device or with a
    specified usm_type:

    >>> x = np.hanning(4) # default case
    >>> x, x.device, x.usm_type
    (array([0.  , 0.75, 0.75, 0.  ]), Device(level_zero:gpu:0), 'device')

    >>> y = np.hanning(4, device="cpu")
    >>> y, y.device, y.usm_type
    (array([0.  , 0.75, 0.75, 0.  ]), Device(opencl:cpu:0), 'device')

    >>> z = np.hanning(4, usm_type="host")
    >>> z, z.device, z.usm_type
    (array([0.  , 0.75, 0.75, 0.  ]), Device(level_zero:gpu:0), 'host')

    """

    return _call_window_kernel(
        M, wi._hanning, device=device, usm_type=usm_type, sycl_queue=sycl_queue
    )


def kaiser(M, beta, *, device=None, usm_type=None, sycl_queue=None):
    r"""
    Return the Kaiser window.

    The Kaiser window is a taper formed by using a Bessel function.

    For full documentation refer to :obj:`numpy.kaiser`.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty array
        is returned.
    beta : float
        Shape parameter for window.
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
    out : dpnp.ndarray of shape (M,)
        The window, with the maximum value normalized to one (the value one
        appears only if the number of samples is odd).

    See Also
    --------
    :obj:`dpnp.bartlett` : Return the Bartlett window.
    :obj:`dpnp.blackman` : Return the Blackman window.
    :obj:`dpnp.hamming` : Return the Hamming window.
    :obj:`dpnp.hanning` : Return the Hanning window.

    Notes
    -----
    The Kaiser window is defined as

    .. math::  w(n) = I_0\left( \beta \sqrt{1-\frac{4n^2}{(M-1)^2}}
               \right)/I_0(\beta)

    with

    .. math:: \quad -\frac{M-1}{2} \leq n \leq \frac{M-1}{2},

    where :math:`I_0` is the modified zeroth-order Bessel function.

    The Kaiser can approximate many other windows by varying the beta
    parameter.

    ====  =======================
    beta  Window shape
    ====  =======================
    0     Rectangular
    5     Similar to a Hamming
    6     Similar to a Hanning
    8.6   Similar to a Blackman
    ====  =======================

    A beta value of ``14`` is probably a good starting point. Note that as beta
    gets large, the window narrows, and so the number of samples needs to be
    large enough to sample the increasingly narrow spike, otherwise NaNs will
    get returned.

    Examples
    --------
    >>> import dpnp as np
    >>> np.kaiser(12, 14)
    array([7.72686638e-06, 3.46009173e-03, 4.65200161e-02, 2.29737107e-01,
           5.99885281e-01, 9.45674843e-01, 9.45674843e-01, 5.99885281e-01,
           2.29737107e-01, 4.65200161e-02, 3.46009173e-03, 7.72686638e-06])

    Creating the output array on a different device or with a
    specified usm_type:

    >>> x = np.kaiser(3, 14) # default case
    >>> x, x.device, x.usm_type
    (array([7.72686638e-06, 9.99999941e-01, 7.72686638e-06]),
     Device(level_zero:gpu:0),
     'device')

    >>> y = np.kaiser(3, 14, device="cpu")
    >>> y, y.device, y.usm_type
    (array([7.72686638e-06, 9.99999941e-01, 7.72686638e-06]),
     Device(opencl:cpu:0),
     'device')

    >>> z = np.kaiser(3, 14, usm_type="host")
    >>> z, z.device, z.usm_type
    (array([7.72686638e-06, 9.99999941e-01, 7.72686638e-06]),
     Device(level_zero:gpu:0),
     'host')

    """

    try:
        beta = float(beta)
    except Exception as e:
        raise TypeError("beta must be a float") from e

    return _call_window_kernel(
        M,
        wi._kaiser,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
        beta=beta,
    )
