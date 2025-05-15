# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2025, Intel Corporation
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
Interface of the Discrete Fourier Transform part of the DPNP

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions

"""

import dpnp

from .dpnp_utils_fft import dpnp_fft, dpnp_fftn, dpnp_fillfreq, swap_direction

__all__ = [
    "fft",
    "fft2",
    "fftfreq",
    "fftn",
    "fftshift",
    "hfft",
    "ifft",
    "ifft2",
    "ifftn",
    "ifftshift",
    "ihfft",
    "irfft",
    "irfft2",
    "irfftn",
    "rfft",
    "rfft2",
    "rfftfreq",
    "rfftn",
]


def fft(a, n=None, axis=-1, norm=None, out=None):
    """
    Compute the one-dimensional discrete Fourier Transform.

    This function computes the one-dimensional *n*-point discrete Fourier
    Transform (DFT) with the efficient Fast Fourier Transform (FFT) algorithm.

    For full documentation refer to :obj:`numpy.fft.fft`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array, can be complex.
    n : {None, int}, optional
        Length of the transformed axis of the output.
        If `n` is smaller than the length of the input, the input is cropped.
        If it is larger, the input is padded with zeros. If `n` is not given,
        the length of the input along the axis specified by `axis` is used.

        Default: ``None``.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last axis is
        used.

        Default: ``-1``.
    norm : {None, "backward", "ortho", "forward"}, optional
        Normalization mode (see :obj:`dpnp.fft`).
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor. ``None`` is an alias of
        the default option ``"backward"``.

        Default: ``"backward"``.
    out : {None, dpnp.ndarray or usm_ndarray of complex dtype}, optional
        If provided, the result will be placed in this array. It should be of
        the appropriate shape (consistent with the choice of `n`) and dtype.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray of complex dtype
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.

    See Also
    --------
    :obj:`dpnp.fft` : For definition of the DFT and conventions used.
    :obj:`dpnp.fft.ifft` : The inverse of :obj:`dpnp.fft.fft`.
    :obj:`dpnp.fft.fft2` : The two-dimensional FFT.
    :obj:`dpnp.fft.fftn` : The *N*-dimensional FFT.
    :obj:`dpnp.fft.rfftn` : The *N*-dimensional FFT of real input.
    :obj:`dpnp.fft.fftfreq` : Frequency bins for given FFT parameters.

    Notes
    -----
    FFT (Fast Fourier Transform) refers to a way the discrete Fourier
    Transform (DFT) can be calculated efficiently, by using symmetries in the
    calculated terms. The symmetry is highest when `n` is a power of 2, and
    the transform is therefore most efficient for these sizes.

    The DFT is defined, with the conventions used in this implementation,
    in the documentation for the :obj:`dpnp.fft` module.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.exp(2j * np.pi * np.arange(8) / 8)
    >>> np.fft.fft(a)
    array([-3.44509285e-16+1.14423775e-17j,  8.00000000e+00-8.52069395e-16j,
            2.33486982e-16+1.22464680e-16j,  0.00000000e+00+1.22464680e-16j,
            9.95799250e-17+2.33486982e-16j, -8.88178420e-16+1.17281316e-16j,
            1.14423775e-17+1.22464680e-16j,  0.00000000e+00+1.22464680e-16j])

    """

    dpnp.check_supported_arrays_type(a)
    return dpnp_fft(
        a, forward=True, real=False, n=n, axis=axis, norm=norm, out=out
    )


def fft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    """
    Compute the 2-dimensional discrete Fourier Transform.

    This function computes the *N*-dimensional discrete Fourier Transform over
    any axes in an *M*-dimensional array by means of the Fast Fourier
    Transform (FFT). By default, the transform is computed over the last two
    axes of the input array, i.e., a 2-dimensional FFT.

    For full documentation refer to :obj:`numpy.fft.fft2`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array, can be complex.
    s : {None, sequence of ints}, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        This corresponds to `n` for ``fft(x, n)``.
        Along each axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.
        If it is ``-1``, the whole input is used (no padding/trimming).
        If `s` is not given, the shape of the input along the axes specified
        by `axes` is used. If `s` is not ``None``, `axes` must not be ``None``
        either.

        Default: ``None``.
    axes : {None, sequence of ints}, optional
        Axes over which to compute the FFT. If not given, the last two axes are
        used. A repeated index in `axes` means the transform over that axis is
        performed multiple times. If `s` is specified, the corresponding `axes`
        to be transformed must be explicitly specified too. A one-element
        sequence means that a one-dimensional FFT is performed. An empty
        sequence means that no FFT is performed.

        Default: ``(-2, -1)``.
    norm : {None, "backward", "ortho", "forward"}, optional
        Normalization mode (see :obj:`dpnp.fft`).
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor. ``None`` is an alias of
        the default option ``"backward"``.

        Default: ``"backward"``.
    out : {None, dpnp.ndarray or usm_ndarray of complex dtype}, optional
        If provided, the result will be placed in this array. It should be of
        the appropriate shape (consistent with the choice of `s`) and dtype.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray of complex dtype
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or the last two axes if `axes` is not given.

    See Also
    --------
    :obj:`dpnp.fft` : Overall view of discrete Fourier transforms, with
        definitions and conventions used.
    :obj:`dpnp.fft.ifft2` : The inverse two-dimensional FFT.
    :obj:`dpnp.fft.fft` : The one-dimensional FFT.
    :obj:`dpnp.fft.fftn` : The *N*-dimensional FFT.
    :obj:`dpnp.fft.fftshift` : Shifts zero-frequency terms to the center of
        the array. For two-dimensional input, swaps first and third quadrants,
        and second and fourth quadrants.

    Notes
    -----
    :obj:`dpnp.fft.fft2` is just :obj:`dpnp.fft.fftn` with a different
    default for `axes`.

    The output, analogously to :obj:`dpnp.fft.fft`, contains the term for zero
    frequency in the low-order corner of the transformed axes, the positive
    frequency terms in the first half of these axes, the term for the Nyquist
    frequency in the middle of the axes and the negative frequency terms in
    the second half of the axes, in order of decreasingly negative frequency.

    See :obj:`dpnp.fft` for details, definitions and conventions used.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.mgrid[:5, :5][0]
    >>> np.fft.fft2(a)
    array([[ 50.  +0.j        ,   0.  +0.j        ,   0.  +0.j        ,
              0.  +0.j        ,   0.  +0.j        ],
           [-12.5+17.20477401j,   0.  +0.j        ,   0.  +0.j        ,
              0.  +0.j        ,   0.  +0.j        ],
           [-12.5 +4.0614962j ,   0.  +0.j        ,   0.  +0.j        ,
              0.  +0.j        ,   0.  +0.j        ],
           [-12.5 -4.0614962j ,   0.  +0.j        ,   0.  +0.j        ,
              0.  +0.j        ,   0.  +0.j        ],
           [-12.5-17.20477401j,   0.  +0.j        ,   0.  +0.j        ,
              0.  +0.j        ,   0.  +0.j        ]])  # may vary

    """

    dpnp.check_supported_arrays_type(a)
    return dpnp_fftn(
        a, forward=True, real=False, s=s, axes=axes, norm=norm, out=out
    )


def fftfreq(
    n, /, d=1.0, *, dtype=None, device=None, usm_type=None, sycl_queue=None
):
    """
    Return the Discrete Fourier Transform sample frequencies.

    The returned float array `f` contains the frequency bin centers in cycles
    per unit of the sample spacing (with zero at the start). For instance, if
    the sample spacing is in seconds, then the frequency unit is cycles/second.

    Given a window length `n` and a sample spacing `d`::

      f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
      f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd

    For full documentation refer to :obj:`numpy.fft.fftfreq`.

    Parameters
    ----------
    n : int
        Window length.
    d : scalar, optional
        Sample spacing (inverse of the sampling rate).

        Default: ``1.0``.
    dtype : {None, str, dtype object}, optional
        The output array data type. Must be a real-valued floating-point data
        type. If `dtype` is ``None``, the output array data type must be the
        default real-valued floating-point data type.

        Default: ``None``.
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
    f : dpnp.ndarray
        Array of length `n` containing the sample frequencies.

    See Also
    --------
    :obj:`dpnp.fft.rfftfreq` : Return the Discrete Fourier Transform sample
                        frequencies (for usage with :obj:`dpnp.fft.rfft` and
                        :obj:`dpnp.fft.irfft`).

    Examples
    --------
    >>> import dpnp as np
    >>> signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5])
    >>> fourier = np.fft.fft(signal)
    >>> n = signal.size
    >>> timestep = 0.1
    >>> freq = np.fft.fftfreq(n, d=timestep)
    >>> freq
    array([ 0.  ,  1.25,  2.5 ,  3.75, -5.  , -3.75, -2.5 , -1.25])

    Creating the output array on a different device or with a
    specified usm_type:

    >>> x = np.fft.fftfreq(n, d=timestep) # default case
    >>> x.shape, x.device, x.usm_type
    ((8,), Device(level_zero:gpu:0), 'device')

    >>> y = np.fft.fftfreq(n, d=timestep, device="cpu")
    >>> y.shape, y.device, y.usm_type
    ((8,), Device(opencl:cpu:0), 'device')

    >>> z = np.fft.fftfreq(n, d=timestep, usm_type="host")
    >>> z.shape, z.device, z.usm_type
    ((8,), Device(level_zero:gpu:0), 'host')

    """

    if not isinstance(n, int):
        raise ValueError("`n` should be an integer")
    if not dpnp.isscalar(d):
        raise ValueError("`d` should be an scalar")

    if dtype and not dpnp.issubdtype(dtype, dpnp.floating):
        raise ValueError(
            "dtype must a real-valued floating-point data type, "
            f"but got {dtype}"
        )

    val = 1.0 / (n * d)
    results = dpnp.empty(
        n, dtype=dtype, device=device, usm_type=usm_type, sycl_queue=sycl_queue
    )

    m = (n - 1) // 2 + 1
    return dpnp_fillfreq(results, m, n, val)


def fftn(a, s=None, axes=None, norm=None, out=None):
    """
    Compute the *N*-dimensional discrete Fourier Transform.

    This function computes the *N*-dimensional discrete Fourier Transform over
    any number of axes in an *M*-dimensional array by means of the
    Fast Fourier Transform (FFT).

    For full documentation refer to :obj:`numpy.fft.fftn`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array, can be complex.
    s : {None, sequence of ints}, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        This corresponds to `n` for ``fft(x, n)``.
        Along each axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.
        If it is ``-1``, the whole input is used (no padding/trimming).
        If `s` is not given, the shape of the input along the axes specified
        by `axes` is used. If `s` is not ``None``, `axes` must not be ``None``
        either.

        Default: ``None``.
    axes : {None, sequence of ints}, optional
        Axes over which to compute the FFT. If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
        Repeated indices in `axes` means that the transform over that axis is
        performed multiple times. If `s` is specified, the corresponding `axes`
        to be transformed must be explicitly specified too. A one-element
        sequence means that a one-dimensional FFT is performed. An empty
        sequence means that no FFT is performed.

        Default: ``None``.
    norm : {None, "backward", "ortho", "forward"}, optional
        Normalization mode (see :obj:`dpnp.fft`).
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor. ``None`` is an alias of
        the default option ``"backward"``.

        Default: ``"backward"``.
    out : {None, dpnp.ndarray or usm_ndarray of complex dtype}, optional
        If provided, the result will be placed in this array. It should be of
        the appropriate shape (consistent with the choice of `s`) and dtype.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray of complex dtype
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` and `a`,
        as explained in the parameters section above.

    See Also
    --------
    :obj:`dpnp.fft` : Overall view of discrete Fourier transforms, with
        definitions and conventions used.
    :obj:`dpnp.fft.ifftn` : The inverse *N*-dimensional FFT.
    :obj:`dpnp.fft.fft` : The one-dimensional FFT.
    :obj:`dpnp.fft.rfftn` : The *N*-dimensional FFT of real input.
    :obj:`dpnp.fft.fft2` : The two-dimensional FFT.
    :obj:`dpnp.fft.fftshift` : Shifts zero-frequency terms to the center of
        the array.

    Notes
    -----
    The output, analogously to :obj:`dpnp.fft.fft`, contains the term for zero
    frequency in the low-order corner of the transformed axes, the positive
    frequency terms in the first half of these axes, the term for the Nyquist
    frequency in the middle of the axes and the negative frequency terms in
    the second half of the axes, in order of decreasingly negative frequency.

    See :obj:`dpnp.fft` for details, definitions and conventions used.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.mgrid[:3, :3, :3][0]
    >>> np.fft.fftn(a, axes=(1, 2))
    array([[[ 0.+0.j,   0.+0.j,   0.+0.j], # may vary
            [ 0.+0.j,   0.+0.j,   0.+0.j],
            [ 0.+0.j,   0.+0.j,   0.+0.j]],
           [[ 9.+0.j,   0.+0.j,   0.+0.j],
            [ 0.+0.j,   0.+0.j,   0.+0.j],
            [ 0.+0.j,   0.+0.j,   0.+0.j]],
           [[18.+0.j,   0.+0.j,   0.+0.j],
            [ 0.+0.j,   0.+0.j,   0.+0.j],
            [ 0.+0.j,   0.+0.j,   0.+0.j]]])

    >>> np.fft.fftn(a, (2, 2), axes=(0, 1))
    array([[[ 2.+0.j,  2.+0.j,  2.+0.j], # may vary
            [ 0.+0.j,  0.+0.j,  0.+0.j]],
           [[-2.+0.j, -2.+0.j, -2.+0.j],
            [ 0.+0.j,  0.+0.j,  0.+0.j]]])

    """

    dpnp.check_supported_arrays_type(a)
    return dpnp_fftn(
        a, forward=True, real=False, s=s, axes=axes, norm=norm, out=out
    )


def fftshift(x, axes=None):
    """
    Shift the zero-frequency component to the center of the spectrum.

    This function swaps half-spaces for all axes listed (defaults to all).
    Note that ``out[0]`` is the Nyquist component only if ``len(x)`` is even.

    For full documentation refer to :obj:`numpy.fft.fftshift`.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        Input array.
    axes : {None, int, list or tuple of ints}, optional
        Axes over which to shift. By default, it shifts over all axes.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        The shifted array.

    See Also
    --------
    :obj:`dpnp.fft.ifftshift` : The inverse of :obj:`dpnp.fft.fftshift`.

    Examples
    --------
    >>> import dpnp as np
    >>> freqs = np.fft.fftfreq(10, 0.1)
    >>> freqs
    array([ 0.,  1.,  2.,  3.,  4., -5., -4., -3., -2., -1.])
    >>> np.fft.fftshift(freqs)
    array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])

    Shift the zero-frequency component only along the second axis:

    >>> freqs = np.fft.fftfreq(9, d=1./9).reshape(3, 3)
    >>> freqs
    array([[ 0.,  1.,  2.],
           [ 3.,  4., -4.],
           [-3., -2., -1.]])
    >>> np.fft.fftshift(freqs, axes=(1,))
    array([[ 2.,  0.,  1.],
           [-4.,  3.,  4.],
           [-1., -3., -2.]])

    """

    dpnp.check_supported_arrays_type(x)
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, int):
        shift = x.shape[axes] // 2
    else:
        x_shape = x.shape
        shift = [x_shape[ax] // 2 for ax in axes]

    return dpnp.roll(x, shift, axes)


def hfft(a, n=None, axis=-1, norm=None, out=None):
    """
    Compute the FFT of a signal that has Hermitian symmetry, i.e.,
    a real spectrum.

    For full documentation refer to :obj:`numpy.fft.hfft`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    n : {None, int}, optional
        Length of the transformed axis of the output.
        For `n` output points, ``n//2+1`` input points are necessary. If the
        input is longer than this, it is cropped. If it is shorter than this,
        it is padded with zeros. If `n` is not given, it is taken to be
        ``2*(m-1)`` where `m` is the length of the input along the axis
        specified by `axis`.

        Default: ``None``.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last axis is
        used.

        Default: ``-1``.
    norm : {None, "backward", "ortho", "forward"}, optional
        Normalization mode (see :obj:`dpnp.fft`).
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor. ``None`` is an alias of
        the default option ``"backward"``.

        Default: ``"backward"``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        If provided, the result will be placed in this array. It should be
        of the appropriate shape and dtype.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.
        The length of the transformed axis is `n`, or, if `n` is not given,
        ``2*(m-1)`` where `m` is the length of the transformed axis of the
        input. To get an odd number of output points, `n` must be specified,
        for instance as ``2*m - 1`` in the typical case.

    See Also
    --------
    :obj:`dpnp.fft` : For definition of the DFT and conventions used.
    :obj:`dpnp.fft.rfft` : The one-dimensional FFT of real input.
    :obj:`dpnp.fft.ihfft` :The inverse of :obj:`dpnp.fft.hfft`.


    Notes
    -----
    :obj:`dpnp.fft.hfft`/:obj:`dpnp.fft.ihfft` are a pair analogous to
    :obj:`dpnp.fft.rfft`/:obj:`dpnp.fft.irfft`, but for the opposite case:
    here the signal has Hermitian symmetry in the time domain and is real in
    the frequency domain. So here it's :obj:`dpnp.fft.hfft` for which you must
    supply the length of the result if it is to be odd.

    * even: ``ihfft(hfft(a, 2*len(a) - 2)) == a``, within round-off error,
    * odd: ``ihfft(hfft(a, 2*len(a) - 1)) == a``, within round-off error.

    The correct interpretation of the Hermitian input depends on the length of
    the original data, as given by `n`. This is because each input shape could
    correspond to either an odd or even length signal. By default,
    :obj:`dpnp.fft.hfft` assumes an even output length which puts the last
    entry at the Nyquist frequency; aliasing with its symmetric counterpart.
    By Hermitian symmetry, the value is thus treated as purely real. To avoid
    losing information, the correct length of the real input **must** be given.

    Examples
    --------
    >>> import dpnp as np
    >>> signal = np.array([1, 2, 3, 4, 3, 2])
    >>> np.fft.fft(signal)
    array([15.+0.j, -4.+0.j,  0.+0.j, -1.-0.j,  0.+0.j, -4.+0.j])
    >>> np.fft.hfft(signal[:4]) # Input first half of signal
    array([15., -4.,  0., -1.,  0., -4.])
    >>> np.fft.hfft(signal, 6)  # Input entire signal and truncate
    array([15., -4.,  0., -1.,  0., -4.])

    >>> signal = np.array([[1, 1.j], [-1.j, 2]])
    >>> np.conj(signal.T) - signal   # check Hermitian symmetry
    array([[ 0.-0.j, -0.+0.j], # may vary
           [ 0.+0.j,  0.-0.j]])
    >>> freq_spectrum = np.fft.hfft(signal)
    >>> freq_spectrum
    array([[ 1.,  1.],
           [ 2., -2.]])

    """

    new_norm = swap_direction(norm)
    return irfft(dpnp.conjugate(a), n=n, axis=axis, norm=new_norm, out=out)


def ifft(a, n=None, axis=-1, norm=None, out=None):
    """
    Compute the one-dimensional inverse discrete Fourier Transform.

    This function computes the inverse of the one-dimensional *n*-point
    discrete Fourier transform computed by :obj:`dpnp.fft.fft`. In other words,
    ``ifft(fft(a)) == a`` to within numerical accuracy.
    For a general description of the algorithm and definitions,
    see :obj:`dpnp.fft`.

    The input should be ordered in the same way as is returned by
    :obj:`dpnp.fft.fft`, i.e.,

    * ``a[0]`` should contain the zero frequency term,
    * ``a[1:(n+1)//2]`` should contain the positive-frequency terms,
    * ``a[n//2 + 1:]`` should contain the negative-frequency terms, in
      increasing order starting from the most negative frequency.

    For an even number of input points, ``a[n//2]`` represents the sum of
    the values at the positive and negative Nyquist frequencies, as the two
    are aliased together.

    For full documentation refer to :obj:`numpy.fft.ifft`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array, can be complex.
    n : {None, int}, optional
        Length of the transformed axis of the output.
        If `n` is smaller than the length of the input, the input is cropped.
        If it is larger, the input is padded with zeros. If `n` is not given,
        the length of the input along the axis specified by `axis` is used.

        Default: ``None``.
    axis : int, optional
        Axis over which to compute the inverse FFT. If not given, the last
        axis is used.

        Default: ``-1``.
    norm : {None, "backward", "ortho", "forward"}, optional
        Normalization mode (see :obj:`dpnp.fft`).
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor. ``None`` is an alias of
        the default option ``"backward"``.

        Default: ``"backward"``.
    out : {None, dpnp.ndarray or usm_ndarray of complex dtype}, optional
        If provided, the result will be placed in this array. It should be of
        the appropriate shape (consistent with the choice of `n`) and dtype.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray of complex dtype
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.

    See Also
    --------
    :obj:`dpnp.fft` : For definition of the DFT and conventions used.
    :obj:`dpnp.fft.fft` : The one-dimensional (forward) FFT,
                          of which :obj:`dpnp.fft.ifft` is the inverse.
    :obj:`dpnp.fft.ifft2` : The two-dimensional inverse FFT.
    :obj:`dpnp.fft.ifftn` : The *N*-dimensional inverse FFT.

    Notes
    -----
    If the input parameter `n` is larger than the size of the input, the input
    is padded by appending zeros at the end. Even though this is the common
    approach, it might lead to surprising results. If a different padding is
    desired, it must be performed before calling :obj:`dpnp.fft.ifft`.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([0, 4, 0, 0])
    >>> np.fft.ifft(a)
    array([ 1.+0.j,  0.+1.j, -1.+0.j,  0.-1.j]) # may vary

    """

    dpnp.check_supported_arrays_type(a)
    return dpnp_fft(
        a, forward=False, real=False, n=n, axis=axis, norm=norm, out=out
    )


def ifft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    """
    Compute the 2-dimensional inverse discrete Fourier Transform.

    This function computes the inverse of the 2-dimensional discrete Fourier
    Transform over any number of axes in an *M*-dimensional array by means of
    the Fast Fourier Transform (FFT). In other words, ``ifft2(fft2(a)) == a``
    to within numerical accuracy. By default, the inverse transform is
    computed over the last two axes of the input array.

    The input, analogously to :obj:`dpnp.fft.ifft`, should be ordered in the
    same way as is returned by :obj:`dpnp.fft.fft2`, i.e. it should have the
    term for zero frequency in the low-order corner of the two axes, the
    positive frequency terms in the first half of these axes, the term for the
    Nyquist frequency in the middle of the axes and the negative frequency
    terms in the second half of both axes, in order of decreasingly negative
    frequency.

    For full documentation refer to :obj:`numpy.fft.ifft2`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array, can be complex.
    s : {None, sequence of ints}, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        This corresponds to `n` for ``ifft(x, n)``.
        Along each axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.
        If it is ``-1``, the whole input is used (no padding/trimming).
        If `s` is not given, the shape of the input along the axes specified
        by `axes` is used. See notes for issue on :obj:`dpnp.fft.ifft`
        zero padding. If `s` is not ``None``, `axes` must not be ``None``
        either.

        Default: ``None``.
    axes : {None, sequence of ints}, optional
        Axes over which to compute the inverse FFT. If not given, the last two
        axes are used. A repeated index in `axes` means the transform over that
        axis is performed multiple times. If `s` is specified, the
        corresponding `axes` to be transformed must be explicitly specified
        too. A one-element sequence means that a one-dimensional FFT is
        performed. An empty sequence means that no FFT is performed.

        Default: ``(-2, -1)``.
    norm : {None, "backward", "ortho", "forward"}, optional
        Normalization mode (see :obj:`dpnp.fft`).
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor. ``None`` is an alias of
        the default option ``"backward"``.

        Default: ``"backward"``.
    out : {None, dpnp.ndarray or usm_ndarray of complex dtype}, optional
        If provided, the result will be placed in this array. It should be of
        the appropriate shape (consistent with the choice of `s`) and dtype.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray of complex dtype
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or the last two axes if `axes` is not given.

    See Also
    --------
    :obj:`dpnp.fft` : Overall view of discrete Fourier transforms, with
        definitions and conventions used.
    :obj:`dpnp.fft.fft2` : The forward two-dimensional FFT, of which
        :obj:`dpnp.fft.ifft2` is the inverse.
    :obj:`dpnp.fft.ifftn` : The inverse of *N*-dimensional FFT.
    :obj:`dpnp.fft.fft` : The one-dimensional FFT.
    :obj:`dpnp.fft.ifft` : The one-dimensional inverse FFT.

    Notes
    -----
    :obj:`dpnp.fft.ifft2` is just :obj:`dpnp.fft.ifftn` with a different
    default for `axes`. See :obj:`dpnp.fft` for details, definitions and
    conventions used.

    Zero-padding, analogously with :obj:`dpnp.fft.ifft`, is performed by
    appending zeros to the input along the specified dimension. Although this
    is the common approach, it might lead to surprising results. If another
    form of zero padding is desired, it must be performed before
    :obj:`dpnp.fft.ifft2` is called.

    Examples
    --------
    >>> import dpnp as np
    >>> a = 4 * np.eye(4)
    >>> np.fft.ifft2(a)
    array([[1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j], # may vary
           [0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j],
           [0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
           [0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j]])

    """

    dpnp.check_supported_arrays_type(a)
    return dpnp_fftn(
        a, forward=False, real=False, s=s, axes=axes, norm=norm, out=out
    )


def ifftn(a, s=None, axes=None, norm=None, out=None):
    """
    Compute the *N*-dimensional inverse discrete Fourier Transform.

    This function computes the inverse of the *N*-dimensional discrete
    Fourier Transform over any number of axes in an *M*-dimensional array by
    means of the Fast Fourier Transform (FFT). In other words,
    ``ifftn(fftn(a)) == a`` to within numerical accuracy. For a description
    of the definitions and conventions used, see :obj:`dpnp.fft`.

    The input, analogously to :obj:`dpnp.fft.ifft`, should be ordered in the
    same way as is returned by :obj:`dpnp.fft.fftn`, i.e. it should have the
    term for zero frequency in all axes in the low-order corner, the positive
    frequency terms in the first half of all axes, the term for the Nyquist
    frequency in the middle of all axes and the negative frequency terms in
    the second half of all axes, in order of decreasingly negative frequency.

    For full documentation refer to :obj:`numpy.fft.ifftn`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array, can be complex.
    s : {None, sequence of ints}, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        This corresponds to `n` for ``ifft(x, n)``.
        Along each axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.
        If it is ``-1``, the whole input is used (no padding/trimming).
        if `s` is not given, the shape of the input along the axes specified
        by `axes` is used. If `s` is not ``None``, `axes` must not be ``None``
        either.

        Default: ``None``.
    axes : {None, sequence of ints}, optional
        Axes over which to compute the inverse FFT. If not given, the last
        ``len(s)`` axes are used, or all axes if `s` is also not specified.
        Repeated indices in `axes` means that the transform over that axis is
        performed multiple times. If `s` is specified, the corresponding `axes`
        to be transformed must be explicitly specified too. A one-element
        sequence means that a one-dimensional FFT is performed. An empty
        sequence means that no FFT is performed.

        Default: ``None``.
    norm : {None, "backward", "ortho", "forward"}, optional
        Normalization mode (see :obj:`dpnp.fft`).
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor. ``None`` is an alias of
        the default option ``"backward"``.

        Default: ``"backward"``.
    out : {None, dpnp.ndarray or usm_ndarray of complex dtype}, optional
        If provided, the result will be placed in this array. It should be of
        the appropriate shape (consistent with the choice of `s`) and dtype.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray of complex dtype
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` and `a`,
        as explained in the parameters section above.

    See Also
    --------
    :obj:`dpnp.fft` : Overall view of discrete Fourier transforms, with
        definitions and conventions used.
    :obj:`dpnp.fft.fftn` : The *N*-dimensional FFT.
    :obj:`dpnp.fft.ifft` : The one-dimensional inverse FFT.
    :obj:`dpnp.fft.ifft2` : The two-dimensional inverse FFT.
    :obj:`dpnp.fft.ifftshift` : Undoes :obj:`dpnp.fft.fftshift`, shifts
        zero-frequency terms to the center of the array.

    Notes
    -----
    See :obj:`dpnp.fft` for details, definitions and conventions used.

    Zero-padding, analogously with :obj:`dpnp.fft.ifft`, is performed by
    appending zeros to the input along the specified dimension. Although this
    is the common approach, it might lead to surprising results. If another
    form of zero padding is desired, it must be performed before
    :obj:`dpnp.fft.ifftn` is called.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.eye(4)
    >>> np.fft.ifftn(np.fft.fftn(a, axes=(0,)), axes=(1,))
    array([[1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j], # may vary
           [0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
           [0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
           [0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j]])

    """

    dpnp.check_supported_arrays_type(a)
    return dpnp_fftn(
        a, forward=False, real=False, s=s, axes=axes, norm=norm, out=out
    )


def ifftshift(x, axes=None):
    """
    Inverse shift the zero-frequency component to the center of the spectrum.

    Although identical for even-length `x`, the functions differ by one sample
    for odd-length `x`.

    For full documentation refer to :obj:`numpy.fft.ifftshift`.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        Input array.
    axes : {None, int, list or tuple of ints}, optional
        Axes over which to shift. By default, it shifts over all axes.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        The shifted array.

    See Also
    --------
    :obj:`dpnp.fft.fftshift` : Shift zero-frequency component to the center
                of the spectrum.

    Examples
    --------
    >>> import dpnp as np
    >>> freqs = np.fft.fftfreq(9, d=1./9).reshape(3, 3)
    >>> freqs
    array([[ 0.,  1.,  2.],
           [ 3.,  4., -4.],
           [-3., -2., -1.]])
    >>> np.fft.ifftshift(np.fft.fftshift(freqs))
    array([[ 0.,  1.,  2.],
           [ 3.,  4., -4.],
           [-3., -2., -1.]])

    """

    dpnp.check_supported_arrays_type(x)
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, int):
        shift = -(x.shape[axes] // 2)
    else:
        x_shape = x.shape
        shift = [-(x_shape[ax] // 2) for ax in axes]

    return dpnp.roll(x, shift, axes)


def ihfft(a, n=None, axis=-1, norm=None, out=None):
    """
    Compute the inverse FFT of a signal that has Hermitian symmetry.

    For full documentation refer to :obj:`numpy.fft.ihfft`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    n : {None, int}, optional
        Length of the inverse FFT, the number of points along
        transformation axis in the input to use. If `n` is smaller than
        the length of the input, the input is cropped. If it is larger,
        the input is padded with zeros. If `n` is not given, the length of
        the input along the axis specified by `axis` is used.

        Default: ``None``.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last axis is
        used.

        Default: ``-1``.
    norm : {None, "backward", "ortho", "forward"}, optional
        Normalization mode (see :obj:`dpnp.fft`).
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor. ``None`` is an alias of
        the default option ``"backward"``.

        Default: ``"backward"``.
    out : {None, dpnp.ndarray or usm_ndarray of complex dtype}, optional
        If provided, the result will be placed in this array. It should be
        of the appropriate shape and dtype.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray of complex dtype
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.
        The length of the transformed axis is ``n//2 + 1``.

    See Also
    --------
    :obj:`dpnp.fft` : For definition of the DFT and conventions used.
    :obj:`dpnp.fft.hfft` : Compute the FFT of a signal that has
                Hermitian symmetry.
    :obj:`dpnp.fft.irfft` : The inverse of :obj:`dpnp.fft.rfft`.

    Notes
    -----
    :obj:`dpnp.fft.hfft`/:obj:`dpnp.fft.ihfft` are a pair analogous to
    :obj:`dpnp.fft.rfft`/:obj:`dpnp.fft.irfft`, but for the opposite case:
    here the signal has Hermitian symmetry in the time domain and is real in
    the frequency domain. So here it's :obj:`dpnp.fft.hfft` for which you must
    supply the length of the result if it is to be odd.

    * even: ``ihfft(hfft(a, 2*len(a) - 2)) == a``, within round-off error,
    * odd: ``ihfft(hfft(a, 2*len(a) - 1)) == a``, within round-off error.

    Examples
    --------
    >>> import dpnp as np
    >>> spectrum = np.array([ 15, -4, 0, -1, 0, -4])
    >>> np.fft.ifft(spectrum)
    array([1.+0.j, 2.+0.j, 3.+0.j, 4.+0.j, 3.+0.j, 2.+0.j]) # may vary
    >>> np.fft.ihfft(spectrum)
    array([1.-0.j, 2.-0.j, 3.-0.j, 4.-0.j]) # may vary

    """

    new_norm = swap_direction(norm)
    res = rfft(a, n=n, axis=axis, norm=new_norm, out=out)
    return dpnp.conjugate(res, out=out)


def irfft(a, n=None, axis=-1, norm=None, out=None):
    """
    Computes the inverse of :obj:`dpnp.fft.rfft`.

    This function computes the inverse of the one-dimensional *n*-point
    discrete Fourier Transform of real input computed by :obj:`dpnp.fft.rfft`.
    In other words, ``irfft(rfft(a), n=len(a)) == a`` to within numerical
    accuracy. (See Notes below for why ``len(a)`` is necessary here.)

    The input is expected to be in the form returned by :obj:`dpnp.fft.rfft`,
    i.e. the real zero-frequency term followed by the complex positive
    frequency terms in order of increasing frequency. Since the discrete
    Fourier Transform of real input is Hermitian-symmetric, the negative
    frequency terms are taken to be the complex conjugates of the corresponding
    positive frequency terms.

    For full documentation refer to :obj:`numpy.fft.irfft`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    n : {None, int}, optional
        Length of the transformed axis of the output.
        For `n` output points, ``n//2+1`` input points are necessary. If the
        input is longer than this, it is cropped. If it is shorter than this,
        it is padded with zeros. If `n` is not given, it is taken to be
        ``2*(m-1)`` where `m` is the length of the input along the axis
        specified by `axis`.

        Default: ``None``.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last axis is
        used.

        Default: ``-1``.
    norm : {None, "backward", "ortho", "forward"}, optional
        Normalization mode (see :obj:`dpnp.fft`).
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor. ``None`` is an alias of
        the default option ``"backward"``.

        Default: ``"backward"``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        If provided, the result will be placed in this array. It should be
        of the appropriate shape and dtype.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.
        The length of the transformed axis is `n`, or, if `n` is not given,
        ``2*(m-1)`` where `m` is the length of the transformed axis of the
        input. To get an odd number of output points, `n` must be specified.

    See Also
    --------
    :obj:`dpnp.fft` : For definition of the DFT and conventions used.
    :obj:`dpnp.fft.rfft` : The one-dimensional FFT of real input, of which
                        :obj:`dpnp.fft.irfft` is inverse.
    :obj:`dpnp.fft.fft` : The one-dimensional FFT of general (complex) input.
    :obj:`dpnp.fft.irfft2` :The inverse of the two-dimensional FFT of
                        real input.
    :obj:`dpnp.fft.irfftn` : The inverse of the *N*-dimensional FFT of
                        real input.

    Notes
    -----
    Returns the real valued *n*-point inverse discrete Fourier transform
    of `a`, where `a` contains the non-negative frequency terms of a
    Hermitian-symmetric sequence. `n` is the length of the result, not the
    input.

    If you specify an `n` such that `a` must be zero-padded or truncated, the
    extra/removed values will be added/removed at high frequencies. One can
    thus re-sample a series to `m` points via Fourier interpolation by:
    ``a_resamp = irfft(rfft(a), m)``.

    The correct interpretation of the Hermitian input depends on the length of
    the original data, as given by `n`. This is because each input shape could
    correspond to either an odd or even length signal. By default,
    :obj:`dpnp.fft.irfft` assumes an even output length which puts the last
    entry at the Nyquist frequency; aliasing with its symmetric counterpart.
    By Hermitian symmetry, the value is thus treated as purely real. To avoid
    losing information, the correct length of the real input **must** be given.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, -1j, -1, 1j])
    >>> np.fft.ifft(a)
    array([0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j]) # may vary
    >>> np.fft.irfft(a[:-1])
    array([0.,  1.,  0.,  0.])

    Notice how the last term in the input to the ordinary :obj:`dpnp.fft.ifft`
    is the complex conjugate of the second term, and the output has zero
    imaginary part everywhere. When calling :obj:`dpnp.fft.irfft`, the negative
    frequencies are not specified, and the output array is purely real.

    """

    dpnp.check_supported_arrays_type(a)
    return dpnp_fft(
        a, forward=False, real=True, n=n, axis=axis, norm=norm, out=out
    )


def irfft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    """
    Computes the inverse of :obj:`dpnp.fft.rfft2`.

    For full documentation refer to :obj:`numpy.fft.irfft2`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array, can be complex.
    s : {None, sequence of ints}, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.). `s` is also the
        number of input points used along this axis, except for the last axis,
        where ``s[-1]//2+1`` points of the input are used.
        Along each axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.
        If it is ``-1``, the whole input is used (no padding/trimming).
        If `s` is not given, the shape of the input along the axes
        specified by `axes` is used. Except for the last axis which is taken to
        be ``2*(m-1)`` where `m` is the length of the input along that axis.
        If `s` is not ``None``, `axes` must not be ``None`` either.

        Default: ``None``.
    axes : {None, sequence of ints}, optional
        Axes over which to compute the inverse FFT. If not given, the last
        ``len(s)`` axes are used, or all axes if `s` is also not specified.
        Repeated indices in `axes` means that the transform over that axis is
        performed multiple times. If `s` is specified, the corresponding `axes`
        to be transformed must be explicitly specified too. A one-element
        sequence means that a one-dimensional FFT is performed. An empty
        sequence means that no FFT is performed.

        Default: ``(-2, -1)``.
    norm : {None, "backward", "ortho", "forward"}, optional
        Normalization mode (see :obj:`dpnp.fft`).
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor. ``None`` is an alias of
        the default option ``"backward"``.

        Default: ``"backward"``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        If provided, the result will be placed in this array. It should be of
        the appropriate dtype and shape for the last transformation
        (consistent with the choice of `s`).

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or the last two axes if `axes` is not given.

    See Also
    --------
    :obj:`dpnp.fft` : Overall view of discrete Fourier transforms, with
        definitions and conventions used.
    :obj:`dpnp.fft.rfft2` : The forward two-dimensional FFT of real input,
                        of which :obj:`dpnp.fft.irfft2` is the inverse.
    :obj:`dpnp.fft.rfft` : The one-dimensional FFT for real input.
    :obj:`dpnp.fft.irfft` : The inverse of the one-dimensional FFT of
                        real input.
    :obj:`dpnp.fft.irfftn` : The inverse of the *N*-dimensional FFT of
                        real input.

    Notes
    -----
    :obj:`dpnp.fft.irfft2` is just :obj:`dpnp.fft.irfftn` with a different
    default for `axes`. For more details see :obj:`dpnp.fft.irfftn`.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.mgrid[:5, :5][0]
    >>> A = np.fft.rfft2(a)
    >>> np.fft.irfft2(A, s=a.shape)
    array([[0., 0., 0., 0., 0.],
           [1., 1., 1., 1., 1.],
           [2., 2., 2., 2., 2.],
           [3., 3., 3., 3., 3.],
           [4., 4., 4., 4., 4.]])

    """

    dpnp.check_supported_arrays_type(a)
    return dpnp_fftn(
        a, forward=False, real=True, s=s, axes=axes, norm=norm, out=out
    )


def irfftn(a, s=None, axes=None, norm=None, out=None):
    """
    Computes the inverse of :obj:`dpnp.fft.rfftn`.

    This function computes the inverse of the *N*-dimensional discrete Fourier
    Transform for real input over any number of axes in an *M*-dimensional
    array by means of the Fast Fourier Transform (FFT). In other words,
    ``irfftn(rfftn(a), s=a.shape, axes=range(a.ndim)) == a`` to within
    numerical accuracy. (The ``a.shape`` is necessary like ``len(a)`` is for
    :obj:`dpnp.fft.irfft`, and for the same reason.)

    The input should be ordered in the same way as is returned by
    :obj:`dpnp.fft.rfftn`, i.e. as for :obj:`dpnp.fft.irfft` for the final
    transformation axis, and as for :obj:`dpnp.fft.irfftn` along all the other
    axes.

    For full documentation refer to :obj:`numpy.fft.irfftn`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array, can be complex.
    s : {None, sequence of ints}, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.). `s` is also the
        number of input points used along this axis, except for the last axis,
        where ``s[-1]//2+1`` points of the input are used.
        Along each axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.
        If it is ``-1``, the whole input is used (no padding/trimming).
        If `s` is not given, the shape of the input along the axes
        specified by `axes` is used. Except for the last axis which is taken to
        be ``2*(m-1)`` where `m` is the length of the input along that axis.
        If `s` is not ``None``, `axes` must not be ``None`` either.

        Default: ``None``.
    axes : {None, sequence of ints}, optional
        Axes over which to compute the inverse FFT. If not given, the last
        ``len(s)`` axes are used, or all axes if `s` is also not specified.
        Repeated indices in `axes` means that the transform over that axis is
        performed multiple times. If `s` is specified, the corresponding `axes`
        to be transformed must be explicitly specified too. A one-element
        sequence means that a one-dimensional FFT is performed. An empty
        sequence means that no FFT is performed.

        Default: ``None``.
    norm : {None, "backward", "ortho", "forward"}, optional
        Normalization mode (see :obj:`dpnp.fft`).
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor. ``None`` is an alias of
        the default option ``"backward"``.

        Default: ``"backward"``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        If provided, the result will be placed in this array. It should be of
        the appropriate dtype and shape for the last transformation
        (consistent with the choice of `s`).

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` and `a`,
        as explained in the parameters section above.
        The length of each transformed axis is as given by the corresponding
        element of `s`, or the length of the input in every axis except for the
        last one if `s` is not given. In the final transformed axis the length
        of the output when `s` is not given is ``2*(m-1)`` where `m` is the
        length of the final transformed axis of the input. To get an odd
        number of output points in the final axis, `s` must be specified.

    See Also
    --------
    :obj:`dpnp.fft` : Overall view of discrete Fourier transforms, with
        definitions and conventions used.
    :obj:`dpnp.fft.rfftn` : The `n`-dimensional FFT of real input.
    :obj:`dpnp.fft.fft` : The one-dimensional FFT, with definitions and
                        conventions used.
    :obj:`dpnp.fft.irfft` : The inverse of the one-dimensional FFT of
                        real input.
    :obj:`dpnp.fft.irfft2` : The inverse of the two-dimensional FFT of
                        real input.

    Notes
    -----
    See :obj:`dpnp.fft` for details, definitions and conventions used.

    See :obj:`dpnp.fft.rfft` for definitions and conventions used for real
    input.

    The correct interpretation of the Hermitian input depends on the shape of
    the original data, as given by `s`. This is because each input shape could
    correspond to either an odd or even length signal. By default,
    :obj:`dpnp.fft.irfftn` assumes an even output length which puts the last
    entry at the Nyquist frequency; aliasing with its symmetric counterpart.
    When performing the final complex to real transform, the last value is thus
    treated as purely real. To avoid losing information, the correct shape of
    the real input **must** be given.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.zeros((3, 2, 2))
    >>> a[0, 0, 0] = 3 * 2 * 2
    >>> np.fft.irfftn(a)
    array([[[1.,  1.],
            [1.,  1.]],
           [[1.,  1.],
            [1.,  1.]],
           [[1.,  1.],
            [1.,  1.]]])

    """

    dpnp.check_supported_arrays_type(a)
    return dpnp_fftn(
        a, forward=False, real=True, s=s, axes=axes, norm=norm, out=out
    )


def rfft(a, n=None, axis=-1, norm=None, out=None):
    """
    Compute the one-dimensional discrete Fourier Transform for real input.

    This function computes the one-dimensional *n*-point discrete Fourier
    Transform (DFT) of a real-valued array by means of an efficient algorithm
    called the Fast Fourier Transform (FFT).

    For full documentation refer to :obj:`numpy.fft.rfft`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array, taken to be real.
    n : {None, int}, optional
        Number of points along transformation axis in the input to use.
        If `n` is smaller than the length of the input, the input is cropped.
        If it is larger, the input is padded with zeros. If `n` is not given,
        the length of the input along the axis specified by `axis` is used.

        Default: ``None``.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last axis is
        used.

        Default: ``-1``.
    norm : {None, "backward", "ortho", "forward"}, optional
        Normalization mode (see :obj:`dpnp.fft`).
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor. ``None`` is an alias of
        the default option ``"backward"``.

        Default: ``"backward"``.
    out : {None, dpnp.ndarray or usm_ndarray of complex dtype}, optional
        If provided, the result will be placed in this array. It should be
        of the appropriate shape and dtype.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray of complex dtype
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.
        If `n` is even, the length of the transformed axis is ``(n/2)+1``.
        If `n` is odd, the length is ``(n+1)/2``.

    See Also
    --------
    :obj:`dpnp.fft` : For definition of the DFT and conventions used.
    :obj:`dpnp.fft.irfft` : The inverse of :obj:`dpnp.fft.rfft`.
    :obj:`dpnp.fft.fft` : The one-dimensional FFT of general (complex) input.
    :obj:`dpnp.fft.fftn` : The *N*-dimensional FFT.
    :obj:`dpnp.fft.rfftn` : The *N*-dimensional FFT of real input.

    Notes
    -----
    When the DFT is computed for purely real input, the output is
    Hermitian-symmetric, i.e. the negative frequency terms are just the complex
    conjugates of the corresponding positive-frequency terms, and the
    negative-frequency terms are therefore redundant. This function does not
    compute the negative frequency terms, and the length of the transformed
    axis of the output is therefore ``n//2 + 1``.

    When ``A = dpnp.fft.rfft(a)`` and fs is the sampling frequency, ``A[0]``
    contains the zero-frequency term 0*fs, which is real due to Hermitian
    symmetry.

    If `n` is even, ``A[-1]`` contains the term representing both positive
    and negative Nyquist frequency (+fs/2 and -fs/2), and must also be purely
    real. If `n` is odd, there is no term at fs/2; ``A[-1]`` contains
    the largest positive frequency (fs/2*(n-1)/n), and is complex in the
    general case.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([0, 1, 0, 0])
    >>> np.fft.fft(a)
    array([ 1.+0.j,  0.-1.j, -1.+0.j,  0.+1.j]) # may vary
    >>> np.fft.rfft(a)
    array([ 1.+0.j,  0.-1.j, -1.+0.j]) # may vary

    Notice how the final element of the :obj:`dpnp.fft.fft` output is the
    complex conjugate of the second element, for real input.
    For :obj:`dpnp.fft.rfft`, this symmetry is exploited to compute only the
    non-negative frequency terms.

    """

    dpnp.check_supported_arrays_type(a)
    return dpnp_fft(
        a, forward=True, real=True, n=n, axis=axis, norm=norm, out=out
    )


def rfft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    """
    Compute the 2-dimensional FFT of a real array.

    For full documentation refer to :obj:`numpy.fft.rfft2`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array, taken to be real.
    s : {None, sequence of ints}, optional
        Shape (length of each transformed axis) to use from the input.
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        The final element of `s` corresponds to `n` for ``rfft(x, n)``, while
        for the remaining axes, it corresponds to `n` for ``fft(x, n)``.
        Along each axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.
        If it is ``-1``, the whole input is used (no padding/trimming).
        If `s` is not given, the shape of the input along the axes specified
        by `axes` is used. If `s` is not ``None``, `axes` must not be ``None``
        either.

        Default: ``None``.
    axes : {None, sequence of ints}, optional
        Axes over which to compute the FFT. If not given, the last two axes are
        used. A repeated index in `axes` means the transform over that axis is
        performed multiple times. If `s` is specified, the corresponding `axes`
        to be transformed must be explicitly specified too. A one-element
        sequence means that a one-dimensional FFT is performed. An empty
        sequence means that no FFT is performed.

        Default: ``(-2, -1)``.
    norm : {None, "backward", "ortho", "forward"}, optional
        Normalization mode (see :obj:`dpnp.fft`).
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor. ``None`` is an alias of
        the default option ``"backward"``.

        Default: ``"backward"``.
    out : {None, dpnp.ndarray or usm_ndarray of complex dtype}, optional
        If provided, the result will be placed in this array. It should be of
        the appropriate dtype and shape for the last transformation
        (consistent with the choice of `s`).

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray of complex dtype
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or the last two axes if `axes` is not given.

    See Also
    --------
    :obj:`dpnp.fft` : Overall view of discrete Fourier transforms, with
        definitions and conventions used.
    :obj:`dpnp.fft.rfft` : The one-dimensional FFT of real input.
    :obj:`dpnp.fft.rfftn` : The `n`-dimensional FFT of real input.
    :obj:`dpnp.fft.irfft2` : The inverse two-dimensional real FFT.

    Notes
    -----
    This is just :obj:`dpnp.fft.rfftn` with different default behavior.
    For more details see :obj:`dpnp.fft.rfftn`.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.mgrid[:5, :5][0]
    >>> np.fft.rfft2(a)
    array([[ 50.  +0.j        ,   0.  +0.j        ,   0.  +0.j        ],
           [-12.5+17.20477401j,   0.  +0.j        ,   0.  +0.j        ],
           [-12.5 +4.0614962j ,   0.  +0.j        ,   0.  +0.j        ],
           [-12.5 -4.0614962j ,   0.  +0.j        ,   0.  +0.j        ],
           [-12.5-17.20477401j,   0.  +0.j        ,   0.  +0.j        ]])

    """

    dpnp.check_supported_arrays_type(a)
    return dpnp_fftn(
        a, forward=True, real=True, s=s, axes=axes, norm=norm, out=out
    )


def rfftfreq(
    n, /, d=1.0, *, dtype=None, device=None, usm_type=None, sycl_queue=None
):
    """
    Return the Discrete Fourier Transform sample frequencies
    (for usage with :obj:`dpnp.fft.rfft`, :obj:`dpnp.fft.irfft`).

    The returned float array `f` contains the frequency bin centers in cycles
    per unit of the sample spacing (with zero at the start). For instance, if
    the sample spacing is in seconds, then the frequency unit is cycles/second.

    Given a window length `n` and a sample spacing `d`::

      f = [0, 1, ...,     n/2-1,     n/2] / (d*n)   if n is even
      f = [0, 1, ..., (n-1)/2-1, (n-1)/2] / (d*n)   if n is odd

    Unlike :obj:`dpnp.fft.fftfreq` the Nyquist frequency component is
    considered to be positive.

    For full documentation refer to :obj:`numpy.fft.rfftfreq`.

    Parameters
    ----------
    n : int
        Window length.
    d : scalar, optional
        Sample spacing (inverse of the sampling rate).

        Default: ``1.0``.
    dtype : {None, str, dtype object}, optional
        The output array data type. Must be a real-valued floating-point data
        type. If `dtype` is ``None``, the output array data type must be the
        default real-valued floating-point data type.

        Default: ``None``.
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
    f : dpnp.ndarray
        Array of length ``n//2 + 1`` containing the sample frequencies.

    See Also
    --------
    :obj:`dpnp.fft.fftfreq` : Return the Discrete Fourier Transform sample
                        frequencies (for usage with :obj:`dpnp.fft.fft` and
                        :obj:`dpnp.fft.ifft`).

    Examples
    --------
    >>> import dpnp as np
    >>> signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5, -3, 4])
    >>> fourier = np.fft.fft(signal)
    >>> n = signal.size
    >>> sample_rate = 100
    >>> freq = np.fft.fftfreq(n, d=1./sample_rate)
    >>> freq
    array([  0.,  10.,  20.,  30.,  40., -50., -40., -30., -20., -10.])
    >>> freq = np.fft.rfftfreq(n, d=1./sample_rate)
    >>> freq
    array([ 0., 10., 20., 30., 40., 50.])

    Creating the output array on a different device or with a
    specified usm_type:

    >>> x = np.fft.rfftfreq(n, d=1./sample_rate) # default case
    >>> x.shape, x.device, x.usm_type
    ((6,), Device(level_zero:gpu:0), 'device')

    >>> y = np.fft.rfftfreq(n, d=1./sample_rate, device="cpu")
    >>> y.shape, y.device, y.usm_type
    ((6,), Device(opencl:cpu:0), 'device')

    >>> z = np.fft.rfftfreq(n, d=1./sample_rate, usm_type="host")
    >>> z.shape, z.device, z.usm_type
    ((6,), Device(level_zero:gpu:0), 'host')

    """

    if not isinstance(n, int):
        raise ValueError("`n` should be an integer")
    if not dpnp.isscalar(d):
        raise ValueError("`d` should be an scalar")

    if dtype and not dpnp.issubdtype(dtype, dpnp.floating):
        raise ValueError(
            "dtype must a real-valued floating-point data type, "
            f"but got {dtype}"
        )

    val = 1.0 / (n * d)
    m = n // 2 + 1
    results = dpnp.arange(
        0,
        m,
        dtype=dtype,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )
    return results * val


def rfftn(a, s=None, axes=None, norm=None, out=None):
    """
    Compute the *N*-dimensional discrete Fourier Transform for real input.

    This function computes the *N*-dimensional discrete Fourier Transform over
    any number of axes in an *M*-dimensional real array by means of the Fast
    Fourier Transform (FFT). By default, all axes are transformed, with the
    real transform performed over the last axis, while the remaining
    transforms are complex.

    For full documentation refer to :obj:`numpy.fft.rfftn`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array, taken to be real.
    s : {None, sequence of ints}, optional
        Shape (length of each transformed axis) to use from the input.
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        The final element of `s` corresponds to `n` for ``rfft(x, n)``, while
        for the remaining axes, it corresponds to `n` for ``fft(x, n)``.
        Along each axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.
        If it is ``-1``, the whole input is used (no padding/trimming).
        If `s` is not given, the shape of the input along the axes specified
        by `axes` is used. If `s` is not ``None``, `axes` must not be ``None``
        either.

        Default: ``None``.
    axes : {None, sequence of ints}, optional
        Axes over which to compute the FFT. If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
        Repeated indices in `axes` means that the transform over that axis is
        performed multiple times. If `s` is specified, the corresponding `axes`
        to be transformed must be explicitly specified too. A one-element
        sequence means that a one-dimensional FFT is performed. An empty
        sequence means that no FFT is performed.

        Default: ``None``.
    norm : {None, "backward", "ortho", "forward"}, optional
        Normalization mode (see :obj:`dpnp.fft`).
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor. ``None`` is an alias of
        the default option ``"backward"``.

        Default: ``"backward"``.
    out : {None, dpnp.ndarray or usm_ndarray of complex dtype}, optional
        If provided, the result will be placed in this array. It should be of
        the appropriate dtype and shape for the last transformation
        (consistent with the choice of `s`).

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray of complex dtype
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` and `a`,
        as explained in the parameters section above.
        The length of the last axis transformed will be ``s[-1]//2+1``,
        while the remaining transformed axes will have lengths according to
        `s`, or unchanged from the input.

    See Also
    --------
    :obj:`dpnp.fft` : Overall view of discrete Fourier transforms, with
        definitions and conventions used.
    :obj:`dpnp.fft.irfftn` : The inverse of the *N*-dimensional FFT of
                        real input.
    :obj:`dpnp.fft.fft` : The one-dimensional FFT of general (complex) input.
    :obj:`dpnp.fft.rfft` : The one-dimensional FFT of real input.
    :obj:`dpnp.fft.fftn` : The *N*-dimensional FFT.
    :obj:`dpnp.fft.fftn` : The two-dimensional FFT.

    Notes
    -----
    The transform for real input is performed over the last transformation
    axis, as by :obj:`dpnp.fft.rfft`, then the transform over the remaining
    axes is performed as by :obj:`dpnp.fft.fftn`. The order of the output
    is as for :obj:`dpnp.fft.rfft` for the final transformation axis, and
    as for :obj:`dpnp.fft.fftn` for the remaining transformation axes.

    See :obj:`dpnp.fft` for details, definitions and conventions used.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.ones((2, 2, 2))
    >>> np.fft.rfftn(a)
    array([[[8.+0.j,  0.+0.j], # may vary
            [0.+0.j,  0.+0.j]],
           [[0.+0.j,  0.+0.j],
            [0.+0.j,  0.+0.j]]])

    >>> np.fft.rfftn(a, axes=(2, 0))
    array([[[4.+0.j,  0.+0.j], # may vary
            [4.+0.j,  0.+0.j]],
           [[0.+0.j,  0.+0.j],
            [0.+0.j,  0.+0.j]]])

    """

    dpnp.check_supported_arrays_type(a)
    return dpnp_fftn(
        a, forward=True, real=True, s=s, axes=axes, norm=norm, out=out
    )
