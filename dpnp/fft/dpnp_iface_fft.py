# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2024, Intel Corporation
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
 - The functions parameters check

"""

# pylint: disable=invalid-name

from enum import Enum

import numpy

import dpnp

# pylint: disable=no-name-in-module
from dpnp.dpnp_utils import (
    call_origin,
    checker_throw_axis_error,
)
from dpnp.fft.dpnp_algo_fft import (
    dpnp_fft_deprecated,
    dpnp_rfft,
)

from .dpnp_utils_fft import (
    dpnp_fft,
)

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


# TODO: remove pylint disable, once new implementation is ready
# pylint: disable=missing-class-docstring
class Norm(Enum):
    backward = 0
    forward = 1
    ortho = 2


# TODO: remove pylint disable, once new implementation is ready
# pylint: disable=missing-function-docstring
def get_validated_norm(norm):
    if norm is None or norm == "backward":
        return Norm.backward
    if norm == "forward":
        return Norm.forward
    if norm == "ortho":
        return Norm.ortho
    raise ValueError("Unknown norm value.")


def fft(a, n=None, axis=-1, norm=None):
    """
    Compute the one-dimensional discrete Fourier Transform.

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
        used. Default: ``-1``.
    norm : {None, "backward", "ortho", "forward"}, optional
        Normalization mode (see :obj:`numpy.fft`).
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor. ``None`` is an alias of
        the default option "backward".
        Default: "backward".

    Returns
    -------
    out : dpnp.ndarray of complex dtype
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.

    See Also
    --------
    :obj:`numpy.fft` : For definition of the DFT and conventions used.
    :obj:`dpnp.fft.ifft` : The inverse of :obj:`dpnp.fft.fft`.
    :obj:`dpnp.fft.fft2` : The two-dimensional FFT.
    :obj:`dpnp.fft.fftn` : The `n`-dimensional FFT.
    :obj:`dpnp.fft.rfftn` : The `n`-dimensional FFT of real input.
    :obj:`dpnp.fft.fftfreq` : Frequency bins for given FFT parameters.

    Notes
    -----
    FFT (Fast Fourier Transform) refers to a way the discrete Fourier
    Transform (DFT) can be calculated efficiently, by using symmetries in the
    calculated terms. The symmetry is highest when `n` is a power of 2, and
    the transform is therefore most efficient for these sizes.

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
    return dpnp_fft(a, is_forward=True, n=n, axis=axis, norm=norm)


def fft2(x, s=None, axes=(-2, -1), norm=None):
    """
    Compute the 2-dimensional discrete Fourier Transform.

    Multi-dimensional arrays computed as batch of 1-D arrays.

    For full documentation refer to :obj:`numpy.fft.fft2`.

    Limitations
    -----------
    Parameter `x` is supported either as :class:`dpnp.ndarray`.
    Parameter `norm` is unsupported.
    Only `dpnp.float64`, `dpnp.float32`, `dpnp.int64`, `dpnp.int32`,
    `dpnp.complex128` data types are supported.
    Otherwise the function will be executed sequentially on CPU.

    """

    x_desc = dpnp.get_dpnp_descriptor(x, copy_when_nondefault_queue=False)
    if x_desc:
        if norm is not None:
            pass
        else:
            return fftn(x, s, axes, norm)

    return call_origin(numpy.fft.fft2, x, s, axes, norm)


def fftfreq(n=None, d=1.0):
    """
    Compute the one-dimensional discrete Fourier Transform sample frequencies.

    For full documentation refer to :obj:`numpy.fft.fftfreq`.

    Limitations
    -----------
    Parameter `d` is unsupported.

    """

    return call_origin(numpy.fft.fftfreq, n, d)


def fftn(x, s=None, axes=None, norm=None):
    """
    Compute the N-dimensional FFT.

    Multi-dimensional arrays computed as batch of 1-D arrays.

    For full documentation refer to :obj:`numpy.fft.fftn`.

    Limitations
    -----------
    Parameter `x` is supported either as :class:`dpnp.ndarray`.
    Parameter `norm` is unsupported.
    Only `dpnp.float64`, `dpnp.float32`, `dpnp.int64`, `dpnp.int32`,
    `dpnp.complex128` data types are supported.
    Otherwise the function will be executed sequentially on CPU.

    """

    x_desc = dpnp.get_dpnp_descriptor(x, copy_when_nondefault_queue=False)
    if x_desc:
        if s is None:
            boundaries = tuple(x_desc.shape[i] for i in range(x_desc.ndim))
        else:
            boundaries = s

        if axes is None:
            axes_param = list(range(x_desc.ndim))
        else:
            axes_param = axes

        if norm is not None:
            pass
        else:
            x_iter = x
            iteration_list = list(range(len(axes_param)))
            iteration_list.reverse()  # inplace operation
            for it in iteration_list:
                param_axis = axes_param[it]
                try:
                    param_n = boundaries[param_axis]
                except IndexError:
                    checker_throw_axis_error(
                        "fft.fftn",
                        "is out of bounds",
                        param_axis,
                        f"< {len(boundaries)}",
                    )

                x_iter = fft(x_iter, n=param_n, axis=param_axis, norm=norm)

            return x_iter

    return call_origin(numpy.fft.fftn, x, s, axes, norm)


def fftshift(x, axes=None):
    """
    Shift the zero-frequency component to the center of the spectrum.

    For full documentation refer to :obj:`numpy.fft.fftshift`.

    Limitations
    -----------
    Parameter `x` is supported either as :class:`dpnp.ndarray`.
    Parameter `axes` is unsupported.
    Only `dpnp.float64`, `dpnp.float32`, `dpnp.int64`, `dpnp.int32`,
    `dpnp.complex128` data types are supported.
    Otherwise the function will be executed sequentially on CPU.

    """

    x_desc = dpnp.get_dpnp_descriptor(x, copy_when_nondefault_queue=False)
    # TODO: enable implementation
    # pylint: disable=condition-evals-to-constant
    if x_desc and 0:
        norm_ = Norm.backward

        if axes is None:
            axis_param = -1  # the most right dimension (default value)
        else:
            axis_param = axes

        if x_desc.size < 1:
            pass  # let fallback to handle exception
        else:
            input_boundarie = x_desc.shape[axis_param]
            output_boundarie = input_boundarie

            return dpnp_fft_deprecated(
                x_desc,
                input_boundarie,
                output_boundarie,
                axis_param,
                False,
                norm_.value,
            ).get_pyobj()

    return call_origin(numpy.fft.fftshift, x, axes)


def hfft(x, n=None, axis=-1, norm=None):
    """
    Compute the one-dimensional discrete Fourier Transform of a signal that has
    Hermitian symmetry.

    For full documentation refer to :obj:`numpy.fft.hfft`.

    Limitations
    -----------
    Parameter `x` is supported either as :class:`dpnp.ndarray`.
    Parameter `norm` is unsupported.
    Only `dpnp.float64`, `dpnp.float32`, `dpnp.int64`, `dpnp.int32`,
    `dpnp.complex128` data types are supported.
    Otherwise the function will be executed sequentially on CPU.

    """

    x_desc = dpnp.get_dpnp_descriptor(x, copy_when_nondefault_queue=False)
    # TODO: enable implementation
    # pylint: disable=condition-evals-to-constant
    if x_desc and 0:
        norm_ = get_validated_norm(norm)

        if axis is None:
            axis_param = -1  # the most right dimension (default value)
        else:
            axis_param = axis

        if n is None:
            input_boundarie = x_desc.shape[axis_param]
        else:
            input_boundarie = n

        if x.size < 1:
            pass  # let fallback to handle exception
        elif input_boundarie < 1:
            pass  # let fallback to handle exception
        elif norm is not None:
            pass
        else:
            output_boundarie = input_boundarie

            return dpnp_fft_deprecated(
                x_desc,
                input_boundarie,
                output_boundarie,
                axis_param,
                False,
                norm_.value,
            ).get_pyobj()

    return call_origin(numpy.fft.hfft, x, n, axis, norm)


def ifft(a, n=None, axis=-1, norm=None):
    """
    Compute the one-dimensional inverse discrete Fourier Transform.

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
        axis is used. Default: ``-1``.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see :obj:`numpy.fft`).
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor. ``None`` is an alias of
        the default option "backward".
        Default: "backward"

    Returns
    -------
    out : dpnp.ndarray of complex dtype
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.

    See Also
    --------
    :obj:`numpy.fft` : For definition of the DFT and conventions used.
    :obj:`dpnp.fft.fft` : The one-dimensional (forward) FFT,
                          of which :obj:`dpnp.fft.ifft` is the inverse.
    :obj:`dpnp.fft.ifft2` : The two-dimensional inverse FFT.
    :obj:`dpnp.fft.ifftn` : The `n`-dimensional inverse FFT.

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
    return dpnp_fft(a, is_forward=False, n=n, axis=axis, norm=norm)


def ifft2(x, s=None, axes=(-2, -1), norm=None):
    """
    Compute the 2-dimensional inverse discrete Fourier Transform.

    Multi-dimensional arrays computed as batch of 1-D arrays.

    For full documentation refer to :obj:`numpy.fft.ifft2`.

    Limitations
    -----------
    Parameter `x` is supported either as :class:`dpnp.ndarray`.
    Parameter `norm` is unsupported.
    Only `dpnp.float64`, `dpnp.float32`, `dpnp.int64`, `dpnp.int32`,
    `dpnp.complex128` data types are supported.
    Otherwise the function will be executed sequentially on CPU.

    """

    x_desc = dpnp.get_dpnp_descriptor(x, copy_when_nondefault_queue=False)
    if x_desc:
        if norm is not None:
            pass
        else:
            return ifftn(x, s, axes, norm)

    return call_origin(numpy.fft.ifft2, x, s, axes, norm)


def ifftn(x, s=None, axes=None, norm=None):
    """
    Compute the N-dimensional inverse discrete Fourier Transform.

    Multi-dimensional arrays computed as batch of 1-D arrays.

    For full documentation refer to :obj:`numpy.fft.ifftn`.

    Limitations
    -----------
    Parameter `x` is supported either as :class:`dpnp.ndarray`.
    Parameter `norm` is unsupported.
    Only `dpnp.float64`, `dpnp.float32`, `dpnp.int64`, `dpnp.int32`,
    `dpnp.complex128` data types are supported.
    Otherwise the function will be executed sequentially on CPU.

    """

    x_desc = dpnp.get_dpnp_descriptor(x, copy_when_nondefault_queue=False)
    # TODO: enable implementation
    # pylint: disable=condition-evals-to-constant
    if x_desc and 0:
        if s is None:
            boundaries = tuple(x_desc.shape[i] for i in range(x_desc.ndim))
        else:
            boundaries = s

        if axes is None:
            axes_param = list(range(x_desc.ndim))
        else:
            axes_param = axes

        if norm is not None:
            pass
        else:
            x_iter = x
            iteration_list = list(range(len(axes_param)))
            iteration_list.reverse()  # inplace operation
            for it in iteration_list:
                param_axis = axes_param[it]
                try:
                    param_n = boundaries[param_axis]
                except IndexError:
                    checker_throw_axis_error(
                        "fft.ifftn",
                        "is out of bounds",
                        param_axis,
                        f"< {len(boundaries)}",
                    )

                x_iter_desc = dpnp.get_dpnp_descriptor(x_iter)
                x_iter = ifft(
                    x_iter_desc.get_pyobj(),
                    n=param_n,
                    axis=param_axis,
                    norm=norm,
                )

            return x_iter

    return call_origin(numpy.fft.ifftn, x, s, axes, norm)


def ifftshift(x, axes=None):
    """
    Inverse shift the zero-frequency component to the center of the spectrum.

    For full documentation refer to :obj:`numpy.fft.ifftshift`.

    Limitations
    -----------
    Parameter `x` is supported either as :class:`dpnp.ndarray`.
    Parameter `axes` is unsupported.
    Only `dpnp.float64`, `dpnp.float32`, `dpnp.int64`, `dpnp.int32`,
    `dpnp.complex128` data types are supported.
    Otherwise the function will be executed sequentially on CPU.

    """

    x_desc = dpnp.get_dpnp_descriptor(x, copy_when_nondefault_queue=False)
    # TODO: enable implementation
    # pylint: disable=condition-evals-to-constant
    if x_desc and 0:
        norm_ = Norm.backward

        if axes is None:
            axis_param = -1  # the most right dimension (default value)
        else:
            axis_param = axes

        input_boundarie = x_desc.shape[axis_param]

        if x_desc.size < 1:
            pass  # let fallback to handle exception
        elif input_boundarie < 1:
            pass  # let fallback to handle exception
        else:
            output_boundarie = input_boundarie

            return dpnp_fft_deprecated(
                x_desc,
                input_boundarie,
                output_boundarie,
                axis_param,
                True,
                norm_.value,
            ).get_pyobj()

    return call_origin(numpy.fft.ifftshift, x, axes)


def ihfft(x, n=None, axis=-1, norm=None):
    """
    Compute inverse one-dimensional discrete Fourier Transform of a signal that
    has Hermitian symmetry.

    For full documentation refer to :obj:`numpy.fft.ihfft`.

    Limitations
    -----------
    Parameter `x` is supported either as :class:`dpnp.ndarray`.
    Parameter `norm` is unsupported.
    Only `dpnp.float64`, `dpnp.float32`, `dpnp.int64`, `dpnp.int32`,
    `dpnp.complex128` data types are supported.
    Otherwise the function will be executed sequentially on CPU.

    """

    x_desc = dpnp.get_dpnp_descriptor(x, copy_when_nondefault_queue=False)
    # TODO: enable implementation
    # pylint: disable=condition-evals-to-constant
    if x_desc and 0:
        norm_ = get_validated_norm(norm)

        if axis is None:
            axis_param = -1  # the most right dimension (default value)
        else:
            axis_param = axis

        if n is None:
            input_boundarie = x_desc.shape[axis_param]
        else:
            input_boundarie = n

        if x_desc.size < 1:
            pass  # let fallback to handle exception
        elif input_boundarie < 1:
            pass  # let fallback to handle exception
        elif norm is not None:
            pass
        elif n is not None:
            pass
        else:
            output_boundarie = input_boundarie

            return dpnp_fft_deprecated(
                x_desc,
                input_boundarie,
                output_boundarie,
                axis_param,
                True,
                norm_.value,
            ).get_pyobj()

    return call_origin(numpy.fft.ihfft, x, n, axis, norm)


def irfft(x, n=None, axis=-1, norm=None):
    """
    Compute the one-dimensional inverse discrete Fourier Transform for real
    input.

    For full documentation refer to :obj:`numpy.fft.irfft`.

    Limitations
    -----------
    Parameter `x` is supported either as :class:`dpnp.ndarray`.
    Parameter `norm` is unsupported.
    Only `dpnp.float64`, `dpnp.float32`, `dpnp.int64`, `dpnp.int32`,
    `dpnp.complex128` data types are supported.
    Otherwise the function will be executed sequentially on CPU.

    """

    x_desc = dpnp.get_dpnp_descriptor(x, copy_when_nondefault_queue=False)
    # TODO: enable implementation
    # pylint: disable=condition-evals-to-constant
    if x_desc and 0:
        norm_ = get_validated_norm(norm)

        if axis is None:
            axis_param = -1  # the most right dimension (default value)
        else:
            axis_param = axis

        if n is None:
            input_boundarie = x_desc.shape[axis_param]
        else:
            input_boundarie = n

        if x_desc.size < 1:
            pass  # let fallback to handle exception
        elif input_boundarie < 1:
            pass  # let fallback to handle exception
        elif norm is not None:
            pass
        elif n is not None:
            pass
        else:
            output_boundarie = 2 * (input_boundarie - 1)

            result = dpnp_rfft(
                x_desc,
                input_boundarie,
                output_boundarie,
                axis_param,
                True,
                norm_.value,
            ).get_pyobj()
            # TODO:
            # tmp = utils.create_output_array(result_shape, result_c_type, out)
            # tmp = dparray(result.shape, dtype=dpnp.float64)
            # for it in range(tmp.size):
            #     tmp[it] = result[it].real
            return result

    return call_origin(numpy.fft.irfft, x, n, axis, norm)


def irfft2(x, s=None, axes=(-2, -1), norm=None):
    """
    Compute the 2-dimensional inverse discrete Fourier Transform for real input.

    Multi-dimensional arrays computed as batch of 1-D arrays.

    For full documentation refer to :obj:`numpy.fft.irfft2`.

    Limitations
    -----------
    Parameter `x` is supported either as :class:`dpnp.ndarray`.
    Parameter `norm` is unsupported.
    Only `dpnp.float64`, `dpnp.float32`, `dpnp.int64`, `dpnp.int32`,
    `dpnp.complex128` data types are supported.
    Otherwise the function will be executed sequentially on CPU.

    """

    x_desc = dpnp.get_dpnp_descriptor(x, copy_when_nondefault_queue=False)
    if x_desc:
        if norm is not None:
            pass
        else:
            return irfftn(x_desc.get_pyobj(), s, axes, norm)

    return call_origin(numpy.fft.irfft2, x, s, axes, norm)


def irfftn(x, s=None, axes=None, norm=None):
    """
    Compute the N-dimensional inverse discrete Fourier Transform for real input.

    Multi-dimensional arrays computed as batch of 1-D arrays.

    For full documentation refer to :obj:`numpy.fft.irfftn`.

    Limitations
    -----------
    Parameter `x` is supported either as :class:`dpnp.ndarray`.
    Parameter `norm` is unsupported.
    Only `dpnp.float64`, `dpnp.float32`, `dpnp.int64`, `dpnp.int32`,
    `dpnp.complex128` data types are supported.
    Otherwise the function will be executed sequentially on CPU.

    """

    x_desc = dpnp.get_dpnp_descriptor(x, copy_when_nondefault_queue=False)
    # TODO: enable implementation
    # pylint: disable=condition-evals-to-constant
    if x_desc and 0:
        if s is None:
            boundaries = tuple(x_desc.shape[i] for i in range(x_desc.ndim))
        else:
            boundaries = s

        if axes is None:
            axes_param = list(range(x_desc.ndim))
        else:
            axes_param = axes

        if norm is not None:
            pass
        else:
            x_iter = x
            iteration_list = list(range(len(axes_param)))
            iteration_list.reverse()  # inplace operation
            for it in iteration_list:
                param_axis = axes_param[it]
                try:
                    param_n = boundaries[param_axis]
                except IndexError:
                    checker_throw_axis_error(
                        "fft.irfftn",
                        "is out of bounds",
                        param_axis,
                        f"< {len(boundaries)}",
                    )

                x_iter_desc = dpnp.get_dpnp_descriptor(x_iter)
                x_iter = irfft(
                    x_iter_desc.get_pyobj(),
                    n=param_n,
                    axis=param_axis,
                    norm=norm,
                )

            return x_iter

    return call_origin(numpy.fft.irfftn, x, s, axes, norm)


def rfft(x, n=None, axis=-1, norm=None):
    """
    Compute the one-dimensional discrete Fourier Transform for real input.

    For full documentation refer to :obj:`numpy.fft.rfft`.

    Limitations
    -----------
    Parameter `x` is supported either as :class:`dpnp.ndarray`.
    Parameter `norm` is unsupported.
    Only `dpnp.float64`, `dpnp.float32`, `dpnp.int64`, `dpnp.int32`,
    `dpnp.complex128` data types are supported.
    The `dpnp.bool` data type is not supported and will raise a `TypeError`
    exception.
    Otherwise the function will be executed sequentially on CPU.

    """

    x_desc = dpnp.get_dpnp_descriptor(x, copy_when_nondefault_queue=False)
    if x_desc:
        dt = x_desc.dtype
        if dpnp.issubdtype(dt, dpnp.bool):
            raise TypeError(f"The `{dt}` data type is unsupported.")

        norm_ = get_validated_norm(norm)

        if axis is None:
            axis_param = -1  # the most right dimension (default value)
        else:
            axis_param = axis

        if n is None:
            input_boundarie = x_desc.shape[axis_param]
        else:
            input_boundarie = n

        if x_desc.size < 1:
            pass  # let fallback to handle exception
        elif input_boundarie < 1:
            pass  # let fallback to handle exception
        elif axis != -1:
            pass
        elif norm is not None:
            pass
        elif n is not None:
            pass
        elif x_desc.dtype in (numpy.complex128, numpy.complex64):
            pass
        else:
            output_boundarie = (
                input_boundarie // 2 + 1
            )  # rfft specific requirenment
            return dpnp_rfft(
                x_desc,
                input_boundarie,
                output_boundarie,
                axis_param,
                False,
                norm_.value,
            ).get_pyobj()

    return call_origin(numpy.fft.rfft, x, n, axis, norm)


def rfft2(x, s=None, axes=(-2, -1), norm=None):
    """
    Compute the 2-dimensional discrete Fourier Transform for real input.

    Multi-dimensional arrays computed as batch of 1-D arrays.

    For full documentation refer to :obj:`numpy.fft.rfft2`.

    Limitations
    -----------
    Parameter `x` is supported either as :class:`dpnp.ndarray`.
    Parameter `norm` is unsupported.
    Only `dpnp.float64`, `dpnp.float32`, `dpnp.int64`, `dpnp.int32`,
    `dpnp.complex128` data types are supported.
    Otherwise the function will be executed sequentially on CPU.

    """

    x_desc = dpnp.get_dpnp_descriptor(x, copy_when_nondefault_queue=False)
    if x_desc:
        if norm is not None:
            pass
        else:
            return rfftn(x_desc.get_pyobj(), s, axes, norm)

    return call_origin(numpy.fft.rfft2, x, s, axes, norm)


def rfftfreq(n=None, d=1.0):
    """
    Compute the one-dimensional discrete Fourier Transform sample frequencies.

    For full documentation refer to :obj:`numpy.fft.rfftfreq`.

    Limitations
    -----------
    Parameter `d` is unsupported.

    """

    return call_origin(numpy.fft.rfftfreq, n, d)


def rfftn(x, s=None, axes=None, norm=None):
    """
    Compute the N-dimensional discrete Fourier Transform for real input.

    Multi-dimensional arrays computed as batch of 1-D arrays.

    For full documentation refer to :obj:`numpy.fft.rfftn`.

    Limitations
    -----------
    Parameter `x` is supported either as :class:`dpnp.ndarray`.
    Parameter `norm` is unsupported.
    Only `dpnp.float64`, `dpnp.float32`, `dpnp.int64`, `dpnp.int32`,
    `dpnp.complex128` data types are supported.
    Otherwise the function will be executed sequentially on CPU.

    """

    x_desc = dpnp.get_dpnp_descriptor(x, copy_when_nondefault_queue=False)
    # TODO: enable implementation
    # pylint: disable=condition-evals-to-constant
    if x_desc and 0:
        if s is None:
            boundaries = tuple(x_desc.shape[i] for i in range(x_desc.ndim))
        else:
            boundaries = s

        if axes is None:
            axes_param = list(range(x_desc.ndim))
        else:
            axes_param = axes

        if norm is not None:
            pass
        elif len(axes) < 1:
            pass  # let fallback to handle exception
        else:
            x_iter = x
            iteration_list = list(range(len(axes_param)))
            iteration_list.reverse()  # inplace operation
            for it in iteration_list:
                param_axis = axes_param[it]
                try:
                    param_n = boundaries[param_axis]
                except IndexError:
                    checker_throw_axis_error(
                        "fft.rfftn",
                        "is out of bounds",
                        param_axis,
                        f"< {len(boundaries)}",
                    )

                x_iter_desc = dpnp.get_dpnp_descriptor(
                    x_iter, copy_when_nondefault_queue=False
                )
                x_iter = rfft(
                    x_iter_desc.get_pyobj(),
                    n=param_n,
                    axis=param_axis,
                    norm=norm,
                )

            return x_iter

    return call_origin(numpy.fft.rfftn, x, s, axes, norm)
