# cython: language_level=3
# distutils: language = c++
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
    dpnp_fft,
    dpnp_rfft,
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


def fft(x, n=None, axis=-1, norm=None):
    """
    Compute the one-dimensional discrete Fourier Transform.

    For full documentation refer to :obj:`numpy.fft.fft`.

    Limitations
    -----------
    Parameter `x` is supported either as :class:`dpnp.ndarray`.
    Parameter `axis` is supported with its default value.
    Only `dpnp.float64`, `dpnp.float32`, `dpnp.int64`, `dpnp.int32`,
    `dpnp.complex128`, `dpnp.complex64` data types are supported.
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
        elif n is not None:
            pass
        elif axis != -1:
            pass
        else:
            output_boundarie = input_boundarie
            return dpnp_fft(
                x_desc,
                input_boundarie,
                output_boundarie,
                axis_param,
                False,
                norm_.value,
            ).get_pyobj()
    return call_origin(numpy.fft.fft, x, n, axis, norm)


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

    This function swaps half-spaces for all axes listed (defaults to all).
    Note that ``out[0]`` is the Nyquist component only if ``len(x)`` is even.

    For full documentation refer to :obj:`numpy.fft.fftshift`.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        Input array.
    axes : {None, int, list or tuple of ints}, optional
        Axes over which to shift.
        Default is ``None``, which shifts all axes.

    Returns
    -------
    out : dpnp.ndarray
        The shifted array.

    See Also
    --------
    :obj:`dpnp.ifftshift` : The inverse of :obj:`dpnp.fftshift`.

    Examples
    --------
    >>> import dpnp as np
    >>> freqs = np.fft.fftfreq(10, 0.1)
    >>> freqs
    array([ 0.,  1.,  2., ..., -3., -2., -1.])
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

            return dpnp_fft(
                x_desc,
                input_boundarie,
                output_boundarie,
                axis_param,
                False,
                norm_.value,
            ).get_pyobj()

    return call_origin(numpy.fft.hfft, x, n, axis, norm)


def ifft(x, n=None, axis=-1, norm=None):
    """
    Compute the one-dimensional inverse discrete Fourier Transform.

    For full documentation refer to :obj:`numpy.fft.ifft`.

    Limitations
    -----------
    Parameter `x` is supported either as :class:`dpnp.ndarray`.
    Parameter `axis` is supported with its default value.
    Only `dpnp.float64`, `dpnp.float32`, `dpnp.int64`, `dpnp.int32`,,
    `dpnp.complex128`, `dpnp.complex64` data types are supported.
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
        elif n is not None:
            pass
        else:
            output_boundarie = input_boundarie
            return dpnp_fft(
                x_desc,
                input_boundarie,
                output_boundarie,
                axis_param,
                True,
                norm_.value,
            ).get_pyobj()

    return call_origin(numpy.fft.ifft, x, n, axis, norm)


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
        Axes over which to calculate.
        Defaults to ``None``, which shifts all axes.

    Returns
    -------
    out : dpnp.ndarray
        The shifted array.

    See Also
    --------
    :obj:`dpnp.fftshift` : Shift zero-frequency component to the center
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

            return dpnp_fft(
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
