# cython: language_level=3
# distutils: language = c++
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2020, Intel Corporation
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


import dpnp
import numpy

from dpnp.dpnp_utils import *
from dpnp.fft.dpnp_algo_fft import *


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
    "rfftn"
]


def fft(x1, n=None, axis=-1, norm=None):
    """
    Compute the one-dimensional discrete Fourier Transform.

    Limitations
    -----------
    Parameter ``norm`` is unsupported.
    Parameter ``x1`` supports ``dpnp.int32``, ``dpnp.int64``, ``dpnp.float32``, ``dpnp.float64``,
    ``dpnp.complex64`` and ``dpnp.complex128`` datatypes only.

    For full documentation refer to :obj:`numpy.fft.fft`.

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        # if norm is None or norm is 'backward':
        #     norm_val = 0
        # else:
        #     norm_val = 1
        if axis is None:
            axis_param = -1      # the most right dimension (default value)
        else:
            axis_param = axis

        if n is None:
            input_boundarie = x1_desc.shape[axis_param]
        else:
            input_boundarie = n

        if x1_desc.size < 1:
            pass                 # let fallback to handle exception
        elif input_boundarie < 1:
            pass                 # let fallback to handle exception
        elif norm is not None:
            pass
        else:
            output_boundarie = input_boundarie

            return dpnp_fft(x1_desc, input_boundarie, output_boundarie, axis_param, False, 0).get_pyobj()

    return call_origin(numpy.fft.fft, x1, n, axis, norm)


def fft2(x1, s=None, axes=(-2, -1), norm=None):
    """
    Compute the 2-dimensional discrete Fourier Transform

    Multi-dimensional arrays computed as batch of 1-D arrays

    Limitations
    -----------
    Parameter ``norm`` is unsupported.
    Parameter ``x1`` supports ``dpnp.int32``, ``dpnp.int64``, ``dpnp.float32``, ``dpnp.float64`` and
    ``dpnp.complex128`` datatypes only.

    For full documentation refer to :obj:`numpy.fft.fft2`.

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if norm is not None:
            pass
        else:
            return fftn(x1, s, axes, norm)

    return call_origin(numpy.fft.fft2, x1, s, axes, norm)


def fftfreq(n=None, d=1.0):
    """
    Compute the one-dimensional discrete Fourier Transform sample frequencies.

    Limitations
    -----------
    Parameter ``d`` is unsupported.

    For full documentation refer to :obj:`numpy.fft.fftfreq`.

    """

    return call_origin(numpy.fft.fftfreq, n, d)


def fftn(x1, s=None, axes=None, norm=None):
    """
    Compute the N-dimensional FFT.

    Multi-dimensional arrays computed as batch of 1-D arrays

    Limitations
    -----------
    Parameter ``norm`` is unsupported.
    Parameter ``x1`` supports ``dpnp.int32``, ``dpnp.int64``, ``dpnp.float32``, ``dpnp.float64`` and
    ``dpnp.complex128`` datatypes only.

    For full documentation refer to :obj:`numpy.fft.fftn`.

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if s is None:
            boundaries = tuple([x1_desc.shape[i] for i in range(x1_desc.ndim)])
        else:
            boundaries = s

        if axes is None:
            axes_param = tuple([i for i in range(x1_desc.ndim)])
        else:
            axes_param = axes

        if norm is not None:
            pass
        else:
            x1_iter = x1
            iteration_list = list(range(len(axes_param)))
            iteration_list.reverse()  # inplace operation
            for it in iteration_list:
                param_axis = axes_param[it]
                try:
                    param_n = boundaries[param_axis]
                except IndexError:
                    checker_throw_axis_error("fft.fftn", "is out of bounds", param_axis, f"< {len(boundaries)}")

                x1_iter = fft(x1_iter, n=param_n, axis=param_axis, norm=norm)

            return x1_iter

    return call_origin(numpy.fft.fftn, x1, s, axes, norm)


def fftshift(x1, axes=None):
    """
    Shift the zero-frequency component to the center of the spectrum.

    Limitations
    -----------
    Parameter ``axes`` is unsupported.
    Parameter ``x1`` supports ``dpnp.int32``, ``dpnp.int64``, ``dpnp.float32``, ``dpnp.float64`` and
    ``dpnp.complex128`` datatypes only.

    For full documentation refer to :obj:`numpy.fft.fftshift`.

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc and 0:
        if axis is None:
            axis_param = -1      # the most right dimension (default value)
        else:
            axis_param = axes

        if x1_desc.size < 1:
            pass                 # let fallback to handle exception
        else:
            return dpnp_fft(x1_desc, input_boundarie, output_boundarie, axis_param, False).get_pyobj()

    return call_origin(numpy.fft.fftshift, x1, axes)


def hfft(x1, n=None, axis=-1, norm=None):
    """
    Compute the one-dimensional discrete Fourier Transform of a signal that has Hermitian symmetry.

    Limitations
    -----------
    Parameter ``norm`` is unsupported.
    Parameter ``x1`` supports ``dpnp.int32``, ``dpnp.int64``, ``dpnp.float32``, ``dpnp.float64`` and
    ``dpnp.complex128`` datatypes only.

    For full documentation refer to :obj:`numpy.fft.hfft`.

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc and 0:
        if axis is None:
            axis_param = -1      # the most right dimension (default value)
        else:
            axis_param = axis

        if n is None:
            input_boundarie = x1_desc.shape[axis_param]
        else:
            input_boundarie = n

        if x1.size < 1:
            pass                 # let fallback to handle exception
        elif input_boundarie < 1:
            pass                 # let fallback to handle exception
        elif norm is not None:
            pass
        else:
            output_boundarie = input_boundarie

            return dpnp_fft(x1_desc, input_boundarie, output_boundarie, axis_param, False).get_pyobj()

    return call_origin(numpy.fft.hfft, x1, n, axis, norm)


def ifft(x1, n=None, axis=-1, norm=None):
    """
    Compute the one-dimensional inverse discrete Fourier Transform.

    Limitations
    -----------
    Parameter ``norm`` is unsupported.
    Parameter ``x1`` supports ``dpnp.int32``, ``dpnp.int64``, ``dpnp.float32``, ``dpnp.float64`` and
    ``dpnp.complex128`` datatypes only.

    For full documentation refer to :obj:`numpy.fft.ifft`.

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if axis is None:
            axis_param = -1      # the most right dimension (default value)
        else:
            axis_param = axis

        if n is None:
            input_boundarie = x1_desc.shape[axis_param]
        else:
            input_boundarie = n

        if x1_desc.size < 1:
            pass                 # let fallback to handle exception
        elif input_boundarie < 1:
            pass                 # let fallback to handle exception
        elif norm is not None:
            pass
        else:
            output_boundarie = input_boundarie

            return dpnp_fft(x1_desc, input_boundarie, output_boundarie, axis_param, True).get_pyobj()

    return call_origin(numpy.fft.ifft, x1, n, axis, norm)


def ifft2(x1, s=None, axes=(-2, -1), norm=None):
    """
    Compute the 2-dimensional inverse discrete Fourier Transform

    Multi-dimensional arrays computed as batch of 1-D arrays

    Limitations
    -----------
    Parameter ``norm`` is unsupported.
    Parameter ``x1`` supports ``dpnp.int32``, ``dpnp.int64``, ``dpnp.float32``, ``dpnp.float64`` and
    ``dpnp.complex128`` datatypes only.

    For full documentation refer to :obj:`numpy.fft.ifft2`.

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if norm is not None:
            pass
        else:
            return ifftn(x1, s, axes, norm)

    return call_origin(numpy.fft.ifft2, x1, s, axes, norm)


def ifftshift(x1, axes=None):
    """
    Inverse shift the zero-frequency component to the center of the spectrum.

    Limitations
    -----------
    Parameter ``axes`` is unsupported.
    Parameter ``x1`` supports ``dpnp.int32``, ``dpnp.int64``, ``dpnp.float32``, ``dpnp.float64`` and
    ``dpnp.complex128`` datatypes only.

    For full documentation refer to :obj:`numpy.fft.ifftshift`.

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc and 0:
        if axis is None:
            axis_param = -1      # the most right dimension (default value)
        else:
            axis_param = axes

        if x1_desc.size < 1:
            pass                 # let fallback to handle exception
        else:
            return dpnp_fft(x1_desc, input_boundarie, output_boundarie, axis_param, False).get_pyobj()

    return call_origin(numpy.fft.ifftshift, x1, axes)


def ifftn(x1, s=None, axes=None, norm=None):
    """
    Compute the N-dimensional inverse discrete Fourier Transform.

    Multi-dimensional arrays computed as batch of 1-D arrays

    Limitations
    -----------
    Parameter ``norm`` is unsupported.
    Parameter ``x1`` supports ``dpnp.int32``, ``dpnp.int64``, ``dpnp.float32``, ``dpnp.float64`` and
    ``dpnp.complex128`` datatypes only.

    For full documentation refer to :obj:`numpy.fft.ifftn`.

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if s is None:
            boundaries = tuple([x1_desc.shape[i] for i in range(x1_desc.ndim)])
        else:
            boundaries = s

        if axes is None:
            axes_param = tuple([i for i in range(x1_desc.ndim)])
        else:
            axes_param = axes

        if norm is not None:
            pass
        else:
            x1_iter = x1
            iteration_list = list(range(len(axes_param)))
            iteration_list.reverse()  # inplace operation
            for it in iteration_list:
                param_axis = axes_param[it]
                try:
                    param_n = boundaries[param_axis]
                except IndexError:
                    checker_throw_axis_error("fft.ifftn", "is out of bounds", param_axis, f"< {len(boundaries)}")

                x1_iter_desc = dpnp.get_dpnp_descriptor(x1_iter)
                x1_iter = ifft(x1_iter_desc.get_pyobj(), n=param_n, axis=param_axis, norm=norm)

            return x1_iter

    return call_origin(numpy.fft.ifftn, x1, s, axes, norm)


def ihfft(x1, n=None, axis=-1, norm=None):
    """
    Compute inverse one-dimensional discrete Fourier Transform of a signal that has Hermitian symmetry.

    Limitations
    -----------
    Parameter ``norm`` is unsupported.
    Parameter ``x1`` supports ``dpnp.int32``, ``dpnp.int64``, ``dpnp.float32``, ``dpnp.float64`` and
    ``dpnp.complex128`` datatypes only.

    For full documentation refer to :obj:`numpy.fft.ihfft`.

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc and 0:
        if axis is None:
            axis_param = -1      # the most right dimension (default value)
        else:
            axis_param = axis

        if n is None:
            input_boundarie = x1_desc.shape[axis_param]
        else:
            input_boundarie = n

        if x1_desc.size < 1:
            pass                 # let fallback to handle exception
        elif input_boundarie < 1:
            pass                 # let fallback to handle exception
        elif norm is not None:
            pass
        else:
            output_boundarie = input_boundarie

            return dpnp_fft(x1_desc, input_boundarie, output_boundarie, axis_param, False).get_pyobj()

    return call_origin(numpy.fft.ihfft, x1, n, axis, norm)


def irfft(x1, n=None, axis=-1, norm=None):
    """
    Compute the one-dimensional inverse discrete Fourier Transform for real input..

    Limitations
    -----------
    Parameter ``norm`` is unsupported.
    Parameter ``x1`` supports ``dpnp.int32``, ``dpnp.int64``, ``dpnp.float32``, ``dpnp.float64`` and
    ``dpnp.complex128`` datatypes only.

    For full documentation refer to :obj:`numpy.fft.irfft`.

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc and 0:
        if axis is None:
            axis_param = -1      # the most right dimension (default value)
        else:
            axis_param = axis

        if n is None:
            input_boundarie = x1_desc.shape[axis_param]
        else:
            input_boundarie = n

        if x1_desc.size < 1:
            pass                 # let fallback to handle exception
        elif input_boundarie < 1:
            pass                 # let fallback to handle exception
        elif norm is not None:
            pass
        else:
            output_boundarie = 2 * (input_boundarie - 1)

            result = dpnp_fft(x1_desc, input_boundarie, output_boundarie, axis_param, True).get_pyobj()
            # TODO tmp = utils.create_output_array(result_shape, result_c_type, out)
            # tmp = dparray(result.shape, dtype=dpnp.float64)
            # for it in range(tmp.size):
            #     tmp[it] = result[it].real
            return result

    return call_origin(numpy.fft.irfft, x1, n, axis, norm)


def irfft2(x1, s=None, axes=(-2, -1), norm=None):
    """
    Compute the 2-dimensional inverse discrete Fourier Transform for real input.

    Multi-dimensional arrays computed as batch of 1-D arrays

    Limitations
    -----------
    Parameter ``norm`` is unsupported.
    Parameter ``x1`` supports ``dpnp.int32``, ``dpnp.int64``, ``dpnp.float32``, ``dpnp.float64`` and
    ``dpnp.complex128`` datatypes only.

    For full documentation refer to :obj:`numpy.fft.irfft2`.

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if norm is not None:
            pass
        else:
            return irfftn(x1_desc.get_pyobj(), s, axes, norm)

    return call_origin(numpy.fft.irfft2, x1, s, axes, norm)


def irfftn(x1, s=None, axes=None, norm=None):
    """
    Compute the N-dimensional inverse discrete Fourier Transform for real input.

    Multi-dimensional arrays computed as batch of 1-D arrays

    Limitations
    -----------
    Parameter ``norm`` is unsupported.
    Parameter ``x1`` supports ``dpnp.int32``, ``dpnp.int64``, ``dpnp.float32``, ``dpnp.float64`` and
    ``dpnp.complex128`` datatypes only.

    For full documentation refer to :obj:`numpy.fft.irfftn`.

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc and 0:
        if s is None:
            boundaries = tuple([x1_desc.shape[i] for i in range(x1_desc.ndim)])
        else:
            boundaries = s

        if axes is None:
            axes_param = tuple([i for i in range(x1_desc.ndim)])
        else:
            axes_param = axes

        if norm is not None:
            pass
        else:
            x1_iter = x1
            iteration_list = list(range(len(axes_param)))
            iteration_list.reverse()  # inplace operation
            for it in iteration_list:
                param_axis = axes_param[it]
                try:
                    param_n = boundaries[param_axis]
                except IndexError:
                    checker_throw_axis_error("fft.irfftn", "is out of bounds", param_axis, f"< {len(boundaries)}")

                x1_iter_desc = dpnp.get_dpnp_descriptor(x1_iter)
                x1_iter = irfft(x1_iter_desc.get_pyobj(), n=param_n, axis=param_axis, norm=norm)

            return x1_iter

    return call_origin(numpy.fft.irfftn, x1, s, axes, norm)


def rfft(x1, n=None, axis=-1, norm=None):
    """
    Compute the one-dimensional discrete Fourier Transform for real input.

    Limitations
    -----------
    Parameter ``norm`` is unsupported.
    Parameter ``x1`` supports ``dpnp.int32``, ``dpnp.int64``, ``dpnp.float32``, ``dpnp.float64`` and
    ``dpnp.complex128`` datatypes only.

    For full documentation refer to :obj:`numpy.fft.rfft`.

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if axis is None:
            axis_param = -1                             # the most right dimension (default value)
        else:
            axis_param = axis

        if n is None:
            input_boundarie = x1_desc.shape[axis_param]
        else:
            input_boundarie = n

        if x1_desc.size < 1:
            pass                                        # let fallback to handle exception
        elif input_boundarie < 1:
            pass                                        # let fallback to handle exception
        elif norm is not None:
            pass
        else:
            output_boundarie = input_boundarie // 2 + 1  # rfft specific requirenment

            return dpnp_fft(x1_desc, input_boundarie, output_boundarie, axis_param, False).get_pyobj()

    return call_origin(numpy.fft.rfft, x1, n, axis, norm)


def rfft2(x1, s=None, axes=(-2, -1), norm=None):
    """
    Compute the 2-dimensional discrete Fourier Transform for real input.

    Multi-dimensional arrays computed as batch of 1-D arrays

    Limitations
    -----------
    Parameter ``norm`` is unsupported.
    Parameter ``x1`` supports ``dpnp.int32``, ``dpnp.int64``, ``dpnp.float32``, ``dpnp.float64`` and
    ``dpnp.complex128`` datatypes only.

    For full documentation refer to :obj:`numpy.fft.rfft2`.

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if norm is not None:
            pass
        else:
            return rfftn(x1_desc.get_pyobj(), s, axes, norm)

    return call_origin(numpy.fft.rfft2, x1, s, axes, norm)


def rfftfreq(n=None, d=1.0):
    """
    Compute the one-dimensional discrete Fourier Transform sample frequencies.

    Limitations
    -----------
    Parameter ``d`` is unsupported.

    For full documentation refer to :obj:`numpy.fft.rfftfreq`.

    """

    return call_origin(numpy.fft.rfftfreq, n, d)


def rfftn(x1, s=None, axes=None, norm=None):
    """
    Compute the N-dimensional discrete Fourier Transform for real input.

    Multi-dimensional arrays computed as batch of 1-D arrays

    Limitations
    -----------
    Parameter ``norm`` is unsupported.
    Parameter ``x1`` supports ``dpnp.int32``, ``dpnp.int64``, ``dpnp.float32``, ``dpnp.float64`` and
    ``dpnp.complex128`` datatypes only.

    For full documentation refer to :obj:`numpy.fft.rfftn`.

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if s is None:
            boundaries = tuple([x1_desc.shape[i] for i in range(x1_desc.ndim)])
        else:
            boundaries = s

        if axes is None:
            axes_param = tuple([i for i in range(x1_desc.ndim)])
        else:
            axes_param = axes

        if norm is not None:
            pass
        elif len(axes) < 1:
            pass                      # let fallback to handle exception
        else:
            x1_iter = x1
            iteration_list = list(range(len(axes_param)))
            iteration_list.reverse()  # inplace operation
            for it in iteration_list:
                param_axis = axes_param[it]
                try:
                    param_n = boundaries[param_axis]
                except IndexError:
                    checker_throw_axis_error("fft.rfftn", "is out of bounds", param_axis, f"< {len(boundaries)}")

                x1_iter_desc = dpnp.get_dpnp_descriptor(x1_iter)
                x1_iter = rfft(x1_iter_desc.get_pyobj(), n=param_n, axis=param_axis, norm=norm)

            return x1_iter

    return call_origin(numpy.fft.rfftn, x1, s, axes, norm)
