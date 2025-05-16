# *****************************************************************************
# Copyright (c) 2024-2025, Intel Corporation
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
Helping functions to implement the FFT interface.

These include assertion functions to validate input array and
functions with the main implementation part to fulfill the interface.
The main computational work is performed by enabling FFT functions
available as a pybind11 extension.

"""

# pylint: disable=protected-access
# pylint: disable=no-name-in-module

from collections.abc import Sequence

import dpctl
import dpctl.tensor._tensor_impl as ti
import dpctl.utils as dpu
import numpy
from dpctl.tensor._numpy_helper import (
    normalize_axis_index,
    normalize_axis_tuple,
)
from dpctl.utils import ExecutionPlacementError

import dpnp
import dpnp.backend.extensions.fft._fft_impl as fi

from ..dpnp_array import dpnp_array
from ..dpnp_utils import map_dtype_to_device
from ..dpnp_utils.dpnp_utils_linearalgebra import (
    _standardize_strides_to_nonzero,
)

__all__ = ["dpnp_fft", "dpnp_fftn", "dpnp_fillfreq", "swap_direction"]


def _check_norm(norm):
    if norm not in (None, "ortho", "forward", "backward"):
        raise ValueError(
            f"Invalid norm value {norm}; should be None, "
            '"ortho", "forward", or "backward".'
        )


def _commit_descriptor(a, forward, in_place, c2c, a_strides, index, batch_fft):
    """Commit the FFT descriptor for the input array."""

    a_shape = a.shape
    shape = a_shape[index:]
    strides = (0,) + a_strides[index:]
    if c2c:  # c2c FFT
        assert dpnp.issubdtype(a.dtype, dpnp.complexfloating)
        if a.dtype == dpnp.complex64:
            dsc = fi.Complex64Descriptor(shape)
        else:
            dsc = fi.Complex128Descriptor(shape)
    else:  # r2c/c2r FFT
        assert dpnp.issubdtype(a.dtype, dpnp.inexact)
        if a.dtype in [dpnp.float32, dpnp.complex64]:
            dsc = fi.Real32Descriptor(shape)
        else:
            dsc = fi.Real64Descriptor(shape)

    dsc.fwd_strides = strides
    dsc.bwd_strides = dsc.fwd_strides
    dsc.transform_in_place = in_place
    out_strides = dsc.bwd_strides[1:]
    if batch_fft:
        dsc.fwd_distance = a_strides[0]
        if c2c:
            dsc.bwd_distance = dsc.fwd_distance
        elif dsc.fwd_strides[-1] == 1:
            if forward:
                dsc.bwd_distance = shape[-1] // 2 + 1
            else:
                dsc.bwd_distance = dsc.fwd_distance
        else:
            dsc.bwd_distance = dsc.fwd_distance
        dsc.number_of_transforms = a_shape[0]  # batch_size
        out_strides.insert(0, dsc.bwd_distance)

    dsc.commit(a.sycl_queue)

    return dsc, out_strides


def _complex_nd_fft(a, s, norm, out, forward, in_place, c2c, axes, batch_fft):
    """Computes complex-to-complex FFT of the input N-D array."""

    len_axes = len(axes)
    # OneMKL supports up to 3-dimensional FFT on GPU
    # repeated axis in OneMKL FFT is not allowed
    if len_axes > 3 or len(set(axes)) < len_axes:
        axes_chunk, shape_chunk = _extract_axes_chunk(axes, s, chunk_size=3)
        for i, (s_chunk, a_chunk) in enumerate(zip(shape_chunk, axes_chunk)):
            a = _truncate_or_pad(a, shape=s_chunk, axes=a_chunk)
            # if out is used in an intermediate step, it will have memory
            # overlap with input and cannot be used in the final step (a new
            # result array will be created for the final step), so there is no
            # benefit in using out in an intermediate step
            if i == len(axes_chunk) - 1:
                tmp_out = out
            else:
                tmp_out = None

            a = _fft(
                a,
                norm=norm,
                out=tmp_out,
                forward=forward,
                # TODO: in-place FFT is only implemented for c2c, see SAT-7154
                in_place=in_place and c2c,
                c2c=c2c,
                axes=a_chunk,
            )

        return a

    a = _truncate_or_pad(a, s, axes)
    if a.size == 0:
        return dpnp.get_result_array(a, out=out, casting="same_kind")

    return _fft(
        a,
        norm=norm,
        out=out,
        forward=forward,
        # TODO: in-place FFT is only implemented for c2c, see SAT-7154
        in_place=in_place and c2c,
        c2c=c2c,
        axes=axes,
        batch_fft=batch_fft,
    )


def _compute_result(dsc, a, out, forward, c2c, out_strides):
    """Compute the result of the FFT."""

    exec_q = a.sycl_queue
    _manager = dpu.SequentialOrderManager[exec_q]
    dep_evs = _manager.submitted_events

    a_usm = dpnp.get_usm_ndarray(a)
    if dsc.transform_in_place:
        # in-place transform
        # TODO: investigate the performance of in-place implementation
        # for r2c/c2r, see SAT-7154
        ht_fft_event, fft_event = fi._fft_in_place(
            dsc, a_usm, forward, depends=dep_evs
        )
        result = a
    else:
        if (
            out is not None
            and out.strides == tuple(out_strides)
            and not ti._array_overlap(a_usm, dpnp.get_usm_ndarray(out))
        ):
            res_usm = dpnp.get_usm_ndarray(out)
            result = out
        else:
            # Result array that is used in OneMKL must have the exact same
            # stride as input array

            if c2c:  # c2c FFT
                out_shape = a.shape
                out_dtype = a.dtype
            else:
                if forward:  # r2c FFT
                    tmp = a.shape[-1] // 2 + 1
                    out_shape = a.shape[:-1] + (tmp,)
                    out_dtype = (
                        dpnp.complex64
                        if a.dtype == dpnp.float32
                        else dpnp.complex128
                    )
                else:  # c2r FFT
                    out_shape = a.shape  # a is already zero-padded
                    out_dtype = (
                        dpnp.float32
                        if a.dtype == dpnp.complex64
                        else dpnp.float64
                    )
            result = dpnp_array(
                out_shape,
                dtype=out_dtype,
                strides=out_strides,
                usm_type=a.usm_type,
                sycl_queue=exec_q,
            )
            res_usm = result.get_array()
        ht_fft_event, fft_event = fi._fft_out_of_place(
            dsc, a_usm, res_usm, forward, depends=dep_evs
        )
    _manager.add_event_pair(ht_fft_event, fft_event)

    if not isinstance(result, dpnp_array):
        return dpnp_array._create_from_usm_ndarray(result)
    return result


def _cook_nd_args(a, s=None, axes=None, c2r=False):
    if s is None:
        shapeless = True
        if axes is None:
            s = list(a.shape)
        else:
            s = numpy.take(a.shape, axes)
    else:
        shapeless = False

    for s_i in s:
        if s_i is not None and s_i < 1 and s_i != -1:
            raise ValueError(
                f"Invalid number of FFT data points ({s_i}) specified."
            )

    if axes is None:
        axes = list(range(-len(s), 0))

    if len(s) != len(axes):
        raise ValueError("Shape and axes have different lengths.")

    s = list(s)
    if c2r and shapeless:
        s[-1] = (a.shape[axes[-1]] - 1) * 2
    # use the whole input array along axis `i` if `s[i] == -1`
    s = [a.shape[_a] if _s == -1 else _s for _s, _a in zip(s, axes)]
    return s, axes


def _copy_array(x, complex_input):
    """
    Creating a C-contiguous copy of input array if input array has a negative
    stride or it does not have a complex data types. In this situation, an
    in-place FFT can be performed.
    """
    dtype = x.dtype
    copy_flag = False
    if numpy.min(x.strides) < 0:
        # negative stride is not allowed in OneMKL FFT
        # TODO: support for negative strides will be added in the future
        # versions of OneMKL, see discussion in MKLD-17597
        copy_flag = True

    if complex_input and not dpnp.issubdtype(dtype, dpnp.complexfloating):
        # c2c/c2r FFT, if input is not complex, convert to complex
        copy_flag = True
        if dtype in [dpnp.float16, dpnp.float32]:
            dtype = dpnp.complex64
        else:
            dtype = map_dtype_to_device(dpnp.complex128, x.sycl_device)
    elif not complex_input and dtype not in [dpnp.float32, dpnp.float64]:
        # r2c FFT, if input is integer or float16 dtype, convert to
        # float32 or float64 depending on device capabilities
        copy_flag = True
        if dtype == dpnp.float16:
            dtype = dpnp.float32
        else:
            dtype = map_dtype_to_device(dpnp.float64, x.sycl_device)

    if copy_flag:
        x = x.astype(dtype, order="C", copy=True)

    # if copying is done, FFT can be in-place (copy_flag = in_place flag)
    return x, copy_flag


def _extract_axes_chunk(a, s, chunk_size=3):
    """
    Classify the first input into a list of lists with each list containing
    only unique values in reverse order and its length is at most `chunk_size`.
    The second input is also classified into a list of lists with each list
    containing the corresponding values of the first input.

    Parameters
    ----------
    a : list or tuple of ints
        The first input.
    s : list or tuple of ints
        The second input.
    chunk_size : int
        Maximum number of elements in each chunk.

    Return
    ------
    out : a tuple of two lists
        The first element of output is a list of lists with each list
        containing only unique values in revere order and its length is
        at most `chunk_size`.
        The second element of output is a list of lists with each list
        containing the corresponding values of the first input.

    Examples
    --------
    >>> axes = (0, 1, 2, 3, 4)
    >>> shape = (7, 8, 10, 9, 5)
    >>> _extract_axes_chunk(axes, shape, chunk_size=3)
    ([[4, 3], [2, 1, 0]], [[5, 9], [10, 8, 7]])

    >>> axes = (1, 0, 3, 2, 4, 4)
    >>> shape = (7, 8, 10, 5, 7, 6)
    >>> _extract_axes_chunk(axes, shape, chunk_size=3)
    ([[4], [4, 2], [3, 0, 1]], [[6], [7, 5], [10, 8, 7]])

    """

    a_chunks = []
    a_current_chunk = []
    seen_elements = set()

    s_chunks = []
    s_current_chunk = []

    for a_elem, s_elem in zip(a, s):
        if a_elem in seen_elements:
            # If element is already seen, start a new chunk
            a_chunks.append(a_current_chunk[::-1])
            s_chunks.append(s_current_chunk[::-1])
            a_current_chunk = [a_elem]
            s_current_chunk = [s_elem]
            seen_elements = {a_elem}
        else:
            a_current_chunk.append(a_elem)
            s_current_chunk.append(s_elem)
            seen_elements.add(a_elem)

        if len(a_current_chunk) == chunk_size:
            a_chunks.append(a_current_chunk[::-1])
            s_chunks.append(s_current_chunk[::-1])
            a_current_chunk = []
            s_current_chunk = []
            seen_elements = set()

    # Add the last chunk if it's not empty
    if a_current_chunk:
        a_chunks.append(a_current_chunk[::-1])
        s_chunks.append(s_current_chunk[::-1])

    return a_chunks[::-1], s_chunks[::-1]


def _fft(a, norm, out, forward, in_place, c2c, axes, batch_fft=True):
    """Calculates FFT of the input array along the specified axes."""

    index = 0
    fft_1d = isinstance(axes, int)
    if batch_fft:
        len_axes = 1 if fft_1d else len(axes)
        local_axes = numpy.arange(-len_axes, 0)
        a = dpnp.moveaxis(a, axes, local_axes)
        a_shape_orig = a.shape
        local_shape = (-1,) + a_shape_orig[-len_axes:]
        a = dpnp.reshape(a, local_shape)
        index = 1

        # cuFFT requires input arrays to be C-contiguous (row-major)
        # for correct execution
        if (
            dpnp.is_cuda_backend(a) and not a.flags.c_contiguous
        ):  # pragma: no cover
            a = dpnp.ascontiguousarray(a)

    # w/a for cuFFT to avoid "Invalid strides" error when
    # the last dimension is 1 and there are multiple axes
    # by swapping the last two axes to correct the input.
    # TODO: Remove this ones the OneMath issue is resolved
    # https://github.com/uxlfoundation/oneMath/issues/631
    cufft_wa = dpnp.is_cuda_backend(a) and a.shape[-1] == 1 and len(axes) > 1
    if cufft_wa:  # pragma: no cover
        a = dpnp.moveaxis(a, -1, -2)

    a_strides = _standardize_strides_to_nonzero(a.strides, a.shape)
    dsc, out_strides = _commit_descriptor(
        a, forward, in_place, c2c, a_strides, index, batch_fft
    )
    res = _compute_result(dsc, a, out, forward, c2c, out_strides)
    res = _scale_result(res, a.shape, norm, forward, index)

    # Revert swapped axes
    if cufft_wa:  # pragma: no cover
        res = dpnp.moveaxis(res, -1, -2)

    if batch_fft:
        tmp_shape = a_shape_orig[:-1] + (res.shape[-1],)
        res = dpnp.reshape(res, tmp_shape)
        res = dpnp.moveaxis(res, local_axes, axes)

    result = dpnp.get_result_array(res, out=out, casting="same_kind")
    if out is None and not (
        result.flags.c_contiguous or result.flags.f_contiguous
    ):
        result = dpnp.ascontiguousarray(result)

    return result


def _make_array_hermitian(a, axis, copy_needed):
    """
    For complex-to-real FFT, the input array should be Hermitian. If it is not,
    the behavior is undefined. This function makes necessary changes to make
    sure the given array is Hermitian.

    It is assumed that this function is called after `_cook_nd_args` and so
    `n` is always ``None``. It is also assumed that it is called after
    `_truncate_or_pad`, so the array has enough length.
    """

    a = dpnp.moveaxis(a, axis, 0)
    n = a.shape[0]

    # TODO: if the input array is already Hermitian, the following steps are
    # not needed, however, validating the input array is hermitian results in
    # synchronization of the SYCL queue, find an alternative.
    if copy_needed:
        a = a.astype(a.dtype, order="C", copy=True)

    a[0].imag = 0
    assert n is not None
    if n % 2 == 0:
        # Nyquist mode (n//2+1 mode) is n//2-th element
        f_ny = n // 2
        assert a.shape[0] > f_ny
        a[f_ny].imag = 0
    else:
        # No Nyquist mode
        pass

    return dpnp.moveaxis(a, 0, axis)


def _scale_result(res, a_shape, norm, forward, index):
    """Scale the result of the FFT according to `norm`."""
    if res.dtype in [dpnp.float32, dpnp.complex64]:
        dtype = dpnp.float32
    else:
        dtype = dpnp.float64
    scale = numpy.prod(a_shape[index:], dtype=dtype)
    norm_factor = 1
    if norm == "ortho":
        norm_factor = numpy.sqrt(scale)
    elif norm == "forward" and forward:
        norm_factor = scale
    elif norm in [None, "backward"] and not forward:
        norm_factor = scale

    res /= norm_factor
    return res


def _truncate_or_pad(a, shape, axes):
    """Truncating or zero-padding the input array along the specified axes."""

    for s, axis in zip(shape, axes):
        a_shape = list(a.shape)
        index = [slice(None)] * a.ndim
        if s == a_shape[axis]:
            pass
        elif s < a_shape[axis]:
            # truncating
            index[axis] = slice(0, s)
            a = a[tuple(index)]
        else:
            # zero-padding
            exec_q = a.sycl_queue
            index[axis] = slice(0, a_shape[axis])  # orig shape
            a_shape[axis] = s  # modified shape
            order = "F" if a.flags.fnc else "C"
            z = dpnp.zeros(
                a_shape,
                dtype=a.dtype,
                order=order,
                usm_type=a.usm_type,
                sycl_queue=exec_q,
            )
            _manager = dpu.SequentialOrderManager[exec_q]
            dep_evs = _manager.submitted_events
            ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                src=dpnp.get_usm_ndarray(a),
                dst=z.get_array()[tuple(index)],
                sycl_queue=exec_q,
                depends=dep_evs,
            )
            _manager.add_event_pair(ht_copy_ev, copy_ev)
            a = z

    return a


def _validate_out_keyword(a, out, s, axes, c2c, c2r, r2c):
    """Validate out keyword argument."""
    if out is not None:
        dpnp.check_supported_arrays_type(out)
        if (
            dpctl.utils.get_execution_queue((a.sycl_queue, out.sycl_queue))
            is None
        ):
            raise ExecutionPlacementError(
                "Input and output allocation queues are not compatible"
            )

        # validate out shape against the final shape,
        # intermediate shapes may vary
        expected_shape = list(a.shape)
        if r2c:
            expected_shape[axes[-1]] = s[-1] // 2 + 1
        elif c2c:
            expected_shape[axes[-1]] = s[-1]
        for s_i, axis in zip(s[-2::-1], axes[-2::-1]):
            expected_shape[axis] = s_i
        if c2r:
            expected_shape[axes[-1]] = s[-1]

        if out.shape != tuple(expected_shape):
            raise ValueError(
                "output array has incorrect shape, expected "
                f"{tuple(expected_shape)}, got {out.shape}."
            )

        # validate out data type
        if c2r:
            if not dpnp.issubdtype(out.dtype, dpnp.floating):
                raise TypeError(
                    "output array should have real floating data type."
                )
        else:  # c2c/r2c FFT
            if not dpnp.issubdtype(out.dtype, dpnp.complexfloating):
                raise TypeError("output array should have complex data type.")


def _validate_s_axes(a, s, axes):
    if axes is not None:
        # validate axes is a sequence and
        # each axis is an integer within the range
        normalize_axis_tuple(list(set(axes)), a.ndim, "axes")

    if s is not None:
        raise_error = False
        if isinstance(s, Sequence):
            if any(not isinstance(s_i, int) for s_i in s):
                raise_error = True
        else:
            raise_error = True

        if raise_error:
            raise TypeError("`s` must be `None` or a sequence of integers.")

        if axes is None:
            raise ValueError(
                "`axes` should not be `None` if `s` is not `None`."
            )


def dpnp_fft(a, forward, real, n=None, axis=-1, norm=None, out=None):
    """Calculates 1-D FFT of the input array along axis"""

    _check_norm(norm)
    a_orig = a
    a_ndim = a.ndim
    if a_ndim == 0:
        raise ValueError("Input array must be at least 1D")

    c2c = not real  # complex-to-complex FFT
    r2c = real and forward  # real-to-complex FFT
    c2r = real and not forward  # complex-to-real FFT
    if r2c and dpnp.issubdtype(a.dtype, dpnp.complexfloating):
        raise TypeError("Input array must be real")

    axis = normalize_axis_index(axis, a_ndim)
    if n is None:
        if c2r:
            n = (a.shape[axis] - 1) * 2
        else:
            n = a.shape[axis]
    elif not isinstance(n, int):
        raise TypeError("`n` should be None or an integer")
    if n < 1:
        raise ValueError(f"Invalid number of FFT data points ({n}) specified")

    a = _truncate_or_pad(a, (n,), (axis,))
    _validate_out_keyword(a, out, (n,), (axis,), c2c, c2r, r2c)
    # if input array is copied, in-place FFT can be used
    a, in_place = _copy_array(a, c2c or c2r)
    if not in_place and out is not None:
        # if input is also given for out, in-place FFT can be used
        in_place = dpnp.are_same_logical_tensors(a, out)

    if a.size == 0:
        return dpnp.get_result_array(a, out=out, casting="same_kind")

    if c2r:
        # input array should be Hermitian for c2r FFT
        a = _make_array_hermitian(
            a, axis, dpnp.are_same_logical_tensors(a, a_orig)
        )

    return _fft(
        a,
        norm=norm,
        out=out,
        forward=forward,
        # TODO: currently in-place is only implemented for c2c, see SAT-7154
        in_place=in_place and c2c,
        c2c=c2c,
        axes=axis,
        batch_fft=a_ndim != 1,
    )


def dpnp_fftn(a, forward, real, s=None, axes=None, norm=None, out=None):
    """Calculates N-D FFT of the input array along axes"""

    a_orig = a
    if isinstance(axes, Sequence) and len(axes) == 0:
        if real:
            raise IndexError("Empty axes.")

        return a

    _check_norm(norm)
    if a.ndim == 0:
        if axes is not None:
            raise IndexError(
                "Input array is 0-dimensional while axis is not `None`."
            )

        return a

    c2c = not real  # complex-to-complex FFT
    r2c = real and forward  # real-to-complex FFT
    c2r = real and not forward  # complex-to-real FFT
    if r2c and dpnp.issubdtype(a.dtype, dpnp.complexfloating):
        raise TypeError("Input array must be real")

    _validate_s_axes(a, s, axes)
    s, axes = _cook_nd_args(a, s, axes, c2r)
    _validate_out_keyword(a, out, s, axes, c2c, c2r, r2c)
    a, in_place = _copy_array(a, c2c or c2r)

    len_axes = len(axes)
    if len_axes == 1:
        a = _truncate_or_pad(a, (s[-1],), (axes[-1],))
        if c2r:
            a = _make_array_hermitian(
                a, axes[-1], dpnp.are_same_logical_tensors(a, a_orig)
            )
        return _fft(
            a, norm, out, forward, in_place and c2c, c2c, axes[-1], a.ndim != 1
        )

    if r2c:
        # a 1D real-to-complext FFT is performed on the last axis and then
        # an N-D complex-to-complex FFT over the remaining axes
        a = _truncate_or_pad(a, (s[-1],), (axes[-1],))
        a = _fft(
            a,
            norm=norm,
            # if out is used in an intermediate step, it will have memory
            # overlap with input and cannot be used in the final step (a new
            # result array will be created for the final step), so there is no
            # benefit in using out in an intermediate step
            out=None,
            forward=forward,
            in_place=in_place and c2c,
            c2c=c2c,
            axes=axes[-1],
            batch_fft=a.ndim != 1,
        )
        return _complex_nd_fft(
            a,
            s=s,
            norm=norm,
            out=out,
            forward=forward,
            in_place=in_place,
            c2c=True,
            axes=axes[:-1],
            batch_fft=a.ndim != len_axes - 1,
        )

    if c2r:
        # an N-D complex-to-complex FFT is performed on all axes except the
        # last one then a 1D complex-to-real FFT is performed on the last axis
        a = _complex_nd_fft(
            a,
            s=s,
            norm=norm,
            # out has real dtype and cannot be used in intermediate steps
            out=None,
            forward=forward,
            in_place=in_place,
            c2c=True,
            axes=axes[:-1],
            batch_fft=a.ndim != len_axes - 1,
        )
        a = _truncate_or_pad(a, (s[-1],), (axes[-1],))
        if c2r:
            a = _make_array_hermitian(
                a, axes[-1], dpnp.are_same_logical_tensors(a, a_orig)
            )
        return _fft(
            a, norm, out, forward, in_place and c2c, c2c, axes[-1], a.ndim != 1
        )

    # c2c
    return _complex_nd_fft(
        a, s, norm, out, forward, in_place, c2c, axes, a.ndim != len_axes
    )


def dpnp_fillfreq(a, m, n, val):
    """Fill an array with the sample frequencies"""

    exec_q = a.sycl_queue
    _manager = dpctl.utils.SequentialOrderManager[exec_q]

    # it's assumed there are no dependent events to populate the array
    ht_lin_ev, lin_ev = ti._linspace_step(0, 1, a[:m].get_array(), exec_q)
    _manager.add_event_pair(ht_lin_ev, lin_ev)

    ht_lin_ev, lin_ev = ti._linspace_step(m - n, 1, a[m:].get_array(), exec_q)
    _manager.add_event_pair(ht_lin_ev, lin_ev)
    return a * val


def swap_direction(norm):
    """Swap the direction of the FFT."""

    _check_norm(norm)
    _swap_direction_map = {
        "backward": "forward",
        None: "forward",
        "ortho": "ortho",
        "forward": "backward",
    }

    return _swap_direction_map[norm]
