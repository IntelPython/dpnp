# *****************************************************************************
# Copyright (c) 2024, Intel Corporation
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

import copy
import itertools
import operator
import warnings

import dpctl
import dpctl.tensor as dpt
import dpctl.tensor._tensor_elementwise_impl as tei
import dpctl.tensor._tensor_impl as ti
import numpy
from numpy.core.numeric import normalize_axis_tuple

import dpnp
import dpnp.backend.extensions.blas._blas_impl as bi
from dpnp.dpnp_array import dpnp_array
from dpnp.dpnp_utils import get_usm_allocations

_einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


__all__ = ["dpnp_cross", "dpnp_dot", "dpnp_einsum", "dpnp_kron", "dpnp_matmul"]


def _calc_offset(shape, linear_id, strides):
    """
    Calculate the offset in a multi-dimensional array given the shape, linear_id, and strides.

    Parameters
    ----------
    shape : tuple
        The shape of the multi-dimensional array.
    linear_id : int
        The linear index in the multi-dimensional array.
    strides : tuple
        The strides of the multi-dimensional array.

    Returns
    -------
    out : int
        The offset in the multi-dimensional array.

    """

    offset = 0
    indices = _index_linear_to_tuple(shape, linear_id)
    for i in range(len(indices)):
        offset += indices[i] * strides[i]
    return offset


def _chr(label):
    """
    Copied from _chr in cupy/core/_einsum.py

    Converts an integer label to a character representation.

    Parameters
    ----------
    label : int
        The integer label to be converted.

    Returns
    -------
    out : str
        A string representation of the label. If the label is negative,
        it returns a string in the format '...[label]', where label is the
        negative integer. Otherwise, it returns a single character string
        representing the label.

    Examples
    --------
    >>> import dpnp.dpnp_utils.dpnp_utils_linearalgebra as np_util
    >>> np_util._chr(97)
    'a'
    >>> np_util._chr(-1)
    '...[-1]'

    """

    if label < 0:
        return f"...[{label}]"
    else:
        return chr(label)


def _compute_res_dtype(*arrays, dtype, casting, sycl_queue):
    """
    Determines the output array data type and an intermediate data type
    used in performing calculations related to a specific math function.
    If dtype is ``None``, the output array data type of the operation is
    determined based on the Promotion Type Rule and device capabilities.
    Otherwise, `dtype` is used as output array dtype, if input arrays
    can cast to it according to the casting rule determined. If casting
    cannot be done, a ``TypeError`` is raised.
    The intermediate data type is the data type used for performing the math
    function calculations. If output array dtype is a floating-point data type,
    it is also used for the intermediate data type. If output array dtype is an
    integral data type, the default floating point data type of the device where
    input arrays are allocated on are used for intermediate data type.

    Parameters
    ----------
    arrays : {dpnp.ndarray, usm_ndarray}
        Input arrays.
    dtype : dtype
        If not ``None``, data type of the output array.
    casting : {"no", "equiv", "safe", "same_kind", "unsafe"}, optional
        Controls what kind of data casting may occur.
    sycl_queue : {SyclQueue}
        A SYCL queue to use for determining default floating point datat type.

    Returns
    -------
    compute_dtype, res_dtype :
        `compute_dtype` is the data type used in performing math function calculations.
        The input arrays of the math function are cast to `compute_dtype` and then
        the calculations are performed.
        `res_dtype` is the output data type. When the result is obtained, it is cast
        to `res_dtype`.

    """

    res_dtype = dpnp.result_type(*arrays)
    default_dtype = dpnp.default_float_type(sycl_queue=sycl_queue)

    if dtype is not None:
        if dpnp.can_cast(res_dtype, dtype, casting=casting):
            res_dtype = dtype
        else:
            raise TypeError(
                f"Cannot cast from dtype({res_dtype}) to dtype({dtype}) with casting rule {casting}"
            )

    compute_dtype = (
        res_dtype if dpnp.issubdtype(res_dtype, dpnp.inexact) else default_dtype
    )

    return compute_dtype, res_dtype


def _compute_size_by_dict(indices, idx_dict):
    """
    Copied from _compute_size_by_dict in numpy/core/einsumfunc.py

    Computes the product of the elements in `indices` based on the dictionary
    `idx_dict`.

    Parameters
    ----------
    indices : iterable
        Indices to base the product on.
    idx_dict : dictionary
        Dictionary of index sizes

    Returns
    -------
    ret : int
        The resulting product.

    Examples
    --------
    >>> import dpnp.dpnp_utils.dpnp_utils_linearalgebra as np_util
    >>> np_util._compute_size_by_dict("abbc", {"a": 2, "b":3, "c":5})
    90

    """
    ret = 1
    for i in indices:
        ret *= idx_dict[i]
    return ret


def _compute_size(start, shape):
    """
    Compute the total size of a multi-dimensional array starting from a given index.

    Parameters
    ----------
    start : int
        The starting index from which to compute the size.
    shape : tuple
        The shape of the multi-dimensional array.

    Returns
    -------
    out : int
        The total size of the array.

    """
    ret = 1
    for i in range(start, len(shape)):
        ret *= shape[i]
    return ret


def _copy_array(x, dep_events, host_events, copy_flag=False, dtype=None):
    """
    Creating a copy of input array if needed.

    If `copy_flag` is ``True``, a C-contiguous copy of input array is returned.
    In this case, the copy array has the input array data type unless `dtype` is
    determined.
    If `copy_flag` is ``False`` and input array data type is different than `dtype`,
    a C-contiguous copy of input array with specified `dtype` is returned.

    """

    if copy_flag:
        copy = copy_flag
    else:
        copy = x.dtype != dtype if dtype is not None else False

    if copy:
        x_copy = dpnp.empty_like(x, dtype=dtype, order="C")
        ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=dpnp.get_usm_ndarray(x),
            dst=x_copy.get_array(),
            sycl_queue=x.sycl_queue,
        )
        dep_events.append(copy_ev)
        host_events.append(ht_copy_ev)
        return x_copy
    return x


def _create_result_array(x1, x2, out, shape, dtype, usm_type, sycl_queue):
    """
    Create the result array.

    If `out` is not ``None`` and its features match the specified `shape`, `dtype,
    `usm_type`, and `sycl_queue` and it is C-contiguous or F-contiguous and
    does not have any memory overlap with `x1` and `x2`, `out` itself is returned.
    If these conditions are not satisfied, an empty array is returned with the
    specified `shape`, `dtype, `usm_type`, and `sycl_queue`.
    """

    if out is not None:
        x1_usm = dpnp.get_usm_ndarray(x1)
        x2_usm = dpnp.get_usm_ndarray(x2)
        out_usm = dpnp.get_usm_ndarray(out)
        contig_flag = _define_contig_flag(out)

        if (
            out.dtype == dtype
            and out.shape == shape
            and out.usm_type == usm_type
            and out.sycl_queue == sycl_queue
            and contig_flag
            and not ti._array_overlap(x1_usm, out_usm)
            and not ti._array_overlap(x2_usm, out_usm)
        ):
            return out

    return dpnp.empty(
        shape,
        dtype=dtype,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )


def _define_contig_flag(x):
    """
    Determines if the data in last two dimensions of array `x` are
    c_contiguous or f_contiguous. For 2D arrays, it is the same as using
    x.flags.c_contiguous or x.flags.f_contiguous.
    """

    flag = False
    x_strides = x.strides
    x_shape = x.shape
    if x.ndim < 2:
        return True

    x_strides = _standardize_strides_to_nonzero(x_strides, x_shape)
    x_is_c_contiguous = x_strides[-1] == 1 and x_strides[-2] == x_shape[-1]
    x_is_f_contiguous = x_strides[-2] == 1 and x_strides[-1] == x_shape[-2]
    if x_is_c_contiguous or x_is_f_contiguous:
        flag = True
    return flag


def _define_dim_flags(x, pos):
    """
    Define useful flags for the main calculation in dpnp_matmul.
    x_is_1D: `x` is 1D array or inherently 1D (all dimensions are equal to one
    except for one of them), for instance, if x.shape = (1, 1, 1, 2),
    then x_is_1D = True
    x_is_2D: `x` is 2D array or inherently 2D (all dimensions are equal to one
    except for the last two of them), for instance, if x.shape = (1, 1, 3, 2),
    then x_is_2D = True
    x_base_is_1D: `x` is 1D considering only its last two dimensions, for instance,
    if x.shape = (3, 4, 1, 2), then x_base_is_1D = True
    """

    index = -1 if pos == 0 else -2
    x_shape = x.shape
    x_ndim = x.ndim
    x_is_1D = x_ndim == 1
    if numpy.prod(x_shape) != 0:
        # the 2nd condition is only expected to be checked for
        # ND-arrays (N>1) since index is in [-2, -1]
        x_is_1D = x_is_1D or numpy.prod(x_shape) == x_shape[index]

    x_is_2D = False
    if not x_is_1D:
        x_is_2D = x_ndim == 2 or numpy.prod(x_shape[:-2]) == 1

    x_base_is_1D = x_is_1D
    if not x_is_1D:
        x_base_is_1D = x_shape[-1] == 1 or x_shape[-2] == 1

    return x_is_2D, x_is_1D, x_base_is_1D


def _einsum_diagonals(input_subscripts, operands):
    """
    Adopted from _einsum_diagonals in cupy/core/_einsum.py

    Compute the diagonal for each operand.

    Parameters
    ----------
    input_subscripts : tuple or list of str
        Strings representing the Einstein summation notation for each operand.
    operands : tuple or list of dpnp.ndarray or usm_ndarray
        Input arrays.

    Raises
    ------
    ValueError
        If dimensions in the operands for collapsing indices don't match.

    Notes
    -----
    This function mutates `input_subscripts` and `operands`.

    Examples
    --------
    >>> import dpnp as np
    >>> import dpnp.dpnp_utils.dpnp_utils_linearalgebra as np_util
    >>> a = np.arange(9).reshape(3, 3)
    >>> input_subscripts = ["ii"]
    >>> operands = [a]
    >>> np_util._einsum_diagonals(input_subscripts, operands)
    >>> input_subscripts
    [['i']]
    >>> operands
    [array([0, 4, 8])]

    """

    for idx in range(len(input_subscripts)):
        sub = input_subscripts[idx]
        arr = operands[idx]

        # repetitive index in the input_subscripts
        if len(set(sub)) < len(sub):
            axeses = {}
            for axis, label in enumerate(sub):
                axeses.setdefault(label, []).append(axis)

            axeses = list(axeses.items())
            for label, axes in axeses:
                dims = {arr.shape[axis] for axis in axes}
                if len(dims) >= 2:
                    dim1 = dims.pop()
                    dim0 = dims.pop()
                    raise ValueError(
                        f"dimensions in operand {idx} "
                        f"for collapsing index '{label}' don't match ({dim0} != {dim1})"
                    )

            sub, axeses = zip(*axeses)  # axeses is not empty
            input_subscripts[idx] = list(sub)
            operands[idx] = _transpose_ex(arr, axeses)


def _expand_dims_transpose(arr, mode, mode_out):
    """
    Copied from _expand_dims_transpose in cupy/core/_einsum.py

    Return a reshaped and transposed array.

    The input array `arr` having `mode` as its modes is reshaped and
    transposed so that modes of the output becomes `mode_out`.

    Parameters
    ----------
    arr : {dpnp.ndarray, usm_ndarray}
        Input array.
    mode : tuple or list
        The modes of input array.
    mode_out : tuple or list
        The modes of output array.

    Returns
    -------
    out : dpnp.ndarray
        The reshaped and transposed array.

    Example
    -------
    >>> import dpnp
    >>> import dpnp.dpnp_utils.dpnp_utils_linearalgebra as np_util
    >>> a = dpnp.zeros((10, 20))
    >>> mode_a = ("A", "B")
    >>> mode_out = ("B", "C", "A")
    >>> out = np_util._expand_dims_transpose(a, mode_a, mode_out)
    >>> out.shape
    (20, 1, 10)

    """
    mode = list(mode)
    shape = list(arr.shape)
    axes = []
    for i in mode_out:
        if i not in mode:
            mode.append(i)
            shape.append(1)
        axes.append(mode.index(i))
    return dpnp.transpose(arr.reshape(shape), axes)


def _find_contraction(positions, input_sets, output_set):
    """
    Copied from _find_contraction in numpy/core/einsumfunc.py

    Finds the contraction for a given set of input and output sets.

    Parameters
    ----------
    positions : iterable
        Integer positions of terms used in the contraction.
    input_sets : list of sets
        List of sets that represent the lhs side of the einsum subscript
    output_set : set
        Set that represents the rhs side of the overall einsum subscript

    Returns
    -------
    new_result : set
        The indices of the resulting contraction
    remaining : list of sets
        List of sets that have not been contracted, the new set is appended to
        the end of this list
    idx_removed : set
        Indices removed from the entire contraction
    idx_contraction : set
        The indices used in the current contraction

    Examples
    --------
    # A simple dot product test case
    >>> import dpnp.dpnp_utils.dpnp_utils_linearalgebra as np_util
    >>> pos = (0, 1)
    >>> isets = [set("ab"), set("bc")]
    >>> oset = set("ac")
    >>> np_util._find_contraction(pos, isets, oset)
    ({'a', 'c'}, [{'a', 'c'}], {'b'}, {'a', 'b', 'c'})

    # A more complex case with additional terms in the contraction
    >>> pos = (0, 2)
    >>> isets = [set("abd"), set("ac"), set("bdc")]
    >>> oset = set("ac")
    >>> np_util._find_contraction(pos, isets, oset)
     ({'a', 'c'}, [{'a', 'c'}, {'a', 'c'}], {'b', 'd'}, {'a', 'b', 'c', 'd'})

    """

    idx_contract = set()
    idx_remain = output_set.copy()
    remaining = []
    for ind, value in enumerate(input_sets):
        if ind in positions:
            idx_contract |= value
        else:
            remaining.append(value)
            idx_remain |= value

    new_result = idx_remain & idx_contract
    idx_removed = idx_contract - new_result
    remaining.append(new_result)

    return (new_result, remaining, idx_removed, idx_contract)


def _flatten_transpose(a, axeses):
    """
    Copied from _flatten_transpose in cupy/core/_einsum.py

    Transpose and flatten each

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axeses : sequence of sequences of ints
        Axeses

    Returns
    -------
    out : dpnp.ndarray
        `a` with its axes permutated and flatten.
    shapes : tuple
        flattened shapes.

    Examples
    --------
    >>> import dpnp as np
    >>> import dpnp.dpnp_utils.dpnp_utils_linearalgebra as np_util
    >>> a = np.arange(24).reshape(2, 3, 4)
    >>> axeses = [(0, 2), (1,)]
    >>> out, shapes = np_util._flatten_transpose(a, axeses)
    >>> out.shape
    (8, 3)
    >>> shapes
    [(2, 4), (3,)]

    """

    transpose_axes = []
    shapes = []
    for axes in axeses:
        transpose_axes.extend(axes)
        shapes.append([a.shape[axis] for axis in axes])

    return (
        dpnp.transpose(a, transpose_axes).reshape(
            tuple([int(numpy.prod(shape)) for shape in shapes])
        ),
        shapes,
    )


def _flop_count(idx_contraction, inner, num_terms, size_dictionary):
    """
    Copied from _flop_count in numpy/core/einsumfunc.py

    Computes the number of FLOPS in the contraction.

    Parameters
    ----------
    idx_contraction : iterable
        The indices involved in the contraction
    inner : bool
        Does this contraction require an inner product?
    num_terms : int
        The number of terms in a contraction
    size_dictionary : dict
        The size of each of the indices in idx_contraction

    Returns
    -------
    flop_count : int
        The total number of FLOPS required for the contraction.

    Examples
    --------
    >>> import dpnp.dpnp_utils.dpnp_utils_linearalgebra as np_util
    >>> np_util._flop_count("abc", False, 1, {"a": 2, "b":3, "c":5})
    30

    >>> np_util._flop_count("abc", True, 2, {"a": 2, "b":3, "c":5})
    60

    """

    overall_size = _compute_size_by_dict(idx_contraction, size_dictionary)
    op_factor = max(1, num_terms - 1)
    if inner:
        op_factor += 1

    return overall_size * op_factor


def _get_result_shape(x1, x2, out, np_flag):
    """
    Three task are completed in this function:
        - Get the shape of the result array.
        - Validate the shape of output array, if provided.
        - Align the input arrays if they could be broadcast together.
    """
    x1_ndim = x1.ndim
    x2_ndim = x2.ndim

    x1_shape = x1.shape
    x2_shape = x2.shape

    if x1_ndim == 0:
        raise ValueError(
            "Input array 0 does not have enough dimensions (has 0, but requires at least 1)"
        )
    if x2_ndim == 0:
        raise ValueError(
            "Input array 1 does not have enough dimensions (has 0, but requires at least 1)"
        )

    if x1_ndim == 1 and x2_ndim == 1:
        if x1_shape[-1] != x2_shape[-1]:
            _shape_error(x1_shape[-1], x2_shape[-1], None, err_msg=0)
        result_shape = ()
    elif x1_ndim == 1:
        if x1_shape[-1] != x2_shape[-2]:
            _shape_error(x1_shape[-1], x2_shape[-2], None, err_msg=0)
        result_shape = x2_shape[:-2] + (x2_shape[-1],)
    elif x2_ndim == 1:
        if x1_shape[-1] != x2_shape[-1]:
            _shape_error(x1_shape[-1], x2_shape[-1], None, err_msg=0)
        result_shape = x1_shape[:-1]
    else:  # at least 2D
        x1_is_2D, x1_is_1D, _ = _define_dim_flags(x1, pos=0)
        x2_is_2D, x2_is_1D, _ = _define_dim_flags(x2, pos=1)

        if x1_shape[-1] != x2_shape[-2]:
            _shape_error(x1_shape[-1], x2_shape[-2], None, err_msg=0)

        if x1_ndim == 2 and x2_ndim == 2:
            result_shape = (x1_shape[-2], x2_shape[-1])
        else:
            if x1_ndim != x2_ndim:
                diff = abs(x1_ndim - x2_ndim)
                if x1_ndim < x2_ndim:
                    x1 = dpnp.reshape(x1, ((1,) * diff + x1.shape))
                    x1_shape = x1.shape
                else:
                    x2 = dpnp.reshape(x2, ((1,) * diff + x2.shape))
                    x2_shape = x2.shape

            # examining the option to align inputs when their
            # shapes differ but the shape of one of them is 1
            # in that dimension (similar to braodcasting concept)
            tmp_shape = list(x1_shape[:-2])
            for i in range(len(tmp_shape)):
                if x1_shape[i] != x2_shape[i]:
                    if x1_shape[i] == 1:
                        tmp_shape[i] = x2_shape[i]
                        # If array `x1` is inherently 1D or 2D, there's
                        # no need to duplicate the data for the dimension
                        # with shape equal to one; gemv_batch or gemm_batch
                        # can handle it by using zero as the stride between
                        # different `x1` matrices
                        if not (x1_is_2D or x1_is_1D):
                            x1 = dpnp.repeat(x1, x2_shape[i], axis=i)
                    elif x2_shape[i] == 1:
                        if not (x2_is_2D or x2_is_1D):
                            x2 = dpnp.repeat(x2, x1_shape[i], axis=i)
                    else:
                        _shape_error(
                            x1_shape[:-2], x2_shape[:-2], None, err_msg=2
                        )

            result_shape = tuple(tmp_shape) + (x1.shape[-2], x2.shape[-1])

    if out is not None:
        out_shape = out.shape
        if out_shape != result_shape and not np_flag:
            len_out = len(out_shape)
            len_res = len(result_shape)
            if len_out != len_res:
                _shape_error(len_out, len_res, None, err_msg=4)
            for i in range(len_out):
                if out_shape[i] != result_shape[i]:
                    if i == len_out - 1:
                        _shape_error(
                            out_shape[i], result_shape[i], 1, err_msg=1
                        )
                    elif i == len_out - 2:
                        _shape_error(
                            out_shape[i], result_shape[i], 0, err_msg=1
                        )
                    else:
                        _shape_error(
                            out_shape[:-2], result_shape[:-2], None, err_msg=3
                        )

    return x1, x2, result_shape


def _gemm_batch_matmul(exec_q, x1, x2, res, dev_tasks_list):
    # arrays here are already at least 3D, make them 3D
    x1 = dpnp.reshape(x1, (-1, x1.shape[-2], x1.shape[-1]))
    x2 = dpnp.reshape(x2, (-1, x2.shape[-2], x2.shape[-1]))
    orig_shape = res.shape
    res = dpnp.reshape(res, (-1, res.shape[-2], res.shape[-1]))

    ht_tasks_list = []
    # gemm_batch does not handle negative strides, make a copy if needed
    x1 = _copy_array(
        x1, dev_tasks_list, ht_tasks_list, copy_flag=x1.strides[0] < 0
    )
    x2 = _copy_array(
        x2, dev_tasks_list, ht_tasks_list, copy_flag=x2.strides[0] < 0
    )
    res = _copy_array(
        res, dev_tasks_list, ht_tasks_list, copy_flag=res.strides[0] < 0
    )
    # onemkl::blas::gemm_bacth throws an exception (Provided range is out
    # of integer limits) if the batch_size is too large (>=4096*4096), so
    # we need to split the batch into smaller chunks
    chunk = 2048 * 2048
    batch_size = res.shape[0]
    for i in range(0, batch_size, chunk):
        x1_usm = dpnp.get_usm_ndarray(x1[i : i + chunk, ...])
        x2_usm = dpnp.get_usm_ndarray(x2[i : i + chunk, ...])
        res_usm = dpnp.get_usm_ndarray(res[i : i + chunk, ...])
        ht_blas_ev, _, row_major = bi._gemm_batch(
            exec_q,
            x1_usm,
            x2_usm,
            res_usm,
            dev_tasks_list,
        )
        ht_tasks_list.append(ht_blas_ev)
    dpctl.SyclEvent.wait_for(ht_tasks_list)
    res_shape = res.shape
    if not row_major:
        res = dpnp.reshape(
            res.ravel(), (batch_size, res_shape[2], res_shape[1])
        ).transpose(0, 2, 1)

    if res_shape != orig_shape:
        res = res.reshape(orig_shape)

    res = dpnp.ascontiguousarray(res)
    return res


def _gemm_matmul(exec_q, x1, x2, res, dev_tasks_list):
    ht_blas_ev, _, row_major = bi._gemm(
        exec_q,
        dpnp.get_usm_ndarray(x1),
        dpnp.get_usm_ndarray(x2),
        dpnp.get_usm_ndarray(res),
        dev_tasks_list,
    )
    ht_blas_ev.wait()

    if not row_major:
        # TODO: investigate the possibility of defining result
        # array with "F" order for this case
        res = dpnp.ascontiguousarray(
            dpnp.reshape(res.ravel(), res.shape, order="F")
        )

    return res


def _greedy_path(input_sets, output_set, idx_dict, memory_limit):
    """
    Copied from _greedy_path in numpy/core/einsumfunc.py

    Finds the path by contracting the best pair until the input list is
    exhausted. The best pair is found by minimizing the tuple
    ``(-prod(indices_removed), cost)``.  What this amounts to is prioritizing
    matrix multiplication or inner product operations, then Hadamard like
    operations, and finally outer operations. Outer products are limited by
    ``memory_limit``. This algorithm scales cubically with respect to the
    number of elements in the list ``input_sets``.

    Parameters
    ----------
    input_sets : list of sets
        List of sets that represent the lhs side of the einsum subscript
    output_set : set
        Set that represents the rhs side of the overall einsum subscript
    idx_dict : dictionary
        Dictionary of index sizes
    memory_limit : int
        The maximum number of elements in a temporary array

    Returns
    -------
    path : list
        The greedy contraction order within the memory limit constraint.

    Examples
    --------
    >>> import dpnp.dpnp_utils.dpnp_utils_linearalgebra as np_util
    >>> isets = [set("abd"), set("ac"), set("bdc")]
    >>> oset = set("")
    >>> idx_sizes = {"a": 1, "b":2, "c":3, "d":4}
    >>> _greedy_path(isets, oset, idx_sizes, 5000)
    [(0, 2), (0, 1)]

    """

    # Handle trivial cases that leaked through
    if len(input_sets) == 1:
        return [(0,)]
    elif len(input_sets) == 2:
        return [(0, 1)]

    # Build up a naive cost
    _, _, idx_removed, idx_contract = _find_contraction(
        range(len(input_sets)), input_sets, output_set
    )
    naive_cost = _flop_count(
        idx_contract, idx_removed, len(input_sets), idx_dict
    )

    # Initially iterate over all pairs
    comb_iter = itertools.combinations(range(len(input_sets)), 2)
    known_contractions = []

    path_cost = 0
    path = []
    for _ in range(len(input_sets) - 1):
        # Iterate over all pairs on first step, only previously found pairs on subsequent steps
        for positions in comb_iter:
            # Always initially ignore outer products
            if input_sets[positions[0]].isdisjoint(input_sets[positions[1]]):
                continue

            result = _parse_possible_contraction(
                positions,
                input_sets,
                output_set,
                idx_dict,
                memory_limit,
                path_cost,
                naive_cost,
            )
            if result is not None:
                known_contractions.append(result)

        # If we do not have a inner contraction, rescan pairs including outer products
        if len(known_contractions) == 0:
            # Then check the outer products
            for positions in itertools.combinations(range(len(input_sets)), 2):
                result = _parse_possible_contraction(
                    positions,
                    input_sets,
                    output_set,
                    idx_dict,
                    memory_limit,
                    path_cost,
                    naive_cost,
                )
                if result is not None:
                    known_contractions.append(result)

            # If we still did not find any remaining contractions, default back to einsum like behavior
            if len(known_contractions) == 0:
                path.append(tuple(range(len(input_sets))))
                break

        # Sort based on first index
        best = min(known_contractions, key=lambda x: x[0])

        # Now propagate as many unused contractions as possible to next iteration
        known_contractions = _update_other_results(known_contractions, best)

        # Next iteration only compute contractions with the new tensor
        # All other contractions have been accounted for
        input_sets = best[2]
        new_tensor_pos = len(input_sets) - 1
        comb_iter = ((i, new_tensor_pos) for i in range(new_tensor_pos))

        # Update path and total cost
        path.append(best[1])
        path_cost += best[0][1]

    return path


def _index_linear_to_tuple(shape, linear_id):
    """
    Convert a linear index to a tuple of indices in a multi-dimensional array.

    Parameters
    ----------
    shape : tuple
        The shape of the multi-dimensional array.
    linear_id : int
        The linear index to convert.

    Returns
    -------
    out: tuple
        A tuple of indices corresponding to the linear index.

    """

    len_shape = len(shape)
    indices = [0] * len_shape
    for i in range(len_shape):
        prod_res = _compute_size(i + 1, shape)
        indices[i] = linear_id // prod_res
        linear_id %= prod_res

    return tuple(indices)


def _iter_path_pairs(path):
    """
    Copied from _iter_path_pairs in cupy/core/_einsum.py

    Decompose path into binary path

    Parameters
    ----------
    path : sequence of tuples of ints

    Yields
    ------
    tuple of ints
        pair (idx0, idx1) that represents the operation
        {pop(idx0); pop(idx1); append();}

    """

    for indices in path:
        assert all(idx >= 0 for idx in indices)
        # [3, 1, 4, 9] -> [(9, 4), (-1, 3), (-1, 1)]
        if len(indices) >= 2:
            indices = sorted(indices, reverse=True)
            yield indices[0], indices[1]
            for idx in indices[2:]:
                yield -1, idx


def _make_transpose_axes(sub, b_dims, c_dims):
    """Copied from _make_transpose_axes in cupy/core/_einsum.py"""
    bs = []
    cs = []
    ts = []
    for axis, label in enumerate(sub):
        if label in b_dims:
            bs.append((label, axis))
        elif label in c_dims:
            cs.append((label, axis))
        else:
            ts.append((label, axis))
    return (
        _tuple_sorted_by_0(bs),
        _tuple_sorted_by_0(cs),
        _tuple_sorted_by_0(ts),
    )


def _optimal_path(input_sets, output_set, idx_dict, memory_limit):
    """
    Copied from _optimal_path in numpy/core/einsumfunc.py

    Computes all possible pair contractions, sieves the results based
    on ``memory_limit`` and returns the lowest cost path. This algorithm
    scales factorial with respect to the elements in the list ``input_sets``.

    Parameters
    ----------
    input_sets : list of sets
        List of sets that represent the lhs side of the einsum subscript
    output_set : set
        Set that represents the rhs side of the overall einsum subscript
    idx_dict : dictionary
        Dictionary of index sizes
    memory_limit : int
        The maximum number of elements in a temporary array

    Returns
    -------
    path : list
        The optimal contraction order within the memory limit constraint.

    Examples
    --------
    >>> import dpnp.dpnp_utils.dpnp_utils_linearalgebra as np_util
    >>> isets = [set("abd"), set("ac"), set("bdc")]
    >>> oset = set("")
    >>> idx_sizes = {"a": 1, "b":2, "c":3, "d":4}
    >>> np_util._optimal_path(isets, oset, idx_sizes, 5000)
    [(0, 2), (0, 1)]

    """

    full_results = [(0, [], input_sets)]
    for iteration in range(len(input_sets) - 1):
        iter_results = []

        # Compute all unique pairs
        for curr in full_results:
            cost, positions, remaining = curr
            for con in itertools.combinations(
                range(len(input_sets) - iteration), 2
            ):
                # Find the contraction
                cont = _find_contraction(con, remaining, output_set)
                new_result, new_input_sets, idx_removed, idx_contract = cont

                # Sieve the results based on memory_limit
                new_size = _compute_size_by_dict(new_result, idx_dict)
                if new_size > memory_limit:
                    continue

                # Build (total_cost, positions, indices_remaining)
                total_cost = cost + _flop_count(
                    idx_contract, idx_removed, len(con), idx_dict
                )
                new_pos = positions + [con]
                iter_results.append((total_cost, new_pos, new_input_sets))

        # Update combinatorial list, if we did not find anything return best
        # path + remaining contractions
        if iter_results:
            full_results = iter_results
        else:
            path = min(full_results, key=lambda x: x[0])[1]
            path += [tuple(range(len(input_sets) - iteration))]
            return path

    path = min(full_results, key=lambda x: x[0])[1]
    return path


def _parse_einsum_input(args):
    """
    Copied from _parse_einsum_input in cupy/core/_einsum.py

    Parse einsum operands.

    Parameters
    ----------
    args : tuple
        The non-keyword arguments to einsum.

    Returns
    -------
    input_strings : str
        Parsed input strings.
    output_string : str
        Parsed output string.
    operands : list of array_like
        The operands to use in the contraction.

    Examples
    --------
    The operand list is simplified to reduce printing:

    >>> import dpnp as np
    >>> import dpnp.dpnp_utils.dpnp_utils_linearalgebra as np_util
    >>> a = np.random.rand(4, 4)
    >>> b = np.random.rand(4, 4, 4)
    >>> np_util._parse_einsum_input(("...a,...a->...", a, b))
    (['@a', '@a'], '@', [a, b])

    >>> np_util._parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))
    (['@a', '@b'], None, [a, b])

    """

    if len(args) == 0:
        raise ValueError(
            "must specify the einstein sum subscripts string and at least one "
            "operand, or at least one operand and its corresponding "
            "subscripts list"
        )

    if isinstance(args[0], str):
        subscripts = args[0].replace(" ", "")
        operands = list(args[1:])

        # Ensure all characters are valid
        for s in subscripts:
            if s in ".,->":
                continue
            if s not in _einsum_symbols:
                raise ValueError(
                    f"invalid subscript '{s}' in einstein sum subscripts "
                    "string, subscripts must be letters"
                )

        # Parse "..."
        subscripts = subscripts.replace("...", "@")
        if "." in subscripts:
            raise ValueError(
                "einstein sum subscripts string contains a '.' that is not "
                "part of an ellipsis ('...')"
            )

        # Parse "->"
        if ("-" in subscripts) or (">" in subscripts):
            # Check for proper "->"
            invalid = subscripts.count("-") > 1 or subscripts.count(">") > 1
            subscripts = subscripts.split("->")
            if invalid or len(subscripts) != 2:
                raise ValueError(
                    "einstein sum subscript string does not contain proper "
                    "'->' output specified"
                )
            input_subscripts, output_subscript = subscripts

        else:
            input_subscripts = subscripts
            output_subscript = None

        input_subscripts = input_subscripts.split(",")
        if len(input_subscripts) != len(operands):
            msg = "more" if len(operands) > len(input_subscripts) else "fewer"
            raise ValueError(
                msg + " operands provided to einstein sum function than "
                "specified in the subscripts string"
            )
    else:
        args = list(args)
        operands = []
        input_subscripts = []
        while len(args) >= 2:
            operands.append(args.pop(0))
            input_subscripts.append(_parse_int_subscript(args.pop(0)))
        if args:
            output_subscript = _parse_int_subscript(args[0])
        else:
            output_subscript = None

    return input_subscripts, output_subscript, operands


def _parse_ellipsis_subscript(subscript, idx, ndim=None, ellipsis_len=None):
    """
    Copied from _parse_ellipsis_subscript in cupy/core/_einsum.py

    Parse a subscript that may contain ellipsis

    Parameters
    ----------
    subscript : str
        An einsum subscript of an operand or an output. "..."
        should be replaced by "@".
    idx : {int, ``None``}
        For error messages, give int idx for the idx-th operand or ``None``
        for the output.
    ndim : int, optional
        ndim of the operand
    ellipsis_len : int, optional
        number of broadcast dimensions of the output.

    Returns
    -------
    out : list of ints
        The parsed subscript

    """
    subs = subscript.split("@")
    if len(subs) == 1:
        (sub,) = subs
        if ndim is not None and len(sub) != ndim:
            if len(sub) > ndim:
                raise ValueError(
                    f"einstein sum subscripts string {sub} contains too many "
                    f"subscripts for operand {idx}"
                )
            raise ValueError(
                f"operand {idx} has more dimensions than subscripts string "
                f"{sub} given in einstein sum, but no '...' ellipsis "
                "provided to broadcast the extra dimensions."
            )
        return [ord(label) for label in sub]
    elif len(subs) == 2:
        left_sub, right_sub = subs
        if ndim is not None:
            ellipsis_len = ndim - (len(left_sub) + len(right_sub))
        if ellipsis_len < 0:
            raise ValueError(
                f"einstein sum subscripts string {left_sub}...{right_sub} "
                f"contains too many subscripts for operand {idx}"
            )
        ret = []
        ret.extend(ord(label) for label in left_sub)
        ret.extend(range(-ellipsis_len, 0))
        ret.extend(ord(label) for label in right_sub)
        return ret
    else:
        # >= 2 ellipses for an operand
        raise ValueError(
            "einstein sum subscripts string contains a '.' that is not "
            "part of an ellipsis ('...') "
            + ("in the output" if idx is None else f"for operand {idx}")
        )


def _parse_int_subscript(list_subscript):
    """Copied from _parse_int_subscript in cupy/core/_einsum.py"""

    str_subscript = ""
    for s in list_subscript:
        if s is Ellipsis:
            str_subscript += "@"
        else:
            try:
                s = operator.index(s)
            except TypeError as e:
                raise TypeError(
                    "For this input type lists must contain "
                    "either int or Ellipsis"
                ) from e
            str_subscript += _einsum_symbols[s]
    return str_subscript


def _parse_possible_contraction(
    positions,
    input_sets,
    output_set,
    idx_dict,
    memory_limit,
    path_cost,
    naive_cost,
):
    """
    Copied from _parse_possible_contraction in numpy/core/einsumfunc.py

    Compute the cost (removed size + flops) and resultant indices for
    performing the contraction specified by ``positions``.

    Parameters
    ----------
    positions : tuple of int
        The locations of the proposed tensors to contract.
    input_sets : list of sets
        The indices found on each tensors.
    output_set : set
        The output indices of the expression.
    idx_dict : dict
        Mapping of each index to its size.
    memory_limit : int
        The total allowed size for an intermediary tensor.
    path_cost : int
        The contraction cost so far.
    naive_cost : int
        The cost of the unoptimized expression.

    Returns
    -------
    cost : (int, int)
        A tuple containing the size of any indices removed, and the flop cost.
    positions : tuple of int
        The locations of the proposed tensors to contract.
    new_input_sets : list of sets
        The resulting new list of indices if this proposed contraction is performed.

    """

    # Find the contraction
    contract = _find_contraction(positions, input_sets, output_set)
    idx_result, new_input_sets, idx_removed, idx_contract = contract

    # Sieve the results based on memory_limit
    new_size = _compute_size_by_dict(idx_result, idx_dict)
    if new_size > memory_limit:
        return None

    # Build sort tuple
    old_sizes = (
        _compute_size_by_dict(input_sets[p], idx_dict) for p in positions
    )
    removed_size = sum(old_sizes) - new_size

    # NB: removed_size used to be just the size of any removed indices i.e.:
    #     helpers.compute_size_by_dict(idx_removed, idx_dict)
    cost = _flop_count(idx_contract, idx_removed, len(positions), idx_dict)
    sort = (-removed_size, cost)

    # Sieve based on total cost as well
    if (path_cost + cost) >= naive_cost:
        return None

    # Add contraction to possible choices
    return [sort, positions, new_input_sets]


def _reduced_binary_einsum(arr0, sub0, arr1, sub1, sub_others):
    """Copied from _reduced_binary_einsum in cupy/core/_einsum.py"""

    set0 = set(sub0)
    set1 = set(sub1)
    assert len(set0) == len(sub0), "operand 0 should be reduced: diagonal"
    assert len(set1) == len(sub1), "operand 1 should be reduced: diagonal"

    if len(sub0) == 0 or len(sub1) == 0:
        return arr0 * arr1, sub0 + sub1

    set_others = set(sub_others)
    shared = set0 & set1
    batch_dims = shared & set_others
    contract_dims = shared - batch_dims

    bs0, cs0, ts0 = _make_transpose_axes(sub0, batch_dims, contract_dims)
    bs1, cs1, ts1 = _make_transpose_axes(sub1, batch_dims, contract_dims)

    sub_b = [sub0[axis] for axis in bs0]
    assert sub_b == [sub1[axis] for axis in bs1]
    sub_l = [sub0[axis] for axis in ts0]
    sub_r = [sub1[axis] for axis in ts1]

    sub_out = sub_b + sub_l + sub_r
    assert set(sub_out) <= set_others, "operands should be reduced: unary sum"

    if len(contract_dims) == 0:
        # Use element-wise multiply when no contraction is needed
        if len(sub_out) == len(sub_others):
            # to assure final output of einsum is C-contiguous
            sub_out = sub_others
        arr0 = _expand_dims_transpose(arr0, sub0, sub_out)
        arr1 = _expand_dims_transpose(arr1, sub1, sub_out)
        return arr0 * arr1, sub_out

    tmp0, shapes0 = _flatten_transpose(arr0, [bs0, ts0, cs0])
    tmp1, shapes1 = _flatten_transpose(arr1, [bs1, cs1, ts1])
    shapes_out = shapes0[0] + shapes0[1] + shapes1[2]
    assert shapes0[0] == shapes1[0]
    arr_out = dpnp.matmul(tmp0, tmp1).reshape(shapes_out)
    return arr_out, sub_out


def _shape_error(a, b, core_dim, err_msg):
    if err_msg == 0:
        raise ValueError(
            "Input arrays have a mismatch in their core dimensions. "
            "The core dimensions should follow this signature: "
            f"(n?,k),(k,m?)->(n?,m?) (size {a} is different from {b})"
        )
    elif err_msg == 1:
        raise ValueError(
            f"Output array has a mismatch in its core dimension {core_dim}. "
            "The core dimensions should follow this signature: "
            f"(n?,k),(k,m?)->(n?,m?) (size {a} is different from {b})"
        )
    elif err_msg == 2:
        raise ValueError(
            "Leading element(s) of the input arrays' shape do not match and "
            "they could not be broadcast together, the leading element(s) of "
            f"the input array 0 is {a} and it is different from leading "
            f"element(s) of the input array 1 which is {b}."
        )
    elif err_msg == 3:
        raise ValueError(
            "The shape of the output array does not have similar leading "
            "elements to the shape of input arrays, the leading element(s) of "
            f"the output array is {a} and it is different from the leading "
            f"element(s) of the input arrays which is {b}."
        )
    elif err_msg == 4:
        raise ValueError(
            "Output array does not have enough dimensions "
            f"(has {a} while requires {b})"
        )


def _standardize_strides_to_nonzero(strides, shape):
    """
    Standardizing the strides.
    When shape of an array along any particular dimension is 1, the stride
    along that dimension is undefined. This function standardize the strides
    by calculating the non-zero value of the strides.
    """

    ndim = len(strides)
    if numpy.prod(strides) == 0:
        stndrd_strides = tuple(
            numpy.prod(shape[i + 1 :]) if strides[i] == 0 else strides[i]
            for i in range(ndim - 1)
        )
        last_stride = 1 if strides[ndim - 1] == 0 else strides[ndim - 1]
        stndrd_strides += (last_stride,)
    else:
        stndrd_strides = strides

    return stndrd_strides


def _transpose_ex(a, axeses):
    """
    Copied from _transpose_ex in cupy/core/_einsum.py

    Transpose and diagonal

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axeses : sequence of sequences of ints
        Axeses

    Returns
    -------
    out : dpnp.ndarray
        `a` with its axes permutated. A writeable view is returned
        whenever possible.
    """

    shape = []
    strides = []
    for axes in axeses:
        shape.append(a.shape[axes[0]] if axes else 1)
        stride = sum(a.strides[axis] for axis in axes)
        strides.append(stride)

    # TODO: replace with a.view() when it is implemented in dpnp
    a = _view_work_around(a, shape, strides)
    return a


def _tuple_sorted_by_0(zs):
    """Copied from _tuple_sorted_by_0 in cupy/core/_einsum.py"""
    return tuple(i for _, i in sorted(zs))


def _update_other_results(results, best):
    """
    Copied from _update_other_results in numpy/core/einsumfunc.py

    Update the positions and provisional input_sets of ``results`` based on
    performing the contraction result ``best``. Remove any involving the tensors
    contracted.

    Parameters
    ----------
    results : list
        List of contraction results produced by ``_parse_possible_contraction``.
    best : list
        The best contraction of ``results`` i.e. the one that will be performed.

    Returns
    -------
    mod_results : list
        The list of modified results, updated with outcome of ``best`` contraction.
    """

    best_con = best[1]
    bx, by = best_con
    mod_results = []

    for cost, (x, y), con_sets in results:
        # Ignore results involving tensors just contracted
        if x in best_con or y in best_con:
            continue

        # Update the input_sets
        del con_sets[by - int(by > x) - int(by > y)]
        del con_sets[bx - int(bx > x) - int(bx > y)]
        con_sets.insert(-1, best[2][-1])

        # Update the position indices
        mod_con = x - int(x > bx) - int(x > by), y - int(y > bx) - int(y > by)
        mod_results.append((cost, mod_con, con_sets))

    return mod_results


def _validate_axes(x1, x2, axes):
    """Check axes is valid for matmul function."""

    def _validate_internal(axes, i, ndim):
        if ndim == 1:
            iter = 1
            if isinstance(axes, int):
                axes = (axes,)
            elif not isinstance(axes, tuple):
                raise TypeError(
                    f"Axes item {i}: {type(axes)} object cannot be interpreted as an integer."
                )

            if len(axes) != 1:
                raise ValueError(
                    f"Axes item {i} should be a tuple with a single element, or an integer."
                )
        else:
            iter = 2
            if not isinstance(axes, tuple):
                raise TypeError(f"Axes item {i} should be a tuple.")
            if len(axes) != 2:
                raise ValueError(
                    f"Axes item {i} should be a tuple with 2 elements."
                )

        for j in range(iter):
            if not isinstance(axes[j], int):
                raise TypeError(
                    f"Axes item {i}: {type(axes[j])} object cannot be interpreted as an integer."
                )
        return axes

    if not isinstance(axes, list):
        raise TypeError("Axes should be a list.")
    else:
        if len(axes) != 3:
            raise ValueError(
                "Axes should be a list of three tuples for inputs and output."
            )

    axes[0] = _validate_internal(axes[0], 0, x1.ndim)
    axes[1] = _validate_internal(axes[1], 1, x2.ndim)

    if x1.ndim == 1 and x2.ndim == 1:
        if axes[2] != ():
            raise TypeError("Axes item 2 should be an empty tuple.")
    elif x1.ndim == 1 or x2.ndim == 1:
        axes[2] = _validate_internal(axes[2], 2, 1)
    else:
        axes[2] = _validate_internal(axes[2], 2, 2)

    return axes


def _view_work_around(a, shape, strides):
    """
    Create a copy of the input array with the specified shape and strides.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        The input array.
    shape : tuple
        The desired shape of the output array.
    strides : tuple
        The desired strides of the output array.

    Returns
    -------
    out : dpnp.ndarray
        A copy of the input array with the specified shape and strides.

    """

    n_size = numpy.prod(shape)
    b = dpnp.empty(
        n_size, dtype=a.dtype, usm_type=a.usm_type, sycl_queue=a.sycl_queue
    )
    for linear_id in range(n_size):
        offset = _calc_offset(shape, linear_id, strides)
        indices = _index_linear_to_tuple(a.shape, offset)
        b[linear_id] = a[indices]
    b = b.reshape(tuple(shape))

    return b


def dpnp_cross(a, b, cp, exec_q):
    """Return the cross product of two (arrays of) vectors."""

    # create local aliases for readability
    a0 = a[..., 0]
    a1 = a[..., 1]
    if a.shape[-1] == 3:
        a2 = a[..., 2]
    b0 = b[..., 0]
    b1 = b[..., 1]
    if b.shape[-1] == 3:
        b2 = b[..., 2]
    if cp.ndim != 0 and cp.shape[-1] == 3:
        cp0 = cp[..., 0]
        cp1 = cp[..., 1]
        cp2 = cp[..., 2]

    host_events = []
    if a.shape[-1] == 2:
        if b.shape[-1] == 2:
            # a0 * b1 - a1 * b0
            cp_usm = dpnp.get_usm_ndarray(cp)
            ht_ev1, dev_ev1 = tei._multiply(
                dpnp.get_usm_ndarray(a0),
                dpnp.get_usm_ndarray(b1),
                cp_usm,
                exec_q,
            )
            host_events.append(ht_ev1)
            tmp = dpt.empty_like(cp_usm)
            ht_ev2, dev_ev2 = tei._multiply(
                dpnp.get_usm_ndarray(a1), dpnp.get_usm_ndarray(b0), tmp, exec_q
            )
            host_events.append(ht_ev2)
            ht_ev3, _ = tei._subtract_inplace(
                cp_usm, tmp, exec_q, [dev_ev1, dev_ev2]
            )
            host_events.append(ht_ev3)
        else:
            assert b.shape[-1] == 3
            # cp0 = a1 * b2 - 0  (a2 = 0)
            # cp1 = 0 - a0 * b2  (a2 = 0)
            # cp2 = a0 * b1 - a1 * b0
            cp1_usm = dpnp.get_usm_ndarray(cp1)
            cp2_usm = dpnp.get_usm_ndarray(cp2)
            a1_usm = dpnp.get_usm_ndarray(a1)
            b2_usm = dpnp.get_usm_ndarray(b2)
            ht_ev1, _ = tei._multiply(
                a1_usm, b2_usm, dpnp.get_usm_ndarray(cp0), exec_q
            )
            host_events.append(ht_ev1)
            ht_ev2, dev_ev2 = tei._multiply(
                dpnp.get_usm_ndarray(a0), b2_usm, cp1_usm, exec_q
            )
            host_events.append(ht_ev2)
            ht_ev3, _ = tei._negative(cp1_usm, cp1_usm, exec_q, [dev_ev2])
            host_events.append(ht_ev3)
            ht_ev4, dev_ev4 = tei._multiply(
                dpnp.get_usm_ndarray(a0),
                dpnp.get_usm_ndarray(b1),
                cp2_usm,
                exec_q,
            )
            host_events.append(ht_ev4)
            tmp = dpt.empty_like(cp2_usm)
            ht_ev5, dev_ev5 = tei._multiply(
                a1_usm, dpnp.get_usm_ndarray(b0), tmp, exec_q
            )
            host_events.append(ht_ev5)
            ht_ev6, _ = tei._subtract_inplace(
                cp2_usm, tmp, exec_q, [dev_ev4, dev_ev5]
            )
            host_events.append(ht_ev6)
    else:
        assert a.shape[-1] == 3
        if b.shape[-1] == 3:
            # cp0 = a1 * b2 - a2 * b1
            # cp1 = a2 * b0 - a0 * b2
            # cp2 = a0 * b1 - a1 * b0
            cp0_usm = dpnp.get_usm_ndarray(cp0)
            cp1_usm = dpnp.get_usm_ndarray(cp1)
            cp2_usm = dpnp.get_usm_ndarray(cp2)
            a0_usm = dpnp.get_usm_ndarray(a0)
            a1_usm = dpnp.get_usm_ndarray(a1)
            a2_usm = dpnp.get_usm_ndarray(a2)
            b0_usm = dpnp.get_usm_ndarray(b0)
            b1_usm = dpnp.get_usm_ndarray(b1)
            b2_usm = dpnp.get_usm_ndarray(b2)
            ht_ev1, dev_ev1 = tei._multiply(a1_usm, b2_usm, cp0_usm, exec_q)
            host_events.append(ht_ev1)
            tmp = dpt.empty_like(cp0_usm)
            ht_ev2, dev_ev2 = tei._multiply(a2_usm, b1_usm, tmp, exec_q)
            host_events.append(ht_ev2)
            ht_ev3, dev_ev3 = tei._subtract_inplace(
                cp0_usm, tmp, exec_q, [dev_ev1, dev_ev2]
            )
            host_events.append(ht_ev3)
            ht_ev4, dev_ev4 = tei._multiply(a2_usm, b0_usm, cp1_usm, exec_q)
            host_events.append(ht_ev4)
            ht_ev5, dev_ev5 = tei._multiply(
                a0_usm, b2_usm, tmp, exec_q, [dev_ev3]
            )
            host_events.append(ht_ev5)
            ht_ev6, dev_ev6 = tei._subtract_inplace(
                cp1_usm, tmp, exec_q, [dev_ev4, dev_ev5]
            )
            host_events.append(ht_ev6)
            ht_ev7, dev_ev7 = tei._multiply(a0_usm, b1_usm, cp2_usm, exec_q)
            host_events.append(ht_ev7)
            ht_ev8, dev_ev8 = tei._multiply(
                a1_usm, b0_usm, tmp, exec_q, [dev_ev6]
            )
            host_events.append(ht_ev8)
            ht_ev9, _ = tei._subtract_inplace(
                cp2_usm, tmp, exec_q, [dev_ev7, dev_ev8]
            )
            host_events.append(ht_ev9)
        else:
            assert b.shape[-1] == 2
            # cp0 = 0 - a2 * b1  (b2 = 0)
            # cp1 = a2 * b0 - 0  (b2 = 0)
            # cp2 = a0 * b1 - a1 * b0
            cp0_usm = dpnp.get_usm_ndarray(cp0)
            cp2_usm = dpnp.get_usm_ndarray(cp2)
            a2_usm = dpnp.get_usm_ndarray(a2)
            b1_usm = dpnp.get_usm_ndarray(b1)
            ht_ev1, dev_ev1 = tei._multiply(a2_usm, b1_usm, cp0_usm, exec_q)
            host_events.append(ht_ev1)
            ht_ev2, _ = tei._negative(cp0_usm, cp0_usm, exec_q, [dev_ev1])
            host_events.append(ht_ev2)
            ht_ev3, _ = tei._multiply(
                a2_usm,
                dpnp.get_usm_ndarray(b0),
                dpnp.get_usm_ndarray(cp1),
                exec_q,
            )
            host_events.append(ht_ev3)
            ht_ev4, dev_ev4 = tei._multiply(
                dpnp.get_usm_ndarray(a0), b1_usm, cp2_usm, exec_q
            )
            host_events.append(ht_ev4)
            tmp = dpt.empty_like(cp2_usm)
            ht_ev5, dev_ev5 = tei._multiply(
                dpnp.get_usm_ndarray(a1), dpnp.get_usm_ndarray(b0), tmp, exec_q
            )
            host_events.append(ht_ev5)
            ht_ev6, _ = tei._subtract_inplace(
                cp2_usm, tmp, exec_q, [dev_ev4, dev_ev5]
            )
            host_events.append(ht_ev6)

    dpctl.SyclEvent.wait_for(host_events)
    return cp


def dpnp_dot(a, b, /, out=None, *, conjugate=False):
    """
    Return the dot product of two arrays.

    The routine that is used to perform the main calculation
    depends on input arrays data type: 1) For integer and boolean data types,
    `dpctl.tensor.vecdot` form the Data Parallel Control library is used,
    2) For real-valued floating point data types, `dot` routines from
    BLAS library of OneMKL are used, and 3) For complex data types,
    `dotu` or `dotc` routines from BLAS library of OneMKL are used.
    If `conjugate` is ``False``, `dotu` is used. Otherwise, `dotc` is used,
    for which the first array is conjugated before calculating the dot product.

    """

    if a.size != b.size:
        raise ValueError(
            "Input arrays have a mismatch in their size. "
            f"(size {a.size} is different from {b.size})"
        )

    res_usm_type, exec_q = get_usm_allocations([a, b])

    # Determine the appropriate data types
    # casting is irrelevant here since dtype is `None`
    dot_dtype, res_dtype = _compute_res_dtype(
        a, b, dtype=None, casting="no", sycl_queue=exec_q
    )

    result = _create_result_array(
        a, b, out, (), dot_dtype, res_usm_type, exec_q
    )
    # input arrays should have the proper data type
    dep_events_list = []
    host_tasks_list = []
    if dpnp.issubdtype(res_dtype, dpnp.inexact):
        # copying is needed if dtypes of input arrays are different
        a = _copy_array(a, dep_events_list, host_tasks_list, dtype=dot_dtype)
        b = _copy_array(b, dep_events_list, host_tasks_list, dtype=dot_dtype)
        if dpnp.issubdtype(res_dtype, dpnp.complexfloating):
            if conjugate:
                dot_func = "_dotc"
            else:
                dot_func = "_dotu"
            ht_ev, _ = getattr(bi, dot_func)(
                exec_q,
                dpnp.get_usm_ndarray(a),
                dpnp.get_usm_ndarray(b),
                dpnp.get_usm_ndarray(result),
                dep_events_list,
            )
        else:
            ht_ev, _ = bi._dot(
                exec_q,
                dpnp.get_usm_ndarray(a),
                dpnp.get_usm_ndarray(b),
                dpnp.get_usm_ndarray(result),
                dep_events_list,
            )
        host_tasks_list.append(ht_ev)
        dpctl.SyclEvent.wait_for(host_tasks_list)
    else:
        # oneapi::mkl::blas::dot is slow for integer data type,
        # so using dpctl.tensor.vecdot instead
        dpt_a = dpnp.get_usm_ndarray(a)
        dpt_b = dpnp.get_usm_ndarray(b)
        result = dpnp_array._create_from_usm_ndarray(dpt.vecdot(dpt_a, dpt_b))

    if dot_dtype != res_dtype:
        result = result.astype(res_dtype, copy=False)

    # numpy.dot does not allow casting even if it is safe
    return dpnp.get_result_array(result, out, casting="no")


def dpnp_einsum(
    *operands, out=None, dtype=None, order="K", casting="safe", optimize=False
):
    """Evaluates the Einstein summation convention on the operands."""

    input_subscripts, output_subscript, operands = _parse_einsum_input(operands)
    assert isinstance(input_subscripts, list)
    assert isinstance(operands, list)

    dpnp.check_supported_arrays_type(*operands, scalar_type=True)
    arrays = []
    for a in operands:
        if dpnp.is_supported_array_type(a):
            arrays.append(a)

    res_usm_type, exec_q = get_usm_allocations(arrays)
    result_dtype = dpnp.result_type(*arrays) if dtype is None else dtype
    for id, a in enumerate(operands):
        if dpnp.isscalar(a):
            operands[id] = dpnp.array(
                a, dtype=result_dtype, usm_type=res_usm_type, sycl_queue=exec_q
            )

    input_subscripts = [
        _parse_ellipsis_subscript(sub, idx, ndim=arr.ndim)
        for idx, (sub, arr) in enumerate(zip(input_subscripts, operands))
    ]

    # Get length of each unique dimension and ensure all dimensions are correct
    dimension_dict = {}
    for idx, sub in enumerate(input_subscripts):
        sh = operands[idx].shape
        for axis, label in enumerate(sub):
            dim = sh[axis]
            if label in dimension_dict.keys():
                # For broadcasting cases we always want the largest dim size
                if dimension_dict[label] == 1:
                    dimension_dict[label] = dim
                elif dim not in (1, dimension_dict[label]):
                    dim_old = dimension_dict[label]
                    raise ValueError(
                        f"Size of label '{_chr(label)}' for operand {idx} ({dim}) "
                        f"does not match previous terms ({dim_old})."
                    )
            else:
                dimension_dict[label] = dim

    if output_subscript is None:
        # Build output subscripts
        tmp_subscripts = list(itertools.chain.from_iterable(input_subscripts))
        output_subscript = [
            label
            for label in sorted(set(tmp_subscripts))
            if label < 0 or tmp_subscripts.count(label) == 1
        ]
    else:
        if "@" not in output_subscript and -1 in dimension_dict:
            raise ValueError(
                "output has more dimensions than subscripts "
                "given in einstein sum, but no '...' ellipsis "
                "provided to broadcast the extra dimensions."
            )
        output_subscript = _parse_ellipsis_subscript(
            output_subscript,
            None,
            ellipsis_len=sum(label < 0 for label in dimension_dict.keys()),
        )

        # Make sure output subscripts are in the input
        tmp_subscripts = set(itertools.chain.from_iterable(input_subscripts))
        for label in output_subscript:
            if label not in tmp_subscripts:
                raise ValueError(
                    "einstein sum subscripts string included output subscript "
                    f"'{_chr(label)}' which never appeared in an input."
                )
        if len(output_subscript) != len(set(output_subscript)):
            for label in output_subscript:
                if output_subscript.count(label) >= 2:
                    raise ValueError(
                        "einstein sum subscripts string includes output "
                        f"subscript '{_chr(label)}' multiple times."
                    )

    _einsum_diagonals(input_subscripts, operands)

    # no more raises
    if len(operands) >= 2:
        if any(arr.size == 0 for arr in operands):
            return dpnp.zeros(
                tuple(dimension_dict[label] for label in output_subscript),
                dtype=result_dtype,
                usm_type=res_usm_type,
                sycl_queue=exec_q,
            )

        # Don't squeeze if unary, because this affects later (in trivial sum)
        # whether the return is a writeable view.
        for idx in range(len(operands)):
            arr = operands[idx]
            if 1 in arr.shape:
                squeeze_indices = []
                sub = []
                for axis, label in enumerate(input_subscripts[idx]):
                    if arr.shape[axis] == 1:
                        squeeze_indices.append(axis)
                    else:
                        sub.append(label)
                input_subscripts[idx] = sub
                operands[idx] = dpnp.squeeze(arr, axis=tuple(squeeze_indices))
                assert operands[idx].ndim == len(input_subscripts[idx])
            del arr

    # unary einsum without summation should return a (writeable) view
    returns_view = len(operands) == 1

    # unary sum
    for idx, sub in enumerate(input_subscripts):
        other_subscripts = copy.copy(input_subscripts)
        other_subscripts[idx] = output_subscript
        other_subscripts = set(itertools.chain.from_iterable(other_subscripts))
        sum_axes = tuple(
            axis
            for axis, label in enumerate(sub)
            if label not in other_subscripts
        )
        if sum_axes:
            returns_view = False
            input_subscripts[idx] = [
                label for axis, label in enumerate(sub) if axis not in sum_axes
            ]

            operands[idx] = operands[idx].sum(axis=sum_axes, dtype=result_dtype)

    if returns_view:
        # TODO: replace with a.view() when it is implemented in dpnp
        operands = [a for a in operands]
    else:
        operands = [
            dpnp.astype(
                a, result_dtype, copy=False, casting=casting, order=order
            )
            for a in operands
        ]

    # no more casts
    optimize_algorithms = {
        "greedy": _greedy_path,
        "optimal": _optimal_path,
    }
    if optimize is False:
        path = [tuple(range(len(operands)))]
    elif len(optimize) and (optimize[0] == "einsum_path"):
        path = optimize[1:]
    else:
        try:
            if len(optimize) == 2 and isinstance(optimize[1], (int, float)):
                algo = optimize_algorithms[optimize[0]]
                memory_limit = int(optimize[1])
            else:
                algo = optimize_algorithms[optimize]
                memory_limit = 2**31
        except (TypeError, KeyError):  # unhashable type or not found
            raise TypeError(
                f"Did not understand the path (optimize): {str(optimize)}"
            )
        input_sets = [set(sub) for sub in input_subscripts]
        output_set = set(output_subscript)
        path = algo(input_sets, output_set, dimension_dict, memory_limit)
        if any(len(indices) > 2 for indices in path):
            warnings.warn(
                "memory efficient einsum is not supported yet",
                RuntimeWarning,
                stacklevel=2,
            )

    for idx0, idx1 in _iter_path_pairs(path):
        # "reduced" binary einsum
        arr0 = operands.pop(idx0)
        sub0 = input_subscripts.pop(idx0)
        arr1 = operands.pop(idx1)
        sub1 = input_subscripts.pop(idx1)
        sub_others = list(
            itertools.chain(
                output_subscript,
                itertools.chain.from_iterable(input_subscripts),
            )
        )
        arr_out, sub_out = _reduced_binary_einsum(
            arr0, sub0, arr1, sub1, sub_others
        )
        operands.append(arr_out)
        input_subscripts.append(sub_out)
        del arr0, arr1

    # unary einsum at last
    (arr0,) = operands
    (sub0,) = input_subscripts

    transpose_axes = []
    for label in output_subscript:
        if label in sub0:
            transpose_axes.append(sub0.index(label))

    arr_out = arr0.transpose(transpose_axes).reshape(
        [dimension_dict[label] for label in output_subscript]
    )
    assert returns_view or arr_out.dtype == result_dtype
    return dpnp.get_result_array(arr_out, out)


def dpnp_kron(a, b, a_ndim, b_ndim):
    """Returns the kronecker product of two arrays."""

    a_shape = a.shape
    b_shape = b.shape
    if not a.flags.contiguous:
        a = dpnp.reshape(a, a_shape)
    if not b.flags.contiguous:
        b = dpnp.reshape(b, b_shape)

    # Equalise the shapes by prepending smaller one with 1s
    a_shape = (1,) * max(0, b_ndim - a_ndim) + a_shape
    b_shape = (1,) * max(0, a_ndim - b_ndim) + b_shape

    # Insert empty dimensions
    a_arr = dpnp.expand_dims(a, axis=tuple(range(b_ndim - a_ndim)))
    b_arr = dpnp.expand_dims(b, axis=tuple(range(a_ndim - b_ndim)))

    # Compute the product
    ndim = max(b_ndim, a_ndim)
    a_arr = dpnp.expand_dims(a_arr, axis=tuple(range(1, 2 * ndim, 2)))
    b_arr = dpnp.expand_dims(b_arr, axis=tuple(range(0, 2 * ndim, 2)))
    result = dpnp.multiply(a_arr, b_arr, order="C")

    # Reshape back
    return result.reshape(tuple(numpy.multiply(a_shape, b_shape)))


def dpnp_matmul(
    x1,
    x2,
    /,
    out=None,
    *,
    casting="same_kind",
    order="K",
    dtype=None,
    axes=None,
):
    """
    Return the matrix product of two arrays.

    The main calculation is done by calling an extension function
    for BLAS library of OneMKL. Depending on dimension of `x1` and `x2` arrays,
    it will be either ``gemm`` (for one- and two-dimentional arrays) or
    ``gemm_batch``(for others).

    """

    x1_ndim = x1.ndim
    x2_ndim = x2.ndim
    res_usm_type, exec_q = get_usm_allocations([x1, x2])

    if axes is not None:
        axes = _validate_axes(x1, x2, axes)

        axes_x1, axes_x2, axes_res = axes
        axes_x1 = normalize_axis_tuple(axes_x1, x1_ndim, "axis")
        axes_x2 = normalize_axis_tuple(axes_x2, x2_ndim, "axis")
        # Move the axes that are going to be used in matrix product,
        # to the end of "x1" and "x2"
        x1 = dpnp.moveaxis(x1, axes_x1, (-2, -1)) if x1_ndim != 1 else x1
        x2 = dpnp.moveaxis(x2, axes_x2, (-2, -1)) if x2_ndim != 1 else x2
        out_orig = out
        if out is not None:
            dpnp.check_supported_arrays_type(out)
            # out that is passed to the backend should have the correct shape
            if len(axes_res) == 2:
                out = dpnp.moveaxis(out, axes_res, (-2, -1))
            elif len(axes_res) == 1:
                out = dpnp.moveaxis(out, axes_res, (-1,))

    # With these conditions, the result is a 0D array. However,
    # NumPy allows out to have any shape and the result is expanded to it
    NumPy_special_behavior = (
        out is not None and x1_ndim == 1 and x2_ndim == 1 and out.shape != ()
    )

    x1, x2, result_shape = _get_result_shape(
        x1, x2, out, NumPy_special_behavior
    )

    # Determine the appropriate data types
    compute_dtype, res_dtype = _compute_res_dtype(
        x1, x2, dtype=dtype, casting=casting, sycl_queue=exec_q
    )

    call_flag = None
    x1_shape = x1.shape
    x2_shape = x2.shape
    x1_is_2D, x1_is_1D, x1_base_is_1D = _define_dim_flags(x1, pos=0)
    x2_is_2D, x2_is_1D, x2_base_is_1D = _define_dim_flags(x2, pos=1)
    # TODO: investigate usage of syrk function from BLAS in
    # case of a.T @ a and a @ a.T to gain performance.
    if numpy.prod(result_shape) == 0:
        res_shape = result_shape
    elif x1_shape[-1] == 1:
        call_flag = "multiply"
    elif x1_is_1D and x2_is_1D:
        call_flag = "dot"
        x1 = dpnp.reshape(x1, x1_shape[-1])
        if x2_ndim != 1:
            x2 = dpnp.reshape(x2, x2_shape[-2])
    elif x1_base_is_1D and x2_base_is_1D:
        # TODO: implement a batch version of dot to use it here
        call_flag = "gemm_batch"
        res_shape = result_shape
    elif x1_is_1D and x2_is_2D:
        # TODO: implement gemv to use it here with transpose
        call_flag = "gemm"
        x1 = dpnp.reshape(x1, (1, x1.size))
        x2 = dpnp.reshape(x2, x2_shape[-2:])
        x1_shape = x1.shape
        res_shape = (x1_shape[-2], x2_shape[-1])
    elif x1_is_2D and x2_is_1D:
        # TODO: implement gemv to use it here without transpose
        call_flag = "gemm"
        x1 = dpnp.reshape(x1, x1_shape[-2:])
        x2 = dpnp.reshape(x2, (x2.size, 1))
        x2_shape = x2.shape
        res_shape = (x1_shape[-2], x2_shape[-1])
    elif x1_is_2D and x2_is_2D:
        call_flag = "gemm"
        x1 = dpnp.reshape(x1, x1_shape[-2:])
        x2 = dpnp.reshape(x2, x2_shape[-2:])
        res_shape = (x1_shape[-2], x2_shape[-1])
    elif x1_base_is_1D:
        # TODO: implement gemv_batch to use it here with transpose
        call_flag = "gemm_batch"
        if x1_ndim == 1:
            x1 = dpnp.reshape(x1, (1, 1, x1.size))
            res_shape = result_shape[:-1] + (1, result_shape[-1])
        else:
            res_shape = result_shape
    elif x2_base_is_1D:
        # TODO: implement gemv_batch to use it here without transpose
        call_flag = "gemm_batch"
        if x2_ndim == 1:
            x2 = dpnp.reshape(x2, (1, x2.size, 1))
            res_shape = result_shape + (1,)
        else:
            res_shape = result_shape
    else:
        call_flag = "gemm_batch"
        res_shape = result_shape

    if call_flag == "multiply":
        res = dpnp.multiply(x1, x2)
        res_shape = res.shape
    elif call_flag == "dot":
        if out is not None and out.shape != ():
            res = dpnp_dot(x1, x2)
        else:
            res = dpnp_dot(x1, x2, out=out)
        res_shape = res.shape
    else:
        res = _create_result_array(
            x1, x2, out, res_shape, compute_dtype, res_usm_type, exec_q
        )

        # calculate result
        if res.size == 0:
            pass
        elif x1.size == 0 or x2.size == 0:
            res.fill(0)
        else:
            # input arrays should have the proper data type and
            # their base (last 2-dimensions) to be c-contiguous or f-contiguous
            dep_events_list = []
            host_tasks_list = []
            contig_flag = _define_contig_flag(x1)
            x1 = _copy_array(
                x1,
                dep_events_list,
                host_tasks_list,
                copy_flag=not contig_flag,
                dtype=compute_dtype,
            )
            contig_flag = _define_contig_flag(x2)
            x2 = _copy_array(
                x2,
                dep_events_list,
                host_tasks_list,
                copy_flag=not contig_flag,
                dtype=compute_dtype,
            )

            if call_flag == "gemm":
                res = _gemm_matmul(
                    exec_q,
                    x1,
                    x2,
                    res,
                    dep_events_list,
                )
            else:  # call_flag == "gemm_batch"
                res = _gemm_batch_matmul(
                    exec_q,
                    x1,
                    x2,
                    res,
                    dep_events_list,
                )

            dpctl.SyclEvent.wait_for(host_tasks_list)

    if NumPy_special_behavior:
        result = dpnp.tile(res, out.shape)
    else:
        result = (
            dpnp.reshape(res, result_shape)
            if res_shape != result_shape
            else res
        )

    if compute_dtype != res_dtype:
        result = dpnp.astype(result, res_dtype, copy=False)

    if out is None:
        if axes is not None:
            # Move the data to the appropriate axes of the result array
            if len(axes_res) == 2:
                result = dpnp.moveaxis(result, (-2, -1), axes_res)
            elif len(axes_res) == 1:
                result = dpnp.moveaxis(result, (-1,), axes_res)
            return result
        # If `order` was not passed as default
        # we need to update it to match the passed `order`.
        elif order not in ["k", "K"]:
            return dpnp.array(result, copy=False, order=order)
        else:
            return result
    else:
        # TODO: There is opportunity to improve performance when out keyword
        # is present. For some cases, out is NOT result but they have the same
        # base (They are views of the same data). In this case, we can avoid
        # copyign result to out.
        result = dpnp.get_result_array(result, out, casting=casting)
        if axes is not None and out is result:
            # out and out_orig contain the same data but they have different shape
            return out_orig
        return result
