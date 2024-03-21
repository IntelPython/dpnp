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

import itertools
import operator

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

# import string

# importing string for string.ascii_letters would be too slow
# the first import before caching has been measured to take 800 µs (#23777)
# einsum_symbols = string.ascii_uppercase + string.ascii_lowercase
einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
einsum_symbols_set = set(einsum_symbols)


__all__ = [
    "dpnp_cross",
    "dpnp_dot",
    "dpnp_einsum_path",
    "dpnp_kron",
    "dpnp_matmul",
]


def _can_dot(inputs, result, idx_removed):
    """
    Checks if we can use BLAS (dpnp.tensordot) call and its beneficial to do so.

    Parameters
    ----------
    inputs : list of str
        Specifies the subscripts for summation.
    result : str
        Resulting summation.
    idx_removed : set
        Indices that are removed in the summation


    Returns
    -------
    type : bool
        Returns true if BLAS should and can be used, else False

    Notes
    -----
    If the operations is BLAS level 1 or 2 and is not already aligned
    we default back to einsum as the memory movement to copy is more
    costly than the operation itself.


    Examples
    --------
    # Standard GEMM operation
    >>> _can_dot(["ij", "jk"], "ik", set("j"))
    True

    # Can use the standard BLAS, but requires odd data movement
    >>> _can_dot(["ijj", "jk"], "ik", set("j"))
    False

    # DDOT where the memory is not aligned
    >>> _can_dot(["ijk", "ikj"], "", set("ijk"))
    False

    """
    # All `dot` calls remove indices
    if len(idx_removed) == 0:
        return False

    # BLAS can only handle two operands
    if len(inputs) != 2:
        return False

    input_left, input_right = inputs

    for c in set(input_left + input_right):
        # can't deal with repeated indices on same input or more than 2 total
        nl, nr = input_left.count(c), input_right.count(c)
        if (nl > 1) or (nr > 1) or (nl + nr > 2):
            return False

        # can't do implicit summation or dimension collapse e.g.
        #     "ab,bc->c" (implicitly sum over 'a')
        #     "ab,ca->ca" (take diagonal of 'a')
        if nl + nr - 1 == int(c in result):
            return False

    # Build a few temporaries
    set_left = set(input_left)
    set_right = set(input_right)
    keep_left = set_left - idx_removed
    keep_right = set_right - idx_removed
    rs = len(idx_removed)

    # At this point we are a DOT, GEMV, or GEMM operation

    # Handle inner products

    # DDOT with aligned data
    if input_left == input_right:
        return True

    # DDOT without aligned data (better to use einsum)
    if set_left == set_right:
        return False

    # Handle the 4 possible (aligned) GEMV or GEMM cases

    # GEMM or GEMV no transpose
    if input_left[-rs:] == input_right[:rs]:
        return True

    # GEMM or GEMV transpose both
    if input_left[:rs] == input_right[-rs:]:
        return True

    # GEMM or GEMV transpose right
    if input_left[-rs:] == input_right[-rs:]:
        return True

    # GEMM or GEMV transpose left
    if input_left[:rs] == input_right[:rs]:
        return True

    # Einsum is faster than GEMV if we have to copy data
    if not keep_left or not keep_right:
        return False

    # We are a matrix-matrix product, but we need to copy data
    return True


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

        if (
            out.dtype == dtype
            and out.shape == shape
            and out.usm_type == usm_type
            and out.sycl_queue == sycl_queue
            and out.flags.c_contiguous
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


def _compute_size_by_dict(indices, idx_dict):
    """
    Computes the product of the elements in indices based on the dictionary
    idx_dict.

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
    >>> _compute_size_by_dict("abbc", {"a": 2, "b":3, "c":5})
    90

    """
    ret = 1
    for i in indices:
        ret *= idx_dict[i]
    return ret


def _copy_array(x, dep_events, host_events, contig_copy=False, dtype=None):
    """
    Creating a copy of input array if needed.

    If `contig_copy` is ``True``, a C-contiguous copy of input array is returned.
    In this case, the copy array has the input array data type unless `dtype` is
    determined.
    If `contig_copy` is ``False`` and input array data type is different than `dtype`,
    a C-contiguous copy of input array with specified `dtype` is returned.

    """

    if contig_copy:
        copy = contig_copy
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


def _find_contraction(positions, input_sets, output_set):
    """
    Finds the contraction for a given set of input and output sets.

    Parameters
    ----------
    positions : iterable
        Integer positions of terms used in the contraction.
    input_sets : list
        List of sets that represent the lhs side of the einsum subscript
    output_set : set
        Set that represents the rhs side of the overall einsum subscript

    Returns
    -------
    new_result : set
        The indices of the resulting contraction
    remaining : list
        List of sets that have not been contracted, the new set is appended to
        the end of this list
    idx_removed : set
        Indices removed from the entire contraction
    idx_contraction : set
        The indices used in the current contraction

    Examples
    --------
    # A simple dot product test case
    >>> pos = (0, 1)
    >>> isets = [set("ab"), set("bc")]
    >>> oset = set("ac")
    >>> _find_contraction(pos, isets, oset)
    ({"a", "c"}, [{"a", "c"}], {"b"}, {"a", "b", "c"})

    # A more complex case with additional terms in the contraction
    >>> pos = (0, 2)
    >>> isets = [set("abd"), set("ac"), set("bdc")]
    >>> oset = set("ac")
    >>> _find_contraction(pos, isets, oset)
    ({"a", "c"}, [{"a", "c"}, {"a", "c"}], {"b", "d"}, {"a", "b", "c", "d"})

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


def _flop_count(idx_contraction, inner, num_terms, size_dictionary):
    """
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
    >>> _flop_count("abc", False, 1, {"a": 2, "b":3, "c":5})
    30

    >>> _flop_count("abc", True, 2, {"a": 2, "b":3, "c":5})
    60

    """
    overall_size = _compute_size_by_dict(idx_contraction, size_dictionary)
    op_factor = max(1, num_terms - 1)
    if inner:
        op_factor += 1

    return overall_size * op_factor


def _gemm_batch_matmul(exec_q, x1, x2, res, x1_is_2D, x2_is_2D, dev_tasks_list):
    # If input array is F-contiguous, we need to change the order to C-contiguous.
    # because mkl::gemm_bacth needs each 2D array to be F-contiguous but
    # when the input array is F-contiguous, the data of 2D array
    # that needs to be called in mkl::gemm_batch are not contiguous.
    ht_tasks_list = []
    contig_copy = not x1.flags.c_contiguous
    x1 = _copy_array(x1, dev_tasks_list, ht_tasks_list, contig_copy=contig_copy)
    contig_copy = not x2.flags.c_contiguous
    x2 = _copy_array(x2, dev_tasks_list, ht_tasks_list, contig_copy=contig_copy)

    x1_strides = x1.strides
    x2_strides = x2.strides
    res_strides = res.strides

    # need to standardize to use in ti._contract_iter2
    x1_strides = _standardize_strides(x1_strides, x1_is_2D, x1.shape, x1.ndim)
    x2_strides = _standardize_strides(x2_strides, x2_is_2D, x2.shape, x2.ndim)

    batch_size = res.shape[:-2][0]
    stridea = x1_strides[0]
    strideb = x2_strides[0]
    stridec = res_strides[-3]

    if x1.ndim > 3:
        iter = ti._contract_iter2(
            res.shape[:-2], x1_strides[:-2], x2_strides[:-2]
        )

        if len(iter[0]) != 1:
            raise ValueError("Input arrays cannot be used in gemm_batch")
        batch_size = iter[0][0]
        stridea = iter[1][0]
        strideb = iter[3][0]

    ht_blas_ev, _ = bi._gemm_batch(
        exec_q,
        dpnp.get_usm_ndarray(x1),
        dpnp.get_usm_ndarray(x2),
        dpnp.get_usm_ndarray(res),
        batch_size,
        stridea,
        strideb,
        stridec,
        dev_tasks_list,
    )

    return ht_blas_ev, ht_tasks_list, res


def _greedy_path(input_sets, output_set, idx_dict, memory_limit):
    """
    Finds the path by contracting the best pair until the input list is
    exhausted. The best pair is found by minimizing the tuple
    ``(-prod(indices_removed), cost)``.  What this amounts to is prioritizing
    matrix multiplication or inner product operations, then Hadamard like
    operations, and finally outer operations. Outer products are limited by
    ``memory_limit``. This algorithm scales cubically with respect to the
    number of elements in the list ``input_sets``.

    Parameters
    ----------
    input_sets : list
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
    >>> isets = [set("abd"), set("ac"), set("bdc")]
    >>> oset = set()
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
    contract = _find_contraction(range(len(input_sets)), input_sets, output_set)
    _, _, idx_removed, idx_contract = contract
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


def _op_res_dtype(*arrays, dtype, casting, sycl_queue):
    """
    _op_res_dtype(*arrays, dtype, casting, sycl_queue)

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
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur.
    sycl_queue : {SyclQueue}
        A SYCL queue to use for determining default floating point datat type.

    Returns
    -------
    op_dtype, res_dtype :
        `op_dtype` is the data type used in performing math function calculations.
        The input arrays of the math function are cast to `op_dtype` and then
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

    op_dtype = (
        res_dtype if dpnp.issubdtype(res_dtype, dpnp.inexact) else default_dtype
    )

    return op_dtype, res_dtype


def _optimal_path(input_sets, output_set, idx_dict, memory_limit):
    """
    Computes all possible pair contractions, sieves the results based
    on ``memory_limit`` and returns the lowest cost path. This algorithm
    scales factorial with respect to the elements in the list ``input_sets``.

    Parameters
    ----------
    input_sets : list
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
    >>> isets = [set("abd"), set("ac"), set("bdc")]
    >>> oset = set()
    >>> idx_sizes = {"a": 1, "b":2, "c":3, "d":4}
    >>> _optimal_path(isets, oset, idx_sizes, 5000)
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

    # If we have not found anything return single einsum contraction
    if len(full_results) == 0:
        return [tuple(range(len(input_sets)))]

    path = min(full_results, key=lambda x: x[0])[1]
    return path


def _parse_einsum_input(operands):
    """
    Parse einsum operands.

    Parameters
    ----------
    args : tuple
        The non-keyword arguments to einsum

    Returns
    -------
    input_strings : list of str
        Parsed input strings
    output_string : str
        Parsed output string
    operands : list of array_like
        The operands to use in the contraction

    Notes
    -----
    If ellipsis is present in the input, it is replaced with ``@`` symbol.

    Examples
    --------
    The operand list is simplified to reduce printing:

    >>> import dpnp as np
    >>> from dpnp.dpnp_utils.dpnp_utils_linearalgebra import _parse_einsum_input
    >>> a = np.random.rand(4, 4)
    >>> b = np.random.rand(4, 4, 4)
    >>> _parse_einsum_input(('...a,...a->...', a, b))
    ('Sa,DSa', 'DS', [a, b]) # may vary

    >>> _parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))
    ('Sa,DSa', 'DS', [a, b]) # may vary

    """

    if len(operands) == 0:
        raise ValueError("No input operands")

    if isinstance(operands[0], str):
        subscripts = operands[0].replace(" ", "")
        operands = [dpnp.asarray(v) for v in operands[1:]]

        # Ensure all characters are valid
        for s in subscripts:
            if s in ".,->":
                continue
            if s not in einsum_symbols:
                raise ValueError("Character %s is not a valid symbol." % s)

    else:
        tmp_operands = list(operands)
        operand_list = []
        subscript_list = []
        for _ in range(len(operands) // 2):
            operand_list.append(tmp_operands.pop(0))
            subscript_list.append(tmp_operands.pop(0))

        output_list = tmp_operands[-1] if len(tmp_operands) else None
        operands = [dpnp.asarray(v) for v in operand_list]
        subscripts = ""
        last = len(subscript_list) - 1
        for num, sub in enumerate(subscript_list):
            for s in sub:
                if s is Ellipsis:
                    subscripts += "..."
                else:
                    try:
                        s = operator.index(s)
                    except TypeError as e:
                        raise TypeError(
                            "For this input type lists must contain "
                            "either int or Ellipsis"
                        ) from e
                    subscripts += einsum_symbols[s]
            if num != last:
                subscripts += ","

        if output_list is not None:
            subscripts += "->"
            for s in output_list:
                if s is Ellipsis:
                    subscripts += "..."
                else:
                    try:
                        s = operator.index(s)
                    except TypeError as e:
                        raise TypeError(
                            "For this input type lists must contain "
                            "either int or Ellipsis"
                        ) from e
                    subscripts += einsum_symbols[s]
    # Check for proper "->"
    if ("-" in subscripts) or (">" in subscripts):
        invalid = (subscripts.count("-") > 1) or (subscripts.count(">") > 1)
        if invalid or (subscripts.count("->") != 1):
            raise ValueError("Subscripts can only contain one '->'.")

    # Parse ellipses
    if "." in subscripts:
        used = subscripts.replace(".", "").replace(",", "").replace("->", "")
        unused = list(einsum_symbols_set - set(used))
        ellipse_inds = "".join(unused)
        longest = 0

        if "->" in subscripts:
            input_tmp, output_sub = subscripts.split("->")
            split_subscripts = input_tmp.split(",")
            out_sub = True
        else:
            split_subscripts = subscripts.split(",")
            out_sub = False

        for num, sub in enumerate(split_subscripts):
            if "." in sub:
                if (sub.count(".") != 3) or (sub.count("...") != 1):
                    raise ValueError("Invalid Ellipses.")

                # Take into account numerical values
                if operands[num].shape == ():
                    ellipse_count = 0
                else:
                    ellipse_count = max(operands[num].ndim, 1)
                    ellipse_count -= len(sub) - 3

                if ellipse_count > longest:
                    longest = ellipse_count

                if ellipse_count < 0:
                    raise ValueError("Ellipses lengths do not match.")
                elif ellipse_count == 0:
                    split_subscripts[num] = sub.replace("...", "")
                else:
                    rep_inds = ellipse_inds[-ellipse_count:]
                    split_subscripts[num] = sub.replace("...", rep_inds)

        subscripts = ",".join(split_subscripts)
        if longest == 0:
            out_ellipse = ""
        else:
            out_ellipse = ellipse_inds[-longest:]

        if out_sub:
            subscripts += "->" + output_sub.replace("...", out_ellipse)
        else:
            # Special care for outputless ellipses
            output_subscript = ""
            tmp_subscripts = subscripts.replace(",", "")
            for s in sorted(set(tmp_subscripts)):
                if s not in (einsum_symbols):
                    raise ValueError("Character %s is not a valid symbol." % s)
                if tmp_subscripts.count(s) == 1:
                    output_subscript += s
            normal_inds = "".join(
                sorted(set(output_subscript) - set(out_ellipse))
            )

            subscripts += "->" + out_ellipse + normal_inds

    # Build output string if does not exist
    if "->" in subscripts:
        input_subscripts, output_subscript = subscripts.split("->")
    else:
        input_subscripts = subscripts
        # Build output subscripts
        tmp_subscripts = subscripts.replace(",", "")
        output_subscript = ""
        for s in sorted(set(tmp_subscripts)):
            if s not in einsum_symbols:
                raise ValueError("Character %s is not a valid symbol." % s)
            if tmp_subscripts.count(s) == 1:
                output_subscript += s

    # Make sure output subscripts are in the input
    for char in output_subscript:
        if char not in input_subscripts:
            raise ValueError(
                "Output character %s did not appear in the input" % char
            )

    # Make sure number operands is equivalent to the number of terms
    if len(input_subscripts.split(",")) != len(operands):
        raise ValueError(
            "Number of einsum subscripts must be equal to the "
            "number of operands."
        )

    return (input_subscripts, output_subscript, operands)


def _parse_possible_contraction(
    positions,
    input_sets,
    output_set,
    idx_dict,
    memory_limit,
    path_cost,
    naive_cost,
):
    """Compute the cost (removed size + flops) and resultant indices for
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
    if (path_cost + cost) > naive_cost:
        return None

    # Add contraction to possible choices
    return [sort, positions, new_input_sets]


def _shape_error(a, b, core_dim, err_msg):
    if err_msg == 0:
        raise ValueError(
            "Input arrays have a mismatch in their core dimensions. "
            "The core dimensions should follow this signature: (n?,k),(k,m?)->(n?,m?) "
            f"(size {a} is different from {b})"
        )
    elif err_msg == 1:
        raise ValueError(
            f"Output array has a mismatch in its core dimension {core_dim}. "
            "The core dimensions should follow this signature: (n?,k),(k,m?)->(n?,m?) "
            f"(size {a} is different from {b})"
        )
    elif err_msg == 2:
        raise ValueError(
            "Input arrays could not be broadcast together with remapped shapes, "
            f"{a} is different from {b}."
        )
    elif err_msg == 3:
        raise ValueError(
            "Output array could not be broadcast to input arrays with remapped shapes, "
            f"{a} is different from {b}."
        )


def _standardize_strides(strides, inherently_2D, shape, ndim):
    """
    Standardizing the strides.

    When shape of an array along any particular dimension is 1, the stride
    along that dimension is undefined. This functions standardize the strides
    in the following way:
    For N-D arrays that are inherently 2D (all dimesnsion are one except for two of them),
    we use zero as the stride for dimensions equal one.
    For other N-D arrays, the non-zero value of strides is calculated and used.

    """

    if inherently_2D:
        stndrd_strides = tuple(
            str_i if sh_i > 1 else 0 for sh_i, str_i in zip(shape, strides)
        )
    else:
        stndrd_strides = [
            numpy.prod(shape[i + 1 :]) if strides[i] == 0 else strides[i]
            for i in range(ndim - 1)
        ]
        # last dimension
        stndrd_strides.append(
            1 if strides[ndim - 1] == 0 else strides[ndim - 1]
        )
        stndrd_strides = tuple(stndrd_strides)

    return stndrd_strides


def _update_other_results(results, best):
    """Update the positions and provisional input_sets of ``results`` based on
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


def dpnp_einsum_path(*operands, optimize="greedy", einsum_call=False):
    """
    Evaluates the lowest cost contraction order for an einsum expression
    by considering the creation of intermediate arrays.
    """
    # Figure out what the path really is
    path_type = optimize
    if path_type is True:
        path_type = "greedy"
    if path_type is None:
        path_type = False

    explicit_einsum_path = False
    memory_limit = None

    if (path_type is False) or isinstance(path_type, str):
        # No optimization or a named path algorithm
        pass
    elif len(path_type) and (path_type[0] == "einsum_path"):
        # Given an explicit path
        explicit_einsum_path = True
    elif (
        (len(path_type) == 2)
        and isinstance(path_type[0], str)
        and isinstance(path_type[1], (int, float))
    ):
        # Path tuple with memory limit
        memory_limit = int(path_type[1])
        path_type = path_type[0]
    else:
        raise TypeError("Did not understand the path: %s" % str(path_type))

    # Hidden option, only einsum should call this
    einsum_call_arg = einsum_call

    # Python side parsing
    input_subscripts, output_subscript, operands = _parse_einsum_input(operands)

    # Build a few useful variables
    input_list = input_subscripts.split(",")
    input_sets = [set(x) for x in input_list]
    output_set = set(output_subscript)
    indices = set(input_subscripts.replace(",", ""))

    # Get length of each unique dimension and ensure all dimensions are correct
    dimension_dict = {}
    broadcast_indices = [[] for _ in range(len(input_list))]
    for tnum, term in enumerate(input_list):
        sh = operands[tnum].shape
        if len(sh) != len(term):
            raise ValueError(
                "Einstein sum subscript %s does not contain the "
                "correct number of indices for operand %d."
                % (input_subscripts[tnum], tnum)
            )
        for cnum, char in enumerate(term):
            dim = sh[cnum]

            # Build out broadcast indices
            if dim == 1:
                broadcast_indices[tnum].append(char)

            if char in dimension_dict.keys():
                # For broadcasting cases we always want the largest dim size
                if dimension_dict[char] == 1:
                    dimension_dict[char] = dim
                elif dim not in (1, dimension_dict[char]):
                    raise ValueError(
                        "Size of label '%s' for operand %d (%d) "
                        "does not match previous terms (%d)."
                        % (char, tnum, dimension_dict[char], dim)
                    )
            else:
                dimension_dict[char] = dim

    # Convert broadcast inds to sets
    broadcast_indices = [set(x) for x in broadcast_indices]

    # Compute size of each input array plus the output array
    size_list = [
        _compute_size_by_dict(term, dimension_dict)
        for term in input_list + [output_subscript]
    ]
    max_size = max(size_list)

    if memory_limit is None:
        memory_arg = max_size
    else:
        memory_arg = memory_limit

    # Compute naive cost
    # This isn't quite right, need to look into exactly how einsum does this
    inner_product = (sum(len(x) for x in input_sets) - len(indices)) > 0
    naive_cost = _flop_count(
        indices, inner_product, len(input_list), dimension_dict
    )

    # Compute the path
    if explicit_einsum_path:
        path = path_type[1:]
    elif (
        (path_type is False)
        or (len(input_list) in [1, 2])
        or (indices == output_set)
    ):
        # Nothing to be optimized, leave it to einsum
        path = [tuple(range(len(input_list)))]
    elif path_type == "greedy":
        path = _greedy_path(input_sets, output_set, dimension_dict, memory_arg)
    elif path_type == "optimal":
        path = _optimal_path(input_sets, output_set, dimension_dict, memory_arg)
    else:
        raise KeyError("Path name %s not found", path_type)

    cost_list, scale_list, size_list, contraction_list = [], [], [], []

    # Build contraction tuple (positions, gemm, einsum_str, remaining)
    for cnum, contract_inds in enumerate(path):
        # Make sure we remove inds from right to left
        contract_inds = tuple(sorted(list(contract_inds), reverse=True))

        contract = _find_contraction(contract_inds, input_sets, output_set)
        out_inds, input_sets, idx_removed, idx_contract = contract

        cost = _flop_count(
            idx_contract, idx_removed, len(contract_inds), dimension_dict
        )
        cost_list.append(cost)
        scale_list.append(len(idx_contract))
        size_list.append(_compute_size_by_dict(out_inds, dimension_dict))

        bcast = set()
        tmp_inputs = []
        for x in contract_inds:
            tmp_inputs.append(input_list.pop(x))
            bcast |= broadcast_indices.pop(x)

        new_bcast_inds = bcast - idx_removed

        # If we're broadcasting, nix blas
        if not len(idx_removed & bcast):
            do_blas = _can_dot(tmp_inputs, out_inds, idx_removed)
        else:
            do_blas = False

        # Last contraction
        if (cnum - len(path)) == -1:
            idx_result = output_subscript
        else:
            sort_result = [(dimension_dict[ind], ind) for ind in out_inds]
            idx_result = "".join([x[1] for x in sorted(sort_result)])

        input_list.append(idx_result)
        broadcast_indices.append(new_bcast_inds)
        einsum_str = ",".join(tmp_inputs) + "->" + idx_result

        contraction = (
            contract_inds,
            idx_removed,
            einsum_str,
            input_list[:],
            do_blas,
        )
        contraction_list.append(contraction)

    opt_cost = sum(cost_list) + 1

    if len(input_list) != 1:
        # Explicit "einsum_path" is usually trusted, but we detect this kind of
        # mistake in order to prevent from returning an intermediate value.
        raise RuntimeError(
            "Invalid einsum_path is specified: {} more operands has to be "
            "contracted.".format(len(input_list) - 1)
        )

    if einsum_call_arg:
        return (operands, contraction_list)

    # Return the path along with a nice string representation
    overall_contraction = input_subscripts + "->" + output_subscript
    header = ("scaling", "current", "remaining")

    speedup = naive_cost / opt_cost
    max_i = max(size_list)

    path_print = "  Complete contraction:  %s\n" % overall_contraction
    path_print += "         Naive scaling:  %d\n" % len(indices)
    path_print += "     Optimized scaling:  %d\n" % max(scale_list)
    path_print += "      Naive FLOP count:  %.3e\n" % naive_cost
    path_print += "  Optimized FLOP count:  %.3e\n" % opt_cost
    path_print += "   Theoretical speedup:  %3.3f\n" % speedup
    path_print += "  Largest intermediate:  %.3e elements\n" % max_i
    path_print += "-" * 74 + "\n"
    path_print += "%6s %24s %40s\n" % header
    path_print += "-" * 74

    for n, contraction in enumerate(contraction_list):
        _, _, einsum_str, remaining, _ = contraction
        remaining_str = ",".join(remaining) + "->" + output_subscript
        path_run = (scale_list[n], einsum_str, remaining_str)
        path_print += "\n%4d    %24s %40s" % path_run

    path = ["einsum_path"] + path
    return (path, path_print)


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
    result = dpnp.multiply(a_arr, b_arr)

    # Reshape back
    return result.reshape(tuple(numpy.multiply(a_shape, b_shape)))


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
    dot_dtype, res_dtype = _op_res_dtype(
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
        dpt_a = dpnp.get_usm_ndarray(a)
        dpt_b = dpnp.get_usm_ndarray(b)
        result = dpnp_array._create_from_usm_ndarray(dpt.vecdot(dpt_a, dpt_b))

    if dot_dtype != res_dtype:
        result = result.astype(res_dtype, copy=False)

    # numpy.dot does not allow casting even if it is safe
    return dpnp.get_result_array(result, out, casting="no")


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

    if x1_ndim == 0:
        raise ValueError(
            "input array 0 does not have enough dimensions (has 0, but requires at least 1)"
        )
    if x2_ndim == 0:
        raise ValueError(
            "input array 1 does not have enough dimensions (has 0, but requires at least 1)"
        )

    res_usm_type, exec_q = get_usm_allocations([x1, x2])

    if axes is not None:
        axes = _validate_axes(x1, x2, axes)

        axes_x1, axes_x2, axes_res = axes
        axes_x1 = normalize_axis_tuple(axes_x1, x1.ndim, "axis")
        axes_x2 = normalize_axis_tuple(axes_x2, x2.ndim, "axis")
        # Move the axes that are going to be used in matrix product,
        # to the end of "x1" and "x2"
        x1 = dpnp.moveaxis(x1, axes_x1, (-2, -1)) if x1.ndim != 1 else x1
        x2 = dpnp.moveaxis(x2, axes_x2, (-2, -1)) if x2.ndim != 1 else x2
        out_orig = out
        if out is not None:
            dpnp.check_supported_arrays_type(out)
            # out that is passed to the backend should have the correct shape
            if len(axes_res) == 2:
                out = dpnp.moveaxis(out, axes_res, (-2, -1))
            elif len(axes_res) == 1:
                out = dpnp.moveaxis(out, axes_res, (-1,))

    appended_axes = []
    if x1_ndim == 1:
        x1 = x1[dpnp.newaxis, :]
        x1_ndim = x1.ndim
        appended_axes.append(-2)

    if x2_ndim == 1:
        x2 = x2[:, dpnp.newaxis]
        x2_ndim = x2.ndim
        appended_axes.append(-1)

    x1_shape = x1.shape
    x2_shape = x2.shape
    if x1_shape[-1] != x2_shape[-2]:
        _shape_error(x1_shape[-1], x2_shape[-2], None, 0)

    if out is not None:
        out_shape = out.shape
        if not appended_axes:
            if out_shape[-2] != x1_shape[-2]:
                _shape_error(out_shape[-2], x1_shape[-2], 0, 1)
            if out_shape[-1] != x2_shape[-1]:
                _shape_error(out_shape[-1], x2_shape[-1], 1, 1)
        elif len(appended_axes) == 1:
            if appended_axes[0] == -1:
                if out_shape[-1] != x1_shape[-2]:
                    _shape_error(out_shape[-1], x1_shape[-2], 0, 1)
            elif appended_axes[0] == -2:
                if out_shape[-1] != x2_shape[-1]:
                    _shape_error(out_shape[-1], x2_shape[-1], 0, 1)

    # Determine the appropriate data types
    gemm_dtype, res_dtype = _op_res_dtype(
        x1, x2, dtype=dtype, casting=casting, sycl_queue=exec_q
    )

    x1_is_2D = x1_ndim == 2 or numpy.prod(x1_shape[:-2]) == 1  # inherently 2D
    x2_is_2D = x2_ndim == 2 or numpy.prod(x2_shape[:-2]) == 1

    # find the result shape
    if x1_is_2D and x2_is_2D:
        x1 = x1.reshape(x1.shape[-2], x1.shape[-1])
        x2 = x2.reshape(x2.shape[-2], x2.shape[-1])
        res_shape = (x1.shape[-2], x2.shape[-1])
    else:
        # makes the dimension of input the same by adding new axis
        if x1_ndim != x2_ndim:
            diff = abs(x1_ndim - x2_ndim)
            if x1_ndim < x2_ndim:
                x1 = x1.reshape((1,) * diff + x1.shape)
                x1_ndim = x1.ndim
                x1_shape = x1.shape
            else:
                x2 = x2.reshape((1,) * diff + x2.shape)
                x2_ndim = x2.ndim
                x2_shape = x2.shape

        # examining the option to align inputs
        # when their shapes differ but they are 1-D in some dimensions.
        tmp_shape = list(x1_shape[:-2])
        for i in range(x1_ndim - 2):
            if x1_shape[i] != x2_shape[i]:
                if x1_shape[i] == 1:
                    tmp_shape[i] = x2_shape[i]
                    # If the `x1` array is inherently 2D, there's no need to
                    # duplicate the data for the 1-D dimension;
                    # GEMM handles it automatically.
                    if not x1_is_2D:
                        x1 = dpnp.repeat(x1, x2_shape[i], axis=i)
                elif x2_shape[i] == 1:
                    tmp_shape[i] = x1_shape[i]
                    if not x2_is_2D:
                        x2 = dpnp.repeat(x2, x1_shape[i], axis=i)
                else:
                    _shape_error(x1_shape[:-2], x2_shape[:-2], None, 2)

        x1_shape = x1.shape
        x2_shape = x2.shape
        if out is not None:
            for i in range(x1_ndim - 2):
                if tmp_shape[i] != out_shape[i]:
                    if not appended_axes:
                        _shape_error(tuple(tmp_shape), out_shape[:-2], None, 3)
                    elif len(appended_axes) == 1:
                        _shape_error(tuple(tmp_shape), out_shape[:-1], None, 3)

        res_shape = tuple(tmp_shape) + (x1_shape[-2], x2_shape[-1])

    # handling a special case to provide a similar result to NumPy
    if out is not None and x1.shape == (1, 0) and x2.shape == (0, 1):
        res_shape = (0,)
        appended_axes = []

    result = _create_result_array(
        x1, x2, out, res_shape, gemm_dtype, res_usm_type, exec_q
    )

    # calculate result
    if result.size == 0:
        pass
    elif x1.size == 0 or x2.size == 0:
        result.fill(0)
    else:
        # input arrays should have the proper data type
        # and be C_CONTIGUOUS or F_CONTIGUOUS
        dep_events_list = []
        host_tasks_list = []
        contig_copy = not (x1.flags.c_contiguous or x1.flags.f_contiguous)
        x1 = _copy_array(
            x1,
            dep_events_list,
            host_tasks_list,
            contig_copy=contig_copy,
            dtype=gemm_dtype,
        )
        contig_copy = not (x2.flags.c_contiguous or x2.flags.f_contiguous)
        x2 = _copy_array(
            x2,
            dep_events_list,
            host_tasks_list,
            contig_copy=contig_copy,
            dtype=gemm_dtype,
        )

        # TODO: investigate usage of gemv (gemv_batch) function
        # from BLAS when one of the inputs is a vector to
        # gain performance.
        # TODO: investigate usage of syrk function from BLAS in
        # case of a.T @ a and a @ a.T to gain performance.
        if x1_is_2D and x2_is_2D:
            ht_blas_ev, _ = bi._gemm(
                exec_q,
                dpnp.get_usm_ndarray(x1),
                dpnp.get_usm_ndarray(x2),
                dpnp.get_usm_ndarray(result),
                dep_events_list,
            )
        else:
            (
                ht_blas_ev,
                ht_copy_ev,
                result,
            ) = _gemm_batch_matmul(
                exec_q,
                x1,
                x2,
                result,
                x1_is_2D,
                x2_is_2D,
                dep_events_list,
            )
            host_tasks_list += ht_copy_ev

        host_tasks_list.append(ht_blas_ev)
        dpctl.SyclEvent.wait_for(host_tasks_list)

    if appended_axes:
        result = dpnp.squeeze(result, tuple(appended_axes))
        if len(appended_axes) == 2 and out is not None:
            result = dpnp.tile(result, out.shape)

    if x1_is_2D and x2_is_2D:
        # add new axes only if one of the input arrays
        # was inehrently 2D
        new_size = max(x1_ndim, x2_ndim)
        for _ in range(new_size - 2):
            result = result[dpnp.newaxis, :]

    if gemm_dtype != res_dtype:
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
        result = dpnp.get_result_array(result, out, casting=casting)
        if axes is not None and out is result:
            # out and out_orig contain the same data but they have different shape
            return out_orig
        return result
