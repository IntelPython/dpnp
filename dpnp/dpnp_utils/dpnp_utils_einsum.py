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

import copy
import itertools
import operator
import warnings

import dpctl
import numpy
from dpctl.utils import ExecutionPlacementError

import dpnp
from dpnp.dpnp_array import dpnp_array
from dpnp.dpnp_utils import get_usm_allocations, map_dtype_to_device

_einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


__all__ = ["dpnp_einsum"]


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
    >>> import dpnp.dpnp_utils.dpnp_utils_einsum as np_util
    >>> np_util._chr(97)
    'a'
    >>> np_util._chr(-1)
    '...[-1]'

    """

    if label < 0:
        return f"...[{label}]"
    return chr(label)


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
    >>> import dpnp.dpnp_utils.dpnp_utils_einsum as np_util
    >>> np_util._compute_size_by_dict("abbc", {"a": 2, "b":3, "c":5})
    90

    """
    ret = 1
    for i in indices:
        ret *= idx_dict[i]
    return ret


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
    >>> import dpnp.dpnp_utils.dpnp_utils_einsum as np_util
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
    >>> import dpnp.dpnp_utils.dpnp_utils_einsum as np_util
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
    >>> import dpnp.dpnp_utils.dpnp_utils_einsum as np_util
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
    >>> import dpnp.dpnp_utils.dpnp_utils_einsum as np_util
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

    new_sh = tuple([int(numpy.prod(shape)) for shape in shapes])
    return (
        dpnp.transpose(a, transpose_axes).reshape(new_sh),
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
    >>> import dpnp.dpnp_utils.dpnp_utils_einsum as np_util
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
    >>> import dpnp.dpnp_utils.dpnp_utils_einsum as np_util
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
    >>> import dpnp.dpnp_utils.dpnp_utils_einsum as np_util
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
    >>> import dpnp.dpnp_utils.dpnp_utils_einsum as np_util
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

    if not isinstance(
        list_subscript, (list, tuple, numpy.ndarray, dpnp.ndarray)
    ):
        raise TypeError(
            "subscripts for each operand must be a list, tuple or ndarray."
        )
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

            if not 0 <= s < len(_einsum_symbols):
                raise ValueError(
                    f"subscript is not within the valid range [0, {len(_einsum_symbols)})."
                )
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
    return dpnp_array(
        shape,
        dtype=a.dtype,
        buffer=a,
        strides=strides,
        usm_type=a.usm_type,
        sycl_queue=a.sycl_queue,
    )


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
    if out is not None:
        dpnp.check_supported_arrays_type(out)
        if dpctl.utils.get_execution_queue((exec_q, out.sycl_queue)) is None:
            raise ExecutionPlacementError(
                "Input and output allocation queues are not compatible"
            )

    for id, a in enumerate(operands):
        if dpnp.isscalar(a):
            scalar_dtype = map_dtype_to_device(type(a), exec_q.sycl_device)
            operands[id] = dpnp.array(
                a, dtype=scalar_dtype, usm_type=res_usm_type, sycl_queue=exec_q
            )
            arrays.append(operands[id])
    result_dtype = dpnp.result_type(*arrays) if dtype is None else dtype
    if order is not None and order in "aA":
        order = "F" if all(arr.flags.fnc for arr in arrays) else "C"

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
            repeated_subscript = []
            for label in output_subscript:
                if output_subscript.count(label) >= 2:
                    repeated_subscript.append(_chr(label))
            raise ValueError(
                "einstein sum subscripts string includes output "
                f"subscript {set(repeated_subscript)} multiple times."
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
            dpnp.astype(a, result_dtype, copy=False, casting=casting)
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

    arr_out = dpnp.asarray(arr_out, order=order)
    assert returns_view or arr_out.dtype == result_dtype
    return dpnp.get_result_array(arr_out, out, casting=casting)
