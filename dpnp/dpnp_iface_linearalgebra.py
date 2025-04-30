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
Interface of the Linear Algebra part of the DPNP

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""

# pylint: disable=no-name-in-module
import numpy

import dpnp

from .dpnp_utils import map_dtype_to_device
from .dpnp_utils.dpnp_utils_einsum import dpnp_einsum
from .dpnp_utils.dpnp_utils_linearalgebra import (
    dpnp_dot,
    dpnp_kron,
    dpnp_multiplication,
    dpnp_tensordot,
    dpnp_vecdot,
)

__all__ = [
    "dot",
    "einsum",
    "einsum_path",
    "inner",
    "kron",
    "matmul",
    "matvec",
    "outer",
    "tensordot",
    "vdot",
    "vecdot",
    "vecmat",
]


# TODO: implement a specific scalar-array kernel
def _call_multiply(a, b, out=None, outer_calc=False):
    """
    Adjusted multiply function for handling special cases of scalar-array dot
    products in linear algebra.

    `dpnp.multiply` cannot directly be used for calculating scalar-array dots,
    because the output dtype of multiply is not the same as the expected dtype
    for scalar-array dots. For example, if `sc` is an scalar and `a` is an
    array of type `float32`, then `dpnp.multiply(a, sc).dtype == dpnp.float32`
    (similar to NumPy). However, for scalar-array dots, such as the dot
    function, we need `dpnp.dot(a, sc).dtype == dpnp.float64` to align with
    NumPy. This functions adjusts the behavior of `dpnp.multiply` function to
    meet this requirement.

    """

    sc, arr = (a, b) if dpnp.isscalar(a) else (b, a)
    sc_dtype = map_dtype_to_device(type(sc), arr.sycl_device)
    res_dtype = dpnp.result_type(sc_dtype, arr)
    multiply_func = dpnp.multiply.outer if outer_calc else dpnp.multiply
    if out is not None and out.dtype == arr.dtype:
        res = multiply_func(a, b, out=out)
    else:
        res = multiply_func(a, b, dtype=res_dtype)
    return dpnp.get_result_array(res, out, casting="no")


def dot(a, b, out=None):
    """
    Dot product of `a` and `b`.

    For full documentation refer to :obj:`numpy.dot`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray, scalar}
        First input array. Both inputs `a` and `b` can not be scalars
        at the same time.
    b : {dpnp.ndarray, usm_ndarray, scalar}
        Second input array. Both inputs `a` and `b` can not be scalars
        at the same time.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. It must have
        the same shape and data type as the expected output and should be
        C-contiguous. If these conditions are not met, an exception is
        raised, instead of attempting to be flexible.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        Returns the dot product of `a` and `b`.
        If `out` is given, then it is returned.

    See Also
    --------
    :obj:`dpnp.ndarray.dot` : Equivalent method.
    :obj:`dpnp.tensordot` : Sum products over arbitrary axes.
    :obj:`dpnp.vdot` : Complex-conjugating dot product.
    :obj:`dpnp.einsum` : Einstein summation convention.
    :obj:`dpnp.matmul` : Matrix product of two arrays.
    :obj:`dpnp.linalg.multi_dot` : Chained dot product.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([1, 2, 3])
    >>> np.dot(a, b)
    array(14)

    Neither argument is complex-conjugated:

    >>> np.dot(np.array([2j, 3j]), np.array([2j, 3j]))
    array(-13+0j)

    For 2-D arrays it is the matrix product:

    >>> a = np.array([[1, 0], [0, 1]])
    >>> b = np.array([[4, 1], [2, 2]])
    >>> np.dot(a, b)
    array([[4, 1],
           [2, 2]])

    >>> a = np.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6))
    >>> b = np.arange(3 * 4 * 5 * 6)[::-1].reshape((5, 4, 6, 3))
    >>> np.dot(a, b)[2, 3, 2, 1, 2, 2]
    array(499128)
    >>> sum(a[2, 3, 2, :] * b[1, 2, :, 2])
    array(499128)

    """

    dpnp.check_supported_arrays_type(a, b, scalar_type=True)

    if out is not None:
        dpnp.check_supported_arrays_type(out)
        if not out.flags.c_contiguous:
            raise ValueError("Only C-contiguous array is acceptable.")

    if dpnp.isscalar(a) or dpnp.isscalar(b):
        return _call_multiply(a, b, out=out)

    a_ndim = a.ndim
    b_ndim = b.ndim
    if a_ndim == 0 or b_ndim == 0:
        # TODO: use specific scalar-vector kernel
        return dpnp.multiply(a, b, out=out)

    # numpy.dot does not allow casting even if it is safe
    # casting="no" is used in the following
    if a_ndim == 1 and b_ndim == 1:
        return dpnp_dot(a, b, out=out, casting="no")

    if a_ndim == 2 and b_ndim == 2:
        return dpnp.matmul(a, b, out=out, casting="no")

    if a_ndim == 1 or b_ndim == 1:
        return dpnp.matmul(a, b, out=out, casting="no")

    # TODO: investigate usage of matmul for some possible
    # use cases instead of dpnp.tensordot
    result = dpnp.tensordot(a, b, axes=(-1, -2))
    return dpnp.get_result_array(result, out, casting="no")


def einsum(
    *operands,
    out=None,
    dtype=None,
    order="K",
    casting="same_kind",
    optimize=False,
):
    """
    einsum(subscripts, *operands, out=None, dtype=None, order="K", \
        casting="same_kind", optimize=False)

    Evaluates the Einstein summation convention on the operands.

    For full documentation refer to :obj:`numpy.einsum`.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation as comma separated list of
        subscript labels. An implicit (classical Einstein summation)
        calculation is performed unless the explicit indicator '->' is
        included as well as subscript labels of the precise output form.
    *operands : sequence of {dpnp.ndarrays, usm_ndarray}
        These are the arrays for the operation.
    out : {dpnp.ndarrays, usm_ndarray, None}, optional
        If provided, the calculation is done into this array.

        Default: ``None``.
    dtype : {None, str, dtype object}, optional
        If provided, forces the calculation to use the data type specified.

        Default: ``None``.
    order : {None, "C", "F", "A", "K"}, optional
        Controls the memory layout of the output. ``"C"`` means it should be
        C-contiguous. ``"F"`` means it should be F-contiguous, ``"A"`` means
        it should be ``"F"`` if the inputs are all ``"F"``, ``"C"`` otherwise.
        ``"K"`` means it should be as close to the layout as the inputs as
        is possible, including arbitrarily permuted axes. ``order=None`` is
        equivalent to ``order="K"``.

        Default: ``"K"``.
    casting : {"no", "equiv", "safe", "same_kind", "unsafe"}, optional
        Controls what kind of data casting may occur. Setting this to
        ``"unsafe"`` is not recommended, as it can adversely affect
        accumulations.

          * ``"no"`` means the data types should not be cast at all.
          * ``"equiv"`` means only byte-order changes are allowed.
          * ``"safe"`` means only casts which can preserve values are allowed.
          * ``"same_kind"`` means only safe casts or casts within a kind,
            like float64 to float32, are allowed.
          * ``"unsafe"`` means any data conversions may be done.

        Please note that, in contrast to NumPy, the default setting here is
        ``"same_kind"``. This is to prevent errors that may occur when data
        needs to be converted to `float64`, but the device does not support it.
        In such cases, the data is instead converted to `float32`.

        Default: ``"same_kind"``.
    optimize : {False, True, "greedy", "optimal"}, optional
        Controls if intermediate optimization should occur. No optimization
        will occur if ``False`` and ``True`` will default to the ``"greedy"``
        algorithm. Also accepts an explicit contraction list from the
        :obj:`dpnp.einsum_path` function.

        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        The calculation based on the Einstein summation convention.

    See Also
    -------
    :obj:`dpnp.einsum_path` : Evaluates the lowest cost contraction order
                              for an einsum expression.
    :obj:`dpnp.dot` : Returns the dot product of two arrays.
    :obj:`dpnp.inner` : Returns the inner product of two arrays.
    :obj:`dpnp.outer` : Returns the outer product of two arrays.
    :obj:`dpnp.tensordot` :  Sum products over arbitrary axes.
    :obj:`dpnp.linalg.multi_dot` : Chained dot product.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.arange(25).reshape(5, 5)
    >>> b = np.arange(5)
    >>> c = np.arange(6).reshape(2, 3)

    Trace of a matrix:

    >>> np.einsum("ii", a)
    array(60)
    >>> np.einsum(a, [0, 0])
    array(60)
    >>> np.trace(a)
    array(60)

    Extract the diagonal (requires explicit form):

    >>> np.einsum("ii->i", a)
    array([ 0,  6, 12, 18, 24])
    >>> np.einsum(a, [0, 0], [0])
    array([ 0,  6, 12, 18, 24])
    >>> np.diag(a)
    array([ 0,  6, 12, 18, 24])

    Sum over an axis (requires explicit form):

    >>> np.einsum("ij->i", a)
    array([ 10,  35,  60,  85, 110])
    >>> np.einsum(a, [0, 1], [0])
    array([ 10,  35,  60,  85, 110])
    >>> np.sum(a, axis=1)
    array([ 10,  35,  60,  85, 110])

    For higher dimensional arrays summing a single axis can be done
    with ellipsis:

    >>> np.einsum("...j->...", a)
    array([ 10,  35,  60,  85, 110])
    >>> np.einsum(a, [Ellipsis,1], [Ellipsis])
    array([ 10,  35,  60,  85, 110])

    Compute a matrix transpose, or reorder any number of axes:

    >>> np.einsum("ji", c)
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> np.einsum("ij->ji", c)
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> np.einsum(c, [1, 0])
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> np.transpose(c)
    array([[0, 3],
           [1, 4],
           [2, 5]])

    Vector inner products:

    >>> np.einsum("i,i", b, b)
    array(30)
    >>> np.einsum(b, [0], b, [0])
    array(30)
    >>> np.inner(b, b)
    array(30)

    Matrix vector multiplication:

    >>> np.einsum("ij,j", a, b)
    array([ 30,  80, 130, 180, 230])
    >>> np.einsum(a, [0, 1], b, [1])
    array([ 30,  80, 130, 180, 230])
    >>> np.dot(a, b)
    array([ 30,  80, 130, 180, 230])
    >>> np.einsum("...j,j", a, b)
    array([ 30,  80, 130, 180, 230])

    Broadcasting and scalar multiplication:

    >>> np.einsum("..., ...", 3, c)
    array([[ 0,  3,  6],
           [ 9, 12, 15]])
    >>> np.einsum(",ij", 3, c)
    array([[ 0,  3,  6],
           [ 9, 12, 15]])
    >>> np.einsum(3, [Ellipsis], c, [Ellipsis])
    array([[ 0,  3,  6],
           [ 9, 12, 15]])
    >>> np.multiply(3, c)
    array([[ 0,  3,  6],
           [ 9, 12, 15]])

    Vector outer product:

    >>> np.einsum("i,j", np.arange(2)+1, b)
    array([[0, 1, 2, 3, 4],
           [0, 2, 4, 6, 8]])
    >>> np.einsum(np.arange(2)+1, [0], b, [1])
    array([[0, 1, 2, 3, 4],
           [0, 2, 4, 6, 8]])
    >>> np.outer(np.arange(2)+1, b)
    array([[0, 1, 2, 3, 4],
           [0, 2, 4, 6, 8]])

    Tensor contraction:

    >>> a = np.arange(60.).reshape(3, 4, 5)
    >>> b = np.arange(24.).reshape(4, 3, 2)
    >>> np.einsum("ijk,jil->kl", a, b)
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])
    >>> np.einsum(a, [0, 1, 2], b, [1, 0, 3], [2, 3])
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])
    >>> np.tensordot(a, b, axes=([1, 0],[0, 1]))
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])

    Example of ellipsis use:

    >>> a = np.arange(6).reshape((3, 2))
    >>> b = np.arange(12).reshape((4, 3))
    >>> np.einsum("ki,jk->ij", a, b)
    array([[10, 28, 46, 64],
           [13, 40, 67, 94]])
    >>> np.einsum("ki,...k->i...", a, b)
    array([[10, 28, 46, 64],
           [13, 40, 67, 94]])
    >>> np.einsum("k...,jk", a, b)
    array([[10, 28, 46, 64],
           [13, 40, 67, 94]])

    Chained array operations. For more complicated contractions, speed ups
    might be achieved by repeatedly computing a "greedy" path or computing
    the "optimal" path in advance and repeatedly applying it, using an
    `einsum_path` insertion. Performance improvements can be particularly
    significant with larger arrays:

    >>> a = np.ones(64000).reshape(20, 40, 80)

    Basic `einsum`: 119 ms ± 26 ms per loop (evaluated on 12th
    Gen Intel\u00ae Core\u2122 i7 processor)

    >>> %timeit np.einsum("ijk,ilm,njm,nlk,abc->",a, a, a, a, a)

    Sub-optimal `einsum`: 32.9 ms ± 5.1 ms per loop

    >>> %timeit np.einsum(
        "ijk,ilm,njm,nlk,abc->", a, a, a, a, a, optimize="optimal"
    )

    Greedy `einsum`: 28.6 ms ± 4.8 ms per loop

    >>> %timeit np.einsum(
        "ijk,ilm,njm,nlk,abc->", a, a, a, a, a, optimize="greedy"
    )

    Optimal `einsum`: 26.9 ms ± 6.3 ms per loop

    >>> path = np.einsum_path(
        "ijk,ilm,njm,nlk,abc->", a, a, a, a, a, optimize="optimal"
    )[0]
    >>> %timeit np.einsum("ijk,ilm,njm,nlk,abc->", a, a, a, a, a, optimize=path)

    """

    if optimize is True:
        optimize = "greedy"

    return dpnp_einsum(
        *operands,
        out=out,
        dtype=dtype,
        order=order,
        casting=casting,
        optimize=optimize,
    )


def einsum_path(*operands, optimize="greedy"):
    """
    einsum_path(subscripts, *operands, optimize="greedy")

    Evaluates the lowest cost contraction order for an einsum expression
    by considering the creation of intermediate arrays.

    For full documentation refer to :obj:`numpy.einsum_path`.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation.
    *operands : sequence of arrays
        These are the arrays for the operation in any form that can be
        converted to an array. This includes scalars, lists, lists of
        tuples, tuples, tuples of tuples, tuples of lists, and ndarrays.
    optimize : {bool, list, tuple, None, "greedy", "optimal"}
        Choose the type of path. If a tuple is provided, the second argument is
        assumed to be the maximum intermediate size created. If only a single
        argument is provided the largest input or output array size is used
        as a maximum intermediate size.

        * if a list is given that starts with ``einsum_path``, uses this as the
          contraction path
        * if ``False`` or ``None`` no optimization is taken
        * if ``True`` defaults to the ``"greedy"`` algorithm
        * ``"optimal"`` is an algorithm that combinatorially explores all
          possible ways of contracting the listed tensors and chooses the
          least costly path. Scales exponentially with the number of terms
          in the contraction.
        * ``"greedy"`` is an algorithm that chooses the best pair contraction
          at each step. Effectively, this algorithm searches the largest inner,
          Hadamard, and then outer products at each step. Scales cubically with
          the number of terms in the contraction. Equivalent to the
          ``"optimal"`` path for most contractions.

        Default: ``"greedy"``.

    Returns
    -------
    path : list of tuples
        A list representation of the einsum path.
    string_repr : str
        A printable representation of the einsum path.

    Notes
    -----
    The resulting path indicates which terms of the input contraction should be
    contracted first, the result of this contraction is then appended to the
    end of the contraction list. This list can then be iterated over until all
    intermediate contractions are complete.

    See Also
    --------
    :obj:`dpnp.einsum` : Evaluates the Einstein summation convention
                         on the operands.
    :obj:`dpnp.linalg.multi_dot` : Chained dot product.
    :obj:`dpnp.dot` : Returns the dot product of two arrays.
    :obj:`dpnp.inner` : Returns the inner product of two arrays.
    :obj:`dpnp.outer` : Returns the outer product of two arrays.

    Examples
    --------
    We can begin with a chain dot example. In this case, it is optimal to
    contract the `b` and `c` tensors first as represented by the first element
    of the path ``(1, 2)``. The resulting tensor is added to the end of the
    contraction and the remaining contraction ``(0, 1)`` is then completed.

    >>> import dpnp as np
    >>> np.random.seed(123)
    >>> a = np.random.rand(2, 2)
    >>> b = np.random.rand(2, 5)
    >>> c = np.random.rand(5, 2)
    >>> path_info = np.einsum_path("ij,jk,kl->il", a, b, c, optimize="greedy")

    >>> print(path_info[0])
    ['einsum_path', (1, 2), (0, 1)]

    >>> print(path_info[1])
      Complete contraction:  ij,jk,kl->il # may vary
             Naive scaling:  4
         Optimized scaling:  3
          Naive FLOP count:  1.200e+02
      Optimized FLOP count:  5.700e+01
       Theoretical speedup:  2.105
      Largest intermediate:  4.000e+00 elements
    -------------------------------------------------------------------------
    scaling                  current                                remaining
    -------------------------------------------------------------------------
       3                   kl,jk->jl                                ij,jl->il
       3                   jl,ij->il                                   il->il

    A more complex index transformation example.

    >>> I = np.random.rand(10, 10, 10, 10)
    >>> C = np.random.rand(10, 10)
    >>> path_info = np.einsum_path(
            "ea,fb,abcd,gc,hd->efgh", C, C, I, C, C, optimize="greedy"
        )
    >>> print(path_info[0])
    ['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)]
    >>> print(path_info[1])
      Complete contraction:  ea,fb,abcd,gc,hd->efgh # may vary
             Naive scaling:  8
         Optimized scaling:  5
          Naive FLOP count:  5.000e+08
      Optimized FLOP count:  8.000e+05
       Theoretical speedup:  624.999
      Largest intermediate:  1.000e+04 elements
    --------------------------------------------------------------------------
    scaling                  current                                remaining
    --------------------------------------------------------------------------
       5               abcd,ea->bcde                      fb,gc,hd,bcde->efgh
       5               bcde,fb->cdef                         gc,hd,cdef->efgh
       5               cdef,gc->defg                            hd,defg->efgh
       5               defg,hd->efgh                               efgh->efgh

    """

    # explicit casting to numpy array if applicable
    operands = [
        dpnp.asnumpy(x) if dpnp.is_supported_array_type(x) else x
        for x in operands
    ]

    return numpy.einsum_path(
        *operands,
        optimize=optimize,
        einsum_call=False,
    )


def inner(a, b):
    """
    Returns the inner product of two arrays.

    For full documentation refer to :obj:`numpy.inner`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray, scalar}
        First input array. Both inputs `a` and `b` can not be scalars
        at the same time.
    b : {dpnp.ndarray, usm_ndarray, scalar}
        Second input array. Both inputs `a` and `b` can not be scalars
        at the same time.

    Returns
    -------
    out : dpnp.ndarray
        If either `a` or `b` is a scalar, the shape of the returned arrays
        matches that of the array between `a` and `b`, whichever is an array.
        If `a` and `b` are both 1-D arrays then a 0-d array is returned;
        otherwise an array with a shape as
        ``out.shape = (*a.shape[:-1], *b.shape[:-1])`` is returned.


    See Also
    --------
    :obj:`dpnp.einsum` : Einstein summation convention.
    :obj:`dpnp.dot` : Generalized matrix product,
                      using second last dimension of `b`.
    :obj:`dpnp.tensordot` : Sum products over arbitrary axes.

    Examples
    --------
    # Ordinary inner product for vectors

    >>> import dpnp as np
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([0, 1, 0])
    >>> np.inner(a, b)
    array(2)

    # Some multidimensional examples

    >>> a = np.arange(24).reshape((2, 3, 4))
    >>> b = np.arange(4)
    >>> c = np.inner(a, b)
    >>> c.shape
    (2, 3)
    >>> c
    array([[ 14,  38,  62],
           [86, 110, 134]])

    >>> a = np.arange(2).reshape((1, 1, 2))
    >>> b = np.arange(6).reshape((3, 2))
    >>> c = np.inner(a, b)
    >>> c.shape
    (1, 1, 3)
    >>> c
    array([[[1, 3, 5]]])

    An example where `b` is a scalar

    >>> np.inner(np.eye(2), 7)
    array([[7., 0.],
           [0., 7.]])

    """

    dpnp.check_supported_arrays_type(a, b, scalar_type=True)

    if dpnp.isscalar(a) or dpnp.isscalar(b):
        return _call_multiply(a, b)

    if a.ndim == 0 or b.ndim == 0:
        # TODO: use specific scalar-vector kernel
        return dpnp.multiply(a, b)

    if a.shape[-1] != b.shape[-1]:
        raise ValueError(
            "shape of input arrays is not similar at the last axis."
        )

    if a.ndim == 1 and b.ndim == 1:
        return dpnp_dot(a, b)

    return dpnp.tensordot(a, b, axes=(-1, -1))


def kron(a, b):
    """
    Kronecker product of two arrays.

    Computes the Kronecker product, a composite array made of blocks of the
    second array scaled by the first.

    For full documentation refer to :obj:`numpy.kron`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray, scalar}
        First input array. Both inputs `a` and `b` can not be scalars
        at the same time.
    b : {dpnp.ndarray, usm_ndarray, scalar}
        Second input array. Both inputs `a` and `b` can not be scalars
        at the same time.

    Returns
    -------
    out : dpnp.ndarray
        Returns the Kronecker product.

    See Also
    --------
    :obj:`dpnp.outer` : Returns the outer product of two arrays.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 10, 100])
    >>> b = np.array([5, 6, 7])
    >>> np.kron(a, b)
    array([  5,   6,   7, ..., 500, 600, 700])
    >>> np.kron(b, a)
    array([  5,  50, 500, ...,   7,  70, 700])

    >>> np.kron(np.eye(2), np.ones((2, 2)))
    array([[1.,  1.,  0.,  0.],
           [1.,  1.,  0.,  0.],
           [0.,  0.,  1.,  1.],
           [0.,  0.,  1.,  1.]])

    >>> a = np.arange(100).reshape((2, 5, 2, 5))
    >>> b = np.arange(24).reshape((2, 3, 4))
    >>> c = np.kron(a, b)
    >>> c.shape
    (2, 10, 6, 20)
    >>> I = (1, 3, 0, 2)
    >>> J = (0, 2, 1)
    >>> J1 = (0,) + J             # extend to ndim=4
    >>> S1 = (1,) + b.shape
    >>> K = tuple(np.array(I) * np.array(S1) + np.array(J1))
    >>> c[K] == a[I]*b[J]
    array(True)

    """

    dpnp.check_supported_arrays_type(a, b, scalar_type=True)

    if dpnp.isscalar(a) or dpnp.isscalar(b):
        return _call_multiply(a, b)

    a_ndim = a.ndim
    b_ndim = b.ndim
    if a_ndim == 0 or b_ndim == 0:
        # TODO: use specific scalar-vector kernel
        return dpnp.multiply(a, b)

    return dpnp_kron(a, b, a_ndim, b_ndim)


def matmul(
    x1,
    x2,
    /,
    out=None,
    *,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
    signature=None,
    axes=None,
    axis=None,
):
    """
    Matrix product of two arrays.

    For full documentation refer to :obj:`numpy.matmul`.

    Parameters
    ----------
    x1 : {dpnp.ndarray, usm_ndarray}
        First input array.
    x2 : {dpnp.ndarray, usm_ndarray}
        Second input array.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. It must have
        a shape that matches the signature `(n,k),(k,m)->(n,m)` but the type
        (of the calculated values) will be cast if necessary.

        Default: ``None``.
    dtype : {None, str, dtype object}, optional
        Type to use in computing the matrix product. By default, the returned
        array will have data type that is determined by considering
        Promotion Type Rule and device capabilities.

        Default: ``None``.
    casting : {"no", "equiv", "safe", "same_kind", "unsafe"}, optional
        Controls what kind of data casting may occur.

        Default: ``"same_kind"``.
    order : {None, "C", "F", "A", "K"}, optional
        Memory layout of the newly output array, if parameter `out` is ``None``.

        Default: ``"K"``.
    axes : {None, list of tuples}, optional
        A list of tuples with indices of axes the matrix product should operate
        on. For instance, for the signature of ``(i,j),(j,k)->(i,k)``, the base
        elements are 2d matrices and these are taken to be stored in the two
        last axes of each argument. The corresponding axes keyword would be
        ``[(-2, -1), (-2, -1), (-2, -1)]``.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        Returns the matrix product of the inputs.
        This is a 0-d array only when both `x1`, `x2` are 1-d vectors.

    Limitations
    -----------
    Keyword arguments `subok`, and `signature`, are only supported with their
    default values. Otherwise ``NotImplementedError`` exception will be raised.

    See Also
    --------
    :obj:`dpnp.linalg.matmul` : Array API compatible version.
    :obj:`dpnp.vecdot` : Complex-conjugating dot product for stacks of vectors.
    :obj:`dpnp.matvec` : Matrix-vector product for stacks of
                         matrices and vectors.
    :obj:`dpnp.vecmat` : Vector-matrix product for stacks of
                         vectors and matrices.
    :obj:`dpnp.tensordot` : Sum products over arbitrary axes.
    :obj:`dpnp.einsum` : Einstein summation convention.
    :obj:`dpnp.dot` : Alternative matrix product with
                      different broadcasting rules.

    Notes
    -----
    The behavior depends on the arguments in the following way.

    - If both arguments are 2-D they are multiplied like conventional matrices.
    - If either argument is N-D, N > 2, it is treated as a stack of matrices
      residing in the last two indexes and broadcast accordingly.
    - If the first argument is 1-D, it is promoted to a matrix by prepending
      a 1 to its dimensions. After matrix multiplication the prepended 1 is
      removed. (For stacks of vectors, use :obj:`dpnp.vecmat`.)
    - If the second argument is 1-D, it is promoted to a matrix by appending
      a 1 to its dimensions. After matrix multiplication the appended 1 is
      removed. (For stacks of vectors, use :obj:`dpnp.matvec`.)

    :obj:`dpnp.matmul` differs from :obj:`dpnp.dot` in two important ways:

    - Multiplication by scalars is not allowed, use ``*`` instead.
    - Stacks of matrices are broadcast together as if the matrices
      were elements, respecting the signature ``(n,k),(k,m)->(n,m)``:

    >>> import dpnp as np
    >>> a = np.ones([9, 5, 7, 4])
    >>> c = np.ones([9, 5, 4, 3])
    >>> np.dot(a, c).shape
    (9, 5, 7, 9, 5, 3)
    >>> np.matmul(a, c).shape
    (9, 5, 7, 3) # n is 7, k is 4, m is 3

    Examples
    --------
    For 2-D arrays it is the matrix product:

    >>> import dpnp as np
    >>> a = np.array([[1, 0], [0, 1]])
    >>> b = np.array([[4, 1], [2, 2]])
    >>> np.matmul(a, b)
    array([[4, 1],
           [2, 2]])

    For 2-D mixed with 1-D, the result is the usual.

    >>> a = np.array([[1, 0], [0, 1]])
    >>> b = np.array([1, 2])
    >>> np.matmul(a, b)
    array([1, 2])
    >>> np.matmul(b, a)
    array([1, 2])

    Broadcasting is conventional for stacks of arrays

    >>> a = np.arange(2 * 2 * 4).reshape((2, 2, 4))
    >>> b = np.arange(2 * 2 * 4).reshape((2, 4, 2))
    >>> np.matmul(a, b).shape
    (2, 2, 2)
    >>> np.matmul(a, b)[0, 1, 1]
    array(98)
    >>> np.sum(a[0, 1, :] * b[0 , :, 1])
    array(98)

    Vector, vector returns the scalar inner product, but neither argument
    is complex-conjugated:

    >>> x1 = np.array([2j, 3j])
    >>> x2 = np.array([2j, 3j])
    >>> np.matmul(x1, x2)
    array(-13+0j)

    The ``@`` operator can be used as a shorthand for :obj:`dpnp.matmul` on
    :class:`dpnp.ndarray`.

    >>> x1 @ x2
    array(-13+0j)

    """

    dpnp.check_limitations(subok_linalg=subok, signature=signature)
    dpnp.check_supported_arrays_type(x1, x2)

    return dpnp_multiplication(
        "matmul",
        x1,
        x2,
        out=out,
        casting=casting,
        order=order,
        dtype=dtype,
        axes=axes,
        axis=axis,
    )


def matvec(
    x1,
    x2,
    /,
    out=None,
    *,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
    signature=None,
    axes=None,
    axis=None,
):
    r"""
    Matrix-vector dot product of two arrays.

    Given a matrix (or stack of matrices) :math:`\mathbf{A}` in `x1` and
    a vector (or stack of vectors) :math:`\mathbf{v}` in `x2`, the
    matrix-vector product is defined as:

    .. math::
       \mathbf{A} \cdot \mathbf{b} = \sum_{i=0}^{n-1} A_{ij}v_j

    where the sum is over the last dimensions in `x1` and `x2` (unless `axes`
    is specified). (For a matrix-vector product with the vector conjugated,
    use ``dpnp.vecmat(x2, x1.mT)``.)

    For full documentation refer to :obj:`numpy.matvec`.

    Parameters
    ----------
    x1 : {dpnp.ndarray, usm_ndarray}
        First input array.
    x2 : {dpnp.ndarray, usm_ndarray}
        Second input array.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        A location into which the result is stored. If provided, it must have
        the broadcasted shape of `x1` and `x2` with the summation axis removed.
        If not provided or ``None``, a freshly-allocated array is used.

        Default: ``None``.
    dtype : {None, str, dtype object}, optional
        Type to use in computing the matrix product. By default, the returned
        array will have data type that is determined by considering
        Promotion Type Rule and device capabilities.

        Default: ``None``.
    casting : {"no", "equiv", "safe", "same_kind", "unsafe"}, optional
        Controls what kind of data casting may occur.

        Default: ``"same_kind"``.
    order : {None, "C", "F", "A", "K"}, optional
        Memory layout of the newly output array, if parameter `out` is ``None``.

        Default: ``"K"``.
    axes : {None, list of tuples}, optional
        A list of tuples with indices of axes the matrix-vector product should
        operate on. For instance, for the signature of ``(i,j),(j)->(i)``, the
        base elements are 2d matrices and 1d vectors, where the matrices are
        assumed to be stored in the last two axes of the first argument, and
        the vectors in the last axis of the second argument. The corresponding
        axes keyword would be ``[(-2, -1), (-1,), (-1,)]``.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        Returns the matrix-vector product of the inputs.

    Limitations
    -----------
    Keyword arguments `subok`, and `signature` are only supported with their
    default values. Otherwise ``NotImplementedError`` exception will be raised.

    See Also
    --------
    :obj:`dpnp.vecdot` : Vector-vector product.
    :obj:`dpnp.vecmat` : Vector-matrix product.
    :obj:`dpnp.matmul` : Matrix-matrix product.
    :obj:`dpnp.einsum` : Einstein summation convention.


    Examples
    --------
    Rotate a set of vectors from Y to X along Z.

    >>> import dpnp as np
    >>> a = np.array([[0., 1., 0.],
    ...               [-1., 0., 0.],
    ...               [0., 0., 1.]])
    >>> v = np.array([[1., 0., 0.],
    ...               [0., 1., 0.],
    ...               [0., 0., 1.],
    ...               [0., 6., 8.]])
    >>> np.matvec(a, v)
    array([[ 0., -1.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 6.,  0.,  8.]])

    """

    dpnp.check_limitations(subok_linalg=subok, signature=signature)
    dpnp.check_supported_arrays_type(x1, x2)

    return dpnp_multiplication(
        "matvec",
        x1,
        x2,
        out=out,
        casting=casting,
        order=order,
        dtype=dtype,
        axes=axes,
        axis=axis,
    )


def outer(a, b, out=None):
    """
    Returns the outer product of two arrays.

    For full documentation refer to :obj:`numpy.outer`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        First input vector. Input is flattened if not already 1-dimensional.
    b : {dpnp.ndarray, usm_ndarray}
        Second input vector. Input is flattened if not already 1-dimensional.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        A location where the result is stored.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        ``out[i, j] = a[i] * b[j]``

    See Also
    --------
    :obj:`dpnp.linalg.outer` : Array API compatible version.
    :obj:`dpnp.einsum` : Evaluates the Einstein summation convention
                         on the operands.
    :obj:`dpnp.inner` : Returns the inner product of two arrays.
    :obj:`dpnp.tensordot` : dpnp.tensordot(a.ravel(), b.ravel(), axes=((), ()))
                            is the equivalent.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 1, 1])
    >>> b = np.array([1, 2, 3])
    >>> np.outer(a, b)
    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]])

    Make a (*very* coarse) grid for computing a Mandelbrot set:

    >>> rl = np.outer(np.ones((5,)), np.linspace(-2, 2, 5))
    >>> rl
    array([[-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.]])
    >>> im = np.outer(1j*np.linspace(2, -2, 5), np.ones((5,)))
    >>> im
    array([[0.+2.j, 0.+2.j, 0.+2.j, 0.+2.j, 0.+2.j],
           [0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.-1.j, 0.-1.j, 0.-1.j, 0.-1.j, 0.-1.j],
           [0.-2.j, 0.-2.j, 0.-2.j, 0.-2.j, 0.-2.j]])
    >>> grid = rl + im
    >>> grid
    array([[-2.+2.j, -1.+2.j,  0.+2.j,  1.+2.j,  2.+2.j],
           [-2.+1.j, -1.+1.j,  0.+1.j,  1.+1.j,  2.+1.j],
           [-2.+0.j, -1.+0.j,  0.+0.j,  1.+0.j,  2.+0.j],
           [-2.-1.j, -1.-1.j,  0.-1.j,  1.-1.j,  2.-1.j],
           [-2.-2.j, -1.-2.j,  0.-2.j,  1.-2.j,  2.-2.j]])

    """

    dpnp.check_supported_arrays_type(a, b, scalar_type=True, all_scalars=False)
    if dpnp.isscalar(a):
        x2 = dpnp.ravel(b)[None, :]
        result = _call_multiply(a, x2, out=out, outer_calc=True)
    elif dpnp.isscalar(b):
        x1 = dpnp.ravel(a)[:, None]
        result = _call_multiply(x1, b, out=out, outer_calc=True)
    else:
        result = dpnp.multiply.outer(dpnp.ravel(a), dpnp.ravel(b), out=out)

    return result


def tensordot(a, b, axes=2):
    r"""
    Compute tensor dot product along specified axes.

    Given two tensors, `a` and `b`, and an array_like object containing
    two array_like objects, ``(a_axes, b_axes)``, sum the products of
    `a`'s and `b`'s elements (components) over the axes specified by
    ``a_axes`` and ``b_axes``. The third argument can be a single non-negative
    integer_like scalar, ``N``; if it is such, then the last ``N`` dimensions
    of `a` and the first ``N`` dimensions of `b` are summed over.

    For full documentation refer to :obj:`numpy.tensordot`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray, scalar}
        First input array. Both inputs `a` and `b` can not be scalars
        at the same time.
    b : {dpnp.ndarray, usm_ndarray, scalar}
        Second input array. Both inputs `a` and `b` can not be scalars
        at the same time.
    axes : int or (2,) array_like
        * integer_like: If an int `N`, sum over the last `N` axes of `a` and
          the first `N` axes of `b` in order. The sizes of the corresponding
          axes must match.
        * (2,) array_like: A list of axes to be summed over, first sequence
          applying to `a`, second to `b`. Both elements array_like must be of
          the same length.

    Returns
    -------
    out : dpnp.ndarray
        Returns the tensor dot product of `a` and `b`.

    See Also
    --------
    :obj:`dpnp.linalg.tensordot` : Equivalent function.
    :obj:`dpnp.dot` : Returns the dot product.
    :obj:`dpnp.einsum` : Evaluates the Einstein summation convention
                         on the operands.

    Notes
    -----
    Three common use cases are:
        * ``axes = 0`` : tensor product :math:`a \otimes b`
        * ``axes = 1`` : tensor dot product :math:`a \cdot b`
        * ``axes = 2`` : (default) tensor double contraction :math:`a:b`

    When `axes` is integer, the sequence for evaluation will be: first
    the -Nth axis in `a` and 0th axis in `b`, and the -1th axis in `a` and
    Nth axis in `b` last.

    When there is more than one axis to sum over - and they are not the last
    (first) axes of `a` (`b`) - the argument `axes` should consist of
    two sequences of the same length, with the first axis to sum over given
    first in both sequences, the second axis second, and so forth.

    The shape of the result consists of the non-contracted axes of the
    first tensor, followed by the non-contracted axes of the second.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> b = np.array([1, 2, 3])
    >>> np.tensordot(a, b, 1)
    array([14, 32, 50])

    >>> a = np.arange(60.).reshape(3, 4, 5)
    >>> b = np.arange(24.).reshape(4, 3, 2)
    >>> c = np.tensordot(a, b, axes=([1, 0], [0, 1]))
    >>> c.shape
    (5, 2)
    >>> c
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])

    A slower but equivalent way of computing the same...

    >>> d = np.zeros((5, 2))
    >>> for i in range(5):
    ...   for j in range(2):
    ...     for k in range(3):
    ...       for n in range(4):
    ...         d[i, j] += a[k, n, i] * b[n, k, j]
    >>> c == d
    array([[ True,  True],
           [ True,  True],
           [ True,  True],
           [ True,  True],
           [ True,  True]])

    """

    dpnp.check_supported_arrays_type(a, b, scalar_type=True)

    if dpnp.isscalar(a) or dpnp.isscalar(b):
        if not isinstance(axes, int) or axes != 0:
            raise ValueError(
                "One of the inputs is scalar, axes should be zero."
            )
        return _call_multiply(a, b)

    return dpnp_tensordot(a, b, axes=axes)


def vdot(a, b):
    """
    Return the dot product of two vectors.

    For full documentation refer to :obj:`numpy.dot`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray, scalar}
        First input array. Both inputs `a` and `b` can not be
        scalars at the same time. If `a` is complex, the complex
        conjugate is taken before the calculation of the dot product.
    b : {dpnp.ndarray, usm_ndarray, scalar}
        Second input array. Both inputs `a` and `b` can not be
        scalars at the same time.

    Returns
    -------
    out : dpnp.ndarray
        Returns the dot product of `a` and `b`.

    See Also
    --------
    :obj:`dpnp.dot` : Returns the dot product.
    :obj:`dpnp.matmul` : Returns the matrix product.
    :obj:`dpnp.vecdot` : Vector dot product of two arrays.
    :obj:`dpnp.linalg.vecdot` : Array API compatible version of
                    :obj:`dpnp.vecdot`.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1+2j,3+4j])
    >>> b = np.array([5+6j,7+8j])
    >>> np.vdot(a, b)
    array(70-8j)
    >>> np.vdot(b, a)
    array(70+8j)

    Note that higher-dimensional arrays are flattened!

    >>> a = np.array([[1, 4], [5, 6]])
    >>> b = np.array([[4, 1], [2, 2]])
    >>> np.vdot(a, b)
    array(30)
    >>> np.vdot(b, a)
    array(30)
    >>> 1*4 + 4*1 + 5*2 + 6*2
    30

    """

    dpnp.check_supported_arrays_type(a, b, scalar_type=True)

    if dpnp.isscalar(a):
        if b.size != 1:
            raise ValueError("The second array should be of size one.")
        a_conj = numpy.conj(a)
        return dpnp.squeeze(_call_multiply(a_conj, b))

    if dpnp.isscalar(b):
        if a.size != 1:
            raise ValueError("The first array should be of size one.")
        a_conj = dpnp.conj(a)
        return dpnp.squeeze(_call_multiply(a_conj, b))

    if a.ndim == 1 and b.ndim == 1:
        return dpnp_dot(a, b, out=None, conjugate=True)

    # dot product of flatten arrays
    return dpnp_dot(dpnp.ravel(a), dpnp.ravel(b), out=None, conjugate=True)


def vecdot(
    x1,
    x2,
    /,
    out=None,
    *,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
    signature=None,
    axes=None,
    axis=None,
):
    r"""
    Computes the vector dot product.

    Let :math:`\mathbf{a}` be a vector in `x1` and :math:`\mathbf{b}` be
    a corresponding vector in `x2`. The dot product is defined as:

    .. math::
       \mathbf{a} \cdot \mathbf{b} = \sum_{i=0}^{n-1} \overline{a_i}b_i

    where the sum is over the last dimension (unless `axis` is specified) and
    where :math:`\overline{a_i}` denotes the complex conjugate if :math:`a_i`
    is complex and the identity otherwise.

    For full documentation refer to :obj:`numpy.vecdot`.

    Parameters
    ----------
    x1 : {dpnp.ndarray, usm_ndarray}
        First input array.
    x2 : {dpnp.ndarray, usm_ndarray}
        Second input array.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        A location into which the result is stored. If provided, it must have
        the broadcasted shape of `x1` and `x2` with the last axis removed.
        If not provided or ``None``, a freshly-allocated array is used.

        Default: ``None``.
    casting : {"no", "equiv", "safe", "same_kind", "unsafe"}, optional
        Controls what kind of data casting may occur.

        Default: ``"same_kind"``.
    order : {None, "C", "F", "A", "K"}, optional
        Memory layout of the newly output array, if parameter `out` is ``None``.

        Default: ``"K"``.
    dtype : {None, str, dtype object}, optional
        Type to use in computing the vector dot product. By default, the
        returned array will have data type that is determined by considering
        Promotion Type Rule and device capabilities.

        Default: ``None``.
    axes : {None, list of tuples}, optional
        A list of tuples with indices of axes the dot product should operate
        on. For instance, for the signature of ``(i),(i)->()``, the base
        elements are vectors and these are taken to be stored in the last axes
        of each argument. The corresponding axes keyword would be
        ``[(-1,), (-1,), ()]``.

        Default: ``None``.
    axis : {None, int}, optional
        Axis over which to compute the dot product. This is a short-cut for
        passing in axes with entries of ``(axis,)`` for each
        single-core-dimension argument and ``()`` for all others. For instance,
        for a signature ``(i),(i)->()``, it is equivalent to passing in
        ``axes=[(axis,), (axis,), ()]``.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        The vector dot product of the inputs.
        This is a 0-d array only when both `x1`, `x2` are 1-d vectors.

    Limitations
    -----------
    Keyword arguments `subok`, and `signature` are only supported with their
    default values. Otherwise ``NotImplementedError`` exception will be raised.

    See Also
    --------
    :obj:`dpnp.linalg.vecdot` : Array API compatible version.
    :obj:`dpnp.vdot` : Complex-conjugating dot product but flattens
                       arguments first.
    :obj:`dpnp.matmul` : Matrix-matrix product.
    :obj:`dpnp.vecmat` : Vector-matrix product.
    :obj:`dpnp.matvec` : Matrix-vector product.
    :obj:`dpnp.einsum` : Einstein summation convention.

    Examples
    --------
    Get the projected size along a given normal for an array of vectors.

    >>> import dpnp as np
    >>> v = np.array([[0., 5., 0.], [0., 0., 10.], [0., 6., 8.]])
    >>> n = np.array([0., 0.6, 0.8])
    >>> np.vecdot(v, n)
    array([ 3.,  8., 10.])

    """

    dpnp.check_limitations(subok_linalg=subok, signature=signature)

    return dpnp_vecdot(
        x1,
        x2,
        out=out,
        casting=casting,
        order=order,
        dtype=dtype,
        axes=axes,
        axis=axis,
    )


def vecmat(
    x1,
    x2,
    /,
    out=None,
    *,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
    signature=None,
    axes=None,
    axis=None,
):
    r"""
    Vector-matrix dot product of two arrays.

    Given a vector (or stack of vector) :math:`\mathbf{v}` in `x1` and a matrix
    (or stack of matrices) :math:`\mathbf{A}` in `x2`, the vector-matrix product
    is defined as:

    .. math::
       \mathbf{b} \cdot \mathbf{A} = \sum_{i=0}^{n-1} \overline{v_i}A_{ij}

    where the sum is over the last dimension of `x1` and the one-but-last
    dimensions in `x2` (unless `axes` is specified) and where
    :math:`\overline{v_i}` denotes the complex conjugate if :math:`{v}` is
    complex and the identity otherwise. (For a non-conjugated
    vector-matrix product, use ``dpnp.matvec(x2.mT, x1)``.)

    For full documentation refer to :obj:`numpy.vecmat`.

    Parameters
    ----------
    x1 : {dpnp.ndarray, usm_ndarray}
        First input array.
    x2 : {dpnp.ndarray, usm_ndarray}
        Second input array.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        A location into which the result is stored. If provided, it must have
        the broadcasted shape of `x1` and `x2` with the summation axis removed.
        If not provided or ``None``, a freshly-allocated array is used.

        Default: ``None``.
    dtype : {None, str, dtype object}, optional
        Type to use in computing the matrix product. By default, the returned
        array will have data type that is determined by considering
        Promotion Type Rule and device capabilities.

        Default: ``None``.
    casting : {"no", "equiv", "safe", "same_kind", "unsafe"}, optional
        Controls what kind of data casting may occur.

        Default: ``"same_kind"``.
    order : {None, "C", "F", "A", "K"}, optional
        Memory layout of the newly output array, if parameter `out` is ``None``.

        Default: ``"K"``.
    axes : {None, list of tuples}, optional
        A list of tuples with indices of axes the vector-matrix product should
        operate on. For instance, for the signature of ``(i),(i,j)->(j)``, the
        base elements are 1D vectors and 2D matrices, where the vectors are
        assumed to be stored in the last axis of the first argument, and the
        matrices in the last two axes of the second argument. The corresponding
        axes keyword would be ``[(-1,), (-2, -1), (-1,)]``.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        The vector-matrix product of the inputs.

    Limitations
    -----------
    Keyword arguments `subok`, and `signature`, are only supported with their
    default values. Otherwise ``NotImplementedError`` exception will be raised.

    See Also
    --------
    :obj:`dpnp.vecdot` : Vector-vector product.
    :obj:`dpnp.matvec` : Matrix-vector product.
    :obj:`dpnp.matmul` : Matrix-matrix product.
    :obj:`dpnp.einsum` : Einstein summation convention.


    Examples
    --------
    Project a vector along X and Y.

    >>> import dpnp as np
    >>> v = np.array([0., 4., 2.])
    >>> a = np.array([[1., 0., 0.],
    ...               [0., 1., 0.],
    ...               [0., 0., 0.]])
    >>> np.vecmat(v, a)
    array([0., 4., 0.])

    """

    dpnp.check_limitations(subok_linalg=subok, signature=signature)
    dpnp.check_supported_arrays_type(x1, x2)

    # cannot directly use dpnp.conj(x1) as it returns `int8` for boolean input
    if dpnp.issubdtype(x1.dtype, dpnp.complexfloating):
        x1 = dpnp.conj(x1)

    return dpnp_multiplication(
        "vecmat",
        x1,
        x2,
        out=out,
        casting=casting,
        order=order,
        dtype=dtype,
        axes=axes,
        axis=axis,
    )
