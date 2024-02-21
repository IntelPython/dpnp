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
Interface of the Linear Algebra part of the DPNP

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""


import numpy
from numpy.core.numeric import normalize_axis_tuple

import dpnp

# pylint: disable=no-name-in-module
from .dpnp_algo import (
    dpnp_inner,
    dpnp_kron,
)
from .dpnp_utils import (
    call_origin,
)
from .dpnp_utils.dpnp_utils_linearalgebra import (
    dpnp_dot,
    dpnp_matmul,
)

__all__ = [
    "dot",
    "einsum",
    "einsum_path",
    "inner",
    "kron",
    "matmul",
    "outer",
    "tensordot",
    "vdot",
]


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
    out : {dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. It must have
        the same shape and data type as the expected output and should be
        C-contiguous. If these conditions are not met, an exception is
        raised, instead of attempting to be flexible.

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

    >>> a = np.arange(3*4*5*6).reshape((3,4,5,6))
    >>> b = np.arange(3*4*5*6)[::-1].reshape((5,4,6,3))
    >>> np.dot(a, b)[2,3,2,1,2,2]
    array(499128)
    >>> sum(a[2,3,2,:] * b[1,2,:,2])
    array(499128)

    """

    dpnp.check_supported_arrays_type(a, b, scalar_type=True)

    if out is not None:
        dpnp.check_supported_arrays_type(out)
        if not out.flags.c_contiguous:
            raise ValueError("Only C-contiguous array is acceptable.")

    if dpnp.isscalar(a) or dpnp.isscalar(b):
        # TODO: investigate usage of axpy (axpy_batch) or scal
        # functions from BLAS here instead of dpnp.multiply
        return dpnp.multiply(a, b, out=out)

    if a.ndim == 0 or b.ndim == 0:
        # TODO: investigate usage of axpy (axpy_batch) or scal
        # functions from BLAS here instead of dpnp.multiply
        return dpnp.multiply(a, b, out=out)

    if a.ndim == 1 and b.ndim == 1:
        return dpnp_dot(a, b, out=out)

    if a.ndim == 2 and b.ndim == 2:
        # NumPy does not allow casting even if it is safe
        return dpnp.matmul(a, b, out=out, casting="no")

    if a.ndim == 1 or b.ndim == 1:
        # NumPy does not allow casting even if it is safe
        return dpnp.matmul(a, b, out=out, casting="no")

    # TODO: investigate usage of matmul for some possible
    # use cases instead of dpnp.tensordot
    result = dpnp.tensordot(a, b, axes=(-1, -2))
    # NumPy does not allow casting even if it is safe
    return dpnp.get_result_array(result, out, casting="no")


def einsum(*args, **kwargs):
    """
    Evaluates the Einstein summation convention on the operands.

    For full documentation refer to :obj:`numpy.einsum`.

    Limitations
    -----------
    Function is executed sequentially on CPU.

    See Also
    -------
    :obj:`dpnp.einsum_path` : Evaluates the lowest cost contraction order
                              for an einsum expression.
    :obj:`dpnp.dot` : Returns the dot product of two arrays.
    :obj:`dpnp.inner` : Returns the inner product of two arrays.
    :obj:`dpnp.outer` : Returns the outer product of two arrays.

    """

    return call_origin(numpy.einsum, *args, **kwargs)


def einsum_path(*args, **kwargs):
    """
    einsum_path(subscripts, *operands, optimize='greedy')

    Evaluates the lowest cost contraction order for an einsum expression
    by considering the creation of intermediate arrays.

    For full documentation refer to :obj:`numpy.einsum_path`.

    Limitations
    -----------
    Function is executed sequentially on CPU.

    See Also
    --------
    :obj:`dpnp.einsum` : Evaluates the Einstein summation convention
                         on the operands.
    :obj:`dpnp.dot` : Returns the dot product of two arrays.
    :obj:`dpnp.inner` : Returns the inner product of two arrays.
    :obj:`dpnp.outer` : Returns the outer product of two arrays.

    """

    return call_origin(numpy.einsum_path, *args, **kwargs)


def inner(x1, x2, **kwargs):
    """
    Returns the inner product of two arrays.

    For full documentation refer to :obj:`numpy.inner`.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as :obj:`dpnp.ndarray`.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the functions will be executed sequentially on CPU.

    See Also
    --------
    :obj:`dpnp.einsum` : Evaluates the Einstein summation convention
                         on the operands.
    :obj:`dpnp.dot` : Returns the dot product of two arrays.
    :obj:`dpnp.tensordot` : Compute tensor dot product along specified axes.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1,2,3])
    >>> b = np.array([0, 1, 0])
    >>> result = np.inner(a, b)
    >>> [x for x in result]
    [2]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    x2_desc = dpnp.get_dpnp_descriptor(x2, copy_when_nondefault_queue=False)
    if x1_desc and x2_desc and not kwargs:
        return dpnp_inner(x1_desc, x2_desc).get_pyobj()

    return call_origin(numpy.inner, x1, x2, **kwargs)


def kron(x1, x2):
    """
    Returns the kronecker product of two arrays.

    For full documentation refer to :obj:`numpy.kron`.

    .. seealso:: :obj:`dpnp.outer` returns the outer product of two arrays.

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    x2_desc = dpnp.get_dpnp_descriptor(x2, copy_when_nondefault_queue=False)
    if x1_desc and x2_desc:
        return dpnp_kron(x1_desc, x2_desc).get_pyobj()

    return call_origin(numpy.kron, x1, x2)


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
    extobj=None,
    axes=None,
    axis=None,
):
    """
    Matrix product of two arrays.

    For full documentation refer to :obj:`numpy.matmul`.

    Parameters
    ----------
    x1 : {dpnp_array, usm_ndarray}
        First input array.
    x2 : {dpnp_array, usm_ndarray}
        Second input array.
    out : {dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. It must have
        a shape that matches the signature `(n,k),(k,m)->(n,m)` but the type
        (of the calculated values) will be cast if necessary. Default: ``None``.
    dtype : dtype, optional
        Type to use in computing the matrix product. By default, the returned
        array will have data type that is determined by considering
        Promotion Type Rule and device capabilities.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur. Default: ``"same_kind"``.
    order : {"C", "F", "A", "K", None}, optional
        Memory layout of the newly output array, if parameter `out` is ``None``.
        Default: "K".
    axes : list of tuples, optional
        A list of tuples with indices of axes the matrix product should operate
        on. For instance, for the signature of ``(i,j),(j,k)->(i,k)``, the base
        elements are 2d matrices and these are taken to be stored in the two
        last axes of each argument. The corresponding axes keyword would be
        [(-2, -1), (-2, -1), (-2, -1)].
        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        Returns the matrix product of the inputs.
        This is a 0-d array only when both `x1`, `x2` are 1-d vectors.

    Limitations
    -----------
    Keyword arguments `subok`, `signature`, `extobj`, and `axis` are
    only supported with their default value.
    Otherwise ``NotImplementedError`` exception will be raised.

    See Also
    --------
    :obj:`dpnp.vdot` : Complex-conjugating dot product.
    :obj:`dpnp.tensordot` : Sum products over arbitrary axes.
    :obj:`dpnp.einsum` : Einstein summation convention.
    :obj:`dpnp.dot` : Alternative matrix product with
                      different broadcasting rules.

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
    >>> np.matmul(a,b).shape
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

    The ``@`` operator can be used as a shorthand for ``matmul`` on
    :class:`dpnp.ndarray`.

    >>> x1 @ x2
    array(-13+0j)

    """

    dpnp.check_supported_arrays_type(x1, x2)
    if subok is False:
        raise NotImplementedError(
            "subok keyword argument is only supported by its default value."
        )
    if signature is not None:
        raise NotImplementedError(
            "signature keyword argument is only supported by its default value."
        )
    if extobj is not None:
        raise NotImplementedError(
            "extobj keyword argument is only supported by its default value."
        )
    if axis is not None:
        raise NotImplementedError(
            "axis keyword argument is only supported by its default value."
        )

    return dpnp_matmul(
        x1,
        x2,
        out=out,
        casting=casting,
        order=order,
        dtype=dtype,
        axes=axes,
    )


def outer(x1, x2, out=None):
    """
    Returns the outer product of two arrays.

    For full documentation refer to :obj:`numpy.outer`.

    Limitations
    -----------
        Parameters `x1` and `x2` are supported as either scalar,
        :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`, but both
        `x1` and `x2` can not be scalars at the same time. Otherwise
        the functions will be executed sequentially on CPU.
        Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.einsum` : Evaluates the Einstein summation convention
                         on the operands.
    :obj:`dpnp.inner` : Returns the inner product of two arrays.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 1, 1])
    >>> b = np.array([1, 2, 3])
    >>> result = np.outer(a, b)
    >>> [x for x in result]
    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]])

    """

    x1_is_scalar = dpnp.isscalar(x1)
    x2_is_scalar = dpnp.isscalar(x2)

    if x1_is_scalar and x2_is_scalar:
        pass
    elif not dpnp.is_supported_array_or_scalar(x1):
        pass
    elif not dpnp.is_supported_array_or_scalar(x2):
        pass
    else:
        x1_in = (
            x1
            if x1_is_scalar
            else (x1.reshape(-1) if x1.ndim > 1 else x1)[:, None]
        )
        x2_in = (
            x2
            if x2_is_scalar
            else (x2.reshape(-1) if x2.ndim > 1 else x2)[None, :]
        )
        return dpnp.multiply(x1_in, x2_in, out=out)

    return call_origin(numpy.outer, x1, x2, out=out)


def tensordot(a, b, axes=2):
    r"""
    Compute tensor dot product along specified axes.

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
        * integer_like
          If an int `N`, sum over the last `N` axes of `a` and the first `N`
          axes of `b` in order. The sizes of the corresponding axes must match.
        * (2,) array_like
          Or, a list of axes to be summed over, first sequence applying to `a`,
          second to `b`. Both elements array_like must be of the same length.

    Returns
    -------
    out : dpnp.ndarray
        Returns the tensordot product of `a` and `b`.

    See Also
    --------
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

    >>> a = np.arange(60.).reshape(3,4,5)
    >>> b = np.arange(24.).reshape(4,3,2)
    >>> c = np.tensordot(a,b, axes=([1,0],[0,1]))
    >>> c.shape
    (5, 2)
    >>> c
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])

    A slower but equivalent way of computing the same...

    >>> d = np.zeros((5,2))
    >>> for i in range(5):
    ...   for j in range(2):
    ...     for k in range(3):
    ...       for n in range(4):
    ...         d[i,j] += a[k,n,i] * b[n,k,j]
    >>> c == d
    array([[ True,  True],
           [ True,  True],
           [ True,  True],
           [ True,  True],
           [ True,  True]])

    """

    dpnp.check_supported_arrays_type(a, b, scalar_type=True)

    if dpnp.isscalar(a):
        a = dpnp.array(a, sycl_queue=b.sycl_queue, usm_type=b.usm_type)
    elif dpnp.isscalar(b):
        b = dpnp.array(b, sycl_queue=a.sycl_queue, usm_type=a.usm_type)

    try:
        iter(axes)
    except Exception as e:  # pylint: disable=broad-exception-caught
        if not isinstance(axes, int):
            raise TypeError("Axes must be an integer.") from e
        axes_a = tuple(range(-axes, 0))
        axes_b = tuple(range(0, axes))
    else:
        if len(axes) != 2:
            raise ValueError("Axes must consist of two sequences.")

        axes_a, axes_b = axes
        axes_a = (axes_a,) if dpnp.isscalar(axes_a) else axes_a
        axes_b = (axes_b,) if dpnp.isscalar(axes_b) else axes_b

        if len(axes_a) != len(axes_b):
            raise ValueError("Axes length mismatch.")

    a_shape = a.shape
    b_shape = b.shape
    for axis_a, axis_b in zip(axes_a, axes_b):
        if a_shape[axis_a] != b_shape[axis_b]:
            raise ValueError(
                "shape of input arrays is not similar at requested axes."
            )

    # Make the axes non-negative
    a_ndim = a.ndim
    b_ndim = b.ndim
    axes_a = normalize_axis_tuple(axes_a, a_ndim, "axis")
    axes_b = normalize_axis_tuple(axes_b, b_ndim, "axis")

    # Move the axes to sum over, to the end of "a"
    notin = tuple(k for k in range(a_ndim) if k not in axes_a)
    newaxes_a = notin + axes_a
    n1 = int(numpy.prod([a_shape[ax] for ax in notin]))
    n2 = int(numpy.prod([a_shape[ax] for ax in axes_a]))
    newshape_a = (n1, n2)
    olda = [a_shape[axis] for axis in notin]

    # Move the axes to sum over, to the front of "b"
    notin = tuple(k for k in range(b_ndim) if k not in axes_b)
    newaxes_b = tuple(axes_b + notin)
    n1 = int(numpy.prod([b_shape[ax] for ax in axes_b]))
    n2 = int(numpy.prod([b_shape[ax] for ax in notin]))
    newshape_b = (n1, n2)
    oldb = [b_shape[axis] for axis in notin]

    at = a.transpose(newaxes_a).reshape(newshape_a)
    bt = b.transpose(newaxes_b).reshape(newshape_b)
    res = dpnp.matmul(at, bt)

    return res.reshape(olda + oldb)


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

    if dpnp.isscalar(a) or dpnp.isscalar(b):
        if dpnp.isscalar(b) and a.size != 1:
            raise ValueError("The first array should be of size one.")
        if dpnp.isscalar(a) and b.size != 1:
            raise ValueError("The second array should be of size one.")
        a_conj = numpy.conj(a) if dpnp.isscalar(a) else dpnp.conj(a)
        # TODO: investigate usage of axpy (axpy_batch) or scal
        # functions from BLAS here instead of dpnp.multiply
        return dpnp.multiply(a_conj, b)

    if a.ndim == 1 and b.ndim == 1:
        return dpnp_dot(a, b, out=None, conjugate=True)

    # dot product of flatten arrays
    return dpnp_dot(dpnp.ravel(a), dpnp.ravel(b), out=None, conjugate=True)
