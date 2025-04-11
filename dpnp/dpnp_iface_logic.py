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
Interface of the Logic part of the DPNP

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""

# pylint: disable=protected-access
# pylint: disable=duplicate-code
# pylint: disable=no-name-in-module


import dpctl.tensor as dpt
import dpctl.tensor._tensor_elementwise_impl as ti
import numpy

import dpnp
from dpnp.dpnp_algo.dpnp_elementwise_common import DPNPBinaryFunc, DPNPUnaryFunc

from .dpnp_utils import get_usm_allocations

__all__ = [
    "all",
    "allclose",
    "any",
    "array_equal",
    "array_equiv",
    "equal",
    "greater",
    "greater_equal",
    "isclose",
    "iscomplex",
    "iscomplexobj",
    "isfinite",
    "isfortran",
    "isinf",
    "isnan",
    "isneginf",
    "isposinf",
    "isreal",
    "isrealobj",
    "isscalar",
    "less",
    "less_equal",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "not_equal",
]


def all(a, /, axis=None, out=None, keepdims=False, *, where=True):
    """
    Test whether all array elements along a given axis evaluate to ``True``.

    For full documentation refer to :obj:`numpy.all`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int, tuple of ints}, optional
        Axis or axes along which a logical AND reduction is performed.
        The default is to perform a logical AND over all the dimensions
        of the input array.`axis` may be negative, in which case it counts
        from the last to the first axis.

        Default: ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the returned
        values) will be cast if necessary.

        Default: ``None``.
    keepdims : {None, bool}, optional
        If ``True``, the reduced axes (dimensions) are included in the result
        as singleton dimensions, so that the returned array remains
        compatible with the input array according to Array Broadcasting
        rules. Otherwise, if ``False``, the reduced axes are not included in
        the returned array.

        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray of bool dtype
        An array containing the results of the logical AND reduction is
        returned unless `out` is specified. Otherwise, a reference to `out` is
        returned. The result has the same shape as `a` if `axis` is not ``None``
        or `a` is a 0-d array.

    Limitations
    -----------
    Parameters `where` is only supported with its default value.
    Otherwise ``NotImplementedError`` exception will be raised.

    See Also
    --------
    :obj:`dpnp.ndarray.all` : Equivalent method.
    :obj:`dpnp.any` : Test whether any element along a given axis evaluates
        to ``True``.

    Notes
    -----
    Not a Number (NaN), positive infinity and negative infinity
    evaluate to ``True`` because these are not equal to zero.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([[True, False], [True, True]])
    >>> np.all(x)
    array(False)

    >>> np.all(x, axis=0)
    array([ True, False])

    >>> x2 = np.array([-1, 4, 5])
    >>> np.all(x2)
    array(True)

    >>> x3 = np.array([1.0, np.nan])
    >>> np.all(x3)
    array(True)

    >>> o = np.array(False)
    >>> z = np.all(x2, out=o)
    >>> z, o
    (array(True), array(True))
    >>> # Check now that `z` is a reference to `o`
    >>> z is o
    True
    >>> id(z), id(o) # identity of `z` and `o`
    (139884456208480, 139884456208480) # may vary

    """

    dpnp.check_limitations(where=where)

    usm_a = dpnp.get_usm_ndarray(a)
    usm_res = dpt.all(usm_a, axis=axis, keepdims=keepdims)

    # TODO: temporary solution until dpt.all supports out parameter
    return dpnp.get_result_array(usm_res, out)


def allclose(a, b, rtol=1.0e-5, atol=1.0e-8, equal_nan=False):
    """
    Returns ``True`` if two arrays are element-wise equal within a tolerance.

    The tolerance values are positive, typically very small numbers. The
    relative difference (`rtol` * abs(`b`)) and the absolute difference `atol`
    are added together to compare against the absolute difference between `a`
    and `b`.

    ``NaNs`` are treated as equal if they are in the same place and if
    ``equal_nan=True``. ``Infs`` are treated as equal if they are in the same
    place and of the same sign in both arrays.

    For full documentation refer to :obj:`numpy.allclose`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray, scalar}
        First input array, expected to have a numeric data type.
    b : {dpnp.ndarray, usm_ndarray, scalar}
        Second input array, also expected to have a numeric data type.
    rtol : {dpnp.ndarray, usm_ndarray, scalar}, optional
        The relative tolerance parameter.

        Default: ``1e-05``.
    atol : {dpnp.ndarray, usm_ndarray, scalar}, optional
        The absolute tolerance parameter.

        Default: ``1e-08``.
    equal_nan : bool
        Whether to compare ``NaNs`` as equal. If ``True``, ``NaNs`` in `a` will
        be considered equal to ``NaNs`` in `b` in the output array.

        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray of bool dtype
        A 0-d array with ``True`` value if the two arrays are equal within the
        given tolerance; with ``False`` otherwise.

    See Also
    --------
    :obj:`dpnp.isclose` : Test whether two arrays are element-wise equal.
    :obj:`dpnp.all` : Test whether all elements evaluate to True.
    :obj:`dpnp.any` : Test whether any element evaluates to True.
    :obj:`dpnp.equal` : Return (x1 == x2) element-wise.

    Notes
    -----
    At least one of `x1` or `x2` must be an array.

    The comparison of `a` and `b` uses standard broadcasting, which means that
    `a` and `b` need not have the same shape in order for
    ``dpnp.allclose(a, b)`` to evaluate to ``True``.
    The same is true for :obj:`dpnp.equal` but not :obj:`dpnp.array_equal`.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1e10, 1e-7])
    >>> b = np.array([1.00001e10, 1e-8])
    >>> np.allclose(a, b)
    array(False)

    >>> a = np.array([1.0, np.nan])
    >>> b = np.array([1.0, np.nan])
    >>> np.allclose(a, b)
    array(False)
    >>> np.allclose(a, b, equal_nan=True)
    array(True)

    >>> a = np.array([1.0, np.inf])
    >>> b = np.array([1.0, np.inf])
    >>> np.allclose(a, b)
    array(True)

    """

    return all(isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))


def any(a, /, axis=None, out=None, keepdims=False, *, where=True):
    """
    Test whether any array element along a given axis evaluates to ``True``.

    For full documentation refer to :obj:`numpy.any`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int, tuple of ints}, optional
        Axis or axes along which a logical OR reduction is performed.
        The default is to perform a logical OR over all the dimensions
        of the input array.`axis` may be negative, in which case it counts
        from the last to the first axis.

        Default: ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the returned
        values) will be cast if necessary.

        Default: ``None``.
    keepdims : {None, bool}, optional
        If ``True``, the reduced axes (dimensions) are included in the result
        as singleton dimensions, so that the returned array remains
        compatible with the input array according to Array Broadcasting
        rules. Otherwise, if ``False``, the reduced axes are not included in
        the returned array.

        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray of bool dtype
        An array containing the results of the logical OR reduction is returned
        unless `out` is specified. Otherwise, a reference to `out` is returned.
        The result has the same shape as `a` if `axis` is not ``None`` or `a`
        is a 0-d array.

    Limitations
    -----------
    Parameters `where` is only supported with its default value.
    Otherwise ``NotImplementedError`` exception will be raised.

    See Also
    --------
    :obj:`dpnp.ndarray.any` : Equivalent method.
    :obj:`dpnp.all` : Test whether all elements along a given axis evaluate
        to ``True``.

    Notes
    -----
    Not a Number (NaN), positive infinity and negative infinity evaluate
    to ``True`` because these are not equal to zero.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([[True, False], [True, True]])
    >>> np.any(x)
    array(True)

    >>> np.any(x, axis=0)
    array([ True,  True])

    >>> x2 = np.array([-1, 0, 5])
    >>> np.any(x2)
    array(True)

    >>> x3 = np.array([1.0, np.nan])
    >>> np.any(x3)
    array(True)

    >>> o = np.array(False)
    >>> z = np.any(x2, out=o)
    >>> z, o
    (array(True), array(True))
    >>> # Check now that `z` is a reference to `o`
    >>> z is o
    True
    >>> id(z), id(o) # identity of `z` and `o`
    >>> (140053638309840, 140053638309840) # may vary

    """

    dpnp.check_limitations(where=where)

    usm_a = dpnp.get_usm_ndarray(a)
    usm_res = dpt.any(usm_a, axis=axis, keepdims=keepdims)

    # TODO: temporary solution until dpt.any supports out parameter
    return dpnp.get_result_array(usm_res, out)


def array_equal(a1, a2, equal_nan=False):
    """
    ``True`` if two arrays have the same shape and elements, ``False``
    otherwise.

    For full documentation refer to :obj:`numpy.array_equal`.

    Parameters
    ----------
    a1 : {dpnp.ndarray, usm_ndarray, scalar}
        First input array.
    a2 : {dpnp.ndarray, usm_ndarray, scalar}
        Second input array.
    equal_nan : bool, optional
        Whether to compare ``NaNs`` as equal. If the dtype of `a1` and `a2` is
        complex, values will be considered equal if either the real or the
        imaginary component of a given value is ``NaN``.

        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray of bool dtype
        A 0-d array with ``True`` value if the arrays are equal.

    See Also
    --------
    :obj:`dpnp.allclose`: Returns ``True`` if two arrays are element-wise equal
                          within a tolerance.
    :obj:`dpnp.array_equiv`: Returns ``True`` if input arrays are shape
                             consistent and all elements equal.

    Notes
    -----
    At least one of `x1` or `x2` must be an array.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 2])
    >>> b = np.array([1, 2])
    >>> np.array_equal(a, b)
    array(True)

    >>> b = np.array([1, 2, 3])
    >>> np.array_equal(a, b)
    array(False)

    >>> b = np.array([1, 4])
    >>> np.array_equal(a, b)
    array(False)

    >>> a = np.array([1, np.nan])
    >>> np.array_equal(a, a)
    array(False)

    >>> np.array_equal(a, a, equal_nan=True)
    array(True)

    When ``equal_nan`` is ``True``, complex values with NaN components are
    considered equal if either the real *or* the imaginary components are
    ``NaNs``.

    >>> a = np.array([1 + 1j])
    >>> b = a.copy()
    >>> a.real = np.nan
    >>> b.imag = np.nan
    >>> np.array_equal(a, b, equal_nan=True)
    array(True)

    """

    dpnp.check_supported_arrays_type(a1, a2, scalar_type=True)
    if dpnp.isscalar(a1):
        usm_type_alloc = a2.usm_type
        sycl_queue_alloc = a2.sycl_queue
        a1 = dpnp.array(
            a1,
            dtype=dpnp.result_type(a1, a2),
            usm_type=usm_type_alloc,
            sycl_queue=sycl_queue_alloc,
        )
    elif dpnp.isscalar(a2):
        usm_type_alloc = a1.usm_type
        sycl_queue_alloc = a1.sycl_queue
        a2 = dpnp.array(
            a2,
            dtype=dpnp.result_type(a1, a2),
            usm_type=usm_type_alloc,
            sycl_queue=sycl_queue_alloc,
        )
    else:
        usm_type_alloc, sycl_queue_alloc = get_usm_allocations([a1, a2])

    if a1.shape != a2.shape:
        return dpnp.array(
            False, usm_type=usm_type_alloc, sycl_queue=sycl_queue_alloc
        )

    if not equal_nan:
        return (a1 == a2).all()

    if a1 is a2:
        # NaN will compare equal so an array will compare equal to itself
        return dpnp.array(
            True, usm_type=usm_type_alloc, sycl_queue=sycl_queue_alloc
        )

    if not (
        dpnp.issubdtype(a1, dpnp.inexact) or dpnp.issubdtype(a2, dpnp.inexact)
    ):
        return (a1 == a2).all()

    # Handling NaN values if equal_nan is True
    a1nan, a2nan = isnan(a1), isnan(a2)
    # NaNs occur at different locations
    if not (a1nan == a2nan).all():
        return dpnp.array(
            False, usm_type=usm_type_alloc, sycl_queue=sycl_queue_alloc
        )
    # Shapes of a1, a2 and masks are guaranteed to be consistent by this point
    return (a1[~a1nan] == a2[~a1nan]).all()


def array_equiv(a1, a2):
    """
    Returns ``True`` if input arrays are shape consistent and all elements
    equal.

    Shape consistent means they are either the same shape, or one input array
    can be broadcasted to create the same shape as the other one.

    For full documentation refer to :obj:`numpy.array_equiv`.

    Parameters
    ----------
    a1 : {dpnp.ndarray, usm_ndarray, scalar}
        First input array.
    a2 : {dpnp.ndarray, usm_ndarray, scalar}
        Second input array.

    Returns
    -------
    out : dpnp.ndarray of bool dtype
        A 0-d array with ``True`` value if the arrays are equivalent, ``False``
        otherwise.

    Notes
    -----
    At least one of `x1` or `x2` must be an array.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 2])
    >>> b = np.array([1, 2])
    >>> c = np.array([1, 3])
    >>> np.array_equiv(a, b)
    array(True)
    >>> np.array_equiv(a, c)
    array(False)

    Showing the shape equivalence:

    >>> b = np.array([[1, 2], [1, 2]])
    >>> c = np.array([[1, 2, 1, 2], [1, 2, 1, 2]])
    >>> np.array_equiv(a, b)
    array(True)
    >>> np.array_equiv(a, c)
    array(False)

    >>> b = np.array([[1, 2], [1, 3]])
    >>> np.array_equiv(a, b)
    array(False)

    """

    dpnp.check_supported_arrays_type(a1, a2, scalar_type=True)
    if not dpnp.isscalar(a1) and not dpnp.isscalar(a2):
        usm_type_alloc, sycl_queue_alloc = get_usm_allocations([a1, a2])
        try:
            dpnp.broadcast_arrays(a1, a2)
        except ValueError:
            return dpnp.array(
                False, usm_type=usm_type_alloc, sycl_queue=sycl_queue_alloc
            )
    return (a1 == a2).all()


_EQUAL_DOCSTRING = """
Calculates equality test results for each element :math:`x1_i` of the input
array `x1` with the respective element :math:`x2_i` of the input array `x2`.

For full documentation refer to :obj:`numpy.equal`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, may have any data type.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also may have any data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray of bool dtype
    An array containing the result of element-wise equality comparison.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.not_equal` : Return (x1 != x2) element-wise.
:obj:`dpnp.greater_equal` : Return the truth value of (x1 >= x2) element-wise.
:obj:`dpnp.less_equal` : Return the truth value of (x1 =< x2) element-wise.
:obj:`dpnp.greater` : Return the truth value of (x1 > x2) element-wise.
:obj:`dpnp.less` : Return the truth value of (x1 < x2) element-wise.

Notes
-----
At least one of `x1` or `x2` must be an array.

If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
(which becomes the shape of the output).

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([0, 1, 3])
>>> x2 = np.arange(3)
>>> np.equal(x1, x2)
array([ True,  True, False])

What is compared are values, not types. So an int (1) and an array of
length one can evaluate as True:

>>> np.equal(1, np.ones(1))
array([ True])

The ``==`` operator can be used as a shorthand for ``equal`` on
:class:`dpnp.ndarray`.

>>> a = np.array([2, 4, 6])
>>> b = np.array([2, 4, 2])
>>> a == b
array([ True,  True, False])

"""

equal = DPNPBinaryFunc(
    "equal",
    ti._equal_result_type,
    ti._equal,
    _EQUAL_DOCSTRING,
)


_GREATER_DOCSTRING = """
Computes the greater-than test results for each element :math:`x1_i` of the
input array `x1` with the respective element :math:`x2_i` of the input array
`x2`.

For full documentation refer to :obj:`numpy.greater`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, may have any data type.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also may have any data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray of bool dtype
    An array containing the result of element-wise greater-than comparison.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.greater_equal` : Return the truth value of (x1 >= x2) element-wise.
:obj:`dpnp.less` : Return the truth value of (x1 < x2) element-wise.
:obj:`dpnp.less_equal` : Return the truth value of (x1 =< x2) element-wise.
:obj:`dpnp.equal` : Return (x1 == x2) element-wise.
:obj:`dpnp.not_equal` : Return (x1 != x2) element-wise.

Notes
-----
At least one of `x1` or `x2` must be an array.

If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
(which becomes the shape of the output).

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([4, 2])
>>> x2 = np.array([2, 2])
>>> np.greater(x1, x2)
array([ True, False])

The ``>`` operator can be used as a shorthand for ``greater`` on
:class:`dpnp.ndarray`.

>>> a = np.array([4, 2])
>>> b = np.array([2, 2])
>>> a > b
array([ True, False])

"""

greater = DPNPBinaryFunc(
    "greater",
    ti._greater_result_type,
    ti._greater,
    _GREATER_DOCSTRING,
)


_GREATER_EQUAL_DOCSTRING = """
Computes the greater-than or equal-to test results for each element :math:`x1_i`
of the input array `x1` with the respective element :math:`x2_i` of the input
array `x2`.

For full documentation refer to :obj:`numpy.greater_equal`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, may have any data type.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also may have any data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray of bool dtype
    An array containing the result of element-wise greater-than or equal-to
    comparison.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.greater` : Return the truth value of (x1 > x2) element-wise.
:obj:`dpnp.less` : Return the truth value of (x1 < x2) element-wise.
:obj:`dpnp.less_equal` : Return the truth value of (x1 =< x2) element-wise.
:obj:`dpnp.equal` : Return (x1 == x2) element-wise.
:obj:`dpnp.not_equal` : Return (x1 != x2) element-wise.

Notes
-----
At least one of `x1` or `x2` must be an array.

If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
(which becomes the shape of the output).

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([4, 2, 1])
>>> x2 = np.array([2, 2, 2])
>>> np.greater_equal(x1, x2)
array([ True,  True, False])

The ``>=`` operator can be used as a shorthand for ``greater_equal`` on
:class:`dpnp.ndarray`.

>>> a = np.array([4, 2, 1])
>>> b = np.array([2, 2, 2])
>>> a >= b
array([ True,  True, False])

"""

greater_equal = DPNPBinaryFunc(
    "greater_equal",
    ti._greater_equal_result_type,
    ti._greater_equal,
    _GREATER_EQUAL_DOCSTRING,
)


def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """
    Returns a boolean array where two arrays are element-wise equal within
    a tolerance.

    The tolerance values are positive, typically very small numbers. The
    relative difference (`rtol` * abs(`b`)) and the absolute difference `atol`
    are added together to compare against the absolute difference between `a`
    and `b`.

    ``NaNs`` are treated as equal if they are in the same place and if
    ``equal_nan=True``. ``Infs`` are treated as equal if they are in the same
    place and of the same sign in both arrays.

    For full documentation refer to :obj:`numpy.isclose`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray, scalar}
        First input array, expected to have a numeric data type.
    b : {dpnp.ndarray, usm_ndarray, scalar}
        Second input array, also expected to have a numeric data type.
    rtol : {dpnp.ndarray, usm_ndarray, scalar}, optional
        The relative tolerance parameter.

        Default: ``1e-05``.
    atol : {dpnp.ndarray, usm_ndarray, scalar}, optional
        The absolute tolerance parameter.

        Default: ``1e-08``.
    equal_nan : bool
        Whether to compare ``NaNs`` as equal. If ``True``, ``NaNs`` in `a` will
        be considered equal to ``NaNs`` in `b` in the output array.

        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray of bool dtype
        Returns a boolean array of where `a` and `b` are equal within the given
        tolerance.

    See Also
    --------
    :obj:`dpnp.allclose` : Returns ``True`` if two arrays are element-wise
                           equal within a tolerance.

    Notes
    -----
    At least one of `x1` or `x2` must be an array.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1e10, 1e-7])
    >>> b = np.array([1.00001e10, 1e-8])
    >>> np.isclose(a, b)
    array([ True, False])

    >>> a = np.array([1e10, 1e-8])
    >>> b = np.array([1.00001e10, 1e-9])
    >>> np.isclose(a, b)
    array([ True,  True])

    >>> a = np.array([1e10, 1e-8])
    >>> b = np.array([1.0001e10, 1e-9])
    >>> np.isclose(a, b)
    array([False,  True])

    >>> a = np.array([1.0, np.nan])
    >>> b = np.array([1.0, np.nan])
    >>> np.isclose(a, b)
    array([ True, False])
    >>> np.isclose(a, b, equal_nan=True)
    array([ True,  True])

    >>> a = np.array([0.0, 0.0])
    >>> b = np.array([1e-8, 1e-7])
    >>> np.isclose(a, b)
    array([ True, False])
    >>> b = np.array([1e-100, 1e-7])
    >>> np.isclose(a, b, atol=0.0)
    array([False, False])

    >>> a = np.array([1e-10, 1e-10])
    >>> b = np.array([1e-20, 0.0])
    >>> np.isclose(a, b)
    array([ True,  True])
    >>> b = np.array([1e-20, 0.999999e-10])
    >>> np.isclose(a, b, atol=0.0)
    array([False,  True])

    """

    dpnp.check_supported_arrays_type(a, b, scalar_type=True)
    dpnp.check_supported_arrays_type(
        rtol, atol, scalar_type=True, all_scalars=True
    )

    # make sure b is an inexact type to avoid bad behavior on abs(MIN_INT)
    if dpnp.isscalar(b):
        dt = dpnp.result_type(a, b, 1.0, rtol, atol)
        b = dpnp.asarray(
            b, dtype=dt, sycl_queue=a.sycl_queue, usm_type=a.usm_type
        )
    elif dpnp.issubdtype(b, dpnp.integer):
        dt = dpnp.result_type(b, 1.0, rtol, atol)
        b = dpnp.astype(b, dt)

    # Firstly handle finite values:
    # result = absolute(a - b) <= atol + rtol * absolute(b)
    dt = dpnp.result_type(b, rtol, atol)
    _b = dpnp.abs(b, dtype=dt)
    _b *= rtol
    _b += atol
    result = less_equal(dpnp.abs(a - b), _b)

    # Handle "inf" values: they are treated as equal if they are in the same
    # place and of the same sign in both arrays
    result &= isfinite(b)
    result |= a == b

    if equal_nan:
        result |= isnan(a) & isnan(b)
    return result


def iscomplex(x):
    """
    Returns a bool array, where ``True`` if input element is complex.

    What is tested is whether the input has a non-zero imaginary part, not if
    the input type is complex.

    For full documentation refer to :obj:`numpy.iscomplex`.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        Input array.

    Returns
    -------
    out : dpnp.ndarray of bool dtype
        Output array.

    See Also
    --------
    :obj:`dpnp.isreal` : Returns a bool array, where ``True`` if input element
                         is real.
    :obj:`dpnp.iscomplexobj` : Return ``True`` if `x` is a complex type or an
                               array of complex numbers.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1+1j, 1+0j, 4.5, 3, 2, 2j])
    >>> np.iscomplex(a)
    array([ True, False, False, False, False,  True])

    """

    dpnp.check_supported_arrays_type(x)
    if dpnp.issubdtype(x.dtype, dpnp.complexfloating):
        return x.imag != 0
    return dpnp.zeros_like(x, dtype=dpnp.bool)


def iscomplexobj(x):
    """
    Check for a complex type or an array of complex numbers.

    The type of the input is checked, not the value. Even if the input has an
    imaginary part equal to zero, :obj:`dpnp.iscomplexobj` evaluates to
    ``True``.

    For full documentation refer to :obj:`numpy.iscomplexobj`.

    Parameters
    ----------
    x : array_like
        Input data, in any form that can be converted to an array. This
        includes scalars, lists, lists of tuples, tuples, tuples of tuples,
        tuples of lists, and ndarrays.

    Returns
    -------
    out : bool
        The return value, ``True`` if `x` is of a complex type or has at least
        one complex element.

    See Also
    --------
    :obj:`dpnp.isrealobj` : Return ``True`` if `x` is a not complex type or an
                            array of complex numbers.
    :obj:`dpnp.iscomplex` : Returns a bool array, where ``True`` if input
                            element is complex.

    Examples
    --------
    >>> import dpnp as np
    >>> np.iscomplexobj(1)
    False
    >>> np.iscomplexobj(1+0j)
    True
    >>> np.iscomplexobj([3, 1+0j, True])
    True

    """

    return numpy.iscomplexobj(x)


_ISFINITE_DOCSTRING = """
Test each element :math:`x_i` of the input array `x` to determine if finite.

For full documentation refer to :obj:`numpy.isfinite`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, may have any data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray of bool dtype
    An array which is True where `x` is not positive infinity,
    negative infinity, or ``NaN``, False otherwise.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.isinf` : Test element-wise for positive or negative infinity.
:obj:`dpnp.isneginf` : Test element-wise for negative infinity,
                        return result as bool array.
:obj:`dpnp.isposinf` : Test element-wise for positive infinity,
                        return result as bool array.
:obj:`dpnp.isnan` : Test element-wise for NaN and
                    return result as a boolean array.

Notes
-----
Not a Number, positive infinity and negative infinity are considered
to be non-finite.

Examples
--------
>>> import dpnp as np
>>> x = np.array([-np.inf, 0., np.inf])
>>> np.isfinite(x)
array([False,  True, False])

"""

isfinite = DPNPUnaryFunc(
    "isfinite",
    ti._isfinite_result_type,
    ti._isfinite,
    _ISFINITE_DOCSTRING,
)


def isfortran(a):
    """
    Check if the array is Fortran contiguous but *not* C contiguous.

    This function is obsolete. If you only want to check if an array is Fortran
    contiguous use ``a.flags.f_contiguous`` instead.

    For full documentation refer to :obj:`numpy.isfortran`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.

    Returns
    -------
    isfortran : bool
        Returns ``True`` if the array is Fortran contiguous
        but *not* C contiguous.

    Examples
    --------
    :obj:`dpnp.array` allows to specify whether the array is written in
    C-contiguous order (last index varies the fastest), or FORTRAN-contiguous
    order in memory (first index varies the fastest).

    >>> import dpnp as np
    >>> a = np.array([[1, 2, 3], [4, 5, 6]], order='C')
    >>> a
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> np.isfortran(a)
    False

    >>> b = np.array([[1, 2, 3], [4, 5, 6]], order='F')
    >>> b
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> np.isfortran(b)
    True

    The transpose of a C-ordered array is a FORTRAN-ordered array.

    >>> a = np.array([[1, 2, 3], [4, 5, 6]], order='C')
    >>> a
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> np.isfortran(a)
    False
    >>> b = a.T
    >>> b
    array([[1, 4],
           [2, 5],
           [3, 6]])
    >>> np.isfortran(b)
    True

    C-ordered arrays evaluate as ``False`` even if they are also
    FORTRAN-ordered.

    >>> np.isfortran(np.array([1, 2], order='F'))
    False

    """

    dpnp.check_supported_arrays_type(a)
    return a.flags.fnc


_ISINF_DOCSTRING = """
Tests each element :math:`x_i` of the input array `x` to determine if equal to
positive or negative infinity.

For full documentation refer to :obj:`numpy.isinf`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, may have any data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray of bool dtype
    An array which is True where `x` is positive or negative infinity,
    False otherwise.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.isneginf` : Test element-wise for negative infinity,
                        return result as bool array.
:obj:`dpnp.isposinf` : Test element-wise for positive infinity,
                        return result as bool array.
:obj:`dpnp.isnan` : Test element-wise for NaN and
                    return result as a boolean array.
:obj:`dpnp.isfinite` : Test element-wise for finiteness.

Examples
--------
>>> import dpnp as np
>>> x = np.array([-np.inf, 0., np.inf])
>>> np.isinf(x)
array([ True, False,  True])

"""

isinf = DPNPUnaryFunc(
    "isinf",
    ti._isinf_result_type,
    ti._isinf,
    _ISINF_DOCSTRING,
)


_ISNAN_DOCSTRING = """
Tests each element :math:`x_i` of the input array `x` to determine whether the
element is ``NaN``.

For full documentation refer to :obj:`numpy.isnan`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, may have any data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray of bool dtype
    An array which is True where `x` is ``NaN``, False otherwise.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.isinf` : Test element-wise for positive or negative infinity.
:obj:`dpnp.isneginf` : Test element-wise for negative infinity,
                        return result as bool array.
:obj:`dpnp.isposinf` : Test element-wise for positive infinity,
                        return result as bool array.
:obj:`dpnp.isfinite` : Test element-wise for finiteness.

Examples
--------
>>> import dpnp as np
>>> x = np.array([np.inf, 0., np.nan])
>>> np.isnan(x)
array([False, False,  True])

"""

isnan = DPNPUnaryFunc(
    "isnan",
    ti._isnan_result_type,
    ti._isnan,
    _ISNAN_DOCSTRING,
)


def isneginf(x, out=None):
    """
    Test element-wise for negative infinity, return result as bool array.

    For full documentation refer to :obj:`numpy.isneginf`.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        Input array.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        A location into which the result is stored. If provided, it must have a
        shape that the input broadcasts to and a boolean data type.
        If not provided or ``None``, a freshly-allocated boolean array
        is returned.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray of bool dtype
        An array with the same shape as `x`.
        If `out` is ``None`` then an array is returned with values ``True``
        where the corresponding element of the input is negative infinity
        and values ``False`` where the element of the input is not negative
        infinity.
        If `out` is not ``None`` then the result is stored there and `out` is
        a reference to that array.

    See Also
    --------
    :obj:`dpnp.isinf` : Test element-wise for positive or negative infinity.
    :obj:`dpnp.isposinf` : Test element-wise for positive infinity,
                            return result as bool array.
    :obj:`dpnp.isnan` : Test element-wise for NaN and
                    return result as a boolean array.
    :obj:`dpnp.isfinite` : Test element-wise for finiteness.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array(np.inf)
    >>> np.isneginf(-x)
    array(True)
    >>> np.isneginf(x)
    array(False)

    >>> x = np.array([-np.inf, 0., np.inf])
    >>> np.isneginf(x)
    array([ True, False, False])

    >>> x = np.array([-np.inf, 0., np.inf])
    >>> y = np.zeros(x.shape, dtype='bool')
    >>> np.isneginf(x, y)
    array([ True, False, False])
    >>> y
    array([ True, False, False])

    """

    dpnp.check_supported_arrays_type(x)

    if out is not None:
        dpnp.check_supported_arrays_type(out)

    x_dtype = x.dtype
    if dpnp.issubdtype(x_dtype, dpnp.complexfloating):
        raise TypeError(
            f"This operation is not supported for {x_dtype} values "
            "because it would be ambiguous."
        )

    is_inf = dpnp.isinf(x)
    signbit = dpnp.signbit(x)

    # TODO: support different out dtype #1717(dpctl)
    return dpnp.logical_and(is_inf, signbit, out=out)


def isposinf(x, out=None):
    """
    Test element-wise for positive infinity, return result as bool array.

    For full documentation refer to :obj:`numpy.isposinf`.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        Input array.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        A location into which the result is stored. If provided, it must have a
        shape that the input broadcasts to and a boolean data type.
        If not provided or ``None``, a freshly-allocated boolean array
        is returned.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray of bool dtype
        An array with the same shape as `x`.
        If `out` is ``None`` then an array is returned with values ``True``
        where the corresponding element of the input is positive infinity
        and values ``False`` where the element of the input is not positive
        infinity.
        If `out` is not ``None`` then the result is stored there and `out` is
        a reference to that array.

    See Also
    --------
    :obj:`dpnp.isinf` : Test element-wise for positive or negative infinity.
    :obj:`dpnp.isneginf` : Test element-wise for negative infinity,
                            return result as bool array.
    :obj:`dpnp.isnan` : Test element-wise for NaN and
                    return result as a boolean array.
    :obj:`dpnp.isfinite` : Test element-wise for finiteness.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array(np.inf)
    >>> np.isposinf(x)
    array(True)
    >>> np.isposinf(-x)
    array(False)

    >>> x = np.array([-np.inf, 0., np.inf])
    >>> np.isposinf(x)
    array([False, False,  True])

    >>> x = np.array([-np.inf, 0., np.inf])
    >>> y = np.zeros(x.shape, dtype='bool')
    >>> np.isposinf(x, y)
    array([False, False,  True])
    >>> y
    array([False, False,  True])

    """

    dpnp.check_supported_arrays_type(x)

    if out is not None:
        dpnp.check_supported_arrays_type(out)

    x_dtype = x.dtype
    if dpnp.issubdtype(x_dtype, dpnp.complexfloating):
        raise TypeError(
            f"This operation is not supported for {x_dtype} values "
            "because it would be ambiguous."
        )

    is_inf = dpnp.isinf(x)
    signbit = ~dpnp.signbit(x)

    # TODO: support different out dtype #1717(dpctl)
    return dpnp.logical_and(is_inf, signbit, out=out)


def isreal(x):
    """
    Returns a bool array, where ``True`` if input element is real.

    If element has complex type with zero imaginary part, the return value
    for that element is ``True``.

    For full documentation refer to :obj:`numpy.isreal`.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        Input array.

    Returns
    -------
    out : dpnp.ndarray of bool dtype
        An array of same shape as `x`.

    See Also
    --------
    :obj:`dpnp.iscomplex` : Returns a bool array, where ``True`` if input
                            element is complex.
    :obj:`dpnp.isrealobj` : Return ``True`` if `x` is not a complex type.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1+1j, 1+0j, 4.5, 3, 2, 2j])
    >>> np.isreal(a)
    array([False,  True,  True,  True,  True, False])

    """

    dpnp.check_supported_arrays_type(x)
    if dpnp.issubdtype(x.dtype, dpnp.complexfloating):
        return x.imag == 0
    return dpnp.ones_like(x, dtype=dpnp.bool)


def isrealobj(x):
    """
    Return ``True`` if `x` is a not complex type or an array of complex numbers.

    The type of the input is checked, not the value. So even if the input has
    an imaginary part equal to zero, :obj:`dpnp.isrealobj` evaluates to
    ``False`` if the data type is complex.

    For full documentation refer to :obj:`numpy.isrealobj`.

    Parameters
    ----------
    x : array_like
        Input data, in any form that can be converted to an array. This
        includes scalars, lists, lists of tuples, tuples, tuples of tuples,
        tuples of lists, and ndarrays.

    Returns
    -------
    out : bool
        The return value, ``False`` if `x` is of a complex type.

    See Also
    --------
    :obj:`dpnp.iscomplexobj` : Check for a complex type or an array of complex
                               numbers.
    :obj:`dpnp.isreal` : Returns a bool array, where ``True`` if input element
                         is real.

    Examples
    --------
    >>> import dpnp as np
    >>> np.isrealobj(False)
    True
    >>> np.isrealobj(1)
    True
    >>> np.isrealobj(1+0j)
    False
    >>> np.isrealobj([3, 1+0j, True])
    False

    """

    return not iscomplexobj(x)


def isscalar(element):
    """
    Returns ``True`` if the type of `element` is a scalar type.

    For full documentation refer to :obj:`numpy.isscalar`.

    Parameters
    ----------
    element : any
        Input argument, can be of any type and shape.

    Returns
    -------
    out : bool
        ``True`` if `element` is a scalar type, ``False`` if it is not.

    Examples
    --------
    >>> import dpnp as np
    >>> np.isscalar(3.1)
    True
    >>> np.isscalar(np.array(3.1))
    False
    >>> np.isscalar([3.1])
    False
    >>> np.isscalar(False)
    True
    >>> np.isscalar("dpnp")
    True

    """

    return numpy.isscalar(element)


_LESS_DOCSTRING = """
Computes the less-than test results for each element :math:`x1_i` of the input
array `x1` with the respective element :math:`x2_i` of the input array `x2`.

For full documentation refer to :obj:`numpy.less`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, may have any data type.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also may have any data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray of bool dtype
    An array containing the result of element-wise less-than comparison.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.greater` : Return the truth value of (x1 > x2) element-wise.
:obj:`dpnp.less_equal` : Return the truth value of (x1 =< x2) element-wise.
:obj:`dpnp.greater_equal` : Return the truth value of (x1 >= x2) element-wise.
:obj:`dpnp.equal` : Return (x1 == x2) element-wise.
:obj:`dpnp.not_equal` : Return (x1 != x2) element-wise.

Notes
-----
At least one of `x1` or `x2` must be an array.

If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
(which becomes the shape of the output).

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([1, 2])
>>> x2 = np.array([2, 2])
>>> np.less(x1, x2)
array([ True, False])

The ``<`` operator can be used as a shorthand for ``less`` on
:class:`dpnp.ndarray`.

>>> a = np.array([1, 2])
>>> b = np.array([2, 2])
>>> a < b
array([ True, False])

"""

less = DPNPBinaryFunc(
    "less",
    ti._less_result_type,
    ti._less,
    _LESS_DOCSTRING,
)


_LESS_EQUAL_DOCSTRING = """
Computes the less-than or equal-to test results for each element :math:`x1_i`
of the input array `x1` with the respective element :math:`x2_i` of the input
array `x2`.

For full documentation refer to :obj:`numpy.less_equal`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, may have any data type.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also may have any data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray of bool dtype
    An array containing the result of element-wise less-than or equal-to

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.greater` : Return the truth value of (x1 > x2) element-wise.
:obj:`dpnp.less` : Return the truth value of (x1 < x2) element-wise.
:obj:`dpnp.greater_equal` : Return the truth value of (x1 >= x2) element-wise.
:obj:`dpnp.equal` : Return (x1 == x2) element-wise.
:obj:`dpnp.not_equal` : Return (x1 != x2) element-wise.

Notes
-----
At least one of `x1` or `x2` must be an array.

If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
(which becomes the shape of the output).

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([4, 2, 1])
>>> x2 = np.array([2, 2, 2]
>>> np.less_equal(x1, x2)
array([False,  True,  True])

The ``<=`` operator can be used as a shorthand for ``less_equal`` on
:class:`dpnp.ndarray`.

>>> a = np.array([4, 2, 1])
>>> b = np.array([2, 2, 2])
>>> a <= b
array([False,  True,  True])

"""

less_equal = DPNPBinaryFunc(
    "less_equal",
    ti._less_equal_result_type,
    ti._less_equal,
    _LESS_EQUAL_DOCSTRING,
)


_LOGICAL_AND_DOCSTRING = """
Computes the logical AND for each element :math:`x1_i` of the input array `x1`
with the respective element :math:`x2_i` of the input array `x2`.

For full documentation refer to :obj:`numpy.logical_and`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, may have any data type.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also may have any data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray of bool dtype
    An array containing the element-wise logical AND results.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.logical_or` : Compute the truth value of x1 OR x2 element-wise.
:obj:`dpnp.logical_not` : Compute the truth value of NOT x element-wise.
:obj:`dpnp.logical_xor` : Compute the truth value of x1 XOR x2, element-wise.
:obj:`dpnp.bitwise_and` : Compute the bit-wise AND of two arrays element-wise.

Notes
-----
At least one of `x1` or `x2` must be an array.

If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
(which becomes the shape of the output).

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([True, False])
>>> x2 = np.array([False, False])
>>> np.logical_and(x1, x2)
array([False, False])

>>> x = np.arange(5)
>>> np.logical_and(x > 1, x < 4)
array([False, False,  True,  True, False])

The ``&`` operator can be used as a shorthand for ``logical_and`` on
boolean :class:`dpnp.ndarray`.

>>> a = np.array([True, False])
>>> b = np.array([False, False])
>>> a & b
array([False, False])

"""

logical_and = DPNPBinaryFunc(
    "logical_and",
    ti._logical_and_result_type,
    ti._logical_and,
    _LOGICAL_AND_DOCSTRING,
)


_LOGICAL_NOT_DOCSTRING = """
Computes the logical NOT for each element :math:`x_i` of input array `x`.

For full documentation refer to :obj:`numpy.logical_not`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, may have any data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray of bool dtype
    An array containing the element-wise logical NOT results.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.logical_and` : Compute the truth value of x1 AND x2 element-wise.
:obj:`dpnp.logical_or` : Compute the truth value of x1 OR x2 element-wise.
:obj:`dpnp.logical_xor` : Compute the truth value of x1 XOR x2, element-wise.

Notes
-----
At least one of `x1` or `x2` must be an array.

If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
(which becomes the shape of the output).

Examples
--------
>>> import dpnp as np
>>> x = np.array([True, False, 0, 1])
>>> np.logical_not(x)
array([False,  True,  True, False])

>>> x = np.arange(5)
>>> np.logical_not(x < 3)
array([False, False, False,  True,  True])

"""

logical_not = DPNPUnaryFunc(
    "logical_not",
    ti._logical_not_result_type,
    ti._logical_not,
    _LOGICAL_NOT_DOCSTRING,
)


_LOGICAL_OR_DOCSTRING = """
Computes the logical OR for each element :math:`x1_i` of the input array `x1`
with the respective element :math:`x2_i` of the input array `x2`.

For full documentation refer to :obj:`numpy.logical_or`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, may have any data type.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also may have any data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray of bool dtype
    An array containing the element-wise logical OR results.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.logical_and` : Compute the truth value of x1 AND x2 element-wise.
:obj:`dpnp.logical_not` : Compute the truth value of NOT x element-wise.
:obj:`dpnp.logical_xor` : Compute the truth value of x1 XOR x2, element-wise.
:obj:`dpnp.bitwise_or` : Compute the bit-wise OR of two arrays element-wise.

Notes
-----
At least one of `x1` or `x2` must be an array.

If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
(which becomes the shape of the output).

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([True, False])
>>> x2 = np.array([False, False])
>>> np.logical_or(x1, x2)
array([ True, False])

>>> x = np.arange(5)
>>> np.logical_or(x < 1, x > 3)
array([ True, False, False, False,  True])

The ``|`` operator can be used as a shorthand for ``logical_or`` on
boolean :class:`dpnp.ndarray`.

>>> a = np.array([True, False])
>>> b = np.array([False, False])
>>> a | b
array([ True, False])

"""

logical_or = DPNPBinaryFunc(
    "logical_or",
    ti._logical_or_result_type,
    ti._logical_or,
    _LOGICAL_OR_DOCSTRING,
)


_LOGICAL_XOR_DOCSTRING = """
Computes the logical XOR for each element :math:`x1_i` of the input array `x1`
with the respective element :math:`x2_i` of the input array `x2`.

For full documentation refer to :obj:`numpy.logical_xor`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, may have any data type.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also may have any data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray of bool dtype
    An array containing the element-wise logical XOR results.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.logical_and` : Compute the truth value of x1 AND x2 element-wise.
:obj:`dpnp.logical_or` : Compute the truth value of x1 OR x2 element-wise.
:obj:`dpnp.logical_not` : Compute the truth value of NOT x element-wise.
:obj:`dpnp.bitwise_xor` : Compute the bit-wise XOR of two arrays element-wise.

Notes
-----
At least one of `x1` or `x2` must be an array.

If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
(which becomes the shape of the output).

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([True, True, False, False])
>>> x2 = np.array([True, False, True, False])
>>> np.logical_xor(x1, x2)
array([False,  True,  True, False])

>>> x = np.arange(5)
>>> np.logical_xor(x < 1, x > 3)
array([ True, False, False, False,  True])

Simple example showing support of broadcasting

>>> np.logical_xor(0, np.eye(2))
array([[ True, False],
       [False,  True]])

"""

logical_xor = DPNPBinaryFunc(
    "logical_xor",
    ti._logical_xor_result_type,
    ti._logical_xor,
    _LOGICAL_XOR_DOCSTRING,
)


_NOT_EQUAL_DOCSTRING = """
Calculates inequality test results for each element :math:`x1_i` of the input
array `x1` with the respective element :math:`x2_i` of the input array `x2`.

For full documentation refer to :obj:`numpy.not_equal`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, may have any data type.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also may have any data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.

    Default: ``None``.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.

    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray of bool dtype
    An array containing the result of element-wise inequality comparison.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.equal` : Return (x1 == x2) element-wise.
:obj:`dpnp.greater` : Return the truth value of (x1 > x2) element-wise.
:obj:`dpnp.greater_equal` : Return the truth value of (x1 >= x2) element-wise.
:obj:`dpnp.less` : Return the truth value of (x1 < x2) element-wise.
:obj:`dpnp.less_equal` : Return the truth value of (x1 =< x2) element-wise.

Notes
-----
At least one of `x1` or `x2` must be an array.

If ``x1.shape != x2.shape``, they must be broadcastable to a common shape
(which becomes the shape of the output).

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([1., 2.])
>>> x2 = np.arange(1., 3.)
>>> np.not_equal(x1, x2)
array([False, False])

The ``!=`` operator can be used as a shorthand for ``not_equal`` on
:class:`dpnp.ndarray`.

>>> a = np.array([1., 2.])
>>> b = np.array([1., 3.])
>>> a != b
array([False,  True])

"""

not_equal = DPNPBinaryFunc(
    "not_equal",
    ti._not_equal_result_type,
    ti._not_equal,
    _NOT_EQUAL_DOCSTRING,
)
