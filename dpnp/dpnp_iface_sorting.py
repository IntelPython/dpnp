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
Interface of the sorting function of the dpnp

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""


import numpy

from dpnp.backend import *
from dpnp.dparray import dparray
from dpnp.dpnp_utils import *


__all__ = [
    'argsort',
    'sort'
]


def argsort(in_array1, axis=-1, kind=None, order=None):
    """
    Return an ndarray of indices that sort the array along the
    specified axis.  Masked values are filled beforehand to
    `fill_value`.
    Parameters
    ----------
    axis : int, optional
        Axis along which to sort. If None, the default, the flattened array
        is used.
        ..  versionchanged:: 1.13.0
            Previously, the default was documented to be -1, but that was
            in error. At some future date, the default will change to -1, as
            originally intended.
            Until then, the axis should be given explicitly when
            ``arr.ndim > 1``, to avoid a FutureWarning.
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
        The sorting algorithm used.
    order : list, optional
        When `a` is an array with fields defined, this argument specifies
        which fields to compare first, second, etc.  Not all fields need be
        specified.
    endwith : {True, False}, optional
        Whether missing values (if any) should be treated as the largest values
        (True) or the smallest values (False)
        When the array contains unmasked values at the same extremes of the
        datatype, the ordering of these values and the masked values is
        undefined.
    fill_value : {var}, optional
        Value used internally for the masked values.
        If ``fill_value`` is not None, it supersedes ``endwith``.
    Returns
    -------
    index_array : ndarray, int
        Array of indices that sort `a` along the specified axis.
        In other words, ``a[index_array]`` yields a sorted `a`.
    See Also
    --------
    MaskedArray.sort : Describes sorting algorithms used.
    lexsort : Indirect stable sort with multiple keys.
    numpy.ndarray.sort : Inplace sort.
    Notes
    -----
    See `sort` for notes on the different sorting algorithms.
    Examples
    --------
    >>> a = np.ma.array([3,2,1], mask=[False, False, True])
    >>> a
    masked_array(data=[3, 2, --],
                 mask=[False, False,  True],
           fill_value=999999)
    >>> a.argsort()
    array([1, 0, 2])

    """

    is_dparray1 = isinstance(in_array1, dparray)

    if (not use_origin_backend(in_array1) and is_dparray1):
        if axis != -1:
            checker_throw_value_error("argsort", "axis", axis, -1)
        if kind is not None:
            checker_throw_value_error("argsort", "kind", type(kind), None)
        if order is not None:
            checker_throw_value_error("argsort", "order", type(order), None)

        return dpnp_argsort(in_array1)

    return numpy.argsort(in_array1, axis, kind, order)


def sort(x1, **kwargs):
    """
    Return a sorted copy of an array.
    Parameters
    ----------
    x1 : array_like
        Array to be sorted.
    axis : int or None, optional
        Axis along which to sort. If None, the array is flattened before
        sorting. The default is -1, which sorts along the last axis.
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
        Sorting algorithm. The default is 'quicksort'. Note that both 'stable'
        and 'mergesort' use timsort or radix sort under the covers and, in general,
        the actual implementation will vary with data type. The 'mergesort' option
        is retained for backwards compatibility.
        .. versionchanged:: 1.15.0.
           The 'stable' option was added.
    order : str or list of str, optional
        When `a` is an array with fields defined, this argument specifies
        which fields to compare first, second, etc.  A single field can
        be specified as a string, and not all fields need be specified,
        but unspecified fields will still be used, in the order in which
        they come up in the dtype, to break ties.
    Returns
    -------
    sorted_array : ndarray
        Array of the same type and shape as `a`.
    See Also
    --------
    ndarray.sort : Method to sort an array in-place.
    argsort : Indirect sort.
    lexsort : Indirect stable sort on multiple keys.
    searchsorted : Find elements in a sorted array.
    partition : Partial sort.
    Notes
    -----
    The various sorting algorithms are characterized by their average speed,
    worst case performance, work space size, and whether they are stable. A
    stable sort keeps items with the same key in the same relative
    order. The four algorithms implemented in NumPy have the following
    properties:
    =========== ======= ============= ============ ========
       kind      speed   worst case    work space   stable
    =========== ======= ============= ============ ========
    'quicksort'    1     O(n^2)            0          no
    'heapsort'     3     O(n*log(n))       0          no
    'mergesort'    2     O(n*log(n))      ~n/2        yes
    'timsort'      2     O(n*log(n))      ~n/2        yes
    =========== ======= ============= ============ ========
    .. note:: The datatype determines which of 'mergesort' or 'timsort'
       is actually used, even if 'mergesort' is specified. User selection
       at a finer scale is not currently available.
    All the sort algorithms make temporary copies of the data when
    sorting along any but the last axis.  Consequently, sorting along
    the last axis is faster and uses less space than sorting along
    any other axis.
    The sort order for complex numbers is lexicographic. If both the real
    and imaginary parts are non-nan then the order is determined by the
    real parts except when they are equal, in which case the order is
    determined by the imaginary parts.
    Previous to numpy 1.4.0 sorting real and complex arrays containing nan
    values led to undefined behaviour. In numpy versions >= 1.4.0 nan
    values are sorted to the end. The extended sort order is:
      * Real: [R, nan]
      * Complex: [R + Rj, R + nanj, nan + Rj, nan + nanj]
    where R is a non-nan real value. Complex values with the same nan
    placements are sorted according to the non-nan part if it exists.
    Non-nan values are sorted as before.
    .. versionadded:: 1.12.0
    quicksort has been changed to `introsort <https://en.wikipedia.org/wiki/Introsort>`_.
    When sorting does not make enough progress it switches to
    `heapsort <https://en.wikipedia.org/wiki/Heapsort>`_.
    This implementation makes quicksort O(n*log(n)) in the worst case.
    'stable' automatically chooses the best stable sorting algorithm
    for the data type being sorted.
    It, along with 'mergesort' is currently mapped to
    `timsort <https://en.wikipedia.org/wiki/Timsort>`_
    or `radix sort <https://en.wikipedia.org/wiki/Radix_sort>`_
    depending on the data type.
    API forward compatibility currently limits the
    ability to select the implementation and it is hardwired for the different
    data types.
    .. versionadded:: 1.17.0
    Timsort is added for better performance on already or nearly
    sorted data. On random data timsort is almost identical to
    mergesort. It is now used for stable sort while quicksort is still the
    default sort if none is chosen. For timsort details, refer to
    `CPython listsort.txt <https://github.com/python/cpython/blob/3.7/Objects/listsort.txt>`_.
    'mergesort' and 'stable' are mapped to radix sort for integer data types. Radix sort is an
    O(n) sort instead of O(n log n).
    .. versionchanged:: 1.18.0
    NaT now sorts to the end of arrays for consistency with NaN.
    Examples
    --------
    >>> a = np.array([[1,4],[3,1]])
    >>> np.sort(a)                # sort along the last axis
    array([[1, 4],
           [1, 3]])
    >>> np.sort(a, axis=None)     # sort the flattened array
    array([1, 1, 3, 4])
    >>> np.sort(a, axis=0)        # sort along the first axis
    array([[1, 1],
           [3, 4]])
    Use the `order` keyword to specify a field to use when sorting a
    structured array:
    >>> dtype = [('name', 'S10'), ('height', float), ('age', int)]
    >>> values = [('Arthur', 1.8, 41), ('Lancelot', 1.9, 38),
    ...           ('Galahad', 1.7, 38)]
    >>> a = np.array(values, dtype=dtype)       # create a structured array
    >>> np.sort(a, order='height')                        # doctest: +SKIP
    array([('Galahad', 1.7, 38), ('Arthur', 1.8, 41),
           ('Lancelot', 1.8999999999999999, 38)],
          dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])
    Sort by age, then height if ages are equal:
    >>> np.sort(a, order=['age', 'height'])               # doctest: +SKIP
    array([('Galahad', 1.7, 38), ('Lancelot', 1.8999999999999999, 38),
           ('Arthur', 1.8, 41)],
          dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])
    """
    if not use_origin_backend(x1) and not kwargs:
        if not isinstance(x1, dparray):
            pass
        elif x1.ndim != 1:
            pass
        else:
            return dpnp_sort(x1)

    return call_origin(numpy.sort, x1, **kwargs)
