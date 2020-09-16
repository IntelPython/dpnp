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
Interface of the Intel NumPy

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
from dpnp.random import *
from dpnp.linalg import *
from dpnp.dparray import dparray
from dpnp.dpnp_utils import checker_throw_value_error, use_origin_backend
import collections

__all__ = [
    "arange",
    "array",
    "array_equal",
    "asarray",
    "asnumpy",
    "empty",
    "dpnp_queue_initialize",
    "matmul",
    "ones",
    "remainder",
    "zeros"
]

from dpnp.dpnp_iface_libmath import *
from dpnp.dpnp_iface_linearalgebra import *
from dpnp.dpnp_iface_logic import *
from dpnp.dpnp_iface_manipulation import *
from dpnp.dpnp_iface_mathematical import *
from dpnp.dpnp_iface_searching import *
from dpnp.dpnp_iface_sorting import *
from dpnp.dpnp_iface_statistics import *
from dpnp.dpnp_iface_trigonometric import *

from dpnp.dpnp_iface_libmath import __all__ as __all__libmath
from dpnp.dpnp_iface_linearalgebra import __all__ as __all__linearalgebra
from dpnp.dpnp_iface_logic import __all__ as __all__logic
from dpnp.dpnp_iface_manipulation import __all__ as __all__manipulation
from dpnp.dpnp_iface_mathematical import __all__ as __all__mathematical
from dpnp.dpnp_iface_searching import __all__ as __all__searching
from dpnp.dpnp_iface_sorting import __all__ as __all__sorting
from dpnp.dpnp_iface_statistics import __all__ as __all__statistics
from dpnp.dpnp_iface_trigonometric import __all__ as __all__trigonometric

__all__ += __all__libmath
__all__ += __all__linearalgebra
__all__ += __all__logic
__all__ += __all__manipulation
__all__ += __all__mathematical
__all__ += __all__searching
__all__ += __all__sorting
__all__ += __all__statistics
__all__ += __all__trigonometric


def arange(*args, **kwargs):
    """Returns an array with evenly spaced values within a given interval.

    Values are generated within the half-open interval [start, stop). The first
    three arguments are mapped like the ``range`` built-in function, i.e. start
    and step are optional.

    Parameters
    ----------
        start: Start of the interval.
        stop: End of the interval.
        step: Step width between each pair of consecutive values.
        dtype: Data type specifier. It is inferred from other arguments by
            default.

    Returns
    -------
        inumpy.dparray: The 1-D array of range values.

    .. seealso:: :func:`numpy.arange`

    """

    if (use_origin_backend()):
        return numpy.arange(*args, **kwargs)

    if not isinstance(args[0], (int)):
        raise TypeError(f"Intel NumPy arange(): scalar arguments expected. Given:{type(args[0])}")

    start_param = 0
    stop_param = 0
    step_param = 1
    dtype_param = kwargs.pop("dtype", None)
    if dtype_param is None:
        dtype_param = numpy.float64

    if kwargs:
        raise TypeError("Intel NumPy arange(): unexpected keyword argument(s): %s" % ",".join(kwargs.keys()))

    args_len = len(args)
    if args_len == 1:
        stop_param = args[0]
    elif args_len == 2:
        start_param = args[0]
        stop_param = args[1]
    elif args_len == 3:
        start_param, stop_param, step_param = args
    else:
        raise TypeError("Intel NumPy arange() takes 3 positional arguments: arange([start], stop, [step])")

    return dpnp_arange(start_param, stop_param, step_param, dtype_param)


def array(obj, dtype=None, copy=True, order='C', subok=False, ndmin=0):
    """
    Creates an array.

    This function currently does not support the ``subok`` option.

    Args:
        obj: :class:`inumpy.dparray` object or any other object that can be
            passed to :func:`numpy.array`.
        dtype: Data type specifier.
        copy (bool): If ``False``, this function returns ``obj`` if possible.
            Otherwise this function always returns a new array.
        order ({'C', 'F', 'A', 'K'}): Row-major (C-style) or column-major
            (Fortran-style) order.
            When ``order`` is 'A', it uses 'F' if ``a`` is column-major and
            uses 'C' otherwise.
            And when ``order`` is 'K', it keeps strides as closely as
            possible.
            If ``obj`` is :class:`numpy.ndarray`, the function returns 'C' or
            'F' order array.
        subok (bool): If True, then sub-classes will be passed-through,
            otherwise the returned array will be forced to be a base-class
            array (default).
        ndmin (int): Minimum number of dimensions. Ones are inserted to the
            head of the shape if needed.

    Returns:
        inumpy.dparray: An array on the current device.



    .. note::
       This method currently does not support ``subok`` argument.

    .. seealso:: :func:`numpy.array`

    """

    if (use_origin_backend(obj)):
        return numpy.array(obj, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin)

    # if not isinstance(obj, collections.abc.Sequence):
    #     return numpy.array(obj, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin)

    # if isinstance(obj, numpy.object):
    #     return numpy.array(obj, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin)

    if subok is not False:
        checker_throw_value_error("array", "subok", subok, False)

    if copy is not True:
        checker_throw_value_error("array", "copy", copy, True)

    if order is not 'C':
        checker_throw_value_error("array", "order", order, 'K')

    if ndmin is not 0:
        checker_throw_value_error("array", "ndmin", ndmin, 0)

    return dpnp_array(obj, dtype)


def array_equal(a1, a2, equal_nan=False):
    """True if two arrays have the same shape and elements, False otherwise.

    Parameters
        a1, a2: array_like
            Input arrays.

        equal_nanbool
            Whether to compare NaN’s as equal. If the dtype of a1 and a2 is complex,
            values will be considered equal if either the real or the imaginary component of a given value is nan.
            New in version 1.19.0.

    Returns
        b: bool
            Returns True if the arrays are equal.

    .. seealso:: :func:`numpy.allclose` :func:`numpy.array_equiv`

    """

    return numpy.array_equal(a1, a2)


def asnumpy(input, order='C'):
    """Returns the NumPy array with input data.

    Args:
        input: Arbitrary object that can be converted to :class:`numpy.ndarray`.
    Returns:
        numpy.ndarray: array with input data.

    """

    return numpy.asarray(input, order=order)


def asarray(input, dtype=None, order='C'):
    """Converts an input object into array.

    This is equivalent to ``array(a, dtype, copy=False)``.

    Args:
        input: The source object.
        dtype: Data type specifier. It is inferred from the input by default.
        order{‘C’, ‘F’}, optional
            Whether to use row-major (C-style) or column-major (Fortran-style) memory representation.
            Defaults to ‘C’.

    Returns:
        inumpy.dparray populated with input data

    .. seealso:: :func:`numpy.asarray`

    """

    if (use_origin_backend(input)):
        return numpy.asarray(input, dtype=dtype, order=order)

    return array(input, dtype=dtype, order=order)


def empty(shape, dtype=None, order='C'):
    """Return a new matrix of given shape and type, without initializing entries.
    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty matrix.
    dtype : data-type, optional
        Desired output data-type.
    order : {'C', 'F'}, optional
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in
        memory.
    See Also
    --------
    empty_like, zeros
    Notes
    -----
    `empty`, unlike `zeros`, does not set the matrix values to zero,
    and may therefore be marginally faster.  On the other hand, it requires
    the user to manually set all the values in the array, and should be
    used with caution.
    Examples
    --------
    >>> import numpy.matlib
    >>> np.matlib.empty((2, 2))    # filled with random data
    matrix([[  6.76425276e-320,   9.79033856e-307], # random
            [  7.39337286e-309,   3.22135945e-309]])
    >>> np.matlib.empty((2, 2), dtype=int)
    matrix([[ 6600475,        0], # random
            [ 6586976, 22740995]])
    """

    if (use_origin_backend()):
        return numpy.empty(shape, dtype, order=order)

    # only 'C' order is supported for now
    if order not in ('C', 'c', None):
        checker_throw_value_error("empty", "order", order, 'C')

    return dparray(shape, dtype)


def matmul(in_array1, in_array2, out=None):
    """
    Returns the matrix product of two arrays and is the implementation of
    the `@` operator introduced in Python 3.5 following PEP465.

    The main difference against dpnp.dot are the handling of arrays with more
    than 2 dimensions. For more information see :func:`numpy.matmul`.

    .. note::
        The out array as input is currently not supported.

    Args:
        in_array1 (dpnp.dparray): The left argument.
        in_array2 (dpnp.dparray): The right argument.
        out (dpnp.dparray): Output array.

    Returns:
        dpnp.dparray: Output array.

    .. seealso:: :func:`numpy.matmul`

    """

    is_dparray1 = isinstance(in_array1, dparray)
    is_dparray2 = isinstance(in_array2, dparray)

    if (not use_origin_backend(in_array1) and is_dparray1 and is_dparray2):

        if out is not None:
            checker_throw_value_error("matmul", "out", type(out), None)

        """
        Cost model checks
        """
        cost_size = 4096  # 2D array shape(64, 64)
        if ((in_array1.dtype == numpy.float64) or (in_array1.dtype == numpy.float32)):
            """
            Floating point types are handled via original MKL better than SYCL MKL
            """
            cost_size = 262144  # 2D array shape(512, 512)

        dparray1_size = in_array1.size
        dparray2_size = in_array2.size

        if (dparray1_size > cost_size) and (dparray2_size > cost_size):
            # print(f"dparray1_size={dparray1_size}")
            return dpnp_matmul(in_array1, in_array2)

    input1 = asnumpy(in_array1) if is_dparray1 else in_array1
    input2 = asnumpy(in_array2) if is_dparray2 else in_array2

    # TODO need to return dparray instead ndarray
    return numpy.matmul(input1, input2, out=out)


def ones(shape, dtype=None, order='C'):
    """
    Return a new array of given shape and type, filled with ones.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int64`.  Default is
        `numpy.float64`.
    order : {'C', 'F'}, optional, default: C
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in
        memory.

    Returns
    -------
    out : dparray
        Array of ones with the given shape, dtype, and order.

    See Also
    --------
    ones_like : Return an array of ones with shape and type of input.
    empty : Return a new uninitialized array.
    zeros : Return a new array setting values to zero.
    full : Return a new array of given shape filled with value.

    Examples
    --------
    >>> np.ones(5)
    array([1., 1., 1., 1., 1.])
    >>> np.ones((5,), dtype=int64)
    array([1, 1, 1, 1, 1])
    >>> np.ones((2, 1))
    array([[1.],
           [1.]])
    >>> s = (2,2)
    >>> np.ones(s)
    array([[1.,  1.],
           [1.,  1.]])
    """

    if (use_origin_backend()):
        return numpy.ones(shape, dtype=dtype, order=order)

    # only 'C' order is supported for now
    if order not in ('C', 'c', None):
        checker_throw_value_error("ones", "order", order, 'C')

    return dpnp_init_val(shape, dtype, 1)


def remainder(x1, x2):
    """
    Return element-wise remainder of division.

    Computes the remainder complementary to the `floor_divide` function.  It is
    equivalent to the Python modulus operator``x1 % x2`` and has the same sign
    as the divisor `x2`. The MATLAB function equivalent to ``np.remainder``
    is ``mod``.

    .. warning::

    This should not be confused with:

    * Python 3.7's `math.remainder` and C's ``remainder``, which
      computes the IEEE remainder, which are the complement to
      ``round(x1 / x2)``.
    * The MATLAB ``rem`` function and or the C ``%`` operator which is the
      complement to ``int(x1 / x2)``.

    Parameters
    ----------
    x1 : array_like
        Dividend array.
    x2 : array_like
        Divisor array. If ``x1.shape != x2.shape``, they must be broadcastable
        to a common shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : ndarray
        The element-wise remainder of the quotient ``floor_divide(x1, x2)``.
        This is a scalar if both `x1` and `x2` are scalars.

    See Also
    --------
    floor_divide : Equivalent of Python ``//`` operator.
    divmod : Simultaneous floor division and remainder.
    fmod : Equivalent of the MATLAB ``rem`` function.
    divide, floor

    Notes
    -----
    Returns 0 when `x2` is 0 and both `x1` and `x2` are (arrays of)
    integers.
    ``mod`` is an alias of ``remainder``.

    Examples
    --------
    >>> np.remainder([4, 7], [2, 3])
    array([0, 1])
    >>> np.remainder(np.arange(7), 5)
    array([0, 1, 2, 3, 4, 0, 1])

    """

    if (use_origin_backend(x1)):
        return numpy.remainder(x1, x2)

    if not isinstance(x1, dparray):
        raise TypeError(f"Intel NumPy remainder(): Unsupported input1={type(x1)}")

    if not isinstance(x2, int):
        raise TypeError(f"Intel NumPy remainder(): Unsupported input2={type(x2)}")

    return dpnp_remainder(x1, x2)


def zeros(shape, dtype=None, order='C'):
    """
    Return a new array of given shape and type, filled with zeros.

    .. seealso:: :func:`numpy.zeros`

    """

    if (use_origin_backend()):
        return numpy.zeros(shape, dtype=dtype, order=order)

    # only 'C' order is supported for now
    if order not in ('C', 'c', None):
        checker_throw_value_error("zeros", "order", order, 'C')

    return dpnp_init_val(shape, dtype, 0)
