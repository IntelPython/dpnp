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
    "array_equal",
    "asnumpy",
    "dpnp_queue_initialize",
    "matmul",
    "remainder"
]

from dpnp.dpnp_iface_arraycreation import *
from dpnp.dpnp_iface_libmath import *
from dpnp.dpnp_iface_linearalgebra import *
from dpnp.dpnp_iface_logic import *
from dpnp.dpnp_iface_manipulation import *
from dpnp.dpnp_iface_mathematical import *
from dpnp.dpnp_iface_sorting import *
from dpnp.dpnp_iface_statistics import *
from dpnp.dpnp_iface_trigonometric import *

from dpnp.dpnp_iface_arraycreation import __all__ as __all__arraycreation
from dpnp.dpnp_iface_libmath import __all__ as __all__libmath
from dpnp.dpnp_iface_linearalgebra import __all__ as __all__linearalgebra
from dpnp.dpnp_iface_logic import __all__ as __all__logic
from dpnp.dpnp_iface_manipulation import __all__ as __all__manipulation
from dpnp.dpnp_iface_mathematical import __all__ as __all__mathematical
from dpnp.dpnp_iface_sorting import __all__ as __all__sorting
from dpnp.dpnp_iface_statistics import __all__ as __all__statistics
from dpnp.dpnp_iface_trigonometric import __all__ as __all__trigonometric

__all__ += __all__arraycreation
__all__ += __all__libmath
__all__ += __all__linearalgebra
__all__ += __all__logic
__all__ += __all__manipulation
__all__ += __all__mathematical
__all__ += __all__sorting
__all__ += __all__statistics
__all__ += __all__trigonometric


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
