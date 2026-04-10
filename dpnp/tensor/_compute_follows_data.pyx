# *****************************************************************************
# Copyright (c) 2026, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
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

# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True

"""Compute-follows-data utilities for execution queue and USM type management.

This module provides utilities to determine execution placement and USM allocation
types when combining arrays under the compute-follows-data paradigm.
"""


import dpctl
from dpctl._sycl_queue cimport SyclQueue

__all__ = [
    "get_execution_queue", "get_coerced_usm_type", "ExecutionPlacementError"
]


class ExecutionPlacementError(Exception):
    """Exception raised when execution placement target can not
    be unambiguously determined from input arrays.

    Make sure that input arrays are associated with the same
    :class:`dpctl.SyclQueue`,
    or migrate data to the same :class:`dpctl.SyclQueue` using
    :meth:`dpctl.tensor.usm_ndarray.to_device` method.
    """
    pass


cdef bint queue_equiv(SyclQueue q1, SyclQueue q2):
    """Queues are equivalent if ``q1 == q2``, that is they are copies
    of the same underlying SYCL object and hence are the same."""
    return q1.__eq__(q2)


def get_execution_queue(qs, /):
    """
    Get execution queue from queues associated with input arrays.

    Args:
        qs (List[:class:`dpctl.SyclQueue`], Tuple[:class:`dpctl.SyclQueue`]):
            a list or a tuple of :class:`dpctl.SyclQueue` objects
            corresponding to arrays that are being combined.

    Returns:
        SyclQueue:
            execution queue under compute follows data paradigm,
            or ``None`` if queues are not equal.
    """
    if not isinstance(qs, (list, tuple)):
        raise TypeError(
            "Expected a list or a tuple, got {}".format(type(qs))
        )
    if len(qs) == 0:
        return None
    elif len(qs) == 1:
        return qs[0] if isinstance(qs[0], dpctl.SyclQueue) else None
    for q1, q2 in zip(qs[:-1], qs[1:]):
        if not isinstance(q1, dpctl.SyclQueue):
            return None
        elif not isinstance(q2, dpctl.SyclQueue):
            return None
        elif not queue_equiv(<SyclQueue> q1, <SyclQueue> q2):
            return None
    return qs[0]


def get_coerced_usm_type(usm_types, /):
    """
    Get USM type of the output array for a function combining
    arrays of given USM types using compute-follows-data execution
    model.

    Args:
        usm_types (List[str], Tuple[str]):
            a list or a tuple of strings of ``.usm_types`` attributes
            for input arrays

    Returns:
         str
            type of USM allocation for the output arrays (s).
            ``None`` if any of the input strings are not recognized.
    """
    if not isinstance(usm_types, (list, tuple)):
        raise TypeError(
            "Expected a list or a tuple, got {}".format(type(usm_types))
        )
    if len(usm_types) == 0:
        return None
    _k = ["device", "shared", "host"]
    _m = {k: i for i, k in enumerate(_k)}
    res = len(_k)
    for t in usm_types:
        if not isinstance(t, str):
            return None
        if t not in _m:
            return None
        res = min(res, _m[t])
    return _k[res]


def _validate_usm_type_allow_none(usm_type):
    "Validates usm_type argument"
    if usm_type is not None:
        if isinstance(usm_type, str):
            if usm_type not in ["device", "shared", "host"]:
                raise ValueError(
                    f"Unrecognized value of usm_type={usm_type}, "
                    "expected 'device', 'shared', 'host', or None."
                )
        else:
            raise TypeError(
                f"Expected usm_type to be a str or None, got {type(usm_type)}"
            )


def _validate_usm_type_disallow_none(usm_type):
    "Validates usm_type argument"
    if isinstance(usm_type, str):
        if usm_type not in ["device", "shared", "host"]:
            raise ValueError(
                f"Unrecognized value of usm_type={usm_type}, "
                "expected 'device', 'shared', or 'host'."
            )
    else:
        raise TypeError(
            f"Expected usm_type to be a str, got {type(usm_type)}"
        )


def validate_usm_type(usm_type, /, *, allow_none=True):
    """ validate_usm_type(usm_type, allow_none=True)

    Raises an exception if `usm_type` is invalid.

    Args:
        usm_type:
            Specification for USM allocation type. Valid specifications
            are:

            * ``"device"``
            * ``"shared"``
            * ``"host"``

            If ``allow_none`` keyword argument is set, a value of
            ``None`` is also permitted.
        allow_none (bool, optional):
            Whether ``usm_type`` value of ``None`` is considered valid.
            Default: `True`.

    Raises:
        ValueError:
            if ``usm_type`` is not a recognized string.
        TypeError:
            if ``usm_type`` is not a string, and ``usm_type`` is
            not ``None`` provided ``allow_none`` is ``True``.
    """
    if allow_none:
        _validate_usm_type_allow_none(usm_type)
    else:
        _validate_usm_type_disallow_none(usm_type)
