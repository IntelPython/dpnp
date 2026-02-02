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

from libcpp cimport bool as cpp_bool

# TODO: replace with
# from dpctl.tensor._usmarray cimport ...
# when dpctl.tensor is removed from dpctl
from ._usmarray cimport (
    USM_ARRAY_C_CONTIGUOUS,
    USM_ARRAY_F_CONTIGUOUS,
    USM_ARRAY_WRITABLE,
    usm_ndarray,
)


cdef cpp_bool _check_bit(int flag, int mask):
    return (flag & mask) == mask


cdef class Flags:
    """
    Helper class to query the flags of a :class:`dpctl.tensor.usm_ndarray`
    instance, which describe how the instance interfaces with its underlying
    memory.
    """
    cdef int flags_
    cdef usm_ndarray arr_

    def __cinit__(self, usm_ndarray arr, int flags):
        self.arr_ = arr
        self.flags_ = flags

    @property
    def flags(self):
        """
        Integer representation of the memory layout flags of
        :class:`dpctl.tensor.usm_ndarray` instance.
        """
        return self.flags_

    @property
    def c_contiguous(self):
        """
        True if the memory layout of the
        :class:`dpctl.tensor.usm_ndarray` instance is C-contiguous.
        """
        return _check_bit(self.flags_, USM_ARRAY_C_CONTIGUOUS)

    @property
    def f_contiguous(self):
        """
        True if the memory layout of the
        :class:`dpctl.tensor.usm_ndarray` instance is F-contiguous.
        """
        return _check_bit(self.flags_, USM_ARRAY_F_CONTIGUOUS)

    @property
    def writable(self):
        """
        True if :class:`dpctl.tensor.usm_ndarray` instance is writable.
        """
        return _check_bit(self.flags_, USM_ARRAY_WRITABLE)

    @writable.setter
    def writable(self, new_val):
        if not isinstance(new_val, bool):
            raise TypeError("Expecting a boolean value")
        self.arr_._set_writable_flag(new_val)

    @property
    def fc(self):
        """
        True if the memory layout of the :class:`dpctl.tensor.usm_ndarray`
        instance is C-contiguous and F-contiguous.
        """
        return (
           _check_bit(self.flags_, USM_ARRAY_C_CONTIGUOUS)
           and _check_bit(self.flags_, USM_ARRAY_F_CONTIGUOUS)
        )

    @property
    def forc(self):
        """
        True if the memory layout of the :class:`dpctl.tensor.usm_ndarray`
        instance is C-contiguous or F-contiguous.
        """
        return (
           _check_bit(self.flags_, USM_ARRAY_C_CONTIGUOUS)
           or _check_bit(self.flags_, USM_ARRAY_F_CONTIGUOUS)
        )

    @property
    def fnc(self):
        """
        True if the memory layout of the :class:`dpctl.tensor.usm_ndarray`
        instance is F-contiguous and not C-contiguous.
        """
        return (
           _check_bit(self.flags_, USM_ARRAY_F_CONTIGUOUS)
           and not _check_bit(self.flags_, USM_ARRAY_C_CONTIGUOUS)
        )

    @property
    def contiguous(self):
        """
        True if the memory layout of the :class:`dpctl.tensor.usm_ndarray`
        instance is C-contiguous and F-contiguous.
        Equivalent to `forc.`
        """
        return self.forc

    def __getitem__(self, name):
        if name in ["C_CONTIGUOUS", "C"]:
            return self.c_contiguous
        elif name in ["F_CONTIGUOUS", "F"]:
            return self.f_contiguous
        elif name in ["WRITABLE", "W"]:
            return self.writable
        elif name == "FC":
            return self.fc
        elif name == "FNC":
            return self.fnc
        elif name in ["FORC", "CONTIGUOUS"]:
            return self.forc

    def __setitem__(self, name, val):
        if name in ["WRITABLE", "W"]:
            self.writable = val
        else:
            raise ValueError(
                "Only writable ('W' or 'WRITABLE') flag can be set"
            )

    def __repr__(self):
        out = []
        for name in "C_CONTIGUOUS", "F_CONTIGUOUS", "WRITABLE":
            out.append("  {} : {}".format(name, self[name]))
        return "\n".join(out)

    def __eq__(self, other):
        cdef Flags other_
        if isinstance(other, self.__class__):
            other_ = <Flags>other
            return self.flags_ == other_.flags_
        elif isinstance(other, int):
            return self.flags_ == <int>other
        else:
            return False
