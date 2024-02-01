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

"""Implementation of flatiter."""

import dpnp


class flatiter:
    """Flat iterator object to iterate over arrays."""

    def __init__(self, X):
        if type(X) is not dpnp.ndarray:
            raise TypeError(
                "Argument must be of type dpnp.ndarray, got {}".format(type(X))
            )
        self.arr_ = X
        self.size_ = X.size
        self.i_ = 0

    def _multiindex(self, i):
        nd = self.arr_.ndim
        if nd == 0:
            if i == 0:
                return ()
            raise KeyError
        elif nd == 1:
            return (i,)
        sh = self.arr_.shape
        i_ = i
        multi_index = [0] * nd
        for k in reversed(range(1, nd)):
            si = sh[k]
            q = i_ // si
            multi_index[k] = i_ - q * si
            i_ = q
        multi_index[0] = i_
        return tuple(multi_index)

    def __getitem__(self, key):
        idx = getattr(key, "__index__", None)
        if not callable(idx):
            raise TypeError(key)
        i = idx()
        mi = self._multiindex(i)
        return self.arr_.__getitem__(mi)

    def __setitem__(self, key, val):
        idx = getattr(key, "__index__", None)
        if not callable(idx):
            raise TypeError(key)
        i = idx()
        mi = self._multiindex(i)
        return self.arr_.__setitem__(mi, val)

    def __iter__(self):
        return self

    def __next__(self):
        if self.i_ < self.size_:
            val = self.__getitem__(self.i_)
            self.i_ = self.i_ + 1
            return val
        else:
            raise StopIteration
