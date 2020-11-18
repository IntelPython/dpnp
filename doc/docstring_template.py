# cython: language_level=3
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

def func(a, b=None, c=True):
    """
    Short (one-line) description.

    Long (perhaps multi-line) description.
    Both short and long descriptions,
    descriptions of input parameters and outputs 
    maybe partially taken from baseline docstring.

    Parameters
    ----------
    a : array_like
        Description of the parameter `a`.
    b : array_like, optional
        Description of the parameter `b`.
    c : bool, optional
        Description of the parameter `c`.

    Returns
    -------
    out : dparray
        Description of the output.

    Limitations
    -----------
    Some limitations in comparison to baseline, ex:
    Prameters `c` is supported only with default value `True`.
    Otherwise the functions will be executed sequentially on CPU.

    See Also
    --------
    base_func, func2, func3

    Notes
    -----
    Some notes if required.

    Examples
    --------
    Some examples, ex:

    >>> a = np.array([0, 1, 2])
    >>> b = np.array([2, 1, 0])
    >>> x = np.func(a, b=b, c=True)
    >>> x
    array([0, 1, 2, 2, 1, 0])

    """
    pass
