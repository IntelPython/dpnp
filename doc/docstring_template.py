# cython: language_level=3
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2023, Intel Corporation
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
This is template that should be used when writing docstrings in the project.

According to this template docstrings should consist of below sections:
1. Short description.
2. Link to related functionality in `numpy` for full documentation.
3. Limitations is a special section that should describe:
   - the limited functionality relative to `numpy` (exception raised)
   - cases when the functionality will be executed sequentially on CPU
     (fallback to `numpy`)
   - list of supported data types for input array
4. See Also
5. Notes, optional
6. Examples

Recommendations:
1. Short description
   maybe partially taken/combinated from `numpy` and `cupy` docstrings.
2. Limitations basically should be described according to the code,
   paying attention to raised exceptions and fallback to `numpy`.
3. See Also may include links to similar functionality in `dpnp`
   with short description.

   In case of only one link better to use one-line section:
   .. seealso:: :func:`numpy.func` short description.

4. Examples maybe partially taken from `numpy` docstrings
   with some modifications, ex:

   >>> import dpnp as np
   >>> a = np.array([0, 1, 2])
   >>> a  # prints unclear result <DPNP DParray:...

   So the array may be printed through iterating:

   >>> [i for i in a]
   [0, 1, 2]

`dpnp.array` contains a good example of docstring.

"""


def func(a, b=None, c=True):
    """
    Short (one-line) description.

    For full documentation refer to :obj:`numpy.func`.

    Limitations
    -----------
    Some limitations in comparison to baseline, ex:
    Input array data types are limited by supported DPNP :ref:`Data types`.
    Parameter `c` is supported only with default value ``True``.
    Otherwise the function will be executed sequentially on CPU.

    See Also
    --------
    :obj:`dpnp.func2` : Short (one-line) description of the function `dpnp.func2`.
    :obj:`dpnp.func3` : Short (one-line) description of the function `dpnp.func3`.

    Notes
    -----
    Some notes if required.

    Examples
    --------
    Some examples, ex:

    >>> import dpnp as np
    >>> a = np.array([0, 1, 2])
    >>> b = np.array([2, 1, 0])
    >>> x = np.func(a, b=b, c=True)
    >>> [i for i in x]
    [2, 2, 2]

    """
    pass
