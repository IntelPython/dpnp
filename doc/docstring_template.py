# cython: language_level=3
# *****************************************************************************
# Copyright (c) 2016, Intel Corporation
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

"""
Template for docstrings of dpnp functions.

Docstrings use the numpydoc format and are rendered by ``doc/conf.py``. They
consist of the sections below, in this order; skip the ones that don't apply:

1. Short (one-line) description.
2. Link to the reference function: ``For full documentation refer to
   :obj:`numpy.func`.`` (or ``:obj:`scipy...``` for ``dpnp.scipy``).
3. Parameters. Array inputs are typed ``{dpnp.ndarray, usm_ndarray}``. Every
   keyword with a default ends with a blank line and ``Default: ``value``.``.
4. Returns, typically ``out : dpnp.ndarray``.
5. Raises, Warns, Warnings.
6. Limitations: what is not supported compared to the reference function and
   the exception raised for it, usually ``NotImplementedError``.
7. See Also: related dpnp functions, each with a short description.
8. Notes, References.
9. Examples, starting with ``>>> import dpnp as np``.

Use an existing function, e.g. :obj:`dpnp.cbrt`, as a model, and copy the
`device`, `usm_type` and `sycl_queue` parameter descriptions from an array
creation function such as :obj:`dpnp.empty`.

"""


def func(a, /, out=None, *, order="K", where=True):
    """
    Short (one-line) description.

    For full documentation refer to :obj:`numpy.func`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Output array to populate. Array must have the correct shape and the
        expected data type.

        Default: ``None``.
    order : {None, "C", "F", "A", "K"}, optional
        Memory layout of the newly output array, if parameter `out` is
        ``None``.

        Default: ``"K"``.

    Returns
    -------
    out : dpnp.ndarray
        An array containing the result.

    Limitations
    -----------
    Parameter `where` is supported only with default value ``True``.
    Otherwise ``NotImplementedError`` exception will be raised.

    See Also
    --------
    :obj:`dpnp.func2` : Short (one-line) description of `dpnp.func2`.

    Notes
    -----
    Some notes, if needed.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([0, 1, 2])
    >>> np.func(a)
    array([0, 1, 2])

    """
