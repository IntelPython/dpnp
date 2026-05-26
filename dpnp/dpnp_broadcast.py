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

"""Implementation of broadcast class."""

import dpnp
import dpnp.tensor as dpt
from dpnp.tensor._manipulation_functions import _broadcast_shapes


class broadcast:
    """
    Produce an object that mimics broadcasting.

    For full documentation refer to :obj:`numpy.broadcast`.

    Parameters
    ----------
    *args : object
        Input parameters. Every argument must define ``shape`` attribute.

    Returns
    -------
    broadcast : broadcast object
        Broadcast the input parameters against one another, and
        return an object that encapsulates the result.
        Amongst others, it has ``shape`` and ``nd`` properties, and
        may be used as an iterator.

    See Also
    --------
    :obj:`dpnp.broadcast_arrays` : Broadcast any number of arrays against
        each other.
    :obj:`dpnp.broadcast_to` : Broadcast an array to a new shape.
    :obj:`dpnp.broadcast_shapes` : Broadcast the input shapes into a single
        shape.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([[1], [2], [3]])
    >>> y = np.array([4, 5, 6])
    >>> b = np.broadcast(x, y)
    >>> b.shape
    (3, 3)
    >>> b.nd
    2
    >>> b.size
    9

    Notes
    -----
    Iterator functionality is not supported.

    """

    def __init__(self, *args):
        for i, arg in enumerate(args):
            if not hasattr(arg, "shape"):
                raise TypeError(
                    f"Argument at position {i} must define shape attribute"
                )

        self._arrays = tuple(args)

        dpnp_arrays = [arg for arg in self._arrays if isinstance(arg, dpnp.ndarray)]
        if len(dpnp_arrays) > 1:
            exec_q = dpt.get_execution_queue(
                tuple(array.sycl_queue for array in dpnp_arrays)
            )
            if exec_q is None:
                raise dpt.ExecutionPlacementError(
                    "Execution placement can not be unambiguously inferred "
                    "from input arguments."
                )

        if len(self._arrays) == 0:
            self._shape = ()
            self._size = 1
            self._nd = 0
            return

        # Compute the broadcasted shape using _broadcast_shapes
        self._shape = _broadcast_shapes(*self._arrays)

        # Calculate size and ndim
        self._size = 1
        for dim in self._shape:
            self._size *= dim
        self._nd = len(self._shape)

    @property
    def shape(self):
        """
        Shape of the broadcasted result.

        Returns
        -------
        out : tuple
            A tuple containing the shape of the broadcasted result.

        """
        return self._shape

    @property
    def size(self):
        """
        Total size of the broadcasted result.

        Returns
        -------
        out : int
            The total size (number of elements) of the broadcasted result.

        """
        return self._size

    @property
    def nd(self):
        """
        Number of dimensions of the broadcasted result.

        Returns
        -------
        out : int
            The number of dimensions of the broadcasted result.

        """
        return self._nd

    @property
    def ndim(self):
        """
        Number of dimensions of the broadcasted result.

        Returns
        -------
        out : int
            The number of dimensions of the broadcasted result.

        """
        return self._nd

    @property
    def numiter(self):
        """
        Number of iterators possessed by the broadcast object.

        Returns
        -------
        out : int
            The number of iterators.

        """
        return len(self._arrays)

    def __repr__(self):
        return f"<broadcast shape={self.shape}, nd={self.nd}, size={self.size}>"
