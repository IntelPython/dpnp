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

"""
Bit generators of the device random number generation API.

A bit generator here keeps no state on the device. It holds its state on
the host, and every kernel launch rebuilds its engines on the device from
it, so sampling never waits for the device. Each launch advances the state
past the part of the stream it takes up front, so a bit generator can be
used from several threads, and launches never share engine output.

"""

# pylint: disable=no-name-in-module
# pylint: disable=protected-access

import math
import operator
import threading

import dpctl.utils as dpu
import numpy

import dpnp
import dpnp.libtensor._rng_impl as ri

__all__ = [
    "BitGenerator",
    "Philox4x3210",
    "SeedSequence",
]

# NumPy's SeedSequence is pure host-side code, so it is reused as is
SeedSequence = numpy.random.SeedSequence

_ISeedSequence = numpy.random.bit_generator.ISeedSequence

_MASK64 = (1 << 64) - 1
_MASK128 = (1 << 128) - 1


def _size_dim(s):
    # booleans have __index__, but NumPy does not take them as dimensions
    if isinstance(s, (bool, numpy.bool_)):
        raise TypeError("an integer is required")
    return operator.index(s)


def _normalize_size(size):
    """`size` as a shape, with the checks and the messages of NumPy."""

    if size is None:
        return None
    if numpy.ndim(size) == 0:
        if isinstance(size, (bool, numpy.bool_)) or (
            getattr(size, "dtype", None) == numpy.bool_
        ):
            index = None
        else:
            try:
                index = operator.index(size)
            except TypeError:
                index = None
        if index is None:
            raise TypeError(
                "expected a sequence of integers or a single integer, "
                f"got '{size!r}'"
            )
        shape = (index,)
    else:
        shape = tuple(_size_dim(s) for s in size)
    if any(s < 0 for s in shape):
        raise ValueError("negative dimensions are not allowed")
    return shape


class BitGenerator:
    """
    Base class for the bit generators of :class:`dpnp.random.Generator`.

    A bit generator is bound to one SYCL queue, and every array it produces
    is allocated on that queue.

    Parameters
    ----------
    seed : {None, int, array_like[ints], SeedSequence}, optional
        The entropy the initial state is derived from. With ``None``, fresh
        entropy is pulled from the operating system. An integer or a
        sequence of integers is passed to :class:`numpy.random.SeedSequence`
        to derive the state.

        Default: ``None``.
    device : {None, string, SyclDevice, SyclQueue, Device}, optional
        An array API concept of device where arrays are created.
        `device` can be ``None``, a oneAPI filter selector string, an instance
        of :class:`dpctl.SyclDevice` corresponding to a non-partitioned SYCL
        device, an instance of :class:`dpctl.SyclQueue`, or a
        :class:`dpnp.tensor.Device` object returned by
        :attr:`dpnp.ndarray.device`.

        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for arrays created by the bit generator.

        Default: ``None``.

    """

    # The engine_id of the extension, set by every subclass
    _engine = None

    def __init__(self, seed=None, *, device=None, sycl_queue=None):
        if type(self) is BitGenerator:  # pylint: disable=unidiomatic-typecheck
            raise TypeError(
                "BitGenerator is a base class and cannot be instantiated"
            )

        self._sycl_queue = dpnp.get_normalized_queue_device(
            device=device, sycl_queue=sycl_queue
        )
        # reentrant, so that `state` can be used while holding `lock`
        self._lock = threading.RLock()

        if isinstance(seed, _ISeedSequence):
            self._seed_seq = seed
        else:
            if dpnp.is_supported_array_type(seed):
                seed = dpnp.asnumpy(seed)
            self._seed_seq = SeedSequence(seed)

    def __repr__(self):
        return f"{self.__class__.__name__} at 0x{id(self):X}"

    def __str__(self):
        return self.__class__.__name__

    @property
    def lock(self):
        """
        The lock guarding the state of the bit generator.

        Hold it to read or set :attr:`state` consistently while other threads
        use the bit generator. The lock is reentrant, so :attr:`state` can be
        read and set while holding it.

        """
        return self._lock

    @property
    def seed_seq(self):
        """
        The seed sequence used to initialize the bit generator, or ``None``
        when the state was given directly, such as a Philox key.

        """
        return self._seed_seq

    @property
    def sycl_queue(self):
        """The SYCL queue arrays are created on."""
        return self._sycl_queue

    @property
    def sycl_device(self):
        """The SYCL device arrays are created on."""
        return self._sycl_queue.sycl_device

    @property
    def state(self):
        """The state of the bit generator, as a dictionary."""
        raise NotImplementedError

    def _num_work_items(self, size):
        """The work-items of a strided launch producing `size` values."""
        return -(-size // ri.items_per_work_item[self._engine])

    def _launch(self, kernel, dtype, size):
        """
        Take the part of the stream a launch of `kernel` producing `size`
        values of `dtype` draws from, and advance the state past it.

        Returns the seed words and the low and the high 64-bit words of the
        counter the extension takes.

        """
        raise NotImplementedError

    def _fill(self, kernel, params, result):
        """
        Fill the C-contiguous array `result` by `kernel` of the extension,
        with the distribution parameters `params`.

        Each parameter is either a Python scalar or a C-contiguous array of
        the size of `result` and of the data type the kernel expects for it.

        """

        result_usm = dpnp.get_usm_ndarray(result)
        params = [
            dpnp.get_usm_ndarray(p) if isinstance(p, dpnp.ndarray) else p
            for p in params
        ]

        seed_words, counter_lo, counter_hi = self._launch(
            kernel, result_usm.dtype, result_usm.size
        )

        exec_q = self._sycl_queue
        _manager = dpu.SequentialOrderManager[exec_q]
        ht_ev, ev = kernel(
            exec_q,
            int(self._engine),
            seed_words,
            counter_lo,
            counter_hi,
            params,
            result_usm,
            depends=_manager.submitted_events,
        )
        _manager.add_event_pair(ht_ev, ev)
        return result

    def random_raw(self, size=None, output=True, *, usm_type=None):
        """
        Return raw 64-bit outputs of the bit generator.

        Parameters
        ----------
        size : {None, int, tuple of ints}, optional
            Output shape. With ``None``, a zero-dimensional array is
            returned.

            Default: ``None``.
        output : bool, optional
            If ``False``, the state is advanced as if the values had been
            drawn, and ``None`` is returned.

            Default: ``True``.
        usm_type : {None, "device", "shared", "host"}, optional
            The type of SYCL USM allocation for the output array.

            Default: ``None``, which means ``"device"``.

        Returns
        -------
        out : {None, dpnp.ndarray}
            An array of ``uint64`` values.

        Limitations
        -----------
        Unlike NumPy, a zero-dimensional array is returned rather than a
        Python integer when `size` is ``None``. With ``output=False``, the
        state advances by the number of elements of `size`, and an invalid
        `size` raises as it does with ``output=True``. NumPy advances by
        the sum of the dimensions of `size` instead, and ignores a negative
        one.

        """

        shape = _normalize_size(size) or ()
        if not output:
            self._launch(
                ri._random_raw, numpy.dtype(numpy.uint64), math.prod(shape)
            )
            return None

        result = dpnp.empty(
            shape,
            dtype=dpnp.uint64,
            usm_type="device" if usm_type is None else usm_type,
            sycl_queue=self._sycl_queue,
        )
        return self._fill(ri._random_raw, (), result)

    def spawn(self, n_children):
        """
        Create new independent child bit generators on the same SYCL queue.

        Children are seeded by :meth:`numpy.random.SeedSequence.spawn` of
        :attr:`seed_seq`.

        Parameters
        ----------
        n_children : int
            The number of children to create.

        Returns
        -------
        out : list of BitGenerator
            The child bit generators.

        """

        if not hasattr(self._seed_seq, "spawn"):
            raise TypeError(
                "The underlying SeedSequence does not implement spawning."
            )
        return [
            type(self)(child, sycl_queue=self._sycl_queue)
            for child in self._seed_seq.spawn(n_children)
        ]


class Philox4x3210(BitGenerator):
    """
    A Philox4x32-10 counter-based bit generator on the device.

    The generator of Salmon et al. (2011), with a 64-bit key and a 128-bit
    counter, as implemented by the oneMKL device RNG API. It is not the same
    as :class:`numpy.random.Philox`, which is Philox4x64-10.

    Parameters
    ----------
    seed : {None, int, array_like[ints], SeedSequence}, optional
        The entropy the key is derived from, unless `key` is given. See
        :class:`dpnp.random.BitGenerator`.

        Default: ``None``.
    counter : {None, int}, optional
        The initial 128-bit counter.

        Default: ``None``, which starts from 0.
    key : {None, int}, optional
        The 64-bit key. Mutually exclusive with `seed`.

        Default: ``None``, which derives the key from `seed`.
    device : {None, string, SyclDevice, SyclQueue, Device}, optional
        See :class:`dpnp.random.BitGenerator`.

        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        See :class:`dpnp.random.BitGenerator`.

        Default: ``None``.

    Notes
    -----
    Every call that produces ``n`` values takes ``ceil(n / 8)`` blocks of
    2**64 counter values, one for each work-item of the kernel, and every
    work-item draws as many values as its rejection samplers need from its
    own block. Element ``i`` comes from the block ``i % ceil(n / 8)``, so the
    values depend on the output size: a draw is not a prefix of a larger
    one. For a given state, output size and data type, the raw bits and the
    uniform floats are the same on every device, while the samplers built on
    transcendental functions agree to a few ULP, or differ where a rejection
    sampler then takes another candidate.

    """

    _engine = ri.engine_id.philox4x32x10

    def __init__(
        self, seed=None, *, counter=None, key=None, device=None, sycl_queue=None
    ):
        if seed is not None and key is not None:
            raise ValueError("seed and key cannot be both used")

        super().__init__(seed, device=device, sycl_queue=sycl_queue)

        if key is None:
            key = int(self._seed_seq.generate_state(1, numpy.uint64)[0])
        else:
            # as in NumPy, there is no seed sequence to spawn from
            self._seed_seq = None
        self._key = self._check_word(key, _MASK64, "key")
        self._counter = self._check_word(
            0 if counter is None else counter, _MASK128, "counter"
        )

    @staticmethod
    def _check_word(value, mask, name):
        value = int(value)
        if value < 0 or value > mask:
            raise ValueError(
                f"{name} must be in the range [0, {mask}], got {value}"
            )
        return value

    @property
    def state(self):
        """
        The state of the bit generator.

        A dictionary ``{"bit_generator": "Philox4x3210", "state": {"key":
        key, "counter": counter}}`` with the 64-bit key and the 128-bit
        counter as Python integers.

        """

        with self._lock:
            return {
                "bit_generator": self.__class__.__name__,
                "state": {"key": self._key, "counter": self._counter},
            }

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError("state must be a dict")

        bitgen = value.get("bit_generator", None)
        if bitgen != self.__class__.__name__:
            raise ValueError(
                f"state must be for a {self.__class__.__name__} PRNG"
            )

        state = value["state"]
        key = self._check_word(state["key"], _MASK64, "key")
        counter = self._check_word(state["counter"], _MASK128, "counter")
        with self._lock:
            self._key = key
            self._counter = counter

    def _launch(self, kernel, dtype, size):
        # every work-item takes a block of 2**64 counter values
        n_blocks = self._num_work_items(size)
        with self._lock:
            key = self._key
            counter = self._counter
            self._counter = (counter + (n_blocks << 64)) & _MASK128
        return [key], counter & _MASK64, counter >> 64
