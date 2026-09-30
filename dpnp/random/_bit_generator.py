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

import operator
import threading

import dpctl.utils as dpu
import numpy

import dpnp
import dpnp.libtensor._rng_impl as ri

__all__ = ["BitGenerator", "PCG64DXSM", "Philox4x3210", "SeedSequence"]

# NumPy's SeedSequence is pure host-side code, so it is reused as is
SeedSequence = numpy.random.SeedSequence

_ISeedSequence = numpy.random.bit_generator.ISeedSequence

_MASK64 = (1 << 64) - 1
_MASK128 = (1 << 128) - 1


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
        :class:`dpctl.tensor.Device` object returned by
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
        self._lock = threading.Lock()

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
        use the bit generator.

        """
        return self._lock

    @property
    def seed_seq(self):
        """The seed sequence used to initialize the bit generator."""
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

    def _launch(self, kernel, dtype, size):
        """
        Take the part of the stream a launch of `kernel` producing `size`
        values of `dtype` draws from, and advance the state past it.

        Returns the seed words and the low and the high 64-bit words of the
        counter the extension takes.

        """
        raise NotImplementedError

    def _fill(self, kernel, params, result, **kwargs):
        """
        Fill the C-contiguous array `result` by `kernel` of the extension,
        with the distribution parameters `params`.

        Each parameter is either a Python scalar or a C-contiguous array of
        the size of `result` and of the data type the kernel expects for it.
        Keyword arguments are passed to `kernel` as is.

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
            **kwargs,
        )
        _manager.add_event_pair(ht_ev, ev)
        return result

    def random_raw(self, size=None, output=True, *, usm_type="device"):
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
        usm_type : {"device", "shared", "host"}, optional
            The type of SYCL USM allocation for the output array.

            Default: ``"device"``.

        Returns
        -------
        out : {None, dpnp.ndarray}
            An array of ``uint64`` values.

        Limitations
        -----------
        Unlike NumPy, a zero-dimensional array is returned rather than a
        Python integer when `size` is ``None``.

        """

        shape = () if size is None else size
        if not output:
            n = numpy.prod(shape, dtype=numpy.int64)
            self._launch(ri._random_raw, numpy.dtype(numpy.uint64), int(n))
            return None

        result = dpnp.empty(
            shape,
            dtype=dpnp.uint64,
            usm_type=usm_type,
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
    own block. The values therefore depend only on the state and the output
    size, and are the same on every device.

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
        n_blocks = -(-size // ri.items_per_work_item)
        with self._lock:
            counter = self._counter
            self._counter = (counter + (n_blocks << 64)) & _MASK128
        return [self._key], counter & _MASK64, counter >> 64


# NumPy's PCG64DXSM: the multiplier of its LCG, and the step of jumped(),
# (phi - 1) * 2**128 made odd
_PCG_MULT = 0xDA942042E4DD58B5
_PCG_JUMP = 0x9E3779B97F4A7C15F39CC0605CEDC835

# The kernels that take a fixed number of 32-bit halves of raw outputs for
# every value of a data type, and so draw the stream of PCG64DXSM in order,
# as NumPy does. It must match halves_per_item of the samplers.
_PCG_SEQUENTIAL = {
    ri._random_raw: {
        numpy.dtype(numpy.uint64): 2,
        numpy.dtype(numpy.uint32): 1,
    },
    ri._uniform: {numpy.dtype(numpy.float64): 2, numpy.dtype(numpy.float32): 1},
}


def _pcg_power(inc, k):
    """The step of the LCG applied `k` times, as a map (mult, inc)."""

    acc_mult, acc_inc = 1, 0
    mult = _PCG_MULT
    while k:
        if k & 1:
            acc_mult = (acc_mult * mult) & _MASK128
            acc_inc = (acc_inc * mult + inc) & _MASK128
        inc = ((mult + 1) * inc) & _MASK128
        mult = (mult * mult) & _MASK128
        k >>= 1
    return acc_mult, acc_inc


def _pcg_advance(state, inc, k):
    """The state of the LCG after `k` steps."""

    mult, add = _pcg_power(inc, k & _MASK128)
    return (mult * state + add) & _MASK128


def _pcg_output(state):
    """The output of PCG64DXSM for the state before a step."""

    hi = state >> 64
    lo = (state & _MASK64) | 1
    hi ^= hi >> 32
    hi = (hi * _PCG_MULT) & _MASK64
    hi ^= hi >> 48
    return (hi * lo) & _MASK64


class PCG64DXSM(BitGenerator):
    """
    NumPy's PCG64DXSM bit generator on the device.

    The permuted congruential generator of :class:`numpy.random.PCG64DXSM`,
    a 128-bit LCG with the DXSM output function. It is seeded and has its
    state as NumPy's, so the two can exchange states.

    Parameters
    ----------
    seed : {None, int, array_like[ints], SeedSequence}, optional
        The entropy the state is derived from, as by
        :class:`numpy.random.PCG64DXSM`. See
        :class:`dpnp.random.BitGenerator`.

        Default: ``None``.
    device : {None, string, SyclDevice, SyclQueue, Device}, optional
        See :class:`dpnp.random.BitGenerator`.

        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        See :class:`dpnp.random.BitGenerator`.

        Default: ``None``.

    Notes
    -----
    The raw outputs of :meth:`random_raw`, the floating-point values of
    :meth:`dpnp.random.Generator.random` and
    :meth:`dpnp.random.Generator.uniform`, and :meth:`dpnp.random.Generator.
    bytes` take a fixed number of engine outputs for every value. They are
    drawn in the order of the stream, and are the same as NumPy's for the
    same state, where NumPy draws them with the same algorithm.

    Every other method is sampled as by :meth:`jumped`: each work-item of a
    call producing ``n`` values draws as many values as its rejection
    samplers need from its own stream, the ``w``-th of which is that of
    ``jumped(w)``, and the call advances the state as ``jumped(ceil(n /
    8))``, but keeps the 32-bit half NumPy buffers. The values therefore
    depend only on the state and the output size, and are the same on every
    device, but not the same as NumPy's.

    """

    _engine = ri.engine_id.pcg64dxsm

    def __init__(self, seed=None, *, device=None, sycl_queue=None):
        super().__init__(seed, device=device, sycl_queue=sycl_queue)

        state = numpy.random.PCG64DXSM(self._seed_seq).state
        self._state = state["state"]["state"]
        self._inc = state["state"]["inc"]
        self._has_uint32 = 0
        self._uinteger = 0
        self._jump = _pcg_power(self._inc, _PCG_JUMP)

    def _set_state(self, state, inc, has_uint32, uinteger):
        self._state = state
        self._inc = inc
        self._has_uint32 = has_uint32
        self._uinteger = uinteger
        self._jump = _pcg_power(inc, _PCG_JUMP)

    @property
    def state(self):
        """
        The state of the bit generator.

        A dictionary ``{"bit_generator": "PCG64DXSM", "state": {"state":
        state, "inc": inc}, "has_uint32": has_uint32, "uinteger":
        uinteger}``, as that of :class:`numpy.random.PCG64DXSM`.

        """

        with self._lock:
            return {
                "bit_generator": self.__class__.__name__,
                "state": {"state": self._state, "inc": self._inc},
                "has_uint32": self._has_uint32,
                "uinteger": self._uinteger,
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

        state = int(value["state"]["state"]) & _MASK128
        inc = int(value["state"]["inc"]) & _MASK128
        has_uint32 = int(value["has_uint32"])
        uinteger = int(value["uinteger"]) & 0xFFFFFFFF
        with self._lock:
            self._set_state(state, inc, has_uint32, uinteger)

    def advance(self, delta):
        """
        Advance the state as if `delta` raw values had been drawn.

        For full documentation refer to
        :obj:`numpy.random.PCG64DXSM.advance`.

        """

        delta = operator.index(delta)
        with self._lock:
            self._state = _pcg_advance(self._state, self._inc, delta)
            self._has_uint32 = 0
            self._uinteger = 0
        return self

    def jumped(self, jumps=1):
        """
        Return a new bit generator with the state jumped ahead.

        For full documentation refer to
        :obj:`numpy.random.PCG64DXSM.jumped`.

        """

        bit_generator = self.__class__(
            self._seed_seq, sycl_queue=self._sycl_queue
        )
        bit_generator.state = self.state
        bit_generator.advance(_PCG_JUMP * operator.index(jumps))
        return bit_generator

    def _words(self):
        return [
            self._state & _MASK64,
            self._state >> 64,
            self._inc & _MASK64,
            self._inc >> 64,
            self._jump[0] & _MASK64,
            self._jump[0] >> 64,
            self._jump[1] & _MASK64,
            self._jump[1] >> 64,
            self._has_uint32,
            self._uinteger,
        ]

    def _launch(self, kernel, dtype, size):
        halves = _PCG_SEQUENTIAL.get(kernel, {}).get(dtype, 0)
        with self._lock:
            words = self._words()
            state, inc = self._state, self._inc
            if halves == 2:
                state = _pcg_advance(state, inc, size)
            elif halves == 1:
                if size and self._has_uint32:
                    size -= 1
                    self._has_uint32 = 0
                raws = -(-size // 2)
                if raws:
                    # as NumPy, keep the high half of the last output, which
                    # the next 32-bit draw takes if `size` is odd
                    state = _pcg_advance(state, inc, raws - 1)
                    self._uinteger = _pcg_output(state) >> 32
                    self._has_uint32 = size % 2
                    state = _pcg_advance(state, inc, 1)
            else:
                # work-item w draws from the stream jumped w times
                n_wi = -(-size // ri.items_per_work_item)
                state = _pcg_advance(state, inc, _PCG_JUMP * n_wi)
            self._state = state
        return words, 0, 0
