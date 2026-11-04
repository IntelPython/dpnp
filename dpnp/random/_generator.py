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
The device random number generation API modelled on
:class:`numpy.random.Generator`.

Every distribution is sampled on the device by a port of NumPy's algorithm,
driven by a oneMKL device engine. The values differ from NumPy's.

"""

# pylint: disable=no-name-in-module
# pylint: disable=protected-access

import operator

import numpy

import dpnp
import dpnp.libtensor._rng_impl as ri
import dpnp.tensor as dpt
from dpnp.exceptions import ExecutionPlacementError
from dpnp.tensor import validate_usm_type

from ._bit_generator import BitGenerator, Philox4x3210

__all__ = ["Generator", "default_rng"]

_float_types = (numpy.dtype(numpy.float32), numpy.dtype(numpy.float64))
# The parameter constraints of NumPy's distributions, see
# numpy/random/_common.pyx
_CONS_NONE = 0
_CONS_NON_NEGATIVE = 1


def _is_array(x):
    return dpnp.is_supported_array_type(x) or numpy.ndim(x) > 0


def _is_scalar_path(*params):
    """
    Whether NumPy takes the path of scalar parameters, whose error messages
    differ slightly from the ones of array parameters.

    """
    return all(numpy.ndim(p) == 0 for p in params)


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


def _check_real(value):
    """
    Raise TypeError, as NumPy does on the path of scalar parameters, unless
    the host scalar `value` converts to a Python float.

    """

    t = type(value)
    if not (hasattr(t, "__float__") or hasattr(t, "__index__")):
        raise TypeError(f"must be real number, not {t.__name__}")


def _uniform_range(low, high, overflow_msg):
    """
    ``high - low``, raising OverflowError with `overflow_msg` unless it is
    finite, or ``None`` when either param is on the device.

    """

    xp, (flow, fhigh) = _check_arrays((low, high), numpy.float64)
    if xp is None:
        return None
    with numpy.errstate(over="ignore", invalid="ignore"):
        rng = numpy.subtract(fhigh, flow)
    if _is_scalar_path(low, high):
        _check_real(low)
        _check_real(high)
    if not numpy.all(numpy.isfinite(rng)):
        raise OverflowError(overflow_msg)
    return rng


def _check_arrays(params, dtype=None):
    """
    `params` as arrays of `dtype` to check, with the module to check them
    by, which is ``None`` when some are on the device, as the check would
    wait for the device.

    """

    if any(dpnp.is_supported_array_type(p) for p in params):
        return None, [None] * len(params)
    return numpy, [numpy.asarray(p, dtype) for p in params]


def _check_constraint(value, name, cons, scalar):
    """
    Raise ValueError, as NumPy does, when the parameter `value` violates the
    constraint `cons`.

    Parameters on the device are not checked, as the check waits for the
    device. `scalar` selects the messages of NumPy's scalar path.

    """

    if dpnp.is_supported_array_type(value):
        return
    val = numpy.asarray(value, dtype=numpy.float64)
    if scalar:
        _check_real(value)
    if cons == _CONS_NONE:
        return

    if cons == _CONS_NON_NEGATIVE:
        if numpy.any(~numpy.isnan(val) & numpy.signbit(val)):
            raise ValueError(f"{name} < 0")


class Generator:
    """
    A container for the bit generator and the distribution samplers.

    The analogue of :class:`numpy.random.Generator`, drawing all samples on
    the SYCL queue of its bit generator. Use :func:`dpnp.random.default_rng`
    to create one.

    For full documentation refer to :obj:`numpy.random.Generator`.

    Parameters
    ----------
    bit_generator : BitGenerator
        The bit generator, such as :class:`dpnp.random.Philox4x3210`.

    Limitations
    -----------
    Every method returns a :class:`dpnp.ndarray`, zero-dimensional when
    `size` is ``None`` and the parameters are scalars, rather than a Python
    scalar. The methods take a `usm_type` keyword, and the floating-point
    ones a `dtype` keyword, as in CuPy, whose default is the default
    floating-point type of the device.

    Parameters on the host (scalars, lists and NumPy arrays) are checked as
    in NumPy. Parameters on the device are not, since that would wait for
    the device. Invalid ones give undefined values, such as NaN, but never
    make the kernels loop without end or access memory out of bounds.

    """

    def __init__(self, bit_generator):
        if not isinstance(bit_generator, BitGenerator):
            raise TypeError(
                "bit_generator must be an instance of dpnp.random.BitGenerator"
            )
        self._bit_generator = bit_generator

    def __repr__(self):
        return self.__str__() + f" at 0x{id(self):X}"

    def __str__(self):
        return f"{self.__class__.__name__}({self._bit_generator})"

    @property
    def bit_generator(self):
        """
        The bit generator used by the generator.

        Unlike NumPy, and as in CuPy, it can be replaced.

        """
        return self._bit_generator

    @bit_generator.setter
    def bit_generator(self, value):
        if not isinstance(value, BitGenerator):
            raise TypeError(
                "bit_generator must be an instance of dpnp.random.BitGenerator"
            )
        self._bit_generator = value

    @property
    def sycl_queue(self):
        """The SYCL queue samples are drawn on."""
        return self._bit_generator.sycl_queue

    @property
    def sycl_device(self):
        """The SYCL device samples are drawn on."""
        return self._bit_generator.sycl_device

    def _check_queues(self, *values):
        """
        Raise ExecutionPlacementError unless every array of `values` is on
        the queue of the generator.

        """

        queues = [
            x.sycl_queue for x in values if dpnp.is_supported_array_type(x)
        ]
        if (
            queues
            and dpt.get_execution_queue([self.sycl_queue] + queues) is None
        ):
            raise ExecutionPlacementError(
                "Input arrays and the generator have incompatible allocation "
                "queues"
            )

    def spawn(self, n_children):
        """
        Create new independent child generators on the same SYCL queue.

        For full documentation refer to :obj:`numpy.random.Generator.spawn`.

        """
        return [type(self)(bg) for bg in self._bit_generator.spawn(n_children)]

    def _float_dtype(self, dtype, name):
        if dtype is None:
            return numpy.dtype(
                dpnp.default_float_type(sycl_queue=self.sycl_queue)
            )

        dtype = numpy.dtype(dtype)
        if dtype not in _float_types:
            raise TypeError(f"Unsupported dtype {dtype!r} for {name}")
        return dtype

    def _prepare_out(self, out, size, dtype, name):
        """Resolve `size` and `dtype` against a user-supplied `out`."""

        if not isinstance(out, dpnp.ndarray):
            raise TypeError(
                f"out must be a dpnp.ndarray, got {type(out)} for {name}"
            )
        # as in NumPy, the default dtype applies even when `out` is given
        dtype = self._float_dtype(dtype, name)
        if out.dtype != dtype:
            raise TypeError(
                "Supplied output array has the wrong type. "
                f"Expected {dtype}, got {out.dtype}"
            )
        if not (out.flags.c_contiguous and out.flags.writable):
            raise ValueError(
                "Supplied output array is not C-contiguous or writable."
            )
        if size is not None and _normalize_size(size) != out.shape:
            raise ValueError("size must match out.shape when used together")
        return out.shape, dtype

    def _sample(
        self,
        kernel,
        params,
        size,
        dtype,
        usm_type,
        *,
        out=None,
        kernel_params=None,
    ):
        """
        Draw samples by `kernel` of the extension into a new array, or into
        `out`.

        `params` are the user's distribution parameters, which set the output
        shape along with `size`. `kernel_params` are the parameters passed to
        the kernel, `params` by default, of data type `dtype`. Every kernel
        parameter that is an array is broadcast to the output shape.

        """

        self._check_queues(*params, *(kernel_params or ()), out)
        queue = self.sycl_queue

        param_shape = numpy.broadcast_shapes(
            *(numpy.shape(p) if _is_array(p) else () for p in params)
        )
        shape = _normalize_size(size)
        if shape is None:
            shape = param_shape
        else:
            try:
                compatible = numpy.broadcast_shapes(shape, param_shape) == shape
            except ValueError:
                compatible = False
            if not compatible:
                raise ValueError(
                    f"Output size {shape} is not compatible with broadcast "
                    f"dimensions of inputs {param_shape}."
                )

        if kernel_params is None:
            kernel_params = params

        def _to_kernel_param(p):
            if _is_array(p):
                if not dpnp.is_supported_array_type(p):
                    # NumPy converts None to NaN, dpnp.asarray does not
                    p = numpy.asarray(p, dtype=dtype)
                p = dpnp.asarray(p, dtype=dtype, sycl_queue=queue)
                return dpnp.ascontiguousarray(dpnp.broadcast_to(p, shape))
            # as in NumPy, None is NaN along with array parameters
            return float(numpy.asarray(p, dtype=numpy.float64))

        kernel_params = tuple(_to_kernel_param(p) for p in kernel_params)

        if out is None:
            out = dpnp.empty(
                shape, dtype=dtype, usm_type=usm_type, sycl_queue=queue
            )
        return self._bit_generator._fill(kernel, kernel_params, out)

    def _float_sample(
        self,
        kernel,
        params,
        size,
        dtype,
        usm_type,
        name,
        *,
        out=None,
        kernel_params=None,
    ):
        validate_usm_type(usm_type, allow_none=False)
        if out is not None:
            size, dtype = self._prepare_out(out, size, dtype, name)
        else:
            dtype = self._float_dtype(dtype, name)
        return self._sample(
            kernel,
            params,
            size,
            dtype,
            usm_type,
            out=out,
            kernel_params=kernel_params,
        )

    def _cont(self, kernel, name, size, dtype, usm_type, *params, out=None):
        """
        Draw samples of a continuous distribution after checking `params`,
        triples of the value, the name and the constraint of a parameter.

        """

        values = tuple(p[0] for p in params)
        self._check_queues(*values, out)
        scalar = _is_scalar_path(*values)
        for value, pname, cons in params:
            _check_constraint(value, pname, cons, scalar)
        return self._float_sample(
            kernel, values, size, dtype, usm_type, name, out=out
        )

    def random(self, size=None, dtype=None, out=None, *, usm_type="device"):
        """
        Return random floats in the half-open interval [0.0, 1.0).

        For full documentation refer to :obj:`numpy.random.Generator.random`.

        Limitations
        -----------
        `dtype` defaults to the default floating-point type of the device.
        `usm_type` sets the type of USM allocation of the output.

        """
        return self._float_sample(
            ri._uniform,
            (),
            size,
            dtype,
            usm_type,
            "random",
            out=out,
            kernel_params=(0.0, 1.0),
        )

    def uniform(
        self, low=0.0, high=1.0, size=None, dtype=None, *, usm_type="device"
    ):
        """
        Draw samples from a uniform distribution over [low, high).

        For full documentation refer to :obj:`numpy.random.Generator.uniform`.

        Limitations
        -----------
        `dtype` and `usm_type` are not in NumPy's signature.

        """

        self._check_queues(low, high)
        scalar = _is_scalar_path(low, high)
        rng = _uniform_range(
            low,
            high,
            (
                "high - low range exceeds valid bounds"
                if scalar
                else "Range exceeds valid bounds"
            ),
        )
        if rng is not None:
            _check_constraint(
                rng,
                "high - low",
                _CONS_NON_NEGATIVE,
                scalar,
            )
        return self._float_sample(
            ri._uniform, (low, high), size, dtype, usm_type, "uniform"
        )

    def standard_normal(
        self, size=None, dtype=None, out=None, *, usm_type="device"
    ):
        """
        Draw samples from a standard normal distribution.

        For full documentation refer to
        :obj:`numpy.random.Generator.standard_normal`.

        Limitations
        -----------
        The Box-Muller transform is used rather than NumPy's ziggurat.

        """
        return self._float_sample(
            ri._normal,
            (),
            size,
            dtype,
            usm_type,
            "standard_normal",
            out=out,
            kernel_params=(0.0, 1.0),
        )

    def normal(
        self, loc=0.0, scale=1.0, size=None, dtype=None, *, usm_type="device"
    ):
        """
        Draw samples from a normal (Gaussian) distribution.

        For full documentation refer to :obj:`numpy.random.Generator.normal`.

        Limitations
        -----------
        `dtype` and `usm_type` are not in NumPy's signature.

        """
        return self._cont(
            ri._normal,
            "normal",
            size,
            dtype,
            usm_type,
            (loc, "loc", _CONS_NONE),
            (scale, "scale", _CONS_NON_NEGATIVE),
        )


def default_rng(seed=None, *, device=None, sycl_queue=None):
    """
    Construct a new :class:`dpnp.random.Generator` with the default bit
    generator, :class:`dpnp.random.Philox4x3210`.

    For full documentation refer to :obj:`numpy.random.default_rng`.

    Parameters
    ----------
    seed : {None, int, array_like[ints], SeedSequence, BitGenerator, \
            Generator}, optional
        The entropy to seed the bit generator with. A
        :class:`dpnp.random.BitGenerator` is wrapped in a new generator, and
        a :class:`dpnp.random.Generator` is returned unaltered.

        Default: ``None``.
    device : {None, string, SyclDevice, SyclQueue, Device}, optional
        An array API concept of device where the samples are drawn and
        allocated. Ignored when `seed` is a bit generator or a generator.

        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to draw samples on. Ignored when `seed` is a bit
        generator or a generator.

        Default: ``None``.

    Returns
    -------
    out : Generator
        The initialized generator.

    Limitations
    -----------
    NumPy's default bit generator is PCG64, so the same seed gives different
    values.

    Examples
    --------
    >>> import dpnp as np
    >>> rng = np.random.default_rng(12345)
    >>> x = rng.random(3)
    >>> x.shape, x.device == rng.bit_generator.sycl_queue.sycl_device
    ((3,), True)

    """

    if isinstance(seed, Generator):
        return seed
    if isinstance(seed, BitGenerator):
        return Generator(seed)
    return Generator(Philox4x3210(seed, device=device, sycl_queue=sycl_queue))
