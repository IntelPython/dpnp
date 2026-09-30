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
from dpnp.tensor import validate_usm_type

from ._bit_generator import BitGenerator, Philox4x3210

__all__ = ["Generator", "default_rng"]

_float_types = (numpy.dtype(numpy.float32), numpy.dtype(numpy.float64))
_integer_types = tuple(
    numpy.dtype(t)
    for t in (
        numpy.bool_,
        numpy.int8,
        numpy.uint8,
        numpy.int16,
        numpy.uint16,
        numpy.int32,
        numpy.uint32,
        numpy.int64,
        numpy.uint64,
    )
)

# NumPy's Generator methods not supported yet
_not_implemented = (
    "binomial",
    "choice",
    "dirichlet",
    "f",
    "geometric",
    "gumbel",
    "hypergeometric",
    "laplace",
    "logistic",
    "logseries",
    "multinomial",
    "multivariate_hypergeometric",
    "multivariate_normal",
    "negative_binomial",
    "noncentral_chisquare",
    "noncentral_f",
    "pareto",
    "permutation",
    "permuted",
    "poisson",
    "power",
    "rayleigh",
    "shuffle",
    "standard_cauchy",
    "standard_t",
    "triangular",
    "vonmises",
    "wald",
    "weibull",
    "zipf",
)


def _is_array(x):
    return dpnp.is_supported_array_type(x) or numpy.ndim(x) > 0


def _normalize_size(size):
    if size is None:
        return None
    try:
        return (operator.index(size),)
    except TypeError:
        return tuple(operator.index(s) for s in size)


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
    scalar. The floating-point methods take a `dtype` keyword, as in CuPy,
    whose default is the default floating-point type of the device.
    Parameters are checked on the host when they are scalars; array
    parameters are not checked, and invalid ones give undefined values.
    Methods not listed here raise :class:`NotImplementedError`.

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
        self, kernel, params, size, dtype, usm_type, *, out=None, a=None, b=None
    ):
        """
        Draw samples by `kernel` of the extension into a new array, or into
        `out`.

        `params` are the user's distribution parameters, which set the output
        shape along with `size`; `a` and `b` are the kernel parameters. Any
        of `a` and `b` that is an array is broadcast to the output shape.

        """

        queue = self.sycl_queue

        param_shape = numpy.broadcast_shapes(
            *(dpnp.shape(p) if _is_array(p) else () for p in params)
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

        def _to_kernel_param(p):
            if not _is_array(p):
                return p
            p = dpnp.asarray(p, dtype=dtype, sycl_queue=queue)
            return dpnp.ascontiguousarray(dpnp.broadcast_to(p, shape))

        a = _to_kernel_param(a)
        b = _to_kernel_param(b)

        if out is None:
            out = dpnp.empty(
                shape, dtype=dtype, usm_type=usm_type, sycl_queue=queue
            )
        return self._bit_generator._fill(kernel, a, b, out)

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
        a=None,
        b=None,
    ):
        validate_usm_type(usm_type, allow_none=False)
        if out is not None:
            size, dtype = self._prepare_out(out, size, dtype, name)
        else:
            dtype = self._float_dtype(dtype, name)
        return self._sample(
            kernel, params, size, dtype, usm_type, out=out, a=a, b=b
        )

    @staticmethod
    def _check_scalar(value, name, cond, msg=None):
        """
        Raise ValueError when `value` is a scalar and fails `cond`.

        Arrays are not checked.

        """

        if _is_array(value):
            return
        # NaN fails every condition, as in NumPy
        if not cond(value):
            raise ValueError(msg if msg else f"{name} < 0")

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
            a=0.0,
            b=1.0,
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

        if not (_is_array(low) or _is_array(high)):
            low, high = float(low), float(high)
            if not numpy.isfinite(high - low):
                raise OverflowError("high - low range exceeds valid bounds")
        return self._float_sample(
            ri._uniform,
            (low, high),
            size,
            dtype,
            usm_type,
            "uniform",
            a=low,
            b=high,
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
            a=0.0,
            b=1.0,
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

        self._check_scalar(scale, "scale", lambda x: x >= 0)
        return self._float_sample(
            ri._normal,
            (loc, scale),
            size,
            dtype,
            usm_type,
            "normal",
            a=loc,
            b=scale,
        )

    def lognormal(
        self, mean=0.0, sigma=1.0, size=None, dtype=None, *, usm_type="device"
    ):
        """
        Draw samples from a log-normal distribution.

        For full documentation refer to
        :obj:`numpy.random.Generator.lognormal`.

        Limitations
        -----------
        `dtype` and `usm_type` are not in NumPy's signature.

        """

        self._check_scalar(sigma, "sigma", lambda x: x >= 0)
        return self._float_sample(
            ri._lognormal,
            (mean, sigma),
            size,
            dtype,
            usm_type,
            "lognormal",
            a=mean,
            b=sigma,
        )

    def standard_exponential(
        self,
        size=None,
        dtype=None,
        method="zig",
        out=None,
        *,
        usm_type="device",
    ):
        """
        Draw samples from the standard exponential distribution.

        For full documentation refer to
        :obj:`numpy.random.Generator.standard_exponential`.

        Limitations
        -----------
        Both methods use inversion; ``"zig"`` is accepted for compatibility.

        """

        if method not in ("zig", "inv"):
            raise ValueError("method must be either 'inv' or 'zig'")
        return self._float_sample(
            ri._exponential,
            (),
            size,
            dtype,
            usm_type,
            "standard_exponential",
            out=out,
            a=1.0,
        )

    def exponential(
        self, scale=1.0, size=None, dtype=None, *, usm_type="device"
    ):
        """
        Draw samples from an exponential distribution.

        For full documentation refer to
        :obj:`numpy.random.Generator.exponential`.

        Limitations
        -----------
        `dtype` and `usm_type` are not in NumPy's signature.

        """

        self._check_scalar(scale, "scale", lambda x: x >= 0)
        return self._float_sample(
            ri._exponential,
            (scale,),
            size,
            dtype,
            usm_type,
            "exponential",
            a=scale,
        )

    def standard_gamma(
        self, shape, size=None, dtype=None, out=None, *, usm_type="device"
    ):
        """
        Draw samples from a standard gamma distribution.

        For full documentation refer to
        :obj:`numpy.random.Generator.standard_gamma`.

        Limitations
        -----------
        `usm_type` is not in NumPy's signature.

        """

        self._check_scalar(shape, "shape", lambda x: x >= 0)
        return self._float_sample(
            ri._gamma,
            (shape,),
            size,
            dtype,
            usm_type,
            "standard_gamma",
            out=out,
            a=shape,
            b=1.0,
        )

    def gamma(
        self, shape, scale=1.0, size=None, dtype=None, *, usm_type="device"
    ):
        """
        Draw samples from a gamma distribution.

        For full documentation refer to :obj:`numpy.random.Generator.gamma`.

        Limitations
        -----------
        `dtype` and `usm_type` are not in NumPy's signature.

        """

        self._check_scalar(shape, "shape", lambda x: x >= 0)
        self._check_scalar(scale, "scale", lambda x: x >= 0)
        return self._float_sample(
            ri._gamma,
            (shape, scale),
            size,
            dtype,
            usm_type,
            "gamma",
            a=shape,
            b=scale,
        )

    def chisquare(self, df, size=None, dtype=None, *, usm_type="device"):
        """
        Draw samples from a chi-square distribution.

        For full documentation refer to
        :obj:`numpy.random.Generator.chisquare`.

        Limitations
        -----------
        `dtype` and `usm_type` are not in NumPy's signature.

        """

        self._check_scalar(df, "df", lambda x: x > 0, "df <= 0")
        # chi-square(df) is gamma(df / 2, 2)
        if _is_array(df):
            half_df = (
                dpnp.asarray(
                    df,
                    dtype=self._float_dtype(dtype, "chisquare"),
                    sycl_queue=self.sycl_queue,
                )
                / 2
            )
        else:
            half_df = float(df) / 2
        return self._float_sample(
            ri._gamma,
            (df,),
            size,
            dtype,
            usm_type,
            "chisquare",
            a=half_df,
            b=2.0,
        )

    def beta(self, a, b, size=None, dtype=None, *, usm_type="device"):
        """
        Draw samples from a beta distribution.

        For full documentation refer to :obj:`numpy.random.Generator.beta`.

        Limitations
        -----------
        `dtype` and `usm_type` are not in NumPy's signature.

        """

        self._check_scalar(a, "a", lambda x: x > 0, "a <= 0")
        self._check_scalar(b, "b", lambda x: x > 0, "b <= 0")
        return self._float_sample(
            ri._beta, (a, b), size, dtype, usm_type, "beta", a=a, b=b
        )

    def integers(
        self,
        low,
        high=None,
        size=None,
        dtype=numpy.int64,
        endpoint=False,
        *,
        usm_type="device",
    ):
        """
        Return random integers from `low` (inclusive) to `high` (exclusive),
        or if ``endpoint=True``, `low` (inclusive) to `high` (inclusive).

        For full documentation refer to
        :obj:`numpy.random.Generator.integers`.

        Limitations
        -----------
        `usm_type` is not in NumPy's signature. The values drawn differ from
        NumPy's for the same range, as there is no buffering of 32-bit
        draws for the narrow types.

        """

        validate_usm_type(usm_type, allow_none=False)

        if high is None:
            low, high = 0, low

        dtype = numpy.dtype(dtype)
        if dtype not in _integer_types:
            raise TypeError(f"Unsupported dtype {dtype!r} for integers")

        # The bounds are passed to the kernel as a closed interval
        if dtype == numpy.bool_:
            lo_lim, hi_lim = 0, 1
        else:
            info = numpy.iinfo(dtype)
            lo_lim, hi_lim = int(info.min), int(info.max)

        if _is_array(low) or _is_array(high):
            # Widen to 64 bits with the sign of the result type, so that an
            # exclusive high of max + 1 survives the subtraction
            wide = (
                numpy.uint64
                if dtype.kind == "u" or dtype == numpy.bool_
                else numpy.int64
            )
            queue = self.sycl_queue
            params = (low, high)

            def _to_bound(x, shift):
                if _is_array(x):
                    x = dpnp.asarray(x, dtype=wide, sycl_queue=queue)
                    return (x - shift if shift else x).astype(dtype)
                return int(x) - shift

            low = _to_bound(low, 0)
            high = _to_bound(high, 0 if endpoint else 1)
            return self._sample(
                ri._integers, params, size, dtype, usm_type, a=low, b=high
            )

        low = int(low)
        high = int(high) if endpoint else int(high) - 1
        dt_name = "bool" if dtype == numpy.bool_ else str(dtype)
        if low < lo_lim:
            raise ValueError(f"low is out of bounds for {dt_name}")
        if high > hi_lim:
            raise ValueError(f"high is out of bounds for {dt_name}")
        if low > high:
            raise ValueError("low > high" if endpoint else "low >= high")

        if dtype == numpy.bool_:
            low, high = bool(low), bool(high)
        return self._sample(
            ri._integers, (), size, dtype, usm_type, a=low, b=high
        )

    def bytes(self, length):
        """
        Return random bytes.

        For full documentation refer to :obj:`numpy.random.Generator.bytes`.

        Limitations
        -----------
        The bytes are drawn on the device and copied to the host.

        """

        length = operator.index(length)
        values = self.integers(0, 256, size=length, dtype=numpy.uint8)
        return dpnp.asnumpy(values).tobytes()


def _make_not_implemented(name):
    def method(self, *args, **kwargs):
        raise NotImplementedError(
            f"dpnp.random.Generator.{name} is not supported yet"
        )

    method.__name__ = name
    method.__qualname__ = f"Generator.{name}"
    method.__doc__ = (
        f"Not supported yet; raises :class:`NotImplementedError`.\n\n"
        f"See :obj:`numpy.random.Generator.{name}`."
    )
    return method


for _name in _not_implemented:
    setattr(Generator, _name, _make_not_implemented(_name))
del _name


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
