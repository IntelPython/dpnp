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
# pylint: disable=too-many-lines

import operator
import warnings

import numpy

import dpnp
import dpnp.libtensor._rng_impl as ri
from dpnp.tensor import validate_usm_type
from dpnp.tensor._numpy_helper import normalize_axis_index

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
_int64 = numpy.dtype(numpy.int64)
_INT64_MAX = int(numpy.iinfo(numpy.int64).max)

# The parameter constraints of NumPy's distributions, see
# numpy/random/_common.pyx
_CONS_NONE = 0
_CONS_NON_NEGATIVE = 1
_CONS_POSITIVE = 2
_CONS_POSITIVE_NOT_NAN = 3
_CONS_BOUNDED_0_1 = 4
_CONS_BOUNDED_GT_0_1 = 5
_CONS_BOUNDED_LT_0_1 = 6
_CONS_GT_1 = 7
_CONS_POISSON = 8
_CONS_GTE_1 = 9

_POISSON_LAM_MAX = _INT64_MAX - numpy.sqrt(_INT64_MAX) * 10
_HYPERGEOM_MAX = 10**9

# the messages of Generator.choice
_CHOICE_MSGS = {
    "a_type": "a must be a sequence or an integer, not {}",
    "a_int": "a must be a positive integer unless no samples are taken",
    "a_1d": None,
    "a_empty": "a cannot be empty unless no samples are taken",
    "p_1d": "p must be 1-dimensional",
    "p_size": "a and p must have same size",
    "p_nan": "Probabilities contain NaN",
    "p_neg": "Probabilities are not non-negative",
    "p_sum": (
        "Probabilities do not sum to 1. See Notes section of docstring for "
        "more information."
    ),
    "too_large": (
        "Cannot take a larger sample than population when replace is False"
    ),
}


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
    ``high - low`` of host params, raising OverflowError with `overflow_msg`
    unless it is finite, or ``None`` when either param is on the device.

    """

    if dpnp.is_supported_array_type(low) or dpnp.is_supported_array_type(high):
        return None
    with numpy.errstate(over="ignore", invalid="ignore"):
        rng = numpy.subtract(
            numpy.asarray(high, dtype=numpy.float64),
            numpy.asarray(low, dtype=numpy.float64),
        )
    if _is_scalar_path(low, high):
        _check_real(low)
        _check_real(high)
    if not numpy.all(numpy.isfinite(rng)):
        raise OverflowError(overflow_msg)
    return rng


# pylint: disable=too-many-branches
def _check_constraint(value, name, cons, scalar):
    """
    Raise ValueError, as NumPy does, when the host parameter `value`
    violates the constraint `cons`.

    Parameters on the device are not checked, as that would wait for the
    device. `scalar` selects the messages of NumPy's scalar path.

    """

    if dpnp.is_supported_array_type(value):
        return

    val = numpy.asarray(value, dtype=numpy.float64)
    if scalar:
        _check_real(value)
    if cons == _CONS_NONE:
        return

    nan_msg = "is NaN" if scalar else "contains NaNs"
    if cons == _CONS_NON_NEGATIVE:
        if numpy.any(~numpy.isnan(val) & numpy.signbit(val)):
            raise ValueError(f"{name} < 0")
    elif cons in (_CONS_POSITIVE, _CONS_POSITIVE_NOT_NAN):
        if cons == _CONS_POSITIVE_NOT_NAN and numpy.any(numpy.isnan(val)):
            raise ValueError(f"{name} must not be NaN")
        if numpy.any(val <= 0):
            raise ValueError(f"{name} <= 0")
    elif cons == _CONS_BOUNDED_0_1:
        if not numpy.all((val >= 0) & (val <= 1)):
            raise ValueError(f"{name} < 0, {name} > 1 or {name} {nan_msg}")
    elif cons == _CONS_BOUNDED_GT_0_1:
        if not numpy.all((val > 0) & (val <= 1)):
            raise ValueError(f"{name} <= 0, {name} > 1 or {name} contains NaNs")
    elif cons == _CONS_BOUNDED_LT_0_1:
        if not numpy.all((val >= 0) & (val < 1)):
            raise ValueError(f"{name} < 0, {name} >= 1 or {name} {nan_msg}")
    elif cons == _CONS_GT_1:
        if not numpy.all(val > 1):
            raise ValueError(f"{name} <= 1 or {name} {nan_msg}")
    elif cons == _CONS_GTE_1:
        if not numpy.all(val >= 1):
            raise ValueError(f"{name} < 1 or {name} {nan_msg}")
    elif cons == _CONS_POISSON:
        # the array path checks the upper bound first, so NaN fails it
        if scalar:
            if not val >= 0:
                raise ValueError(f"{name} < 0 or {name} is NaN")
            if val > _POISSON_LAM_MAX:
                raise ValueError(f"{name} value too large")
        else:
            if not numpy.all(val <= _POISSON_LAM_MAX):
                raise ValueError(f"{name} value too large")
            if not numpy.all(val >= 0):
                raise ValueError(f"{name} < 0 or {name} contains NaNs")


def _as_int64(x):
    """Truncate an integer parameter to int64, as NumPy does."""

    if dpnp.is_supported_array_type(x):
        return dpnp.asarray(x).astype(_int64)
    if numpy.ndim(x) > 0:
        return numpy.asarray(x).astype(_int64)
    return int(x)


# pylint: disable=too-many-public-methods
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
    the device, and invalid ones give undefined values.

    The discrete distributions return ``int64`` values, and compute in
    double precision on devices supporting it, and in single precision
    otherwise. On the latter, the samples for very large parameters, such
    as a Poisson `lam` beyond about ``1e7``, are less accurate than NumPy's.

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

    def _discrete_float_type(self):
        """The floating-point type the discrete distributions compute in."""

        if self.sycl_device.has_aspect_fp64:
            return numpy.dtype(numpy.float64)
        return numpy.dtype(numpy.float32)

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
        param_types=None,
        **kwargs,
    ):
        """
        Draw samples by `kernel` of the extension into a new array, or into
        `out`.

        `params` are the user's distribution parameters, which set the output
        shape along with `size`. `kernel_params` are the parameters passed to
        the kernel, `params` by default, and `param_types` their data types,
        `dtype` by default. Every kernel parameter that is an array is
        broadcast to the output shape. `kwargs` are passed to the kernel.

        """

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
        if param_types is None:
            param_types = (dtype,) * len(kernel_params)

        def _to_kernel_param(p, dt):
            if _is_array(p):
                if not dpnp.is_supported_array_type(p):
                    # NumPy converts None to NaN, dpnp.asarray does not
                    p = numpy.asarray(p, dtype=dt)
                p = dpnp.asarray(p, dtype=dt, sycl_queue=queue)
                return dpnp.ascontiguousarray(dpnp.broadcast_to(p, shape))
            if dt.kind == "f":
                # as in NumPy, None is NaN along with array parameters
                return float(numpy.asarray(p, dtype=numpy.float64))
            if dt.kind == "b":
                return bool(p)
            return int(p)

        kernel_params = tuple(
            _to_kernel_param(p, dt) for p, dt in zip(kernel_params, param_types)
        )

        if out is None:
            out = dpnp.empty(
                shape, dtype=dtype, usm_type=usm_type, sycl_queue=queue
            )
        return self._bit_generator._fill(kernel, kernel_params, out, **kwargs)

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
        scalar = _is_scalar_path(*values)
        for value, pname, cons in params:
            _check_constraint(value, pname, cons, scalar)
        return self._float_sample(
            kernel, values, size, dtype, usm_type, name, out=out
        )

    def _discrete_sample(
        self,
        kernel,
        params,
        size,
        usm_type,
        *,
        kernel_params=None,
        param_types=None,
    ):
        """
        Draw ``int64`` samples of a discrete distribution.

        A ``None`` in `param_types` stands for the floating-point type the
        distribution computes in, which is also the default for every
        parameter.

        """

        validate_usm_type(usm_type, allow_none=False)
        ft = self._discrete_float_type()
        if kernel_params is None:
            kernel_params = params
        if param_types is None:
            param_types = (None,) * len(kernel_params)
        param_types = tuple(ft if t is None else t for t in param_types)
        return self._sample(
            kernel,
            params,
            size,
            _int64,
            usm_type,
            kernel_params=kernel_params,
            param_types=param_types,
            double_precision=ft == numpy.float64,
        )

    def _permutation_index(self, n, queue=None):
        """
        A random permutation of ``arange(n)``, as ``int64`` values on `queue`
        or on the queue of the generator.

        """

        keys = self._bit_generator.random_raw(max(n, 0))
        idx = dpnp.argsort(keys)
        if queue is not None:
            idx = dpnp.asarray(idx, sycl_queue=queue)
        return idx

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
            _check_constraint(rng, "high - low", _CONS_NON_NEGATIVE, scalar)
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
        return self._cont(
            ri._lognormal,
            "lognormal",
            size,
            dtype,
            usm_type,
            (mean, "mean", _CONS_NONE),
            (sigma, "sigma", _CONS_NON_NEGATIVE),
        )

    def standard_cauchy(self, size=None, dtype=None, *, usm_type="device"):
        """
        Draw samples from a standard Cauchy distribution with mode = 0.

        For full documentation refer to
        :obj:`numpy.random.Generator.standard_cauchy`.

        Limitations
        -----------
        `dtype` and `usm_type` are not in NumPy's signature.

        """
        return self._float_sample(
            ri._standard_cauchy, (), size, dtype, usm_type, "standard_cauchy"
        )

    def wald(self, mean, scale, size=None, dtype=None, *, usm_type="device"):
        """
        Draw samples from a Wald, or inverse Gaussian, distribution.

        For full documentation refer to :obj:`numpy.random.Generator.wald`.

        Limitations
        -----------
        `dtype` and `usm_type` are not in NumPy's signature.

        """
        return self._cont(
            ri._wald,
            "wald",
            size,
            dtype,
            usm_type,
            (mean, "mean", _CONS_POSITIVE),
            (scale, "scale", _CONS_POSITIVE),
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
            kernel_params=(1.0,),
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
        return self._cont(
            ri._exponential,
            "exponential",
            size,
            dtype,
            usm_type,
            (scale, "scale", _CONS_NON_NEGATIVE),
        )

    def pareto(self, a, size=None, dtype=None, *, usm_type="device"):
        """
        Draw samples from a Pareto II (AKA Lomax) distribution with the
        specified shape.

        For full documentation refer to :obj:`numpy.random.Generator.pareto`.

        Limitations
        -----------
        `dtype` and `usm_type` are not in NumPy's signature.

        """
        return self._cont(
            ri._pareto,
            "pareto",
            size,
            dtype,
            usm_type,
            (a, "a", _CONS_POSITIVE),
        )

    def weibull(self, a, size=None, dtype=None, *, usm_type="device"):
        """
        Draw samples from a Weibull distribution.

        For full documentation refer to :obj:`numpy.random.Generator.weibull`.

        Limitations
        -----------
        `dtype` and `usm_type` are not in NumPy's signature.

        """
        return self._cont(
            ri._weibull,
            "weibull",
            size,
            dtype,
            usm_type,
            (a, "a", _CONS_NON_NEGATIVE),
        )

    def power(self, a, size=None, dtype=None, *, usm_type="device"):
        """
        Draw samples in [0, 1] from a power distribution with positive
        exponent ``a - 1``.

        For full documentation refer to :obj:`numpy.random.Generator.power`.

        Limitations
        -----------
        `dtype` and `usm_type` are not in NumPy's signature.

        """
        return self._cont(
            ri._power, "power", size, dtype, usm_type, (a, "a", _CONS_POSITIVE)
        )

    def rayleigh(self, scale=1.0, size=None, dtype=None, *, usm_type="device"):
        """
        Draw samples from a Rayleigh distribution.

        For full documentation refer to
        :obj:`numpy.random.Generator.rayleigh`.

        Limitations
        -----------
        `dtype` and `usm_type` are not in NumPy's signature.

        """
        return self._cont(
            ri._rayleigh,
            "rayleigh",
            size,
            dtype,
            usm_type,
            (scale, "scale", _CONS_NON_NEGATIVE),
        )

    def laplace(
        self, loc=0.0, scale=1.0, size=None, dtype=None, *, usm_type="device"
    ):
        """
        Draw samples from the Laplace or double exponential distribution
        with specified location (or mean) and scale (decay).

        For full documentation refer to :obj:`numpy.random.Generator.laplace`.

        Limitations
        -----------
        `dtype` and `usm_type` are not in NumPy's signature.

        """
        return self._cont(
            ri._laplace,
            "laplace",
            size,
            dtype,
            usm_type,
            (loc, "loc", _CONS_NONE),
            (scale, "scale", _CONS_NON_NEGATIVE),
        )

    def gumbel(
        self, loc=0.0, scale=1.0, size=None, dtype=None, *, usm_type="device"
    ):
        """
        Draw samples from a Gumbel distribution.

        For full documentation refer to :obj:`numpy.random.Generator.gumbel`.

        Limitations
        -----------
        `dtype` and `usm_type` are not in NumPy's signature.

        """
        return self._cont(
            ri._gumbel,
            "gumbel",
            size,
            dtype,
            usm_type,
            (loc, "loc", _CONS_NONE),
            (scale, "scale", _CONS_NON_NEGATIVE),
        )

    def logistic(
        self, loc=0.0, scale=1.0, size=None, dtype=None, *, usm_type="device"
    ):
        """
        Draw samples from a logistic distribution.

        For full documentation refer to
        :obj:`numpy.random.Generator.logistic`.

        Limitations
        -----------
        `dtype` and `usm_type` are not in NumPy's signature.

        """
        return self._cont(
            ri._logistic,
            "logistic",
            size,
            dtype,
            usm_type,
            (loc, "loc", _CONS_NONE),
            (scale, "scale", _CONS_NON_NEGATIVE),
        )

    def triangular(
        self, left, mode, right, size=None, dtype=None, *, usm_type="device"
    ):
        """
        Draw samples from the triangular distribution over the interval
        [left, right].

        For full documentation refer to
        :obj:`numpy.random.Generator.triangular`.

        Limitations
        -----------
        `dtype` and `usm_type` are not in NumPy's signature.

        """

        params = (left, mode, right)
        if not any(dpnp.is_supported_array_type(p) for p in params):
            fleft, fmode, fright = (
                numpy.asarray(p, dtype=numpy.float64) for p in params
            )
            if _is_scalar_path(*params):
                for p in params:
                    _check_real(p)
            if numpy.any(fleft > fmode):
                raise ValueError("left > mode")
            if numpy.any(fmode > fright):
                raise ValueError("mode > right")
            if numpy.any(fleft == fright):
                raise ValueError("left == right")
        return self._float_sample(
            ri._triangular, params, size, dtype, usm_type, "triangular"
        )

    def vonmises(self, mu, kappa, size=None, dtype=None, *, usm_type="device"):
        """
        Draw samples from a von Mises distribution.

        For full documentation refer to
        :obj:`numpy.random.Generator.vonmises`.

        Limitations
        -----------
        `dtype` and `usm_type` are not in NumPy's signature.

        """
        return self._cont(
            ri._vonmises,
            "vonmises",
            size,
            dtype,
            usm_type,
            (mu, "mu", _CONS_NONE),
            (kappa, "kappa", _CONS_NON_NEGATIVE),
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

        _check_constraint(
            shape, "shape", _CONS_NON_NEGATIVE, _is_scalar_path(shape)
        )
        return self._float_sample(
            ri._gamma,
            (shape,),
            size,
            dtype,
            usm_type,
            "standard_gamma",
            out=out,
            kernel_params=(shape, 1.0),
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
        return self._cont(
            ri._gamma,
            "gamma",
            size,
            dtype,
            usm_type,
            (shape, "shape", _CONS_NON_NEGATIVE),
            (scale, "scale", _CONS_NON_NEGATIVE),
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
        return self._cont(
            ri._chisquare,
            "chisquare",
            size,
            dtype,
            usm_type,
            (df, "df", _CONS_POSITIVE),
        )

    def noncentral_chisquare(
        self, df, nonc, size=None, dtype=None, *, usm_type="device"
    ):
        """
        Draw samples from a noncentral chi-square distribution.

        For full documentation refer to
        :obj:`numpy.random.Generator.noncentral_chisquare`.

        Limitations
        -----------
        `dtype` and `usm_type` are not in NumPy's signature.

        """
        return self._cont(
            ri._noncentral_chisquare,
            "noncentral_chisquare",
            size,
            dtype,
            usm_type,
            (df, "df", _CONS_POSITIVE),
            (nonc, "nonc", _CONS_NON_NEGATIVE),
        )

    def standard_t(self, df, size=None, dtype=None, *, usm_type="device"):
        """
        Draw samples from a standard Student's t distribution with `df`
        degrees of freedom.

        For full documentation refer to
        :obj:`numpy.random.Generator.standard_t`.

        Limitations
        -----------
        `dtype` and `usm_type` are not in NumPy's signature.

        """
        return self._cont(
            ri._standard_t,
            "standard_t",
            size,
            dtype,
            usm_type,
            (df, "df", _CONS_POSITIVE),
        )

    def f(self, dfnum, dfden, size=None, dtype=None, *, usm_type="device"):
        """
        Draw samples from an F distribution.

        For full documentation refer to :obj:`numpy.random.Generator.f`.

        Limitations
        -----------
        `dtype` and `usm_type` are not in NumPy's signature.

        """
        return self._cont(
            ri._f,
            "f",
            size,
            dtype,
            usm_type,
            (dfnum, "dfnum", _CONS_POSITIVE),
            (dfden, "dfden", _CONS_POSITIVE),
        )

    def noncentral_f(
        self, dfnum, dfden, nonc, size=None, dtype=None, *, usm_type="device"
    ):
        """
        Draw samples from the noncentral F distribution.

        For full documentation refer to
        :obj:`numpy.random.Generator.noncentral_f`.

        Limitations
        -----------
        `dtype` and `usm_type` are not in NumPy's signature.

        """
        return self._cont(
            ri._noncentral_f,
            "noncentral_f",
            size,
            dtype,
            usm_type,
            (dfnum, "dfnum", _CONS_POSITIVE),
            (dfden, "dfden", _CONS_POSITIVE),
            (nonc, "nonc", _CONS_NON_NEGATIVE),
        )

    def beta(self, a, b, size=None, dtype=None, *, usm_type="device"):
        """
        Draw samples from a beta distribution.

        For full documentation refer to :obj:`numpy.random.Generator.beta`.

        Limitations
        -----------
        `dtype` and `usm_type` are not in NumPy's signature.

        """
        return self._cont(
            ri._beta,
            "beta",
            size,
            dtype,
            usm_type,
            (a, "a", _CONS_POSITIVE),
            (b, "b", _CONS_POSITIVE),
        )

    def poisson(self, lam=1.0, size=None, *, usm_type="device"):
        """
        Draw samples from a Poisson distribution.

        For full documentation refer to :obj:`numpy.random.Generator.poisson`.

        Limitations
        -----------
        `usm_type` is not in NumPy's signature.

        """

        _check_constraint(lam, "lam", _CONS_POISSON, _is_scalar_path(lam))
        return self._discrete_sample(ri._poisson, (lam,), size, usm_type)

    def binomial(self, n, p, size=None, *, usm_type="device"):
        """
        Draw samples from a binomial distribution.

        For full documentation refer to
        :obj:`numpy.random.Generator.binomial`.

        Limitations
        -----------
        `usm_type` is not in NumPy's signature.

        """

        scalar = _is_scalar_path(n, p)
        # NumPy converts both parameters before checking either
        n = _as_int64(n)
        _check_constraint(p, "p", _CONS_BOUNDED_0_1, scalar)
        _check_constraint(n, "n", _CONS_NON_NEGATIVE, scalar)
        return self._discrete_sample(
            ri._binomial, (n, p), size, usm_type, param_types=(_int64, None)
        )

    def negative_binomial(self, n, p, size=None, *, usm_type="device"):
        """
        Draw samples from a negative binomial distribution.

        For full documentation refer to
        :obj:`numpy.random.Generator.negative_binomial`.

        Limitations
        -----------
        `usm_type` is not in NumPy's signature.

        """

        # NumPy takes scalar parameters as arrays here, so None gives NaN
        _check_constraint(n, "n", _CONS_POSITIVE_NOT_NAN, False)
        _check_constraint(p, "p", _CONS_BOUNDED_GT_0_1, False)
        if not (
            dpnp.is_supported_array_type(n) or dpnp.is_supported_array_type(p)
        ):
            dn = numpy.asarray(n, dtype=numpy.float64)
            dp = numpy.asarray(p, dtype=numpy.float64)
            with numpy.errstate(all="ignore"):
                max_lam = (1 - dp) / dp * (dn + 10 * numpy.sqrt(dn))
            if numpy.any(max_lam > _POISSON_LAM_MAX):
                raise ValueError(
                    "n too large or p too small, see "
                    "Generator.negative_binomial Notes"
                )
        return self._discrete_sample(
            ri._negative_binomial, (n, p), size, usm_type
        )

    def geometric(self, p, size=None, *, usm_type="device"):
        """
        Draw samples from the geometric distribution.

        For full documentation refer to
        :obj:`numpy.random.Generator.geometric`.

        Limitations
        -----------
        `usm_type` is not in NumPy's signature.

        """

        _check_constraint(p, "p", _CONS_BOUNDED_GT_0_1, _is_scalar_path(p))
        return self._discrete_sample(ri._geometric, (p,), size, usm_type)

    def logseries(self, p, size=None, *, usm_type="device"):
        """
        Draw samples from a logarithmic series distribution.

        For full documentation refer to
        :obj:`numpy.random.Generator.logseries`.

        Limitations
        -----------
        `usm_type` is not in NumPy's signature.

        """

        _check_constraint(p, "p", _CONS_BOUNDED_LT_0_1, _is_scalar_path(p))
        return self._discrete_sample(ri._logseries, (p,), size, usm_type)

    def zipf(self, a, size=None, *, usm_type="device"):
        """
        Draw samples from a Zipf distribution.

        For full documentation refer to :obj:`numpy.random.Generator.zipf`.

        Limitations
        -----------
        `usm_type` is not in NumPy's signature.

        """

        _check_constraint(a, "a", _CONS_GT_1, _is_scalar_path(a))
        return self._discrete_sample(ri._zipf, (a,), size, usm_type)

    def hypergeometric(
        self, ngood, nbad, nsample, size=None, *, usm_type="device"
    ):
        """
        Draw samples from a hypergeometric distribution.

        For full documentation refer to
        :obj:`numpy.random.Generator.hypergeometric`.

        Limitations
        -----------
        `usm_type` is not in NumPy's signature.

        """

        params = tuple(_as_int64(x) for x in (ngood, nbad, nsample))
        names = ("ngood", "nbad", "nsample")
        scalar = _is_scalar_path(*params)

        on_host = [not dpnp.is_supported_array_type(x) for x in params]
        for x, host in zip(params[:2], on_host):
            if host and numpy.any(numpy.asarray(x) >= _HYPERGEOM_MAX):
                raise ValueError(
                    "both ngood and nbad must be less than " f"{_HYPERGEOM_MAX}"
                )
        if all(on_host):
            good, bad, sample = (numpy.asarray(x) for x in params)
            if numpy.any(good + bad < sample):
                raise ValueError("ngood + nbad < nsample")
        for x, name in zip(params, names):
            _check_constraint(x, name, _CONS_NON_NEGATIVE, scalar)

        return self._discrete_sample(
            ri._hypergeometric,
            params,
            size,
            usm_type,
            param_types=(_int64,) * 3,
        )

    def dirichlet(self, alpha, size=None, dtype=None, *, usm_type="device"):
        """
        Draw samples from the Dirichlet distribution.

        For full documentation refer to
        :obj:`numpy.random.Generator.dirichlet`.

        Limitations
        -----------
        `dtype` and `usm_type` are not in NumPy's signature. `alpha` is
        copied to the host, where it is checked, and one kernel is launched
        for every component when all of `alpha` is less than 0.1.

        """

        validate_usm_type(usm_type, allow_none=False)
        dtype = self._float_dtype(dtype, "dirichlet")
        queue = self.sycl_queue

        k = len(alpha)
        if dpnp.is_supported_array_type(alpha):
            alpha = dpnp.asnumpy(alpha)
        alpha = numpy.asarray(alpha, dtype=numpy.float64)
        if alpha.ndim != 1:
            raise ValueError(
                "setting an array element with a sequence. The requested "
                "array would exceed the maximum number of dimension of 1."
            )
        if numpy.any(alpha < 0):
            raise ValueError("alpha < 0")

        size = _normalize_size(size)
        shape = (() if size is None else size) + (k,)

        if k == 0 or not alpha.max() < 0.1:
            # normalize gamma variates, NaN in alpha giving NaN
            res = self._float_sample(
                ri._gamma,
                (),
                shape,
                dtype,
                usm_type,
                "dirichlet",
                kernel_params=(alpha, 1.0),
            )
            res /= res.sum(axis=-1, keepdims=True)
            return res

        # With small alpha the gamma variates underflow, so break a stick
        # by beta variates instead, as NumPy does. csum[j] is sum(alpha[j:]).
        csum = numpy.cumsum(alpha[::-1])[::-1]
        res = dpnp.zeros(
            shape, dtype=dtype, usm_type=usm_type, sycl_queue=queue
        )
        if csum[0] == 0:
            return res

        batch = shape[:-1]
        acc = dpnp.ones(batch, dtype=dtype, usm_type=usm_type, sycl_queue=queue)
        last = k - 1
        for j in range(k - 1):
            if csum[j + 1] == 0:
                # the stick left goes to component j
                last = j
                break
            if alpha[j] == 0:
                continue
            v = self._float_sample(
                ri._beta,
                (),
                batch,
                dtype,
                usm_type,
                "dirichlet",
                kernel_params=(alpha[j], csum[j + 1]),
            )
            res[..., j] = acc * v
            acc *= 1 - v
        res[..., last] = acc
        return res

    def multinomial(self, n, pvals, size=None, *, usm_type="device"):
        """
        Draw samples from a multinomial distribution.

        For full documentation refer to
        :obj:`numpy.random.Generator.multinomial`.

        Limitations
        -----------
        `usm_type` is not in NumPy's signature. One binomial kernel is
        launched for every category but the last.

        """

        validate_usm_type(usm_type, allow_none=False)
        queue = self.sycl_queue
        ft = self._discrete_float_type()

        on_host = not dpnp.is_supported_array_type(pvals)
        if on_host:
            pvals = numpy.asarray(pvals, dtype=numpy.float64)
        if pvals.ndim == 0 or pvals.shape[-1] == 0:
            raise ValueError(
                "pvals must have at least 1 dimension and the last dimension "
                "of pvals must be greater than 0."
            )
        if on_host:
            _check_constraint(pvals, "pvals", _CONS_BOUNDED_0_1, False)
            if numpy.any(pvals[..., :-1].sum(axis=-1) > 1.0 + 1e-12):
                raise ValueError(
                    "sum(pvals[:-1]) > 1.0"
                    if pvals.ndim == 1
                    else "sum(pvals[...,:-1]) > 1.0"
                )

        n = _as_int64(n)
        _check_constraint(n, "n", _CONS_NON_NEGATIVE, numpy.ndim(n) == 0)

        d = pvals.shape[-1]
        batch = numpy.broadcast_shapes(numpy.shape(n), pvals.shape[:-1])
        size = _normalize_size(size)
        if size is not None:
            try:
                compatible = numpy.broadcast_shapes(size, batch) == size
            except ValueError:
                compatible = False
            if not compatible:
                raise ValueError(
                    f"Output size {size} is not compatible with broadcast "
                    f"dimensions of inputs {batch}."
                )
            batch = size

        res = dpnp.empty(
            batch + (d,), dtype=_int64, usm_type=usm_type, sycl_queue=queue
        )

        # Draw the categories in turn, each from a binomial of the trials
        # left with the probability of the category conditioned on not
        # being any of the previous ones
        host_p = on_host and pvals.ndim == 1
        if host_p:
            remaining = 1.0
        else:
            pvals = dpnp.asarray(pvals, dtype=ft, sycl_queue=queue)
            remaining = dpnp.ones_like(pvals[..., 0])
        dn = dpnp.asarray(n, sycl_queue=queue) if _is_array(n) else n
        for j in range(d - 1):
            pj = pvals[..., j]
            if host_p:
                prob = (
                    min(max(pj / remaining, 0.0), 1.0) if remaining > 0 else 0
                )
            else:
                prob = dpnp.where(
                    remaining > 0, dpnp.clip(pj / remaining, 0, 1), 0
                )
            x = self._discrete_sample(
                ri._binomial,
                (),
                batch,
                usm_type,
                kernel_params=(dn, prob),
                param_types=(_int64, None),
            )
            res[..., j] = x
            dn = dn - x
            remaining = remaining - pj
        res[..., d - 1] = dn
        return res

    def multivariate_hypergeometric(
        self,
        colors,
        nsample,
        size=None,
        method="marginals",
        *,
        usm_type="device",
    ):
        """
        Generate variates from a multivariate hypergeometric distribution.

        For full documentation refer to
        :obj:`numpy.random.Generator.multivariate_hypergeometric`.

        Limitations
        -----------
        `usm_type` is not in NumPy's signature. Both methods draw by the
        marginals algorithm, launching one kernel for every color but the
        last, so ``sum(colors)`` must be less than 1000000000 for the
        ``"count"`` method as well.

        """

        validate_usm_type(usm_type, allow_none=False)
        if method not in ("count", "marginals"):
            raise ValueError('method must be "count" or "marginals".')

        try:
            nsample = operator.index(nsample)
        except TypeError:
            raise ValueError("nsample must be an integer") from None
        if nsample < 0:
            raise ValueError("nsample must be nonnegative.")
        if nsample > _INT64_MAX:
            raise ValueError(f"nsample must not exceed {_INT64_MAX}")

        if dpnp.is_supported_array_type(colors):
            colors = dpnp.asnumpy(colors)
        invalid_colors = False
        try:
            colors = numpy.asarray(colors)
            if colors.ndim != 1:
                invalid_colors = True
            elif colors.size > 0 and not numpy.issubdtype(
                colors.dtype, numpy.integer
            ):
                invalid_colors = True
            elif numpy.any((colors < 0) | (colors > _INT64_MAX)):
                invalid_colors = True
        except ValueError:
            invalid_colors = True
        if invalid_colors:
            raise ValueError(
                "colors must be a one-dimensional sequence of nonnegative "
                f"integers not exceeding {_INT64_MAX}."
            )

        colors = numpy.ascontiguousarray(colors, dtype=_int64)
        total = sum(int(c) for c in colors)
        if total > _INT64_MAX:
            raise ValueError(
                "sum(colors) must not exceed the maximum value of a 64 bit "
                f"signed integer ({_INT64_MAX})"
            )
        if total >= _HYPERGEOM_MAX:
            raise ValueError(
                f'When method is "{method}", sum(colors) must be less than '
                f"{_HYPERGEOM_MAX}."
            )
        if nsample > total:
            raise ValueError("nsample > sum(colors)")

        size = _normalize_size(size)
        batch = () if size is None else size
        k = colors.size
        res = dpnp.zeros(
            batch + (k,),
            dtype=_int64,
            usm_type=usm_type,
            sycl_queue=self.sycl_queue,
        )
        if k == 0 or nsample == 0 or res.size == 0:
            return res

        # draw the complement when it is smaller
        more_than_half = nsample > total // 2
        num_to_sample = total - nsample if more_than_half else nsample
        remaining = total
        for j in range(k - 1):
            remaining -= int(colors[j])
            x = self._discrete_sample(
                ri._hypergeometric,
                (),
                batch,
                usm_type,
                kernel_params=(int(colors[j]), remaining, num_to_sample),
                param_types=(_int64,) * 3,
            )
            res[..., j] = x
            num_to_sample = num_to_sample - x
        res[..., k - 1] = num_to_sample

        if more_than_half:
            dpnp.subtract(
                dpnp.asarray(
                    colors, usm_type=usm_type, sycl_queue=res.sycl_queue
                ),
                res,
                out=res,
            )
        return res

    def multivariate_normal(
        self,
        mean,
        cov,
        size=None,
        check_valid="warn",
        tol=1e-8,
        *,
        method="svd",
        dtype=None,
        usm_type="device",
    ):
        """
        Draw random samples from a multivariate normal distribution.

        For full documentation refer to
        :obj:`numpy.random.Generator.multivariate_normal`.

        Limitations
        -----------
        `dtype` and `usm_type` are not in NumPy's signature. The covariance
        matrix is factorized on the device by :mod:`dpnp.linalg`, in double
        precision if the device supports it. Otherwise it is factorized in
        single precision, and `tol` is taken to be at least ``100 * eps`` of
        :obj:`dpnp.float32`.

        """

        if method not in {"eigh", "svd", "cholesky"}:
            raise ValueError(
                "method must be one of {'eigh', 'svd', 'cholesky'}"
            )

        validate_usm_type(usm_type, allow_none=False)
        dtype = self._float_dtype(dtype, "multivariate_normal")
        queue = self.sycl_queue
        mean = dpnp.asarray(mean, usm_type=usm_type, sycl_queue=queue)
        cov = dpnp.asarray(cov, usm_type=usm_type, sycl_queue=queue)
        if dpnp.issubdtype(mean.dtype, dpnp.complexfloating) or dpnp.issubdtype(
            cov.dtype, dpnp.complexfloating
        ):
            raise TypeError("mean and cov must not be complex")

        if mean.ndim != 1:
            raise ValueError("mean must be 1 dimensional")
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("cov must be 2 dimensional and square")
        if mean.shape[0] != cov.shape[0]:
            raise ValueError("mean and cov must have same length")

        cov = cov.astype(dpnp.default_float_type(sycl_queue=queue))
        if cov.dtype == dpnp.float32:
            tol = max(tol, 100 * dpnp.finfo(dpnp.float32).eps)
        # as in NumPy, check_valid is ignored by the cholesky method
        if method != "cholesky" and check_valid not in (
            "warn",
            "raise",
            "ignore",
        ):
            raise ValueError(
                "check_valid must equal 'warn', 'raise', or 'ignore'"
            )

        psd = True
        if method == "svd":
            u, s, vh = dpnp.linalg.svd(cov)
            if check_valid != "ignore":
                psd = dpnp.allclose(
                    dpnp.dot(vh.T * s, vh), cov, rtol=tol, atol=tol
                )
            factor = u * dpnp.sqrt(s)
        elif method == "eigh":
            s, u = dpnp.linalg.eigh(cov)
            if check_valid != "ignore":
                psd = not dpnp.any(s < -tol)
            factor = u * dpnp.sqrt(dpnp.abs(s))
        else:
            factor = dpnp.linalg.cholesky(cov)

        if not psd:
            msg = "covariance is not symmetric positive-semidefinite."
            if check_valid == "warn":
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
            else:
                raise ValueError(msg)

        size = _normalize_size(size)
        shape = (() if size is None else size) + mean.shape
        x = self._float_sample(
            ri._normal,
            (),
            shape,
            dtype,
            usm_type,
            "multivariate_normal",
            kernel_params=(0.0, 1.0),
        )
        factor = factor.astype(dtype)
        x = mean.astype(dtype) + x.reshape(-1, mean.shape[0]) @ factor.T
        return x.reshape(shape)

    def permutation(self, x, axis=0):
        """
        Randomly permute a sequence, or return a permuted range.

        For full documentation refer to
        :obj:`numpy.random.Generator.permutation`.

        Limitations
        -----------
        A permutation of an array on another SYCL queue is allocated on that
        queue.

        """

        if isinstance(x, (int, numpy.integer)):
            return self._permutation_index(int(x))

        if dpnp.is_supported_array_type(x):
            arr = dpnp.asarray(x)
        else:
            arr = dpnp.asarray(x, sycl_queue=self.sycl_queue)
        axis = normalize_axis_index(axis, arr.ndim)
        idx = self._permutation_index(arr.shape[axis], arr.sycl_queue)
        return dpnp.take(arr, idx, axis=axis)

    def shuffle(self, x, axis=0):
        """
        Modify an array in place by shuffling its contents.

        For full documentation refer to
        :obj:`numpy.random.Generator.shuffle`.

        Limitations
        -----------
        As in CuPy, `x` must be a :class:`dpnp.ndarray`.

        """

        if not isinstance(x, dpnp.ndarray):
            raise TypeError(f"x must be a dpnp.ndarray, got {type(x)}")
        if x.ndim == 0:
            raise TypeError("len() of unsized object")
        axis = normalize_axis_index(axis, x.ndim)
        idx = self._permutation_index(x.shape[axis], x.sycl_queue)
        x[...] = dpnp.take(x, idx, axis=axis)

    def permuted(self, x, *, axis=None, out=None):
        """
        Randomly permute `x` along axis `axis`.

        For full documentation refer to
        :obj:`numpy.random.Generator.permuted`.

        Limitations
        -----------
        `out` must be a :class:`dpnp.ndarray`.

        """

        if dpnp.is_supported_array_type(x):
            x = dpnp.asarray(x)
        else:
            x = dpnp.asarray(x, sycl_queue=self.sycl_queue)

        if out is not None:
            if not isinstance(out, dpnp.ndarray):
                raise TypeError("out must be a dpnp.ndarray")
            if out.shape != x.shape:
                raise ValueError("out must have the same shape as x")
            if not dpnp.can_cast(x.dtype, out.dtype, casting="safe"):
                raise TypeError(
                    f"Cannot cast array data from {x.dtype} to {out.dtype} "
                    "according to the rule 'safe'"
                )

        if axis is None:
            if x.ndim == 0:
                raise TypeError("len() of unsized object")
            idx = self._permutation_index(x.size, x.sycl_queue)
            res = dpnp.take(dpnp.ravel(x), idx).reshape(x.shape)
        else:
            axis = normalize_axis_index(axis, x.ndim)
            keys = self._bit_generator.random_raw(x.shape)
            idx = dpnp.asarray(
                dpnp.argsort(keys, axis=axis), sycl_queue=x.sycl_queue
            )
            res = dpnp.take_along_axis(x, idx, axis=axis)

        if out is None:
            return res
        out[...] = res
        return out

    def choice(
        self,
        a,
        size=None,
        replace=True,
        p=None,
        axis=0,
        shuffle=True,
        *,
        usm_type="device",
    ):
        """
        Generate a random sample from a given array.

        For full documentation refer to :obj:`numpy.random.Generator.choice`.

        Limitations
        -----------
        `usm_type` is not in NumPy's signature. Without replacement, the
        sample is always in random order, so `shuffle` has no effect, and
        memory and time proportional to the population size are used. `p`
        is checked only when it is on the host.

        """

        # pylint: disable=unused-argument
        return self._choice(a, size, replace, p, axis, usm_type, _CHOICE_MSGS)

    # pylint: disable=too-many-branches, too-many-statements
    def _choice(self, a, size, replace, p, axis, usm_type, msgs):
        """
        The implementation of :meth:`choice`, raising the messages of the
        table `msgs`, as those of NumPy's legacy ``RandomState.choice``
        differ.

        """

        validate_usm_type(usm_type, allow_none=False)
        queue = self.sycl_queue
        no_samples = size is not None and numpy.prod(size) == 0

        a_original = a
        a = (
            dpnp.asarray(a)
            if dpnp.is_supported_array_type(a)
            else numpy.asarray(a)
        )
        if a.ndim == 0:
            try:
                pop_size = operator.index(a.item())
            except TypeError as exc:
                raise ValueError(
                    msgs["a_type"].format(type(a_original))
                ) from exc
            if pop_size <= 0 and not no_samples:
                raise ValueError(msgs["a_int"])
        else:
            if msgs["a_1d"] is not None and a.ndim != 1:
                raise ValueError(msgs["a_1d"])
            axis = normalize_axis_index(axis, a.ndim)
            pop_size = a.shape[axis]
            if pop_size == 0 and not no_samples:
                raise ValueError(msgs["a_empty"])

        p_host = None
        if p is not None:
            if dpnp.is_supported_array_type(p):
                p_dev = dpnp.asarray(p)
            else:
                atol = numpy.sqrt(numpy.finfo(numpy.float64).eps)
                if isinstance(p, numpy.ndarray) and numpy.issubdtype(
                    p.dtype, numpy.floating
                ):
                    atol = max(atol, numpy.sqrt(numpy.finfo(p.dtype).eps))
                p_host = numpy.asarray(p, dtype=numpy.float64)
                p_dev = p_host
            if p_dev.ndim != 1:
                raise ValueError(msgs["p_1d"])
            if p_dev.size != pop_size:
                raise ValueError(msgs["p_size"])
            if p_host is not None:
                p_sum = p_host.sum()
                if numpy.isnan(p_sum):
                    raise ValueError(msgs["p_nan"])
                if numpy.any(p_host < 0):
                    raise ValueError(msgs["p_neg"])
                if abs(p_sum - 1.0) > atol:
                    raise ValueError(msgs["p_sum"])
            ft = dpnp.default_float_type(sycl_queue=queue)
            p_dev = dpnp.asarray(p_dev, dtype=ft, sycl_queue=queue)

        shape = () if size is None else _normalize_size(size)
        n = int(numpy.prod(shape, dtype=numpy.int64))
        if not replace:
            if n > pop_size:
                raise ValueError(msgs["too_large"])
            if p_host is not None and numpy.count_nonzero(p_host > 0) < n:
                raise ValueError("Fewer non-zero entries in p than size")

        if n == 0:
            idx = dpnp.empty(
                shape, dtype=_int64, usm_type=usm_type, sycl_queue=queue
            )
        elif replace:
            if p is not None:
                cdf = dpnp.cumsum(p_dev)
                cdf /= cdf[-1]
                u = self.random(shape, dtype=ft, usm_type=usm_type)
                idx = dpnp.searchsorted(cdf, u, side="right")
            else:
                idx = self.integers(0, pop_size, size=shape, usm_type=usm_type)
        elif p is not None:
            # The Efraimidis-Spirakis keys: taking the smallest ones gives
            # the successive weighted draws of NumPy, in the order drawn
            keys = self.standard_exponential(pop_size, dtype=ft) / p_dev
            idx = dpnp.argsort(keys)[:n].reshape(shape)
        else:
            idx = self._permutation_index(pop_size)[:n].reshape(shape)

        if a.ndim == 0:
            return idx
        if not isinstance(a, dpnp.ndarray):
            a = dpnp.asarray(a, usm_type=usm_type, sycl_queue=queue)
        idx = dpnp.asarray(idx, sycl_queue=a.sycl_queue)
        return dpnp.take(a, idx, axis=axis)

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

        return self._integers(
            low, high, size, dtype, endpoint, usm_type, "integers"
        )

    def _integers(self, low, high, size, dtype, endpoint, usm_type, name):
        """The implementation of :meth:`integers`, for the method `name`."""

        validate_usm_type(usm_type, allow_none=False)

        if high is None:
            low, high = 0, low

        dtype = numpy.dtype(dtype)
        if dtype not in _integer_types:
            raise TypeError(f"Unsupported dtype {dtype!r} for {name}")

        # The bounds are passed to the kernel as a closed interval
        if dtype == numpy.bool_:
            lo_lim, hi_lim = 0, 1
        else:
            info = numpy.iinfo(dtype)
            lo_lim, hi_lim = int(info.min), int(info.max)
        dt_name = "bool" if dtype == numpy.bool_ else str(dtype)

        def _check_bounds(low, high):
            # the checks of NumPy, for host bounds only
            if low is not None and numpy.any(numpy.less(low, lo_lim)):
                raise ValueError(f"low is out of bounds for {dt_name}")
            if high is not None and numpy.any(
                numpy.greater(high, hi_lim if endpoint else hi_lim + 1)
            ):
                raise ValueError(f"high is out of bounds for {dt_name}")
            if low is None or high is None:
                return
            if numpy.any(
                numpy.greater(low, high)
                if endpoint
                else numpy.greater_equal(low, high)
            ):
                if not numpy.any(low):
                    # low is 0 when only one bound is given
                    raise ValueError("high < 0" if endpoint else "high <= 0")
                raise ValueError("low > high" if endpoint else "low >= high")

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

            def _on_host(x):
                return None if dpnp.is_supported_array_type(x) else x

            _check_bounds(
                *(
                    None if x is None else numpy.asarray(x)
                    for x in map(_on_host, params)
                )
            )

            def _to_bound(x, shift):
                if dpnp.is_supported_array_type(x):
                    x = dpnp.asarray(x, dtype=wide, sycl_queue=queue)
                    return (x - shift if shift else x).astype(dtype)
                if not _is_array(x):
                    return int(x) - shift
                # exact integers, truncated as in NumPy, since max + 1 fits
                # the wide type only after the shift
                x = numpy.frompyfunc(int, 1, 1)(numpy.asarray(x)) - shift
                x = dpnp.asarray(x.astype(wide), sycl_queue=queue)
                return x.astype(dtype)

            low = _to_bound(low, 0)
            high = _to_bound(high, 0 if endpoint else 1)
            return self._sample(
                ri._integers,
                params,
                size,
                dtype,
                usm_type,
                kernel_params=(low, high),
            )

        low = int(low)
        high = int(high)
        _check_bounds(low, high)
        if not endpoint:
            high -= 1

        return self._sample(
            ri._integers, (), size, dtype, usm_type, kernel_params=(low, high)
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
