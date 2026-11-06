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
The device random number generation API modeled on
:class:`numpy.random.Generator`.

Every distribution is sampled on the device by a port of NumPy's algorithm,
driven by a oneMKL device engine. The values differ from NumPy's.

"""

# pylint: disable=no-name-in-module
# pylint: disable=protected-access
# pylint: disable=too-many-lines

import numpy

import dpnp
import dpnp.libtensor._rng_impl as ri
import dpnp.libtensor._tensor_impl as ti
import dpnp.tensor as dpt
from dpnp.exceptions import ExecutionPlacementError
from dpnp.tensor import validate_usm_type

from ._bit_generator import BitGenerator, Philox4x3210, _normalize_size

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

_POISSON_LAM_MAX = _INT64_MAX - numpy.sqrt(_INT64_MAX) * 10
_HYPERGEOM_MAX = 10**9

# bounds keeping valid host parameters valid once rounded to float32; the
# smallest normal also survives a device flushing subnormals to zero
_F32_GT_0 = (numpy.finfo(numpy.float32).tiny, None)
_F32_GT_1 = (numpy.nextafter(numpy.float32(1), numpy.float32(2)), None)
_F32_LT_1 = (None, numpy.nextafter(numpy.float32(1), numpy.float32(0)))
_F32_LT_2_63 = (None, numpy.nextafter(numpy.float32(2**63), numpy.float32(0)))


def _is_array(x):
    return dpnp.is_supported_array_type(x) or numpy.ndim(x) > 0


def _is_scalar_path(*params):
    """
    Whether NumPy takes the path of scalar parameters, whose error messages
    differ slightly from the ones of array parameters.

    """
    return all(numpy.ndim(p) == 0 for p in params)


def _check_real(value):
    """
    Raise TypeError, as NumPy does on the path of scalar parameters, unless
    the host scalar `value` converts to a Python float.

    """

    t = type(value)
    if not (hasattr(t, "__float__") or hasattr(t, "__index__")):
        raise TypeError(f"must be real number, not {t.__name__}")


def _check_host_cast(value):
    """
    Raise TypeError, as NumPy does, when the host array `value` does not
    cast safely to float64, such as a complex or an object array.

    """

    if isinstance(value, numpy.ndarray) and not numpy.can_cast(
        value.dtype, numpy.float64, "safe"
    ):
        what = "scalar" if value.ndim == 0 else "array data"
        raise TypeError(
            f"Cannot cast {what} from {value.dtype!r} to "
            "dtype('float64') according to the rule 'safe'"
        )


def _check_device_cast(value, dtype):
    """
    Raise TypeError unless the device array `value` casts to `dtype` by the
    ``"same_kind"`` rule, which lets a float64 array feed a float32 draw.

    An integer or boolean parameter only rejects a complex array, as it takes
    floating-point arrays converted as by :meth:`dpnp.ndarray.astype`.

    """

    if dtype.kind == "f":
        castable = dpnp.can_cast(value.dtype, dtype, "same_kind")
    else:
        castable = value.dtype.kind != "c"
    if not castable:
        raise TypeError(
            f"Cannot cast array data from {value.dtype!r} to {dtype!r} "
            "according to the rule 'same_kind'"
        )


def _as_host_float(value):
    _check_host_cast(value)
    return numpy.asarray(value, dtype=numpy.float64)


def _check_arrays(params, dtype=None):
    """
    `params` as arrays of `dtype` to check, with the module to check them
    by, which is ``None`` when some are on the device, as the check would
    wait for the device.

    """

    if any(dpnp.is_supported_array_type(p) for p in params):
        return None, [None] * len(params)
    for p in params:
        _check_host_cast(p)
    return numpy, [numpy.asarray(p, dtype) for p in params]


def _uniform_range(low, high, dtype, overflow_msg):
    """
    ``high - low``, raising OverflowError with `overflow_msg` unless it is
    finite in float64, as NumPy does, and in `dtype`, the sampling type.

    Returns ``None`` when either param is on the device, after checking
    that the other one is finite.

    """

    host = [p for p in (low, high) if not dpnp.is_supported_array_type(p)]
    values = [_as_host_float(p) for p in host]
    if _is_scalar_path(low, high):
        for p in host:
            _check_real(p)

    with numpy.errstate(over="ignore", invalid="ignore"):
        if len(host) < 2:
            # the range is unknown, but a host bound that is not finite in
            # `dtype` makes it overflow
            if not all(numpy.isfinite(v.astype(dtype)).all() for v in values):
                raise OverflowError(overflow_msg)
            return None

        flow, fhigh = values
        rng = numpy.subtract(fhigh, flow)
        # the kernel subtracts in `dtype`
        rng_dtype = numpy.subtract(fhigh.astype(dtype), flow.astype(dtype))
    if not (numpy.isfinite(rng).all() and numpy.isfinite(rng_dtype).all()):
        raise OverflowError(overflow_msg)
    return rng


# pylint: disable=too-many-branches
def _usm_type_of(*values):
    """The coerced USM type of the arrays in `values`, or ``"device"``."""

    usm_types = [x.usm_type for x in values if dpnp.is_supported_array_type(x)]
    return dpt.get_coerced_usm_type(usm_types) if usm_types else "device"


def _check_constraint(value, name, cons, scalar):
    """
    Raise ValueError, as NumPy does, when the parameter `value` violates the
    constraint `cons`.

    Parameters on the device are not checked, as the check waits for the
    device. `scalar` selects the messages of NumPy's scalar path.

    """

    if dpnp.is_supported_array_type(value):
        return
    val = _as_host_float(value)
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


def _host_float32(value, bounds):
    """
    Round a host parameter to ``float32`` and clip it to `bounds`, which
    rounding a valid parameter to the nearest ``float32`` may cross.

    """

    if dpnp.is_supported_array_type(value):
        return value
    with numpy.errstate(over="ignore"):
        value = numpy.asarray(value, dtype=numpy.float64).astype(numpy.float32)
    value = numpy.clip(value, *bounds)
    return value if value.ndim else float(value)


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

    The counterpart of :class:`numpy.random.Generator`, drawing all samples on
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
    floating-point type of the device. With ``usm_type=None``, the output
    takes the USM type of the array parameters, or ``"device"`` when there
    are none.

    Pickling, copying, and the ``advance`` and ``jumped`` methods of NumPy's
    bit generators are not supported.

    Parameters on the host (scalars, lists and NumPy arrays) are checked as
    in NumPy. Parameters on the device are not, since that would wait for
    the device. Invalid ones give undefined values, such as NaN or the
    smallest ``int64``, but never make the kernels loop without end or
    access memory out of bounds.

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

    def _discrete_float_type(self):
        """The floating-point type the discrete distributions compute in."""

        if self.sycl_device.has_aspect_fp64:
            return numpy.dtype(numpy.float64)
        return numpy.dtype(numpy.float32)

    def _prepare_out(self, out, size, dtype, name, *, allow_f=False):
        """
        Resolve `size` and `dtype` against a user-supplied `out`.

        As in NumPy, `out` may be F-contiguous when `allow_f` is set, which
        the methods without array parameters do.

        """

        dpnp.check_supported_arrays_type(out)
        # as in NumPy, the default dtype applies even when `out` is given
        dtype = self._float_dtype(dtype, name)
        if out.dtype != dtype:
            raise TypeError(
                "Supplied output array has the wrong type. "
                f"Expected {dtype}, got {out.dtype}"
            )
        flags = out.flags
        if allow_f:
            if not (
                (flags.c_contiguous or flags.f_contiguous) and flags.writable
            ):
                raise ValueError(
                    "Supplied output array must be contiguous, writable, "
                    "aligned, and in machine byte-order."
                )
        elif not (flags.c_contiguous and flags.writable):
            raise ValueError(
                "Supplied output array must be C-contiguous, writable, "
                "aligned, and in machine byte-order."
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

        With ``usm_type=None``, the output and the temporaries take the USM
        type of the array parameters and `out`, or ``"device"``.

        """

        self._check_queues(*params, *(kernel_params or ()), out)
        queue = self.sycl_queue
        if usm_type is None:
            usm_type = _usm_type_of(*params, *(kernel_params or ()), out)

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
                if dpnp.is_supported_array_type(p):
                    _check_device_cast(p, dt)
                else:
                    _check_host_cast(p)
                    # NumPy converts None to NaN, dpnp.asarray does not
                    p = numpy.asarray(p, dtype=dt)
                p = dpnp.asarray(
                    p, dtype=dt, usm_type=usm_type, sycl_queue=queue
                )
                p = dpnp.ascontiguousarray(dpnp.broadcast_to(p, shape))
                if out is not None:
                    usm_p, usm_out = map(dpnp.get_usm_ndarray, (p, out))
                    # the kernel reads element i of `p` before writing
                    # element i of `out`, so only an exact alias is safe
                    if ti._array_overlap(
                        usm_p, usm_out
                    ) and not ti._same_logical_tensors(usm_p, usm_out):
                        p = dpnp.copy(p)
                return p
            if dt.kind == "f":
                # as in NumPy, None is NaN along with array parameters
                return float(_as_host_float(p))
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
            target = out
        elif out.flags.c_contiguous:
            target = out
        else:
            # an F-contiguous `out` of a method without array parameters is
            # filled in memory order, through its C-contiguous transpose
            usm_out = dpnp.get_usm_ndarray(out)
            target = dpt.permute_dims(usm_out, tuple(range(out.ndim))[::-1])
        self._bit_generator._fill(kernel, kernel_params, target, **kwargs)
        return out

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
        validate_usm_type(usm_type, allow_none=True)
        if out is not None:
            size, dtype = self._prepare_out(
                out, size, dtype, name, allow_f=not params
            )
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

    def _discrete_sample(
        self,
        kernel,
        params,
        size,
        usm_type,
        *,
        kernel_params=None,
        param_types=None,
        f32_bounds=None,
    ):
        """
        Draw ``int64`` samples of a discrete distribution.

        A ``None`` in `param_types` stands for the floating-point type the
        distribution computes in, which is also the default for every
        parameter. When that type is ``float32``, the host kernel parameters
        are clipped to the matching `f32_bounds`, unless they are ``None``.

        """

        validate_usm_type(usm_type, allow_none=True)
        ft = self._discrete_float_type()
        if kernel_params is None:
            kernel_params = params
        if f32_bounds is not None and ft == numpy.float32:
            kernel_params = tuple(
                p if b is None else _host_float32(p, b)
                for p, b in zip(kernel_params, f32_bounds)
            )
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

    def random(self, size=None, dtype=None, out=None, *, usm_type=None):
        """
        Return random floats in the half-open interval [0.0, 1.0).

        For full documentation refer to :obj:`numpy.random.Generator.random`.

        Parameters
        ----------
        size : {None, int, tuple of ints}, optional
            Output shape. With ``None``, a zero-dimensional array is returned.

            Default: ``None``.
        dtype : {None, dtype}, optional
            The data type of the output, :obj:`dpnp.float32` or
            :obj:`dpnp.float64`.

            Default: ``None``, which means the default floating-point type of
            the device.
        out : {None, dpnp.ndarray, usm_ndarray}, optional
            A C- or F-contiguous array to fill, of the data type `dtype` and
            on the SYCL queue of the generator. If `size` is given as well,
            it must be the shape of `out`.

            Default: ``None``.
        usm_type : {None, "device", "shared", "host"}, optional
            The type of SYCL USM allocation for the output array. Ignored
            when `out` is given.

            Default: ``None``, which means ``"device"``.

        Returns
        -------
        out : dpnp.ndarray
            The samples, or `out` when it is given.

        Limitations
        -----------
        `dtype` defaults to the default floating-point type of the device, and
        `usm_type` is not in NumPy's signature.

        Examples
        --------
        >>> import dpnp as np
        >>> rng = np.random.default_rng(42)
        >>> x = rng.random((2, 3))
        >>> x.shape, bool(((0 <= x) & (x < 1)).all())
        ((2, 3), True)

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
        self, low=0.0, high=1.0, size=None, dtype=None, *, usm_type=None
    ):
        """
        Draw samples from a uniform distribution over [low, high).

        For full documentation refer to :obj:`numpy.random.Generator.uniform`.

        Parameters
        ----------
        low : {float, array_like of floats, dpnp.ndarray, usm_ndarray}, \
                optional
            The lower bound of the output interval.

            Default: ``0.0``.
        high : {float, array_like of floats, dpnp.ndarray, usm_ndarray}, \
                optional
            The upper bound of the output interval.

            Default: ``1.0``.
        size : {None, int, tuple of ints}, optional
            Output shape. With ``None``, the broadcast shape of `low` and `high`.

            Default: ``None``.
        dtype : {None, dtype}, optional
            The data type of the output, :obj:`dpnp.float32` or
            :obj:`dpnp.float64`.

            Default: ``None``, which means the default floating-point type of
            the device.
        usm_type : {None, "device", "shared", "host"}, optional
            The type of SYCL USM allocation for the output array.

            Default: ``None``, which means the USM type of the array
            parameters, or ``"device"`` when there are none.

        Returns
        -------
        out : dpnp.ndarray
            The samples.

        Limitations
        -----------
        `dtype` and `usm_type` are not in NumPy's signature. A `high - low`
        range that overflows `dtype` raises :class:`OverflowError`, as it
        does in float64.

        Examples
        --------
        >>> import dpnp as np
        >>> rng = np.random.default_rng(42)
        >>> x = rng.uniform(-1, 0, 1000)
        >>> bool(((-1 <= x) & (x < 0)).all())
        True

        """

        self._check_queues(low, high)
        scalar = _is_scalar_path(low, high)
        dtype = self._float_dtype(dtype, "uniform")
        rng = _uniform_range(
            low,
            high,
            dtype,
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
        self, size=None, dtype=None, out=None, *, usm_type=None
    ):
        """
        Draw samples from a standard normal distribution.

        For full documentation refer to
        :obj:`numpy.random.Generator.standard_normal`.

        Parameters
        ----------
        size : {None, int, tuple of ints}, optional
            Output shape. With ``None``, a zero-dimensional array is returned.

            Default: ``None``.
        dtype : {None, dtype}, optional
            The data type of the output, :obj:`dpnp.float32` or
            :obj:`dpnp.float64`.

            Default: ``None``, which means the default floating-point type of
            the device.
        out : {None, dpnp.ndarray, usm_ndarray}, optional
            A C- or F-contiguous array to fill, of the data type `dtype` and
            on the SYCL queue of the generator. If `size` is given as well,
            it must be the shape of `out`.

            Default: ``None``.
        usm_type : {None, "device", "shared", "host"}, optional
            The type of SYCL USM allocation for the output array. Ignored
            when `out` is given.

            Default: ``None``, which means ``"device"``.

        Returns
        -------
        out : dpnp.ndarray
            The samples, or `out` when it is given.

        Limitations
        -----------
        `dtype` and `usm_type` are not in NumPy's signature. The Box-Muller
        transform is used rather than NumPy's ziggurat, so the magnitude of
        the samples is bounded: by about 6.8 in float32, and 8.6 in float64.

        Examples
        --------
        >>> import dpnp as np
        >>> rng = np.random.default_rng(42)
        >>> x = rng.standard_normal(10**6)
        >>> bool(abs(x.mean()) < 0.01), bool(abs(x.std() - 1) < 0.01)
        (True, True)

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
        self, loc=0.0, scale=1.0, size=None, dtype=None, *, usm_type=None
    ):
        """
        Draw samples from a normal (Gaussian) distribution.

        For full documentation refer to :obj:`numpy.random.Generator.normal`.

        Parameters
        ----------
        loc : {float, array_like of floats, dpnp.ndarray, usm_ndarray}, \
                optional
            The mean of the distribution.

            Default: ``0.0``.
        scale : {float, array_like of floats, dpnp.ndarray, usm_ndarray}, \
                optional
            The standard deviation of the distribution, non-negative.

            Default: ``1.0``.
        size : {None, int, tuple of ints}, optional
            Output shape. With ``None``, the broadcast shape of `loc` and `scale`.

            Default: ``None``.
        dtype : {None, dtype}, optional
            The data type of the output, :obj:`dpnp.float32` or
            :obj:`dpnp.float64`.

            Default: ``None``, which means the default floating-point type of
            the device.
        usm_type : {None, "device", "shared", "host"}, optional
            The type of SYCL USM allocation for the output array.

            Default: ``None``, which means the USM type of the array
            parameters, or ``"device"`` when there are none.

        Returns
        -------
        out : dpnp.ndarray
            The samples.

        Limitations
        -----------
        `dtype` and `usm_type` are not in NumPy's signature. The samples are
        drawn as in :meth:`standard_normal`.

        Examples
        --------
        >>> import dpnp as np
        >>> rng = np.random.default_rng(42)
        >>> x = rng.normal(np.array([0.0, 10.0]), 1.0, size=(10**5, 2))
        >>> m = x.mean(axis=0)
        >>> bool(abs(m[0]) < 0.05), bool(abs(m[1] - 10) < 0.05)
        (True, True)

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
        self, mean=0.0, sigma=1.0, size=None, dtype=None, *, usm_type=None
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

    def standard_cauchy(self, size=None, dtype=None, *, usm_type=None):
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

    def wald(self, mean, scale, size=None, dtype=None, *, usm_type=None):
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
        usm_type=None,
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

    def exponential(self, scale=1.0, size=None, dtype=None, *, usm_type=None):
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

    def pareto(self, a, size=None, dtype=None, *, usm_type=None):
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

    def weibull(self, a, size=None, dtype=None, *, usm_type=None):
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

    def power(self, a, size=None, dtype=None, *, usm_type=None):
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

    def rayleigh(self, scale=1.0, size=None, dtype=None, *, usm_type=None):
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
        self, loc=0.0, scale=1.0, size=None, dtype=None, *, usm_type=None
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
        self, loc=0.0, scale=1.0, size=None, dtype=None, *, usm_type=None
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
        self, loc=0.0, scale=1.0, size=None, dtype=None, *, usm_type=None
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
        self, left, mode, right, size=None, dtype=None, *, usm_type=None
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
        self._check_queues(*params)
        xp, (fleft, fmode, fright) = _check_arrays(params, numpy.float64)
        if xp is not None:
            if xp is numpy and _is_scalar_path(*params):
                for p in params:
                    _check_real(p)
            if xp.any(fleft > fmode):
                raise ValueError("left > mode")
            if xp.any(fmode > fright):
                raise ValueError("mode > right")
            if xp.any(fleft == fright):
                raise ValueError("left == right")
        return self._float_sample(
            ri._triangular, params, size, dtype, usm_type, "triangular"
        )

    def vonmises(self, mu, kappa, size=None, dtype=None, *, usm_type=None):
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
        self, shape, size=None, dtype=None, out=None, *, usm_type=None
    ):
        """
        Draw samples from a standard gamma distribution.

        For full documentation refer to
        :obj:`numpy.random.Generator.standard_gamma`.

        Limitations
        -----------
        `usm_type` is not in NumPy's signature.

        """

        self._check_queues(shape, out)
        _check_constraint(
            shape,
            "shape",
            _CONS_NON_NEGATIVE,
            _is_scalar_path(shape),
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

    def gamma(self, shape, scale=1.0, size=None, dtype=None, *, usm_type=None):
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

    def chisquare(self, df, size=None, dtype=None, *, usm_type=None):
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
        self, df, nonc, size=None, dtype=None, *, usm_type=None
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

    def standard_t(self, df, size=None, dtype=None, *, usm_type=None):
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

    def f(self, dfnum, dfden, size=None, dtype=None, *, usm_type=None):
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
        self, dfnum, dfden, nonc, size=None, dtype=None, *, usm_type=None
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

    def beta(self, a, b, size=None, dtype=None, *, usm_type=None):
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

    def poisson(self, lam=1.0, size=None, *, usm_type=None):
        """
        Draw samples from a Poisson distribution.

        For full documentation refer to :obj:`numpy.random.Generator.poisson`.

        Limitations
        -----------
        `usm_type` is not in NumPy's signature.

        """

        self._check_queues(lam)
        _check_constraint(
            lam,
            "lam",
            _CONS_POISSON,
            _is_scalar_path(lam),
        )
        return self._discrete_sample(
            ri._poisson, (lam,), size, usm_type, f32_bounds=(_F32_LT_2_63,)
        )

    def binomial(self, n, p, size=None, *, usm_type=None):
        """
        Draw samples from a binomial distribution.

        For full documentation refer to
        :obj:`numpy.random.Generator.binomial`.

        Limitations
        -----------
        `usm_type` is not in NumPy's signature.

        """

        self._check_queues(n, p)
        scalar = _is_scalar_path(n, p)
        # NumPy converts both parameters before checking either
        n = _as_int64(n)
        _check_constraint(p, "p", _CONS_BOUNDED_0_1, scalar)
        _check_constraint(n, "n", _CONS_NON_NEGATIVE, scalar)
        return self._discrete_sample(
            ri._binomial, (n, p), size, usm_type, param_types=(_int64, None)
        )

    def negative_binomial(self, n, p, size=None, *, usm_type=None):
        """
        Draw samples from a negative binomial distribution.

        For full documentation refer to
        :obj:`numpy.random.Generator.negative_binomial`.

        Limitations
        -----------
        `usm_type` is not in NumPy's signature.

        """

        self._check_queues(n, p)
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

    def geometric(self, p, size=None, *, usm_type=None):
        """
        Draw samples from the geometric distribution.

        For full documentation refer to
        :obj:`numpy.random.Generator.geometric`.

        Limitations
        -----------
        `usm_type` is not in NumPy's signature.

        """

        self._check_queues(p)
        _check_constraint(
            p,
            "p",
            _CONS_BOUNDED_GT_0_1,
            _is_scalar_path(p),
        )
        return self._discrete_sample(
            ri._geometric, (p,), size, usm_type, f32_bounds=(_F32_GT_0,)
        )

    def logseries(self, p, size=None, *, usm_type=None):
        """
        Draw samples from a logarithmic series distribution.

        For full documentation refer to
        :obj:`numpy.random.Generator.logseries`.

        Limitations
        -----------
        `usm_type` is not in NumPy's signature.

        """

        self._check_queues(p)
        _check_constraint(
            p,
            "p",
            _CONS_BOUNDED_LT_0_1,
            _is_scalar_path(p),
        )
        return self._discrete_sample(
            ri._logseries, (p,), size, usm_type, f32_bounds=(_F32_LT_1,)
        )

    def zipf(self, a, size=None, *, usm_type=None):
        """
        Draw samples from a Zipf distribution.

        For full documentation refer to :obj:`numpy.random.Generator.zipf`.

        Limitations
        -----------
        `usm_type` is not in NumPy's signature. The samples follow the Zipf
        distribution truncated to the largest ``int64``, while NumPy's
        rejection test rounds to accepting every sample above ``2**53``,
        which makes its tail too heavy for `a` close to 1.

        """

        self._check_queues(a)
        _check_constraint(a, "a", _CONS_GT_1, _is_scalar_path(a))
        return self._discrete_sample(
            ri._zipf, (a,), size, usm_type, f32_bounds=(_F32_GT_1,)
        )

    def hypergeometric(self, ngood, nbad, nsample, size=None, *, usm_type=None):
        """
        Draw samples from a hypergeometric distribution.

        For full documentation refer to
        :obj:`numpy.random.Generator.hypergeometric`.

        Limitations
        -----------
        `usm_type` is not in NumPy's signature.

        """

        self._check_queues(ngood, nbad, nsample)
        params = tuple(_as_int64(x) for x in (ngood, nbad, nsample))
        names = ("ngood", "nbad", "nsample")
        scalar = _is_scalar_path(*params)

        for x in params[:2]:
            xp, (xa,) = _check_arrays((x,))
            if xp is not None and xp.any(xa >= _HYPERGEOM_MAX):
                raise ValueError(
                    "both ngood and nbad must be less than " f"{_HYPERGEOM_MAX}"
                )
        xp, (good, bad, sample) = _check_arrays(params)
        if xp is not None and xp.any(good + bad < sample):
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

    def integers(
        self,
        low,
        high=None,
        size=None,
        dtype=numpy.int64,
        endpoint=False,
        *,
        usm_type=None,
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

        self._check_queues(low, high)
        validate_usm_type(usm_type, allow_none=True)

        if high is None:
            low, high = 0, low
        if usm_type is None:
            # resolved here, as the bounds on the host become temporaries
            usm_type = _usm_type_of(low, high)

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
                # bounds on the device are not checked
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
                x = dpnp.asarray(
                    x.astype(wide), usm_type=usm_type, sycl_queue=queue
                )
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
    >>> x.shape, x.device.sycl_device == rng.sycl_device
    ((3,), True)

    """

    if isinstance(seed, Generator):
        return seed
    if isinstance(seed, BitGenerator):
        return Generator(seed)
    return Generator(Philox4x3210(seed, device=device, sycl_queue=sycl_queue))
