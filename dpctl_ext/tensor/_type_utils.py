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

from __future__ import annotations

import dpctl.tensor as dpt
import numpy as np

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt_ext
import dpctl_ext.tensor._tensor_impl as ti


def _all_data_types(_fp16, _fp64):
    _non_fp_types = [
        dpt.bool,
        dpt.int8,
        dpt.uint8,
        dpt.int16,
        dpt.uint16,
        dpt.int32,
        dpt.uint32,
        dpt.int64,
        dpt.uint64,
    ]
    if _fp64:
        if _fp16:
            return _non_fp_types + [
                dpt.float16,
                dpt.float32,
                dpt.float64,
                dpt.complex64,
                dpt.complex128,
            ]
        else:
            return _non_fp_types + [
                dpt.float32,
                dpt.float64,
                dpt.complex64,
                dpt.complex128,
            ]
    else:
        if _fp16:
            return _non_fp_types + [
                dpt.float16,
                dpt.float32,
                dpt.complex64,
            ]
        else:
            return _non_fp_types + [
                dpt.float32,
                dpt.complex64,
            ]


def _acceptance_fn_default_binary(
    arg1_dtype, arg2_dtype, ret_buf1_dt, ret_buf2_dt, res_dt, sycl_dev
):
    return True


def _acceptance_fn_default_unary(arg_dtype, ret_buf_dt, res_dt, sycl_dev):
    return True


def _acceptance_fn_divide(
    arg1_dtype, arg2_dtype, ret_buf1_dt, ret_buf2_dt, res_dt, sycl_dev
):
    # both are being promoted, if the kind of result is
    # different than the kind of original input dtypes,
    # we use default dtype for the resulting kind.
    # This covers, e.g. (array_dtype_i1 / array_dtype_u1)
    # result of which in divide is double (in NumPy), but
    # regular type promotion rules peg at float16
    if (ret_buf1_dt.kind != arg1_dtype.kind) and (
        ret_buf2_dt.kind != arg2_dtype.kind
    ):
        default_dt = _get_device_default_dtype(res_dt.kind, sycl_dev)
        if res_dt == default_dt:
            return True
        else:
            return False
    else:
        return True


def _acceptance_fn_negative(arg_dtype, buf_dt, res_dt, sycl_dev):
    # negative is not defined for boolean data type
    if arg_dtype.char == "?":
        raise ValueError(
            "The `negative` function, the `-` operator, is not supported "
            "for inputs of data type bool, use the `~` operator or the "
            "`logical_not` function instead"
        )
    else:
        return True


def _acceptance_fn_reciprocal(arg_dtype, buf_dt, res_dt, sycl_dev):
    # if the kind of result is different from the kind of input, we use the
    # default floating-point dtype for the resulting kind. This guarantees
    # alignment of reciprocal and divide output types.
    if buf_dt.kind != arg_dtype.kind:
        default_dt = _get_device_default_dtype(res_dt.kind, sycl_dev)
        if res_dt == default_dt:
            return True
        else:
            return False
    else:
        return True


def _acceptance_fn_subtract(
    arg1_dtype, arg2_dtype, buf1_dt, buf2_dt, res_dt, sycl_dev
):
    # subtract is not defined for boolean data type
    if arg1_dtype.char == "?" and arg2_dtype.char == "?":
        raise ValueError(
            "The `subtract` function, the `-` operator, is not supported "
            "for inputs of data type bool, use the `^` operator,  the "
            "`bitwise_xor`, or the `logical_xor` function instead"
        )
    else:
        return True


def _can_cast(
    from_: dpt.dtype, to_: dpt.dtype, _fp16: bool, _fp64: bool, casting="safe"
) -> bool:
    """
    Can `from_` be cast to `to_` safely on a device with
    fp16 and fp64 aspects as given?
    """
    if not _dtype_supported_by_device_impl(to_, _fp16, _fp64):
        return False
    can_cast_v = np.can_cast(from_, to_, casting=casting)  # ask NumPy
    if _fp16 and _fp64:
        return can_cast_v
    if not can_cast_v:
        if (
            from_.kind in "biu"
            and to_.kind in "fc"
            and _is_maximal_inexact_type(to_, _fp16, _fp64)
        ):
            return True

    return can_cast_v


def _dtype_supported_by_device_impl(
    dt: dpt.dtype, has_fp16: bool, has_fp64: bool
) -> bool:
    if has_fp64:
        if not has_fp16:
            if dt is dpt.float16:
                return False
    else:
        if dt is dpt.float64:
            return False
        elif dt is dpt.complex128:
            return False
        if not has_fp16 and dt is dpt.float16:
            return False
    return True


def _find_buf_dtype(arg_dtype, query_fn, sycl_dev, acceptance_fn):
    res_dt = query_fn(arg_dtype)
    if res_dt:
        return None, res_dt

    _fp16 = sycl_dev.has_aspect_fp16
    _fp64 = sycl_dev.has_aspect_fp64
    all_dts = _all_data_types(_fp16, _fp64)
    for buf_dt in all_dts:
        if _can_cast(arg_dtype, buf_dt, _fp16, _fp64):
            res_dt = query_fn(buf_dt)
            if res_dt:
                acceptable = acceptance_fn(arg_dtype, buf_dt, res_dt, sycl_dev)
                if acceptable:
                    return buf_dt, res_dt
                else:
                    continue

    return None, None


def _find_buf_dtype2(arg1_dtype, arg2_dtype, query_fn, sycl_dev, acceptance_fn):
    res_dt = query_fn(arg1_dtype, arg2_dtype)
    if res_dt:
        return None, None, res_dt

    _fp16 = sycl_dev.has_aspect_fp16
    _fp64 = sycl_dev.has_aspect_fp64
    all_dts = _all_data_types(_fp16, _fp64)
    for buf1_dt in all_dts:
        for buf2_dt in all_dts:
            if _can_cast(arg1_dtype, buf1_dt, _fp16, _fp64) and _can_cast(
                arg2_dtype, buf2_dt, _fp16, _fp64
            ):
                res_dt = query_fn(buf1_dt, buf2_dt)
                if res_dt:
                    ret_buf1_dt = None if buf1_dt == arg1_dtype else buf1_dt
                    ret_buf2_dt = None if buf2_dt == arg2_dtype else buf2_dt
                    if ret_buf1_dt is None or ret_buf2_dt is None:
                        return ret_buf1_dt, ret_buf2_dt, res_dt
                    else:
                        acceptable = acceptance_fn(
                            arg1_dtype,
                            arg2_dtype,
                            ret_buf1_dt,
                            ret_buf2_dt,
                            res_dt,
                            sycl_dev,
                        )
                        if acceptable:
                            return ret_buf1_dt, ret_buf2_dt, res_dt
                        else:
                            continue

    return None, None, None


def _find_buf_dtype_in_place_op(arg1_dtype, arg2_dtype, query_fn, sycl_dev):
    res_dt = query_fn(arg1_dtype, arg2_dtype)
    if res_dt:
        return None, res_dt

    _fp16 = sycl_dev.has_aspect_fp16
    _fp64 = sycl_dev.has_aspect_fp64
    if _can_cast(arg2_dtype, arg1_dtype, _fp16, _fp64, casting="same_kind"):
        res_dt = query_fn(arg1_dtype, arg1_dtype)
        if res_dt:
            return arg1_dtype, res_dt

    return None, None


def _get_device_default_dtype(dt_kind, sycl_dev):
    if dt_kind == "b":
        return dpt.dtype(ti.default_device_bool_type(sycl_dev))
    elif dt_kind == "i":
        return dpt.dtype(ti.default_device_int_type(sycl_dev))
    elif dt_kind == "u":
        return dpt.dtype(ti.default_device_uint_type(sycl_dev))
    elif dt_kind == "f":
        return dpt.dtype(ti.default_device_fp_type(sycl_dev))
    elif dt_kind == "c":
        return dpt.dtype(ti.default_device_complex_type(sycl_dev))
    raise RuntimeError


def _is_maximal_inexact_type(dt: dpt.dtype, _fp16: bool, _fp64: bool):
    """
    Return True if data type `dt` is the
    maximal size inexact data type
    """
    if _fp64:
        return dt in [dpt.float64, dpt.complex128]
    return dt in [dpt.float32, dpt.complex64]


def _to_device_supported_dtype(dt, dev):
    has_fp16 = dev.has_aspect_fp16
    has_fp64 = dev.has_aspect_fp64

    return _to_device_supported_dtype_impl(dt, has_fp16, has_fp64)


def _to_device_supported_dtype_impl(dt, has_fp16, has_fp64):
    if has_fp64:
        if not has_fp16:
            if dt is dpt.float16:
                return dpt.float32
    else:
        if dt is dpt.float64:
            return dpt.float32
        elif dt is dpt.complex128:
            return dpt.complex64
        if not has_fp16 and dt is dpt.float16:
            return dpt.float32
    return dt


class WeakBooleanType:
    """Python type representing type of Python boolean objects"""

    def __init__(self, o):
        self.o_ = o

    def get(self):
        return self.o_


class WeakIntegralType:
    """Python type representing type of Python integral objects"""

    def __init__(self, o):
        self.o_ = o

    def get(self):
        return self.o_


class WeakFloatingType:
    """Python type representing type of Python floating point objects"""

    def __init__(self, o):
        self.o_ = o

    def get(self):
        return self.o_


class WeakComplexType:
    """Python type representing type of Python complex floating point objects"""

    def __init__(self, o):
        self.o_ = o

    def get(self):
        return self.o_


def _weak_type_num_kind(o):
    _map = {"?": 0, "i": 1, "f": 2, "c": 3}
    if isinstance(o, WeakBooleanType):
        return _map["?"]
    if isinstance(o, WeakIntegralType):
        return _map["i"]
    if isinstance(o, WeakFloatingType):
        return _map["f"]
    if isinstance(o, WeakComplexType):
        return _map["c"]
    raise TypeError(
        f"Unexpected type {o} while expecting "
        "`WeakBooleanType`, `WeakIntegralType`,"
        "`WeakFloatingType`, or `WeakComplexType`."
    )


def _strong_dtype_num_kind(o):
    _map = {"b": 0, "i": 1, "u": 1, "f": 2, "c": 3}
    if not isinstance(o, dpt.dtype):
        raise TypeError
    k = o.kind
    if k in _map:
        return _map[k]
    raise ValueError(f"Unrecognized kind {k} for dtype {o}")


def _is_weak_dtype(dtype):
    return isinstance(
        dtype,
        (WeakBooleanType, WeakIntegralType, WeakFloatingType, WeakComplexType),
    )


def _resolve_weak_types(o1_dtype, o2_dtype, dev):
    """Resolves weak data type per NEP-0050"""
    if _is_weak_dtype(o1_dtype):
        if _is_weak_dtype(o2_dtype):
            raise ValueError
        o1_kind_num = _weak_type_num_kind(o1_dtype)
        o2_kind_num = _strong_dtype_num_kind(o2_dtype)
        if o1_kind_num > o2_kind_num:
            if isinstance(o1_dtype, WeakIntegralType):
                return dpt.dtype(ti.default_device_int_type(dev)), o2_dtype
            if isinstance(o1_dtype, WeakComplexType):
                if o2_dtype is dpt.float16 or o2_dtype is dpt.float32:
                    return dpt.complex64, o2_dtype
                return (
                    _to_device_supported_dtype(dpt.complex128, dev),
                    o2_dtype,
                )
            return _to_device_supported_dtype(dpt.float64, dev), o2_dtype
        else:
            return o2_dtype, o2_dtype
    elif _is_weak_dtype(o2_dtype):
        o1_kind_num = _strong_dtype_num_kind(o1_dtype)
        o2_kind_num = _weak_type_num_kind(o2_dtype)
        if o2_kind_num > o1_kind_num:
            if isinstance(o2_dtype, WeakIntegralType):
                return o1_dtype, dpt.dtype(ti.default_device_int_type(dev))
            if isinstance(o2_dtype, WeakComplexType):
                if o1_dtype is dpt.float16 or o1_dtype is dpt.float32:
                    return o1_dtype, dpt.complex64
                return o1_dtype, _to_device_supported_dtype(dpt.complex128, dev)
            return (
                o1_dtype,
                _to_device_supported_dtype(dpt.float64, dev),
            )
        else:
            return o1_dtype, o1_dtype
    else:
        return o1_dtype, o2_dtype


def _resolve_weak_types_all_py_ints(o1_dtype, o2_dtype, dev):
    """
    Resolves weak data type per NEP-0050 for comparisons and
    divide, where result type is known and special behavior
    is needed to handle mixed integer kinds and Python integers
    without overflow
    """
    if _is_weak_dtype(o1_dtype):
        if _is_weak_dtype(o2_dtype):
            raise ValueError
        o1_kind_num = _weak_type_num_kind(o1_dtype)
        o2_kind_num = _strong_dtype_num_kind(o2_dtype)
        if o1_kind_num > o2_kind_num:
            if isinstance(o1_dtype, WeakIntegralType):
                return dpt.dtype(ti.default_device_int_type(dev)), o2_dtype
            if isinstance(o1_dtype, WeakComplexType):
                if o2_dtype is dpt.float16 or o2_dtype is dpt.float32:
                    return dpt.complex64, o2_dtype
                return (
                    _to_device_supported_dtype(dpt.complex128, dev),
                    o2_dtype,
                )
            return _to_device_supported_dtype(dpt.float64, dev), o2_dtype
        else:
            if o1_kind_num == o2_kind_num and isinstance(
                o1_dtype, WeakIntegralType
            ):
                o1_val = o1_dtype.get()
                o2_iinfo = dpt.iinfo(o2_dtype)
                if (o1_val < o2_iinfo.min) or (o1_val > o2_iinfo.max):
                    return dpt.dtype(np.min_scalar_type(o1_val)), o2_dtype
            return o2_dtype, o2_dtype
    elif _is_weak_dtype(o2_dtype):
        o1_kind_num = _strong_dtype_num_kind(o1_dtype)
        o2_kind_num = _weak_type_num_kind(o2_dtype)
        if o2_kind_num > o1_kind_num:
            if isinstance(o2_dtype, WeakIntegralType):
                return o1_dtype, dpt.dtype(ti.default_device_int_type(dev))
            if isinstance(o2_dtype, WeakComplexType):
                if o1_dtype is dpt.float16 or o1_dtype is dpt.float32:
                    return o1_dtype, dpt.complex64
                return o1_dtype, _to_device_supported_dtype(dpt.complex128, dev)
            return (
                o1_dtype,
                _to_device_supported_dtype(dpt.float64, dev),
            )
        else:
            if o1_kind_num == o2_kind_num and isinstance(
                o2_dtype, WeakIntegralType
            ):
                o2_val = o2_dtype.get()
                o1_iinfo = dpt.iinfo(o1_dtype)
                if (o2_val < o1_iinfo.min) or (o2_val > o1_iinfo.max):
                    return o1_dtype, dpt.dtype(np.min_scalar_type(o2_val))
            return o1_dtype, o1_dtype
    else:
        return o1_dtype, o2_dtype


def _resolve_one_strong_two_weak_types(st_dtype, dtype1, dtype2, dev):
    """
    Resolves weak data types per NEP-0050,
    where the second and third arguments are
    permitted to be weak types
    """
    if _is_weak_dtype(st_dtype):
        raise ValueError
    if _is_weak_dtype(dtype1):
        if _is_weak_dtype(dtype2):
            kind_num1 = _weak_type_num_kind(dtype1)
            kind_num2 = _weak_type_num_kind(dtype2)
            st_kind_num = _strong_dtype_num_kind(st_dtype)

            if kind_num1 > st_kind_num:
                if isinstance(dtype1, WeakIntegralType):
                    ret_dtype1 = dpt.dtype(ti.default_device_int_type(dev))
                elif isinstance(dtype1, WeakComplexType):
                    if st_dtype is dpt.float16 or st_dtype is dpt.float32:
                        ret_dtype1 = dpt.complex64
                    ret_dtype1 = _to_device_supported_dtype(dpt.complex128, dev)
                else:
                    ret_dtype1 = _to_device_supported_dtype(dpt.float64, dev)
            else:
                ret_dtype1 = st_dtype

            if kind_num2 > st_kind_num:
                if isinstance(dtype2, WeakIntegralType):
                    ret_dtype2 = dpt.dtype(ti.default_device_int_type(dev))
                elif isinstance(dtype2, WeakComplexType):
                    if st_dtype is dpt.float16 or st_dtype is dpt.float32:
                        ret_dtype2 = dpt.complex64
                    ret_dtype2 = _to_device_supported_dtype(dpt.complex128, dev)
                else:
                    ret_dtype2 = _to_device_supported_dtype(dpt.float64, dev)
            else:
                ret_dtype2 = st_dtype

            return ret_dtype1, ret_dtype2

        max_dt_num_kind, max_dtype = max(
            [
                (_strong_dtype_num_kind(st_dtype), st_dtype),
                (_strong_dtype_num_kind(dtype2), dtype2),
            ]
        )
        dt1_kind_num = _weak_type_num_kind(dtype1)
        if dt1_kind_num > max_dt_num_kind:
            if isinstance(dtype1, WeakIntegralType):
                return dpt.dtype(ti.default_device_int_type(dev)), dtype2
            if isinstance(dtype1, WeakComplexType):
                if max_dtype is dpt.float16 or max_dtype is dpt.float32:
                    return dpt.complex64, dtype2
                return (
                    _to_device_supported_dtype(dpt.complex128, dev),
                    dtype2,
                )
            return _to_device_supported_dtype(dpt.float64, dev), dtype2
        else:
            return max_dtype, dtype2
    elif _is_weak_dtype(dtype2):
        max_dt_num_kind, max_dtype = max(
            [
                (_strong_dtype_num_kind(st_dtype), st_dtype),
                (_strong_dtype_num_kind(dtype1), dtype1),
            ]
        )
        dt2_kind_num = _weak_type_num_kind(dtype2)
        if dt2_kind_num > max_dt_num_kind:
            if isinstance(dtype2, WeakIntegralType):
                return dtype1, dpt.dtype(ti.default_device_int_type(dev))
            if isinstance(dtype2, WeakComplexType):
                if max_dtype is dpt.float16 or max_dtype is dpt.float32:
                    return dtype1, dpt.complex64
                return (
                    dtype1,
                    _to_device_supported_dtype(dpt.complex128, dev),
                )
            return dtype1, _to_device_supported_dtype(dpt.float64, dev)
        else:
            return dtype1, max_dtype
    else:
        # both are strong dtypes
        # return unmodified
        return dtype1, dtype2


def _resolve_one_strong_one_weak_types(st_dtype, dtype, dev):
    """Resolves one weak data type with one strong data type per NEP-0050"""
    if _is_weak_dtype(st_dtype):
        raise ValueError
    if _is_weak_dtype(dtype):
        st_kind_num = _strong_dtype_num_kind(st_dtype)
        kind_num = _weak_type_num_kind(dtype)
        if kind_num > st_kind_num:
            if isinstance(dtype, WeakIntegralType):
                return dpt.dtype(ti.default_device_int_type(dev))
            if isinstance(dtype, WeakComplexType):
                if st_dtype is dpt.float16 or st_dtype is dpt.float32:
                    return dpt.complex64
                return _to_device_supported_dtype(dpt.complex128, dev)
            return _to_device_supported_dtype(dpt.float64, dev)
        else:
            return st_dtype
    else:
        return dtype


class finfo_object:
    """
    `numpy.finfo` subclass which returns Python floating-point scalars for
    `eps`, `max`, `min`, and `smallest_normal` attributes.
    """

    def __init__(self, dtype):
        _supported_dtype([dpt.dtype(dtype)])
        self._finfo = np.finfo(dtype)

    @property
    def bits(self):
        """Number of bits occupied by the real-valued floating-point data type."""
        return int(self._finfo.bits)

    @property
    def smallest_normal(self):
        """
        Smallest positive real-valued floating-point number with full
        precision.
        """
        return float(self._finfo.smallest_normal)

    @property
    def tiny(self):
        """An alias for `smallest_normal`"""
        return float(self._finfo.tiny)

    @property
    def eps(self):
        """
        Difference between 1.0 and the next smallest representable real-valued
        floating-point number larger than 1.0 according to the IEEE-754
        standard.
        """
        return float(self._finfo.eps)

    @property
    def epsneg(self):
        """
        Difference between 1.0 and the next smallest representable real-valued
        floating-point number smaller than 1.0 according to the IEEE-754
        standard.
        """
        return float(self._finfo.epsneg)

    @property
    def min(self):
        """Smallest representable real-valued number."""
        return float(self._finfo.min)

    @property
    def max(self):
        """Largest representable real-valued number."""
        return float(self._finfo.max)

    @property
    def resolution(self):
        """The approximate decimal resolution of this type."""
        return float(self._finfo.resolution)

    @property
    def precision(self):
        """
        The approximate number of decimal digits to which this kind of
        floating point type is precise.
        """
        return float(self._finfo.precision)

    @property
    def dtype(self):
        """
        The dtype for which finfo returns information. For complex input, the
        returned dtype is the associated floating point dtype for its real and
        complex components.
        """
        return self._finfo.dtype

    def __str__(self):
        return self._finfo.__str__()

    def __repr__(self):
        return self._finfo.__repr__()


def can_cast(from_, to, /, *, casting="safe") -> bool:
    """can_cast(from, to, casting="safe")

    Determines if one data type can be cast to another data type according \
    to Type Promotion Rules.

    Args:
       from_ (Union[usm_ndarray, dtype]):
           source data type. If `from_` is an array, a device-specific type
           promotion rules apply.
       to (dtype):
           target data type
       casting (Optional[str]):
            controls what kind of data casting may occur.

                * "no" means data types should not be cast at all.
                * "safe" means only casts that preserve values are allowed.
                * "same_kind" means only safe casts and casts within a kind,
                  like `float64` to `float32`, are allowed.
                * "unsafe" means any data conversion can be done.

            Default: `"safe"`.

    Returns:
        bool:
            Gives `True` if cast can occur according to the casting rule.

    Device-specific type promotion rules take into account which data type are
    and are not supported by a specific device.
    """
    if isinstance(to, dpt.usm_ndarray):
        raise TypeError(f"Expected `dpt.dtype` type, got {type(to)}.")

    dtype_to = dpt.dtype(to)
    _supported_dtype([dtype_to])

    if isinstance(from_, dpt.usm_ndarray):
        dtype_from = from_.dtype
        return _can_cast(
            dtype_from,
            dtype_to,
            from_.sycl_device.has_aspect_fp16,
            from_.sycl_device.has_aspect_fp64,
            casting=casting,
        )
    else:
        dtype_from = dpt.dtype(from_)
        _supported_dtype([dtype_from])
        # query casting as if all dtypes are supported
        return _can_cast(dtype_from, dtype_to, True, True, casting=casting)


def result_type(*arrays_and_dtypes):
    """
    result_type(*arrays_and_dtypes)

    Returns the dtype that results from applying the Type Promotion Rules to \
        the arguments.

    Args:
        arrays_and_dtypes (Union[usm_ndarray, dtype]):
            An arbitrary length sequence of usm_ndarray objects or dtypes.

    Returns:
        dtype:
            The dtype resulting from an operation involving the
            input arrays and dtypes.
    """
    dtypes = []
    devices = []
    weak_dtypes = []
    for arg_i in arrays_and_dtypes:
        if isinstance(arg_i, dpt.usm_ndarray):
            devices.append(arg_i.sycl_device)
            dtypes.append(arg_i.dtype)
        elif isinstance(arg_i, int):
            weak_dtypes.append(WeakIntegralType(arg_i))
        elif isinstance(arg_i, float):
            weak_dtypes.append(WeakFloatingType(arg_i))
        elif isinstance(arg_i, complex):
            weak_dtypes.append(WeakComplexType(arg_i))
        elif isinstance(arg_i, bool):
            weak_dtypes.append(WeakBooleanType(arg_i))
        else:
            dt = dpt.dtype(arg_i)
            _supported_dtype([dt])
            dtypes.append(dt)

    has_fp16 = True
    has_fp64 = True
    target_dev = None
    if devices:
        inspected = False
        for d in devices:
            if inspected:
                unsame_fp16_support = d.has_aspect_fp16 != has_fp16
                unsame_fp64_support = d.has_aspect_fp64 != has_fp64
                if unsame_fp16_support or unsame_fp64_support:
                    raise ValueError(
                        "Input arrays reside on devices "
                        "with different device supports; "
                        "unable to determine which "
                        "device-specific type promotion rules "
                        "to use."
                    )
            else:
                has_fp16 = d.has_aspect_fp16
                has_fp64 = d.has_aspect_fp64
                target_dev = d
                inspected = True

    if not dtypes and weak_dtypes:
        dtypes.append(weak_dtypes[0].get())

    if not (has_fp16 and has_fp64):
        for dt in dtypes:
            if not _dtype_supported_by_device_impl(dt, has_fp16, has_fp64):
                raise ValueError(
                    f"Argument {dt} is not supported by the device"
                )
        res_dt = np.result_type(*dtypes)
        res_dt = _to_device_supported_dtype_impl(res_dt, has_fp16, has_fp64)
        for wdt in weak_dtypes:
            pair = _resolve_weak_types(wdt, res_dt, target_dev)
            res_dt = np.result_type(*pair)
            res_dt = _to_device_supported_dtype_impl(res_dt, has_fp16, has_fp64)
    else:
        res_dt = np.result_type(*dtypes)
        if weak_dtypes:
            weak_dt_obj = [wdt.get() for wdt in weak_dtypes]
            res_dt = np.result_type(res_dt, *weak_dt_obj)

    return res_dt


def iinfo(dtype, /):
    """iinfo(dtype)

    Returns machine limits for integer data types.

    Args:
        dtype (dtype, usm_ndarray):
            integer dtype or
            an array with integer dtype.

    Returns:
        iinfo_object:
            An object with the following attributes:

            * bits: int
                number of bits occupied by the data type
            * max: int
                largest representable number.
            * min: int
                smallest representable number.
            * dtype: dtype
                integer data type.
    """
    if isinstance(dtype, dpt.usm_ndarray):
        dtype = dtype.dtype
    _supported_dtype([dpt.dtype(dtype)])
    return np.iinfo(dtype)


def finfo(dtype, /):
    """finfo(type)

    Returns machine limits for floating-point data types.

    Args:
        dtype (dtype, usm_ndarray): floating-point dtype or
            an array with floating point data type.
            If complex, the information is about its component
            data type.

    Returns:
        finfo_object:
            an object have the following attributes:

                * bits: int
                    number of bits occupied by dtype.
                * eps: float
                    difference between 1.0 and the next smallest representable
                    real-valued floating-point number larger than 1.0 according
                    to the IEEE-754 standard.
                * max: float
                    largest representable real-valued number.
                * min: float
                    smallest representable real-valued number.
                * smallest_normal: float
                    smallest positive real-valued floating-point number with
                    full precision.
                * dtype: dtype
                    real-valued floating-point data type.

    """
    if isinstance(dtype, dpt.usm_ndarray):
        dtype = dtype.dtype
    _supported_dtype([dpt.dtype(dtype)])
    return finfo_object(dtype)


def _supported_dtype(dtypes):
    for dtype in dtypes:
        if dtype.char not in "?bBhHiIlLqQefdFD":
            raise ValueError(f"Dpctl doesn't support dtype {dtype}.")
    return True


def isdtype(dtype, kind):
    """isdtype(dtype, kind)

    Returns a boolean indicating whether a provided `dtype` is
    of a specified data type `kind`.

    See [array API](array_api) for more information.

    [array_api]: https://data-apis.org/array-api/latest/
    """

    if not isinstance(dtype, np.dtype):
        raise TypeError(f"Expected instance of `dpt.dtype`, got {dtype}")

    if isinstance(kind, np.dtype):
        return dtype == kind

    elif isinstance(kind, str):
        if kind == "bool":
            return dtype == np.dtype("bool")
        elif kind == "signed integer":
            return dtype.kind == "i"
        elif kind == "unsigned integer":
            return dtype.kind == "u"
        elif kind == "integral":
            return dtype.kind in "iu"
        elif kind == "real floating":
            return dtype.kind == "f"
        elif kind == "complex floating":
            return dtype.kind == "c"
        elif kind == "numeric":
            return dtype.kind in "iufc"
        else:
            raise ValueError(f"Unrecognized data type kind: {kind}")

    elif isinstance(kind, tuple):
        return any(isdtype(dtype, k) for k in kind)

    else:
        raise TypeError(f"Unsupported data type kind: {kind}")


def _default_accumulation_dtype(inp_dt, q):
    """Gives default output data type for given input data
    type `inp_dt` when accumulation is performed on queue `q`
    """
    inp_kind = inp_dt.kind
    if inp_kind in "bi":
        res_dt = dpt.dtype(ti.default_device_int_type(q))
        if inp_dt.itemsize > res_dt.itemsize:
            res_dt = inp_dt
    elif inp_kind in "u":
        res_dt = dpt.dtype(ti.default_device_uint_type(q))
        res_ii = dpt.iinfo(res_dt)
        inp_ii = dpt.iinfo(inp_dt)
        if inp_ii.min >= res_ii.min and inp_ii.max <= res_ii.max:
            pass
        else:
            res_dt = inp_dt
    elif inp_kind in "fc":
        res_dt = inp_dt

    return res_dt


def _default_accumulation_dtype_fp_types(inp_dt, q):
    """Gives default output data type for given input data
    type `inp_dt` when accumulation is performed on queue `q`
    and the accumulation supports only floating-point data types
    """
    inp_kind = inp_dt.kind
    if inp_kind in "biu":
        res_dt = dpt.dtype(ti.default_device_fp_type(q))
        can_cast_v = dpt_ext.can_cast(inp_dt, res_dt)
        if not can_cast_v:
            _fp64 = q.sycl_device.has_aspect_fp64
            res_dt = dpt.float64 if _fp64 else dpt.float32
    elif inp_kind in "f":
        res_dt = inp_dt
    elif inp_kind in "c":
        raise ValueError("function not defined for complex types")

    return res_dt


__all__ = [
    "_find_buf_dtype",
    "_find_buf_dtype2",
    "_to_device_supported_dtype",
    "_acceptance_fn_default_unary",
    "_acceptance_fn_reciprocal",
    "_acceptance_fn_default_binary",
    "_acceptance_fn_divide",
    "_acceptance_fn_negative",
    "_acceptance_fn_subtract",
    "_resolve_one_strong_one_weak_types",
    "_resolve_one_strong_two_weak_types",
    "_resolve_weak_types",
    "_resolve_weak_types_all_py_ints",
    "_weak_type_num_kind",
    "_strong_dtype_num_kind",
    "can_cast",
    "finfo",
    "iinfo",
    "isdtype",
    "result_type",
    "WeakBooleanType",
    "WeakIntegralType",
    "WeakFloatingType",
    "WeakComplexType",
    "_default_accumulation_dtype",
    "_default_accumulation_dtype_fp_types",
    "_find_buf_dtype_in_place_op",
]
