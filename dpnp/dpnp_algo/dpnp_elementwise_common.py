# *****************************************************************************
# Copyright (c) 2023, Intel Corporation
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

import warnings
from functools import wraps

import dpctl.tensor as dpt
import dpctl.tensor._copy_utils as dtc
import dpctl.tensor._type_utils as dtu
import dpctl.utils as dpu
import numpy
from dpctl.tensor._elementwise_common import (
    BinaryElementwiseFunc,
    UnaryElementwiseFunc,
)
from dpctl.tensor._scalar_utils import (
    _get_dtype,
    _get_shape,
    _validate_dtype,
)

import dpctl_ext.tensor._tensor_impl as dti
import dpnp
import dpnp.backend.extensions.vm._vm_impl as vmi
from dpnp.dpnp_array import dpnp_array
from dpnp.dpnp_utils import get_usm_allocations
from dpnp.dpnp_utils.dpnp_utils_common import (
    find_buf_dtype_3out,
    find_buf_dtype_4out,
)

__all__ = [
    "DPNPI0",
    "DPNPAngle",
    "DPNPBinaryFunc",
    "DPNPBinaryFuncOutKw",
    "DPNPBinaryTwoOutputsFunc",
    "DPNPDeprecatedUnaryFunc",
    "DPNPImag",
    "DPNPReal",
    "DPNPRound",
    "DPNPSinc",
    "DPNPUnaryFunc",
    "DPNPUnaryTwoOutputsFunc",
    "acceptance_fn_gcd_lcm",
    "acceptance_fn_negative",
    "acceptance_fn_positive",
    "acceptance_fn_sign",
    "acceptance_fn_subtract",
    "resolve_weak_types_2nd_arg_int",
]


class DPNPUnaryFunc(UnaryElementwiseFunc):
    """
    Class that implements unary element-wise functions.

    Parameters
    ----------
    name : {str}
        Name of the unary function
    result_type_resolver_fn : {callable}
        Function that takes dtype of the input and returns the dtype of
        the result if the implementation functions supports it, or
        returns `None` otherwise.
    unary_dp_impl_fn : {callable}
        Data-parallel implementation function with signature
        `impl_fn(src: usm_ndarray, dst: usm_ndarray,
            sycl_queue: SyclQueue, depends: Optional[List[SyclEvent]])`
        where the `src` is the argument array, `dst` is the
        array to be populated with function values, effectively
        evaluating `dst = func(src)`.
        The `impl_fn` is expected to return a 2-tuple of `SyclEvent`s.
        The first event corresponds to data-management host tasks,
        including lifetime management of argument Python objects to ensure
        that their associated USM allocation is not freed before offloaded
        computational tasks complete execution, while the second event
        corresponds to computational tasks associated with function evaluation.
    docs : {str}
        Documentation string for the unary function.
    mkl_fn_to_call : {None, str}
        Check input arguments to answer if function from OneMKL VM library
        can be used.
    mkl_impl_fn : {None, str}
        Function from OneMKL VM library to call.
    acceptance_fn : {None, callable}, optional
        Function to influence type promotion behavior of this unary
        function. The function takes 4 arguments:
            arg_dtype - Data type of the first argument
            buf_dtype - Data type the argument would be cast to
            res_dtype - Data type of the output array with function values
            sycl_dev - The :class:`dpctl.SyclDevice` where the function
                evaluation is carried out.
        The function is invoked when the argument of the unary function
        requires casting, e.g. the argument of `dpctl.tensor.log` is an
        array with integral data type.

    """

    def __init__(
        self,
        name,
        result_type_resolver_fn,
        unary_dp_impl_fn,
        docs,
        mkl_fn_to_call=None,
        mkl_impl_fn=None,
        acceptance_fn=None,
    ):
        def _call_func(src, dst, sycl_queue, depends=None):
            """
            A callback to register in UnaryElementwiseFunc class of
            dpctl.tensor
            """

            if depends is None:
                depends = []

            if vmi._is_available() and not (
                mkl_impl_fn is None or mkl_fn_to_call is None
            ):
                if getattr(vmi, mkl_fn_to_call)(sycl_queue, src, dst):
                    # call pybind11 extension for unary function from OneMKL VM
                    return getattr(vmi, mkl_impl_fn)(
                        sycl_queue, src, dst, depends
                    )
            return unary_dp_impl_fn(src, dst, sycl_queue, depends)

        super().__init__(
            name,
            result_type_resolver_fn,
            _call_func,
            docs,
            acceptance_fn=acceptance_fn,
        )
        self.__name__ = "DPNPUnaryFunc"

    def __call__(
        self,
        x,
        /,
        out=None,
        *,
        where=True,
        order="K",
        dtype=None,
        subok=True,
        **kwargs,
    ):
        if kwargs:
            raise NotImplementedError(
                f"Requested function={self.name_} with kwargs={kwargs} "
                "isn't currently supported."
            )
        elif where is not True:
            raise NotImplementedError(
                f"Requested function={self.name_} with where={where} "
                "isn't currently supported."
            )
        elif subok is not True:
            raise NotImplementedError(
                f"Requested function={self.name_} with subok={subok} "
                "isn't currently supported."
            )
        elif not dpnp.is_supported_array_type(x):
            raise TypeError(
                "Input array must be any of supported type, "
                f"but got {type(x)}"
            )
        elif dtype is not None and out is not None:
            raise TypeError(
                f"Requested function={self.name_} only takes `out` or `dtype` "
                "as an argument, but both were provided."
            )

        if order is None:
            order = "K"
        elif order in "afkcAFKC":
            order = order.upper()
        else:
            raise ValueError(
                "order must be one of 'C', 'F', 'A', or 'K' " f"(got '{order}')"
            )

        x_usm = dpnp.get_usm_ndarray(x)
        if dtype is not None:
            x_usm = dpt.astype(x_usm, dtype, copy=False)

        out = self._unpack_out_kw(out)
        out_usm = None if out is None else dpnp.get_usm_ndarray(out)

        res_usm = super().__call__(x_usm, out=out_usm, order=order)
        if out is not None and isinstance(out, dpnp_array):
            return out
        return dpnp_array._create_from_usm_ndarray(res_usm)

    def _unpack_out_kw(self, out):
        """Unpack `out` keyword if passed as a tuple."""

        if isinstance(out, tuple):
            if len(out) != self.nout:
                raise ValueError(
                    "'out' tuple must have exactly one entry per ufunc output"
                )
            return out[0]
        return out


class DPNPDeprecatedUnaryFunc(DPNPUnaryFunc):
    """
    Class that implements a deprecated unary element-wise function.

    Parameters
    ----------
    deprecated_msg : {str, None}, optional
        Warning message to emit. If None, no warning is issued.

        Default: ``None``.

    """

    def __init__(self, *args, deprecated_msg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._deprecated_msg = deprecated_msg

    @wraps(DPNPUnaryFunc.__call__)
    def __call__(self, *args, **kwargs):
        if self._deprecated_msg:
            warnings.warn(
                self._deprecated_msg, DeprecationWarning, stacklevel=2
            )
        return super().__call__(*args, **kwargs)


class DPNPUnaryTwoOutputsFunc(UnaryElementwiseFunc):
    """
    Class that implements unary element-wise functions with two output arrays.

    Parameters
    ----------
    name : {str}
        Name of the unary function
    result_type_resolver_fn : {callable}
        Function that takes dtype of the input and returns the dtype of
        the result if the implementation functions supports it, or
        returns `None` otherwise.
    unary_dp_impl_fn : {callable}
        Data-parallel implementation function with signature
        `impl_fn(src: usm_ndarray, dst: usm_ndarray,
            sycl_queue: SyclQueue, depends: Optional[List[SyclEvent]])`
        where the `src` is the argument array, `dst` is the
        array to be populated with function values, effectively
        evaluating `dst = func(src)`.
        The `impl_fn` is expected to return a 2-tuple of `SyclEvent`s.
        The first event corresponds to data-management host tasks,
        including lifetime management of argument Python objects to ensure
        that their associated USM allocation is not freed before offloaded
        computational tasks complete execution, while the second event
        corresponds to computational tasks associated with function evaluation.
    docs : {str}
        Documentation string for the unary function.
    mkl_fn_to_call : {None, str}
        Check input arguments to answer if function from OneMKL VM library
        can be used.
    mkl_impl_fn : {None, str}
        Function from OneMKL VM library to call.

    """

    def __init__(
        self,
        name,
        result_type_resolver_fn,
        unary_dp_impl_fn,
        docs,
        mkl_fn_to_call=None,
        mkl_impl_fn=None,
    ):
        def _call_func(src, dst1, dst2, sycl_queue, depends=None):
            """A callback to register in UnaryElementwiseFunc class."""

            if depends is None:
                depends = []

            if vmi._is_available() and not (
                mkl_impl_fn is None or mkl_fn_to_call is None
            ):
                if getattr(vmi, mkl_fn_to_call)(sycl_queue, src, dst1, dst2):
                    # call pybind11 extension for unary function from OneMKL VM
                    return getattr(vmi, mkl_impl_fn)(
                        sycl_queue, src, dst1, dst2, depends
                    )
            return unary_dp_impl_fn(src, dst1, dst2, sycl_queue, depends)

        super().__init__(
            name,
            result_type_resolver_fn,
            _call_func,
            docs,
        )
        self.__name__ = "DPNPUnaryTwoOutputsFunc"

    @property
    def nout(self):
        """Returns the number of arguments treated as outputs."""
        return 2

    @property
    def types(self):
        """
        Returns information about types supported by implementation function,
        using NumPy's character encoding for data type.

        Examples
        --------
        >>> import dpnp as np
        >>> np.frexp.types
        ['e->ei', 'f->fi', 'd->di']

        """

        types = self.types_
        if not types:
            types = []
            for dt1 in dtu._all_data_types(True, True):
                dt2 = self.result_type_resolver_fn_(dt1)
                if all(dt for dt in dt2):
                    types.append(f"{dt1.char}->{dt2[0].char}{dt2[1].char}")
            self.types_ = types
        return types

    def __call__(
        self,
        x,
        out1=None,
        out2=None,
        /,
        *,
        out=(None, None),
        where=True,
        order="K",
        dtype=None,
        subok=True,
        **kwargs,
    ):
        if kwargs:
            raise NotImplementedError(
                f"Requested function={self.name_} with kwargs={kwargs} "
                "isn't currently supported."
            )
        elif where is not True:
            raise NotImplementedError(
                f"Requested function={self.name_} with where={where} "
                "isn't currently supported."
            )
        elif dtype is not None:
            raise NotImplementedError(
                f"Requested function={self.name_} with dtype={dtype} "
                "isn't currently supported."
            )
        elif subok is not True:
            raise NotImplementedError(
                f"Requested function={self.name_} with subok={subok} "
                "isn't currently supported."
            )

        x = dpnp.get_usm_ndarray(x)
        exec_q = x.sycl_queue

        if order is None:
            order = "K"
        elif order in "afkcAFKC":
            order = order.upper()
            if order == "A":
                order = "F" if x.flags.f_contiguous else "C"
        else:
            raise ValueError(
                "order must be one of 'C', 'F', 'A', or 'K' " f"(got '{order}')"
            )

        buf_dt, res1_dt, res2_dt = find_buf_dtype_3out(
            x.dtype,
            self.get_type_result_resolver_function(),
            x.sycl_device,
        )
        if res1_dt is None or res2_dt is None:
            raise ValueError(
                f"function '{self.name_}' does not support input type "
                f"({x.dtype}), "
                "and the input could not be safely coerced to any "
                "supported types according to the casting rule ''safe''."
            )

        if not isinstance(out, tuple):
            raise TypeError("'out' must be a tuple of arrays")

        if len(out) != self.nout:
            raise ValueError(
                "'out' tuple must have exactly one entry per ufunc output"
            )

        if not (out1 is None and out2 is None):
            if all(res is None for res in out):
                out = (out1, out2)
            else:
                raise TypeError(
                    "cannot specify 'out' as both a positional and keyword argument"
                )

        orig_out, out = list(out), list(out)
        res_dts = [res1_dt, res2_dt]

        for i in range(self.nout):
            if out[i] is None:
                continue

            res = dpnp.get_usm_ndarray(out[i])
            if not res.flags.writable:
                raise ValueError("output array is read-only")

            if res.shape != x.shape:
                raise ValueError(
                    "The shape of input and output arrays are inconsistent. "
                    f"Expected output shape is {x.shape}, got {res.shape}"
                )

            if dpu.get_execution_queue((exec_q, res.sycl_queue)) is None:
                raise dpnp.exceptions.ExecutionPlacementError(
                    "Input and output allocation queues are not compatible"
                )

            res_dt = res_dts[i]
            if res_dt != res.dtype:
                if not dpnp.can_cast(res_dt, res.dtype, casting="same_kind"):
                    raise TypeError(
                        f"Cannot cast ufunc '{self.name_}' output {i + 1} from "
                        f"{res_dt} to {res.dtype} with casting rule 'same_kind'"
                    )

                # Allocate a temporary buffer with the required dtype
                out[i] = dpt.empty_like(res, dtype=res_dt)
            elif (
                buf_dt is None
                and dti._array_overlap(x, res)
                and not dti._same_logical_tensors(x, res)
            ):
                # Allocate a temporary buffer to avoid memory overlapping.
                # Note if `buf_dt` is not None, a temporary copy of `x` will be
                # created, so the array overlap check isn't needed.
                out[i] = dpt.empty_like(res)

        _manager = dpu.SequentialOrderManager[exec_q]
        dep_evs = _manager.submitted_events

        # Cast input array to the supported type if needed
        if buf_dt is not None:
            if order == "K":
                buf = dtc._empty_like_orderK(x, buf_dt)
            else:
                buf = dpt.empty_like(x, dtype=buf_dt, order=order)

            ht_copy_ev, copy_ev = dti._copy_usm_ndarray_into_usm_ndarray(
                src=x, dst=buf, sycl_queue=exec_q, depends=dep_evs
            )
            _manager.add_event_pair(ht_copy_ev, copy_ev)

            x = buf
            dep_evs = copy_ev

        # Allocate a buffer for the output arrays if needed
        for i in range(self.nout):
            if out[i] is None:
                res_dt = res_dts[i]
                if order == "K":
                    out[i] = dtc._empty_like_orderK(x, res_dt)
                else:
                    out[i] = dpt.empty_like(x, dtype=res_dt, order=order)

        # Call the unary function with input and output arrays
        ht_unary_ev, unary_ev = self.get_implementation_function()(
            x,
            dpnp.get_usm_ndarray(out[0]),
            dpnp.get_usm_ndarray(out[1]),
            sycl_queue=exec_q,
            depends=_manager.submitted_events,
        )
        _manager.add_event_pair(ht_unary_ev, unary_ev)

        for i in range(self.nout):
            orig_res, res = orig_out[i], out[i]
            if not (orig_res is None or orig_res is res):
                # Copy the out data from temporary buffer to original memory
                ht_copy_ev, copy_ev = dti._copy_usm_ndarray_into_usm_ndarray(
                    src=res,
                    dst=dpnp.get_usm_ndarray(orig_res),
                    sycl_queue=exec_q,
                    depends=[unary_ev],
                )
                _manager.add_event_pair(ht_copy_ev, copy_ev)
                res = out[i] = orig_res

            if not isinstance(res, dpnp_array):
                # Always return dpnp.ndarray
                out[i] = dpnp_array._create_from_usm_ndarray(res)
        return tuple(out)


class DPNPBinaryFunc(BinaryElementwiseFunc):
    """
    Class that implements binary element-wise functions.

    Args:
    name : {str}
        Name of the binary function
    result_type_resovle_fn : {callable}
        Function that takes dtype of the input and returns the dtype of
        the result if the implementation functions supports it, or
        returns `None` otherwise..
    binary_dp_impl_fn : {callable}
        Data-parallel implementation function with signature
        `impl_fn(src1: usm_ndarray, src2: usm_ndarray, dst: usm_ndarray,
            sycl_queue: SyclQueue, depends: Optional[List[SyclEvent]])`
        where the `src1` and `src2` are the argument arrays, `dst` is the
        array to be populated with function values,
        i.e. `dst=func(src1, src2)`.
        The `impl_fn` is expected to return a 2-tuple of `SyclEvent`s.
        The first event corresponds to data-management host tasks,
        including lifetime management of argument Python objects to ensure
        that their associated USM allocation is not freed before offloaded
        computational tasks complete execution, while the second event
        corresponds to computational tasks associated with function
        evaluation.
    docs : {str}
        Documentation string for the binary function.
    mkl_fn_to_call : {None, str}
        Check input arguments to answer if function from OneMKL VM library
        can be used.
    mkl_impl_fn : {None, str}
        Function from OneMKL VM library to call.
    binary_inplace_fn : {None, callable}, optional
        Data-parallel implementation function with signature
        `impl_fn(src: usm_ndarray, dst: usm_ndarray,
            sycl_queue: SyclQueue, depends: Optional[List[SyclEvent]])`
        where the `src` is the argument array, `dst` is the
        array to be populated with function values,
        i.e. `dst=func(dst, src)`.
        The `impl_fn` is expected to return a 2-tuple of `SyclEvent`s.
        The first event corresponds to data-management host tasks,
        including async lifetime management of Python arguments,
        while the second event corresponds to computational tasks
        associated with function evaluation.
    acceptance_fn : {None, callable}, optional
        Function to influence type promotion behavior of this binary
        function. The function takes 6 arguments:
            arg1_dtype - Data type of the first argument
            arg2_dtype - Data type of the second argument
            ret_buf1_dtype - Data type the first argument would be cast to
            ret_buf2_dtype - Data type the second argument would be cast to
            res_dtype - Data type of the output array with function values
            sycl_dev - The :class:`dpctl.SyclDevice` where the function
                evaluation is carried out.
        The function is only called when both arguments of the binary
        function require casting, e.g. both arguments of
        `dpctl.tensor.logaddexp` are arrays with integral data type.
    weak_type_resolver : {None, callable}, optional
        Function to influence type promotion behavior for Python scalar types
        of this binary function. The function takes 3 arguments:
            o1_dtype - Data type or Python scalar type of the first argument
            o2_dtype - Data type or Python scalar type of of the second argument
            sycl_dev - The :class:`dpctl.SyclDevice` where the function
                evaluation is carried out.
        One of `o1_dtype` and `o2_dtype` must be a ``dtype`` instance.

    """

    def __init__(
        self,
        name,
        result_type_resolver_fn,
        binary_dp_impl_fn,
        docs,
        mkl_fn_to_call=None,
        mkl_impl_fn=None,
        binary_inplace_fn=None,
        acceptance_fn=None,
        weak_type_resolver=None,
    ):
        def _call_func(src1, src2, dst, sycl_queue, depends=None):
            """
            A callback to register in UnaryElementwiseFunc class of
            dpctl.tensor
            """

            if depends is None:
                depends = []

            if vmi._is_available() and not (
                mkl_impl_fn is None or mkl_fn_to_call is None
            ):
                if getattr(vmi, mkl_fn_to_call)(sycl_queue, src1, src2, dst):
                    # call pybind11 extension for binary function from OneMKL VM
                    return getattr(vmi, mkl_impl_fn)(
                        sycl_queue, src1, src2, dst, depends
                    )
            return binary_dp_impl_fn(src1, src2, dst, sycl_queue, depends)

        super().__init__(
            name,
            result_type_resolver_fn,
            _call_func,
            docs,
            binary_inplace_fn,
            acceptance_fn=acceptance_fn,
            weak_type_resolver=weak_type_resolver,
        )
        self.__name__ = "DPNPBinaryFunc"

    def __call__(
        self,
        x1,
        x2,
        /,
        out=None,
        *,
        where=True,
        order="K",
        dtype=None,
        subok=True,
        **kwargs,
    ):
        dpnp.check_supported_arrays_type(
            x1, x2, scalar_type=True, all_scalars=False
        )
        if kwargs:
            raise NotImplementedError(
                f"Requested function={self.name_} with kwargs={kwargs} "
                "isn't currently supported."
            )
        elif where is not True:
            raise NotImplementedError(
                f"Requested function={self.name_} with where={where} "
                "isn't currently supported."
            )
        elif subok is not True:
            raise NotImplementedError(
                f"Requested function={self.name_} with subok={subok} "
                "isn't currently supported."
            )
        elif dtype is not None and out is not None:
            raise TypeError(
                f"Requested function={self.name_} only takes `out` or `dtype` "
                "as an argument, but both were provided."
            )

        x1_usm = dpnp.get_usm_ndarray_or_scalar(x1)
        x2_usm = dpnp.get_usm_ndarray_or_scalar(x2)

        if isinstance(out, tuple):
            if len(out) != self.nout:
                raise ValueError(
                    "'out' tuple must have exactly one entry per ufunc output"
                )
            out = out[0]
        out_usm = None if out is None else dpnp.get_usm_ndarray(out)

        if (
            isinstance(x1, dpnp_array)
            and x1 is out
            and order == "K"
            and dtype is None
        ):
            # in-place operation
            super()._inplace_op(x1_usm, x2_usm)
            return x1

        if order is None:
            order = "K"
        elif order in "afkcAFKC":
            order = order.upper()
        else:
            raise ValueError(
                "order must be one of 'C', 'F', 'A', or 'K' (got '{order}')"
            )

        if dtype is not None:
            if dpnp.isscalar(x1):
                x1_usm = dpt.asarray(
                    x1,
                    dtype=dtype,
                    sycl_queue=x2.sycl_queue,
                    usm_type=x2.usm_type,
                )
                x2_usm = dpt.astype(x2_usm, dtype, copy=False)
            elif dpnp.isscalar(x2):
                x1_usm = dpt.astype(x1_usm, dtype, copy=False)
                x2_usm = dpt.asarray(
                    x2,
                    dtype=dtype,
                    sycl_queue=x1.sycl_queue,
                    usm_type=x1.usm_type,
                )
            else:
                x1_usm = dpt.astype(x1_usm, dtype, copy=False)
                x2_usm = dpt.astype(x2_usm, dtype, copy=False)

        res_usm = super().__call__(x1_usm, x2_usm, out=out_usm, order=order)

        if out is not None and isinstance(out, dpnp_array):
            return out
        return dpnp_array._create_from_usm_ndarray(res_usm)

    def outer(
        self,
        x1,
        x2,
        out=None,
        where=True,
        order="K",
        dtype=None,
        subok=True,
        **kwargs,
    ):
        """
        Apply the ufunc op to all pairs (a, b) with a in A and b in B.

        Parameters
        ----------
        x1 : {dpnp.ndarray, usm_ndarray}
            First input array.
        x2 : {dpnp.ndarray, usm_ndarray}
            Second input array.
        out : {None, dpnp.ndarray, usm_ndarray}, optional
            Output array to populate.
            Array must have the correct shape and the expected data type.
        order : {None, "C", "F", "A", "K"}, optional
            Memory layout of the newly output array, Cannot be provided
            together with `out`. Default: ``"K"``.
        dtype : {None, str, dtype object}, optional
            If provided, the destination array will have this dtype. Cannot be
            provided together with `out`. Default: ``None``.

        Returns
        -------
        out : dpnp.ndarray
            Output array. The data type of the returned array is determined by
            the Type Promotion Rules.

        Limitations
        -----------
        Parameters `where` and `subok` are supported with their default values.
        Keyword argument `kwargs` is currently unsupported.
        Otherwise ``NotImplementedError`` exception will be raised.

        See also
        --------
        :obj:`dpnp.outer` : A less powerful version of dpnp.multiply.outer
                            that ravels all inputs to 1D. This exists primarily
                            for compatibility with old code.

        :obj:`dpnp.tensordot` : dpnp.tensordot(a, b, axes=((), ())) and
                                dpnp.multiply.outer(a, b) behave same for all
                                dimensions of a and b.

        Examples
        --------
        >>> import dpnp as np
        >>> A = np.array([1, 2, 3])
        >>> B = np.array([4, 5, 6])
        >>> np.multiply.outer(A, B)
        array([[ 4,  5,  6],
               [ 8, 10, 12],
               [12, 15, 18]])

        A multi-dimensional example:
        >>> A = np.array([[1, 2, 3], [4, 5, 6]])
        >>> A.shape
        (2, 3)
        >>> B = np.array([[1, 2, 3, 4]])
        >>> B.shape
        (1, 4)
        >>> C = np.multiply.outer(A, B)
        >>> C.shape; C
        (2, 3, 1, 4)
        array([[[[ 1,  2,  3,  4]],
                [[ 2,  4,  6,  8]],
                [[ 3,  6,  9, 12]]],
               [[[ 4,  8, 12, 16]],
                [[ 5, 10, 15, 20]],
                [[ 6, 12, 18, 24]]]])

        """

        dpnp.check_supported_arrays_type(
            x1, x2, scalar_type=True, all_scalars=False
        )
        if dpnp.isscalar(x1) or dpnp.isscalar(x2):
            _x1 = x1
            _x2 = x2
        else:
            _x1 = x1[(Ellipsis,) + (None,) * x2.ndim]
            _x2 = x2[(None,) * x1.ndim + (Ellipsis,)]
        return self.__call__(
            _x1,
            _x2,
            out=out,
            where=where,
            order=order,
            dtype=dtype,
            subok=subok,
            **kwargs,
        )


class DPNPBinaryFuncOutKw(DPNPBinaryFunc):
    """DPNPBinaryFunc that deprecates positional `out` argument."""

    @wraps(DPNPBinaryFunc.__call__)
    def __call__(self, *args, **kwargs):
        if len(args) > self.nin:
            warnings.warn(
                "Passing more than 2 positional arguments is deprecated. "
                "If you meant to use the third argument as an output, "
                "use the `out` keyword argument instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return super().__call__(*args, **kwargs)


class DPNPBinaryTwoOutputsFunc(BinaryElementwiseFunc):
    """
    Class that implements binary element-wise functions with two output arrays.

    Parameters
    ----------
    name : {str}
        Name of the binary function
    result_type_resolver_fn : {callable}
        Function that takes dtype of the input and returns the dtype of
        the result if the implementation functions supports it, or
        returns `None` otherwise.
    binary_dp_impl_fn : {callable}
        Data-parallel implementation function with signature
        `impl_fn(src: usm_ndarray, dst: usm_ndarray,
            sycl_queue: SyclQueue, depends: Optional[List[SyclEvent]])`
        where the `src` is the argument array, `dst` is the
        array to be populated with function values, effectively
        evaluating `dst = func(src)`.
        The `impl_fn` is expected to return a 2-tuple of `SyclEvent`s.
        The first event corresponds to data-management host tasks,
        including lifetime management of argument Python objects to ensure
        that their associated USM allocation is not freed before offloaded
        computational tasks complete execution, while the second event
        corresponds to computational tasks associated with function evaluation.
    docs : {str}
        Documentation string for the binary function.

    """

    def __init__(
        self,
        name,
        result_type_resolver_fn,
        binary_dp_impl_fn,
        docs,
    ):
        super().__init__(
            name,
            result_type_resolver_fn,
            binary_dp_impl_fn,
            docs,
        )
        self.__name__ = "DPNPBinaryTwoOutputsFunc"

    @property
    def nout(self):
        """Returns the number of arguments treated as outputs."""
        return 2

    @property
    def types(self):
        """
        Returns information about types supported by implementation function,
        using NumPy's character encoding for data types, e.g.

        Examples
        --------
        >>> import dpnp as np
        >>> np.divmod.types
        ['bb->bb', 'BB->BB', 'hh->hh', 'HH->HH', 'ii->ii', 'II->II',
         'll->ll', 'LL->LL', 'ee->ee', 'ff->ff', 'dd->dd']

        """

        types = self.types_
        if not types:
            types = []
            _all_dtypes = dtu._all_data_types(True, True)
            for dt1 in _all_dtypes:
                for dt2 in _all_dtypes:
                    dt3 = self.result_type_resolver_fn_(dt1, dt2)
                    if all(dt for dt in dt3):
                        types.append(
                            f"{dt1.char}{dt2.char}->{dt3[0].char}{dt3[1].char}"
                        )
            self.types_ = types
        return types

    def __call__(
        self,
        x1,
        x2,
        out1=None,
        out2=None,
        /,
        *,
        out=(None, None),
        where=True,
        order="K",
        dtype=None,
        subok=True,
        **kwargs,
    ):
        if kwargs:
            raise NotImplementedError(
                f"Requested function={self.name_} with kwargs={kwargs} "
                "isn't currently supported."
            )
        elif where is not True:
            raise NotImplementedError(
                f"Requested function={self.name_} with where={where} "
                "isn't currently supported."
            )
        elif dtype is not None:
            raise NotImplementedError(
                f"Requested function={self.name_} with dtype={dtype} "
                "isn't currently supported."
            )
        elif subok is not True:
            raise NotImplementedError(
                f"Requested function={self.name_} with subok={subok} "
                "isn't currently supported."
            )

        dpnp.check_supported_arrays_type(x1, x2, scalar_type=True)

        if order is None:
            order = "K"
        elif order in "afkcAFKC":
            order = order.upper()
        else:
            raise ValueError(
                "order must be one of 'C', 'F', 'A', or 'K' " f"(got '{order}')"
            )

        res_usm_type, exec_q = get_usm_allocations([x1, x2])
        x1 = dpnp.get_usm_ndarray_or_scalar(x1)
        x2 = dpnp.get_usm_ndarray_or_scalar(x2)

        x1_sh = _get_shape(x1)
        x2_sh = _get_shape(x2)
        try:
            res_shape = dpnp.broadcast_shapes(x1_sh, x2_sh)
        except ValueError:
            raise ValueError(
                "operands could not be broadcast together with shapes "
                f"{x1_sh} and {x2_sh}"
            )

        sycl_dev = exec_q.sycl_device
        x1_dt = _get_dtype(x1, sycl_dev)
        x2_dt = _get_dtype(x2, sycl_dev)
        if not all(
            _validate_dtype(dt) for dt in [x1_dt, x2_dt]
        ):  # pragma: no cover
            raise ValueError("Operands have unsupported data types")

        x1_dt, x2_dt = self.get_array_dtype_scalar_type_resolver_function()(
            x1_dt, x2_dt, sycl_dev
        )

        buf1_dt, buf2_dt, res1_dt, res2_dt = find_buf_dtype_4out(
            x1_dt,
            x2_dt,
            self.get_type_result_resolver_function(),
            sycl_dev,
        )
        if res1_dt is None or res2_dt is None:
            raise ValueError(
                f"function '{self.name_}' does not support input type "
                f"({x1_dt}, {x2_dt}), "
                "and the input could not be safely coerced to any "
                "supported types according to the casting rule ''safe''."
            )
        buf_dts = [buf1_dt, buf2_dt]

        if not isinstance(out, tuple):
            raise TypeError("'out' must be a tuple of arrays")

        if len(out) != self.nout:
            raise ValueError(
                "'out' tuple must have exactly one entry per ufunc output"
            )

        if not (out1 is None and out2 is None):
            if all(res is None for res in out):
                out = (out1, out2)
            else:
                raise TypeError(
                    "cannot specify 'out' as both a positional and keyword argument"
                )

        orig_out, out = list(out), list(out)
        res_dts = [res1_dt, res2_dt]

        for i in range(self.nout):
            if out[i] is None:
                continue

            res = dpnp.get_usm_ndarray(out[i])
            if not res.flags.writable:
                raise ValueError("output array is read-only")

            for other_out in out[:i]:
                if other_out is None:
                    continue

                other_out = dpnp.get_usm_ndarray(other_out)
                if dti._array_overlap(res, other_out):
                    raise ValueError("Output arrays cannot overlap")

            if res.shape != res_shape:
                raise ValueError(
                    "The shape of input and output arrays are inconsistent. "
                    f"Expected output shape is {res_shape}, got {res.shape}"
                )

            if dpu.get_execution_queue((exec_q, res.sycl_queue)) is None:
                raise dpnp.exceptions.ExecutionPlacementError(
                    "Input and output allocation queues are not compatible"
                )

            res_dt = res_dts[i]
            if res_dt != res.dtype:
                if not dpnp.can_cast(res_dt, res.dtype, casting="same_kind"):
                    raise TypeError(
                        f"Cannot cast ufunc '{self.name_}' output {i + 1} from "
                        f"{res_dt} to {res.dtype} with casting rule 'same_kind'"
                    )

                # Allocate a temporary buffer with the required dtype
                out[i] = dpt.empty_like(res, dtype=res_dt)
            else:
                # If `dt` is not None, a temporary copy of `x` will be created,
                # so the array overlap check isn't needed.
                x_to_check = [
                    x
                    for x, dt in zip([x1, x2], buf_dts)
                    if not dpnp.isscalar(x) and dt is None
                ]

                if any(
                    dti._array_overlap(x, res)
                    and not dti._same_logical_tensors(x, res)
                    for x in x_to_check
                ):
                    # allocate a temporary buffer to avoid memory overlapping
                    out[i] = dpt.empty_like(res)

        x1 = dpnp.as_usm_ndarray(x1, dtype=x1_dt, sycl_queue=exec_q)
        x2 = dpnp.as_usm_ndarray(x2, dtype=x2_dt, sycl_queue=exec_q)

        if order == "A":
            if x1.flags.f_contiguous and x2.flags.f_contiguous:
                order = "F"
            else:
                order = "C"

        _manager = dpu.SequentialOrderManager[exec_q]
        dep_evs = _manager.submitted_events

        # Cast input array to the supported type if needed
        if any(dt is not None for dt in buf_dts):
            if all(dt is not None for dt in buf_dts):
                if x1.flags.c_contiguous and x2.flags.c_contiguous:
                    order = "C"
                elif x1.flags.f_contiguous and x2.flags.f_contiguous:
                    order = "F"

            arrs = [x1, x2]
            buf_dts = [buf1_dt, buf2_dt]
            for i in range(self.nout):
                buf_dt = buf_dts[i]
                if buf_dt is None:
                    continue

                x = arrs[i]
                if order == "K":
                    buf = dtc._empty_like_orderK(x, buf_dt)
                else:
                    buf = dpt.empty_like(x, dtype=buf_dt, order=order)

                ht_copy_ev, copy_ev = dti._copy_usm_ndarray_into_usm_ndarray(
                    src=x, dst=buf, sycl_queue=exec_q, depends=dep_evs
                )
                _manager.add_event_pair(ht_copy_ev, copy_ev)

                arrs[i] = buf
            x1, x2 = arrs

        # Allocate a buffer for the output arrays if needed
        for i in range(self.nout):
            if out[i] is None:
                res_dt = res_dts[i]
                if order == "K":
                    out[i] = dtc._empty_like_pair_orderK(
                        x1, x2, res_dt, res_shape, res_usm_type, exec_q
                    )
                else:
                    out[i] = dpt.empty(
                        res_shape,
                        dtype=res_dt,
                        order=order,
                        usm_type=res_usm_type,
                        sycl_queue=exec_q,
                    )

        # Broadcast shapes of input arrays
        if x1.shape != res_shape:
            x1 = dpt.broadcast_to(x1, res_shape)
        if x2.shape != res_shape:
            x2 = dpt.broadcast_to(x2, res_shape)

        # Call the binary function with input and output arrays
        ht_binary_ev, binary_ev = self.get_implementation_function()(
            x1,
            x2,
            dpnp.get_usm_ndarray(out[0]),
            dpnp.get_usm_ndarray(out[1]),
            sycl_queue=exec_q,
            depends=_manager.submitted_events,
        )
        _manager.add_event_pair(ht_binary_ev, binary_ev)

        for i in range(self.nout):
            orig_res, res = orig_out[i], out[i]
            if not (orig_res is None or orig_res is res):
                # Copy the out data from temporary buffer to original memory
                ht_copy_ev, copy_ev = dti._copy_usm_ndarray_into_usm_ndarray(
                    src=res,
                    dst=dpnp.get_usm_ndarray(orig_res),
                    sycl_queue=exec_q,
                    depends=[binary_ev],
                )
                _manager.add_event_pair(ht_copy_ev, copy_ev)
                res = out[i] = orig_res

            if not isinstance(res, dpnp_array):
                # Always return dpnp.ndarray
                out[i] = dpnp_array._create_from_usm_ndarray(res)
        return tuple(out)


class DPNPAngle(DPNPUnaryFunc):
    """Class that implements dpnp.angle unary element-wise functions."""

    def __init__(
        self,
        name,
        result_type_resolver_fn,
        unary_dp_impl_fn,
        docs,
        mkl_fn_to_call=None,
        mkl_impl_fn=None,
    ):
        super().__init__(
            name,
            result_type_resolver_fn,
            unary_dp_impl_fn,
            docs,
            mkl_fn_to_call=mkl_fn_to_call,
            mkl_impl_fn=mkl_impl_fn,
        )

    def __call__(self, x, /, deg=False, *, out=None, order="K"):
        res = super().__call__(x, out=out, order=order)
        if deg is True:
            res *= 180 / dpnp.pi
        return res


class DPNPI0(DPNPUnaryFunc):
    """Class that implements dpnp.i0 unary element-wise functions."""

    def __init__(
        self,
        name,
        result_type_resolver_fn,
        unary_dp_impl_fn,
        docs,
        mkl_fn_to_call=None,
        mkl_impl_fn=None,
    ):
        super().__init__(
            name,
            result_type_resolver_fn,
            unary_dp_impl_fn,
            docs,
            mkl_fn_to_call=mkl_fn_to_call,
            mkl_impl_fn=mkl_impl_fn,
        )

    def __call__(self, x, /, *, out=None, order="K"):
        return super().__call__(x, out=out, order=order)


class DPNPImag(DPNPUnaryFunc):
    """Class that implements dpnp.imag unary element-wise functions."""

    def __init__(
        self,
        name,
        result_type_resolver_fn,
        unary_dp_impl_fn,
        docs,
    ):
        super().__init__(
            name,
            result_type_resolver_fn,
            unary_dp_impl_fn,
            docs,
        )

    def __call__(self, x, /, *, out=None, order="K"):
        return super().__call__(x, out=out, order=order)


class DPNPReal(DPNPUnaryFunc):
    """Class that implements dpnp.real unary element-wise functions."""

    def __init__(
        self,
        name,
        result_type_resolver_fn,
        unary_dp_impl_fn,
        docs,
    ):
        super().__init__(
            name,
            result_type_resolver_fn,
            unary_dp_impl_fn,
            docs,
        )

    def __call__(self, x, /, *, out=None, order="K"):
        if numpy.iscomplexobj(x):
            return super().__call__(x, out=out, order=order)
        return x


class DPNPRound(DPNPUnaryFunc):
    """Class that implements dpnp.round unary element-wise functions."""

    def __init__(
        self,
        name,
        result_type_resolver_fn,
        unary_dp_impl_fn,
        docs,
        mkl_fn_to_call=None,
        mkl_impl_fn=None,
    ):
        super().__init__(
            name,
            result_type_resolver_fn,
            unary_dp_impl_fn,
            docs,
            mkl_fn_to_call=mkl_fn_to_call,
            mkl_impl_fn=mkl_impl_fn,
        )

    def __call__(self, x, /, decimals=0, out=None, *, dtype=None):
        if decimals != 0:
            x_usm = dpnp.get_usm_ndarray(x)
            out = self._unpack_out_kw(out)
            out_usm = None if out is None else dpnp.get_usm_ndarray(out)

            if dpnp.issubdtype(x_usm.dtype, dpnp.integer):
                if decimals < 0:
                    dtype = x_usm.dtype
                    x_usm = dpt.round(x_usm * 10**decimals, out=out_usm)
                    res_usm = dpt.divide(x_usm, 10**decimals, out=out_usm)
                else:
                    res_usm = dpt.round(x_usm, out=out_usm)
            else:
                x_usm = dpt.round(x_usm * 10**decimals, out=out_usm)
                res_usm = dpt.divide(x_usm, 10**decimals, out=out_usm)

            if dtype is not None:
                res_usm = dpt.astype(res_usm, dtype, copy=False)

            if out is not None and isinstance(out, dpnp_array):
                return out
            return dpnp_array._create_from_usm_ndarray(res_usm)
        else:
            return super().__call__(x, out=out, dtype=dtype)


class DPNPSinc(DPNPUnaryFunc):
    """Class that implements dpnp.sinc unary element-wise functions."""

    def __init__(
        self,
        name,
        result_type_resolver_fn,
        unary_dp_impl_fn,
        docs,
    ):
        super().__init__(
            name,
            result_type_resolver_fn,
            unary_dp_impl_fn,
            docs,
        )

    def __call__(self, x, /, *, out=None, order="K"):
        return super().__call__(x, out=out, order=order)


def acceptance_fn_gcd_lcm(
    arg1_dtype, arg2_dtype, buf1_dt, buf2_dt, res_dt, sycl_dev
):
    # gcd/lcm are not defined for boolean data type
    if arg1_dtype.char == "?" and arg2_dtype.char == "?":
        raise ValueError(
            "The function is not supported for inputs of data type bool"
        )
    else:
        return True


def acceptance_fn_negative(arg_dtype, buf_dt, res_dt, sycl_dev):
    # negative is not defined for boolean data type
    if arg_dtype.char == "?":
        raise TypeError(
            "The `negative` function, the `-` operator, is not supported "
            "for inputs of data type bool, use the `~` operator or the "
            "`logical_not` function instead"
        )
    else:
        return True


def acceptance_fn_positive(arg_dtype, buf_dt, res_dt, sycl_dev):
    # positive is not defined for boolean data type
    if arg_dtype.char == "?":
        raise TypeError(
            "The `positive` function is not supported for inputs of data type "
            "bool"
        )
    else:
        return True


def acceptance_fn_sign(arg_dtype, buf_dt, res_dt, sycl_dev):
    # sign is not defined for boolean data type
    if arg_dtype.char == "?":
        raise TypeError(
            "The `sign` function is not supported for inputs of data type bool"
        )
    else:
        return True


def acceptance_fn_subtract(
    arg1_dtype, arg2_dtype, buf1_dt, buf2_dt, res_dt, sycl_dev
):
    # subtract is not defined for boolean data type
    if arg1_dtype.char == "?" and arg2_dtype.char == "?":
        raise TypeError(
            "The `subtract` function, the `-` operator, is not supported "
            "for inputs of data type bool, use the `^` operator,  the "
            "`bitwise_xor`, or the `logical_xor` function instead"
        )
    else:
        return True


def resolve_weak_types_2nd_arg_int(o1_dtype, o2_dtype, sycl_dev):
    """
    The second weak dtype has to be upcasting up to default integer dtype
    for a SYCL device where it is possible.
    For other cases the default weak types resolving will be applied.

    """

    if dtu._is_weak_dtype(o2_dtype):
        o1_kind_num = dtu._strong_dtype_num_kind(o1_dtype)
        o2_kind_num = dtu._weak_type_num_kind(o2_dtype)
        if o2_kind_num < o1_kind_num:
            if isinstance(
                o2_dtype, (dtu.WeakBooleanType, dtu.WeakIntegralType)
            ):
                return o1_dtype, dpt.dtype(
                    dti.default_device_int_type(sycl_dev)
                )
    return dtu._resolve_weak_types(o1_dtype, o2_dtype, sycl_dev)
