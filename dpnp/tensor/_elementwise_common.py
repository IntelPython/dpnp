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

from dpctl.utils import SequentialOrderManager

import dpnp.tensor as dpt
import dpnp.tensor._tensor_impl as ti

from ._copy_utils import _empty_like_orderK, _empty_like_pair_orderK
from ._manipulation_functions import _broadcast_shape_impl
from ._scalar_utils import (
    _get_dtype,
    _get_queue_usm_type,
    _get_shape,
    _validate_dtype,
)
from ._type_utils import (
    _acceptance_fn_default_binary,
    _acceptance_fn_default_unary,
    _all_data_types,
    _find_buf_dtype,
    _find_buf_dtype2,
    _find_buf_dtype_in_place_op,
    _resolve_weak_types,
)


class UnaryElementwiseFunc:
    """
    Class that implements unary element-wise functions.

    Args:
        name (str):
            Name of the unary function
        result_type_resovler_fn (callable):
            Function that takes dtype of the input and
            returns the dtype of the result if the
            implementation functions supports it, or
            returns `None` otherwise.
        unary_dp_impl_fn (callable):
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
            corresponds to computational tasks associated with function
            evaluation.
        acceptance_fn (callable, optional):
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
        docs (str):
            Documentation string for the unary function.
    """

    def __init__(
        self,
        name,
        result_type_resolver_fn,
        unary_dp_impl_fn,
        docs,
        acceptance_fn=None,
    ):
        self.__name__ = "UnaryElementwiseFunc"
        self.name_ = name
        self.result_type_resolver_fn_ = result_type_resolver_fn
        self.types_ = None
        self.unary_fn_ = unary_dp_impl_fn
        self.__doc__ = docs
        if callable(acceptance_fn):
            self.acceptance_fn_ = acceptance_fn
        else:
            self.acceptance_fn_ = _acceptance_fn_default_unary

    def __str__(self):
        return f"<{self.__name__} '{self.name_}'>"

    def __repr__(self):
        return f"<{self.__name__} '{self.name_}'>"

    def get_implementation_function(self):
        """Returns the implementation function for
        this elementwise unary function.

        """
        return self.unary_fn_

    def get_type_result_resolver_function(self):
        """Returns the type resolver function for this
        elementwise unary function.
        """
        return self.result_type_resolver_fn_

    def get_type_promotion_path_acceptance_function(self):
        """Returns the acceptance function for this
        elementwise binary function.

        Acceptance function influences the type promotion
        behavior of this unary function.
        The function takes 4 arguments:
            arg_dtype - Data type of the first argument
            buf_dtype - Data type the argument would be cast to
            res_dtype - Data type of the output array with function values
            sycl_dev - The :class:`dpctl.SyclDevice` where the function
                evaluation is carried out.
        The function is invoked when the argument of the unary function
        requires casting, e.g. the argument of `dpctl.tensor.log` is an
        array with integral data type.
        """
        return self.acceptance_fn_

    @property
    def nin(self):
        """Returns the number of arguments treated as inputs."""
        return 1

    @property
    def nout(self):
        """Returns the number of arguments treated as outputs."""
        return 1

    @property
    def types(self):
        """Returns information about types supported by
        implementation function, using NumPy's character
        encoding for data types, e.g.

        :Example:
            .. code-block:: python

                dpctl.tensor.sin.types
                # Outputs: ['e->e', 'f->f', 'd->d', 'F->F', 'D->D']
        """
        types = self.types_
        if not types:
            types = []
            for dt1 in _all_data_types(True, True):
                dt2 = self.result_type_resolver_fn_(dt1)
                if dt2:
                    types.append(f"{dt1.char}->{dt2.char}")
            self.types_ = types
        return types

    def __call__(self, x, /, *, out=None, order="K"):
        if not isinstance(x, dpt.usm_ndarray):
            raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x)}")

        if order not in ["C", "F", "K", "A"]:
            order = "K"
        buf_dt, res_dt = _find_buf_dtype(
            x.dtype,
            self.result_type_resolver_fn_,
            x.sycl_device,
            acceptance_fn=self.acceptance_fn_,
        )
        if res_dt is None:
            raise ValueError(
                f"function '{self.name_}' does not support input type "
                f"({x.dtype}), "
                "and the input could not be safely coerced to any "
                "supported types according to the casting rule ''safe''."
            )

        orig_out = out
        if out is not None:
            if not isinstance(out, dpt.usm_ndarray):
                raise TypeError(
                    f"output array must be of usm_ndarray type, got {type(out)}"
                )

            if not out.flags.writable:
                raise ValueError("provided `out` array is read-only")

            if out.shape != x.shape:
                raise ValueError(
                    "The shape of input and output arrays are inconsistent. "
                    f"Expected output shape is {x.shape}, got {out.shape}"
                )

            if res_dt != out.dtype:
                raise ValueError(
                    f"Output array of type {res_dt} is needed, "
                    f"got {out.dtype}"
                )

            if (
                buf_dt is None
                and ti._array_overlap(x, out)
                and not ti._same_logical_tensors(x, out)
            ):
                # Allocate a temporary buffer to avoid memory overlapping.
                # Note if `buf_dt` is not None, a temporary copy of `x` will be
                # created, so the array overlap check isn't needed.
                out = dpt.empty_like(out)

            if dpt.get_execution_queue((x.sycl_queue, out.sycl_queue)) is None:
                raise dpt.ExecutionPlacementError(
                    "Input and output allocation queues are not compatible"
                )

        exec_q = x.sycl_queue
        _manager = SequentialOrderManager[exec_q]
        if buf_dt is None:
            if out is None:
                if order == "K":
                    out = _empty_like_orderK(x, res_dt)
                else:
                    if order == "A":
                        order = "F" if x.flags.f_contiguous else "C"
                    out = dpt.empty_like(x, dtype=res_dt, order=order)

            dep_evs = _manager.submitted_events
            ht_unary_ev, unary_ev = self.unary_fn_(
                x, out, sycl_queue=exec_q, depends=dep_evs
            )
            _manager.add_event_pair(ht_unary_ev, unary_ev)

            if not (orig_out is None or orig_out is out):
                # Copy the out data from temporary buffer to original memory
                ht_copy_ev, cpy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                    src=out, dst=orig_out, sycl_queue=exec_q, depends=[unary_ev]
                )
                _manager.add_event_pair(ht_copy_ev, cpy_ev)
                out = orig_out

            return out

        if order == "K":
            buf = _empty_like_orderK(x, buf_dt)
        else:
            if order == "A":
                order = "F" if x.flags.f_contiguous else "C"
            buf = dpt.empty_like(x, dtype=buf_dt, order=order)

        dep_evs = _manager.submitted_events
        ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=x, dst=buf, sycl_queue=exec_q, depends=dep_evs
        )
        _manager.add_event_pair(ht_copy_ev, copy_ev)
        if out is None:
            if order == "K":
                out = _empty_like_orderK(buf, res_dt)
            else:
                out = dpt.empty_like(buf, dtype=res_dt, order=order)

        ht, uf_ev = self.unary_fn_(
            buf, out, sycl_queue=exec_q, depends=[copy_ev]
        )
        _manager.add_event_pair(ht, uf_ev)

        return out


class BinaryElementwiseFunc:
    """
    Class that implements binary element-wise functions.

    Args:
        name (str):
            Name of the unary function
        result_type_resovle_fn (callable):
            Function that takes dtypes of the input and
            returns the dtype of the result if the
            implementation functions supports it, or
            returns `None` otherwise.
        binary_dp_impl_fn (callable):
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
        docs (str):
            Documentation string for the unary function.
        binary_inplace_fn (callable, optional):
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
        acceptance_fn (callable, optional):
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
    """

    def __init__(
        self,
        name,
        result_type_resolver_fn,
        binary_dp_impl_fn,
        docs,
        binary_inplace_fn=None,
        acceptance_fn=None,
        weak_type_resolver=None,
    ):
        self.__name__ = "BinaryElementwiseFunc"
        self.name_ = name
        self.result_type_resolver_fn_ = result_type_resolver_fn
        self.types_ = None
        self.binary_fn_ = binary_dp_impl_fn
        self.binary_inplace_fn_ = binary_inplace_fn
        self.__doc__ = docs
        if callable(acceptance_fn):
            self.acceptance_fn_ = acceptance_fn
        else:
            self.acceptance_fn_ = _acceptance_fn_default_binary
        if callable(weak_type_resolver):
            self.weak_type_resolver_ = weak_type_resolver
        else:
            self.weak_type_resolver_ = _resolve_weak_types

    def __str__(self):
        return f"<{self.__name__} '{self.name_}'>"

    def __repr__(self):
        return f"<{self.__name__} '{self.name_}'>"

    def get_implementation_function(self):
        """Returns the out-of-place implementation
        function for this elementwise binary function.

        """
        return self.binary_fn_

    def get_implementation_inplace_function(self):
        """Returns the in-place implementation
        function for this elementwise binary function.

        """
        return self.binary_inplace_fn_

    def get_type_result_resolver_function(self):
        """Returns the type resolver function for this
        elementwise binary function.
        """
        return self.result_type_resolver_fn_

    def get_type_promotion_path_acceptance_function(self):
        """Returns the acceptance function for this
        elementwise binary function.

        Acceptance function influences the type promotion
        behavior of this binary function.
        The function takes 6 arguments:
            arg1_dtype - Data type of the first argument
            arg2_dtype - Data type of the second argument
            ret_buf1_dtype - Data type the first argument would be cast to
            ret_buf2_dtype - Data type the second argument would be cast to
            res_dtype - Data type of the output array with function values
            sycl_dev - :class:`dpctl.SyclDevice` on which function evaluation
                is carried out.

        The acceptance function is only invoked if both input arrays must be
        cast to intermediary data types, as would happen during call of
        `dpctl.tensor.hypot` with both arrays being of integral data type.
        """
        return self.acceptance_fn_

    def get_array_dtype_scalar_type_resolver_function(self):
        """Returns the function which determines how to treat
        Python scalar types for this elementwise binary function.

        Resolver influences what type the scalar will be
        treated as prior to type promotion behavior.
        The function takes 3 arguments:

        Args:
            o1_dtype (object, dtype):
                A class representing a Python scalar type or a ``dtype``
            o2_dtype (object, dtype):
                A class representing a Python scalar type or a ``dtype``
            sycl_dev (:class:`dpctl.SyclDevice`):
                Device on which function evaluation is carried out.

        One of ``o1_dtype`` and ``o2_dtype`` must be a ``dtype`` instance.
        """
        return self.weak_type_resolver_

    @property
    def nin(self):
        """Returns the number of arguments treated as inputs."""
        return 2

    @property
    def nout(self):
        """Returns the number of arguments treated as outputs."""
        return 1

    @property
    def types(self):
        """Returns information about types supported by
        implementation function, using NumPy's character
        encoding for data types, e.g.

        :Example:
            .. code-block:: python

                dpctl.tensor.divide.types
                # Outputs: ['ee->e', 'ff->f', 'fF->F', 'dd->d', 'dD->D',
                #    'Ff->F', 'FF->F', 'Dd->D', 'DD->D']
        """
        types = self.types_
        if not types:
            types = []
            _all_dtypes = _all_data_types(True, True)
            for dt1 in _all_dtypes:
                for dt2 in _all_dtypes:
                    dt3 = self.result_type_resolver_fn_(dt1, dt2)
                    if dt3:
                        types.append(f"{dt1.char}{dt2.char}->{dt3.char}")
            self.types_ = types
        return types

    def __call__(self, o1, o2, /, *, out=None, order="K"):
        if order not in ["K", "C", "F", "A"]:
            order = "K"
        q1, o1_usm_type = _get_queue_usm_type(o1)
        q2, o2_usm_type = _get_queue_usm_type(o2)
        if q1 is None and q2 is None:
            raise dpt.ExecutionPlacementError(
                "Execution placement can not be unambiguously inferred "
                "from input arguments. "
                "One of the arguments must represent USM allocation and "
                "expose `__sycl_usm_array_interface__` property"
            )
        if q1 is None:
            exec_q = q2
            res_usm_type = o2_usm_type
        elif q2 is None:
            exec_q = q1
            res_usm_type = o1_usm_type
        else:
            exec_q = dpt.get_execution_queue((q1, q2))
            if exec_q is None:
                raise dpt.ExecutionPlacementError(
                    "Execution placement can not be unambiguously inferred "
                    "from input arguments."
                )
            res_usm_type = dpt.get_coerced_usm_type(
                (
                    o1_usm_type,
                    o2_usm_type,
                )
            )
        dpt.validate_usm_type(res_usm_type, allow_none=False)
        o1_shape = _get_shape(o1)
        o2_shape = _get_shape(o2)
        if not all(
            isinstance(s, (tuple, list))
            for s in (
                o1_shape,
                o2_shape,
            )
        ):
            raise TypeError(
                "Shape of arguments can not be inferred. "
                "Arguments are expected to be "
                "lists, tuples, or both"
            )
        try:
            res_shape = _broadcast_shape_impl(
                [
                    o1_shape,
                    o2_shape,
                ]
            )
        except ValueError:
            raise ValueError(
                "operands could not be broadcast together with shapes "
                f"{o1_shape} and {o2_shape}"
            )
        sycl_dev = exec_q.sycl_device
        o1_dtype = _get_dtype(o1, sycl_dev)
        o2_dtype = _get_dtype(o2, sycl_dev)
        if not all(_validate_dtype(o) for o in (o1_dtype, o2_dtype)):
            raise ValueError("Operands have unsupported data types")

        o1_dtype, o2_dtype = self.weak_type_resolver_(
            o1_dtype, o2_dtype, sycl_dev
        )

        buf1_dt, buf2_dt, res_dt = _find_buf_dtype2(
            o1_dtype,
            o2_dtype,
            self.result_type_resolver_fn_,
            sycl_dev,
            acceptance_fn=self.acceptance_fn_,
        )

        if res_dt is None:
            raise ValueError(
                f"function '{self.name_}' does not support input types "
                f"({o1_dtype}, {o2_dtype}), "
                "and the inputs could not be safely coerced to any "
                "supported types according to the casting rule ''safe''."
            )

        orig_out = out
        _manager = SequentialOrderManager[exec_q]
        if out is not None:
            if not isinstance(out, dpt.usm_ndarray):
                raise TypeError(
                    f"output array must be of usm_ndarray type, got {type(out)}"
                )

            if not out.flags.writable:
                raise ValueError("provided `out` array is read-only")

            if out.shape != res_shape:
                raise ValueError(
                    "The shape of input and output arrays are inconsistent. "
                    f"Expected output shape is {res_shape}, got {out.shape}"
                )

            if res_dt != out.dtype:
                raise ValueError(
                    f"Output array of type {res_dt} is needed, "
                    f"got {out.dtype}"
                )

            if dpt.get_execution_queue((exec_q, out.sycl_queue)) is None:
                raise dpt.ExecutionPlacementError(
                    "Input and output allocation queues are not compatible"
                )

            if isinstance(o1, dpt.usm_ndarray):
                if ti._array_overlap(o1, out) and buf1_dt is None:
                    if not ti._same_logical_tensors(o1, out):
                        out = dpt.empty_like(out)
                    elif self.binary_inplace_fn_ is not None:
                        # if there is a dedicated in-place kernel
                        # it can be called here, otherwise continues
                        if isinstance(o2, dpt.usm_ndarray):
                            src2 = o2
                            if (
                                ti._array_overlap(o2, out)
                                and not ti._same_logical_tensors(o2, out)
                                and buf2_dt is None
                            ):
                                buf2_dt = o2_dtype
                        else:
                            src2 = dpt.asarray(
                                o2, dtype=o2_dtype, sycl_queue=exec_q
                            )
                        if buf2_dt is None:
                            if src2.shape != res_shape:
                                src2 = dpt.broadcast_to(src2, res_shape)
                            dep_evs = _manager.submitted_events
                            ht_, comp_ev = self.binary_inplace_fn_(
                                lhs=o1,
                                rhs=src2,
                                sycl_queue=exec_q,
                                depends=dep_evs,
                            )
                            _manager.add_event_pair(ht_, comp_ev)
                        else:
                            buf2 = dpt.empty_like(src2, dtype=buf2_dt)
                            dep_evs = _manager.submitted_events
                            (
                                ht_copy_ev,
                                copy_ev,
                            ) = ti._copy_usm_ndarray_into_usm_ndarray(
                                src=src2,
                                dst=buf2,
                                sycl_queue=exec_q,
                                depends=dep_evs,
                            )
                            _manager.add_event_pair(ht_copy_ev, copy_ev)

                            buf2 = dpt.broadcast_to(buf2, res_shape)
                            ht_, bf_ev = self.binary_inplace_fn_(
                                lhs=o1,
                                rhs=buf2,
                                sycl_queue=exec_q,
                                depends=[copy_ev],
                            )
                            _manager.add_event_pair(ht_, bf_ev)

                        return out

            if isinstance(o2, dpt.usm_ndarray):
                if (
                    ti._array_overlap(o2, out)
                    and not ti._same_logical_tensors(o2, out)
                    and buf2_dt is None
                ):
                    # should not reach if out is reallocated
                    # after being checked against o1
                    out = dpt.empty_like(out)

        if isinstance(o1, dpt.usm_ndarray):
            src1 = o1
        else:
            src1 = dpt.asarray(o1, dtype=o1_dtype, sycl_queue=exec_q)
        if isinstance(o2, dpt.usm_ndarray):
            src2 = o2
        else:
            src2 = dpt.asarray(o2, dtype=o2_dtype, sycl_queue=exec_q)

        if order == "A":
            order = (
                "F"
                if all(
                    arr.flags.f_contiguous
                    for arr in (
                        src1,
                        src2,
                    )
                )
                else "C"
            )

        if buf1_dt is None and buf2_dt is None:
            if out is None:
                if order == "K":
                    out = _empty_like_pair_orderK(
                        src1, src2, res_dt, res_shape, res_usm_type, exec_q
                    )
                else:
                    out = dpt.empty(
                        res_shape,
                        dtype=res_dt,
                        usm_type=res_usm_type,
                        sycl_queue=exec_q,
                        order=order,
                    )
            if src1.shape != res_shape:
                src1 = dpt.broadcast_to(src1, res_shape)
            if src2.shape != res_shape:
                src2 = dpt.broadcast_to(src2, res_shape)
            deps_ev = _manager.submitted_events
            ht_binary_ev, binary_ev = self.binary_fn_(
                src1=src1,
                src2=src2,
                dst=out,
                sycl_queue=exec_q,
                depends=deps_ev,
            )
            _manager.add_event_pair(ht_binary_ev, binary_ev)
            if not (orig_out is None or orig_out is out):
                # Copy the out data from temporary buffer to original memory
                ht_copy_out_ev, cpy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                    src=out,
                    dst=orig_out,
                    sycl_queue=exec_q,
                    depends=[binary_ev],
                )
                _manager.add_event_pair(ht_copy_out_ev, cpy_ev)
                out = orig_out
            return out
        elif buf1_dt is None:
            if order == "K":
                buf2 = _empty_like_orderK(src2, buf2_dt)
            else:
                buf2 = dpt.empty_like(src2, dtype=buf2_dt, order=order)
            dep_evs = _manager.submitted_events
            ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                src=src2, dst=buf2, sycl_queue=exec_q, depends=dep_evs
            )
            _manager.add_event_pair(ht_copy_ev, copy_ev)
            if out is None:
                if order == "K":
                    out = _empty_like_pair_orderK(
                        src1, buf2, res_dt, res_shape, res_usm_type, exec_q
                    )
                else:
                    out = dpt.empty(
                        res_shape,
                        dtype=res_dt,
                        usm_type=res_usm_type,
                        sycl_queue=exec_q,
                        order=order,
                    )

            if src1.shape != res_shape:
                src1 = dpt.broadcast_to(src1, res_shape)
            buf2 = dpt.broadcast_to(buf2, res_shape)
            ht_binary_ev, binary_ev = self.binary_fn_(
                src1=src1,
                src2=buf2,
                dst=out,
                sycl_queue=exec_q,
                depends=[copy_ev],
            )
            _manager.add_event_pair(ht_binary_ev, binary_ev)
            if not (orig_out is None or orig_out is out):
                # Copy the out data from temporary buffer to original memory
                ht_copy_out_ev, cpy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                    src=out,
                    dst=orig_out,
                    sycl_queue=exec_q,
                    depends=[binary_ev],
                )
                _manager.add_event_pair(ht_copy_out_ev, cpy_ev)
                out = orig_out
            return out
        elif buf2_dt is None:
            if order == "K":
                buf1 = _empty_like_orderK(src1, buf1_dt)
            else:
                buf1 = dpt.empty_like(src1, dtype=buf1_dt, order=order)
            dep_evs = _manager.submitted_events
            ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                src=src1, dst=buf1, sycl_queue=exec_q, depends=dep_evs
            )
            _manager.add_event_pair(ht_copy_ev, copy_ev)
            if out is None:
                if order == "K":
                    out = _empty_like_pair_orderK(
                        buf1, src2, res_dt, res_shape, res_usm_type, exec_q
                    )
                else:
                    out = dpt.empty(
                        res_shape,
                        dtype=res_dt,
                        usm_type=res_usm_type,
                        sycl_queue=exec_q,
                        order=order,
                    )

            buf1 = dpt.broadcast_to(buf1, res_shape)
            if src2.shape != res_shape:
                src2 = dpt.broadcast_to(src2, res_shape)
            ht_binary_ev, binary_ev = self.binary_fn_(
                src1=buf1,
                src2=src2,
                dst=out,
                sycl_queue=exec_q,
                depends=[copy_ev],
            )
            _manager.add_event_pair(ht_binary_ev, binary_ev)
            if not (orig_out is None or orig_out is out):
                # Copy the out data from temporary buffer to original memory
                ht_copy_out_ev, cpy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                    src=out,
                    dst=orig_out,
                    sycl_queue=exec_q,
                    depends=[binary_ev],
                )
                _manager.add_event_pair(ht_copy_out_ev, cpy_ev)
                out = orig_out
            return out

        if order == "K":
            if src1.flags.c_contiguous and src2.flags.c_contiguous:
                order = "C"
            elif src1.flags.f_contiguous and src2.flags.f_contiguous:
                order = "F"
        if order == "K":
            buf1 = _empty_like_orderK(src1, buf1_dt)
        else:
            buf1 = dpt.empty_like(src1, dtype=buf1_dt, order=order)
        dep_evs = _manager.submitted_events
        ht_copy1_ev, copy1_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=src1, dst=buf1, sycl_queue=exec_q, depends=dep_evs
        )
        _manager.add_event_pair(ht_copy1_ev, copy1_ev)
        if order == "K":
            buf2 = _empty_like_orderK(src2, buf2_dt)
        else:
            buf2 = dpt.empty_like(src2, dtype=buf2_dt, order=order)
        ht_copy2_ev, copy2_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=src2, dst=buf2, sycl_queue=exec_q, depends=dep_evs
        )
        _manager.add_event_pair(ht_copy2_ev, copy2_ev)
        if out is None:
            if order == "K":
                out = _empty_like_pair_orderK(
                    buf1, buf2, res_dt, res_shape, res_usm_type, exec_q
                )
            else:
                out = dpt.empty(
                    res_shape,
                    dtype=res_dt,
                    usm_type=res_usm_type,
                    sycl_queue=exec_q,
                    order=order,
                )

        buf1 = dpt.broadcast_to(buf1, res_shape)
        buf2 = dpt.broadcast_to(buf2, res_shape)
        ht_, bf_ev = self.binary_fn_(
            src1=buf1,
            src2=buf2,
            dst=out,
            sycl_queue=exec_q,
            depends=[copy1_ev, copy2_ev],
        )
        _manager.add_event_pair(ht_, bf_ev)
        return out

    def _inplace_op(self, o1, o2):
        if self.binary_inplace_fn_ is None:
            raise ValueError(
                "binary function does not have a dedicated in-place "
                "implementation"
            )
        if not isinstance(o1, dpt.usm_ndarray):
            raise TypeError(
                "Expected first argument to be "
                f"dpctl.tensor.usm_ndarray, got {type(o1)}"
            )
        if not o1.flags.writable:
            raise ValueError("provided left-hand side array is read-only")
        q1, o1_usm_type = o1.sycl_queue, o1.usm_type
        q2, o2_usm_type = _get_queue_usm_type(o2)
        if q2 is None:
            exec_q = q1
            res_usm_type = o1_usm_type
        else:
            exec_q = dpt.get_execution_queue((q1, q2))
            if exec_q is None:
                raise dpt.ExecutionPlacementError(
                    "Execution placement can not be unambiguously inferred "
                    "from input arguments."
                )
            res_usm_type = dpt.get_coerced_usm_type(
                (
                    o1_usm_type,
                    o2_usm_type,
                )
            )
        dpt.validate_usm_type(res_usm_type, allow_none=False)
        o1_shape = o1.shape
        o2_shape = _get_shape(o2)
        if not isinstance(o2_shape, (tuple, list)):
            raise TypeError(
                "Shape of second argument can not be inferred. "
                "Expected list or tuple."
            )
        try:
            res_shape = _broadcast_shape_impl(
                [
                    o1_shape,
                    o2_shape,
                ]
            )
        except ValueError:
            raise ValueError(
                "operands could not be broadcast together with shapes "
                f"{o1_shape} and {o2_shape}"
            )

        if res_shape != o1_shape:
            raise ValueError(
                "The shape of the non-broadcastable left-hand "
                f"side {o1_shape} is inconsistent with the "
                f"broadcast shape {res_shape}."
            )

        sycl_dev = exec_q.sycl_device
        o1_dtype = o1.dtype
        o2_dtype = _get_dtype(o2, sycl_dev)
        if not _validate_dtype(o2_dtype):
            raise ValueError("Operand has an unsupported data type")

        o1_dtype, o2_dtype = self.weak_type_resolver_(
            o1_dtype, o2_dtype, sycl_dev
        )

        buf_dt, res_dt = _find_buf_dtype_in_place_op(
            o1_dtype,
            o2_dtype,
            self.result_type_resolver_fn_,
            sycl_dev,
        )

        if res_dt is None:
            raise ValueError(
                f"function '{self.name_}' does not support input types "
                f"({o1_dtype}, {o2_dtype}), "
                "and the inputs could not be safely coerced to any "
                "supported types according to the casting rule "
                "''same_kind''."
            )

        if res_dt != o1_dtype:
            raise ValueError(
                f"Output array of type {res_dt} is needed, " f"got {o1_dtype}"
            )

        _manager = SequentialOrderManager[exec_q]
        if isinstance(o2, dpt.usm_ndarray):
            src2 = o2
            if (
                ti._array_overlap(o2, o1)
                and not ti._same_logical_tensors(o2, o1)
                and buf_dt is None
            ):
                buf_dt = o2_dtype
        else:
            src2 = dpt.asarray(o2, dtype=o2_dtype, sycl_queue=exec_q)
        if buf_dt is None:
            if src2.shape != res_shape:
                src2 = dpt.broadcast_to(src2, res_shape)
            dep_evs = _manager.submitted_events
            ht_, comp_ev = self.binary_inplace_fn_(
                lhs=o1,
                rhs=src2,
                sycl_queue=exec_q,
                depends=dep_evs,
            )
            _manager.add_event_pair(ht_, comp_ev)
        else:
            buf = dpt.empty_like(src2, dtype=buf_dt)
            dep_evs = _manager.submitted_events
            (
                ht_copy_ev,
                copy_ev,
            ) = ti._copy_usm_ndarray_into_usm_ndarray(
                src=src2,
                dst=buf,
                sycl_queue=exec_q,
                depends=dep_evs,
            )
            _manager.add_event_pair(ht_copy_ev, copy_ev)

            buf = dpt.broadcast_to(buf, res_shape)
            ht_, bf_ev = self.binary_inplace_fn_(
                lhs=o1,
                rhs=buf,
                sycl_queue=exec_q,
                depends=[copy_ev],
            )
            _manager.add_event_pair(ht_, bf_ev)

        return o1
