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

import dpctl
import dpctl.tensor as dpt
from dpctl.utils import ExecutionPlacementError, SequentialOrderManager

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt_ext
import dpctl_ext.tensor._tensor_impl as ti

from ._copy_utils import _empty_like_orderK
from ._type_utils import (
    _acceptance_fn_default_unary,
    _all_data_types,
    _find_buf_dtype,
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
                out = dpt_ext.empty_like(out)

            if (
                dpctl.utils.get_execution_queue((x.sycl_queue, out.sycl_queue))
                is None
            ):
                raise ExecutionPlacementError(
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
                    out = dpt_ext.empty_like(x, dtype=res_dt, order=order)

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
            buf = dpt_ext.empty_like(x, dtype=buf_dt, order=order)

        dep_evs = _manager.submitted_events
        ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=x, dst=buf, sycl_queue=exec_q, depends=dep_evs
        )
        _manager.add_event_pair(ht_copy_ev, copy_ev)
        if out is None:
            if order == "K":
                out = _empty_like_orderK(buf, res_dt)
            else:
                out = dpt_ext.empty_like(buf, dtype=res_dt, order=order)

        ht, uf_ev = self.unary_fn_(
            buf, out, sycl_queue=exec_q, depends=[copy_ev]
        )
        _manager.add_event_pair(ht, uf_ev)

        return out
