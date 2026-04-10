//*****************************************************************************
// Copyright (c) 2026, Intel Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// - Neither the name of the copyright holder nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
// THE POSSIBILITY OF SUCH DAMAGE.
//*****************************************************************************
//
//===---------------------------------------------------------------------===//
///
/// \file
/// This file defines functions of dpnp.tensor._tensor_reductions_impl
/// extension.
//===---------------------------------------------------------------------===//

#include <cstdint>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kernels/reductions.hpp"
#include "reduction_atomic_support.hpp"
#include "reduction_over_axis.hpp"
#include "utils/type_dispatch.hpp"

namespace dpctl::tensor::py_internal
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

namespace impl
{

using dpctl::tensor::kernels::reduction_strided_impl_fn_ptr;
static reduction_strided_impl_fn_ptr
    any_reduction_strided_dispatch_vector[td_ns::num_types];

using dpctl::tensor::kernels::reduction_contig_impl_fn_ptr;
static reduction_contig_impl_fn_ptr
    any_reduction_axis1_contig_dispatch_vector[td_ns::num_types];
static reduction_contig_impl_fn_ptr
    any_reduction_axis0_contig_dispatch_vector[td_ns::num_types];

template <typename fnT, typename srcTy>
struct AnyStridedFactory
{
    fnT get() const
    {
        using dstTy = std::int32_t;
        using ReductionOpT = sycl::logical_or<dstTy>;
        return dpctl::tensor::kernels::
            reduction_over_group_with_atomics_strided_impl<srcTy, dstTy,
                                                           ReductionOpT>;
    }
};

template <typename fnT, typename srcTy>
struct AnyAxis1ContigFactory
{
    fnT get() const
    {
        using dstTy = std::int32_t;
        using ReductionOpT = sycl::logical_or<dstTy>;
        return dpctl::tensor::kernels::
            reduction_axis1_over_group_with_atomics_contig_impl<srcTy, dstTy,
                                                                ReductionOpT>;
    }
};

template <typename fnT, typename srcTy>
struct AnyAxis0ContigFactory
{
    fnT get() const
    {
        using dstTy = std::int32_t;
        using ReductionOpT = sycl::logical_or<dstTy>;
        return dpctl::tensor::kernels::
            reduction_axis0_over_group_with_atomics_contig_impl<srcTy, dstTy,
                                                                ReductionOpT>;
    }
};

void populate_any_dispatch_vectors(void)
{
    using td_ns::DispatchVectorBuilder;

    DispatchVectorBuilder<reduction_strided_impl_fn_ptr, AnyStridedFactory,
                          td_ns::num_types>
        any_dvb1;
    any_dvb1.populate_dispatch_vector(any_reduction_strided_dispatch_vector);

    DispatchVectorBuilder<reduction_contig_impl_fn_ptr, AnyAxis1ContigFactory,
                          td_ns::num_types>
        any_dvb2;
    any_dvb2.populate_dispatch_vector(
        any_reduction_axis1_contig_dispatch_vector);

    DispatchVectorBuilder<reduction_contig_impl_fn_ptr, AnyAxis0ContigFactory,
                          td_ns::num_types>
        any_dvb3;
    any_dvb3.populate_dispatch_vector(
        any_reduction_axis0_contig_dispatch_vector);
};

using atomic_support::atomic_support_fn_ptr_t;
using atomic_support::check_atomic_support;
static atomic_support_fn_ptr_t any_atomic_support =
    check_atomic_support<std::int32_t>;

} // namespace impl

void init_any(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;
    {
        impl::populate_any_dispatch_vectors();
        using impl::any_reduction_axis0_contig_dispatch_vector;
        using impl::any_reduction_axis1_contig_dispatch_vector;
        using impl::any_reduction_strided_dispatch_vector;

        using impl::any_atomic_support;

        auto any_pyapi = [&](const arrayT &src, int trailing_dims_to_reduce,
                             const arrayT &dst, sycl::queue &exec_q,
                             const event_vecT &depends = {}) {
            return py_boolean_reduction(
                src, trailing_dims_to_reduce, dst, exec_q, depends,
                any_reduction_axis1_contig_dispatch_vector,
                any_reduction_axis0_contig_dispatch_vector,
                any_reduction_strided_dispatch_vector, any_atomic_support);
        };
        m.def("_any", any_pyapi, "", py::arg("src"),
              py::arg("trailing_dims_to_reduce"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());
    }
}

} // namespace dpctl::tensor::py_internal
