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
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines functions of dpctl.tensor._tensor_sorting_impl
/// extension.
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils/rich_comparisons.hpp"
#include "utils/type_dispatch.hpp"

#include "kernels/sorting/merge_sort.hpp"
#include "kernels/sorting/sort_impl_fn_ptr_t.hpp"

#include "merge_sort.hpp"
#include "py_sort_common.hpp"

namespace td_ns = dpctl::tensor::type_dispatch;

namespace dpctl::tensor::py_internal
{

using dpctl::tensor::kernels::sort_contig_fn_ptr_t;
static sort_contig_fn_ptr_t
    ascending_sort_contig_dispatch_vector[td_ns::num_types];
static sort_contig_fn_ptr_t
    descending_sort_contig_dispatch_vector[td_ns::num_types];

template <typename fnT, typename argTy>
struct AscendingSortContigFactory
{
    fnT get()
    {
        using dpctl::tensor::rich_comparisons::AscendingSorter;
        using Comp = typename AscendingSorter<argTy>::type;

        using dpctl::tensor::kernels::stable_sort_axis1_contig_impl;
        return stable_sort_axis1_contig_impl<argTy, Comp>;
    }
};

template <typename fnT, typename argTy>
struct DescendingSortContigFactory
{
    fnT get()
    {
        using dpctl::tensor::rich_comparisons::DescendingSorter;
        using Comp = typename DescendingSorter<argTy>::type;

        using dpctl::tensor::kernels::stable_sort_axis1_contig_impl;
        return stable_sort_axis1_contig_impl<argTy, Comp>;
    }
};

void init_merge_sort_dispatch_vectors(void)
{
    using dpctl::tensor::kernels::sort_contig_fn_ptr_t;

    td_ns::DispatchVectorBuilder<sort_contig_fn_ptr_t,
                                 AscendingSortContigFactory, td_ns::num_types>
        dtv1;
    dtv1.populate_dispatch_vector(ascending_sort_contig_dispatch_vector);

    td_ns::DispatchVectorBuilder<sort_contig_fn_ptr_t,
                                 DescendingSortContigFactory, td_ns::num_types>
        dtv2;
    dtv2.populate_dispatch_vector(descending_sort_contig_dispatch_vector);
}

void init_merge_sort_functions(py::module_ m)
{
    dpctl::tensor::py_internal::init_merge_sort_dispatch_vectors();

    auto py_sort_ascending = [](const dpctl::tensor::usm_ndarray &src,
                                const int trailing_dims_to_sort,
                                const dpctl::tensor::usm_ndarray &dst,
                                sycl::queue &exec_q,
                                const std::vector<sycl::event> &depends)
        -> std::pair<sycl::event, sycl::event> {
        return dpctl::tensor::py_internal::py_sort(
            src, trailing_dims_to_sort, dst, exec_q, depends,
            dpctl::tensor::py_internal::ascending_sort_contig_dispatch_vector);
    };
    m.def("_sort_ascending", py_sort_ascending, py::arg("src"),
          py::arg("trailing_dims_to_sort"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    auto py_sort_descending = [](const dpctl::tensor::usm_ndarray &src,
                                 const int trailing_dims_to_sort,
                                 const dpctl::tensor::usm_ndarray &dst,
                                 sycl::queue &exec_q,
                                 const std::vector<sycl::event> &depends)
        -> std::pair<sycl::event, sycl::event> {
        return dpctl::tensor::py_internal::py_sort(
            src, trailing_dims_to_sort, dst, exec_q, depends,
            dpctl::tensor::py_internal::descending_sort_contig_dispatch_vector);
    };
    m.def("_sort_descending", py_sort_descending, py::arg("src"),
          py::arg("trailing_dims_to_sort"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    return;
}

} // namespace dpctl::tensor::py_internal
