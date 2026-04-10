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
/// This file defines functions of dpnp.tensor._tensor_sorting_impl
/// extension.
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils/rich_comparisons.hpp"
#include "utils/type_dispatch.hpp"

#include "kernels/sorting/merge_sort.hpp"
#include "kernels/sorting/sort_impl_fn_ptr_t.hpp"

#include "merge_argsort.hpp"
#include "py_argsort_common.hpp"

namespace dpctl::tensor::py_internal
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::kernels::sort_contig_fn_ptr_t;
static sort_contig_fn_ptr_t
    ascending_argsort_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static sort_contig_fn_ptr_t
    descending_argsort_contig_dispatch_table[td_ns::num_types]
                                            [td_ns::num_types];

template <typename fnT, typename argTy, typename IndexTy>
struct AscendingArgSortContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<IndexTy, std::int64_t> ||
                      std::is_same_v<IndexTy, std::int32_t>)
        {
            using dpctl::tensor::rich_comparisons::AscendingSorter;
            using Comp = typename AscendingSorter<argTy>::type;

            using dpctl::tensor::kernels::stable_argsort_axis1_contig_impl;
            return stable_argsort_axis1_contig_impl<argTy, IndexTy, Comp>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename argTy, typename IndexTy>
struct DescendingArgSortContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<IndexTy, std::int64_t> ||
                      std::is_same_v<IndexTy, std::int32_t>)
        {
            using dpctl::tensor::rich_comparisons::DescendingSorter;
            using Comp = typename DescendingSorter<argTy>::type;

            using dpctl::tensor::kernels::stable_argsort_axis1_contig_impl;
            return stable_argsort_axis1_contig_impl<argTy, IndexTy, Comp>;
        }
        else {
            return nullptr;
        }
    }
};

void init_merge_argsort_dispatch_tables(void)
{
    using dpctl::tensor::kernels::sort_contig_fn_ptr_t;

    td_ns::DispatchTableBuilder<sort_contig_fn_ptr_t,
                                AscendingArgSortContigFactory, td_ns::num_types>
        dtb1;
    dtb1.populate_dispatch_table(ascending_argsort_contig_dispatch_table);

    td_ns::DispatchTableBuilder<
        sort_contig_fn_ptr_t, DescendingArgSortContigFactory, td_ns::num_types>
        dtb2;
    dtb2.populate_dispatch_table(descending_argsort_contig_dispatch_table);
}

void init_merge_argsort_functions(py::module_ m)
{
    init_merge_argsort_dispatch_tables();

    auto py_argsort_ascending = [](const dpctl::tensor::usm_ndarray &src,
                                   const int trailing_dims_to_sort,
                                   const dpctl::tensor::usm_ndarray &dst,
                                   sycl::queue &exec_q,
                                   const std::vector<sycl::event> &depends)
        -> std::pair<sycl::event, sycl::event> {
        return py_argsort(src, trailing_dims_to_sort, dst, exec_q, depends,
                          ascending_argsort_contig_dispatch_table);
    };
    m.def("_argsort_ascending", py_argsort_ascending, py::arg("src"),
          py::arg("trailing_dims_to_sort"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    auto py_argsort_descending = [](const dpctl::tensor::usm_ndarray &src,
                                    const int trailing_dims_to_sort,
                                    const dpctl::tensor::usm_ndarray &dst,
                                    sycl::queue &exec_q,
                                    const std::vector<sycl::event> &depends)
        -> std::pair<sycl::event, sycl::event> {
        return py_argsort(src, trailing_dims_to_sort, dst, exec_q, depends,
                          descending_argsort_contig_dispatch_table);
    };
    m.def("_argsort_descending", py_argsort_descending, py::arg("src"),
          py::arg("trailing_dims_to_sort"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    return;
}

} // namespace dpctl::tensor::py_internal
