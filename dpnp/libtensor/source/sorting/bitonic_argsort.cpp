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
/// This file defines functions of dpnp.libtensor._tensor_sorting_impl
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

#include "kernels/sorting/bitonic_sort.hpp"
#include "kernels/sorting/sort_impl_fn_ptr_t.hpp"

#include "bitonic_argsort.hpp"
#include "py_argsort_common.hpp"

namespace dpnp::tensor::py_internal
{

namespace py = pybind11;
namespace td_ns = dpnp::tensor::type_dispatch;

using dpnp::tensor::kernels::sort_contig_fn_ptr_t;
static sort_contig_fn_ptr_t
    ascending_bitonic_argsort_contig_dispatch_table[td_ns::num_types]
                                                   [td_ns::num_types];
static sort_contig_fn_ptr_t
    descending_bitonic_argsort_contig_dispatch_table[td_ns::num_types]
                                                    [td_ns::num_types];

template <typename fnT, typename argTy, typename IndexTy>
struct AscendingBitonicArgSortContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<IndexTy, std::int64_t> ||
                      std::is_same_v<IndexTy, std::int32_t>) {
            using dpnp::tensor::rich_comparisons::AscendingSorter;
            using Comp = typename AscendingSorter<argTy>::type;

            using dpnp::tensor::kernels::bitonic_argsort_axis1_contig_impl;
            return bitonic_argsort_axis1_contig_impl<argTy, IndexTy, Comp,
                                                     /*ascending*/ true>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename argTy, typename IndexTy>
struct DescendingBitonicArgSortContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<IndexTy, std::int64_t> ||
                      std::is_same_v<IndexTy, std::int32_t>) {
            using dpnp::tensor::rich_comparisons::DescendingSorter;
            using Comp = typename DescendingSorter<argTy>::type;

            using dpnp::tensor::kernels::bitonic_argsort_axis1_contig_impl;
            return bitonic_argsort_axis1_contig_impl<argTy, IndexTy, Comp,
                                                     /*ascending*/ false>;
        }
        else {
            return nullptr;
        }
    }
};

using bitonic_local_memory_nelems_fn_ptr_t =
    std::size_t (*)(const sycl::device &);

static bitonic_local_memory_nelems_fn_ptr_t
    bitonic_argsort_local_memory_nelems_dispatch_table[td_ns::num_types]
                                                      [td_ns::num_types];

template <typename fnT, typename argTy, typename IndexTy>
struct BitonicArgSortLocalMemoryNelemsFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<IndexTy, std::int64_t> ||
                      std::is_same_v<IndexTy, std::int32_t>) {
            using dpnp::tensor::kernels::bitonic_argsort_local_memory_nelems;
            return bitonic_argsort_local_memory_nelems<argTy, IndexTy>;
        }
        else {
            return nullptr;
        }
    }
};

void init_bitonic_argsort_dispatch_tables(void)
{
    using dpnp::tensor::kernels::sort_contig_fn_ptr_t;

    td_ns::DispatchTableBuilder<sort_contig_fn_ptr_t,
                                AscendingBitonicArgSortContigFactory,
                                td_ns::num_types>
        dtb1;
    dtb1.populate_dispatch_table(
        ascending_bitonic_argsort_contig_dispatch_table);

    td_ns::DispatchTableBuilder<sort_contig_fn_ptr_t,
                                DescendingBitonicArgSortContigFactory,
                                td_ns::num_types>
        dtb2;
    dtb2.populate_dispatch_table(
        descending_bitonic_argsort_contig_dispatch_table);

    td_ns::DispatchTableBuilder<bitonic_local_memory_nelems_fn_ptr_t,
                                BitonicArgSortLocalMemoryNelemsFactory,
                                td_ns::num_types>
        dtb3;
    dtb3.populate_dispatch_table(
        bitonic_argsort_local_memory_nelems_dispatch_table);
}

/*! @brief Longest row the bitonic argsort of arrays of the given types sorts in
 *  local memory, or zero for types it does not support */
std::size_t py_bitonic_argsort_local_memory_nelems(int typenum,
                                                   int index_typenum,
                                                   const sycl::queue &exec_q)
{
    const auto &array_types = td_ns::usm_ndarray_types();

    try {
        const int type_id = array_types.typenum_to_lookup_id(typenum);
        const int index_type_id =
            array_types.typenum_to_lookup_id(index_typenum);
        auto fn =
            bitonic_argsort_local_memory_nelems_dispatch_table[type_id]
                                                              [index_type_id];

        return (nullptr == fn) ? std::size_t(0) : fn(exec_q.get_device());
    } catch (const std::exception &e) {
        return std::size_t(0);
    }
}

void init_bitonic_argsort_functions(py::module_ m)
{
    init_bitonic_argsort_dispatch_tables();

    auto py_bitonic_argsort_ascending =
        [](const dpnp::tensor::usm_ndarray &src,
           const int trailing_dims_to_sort,
           const dpnp::tensor::usm_ndarray &dst, sycl::queue &exec_q,
           const std::vector<sycl::event> &depends)
        -> std::pair<sycl::event, sycl::event> {
        return py_argsort(src, trailing_dims_to_sort, dst, exec_q, depends,
                          ascending_bitonic_argsort_contig_dispatch_table);
    };
    m.def("_bitonic_argsort_ascending", py_bitonic_argsort_ascending,
          py::arg("src"), py::arg("trailing_dims_to_sort"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    auto py_bitonic_argsort_descending =
        [](const dpnp::tensor::usm_ndarray &src,
           const int trailing_dims_to_sort,
           const dpnp::tensor::usm_ndarray &dst, sycl::queue &exec_q,
           const std::vector<sycl::event> &depends)
        -> std::pair<sycl::event, sycl::event> {
        return py_argsort(src, trailing_dims_to_sort, dst, exec_q, depends,
                          descending_bitonic_argsort_contig_dispatch_table);
    };
    m.def("_bitonic_argsort_descending", py_bitonic_argsort_descending,
          py::arg("src"), py::arg("trailing_dims_to_sort"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    m.def("_bitonic_argsort_local_memory_nelems",
          py_bitonic_argsort_local_memory_nelems, py::arg("typenum"),
          py::arg("index_typenum"), py::arg("sycl_queue"));

    return;
}

} // namespace dpnp::tensor::py_internal
