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

#include <cstddef>
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

#include "bitonic_sort.hpp"
#include "py_sort_common.hpp"

namespace td_ns = dpnp::tensor::type_dispatch;

namespace dpnp::tensor::py_internal
{

using dpnp::tensor::kernels::sort_contig_fn_ptr_t;
static sort_contig_fn_ptr_t
    ascending_bitonic_sort_contig_dispatch_vector[td_ns::num_types];
static sort_contig_fn_ptr_t
    descending_bitonic_sort_contig_dispatch_vector[td_ns::num_types];

template <typename fnT, typename argTy>
struct AscendingBitonicSortContigFactory
{
    fnT get()
    {
        using dpnp::tensor::rich_comparisons::AscendingSorter;
        using Comp = typename AscendingSorter<argTy>::type;

        using dpnp::tensor::kernels::bitonic_sort_axis1_contig_impl;
        return bitonic_sort_axis1_contig_impl<argTy, Comp,
                                              /*ascending*/ true>;
    }
};

template <typename fnT, typename argTy>
struct DescendingBitonicSortContigFactory
{
    fnT get()
    {
        using dpnp::tensor::rich_comparisons::DescendingSorter;
        using Comp = typename DescendingSorter<argTy>::type;

        using dpnp::tensor::kernels::bitonic_sort_axis1_contig_impl;
        return bitonic_sort_axis1_contig_impl<argTy, Comp,
                                              /*ascending*/ false>;
    }
};

using bitonic_local_memory_nelems_fn_ptr_t =
    std::size_t (*)(const sycl::device &);

static bitonic_local_memory_nelems_fn_ptr_t
    bitonic_sort_local_memory_nelems_dispatch_vector[td_ns::num_types];

template <typename fnT, typename argTy>
struct BitonicSortLocalMemoryNelemsFactory
{
    fnT get()
    {
        using dpnp::tensor::kernels::bitonic_sort_local_memory_nelems;
        return bitonic_sort_local_memory_nelems<argTy>;
    }
};

void init_bitonic_sort_dispatch_vectors(void)
{
    using dpnp::tensor::kernels::sort_contig_fn_ptr_t;

    td_ns::DispatchVectorBuilder<sort_contig_fn_ptr_t,
                                 AscendingBitonicSortContigFactory,
                                 td_ns::num_types>
        dtv1;
    dtv1.populate_dispatch_vector(
        ascending_bitonic_sort_contig_dispatch_vector);

    td_ns::DispatchVectorBuilder<sort_contig_fn_ptr_t,
                                 DescendingBitonicSortContigFactory,
                                 td_ns::num_types>
        dtv2;
    dtv2.populate_dispatch_vector(
        descending_bitonic_sort_contig_dispatch_vector);

    td_ns::DispatchVectorBuilder<bitonic_local_memory_nelems_fn_ptr_t,
                                 BitonicSortLocalMemoryNelemsFactory,
                                 td_ns::num_types>
        dtv3;
    dtv3.populate_dispatch_vector(
        bitonic_sort_local_memory_nelems_dispatch_vector);
}

/*! @brief Longest row the bitonic sort of arrays of the given type sorts in
 *  local memory, or zero for a type it does not support */
std::size_t py_bitonic_sort_local_memory_nelems(int typenum,
                                                const sycl::queue &exec_q)
{
    const auto &array_types = td_ns::usm_ndarray_types();

    try {
        const int type_id = array_types.typenum_to_lookup_id(typenum);
        auto fn = bitonic_sort_local_memory_nelems_dispatch_vector[type_id];

        return (nullptr == fn) ? std::size_t(0) : fn(exec_q.get_device());
    } catch (const std::exception &e) {
        return std::size_t(0);
    }
}

void init_bitonic_sort_functions(py::module_ m)
{
    init_bitonic_sort_dispatch_vectors();

    auto py_bitonic_sort_ascending = [](const dpnp::tensor::usm_ndarray &src,
                                        const int trailing_dims_to_sort,
                                        const dpnp::tensor::usm_ndarray &dst,
                                        sycl::queue &exec_q,
                                        const std::vector<sycl::event> &depends)
        -> std::pair<sycl::event, sycl::event> {
        return py_sort(src, trailing_dims_to_sort, dst, exec_q, depends,
                       ascending_bitonic_sort_contig_dispatch_vector);
    };
    m.def("_bitonic_sort_ascending", py_bitonic_sort_ascending, py::arg("src"),
          py::arg("trailing_dims_to_sort"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    auto py_bitonic_sort_descending =
        [](const dpnp::tensor::usm_ndarray &src,
           const int trailing_dims_to_sort,
           const dpnp::tensor::usm_ndarray &dst, sycl::queue &exec_q,
           const std::vector<sycl::event> &depends)
        -> std::pair<sycl::event, sycl::event> {
        return py_sort(src, trailing_dims_to_sort, dst, exec_q, depends,
                       descending_bitonic_sort_contig_dispatch_vector);
    };
    m.def("_bitonic_sort_descending", py_bitonic_sort_descending,
          py::arg("src"), py::arg("trailing_dims_to_sort"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    m.def("_bitonic_sort_local_memory_nelems",
          py_bitonic_sort_local_memory_nelems, py::arg("typenum"),
          py::arg("sycl_queue"));

    return;
}

} // namespace dpnp::tensor::py_internal
