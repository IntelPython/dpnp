//*****************************************************************************
// Copyright (c) 2024, Intel Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// maxification, are permitted provided that the following conditions are met:
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
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

#include <sycl/sycl.hpp>

#include "dpctl4pybind11.hpp"

#include "heaviside.hpp"
#include "kernels/elementwise_functions/heaviside.hpp"
#include "populate.hpp"

// include a local copy of elementwise common header from dpctl tensor:
// dpctl/tensor/libtensor/source/elementwise_functions/elementwise_functions.hpp
// TODO: replace by including dpctl header once available
#include "../../elementwise_functions/elementwise_functions.hpp"

// dpctl tensor headers
#include "kernels/elementwise_functions/common.hpp"
#include "utils/type_dispatch.hpp"

namespace dpnp::extensions::ufunc
{
namespace py = pybind11;
namespace py_int = dpnp::extensions::py_internal;
namespace td_ns = dpctl::tensor::type_dispatch;

namespace impl
{
namespace ew_cmn_ns = dpctl::tensor::kernels::elementwise_common;

template <typename T1, typename T2>
struct OutputType
{
    using value_type = typename std::disjunction< // disjunction is C++17
                                                  // feature, supported by DPC++
        td_ns::BinaryTypeMapResultEntry<T1,
                                        sycl::half,
                                        T2,
                                        sycl::half,
                                        sycl::half>,
        td_ns::BinaryTypeMapResultEntry<T1, float, T2, float, float>,
        td_ns::BinaryTypeMapResultEntry<T1, double, T2, double, double>,
        td_ns::DefaultResultEntry<void>>::result_type;
};

using dpnp::kernels::heaviside::HeavisideFunctor;

template <typename argT1,
          typename argT2,
          typename resT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using ContigFunctor =
    ew_cmn_ns::BinaryContigFunctor<argT1,
                                   argT2,
                                   resT,
                                   HeavisideFunctor<argT1, argT2, resT>,
                                   vec_sz,
                                   n_vecs,
                                   enable_sg_loadstore>;

template <typename argT1, typename argT2, typename resT, typename IndexerT>
using StridedFunctor =
    ew_cmn_ns::BinaryStridedFunctor<argT1,
                                    argT2,
                                    resT,
                                    IndexerT,
                                    HeavisideFunctor<argT1, argT2, resT>>;

using ew_cmn_ns::binary_contig_impl_fn_ptr_t;
using ew_cmn_ns::binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t;
using ew_cmn_ns::binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t;
using ew_cmn_ns::binary_strided_impl_fn_ptr_t;

static binary_contig_impl_fn_ptr_t
    heaviside_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static int heaviside_output_typeid_table[td_ns::num_types][td_ns::num_types];
static binary_strided_impl_fn_ptr_t
    heaviside_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

MACRO_POPULATE_DISPATCH_TABLES(heaviside);
} // namespace impl

void init_heaviside(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;
    {
        impl::populate_heaviside_dispatch_tables();
        using impl::heaviside_contig_dispatch_table;
        using impl::heaviside_output_typeid_table;
        using impl::heaviside_strided_dispatch_table;

        auto heaviside_pyapi = [&](const arrayT &src1, const arrayT &src2,
                                   const arrayT &dst, sycl::queue &exec_q,
                                   const event_vecT &depends = {}) {
            return py_int::py_binary_ufunc(
                src1, src2, dst, exec_q, depends, heaviside_output_typeid_table,
                heaviside_contig_dispatch_table,
                heaviside_strided_dispatch_table,
                // no dedicated kernel for C-contig row with broadcasting
                td_ns::NullPtrTable<
                    impl::
                        binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                td_ns::NullPtrTable<
                    impl::
                        binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        m.def("_heaviside", heaviside_pyapi, "", py::arg("src1"),
              py::arg("src2"), py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());

        auto heaviside_result_type_pyapi = [&](const py::dtype &dtype1,
                                               const py::dtype &dtype2) {
            return py_int::py_binary_ufunc_result_type(
                dtype1, dtype2, heaviside_output_typeid_table);
        };
        m.def("_heaviside_result_type", heaviside_result_type_pyapi);
    }
}
} // namespace dpnp::extensions::ufunc