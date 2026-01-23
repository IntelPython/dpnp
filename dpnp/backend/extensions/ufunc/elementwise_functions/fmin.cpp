//*****************************************************************************
// Copyright (c) 2024, Intel Corporation
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

#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"

#include "fmin.hpp"
#include "kernels/elementwise_functions/fmin.hpp"
#include "populate.hpp"

// include a local copy of elementwise common header from dpctl tensor:
// dpctl/tensor/libtensor/source/elementwise_functions/elementwise_functions.hpp
// TODO: replace by including dpctl header once available
#include "../../elementwise_functions/elementwise_functions.hpp"

// dpctl tensor headers
#include "kernels/elementwise_functions/common.hpp"
#include "kernels/elementwise_functions/minimum.hpp"
#include "utils/type_dispatch.hpp"

namespace dpnp::extensions::ufunc
{
namespace py = pybind11;
namespace py_int = dpnp::extensions::py_internal;
namespace td_ns = dpctl::tensor::type_dispatch;

namespace impl
{
namespace ew_cmn_ns = dpctl::tensor::kernels::elementwise_common;
namespace min_ns = dpctl::tensor::kernels::minimum;

// Supports the same types table as for minimum function in dpctl
template <typename T1, typename T2>
using OutputType = min_ns::MinimumOutputType<T1, T2>;

using dpnp::kernels::fmin::FminFunctor;

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
                                   FminFunctor<argT1, argT2, resT>,
                                   vec_sz,
                                   n_vecs,
                                   enable_sg_loadstore>;

template <typename argT1, typename argT2, typename resT, typename IndexerT>
using StridedFunctor =
    ew_cmn_ns::BinaryStridedFunctor<argT1,
                                    argT2,
                                    resT,
                                    IndexerT,
                                    FminFunctor<argT1, argT2, resT>>;

using ew_cmn_ns::binary_contig_impl_fn_ptr_t;
using ew_cmn_ns::binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t;
using ew_cmn_ns::binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t;
using ew_cmn_ns::binary_strided_impl_fn_ptr_t;

static binary_contig_impl_fn_ptr_t fmin_contig_dispatch_table[td_ns::num_types]
                                                             [td_ns::num_types];
static int fmin_output_typeid_table[td_ns::num_types][td_ns::num_types];
static binary_strided_impl_fn_ptr_t
    fmin_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

MACRO_POPULATE_DISPATCH_TABLES(fmin);
} // namespace impl

void init_fmin(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;
    {
        impl::populate_fmin_dispatch_tables();
        using impl::fmin_contig_dispatch_table;
        using impl::fmin_output_typeid_table;
        using impl::fmin_strided_dispatch_table;

        auto fmin_pyapi = [&](const arrayT &src1, const arrayT &src2,
                              const arrayT &dst, sycl::queue &exec_q,
                              const event_vecT &depends = {}) {
            return py_int::py_binary_ufunc(
                src1, src2, dst, exec_q, depends, fmin_output_typeid_table,
                fmin_contig_dispatch_table, fmin_strided_dispatch_table,
                // no support of C-contig row with broadcasting in OneMKL
                td_ns::NullPtrTable<
                    impl::
                        binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                td_ns::NullPtrTable<
                    impl::
                        binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        m.def("_fmin", fmin_pyapi, "", py::arg("src1"), py::arg("src2"),
              py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());

        auto fmin_result_type_pyapi = [&](const py::dtype &dtype1,
                                          const py::dtype &dtype2) {
            return py_int::py_binary_ufunc_result_type(
                dtype1, dtype2, fmin_output_typeid_table);
        };
        m.def("_fmin_result_type", fmin_result_type_pyapi);
    }
}
} // namespace dpnp::extensions::ufunc
