//*****************************************************************************
// Copyright (c) 2025, Intel Corporation
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

#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"

#include "frexp.hpp"
#include "kernels/elementwise_functions/frexp.hpp"
#include "populate.hpp"

// include a local copy of elementwise common header from dpctl tensor:
// dpctl/tensor/libtensor/source/elementwise_functions/elementwise_functions.hpp
// TODO: replace by including dpctl header once available
#include "../../elementwise_functions/elementwise_functions.hpp"

#include "../../elementwise_functions/common.hpp"
#include "../../elementwise_functions/type_dispatch_building.hpp"

// utils extension header
#include "ext/common.hpp"

// dpctl tensor headers
#include "kernels/elementwise_functions/common.hpp"
#include "utils/type_dispatch.hpp"

namespace dpnp::extensions::ufunc
{
namespace py = pybind11;
namespace py_int = dpnp::extensions::py_internal;

namespace impl
{
namespace ew_cmn_ns = dpnp::extensions::py_internal::elementwise_common;
namespace td_int_ns = py_int::type_dispatch;
namespace td_ns = dpctl::tensor::type_dispatch;

using dpnp::kernels::frexp::FrexpFunctor;

template <typename T>
struct OutputType
{
    using table_type = std::disjunction< // disjunction is C++17
                                         // feature, supported by DPC++
        td_int_ns::
            TypeMapTwoResultsEntry<T, sycl::half, sycl::half, std::int32_t>,
        td_int_ns::TypeMapTwoResultsEntry<T, float, float, std::int32_t>,
        td_int_ns::TypeMapTwoResultsEntry<T, double, double, std::int32_t>,
        td_int_ns::DefaultTwoResultsEntry<void>>;
    using value_type1 = typename table_type::result_type1;
    using value_type2 = typename table_type::result_type2;
};

template <typename argTy,
          typename resTy1 = argTy,
          typename resTy2 = argTy,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
using ContigFunctor =
    ew_cmn_ns::UnaryTwoOutputsContigFunctor<argTy,
                                            resTy1,
                                            resTy2,
                                            FrexpFunctor<argTy, resTy1, resTy2>,
                                            vec_sz,
                                            n_vecs,
                                            enable_sg_loadstore>;

template <typename argTy, typename resTy1, typename resTy2, typename IndexerT>
using StridedFunctor = ew_cmn_ns::UnaryTwoOutputsStridedFunctor<
    argTy,
    resTy1,
    resTy2,
    IndexerT,
    FrexpFunctor<argTy, resTy1, resTy2>>;

using ew_cmn_ns::unary_two_outputs_contig_impl_fn_ptr_t;
using ew_cmn_ns::unary_two_outputs_strided_impl_fn_ptr_t;

static unary_two_outputs_contig_impl_fn_ptr_t
    frexp_contig_dispatch_vector[td_ns::num_types];
static std::pair<int, int> frexp_output_typeid_vector[td_ns::num_types];
static unary_two_outputs_strided_impl_fn_ptr_t
    frexp_strided_dispatch_vector[td_ns::num_types];

MACRO_POPULATE_DISPATCH_2OUTS_VECTORS(frexp);
} // namespace impl

void init_frexp(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;
    {
        impl::populate_frexp_dispatch_vectors();
        using impl::frexp_contig_dispatch_vector;
        using impl::frexp_output_typeid_vector;
        using impl::frexp_strided_dispatch_vector;

        auto frexp_pyapi = [&](const arrayT &src, const arrayT &dst1,
                               const arrayT &dst2, sycl::queue &exec_q,
                               const event_vecT &depends = {}) {
            return py_int::py_unary_two_outputs_ufunc(
                src, dst1, dst2, exec_q, depends, frexp_output_typeid_vector,
                frexp_contig_dispatch_vector, frexp_strided_dispatch_vector);
        };
        m.def("_frexp", frexp_pyapi, "", py::arg("src"), py::arg("dst1"),
              py::arg("dst2"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());

        auto frexp_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_int::py_unary_two_outputs_ufunc_result_type(
                dtype, frexp_output_typeid_vector);
        };
        m.def("_frexp_result_type", frexp_result_type_pyapi);
    }
}
} // namespace dpnp::extensions::ufunc
