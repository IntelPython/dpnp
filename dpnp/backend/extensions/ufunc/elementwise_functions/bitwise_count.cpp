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
#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"

#include "bitwise_count.hpp"
#include "kernels/elementwise_functions/bitwise_count.hpp"
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

namespace impl
{
namespace ew_cmn_ns = dpctl::tensor::kernels::elementwise_common;
namespace td_ns = dpctl::tensor::type_dispatch;

/**
 * @brief A factory to define pairs of supported types for which
 * sycl::bitwise_count<T> function is available.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct OutputType
{
    using value_type = typename std::disjunction<
        td_ns::TypeMapResultEntry<T, std::uint8_t, std::uint8_t>,
        td_ns::TypeMapResultEntry<T, std::int8_t, std::uint8_t>,
        td_ns::TypeMapResultEntry<T, std::uint16_t, std::uint8_t>,
        td_ns::TypeMapResultEntry<T, std::int16_t, std::uint8_t>,
        td_ns::TypeMapResultEntry<T, std::uint32_t, std::uint8_t>,
        td_ns::TypeMapResultEntry<T, std::int32_t, std::uint8_t>,
        td_ns::TypeMapResultEntry<T, std::uint64_t, std::uint8_t>,
        td_ns::TypeMapResultEntry<T, std::int64_t, std::uint8_t>,
        td_ns::DefaultResultEntry<void>>::result_type;
};

using dpnp::kernels::bitwise_count::BitwiseCountFunctor;

template <typename argT,
          typename resT = argT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using ContigFunctor =
    ew_cmn_ns::UnaryContigFunctor<argT,
                                  resT,
                                  BitwiseCountFunctor<argT, resT>,
                                  vec_sz,
                                  n_vecs,
                                  enable_sg_loadstore>;

template <typename argTy, typename resTy, typename IndexerT>
using StridedFunctor =
    ew_cmn_ns::UnaryStridedFunctor<argTy,
                                   resTy,
                                   IndexerT,
                                   BitwiseCountFunctor<argTy, resTy>>;

using ew_cmn_ns::unary_contig_impl_fn_ptr_t;
using ew_cmn_ns::unary_strided_impl_fn_ptr_t;

static unary_contig_impl_fn_ptr_t
    bitwise_count_contig_dispatch_vector[td_ns::num_types];
static int bitwise_count_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    bitwise_count_strided_dispatch_vector[td_ns::num_types];

MACRO_POPULATE_DISPATCH_VECTORS(bitwise_count);
} // namespace impl

void init_bitwise_count(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;
    {
        impl::populate_bitwise_count_dispatch_vectors();
        using impl::bitwise_count_contig_dispatch_vector;
        using impl::bitwise_count_output_typeid_vector;
        using impl::bitwise_count_strided_dispatch_vector;

        auto bitwise_count_pyapi = [&](const arrayT &src, const arrayT &dst,
                                       sycl::queue &exec_q,
                                       const event_vecT &depends = {}) {
            return py_int::py_unary_ufunc(
                src, dst, exec_q, depends, bitwise_count_output_typeid_vector,
                bitwise_count_contig_dispatch_vector,
                bitwise_count_strided_dispatch_vector);
        };
        m.def("_bitwise_count", bitwise_count_pyapi, "", py::arg("src"),
              py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());

        auto bitwise_count_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_int::py_unary_ufunc_result_type(
                dtype, bitwise_count_output_typeid_vector);
        };
        m.def("_bitwise_count_result_type", bitwise_count_result_type_pyapi);
    }
}
} // namespace dpnp::extensions::ufunc
