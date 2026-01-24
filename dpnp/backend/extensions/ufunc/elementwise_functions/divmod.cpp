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

#include <complex>
#include <type_traits>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"

#include "divmod.hpp"
#include "kernels/elementwise_functions/divmod.hpp"
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

using dpnp::kernels::divmod::DivmodFunctor;

template <typename T1, typename T2>
struct OutputType
{
    using table_type = typename std::disjunction< // disjunction is C++17
                                                  // feature, supported by DPC++
        td_int_ns::
            BinaryTypeMapTwoResultsEntry<T1, std::uint8_t, T2, std::uint8_t>,
        td_int_ns::
            BinaryTypeMapTwoResultsEntry<T1, std::int8_t, T2, std::int8_t>,
        td_int_ns::
            BinaryTypeMapTwoResultsEntry<T1, std::uint16_t, T2, std::uint16_t>,
        td_int_ns::
            BinaryTypeMapTwoResultsEntry<T1, std::int16_t, T2, std::int16_t>,
        td_int_ns::
            BinaryTypeMapTwoResultsEntry<T1, std::uint32_t, T2, std::uint32_t>,
        td_int_ns::
            BinaryTypeMapTwoResultsEntry<T1, std::int32_t, T2, std::int32_t>,
        td_int_ns::
            BinaryTypeMapTwoResultsEntry<T1, std::uint64_t, T2, std::uint64_t>,
        td_int_ns::
            BinaryTypeMapTwoResultsEntry<T1, std::int64_t, T2, std::int64_t>,
        td_int_ns::BinaryTypeMapTwoResultsEntry<T1, sycl::half, T2, sycl::half>,
        td_int_ns::BinaryTypeMapTwoResultsEntry<T1, float, T2, float>,
        td_int_ns::BinaryTypeMapTwoResultsEntry<T1, double, T2, double>,
        td_int_ns::DefaultTwoResultsEntry<void>>;
    using value_type1 = typename table_type::result_type1;
    using value_type2 = typename table_type::result_type2;
};

template <typename argTy1,
          typename argTy2,
          typename resTy1,
          typename resTy2,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using ContigFunctor = ew_cmn_ns::BinaryTwoOutputsContigFunctor<
    argTy1,
    argTy2,
    resTy1,
    resTy2,
    DivmodFunctor<argTy1, argTy2, resTy1, resTy2>,
    vec_sz,
    n_vecs,
    enable_sg_loadstore>;

template <typename argTy1,
          typename argTy2,
          typename resTy1,
          typename resTy2,
          typename IndexerT>
using StridedFunctor = ew_cmn_ns::BinaryTwoOutputsStridedFunctor<
    argTy1,
    argTy2,
    resTy1,
    resTy2,
    IndexerT,
    DivmodFunctor<argTy1, argTy2, resTy1, resTy2>>;

using ew_cmn_ns::binary_two_outputs_contig_impl_fn_ptr_t;
using ew_cmn_ns::binary_two_outputs_strided_impl_fn_ptr_t;

static binary_two_outputs_contig_impl_fn_ptr_t
    divmod_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static std::pair<int, int> divmod_output_typeid_table[td_ns::num_types]
                                                     [td_ns::num_types];
static binary_two_outputs_strided_impl_fn_ptr_t
    divmod_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

MACRO_POPULATE_DISPATCH_2OUTS_TABLES(divmod);
} // namespace impl

void init_divmod(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;
    {
        impl::populate_divmod_dispatch_tables();
        using impl::divmod_contig_dispatch_table;
        using impl::divmod_output_typeid_table;
        using impl::divmod_strided_dispatch_table;

        auto divmod_pyapi = [&](const arrayT &src1, const arrayT &src2,
                                const arrayT &dst1, const arrayT &dst2,
                                sycl::queue &exec_q,
                                const event_vecT &depends = {}) {
            return py_int::py_binary_two_outputs_ufunc(
                src1, src2, dst1, dst2, exec_q, depends,
                divmod_output_typeid_table, divmod_contig_dispatch_table,
                divmod_strided_dispatch_table);
        };
        m.def("_divmod", divmod_pyapi, "", py::arg("src1"), py::arg("src2"),
              py::arg("dst1"), py::arg("dst2"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());

        auto divmod_result_type_pyapi = [&](const py::dtype &dtype1,
                                            const py::dtype &dtype2) {
            return py_int::py_binary_two_outputs_ufunc_result_type(
                dtype1, dtype2, divmod_output_typeid_table);
        };
        m.def("_divmod_result_type", divmod_result_type_pyapi);
    }
}
} // namespace dpnp::extensions::ufunc
