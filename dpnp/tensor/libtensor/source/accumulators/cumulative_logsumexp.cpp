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
/// This file defines functions of dpctl.tensor._tensor_accumulation_impl
//  extensions
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <type_traits>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "accumulate_over_axis.hpp"
#include "kernels/accumulators.hpp"
#include "utils/sycl_utils.hpp"
#include "utils/type_dispatch_building.hpp"

namespace py = pybind11;

namespace dpctl::tensor::py_internal
{

namespace su_ns = dpctl::tensor::sycl_utils;
namespace td_ns = dpctl::tensor::type_dispatch;

namespace impl
{

using dpctl::tensor::kernels::accumulators::accumulate_1d_contig_impl_fn_ptr_t;
static accumulate_1d_contig_impl_fn_ptr_t
    cumlogsumexp_1d_contig_dispatch_table[td_ns::num_types][td_ns::num_types];

using dpctl::tensor::kernels::accumulators::accumulate_strided_impl_fn_ptr_t;
static accumulate_strided_impl_fn_ptr_t
    cumlogsumexp_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

static accumulate_1d_contig_impl_fn_ptr_t
    cumlogsumexp_1d_include_initial_contig_dispatch_table[td_ns::num_types]
                                                         [td_ns::num_types];

static accumulate_strided_impl_fn_ptr_t
    cumlogsumexp_include_initial_strided_dispatch_table[td_ns::num_types]
                                                       [td_ns::num_types];

template <typename argTy, typename outTy>
struct TypePairSupportDataForLogSumExpAccumulation
{
    static constexpr bool is_defined = std::disjunction<
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, sycl::half>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, double>,

        // input int8_t
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, sycl::half>,
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, double>,

        // input uint8_t
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, sycl::half>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, double>,

        // input int16_t
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, double>,

        // input uint16_t
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, double>,

        // input int32_t
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, double>,

        // input uint32_t
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, double>,

        // input int64_t
        td_ns::TypePairDefinedEntry<argTy, std::int64_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::int64_t, outTy, double>,

        // input uint64_t
        td_ns::TypePairDefinedEntry<argTy, std::uint64_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::uint64_t, outTy, double>,

        // input half
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, sycl::half>,
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, double>,

        // input float
        td_ns::TypePairDefinedEntry<argTy, float, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, float, outTy, double>,

        // input double
        td_ns::TypePairDefinedEntry<argTy, double, outTy, double>,

        // fall-through
        td_ns::NotDefinedEntry>::is_defined;
};

template <typename fnT, typename srcTy, typename dstTy>
struct CumLogSumExp1DContigFactory
{
    fnT get()
    {
        if constexpr (TypePairSupportDataForLogSumExpAccumulation<
                          srcTy, dstTy>::is_defined) {
            using ScanOpT = su_ns::LogSumExp<dstTy>;
            static constexpr bool include_initial = false;
            if constexpr (std::is_same_v<srcTy, dstTy>) {
                using dpctl::tensor::kernels::accumulators::NoOpTransformer;
                fnT fn = dpctl::tensor::kernels::accumulators::
                    accumulate_1d_contig_impl<srcTy, dstTy,
                                              NoOpTransformer<dstTy>, ScanOpT,
                                              include_initial>;
                return fn;
            }
            else {
                using dpctl::tensor::kernels::accumulators::CastTransformer;
                fnT fn = dpctl::tensor::kernels::accumulators::
                    accumulate_1d_contig_impl<srcTy, dstTy,
                                              CastTransformer<srcTy, dstTy>,
                                              ScanOpT, include_initial>;
                return fn;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct CumLogSumExp1DIncludeInitialContigFactory
{
    fnT get()
    {
        if constexpr (TypePairSupportDataForLogSumExpAccumulation<
                          srcTy, dstTy>::is_defined) {
            using ScanOpT = su_ns::LogSumExp<dstTy>;
            static constexpr bool include_initial = true;
            if constexpr (std::is_same_v<srcTy, dstTy>) {
                using dpctl::tensor::kernels::accumulators::NoOpTransformer;
                fnT fn = dpctl::tensor::kernels::accumulators::
                    accumulate_1d_contig_impl<srcTy, dstTy,
                                              NoOpTransformer<dstTy>, ScanOpT,
                                              include_initial>;
                return fn;
            }
            else {
                using dpctl::tensor::kernels::accumulators::CastTransformer;
                fnT fn = dpctl::tensor::kernels::accumulators::
                    accumulate_1d_contig_impl<srcTy, dstTy,
                                              CastTransformer<srcTy, dstTy>,
                                              ScanOpT, include_initial>;
                return fn;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct CumLogSumExpStridedFactory
{
    fnT get()
    {
        if constexpr (TypePairSupportDataForLogSumExpAccumulation<
                          srcTy, dstTy>::is_defined) {
            using ScanOpT = su_ns::LogSumExp<dstTy>;
            static constexpr bool include_initial = false;
            if constexpr (std::is_same_v<srcTy, dstTy>) {
                using dpctl::tensor::kernels::accumulators::NoOpTransformer;
                fnT fn = dpctl::tensor::kernels::accumulators::
                    accumulate_strided_impl<srcTy, dstTy,
                                            NoOpTransformer<dstTy>, ScanOpT,
                                            include_initial>;
                return fn;
            }
            else {
                using dpctl::tensor::kernels::accumulators::CastTransformer;
                fnT fn = dpctl::tensor::kernels::accumulators::
                    accumulate_strided_impl<srcTy, dstTy,
                                            CastTransformer<srcTy, dstTy>,
                                            ScanOpT, include_initial>;
                return fn;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct CumLogSumExpIncludeInitialStridedFactory
{
    fnT get()
    {
        if constexpr (TypePairSupportDataForLogSumExpAccumulation<
                          srcTy, dstTy>::is_defined) {
            using ScanOpT = su_ns::LogSumExp<dstTy>;
            static constexpr bool include_initial = true;
            if constexpr (std::is_same_v<srcTy, dstTy>) {
                using dpctl::tensor::kernels::accumulators::NoOpTransformer;
                fnT fn = dpctl::tensor::kernels::accumulators::
                    accumulate_strided_impl<srcTy, dstTy,
                                            NoOpTransformer<dstTy>, ScanOpT,
                                            include_initial>;
                return fn;
            }
            else {
                using dpctl::tensor::kernels::accumulators::CastTransformer;
                fnT fn = dpctl::tensor::kernels::accumulators::
                    accumulate_strided_impl<srcTy, dstTy,
                                            CastTransformer<srcTy, dstTy>,
                                            ScanOpT, include_initial>;
                return fn;
            }
        }
        else {
            return nullptr;
        }
    }
};

void populate_cumlogsumexp_dispatch_tables(void)
{
    td_ns::DispatchTableBuilder<accumulate_1d_contig_impl_fn_ptr_t,
                                CumLogSumExp1DContigFactory, td_ns::num_types>
        dtb1;
    dtb1.populate_dispatch_table(cumlogsumexp_1d_contig_dispatch_table);

    td_ns::DispatchTableBuilder<accumulate_strided_impl_fn_ptr_t,
                                CumLogSumExpStridedFactory, td_ns::num_types>
        dtb2;
    dtb2.populate_dispatch_table(cumlogsumexp_strided_dispatch_table);

    td_ns::DispatchTableBuilder<accumulate_1d_contig_impl_fn_ptr_t,
                                CumLogSumExp1DIncludeInitialContigFactory,
                                td_ns::num_types>
        dtb3;
    dtb3.populate_dispatch_table(
        cumlogsumexp_1d_include_initial_contig_dispatch_table);

    td_ns::DispatchTableBuilder<accumulate_strided_impl_fn_ptr_t,
                                CumLogSumExpIncludeInitialStridedFactory,
                                td_ns::num_types>
        dtb4;
    dtb4.populate_dispatch_table(
        cumlogsumexp_include_initial_strided_dispatch_table);

    return;
}

} // namespace impl

void init_cumulative_logsumexp(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;

    using impl::populate_cumlogsumexp_dispatch_tables;
    populate_cumlogsumexp_dispatch_tables();

    using impl::cumlogsumexp_1d_contig_dispatch_table;
    using impl::cumlogsumexp_strided_dispatch_table;
    auto cumlogsumexp_pyapi = [&](const arrayT &src,
                                  int trailing_dims_to_accumulate,
                                  const arrayT &dst, sycl::queue &exec_q,
                                  const event_vecT &depends = {}) {
        return py_accumulate_over_axis(src, trailing_dims_to_accumulate, dst,
                                       exec_q, depends,
                                       cumlogsumexp_strided_dispatch_table,
                                       cumlogsumexp_1d_contig_dispatch_table);
    };
    m.def("_cumlogsumexp_over_axis", cumlogsumexp_pyapi, "", py::arg("src"),
          py::arg("trailing_dims_to_accumulate"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    using impl::cumlogsumexp_1d_include_initial_contig_dispatch_table;
    using impl::cumlogsumexp_include_initial_strided_dispatch_table;
    auto cumlogsumexp_include_initial_pyapi =
        [&](const arrayT &src, const arrayT &dst, sycl::queue &exec_q,
            const event_vecT &depends = {}) {
            return py_accumulate_final_axis_include_initial(
                src, dst, exec_q, depends,
                cumlogsumexp_include_initial_strided_dispatch_table,
                cumlogsumexp_1d_include_initial_contig_dispatch_table);
        };
    m.def("_cumlogsumexp_final_axis_include_initial",
          cumlogsumexp_include_initial_pyapi, "", py::arg("src"),
          py::arg("dst"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());

    auto cumlogsumexp_dtype_supported = [&](const py::dtype &input_dtype,
                                            const py::dtype &output_dtype) {
        return py_accumulate_dtype_supported(
            input_dtype, output_dtype, cumlogsumexp_strided_dispatch_table);
    };
    m.def("_cumlogsumexp_dtype_supported", cumlogsumexp_dtype_supported, "",
          py::arg("arg_dtype"), py::arg("out_dtype"));
}

} // namespace dpctl::tensor::py_internal
