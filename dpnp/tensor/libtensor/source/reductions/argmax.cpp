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
//===---------------------------------------------------------------------===//
///
/// \file
/// This file defines functions of dpnp.tensor._tensor_reductions_impl
/// extension.
//===---------------------------------------------------------------------===//

#include <complex>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kernels/reductions.hpp"
#include "reduction_over_axis.hpp"
#include "utils/sycl_utils.hpp"
#include "utils/type_dispatch_building.hpp"

namespace dpctl::tensor::py_internal
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;
namespace su_ns = dpctl::tensor::sycl_utils;

namespace impl
{

using dpctl::tensor::kernels::search_strided_impl_fn_ptr;
static search_strided_impl_fn_ptr
    argmax_over_axis_strided_temps_dispatch_table[td_ns::num_types]
                                                 [td_ns::num_types];

using dpctl::tensor::kernels::search_contig_impl_fn_ptr;
static search_contig_impl_fn_ptr
    argmax_over_axis1_contig_temps_dispatch_table[td_ns::num_types]
                                                 [td_ns::num_types];
using dpctl::tensor::kernels::search_contig_impl_fn_ptr;
static search_contig_impl_fn_ptr
    argmax_over_axis0_contig_temps_dispatch_table[td_ns::num_types]
                                                 [td_ns::num_types];

template <typename argTy, typename outTy>
struct TypePairSupportForArgmaxReductionTemps
{

    static constexpr bool is_defined = std::disjunction<
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::int64_t>,
        // input int8_t
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int64_t>,

        // input uint8_t
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::int64_t>,

        // input int16_t
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, std::int64_t>,

        // input uint16_t
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::int64_t>,

        // input int32_t
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, std::int64_t>,
        // input uint32_t
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, std::int64_t>,

        // input int64_t
        td_ns::TypePairDefinedEntry<argTy, std::int64_t, outTy, std::int64_t>,

        // input uint32_t
        td_ns::TypePairDefinedEntry<argTy, std::uint64_t, outTy, std::int64_t>,

        // input half
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, std::int64_t>,

        // input float
        td_ns::TypePairDefinedEntry<argTy, float, outTy, std::int64_t>,

        // input double
        td_ns::TypePairDefinedEntry<argTy, double, outTy, std::int64_t>,

        // input std::complex
        td_ns::TypePairDefinedEntry<argTy,
                                    std::complex<float>,
                                    outTy,
                                    std::int64_t>,

        td_ns::TypePairDefinedEntry<argTy,
                                    std::complex<double>,
                                    outTy,
                                    std::int64_t>,

        // fall-through
        td_ns::NotDefinedEntry>::is_defined;
};

template <typename fnT, typename srcTy, typename dstTy>
struct ArgmaxOverAxisTempsStridedFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportForArgmaxReductionTemps<srcTy,
                                                             dstTy>::is_defined)
        {
            if constexpr (std::is_integral_v<srcTy> &&
                          !std::is_same_v<srcTy, bool>)
            {
                // op for values
                using ReductionOpT = sycl::maximum<srcTy>;
                // op for indices
                using IndexOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    search_over_group_temps_strided_impl<
                        srcTy, dstTy, ReductionOpT, IndexOpT>;
            }
            else {
                // op for values
                using ReductionOpT = su_ns::Maximum<srcTy>;
                // op for indices
                using IndexOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    search_over_group_temps_strided_impl<
                        srcTy, dstTy, ReductionOpT, IndexOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct ArgmaxOverAxis1TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportForArgmaxReductionTemps<srcTy,
                                                             dstTy>::is_defined)
        {
            if constexpr (std::is_integral_v<srcTy> &&
                          !std::is_same_v<srcTy, bool>)
            {
                // op for values
                using ReductionOpT = sycl::maximum<srcTy>;
                // op for indices
                using IndexOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    search_axis1_over_group_temps_contig_impl<
                        srcTy, dstTy, ReductionOpT, IndexOpT>;
            }
            else {
                // op for values
                using ReductionOpT = su_ns::Maximum<srcTy>;
                // op for indices
                using IndexOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    search_axis1_over_group_temps_contig_impl<
                        srcTy, dstTy, ReductionOpT, IndexOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct ArgmaxOverAxis0TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportForArgmaxReductionTemps<srcTy,
                                                             dstTy>::is_defined)
        {
            if constexpr (std::is_integral_v<srcTy> &&
                          !std::is_same_v<srcTy, bool>)
            {
                // op for values
                using ReductionOpT = sycl::maximum<srcTy>;
                // op for indices
                using IndexOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    search_axis0_over_group_temps_contig_impl<
                        srcTy, dstTy, ReductionOpT, IndexOpT>;
            }
            else {
                // op for values
                using ReductionOpT = su_ns::Maximum<srcTy>;
                // op for indices
                using IndexOpT = sycl::minimum<dstTy>;
                return dpctl::tensor::kernels::
                    search_axis0_over_group_temps_contig_impl<
                        srcTy, dstTy, ReductionOpT, IndexOpT>;
            }
        }
        else {
            return nullptr;
        }
    }
};

void populate_argmax_over_axis_dispatch_tables(void)
{
    using td_ns::DispatchTableBuilder;

    DispatchTableBuilder<search_strided_impl_fn_ptr,
                         ArgmaxOverAxisTempsStridedFactory, td_ns::num_types>
        dtb1;
    dtb1.populate_dispatch_table(argmax_over_axis_strided_temps_dispatch_table);

    DispatchTableBuilder<search_contig_impl_fn_ptr,
                         ArgmaxOverAxis1TempsContigFactory, td_ns::num_types>
        dtb2;
    dtb2.populate_dispatch_table(argmax_over_axis1_contig_temps_dispatch_table);

    DispatchTableBuilder<search_contig_impl_fn_ptr,
                         ArgmaxOverAxis0TempsContigFactory, td_ns::num_types>
        dtb3;
    dtb3.populate_dispatch_table(argmax_over_axis0_contig_temps_dispatch_table);
}

} // namespace impl

void init_argmax(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;
    {
        using impl::populate_argmax_over_axis_dispatch_tables;
        populate_argmax_over_axis_dispatch_tables();
        using impl::argmax_over_axis0_contig_temps_dispatch_table;
        using impl::argmax_over_axis1_contig_temps_dispatch_table;
        using impl::argmax_over_axis_strided_temps_dispatch_table;

        auto argmax_pyapi = [&](const arrayT &src, int trailing_dims_to_reduce,
                                const arrayT &dst, sycl::queue &exec_q,
                                const event_vecT &depends = {}) {
            return py_search_over_axis(
                src, trailing_dims_to_reduce, dst, exec_q, depends,
                argmax_over_axis_strided_temps_dispatch_table,
                argmax_over_axis0_contig_temps_dispatch_table,
                argmax_over_axis1_contig_temps_dispatch_table);
        };
        m.def("_argmax_over_axis", argmax_pyapi, "", py::arg("src"),
              py::arg("trailing_dims_to_reduce"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());
    }
}

} // namespace dpctl::tensor::py_internal
