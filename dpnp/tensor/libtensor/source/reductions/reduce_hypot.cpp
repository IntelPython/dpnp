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

#include <cstdint>
#include <type_traits>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kernels/reductions.hpp"
#include "utils/sycl_utils.hpp"
#include "utils/type_dispatch_building.hpp"

#include "reduction_over_axis.hpp"

namespace dpctl::tensor::py_internal
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;
namespace su_ns = dpctl::tensor::sycl_utils;

namespace impl
{

using dpctl::tensor::kernels::reduction_strided_impl_fn_ptr;
static reduction_strided_impl_fn_ptr
    hypot_over_axis_strided_temps_dispatch_table[td_ns::num_types]
                                                [td_ns::num_types];

using dpctl::tensor::kernels::reduction_contig_impl_fn_ptr;
static reduction_contig_impl_fn_ptr
    hypot_over_axis1_contig_temps_dispatch_table[td_ns::num_types]
                                                [td_ns::num_types];
static reduction_contig_impl_fn_ptr
    hypot_over_axis0_contig_temps_dispatch_table[td_ns::num_types]
                                                [td_ns::num_types];

template <typename argTy, typename outTy>
struct TypePairSupportDataForHypotReductionTemps
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
struct HypotOverAxisTempsStridedFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForHypotReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            using ReductionOpT = su_ns::Hypot<dstTy>;
            return dpctl::tensor::kernels::
                reduction_over_group_temps_strided_impl<srcTy, dstTy,
                                                        ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct HypotOverAxis1TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForHypotReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            using ReductionOpT = su_ns::Hypot<dstTy>;
            return dpctl::tensor::kernels::
                reduction_axis1_over_group_temps_contig_impl<srcTy, dstTy,
                                                             ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct HypotOverAxis0TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForHypotReductionTemps<
                          srcTy, dstTy>::is_defined)
        {
            using ReductionOpT = su_ns::Hypot<dstTy>;
            return dpctl::tensor::kernels::
                reduction_axis0_over_group_temps_contig_impl<srcTy, dstTy,
                                                             ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

void populate_hypot_over_axis_dispatch_tables(void)
{
    using dpctl::tensor::kernels::reduction_contig_impl_fn_ptr;
    using dpctl::tensor::kernels::reduction_strided_impl_fn_ptr;
    using namespace td_ns;

    DispatchTableBuilder<reduction_strided_impl_fn_ptr,
                         HypotOverAxisTempsStridedFactory, num_types>
        dtb1;
    dtb1.populate_dispatch_table(hypot_over_axis_strided_temps_dispatch_table);

    DispatchTableBuilder<reduction_contig_impl_fn_ptr,
                         HypotOverAxis1TempsContigFactory, td_ns::num_types>
        dtb2;
    dtb2.populate_dispatch_table(hypot_over_axis1_contig_temps_dispatch_table);

    DispatchTableBuilder<reduction_contig_impl_fn_ptr,
                         HypotOverAxis0TempsContigFactory, td_ns::num_types>
        dtb3;
    dtb3.populate_dispatch_table(hypot_over_axis0_contig_temps_dispatch_table);
}

} // namespace impl

void init_reduce_hypot(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;
    {
        using impl::populate_hypot_over_axis_dispatch_tables;
        populate_hypot_over_axis_dispatch_tables();
        using impl::hypot_over_axis0_contig_temps_dispatch_table;
        using impl::hypot_over_axis1_contig_temps_dispatch_table;
        using impl::hypot_over_axis_strided_temps_dispatch_table;

        using dpctl::tensor::kernels::reduction_contig_impl_fn_ptr;
        using dpctl::tensor::kernels::reduction_strided_impl_fn_ptr;

        auto hypot_pyapi = [&](const arrayT &src, int trailing_dims_to_reduce,
                               const arrayT &dst, sycl::queue &exec_q,
                               const event_vecT &depends = {}) {
            return py_tree_reduction_over_axis(
                src, trailing_dims_to_reduce, dst, exec_q, depends,
                hypot_over_axis_strided_temps_dispatch_table,
                hypot_over_axis0_contig_temps_dispatch_table,
                hypot_over_axis1_contig_temps_dispatch_table);
        };
        m.def("_hypot_over_axis", hypot_pyapi, "", py::arg("src"),
              py::arg("trailing_dims_to_reduce"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto hypot_dtype_supported = [&](const py::dtype &input_dtype,
                                         const py::dtype &output_dtype) {
            return py_tree_reduction_dtype_supported(
                input_dtype, output_dtype,
                hypot_over_axis_strided_temps_dispatch_table);
        };
        m.def("_hypot_over_axis_dtype_supported", hypot_dtype_supported, "",
              py::arg("arg_dtype"), py::arg("out_dtype"));
    }
}

} // namespace dpctl::tensor::py_internal
