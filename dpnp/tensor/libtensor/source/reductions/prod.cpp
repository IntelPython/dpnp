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
/// This file defines functions of dpctl.tensor._tensor_reductions_impl
/// extension.
//===---------------------------------------------------------------------===//

#include <complex>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kernels/reductions.hpp"
#include "utils/type_dispatch_building.hpp"

#include "reduction_atomic_support.hpp"
#include "reduction_over_axis.hpp"

namespace dpctl::tensor::py_internal
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

namespace impl
{

using dpctl::tensor::kernels::reduction_strided_impl_fn_ptr;
static reduction_strided_impl_fn_ptr
    prod_over_axis_strided_atomic_dispatch_table[td_ns::num_types]
                                                [td_ns::num_types];
static reduction_strided_impl_fn_ptr
    prod_over_axis_strided_temps_dispatch_table[td_ns::num_types]
                                               [td_ns::num_types];

using dpctl::tensor::kernels::reduction_contig_impl_fn_ptr;
static reduction_contig_impl_fn_ptr
    prod_over_axis1_contig_atomic_dispatch_table[td_ns::num_types]
                                                [td_ns::num_types];
static reduction_contig_impl_fn_ptr
    prod_over_axis0_contig_atomic_dispatch_table[td_ns::num_types]
                                                [td_ns::num_types];
static reduction_contig_impl_fn_ptr
    prod_over_axis1_contig_temps_dispatch_table[td_ns::num_types]
                                               [td_ns::num_types];
static reduction_contig_impl_fn_ptr
    prod_over_axis0_contig_temps_dispatch_table[td_ns::num_types]
                                               [td_ns::num_types];

/* @brief Types supported by plus-reduction code based on atomic_ref */
template <typename argTy, typename outTy>
struct TypePairSupportDataForProductReductionAtomic
{

    /* value if true a kernel for <argTy, outTy> must be instantiated, false
     * otherwise */
    static constexpr bool is_defined = std::disjunction<
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::uint64_t>,
        // input int8
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int64_t>,
        // input uint8
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::uint64_t>,
        // input int16
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, std::int64_t>,
        // input uint16
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::uint64_t>,
        // input int32
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, std::int64_t>,
        // input uint32
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, std::uint64_t>,
        // input int64
        td_ns::TypePairDefinedEntry<argTy, std::int64_t, outTy, std::int64_t>,
        // input uint64
        td_ns::TypePairDefinedEntry<argTy, std::uint64_t, outTy, std::uint64_t>,
        // fall-through
        td_ns::NotDefinedEntry>::is_defined;
};

template <typename argTy, typename outTy>
struct TypePairSupportDataForProductReductionTemps
{

    static constexpr bool is_defined = std::disjunction<
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, bool>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::int8_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::uint8_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::int16_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::uint16_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, std::uint64_t>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, bool, outTy, double>,

        // input int8_t
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int8_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int16_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, outTy, double>,

        // input uint8_t
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::uint8_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::int16_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::uint16_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, std::uint64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, outTy, double>,

        // input int16_t
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, std::int16_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, outTy, double>,

        // input uint16_t
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::uint16_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, std::uint64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, outTy, double>,

        // input int32_t
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, outTy, double>,

        // input uint32_t
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, std::uint64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, outTy, double>,

        // input int64_t
        td_ns::TypePairDefinedEntry<argTy, std::int64_t, outTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int64_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::int64_t, outTy, double>,

        // input uint32_t
        td_ns::TypePairDefinedEntry<argTy, std::uint64_t, outTy, std::uint64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint64_t, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, std::uint64_t, outTy, double>,

        // input half
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, sycl::half>,
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, sycl::half, outTy, double>,
        td_ns::
            TypePairDefinedEntry<argTy, sycl::half, outTy, std::complex<float>>,
        td_ns::TypePairDefinedEntry<argTy,
                                    sycl::half,
                                    outTy,
                                    std::complex<double>>,

        // input float
        td_ns::TypePairDefinedEntry<argTy, float, outTy, float>,
        td_ns::TypePairDefinedEntry<argTy, float, outTy, double>,
        td_ns::TypePairDefinedEntry<argTy, float, outTy, std::complex<float>>,
        td_ns::TypePairDefinedEntry<argTy, float, outTy, std::complex<double>>,

        // input double
        td_ns::TypePairDefinedEntry<argTy, double, outTy, double>,
        td_ns::TypePairDefinedEntry<argTy, double, outTy, std::complex<double>>,

        // input std::complex
        td_ns::TypePairDefinedEntry<argTy,
                                    std::complex<float>,
                                    outTy,
                                    std::complex<float>>,
        td_ns::TypePairDefinedEntry<argTy,
                                    std::complex<float>,
                                    outTy,
                                    std::complex<double>>,

        td_ns::TypePairDefinedEntry<argTy,
                                    std::complex<double>,
                                    outTy,
                                    std::complex<double>>,

        // fall-through
        td_ns::NotDefinedEntry>::is_defined;
};

template <typename fnT, typename srcTy, typename dstTy>
struct ProductOverAxisAtomicStridedFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForProductReductionAtomic<
                          srcTy, dstTy>::is_defined) {
            using ReductionOpT = sycl::multiplies<dstTy>;
            return dpctl::tensor::kernels::
                reduction_over_group_with_atomics_strided_impl<srcTy, dstTy,
                                                               ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct ProductOverAxisTempsStridedFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForProductReductionTemps<
                          srcTy, dstTy>::is_defined) {
            using ReductionOpT = std::conditional_t<std::is_same_v<dstTy, bool>,
                                                    sycl::logical_and<dstTy>,
                                                    sycl::multiplies<dstTy>>;
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
struct ProductOverAxis1AtomicContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForProductReductionAtomic<
                          srcTy, dstTy>::is_defined) {
            using ReductionOpT = sycl::multiplies<dstTy>;
            return dpctl::tensor::kernels::
                reduction_axis1_over_group_with_atomics_contig_impl<
                    srcTy, dstTy, ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct ProductOverAxis0AtomicContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForProductReductionAtomic<
                          srcTy, dstTy>::is_defined) {
            using ReductionOpT = sycl::multiplies<dstTy>;
            return dpctl::tensor::kernels::
                reduction_axis0_over_group_with_atomics_contig_impl<
                    srcTy, dstTy, ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename srcTy, typename dstTy>
struct ProductOverAxis1TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForProductReductionTemps<
                          srcTy, dstTy>::is_defined) {
            using ReductionOpT = std::conditional_t<std::is_same_v<dstTy, bool>,
                                                    sycl::logical_and<dstTy>,
                                                    sycl::multiplies<dstTy>>;
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
struct ProductOverAxis0TempsContigFactory
{
    fnT get() const
    {
        if constexpr (TypePairSupportDataForProductReductionTemps<
                          srcTy, dstTy>::is_defined) {
            using ReductionOpT = std::conditional_t<std::is_same_v<dstTy, bool>,
                                                    sycl::logical_and<dstTy>,
                                                    sycl::multiplies<dstTy>>;
            return dpctl::tensor::kernels::
                reduction_axis0_over_group_temps_contig_impl<srcTy, dstTy,
                                                             ReductionOpT>;
        }
        else {
            return nullptr;
        }
    }
};

void populate_prod_over_axis_dispatch_tables(void)
{
    using dpctl::tensor::kernels::reduction_contig_impl_fn_ptr;
    using dpctl::tensor::kernels::reduction_strided_impl_fn_ptr;
    using namespace td_ns;

    DispatchTableBuilder<reduction_strided_impl_fn_ptr,
                         ProductOverAxisAtomicStridedFactory, num_types>
        dtb1;
    dtb1.populate_dispatch_table(prod_over_axis_strided_atomic_dispatch_table);

    DispatchTableBuilder<reduction_strided_impl_fn_ptr,
                         ProductOverAxisTempsStridedFactory, num_types>
        dtb2;
    dtb2.populate_dispatch_table(prod_over_axis_strided_temps_dispatch_table);

    DispatchTableBuilder<reduction_contig_impl_fn_ptr,
                         ProductOverAxis1AtomicContigFactory, num_types>
        dtb3;
    dtb3.populate_dispatch_table(prod_over_axis1_contig_atomic_dispatch_table);

    DispatchTableBuilder<reduction_contig_impl_fn_ptr,
                         ProductOverAxis0AtomicContigFactory, num_types>
        dtb4;
    dtb4.populate_dispatch_table(prod_over_axis0_contig_atomic_dispatch_table);

    DispatchTableBuilder<reduction_contig_impl_fn_ptr,
                         ProductOverAxis1TempsContigFactory, td_ns::num_types>
        dtb5;
    dtb5.populate_dispatch_table(prod_over_axis1_contig_temps_dispatch_table);

    DispatchTableBuilder<reduction_contig_impl_fn_ptr,
                         ProductOverAxis0TempsContigFactory, td_ns::num_types>
        dtb6;
    dtb6.populate_dispatch_table(prod_over_axis0_contig_temps_dispatch_table);
}

using atomic_support::atomic_support_fn_ptr_t;
static atomic_support_fn_ptr_t prod_atomic_support_vector[td_ns::num_types];

void populate_prod_atomic_support_dispatch_vector(void)
{
    using td_ns::DispatchVectorBuilder;

    using atomic_support::ProductAtomicSupportFactory;
    DispatchVectorBuilder<atomic_support_fn_ptr_t, ProductAtomicSupportFactory,
                          td_ns::num_types>
        dvb;
    dvb.populate_dispatch_vector(prod_atomic_support_vector);
}

} // namespace impl

void init_prod(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;
    {
        using impl::populate_prod_over_axis_dispatch_tables;
        populate_prod_over_axis_dispatch_tables();
        using impl::prod_over_axis0_contig_atomic_dispatch_table;
        using impl::prod_over_axis0_contig_temps_dispatch_table;
        using impl::prod_over_axis1_contig_atomic_dispatch_table;
        using impl::prod_over_axis1_contig_temps_dispatch_table;
        using impl::prod_over_axis_strided_atomic_dispatch_table;
        using impl::prod_over_axis_strided_temps_dispatch_table;

        using impl::populate_prod_atomic_support_dispatch_vector;
        populate_prod_atomic_support_dispatch_vector();
        using impl::prod_atomic_support_vector;

        auto prod_pyapi = [&](const arrayT &src, int trailing_dims_to_reduce,
                              const arrayT &dst, sycl::queue &exec_q,
                              const event_vecT &depends = {}) {
            return py_reduction_over_axis(
                src, trailing_dims_to_reduce, dst, exec_q, depends,
                prod_over_axis_strided_atomic_dispatch_table,
                prod_over_axis0_contig_atomic_dispatch_table,
                prod_over_axis1_contig_atomic_dispatch_table,
                prod_over_axis_strided_temps_dispatch_table,
                prod_over_axis0_contig_temps_dispatch_table,
                prod_over_axis1_contig_temps_dispatch_table,
                prod_atomic_support_vector);
        };
        m.def("_prod_over_axis", prod_pyapi, "", py::arg("src"),
              py::arg("trailing_dims_to_reduce"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto prod_dtype_supported =
            [&](const py::dtype &input_dtype, const py::dtype &output_dtype,
                const std::string &dst_usm_type, sycl::queue &q) {
                return py_reduction_dtype_supported(
                    input_dtype, output_dtype, dst_usm_type, q,
                    prod_over_axis_strided_atomic_dispatch_table,
                    prod_over_axis_strided_temps_dispatch_table,
                    prod_atomic_support_vector);
            };
        m.def("_prod_over_axis_dtype_supported", prod_dtype_supported, "",
              py::arg("arg_dtype"), py::arg("out_dtype"),
              py::arg("dst_usm_type"), py::arg("sycl_queue"));
    }
}

} // namespace dpctl::tensor::py_internal
