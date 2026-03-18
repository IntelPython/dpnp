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
/// This file defines functions of dpctl.tensor._tensor_elementwise_impl
/// extension, specifically functions for elementwise operations.
//===---------------------------------------------------------------------===//

#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "elementwise_functions.hpp"
#include "floor_divide.hpp"
#include "utils/type_dispatch.hpp"

#include "kernels/elementwise_functions/common.hpp"
#include "kernels/elementwise_functions/common_inplace.hpp"
#include "kernels/elementwise_functions/floor_divide.hpp"

namespace dpctl::tensor::py_internal
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

namespace ew_cmn_ns = dpctl::tensor::kernels::elementwise_common;
using ew_cmn_ns::binary_contig_impl_fn_ptr_t;
using ew_cmn_ns::binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t;
using ew_cmn_ns::binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t;
using ew_cmn_ns::binary_strided_impl_fn_ptr_t;

using ew_cmn_ns::binary_inplace_contig_impl_fn_ptr_t;
using ew_cmn_ns::binary_inplace_row_matrix_broadcast_impl_fn_ptr_t;
using ew_cmn_ns::binary_inplace_strided_impl_fn_ptr_t;

// B10: ===== FLOOR_DIVIDE (x1, x2)
namespace impl
{
namespace floor_divide_fn_ns = dpctl::tensor::kernels::floor_divide;

static binary_contig_impl_fn_ptr_t
    floor_divide_contig_dispatch_table[td_ns::num_types][td_ns::num_types];

static int floor_divide_output_id_table[td_ns::num_types][td_ns::num_types];
static int floor_divide_inplace_output_id_table[td_ns::num_types]
                                               [td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    floor_divide_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

static binary_inplace_contig_impl_fn_ptr_t
    floor_divide_inplace_contig_dispatch_table[td_ns::num_types]
                                              [td_ns::num_types];
static binary_inplace_strided_impl_fn_ptr_t
    floor_divide_inplace_strided_dispatch_table[td_ns::num_types]
                                               [td_ns::num_types];

void populate_floor_divide_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = floor_divide_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::FloorDivideTypeMapFactory;
    DispatchTableBuilder<int, FloorDivideTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(floor_divide_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::FloorDivideStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t,
                         FloorDivideStridedFactory, num_types>
        dtb2;
    dtb2.populate_dispatch_table(floor_divide_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::FloorDivideContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, FloorDivideContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(floor_divide_contig_dispatch_table);

    // function pointers for inplace operation on general strided arrays
    using fn_ns::FloorDivideInplaceStridedFactory;
    DispatchTableBuilder<binary_inplace_strided_impl_fn_ptr_t,
                         FloorDivideInplaceStridedFactory, num_types>
        dtb4;
    dtb4.populate_dispatch_table(floor_divide_inplace_strided_dispatch_table);

    // function pointers for inplace operation on contiguous inputs and output
    using fn_ns::FloorDivideInplaceContigFactory;
    DispatchTableBuilder<binary_inplace_contig_impl_fn_ptr_t,
                         FloorDivideInplaceContigFactory, num_types>
        dtb5;
    dtb5.populate_dispatch_table(floor_divide_inplace_contig_dispatch_table);

    // which types are supported by the in-place kernels
    using fn_ns::FloorDivideInplaceTypeMapFactory;
    DispatchTableBuilder<int, FloorDivideInplaceTypeMapFactory, num_types> dtb6;
    dtb6.populate_dispatch_table(floor_divide_inplace_output_id_table);
};

} // namespace impl

void init_floor_divide(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;
    {
        impl::populate_floor_divide_dispatch_tables();
        using impl::floor_divide_contig_dispatch_table;
        using impl::floor_divide_output_id_table;
        using impl::floor_divide_strided_dispatch_table;

        auto floor_divide_pyapi = [&](const arrayT &src1, const arrayT &src2,
                                      const arrayT &dst, sycl::queue &exec_q,
                                      const event_vecT &depends = {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, floor_divide_output_id_table,
                // function pointers to handle operation on contiguous arrays
                // (pointers may be nullptr)
                floor_divide_contig_dispatch_table,
                // function pointers to handle operation on strided arrays (most
                // general case)
                floor_divide_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        auto floor_divide_result_type_pyapi = [&](const py::dtype &dtype1,
                                                  const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               floor_divide_output_id_table);
        };
        m.def("_floor_divide", floor_divide_pyapi, "", py::arg("src1"),
              py::arg("src2"), py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_floor_divide_result_type", floor_divide_result_type_pyapi, "");

        using impl::floor_divide_inplace_contig_dispatch_table;
        using impl::floor_divide_inplace_output_id_table;
        using impl::floor_divide_inplace_strided_dispatch_table;

        auto floor_divide_inplace_pyapi = [&](const arrayT &src,
                                              const arrayT &dst,
                                              sycl::queue &exec_q,
                                              const event_vecT &depends = {}) {
            return py_binary_inplace_ufunc(
                src, dst, exec_q, depends, floor_divide_inplace_output_id_table,
                // function pointers to handle inplace operation on
                // contiguous arrays (pointers may be nullptr)
                floor_divide_inplace_contig_dispatch_table,
                // function pointers to handle inplace operation on strided
                // arrays (most general case)
                floor_divide_inplace_strided_dispatch_table,
                // function pointers to handle inplace operation on
                // c-contig matrix with c-contig row with broadcasting
                // (may be nullptr)
                td_ns::NullPtrTable<
                    binary_inplace_row_matrix_broadcast_impl_fn_ptr_t>{});
        };
        m.def("_floor_divide_inplace", floor_divide_inplace_pyapi, "",
              py::arg("lhs"), py::arg("rhs"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
    }
}

} // namespace dpctl::tensor::py_internal
