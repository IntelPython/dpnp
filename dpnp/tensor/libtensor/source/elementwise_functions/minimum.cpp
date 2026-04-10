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
/// This file defines functions of dpnp.tensor._tensor_elementwise_impl
/// extension, specifically functions for elementwise operations.
//===---------------------------------------------------------------------===//

#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "elementwise_functions.hpp"
#include "minimum.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_dispatch_building.hpp"

#include "kernels/elementwise_functions/common.hpp"
#include "kernels/elementwise_functions/minimum.hpp"

namespace dpctl::tensor::py_internal
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

namespace ew_cmn_ns = dpctl::tensor::kernels::elementwise_common;
using ew_cmn_ns::binary_contig_impl_fn_ptr_t;
using ew_cmn_ns::binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t;
using ew_cmn_ns::binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t;
using ew_cmn_ns::binary_strided_impl_fn_ptr_t;

// B27: ===== MINIMUM (x1, x2)
namespace impl
{
namespace minimum_fn_ns = dpctl::tensor::kernels::minimum;

static binary_contig_impl_fn_ptr_t
    minimum_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static int minimum_output_id_table[td_ns::num_types][td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    minimum_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

void populate_minimum_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = minimum_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::MinimumTypeMapFactory;
    DispatchTableBuilder<int, MinimumTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(minimum_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::MinimumStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t, MinimumStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(minimum_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::MinimumContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, MinimumContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(minimum_contig_dispatch_table);
};

} // namespace impl

void init_minimum(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;
    {
        impl::populate_minimum_dispatch_tables();
        using impl::minimum_contig_dispatch_table;
        using impl::minimum_output_id_table;
        using impl::minimum_strided_dispatch_table;

        auto minimum_pyapi = [&](const arrayT &src1, const arrayT &src2,
                                 const arrayT &dst, sycl::queue &exec_q,
                                 const event_vecT &depends = {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, minimum_output_id_table,
                // function pointers to handle operation on contiguous arrays
                // (pointers may be nullptr)
                minimum_contig_dispatch_table,
                // function pointers to handle operation on strided arrays (most
                // general case)
                minimum_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                td_ns::NullPtrTable<
                    binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
        };
        auto minimum_result_type_pyapi = [&](const py::dtype &dtype1,
                                             const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               minimum_output_id_table);
        };
        m.def("_minimum", minimum_pyapi, "", py::arg("src1"), py::arg("src2"),
              py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_minimum_result_type", minimum_result_type_pyapi, "");
    }
}

} // namespace dpctl::tensor::py_internal
