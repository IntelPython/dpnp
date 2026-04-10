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

#include "atan.hpp"
#include "elementwise_functions.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_dispatch_building.hpp"

#include "kernels/elementwise_functions/atan.hpp"
#include "kernels/elementwise_functions/common.hpp"

namespace dpctl::tensor::py_internal
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

namespace ew_cmn_ns = dpctl::tensor::kernels::elementwise_common;
using ew_cmn_ns::unary_contig_impl_fn_ptr_t;
using ew_cmn_ns::unary_strided_impl_fn_ptr_t;

// U06: ==== ATAN   (x)
namespace impl
{

namespace atan_fn_ns = dpctl::tensor::kernels::atan;

static unary_contig_impl_fn_ptr_t atan_contig_dispatch_vector[td_ns::num_types];
static int atan_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    atan_strided_dispatch_vector[td_ns::num_types];

void populate_atan_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = atan_fn_ns;

    using fn_ns::AtanContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, AtanContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(atan_contig_dispatch_vector);

    using fn_ns::AtanStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, AtanStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(atan_strided_dispatch_vector);

    using fn_ns::AtanTypeMapFactory;
    DispatchVectorBuilder<int, AtanTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(atan_output_typeid_vector);
};

} // namespace impl

void init_atan(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;
    {
        impl::populate_atan_dispatch_vectors();
        using impl::atan_contig_dispatch_vector;
        using impl::atan_output_typeid_vector;
        using impl::atan_strided_dispatch_vector;

        auto atan_pyapi = [&](const arrayT &src, const arrayT &dst,
                              sycl::queue &exec_q,
                              const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, atan_output_typeid_vector,
                atan_contig_dispatch_vector, atan_strided_dispatch_vector);
        };
        m.def("_atan", atan_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto atan_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype, atan_output_typeid_vector);
        };
        m.def("_atan_result_type", atan_result_type_pyapi);
    }
}

} // namespace dpctl::tensor::py_internal
