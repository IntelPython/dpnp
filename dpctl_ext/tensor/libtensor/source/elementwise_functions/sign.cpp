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
#include "sign.hpp"
#include "utils/type_dispatch.hpp"

#include "kernels/elementwise_functions/common.hpp"
#include "kernels/elementwise_functions/sign.hpp"

namespace dpctl::tensor::py_internal
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

namespace ew_cmn_ns = dpctl::tensor::kernels::elementwise_common;
using ew_cmn_ns::unary_contig_impl_fn_ptr_t;
using ew_cmn_ns::unary_strided_impl_fn_ptr_t;

// U29: ==== SIGN   (x)
namespace impl
{

namespace sign_fn_ns = dpctl::tensor::kernels::sign;

static unary_contig_impl_fn_ptr_t sign_contig_dispatch_vector[td_ns::num_types];
static int sign_output_typeid_vector[td_ns::num_types];
static unary_strided_impl_fn_ptr_t
    sign_strided_dispatch_vector[td_ns::num_types];

void populate_sign_dispatch_vectors(void)
{
    using namespace td_ns;
    namespace fn_ns = sign_fn_ns;

    using fn_ns::SignContigFactory;
    DispatchVectorBuilder<unary_contig_impl_fn_ptr_t, SignContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(sign_contig_dispatch_vector);

    using fn_ns::SignStridedFactory;
    DispatchVectorBuilder<unary_strided_impl_fn_ptr_t, SignStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(sign_strided_dispatch_vector);

    using fn_ns::SignTypeMapFactory;
    DispatchVectorBuilder<int, SignTypeMapFactory, num_types> dvb3;
    dvb3.populate_dispatch_vector(sign_output_typeid_vector);
};

} // namespace impl

void init_sign(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;
    {
        impl::populate_sign_dispatch_vectors();
        using impl::sign_contig_dispatch_vector;
        using impl::sign_output_typeid_vector;
        using impl::sign_strided_dispatch_vector;

        auto sign_pyapi = [&](const arrayT &src, const arrayT &dst,
                              sycl::queue &exec_q,
                              const event_vecT &depends = {}) {
            return py_unary_ufunc(
                src, dst, exec_q, depends, sign_output_typeid_vector,
                sign_contig_dispatch_vector, sign_strided_dispatch_vector);
        };
        m.def("_sign", sign_pyapi, "", py::arg("src"), py::arg("dst"),
              py::arg("sycl_queue"), py::arg("depends") = py::list());

        auto sign_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_unary_ufunc_result_type(dtype, sign_output_typeid_vector);
        };
        m.def("_sign_result_type", sign_result_type_pyapi);
    }
}

} // namespace dpctl::tensor::py_internal
