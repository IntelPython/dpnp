//*****************************************************************************
// Copyright (c) 2024, Intel Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
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

#pragma once

#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>

#include <dpctl4pybind11.hpp>

// dpctl tensor headers
#include "utils/type_dispatch.hpp"

// dpctl namespace for operations with types
namespace dpctl_td_ns = dpctl::tensor::type_dispatch;

namespace dpnp::extensions::blas
{
typedef sycl::event (*nrm2_impl_fn_ptr_t)(sycl::queue &,
                                          const std::int64_t,
                                          const char *,
                                          const std::int64_t,
                                          char *,
                                          const std::vector<sycl::event> &);

extern nrm2_impl_fn_ptr_t nrm2_dispatch_table[dpctl_td_ns::num_types]
                                             [dpctl_td_ns::num_types];

extern std::pair<sycl::event, sycl::event>
    nrm2(sycl::queue &exec_q,
         const dpctl::tensor::usm_ndarray &VectorX,
         const dpctl::tensor::usm_ndarray &result,
         const std::vector<sycl::event> &depends);

extern std::pair<sycl::event, std::vector<sycl::event>>
    nrm2_batch(sycl::queue &exec_q,
               const dpctl::tensor::usm_ndarray &arrayX,
               const dpctl::tensor::usm_ndarray &result,
               const std::vector<sycl::event> &depends);

extern void init_nrm2_dispatch_table(void);
} // namespace dpnp::extensions::blas
