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

#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>

#include <dpctl4pybind11.hpp>

namespace dpnp::extensions::lapack
{
extern std::pair<sycl::event, sycl::event>
    orgqr(sycl::queue &exec_q,
          const std::int64_t m,
          const std::int64_t n,
          const std::int64_t k,
          const dpctl::tensor::usm_ndarray &a_array,
          const dpctl::tensor::usm_ndarray &tau_array,
          const std::vector<sycl::event> &depends = {});

extern std::pair<sycl::event, sycl::event>
    orgqr_batch(sycl::queue &exec_q,
                const dpctl::tensor::usm_ndarray &a_array,
                const dpctl::tensor::usm_ndarray &tau_array,
                std::int64_t m,
                std::int64_t n,
                std::int64_t k,
                std::int64_t stride_a,
                std::int64_t stride_tau,
                std::int64_t batch_size,
                const std::vector<sycl::event> &depends = {});

extern void init_orgqr_batch_dispatch_vector(void);
extern void init_orgqr_dispatch_vector(void);
} // namespace dpnp::extensions::lapack
