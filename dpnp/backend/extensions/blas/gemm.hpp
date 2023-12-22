//*****************************************************************************
// Copyright (c) 2023, Intel Corporation
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

namespace dpnp
{
namespace backend
{
namespace ext
{
namespace blas
{
extern std::pair<sycl::event, sycl::event>
    gemm(sycl::queue exec_q,
         dpctl::tensor::usm_ndarray matrixA,
         dpctl::tensor::usm_ndarray matrixB,
         dpctl::tensor::usm_ndarray resultC,
         const std::vector<sycl::event> &depends);

extern std::pair<sycl::event, sycl::event>
    gemm_batch(sycl::queue exec_q,
               dpctl::tensor::usm_ndarray matrixA,
               dpctl::tensor::usm_ndarray matrixB,
               dpctl::tensor::usm_ndarray resultC,
               const std::int64_t batch_size,
               size_t stridea,
               size_t strideb,
               size_t stridec,
               const std::vector<sycl::event> &depends);

extern void init_gemm_dispatch_table(void);
extern void init_gemm_batch_dispatch_table(void);
} // namespace blas
} // namespace ext
} // namespace backend
} // namespace dpnp
