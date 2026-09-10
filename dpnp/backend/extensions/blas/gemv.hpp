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

#pragma once

#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"

namespace dpnp::extensions::blas
{
// y = alpha * op(A) * x + beta * y. alpha/beta are real-valued (double)
// and are cast to the matrix value type in the impl.
//
// ``trans_op`` selects the operation applied to A:
//      0 = N  (no transpose)         y = alpha * A   @ x + beta * y
//      1 = T  (transpose)            y = alpha * A^T @ x + beta * y
//      2 = C  (conjugate-transpose)  y = alpha * A^H @ x + beta * y
//
// For real-valued A, T and C are equivalent. For complex A they
// differ, and C is required for any algorithm that performs a
// Hermitian inner product through gemv -- the GMRES Arnoldi step
// (Gram-Schmidt over a complex Krylov basis) being the canonical
// example. ``trans_op = 2`` is currently only supported for
// F-contiguous (column-major) matrices; the row-major code path
// for conjugate-transpose would require an explicit element-wise
// conjugate pass and is not wired up here.
extern std::pair<sycl::event, sycl::event>
    gemv(sycl::queue &exec_q,
         const dpnp::tensor::usm_ndarray &matrixA,
         const dpnp::tensor::usm_ndarray &vectorX,
         const dpnp::tensor::usm_ndarray &vectorY,
         const int trans_op,
         const double alpha,
         const double beta,
         const std::vector<sycl::event> &depends);

extern void init_gemv_dispatch_vector(void);
} // namespace dpnp::extensions::blas
