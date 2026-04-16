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

namespace dpnp::extensions::lapack
{
extern std::pair<sycl::event, sycl::event>
    gesv(sycl::queue &exec_q,
         const dpctl::tensor::usm_ndarray &coeff_matrix,
         const dpctl::tensor::usm_ndarray &dependent_vals,
         const std::vector<sycl::event> &depends);

extern std::pair<sycl::event, sycl::event>
    gesv_batch(sycl::queue &exec_q,
               const dpctl::tensor::usm_ndarray &coeff_matrix,
               const dpctl::tensor::usm_ndarray &dependent_vals,
               const std::vector<sycl::event> &depends);

extern void common_gesv_checks(sycl::queue &exec_q,
                               const dpctl::tensor::usm_ndarray &coeff_matrix,
                               const dpctl::tensor::usm_ndarray &dependent_vals,
                               const py::ssize_t *coeff_matrix_shape,
                               const py::ssize_t *dependent_vals_shape,
                               const int expected_coeff_matrix_ndim,
                               const int min_dependent_vals_ndim,
                               const int max_dependent_vals_ndim);

extern void init_gesv_dispatch_vector(void);
extern void init_gesv_batch_dispatch_vector(void);
} // namespace dpnp::extensions::lapack
