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
///
/// \file
/// This file defines common code for elementwise tensor operations.
//===---------------------------------------------------------------------===//

#pragma once

#include <cstddef>
#include <vector>

#include <sycl/sycl.hpp>

namespace dpctl::tensor::kernels::elementwise_detail
{
template <typename T>
class populate_padded_vec_krn;

template <typename T>
sycl::event
    populate_padded_vector(sycl::queue &exec_q,
                           const T *vec,
                           std::size_t vec_sz,
                           T *padded_vec,
                           size_t padded_vec_sz,
                           const std::vector<sycl::event> &dependent_events)
{
    sycl::event populate_padded_vec_ev = exec_q.submit([&](sycl::handler &cgh) {
        // ensure vec contains actual data
        cgh.depends_on(dependent_events);

        sycl::range<1> gRange{padded_vec_sz};

        cgh.parallel_for<class populate_padded_vec_krn<T>>(
            gRange, [=](sycl::id<1> id)
        {
            std::size_t i = id[0];
            padded_vec[i] = vec[i % vec_sz];
            });
    });

    return populate_padded_vec_ev;
}
} // namespace dpctl::tensor::kernels::elementwise_detail
