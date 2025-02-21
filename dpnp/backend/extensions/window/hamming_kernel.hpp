//*****************************************************************************
// Copyright (c) 2025, Intel Corporation
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

#include <sycl/sycl.hpp>

#include "kernels/dpctl_tensor_types.hpp"

#include "utils/type_utils.hpp"

namespace dpnp::extensions::window::kernels
{

template <typename T>
class HammingFunctor
{
private:
    T *data;
    size_t N;

public:
    HammingFunctor(T *data, size_t N) : data(data), N(N) {}

    void operator()(sycl::id<1> id) const
    {
        dpctl::tensor::ssize_t i = id[0];

        data[i] = static_cast<T>(0.54) -
                  static_cast<T>(0.46) *
                      sycl::cospi((static_cast<T>(2.0) * i) / (N - 1));
    }
};

typedef sycl::event (*hamming_fn_ptr_t)(sycl::queue &,
                                        char *,
                                        size_t,
                                        const std::vector<sycl::event> &);

template <typename T>
sycl::event hamming_impl(sycl::queue &q,
                         char *result,
                         size_t nelems,
                         const std::vector<sycl::event> &depends)
{
    dpctl::tensor::type_utils::validate_type_for_device<T>(q);

    T *res = reinterpret_cast<T *>(result);

    sycl::event hamming_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        cgh.parallel_for(sycl::range<1>(nelems),
                         HammingFunctor<T>(res, nelems));
    });

    return hamming_ev;
}

} // namespace dpnp::extensions::window::kernels
