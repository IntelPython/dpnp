//*****************************************************************************
// Copyright (c) 2024-2025, Intel Corporation
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

#include <stdexcept>

#include "dot_common.hpp"

namespace dpnp::extensions::blas
{
namespace mkl_blas = oneapi::mkl::blas;
namespace type_utils = dpctl::tensor::type_utils;

template <typename T>
static sycl::event dot_impl(sycl::queue &exec_q,
                            const std::int64_t n,
                            const char *vectorX,
                            const std::int64_t incx,
                            const char *vectorY,
                            const std::int64_t incy,
                            char *result,
                            const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<T>(exec_q);

    const T *x = reinterpret_cast<const T *>(vectorX);
    const T *y = reinterpret_cast<const T *>(vectorY);
    T *res = reinterpret_cast<T *>(result);

    std::stringstream error_msg;
    bool is_exception_caught = false;

    sycl::event dot_event;
    try {
        dot_event = mkl_blas::column_major::dot(exec_q,
                                                n, // size of the input vectors
                                                x, // Pointer to vector x.
                                                incx, // Stride of vector x.
                                                y,    // Pointer to vector y.
                                                incy, // Stride of vector y.
                                                res,  // Pointer to result.
                                                depends);
    } catch (oneapi::mkl::exception const &e) {
        error_msg
            << "Unexpected MKL exception caught during dot() call:\nreason: "
            << e.what();
        is_exception_caught = true;
    } catch (sycl::exception const &e) {
        error_msg << "Unexpected SYCL exception caught during dot() call:\n"
                  << e.what();
        is_exception_caught = true;
    }

    if (is_exception_caught) // an unexpected error occurs
    {
        throw std::runtime_error(error_msg.str());
    }

    return dot_event;
}

template <typename fnT, typename varT>
struct DotContigFactory
{
    fnT get()
    {
        if constexpr (types::DotTypePairSupportFactory<varT>::is_defined) {
            return dot_impl<varT>;
        }
        else {
            return nullptr;
        }
    }
};
} // namespace dpnp::extensions::blas
