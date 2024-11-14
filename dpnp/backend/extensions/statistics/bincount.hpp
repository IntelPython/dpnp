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

#include <dpctl4pybind11.hpp>
#include <sycl/sycl.hpp>

#include "dispatch_table.hpp"

namespace dpctl_td_ns = dpctl::tensor::type_dispatch;

namespace statistics
{
namespace histogram
{
struct Bincount
{
    using FnT = sycl::event (*)(sycl::queue &,
                                const void *,
                                const int64_t,
                                const int64_t,
                                const void *,
                                void *,
                                const size_t,
                                const size_t,
                                const std::vector<sycl::event> &);

    common::DispatchTable2<FnT> dispatch_table;

    Bincount();

    std::tuple<sycl::event, sycl::event>
        call(const dpctl::tensor::usm_ndarray &input,
             const int64_t min,
             const int64_t max,
             const std::optional<const dpctl::tensor::usm_ndarray> &weights,
             dpctl::tensor::usm_ndarray &output,
             const std::vector<sycl::event> &depends);
};

void populate_bincount(py::module_ m);
} // namespace histogram
} // namespace statistics
