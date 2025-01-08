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

#include <string>
#include <unordered_map>
#include <vector>

#include "dpctl4pybind11.hpp"

namespace statistics::validation
{
using array_ptr = const dpctl::tensor::usm_ndarray *;
using array_names = std::unordered_map<array_ptr, std::string>;

std::string name_of(const array_ptr &arr, const array_names &names);

void check_writable(const std::vector<array_ptr> &arrays,
                    const array_names &names);
void check_c_contig(const std::vector<array_ptr> &arrays,
                    const array_names &names);
void check_queue(const std::vector<array_ptr> &arrays,
                 const array_names &names,
                 const sycl::queue &exec_q);

void check_no_overlap(const array_ptr &inputs,
                      const array_ptr &outputs,
                      const array_names &names);
void check_no_overlap(const std::vector<array_ptr> &inputs,
                      const std::vector<array_ptr> &outputs,
                      const array_names &names);

void check_num_dims(const array_ptr &arr,
                    const size_t ndim,
                    const array_names &names);
void check_max_dims(const array_ptr &arr,
                    const size_t max_ndim,
                    const array_names &names);

void check_size_at_least(const array_ptr &arr,
                         const size_t size,
                         const array_names &names);

void common_checks(const std::vector<array_ptr> &inputs,
                   const std::vector<array_ptr> &outputs,
                   const array_names &names);
} // namespace statistics::validation
