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
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines functions of dpctl.tensor._tensor_impl extensions
//===----------------------------------------------------------------------===//

#pragma once
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace dpctl::tensor::py_internal
{

extern std::pair<sycl::event, sycl::event>
    py_extract(const dpctl::tensor::usm_ndarray &src,
               const dpctl::tensor::usm_ndarray &cumsum,
               int axis_start, // axis_start <= mask_i < axis_end
               int axis_end,
               const dpctl::tensor::usm_ndarray &dst,
               sycl::queue &exec_q,
               const std::vector<sycl::event> &depends = {});

extern void populate_masked_extract_dispatch_vectors(void);

extern std::pair<sycl::event, sycl::event>
    py_place(const dpctl::tensor::usm_ndarray &dst,
             const dpctl::tensor::usm_ndarray &cumsum,
             int axis_start, // axis_start <= mask_i < axis_end
             int axis_end,
             const dpctl::tensor::usm_ndarray &rhs,
             sycl::queue &exec_q,
             const std::vector<sycl::event> &depends = {});

extern void populate_masked_place_dispatch_vectors(void);

extern std::pair<sycl::event, sycl::event>
    py_nonzero(const dpctl::tensor::usm_ndarray
                   &cumsum, // int32 input array, 1D, C-contiguous
               const dpctl::tensor::usm_ndarray
                   &indexes, // int32 2D output array, C-contiguous
               const std::vector<py::ssize_t>
                   &mask_shape, // shape of array from which cumsum was computed
               sycl::queue &exec_q,
               const std::vector<sycl::event> &depends = {});

} // namespace dpctl::tensor::py_internal
