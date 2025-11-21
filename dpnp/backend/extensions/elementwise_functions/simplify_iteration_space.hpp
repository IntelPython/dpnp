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

#include <pybind11/pybind11.h>
#include <vector>

namespace dpnp::extensions::py_internal
{
namespace py = pybind11;

void simplify_iteration_space(int &,
                              const py::ssize_t *const &,
                              std::vector<py::ssize_t> const &,
                              std::vector<py::ssize_t> const &,
                              std::vector<py::ssize_t> &,
                              std::vector<py::ssize_t> &,
                              std::vector<py::ssize_t> &,
                              py::ssize_t &,
                              py::ssize_t &);

void simplify_iteration_space_3(int &,
                                const py::ssize_t *const &,
                                // src1
                                std::vector<py::ssize_t> const &,
                                // src2
                                std::vector<py::ssize_t> const &,
                                // dst
                                std::vector<py::ssize_t> const &,
                                // output
                                std::vector<py::ssize_t> &,
                                std::vector<py::ssize_t> &,
                                std::vector<py::ssize_t> &,
                                std::vector<py::ssize_t> &,
                                py::ssize_t &,
                                py::ssize_t &,
                                py::ssize_t &);

void simplify_iteration_space_4(int &,
                                const py::ssize_t *const &,
                                // src1
                                std::vector<py::ssize_t> const &,
                                // src2
                                std::vector<py::ssize_t> const &,
                                // src3
                                std::vector<py::ssize_t> const &,
                                // dst
                                std::vector<py::ssize_t> const &,
                                // output
                                std::vector<py::ssize_t> &,
                                std::vector<py::ssize_t> &,
                                std::vector<py::ssize_t> &,
                                std::vector<py::ssize_t> &,
                                std::vector<py::ssize_t> &,
                                py::ssize_t &,
                                py::ssize_t &,
                                py::ssize_t &,
                                py::ssize_t &);
} // namespace dpnp::extensions::py_internal
