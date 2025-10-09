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

// dpctl tensor headers
#include "utils/memory_overlap.hpp"
#include "utils/output_validation.hpp"

namespace dpnp::extensions::lapack::evd
{
namespace py = pybind11;

inline void common_evd_checks(sycl::queue &exec_q,
                              const dpctl::tensor::usm_ndarray &eig_vecs,
                              const dpctl::tensor::usm_ndarray &eig_vals,
                              const py::ssize_t *eig_vecs_shape,
                              const int expected_eig_vecs_nd,
                              const int expected_eig_vals_nd)
{
    const int eig_vecs_nd = eig_vecs.get_ndim();
    const int eig_vals_nd = eig_vals.get_ndim();

    if (eig_vecs_nd != expected_eig_vecs_nd) {
        throw py::value_error("The output eigenvectors array has ndim=" +
                              std::to_string(eig_vecs_nd) + ", but a " +
                              std::to_string(expected_eig_vecs_nd) +
                              "-dimensional array is expected.");
    }
    else if (eig_vals_nd != expected_eig_vals_nd) {
        throw py::value_error("The output eigenvalues array has ndim=" +
                              std::to_string(eig_vals_nd) + ", but a " +
                              std::to_string(expected_eig_vals_nd) +
                              "-dimensional array is expected.");
    }

    if (eig_vecs_shape[0] != eig_vecs_shape[1]) {
        throw py::value_error("Output array with eigenvectors must be square");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(eig_vecs);
    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(eig_vals);

    // check compatibility of execution queue and allocation queue
    if (!dpctl::utils::queues_are_compatible(exec_q, {eig_vecs, eig_vals})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(eig_vecs, eig_vals)) {
        throw py::value_error("Arrays with eigenvectors and eigenvalues are "
                              "overlapping segments of memory");
    }

    const bool is_eig_vecs_f_contig = eig_vecs.is_f_contiguous();
    const bool is_eig_vals_c_contig = eig_vals.is_c_contiguous();
    if (!is_eig_vecs_f_contig) {
        throw py::value_error(
            "An array with input matrix / output eigenvectors "
            "must be F-contiguous");
    }
    else if (!is_eig_vals_c_contig) {
        throw py::value_error(
            "An array with output eigenvalues must be C-contiguous");
    }
}
} // namespace dpnp::extensions::lapack::evd
