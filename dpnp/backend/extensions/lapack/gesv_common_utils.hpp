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

#include <pybind11/pybind11.h>

// dpctl tensor headers
#include "utils/memory_overlap.hpp"
#include "utils/output_validation.hpp"
#include "utils/sycl_alloc_utils.hpp"
#include "utils/type_dispatch.hpp"

#include "common_helpers.hpp"
#include "linalg_exceptions.hpp"

namespace dpnp::extensions::lapack::gesv_utils
{
namespace dpctl_td_ns = dpctl::tensor::type_dispatch;
namespace py = pybind11;

inline void common_gesv_checks(sycl::queue &exec_q,
                               const dpctl::tensor::usm_ndarray &coeff_matrix,
                               const dpctl::tensor::usm_ndarray &dependent_vals,
                               const py::ssize_t *coeff_matrix_shape,
                               const py::ssize_t *dependent_vals_shape,
                               const int expected_coeff_matrix_ndim,
                               const int min_dependent_vals_ndim,
                               const int max_dependent_vals_ndim)
{
    const int coeff_matrix_nd = coeff_matrix.get_ndim();
    const int dependent_vals_nd = dependent_vals.get_ndim();

    if (coeff_matrix_nd != expected_coeff_matrix_ndim) {
        throw py::value_error("The coefficient matrix has ndim=" +
                              std::to_string(coeff_matrix_nd) + ", but a " +
                              std::to_string(expected_coeff_matrix_ndim) +
                              "-dimensional array is expected.");
    }

    if (dependent_vals_nd < min_dependent_vals_ndim ||
        dependent_vals_nd > max_dependent_vals_ndim)
    {
        throw py::value_error("The dependent values array has ndim=" +
                              std::to_string(dependent_vals_nd) + ", but a " +
                              std::to_string(min_dependent_vals_ndim) +
                              "-dimensional or a " +
                              std::to_string(max_dependent_vals_ndim) +
                              "-dimensional array is expected.");
    }

    // The coeff_matrix and dependent_vals arrays must be F-contiguous arrays
    // for gesv
    // with the shapes (n, n) and (n, nrhs) or (n, ) respectively;
    // for gesv_batch
    // with the shapes (n, n, batch_size) and (n, nrhs, batch_size) or
    // (n, batch_size) respectively
    if (coeff_matrix_shape[0] != coeff_matrix_shape[1]) {
        throw py::value_error("The coefficient matrix must be square,"
                              " but got a shape of (" +
                              std::to_string(coeff_matrix_shape[0]) + ", " +
                              std::to_string(coeff_matrix_shape[1]) + ").");
    }
    if (coeff_matrix_shape[0] != dependent_vals_shape[0]) {
        throw py::value_error("The first dimension (n) of coeff_matrix and"
                              " dependent_vals must be the same, but got " +
                              std::to_string(coeff_matrix_shape[0]) + " and " +
                              std::to_string(dependent_vals_shape[0]) + ".");
    }

    // check compatibility of execution queue and allocation queue
    if (!dpctl::utils::queues_are_compatible(exec_q,
                                             {coeff_matrix, dependent_vals}))
    {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues.");
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(coeff_matrix, dependent_vals)) {
        throw py::value_error(
            "The arrays of coefficients and dependent variables "
            "are overlapping segments of memory.");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(
        dependent_vals);

    const bool is_coeff_matrix_f_contig = coeff_matrix.is_f_contiguous();
    if (!is_coeff_matrix_f_contig) {
        throw py::value_error("The coefficient matrix "
                              "must be F-contiguous.");
    }

    const bool is_dependent_vals_f_contig = dependent_vals.is_f_contiguous();
    if (!is_dependent_vals_f_contig) {
        throw py::value_error("The array of dependent variables "
                              "must be F-contiguous.");
    }

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    const int coeff_matrix_type_id =
        array_types.typenum_to_lookup_id(coeff_matrix.get_typenum());
    const int dependent_vals_type_id =
        array_types.typenum_to_lookup_id(dependent_vals.get_typenum());

    if (coeff_matrix_type_id != dependent_vals_type_id) {
        throw py::value_error("The types of the coefficient matrix and "
                              "dependent variables are mismatched.");
    }
}

template <typename T>
inline void handle_lapack_exc(sycl::queue &exec_q,
                              const std::int64_t lda,
                              T *a,
                              std::int64_t scratchpad_size,
                              T *scratchpad,
                              std::int64_t *ipiv,
                              const oneapi::mkl::lapack::exception &e,
                              std::stringstream &error_msg)
{
    std::int64_t info = e.info();
    if (info < 0) {
        error_msg << "Parameter number " << -info << " had an illegal value.";
    }
    else if (info == scratchpad_size && e.detail() != 0) {
        error_msg << "Insufficient scratchpad size. Required size is at least "
                  << e.detail();
    }
    else if (info > 0) {
        T host_U;
        exec_q.memcpy(&host_U, &a[(info - 1) * lda + info - 1], sizeof(T))
            .wait();

        using ThresholdType = typename helper::value_type_of<T>::type;

        const auto threshold =
            std::numeric_limits<ThresholdType>::epsilon() * 100;
        if (std::abs(host_U) < threshold) {
            using dpctl::tensor::alloc_utils::sycl_free_noexcept;

            if (scratchpad != nullptr)
                sycl_free_noexcept(scratchpad, exec_q);
            if (ipiv != nullptr)
                sycl_free_noexcept(ipiv, exec_q);
            throw LinAlgError("The input coefficient matrix is singular.");
        }
        else {
            error_msg << "Unexpected MKL exception caught during gesv() "
                         "call:\nreason: "
                      << e.what() << "\ninfo: " << e.info();
        }
    }
    else {
        error_msg
            << "Unexpected MKL exception caught during gesv() call:\nreason: "
            << e.what() << "\ninfo: " << e.info();
    }
}
} // namespace dpnp::extensions::lapack::gesv_utils
