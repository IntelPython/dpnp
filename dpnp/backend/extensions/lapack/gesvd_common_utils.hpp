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
#include <oneapi/mkl.hpp>
#include <pybind11/pybind11.h>

// dpctl tensor headers
#include "utils/memory_overlap.hpp"
#include "utils/output_validation.hpp"
#include "utils/type_dispatch.hpp"

#include "common_helpers.hpp"

namespace dpnp::extensions::lapack::gesvd_utils
{
namespace dpctl_td_ns = dpctl::tensor::type_dispatch;
namespace py = pybind11;

// Converts a given character code (ord) to the corresponding
// oneapi::mkl::jobsvd enumeration value
inline oneapi::mkl::jobsvd process_job(const std::int8_t job_val)
{
    switch (job_val) {
    case 'A':
        return oneapi::mkl::jobsvd::vectors;
    case 'S':
        return oneapi::mkl::jobsvd::somevec;
    case 'O':
        return oneapi::mkl::jobsvd::vectorsina;
    case 'N':
        return oneapi::mkl::jobsvd::novec;
    default:
        throw std::invalid_argument("Unknown value for job");
    }
}

inline void common_gesvd_checks(sycl::queue &exec_q,
                                const dpctl::tensor::usm_ndarray &a_array,
                                const dpctl::tensor::usm_ndarray &out_s,
                                const dpctl::tensor::usm_ndarray &out_u,
                                const dpctl::tensor::usm_ndarray &out_vt,
                                const std::int8_t jobu_val,
                                const std::int8_t jobvt_val,
                                const int expected_a_u_vt_ndim,
                                const int expected_s_ndim)
{
    const int a_array_nd = a_array.get_ndim();
    const int out_u_array_nd = out_u.get_ndim();
    const int out_s_array_nd = out_s.get_ndim();
    const int out_vt_array_nd = out_vt.get_ndim();

    if (a_array_nd != expected_a_u_vt_ndim) {
        throw py::value_error(
            "The input array has ndim=" + std::to_string(a_array_nd) +
            ", but a " + std::to_string(expected_a_u_vt_ndim) +
            "-dimensional array is expected.");
    }

    if (out_s_array_nd != expected_s_ndim) {
        throw py::value_error("The output array of singular values has ndim=" +
                              std::to_string(out_s_array_nd) + ", but a " +
                              std::to_string(expected_s_ndim) +
                              "-dimensional array is expected.");
    }

    if (jobu_val == 'N' && jobvt_val == 'N') {
        if (out_u_array_nd != 0) {
            throw py::value_error(
                "The output array of the left singular vectors has ndim=" +
                std::to_string(out_u_array_nd) +
                ", but it is not used and should have ndim=0.");
        }
        if (out_vt_array_nd != 0) {
            throw py::value_error(
                "The output array of the right singular vectors has ndim=" +
                std::to_string(out_vt_array_nd) +
                ", but it is not used and should have ndim=0.");
        }
    }
    else {
        if (out_u_array_nd != expected_a_u_vt_ndim) {
            throw py::value_error(
                "The output array of the left singular vectors has ndim=" +
                std::to_string(out_u_array_nd) + ", but a " +
                std::to_string(expected_a_u_vt_ndim) +
                "-dimensional array is expected.");
        }
        if (out_vt_array_nd != expected_a_u_vt_ndim) {
            throw py::value_error(
                "The output array of the right singular vectors has ndim=" +
                std::to_string(out_vt_array_nd) + ", but a " +
                std::to_string(expected_a_u_vt_ndim) +
                "-dimensional array is expected.");
        }
    }

    // check compatibility of execution queue and allocation queue
    if (!dpctl::utils::queues_are_compatible(exec_q,
                                             {a_array, out_s, out_u, out_vt}))
    {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues.");
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(a_array, out_s) || overlap(a_array, out_u) ||
        overlap(a_array, out_vt) || overlap(out_s, out_u) ||
        overlap(out_s, out_vt) || overlap(out_u, out_vt))
    {
        throw py::value_error("Arrays have overlapping segments of memory");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(a_array);
    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(out_s);
    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(out_u);
    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(out_vt);

    const bool is_a_array_f_contig = a_array.is_f_contiguous();
    if (!is_a_array_f_contig) {
        throw py::value_error("The input array must be F-contiguous");
    }

    const bool is_out_u_array_f_contig = out_u.is_f_contiguous();
    const bool is_out_vt_array_f_contig = out_vt.is_f_contiguous();

    if (!is_out_u_array_f_contig || !is_out_vt_array_f_contig) {
        throw py::value_error("The output arrays of the left and right "
                              "singular vectors must be F-contiguous");
    }

    const bool is_out_s_array_c_contig = out_s.is_c_contiguous();

    if (!is_out_s_array_c_contig) {
        throw py::value_error("The output array of singular values "
                              "must be C-contiguous");
    }

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    const int a_array_type_id =
        array_types.typenum_to_lookup_id(a_array.get_typenum());
    const int out_u_type_id =
        array_types.typenum_to_lookup_id(out_u.get_typenum());
    const int out_vt_type_id =
        array_types.typenum_to_lookup_id(out_vt.get_typenum());

    if (a_array_type_id != out_u_type_id || a_array_type_id != out_vt_type_id) {
        throw py::type_error(
            "Input array, output left singular vectors array, "
            "and outpuy right singular vectors array must have "
            "the same data type");
    }
}

// Check if the shape of input arrays for gesvd has any non-zero dimension.
inline bool check_zeros_shape_gesvd(const dpctl::tensor::usm_ndarray &a_array,
                                    const dpctl::tensor::usm_ndarray &out_s,
                                    const dpctl::tensor::usm_ndarray &out_u,
                                    const dpctl::tensor::usm_ndarray &out_vt,
                                    const std::int8_t jobu_val,
                                    const std::int8_t jobvt_val)
{

    const int a_array_nd = a_array.get_ndim();
    const int out_u_array_nd = out_u.get_ndim();
    const int out_s_array_nd = out_s.get_ndim();
    const int out_vt_array_nd = out_vt.get_ndim();

    const py::ssize_t *a_array_shape = a_array.get_shape_raw();
    const py::ssize_t *s_out_shape = out_s.get_shape_raw();
    const py::ssize_t *u_out_shape = out_u.get_shape_raw();
    const py::ssize_t *vt_out_shape = out_vt.get_shape_raw();

    bool is_zeros_shape = helper::check_zeros_shape(a_array_nd, a_array_shape);
    if (jobu_val == 'N' && jobvt_val == 'N') {
        is_zeros_shape = is_zeros_shape || helper::check_zeros_shape(
                                               out_vt_array_nd, vt_out_shape);
    }
    else {
        is_zeros_shape =
            is_zeros_shape ||
            helper::check_zeros_shape(out_u_array_nd, u_out_shape) ||
            helper::check_zeros_shape(out_s_array_nd, s_out_shape) ||
            helper::check_zeros_shape(out_vt_array_nd, vt_out_shape);
    }

    return is_zeros_shape;
}

inline void handle_lapack_exc(const std::int64_t scratchpad_size,
                              const oneapi::mkl::lapack::exception &e,
                              std::stringstream &error_msg)
{
    const std::int64_t info = e.info();
    if (info < 0) {
        error_msg << "Parameter number " << -info << " had an illegal value.";
    }
    else if (info == scratchpad_size && e.detail() != 0) {
        error_msg << "Insufficient scratchpad size. Required size is at least "
                  << e.detail();
    }
    else if (info > 0) {
        error_msg << "The algorithm computing SVD failed to converge; " << info
                  << " off-diagonal elements of an intermediate "
                  << "bidiagonal form did not converge to zero.\n";
    }
    else {
        error_msg
            << "Unexpected MKL exception caught during gesv() call:\nreason: "
            << e.what() << "\ninfo: " << e.info();
    }
}
} // namespace dpnp::extensions::lapack::gesvd_utils
