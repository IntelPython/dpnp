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

#include <oneapi/mkl.hpp>
#include <pybind11/pybind11.h>

// dpctl tensor headers
#include "utils/memory_overlap.hpp"
#include "utils/output_validation.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

#include "types_matrix.hpp"

namespace dpnp::extensions::lapack::evd
{
typedef sycl::event (*evd_impl_fn_ptr_t)(sycl::queue &,
                                         const oneapi::mkl::job,
                                         const oneapi::mkl::uplo,
                                         const std::int64_t,
                                         char *,
                                         char *,
                                         std::vector<sycl::event> &,
                                         const std::vector<sycl::event> &);

namespace dpctl_td_ns = dpctl::tensor::type_dispatch;
namespace py = pybind11;

template <typename dispatchT>
std::pair<sycl::event, sycl::event>
    evd_func(sycl::queue &exec_q,
             const std::int8_t jobz,
             const std::int8_t upper_lower,
             dpctl::tensor::usm_ndarray &eig_vecs,
             dpctl::tensor::usm_ndarray &eig_vals,
             const std::vector<sycl::event> &depends,
             const dispatchT &evd_dispatch_table)
{
    const int eig_vecs_nd = eig_vecs.get_ndim();
    const int eig_vals_nd = eig_vals.get_ndim();

    if (eig_vecs_nd != 2) {
        throw py::value_error("Unexpected ndim=" + std::to_string(eig_vecs_nd) +
                              " of an output array with eigenvectors");
    }
    else if (eig_vals_nd != 1) {
        throw py::value_error("Unexpected ndim=" + std::to_string(eig_vals_nd) +
                              " of an output array with eigenvalues");
    }

    const py::ssize_t *eig_vecs_shape = eig_vecs.get_shape_raw();
    const py::ssize_t *eig_vals_shape = eig_vals.get_shape_raw();

    if (eig_vecs_shape[0] != eig_vecs_shape[1]) {
        throw py::value_error("Output array with eigenvectors with be square");
    }
    else if (eig_vecs_shape[0] != eig_vals_shape[0]) {
        throw py::value_error(
            "Eigenvectors and eigenvalues have different shapes");
    }

    size_t src_nelems(1);

    for (int i = 0; i < eig_vecs_nd; ++i) {
        src_nelems *= static_cast<size_t>(eig_vecs_shape[i]);
    }

    if (src_nelems == 0) {
        // nothing to do
        return std::make_pair(sycl::event(), sycl::event());
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

    bool is_eig_vecs_f_contig = eig_vecs.is_f_contiguous();
    bool is_eig_vals_c_contig = eig_vals.is_c_contiguous();
    if (!is_eig_vecs_f_contig) {
        throw py::value_error(
            "An array with input matrix / output eigenvectors "
            "must be F-contiguous");
    }
    else if (!is_eig_vals_c_contig) {
        throw py::value_error(
            "An array with output eigenvalues must be C-contiguous");
    }

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    int eig_vecs_type_id =
        array_types.typenum_to_lookup_id(eig_vecs.get_typenum());
    int eig_vals_type_id =
        array_types.typenum_to_lookup_id(eig_vals.get_typenum());

    evd_impl_fn_ptr_t evd_fn =
        evd_dispatch_table[eig_vecs_type_id][eig_vals_type_id];
    if (evd_fn == nullptr) {
        throw py::value_error(
            "Types of input vectors and result array are mismatched.");
    }

    char *eig_vecs_data = eig_vecs.get_data();
    char *eig_vals_data = eig_vals.get_data();

    const std::int64_t n = eig_vecs_shape[0];
    const oneapi::mkl::job jobz_val = static_cast<oneapi::mkl::job>(jobz);
    const oneapi::mkl::uplo uplo_val =
        static_cast<oneapi::mkl::uplo>(upper_lower);

    std::vector<sycl::event> host_task_events;
    sycl::event evd_ev = evd_fn(exec_q, jobz_val, uplo_val, n, eig_vecs_data,
                                eig_vals_data, host_task_events, depends);

    sycl::event args_ev = dpctl::utils::keep_args_alive(
        exec_q, {eig_vecs, eig_vals}, host_task_events);

    return std::make_pair(args_ev, evd_ev);
}

template <typename dispatchT,
          template <typename fnT, typename T, typename RealT>
          typename factoryT>
void init_evd_dispatch_table(
    dispatchT evd_dispatch_table[][dpctl_td_ns::num_types])
{
    dpctl_td_ns::DispatchTableBuilder<dispatchT, factoryT,
                                      dpctl_td_ns::num_types>
        contig;
    contig.populate_dispatch_table(evd_dispatch_table);
}
} // namespace dpnp::extensions::lapack::evd
