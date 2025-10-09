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
#include "utils/type_dispatch.hpp"

#include "common_helpers.hpp"
#include "evd_common_utils.hpp"
#include "types_matrix.hpp"

namespace dpnp::extensions::lapack::evd
{
using dpnp::extensions::lapack::helper::check_zeros_shape;

typedef sycl::event (*evd_impl_fn_ptr_t)(sycl::queue &,
                                         const oneapi::mkl::job,
                                         const oneapi::mkl::uplo,
                                         const std::int64_t,
                                         char *,
                                         char *,
                                         const std::vector<sycl::event> &);

namespace dpctl_td_ns = dpctl::tensor::type_dispatch;
namespace py = pybind11;

template <typename dispatchT>
std::pair<sycl::event, sycl::event>
    evd_func(sycl::queue &exec_q,
             const std::int8_t jobz,
             const std::int8_t upper_lower,
             const dpctl::tensor::usm_ndarray &eig_vecs,
             const dpctl::tensor::usm_ndarray &eig_vals,
             const std::vector<sycl::event> &depends,
             const dispatchT &evd_dispatch_table)
{
    const int eig_vecs_nd = eig_vecs.get_ndim();

    const py::ssize_t *eig_vecs_shape = eig_vecs.get_shape_raw();
    const py::ssize_t *eig_vals_shape = eig_vals.get_shape_raw();

    constexpr int expected_eig_vecs_nd = 2;
    constexpr int expected_eig_vals_nd = 1;

    common_evd_checks(exec_q, eig_vecs, eig_vals, eig_vecs_shape,
                      expected_eig_vecs_nd, expected_eig_vals_nd);

    if (eig_vecs_shape[0] != eig_vals_shape[0]) {
        throw py::value_error(
            "Eigenvectors and eigenvalues have different shapes");
    }

    if (check_zeros_shape(eig_vecs_nd, eig_vecs_shape)) {
        // nothing to do
        return std::make_pair(sycl::event(), sycl::event());
    }

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    const int eig_vecs_type_id =
        array_types.typenum_to_lookup_id(eig_vecs.get_typenum());
    const int eig_vals_type_id =
        array_types.typenum_to_lookup_id(eig_vals.get_typenum());

    evd_impl_fn_ptr_t evd_fn =
        evd_dispatch_table[eig_vecs_type_id][eig_vals_type_id];
    if (evd_fn == nullptr) {
        throw py::value_error(
            "No evd implementation is available for the specified data type "
            "of the input and output arrays.");
    }

    char *eig_vecs_data = eig_vecs.get_data();
    char *eig_vals_data = eig_vals.get_data();

    const std::int64_t n = eig_vecs_shape[0];
    const oneapi::mkl::job jobz_val = static_cast<oneapi::mkl::job>(jobz);
    const oneapi::mkl::uplo uplo_val =
        static_cast<oneapi::mkl::uplo>(upper_lower);

    sycl::event evd_ev = evd_fn(exec_q, jobz_val, uplo_val, n, eig_vecs_data,
                                eig_vals_data, depends);

    sycl::event ht_ev =
        dpctl::utils::keep_args_alive(exec_q, {eig_vecs, eig_vals}, {evd_ev});

    return std::make_pair(ht_ev, evd_ev);
}
} // namespace dpnp::extensions::lapack::evd
