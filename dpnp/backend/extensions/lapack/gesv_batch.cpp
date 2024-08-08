//*****************************************************************************
// Copyright (c) 2023-2024, Intel Corporation
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

#include <pybind11/pybind11.h>

// dpctl tensor headers
#include "utils/type_utils.hpp"

#include "common_helpers.hpp"
#include "gesv.hpp"
#include "gesv_common_utils.hpp"
#include "types_matrix.hpp"

namespace dpnp::extensions::lapack
{
namespace mkl_lapack = oneapi::mkl::lapack;
namespace py = pybind11;
namespace type_utils = dpctl::tensor::type_utils;

typedef sycl::event (*gesv_batch_impl_fn_ptr_t)(
    sycl::queue &,
    const std::int64_t,
    const std::int64_t,
    const std::int64_t,
    char *,
    char *,
    const std::vector<sycl::event> &);

static gesv_batch_impl_fn_ptr_t
    gesv_batch_dispatch_vector[dpctl_td_ns::num_types];

template <typename T>
static sycl::event gesv_batch_impl(sycl::queue &exec_q,
                                   const std::int64_t n,
                                   const std::int64_t nrhs,
                                   const std::int64_t batch_size,
                                   char *in_a,
                                   char *in_b,
                                   const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<T>(exec_q);

    T *a = reinterpret_cast<T *>(in_a);
    T *b = reinterpret_cast<T *>(in_b);

    const std::int64_t a_size = n * n;
    const std::int64_t b_size = n * nrhs;

    const std::int64_t lda = std::max<size_t>(1UL, n);
    const std::int64_t ldb = std::max<size_t>(1UL, n);

    // Get the number of independent linear streams
    const std::int64_t n_linear_streams =
        (batch_size > 16) ? 4 : ((batch_size > 4 ? 2 : 1));

    const std::int64_t scratchpad_size =
        mkl_lapack::gesv_scratchpad_size<T>(exec_q, n, nrhs, lda, ldb);

    T *scratchpad = helper::alloc_scratchpad_batch<T>(scratchpad_size,
                                                      n_linear_streams, exec_q);

    std::int64_t *ipiv = nullptr;
    try {
        ipiv = helper::alloc_ipiv_batch<T>(n, n_linear_streams, exec_q);
    } catch (const std::exception &e) {
        if (scratchpad != nullptr)
            sycl::free(scratchpad, exec_q);
        throw;
    }

    // Computation events to manage dependencies for each linear stream
    std::vector<std::vector<sycl::event>> comp_evs(n_linear_streams, depends);

    std::stringstream error_msg;
    bool is_exception_caught = false;

    // Release GIL to avoid serialization of host task
    // submissions to the same queue in OneMKL
    py::gil_scoped_release release;

    for (std::int64_t batch_id = 0; batch_id < batch_size; ++batch_id) {
        T *a_batch = a + batch_id * a_size;
        T *b_batch = b + batch_id * b_size;

        std::int64_t stream_id = (batch_id % n_linear_streams);

        T *current_scratch_gesv = scratchpad + stream_id * scratchpad_size;
        std::int64_t *current_ipiv = ipiv + stream_id * n;

        // Get the event dependencies for the current stream
        const auto &current_dep = comp_evs[stream_id];

        sycl::event gesv_event;

        try {
            gesv_event = mkl_lapack::gesv(
                exec_q,
                n,    // The order of the square matrix A
                      // and the number of rows in matrix B (0 ≤ n).
                nrhs, // The number of right-hand sides,
                      // i.e., the number of columns in matrix B (0 ≤ nrhs).
                a_batch, // Pointer to the square coefficient matrix A (n x n).
                lda, // The leading dimension of a, must be at least max(1, n).
                current_ipiv, // The pivot indices that define the permutation
                              // matrix P; row i of the matrix was interchanged
                              // with row ipiv(i), must be at least max(1, n).
                b_batch, // Pointer to the right hand side matrix B (n x nrhs).
                ldb,     // The leading dimension of matrix B,
                         // must be at least max(1, n).
                current_scratch_gesv, // Pointer to scratchpad memory to be used
                                      // by MKL routine for storing intermediate
                                      // results.
                scratchpad_size, current_dep);
        } catch (mkl_lapack::exception const &e) {
            is_exception_caught = true;
            gesv_utils::handle_lapack_exc(exec_q, lda, a, scratchpad_size,
                                          scratchpad, ipiv, e, error_msg);
        } catch (sycl::exception const &e) {
            is_exception_caught = true;
            error_msg
                << "Unexpected SYCL exception caught during gesv() call:\n"
                << e.what();
        }

        // Update the event dependencies for the current stream
        comp_evs[stream_id] = {gesv_event};
    }

    if (is_exception_caught) // an unexpected error occurs
    {
        if (scratchpad != nullptr)
            sycl::free(scratchpad, exec_q);
        if (ipiv != nullptr)
            sycl::free(ipiv, exec_q);
        throw std::runtime_error(error_msg.str());
    }

    sycl::event ht_ev = exec_q.submit([&](sycl::handler &cgh) {
        for (const auto &ev : comp_evs) {
            cgh.depends_on(ev);
        }
        auto ctx = exec_q.get_context();
        cgh.host_task([ctx, scratchpad, ipiv]() {
            sycl::free(scratchpad, ctx);
            sycl::free(ipiv, ctx);
        });
    });

    return ht_ev;
}

std::pair<sycl::event, sycl::event>
    gesv_batch(sycl::queue &exec_q,
               dpctl::tensor::usm_ndarray coeff_matrix,
               dpctl::tensor::usm_ndarray dependent_vals,
               const std::vector<sycl::event> &depends)
{
    const int coeff_matrix_nd = coeff_matrix.get_ndim();
    const int dependent_vals_nd = dependent_vals.get_ndim();

    const py::ssize_t *coeff_matrix_shape = coeff_matrix.get_shape_raw();
    const py::ssize_t *dependent_vals_shape = dependent_vals.get_shape_raw();

    constexpr int expected_coeff_matrix_ndim = 3;
    constexpr int min_dependent_vals_ndim = 2;
    constexpr int max_dependent_vals_ndim = 3;

    gesv_utils::common_gesv_checks(
        exec_q, coeff_matrix, dependent_vals, coeff_matrix_shape,
        dependent_vals_shape, expected_coeff_matrix_ndim,
        min_dependent_vals_ndim, max_dependent_vals_ndim);

    // Ensure `batch_size`, `n` and 'nrhs' are non-zero, otherwise return empty
    // events
    if (helper::check_zeros_shape(coeff_matrix_nd, coeff_matrix_shape) ||
        helper::check_zeros_shape(dependent_vals_nd, dependent_vals_shape))
    {
        // nothing to do
        return std::make_pair(sycl::event(), sycl::event());
    }

    if (dependent_vals_nd == 2) {
        if (coeff_matrix_shape[2] != dependent_vals_shape[1]) {
            throw py::value_error(
                "The batch_size of "
                " coeff_matrix and dependent_vals must be"
                " the same, but got " +
                std::to_string(coeff_matrix_shape[2]) + " and " +
                std::to_string(dependent_vals_shape[1]) + ".");
        }
    }
    else if (dependent_vals_nd == 3) {
        if (coeff_matrix_shape[2] != dependent_vals_shape[2]) {
            throw py::value_error(
                "The batch_size of "
                " coeff_matrix and dependent_vals must be"
                " the same, but got " +
                std::to_string(coeff_matrix_shape[2]) + " and " +
                std::to_string(dependent_vals_shape[2]) + ".");
        }
    }

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    const int coeff_matrix_type_id =
        array_types.typenum_to_lookup_id(coeff_matrix.get_typenum());

    gesv_batch_impl_fn_ptr_t gesv_batch_fn =
        gesv_batch_dispatch_vector[coeff_matrix_type_id];
    if (gesv_batch_fn == nullptr) {
        throw py::value_error(
            "No gesv implementation defined for the provided type "
            "of the coefficient matrix.");
    }

    char *coeff_matrix_data = coeff_matrix.get_data();
    char *dependent_vals_data = dependent_vals.get_data();

    const std::int64_t batch_size = coeff_matrix_shape[2];
    const std::int64_t n = coeff_matrix_shape[1];
    const std::int64_t nrhs =
        (dependent_vals_nd > 2) ? dependent_vals_shape[1] : 1;

    sycl::event gesv_ev =
        gesv_batch_fn(exec_q, n, nrhs, batch_size, coeff_matrix_data,
                      dependent_vals_data, depends);

    sycl::event ht_ev = dpctl::utils::keep_args_alive(
        exec_q, {coeff_matrix, dependent_vals}, {gesv_ev});

    return std::make_pair(ht_ev, gesv_ev);
}

template <typename fnT, typename T>
struct GesvBatchContigFactory
{
    fnT get()
    {
        if constexpr (types::GesvTypePairSupportFactory<T>::is_defined) {
            return gesv_batch_impl<T>;
        }
        else {
            return nullptr;
        }
    }
};

void init_gesv_batch_dispatch_vector(void)
{
    dpctl_td_ns::DispatchVectorBuilder<gesv_batch_impl_fn_ptr_t,
                                       GesvBatchContigFactory,
                                       dpctl_td_ns::num_types>
        contig;
    contig.populate_dispatch_vector(gesv_batch_dispatch_vector);
}
} // namespace dpnp::extensions::lapack
