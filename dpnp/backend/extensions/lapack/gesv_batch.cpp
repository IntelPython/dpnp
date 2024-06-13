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
#include "utils/memory_overlap.hpp"
#include "utils/type_utils.hpp"

#include "common_helpers.hpp"
#include "gesv.hpp"
#include "linalg_exceptions.hpp"
#include "types_matrix.hpp"

#include "dpnp_utils.hpp"

namespace dpnp
{
namespace backend
{
namespace ext
{
namespace lapack
{
namespace mkl_lapack = oneapi::mkl::lapack;
namespace py = pybind11;
namespace type_utils = dpctl::tensor::type_utils;

typedef sycl::event (*gesv_batch_impl_fn_ptr_t)(sycl::queue &,
                                                const std::int64_t,
                                                const std::int64_t,
                                                const std::int64_t,
                                                char *,
                                                std::int64_t,
                                                char *,
                                                std::int64_t,
                                                const std::vector<sycl::event> &);

static gesv_batch_impl_fn_ptr_t gesv_batch_dispatch_vector[dpctl_td_ns::num_types];

template <typename T>
static sycl::event gesv_batch_impl(sycl::queue &exec_q,
                                   const std::int64_t n,
                                   const std::int64_t nrhs,
                                   const std::int64_t batch_size,
                                   char *in_a,
                                   std::int64_t lda,
                                   char *in_b,
                                   std::int64_t ldb,
                                   const std::vector<sycl::event> &depends)
{
    // py::gil_scoped_release release;
    type_utils::validate_type_for_device<T>(exec_q);

    T *a = reinterpret_cast<T *>(in_a);
    T *b = reinterpret_cast<T *>(in_b);

    std::int64_t a_size = n * n;
    std::int64_t b_size = n * nrhs;

    const std::int64_t scratchpad_size =
        mkl_lapack::gesv_scratchpad_size<T>(exec_q, n, nrhs, lda, ldb);
    T *scratchpad = nullptr;

    if (scratchpad_size > 0) {
        scratchpad = sycl::malloc_device<T>(scratchpad_size, exec_q);
        if (!scratchpad)
            throw std::runtime_error("Device allocation for scratchpad failed");
    }

    std::int64_t *ipiv = sycl::malloc_device<std::int64_t>(n, exec_q);
    if (!ipiv)
        throw std::runtime_error("Device allocation for ipiv failed");

    std::vector<sycl::event> gesv_task_events;
    gesv_task_events.reserve(batch_size);

    std::stringstream error_msg;
    std::int64_t info = 0;
    bool is_exception_caught = false;

    sycl::event gesv_event;

    for (std::int64_t batch_id = 0; batch_id < batch_size; ++batch_id) {
        T *a_batch = a + batch_id * a_size;
        T *b_batch = b + batch_id * b_size;

        try {
            gesv_event = mkl_lapack::gesv(
                exec_q,
                n,    // The order of the square matrix A
                    // and the number of rows in matrix B (0 ≤ n).
                nrhs, // The number of right-hand sides,
                    // i.e., the number of columns in matrix B (0 ≤ nrhs).
                a_batch,    // Pointer to the square coefficient matrix A (n x n).
                lda,  // The leading dimension of a, must be at least max(1, n).
                ipiv, // The pivot indices that define the permutation matrix P;
                    // row i of the matrix was interchanged with row ipiv(i),
                    // must be at least max(1, n).
                b_batch,    // Pointer to the right hand side matrix B (n x nrhs).
                ldb,  // The leading dimension of matrix B,
                    // must be at least max(1, n).
                scratchpad, // Pointer to scratchpad memory to be used by MKL
                            // routine for storing intermediate results.
                scratchpad_size, depends);
        } catch (mkl_lapack::exception const &e) {
            is_exception_caught = true;
            info = e.info();

            if (info < 0) {
                error_msg << "Parameter number " << -info
                        << " had an illegal value.";
            }
            else if (info == scratchpad_size && e.detail() != 0) {
                error_msg
                    << "Insufficient scratchpad size. Required size is at least "
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
                    sycl::free(scratchpad, exec_q);
                    throw LinAlgError("The input coefficient matrix is singular.");
                }
                else {
                    error_msg << "Unexpected MKL exception caught during gesv() "
                                "call:\nreason: "
                            << e.what() << "\ninfo: " << e.info();
                }
            }
            else {
                error_msg << "Unexpected MKL exception caught during gesv() "
                            "call:\nreason: "
                        << e.what() << "\ninfo: " << e.info();
            }
        } catch (sycl::exception const &e) {
            is_exception_caught = true;
            error_msg << "Unexpected SYCL exception caught during gesv() call:\n"
                    << e.what();
        }

        gesv_task_events.push_back(gesv_event);

    }

    if (is_exception_caught) // an unexpected error occurs
        {
            if (scratchpad != nullptr) {
                sycl::free(scratchpad, exec_q);
            }
            if (ipiv != nullptr) {
                sycl::free(ipiv, exec_q);
            }
            throw std::runtime_error(error_msg.str());
        }

    sycl::event ht_ev = exec_q.submit([&](sycl::handler &cgh) {
        for(const auto &el : gesv_task_events) {
                cgh.depends_on(el);
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

    if (coeff_matrix_nd != 3) {
        throw py::value_error("The coefficient matrix has ndim=" +
                              std::to_string(coeff_matrix_nd) +
                              ", but a 3-dimensional array is expected.");
    }

    if (dependent_vals_nd < 2 || dependent_vals_nd > 3) {
        throw py::value_error(
            "The dependent values array has ndim=" +
            std::to_string(dependent_vals_nd) +
            ", but a 2-dimensional or 3-dimensional array is expected.");
    }

    const py::ssize_t *coeff_matrix_shape = coeff_matrix.get_shape_raw();
    const py::ssize_t *dependent_vals_shape = dependent_vals.get_shape_raw();

    // The coeff_matrix and dependent_vals arrays must be F-contiguous arrays
    // with the shapes (n,n,batch_size) and (n,nrhs,batch_size) or
    // (n,batch_size) respectively
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

    bool is_coeff_matrix_f_contig = coeff_matrix.is_f_contiguous();
    if (!is_coeff_matrix_f_contig) {
        throw py::value_error("The coefficient matrix "
                              "must be F-contiguous.");
    }

    bool is_dependent_vals_f_contig = dependent_vals.is_f_contiguous();
    if (!is_dependent_vals_f_contig) {
        throw py::value_error("The array of dependent variables "
                              "must be F-contiguous.");
    }

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    int coeff_matrix_type_id =
        array_types.typenum_to_lookup_id(coeff_matrix.get_typenum());
    int dependent_vals_type_id =
        array_types.typenum_to_lookup_id(dependent_vals.get_typenum());

    if (coeff_matrix_type_id != dependent_vals_type_id) {
        throw py::value_error("The types of the coefficient matrix and "
                              "dependent variables are mismatched.");
    }

    gesv_batch_impl_fn_ptr_t gesv_batch_fn = gesv_batch_dispatch_vector[coeff_matrix_type_id];
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

    const std::int64_t lda = std::max<size_t>(1UL, n);
    const std::int64_t ldb = std::max<size_t>(1UL, n);

    sycl::event gesv_ev;

    gesv_ev = gesv_batch_fn(exec_q, n, nrhs, batch_size, coeff_matrix_data, lda,
                            dependent_vals_data, ldb, depends);


    sycl::event ht_ev =
        dpctl::utils::keep_args_alive(exec_q, {coeff_matrix, dependent_vals}, {gesv_ev});

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
    dpctl_td_ns::DispatchVectorBuilder<gesv_batch_impl_fn_ptr_t, GesvBatchContigFactory,
                                       dpctl_td_ns::num_types>
        contig;
    contig.populate_dispatch_vector(gesv_batch_dispatch_vector);
}
} // namespace lapack
} // namespace ext
} // namespace backend
} // namespace dpnp
