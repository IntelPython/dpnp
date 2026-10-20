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

#include <stdexcept>

#include <pybind11/pybind11.h>

// utils extension header
#include "ext/common.hpp"

// dpnp tensor headers
#include "utils/memory_overlap.hpp"
#include "utils/output_validation.hpp"
#include "utils/type_utils.hpp"

#include "trsm.hpp"
#include "types_matrix.hpp"

namespace dpnp::extensions::blas
{
namespace mkl_blas = oneapi::mkl::blas;
namespace py = pybind11;
namespace type_utils = dpnp::tensor::type_utils;

using ext::common::init_dispatch_vector;

typedef sycl::event (*trsm_batch_impl_fn_ptr_t)(
    sycl::queue &,
    const oneapi::mkl::uplo,
    const oneapi::mkl::transpose,
    const oneapi::mkl::diag,
    const std::int64_t, // m
    const std::int64_t, // n
    const char *,       // a
    const std::int64_t, // lda
    const std::int64_t, // stride_a
    char *,             // b
    const std::int64_t, // ldb
    const std::int64_t, // stride_b
    const std::int64_t, // batch_size
    const std::vector<sycl::event> &);

static trsm_batch_impl_fn_ptr_t
    trsm_batch_dispatch_vector[dpnp_td_ns::num_types];

template <typename T>
static sycl::event trsm_batch_impl(sycl::queue &exec_q,
                                   const oneapi::mkl::uplo upper_lower,
                                   const oneapi::mkl::transpose trans,
                                   const oneapi::mkl::diag unit_diag,
                                   const std::int64_t m,
                                   const std::int64_t n,
                                   const char *matrixA,
                                   const std::int64_t lda,
                                   const std::int64_t stride_a,
                                   char *matrixB,
                                   const std::int64_t ldb,
                                   const std::int64_t stride_b,
                                   const std::int64_t batch_size,
                                   const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<T>(exec_q);

    const T *a = reinterpret_cast<const T *>(matrixA);
    T *b = reinterpret_cast<T *>(matrixB);

    std::stringstream error_msg;
    bool is_exception_caught = false;

    sycl::event trsm_batch_event;
    try {
        // Release GIL to avoid serialization of host task submissions
        // to the same queue in OneMKL
        py::gil_scoped_release lock{};

        trsm_batch_event = mkl_blas::column_major::trsm_batch(
            exec_q,
            oneapi::mkl::side::left, // op(A) is on the left: op(A) X = B
            upper_lower,             // Whether A is upper or lower triangular.
            trans,      // Defines op(A): 'N' for A, 'T' for A^T, or 'C'
                        // for A^H.
            unit_diag,  // Whether A is assumed to be unit triangular.
            m,          // Number of rows of B, and the order of A.
            n,          // Number of columns of B.
            T(1),       // Scaling factor for B.
            a,          // Pointer to the batch of triangular matrices A.
            lda,        // Leading dimension of each matrix A.
            stride_a,   // Stride between consecutive A matrices.
            b,          // Pointer to the batch of matrices B, overwritten
                        // by the solutions X.
            ldb,        // Leading dimension of each matrix B.
            stride_b,   // Stride between consecutive B matrices.
            batch_size, // Total number of matrices in the batch.
            depends);
    } catch (oneapi::mkl::exception const &e) {
        error_msg << "Unexpected MKL exception caught during trsm_batch() "
                     "call:\nreason: "
                  << e.what();
        is_exception_caught = true;
    } catch (sycl::exception const &e) {
        error_msg
            << "Unexpected SYCL exception caught during trsm_batch() call:\n"
            << e.what();
        is_exception_caught = true;
    }

    if (is_exception_caught) // an unexpected error occurs
    {
        throw std::runtime_error(error_msg.str());
    }

    return trsm_batch_event;
}

std::pair<sycl::event, sycl::event>
    trsm_batch(sycl::queue &exec_q,
               const dpnp::tensor::usm_ndarray &matrixA,
               const dpnp::tensor::usm_ndarray &matrixB,
               const std::int8_t upper_lower,
               const int trans_op,
               const bool unit_diag,
               const std::vector<sycl::event> &depends)
{
    const int a_nd = matrixA.get_ndim();
    const int b_nd = matrixB.get_ndim();

    // batch is the last dimension: (m, m, batch) and (m, n, batch)
    if (a_nd != 3 || b_nd != 3) {
        throw py::value_error(
            "The triangular and right-hand sides arrays must be "
            "3-dimensional, but got ndim=" +
            std::to_string(a_nd) + " and ndim=" + std::to_string(b_nd) + ".");
    }
    if (trans_op < 0 || trans_op > 2) {
        throw py::value_error(
            "trsm_batch: trans_op must be 0 (N), 1 (T), or 2 (C).");
    }

    const py::ssize_t *a_shape = matrixA.get_shape_raw();
    const py::ssize_t *b_shape = matrixB.get_shape_raw();

    if (a_shape[0] != a_shape[1]) {
        throw py::value_error("Expected batch of square matrices, but got "
                              "matrix shape (" +
                              std::to_string(a_shape[0]) + ", " +
                              std::to_string(a_shape[1]) + ") in batch");
    }
    if (a_shape[0] != b_shape[0] || a_shape[2] != b_shape[2]) {
        throw py::value_error("The shapes of the triangular and right-hand "
                              "sides arrays are incompatible.");
    }

    // check compatibility of execution queue and allocation queue
    if (!dpnp::utils::queues_are_compatible(exec_q, {matrixA, matrixB})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    auto const &overlap = dpnp::tensor::overlap::MemoryOverlap();
    if (overlap(matrixA, matrixB)) {
        throw py::value_error("The triangular and right-hand sides arrays "
                              "are overlapping segments of memory");
    }

    dpnp::tensor::validation::CheckWritable::throw_if_not_writable(matrixB);

    if (!matrixA.is_f_contiguous() || !matrixB.is_f_contiguous()) {
        throw py::value_error("The triangular and right-hand sides arrays "
                              "must be F-contiguous");
    }

    auto array_types = dpnp_td_ns::usm_ndarray_types();
    const int a_type_id =
        array_types.typenum_to_lookup_id(matrixA.get_typenum());
    const int b_type_id =
        array_types.typenum_to_lookup_id(matrixB.get_typenum());

    if (a_type_id != b_type_id) {
        throw py::value_error("The types of the triangular and right-hand "
                              "sides arrays are mismatched");
    }

    trsm_batch_impl_fn_ptr_t trsm_batch_fn =
        trsm_batch_dispatch_vector[a_type_id];
    if (trsm_batch_fn == nullptr) {
        throw py::value_error(
            "No trsm_batch implementation defined for the provided type "
            "of the input matrix");
    }

    const std::int64_t m = a_shape[0];
    const std::int64_t n = b_shape[1];
    const std::int64_t batch_size = a_shape[2];

    const std::int64_t lda = std::max<std::int64_t>(1, m);
    const std::int64_t ldb = std::max<std::int64_t>(1, m);
    const std::int64_t stride_a = m * m;
    const std::int64_t stride_b = m * n;

    const oneapi::mkl::uplo uplo_val =
        static_cast<oneapi::mkl::uplo>(upper_lower);
    const oneapi::mkl::transpose trans =
        (trans_op == 0)   ? oneapi::mkl::transpose::N
        : (trans_op == 1) ? oneapi::mkl::transpose::T
                          : oneapi::mkl::transpose::C;
    const oneapi::mkl::diag diag_val =
        unit_diag ? oneapi::mkl::diag::unit : oneapi::mkl::diag::nonunit;

    sycl::event trsm_batch_ev = trsm_batch_fn(
        exec_q, uplo_val, trans, diag_val, m, n, matrixA.get_data(), lda,
        stride_a, matrixB.get_data(), ldb, stride_b, batch_size, depends);

    sycl::event args_ev = dpnp::utils::keep_args_alive(
        exec_q, {matrixA, matrixB}, {trsm_batch_ev});

    return std::make_pair(args_ev, trsm_batch_ev);
}

template <typename fnT, typename T>
struct TrsmBatchContigFactory
{
    fnT get()
    {
        if constexpr (types::TrsmBatchTypePairSupportFactory<T>::is_defined) {
            return trsm_batch_impl<T>;
        }
        else {
            return nullptr;
        }
    }
};

void init_trsm_batch_dispatch_vector(void)
{
    init_dispatch_vector<trsm_batch_impl_fn_ptr_t, TrsmBatchContigFactory>(
        trsm_batch_dispatch_vector);
}
} // namespace dpnp::extensions::blas
