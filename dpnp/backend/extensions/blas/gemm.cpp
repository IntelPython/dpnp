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

#include <stdexcept>

#include <pybind11/pybind11.h>

// dpctl tensor headers
#include "utils/memory_overlap.hpp"
#include "utils/output_validation.hpp"
#include "utils/type_utils.hpp"

#include "gemm.hpp"
#include "types_matrix.hpp"

#include "dpnp_utils.hpp"

namespace dpnp::extensions::blas
{
namespace mkl_blas = oneapi::mkl::blas;
namespace py = pybind11;
namespace type_utils = dpctl::tensor::type_utils;

typedef sycl::event (*gemm_impl_fn_ptr_t)(sycl::queue &,
                                          oneapi::mkl::transpose,
                                          oneapi::mkl::transpose,
                                          const std::int64_t,
                                          const std::int64_t,
                                          const std::int64_t,
                                          const char *,
                                          const std::int64_t,
                                          const char *,
                                          const std::int64_t,
                                          char *,
                                          const std::int64_t,
#if !defined(USE_ONEMATH_CUBLAS)
                                          const bool,
#endif // !USE_ONEMATH_CUBLAS
                                          const std::vector<sycl::event> &);

static gemm_impl_fn_ptr_t gemm_dispatch_table[dpctl_td_ns::num_types]
                                             [dpctl_td_ns::num_types];

template <typename Tab, typename Tc>
static sycl::event gemm_impl(sycl::queue &exec_q,
                             oneapi::mkl::transpose transA,
                             oneapi::mkl::transpose transB,
                             const std::int64_t m,
                             const std::int64_t n,
                             const std::int64_t k,
                             const char *matrixA,
                             const std::int64_t lda,
                             const char *matrixB,
                             const std::int64_t ldb,
                             char *resultC,
                             const std::int64_t ldc,
#if !defined(USE_ONEMATH_CUBLAS)
                             const bool is_row_major,
#endif // !USE_ONEMATH_CUBLAS
                             const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<Tab>(exec_q);
    type_utils::validate_type_for_device<Tc>(exec_q);

    const Tab *a = reinterpret_cast<const Tab *>(matrixA);
    const Tab *b = reinterpret_cast<const Tab *>(matrixB);
    Tc *res = reinterpret_cast<Tc *>(resultC);

    std::stringstream error_msg;
    bool is_exception_caught = false;

    sycl::event gemm_event;
    try {
        auto gemm_func =
            [&](sycl::queue &q, oneapi::mkl::transpose transA,
                oneapi::mkl::transpose transB, const std::int64_t m,
                const std::int64_t n, const std::int64_t k, Tab alpha,
                const Tab *a, const std::int64_t lda, const Tab *b,
                const std::int64_t ldb, Tab beta, Tc *c, const std::int64_t ldc,
                const std::vector<sycl::event> &deps) -> sycl::event {
#if defined(USE_ONEMATH_CUBLAS)
            return mkl_blas::column_major::gemm(q, transA, transB, m, n, k,
                                                alpha, a, lda, b, ldb, beta, c,
                                                ldc, deps);
#else
            if (is_row_major) {
                return mkl_blas::row_major::gemm(q, transA, transB, m, n, k,
                                                 alpha, a, lda, b, ldb, beta, c,
                                                 ldc, deps);
            }
            else {
                return mkl_blas::column_major::gemm(q, transA, transB, m, n, k,
                                                    alpha, a, lda, b, ldb, beta,
                                                    c, ldc, deps);
            }
#endif // USE_ONEMATH_CUBLAS
        };
        gemm_event = gemm_func(
            exec_q,
            transA, // Defines the transpose operation for matrix A:
                    // 'N' indicates no transpose, 'T' for transpose,
                    // or 'C' for a conjugate transpose.
            transB, // Same as transA but for matrix B.
            m,      // Number of rows in matrices A and C.
            n,      // Number of columns in matrices B and C.
            k,      // Number of columns in matrix A and rows in matrix B.
            Tab(1), // Scaling factor for the product of matrices A and B.
            a,      // Pointer to matrix A.
            lda,    // Leading dimension of matrix A, which is the
                    // stride between successive rows (for row major layout).
            b,      // Pointer to matrix B.
            ldb,    // Leading dimension of matrix B, similar to lda.
            Tab(0), // Scaling factor for matrix C.
            res,    // Pointer to matrix C, where the result is stored.
            ldc,    // Leading dimension of matrix C.
            depends);
    } catch (oneapi::mkl::exception const &e) {
        error_msg
            << "Unexpected MKL exception caught during gemm() call:\nreason: "
            << e.what();
        is_exception_caught = true;
    } catch (sycl::exception const &e) {
        error_msg << "Unexpected SYCL exception caught during gemm() call:\n"
                  << e.what();
        is_exception_caught = true;
    }

    if (is_exception_caught) // an unexpected error occurs
    {
        throw std::runtime_error(error_msg.str());
    }

    return gemm_event;
}

std::tuple<sycl::event, sycl::event, bool>
    gemm(sycl::queue &exec_q,
         const dpctl::tensor::usm_ndarray &matrixA,
         const dpctl::tensor::usm_ndarray &matrixB,
         const dpctl::tensor::usm_ndarray &resultC,
         const std::vector<sycl::event> &depends)
{
    const int matrixA_nd = matrixA.get_ndim();
    const int matrixB_nd = matrixB.get_ndim();
    const int resultC_nd = resultC.get_ndim();

    if ((matrixA_nd != 2) || (matrixB_nd != 2) || (resultC_nd != 2)) {
        throw py::value_error(
            "Input and output matrices must be two-dimensional.");
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(matrixA, resultC)) {
        throw py::value_error(
            "The first input array and output array are overlapping "
            "segments of memory");
    }
    if (overlap(matrixB, resultC)) {
        throw py::value_error(
            "The second input array and output array are overlapping "
            "segments of memory");
    }

    if (!dpctl::utils::queues_are_compatible(
            exec_q,
            {matrixA.get_queue(), matrixB.get_queue(), resultC.get_queue()}))
    {
        throw py::value_error(
            "USM allocations are not compatible with the execution queue.");
    }

    const py::ssize_t *a_shape = matrixA.get_shape_raw();
    const py::ssize_t *b_shape = matrixB.get_shape_raw();
    const py::ssize_t *c_shape = resultC.get_shape_raw();
    const std::int64_t m = a_shape[0];
    const std::int64_t n = b_shape[1];
    const std::int64_t k = a_shape[1];
    if (a_shape[1] != b_shape[0]) {
        throw py::value_error("The number of columns in A must be equal to "
                              "the number of rows in B.");
    }
    if (a_shape[0] != c_shape[0]) {
        throw py::value_error("The number of rows in A must be equal to "
                              "the number of rows in result array.");
    }
    if (b_shape[1] != c_shape[1]) {
        throw py::value_error("The number of columns in B must be equal to "
                              "the number of columns in result array.");
    }

    const std::size_t src_nelems = m * n;
    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(resultC);
    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(resultC,
                                                               src_nelems);

    const bool is_matrixA_f_contig = matrixA.is_f_contiguous();
    const bool is_matrixB_f_contig = matrixB.is_f_contiguous();
    const bool is_resultC_f_contig = resultC.is_f_contiguous();
    const bool is_matrixA_c_contig = matrixA.is_c_contiguous();
    const bool is_matrixB_c_contig = matrixB.is_c_contiguous();
    const bool is_resultC_c_contig = resultC.is_c_contiguous();

    if (!is_matrixA_f_contig and !is_matrixA_c_contig) {
        throw py::value_error(
            "The first input array is not c-contiguous nor f-contiguous.");
    }
    if (!is_matrixB_f_contig and !is_matrixB_c_contig) {
        throw py::value_error(
            "The second input array is not c-contiguous nor f-contiguous.");
    }
    if (!is_resultC_f_contig and !is_resultC_c_contig) {
        throw py::value_error(
            "Result array is not c-contiguous nor f-contiguous.");
    }

    oneapi::mkl::transpose transA;
    oneapi::mkl::transpose transB;
    std::int64_t lda;
    std::int64_t ldb;

// cuBLAS supports only column-major storage
#if defined(USE_ONEMATH_CUBLAS)
    const bool is_row_major = false;

    transA = is_matrixA_c_contig ? oneapi::mkl::transpose::T
                                 : oneapi::mkl::transpose::N;
    transB = is_matrixB_c_contig ? oneapi::mkl::transpose::T
                                 : oneapi::mkl::transpose::N;

    if (transA == oneapi::mkl::transpose::N) {
        lda = m;
    }
    else {
        lda = k;
    }
    if (transB == oneapi::mkl::transpose::N) {
        ldb = k;
    }
    else {
        ldb = n;
    }
#else
    bool is_row_major = true;
    if (is_matrixA_f_contig && is_matrixB_f_contig) {
        is_row_major = false;
    }

    if (is_row_major) {
        transA = is_matrixA_f_contig ? oneapi::mkl::transpose::T
                                     : oneapi::mkl::transpose::N;
        transB = is_matrixB_f_contig ? oneapi::mkl::transpose::T
                                     : oneapi::mkl::transpose::N;
        if (transA == oneapi::mkl::transpose::N) {
            lda = k;
        }
        else {
            lda = m;
        }
        if (transB == oneapi::mkl::transpose::N) {
            ldb = n;
        }
        else {
            ldb = k;
        }
    }
    else {
        // both A and B are f_contig so using column-major gemm and
        // no transpose is needed
        transA = oneapi::mkl::transpose::N;
        transB = oneapi::mkl::transpose::N;
        lda = m;
        ldb = k;
    }
#endif // USE_ONEMATH_CUBLAS

    const std::int64_t ldc = is_row_major ? n : m;

    const int matrixA_typenum = matrixA.get_typenum();
    const int matrixB_typenum = matrixB.get_typenum();
    const int resultC_typenum = resultC.get_typenum();

    if (matrixA_typenum != matrixB_typenum) {
        throw py::value_error("matrixA and matrixB must be of the same type.");
    }

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    const int matrixAB_type_id =
        array_types.typenum_to_lookup_id(matrixA_typenum);
    const int resultC_type_id =
        array_types.typenum_to_lookup_id(resultC_typenum);

    gemm_impl_fn_ptr_t gemm_fn =
        gemm_dispatch_table[matrixAB_type_id][resultC_type_id];
    if (gemm_fn == nullptr) {
        throw py::value_error(
            "Types of input matrices and result matrix are mismatched.");
    }

    const char *a_typeless_ptr = matrixA.get_data();
    const char *b_typeless_ptr = matrixB.get_data();
    char *r_typeless_ptr = resultC.get_data();

#if defined(USE_ONEMATH_CUBLAS)
    sycl::event gemm_ev =
        gemm_fn(exec_q, transA, transB, m, n, k, a_typeless_ptr, lda,
                b_typeless_ptr, ldb, r_typeless_ptr, ldc, depends);
#else
    sycl::event gemm_ev = gemm_fn(exec_q, transA, transB, m, n, k,
                                  a_typeless_ptr, lda, b_typeless_ptr, ldb,
                                  r_typeless_ptr, ldc, is_row_major, depends);
#endif // USE_ONEMATH_CUBLAS

    sycl::event args_ev = dpctl::utils::keep_args_alive(
        exec_q, {matrixA, matrixB, resultC}, {gemm_ev});

    return std::make_tuple(args_ev, gemm_ev, is_row_major);
}

template <typename fnT, typename Tab, typename Tc>
struct GemmContigFactory
{
    fnT get()
    {
        if constexpr (types::GemmTypePairSupportFactory<Tab, Tc>::is_defined) {
            return gemm_impl<Tab, Tc>;
        }
        else {
            return nullptr;
        }
    }
};

void init_gemm_dispatch_table(void)
{
    dpctl_td_ns::DispatchTableBuilder<gemm_impl_fn_ptr_t, GemmContigFactory,
                                      dpctl_td_ns::num_types>
        contig;
    contig.populate_dispatch_table(gemm_dispatch_table);
}
} // namespace dpnp::extensions::blas
