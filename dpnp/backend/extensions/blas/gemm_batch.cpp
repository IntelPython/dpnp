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

#include <pybind11/pybind11.h>

// dpctl tensor headers
#include "utils/memory_overlap.hpp"
#include "utils/type_utils.hpp"

#include "gemm.hpp"
#include "types_matrix.hpp"

#include "dpnp_utils.hpp"

namespace dpnp
{
namespace backend
{
namespace ext
{
namespace blas
{
namespace mkl_blas = oneapi::mkl::blas;
namespace py = pybind11;
namespace type_utils = dpctl::tensor::type_utils;

typedef sycl::event (*gemm_batch_impl_fn_ptr_t)(
    sycl::queue,
    const std::int64_t,
    const std::int64_t,
    const std::int64_t,
    const std::int64_t,
    const std::int64_t,
    const std::int64_t,
    const std::int64_t,
    size_t,
    size_t,
    size_t,
    oneapi::mkl::transpose,
    oneapi::mkl::transpose,
    char *,
    char *,
    char *,
    const std::vector<sycl::event> &);

static gemm_batch_impl_fn_ptr_t
    gemm_batch_dispatch_table[dpctl_td_ns::num_types][dpctl_td_ns::num_types];

template <typename Tab, typename Tc>
static sycl::event gemm_batch_impl(sycl::queue exec_q,
                                   const std::int64_t m,
                                   const std::int64_t n,
                                   const std::int64_t k,
                                   const std::int64_t batch_size,
                                   const std::int64_t lda,
                                   const std::int64_t ldb,
                                   const std::int64_t ld_result,
                                   size_t stridea,
                                   size_t strideb,
                                   size_t stridec,
                                   oneapi::mkl::transpose transA,
                                   oneapi::mkl::transpose transB,
                                   char *matrixA,
                                   char *matrixB,
                                   char *resultC,
                                   const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<Tab>(exec_q);
    type_utils::validate_type_for_device<Tc>(exec_q);

    Tab *a = reinterpret_cast<Tab *>(matrixA);
    Tab *b = reinterpret_cast<Tab *>(matrixB);
    Tc *res = reinterpret_cast<Tc *>(resultC);

    std::stringstream error_msg;
    bool is_exception_caught = false;

    sycl::event gemm_batch_event;
    try {
        gemm_batch_event = mkl_blas::row_major::gemm_batch(
            exec_q,
            transA,     // Defines the transpose operation for matrix A:
                        // 'N' indicates no transpose, 'T' for transpose,
                        // or 'C' for a conjugate transpose.
            transB,     // Same as transA but for matrix B.
            m,          // Number of rows in matrices A and C.
            n,          // Number of columns in matrices B and C.
            k,          // Number of columns in matrix A and rows in matrix B.
            Tab(1),     // Scaling factor for the product of matrices A and B.
            a,          // Pointer to matrix A.
            lda,        // Leading dimension of matrix A, which is the
                        // stride between successive rows (for row major
                        // layout).
            stridea,    // Stride between different A matrices.
            b,          // Pointer to matrix B.
            ldb,        // Leading dimension of matrix B, similar to lda.
            strideb,    // Stride between different B matrices.
            Tab(0),     // Scaling factor for matrix C.
            res,        // Pointer to matrix C, where the result is stored.
            ld_result,  // Leading dimension of matrix C.
            stridec,    // Stride between different C matrices.
            batch_size, // Specifies the number of matrix multiply operations to
                        // perform.
            depends);
    } catch (oneapi::mkl::exception const &e) {
        error_msg << "Unexpected MKL exception caught during gemm_batch() "
                     "call:\nreason: "
                  << e.what();
        is_exception_caught = true;
    } catch (sycl::exception const &e) {
        error_msg
            << "Unexpected SYCL exception caught during gemm_batch() call:\n"
            << e.what();
        is_exception_caught = true;
    }

    if (is_exception_caught) // an unexpected error occurs
    {
        throw std::runtime_error(error_msg.str());
    }

    return gemm_batch_event;
}

std::pair<sycl::event, sycl::event>
    gemm_batch(sycl::queue exec_q,
               dpctl::tensor::usm_ndarray matrixA,
               dpctl::tensor::usm_ndarray matrixB,
               dpctl::tensor::usm_ndarray resultC,
               const std::int64_t batch_size,
               size_t stridea,
               size_t strideb,
               size_t stridec,
               const std::vector<sycl::event> &depends = {})
{
    if (!dpctl::utils::queues_are_compatible(
            exec_q,
            {matrixA.get_queue(), matrixB.get_queue(), resultC.get_queue()}))
    {
        throw py::value_error(
            "USM allocations are not compatible with the execution queue.");
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(matrixA, resultC)) {
        throw py::value_error("Input array 1 and output array are overlapping "
                              "segments of memory");
    }
    if (overlap(matrixB, resultC)) {
        throw py::value_error("Input array 2 and output array are overlapping "
                              "segments of memory");
    }

    const int matrixA_nd = matrixA.get_ndim();
    const int matrixB_nd = matrixB.get_ndim();
    const py::ssize_t *a_shape = matrixA.get_shape_raw();
    const py::ssize_t *b_shape = matrixB.get_shape_raw();

    if (a_shape[matrixA_nd - 1] != b_shape[matrixB_nd - 2]) {
        throw py::value_error("The number of columns in A must be equal to "
                              "the number of rows in B.");
    }

    const std::int64_t m = a_shape[matrixA_nd - 2];
    const std::int64_t n = b_shape[matrixB_nd - 1];
    const std::int64_t k = a_shape[matrixA_nd - 1];

    // transA and transB are always True
    oneapi::mkl::transpose transA = oneapi::mkl::transpose::N;
    oneapi::mkl::transpose transB = oneapi::mkl::transpose::N;

    int matrixA_typenum = matrixA.get_typenum();
    int matrixB_typenum = matrixB.get_typenum();
    int resultC_typenum = resultC.get_typenum();

    if (matrixA_typenum != matrixB_typenum) {
        throw py::value_error("matrixA and matrixB must be of the same type.");
    }
    auto array_types = dpctl_td_ns::usm_ndarray_types();
    int matrixAB_type_id = array_types.typenum_to_lookup_id(matrixA_typenum);
    int resultC_type_id = array_types.typenum_to_lookup_id(resultC_typenum);

    gemm_batch_impl_fn_ptr_t gemm_batch_fn =
        gemm_batch_dispatch_table[matrixAB_type_id][resultC_type_id];
    if (gemm_batch_fn == nullptr) {
        throw py::value_error(
            "Types of input matrices and result matrix are mismatched.");
    }

    char *a_typeless_ptr = matrixA.get_data();
    char *b_typeless_ptr = matrixB.get_data();
    char *r_typeless_ptr = resultC.get_data();

    // Note that lda = k, ldb = n, and ld_result = n
    sycl::event gemm_batch_ev = gemm_batch_fn(
        exec_q, m, n, k, batch_size, k, n, n, stridea, strideb, stridec, transA,
        transB, a_typeless_ptr, b_typeless_ptr, r_typeless_ptr, depends);

    sycl::event args_batch_ev = dpctl::utils::keep_args_alive(
        exec_q, {matrixA, matrixB, resultC}, {gemm_batch_ev});

    return std::make_pair(args_batch_ev, gemm_batch_ev);
}

template <typename fnT, typename Tab, typename Tc>
struct GemmBatchContigFactory
{
    fnT get()
    {
        if constexpr (types::GemmBatchTypePairSupportFactory<Tab,
                                                             Tc>::is_defined) {
            return gemm_batch_impl<Tab, Tc>;
        }
        else {
            return nullptr;
        }
    }
};

void init_gemm_batch_dispatch_table(void)
{
    dpctl_td_ns::DispatchTableBuilder<gemm_batch_impl_fn_ptr_t,
                                      GemmBatchContigFactory,
                                      dpctl_td_ns::num_types>
        contig;
    contig.populate_dispatch_table(gemm_batch_dispatch_table);
}
} // namespace blas
} // namespace ext
} // namespace backend
} // namespace dpnp
