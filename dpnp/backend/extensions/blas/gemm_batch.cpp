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

#include <stdexcept>

#include <pybind11/pybind11.h>

// utils extension header
#include "ext/common.hpp"

// dpctl tensor headers
#include "utils/memory_overlap.hpp"
#include "utils/output_validation.hpp"
#include "utils/type_utils.hpp"

#include "gemm.hpp"
#include "types_matrix.hpp"

namespace dpnp::extensions::blas
{
namespace mkl_blas = oneapi::mkl::blas;
namespace py = pybind11;
namespace type_utils = dpctl::tensor::type_utils;

using ext::common::init_dispatch_table;

typedef sycl::event (*gemm_batch_impl_fn_ptr_t)(
    sycl::queue &,
    const std::int64_t,
    const std::int64_t,
    const std::int64_t,
    const std::int64_t,
    const std::int64_t,
    const std::int64_t,
    const std::int64_t,
    const std::int64_t,
    const std::int64_t,
    const std::int64_t,
    oneapi::mkl::transpose,
    oneapi::mkl::transpose,
    const char *,
    const char *,
    char *,
    const bool,
    const std::vector<sycl::event> &);

static gemm_batch_impl_fn_ptr_t
    gemm_batch_dispatch_table[dpctl_td_ns::num_types][dpctl_td_ns::num_types];

template <typename Tab, typename Tc>
static sycl::event gemm_batch_impl(sycl::queue &exec_q,
                                   const std::int64_t m,
                                   const std::int64_t n,
                                   const std::int64_t k,
                                   const std::int64_t batch_size,
                                   const std::int64_t lda,
                                   const std::int64_t ldb,
                                   const std::int64_t ldc,
                                   const std::int64_t stridea,
                                   const std::int64_t strideb,
                                   const std::int64_t stridec,
                                   oneapi::mkl::transpose transA,
                                   oneapi::mkl::transpose transB,
                                   const char *matrixA,
                                   const char *matrixB,
                                   char *resultC,
                                   const bool is_row_major,
                                   const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<Tab>(exec_q);
    type_utils::validate_type_for_device<Tc>(exec_q);

    const Tab *a = reinterpret_cast<const Tab *>(matrixA);
    const Tab *b = reinterpret_cast<const Tab *>(matrixB);
    Tc *res = reinterpret_cast<Tc *>(resultC);

    std::stringstream error_msg;
    bool is_exception_caught = false;

    sycl::event gemm_batch_event;
    try {
        auto gemm_batch_func =
            [&](sycl::queue &q, oneapi::mkl::transpose transA,
                oneapi::mkl::transpose transB, const std::int64_t m,
                const std::int64_t n, const std::int64_t k, Tab alpha,
                const Tab *a, const std::int64_t lda,
                const std::int64_t stridea, const Tab *b,
                const std::int64_t ldb, const std::int64_t strideb, Tab beta,
                Tc *c, const std::int64_t ldc, const std::int64_t stridec,
                const std::int64_t batch_size,
                const std::vector<sycl::event> &deps) -> sycl::event {
            if (is_row_major) {
                return mkl_blas::row_major::gemm_batch(
                    q, transA, transB, m, n, k, alpha, a, lda, stridea, b, ldb,
                    strideb, beta, c, ldc, stridec, batch_size, deps);
            }
            else {
                return mkl_blas::column_major::gemm_batch(
                    q, transA, transB, m, n, k, alpha, a, lda, stridea, b, ldb,
                    strideb, beta, c, ldc, stridec, batch_size, deps);
            }
        };
        gemm_batch_event = gemm_batch_func(
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
            ldc,        // Leading dimension of matrix C.
            stridec,    // Stride between different C matrices.
            batch_size, // Specifies the number of matrix multiply
                        // operations to perform.
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

void standardize_strides_to_nonzero(std::vector<py::ssize_t> &strides,
                                    const py::ssize_t *shape)
{
    // When shape of an array along any particular dimension is 1, the stride
    // along that dimension is undefined. This function standardize the strides
    // by calculating the non-zero value of the strides.
    const std::size_t ndim = strides.size();
    const bool has_zero_stride =
        std::accumulate(strides.begin(), strides.end(), 1,
                        std::multiplies<py::ssize_t>{}) == 0;

    if (has_zero_stride) {
        for (std::size_t i = 0; i < ndim - 1; ++i) {
            strides[i] = strides[i] == 0
                             ? std::accumulate(shape + i + 1, shape + ndim, 1,
                                               std::multiplies<py::ssize_t>{})
                             : strides[i];
        }
        strides[ndim - 1] = strides[ndim - 1] == 0 ? 1 : strides[ndim - 1];
    }
}

void standardize_strides_to_zero(std::vector<py::ssize_t> &strides,
                                 const py::ssize_t *shape)
{
    // When shape of an array along any particular dimension is 1, the stride
    // along that dimension is undefined. This function standardize the strides
    // by defining such a stride as zero. This is because for these cases,
    // instead of copying the array into the additional dimension for batch
    // multiplication, we choose to use zero as the stride between different
    // matrices.  Therefore, the same array is used repeatedly.
    const std::size_t ndim = strides.size();

    for (std::size_t i = 0; i < ndim; ++i) {
        if (shape[i] <= 1) {
            strides[i] = 0;
        }
    }
}

std::tuple<sycl::event, sycl::event, bool>
    gemm_batch(sycl::queue &exec_q,
               const dpctl::tensor::usm_ndarray &matrixA,
               const dpctl::tensor::usm_ndarray &matrixB,
               const dpctl::tensor::usm_ndarray &resultC,
               const std::vector<sycl::event> &depends = {})
{
    const int matrixA_nd = matrixA.get_ndim();
    const int matrixB_nd = matrixB.get_ndim();
    const int resultC_nd = resultC.get_ndim();

    if (matrixA_nd != resultC_nd || matrixB_nd != resultC_nd) {
        throw py::value_error("The given arrays have incorrect dimensions.");
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
    const std::int64_t m = a_shape[1];
    const std::int64_t n = b_shape[2];
    const std::int64_t k = a_shape[2];
    const std::int64_t batch_size = c_shape[0];
    if (a_shape[2] != b_shape[1]) {
        throw py::value_error("The number of columns in A must be equal to "
                              "the number of rows in B.");
    }
    if (a_shape[1] != c_shape[1]) {
        throw py::value_error("The number of rows in A must be equal to "
                              "the number of rows in result array.");
    }
    if (b_shape[2] != c_shape[2]) {
        throw py::value_error("The number of columns in B must be equal to "
                              "the number of columns in result array.");
    }
    const std::int64_t src_nelems = batch_size * m * n;
    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(resultC);
    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(resultC,
                                                               src_nelems);

    std::vector<py::ssize_t> a_stride = matrixA.get_strides_vector();
    std::vector<py::ssize_t> b_stride = matrixB.get_strides_vector();
    std::vector<py::ssize_t> c_stride = resultC.get_strides_vector();
    standardize_strides_to_zero(a_stride, a_shape);
    standardize_strides_to_zero(b_stride, b_shape);
    standardize_strides_to_zero(c_stride, c_shape);
    const std::int64_t stridea = a_stride[0];
    const std::int64_t strideb = b_stride[0];
    const std::int64_t stridec = c_stride[0];

    standardize_strides_to_nonzero(a_stride, a_shape);
    standardize_strides_to_nonzero(b_stride, b_shape);
    standardize_strides_to_nonzero(c_stride, c_shape);
    const bool A_base_is_f_contig =
        a_stride[1] == 1 && a_stride[2] == a_shape[1];
    const bool A_base_is_c_contig =
        a_stride[1] == a_shape[2] && a_stride[2] == 1;
    const bool B_base_is_f_contig =
        b_stride[1] == 1 && b_stride[2] == b_shape[1];
    const bool B_base_is_c_contig =
        b_stride[1] == b_shape[2] && b_stride[2] == 1;
    const bool C_base_is_f_contig =
        c_stride[1] == 1 && c_stride[2] == c_shape[1];
    const bool C_base_is_c_contig =
        c_stride[1] == c_shape[2] && c_stride[2] == 1;

    if (!A_base_is_f_contig and !A_base_is_c_contig) {
        throw py::value_error("The 2D base of the first input array is not "
                              "c-contiguous nor f-contiguous.");
    }
    if (!B_base_is_f_contig and !B_base_is_c_contig) {
        throw py::value_error("The 2D base of the second input array is not "
                              "c-contiguous nor f-contiguous.");
    }
    if (!C_base_is_f_contig and !C_base_is_c_contig) {
        throw py::value_error("The 2D base of result array is not c-contiguous "
                              "nor f-contiguous.");
    }

    oneapi::mkl::transpose transA;
    oneapi::mkl::transpose transB;
    std::int64_t lda;
    std::int64_t ldb;

// cuBLAS supports only column-major storage
#if defined(USE_ONEMATH_CUBLAS)
    constexpr bool is_row_major = false;

    transA = A_base_is_c_contig ? oneapi::mkl::transpose::T
                                : oneapi::mkl::transpose::N;
    transB = B_base_is_c_contig ? oneapi::mkl::transpose::T
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
    if (A_base_is_f_contig && B_base_is_f_contig) {
        is_row_major = false;
    }

    if (is_row_major) {
        transA = A_base_is_f_contig ? oneapi::mkl::transpose::T
                                    : oneapi::mkl::transpose::N;
        transB = B_base_is_f_contig ? oneapi::mkl::transpose::T
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

    gemm_batch_impl_fn_ptr_t gemm_batch_fn =
        gemm_batch_dispatch_table[matrixAB_type_id][resultC_type_id];
    if (gemm_batch_fn == nullptr) {
        throw py::value_error(
            "No gemm_batch implementation is available for the specified data "
            "type of the input and output arrays.");
    }

    const char *a_typeless_ptr = matrixA.get_data();
    const char *b_typeless_ptr = matrixB.get_data();
    char *r_typeless_ptr = resultC.get_data();

    sycl::event gemm_batch_ev =
        gemm_batch_fn(exec_q, m, n, k, batch_size, lda, ldb, ldc, stridea,
                      strideb, stridec, transA, transB, a_typeless_ptr,
                      b_typeless_ptr, r_typeless_ptr, is_row_major, depends);

    sycl::event args_ev = dpctl::utils::keep_args_alive(
        exec_q, {matrixA, matrixB, resultC}, {gemm_batch_ev});

    return std::make_tuple(args_ev, gemm_batch_ev, is_row_major);
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
    init_dispatch_table<gemm_batch_impl_fn_ptr_t, GemmBatchContigFactory>(
        gemm_batch_dispatch_table);
}
} // namespace dpnp::extensions::blas
