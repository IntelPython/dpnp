//*****************************************************************************
// Copyright (c) 2023, Intel Corporation
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

typedef sycl::event (*gemm_impl_fn_ptr_t)(sycl::queue,
                                          oneapi::mkl::transpose,
                                          oneapi::mkl::transpose,
                                          const std::int64_t,
                                          const std::int64_t,
                                          const std::int64_t,
                                          char *,
                                          const std::int64_t,
                                          char *,
                                          const std::int64_t,
                                          char *,
                                          const std::int64_t,
                                          const std::vector<sycl::event> &);

static gemm_impl_fn_ptr_t gemm_dispatch_table[dpctl_td_ns::num_types]
                                             [dpctl_td_ns::num_types];

template <typename Tab, typename Tc>
static sycl::event gemm_impl(sycl::queue exec_q,
                             oneapi::mkl::transpose transA,
                             oneapi::mkl::transpose transB,
                             const std::int64_t m,
                             const std::int64_t n,
                             const std::int64_t k,
                             char *matrixA,
                             const std::int64_t lda,
                             char *matrixB,
                             const std::int64_t ldb,
                             char *resultC,
                             const std::int64_t ldc,
                             const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<Tab>(exec_q);
    type_utils::validate_type_for_device<Tc>(exec_q);

    Tab *a = reinterpret_cast<Tab *>(matrixA);
    Tab *b = reinterpret_cast<Tab *>(matrixB);
    Tc *res = reinterpret_cast<Tc *>(resultC);

    std::stringstream error_msg;
    std::int64_t info = 0;
    bool mkl_exception_caught = false;

    sycl::event gemm_event;
    try {
        gemm_event = mkl_blas::row_major::gemm(
            exec_q,
            transA, // Parameter indicating whether matrix A is not
                    // transposed ('N'), transposed ('T'),
                    // or conjugate transposed ('C').
            transB, // Same as transA but for matrix B.
            m,      // Number of rows in matrices A and C.
            n,      // Number of columns in matrices B and C.
            k,      // Number of columns in matrix A and rows in matrix B.
            Tab(1), // Scaling factor for the product of matrices A and B.
            a,      // Pointer to matrix A.
            lda,    // Leading dimension of matrix A, which is the
                    // stride between successive rows (for row major
                    // layout).
            b,      // Pointer to matrix B.
            ldb,    // Leading dimension of matrix B, similar to lda
            Tab(0), // Scaling factor for matrix C.
            res,    // Pointer to matrix C, where the result is stored.
            ldc,    // Leading dimension of matrix C.
            depends);
    } catch (oneapi::mkl::exception const &e) {
        error_msg
            << "Unexpected MKL exception caught during gemm() call:\nreason: "
            << e.what();
        mkl_exception_caught = true;
    } catch (sycl::exception const &e) {
        error_msg << "Unexpected SYCL exception caught during gemm() call:\n"
                  << e.what();
        info = -1;
    }

    if (info != 0 || mkl_exception_caught) // an unexpected error occurs
    {
        throw std::runtime_error(error_msg.str());
    }

    return gemm_event;
}

std::pair<sycl::event, sycl::event>
    gemm(sycl::queue exec_q,
         dpctl::tensor::usm_ndarray matrixA,
         dpctl::tensor::usm_ndarray matrixB,
         dpctl::tensor::usm_ndarray resultC,
         const std::vector<sycl::event> &depends)
{
    const int matrixA_nd = matrixA.get_ndim();
    const int matrixB_nd = matrixB.get_ndim();
    const int resultC_nd = resultC.get_ndim();

    if ((matrixA_nd != 2) || (matrixB_nd != 2) || (resultC_nd != 2)) {
        throw py::value_error("The input matrices must be of 2 dimensions.");
    }

    // check compatibility of execution queue and allocation queue
    if (!dpctl::utils::queues_are_compatible(
            exec_q,
            {matrixA.get_queue(), matrixB.get_queue(), resultC.get_queue()}))
    {
        throw std::runtime_error(
            "USM allocations are not compatible with the execution queue.");
    }

    bool is_matrixA_f_contig = matrixA.is_f_contiguous();
    bool is_matrixB_f_contig = matrixB.is_f_contiguous();

    const py::ssize_t *a_shape = matrixA.get_shape_raw();
    const py::ssize_t *b_shape = matrixB.get_shape_raw();
    const py::ssize_t *res_shape = resultC.get_shape_raw();

    if (a_shape[1] != b_shape[0]) {
        throw std::runtime_error("The number of columns in A must be equal to "
                                 "the number of rows in B.");
    }

    oneapi::mkl::transpose transA = is_matrixA_f_contig
                                        ? oneapi::mkl::transpose::T
                                        : oneapi::mkl::transpose::N;
    oneapi::mkl::transpose transB = is_matrixB_f_contig
                                        ? oneapi::mkl::transpose::T
                                        : oneapi::mkl::transpose::N;

    const std::int64_t m = a_shape[0];
    const std::int64_t n = b_shape[1];
    const std::int64_t k = a_shape[1];

    const std::int64_t lda =
        (transA == oneapi::mkl::transpose::N) ? a_shape[1] : a_shape[0];
    const std::int64_t ldb =
        (transB == oneapi::mkl::transpose::N) ? b_shape[1] : b_shape[0];
    const std::int64_t ldc = res_shape[1];

    int matrixA_typenum = matrixA.get_typenum();
    int matrixB_typenum = matrixB.get_typenum();
    int resultC_typenum = resultC.get_typenum();

    if (matrixA_typenum != matrixB_typenum) {
        throw py::value_error("matrixA and matrixB must be of the same type.");
    }

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    int matrixAB_type_id = array_types.typenum_to_lookup_id(matrixA_typenum);
    int resultC_type_id = array_types.typenum_to_lookup_id(resultC_typenum);

    gemm_impl_fn_ptr_t gemm_fn =
        gemm_dispatch_table[matrixAB_type_id][resultC_type_id];
    if (gemm_fn == nullptr) {
        throw py::value_error("Type dispatch ran into trouble.");
    }

    char *a_typeless_ptr = matrixA.get_data();
    char *b_typeless_ptr = matrixB.get_data();
    char *r_typeless_ptr = resultC.get_data();

    std::vector<sycl::event> host_task_events;
    sycl::event gemm_ev =
        gemm_fn(exec_q, transA, transB, m, n, k, a_typeless_ptr, lda,
                b_typeless_ptr, ldb, r_typeless_ptr, ldc, depends);

    host_task_events.push_back(gemm_ev);
    sycl::event args_ev = dpctl::utils::keep_args_alive(
        exec_q, {matrixA, matrixB, resultC}, host_task_events);

    return std::make_pair(args_ev, gemm_ev);
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
} // namespace blas
} // namespace ext
} // namespace backend
} // namespace dpnp
