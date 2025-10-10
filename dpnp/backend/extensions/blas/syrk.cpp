//*****************************************************************************
// Copyright (c) 2025, Intel Corporation
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

#include <cassert>
#include <stdexcept>

#include <pybind11/pybind11.h>

// utils extension header
#include "ext/common.hpp"

// dpctl tensor headers
#include "utils/memory_overlap.hpp"
#include "utils/output_validation.hpp"
#include "utils/type_utils.hpp"

#include "syrk.hpp"
#include "types_matrix.hpp"

using ext::common::Align;

namespace dpnp::extensions::blas
{
namespace mkl_blas = oneapi::mkl::blas;
namespace py = pybind11;
namespace type_utils = dpctl::tensor::type_utils;

using ext::common::init_dispatch_vector;

typedef sycl::event (*syrk_impl_fn_ptr_t)(sycl::queue &,
                                          const oneapi::mkl::transpose,
                                          const std::int64_t,
                                          const std::int64_t,
                                          const char *,
                                          const std::int64_t,
                                          char *,
                                          const std::int64_t,
                                          const bool,
                                          const std::vector<sycl::event> &);

static syrk_impl_fn_ptr_t syrk_dispatch_vector[dpctl_td_ns::num_types];

template <typename T>
constexpr void copy_to_lower_triangle(T *res,
                                      const std::size_t i,
                                      const std::size_t j,
                                      const std::int64_t ldc,
                                      const std::size_t n,
                                      const bool is_row_major)
{
    if (i < n && j < n && i > j) {
        // result form row_major::syrk is row major and result form
        // column_major::syrk is column major, so copying upper
        // triangle to lower triangle is different for each case
        if (is_row_major) {
            res[i * ldc + j] = res[j * ldc + i];
        }
        else {
            res[j * ldc + i] = res[i * ldc + j];
        }
    }
}

template <typename T, bool use_wg>
class copy_kernel;

template <typename T, bool use_wg = false>
void submit_copy_kernel(T *res,
                        const std::int64_t ldc,
                        const std::size_t n,
                        const bool is_row_major,
                        sycl::handler &cgh)
{
    using KernelName = copy_kernel<T, use_wg>;

    if constexpr (use_wg) {
        static constexpr std::size_t tile_sz = 8;
        sycl::range<2> global_range(Align(n, tile_sz), Align(n, tile_sz));
        sycl::range<2> local_range(tile_sz, tile_sz);

        cgh.parallel_for<KernelName>(
            sycl::nd_range<2>(global_range, local_range),
            [=](sycl::nd_item<2> item) {
                std::size_t i = item.get_global_id(0);
                std::size_t j = item.get_global_id(1);

                copy_to_lower_triangle(res, i, j, n, ldc, is_row_major);
            });
    }
    else {
        cgh.parallel_for<KernelName>(
            sycl::range<2>{n, n}, [=](sycl::id<2> idx) {
                std::size_t i = idx[0];
                std::size_t j = idx[1];

                copy_to_lower_triangle(res, i, j, n, ldc, is_row_major);
            });
    }
}

// kernel to copy upper triangle to lower triangle
template <typename T>
sycl::event run_copy(sycl::queue &exec_q,
                     T *res,
                     const std::int64_t ldc,
                     const std::int64_t n,
                     const bool is_row_major,
                     sycl::event &depends)
{
    const sycl::device &dev = exec_q.get_device();

    return exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        // two separate kernels are used to have better performance compared
        // to gemm on both CPU and GPU
        if (dev.is_gpu()) {
            submit_copy_kernel<T, false>(res, ldc, n, is_row_major, cgh);
        }
        else {
            assert(dev.is_cpu());
            submit_copy_kernel<T, true>(res, ldc, n, is_row_major, cgh);
        }
    });
}

template <typename T>
static sycl::event syrk_impl(sycl::queue &exec_q,
                             const oneapi::mkl::transpose transA,
                             const std::int64_t n,
                             const std::int64_t k,
                             const char *matrixA,
                             const std::int64_t lda,
                             char *resultC,
                             const std::int64_t ldc,
                             const bool is_row_major,
                             const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<T>(exec_q);

    const T *a = reinterpret_cast<const T *>(matrixA);
    T *res = reinterpret_cast<T *>(resultC);

    std::stringstream error_msg;
    bool is_exception_caught = false;

    sycl::event syrk_event;
    try {
        auto syrk_func =
            [&](sycl::queue &q, oneapi::mkl::uplo upper_lower,
                oneapi::mkl::transpose transA, const std::int64_t n,
                const std::int64_t k, T alpha, const T *a,
                const std::int64_t lda, T beta, T *c, const std::int64_t ldc,
                const std::vector<sycl::event> &deps) -> sycl::event {
            if (is_row_major) {
                return mkl_blas::row_major::syrk(q, upper_lower, transA, n, k,
                                                 alpha, a, lda, beta, c, ldc,
                                                 deps);
            }
            else {
                return mkl_blas::column_major::syrk(q, upper_lower, transA, n,
                                                    k, alpha, a, lda, beta, c,
                                                    ldc, deps);
            }
        };

        // we pass beta = 0, so passing upper or lower does not matter
        static constexpr auto uplo = oneapi::mkl::uplo::upper;
        syrk_event = syrk_func(
            exec_q,
            uplo,   // Specifies whether C’s data is stored in its upper
                    // or lower triangle
            transA, // Defines the transpose operation for matrix A:
                    // 'N' indicates no transpose, 'T' for transpose,
                    // or 'C' for a conjugate transpose.
            n,      // Number of rows in op(A).
                    // Number of rows and columns in C.
            k,      // Number of columns in op(A).
            T(1),   // Scaling factor for the rank-k update.
            a,      // Pointer to the input matrix A.
            lda,    // Leading dimension of matrix A, which is the
                    // stride between successive rows (for row major layout).
            T(0),   // Scaling factor for matrix C.
            res,    // Pointer to output matrix c, where the result is stored.
            ldc,    // Leading dimension of matrix C.
            depends);
    } catch (oneapi::mkl::exception const &e) {
        error_msg << "Unexpected MKL exception caught during syrk() "
                     "call:\nreason: "
                  << e.what();
        is_exception_caught = true;
    } catch (sycl::exception const &e) {
        error_msg << "Unexpected SYCL exception caught during syrk() call:\n"
                  << e.what();
        is_exception_caught = true;
    }

    if (is_exception_caught) // an unexpected error occurs
    {
        throw std::runtime_error(error_msg.str());
    }

    return run_copy(exec_q, res, ldc, n, is_row_major, syrk_event);
}

std::pair<sycl::event, sycl::event>
    syrk(sycl::queue &exec_q,
         const dpctl::tensor::usm_ndarray &matrixA,
         const dpctl::tensor::usm_ndarray &resultC,
         const std::vector<sycl::event> &depends)
{
    const int matrixA_nd = matrixA.get_ndim();
    const int resultC_nd = resultC.get_ndim();

    if ((matrixA_nd != 2) || (resultC_nd != 2)) {
        throw py::value_error("The given arrays have incorrect dimensions.");
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(matrixA, resultC)) {
        throw py::value_error("Input and output matrices are overlapping "
                              "segments of memory");
    }

    if (!dpctl::utils::queues_are_compatible(
            exec_q, {matrixA.get_queue(), resultC.get_queue()}))
    {
        throw py::value_error(
            "USM allocations are not compatible with the execution queue.");
    }

    const py::ssize_t *a_shape = matrixA.get_shape_raw();
    const py::ssize_t *c_shape = resultC.get_shape_raw();
    if (c_shape[0] != c_shape[1]) {
        throw py::value_error("The output matrix should be square.");
    }
    if (a_shape[0] != c_shape[0]) {
        throw py::value_error("The number of rows in A must be equal to "
                              "the number of rows in result array.");
    }

    const bool is_matrixA_f_contig = matrixA.is_f_contiguous();
    const bool is_matrixA_c_contig = matrixA.is_c_contiguous();
    if (!is_matrixA_f_contig && !is_matrixA_c_contig) {
        throw py::value_error(
            "Input matrix is not c-contiguous nor f-contiguous.");
    }

    oneapi::mkl::transpose transA;
    std::size_t src_nelems;

// cuBLAS supports only column-major storage
#if defined(USE_ONEMATH_CUBLAS)
    constexpr bool is_row_major = false;
    std::int64_t n;
    std::int64_t k;

    if (is_matrixA_f_contig) {
        transA = oneapi::mkl::transpose::N;
        n = a_shape[0];
        k = a_shape[1];
        src_nelems = n * n;
    }
    else {
        transA = oneapi::mkl::transpose::T;
        k = a_shape[0];
        n = a_shape[1];
        src_nelems = k * k;
    }
#else
    bool is_row_major = true;
    if (is_matrixA_f_contig) {
        is_row_major = false;
    }

    transA = oneapi::mkl::transpose::N;
    const std::int64_t n = a_shape[0];
    const std::int64_t k = a_shape[1];
    src_nelems = n * n;
#endif // USE_ONEMATH_CUBLAS

    const std::int64_t lda = is_row_major ? k : n;
    const std::int64_t ldc = n;
    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(resultC);
    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(resultC,
                                                               src_nelems);

    const int matrixA_typenum = matrixA.get_typenum();
    const int resultC_typenum = resultC.get_typenum();
    if (matrixA_typenum != resultC_typenum) {
        throw py::value_error("Given arrays must be of the same type.");
    }

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    const int type_id = array_types.typenum_to_lookup_id(matrixA_typenum);
    syrk_impl_fn_ptr_t syrk_fn = syrk_dispatch_vector[type_id];
    if (syrk_fn == nullptr) {
        throw py::value_error("No syrk implementation is available for the "
                              "specified data type "
                              "of the input and output arrays.");
    }

    const char *a_typeless_ptr = matrixA.get_data();
    char *r_typeless_ptr = resultC.get_data();

    sycl::event syrk_ev = syrk_fn(exec_q, transA, n, k, a_typeless_ptr, lda,
                                  r_typeless_ptr, ldc, is_row_major, depends);

    sycl::event args_ev =
        dpctl::utils::keep_args_alive(exec_q, {matrixA, resultC}, {syrk_ev});

    return std::make_pair(args_ev, syrk_ev);
}

template <typename fnT, typename varT>
struct SyrkContigFactory
{
    fnT get()
    {
        if constexpr (types::SyrkTypePairSupportFactory<varT>::is_defined) {
            return syrk_impl<varT>;
        }
        else {
            return nullptr;
        }
    } // namespace dpnp::extensions::blas
};

void init_syrk_dispatch_vector(void)
{
    init_dispatch_vector<syrk_impl_fn_ptr_t, SyrkContigFactory>(
        syrk_dispatch_vector);
}
} // namespace dpnp::extensions::blas
