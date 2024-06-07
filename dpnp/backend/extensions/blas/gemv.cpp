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
#include "utils/output_validation.hpp"
#include "utils/type_utils.hpp"

#include "gemv.hpp"
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

typedef sycl::event (*gemv_impl_fn_ptr_t)(sycl::queue &,
                                          oneapi::mkl::transpose,
                                          const std::int64_t,
                                          const std::int64_t,
                                          char *,
                                          const std::int64_t,
                                          char *,
                                          const std::int64_t,
                                          char *,
                                          const std::int64_t,
                                          bool,
                                          const std::vector<sycl::event> &);

static gemv_impl_fn_ptr_t gemv_dispatch_vector[dpctl_td_ns::num_types];

template <typename T>
static sycl::event gemv_impl(sycl::queue &exec_q,
                             oneapi::mkl::transpose transA,
                             const std::int64_t m,
                             const std::int64_t n,
                             char *matrixA,
                             const std::int64_t lda,
                             char *vectorX,
                             const std::int64_t incx,
                             char *vectorY,
                             const std::int64_t incy,
                             bool is_row_major,
                             const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<T>(exec_q);

    T *a = reinterpret_cast<T *>(matrixA);
    T *x = reinterpret_cast<T *>(vectorX);
    T *y = reinterpret_cast<T *>(vectorY);

    std::stringstream error_msg;
    bool is_exception_caught = false;

    sycl::event gemv_event;
    try {
        auto gemv_func =
            [&](sycl::queue &q, oneapi::mkl::transpose transA, std::int64_t m,
                std::int64_t n, T alpha, const T *a, std::int64_t lda,
                const T *x, std::int64_t incx, T beta, T *y, std::int64_t incy,
                const std::vector<sycl::event> &deps) -> sycl::event {
            if (is_row_major) {
                return mkl_blas::row_major::gemv(q, transA, m, n, alpha, a, lda,
                                                 x, incx, beta, y, incy, deps);
            }
            else {
                return mkl_blas::column_major::gemv(q, transA, m, n, alpha, a,
                                                    lda, x, incx, beta, y, incy,
                                                    deps);
            }
        };
        gemv_event = gemv_func(
            exec_q,
            transA, // Defines the transpose operation for matrix A:
                    // 'N' indicates no transpose, 'T' for transpose,
                    // or 'C' for a conjugate transpose.
            m,      // Number of rows in matrix A.
            n,      // Number of columns in matrix A.
            T(1),   // Scaling factor for the matrix-vector product.
            a,      // Pointer to the input matrix A.
            lda,    // Leading dimension of matrix A, which is the
                    // stride between successive rows (for row major
                    // layout).
            x,      // Pointer to the input vector x.
            incx,   // The stride of vector x.
            T(0),   // Scaling factor for vector y.
            y,      // Pointer to output vector y, where the result is stored.
            incy,   // The stride of vector y.
            depends);
    } catch (oneapi::mkl::exception const &e) {
        error_msg
            << "Unexpected MKL exception caught during gemv() call:\nreason: "
            << e.what();
        is_exception_caught = true;
    } catch (sycl::exception const &e) {
        error_msg << "Unexpected SYCL exception caught during gemv() call:\n"
                  << e.what();
        is_exception_caught = true;
    }

    if (is_exception_caught) // an unexpected error occurs
    {
        throw std::runtime_error(error_msg.str());
    }

    return gemv_event;
}

std::pair<sycl::event, sycl::event>
    gemv(sycl::queue &exec_q,
         dpctl::tensor::usm_ndarray matrixA,
         dpctl::tensor::usm_ndarray vectorX,
         dpctl::tensor::usm_ndarray vectorY,
         bool transpose,
         const std::vector<sycl::event> &depends)
{
    const int matrixA_nd = matrixA.get_ndim();
    const int vectorX_nd = vectorX.get_ndim();
    const int vectorY_nd = vectorY.get_ndim();

    if ((matrixA_nd != 2) || (vectorX_nd != 1) || (vectorY_nd != 1)) {
        throw py::value_error("The arrays have incorrect dimensions.");
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(matrixA, vectorY)) {
        throw py::value_error("Input matrix and output vector are overlapping "
                              "segments of memory");
    }
    if (overlap(vectorX, vectorY)) {
        throw py::value_error("Input vector and output vector are overlapping "
                              "segments of memory");
    }

    if (!dpctl::utils::queues_are_compatible(
            exec_q,
            {matrixA.get_queue(), vectorX.get_queue(), vectorY.get_queue()}))
    {
        throw py::value_error(
            "USM allocations are not compatible with the execution queue.");
    }

    bool is_matrixA_f_contig = matrixA.is_f_contiguous();
    bool is_matrixA_c_contig = matrixA.is_c_contiguous();

    if (!is_matrixA_f_contig and !is_matrixA_c_contig) {
        throw py::value_error(
            "Input matrix is not c-contiguous nor f-contiguous.");
    }

    bool is_row_major = true;
    if (is_matrixA_f_contig) {
        is_row_major = false;
    }

    const py::ssize_t *a_shape = matrixA.get_shape_raw();
    const py::ssize_t *x_shape = vectorX.get_shape_raw();
    const py::ssize_t *y_shape = vectorY.get_shape_raw();
    const std::int64_t m = a_shape[0];
    const std::int64_t n = a_shape[1];
    const std::int64_t lda = is_row_major ? n : m;

    oneapi::mkl::transpose transA;
    size_t src_nelems;
    if (transpose) {
        transA = oneapi::mkl::transpose::T;
        src_nelems = n;
        if (m != x_shape[0]) {
            throw py::value_error("The number of rows in A must be equal to "
                                  "the number of elements in X.");
        }
        if (n != y_shape[0]) {
            throw py::value_error("The number of columns in A must be equal to "
                                  "the number of elements in Y.");
        }
    }
    else {
        transA = oneapi::mkl::transpose::N;
        src_nelems = m;
        if (n != x_shape[0]) {
            throw py::value_error("The number of columns in A must be equal to "
                                  "the number of elements in X.");
        }
        if (m != y_shape[0]) {
            throw py::value_error("The number of rows in A must be equal to "
                                  "the number of elements in Y.");
        }
    }
    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(vectorY);
    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(vectorY,
                                                               src_nelems);

    int matrixA_typenum = matrixA.get_typenum();
    int vectorX_typenum = vectorX.get_typenum();
    int vectorY_typenum = vectorY.get_typenum();

    if (matrixA_typenum != vectorX_typenum ||
        matrixA_typenum != vectorY_typenum) {
        throw py::value_error("Given arrays must be of the same type.");
    }

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    int type_id = array_types.typenum_to_lookup_id(matrixA_typenum);

    gemv_impl_fn_ptr_t gemv_fn = gemv_dispatch_vector[type_id];
    if (gemv_fn == nullptr) {
        throw py::value_error(
            "Types of input arrays and result array are mismatched.");
    }

    char *a_typeless_ptr = matrixA.get_data();
    char *x_typeless_ptr = vectorX.get_data();
    char *y_typeless_ptr = vectorY.get_data();

    std::vector<py::ssize_t> x_stride = vectorX.get_strides_vector();
    std::vector<py::ssize_t> y_stride = vectorY.get_strides_vector();
    const int x_elemsize = vectorX.get_elemsize();
    const int y_elemsize = vectorY.get_elemsize();
    const std::int64_t incx = x_stride[0];
    const std::int64_t incy = y_stride[0];
    if (incx < 0) {
        x_typeless_ptr -= (x_shape[0] - 1) * std::abs(incx) * x_elemsize;
    }
    if (incy < 0) {
        y_typeless_ptr -= (y_shape[0] - 1) * std::abs(incy) * y_elemsize;
    }

    sycl::event gemv_ev =
        gemv_fn(exec_q, transA, m, n, a_typeless_ptr, lda, x_typeless_ptr, incx,
                y_typeless_ptr, incy, is_row_major, depends);

    sycl::event args_ev = dpctl::utils::keep_args_alive(
        exec_q, {matrixA, vectorX, vectorY}, {gemv_ev});

    return std::make_pair(args_ev, gemv_ev);
}

template <typename fnT, typename varT>
struct GemvContigFactory
{
    fnT get()
    {
        if constexpr (types::GemvTypePairSupportFactory<varT>::is_defined) {
            return gemv_impl<varT>;
        }
        else {
            return nullptr;
        }
    }
};

void init_gemv_dispatch_vector(void)
{
    dpctl_td_ns::DispatchVectorBuilder<gemv_impl_fn_ptr_t, GemvContigFactory,
                                       dpctl_td_ns::num_types>
        contig;
    contig.populate_dispatch_vector(gemv_dispatch_vector);
}
} // namespace blas
} // namespace ext
} // namespace backend
} // namespace dpnp
