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

#include <sstream>
#include <stdexcept>

#include <pybind11/pybind11.h>

// dpnp extension infrastructure
#include "ext/common.hpp"

// dpctl tensor validation and utility headers
#include "utils/memory_overlap.hpp"
#include "utils/output_validation.hpp"
#include "utils/type_utils.hpp"

#include "gemv.hpp"
#include "types_matrix.hpp"

namespace mkl_sparse = oneapi::mkl::sparse;
namespace py         = pybind11;
namespace type_utils = dpctl::tensor::type_utils;

using ext::common::init_dispatch_table;

namespace dpnp::extensions::sparse
{

// ---------------------------------------------------------------------------
// Dispatch table: [value_type_id][index_type_id] -> impl function pointer
// Mirrors the 2-D table pattern of blas/gemm.cpp.
// ---------------------------------------------------------------------------

typedef sycl::event (*gemv_impl_fn_ptr_t)(
    sycl::queue &,
    oneapi::mkl::transpose,
    double,                        // alpha (always passed as double; cast inside)
    const char *,                  // row_ptr  (typeless)
    const char *,                  // col_ind  (typeless)
    const char *,                  // values   (typeless)
    std::int64_t,                  // num_rows
    std::int64_t,                  // num_cols
    std::int64_t,                  // nnz
    const char *,                  // x        (typeless)
    double,                        // beta     (always passed as double; cast inside)
    char *,                        // y        (typeless, writable)
    const std::vector<sycl::event> &);

static gemv_impl_fn_ptr_t
    gemv_dispatch_table[dpctl_td_ns::num_types][dpctl_td_ns::num_types];


// ---------------------------------------------------------------------------
// Typed implementation — one instantiation per (Tv, Ti) pair
// ---------------------------------------------------------------------------

template <typename Tv, typename Ti>
static sycl::event
gemv_impl(sycl::queue                      &exec_q,
          oneapi::mkl::transpose            mkl_trans,
          double                            alpha_d,
          const char                       *row_ptr_data,
          const char                       *col_ind_data,
          const char                       *values_data,
          std::int64_t                      num_rows,
          std::int64_t                      num_cols,
          std::int64_t                      nnz,
          const char                       *x_data,
          double                            beta_d,
          char                             *y_data,
          const std::vector<sycl::event>   &depends)
{
    type_utils::validate_type_for_device<Tv>(exec_q);

    const Tv  alpha = static_cast<Tv>(alpha_d);
    const Tv  beta  = static_cast<Tv>(beta_d);
    const Ti *row_ptr = reinterpret_cast<const Ti *>(row_ptr_data);
    const Ti *col_ind = reinterpret_cast<const Ti *>(col_ind_data);
    const Tv *values  = reinterpret_cast<const Tv *>(values_data);
    const Tv *x       = reinterpret_cast<const Tv *>(x_data);
    Tv       *y       = reinterpret_cast<Tv *>(y_data);

    std::stringstream error_msg;
    bool is_exception_caught = false;

    mkl_sparse::matrix_handle_t spmat = nullptr;
    sycl::event gemv_ev;

    try {
        mkl_sparse::init_matrix_handle(&spmat);

        // oneMKL 2025-2 API: set_csr_data now requires explicit nnz and uses
        // `spmat` nomenclature. The old form without nnz is deprecated.
        auto ev_set = mkl_sparse::set_csr_data(
            exec_q, spmat,
            num_rows, num_cols, nnz,
            oneapi::mkl::index_base::zero,
            const_cast<Ti *>(row_ptr),
            const_cast<Ti *>(col_ind),
            const_cast<Tv *>(values),
            depends);

        auto ev_opt = mkl_sparse::optimize_gemv(
            exec_q, mkl_trans, spmat, {ev_set});

        gemv_ev = mkl_sparse::gemv(
            exec_q, mkl_trans,
            alpha, spmat,
            x, beta, y,
            {ev_opt});

        mkl_sparse::release_matrix_handle(exec_q, &spmat, {gemv_ev});

    } catch (oneapi::mkl::exception const &e) {
        error_msg << "Unexpected MKL exception caught during sparse_gemv() "
                     "call:\nreason: " << e.what();
        is_exception_caught = true;
    } catch (sycl::exception const &e) {
        error_msg << "Unexpected SYCL exception caught during sparse_gemv() "
                     "call:\n" << e.what();
        is_exception_caught = true;
    }

    if (is_exception_caught) {
        if (spmat != nullptr)
            mkl_sparse::release_matrix_handle(exec_q, &spmat, {});
        throw std::runtime_error(error_msg.str());
    }

    return gemv_ev;
}


// ---------------------------------------------------------------------------
// Python-facing entry point
// ---------------------------------------------------------------------------

std::pair<sycl::event, sycl::event>
sparse_gemv(sycl::queue                           &exec_q,
            const int                              trans,
            const double                           alpha,
            const dpctl::tensor::usm_ndarray      &row_ptr,
            const dpctl::tensor::usm_ndarray      &col_ind,
            const dpctl::tensor::usm_ndarray      &values,
            const dpctl::tensor::usm_ndarray      &x,
            const double                           beta,
            const dpctl::tensor::usm_ndarray      &y,
            const std::int64_t                     num_rows,
            const std::int64_t                     num_cols,
            const std::int64_t                     nnz,
            const std::vector<sycl::event>        &depends)
{
    // 1. ndim checks
    if (x.get_ndim() != 1)
        throw py::value_error("sparse_gemv: x must be a 1-D array.");
    if (y.get_ndim() != 1)
        throw py::value_error("sparse_gemv: y must be a 1-D array.");

    // 2. Queue compatibility
    if (!dpctl::utils::queues_are_compatible(
            exec_q, {row_ptr.get_queue(), col_ind.get_queue(),
                     values.get_queue(), x.get_queue(), y.get_queue()}))
        throw py::value_error(
            "sparse_gemv: USM allocations are not compatible with the "
            "execution queue.");

    // 3. Memory overlap: x and y must not alias
    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(x, y))
        throw py::value_error(
            "sparse_gemv: input array x and output array y are overlapping "
            "segments of memory.");

    // 4. Output writability and size
    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(y);
    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(
        y, static_cast<std::size_t>(num_rows));

    // 5. Map trans integer to oneMKL enum
    oneapi::mkl::transpose mkl_trans;
    switch (trans) {
        case 0: mkl_trans = oneapi::mkl::transpose::nontrans;  break;
        case 1: mkl_trans = oneapi::mkl::transpose::trans;     break;
        case 2: mkl_trans = oneapi::mkl::transpose::conjtrans; break;
        default:
            throw std::invalid_argument(
                "sparse_gemv: trans must be 0 (N), 1 (T), or 2 (C)");
    }

    // 6. Dispatch table lookup
    auto array_types = dpctl_td_ns::usm_ndarray_types();
    const int val_id = array_types.typenum_to_lookup_id(values.get_typenum());
    const int idx_id = array_types.typenum_to_lookup_id(row_ptr.get_typenum());

    gemv_impl_fn_ptr_t gemv_fn = gemv_dispatch_table[val_id][idx_id];
    if (gemv_fn == nullptr)
        throw py::value_error(
            "sparse_gemv: no implementation for the given value/index dtype "
            "combination. Supported: float32/float64 with int32/int64 indices.");

    sycl::event gemv_ev =
        gemv_fn(exec_q, mkl_trans, alpha,
                row_ptr.get_data(), col_ind.get_data(), values.get_data(),
                num_rows, num_cols, nnz,
                x.get_data(), beta, y.get_data(),
                depends);

    sycl::event args_ev = dpctl::utils::keep_args_alive(
        exec_q, {row_ptr, col_ind, values, x, y}, {gemv_ev});

    return std::make_pair(args_ev, gemv_ev);
}


// ---------------------------------------------------------------------------
// Factory and dispatch table initialisation
// ---------------------------------------------------------------------------

template <typename fnT, typename Tv, typename Ti>
struct GemvContigFactory
{
    fnT get()
    {
        if constexpr (types::SparseGemvTypePairSupportFactory<Tv, Ti>::is_defined)
            return gemv_impl<Tv, Ti>;
        else
            return nullptr;
    }
};

void init_sparse_gemv_dispatch_table(void)
{
    init_dispatch_table<gemv_impl_fn_ptr_t, GemvContigFactory>(
        gemv_dispatch_table);
}

} // namespace dpnp::extensions::sparse
