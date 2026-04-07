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
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//*****************************************************************************

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>

// utils extension header
#include "ext/common.hpp"

// dpctl tensor headers
#include "utils/memory_overlap.hpp"
#include "utils/output_validation.hpp"
#include "utils/type_utils.hpp"

#include "gemv.hpp"
#include "types_matrix.hpp"

namespace dpnp::extensions::sparse
{

namespace mkl_sparse = oneapi::mkl::sparse;
namespace py         = pybind11;
namespace type_utils = dpctl::tensor::type_utils;

using ext::common::init_dispatch_table;

// ---------------------------------------------------------------------------
// Dispatch table types
// ---------------------------------------------------------------------------

/**
 * init_impl: builds the matrix_handle, calls set_csr_data + optimize_gemv.
 * Returns (handle_ptr, optimize_event).
 * All CSR arrays are *not* copied -- they must stay alive until release.
 */
typedef std::pair<std::uintptr_t, sycl::event> (*gemv_init_fn_ptr_t)(
    sycl::queue &,
    oneapi::mkl::transpose,
    const char *,   // row_ptr (typeless)
    const char *,   // col_ind (typeless)
    const char *,   // values  (typeless)
    std::int64_t,   // num_rows
    std::int64_t,   // num_cols
    std::int64_t,   // nnz
    const std::vector<sycl::event> &);

/**
 * compute_impl: fires sparse::gemv using a pre-built handle.
 * Returns the gemv event directly -- no host_task wrapping.
 */
typedef sycl::event (*gemv_compute_fn_ptr_t)(
    sycl::queue &,
    oneapi::mkl::sparse::matrix_handle_t,
    oneapi::mkl::transpose,
    double,         // alpha (cast to Tv inside)
    const char *,   // x (typeless)
    double,         // beta  (cast to Tv inside)
    char *,         // y (typeless, writable)
    const std::vector<sycl::event> &);

// Init dispatch: 2-D on (Tv, Ti).
static gemv_init_fn_ptr_t
    gemv_init_dispatch_table[dpctl_td_ns::num_types][dpctl_td_ns::num_types];

// Compute dispatch: 1-D on Tv. The index type is baked into the handle,
// so compute doesn't need it.
static gemv_compute_fn_ptr_t
    gemv_compute_dispatch_table[dpctl_td_ns::num_types];

// ---------------------------------------------------------------------------
// Per-type init implementation
// ---------------------------------------------------------------------------

template <typename Tv, typename Ti>
static std::pair<std::uintptr_t, sycl::event>
gemv_init_impl(sycl::queue &exec_q,
               oneapi::mkl::transpose mkl_trans,
               const char *row_ptr_data,
               const char *col_ind_data,
               const char *values_data,
               std::int64_t num_rows,
               std::int64_t num_cols,
               std::int64_t nnz,
               const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<Tv>(exec_q);

    const Ti *row_ptr = reinterpret_cast<const Ti *>(row_ptr_data);
    const Ti *col_ind = reinterpret_cast<const Ti *>(col_ind_data);
    const Tv *values  = reinterpret_cast<const Tv *>(values_data);

    mkl_sparse::matrix_handle_t spmat = nullptr;
    mkl_sparse::init_matrix_handle(&spmat);

    auto ev_set = mkl_sparse::set_csr_data(
        exec_q, spmat,
        num_rows, num_cols, nnz,
        oneapi::mkl::index_base::zero,
        const_cast<Ti *>(row_ptr),
        const_cast<Ti *>(col_ind),
        const_cast<Tv *>(values),
        depends);

    sycl::event ev_opt;
    try {
        ev_opt = mkl_sparse::optimize_gemv(
            exec_q, mkl_trans, spmat, {ev_set});
    } catch (oneapi::mkl::exception const &e) {
        mkl_sparse::release_matrix_handle(exec_q, &spmat, {});
        throw std::runtime_error(
            std::string("sparse_gemv_init: MKL exception in optimize_gemv: ")
            + e.what());
    } catch (sycl::exception const &e) {
        mkl_sparse::release_matrix_handle(exec_q, &spmat, {});
        throw std::runtime_error(
            std::string("sparse_gemv_init: SYCL exception in optimize_gemv: ")
            + e.what());
    }

    auto handle_ptr = reinterpret_cast<std::uintptr_t>(spmat);
    return {handle_ptr, ev_opt};
}

// ---------------------------------------------------------------------------
// Per-type compute implementation
// ---------------------------------------------------------------------------

template <typename Tv>
static sycl::event
gemv_compute_impl(sycl::queue &exec_q,
                  mkl_sparse::matrix_handle_t spmat,
                  oneapi::mkl::transpose mkl_trans,
                  double alpha_d,
                  const char *x_data,
                  double beta_d,
                  char *y_data,
                  const std::vector<sycl::event> &depends)
{
    // For complex Tv the single-arg constructor sets imag to zero.
    // Solvers use alpha=1, beta=0 so this is exact; other callers
    // passing complex scalars via this path will lose the imag
    // component silently.
    const Tv alpha = static_cast<Tv>(alpha_d);
    const Tv beta  = static_cast<Tv>(beta_d);

    const Tv *x = reinterpret_cast<const Tv *>(x_data);
    Tv *y       = reinterpret_cast<Tv *>(y_data);

    try {
        return mkl_sparse::gemv(
            exec_q, mkl_trans,
            alpha, spmat,
            x, beta, y,
            depends);
    } catch (oneapi::mkl::exception const &e) {
        throw std::runtime_error(
            std::string("sparse_gemv_compute: MKL exception: ") + e.what());
    } catch (sycl::exception const &e) {
        throw std::runtime_error(
            std::string("sparse_gemv_compute: SYCL exception: ") + e.what());
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

static oneapi::mkl::transpose
decode_trans(const int trans)
{
    switch (trans) {
        case 0: return oneapi::mkl::transpose::nontrans;
        case 1: return oneapi::mkl::transpose::trans;
        case 2: return oneapi::mkl::transpose::conjtrans;
        default:
            throw std::invalid_argument(
                "sparse_gemv: trans must be 0 (N), 1 (T), or 2 (C)");
    }
}

std::tuple<std::uintptr_t, int, sycl::event>
sparse_gemv_init(sycl::queue &exec_q,
                 const int trans,
                 const dpctl::tensor::usm_ndarray &row_ptr,
                 const dpctl::tensor::usm_ndarray &col_ind,
                 const dpctl::tensor::usm_ndarray &values,
                 const std::int64_t num_rows,
                 const std::int64_t num_cols,
                 const std::int64_t nnz,
                 const std::vector<sycl::event> &depends)
{
    if (!dpctl::utils::queues_are_compatible(
            exec_q, {row_ptr.get_queue(), col_ind.get_queue(),
                     values.get_queue()}))
        throw py::value_error(
            "sparse_gemv_init: USM allocations are not compatible with the "
            "execution queue.");

    // Basic CSR shape sanity.
    if (row_ptr.get_ndim() != 1 || col_ind.get_ndim() != 1 ||
        values.get_ndim() != 1)
        throw py::value_error(
            "sparse_gemv_init: row_ptr, col_ind, values must all be 1-D.");

    if (row_ptr.get_shape(0) != num_rows + 1)
        throw py::value_error(
            "sparse_gemv_init: row_ptr length must equal num_rows + 1.");

    if (col_ind.get_shape(0) != nnz || values.get_shape(0) != nnz)
        throw py::value_error(
            "sparse_gemv_init: col_ind and values length must equal nnz.");

    // Index types of row_ptr and col_ind must match.
    if (row_ptr.get_typenum() != col_ind.get_typenum())
        throw py::value_error(
            "sparse_gemv_init: row_ptr and col_ind must have the same dtype.");

    auto mkl_trans = decode_trans(trans);

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    const int val_id = array_types.typenum_to_lookup_id(values.get_typenum());
    const int idx_id = array_types.typenum_to_lookup_id(row_ptr.get_typenum());

    gemv_init_fn_ptr_t init_fn = gemv_init_dispatch_table[val_id][idx_id];
    if (init_fn == nullptr)
        throw py::value_error(
            "sparse_gemv_init: no implementation for the given value/index "
            "dtype combination. Supported: {float32,float64,complex64,"
            "complex128} x {int32,int64}.");

    auto [handle_ptr, ev_opt] = init_fn(
        exec_q, mkl_trans,
        row_ptr.get_data(), col_ind.get_data(), values.get_data(),
        num_rows, num_cols, nnz, depends);

    return {handle_ptr, val_id, ev_opt};
}

sycl::event
sparse_gemv_compute(sycl::queue &exec_q,
                    const std::uintptr_t handle_ptr,
                    const int val_type_id,
                    const int trans,
                    const double alpha,
                    const dpctl::tensor::usm_ndarray &x,
                    const double beta,
                    const dpctl::tensor::usm_ndarray &y,
                    const std::int64_t num_rows,
                    const std::int64_t num_cols,
                    const std::vector<sycl::event> &depends)
{
    if (x.get_ndim() != 1)
        throw py::value_error("sparse_gemv_compute: x must be a 1-D array.");
    if (y.get_ndim() != 1)
        throw py::value_error("sparse_gemv_compute: y must be a 1-D array.");

    if (!dpctl::utils::queues_are_compatible(
            exec_q, {x.get_queue(), y.get_queue()}))
        throw py::value_error(
            "sparse_gemv_compute: USM allocations are not compatible with the "
            "execution queue.");

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(x, y))
        throw py::value_error(
            "sparse_gemv_compute: x and y are overlapping memory segments.");

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(y);

    // Shape validation: op(A) is (num_rows, num_cols) for trans=N,
    // (num_cols, num_rows) for trans={T,C}.
    auto mkl_trans = decode_trans(trans);
    const bool is_non_trans =
        (mkl_trans == oneapi::mkl::transpose::nontrans);
    const std::int64_t op_rows = is_non_trans ? num_rows : num_cols;
    const std::int64_t op_cols = is_non_trans ? num_cols : num_rows;

    if (x.get_shape(0) != op_cols)
        throw py::value_error(
            "sparse_gemv_compute: x length does not match operator columns.");
    if (y.get_shape(0) != op_rows)
        throw py::value_error(
            "sparse_gemv_compute: y length does not match operator rows.");

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(
        y, static_cast<std::size_t>(op_rows));

    // Dtype verification: x, y, and the handle's value type must all match.
    auto array_types = dpctl_td_ns::usm_ndarray_types();
    const int x_val_id = array_types.typenum_to_lookup_id(x.get_typenum());
    const int y_val_id = array_types.typenum_to_lookup_id(y.get_typenum());

    if (x_val_id != val_type_id || y_val_id != val_type_id)
        throw py::value_error(
            "sparse_gemv_compute: x and y dtype must match the value dtype "
            "of the sparse matrix used to build the handle.");

    if (val_type_id < 0 || val_type_id >= dpctl_td_ns::num_types)
        throw py::value_error(
            "sparse_gemv_compute: val_type_id out of range.");

    gemv_compute_fn_ptr_t compute_fn =
        gemv_compute_dispatch_table[val_type_id];

    if (compute_fn == nullptr)
        throw py::value_error(
            "sparse_gemv_compute: unsupported value dtype.");

    auto spmat = reinterpret_cast<mkl_sparse::matrix_handle_t>(handle_ptr);

    return compute_fn(exec_q, spmat, mkl_trans, alpha,
                      x.get_data(), beta,
                      const_cast<char *>(y.get_data()),
                      depends);
}

sycl::event
sparse_gemv_release(sycl::queue &exec_q,
                    const std::uintptr_t handle_ptr,
                    const std::vector<sycl::event> &depends)
{
    auto spmat = reinterpret_cast<mkl_sparse::matrix_handle_t>(handle_ptr);

    // release_matrix_handle takes `depends` so it will not free the handle
    // until all pending compute work on it has completed. In recent oneMKL
    // versions release_matrix_handle returns a sycl::event; older versions
    // returned void. If your pinned oneMKL returns void, replace the body
    // with:
    //     mkl_sparse::release_matrix_handle(exec_q, &spmat, depends);
    //     return exec_q.submit([&](sycl::handler &cgh) {
    //         cgh.depends_on(depends);
    //         cgh.host_task([]() {});
    //     });
    sycl::event release_ev =
        mkl_sparse::release_matrix_handle(exec_q, &spmat, depends);

    return release_ev;
}

// ---------------------------------------------------------------------------
// Dispatch table factories and registration
// ---------------------------------------------------------------------------

template <typename fnT, typename Tv, typename Ti>
struct GemvInitContigFactory
{
    fnT get()
    {
        if constexpr (types::SparseGemvInitTypePairSupportFactory<Tv, Ti>::is_defined)
            return gemv_init_impl<Tv, Ti>;
        else
            return nullptr;
    }
};

template <typename fnT, typename Tv>
struct GemvComputeContigFactory
{
    fnT get()
    {
        if constexpr (types::SparseGemvComputeTypeSupportFactory<Tv>::is_defined)
            return gemv_compute_impl<Tv>;
        else
            return nullptr;
    }
};

void init_sparse_gemv_dispatch_tables(void)
{
    // 2-D table on (Tv, Ti) for init.
    init_dispatch_table<gemv_init_fn_ptr_t, GemvInitContigFactory>(
        gemv_init_dispatch_table);

    // 1-D table on Tv for compute. dpctl's type dispatch headers expose
    // DispatchVectorBuilder as the 1-D analogue of DispatchTableBuilder.
    dpctl_td_ns::DispatchVectorBuilder
        gemv_compute_fn_ptr_t,
        GemvComputeContigFactory,
        dpctl_td_ns::num_types>
        builder;
    builder.populate_dispatch_vector(gemv_compute_dispatch_table);
}

} // namespace dpnp::extensions::sparse
