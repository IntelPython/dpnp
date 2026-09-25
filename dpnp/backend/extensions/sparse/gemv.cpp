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
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//*****************************************************************************

#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>

// utils extension header
#include "ext/common.hpp"

// dpnp tensor headers
#include "utils/memory_overlap.hpp"
#include "utils/output_validation.hpp"
#include "utils/type_utils.hpp"

#include "gemv.hpp"
#include "types_matrix.hpp"

namespace dpnp::extensions::sparse
{

#if defined(USE_ONEMATH)
namespace mkl = oneapi::math;
namespace mkl_sparse = oneapi::math::sparse;
#else
namespace mkl = oneapi::mkl;
namespace mkl_sparse = oneapi::mkl::sparse;
#endif

namespace py = pybind11;
namespace type_utils = dpnp::tensor::type_utils;

using ext::common::init_dispatch_table;

// ---------------------------------------------------------------------------
// Dispatch table types
// ---------------------------------------------------------------------------

/**
 * init_impl: builds the sparse matrix handle from the CSR arrays.
 * Returns (handle_ptr, event)
 */
typedef std::pair<std::uintptr_t, sycl::event> (*gemv_init_fn_ptr_t)(
    sycl::queue &,
    mkl::transpose,
    const char *,       // row_ptr (typeless)
    const char *,       // col_ind (typeless)
    const char *,       // values  (typeless)
    const std::int64_t, // num_rows
    const std::int64_t, // num_cols
    const std::int64_t, // nnz
    const std::vector<sycl::event> &);

/**
 * compute_impl: fires a single sparse matrix-vector product using a
 * pre-built handle. Returns the kernel event directly -- no host_task
 * wrapping.
 */
typedef sycl::event (*gemv_compute_fn_ptr_t)(
    sycl::queue &,
    std::uintptr_t, // pre-built handle (matrix_handle_t or cache ptr)
    mkl::transpose,
    const double,       // alpha (cast to Tv inside)
    const char *,       // x (typeless)
    const double,       // beta  (cast to Tv inside)
    char *,             // y (typeless, writable)
    const std::int64_t, // op_rows (length of y)
    const std::int64_t, // op_cols (length of x)
    const std::vector<sycl::event> &);

// Init dispatch: 2-D on (Tv, Ti).
static gemv_init_fn_ptr_t gemv_init_dispatch_table[dpnp_td_ns::num_types]
                                                  [dpnp_td_ns::num_types];

// Compute dispatch: 1-D on Tv. The index type is baked into the handle,
// so compute doesn't need it.
static gemv_compute_fn_ptr_t gemv_compute_dispatch_table[dpnp_td_ns::num_types];

#if defined(USE_ONEMATH)

// ---------------------------------------------------------------------------
// oneMath sparse API (v2) cache
// ---------------------------------------------------------------------------

/**
 * Owns the five oneMath objects an spmv needs (matrix handle, x / y
 * dense-vector handles, descriptor, workspace), so a single cached
 * handle can drive repeated matvecs. init returns its address as the
 * opaque uintptr_t handle; release frees the workspace and deletes it.
 * spmv_buffer_size + spmv_optimize run once on the first compute;
 * later calls only rebind the x / y data (see gemv_compute_impl).
 */
struct SpmvCache
{
    mkl_sparse::matrix_handle_t A = nullptr;
    mkl_sparse::dense_vector_handle_t x = nullptr;
    mkl_sparse::dense_vector_handle_t y = nullptr;
    mkl_sparse::spmv_descr_t descr = nullptr;
    void *workspace = nullptr;
    mkl_sparse::matrix_view view{};
    bool optimized = false;
};

// ---------------------------------------------------------------------------
// Per-type init implementation (oneMath)
// ---------------------------------------------------------------------------

template <typename Tv, typename Ti>
static std::pair<std::uintptr_t, sycl::event>
    gemv_init_impl(sycl::queue &exec_q,
                   mkl::transpose mkl_trans,
                   const char *row_ptr_data,
                   const char *col_ind_data,
                   const char *values_data,
                   std::int64_t num_rows,
                   std::int64_t num_cols,
                   std::int64_t nnz,
                   const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<Tv>(exec_q);

    // init_csr_matrix has no dependency-list overload in the USM API;
    // the caller-supplied depends are honoured at the first compute
    // (spmv_optimize / spmv accept them).
    static_cast<void>(depends);

    Ti *row_ptr = const_cast<Ti *>(reinterpret_cast<const Ti *>(row_ptr_data));
    Ti *col_ind = const_cast<Ti *>(reinterpret_cast<const Ti *>(col_ind_data));
    Tv *values = const_cast<Tv *>(reinterpret_cast<const Tv *>(values_data));

    // op(A) is (num_rows x num_cols) for trans=N, transposed otherwise;
    // x has op_cols elements, y has op_rows.
    const bool is_non_trans = (mkl_trans == mkl::transpose::nontrans);
    const std::int64_t op_rows = is_non_trans ? num_rows : num_cols;
    const std::int64_t op_cols = is_non_trans ? num_cols : num_rows;

    auto *cache = new SpmvCache;

    // Release whatever was created before the failure, then drop the
    // cache, so a throwing init does not leak oneMath handles.
    auto cleanup_partial = [&]() {
        if (cache->descr != nullptr)
            mkl_sparse::release_spmv_descr(exec_q, cache->descr, {});
        if (cache->x != nullptr)
            mkl_sparse::release_dense_vector(exec_q, cache->x, {});
        if (cache->y != nullptr)
            mkl_sparse::release_dense_vector(exec_q, cache->y, {});
        if (cache->A != nullptr)
            mkl_sparse::release_sparse_matrix(exec_q, cache->A, {});
        delete cache;
    };

    try {
        mkl_sparse::init_csr_matrix(exec_q, &cache->A, num_rows, num_cols, nnz,
                                    mkl::index_base::zero, row_ptr, col_ind,
                                    values);

        // values is a placeholder pointer; the real x / y pointers are
        // bound on every compute call via set_dense_vector_data.
        mkl_sparse::init_dense_vector(exec_q, &cache->x, op_cols, values);
        mkl_sparse::init_dense_vector(exec_q, &cache->y, op_rows, values);

        mkl_sparse::init_spmv_descr(exec_q, &cache->descr);
    } catch (mkl::exception const &e) {
        cleanup_partial();
        throw std::runtime_error(
            std::string("sparse_gemv_init: oneMath exception in init: ") +
            e.what());
    } catch (sycl::exception const &e) {
        cleanup_partial();
        throw std::runtime_error(
            std::string("sparse_gemv_init: SYCL exception in init: ") +
            e.what());
    }

    auto handle_ptr = reinterpret_cast<std::uintptr_t>(cache);
    // No optimize event yet -- optimization is deferred to first compute.
    // Return a completed event so the caller's wait() is a no-op.
    return {handle_ptr, sycl::event{}};
}

#else // legacy oneMKL sparse API (v1)

// ---------------------------------------------------------------------------
// Per-type init implementation (oneMKL)
// ---------------------------------------------------------------------------

template <typename Tv, typename Ti>
static std::pair<std::uintptr_t, sycl::event>
    gemv_init_impl(sycl::queue &exec_q,
                   mkl::transpose mkl_trans,
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
    const Tv *values = reinterpret_cast<const Tv *>(values_data);

    mkl_sparse::matrix_handle_t spmat = nullptr;
    try {
        mkl_sparse::init_matrix_handle(&spmat);
    } catch (mkl::exception const &e) {
        throw std::runtime_error(
            std::string(
                "sparse_gemv_init: MKL exception in init_matrix_handle: ") +
            e.what());
    } catch (sycl::exception const &e) {
        throw std::runtime_error(
            std::string(
                "sparse_gemv_init: SYCL exception in init_matrix_handle: ") +
            e.what());
    }

    sycl::event ev_set;
    try {
        ev_set = mkl_sparse::set_csr_data(
            exec_q, spmat, num_rows, num_cols, nnz, mkl::index_base::zero,
            const_cast<Ti *>(row_ptr), const_cast<Ti *>(col_ind),
            const_cast<Tv *>(values), depends);
    } catch (mkl::exception const &e) {
        mkl_sparse::release_matrix_handle(exec_q, &spmat, {}).wait();
        throw std::runtime_error(
            std::string("sparse_gemv_init: MKL exception in set_csr_data: ") +
            e.what());
    } catch (sycl::exception const &e) {
        mkl_sparse::release_matrix_handle(exec_q, &spmat, {}).wait();
        throw std::runtime_error(
            std::string("sparse_gemv_init: SYCL exception in set_csr_data: ") +
            e.what());
    }

    sycl::event ev_opt;
    try {
        ev_opt = mkl_sparse::optimize_gemv(exec_q, mkl_trans, spmat, {ev_set});
    } catch (mkl::exception const &e) {
        mkl_sparse::release_matrix_handle(exec_q, &spmat, {ev_set}).wait();
        throw std::runtime_error(
            std::string("sparse_gemv_init: MKL exception in optimize_gemv: ") +
            e.what());
    } catch (sycl::exception const &e) {
        mkl_sparse::release_matrix_handle(exec_q, &spmat, {ev_set}).wait();
        throw std::runtime_error(
            std::string("sparse_gemv_init: SYCL exception in optimize_gemv: ") +
            e.what());
    }

    auto handle_ptr = reinterpret_cast<std::uintptr_t>(spmat);
    return {handle_ptr, ev_opt};
}

#endif // USE_ONEMATH

// ---------------------------------------------------------------------------
// Per-type compute implementation
// ---------------------------------------------------------------------------

#if defined(USE_ONEMATH)

template <typename Tv>
static sycl::event gemv_compute_impl(sycl::queue &exec_q,
                                     std::uintptr_t handle_ptr,
                                     mkl::transpose mkl_trans,
                                     double alpha_d,
                                     const char *x_data,
                                     double beta_d,
                                     char *y_data,
                                     std::int64_t op_rows,
                                     std::int64_t op_cols,
                                     const std::vector<sycl::event> &depends)
{
    auto *cache = reinterpret_cast<SpmvCache *>(handle_ptr);

    const Tv alpha = static_cast<Tv>(alpha_d);
    const Tv beta = static_cast<Tv>(beta_d);

    Tv *x = const_cast<Tv *>(reinterpret_cast<const Tv *>(x_data));
    Tv *y = reinterpret_cast<Tv *>(y_data);

    try {
        // The spec permits resetting x / y data (and alpha / beta)
        // before each spmv without re-optimizing, as long as the
        // handles passed to spmv match those passed to spmv_optimize.
        mkl_sparse::set_dense_vector_data(exec_q, cache->x, op_cols, x);
        mkl_sparse::set_dense_vector_data(exec_q, cache->y, op_rows, y);

        constexpr auto alg = mkl_sparse::spmv_alg::default_alg;

        if (!cache->optimized) {
            // spmv_buffer_size + spmv_optimize must each run at least
            // once before spmv; do so on the first matvec only.
            std::size_t workspace_bytes = 0;
            mkl_sparse::spmv_buffer_size(exec_q, mkl_trans, &alpha, cache->view,
                                         cache->A, cache->x, &beta, cache->y,
                                         alg, cache->descr, workspace_bytes);
            if (workspace_bytes > 0) {
                cache->workspace = sycl::malloc_device(workspace_bytes, exec_q);
                if (cache->workspace == nullptr)
                    throw std::runtime_error(
                        "sparse_gemv_compute: failed to allocate spmv "
                        "workspace.");
            }

            sycl::event ev_opt = mkl_sparse::spmv_optimize(
                exec_q, mkl_trans, &alpha, cache->view, cache->A, cache->x,
                &beta, cache->y, alg, cache->descr, cache->workspace, depends);
            cache->optimized = true;

            return mkl_sparse::spmv(exec_q, mkl_trans, &alpha, cache->view,
                                    cache->A, cache->x, &beta, cache->y, alg,
                                    cache->descr, {ev_opt});
        }

        return mkl_sparse::spmv(exec_q, mkl_trans, &alpha, cache->view,
                                cache->A, cache->x, &beta, cache->y, alg,
                                cache->descr, depends);
    } catch (mkl::exception const &e) {
        throw std::runtime_error(
            std::string("sparse_gemv_compute: oneMath exception: ") + e.what());
    } catch (sycl::exception const &e) {
        throw std::runtime_error(
            std::string("sparse_gemv_compute: SYCL exception: ") + e.what());
    }
}

#else // legacy oneMKL sparse API (v1)

template <typename Tv>
static sycl::event gemv_compute_impl(sycl::queue &exec_q,
                                     std::uintptr_t handle_ptr,
                                     mkl::transpose mkl_trans,
                                     double alpha_d,
                                     const char *x_data,
                                     double beta_d,
                                     char *y_data,
                                     std::int64_t op_rows,
                                     std::int64_t op_cols,
                                     const std::vector<sycl::event> &depends)
{
    // op_rows / op_cols are unused here (the handle encodes the
    // dimensions); kept for ABI parity with the oneMath path.
    static_cast<void>(op_rows);
    static_cast<void>(op_cols);

    auto spmat = reinterpret_cast<mkl_sparse::matrix_handle_t>(handle_ptr);

    const Tv alpha = static_cast<Tv>(alpha_d);
    const Tv beta = static_cast<Tv>(beta_d);

    const Tv *x = reinterpret_cast<const Tv *>(x_data);
    Tv *y = reinterpret_cast<Tv *>(y_data);

    try {
        return mkl_sparse::gemv(exec_q, mkl_trans, alpha, spmat, x, beta, y,
                                depends);
    } catch (mkl::exception const &e) {
        throw std::runtime_error(
            std::string("sparse_gemv_compute: MKL exception: ") + e.what());
    } catch (sycl::exception const &e) {
        throw std::runtime_error(
            std::string("sparse_gemv_compute: SYCL exception: ") + e.what());
    }
}

#endif // USE_ONEMATH

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

static mkl::transpose decode_trans(const int trans)
{
    switch (trans) {
    case 0:
        return mkl::transpose::nontrans;
    case 1:
        return mkl::transpose::trans;
    case 2:
        return mkl::transpose::conjtrans;
    default:
        throw std::invalid_argument(
            "sparse_gemv: trans must be 0 (N), 1 (T), or 2 (C)");
    }
}

std::tuple<std::uintptr_t, int, sycl::event>
    sparse_gemv_init(sycl::queue &exec_q,
                     const int trans,
                     const dpnp::tensor::usm_ndarray &row_ptr,
                     const dpnp::tensor::usm_ndarray &col_ind,
                     const dpnp::tensor::usm_ndarray &values,
                     const std::int64_t num_rows,
                     const std::int64_t num_cols,
                     const std::int64_t nnz,
                     const std::vector<sycl::event> &depends)
{
    if (!dpctl::utils::queues_are_compatible(
            exec_q,
            {row_ptr.get_queue(), col_ind.get_queue(), values.get_queue()}))
        throw py::value_error(
            "sparse_gemv_init: USM allocations are not compatible with the "
            "execution queue.");

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

    auto array_types = dpnp_td_ns::usm_ndarray_types();
    const int val_id = array_types.typenum_to_lookup_id(values.get_typenum());
    const int idx_id = array_types.typenum_to_lookup_id(row_ptr.get_typenum());

    gemv_init_fn_ptr_t init_fn = gemv_init_dispatch_table[val_id][idx_id];
    if (init_fn == nullptr)
        throw py::value_error(
            "sparse_gemv_init: no implementation for the given value/index "
            "dtype combination. Supported: {float32,float64,complex64,"
            "complex128} x {int32,int64}.");

    auto [handle_ptr, ev_opt] =
        init_fn(exec_q, mkl_trans, row_ptr.get_data(), col_ind.get_data(),
                values.get_data(), num_rows, num_cols, nnz, depends);

    return {handle_ptr, val_id, ev_opt};
}

std::pair<sycl::event, sycl::event>
    sparse_gemv_compute(sycl::queue &exec_q,
                        const std::uintptr_t handle_ptr,
                        const int val_type_id,
                        const int trans,
                        const double alpha,
                        const dpnp::tensor::usm_ndarray &x,
                        const double beta,
                        const dpnp::tensor::usm_ndarray &y,
                        const std::int64_t num_rows,
                        const std::int64_t num_cols,
                        const std::vector<sycl::event> &depends)
{
    if (x.get_ndim() != 1)
        throw py::value_error("sparse_gemv_compute: x must be a 1-D array.");
    if (y.get_ndim() != 1)
        throw py::value_error("sparse_gemv_compute: y must be a 1-D array.");

    if (!dpctl::utils::queues_are_compatible(exec_q,
                                             {x.get_queue(), y.get_queue()}))
        throw py::value_error(
            "sparse_gemv_compute: USM allocations are not compatible with the "
            "execution queue.");

    auto const &overlap = dpnp::tensor::overlap::MemoryOverlap();
    if (overlap(x, y))
        throw py::value_error(
            "sparse_gemv_compute: x and y are overlapping memory segments.");

    dpnp::tensor::validation::CheckWritable::throw_if_not_writable(y);

    // Shape validation: op(A) is (num_rows, num_cols) for trans=N,
    // (num_cols, num_rows) for trans={T,C}.
    auto mkl_trans = decode_trans(trans);
    const bool is_non_trans = (mkl_trans == mkl::transpose::nontrans);
    const std::int64_t op_rows = is_non_trans ? num_rows : num_cols;
    const std::int64_t op_cols = is_non_trans ? num_cols : num_rows;

    if (x.get_shape(0) != op_cols)
        throw py::value_error(
            "sparse_gemv_compute: x length does not match operator columns.");
    if (y.get_shape(0) != op_rows)
        throw py::value_error(
            "sparse_gemv_compute: y length does not match operator rows.");

    dpnp::tensor::validation::AmpleMemory::throw_if_not_ample(
        y, static_cast<std::size_t>(op_rows));

    auto array_types = dpnp_td_ns::usm_ndarray_types();
    const int x_val_id = array_types.typenum_to_lookup_id(x.get_typenum());
    const int y_val_id = array_types.typenum_to_lookup_id(y.get_typenum());

    if (x_val_id != val_type_id || y_val_id != val_type_id)
        throw py::value_error(
            "sparse_gemv_compute: x and y dtype must match the value dtype "
            "of the sparse matrix used to build the handle.");

    if (val_type_id < 0 || val_type_id >= dpnp_td_ns::num_types)
        throw py::value_error("sparse_gemv_compute: val_type_id out of range.");

    gemv_compute_fn_ptr_t compute_fn = gemv_compute_dispatch_table[val_type_id];

    if (compute_fn == nullptr)
        throw py::value_error("sparse_gemv_compute: unsupported value dtype.");

    sycl::event gemv_ev =
        compute_fn(exec_q, handle_ptr, mkl_trans, alpha, x.get_data(), beta,
                   const_cast<char *>(y.get_data()), op_rows, op_cols, depends);

    sycl::event args_ev =
        dpnp::utils::keep_args_alive(exec_q, {x, y}, {gemv_ev});

    return std::make_pair(args_ev, gemv_ev);
}

#if defined(USE_ONEMATH)

sycl::event sparse_gemv_release(sycl::queue &exec_q,
                                const std::uintptr_t handle_ptr,
                                const std::vector<sycl::event> &depends)
{
    auto *cache = reinterpret_cast<SpmvCache *>(handle_ptr);
    if (cache == nullptr)
        return sycl::event{};

    // Release every owned oneMath object; each takes `depends` so it
    // waits for pending compute before freeing.
    std::vector<sycl::event> rel_evs;
    rel_evs.push_back(
        mkl_sparse::release_spmv_descr(exec_q, cache->descr, depends));
    rel_evs.push_back(
        mkl_sparse::release_dense_vector(exec_q, cache->x, depends));
    rel_evs.push_back(
        mkl_sparse::release_dense_vector(exec_q, cache->y, depends));
    rel_evs.push_back(
        mkl_sparse::release_sparse_matrix(exec_q, cache->A, depends));

    // Free the USM workspace and delete the cache only after all
    // releases complete (the spec forbids freeing the workspace before
    // the spmv using it has finished). A host_task orders this without
    // blocking the caller.
    sycl::event cleanup_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(rel_evs);
        cgh.host_task([cache, exec_q]() {
            if (cache->workspace != nullptr)
                sycl::free(cache->workspace, exec_q);
            delete cache;
        });
    });

    return cleanup_ev;
}

#else // legacy oneMKL sparse API (v1)

sycl::event sparse_gemv_release(sycl::queue &exec_q,
                                const std::uintptr_t handle_ptr,
                                const std::vector<sycl::event> &depends)
{
    auto spmat = reinterpret_cast<mkl_sparse::matrix_handle_t>(handle_ptr);

    // release_matrix_handle takes `depends` so the handle is not freed
    // until pending compute on it completes.
    sycl::event release_ev =
        mkl_sparse::release_matrix_handle(exec_q, &spmat, depends);

    return release_ev;
}

#endif // USE_ONEMATH

// ---------------------------------------------------------------------------
// Dispatch table factories and registration
// ---------------------------------------------------------------------------

template <typename fnT, typename Tv, typename Ti>
struct GemvInitContigFactory
{
    fnT get()
    {
        if constexpr (types::SparseGemvInitTypePairSupportFactory<
                          Tv, Ti>::is_defined)
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
        if constexpr (types::SparseGemvComputeTypeSupportFactory<
                          Tv>::is_defined)
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
    dpnp_td_ns::DispatchVectorBuilder<
        gemv_compute_fn_ptr_t, GemvComputeContigFactory, dpnp_td_ns::num_types>
        builder;
    builder.populate_dispatch_vector(gemv_compute_dispatch_table);
}

} // namespace dpnp::extensions::sparse
