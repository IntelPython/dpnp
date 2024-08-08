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

#include "common_helpers.hpp"
#include "gesvd.hpp"
#include "gesvd_common_utils.hpp"
#include "types_matrix.hpp"

#include "dpnp_utils.hpp"

namespace dpnp::extensions::lapack
{
namespace mkl_lapack = oneapi::mkl::lapack;
namespace py = pybind11;
namespace type_utils = dpctl::tensor::type_utils;

typedef sycl::event (*gesvd_batch_impl_fn_ptr_t)(
    sycl::queue &,
    const oneapi::mkl::jobsvd,
    const oneapi::mkl::jobsvd,
    const std::int64_t,
    const std::int64_t,
    const std::int64_t,
    char *,
    const std::int64_t,
    char *,
    char *,
    const std::int64_t,
    char *,
    const std::int64_t,
    const std::vector<sycl::event> &);

static gesvd_batch_impl_fn_ptr_t
    gesvd_batch_dispatch_table[dpctl_td_ns::num_types][dpctl_td_ns::num_types];

template <typename T, typename RealT>
static sycl::event gesvd_batch_impl(sycl::queue &exec_q,
                                    const oneapi::mkl::jobsvd jobu,
                                    const oneapi::mkl::jobsvd jobvt,
                                    const std::int64_t m,
                                    const std::int64_t n,
                                    const std::int64_t batch_size,
                                    char *in_a,
                                    const std::int64_t lda,
                                    char *out_s,
                                    char *out_u,
                                    const std::int64_t ldu,
                                    char *out_vt,
                                    const std::int64_t ldvt,
                                    const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<T>(exec_q);
    type_utils::validate_type_for_device<RealT>(exec_q);

    T *a = reinterpret_cast<T *>(in_a);
    RealT *s = reinterpret_cast<RealT *>(out_s);
    T *u = reinterpret_cast<T *>(out_u);
    T *vt = reinterpret_cast<T *>(out_vt);

    const std::int64_t k = std::min(m, n);

    const std::int64_t a_size = m * n;
    const std::int64_t s_size = k;

    std::int64_t u_size = 0;
    std::int64_t vt_size = 0;

    if (jobu == oneapi::mkl::jobsvd::somevec ||
        jobu == oneapi::mkl::jobsvd::vectorsina)
    {
        u_size = m * k;
        vt_size = k * n;
    }
    else if (jobu == oneapi::mkl::jobsvd::vectors) {
        u_size = m * m;
        vt_size = n * n;
    }
    else if (jobu == oneapi::mkl::jobsvd::novec) {
        u_size = 0;
        vt_size = 0;
    }

    // Get the number of independent linear streams
    const std::int64_t n_linear_streams =
        (batch_size > 16) ? 4 : ((batch_size > 4 ? 2 : 1));

    const std::int64_t scratchpad_size = mkl_lapack::gesvd_scratchpad_size<T>(
        exec_q, jobu, jobvt, m, n, lda, ldu, ldvt);

    T *scratchpad = helper::alloc_scratchpad_batch<T>(scratchpad_size,
                                                      n_linear_streams, exec_q);

    // Computation events to manage dependencies for each linear stream
    std::vector<std::vector<sycl::event>> comp_evs(n_linear_streams, depends);

    std::stringstream error_msg;
    bool is_exception_caught = false;

    for (std::int64_t batch_id = 0; batch_id < batch_size; ++batch_id) {

        T *a_batch = a + batch_id * a_size;
        T *u_batch = u + batch_id * u_size;
        RealT *s_batch = s + batch_id * s_size;
        T *vt_batch = vt + batch_id * vt_size;

        std::int64_t stream_id = (batch_id % n_linear_streams);

        T *current_scratch_gesvd = scratchpad + stream_id * scratchpad_size;

        // Get the event dependencies for the current stream
        const auto &current_dep = comp_evs[stream_id];

        sycl::event gesvd_event;
        try {
            gesvd_event = mkl_lapack::gesvd(
                exec_q,
                jobu,  // Character specifying how to compute the matrix U:
                       // 'A' computes all columns of U,
                       // 'S' computes the first min(m,n) columns of U,
                       // 'O' overwrites A with the columns of U,
                       // 'N' does not compute U.
                jobvt, // Character specifying how to compute the matrix VT:
                       // 'A' computes all rows of VT,
                       // 'S' computes the first min(m,n) rows of VT,
                       // 'O' overwrites A with the rows of VT,
                       // 'N' does not compute VT.
                m, // The number of rows in the input batch matrix A (0 <= m).
                n, // The number of columns in the input batch matrix A (0 <=
                   // n).
                a_batch, // Pointer to the input batch matrix A of size (m x n)
                         // for the current batch.
                lda, // The leading dimension of A, must be at least max(1, m).
                s_batch, // Pointer to the array containing the singular values
                         // for the current batch.
                u_batch, // Pointer to the matrix U in the singular value
                         // decomposition for the current batch.
                ldu, // The leading dimension of U, must be at least max(1, m).
                vt_batch, // Pointer to the matrix VT in the singular value
                          // decomposition for the current batch.
                ldvt, // The leading dimension of VT, must be at least max(1,
                      // n).
                current_scratch_gesvd, // Pointer to scratchpad memory to be
                                       // used by MKL routine for storing
                                       // intermediate results.
                scratchpad_size, current_dep);
        } catch (mkl_lapack::exception const &e) {
            is_exception_caught = true;
            gesvd_utils::handle_lapack_exc(scratchpad_size, e, error_msg);
        } catch (sycl::exception const &e) {
            is_exception_caught = true;
            error_msg
                << "Unexpected SYCL exception caught during gesvd() call:\n"
                << e.what();
        }

        // Update the event dependencies for the current stream
        comp_evs[stream_id] = {gesvd_event};
    }

    if (is_exception_caught) // an unexpected error occurs
    {
        if (scratchpad != nullptr) {
            sycl::free(scratchpad, exec_q);
        }
        throw std::runtime_error(error_msg.str());
    }

    sycl::event ht_ev = exec_q.submit([&](sycl::handler &cgh) {
        for (const auto &ev : comp_evs) {
            cgh.depends_on(ev);
        }
        auto ctx = exec_q.get_context();
        cgh.host_task([ctx, scratchpad]() { sycl::free(scratchpad, ctx); });
    });

    return ht_ev;
}

std::pair<sycl::event, sycl::event>
    gesvd_batch(sycl::queue &exec_q,
                const std::int8_t jobu_val,
                const std::int8_t jobvt_val,
                dpctl::tensor::usm_ndarray a_array,
                dpctl::tensor::usm_ndarray out_s,
                dpctl::tensor::usm_ndarray out_u,
                dpctl::tensor::usm_ndarray out_vt,
                const std::vector<sycl::event> &depends)
{
    const int a_array_nd = a_array.get_ndim();
    const int out_u_array_nd = out_u.get_ndim();
    const int out_s_array_nd = out_s.get_ndim();
    const int out_vt_array_nd = out_vt.get_ndim();

    if (a_array_nd != 3) {
        throw py::value_error(
            "The input array has ndim=" + std::to_string(a_array_nd) +
            ", but a 3-dimensional array is expected.");
    }

    if (out_s_array_nd != 2) {
        throw py::value_error("The output array of singular values has ndim=" +
                              std::to_string(out_s_array_nd) +
                              ", but a 2-dimensional array is expected.");
    }

    if (jobu_val == 'N' && jobvt_val == 'N') {
        if (out_u_array_nd != 0) {
            throw py::value_error(
                "The output array of the left singular vectors has ndim=" +
                std::to_string(out_u_array_nd) +
                ", but it is not used and should have ndim=0.");
        }
        if (out_vt_array_nd != 0) {
            throw py::value_error(
                "The output array of the right singular vectors has ndim=" +
                std::to_string(out_vt_array_nd) +
                ", but it is not used and should have ndim=0.");
        }
    }
    else {
        if (out_u_array_nd != 3) {
            throw py::value_error(
                "The output array of the left singular vectors has ndim=" +
                std::to_string(out_u_array_nd) +
                ", but a 3-dimensional array is expected.");
        }
        if (out_vt_array_nd != 3) {
            throw py::value_error(
                "The output array of the right singular vectors has ndim=" +
                std::to_string(out_vt_array_nd) +
                ", but a 3-dimensional array is expected.");
        }
    }

    // check compatibility of execution queue and allocation queue
    if (!dpctl::utils::queues_are_compatible(
            exec_q, {a_array.get_queue(), out_s.get_queue(), out_u.get_queue(),
                     out_vt.get_queue()}))
    {
        throw std::runtime_error(
            "USM allocations are not compatible with the execution queue.");
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(a_array, out_s) || overlap(a_array, out_u) ||
        overlap(a_array, out_vt) || overlap(out_s, out_u) ||
        overlap(out_s, out_vt) || overlap(out_u, out_vt))
    {
        throw py::value_error("Arrays have overlapping segments of memory");
    }

    bool is_a_array_f_contig = a_array.is_f_contiguous();
    if (!is_a_array_f_contig) {
        throw py::value_error("The input array must be F-contiguous");
    }

    bool is_out_u_array_f_contig = out_u.is_f_contiguous();
    bool is_out_vt_array_f_contig = out_vt.is_f_contiguous();

    if (!is_out_u_array_f_contig || !is_out_vt_array_f_contig) {
        throw py::value_error("The output arrays of the left and right "
                              "singular vectors must be F-contiguous");
    }

    bool is_out_s_array_c_contig = out_s.is_c_contiguous();

    if (!is_out_s_array_c_contig) {
        throw py::value_error("The output array of singular values "
                              "must be C-contiguous");
    }

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    int a_array_type_id =
        array_types.typenum_to_lookup_id(a_array.get_typenum());
    int out_u_type_id = array_types.typenum_to_lookup_id(out_u.get_typenum());
    int out_s_type_id = array_types.typenum_to_lookup_id(out_s.get_typenum());
    int out_vt_type_id = array_types.typenum_to_lookup_id(out_vt.get_typenum());

    if (a_array_type_id != out_u_type_id || a_array_type_id != out_vt_type_id) {
        throw py::type_error(
            "Input array, output left singular vectors array, "
            "and outpuy right singular vectors array must have "
            "the same data type");
    }

    gesvd_batch_impl_fn_ptr_t gesvd_batch_fn =
        gesvd_batch_dispatch_table[a_array_type_id][out_s_type_id];
    if (gesvd_batch_fn == nullptr) {
        throw py::value_error(
            "No gesvd implementation is defined for the given pair "
            "of array type and output singular values type.");
    }

    char *a_array_data = a_array.get_data();
    char *out_s_data = out_s.get_data();
    char *out_u_data = out_u.get_data();
    char *out_vt_data = out_vt.get_data();

    const py::ssize_t *a_array_shape = a_array.get_shape_raw();

    // Input array have (m, n, batch_size) shape
    const std::int64_t batch_size = a_array_shape[2];
    const std::int64_t m = a_array_shape[0];
    const std::int64_t n = a_array_shape[1];

    const std::int64_t lda = std::max<size_t>(1UL, m);
    const std::int64_t ldu = std::max<size_t>(1UL, m);
    const std::int64_t ldvt =
        std::max<std::size_t>(1UL, jobvt_val == 'S' ? (m > n ? n : m) : n);

    const oneapi::mkl::jobsvd jobu = gesvd_utils::process_job(jobu_val);
    const oneapi::mkl::jobsvd jobvt = gesvd_utils::process_job(jobvt_val);

    sycl::event gesvd_ev =
        gesvd_batch_fn(exec_q, jobu, jobvt, m, n, batch_size, a_array_data, lda,
                       out_s_data, out_u_data, ldu, out_vt_data, ldvt, depends);

    sycl::event ht_ev = dpctl::utils::keep_args_alive(
        exec_q, {a_array, out_s, out_u, out_vt}, {gesvd_ev});

    return std::make_pair(ht_ev, gesvd_ev);
}

template <typename fnT, typename T, typename RealT>
struct GesvdBatchContigFactory
{
    fnT get()
    {
        if constexpr (types::GesvdTypePairSupportFactory<T, RealT>::is_defined)
        {
            return gesvd_batch_impl<T, RealT>;
        }
        else {
            return nullptr;
        }
    }
};

void init_gesvd_batch_dispatch_table(void)
{
    dpctl_td_ns::DispatchTableBuilder<gesvd_batch_impl_fn_ptr_t,
                                      GesvdBatchContigFactory,
                                      dpctl_td_ns::num_types>
        contig;
    contig.populate_dispatch_table(gesvd_batch_dispatch_table);
}
} // namespace dpnp::extensions::lapack
