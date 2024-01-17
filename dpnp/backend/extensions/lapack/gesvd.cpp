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

#include "gesvd.hpp"
#include "types_matrix.hpp"

#include "dpnp_utils.hpp"

namespace dpnp
{
namespace backend
{
namespace ext
{
namespace lapack
{
namespace mkl_lapack = oneapi::mkl::lapack;
namespace py = pybind11;
namespace type_utils = dpctl::tensor::type_utils;

typedef sycl::event (*gesvd_impl_fn_ptr_t)(sycl::queue,
                                           const oneapi::mkl::jobsvd,
                                           const oneapi::mkl::jobsvd,
                                           const std::int64_t,
                                           const std::int64_t,
                                           char *,
                                           const std::int64_t,
                                           char *,
                                           char *,
                                           const std::int64_t,
                                           char *,
                                           const std::int64_t,
                                           std::vector<sycl::event> &,
                                           const std::vector<sycl::event> &);

static gesvd_impl_fn_ptr_t gesvd_dispatch_table[dpctl_td_ns::num_types]
                                               [dpctl_td_ns::num_types];

// Converts a given character code (ord) to the corresponding
// oneapi::mkl::jobsvd enumeration value
static oneapi::mkl::jobsvd process_job(std::int8_t job_val)
{
    switch (job_val) {
    case 'A':
        return oneapi::mkl::jobsvd::vectors;
    case 'S':
        return oneapi::mkl::jobsvd::somevec;
    case 'O':
        return oneapi::mkl::jobsvd::vectorsina;
    case 'N':
        return oneapi::mkl::jobsvd::novec;
    default:
        throw std::invalid_argument("Unknown value for job");
    }
}

template <typename T, typename RealT>
static sycl::event gesvd_impl(sycl::queue exec_q,
                              const oneapi::mkl::jobsvd jobu,
                              const oneapi::mkl::jobsvd jobvt,
                              const std::int64_t m,
                              const std::int64_t n,
                              char *in_a,
                              const std::int64_t lda,
                              char *out_s,
                              char *out_u,
                              const std::int64_t ldu,
                              char *out_vt,
                              const std::int64_t ldvt,
                              std::vector<sycl::event> &host_task_events,
                              const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<T>(exec_q);
    type_utils::validate_type_for_device<RealT>(exec_q);

    T *a = reinterpret_cast<T *>(in_a);
    RealT *s = reinterpret_cast<RealT *>(out_s);
    T *u = reinterpret_cast<T *>(out_u);
    T *vt = reinterpret_cast<T *>(out_vt);

    const std::int64_t scratchpad_size = mkl_lapack::gesvd_scratchpad_size<T>(
        exec_q, jobu, jobvt, m, n, lda, ldu, ldvt);
    T *scratchpad = nullptr;

    std::stringstream error_msg;
    std::int64_t info = 0;
    bool is_exception_caught = false;

    sycl::event gesvd_event;
    try {
        scratchpad = sycl::malloc_device<T>(scratchpad_size, exec_q);

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
            m,     // The number of rows in the input matrix A (0 <= m).
            n,     // The number of columns in the input matrix A (0 <= n).
            a,     // Pointer to the input matrix A of size (m x n).
            lda,   // The leading dimension of A, must be at least max(1, m).
            s,     // Pointer to the array containing the singular values.
            u,   // Pointer to the matrix U in the singular value decomposition.
            ldu, // The leading dimension of U, must be at least max(1, m).
            vt, // Pointer to the matrix VT in the singular value decomposition.
            ldvt, // The leading dimension of VT, must be at least max(1, n).
            scratchpad, // Pointer to scratchpad memory to be used by MKL
                        // routine for storing intermediate results.
            scratchpad_size, depends);
    } catch (mkl_lapack::exception const &e) {
        is_exception_caught = true;
        info = e.info();
        if (info < 0) {
            error_msg << "Parameter number " << -info
                      << " had an illegal value.";
        }
        else if (info == scratchpad_size && e.detail() != 0) {
            error_msg
                << "Insufficient scratchpad size. Required size is at least "
                << e.detail();
        }
        else if (info > 0) {
            error_msg << "The algorithm computing SVD failed to converge; "
                      << info << " off-diagonal elements of an intermediate "
                      << "bidiagonal form did not converge to zero.\n";
        }
        else {
            error_msg << "Unexpected MKL exception caught during gesvd() "
                         "call:\nreason: "
                      << e.what() << "\ninfo: " << e.info();
        }
    } catch (sycl::exception const &e) {
        is_exception_caught = true;
        error_msg << "Unexpected SYCL exception caught during gesvd() call:\n"
                  << e.what();
    }

    if (is_exception_caught) // an unexpected error occurs
    {
        if (scratchpad != nullptr) {
            sycl::free(scratchpad, exec_q);
        }
        throw std::runtime_error(error_msg.str());
    }

    sycl::event clean_up_event = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(gesvd_event);
        auto ctx = exec_q.get_context();
        cgh.host_task([ctx, scratchpad]() { sycl::free(scratchpad, ctx); });
    });
    host_task_events.push_back(clean_up_event);
    return gesvd_event;
}

std::pair<sycl::event, sycl::event>
    gesvd(sycl::queue exec_q,
          const std::int8_t jobu_val,
          const std::int8_t jobvt_val,
          dpctl::tensor::usm_ndarray a_array,
          dpctl::tensor::usm_ndarray out_s,
          dpctl::tensor::usm_ndarray out_u,
          dpctl::tensor::usm_ndarray out_vt,
          const std::vector<sycl::event> &depends)
{
    const int a_array_nd = a_array.get_ndim();

    if (a_array_nd != 2) {
        throw py::value_error(
            "The input array has ndim=" + std::to_string(a_array_nd) +
            ", but a 2-dimensional array is expected.");
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

    bool is_a_array_c_contig = a_array.is_c_contiguous();
    if (!is_a_array_c_contig) {
        throw py::value_error("The input array must be C-contiguous");
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

    gesvd_impl_fn_ptr_t gesvd_fn =
        gesvd_dispatch_table[a_array_type_id][out_s_type_id];
    if (gesvd_fn == nullptr) {
        throw py::value_error(
            "No gesvd implementation is defined for the given pair "
            "of array type and output singular values type.");
    }

    char *a_array_data = a_array.get_data();
    char *out_s_data = out_s.get_data();
    char *out_u_data = out_u.get_data();
    char *out_vt_data = out_vt.get_data();

    const py::ssize_t *a_array_shape = a_array.get_shape_raw();
    const std::int64_t n = a_array_shape[0];
    const std::int64_t m = a_array_shape[1];

    const std::int64_t lda = std::max<size_t>(1UL, m);
    const std::int64_t ldu = std::max<size_t>(1UL, m);
    const std::int64_t ldvt = std::max<size_t>(1UL, n);

    const oneapi::mkl::jobsvd jobu = process_job(jobu_val);
    const oneapi::mkl::jobsvd jobvt = process_job(jobvt_val);

    std::vector<sycl::event> host_task_events;
    sycl::event gesvd_ev =
        gesvd_fn(exec_q, jobu, jobvt, m, n, a_array_data, lda, out_s_data,
                 out_u_data, ldu, out_vt_data, ldvt, host_task_events, depends);

    sycl::event args_ev = dpctl::utils::keep_args_alive(
        exec_q, {a_array, out_s, out_u, out_vt}, host_task_events);

    return std::make_pair(args_ev, gesvd_ev);
}

template <typename fnT, typename T, typename RealT>
struct GesvdContigFactory
{
    fnT get()
    {
        if constexpr (types::GesvdTypePairSupportFactory<T, RealT>::is_defined)
        {
            return gesvd_impl<T, RealT>;
        }
        else {
            return nullptr;
        }
    }
};

void init_gesvd_dispatch_table(void)
{
    dpctl_td_ns::DispatchTableBuilder<gesvd_impl_fn_ptr_t, GesvdContigFactory,
                                      dpctl_td_ns::num_types>
        contig;
    contig.populate_dispatch_table(gesvd_dispatch_table);
}
} // namespace lapack
} // namespace ext
} // namespace backend
} // namespace dpnp
