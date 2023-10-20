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
const oneapi::mkl::jobsvd process_job(std::int8_t job_val)
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

    const std::int64_t scratchpad_size =
        oneapi::mkl::lapack::gesvd_scratchpad_size<T>(exec_q, jobu, jobvt, m, n,
                                                      lda, ldu, ldvt);
    T *scratchpad = nullptr;

    std::stringstream error_msg;
    std::int64_t info = 0;

    sycl::event gesvd_event;
    try {
        scratchpad = sycl::malloc_device<T>(scratchpad_size, exec_q);

        gesvd_event = oneapi::mkl::lapack::gesvd(
            exec_q, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, scratchpad,
            scratchpad_size, depends);
    } catch (mkl_lapack::exception const &e) {
        error_msg
            << "Unexpected MKL exception caught during gesvd() call:\nreason: "
            << e.what() << "\ninfo: " << e.info();
        info = e.info();
    } catch (sycl::exception const &e) {
        error_msg << "Unexpected SYCL exception caught during gesvd() call:\n"
                  << e.what();
        info = -1;
    }

    if (info != 0) // an unexpected error occurs
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

sycl::event gesvd(sycl::queue exec_q,
                  const std::int8_t jobu_val,
                  const std::int8_t jobvt_val,
                  const std::int64_t m,
                  const std::int64_t n,
                  dpctl::tensor::usm_ndarray a_array,
                  dpctl::tensor::usm_ndarray out_s,
                  dpctl::tensor::usm_ndarray out_u,
                  dpctl::tensor::usm_ndarray out_vt,
                  const std::vector<sycl::event> &depends)
{
    // check ndim if required

    // check compatibility of execution queue and allocation queue
    if (!dpctl::utils::queues_are_compatible(
            exec_q, {a_array.get_queue(), out_s.get_queue(), out_u.get_queue(),
                     out_vt.get_queue()}))
    {
        throw std::runtime_error(
            "USM allocations are not compatible with the execution queue.");
    }

    // check shape if required

    // auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    // if (overlap(eig_vecs, eig_vals)) {
    //     throw py::value_error("Arrays with eigenvectors and eigenvalues are "
    //                           "overlapping segments of memory");
    // }

    // need to add the check equality of types a, out_vt, out_u

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    int a_array_type_id =
        array_types.typenum_to_lookup_id(a_array.get_typenum());
    int out_s_type_id = array_types.typenum_to_lookup_id(out_s.get_typenum());

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

    const int a_array_nd = a_array.get_ndim();

    const std::int64_t lda = std::max<size_t>(1UL, m);
    const std::int64_t ldu = std::max<size_t>(1UL, m);
    const std::int64_t ldvt = std::max<size_t>(1UL, n);

    const oneapi::mkl::jobsvd jobu = process_job(jobu_val);
    const oneapi::mkl::jobsvd jobvt = process_job(jobvt_val);

    std::vector<sycl::event> host_task_events;
    sycl::event gesvd_ev =
        gesvd_fn(exec_q, jobu, jobvt, m, n, a_array_data, lda, out_s_data,
                 out_u_data, ldu, out_vt_data, ldvt, host_task_events, depends);

    return gesvd_ev;
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
