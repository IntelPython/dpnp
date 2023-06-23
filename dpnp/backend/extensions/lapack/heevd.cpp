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

#include "heevd.hpp"
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

typedef sycl::event (*heevd_impl_fn_ptr_t)(sycl::queue,
                                           const oneapi::mkl::job,
                                           const oneapi::mkl::uplo,
                                           const std::int64_t,
                                           char *,
                                           char *,
                                           std::vector<sycl::event> &,
                                           const std::vector<sycl::event> &);

static heevd_impl_fn_ptr_t heevd_dispatch_table[dpctl_td_ns::num_types]
                                               [dpctl_td_ns::num_types];

template <typename T, typename RealT>
static sycl::event heevd_impl(sycl::queue exec_q,
                              const oneapi::mkl::job jobz,
                              const oneapi::mkl::uplo upper_lower,
                              const std::int64_t n,
                              char *in_a,
                              char *out_w,
                              std::vector<sycl::event> &host_task_events,
                              const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<T>(exec_q);
    type_utils::validate_type_for_device<RealT>(exec_q);

    T *a = reinterpret_cast<T *>(in_a);
    RealT *w = reinterpret_cast<RealT *>(out_w);

    const std::int64_t lda = std::max<size_t>(1UL, n);
    const std::int64_t scratchpad_size =
        mkl_lapack::heevd_scratchpad_size<T>(exec_q, jobz, upper_lower, n, lda);
    T *scratchpad = nullptr;

    std::stringstream error_msg;
    std::int64_t info = 0;

    sycl::event heevd_event;
    try {
        scratchpad = sycl::malloc_device<T>(scratchpad_size, exec_q);

        heevd_event = mkl_lapack::heevd(
            exec_q,
            jobz, // 'jobz == job::vec' means eigenvalues and eigenvectors are
                  // computed.
            upper_lower, // 'upper_lower == job::upper' means the upper
                         // triangular part of A, or the lower triangular
                         // otherwise
            n,           // The order of the matrix A (0 <= n)
            a, // Pointer to A, size (lda, *), where the 2nd dimension, must be
               // at least max(1, n) If 'jobz == job::vec', then on exit it will
               // contain the eigenvectors of A
            lda, // The leading dimension of a, must be at least max(1, n)
            w,   // Pointer to array of size at least n, it will contain the
                 // eigenvalues of A in ascending order
            scratchpad, // Pointer to scratchpad memory to be used by MKL
                        // routine for storing intermediate results
            scratchpad_size, depends);
    } catch (mkl_lapack::exception const &e) {
        error_msg
            << "Unexpected MKL exception caught during heevd() call:\nreason: "
            << e.what() << "\ninfo: " << e.info();
        info = e.info();
    } catch (sycl::exception const &e) {
        error_msg << "Unexpected SYCL exception caught during heevd() call:\n"
                  << e.what();
        info = -1;
    }

    if (info != 0) // an unexected error occurs
    {
        if (scratchpad != nullptr) {
            sycl::free(scratchpad, exec_q);
        }
        throw std::runtime_error(error_msg.str());
    }

    sycl::event clean_up_event = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(heevd_event);
        auto ctx = exec_q.get_context();
        cgh.host_task([ctx, scratchpad]() { sycl::free(scratchpad, ctx); });
    });
    host_task_events.push_back(clean_up_event);
    return heevd_event;
}

std::pair<sycl::event, sycl::event>
    heevd(sycl::queue exec_q,
          const std::int8_t jobz,
          const std::int8_t upper_lower,
          dpctl::tensor::usm_ndarray eig_vecs,
          dpctl::tensor::usm_ndarray eig_vals,
          const std::vector<sycl::event> &depends)
{
    const int eig_vecs_nd = eig_vecs.get_ndim();
    const int eig_vals_nd = eig_vals.get_ndim();

    if (eig_vecs_nd != 2) {
        throw py::value_error("Unexpected ndim=" + std::to_string(eig_vecs_nd) +
                              " of an output array with eigenvectors");
    }
    else if (eig_vals_nd != 1) {
        throw py::value_error("Unexpected ndim=" + std::to_string(eig_vals_nd) +
                              " of an output array with eigenvalues");
    }

    const py::ssize_t *eig_vecs_shape = eig_vecs.get_shape_raw();
    const py::ssize_t *eig_vals_shape = eig_vals.get_shape_raw();

    if (eig_vecs_shape[0] != eig_vecs_shape[1]) {
        throw py::value_error("Output array with eigenvectors with be square");
    }
    else if (eig_vecs_shape[0] != eig_vals_shape[0]) {
        throw py::value_error(
            "Eigenvectors and eigenvalues have different shapes");
    }

    size_t src_nelems(1);

    for (int i = 0; i < eig_vecs_nd; ++i) {
        src_nelems *= static_cast<size_t>(eig_vecs_shape[i]);
    }

    if (src_nelems == 0) {
        // nothing to do
        return std::make_pair(sycl::event(), sycl::event());
    }

    // check compatibility of execution queue and allocation queue
    if (!dpctl::utils::queues_are_compatible(exec_q, {eig_vecs, eig_vals})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(eig_vecs, eig_vals)) {
        throw py::value_error("Arrays with eigenvectors and eigenvalues are "
                              "overlapping segments of memory");
    }

    bool is_eig_vecs_f_contig = eig_vecs.is_f_contiguous();
    bool is_eig_vals_c_contig = eig_vals.is_c_contiguous();
    if (!is_eig_vecs_f_contig) {
        throw py::value_error("An array with input matrix / ouput eigenvectors "
                              "must be F-contiguous");
    }
    else if (!is_eig_vals_c_contig) {
        throw py::value_error(
            "An array with output eigenvalues must be C-contiguous");
    }

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    int eig_vecs_type_id =
        array_types.typenum_to_lookup_id(eig_vecs.get_typenum());
    int eig_vals_type_id =
        array_types.typenum_to_lookup_id(eig_vals.get_typenum());

    heevd_impl_fn_ptr_t heevd_fn =
        heevd_dispatch_table[eig_vecs_type_id][eig_vals_type_id];
    if (heevd_fn == nullptr) {
        throw py::value_error("No heevd implementation defined for a pair of "
                              "type for eigenvectors and eigenvalues");
    }

    char *eig_vecs_data = eig_vecs.get_data();
    char *eig_vals_data = eig_vals.get_data();

    const std::int64_t n = eig_vecs_shape[0];
    const oneapi::mkl::job jobz_val = static_cast<oneapi::mkl::job>(jobz);
    const oneapi::mkl::uplo uplo_val =
        static_cast<oneapi::mkl::uplo>(upper_lower);

    std::vector<sycl::event> host_task_events;
    sycl::event heevd_ev =
        heevd_fn(exec_q, jobz_val, uplo_val, n, eig_vecs_data, eig_vals_data,
                 host_task_events, depends);

    sycl::event args_ev = dpctl::utils::keep_args_alive(
        exec_q, {eig_vecs, eig_vals}, host_task_events);
    return std::make_pair(args_ev, heevd_ev);
}

template <typename fnT, typename T, typename RealT>
struct HeevdContigFactory
{
    fnT get()
    {
        if constexpr (types::HeevdTypePairSupportFactory<T, RealT>::is_defined)
        {
            return heevd_impl<T, RealT>;
        }
        else {
            return nullptr;
        }
    }
};

void init_heevd_dispatch_table(void)
{
    dpctl_td_ns::DispatchTableBuilder<heevd_impl_fn_ptr_t, HeevdContigFactory,
                                      dpctl_td_ns::num_types>
        contig;
    contig.populate_dispatch_table(heevd_dispatch_table);
}
} // namespace lapack
} // namespace ext
} // namespace backend
} // namespace dpnp
