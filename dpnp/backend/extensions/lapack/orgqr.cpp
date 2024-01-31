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
#include "utils/type_utils.hpp"

#include "orgqr.hpp"
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

typedef sycl::event (*orgqr_impl_fn_ptr_t)(sycl::queue,
                                           const std::int64_t,
                                           const std::int64_t,
                                           const std::int64_t,
                                           char *,
                                           std::int64_t,
                                           char *,
                                           std::vector<sycl::event> &,
                                           const std::vector<sycl::event> &);

static orgqr_impl_fn_ptr_t orgqr_dispatch_vector[dpctl_td_ns::num_types];

template <typename T>
static sycl::event orgqr_impl(sycl::queue exec_q,
                              const std::int64_t m,
                              const std::int64_t n,
                              const std::int64_t k,
                              char *in_a,
                              std::int64_t lda,
                              char *in_tau,
                              std::vector<sycl::event> &host_task_events,
                              const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<T>(exec_q);

    T *a = reinterpret_cast<T *>(in_a);
    T *tau = reinterpret_cast<T *>(in_tau);

    const std::int64_t scratchpad_size =
        oneapi::mkl::lapack::orgqr_scratchpad_size<T>(exec_q, m, n, k, lda);
    T *scratchpad = nullptr;

    std::stringstream error_msg;
    std::int64_t info = 0;
    bool is_exception_caught = false;

    sycl::event orgqr_event;
    try {
        scratchpad = sycl::malloc_device<T>(scratchpad_size, exec_q);

        orgqr_event = oneapi::mkl::lapack::orgqr(
            exec_q,
            m,          // The number of rows in the matrix; (0 ≤ m).
            n,          // The number of columns in the matrix; (0 ≤ n).
            k,          // The number of elementary reflectors
                        // whose product defines the matrix Q; (0 ≤ k ≤ n).
            a,          // Pointer to the m-by-n matrix.
            lda,        // The leading dimension of `a`; (1 ≤ m).
            tau,        // Pointer to the array of scalar factors of the
                        // elementary reflectors.
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
        else {
            error_msg << "Unexpected MKL exception caught during orgqr() "
                         "call:\nreason: "
                      << e.what() << "\ninfo: " << info;
        }
    } catch (sycl::exception const &e) {
        is_exception_caught = true;
        error_msg << "Unexpected SYCL exception caught during orfqr() call:\n"
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
        cgh.depends_on(orgqr_event);
        auto ctx = exec_q.get_context();
        cgh.host_task([ctx, scratchpad]() { sycl::free(scratchpad, ctx); });
    });
    host_task_events.push_back(clean_up_event);

    return orgqr_event;
}

std::pair<sycl::event, sycl::event>
    orgqr(sycl::queue q,
          const std::int64_t m,
          const std::int64_t n,
          const std::int64_t k,
          dpctl::tensor::usm_ndarray a_array,
          dpctl::tensor::usm_ndarray tau_array,
          const std::vector<sycl::event> &depends)
{
    const int a_array_nd = a_array.get_ndim();
    const int tau_array_nd = tau_array.get_ndim();

    if (a_array_nd != 2) {
        throw py::value_error(
            "The input array has ndim=" + std::to_string(a_array_nd) +
            ", but a 2-dimensional array is expected.");
    }

    if (tau_array_nd != 1) {
        throw py::value_error("The array of Householder scalars has ndim=" +
                              std::to_string(tau_array_nd) +
                              ", but a 1-dimensional array is expected.");
    }

    // check compatibility of execution queue and allocation queue
    if (!dpctl::utils::queues_are_compatible(q, {a_array, tau_array})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(a_array, tau_array)) {
        throw py::value_error(
            "The input array and the array of Householder scalars "
            "are overlapping segments of memory");
    }

    bool is_a_array_c_contig = a_array.is_c_contiguous();
    if (!is_a_array_c_contig) {
        throw py::value_error("The input array "
                              "must be C-contiguous");
    }

    bool is_tau_array_c_contig = tau_array.is_c_contiguous();
    bool is_tau_array_f_contig = tau_array.is_f_contiguous();

    if (!is_tau_array_c_contig || !is_tau_array_f_contig) {
        throw py::value_error("The array of Householder scalars "
                              "must be contiguous");
    }

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    int a_array_type_id =
        array_types.typenum_to_lookup_id(a_array.get_typenum());

    orgqr_impl_fn_ptr_t orgqr_fn = orgqr_dispatch_vector[a_array_type_id];
    if (orgqr_fn == nullptr) {
        throw py::value_error(
            "No orgqr implementation defined for the provided type "
            "of the input matrix.");
    }

    char *a_array_data = a_array.get_data();
    const std::int64_t lda = std::max<size_t>(1UL, m);

    char *tau_array_data = tau_array.get_data();

    std::vector<sycl::event> host_task_events;
    sycl::event orgqr_ev = orgqr_fn(q, m, n, k, a_array_data, lda,
                                    tau_array_data, host_task_events, depends);

    sycl::event args_ev = dpctl::utils::keep_args_alive(q, {a_array, tau_array},
                                                        host_task_events);

    return std::make_pair(args_ev, orgqr_ev);
}

template <typename fnT, typename T>
struct OrgqrContigFactory
{
    fnT get()
    {
        if constexpr (types::OrgqrTypePairSupportFactory<T>::is_defined) {
            return orgqr_impl<T>;
        }
        else {
            return nullptr;
        }
    }
};

void init_orgqr_dispatch_vector(void)
{
    dpctl_td_ns::DispatchVectorBuilder<orgqr_impl_fn_ptr_t, OrgqrContigFactory,
                                       dpctl_td_ns::num_types>
        contig;
    contig.populate_dispatch_vector(orgqr_dispatch_vector);
}
} // namespace lapack
} // namespace ext
} // namespace backend
} // namespace dpnp
