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

#include "geqrf.hpp"
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

typedef sycl::event (*geqrf_batch_impl_fn_ptr_t)(
    sycl::queue,
    std::int64_t,
    std::int64_t,
    char *,
    std::int64_t,
    std::int64_t,
    char *,
    std::int64_t,
    std::int64_t,
    std::vector<sycl::event> &,
    const std::vector<sycl::event> &);

static geqrf_batch_impl_fn_ptr_t
    geqrf_batch_dispatch_vector[dpctl_td_ns::num_types];

template <typename T>
static sycl::event geqrf_batch_impl(sycl::queue exec_q,
                                    std::int64_t m,
                                    std::int64_t n,
                                    char *in_a,
                                    std::int64_t lda,
                                    std::int64_t stride_a,
                                    char *in_tau,
                                    std::int64_t stride_tau,
                                    std::int64_t batch_size,
                                    std::vector<sycl::event> &host_task_events,
                                    const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<T>(exec_q);

    T *a = reinterpret_cast<T *>(in_a);
    T *tau = reinterpret_cast<T *>(in_tau);

    const std::int64_t scratchpad_size =
        mkl_lapack::geqrf_batch_scratchpad_size<T>(exec_q, m, n, lda, stride_a,
                                                   stride_tau, batch_size);
    T *scratchpad = nullptr;

    std::stringstream error_msg;
    std::int64_t info = 0;
    bool is_exception_caught = false;

    sycl::event geqrf_batch_event;
    try {
        scratchpad = sycl::malloc_device<T>(scratchpad_size, exec_q);

        geqrf_batch_event = mkl_lapack::geqrf_batch(
            exec_q,
            m, // The number of rows in each matrix in the batch; (0 ≤ m).
               // It must be a non-negative integer.
            n, // The number of columns in each matrix in the batch; (0 ≤ n).
               // It must be a non-negative integer.
            a, // Pointer to the batch of matrices, each of size (m x n).
            lda,      // The leading dimension of each matrix in the batch.
                      // For row major layout, lda ≥ max(1, m).
            stride_a, // Stride between consecutive matrices in the batch.
            tau, // Pointer to the array of scalar factors of the elementary
                 // reflectors for each matrix in the batch.
            stride_tau, // Stride between arrays of scalar factors in the batch.
            batch_size, // The number of matrices in the batch.
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
                << e.detail() << ", but current size is " << scratchpad_size
                << ".";
        }
        else {
            error_msg << "Unexpected MKL exception caught during geqrf_batch() "
                         "call:\nreason: "
                      << e.what() << "\ninfo: " << e.info();
        }
    } catch (sycl::exception const &e) {
        is_exception_caught = true;
        error_msg
            << "Unexpected SYCL exception caught during geqrf_batch() call:\n"
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
        cgh.depends_on(geqrf_batch_event);
        auto ctx = exec_q.get_context();
        cgh.host_task([ctx, scratchpad]() { sycl::free(scratchpad, ctx); });
    });
    host_task_events.push_back(clean_up_event);
    return geqrf_batch_event;
}

std::pair<sycl::event, sycl::event>
    geqrf_batch(sycl::queue q,
                dpctl::tensor::usm_ndarray a_array,
                dpctl::tensor::usm_ndarray tau_array,
                std::int64_t m,
                std::int64_t n,
                std::int64_t stride_a,
                std::int64_t stride_tau,
                std::int64_t batch_size,
                const std::vector<sycl::event> &depends)
{
    const int a_array_nd = a_array.get_ndim();
    const int tau_array_nd = tau_array.get_ndim();

    if (a_array_nd < 3) {
        throw py::value_error(
            "The input array has ndim=" + std::to_string(a_array_nd) +
            ", but an array with ndim >= 3 is expected.");
    }

    if (tau_array_nd != 2) {
        throw py::value_error("The array of Householder scalars has ndim=" +
                              std::to_string(tau_array_nd) +
                              ", but a 2-dimensional array is expected.");
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
    bool is_tau_array_c_contig = tau_array.is_c_contiguous();
    if (!is_a_array_c_contig) {
        throw py::value_error("The input array "
                              "must be C-contiguous");
    }
    if (!is_tau_array_c_contig) {
        throw py::value_error("The array of Householder scalars "
                              "must be C-contiguous");
    }

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    int a_array_type_id =
        array_types.typenum_to_lookup_id(a_array.get_typenum());
    int tau_array_type_id =
        array_types.typenum_to_lookup_id(tau_array.get_typenum());

    if (a_array_type_id != tau_array_type_id) {
        throw py::value_error(
            "The types of the input array and "
            "the array of Householder scalars are mismatched");
    }

    geqrf_batch_impl_fn_ptr_t geqrf_batch_fn =
        geqrf_batch_dispatch_vector[a_array_type_id];
    if (geqrf_batch_fn == nullptr) {
        throw py::value_error(
            "No geqrf_batch implementation defined for the provided type "
            "of the input matrix.");
    }

    // auto tau_types = dpctl_td_ns::usm_ndarray_types();
    // int tau_array_type_id =
    //     tau_types.typenum_to_lookup_id(tau_array.get_typenum());

    // if (tau_array_type_id != static_cast<int>(dpctl_td_ns::typenum_t::INT64))
    // {
    //     throw py::value_error("The type of 'tau_array' must be int64.");
    // }

    char *a_array_data = a_array.get_data();
    char *tau_array_data = tau_array.get_data();

    const std::int64_t lda = std::max<size_t>(1UL, m);

    std::vector<sycl::event> host_task_events;
    sycl::event geqrf_batch_ev =
        geqrf_batch_fn(q, m, n, a_array_data, lda, stride_a, tau_array_data,
                       stride_tau, batch_size, host_task_events, depends);

    sycl::event args_ev = dpctl::utils::keep_args_alive(q, {a_array, tau_array},
                                                        host_task_events);

    return std::make_pair(args_ev, geqrf_batch_ev);
}

template <typename fnT, typename T>
struct GeqrfBatchContigFactory
{
    fnT get()
    {
        if constexpr (types::GeqrfBatchTypePairSupportFactory<T>::is_defined) {
            return geqrf_batch_impl<T>;
        }
        else {
            return nullptr;
        }
    }
};

void init_geqrf_batch_dispatch_vector(void)
{
    dpctl_td_ns::DispatchVectorBuilder<geqrf_batch_impl_fn_ptr_t,
                                       GeqrfBatchContigFactory,
                                       dpctl_td_ns::num_types>
        contig;
    contig.populate_dispatch_vector(geqrf_batch_dispatch_vector);
}
} // namespace lapack
} // namespace ext
} // namespace backend
} // namespace dpnp
