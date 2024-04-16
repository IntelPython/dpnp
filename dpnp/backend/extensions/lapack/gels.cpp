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

#include "gels.hpp"
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

typedef sycl::event (*gels_impl_fn_ptr_t)(sycl::queue,
                                           oneapi::mkl::transpose,
                                           const std::int64_t,
                                           const std::int64_t,
                                           const std::int64_t,
                                           char *,
                                           std::int64_t,
                                           char *,
                                           std::int64_t,
                                           std::vector<sycl::event> &,
                                           const std::vector<sycl::event> &);

static gels_impl_fn_ptr_t gels_dispatch_vector[dpctl_td_ns::num_types];

template <typename T>
static sycl::event gels_impl(sycl::queue exec_q,
                              oneapi::mkl::transpose trans,
                              const std::int64_t m,
                              const std::int64_t n,
                              const std::int64_t nrhs,
                              char *in_a,
                              std::int64_t lda,
                              char *in_b,
                              std::int64_t ldb,
                              std::vector<sycl::event> &host_task_events,
                              const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<T>(exec_q);

    T *a = reinterpret_cast<T *>(in_a);
    T *b = reinterpret_cast<T *>(in_b);

    const std::int64_t scratchpad_size =
        mkl_lapack::gels_scratchpad_size<T>(exec_q, trans, m, n, nrhs, lda, ldb);
    T *scratchpad = nullptr;

    std::stringstream error_msg;
    std::int64_t info = 0;
    bool is_exception_caught = false;

    sycl::event gels_event;
    try {
        scratchpad = sycl::malloc_device<T>(scratchpad_size, exec_q);

        gels_event = mkl_lapack::gels(
            exec_q,
            trans,
            m,          // The number of rows in the matrix; (0 ≤ m).
            n,          // The number of columns in the matrix; (0 ≤ n).
            nrhs,          // The number of elementary reflectors
                        // whose product defines the matrix Q; (0 ≤ k ≤ n).
            a,          // Pointer to the m-by-n matrix.
            lda,        // The leading dimension of `a`; (1 ≤ m).
            b,
            ldb,
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
            error_msg << "Unexpected MKL exception caught during gels() "
                         "call:\nreason: "
                      << e.what() << "\ninfo: " << info;
        }
    } catch (sycl::exception const &e) {
        is_exception_caught = true;
        error_msg << "Unexpected SYCL exception caught during gels() call:\n"
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
        cgh.depends_on(gels_event);
        auto ctx = exec_q.get_context();
        cgh.host_task([ctx, scratchpad]() { sycl::free(scratchpad, ctx); });
    });
    host_task_events.push_back(clean_up_event);

    return gels_event;
}

std::pair<sycl::event, sycl::event>
    gels(sycl::queue q,
          const std::int64_t m,
          const std::int64_t n,
          const std::int64_t nrhs,
          dpctl::tensor::usm_ndarray a_array,
          dpctl::tensor::usm_ndarray b_array,
          const std::vector<sycl::event> &depends)
{
    const int a_array_nd = a_array.get_ndim();
    const int b_array_nd = b_array.get_ndim();

    if (a_array_nd != 2) {
        throw py::value_error(
            "The input array has ndim=" + std::to_string(a_array_nd) +
            ", but a 2-dimensional array is expected.");
    }

    if (b_array_nd != 2) {
        throw py::value_error(
            "The input array has ndim=" + std::to_string(b_array_nd) +
            ", but a 2-dimensional array is expected.");
    }

    // check compatibility of execution queue and allocation queue
    if (!dpctl::utils::queues_are_compatible(q, {a_array, b_array})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(a_array, b_array)) {
        throw py::value_error(
            "The input array and the array of Householder scalars "
            "are overlapping segments of memory");
    }

    // bool is_a_array_c_contig = a_array.is_c_contiguous();
    // if (!is_a_array_c_contig) {
    //     throw py::value_error("The input array "
    //                           "must be C-contiguous");
    // }

    // bool is_b_array_c_contig = b_array.is_c_contiguous();
    // if (!is_b_array_c_contig) {
    //     throw py::value_error("The input array "
    //                           "must be C-contiguous");
    // }


    auto array_types = dpctl_td_ns::usm_ndarray_types();
    int a_array_type_id =
        array_types.typenum_to_lookup_id(a_array.get_typenum());
    int b_array_type_id =
        array_types.typenum_to_lookup_id(b_array.get_typenum());

    if (a_array_type_id != b_array_type_id) {
        throw py::value_error(
            "The types of the input array and "
            "the array of Householder scalars are mismatched");
    }

    gels_impl_fn_ptr_t gels_fn = gels_dispatch_vector[a_array_type_id];
    if (gels_fn == nullptr) {
        throw py::value_error(
            "No gels implementation defined for the provided type "
            "of the input matrix.");
    }

    char *a_array_data = a_array.get_data();
    char *b_array_data = b_array.get_data();

    const std::int64_t lda = std::max<size_t>(1UL, m);
    const std::int64_t ldb = std::max<size_t>(1UL, n);

    //should be T according to mkl
    oneapi::mkl::transpose trans = oneapi::mkl::transpose::N;

    std::vector<sycl::event> host_task_events;
    sycl::event gels_ev = gels_fn(q, trans, m, n, nrhs, a_array_data, lda,
                                    b_array_data, ldb,
                                    host_task_events, depends);

    sycl::event args_ev = dpctl::utils::keep_args_alive(q, {a_array, b_array},
                                                        host_task_events);

    return std::make_pair(args_ev, gels_ev);
}

template <typename fnT, typename T>
struct GelsContigFactory
{
    fnT get()
    {
        if constexpr (types::GelsTypePairSupportFactory<T>::is_defined) {
            return gels_impl<T>;
        }
        else {
            return nullptr;
        }
    }
};

void init_gels_dispatch_vector(void)
{
    dpctl_td_ns::DispatchVectorBuilder<gels_impl_fn_ptr_t, GelsContigFactory,
                                       dpctl_td_ns::num_types>
        contig;
    contig.populate_dispatch_vector(gels_dispatch_vector);
}
} // namespace lapack
} // namespace ext
} // namespace backend
} // namespace dpnp
