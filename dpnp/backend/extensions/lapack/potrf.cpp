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

#include "potrf.hpp"
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

typedef sycl::event (*potrf_impl_fn_ptr_t)(sycl::queue,
                                           oneapi::mkl::uplo,
                                           const std::int64_t,
                                           char *,
                                           std::int64_t,
                                           std::vector<sycl::event> &,
                                           const std::vector<sycl::event> &);

static potrf_impl_fn_ptr_t potrf_dispatch_vector[dpctl_td_ns::num_types];

template <typename T>
static sycl::event potrf_impl(sycl::queue exec_q,
                              oneapi::mkl::uplo upper_lower,
                              const std::int64_t n,
                              char *in_a,
                              std::int64_t lda,
                              std::vector<sycl::event> &host_task_events,
                              const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<T>(exec_q);

    T *a = reinterpret_cast<T *>(in_a);

    const std::int64_t scratchpad_size =
        oneapi::mkl::lapack::potrf_scratchpad_size<T>(exec_q, upper_lower, n,
                                                      lda);
    T *scratchpad = nullptr;

    std::stringstream error_msg;
    std::int64_t info = 0;

    sycl::event potrf_event;
    try {
        scratchpad = sycl::malloc_device<T>(scratchpad_size, exec_q);

        potrf_event = oneapi::mkl::lapack::potrf(
            exec_q,
            upper_lower, //
            n,           // Order of the square matrix; (0 ≤ n).
            a,           // Pointer to the n-by-n matrix.
            lda,         // The leading dimension of `a`.
            scratchpad,  // Pointer to scratchpad memory to be used by MKL
                         // routine for storing intermediate results.
            scratchpad_size, depends);
    } catch (mkl_lapack::exception const &e) {
        error_msg
            << "Unexpected MKL exception caught during potrf() call:\nreason: "
            << e.what() << "\ninfo: " << e.info();
        info = e.info();
    } catch (sycl::exception const &e) {
        error_msg << "Unexpected SYCL exception caught during potrf() call:\n"
                  << e.what();
        info = -1;
    }

    if (info != 0) // an unexpected error occurs
    {
        if (scratchpad != nullptr) {
            sycl::free(scratchpad, exec_q);
        }

        // TODO: use LinAlgError
        if (info == 2) {
            throw py::value_error("Matrix is not positive definite");
        }

        throw std::runtime_error(error_msg.str());
    }

    sycl::event clean_up_event = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(potrf_event);
        auto ctx = exec_q.get_context();
        cgh.host_task([ctx, scratchpad]() { sycl::free(scratchpad, ctx); });
    });
    host_task_events.push_back(clean_up_event);
    return potrf_event;
}

std::pair<sycl::event, sycl::event>
    potrf(sycl::queue q,
          const std::int64_t n,
          dpctl::tensor::usm_ndarray a_array,
          const std::vector<sycl::event> &depends)
{

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    int a_array_type_id =
        array_types.typenum_to_lookup_id(a_array.get_typenum());

    potrf_impl_fn_ptr_t potrf_fn = potrf_dispatch_vector[a_array_type_id];
    if (potrf_fn == nullptr) {
        throw py::value_error(
            "No potrf implementation defined for the provided type "
            "of the input matrix.");
    }

    char *a_array_data = a_array.get_data();
    const std::int64_t lda = std::max<size_t>(1UL, n);
    oneapi::mkl::uplo upper_lower = oneapi::mkl::uplo::upper;

    std::vector<sycl::event> host_task_events;
    sycl::event potrf_ev = potrf_fn(q, upper_lower, n, a_array_data, lda,
                                    host_task_events, depends);

    sycl::event args_ev =
        dpctl::utils::keep_args_alive(q, {a_array}, host_task_events);

    return std::make_pair(args_ev, potrf_ev);
}

template <typename fnT, typename T>
struct PotrfContigFactory
{
    fnT get()
    {
        if constexpr (types::PotrfTypePairSupportFactory<T>::is_defined) {
            return potrf_impl<T>;
        }
        else {
            return nullptr;
        }
    }
};

void init_potrf_dispatch_vector(void)
{
    dpctl_td_ns::DispatchVectorBuilder<potrf_impl_fn_ptr_t, PotrfContigFactory,
                                       dpctl_td_ns::num_types>
        contig;
    contig.populate_dispatch_vector(potrf_dispatch_vector);
}
} // namespace lapack
} // namespace ext
} // namespace backend
} // namespace dpnp
