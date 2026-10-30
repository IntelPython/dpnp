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
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
// THE POSSIBILITY OF SUCH DAMAGE.
//*****************************************************************************

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>

// utils extension header
#include "ext/common.hpp"

// dpnp tensor headers
#include "utils/memory_overlap.hpp"
#include "utils/output_validation.hpp"
#include "utils/sycl_alloc_utils.hpp"
#include "utils/type_utils.hpp"

#include "potrs.hpp"
#include "types_matrix.hpp"

namespace dpnp::extensions::lapack
{
namespace mkl_lapack = oneapi::mkl::lapack;
namespace py = pybind11;
namespace type_utils = dpnp::tensor::type_utils;

using ext::common::init_dispatch_vector;

typedef sycl::event (*potrs_impl_fn_ptr_t)(sycl::queue &,
                                           const oneapi::mkl::uplo,
                                           const std::int64_t,
                                           const std::int64_t,
                                           char *,
                                           const std::int64_t,
                                           char *,
                                           const std::int64_t,
                                           std::vector<sycl::event> &,
                                           const std::vector<sycl::event> &);

static potrs_impl_fn_ptr_t potrs_dispatch_vector[dpnp_td_ns::num_types];

template <typename T>
static sycl::event potrs_impl(sycl::queue &exec_q,
                              const oneapi::mkl::uplo upper_lower,
                              const std::int64_t n,
                              const std::int64_t nrhs,
                              char *in_a,
                              const std::int64_t lda,
                              char *in_b,
                              const std::int64_t ldb,
                              std::vector<sycl::event> &host_task_events,
                              const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<T>(exec_q);

    T *a = reinterpret_cast<T *>(in_a);
    T *b = reinterpret_cast<T *>(in_b);

    const std::int64_t scratchpad_size = mkl_lapack::potrs_scratchpad_size<T>(
        exec_q, upper_lower, n, nrhs, lda, ldb);
    T *scratchpad = nullptr;

    std::stringstream error_msg;
    std::int64_t info = 0;
    bool is_exception_caught = false;

    sycl::event potrs_event;
    try {
        // Release GIL to avoid serialization of host task submissions
        // to the same queue in OneMKL
        py::gil_scoped_release lock{};

        scratchpad = sycl::malloc_device<T>(scratchpad_size, exec_q);

        potrs_event = mkl_lapack::potrs(
            exec_q,
            upper_lower, // An enumeration value of type oneapi::mkl::uplo:
                         // oneapi::mkl::uplo::upper if the Cholesky factor is
                         // stored in the upper triangular part of `a`;
                         // oneapi::mkl::uplo::lower if it is stored in the
                         // lower triangular part.
            n,           // The order of the matrix A (0 ≤ n).
            nrhs,        // The number of right-hand sides (0 ≤ nrhs).
            a,           // Pointer to the Cholesky-factored matrix (n x n).
            lda,         // The leading dimension of `a`.
            b,   // Pointer to the matrix B of right-hand sides (ldb, nrhs),
                 // overwritten by the solution matrix X.
            ldb, // The leading dimension of `b`.
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
            error_msg << "Unexpected MKL exception caught during potrs() "
                         "call:\nreason: "
                      << e.what() << "\ninfo: " << e.info();
        }
    } catch (oneapi::mkl::exception const &e) {
        // not from LAPACK, e.g. unimplemented or a failed allocation
        is_exception_caught = true;
        error_msg << "Unexpected MKL exception caught during potrs() "
                     "call:\nreason: "
                  << e.what();
    } catch (sycl::exception const &e) {
        is_exception_caught = true;
        error_msg << "Unexpected SYCL exception caught during potrs() call:\n"
                  << e.what();
    }

    if (is_exception_caught) // an unexpected error occurs
    {
        if (scratchpad != nullptr) {
            dpnp::tensor::alloc_utils::sycl_free_noexcept(scratchpad, exec_q);
        }
        throw std::runtime_error(error_msg.str());
    }

    sycl::event clean_up_event = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(potrs_event);
        auto ctx = exec_q.get_context();
        cgh.host_task([ctx, scratchpad]() {
            dpnp::tensor::alloc_utils::sycl_free_noexcept(scratchpad, ctx);
        });
    });
    host_task_events.push_back(clean_up_event);
    return potrs_event;
}

std::pair<sycl::event, sycl::event>
    potrs(sycl::queue &exec_q,
          const dpnp::tensor::usm_ndarray &a_array,
          const dpnp::tensor::usm_ndarray &b_array,
          const std::int8_t upper_lower,
          const std::vector<sycl::event> &depends)
{
    const int a_array_nd = a_array.get_ndim();
    const int b_array_nd = b_array.get_ndim();

    if (a_array_nd != 2) {
        throw py::value_error("The Cholesky-factored array has ndim=" +
                              std::to_string(a_array_nd) +
                              ", but a 2-dimensional array is expected.");
    }
    if (b_array_nd < 1 || b_array_nd > 2) {
        throw py::value_error(
            "The right-hand sides array has ndim=" +
            std::to_string(b_array_nd) +
            ", but a 1-dimensional or a 2-dimensional array is expected.");
    }

    const py::ssize_t *a_array_shape = a_array.get_shape_raw();
    const py::ssize_t *b_array_shape = b_array.get_shape_raw();

    if (a_array_shape[0] != a_array_shape[1]) {
        throw py::value_error("The Cholesky-factored array must be square,"
                              " but got a shape of (" +
                              std::to_string(a_array_shape[0]) + ", " +
                              std::to_string(a_array_shape[1]) + ").");
    }
    if (a_array_shape[0] != b_array_shape[0]) {
        throw py::value_error("The right-hand sides array must have " +
                              std::to_string(a_array_shape[0]) +
                              " rows, but got " +
                              std::to_string(b_array_shape[0]) + ".");
    }

    // check compatibility of execution queue and allocation queue
    if (!dpnp::utils::queues_are_compatible(exec_q, {a_array, b_array})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    auto const &overlap = dpnp::tensor::overlap::MemoryOverlap();
    if (overlap(a_array, b_array)) {
        throw py::value_error("The Cholesky-factored and right-hand sides "
                              "arrays are overlapping segments of memory");
    }

    dpnp::tensor::validation::CheckWritable::throw_if_not_writable(b_array);

    if (!a_array.is_f_contiguous()) {
        throw py::value_error("The Cholesky-factored array "
                              "must be F-contiguous");
    }
    if (!b_array.is_f_contiguous()) {
        throw py::value_error("The right-hand sides array "
                              "must be F-contiguous");
    }

    auto array_types = dpnp_td_ns::usm_ndarray_types();
    int a_array_type_id =
        array_types.typenum_to_lookup_id(a_array.get_typenum());
    int b_array_type_id =
        array_types.typenum_to_lookup_id(b_array.get_typenum());

    if (a_array_type_id != b_array_type_id) {
        throw py::value_error("The types of the Cholesky-factored and "
                              "right-hand sides arrays are mismatched");
    }

    potrs_impl_fn_ptr_t potrs_fn = potrs_dispatch_vector[a_array_type_id];
    if (potrs_fn == nullptr) {
        throw py::value_error(
            "No potrs implementation defined for the provided type "
            "of the input matrix.");
    }

    const std::int64_t n = a_array_shape[0];
    const std::int64_t nrhs = (b_array_nd > 1) ? b_array_shape[1] : 1;

    const std::int64_t lda = std::max<size_t>(1UL, n);
    const std::int64_t ldb = std::max<size_t>(1UL, n);

    char *a_array_data = a_array.get_data();
    char *b_array_data = b_array.get_data();
    const oneapi::mkl::uplo uplo_val =
        static_cast<oneapi::mkl::uplo>(upper_lower);

    std::vector<sycl::event> host_task_events;
    sycl::event potrs_ev =
        potrs_fn(exec_q, uplo_val, n, nrhs, a_array_data, lda, b_array_data,
                 ldb, host_task_events, depends);

    sycl::event args_ev = dpnp::utils::keep_args_alive(
        exec_q, {a_array, b_array}, host_task_events);

    return std::make_pair(args_ev, potrs_ev);
}

template <typename fnT, typename T>
struct PotrsContigFactory
{
    fnT get()
    {
        if constexpr (types::PotrsTypePairSupportFactory<T>::is_defined) {
            return potrs_impl<T>;
        }
        else {
            return nullptr;
        }
    }
};

void init_potrs_dispatch_vector(void)
{
    init_dispatch_vector<potrs_impl_fn_ptr_t, PotrsContigFactory>(
        potrs_dispatch_vector);
}
} // namespace dpnp::extensions::lapack
