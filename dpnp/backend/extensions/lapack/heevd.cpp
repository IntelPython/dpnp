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

#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "evd_common.hpp"
#include "heevd.hpp"

// utils extension header
#include "ext/common.hpp"

// dpctl tensor headers
#include "utils/type_utils.hpp"

namespace dpnp::extensions::lapack
{
namespace mkl_lapack = oneapi::mkl::lapack;
namespace type_utils = dpctl::tensor::type_utils;

using ext::common::init_dispatch_table;

template <typename T, typename RealT>
static sycl::event heevd_impl(sycl::queue &exec_q,
                              const oneapi::mkl::job jobz,
                              const oneapi::mkl::uplo upper_lower,
                              const std::int64_t n,
                              char *in_a,
                              char *out_w,
                              const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<T>(exec_q);
    type_utils::validate_type_for_device<RealT>(exec_q);

    T *a = reinterpret_cast<T *>(in_a);
    RealT *w = reinterpret_cast<RealT *>(out_w);

    const std::int64_t lda = std::max<size_t>(1UL, n);
    const std::int64_t scratchpad_size =
        mkl_lapack::heevd_scratchpad_size<T>(exec_q, jobz, upper_lower, n, lda);

    T *scratchpad = helper::alloc_scratchpad<T>(scratchpad_size, exec_q);

    std::stringstream error_msg;
    std::int64_t info = 0;

    sycl::event heevd_event;
    try {
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

    if (info != 0) // an unexpected error occurs
    {
        if (scratchpad != nullptr) {
            dpctl::tensor::alloc_utils::sycl_free_noexcept(scratchpad, exec_q);
        }
        throw std::runtime_error(error_msg.str());
    }

    sycl::event ht_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(heevd_event);
        auto ctx = exec_q.get_context();
        cgh.host_task([ctx, scratchpad]() {
            dpctl::tensor::alloc_utils::sycl_free_noexcept(scratchpad, ctx);
        });
    });

    return ht_ev;
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

using evd::evd_impl_fn_ptr_t;

void init_heevd(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;

    static evd_impl_fn_ptr_t heevd_dispatch_table[dpctl_td_ns::num_types]
                                                 [dpctl_td_ns::num_types];

    {
        init_dispatch_table<evd_impl_fn_ptr_t, HeevdContigFactory>(
            heevd_dispatch_table);

        auto heevd_pyapi = [&](sycl::queue &exec_q, const std::int8_t jobz,
                               const std::int8_t upper_lower,
                               const arrayT &eig_vecs, const arrayT &eig_vals,
                               const event_vecT &depends = {}) {
            return evd::evd_func(exec_q, jobz, upper_lower, eig_vecs, eig_vals,
                                 depends, heevd_dispatch_table);
        };

        m.def("_heevd", heevd_pyapi,
              "Call `heevd` from OneMKL LAPACK library to return "
              "the eigenvalues and eigenvectors of a complex Hermitian matrix",
              py::arg("sycl_queue"), py::arg("jobz"), py::arg("upper_lower"),
              py::arg("eig_vecs"), py::arg("eig_vals"),
              py::arg("depends") = py::list());
    }
}
} // namespace dpnp::extensions::lapack
