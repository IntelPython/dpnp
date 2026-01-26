//*****************************************************************************
// Copyright (c) 2025, Intel Corporation
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

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"

// dpctl tensor headers
#include "utils/output_validation.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

namespace dpnp::extensions::window
{
namespace dpctl_td_ns = dpctl::tensor::type_dispatch;

namespace py = pybind11;

typedef sycl::event (*window_fn_ptr_t)(sycl::queue &,
                                       char *,
                                       const std::size_t,
                                       const std::vector<sycl::event> &);

template <typename T, template <typename> class Functor>
sycl::event window_impl(sycl::queue &exec_q,
                        char *result,
                        const std::size_t nelems,
                        const std::vector<sycl::event> &depends)
{
    dpctl::tensor::type_utils::validate_type_for_device<T>(exec_q);

    T *res = reinterpret_cast<T *>(result);

    sycl::event window_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using WindowKernel = Functor<T>;
        cgh.parallel_for<WindowKernel>(sycl::range<1>(nelems),
                                       WindowKernel(res, nelems));
    });

    return window_ev;
}

template <typename funcPtrT>
std::tuple<size_t, char *, funcPtrT>
    window_fn(sycl::queue &exec_q,
              const dpctl::tensor::usm_ndarray &result,
              const funcPtrT *window_dispatch_vector)
{
    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(result);

    const int nd = result.get_ndim();
    if (nd != 1) {
        throw py::value_error("Array should be 1d");
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {result.get_queue()})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queue.");
    }

    const bool is_result_c_contig = result.is_c_contiguous();
    if (!is_result_c_contig) {
        throw py::value_error("The result array is not c-contiguous.");
    }

    const std::size_t nelems = result.get_size();
    if (nelems == 0) {
        return std::make_tuple(nelems, nullptr, nullptr);
    }

    const int result_typenum = result.get_typenum();
    auto array_types = dpctl_td_ns::usm_ndarray_types();
    const int result_type_id = array_types.typenum_to_lookup_id(result_typenum);
    funcPtrT fn = window_dispatch_vector[result_type_id];

    if (fn == nullptr) {
        throw std::runtime_error("Type of given array is not supported");
    }

    char *result_typeless_ptr = result.get_data();
    return std::make_tuple(nelems, result_typeless_ptr, fn);
}

inline std::pair<sycl::event, sycl::event>
    py_window(sycl::queue &exec_q,
              const dpctl::tensor::usm_ndarray &result,
              const std::vector<sycl::event> &depends,
              const window_fn_ptr_t *window_dispatch_vector)
{
    auto [nelems, result_typeless_ptr, fn] =
        window_fn<window_fn_ptr_t>(exec_q, result, window_dispatch_vector);

    if (nelems == 0) {
        return std::make_pair(sycl::event{}, sycl::event{});
    }

    sycl::event window_ev = fn(exec_q, result_typeless_ptr, nelems, depends);
    sycl::event args_ev =
        dpctl::utils::keep_args_alive(exec_q, {result}, {window_ev});

    return std::make_pair(args_ev, window_ev);
}
} // namespace dpnp::extensions::window
