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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dpnp4pybind11.hpp"

#include "median.hpp"

// dpnp tensor headers
#include "kernels/sorting/radix_select.hpp"
#include "utils/output_validation.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

// utils extension headers
#include "ext/common.hpp"
#include "ext/validation_utils.hpp"

namespace td_ns = dpnp::tensor::type_dispatch;

using dpnp::tensor::usm_ndarray;

using ext::validation::array_names;
using ext::validation::check_c_contig;
using ext::validation::check_no_overlap;
using ext::validation::check_num_dims;
using ext::validation::check_queue;
using ext::validation::check_writable;

namespace statistics::median
{
using ext::common::init_dispatch_table;

typedef sycl::event (*median_fn_ptr_t)(sycl::queue &,
                                       const std::size_t, // n_iters
                                       const std::size_t, // n_values
                                       const bool,        // ignore_nan
                                       const char *,      // src
                                       char *,            // dst
                                       std::uint32_t *,   // all_nan
                                       const std::vector<sycl::event> &);

static median_fn_ptr_t median_dispatch_table[td_ns::num_types]
                                            [td_ns::num_types];

namespace
{
template <typename T, typename ResT>
sycl::event median_call(sycl::queue &exec_q,
                        const std::size_t n_iters,
                        const std::size_t n_values,
                        const bool ignore_nan,
                        const char *src_p,
                        char *dst_p,
                        std::uint32_t *all_nan_p,
                        const std::vector<sycl::event> &depends)
{
    using dpnp::tensor::kernels::radix_select_details::radix_median_impl;
    return radix_median_impl<T, ResT>(exec_q, n_iters, n_values, ignore_nan,
                                      reinterpret_cast<const T *>(src_p),
                                      reinterpret_cast<ResT *>(dst_p),
                                      all_nan_p, depends);
}
} // namespace

/*! @brief Writes the median of each row of C-contiguous `src` along its last
 * axis into C-contiguous `dst` of the shape of `src` without its last axis;
 * the single element of `all_nan`, when it is given, is set to whether
 * `ignore_nan` and a row has NaNs only */
std::pair<sycl::event, sycl::event>
    py_median(const usm_ndarray &src,
              const bool ignore_nan,
              const usm_ndarray &dst,
              sycl::queue &exec_q,
              const std::optional<const usm_ndarray> &all_nan,
              const std::vector<sycl::event> &depends)
{
    array_names names = {{&src, "src"}, {&dst, "dst"}};

    const int nd = src.get_ndim();
    if (nd == 0) {
        throw py::value_error("The input array must have at least one "
                              "dimension");
    }
    check_num_dims(&dst, nd - 1, names);

    const py::ssize_t *src_shape = src.get_shape_raw();
    const py::ssize_t *dst_shape = dst.get_shape_raw();
    std::size_t n_iters(1);
    for (int i = 0; i + 1 < nd; ++i) {
        if (src_shape[i] != dst_shape[i]) {
            throw py::value_error("`dst` shape must match `src` shape "
                                  "without its last axis");
        }
        n_iters *= static_cast<std::size_t>(src_shape[i]);
    }
    const std::size_t n_values = static_cast<std::size_t>(src_shape[nd - 1]);

    std::uint32_t *all_nan_p = nullptr;
    if (all_nan.has_value()) {
        const usm_ndarray &flag = all_nan.value();
        names.emplace(&flag, "all_nan");
        check_queue({&src, &dst, &flag}, names, exec_q);
        check_writable({&dst, &flag}, names);
        auto types = td_ns::usm_ndarray_types();
        if (flag.get_size() != 1 ||
            types.typenum_to_lookup_id(flag.get_typenum()) !=
                static_cast<int>(td_ns::typenum_t::UINT32)) {
            throw py::value_error("`all_nan` must be a single uint32 element");
        }
        check_no_overlap(&dst, &flag, names);
        all_nan_p = reinterpret_cast<std::uint32_t *>(flag.get_data());
    }
    else {
        check_queue({&src, &dst}, names, exec_q);
        check_writable({&dst}, names);
    }

    if (n_iters == 0) {
        return {sycl::event(), sycl::event()};
    }
    if (n_values == 0) {
        throw py::value_error("The median of an empty axis is undefined");
    }

    check_c_contig({&src, &dst}, names);
    check_no_overlap(&src, &dst, names);
    dpnp::tensor::validation::AmpleMemory::throw_if_not_ample(dst, n_iters);

    auto types = td_ns::usm_ndarray_types();
    const int src_typeid = types.typenum_to_lookup_id(src.get_typenum());
    const int dst_typeid = types.typenum_to_lookup_id(dst.get_typenum());

    auto fn = median_dispatch_table[src_typeid][dst_typeid];
    if (fn == nullptr) {
        throw py::value_error("No median implementation is available for "
                              "the data types of the input and the output "
                              "arrays");
    }

    std::vector<sycl::event> median_deps(depends);
    if (all_nan_p != nullptr) {
        median_deps.push_back(
            exec_q.fill<std::uint32_t>(all_nan_p, 0, 1, depends));
    }
    sycl::event comp_ev =
        fn(exec_q, n_iters, n_values, ignore_nan, src.get_data(),
           dst.get_data(), all_nan_p, median_deps);
    sycl::event ht_ev =
        (all_nan.has_value())
            ? dpnp::utils::keep_args_alive(exec_q, {src, dst, all_nan.value()},
                                           {comp_ev})
            : dpnp::utils::keep_args_alive(exec_q, {src, dst}, {comp_ev});

    return std::make_pair(ht_ev, comp_ev);
}

/*! @brief Whether the median of `T` is written as `ResT`: floating types
 * keep their type, the others are averaged in double or, on devices without
 * it, in single precision */
template <typename T, typename ResT>
constexpr bool is_median_type_pair()
{
    using dpnp::tensor::type_utils::is_complex_v;
    // complex types have no order-preserving radix key
    if constexpr (is_complex_v<T> || is_complex_v<ResT>) {
        return false;
    }
    else if constexpr (std::is_floating_point_v<T> ||
                       std::is_same_v<T, sycl::half>) {
        return std::is_same_v<T, ResT>;
    }
    else {
        return std::is_same_v<ResT, double> || std::is_same_v<ResT, float>;
    }
}

template <typename fnT, typename T, typename ResT>
struct MedianFactory
{
    fnT get()
    {
        if constexpr (is_median_type_pair<T, ResT>()) {
            return median_call<T, ResT>;
        }
        else {
            return nullptr;
        }
    }
};

void populate_median(py::module_ m)
{
    init_dispatch_table<median_fn_ptr_t, MedianFactory>(median_dispatch_table);

    m.def("_median", &py_median,
          "Writes the median of each row of C-contiguous `src` along its last "
          "axis into `dst` of the shape of `src` without its last axis; it is "
          "NaN for a row with NaNs, or, if `ignore_nan`, with NaNs only; the "
          "single element of `all_nan`, if given, is set to whether any row "
          "is of the latter",
          py::arg("src"), py::arg("ignore_nan"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("all_nan") = py::none(),
          py::arg("depends") = py::list());
}

} // namespace statistics::median
