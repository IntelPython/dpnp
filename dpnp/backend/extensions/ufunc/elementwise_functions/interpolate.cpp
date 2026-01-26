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

#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"

// dpctl tensor headers
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

#include "kernels/elementwise_functions/interpolate.hpp"

// utils extension headers
#include "ext/common.hpp"
#include "ext/validation_utils.hpp"

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;
namespace type_utils = dpctl::tensor::type_utils;

using ext::common::value_type_of;
using ext::validation::array_names;
using ext::validation::array_ptr;

using ext::common::dtype_from_typenum;
using ext::validation::check_has_dtype;
using ext::validation::check_num_dims;
using ext::validation::check_same_dtype;
using ext::validation::check_same_size;
using ext::validation::common_checks;

namespace dpnp::extensions::ufunc
{

namespace impl
{
using ext::common::init_dispatch_vector;

template <typename T>
using value_type_of_t = typename value_type_of<T>::type;

typedef sycl::event (*interpolate_fn_ptr_t)(sycl::queue &,
                                            const void *,      // x
                                            const void *,      // idx
                                            const void *,      // xp
                                            const void *,      // fp
                                            const void *,      // left
                                            const void *,      // right
                                            void *,            // out
                                            const std::size_t, // n
                                            const std::size_t, // xp_size
                                            const std::vector<sycl::event> &);

template <typename T, typename TIdx = std::int64_t>
sycl::event interpolate_call(sycl::queue &exec_q,
                             const void *vx,
                             const void *vidx,
                             const void *vxp,
                             const void *vfp,
                             const void *vleft,
                             const void *vright,
                             void *vout,
                             const std::size_t n,
                             const std::size_t xp_size,
                             const std::vector<sycl::event> &depends)
{
    using type_utils::is_complex_v;
    using TCoord = std::conditional_t<is_complex_v<T>, value_type_of_t<T>, T>;

    const TCoord *x = static_cast<const TCoord *>(vx);
    const TIdx *idx = static_cast<const TIdx *>(vidx);
    const TCoord *xp = static_cast<const TCoord *>(vxp);
    const T *fp = static_cast<const T *>(vfp);
    const T *left = static_cast<const T *>(vleft);
    const T *right = static_cast<const T *>(vright);
    T *out = static_cast<T *>(vout);

    using dpnp::kernels::interpolate::interpolate_impl;
    sycl::event interpolate_ev = interpolate_impl<TCoord, T>(
        exec_q, x, idx, xp, fp, left, right, out, n, xp_size, depends);

    return interpolate_ev;
}

interpolate_fn_ptr_t interpolate_dispatch_vector[td_ns::num_types];

void common_interpolate_checks(
    const dpctl::tensor::usm_ndarray &x,
    const dpctl::tensor::usm_ndarray &idx,
    const dpctl::tensor::usm_ndarray &xp,
    const dpctl::tensor::usm_ndarray &fp,
    const dpctl::tensor::usm_ndarray &out,
    const std::optional<const dpctl::tensor::usm_ndarray> &left,
    const std::optional<const dpctl::tensor::usm_ndarray> &right)
{
    array_names names = {{&x, "x"}, {&xp, "xp"}, {&fp, "fp"}, {&out, "out"}};

    auto left_v = left ? &left.value() : nullptr;
    if (left_v) {
        names.insert({left_v, "left"});
    }

    auto right_v = right ? &right.value() : nullptr;
    if (right_v) {
        names.insert({right_v, "right"});
    }

    check_same_dtype(&x, &xp, names);
    check_same_dtype({&fp, left_v, right_v, &out}, names);
    check_has_dtype(&idx, td_ns::typenum_t::INT64, names);

    common_checks({&x, &xp, &fp, left_v, right_v}, {&out}, names);

    check_num_dims({&x, &xp, &fp, &idx, &out}, 1, names);
    check_num_dims({left_v, right_v}, 0, names);

    check_same_size(&xp, &fp, names);
    check_same_size({&x, &idx, &out}, names);

    if (xp.get_size() == 0) {
        throw py::value_error("array of sample points is empty");
    }
}

std::pair<sycl::event, sycl::event>
    py_interpolate(const dpctl::tensor::usm_ndarray &x,
                   const dpctl::tensor::usm_ndarray &idx,
                   const dpctl::tensor::usm_ndarray &xp,
                   const dpctl::tensor::usm_ndarray &fp,
                   std::optional<const dpctl::tensor::usm_ndarray> &left,
                   std::optional<const dpctl::tensor::usm_ndarray> &right,
                   dpctl::tensor::usm_ndarray &out,
                   sycl::queue &exec_q,
                   const std::vector<sycl::event> &depends)
{
    common_interpolate_checks(x, idx, xp, fp, out, left, right);

    int out_typenum = out.get_typenum();

    auto array_types = td_ns::usm_ndarray_types();
    int out_type_id = array_types.typenum_to_lookup_id(out_typenum);

    auto fn = interpolate_dispatch_vector[out_type_id];
    if (!fn) {
        py::dtype out_dtype_py = dtype_from_typenum(out_type_id);
        std::string msg = "Unsupported dtype for interpolation: " +
                          std::string(py::str(out_dtype_py));
        throw py::type_error(msg);
    }

    std::size_t n = x.get_size();
    std::size_t xp_size = xp.get_size();

    void *left_ptr = left ? left.value().get_data() : nullptr;
    void *right_ptr = right ? right.value().get_data() : nullptr;

    sycl::event ev =
        fn(exec_q, x.get_data(), idx.get_data(), xp.get_data(), fp.get_data(),
           left_ptr, right_ptr, out.get_data(), n, xp_size, depends);

    sycl::event args_ev;

    if (left && right) {
        args_ev = dpctl::utils::keep_args_alive(
            exec_q, {x, idx, xp, fp, out, left.value(), right.value()}, {ev});
    }
    else if (left || right) {
        args_ev = dpctl::utils::keep_args_alive(
            exec_q, {x, idx, xp, fp, out, left ? left.value() : right.value()},
            {ev});
    }
    else {
        args_ev =
            dpctl::utils::keep_args_alive(exec_q, {x, idx, xp, fp, out}, {ev});
    }

    return std::make_pair(args_ev, ev);
}

/**
 * @brief A factory to define pairs of supported types for which
 * interpolate function is available.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct InterpolateOutputType
{
    using value_type = typename std::disjunction<
        td_ns::TypeMapResultEntry<T, float>,
        td_ns::TypeMapResultEntry<T, double>,
        td_ns::TypeMapResultEntry<T, std::complex<float>>,
        td_ns::TypeMapResultEntry<T, std::complex<double>>,
        td_ns::DefaultResultEntry<void>>::result_type;
};

template <typename fnT, typename T>
struct InterpolateFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename InterpolateOutputType<T>::value_type, void>)
        {
            return nullptr;
        }
        else {
            return interpolate_call<T>;
        }
    }
};

static void init_interpolate_dispatch_vectors()
{
    init_dispatch_vector<interpolate_fn_ptr_t, InterpolateFactory>(
        interpolate_dispatch_vector);
}

} // namespace impl

void init_interpolate(py::module_ m)
{
    impl::init_interpolate_dispatch_vectors();

    using impl::py_interpolate;
    m.def("_interpolate", &py_interpolate, "", py::arg("x"), py::arg("idx"),
          py::arg("xp"), py::arg("fp"), py::arg("left"), py::arg("right"),
          py::arg("out"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());
}

} // namespace dpnp::extensions::ufunc
