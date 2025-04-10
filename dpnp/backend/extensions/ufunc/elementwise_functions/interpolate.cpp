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
#include <vector>

#include "dpctl4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// dpctl tensor headers
#include "utils/output_validation.hpp"
#include "utils/type_dispatch.hpp"

#include "kernels/elementwise_functions/interpolate.hpp"

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

namespace dpnp::extensions::ufunc
{

namespace impl
{

template <typename T>
struct value_type_of
{
    using type = T;
};

template <typename T>
struct value_type_of<std::complex<T>>
{
    using type = T;
};

template <typename T>
using value_type_of_t = typename value_type_of<T>::type;

typedef sycl::event (*interpolate_fn_ptr_t)(sycl::queue &,
                                            const void *, // x
                                            const void *, // idx
                                            const void *, // xp
                                            const void *, // fp
                                            const void *, // left
                                            const void *, // right
                                            void *,       // out
                                            std::size_t,  // n
                                            std::size_t,  // xp_size
                                            const std::vector<sycl::event> &);

template <typename T>
sycl::event interpolate_call(sycl::queue &exec_q,
                             const void *vx,
                             const void *vidx,
                             const void *vxp,
                             const void *vfp,
                             const void *vleft,
                             const void *vright,
                             void *vout,
                             std::size_t n,
                             std::size_t xp_size,
                             const std::vector<sycl::event> &depends)
{
    using dpctl::tensor::type_utils::is_complex_v;
    using TCoord = std::conditional_t<is_complex_v<T>, value_type_of_t<T>, T>;

    const TCoord *x = static_cast<const TCoord *>(vx);
    const std::size_t *idx = static_cast<const std::size_t *>(vidx);
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
    int x_typenum = x.get_typenum();
    int xp_typenum = xp.get_typenum();
    int fp_typenum = fp.get_typenum();
    int out_typenum = out.get_typenum();

    auto array_types = td_ns::usm_ndarray_types();
    int x_type_id = array_types.typenum_to_lookup_id(x_typenum);
    int xp_type_id = array_types.typenum_to_lookup_id(xp_typenum);
    int fp_type_id = array_types.typenum_to_lookup_id(fp_typenum);
    int out_type_id = array_types.typenum_to_lookup_id(out_typenum);

    if (x_type_id != xp_type_id) {
        throw py::value_error("x and xp must have the same dtype");
    }
    if (fp_type_id != out_type_id) {
        throw py::value_error("fp and out must have the same dtype");
    }

    auto fn = interpolate_dispatch_vector[fp_type_id];
    if (!fn) {
        throw py::type_error("Unsupported dtype");
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {x, idx, xp, fp, out})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(out);

    if (x.get_ndim() != 1 || xp.get_ndim() != 1 || fp.get_ndim() != 1 ||
        idx.get_ndim() != 1 || out.get_ndim() != 1)
    {
        throw py::value_error("All arrays must be one-dimensional");
    }

    if (xp.get_size() != fp.get_size()) {
        throw py::value_error("xp and fp must have the same size");
    }

    if (x.get_size() != out.get_size() || x.get_size() != idx.get_size()) {
        throw py::value_error("x, idx, and out must have the same size");
    }

    std::size_t n = x.get_size();
    std::size_t xp_size = xp.get_size();

    void *left_ptr = left.has_value() ? left.value().get_data() : nullptr;

    void *right_ptr = right.has_value() ? right.value().get_data() : nullptr;

    sycl::event ev =
        fn(exec_q, x.get_data(), idx.get_data(), xp.get_data(), fp.get_data(),
           left_ptr, right_ptr, out.get_data(), n, xp_size, depends);

    sycl::event args_ev;

    if (left.has_value() && right.has_value()) {
        args_ev = dpctl::utils::keep_args_alive(
            exec_q, {x, idx, xp, fp, out, left.value(), right.value()}, {ev});
    }
    else if (left.has_value()) {
        args_ev = dpctl::utils::keep_args_alive(
            exec_q, {x, idx, xp, fp, out, left.value()}, {ev});
    }
    else if (right.has_value()) {
        args_ev = dpctl::utils::keep_args_alive(
            exec_q, {x, idx, xp, fp, out, right.value()}, {ev});
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
        td_ns::TypeMapResultEntry<T, sycl::half>,
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

void init_interpolate_dispatch_vectors()
{
    using namespace td_ns;

    DispatchVectorBuilder<interpolate_fn_ptr_t, InterpolateFactory, num_types>
        dtb_interpolate;
    dtb_interpolate.populate_dispatch_vector(interpolate_dispatch_vector);
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
