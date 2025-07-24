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

#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpctl4pybind11.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kernels/elementwise_functions/isclose.hpp"

#include "../../elementwise_functions/simplify_iteration_space.hpp"

// dpctl tensor headers
#include "utils/memory_overlap.hpp"
#include "utils/offset_utils.hpp"
#include "utils/output_validation.hpp"
#include "utils/sycl_alloc_utils.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

#include "ext/common.hpp"

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

using ext::common::value_type_of;

namespace dpnp::extensions::ufunc
{

namespace impl
{

using isclose_strided_scalar_fn_ptr_t =
    sycl::event (*)(sycl::queue &,
                    int,
                    std::size_t,
                    const py::ssize_t *,
                    const py::object &,
                    const py::object &,
                    const py::object &,
                    const char *,
                    py::ssize_t,
                    const char *,
                    py::ssize_t,
                    char *,
                    py::ssize_t,
                    const std::vector<sycl::event> &);

using isclose_contig_scalar_fn_ptr_t =
    sycl::event (*)(sycl::queue &,
                    std::size_t,
                    const py::object &,
                    const py::object &,
                    const py::object &,
                    const char *,
                    const char *,
                    char *,
                    const std::vector<sycl::event> &);

static isclose_strided_scalar_fn_ptr_t
    isclose_strided_scalar_dispatch_vector[td_ns::num_types];
static isclose_contig_scalar_fn_ptr_t
    isclose_contig_dispatch_vector[td_ns::num_types];

template <typename T>
using value_type_of_t = typename value_type_of<T>::type;

/**
 * @brief A factory to define pairs of supported types for which
 * isclose function is available.
 *
 * @tparam T Type of input vector `a` and `b` and of result vector `y`.
 */
template <typename T>
struct IsCloseOutputType
{
    using value_type = typename std::disjunction<
        td_ns::TypeMapResultEntry<T, sycl::half>,
        td_ns::TypeMapResultEntry<T, float>,
        td_ns::TypeMapResultEntry<T, double>,
        td_ns::TypeMapResultEntry<T, std::complex<float>>,
        td_ns::TypeMapResultEntry<T, std::complex<double>>,
        td_ns::DefaultResultEntry<void>>::result_type;
};

template <typename T>
sycl::event isclose_strided_scalar_call(sycl::queue &exec_q,
                                        int nd,
                                        std::size_t nelems,
                                        const py::ssize_t *shape_strides,
                                        const py::object &py_rtol,
                                        const py::object &py_atol,
                                        const py::object &py_equal_nan,
                                        const char *in1_p,
                                        py::ssize_t in1_offset,
                                        const char *in2_p,
                                        py::ssize_t in2_offset,
                                        char *out_p,
                                        py::ssize_t out_offset,
                                        const std::vector<sycl::event> &depends)
{
    using dpctl::tensor::type_utils::is_complex_v;
    using scT = std::conditional_t<is_complex_v<T>, value_type_of_t<T>, T>;

    const scT rtol = py::cast<scT>(py_rtol);
    const scT atol = py::cast<scT>(py_atol);
    const bool equal_nan = py::cast<bool>(py_equal_nan);

    return dpnp::kernels::isclose::isclose_strided_scalar_impl<T, scT>(
        exec_q, nd, nelems, shape_strides, rtol, atol, equal_nan, in1_p,
        in1_offset, in2_p, in2_offset, out_p, out_offset, depends);
}

template <typename T>
sycl::event isclose_contig_scalar_call(sycl::queue &q,
                                       std::size_t nelems,
                                       const py::object &py_rtol,
                                       const py::object &py_atol,
                                       const py::object &py_equal_nan,
                                       const char *in1_p,
                                       const char *in2_p,
                                       char *out_p,
                                       const std::vector<sycl::event> &depends)
{
    using dpctl::tensor::type_utils::is_complex_v;
    using scT = std::conditional_t<is_complex_v<T>, value_type_of_t<T>, T>;

    const scT rtol = py::cast<scT>(py_rtol);
    const scT atol = py::cast<scT>(py_atol);
    const bool equal_nan = py::cast<bool>(py_equal_nan);

    return dpnp::kernels::isclose::isclose_contig_scalar_impl<T, scT>(
        q, nelems, rtol, atol, equal_nan, in1_p, 0, in2_p, 0, out_p, 0,
        depends);
}

template <typename fnT, typename T>
struct IsCloseStridedScalarFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename IsCloseOutputType<T>::value_type,
                                     void>)
        {
            return nullptr;
        }
        else {
            return isclose_strided_scalar_call<T>;
        }
    }
};

template <typename fnT, typename T>
struct IsCloseContigScalarFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename IsCloseOutputType<T>::value_type,
                                     void>)
        {
            return nullptr;
        }
        else {
            return isclose_contig_scalar_call<T>;
        }
    }
};

void populate_isclose_dispatch_vectors()
{
    using namespace td_ns;

    DispatchVectorBuilder<isclose_strided_scalar_fn_ptr_t,
                          IsCloseStridedScalarFactory, num_types>
        dvb1;
    dvb1.populate_dispatch_vector(isclose_strided_scalar_dispatch_vector);

    DispatchVectorBuilder<isclose_contig_scalar_fn_ptr_t,
                          IsCloseContigScalarFactory, num_types>
        dvb2;
    dvb2.populate_dispatch_vector(isclose_contig_dispatch_vector);
}

std::pair<sycl::event, sycl::event>
    py_isclose_scalar(const dpctl::tensor::usm_ndarray &a,
                      const dpctl::tensor::usm_ndarray &b,
                      const py::object &py_rtol,
                      const py::object &py_atol,
                      const py::object &py_equal_nan,
                      const dpctl::tensor::usm_ndarray &res,
                      sycl::queue &exec_q,
                      const std::vector<sycl::event> &depends)
{
    auto types = td_ns::usm_ndarray_types();
    int a_typeid = types.typenum_to_lookup_id(a.get_typenum());
    int b_typeid = types.typenum_to_lookup_id(b.get_typenum());

    if (a_typeid != b_typeid) {
        throw py::type_error("Array data types are not the same.");
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {a, b, res})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(res);

    int res_nd = res.get_ndim();
    if (res_nd != a.get_ndim() || res_nd != b.get_ndim()) {
        throw py::value_error("Array dimensions are not the same.");
    }

    const py::ssize_t *a_shape = a.get_shape_raw();
    const py::ssize_t *b_shape = b.get_shape_raw();
    const py::ssize_t *res_shape = res.get_shape_raw();
    bool shapes_equal(true);
    std::size_t nelems(1);

    for (int i = 0; i < res_nd; ++i) {
        nelems *= static_cast<std::size_t>(a_shape[i]);
        shapes_equal = shapes_equal && (a_shape[i] == res_shape[i] &&
                                        b_shape[i] == res_shape[i]);
    }
    if (!shapes_equal) {
        throw py::value_error("Array shapes are not the same.");
    }

    // if nelems is zero, return
    if (nelems == 0) {
        return std::make_pair(sycl::event(), sycl::event());
    }

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(res, nelems);

    // check memory overlap
    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    auto const &same_logical_tensors =
        dpctl::tensor::overlap::SameLogicalTensors();
    if ((overlap(a, res) && !same_logical_tensors(a, res)) ||
        (overlap(b, res) && !same_logical_tensors(b, res)))
    {
        throw py::value_error("Arrays index overlapping segments of memory");
    }

    const char *a_data = a.get_data();
    const char *b_data = b.get_data();
    char *res_data = res.get_data();

    // handle contiguous inputs
    bool is_a_c_contig = a.is_c_contiguous();
    bool is_a_f_contig = a.is_f_contiguous();

    bool is_b_c_contig = b.is_c_contiguous();
    bool is_b_f_contig = b.is_f_contiguous();

    bool is_res_c_contig = res.is_c_contiguous();
    bool is_res_f_contig = res.is_f_contiguous();

    bool all_c_contig = (is_a_c_contig && is_b_c_contig && is_res_c_contig);
    bool all_f_contig = (is_a_f_contig && is_b_f_contig && is_res_f_contig);

    if (all_c_contig || all_f_contig) {
        // a_typeid == b_typeid
        auto contig_fn = isclose_contig_dispatch_vector[a_typeid];

        if (contig_fn == nullptr) {
            throw std::runtime_error(
                "Contiguous implementation is missing for a_typeid=" +
                std::to_string(a_typeid) +
                " and b_typeid=" + std::to_string(b_typeid));
        }

        auto comp_ev = contig_fn(exec_q, nelems, py_rtol, py_atol, py_equal_nan,
                                 a_data, b_data, res_data, depends);
        sycl::event ht_ev =
            dpctl::utils::keep_args_alive(exec_q, {a, b, res}, {comp_ev});

        return std::make_pair(ht_ev, comp_ev);
    }

    // simplify iteration space
    //     if 1d with strides 1 - input is contig
    //     dispatch to strided

    std::cout << "Strided impl run" << std::endl;

    auto const &a_strides = a.get_strides_vector();
    auto const &b_strides = b.get_strides_vector();
    auto const &res_strides = res.get_strides_vector();

    using shT = std::vector<py::ssize_t>;
    shT simplified_shape;
    shT simplified_a_strides;
    shT simplified_b_strides;
    shT simplified_res_strides;
    py::ssize_t a_offset(0);
    py::ssize_t b_offset(0);
    py::ssize_t res_offset(0);

    int nd = res_nd;
    const py::ssize_t *shape = a_shape;

    py_internal::simplify_iteration_space_3(
        nd, shape, a_strides, b_strides, res_strides,
        // output
        simplified_shape, simplified_a_strides, simplified_b_strides,
        simplified_res_strides, a_offset, b_offset, res_offset);

    if (nd == 1 && simplified_a_strides[0] == 1 &&
        simplified_b_strides[0] == 1 && simplified_res_strides[0] == 1)
    {
        // Special case of contiguous data
        // a_typeid == b_typeid
        auto contig_fn = isclose_contig_dispatch_vector[a_typeid];

        if (contig_fn == nullptr) {
            throw std::runtime_error(
                "Contiguous implementation is missing for a_typeid=" +
                std::to_string(a_typeid) +
                " and b_typeid=" + std::to_string(b_typeid));
        }

        std::cout << "Run contig impl in strided" << std::endl;

        int a_elem_size = a.get_elemsize();
        int b_elem_size = b.get_elemsize();
        int res_elem_size = res.get_elemsize();
        auto comp_ev = contig_fn(
            exec_q, nelems, py_rtol, py_atol, py_equal_nan,
            a_data + a_elem_size * a_offset, b_data + b_elem_size * b_offset,
            res_data + res_elem_size * res_offset, depends);

        sycl::event ht_ev =
            dpctl::utils::keep_args_alive(exec_q, {a, b, res}, {comp_ev});

        return std::make_pair(ht_ev, comp_ev);
    }

    // a_typeid == b_typeid
    auto strided_fn = isclose_strided_scalar_dispatch_vector[a_typeid];

    if (strided_fn == nullptr) {
        throw std::runtime_error(
            "isclose implementation is missing for a_typeid=" +
            std::to_string(a_typeid) +
            " and b_typeid=" + std::to_string(b_typeid));
    }

    std::cout << "Run strided impl" << std::endl;

    using dpctl::tensor::offset_utils::device_allocate_and_pack;

    std::vector<sycl::event> host_tasks{};
    host_tasks.reserve(2);

    auto ptr_size_event_triple_ = device_allocate_and_pack<py::ssize_t>(
        exec_q, host_tasks, simplified_shape, simplified_a_strides,
        simplified_b_strides, simplified_res_strides);
    auto shape_strides_owner = std::move(std::get<0>(ptr_size_event_triple_));
    const sycl::event &copy_shape_ev = std::get<2>(ptr_size_event_triple_);
    const py::ssize_t *shape_strides = shape_strides_owner.get();

    std::vector<sycl::event> all_deps;
    all_deps.reserve(depends.size() + 1);
    all_deps.insert(all_deps.end(), depends.begin(), depends.end());
    all_deps.push_back(copy_shape_ev);

    sycl::event comp_ev = strided_fn(
        exec_q, nd, nelems, shape_strides, py_rtol, py_atol, py_equal_nan,
        a_data, a_offset, b_data, b_offset, res_data, res_offset, all_deps);

    // async free of shape_strides temporary
    sycl::event tmp_cleanup_ev = dpctl::tensor::alloc_utils::async_smart_free(
        exec_q, {comp_ev}, shape_strides_owner);

    host_tasks.push_back(tmp_cleanup_ev);

    return std::make_pair(
        dpctl::utils::keep_args_alive(exec_q, {a, b, res}, host_tasks),
        comp_ev);
}

} // namespace impl

void init_isclose(py::module_ m)
{
    impl::populate_isclose_dispatch_vectors();

    m.def("_isclose_scalar", &impl::py_isclose_scalar, "", py::arg("a"),
          py::arg("b"), py::arg("py_rtol"), py::arg("py_atol"),
          py::arg("py_equal_nan"), py::arg("res"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());
}

} // namespace dpnp::extensions::ufunc
