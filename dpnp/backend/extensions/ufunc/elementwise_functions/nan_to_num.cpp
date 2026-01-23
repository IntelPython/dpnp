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

#include "dpnp4pybind11.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kernels/elementwise_functions/nan_to_num.hpp"

#include "../../elementwise_functions/simplify_iteration_space.hpp"

// dpctl tensor headers
#include "utils/memory_overlap.hpp"
#include "utils/offset_utils.hpp"
#include "utils/output_validation.hpp"
#include "utils/sycl_alloc_utils.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

// utils extension header
#include "ext/common.hpp"

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

using ext::common::value_type_of;

// declare pybind11 wrappers in py_internal namespace
namespace dpnp::extensions::ufunc
{

namespace impl
{
using ext::common::init_dispatch_vector;

template <typename T>
using value_type_of_t = typename value_type_of<T>::type;

typedef sycl::event (*nan_to_num_fn_ptr_t)(sycl::queue &,
                                           int,
                                           std::size_t,
                                           const py::ssize_t *,
                                           const py::object &,
                                           const py::object &,
                                           const py::object &,
                                           const char *,
                                           py::ssize_t,
                                           char *,
                                           py::ssize_t,
                                           const std::vector<sycl::event> &);

template <typename T>
sycl::event nan_to_num_strided_call(sycl::queue &exec_q,
                                    int nd,
                                    std::size_t nelems,
                                    const py::ssize_t *shape_strides,
                                    const py::object &py_nan,
                                    const py::object &py_posinf,
                                    const py::object &py_neginf,
                                    const char *arg_p,
                                    py::ssize_t arg_offset,
                                    char *dst_p,
                                    py::ssize_t dst_offset,
                                    const std::vector<sycl::event> &depends)
{
    using dpctl::tensor::type_utils::is_complex_v;
    using scT = std::conditional_t<is_complex_v<T>, value_type_of_t<T>, T>;

    const scT nan_v = py::cast<const scT>(py_nan);
    const scT posinf_v = py::cast<const scT>(py_posinf);
    const scT neginf_v = py::cast<const scT>(py_neginf);

    using dpnp::kernels::nan_to_num::nan_to_num_strided_impl;
    sycl::event to_num_ev = nan_to_num_strided_impl<T, scT>(
        exec_q, nd, nelems, shape_strides, nan_v, posinf_v, neginf_v, arg_p,
        arg_offset, dst_p, dst_offset, depends);

    return to_num_ev;
}

typedef sycl::event (*nan_to_num_contig_fn_ptr_t)(
    sycl::queue &,
    std::size_t,
    const py::object &,
    const py::object &,
    const py::object &,
    const char *,
    char *,
    const std::vector<sycl::event> &);

template <typename T>
sycl::event nan_to_num_contig_call(sycl::queue &exec_q,
                                   std::size_t nelems,
                                   const py::object &py_nan,
                                   const py::object &py_posinf,
                                   const py::object &py_neginf,
                                   const char *arg_p,
                                   char *dst_p,
                                   const std::vector<sycl::event> &depends)
{
    using dpctl::tensor::type_utils::is_complex_v;
    using scT = std::conditional_t<is_complex_v<T>, value_type_of_t<T>, T>;

    const scT nan_v = py::cast<const scT>(py_nan);
    const scT posinf_v = py::cast<const scT>(py_posinf);
    const scT neginf_v = py::cast<const scT>(py_neginf);

    using dpnp::kernels::nan_to_num::nan_to_num_contig_impl;
    sycl::event to_num_contig_ev = nan_to_num_contig_impl<T, scT>(
        exec_q, nelems, nan_v, posinf_v, neginf_v, arg_p, dst_p, depends);

    return to_num_contig_ev;
}

namespace td_ns = dpctl::tensor::type_dispatch;
nan_to_num_fn_ptr_t nan_to_num_dispatch_vector[td_ns::num_types];
nan_to_num_contig_fn_ptr_t nan_to_num_contig_dispatch_vector[td_ns::num_types];

std::pair<sycl::event, sycl::event>
    py_nan_to_num(const dpctl::tensor::usm_ndarray &src,
                  const py::object &py_nan,
                  const py::object &py_posinf,
                  const py::object &py_neginf,
                  const dpctl::tensor::usm_ndarray &dst,
                  sycl::queue &q,
                  const std::vector<sycl::event> &depends)
{
    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    const auto &array_types = td_ns::usm_ndarray_types();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    if (src_typeid != dst_typeid) {
        throw py::value_error("Array data types are not the same.");
    }

    if (!dpctl::utils::queues_are_compatible(q, {src, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    const int src_nd = src.get_ndim();
    if (src_nd != dst.get_ndim()) {
        throw py::value_error("Array dimensions are not the same.");
    }

    const py::ssize_t *src_shape = src.get_shape_raw();
    const py::ssize_t *dst_shape = dst.get_shape_raw();

    const std::size_t nelems = src.get_size();
    const bool shapes_equal =
        std::equal(src_shape, src_shape + src_nd, dst_shape);
    if (!shapes_equal) {
        throw py::value_error("Array shapes are not the same.");
    }

    // if nelems is zero, return
    if (nelems == 0) {
        return std::make_pair(sycl::event(), sycl::event());
    }

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(dst, nelems);

    // check memory overlap
    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    auto const &same_logical_tensors =
        dpctl::tensor::overlap::SameLogicalTensors();
    if (overlap(src, dst) && !same_logical_tensors(src, dst)) {
        throw py::value_error("Arrays index overlapping segments of memory");
    }

    const char *src_data = src.get_data();
    char *dst_data = dst.get_data();

    // handle contiguous inputs
    const bool is_src_c_contig = src.is_c_contiguous();
    const bool is_src_f_contig = src.is_f_contiguous();

    const bool is_dst_c_contig = dst.is_c_contiguous();
    const bool is_dst_f_contig = dst.is_f_contiguous();

    const bool both_c_contig = (is_src_c_contig && is_dst_c_contig);
    const bool both_f_contig = (is_src_f_contig && is_dst_f_contig);

    if (both_c_contig || both_f_contig) {
        auto contig_fn = nan_to_num_contig_dispatch_vector[src_typeid];

        if (contig_fn == nullptr) {
            throw std::runtime_error(
                "Contiguous implementation is missing for src_typeid=" +
                std::to_string(src_typeid));
        }

        auto comp_ev = contig_fn(q, nelems, py_nan, py_posinf, py_neginf,
                                 src_data, dst_data, depends);
        sycl::event ht_ev =
            dpctl::utils::keep_args_alive(q, {src, dst}, {comp_ev});

        return std::make_pair(ht_ev, comp_ev);
    }

    // simplify iteration space
    //     if 1d with strides 1 - input is contig
    //     dispatch to strided

    auto const &src_strides = src.get_strides_vector();
    auto const &dst_strides = dst.get_strides_vector();

    using shT = std::vector<py::ssize_t>;
    shT simplified_shape;
    shT simplified_src_strides;
    shT simplified_dst_strides;
    py::ssize_t src_offset(0);
    py::ssize_t dst_offset(0);

    int nd = src_nd;
    const py::ssize_t *shape = src_shape;

    py_internal::simplify_iteration_space(
        nd, shape, src_strides, dst_strides,
        // output
        simplified_shape, simplified_src_strides, simplified_dst_strides,
        src_offset, dst_offset);

    if (nd == 1 && simplified_src_strides[0] == 1 &&
        simplified_dst_strides[0] == 1) {
        // Special case of contiguous data
        auto contig_fn = nan_to_num_contig_dispatch_vector[src_typeid];

        if (contig_fn == nullptr) {
            throw std::runtime_error(
                "Contiguous implementation is missing for src_typeid=" +
                std::to_string(src_typeid));
        }

        int src_elem_size = src.get_elemsize();
        int dst_elem_size = dst.get_elemsize();
        auto comp_ev =
            contig_fn(q, nelems, py_nan, py_posinf, py_neginf,
                      src_data + src_elem_size * src_offset,
                      dst_data + dst_elem_size * dst_offset, depends);

        sycl::event ht_ev =
            dpctl::utils::keep_args_alive(q, {src, dst}, {comp_ev});

        return std::make_pair(ht_ev, comp_ev);
    }

    auto fn = nan_to_num_dispatch_vector[src_typeid];

    if (fn == nullptr) {
        throw std::runtime_error(
            "nan_to_num implementation is missing for src_typeid=" +
            std::to_string(src_typeid));
    }

    using dpctl::tensor::offset_utils::device_allocate_and_pack;

    std::vector<sycl::event> host_tasks{};
    host_tasks.reserve(2);

    auto ptr_size_event_triple_ = device_allocate_and_pack<py::ssize_t>(
        q, host_tasks, simplified_shape, simplified_src_strides,
        simplified_dst_strides);
    auto shape_strides_owner = std::move(std::get<0>(ptr_size_event_triple_));
    const sycl::event &copy_shape_ev = std::get<2>(ptr_size_event_triple_);
    const py::ssize_t *shape_strides = shape_strides_owner.get();

    std::vector<sycl::event> all_deps;
    all_deps.reserve(depends.size() + 1);
    all_deps.insert(all_deps.end(), depends.begin(), depends.end());
    all_deps.push_back(copy_shape_ev);

    sycl::event comp_ev =
        fn(q, nelems, nd, shape_strides, py_nan, py_posinf, py_neginf, src_data,
           src_offset, dst_data, dst_offset, all_deps);

    // async free of shape_strides temporary
    sycl::event tmp_cleanup_ev = dpctl::tensor::alloc_utils::async_smart_free(
        q, {comp_ev}, shape_strides_owner);

    host_tasks.push_back(tmp_cleanup_ev);

    return std::make_pair(
        dpctl::utils::keep_args_alive(q, {src, dst}, host_tasks), comp_ev);
}

/**
 * @brief A factory to define pairs of supported types for which
 * nan-to-num function is available.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct NanToNumOutputType
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
struct NanToNumFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename NanToNumOutputType<T>::value_type,
                                     void>) {
            return nullptr;
        }
        else {
            return nan_to_num_strided_call<T>;
        }
    }
};

template <typename fnT, typename T>
struct NanToNumContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename NanToNumOutputType<T>::value_type,
                                     void>) {
            return nullptr;
        }
        else {
            return nan_to_num_contig_call<T>;
        }
    }
};

static void populate_nan_to_num_dispatch_vectors(void)
{
    init_dispatch_vector<nan_to_num_fn_ptr_t, NanToNumFactory>(
        nan_to_num_dispatch_vector);
    init_dispatch_vector<nan_to_num_contig_fn_ptr_t, NanToNumContigFactory>(
        nan_to_num_contig_dispatch_vector);
}

} // namespace impl

void init_nan_to_num(py::module_ m)
{
    {
        impl::populate_nan_to_num_dispatch_vectors();

        using impl::py_nan_to_num;
        m.def("_nan_to_num", &py_nan_to_num, "", py::arg("src"),
              py::arg("py_nan"), py::arg("py_posinf"), py::arg("py_neginf"),
              py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
    }
}

} // namespace dpnp::extensions::ufunc
