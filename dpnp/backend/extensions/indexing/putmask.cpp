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
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dpnp4pybind11.hpp"

#include "kernels/indexing/putmask.hpp"

// dpnp tensor headers
#include "utils/offset_utils.hpp"
#include "utils/output_validation.hpp"
#include "utils/type_dispatch.hpp"

// utils extension headers
#include "ext/common.hpp"
#include "ext/validation_utils.hpp"

namespace py = pybind11;
namespace td_ns = dpnp::tensor::type_dispatch;

using dpnp::tensor::usm_ndarray;

using ext::common::dtype_from_typenum;
using ext::validation::array_names;
using ext::validation::check_c_contig;
using ext::validation::check_has_dtype;
using ext::validation::check_num_dims;
using ext::validation::check_queue;
using ext::validation::check_same_dtype;
using ext::validation::check_same_size;
using ext::validation::check_writable;

namespace dpnp::extensions::indexing
{
using ext::common::init_dispatch_vector;

typedef sycl::event (*putmask_strided_fn_ptr_t)(
    sycl::queue &,
    const int,           // nd
    const std::size_t,   // nelems
    const py::ssize_t *, // shape_strides
    char *,              // dst
    py::ssize_t,         // dst_offset
    const char *,        // mask
    py::ssize_t,         // mask_offset
    const char *,        // values
    const std::size_t,   // values_size
    const std::vector<sycl::event> &);

template <typename T>
sycl::event putmask_strided_call(sycl::queue &q,
                                 const int nd,
                                 const std::size_t nelems,
                                 const py::ssize_t *shape_strides,
                                 char *dst_p,
                                 py::ssize_t dst_offset,
                                 const char *mask_p,
                                 py::ssize_t mask_offset,
                                 const char *values_p,
                                 const std::size_t values_size,
                                 const std::vector<sycl::event> &depends)
{
    return dpnp::kernels::putmask::putmask_strided_impl<T>(
        q, nd, nelems, shape_strides, dst_p, dst_offset, mask_p, mask_offset,
        values_p, values_size, depends);
}

typedef sycl::event (*putmask_contig_fn_ptr_t)(
    sycl::queue &,
    const std::size_t, // nelems
    char *,            // dst
    const char *,      // mask
    const char *,      // values
    const std::size_t, // values_size
    const std::vector<sycl::event> &);

template <typename T>
sycl::event putmask_contig_call(sycl::queue &q,
                                const std::size_t nelems,
                                char *dst_p,
                                const char *mask_p,
                                const char *values_p,
                                const std::size_t values_size,
                                const std::vector<sycl::event> &depends)
{
    return dpnp::kernels::putmask::putmask_contig_impl<T>(
        q, nelems, dst_p, mask_p, values_p, values_size, depends);
}

putmask_strided_fn_ptr_t putmask_strided_dispatch_vector[td_ns::num_types];
putmask_contig_fn_ptr_t putmask_contig_dispatch_vector[td_ns::num_types];

std::pair<sycl::event, sycl::event>
    py_putmask(const usm_ndarray &dst,
               const usm_ndarray &mask,
               const usm_ndarray &values,
               sycl::queue &exec_q,
               const std::vector<sycl::event> &depends = {})
{
    array_names names = {{&dst, "dst"}, {&mask, "mask"}, {&values, "values"}};

    check_same_dtype(&dst, &values, names);
    check_has_dtype(&mask, td_ns::typenum_t::BOOL, names);

    check_same_size({&dst, &mask}, names);
    const int nd = dst.get_ndim();
    check_num_dims({&mask}, nd, names);

    check_queue({&dst, &mask, &values}, names, exec_q);
    check_writable({&dst}, names);

    // values must be C-contiguous
    check_c_contig({&values}, names);

    auto types = td_ns::usm_ndarray_types();
    // dst_typeid == values_typeid (check_same_dtype(&dst, &values, names))
    int dst_values_typeid = types.typenum_to_lookup_id(dst.get_typenum());

    const py::ssize_t *dst_shape = dst.get_shape_raw();
    const py::ssize_t *mask_shape = mask.get_shape_raw();
    bool shapes_equal(true);
    std::size_t nelems(1);

    for (int i = 0; i < std::max(nd, 1); ++i) {
        const py::ssize_t d = (nd == 0 ? 1 : dst_shape[i]);
        const py::ssize_t m = (nd == 0 ? 1 : mask_shape[i]);
        nelems *= static_cast<std::size_t>(d);
        shapes_equal = shapes_equal && (d == m);
    }
    if (!shapes_equal) {
        throw py::value_error("`mask` and `dst` shapes must match");
    }

    const std::size_t values_size = values.get_size();

    // empty output or empty `values` is a no-op
    if (nelems == 0 || values_size == 0) {
        return {sycl::event(), sycl::event()};
    }

    dpnp::tensor::validation::AmpleMemory::throw_if_not_ample(dst, nelems);

    char *dst_p = dst.get_data();
    const char *mask_p = mask.get_data();
    const char *values_p = values.get_data();

    // the contig kernel cycles `values` by the memory-linear index, which
    // matches numpy's C-order `values.flat` only for C-contiguous data
    const bool all_c_contig = dst.is_c_contiguous() && mask.is_c_contiguous() &&
                              values.is_c_contiguous();

    if (all_c_contig) {
        auto contig_fn = putmask_contig_dispatch_vector[dst_values_typeid];

        if (contig_fn == nullptr) {
            py::dtype dst_values_dtype_py =
                dtype_from_typenum(dst_values_typeid);
            throw std::runtime_error(
                "Contiguous implementation is missing for " +
                std::string(py::str(dst_values_dtype_py)) + " data type");
        }

        auto comp_ev = contig_fn(exec_q, nelems, dst_p, mask_p, values_p,
                                 values_size, depends);
        sycl::event ht_ev = dpnp::utils::keep_args_alive(
            exec_q, {dst, mask, values}, {comp_ev});

        return std::make_pair(ht_ev, comp_ev);
    }

    // strided path: the iteration space is intentionally not simplified, so
    // the kernel's linear index stays equal to the C-order flat index used to
    // cycle `values` (simplify_iteration_space may reorder axes and break it)
    const auto &dst_strides = dst.get_strides_vector();
    const auto &mask_strides = mask.get_strides_vector();

    using shT = std::vector<py::ssize_t>;
    shT common_shape;
    shT s_dst_strides;
    shT s_mask_strides;

    int eff_nd = nd;
    if (nd == 0) {
        // scalar arrays: single-element 1D iteration
        eff_nd = 1;
        common_shape = {1};
        s_dst_strides = {0};
        s_mask_strides = {0};
    }
    else {
        common_shape.assign(dst_shape, dst_shape + nd);
        s_dst_strides = dst_strides;
        s_mask_strides = mask_strides;
    }

    // trivial offsets: shape and strides are passed without simplification
    constexpr py::ssize_t dst_off = 0;
    constexpr py::ssize_t mask_off = 0;

    auto strided_fn = putmask_strided_dispatch_vector[dst_values_typeid];
    if (strided_fn == nullptr) {
        py::dtype dt = dtype_from_typenum(dst_values_typeid);
        throw std::runtime_error("Strided implementation is missing for " +
                                 std::string(py::str(dt)) + " data type");
    }

    using dpnp::tensor::offset_utils::device_allocate_and_pack;

    std::vector<sycl::event> host_tasks;
    host_tasks.reserve(2);

    auto pack = device_allocate_and_pack<py::ssize_t>(
        exec_q, host_tasks, common_shape, s_dst_strides, s_mask_strides);

    auto shape_strides_owner = std::move(std::get<0>(pack));
    const py::ssize_t *shape_strides_dev = shape_strides_owner.get();
    const sycl::event &cpy_ev = std::get<2>(pack);

    std::vector<sycl::event> all_deps = depends;
    all_deps.push_back(cpy_ev);

    sycl::event comp_ev =
        strided_fn(exec_q, eff_nd, nelems, shape_strides_dev, dst_p, dst_off,
                   mask_p, mask_off, values_p, values_size, all_deps);

    sycl::event cleanup_ev = dpnp::tensor::alloc_utils::async_smart_free(
        exec_q, {comp_ev}, shape_strides_owner);
    host_tasks.push_back(cleanup_ev);

    sycl::event ht_ev =
        dpnp::utils::keep_args_alive(exec_q, {dst, mask, values}, host_tasks);

    return std::make_pair(ht_ev, comp_ev);
}

/**
 * @brief A factory to define pairs of supported types for which
 * putmask function is available.
 *
 * @tparam T Type of input vector `dst` and `values` and of result vector `dst`.
 */
template <typename T>
struct PutMaskOutputType
{
    using value_type = typename std::disjunction<
        td_ns::TypeMapResultEntry<T, bool>,
        td_ns::TypeMapResultEntry<T, std::uint8_t>,
        td_ns::TypeMapResultEntry<T, std::int8_t>,
        td_ns::TypeMapResultEntry<T, std::uint16_t>,
        td_ns::TypeMapResultEntry<T, std::int16_t>,
        td_ns::TypeMapResultEntry<T, std::uint32_t>,
        td_ns::TypeMapResultEntry<T, std::int32_t>,
        td_ns::TypeMapResultEntry<T, std::uint64_t>,
        td_ns::TypeMapResultEntry<T, std::int64_t>,
        td_ns::TypeMapResultEntry<T, sycl::half>,
        td_ns::TypeMapResultEntry<T, float>,
        td_ns::TypeMapResultEntry<T, double>,
        td_ns::TypeMapResultEntry<T, std::complex<float>>,
        td_ns::TypeMapResultEntry<T, std::complex<double>>,
        td_ns::DefaultResultEntry<void>>::result_type;
};

template <typename fnT, typename T>
struct PutMaskStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename PutMaskOutputType<T>::value_type,
                                     void>) {
            return nullptr;
        }
        else {
            return putmask_strided_call<T>;
        }
    }
};

template <typename fnT, typename T>
struct PutMaskContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename PutMaskOutputType<T>::value_type,
                                     void>) {
            return nullptr;
        }
        else {
            return putmask_contig_call<T>;
        }
    }
};

static void populate_putmask_dispatch_vectors()
{
    init_dispatch_vector<putmask_strided_fn_ptr_t, PutMaskStridedFactory>(
        putmask_strided_dispatch_vector);
    init_dispatch_vector<putmask_contig_fn_ptr_t, PutMaskContigFactory>(
        putmask_contig_dispatch_vector);
}

void init_putmask(py::module_ &m)
{
    populate_putmask_dispatch_vectors();

    m.def("_putmask", &py_putmask, "", py::arg("dst"), py::arg("mask"),
          py::arg("values"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());

    return;
}

} // namespace dpnp::extensions::indexing
