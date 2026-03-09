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
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines functions of dpctl.tensor._tensor_sorting_impl
/// extension.
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kernels/sorting/searchsorted.hpp"
#include "utils/memory_overlap.hpp"
#include "utils/offset_utils.hpp"
#include "utils/output_validation.hpp"
#include "utils/rich_comparisons.hpp"
#include "utils/sycl_alloc_utils.hpp"
#include "utils/type_dispatch.hpp"

#include "simplify_iteration_space.hpp"

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

namespace dpctl::tensor::py_internal
{

namespace detail
{

using dpctl::tensor::kernels::searchsorted_contig_impl_fp_ptr_t;

static searchsorted_contig_impl_fp_ptr_t
    left_side_searchsorted_contig_impl[td_ns::num_types][td_ns::num_types];

static searchsorted_contig_impl_fp_ptr_t
    right_side_searchsorted_contig_impl[td_ns::num_types][td_ns::num_types];

template <typename fnT, typename argTy, typename indTy>
struct LeftSideSearchSortedContigFactory
{
    constexpr LeftSideSearchSortedContigFactory() {}

    fnT get() const
    {
        if constexpr (std::is_same_v<indTy, std::int32_t> ||
                      std::is_same_v<indTy, std::int64_t>)
        {
            static constexpr bool left_side_search(true);
            using dpctl::tensor::kernels::searchsorted_contig_impl;
            using dpctl::tensor::rich_comparisons::AscendingSorter;

            using Compare = typename AscendingSorter<argTy>::type;

            return searchsorted_contig_impl<argTy, indTy, left_side_search,
                                            Compare>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename argTy, typename indTy>
struct RightSideSearchSortedContigFactory
{
    constexpr RightSideSearchSortedContigFactory() {}

    fnT get() const
    {
        if constexpr (std::is_same_v<indTy, std::int32_t> ||
                      std::is_same_v<indTy, std::int64_t>)
        {
            static constexpr bool right_side_search(false);

            using dpctl::tensor::kernels::searchsorted_contig_impl;
            using dpctl::tensor::rich_comparisons::AscendingSorter;

            using Compare = typename AscendingSorter<argTy>::type;

            return searchsorted_contig_impl<argTy, indTy, right_side_search,
                                            Compare>;
        }
        else {
            return nullptr;
        }
    }
};

using dpctl::tensor::kernels::searchsorted_strided_impl_fp_ptr_t;

static searchsorted_strided_impl_fp_ptr_t
    left_side_searchsorted_strided_impl[td_ns::num_types][td_ns::num_types];

static searchsorted_strided_impl_fp_ptr_t
    right_side_searchsorted_strided_impl[td_ns::num_types][td_ns::num_types];

template <typename fnT, typename argTy, typename indTy>
struct LeftSideSearchSortedStridedFactory
{
    constexpr LeftSideSearchSortedStridedFactory() {}

    fnT get() const
    {
        if constexpr (std::is_same_v<indTy, std::int32_t> ||
                      std::is_same_v<indTy, std::int64_t>)
        {
            static constexpr bool left_side_search(true);
            using dpctl::tensor::kernels::searchsorted_strided_impl;
            using dpctl::tensor::rich_comparisons::AscendingSorter;

            using Compare = typename AscendingSorter<argTy>::type;

            return searchsorted_strided_impl<argTy, indTy, left_side_search,
                                             Compare>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename argTy, typename indTy>
struct RightSideSearchSortedStridedFactory
{
    constexpr RightSideSearchSortedStridedFactory() {}

    fnT get() const
    {
        if constexpr (std::is_same_v<indTy, std::int32_t> ||
                      std::is_same_v<indTy, std::int64_t>)
        {
            static constexpr bool right_side_search(false);
            using dpctl::tensor::kernels::searchsorted_strided_impl;
            using dpctl::tensor::rich_comparisons::AscendingSorter;

            using Compare = typename AscendingSorter<argTy>::type;

            return searchsorted_strided_impl<argTy, indTy, right_side_search,
                                             Compare>;
        }
        else {
            return nullptr;
        }
    }
};

void init_searchsorted_dispatch_table(void)
{

    // Contiguous input function dispatch
    td_ns::DispatchTableBuilder<searchsorted_contig_impl_fp_ptr_t,
                                LeftSideSearchSortedContigFactory,
                                td_ns::num_types>
        dtb1;
    dtb1.populate_dispatch_table(left_side_searchsorted_contig_impl);

    td_ns::DispatchTableBuilder<searchsorted_contig_impl_fp_ptr_t,
                                RightSideSearchSortedContigFactory,
                                td_ns::num_types>
        dtb2;
    dtb2.populate_dispatch_table(right_side_searchsorted_contig_impl);

    // Strided input function dispatch
    td_ns::DispatchTableBuilder<searchsorted_strided_impl_fp_ptr_t,
                                LeftSideSearchSortedStridedFactory,
                                td_ns::num_types>
        dtb3;
    dtb3.populate_dispatch_table(left_side_searchsorted_strided_impl);

    td_ns::DispatchTableBuilder<searchsorted_strided_impl_fp_ptr_t,
                                RightSideSearchSortedStridedFactory,
                                td_ns::num_types>
        dtb4;
    dtb4.populate_dispatch_table(right_side_searchsorted_strided_impl);
}

} // namespace detail

/*! @brief search for needle from needles in sorted hay */
std::pair<sycl::event, sycl::event>
    py_searchsorted(const dpctl::tensor::usm_ndarray &hay,
                    const dpctl::tensor::usm_ndarray &needles,
                    const dpctl::tensor::usm_ndarray &positions,
                    sycl::queue &exec_q,
                    const bool search_left_side,
                    const std::vector<sycl::event> &depends)
{
    const int hay_nd = hay.get_ndim();
    const int needles_nd = needles.get_ndim();
    const int positions_nd = positions.get_ndim();

    if (hay_nd != 1 || needles_nd != positions_nd) {
        throw py::value_error("Array dimensions mismatch");
    }

    // check that needle and positions have the same shape
    std::size_t needles_nelems(1);
    bool same_shape(true);

    const std::size_t hay_nelems = static_cast<std::size_t>(hay.get_shape(0));

    const py::ssize_t *needles_shape_ptr = needles.get_shape_raw();
    const py::ssize_t *positions_shape_ptr = positions.get_shape_raw();

    for (int i = 0; (i < needles_nd) && same_shape; ++i) {
        const auto needles_sh_i = needles_shape_ptr[i];
        const auto positions_sh_i = positions_shape_ptr[i];

        same_shape = same_shape && (needles_sh_i == positions_sh_i);
        needles_nelems *= static_cast<std::size_t>(needles_sh_i);
    }

    if (!same_shape) {
        throw py::value_error(
            "Array of values to search for and array of their "
            "positions do not have the same shape");
    }

    // check that positions is ample enough
    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(positions,
                                                               needles_nelems);

    // check that positions is writable
    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(positions);

    // check that queues are compatible
    if (!dpctl::utils::queues_are_compatible(exec_q, {hay, needles, positions}))
    {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    // if output array overlaps with input arrays, race condition results
    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(positions, hay) || overlap(positions, needles)) {
        throw py::value_error("Destination array overlaps with input.");
    }

    const int hay_typenum = hay.get_typenum();
    const int needles_typenum = needles.get_typenum();
    const int positions_typenum = positions.get_typenum();

    auto const &array_types = td_ns::usm_ndarray_types();
    const int hay_typeid = array_types.typenum_to_lookup_id(hay_typenum);
    const int needles_typeid =
        array_types.typenum_to_lookup_id(needles_typenum);
    const int positions_typeid =
        array_types.typenum_to_lookup_id(positions_typenum);

    // check hay and needle have the same data-type
    if (needles_typeid != hay_typeid) {
        throw py::value_error(
            "Hay array and needles array must have the same data types");
    }
    // check that positions has indexing data-type (int32, or int64)
    const auto positions_typenum_t_v =
        static_cast<td_ns::typenum_t>(positions_typeid);
    if (positions_typenum_t_v != td_ns::typenum_t::INT32 &&
        positions_typenum_t_v != td_ns::typenum_t::INT64)
    {
        throw py::value_error(
            "Positions array must have data-type int32, or int64");
    }

    if (needles_nelems == 0) {
        // Nothing to do
        return std::make_pair(sycl::event{}, sycl::event{});
    }

    // if all inputs are contiguous call contiguous implementations
    // otherwise call strided implementation
    const bool hay_is_c_contig = hay.is_c_contiguous();
    const bool hay_is_f_contig = hay.is_f_contiguous();

    const bool needles_is_c_contig = needles.is_c_contiguous();
    const bool needles_is_f_contig = needles.is_f_contiguous();

    const bool positions_is_c_contig = positions.is_c_contiguous();
    const bool positions_is_f_contig = positions.is_f_contiguous();

    const bool all_c_contig =
        (hay_is_c_contig && needles_is_c_contig && positions_is_c_contig);
    const bool all_f_contig =
        (hay_is_f_contig && needles_is_f_contig && positions_is_f_contig);

    const char *hay_data = hay.get_data();
    const char *needles_data = needles.get_data();

    char *positions_data = positions.get_data();

    if (all_c_contig || all_f_contig) {
        auto fn =
            (search_left_side)
                ? detail::left_side_searchsorted_contig_impl[hay_typeid]
                                                            [positions_typeid]
                : detail::right_side_searchsorted_contig_impl[hay_typeid]
                                                             [positions_typeid];

        if (fn) {
            static constexpr py::ssize_t zero_offset(0);

            sycl::event comp_ev =
                fn(exec_q, hay_nelems, needles_nelems, hay_data, zero_offset,
                   needles_data, zero_offset, positions_data, zero_offset,
                   depends);

            return std::make_pair(
                dpctl::utils::keep_args_alive(exec_q, {hay, needles, positions},
                                              {comp_ev}),
                comp_ev);
        }
    }

    // strided case

    const auto &needles_strides = needles.get_strides_vector();
    const auto &positions_strides = positions.get_strides_vector();

    int simplified_nd = needles_nd;

    using shT = std::vector<py::ssize_t>;
    shT simplified_common_shape;
    shT simplified_needles_strides;
    shT simplified_positions_strides;
    py::ssize_t needles_offset(0);
    py::ssize_t positions_offset(0);

    if (simplified_nd == 0) {
        // needles and positions have same nd
        simplified_nd = 1;
        simplified_common_shape.push_back(1);
        simplified_needles_strides.push_back(0);
        simplified_positions_strides.push_back(0);
    }
    else {
        simplify_iteration_space(
            // modified by reference
            simplified_nd,
            // read-only inputs
            needles_shape_ptr, needles_strides, positions_strides,
            // output, modified by reference
            simplified_common_shape, simplified_needles_strides,
            simplified_positions_strides, needles_offset, positions_offset);
    }
    std::vector<sycl::event> host_task_events;
    host_task_events.reserve(2);

    using dpctl::tensor::offset_utils::device_allocate_and_pack;

    auto ptr_size_event_tuple = device_allocate_and_pack<py::ssize_t>(
        exec_q, host_task_events,
        // vectors being packed
        simplified_common_shape, simplified_needles_strides,
        simplified_positions_strides);
    auto packed_shape_strides_owner =
        std::move(std::get<0>(ptr_size_event_tuple));
    const sycl::event &copy_shape_strides_ev =
        std::get<2>(ptr_size_event_tuple);
    const py::ssize_t *packed_shape_strides = packed_shape_strides_owner.get();

    std::vector<sycl::event> all_deps;
    all_deps.reserve(depends.size() + 1);
    all_deps.insert(all_deps.end(), depends.begin(), depends.end());
    all_deps.push_back(copy_shape_strides_ev);

    auto strided_fn =
        (search_left_side)
            ? detail::left_side_searchsorted_strided_impl[hay_typeid]
                                                         [positions_typeid]
            : detail::right_side_searchsorted_strided_impl[hay_typeid]
                                                          [positions_typeid];

    if (!strided_fn) {
        throw std::runtime_error(
            "No implementation for data types of input arrays");
    }

    static constexpr py::ssize_t zero_offset(0);
    py::ssize_t hay_step = hay.get_strides_vector()[0];

    const sycl::event &comp_ev = strided_fn(
        exec_q, hay_nelems, needles_nelems, hay_data, zero_offset, hay_step,
        needles_data, needles_offset, positions_data, positions_offset,
        simplified_nd, packed_shape_strides, all_deps);

    // free packed temporaries
    sycl::event temporaries_cleanup_ev =
        dpctl::tensor::alloc_utils::async_smart_free(
            exec_q, {comp_ev}, packed_shape_strides_owner);

    host_task_events.push_back(temporaries_cleanup_ev);
    const sycl::event &ht_ev = dpctl::utils::keep_args_alive(
        exec_q, {hay, needles, positions}, host_task_events);

    return std::make_pair(ht_ev, comp_ev);
}

/*! @brief search for needle from needles in sorted hay,
 *         hay[pos] <= needle < hay[pos + 1]
 */
std::pair<sycl::event, sycl::event>
    py_searchsorted_left(const dpctl::tensor::usm_ndarray &hay,
                         const dpctl::tensor::usm_ndarray &needles,
                         const dpctl::tensor::usm_ndarray &positions,
                         sycl::queue &exec_q,
                         const std::vector<sycl::event> &depends)
{
    static constexpr bool side_left(true);
    return py_searchsorted(hay, needles, positions, exec_q, side_left, depends);
}

/*! @brief search for needle from needles in sorted hay,
 *         hay[pos] < needle <= hay[pos + 1]
 */
std::pair<sycl::event, sycl::event>
    py_searchsorted_right(const dpctl::tensor::usm_ndarray &hay,
                          const dpctl::tensor::usm_ndarray &needles,
                          const dpctl::tensor::usm_ndarray &positions,
                          sycl::queue &exec_q,
                          const std::vector<sycl::event> &depends)
{
    static constexpr bool side_right(false);
    return py_searchsorted(hay, needles, positions, exec_q, side_right,
                           depends);
}

void init_searchsorted_functions(py::module_ m)
{
    detail::init_searchsorted_dispatch_table();

    m.def("_searchsorted_left", &py_searchsorted_left, py::arg("hay"),
          py::arg("needles"), py::arg("positions"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());
    m.def("_searchsorted_right", &py_searchsorted_right, py::arg("hay"),
          py::arg("needles"), py::arg("positions"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());
}

} // namespace dpctl::tensor::py_internal
