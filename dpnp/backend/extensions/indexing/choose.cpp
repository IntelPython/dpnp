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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dpnp4pybind11.hpp"

#include "choose_kernel.hpp"

// utils extension header
#include "ext/common.hpp"

// dpctl tensor headers
#include "utils/indexing_utils.hpp"
#include "utils/memory_overlap.hpp"
#include "utils/output_validation.hpp"
#include "utils/sycl_alloc_utils.hpp"
#include "utils/type_dispatch.hpp"

namespace dpnp::extensions::indexing
{
namespace td_ns = dpctl::tensor::type_dispatch;

static kernels::choose_fn_ptr_t choose_clip_dispatch_table[td_ns::num_types]
                                                          [td_ns::num_types];
static kernels::choose_fn_ptr_t choose_wrap_dispatch_table[td_ns::num_types]
                                                          [td_ns::num_types];

namespace py = pybind11;

namespace detail
{

using host_ptrs_allocator_t =
    dpctl::tensor::alloc_utils::usm_host_allocator<char *>;
using ptrs_t = std::vector<char *, host_ptrs_allocator_t>;
using host_ptrs_shp_t = std::shared_ptr<ptrs_t>;

host_ptrs_shp_t make_host_ptrs(sycl::queue &exec_q,
                               const std::vector<char *> &ptrs)
{
    host_ptrs_allocator_t ptrs_allocator(exec_q);
    host_ptrs_shp_t host_ptrs_shp =
        std::make_shared<ptrs_t>(ptrs.size(), ptrs_allocator);

    std::copy(ptrs.begin(), ptrs.end(), host_ptrs_shp->begin());

    return host_ptrs_shp;
}

using host_sz_allocator_t =
    dpctl::tensor::alloc_utils::usm_host_allocator<py::ssize_t>;
using sz_t = std::vector<py::ssize_t, host_sz_allocator_t>;
using host_sz_shp_t = std::shared_ptr<sz_t>;

host_sz_shp_t make_host_offsets(sycl::queue &exec_q,
                                const std::vector<py::ssize_t> &offsets)
{
    host_sz_allocator_t offsets_allocator(exec_q);
    host_sz_shp_t host_offsets_shp =
        std::make_shared<sz_t>(offsets.size(), offsets_allocator);

    std::copy(offsets.begin(), offsets.end(), host_offsets_shp->begin());

    return host_offsets_shp;
}

host_sz_shp_t make_host_shape_strides(sycl::queue &exec_q,
                                      py::ssize_t n_chcs,
                                      std::vector<py::ssize_t> &shape,
                                      std::vector<py::ssize_t> &inp_strides,
                                      std::vector<py::ssize_t> &dst_strides,
                                      std::vector<py::ssize_t> &chc_strides)
{
    auto nelems = shape.size();
    host_sz_allocator_t shape_strides_allocator(exec_q);
    host_sz_shp_t host_shape_strides_shp =
        std::make_shared<sz_t>(nelems * (3 + n_chcs), shape_strides_allocator);

    std::copy(shape.begin(), shape.end(), host_shape_strides_shp->begin());
    std::copy(inp_strides.begin(), inp_strides.end(),
              host_shape_strides_shp->begin() + nelems);
    std::copy(dst_strides.begin(), dst_strides.end(),
              host_shape_strides_shp->begin() + 2 * nelems);
    std::copy(chc_strides.begin(), chc_strides.end(),
              host_shape_strides_shp->begin() + 3 * nelems);

    return host_shape_strides_shp;
}

/* This function expects a queue and a non-trivial number of
   std::pairs of raw device pointers and host shared pointers
   (structured as <device_ptr, shared_ptr>),
   then enqueues a copy of the host shared pointer data into
   the device pointer.

   Assumes the device pointer addresses sufficient memory for
   the size of the host memory.
*/
template <typename... DevHostPairs>
std::vector<sycl::event> batched_copy(sycl::queue &exec_q,
                                      DevHostPairs &&...dev_host_pairs)
{
    constexpr std::size_t n = sizeof...(DevHostPairs);
    static_assert(n > 0, "batched_copy requires at least one argument");

    std::vector<sycl::event> copy_evs;
    copy_evs.reserve(n);
    (copy_evs.emplace_back(exec_q.copy(dev_host_pairs.second->data(),
                                       dev_host_pairs.first,
                                       dev_host_pairs.second->size())),
     ...);

    return copy_evs;
}

/* This function takes as input a queue, sycl::event dependencies,
   and a non-trivial number of shared_ptrs and moves them into
   a host_task lambda capture, ensuring their lifetime until the
   host_task executes.
*/
template <typename... Shps>
sycl::event async_shp_free(sycl::queue &exec_q,
                           const std::vector<sycl::event> &depends,
                           Shps &&...shps)
{
    constexpr std::size_t n = sizeof...(Shps);
    static_assert(n > 0, "async_shp_free requires at least one argument");

    const sycl::event &shared_ptr_cleanup_ev =
        exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);
            cgh.host_task([capture = std::tuple(std::move(shps)...)]() {});
        });

    return shared_ptr_cleanup_ev;
}

// copied from dpctl, remove if a similar utility is ever exposed
std::vector<dpctl::tensor::usm_ndarray> parse_py_chcs(const sycl::queue &q,
                                                      const py::object &py_chcs)
{
    py::ssize_t chc_count = py::len(py_chcs);
    std::vector<dpctl::tensor::usm_ndarray> res;
    res.reserve(chc_count);

    for (py::ssize_t i = 0; i < chc_count; ++i) {
        py::object el_i = py_chcs[py::cast(i)];
        dpctl::tensor::usm_ndarray arr_i =
            py::cast<dpctl::tensor::usm_ndarray>(el_i);
        if (!dpctl::utils::queues_are_compatible(q, {arr_i})) {
            throw py::value_error("Choice allocation queue is not compatible "
                                  "with execution queue");
        }
        res.push_back(arr_i);
    }

    return res;
}

} // namespace detail

std::pair<sycl::event, sycl::event>
    py_choose(const dpctl::tensor::usm_ndarray &src,
              const py::object &py_chcs,
              const dpctl::tensor::usm_ndarray &dst,
              uint8_t mode,
              sycl::queue &exec_q,
              const std::vector<sycl::event> &depends)
{
    std::vector<dpctl::tensor::usm_ndarray> chcs =
        detail::parse_py_chcs(exec_q, py_chcs);

    // Python list max size must fit into py_ssize_t
    py::ssize_t n_chcs = chcs.size();

    if (n_chcs == 0) {
        throw py::value_error("List of choices is empty.");
    }

    if (mode != 0 && mode != 1) {
        throw py::value_error("Mode must be 0 or 1.");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    const dpctl::tensor::usm_ndarray chc_rep = chcs[0];

    int nd = src.get_ndim();
    int dst_nd = dst.get_ndim();
    int chc_nd = chc_rep.get_ndim();

    if (nd != dst_nd || nd != chc_nd) {
        throw py::value_error("Array shapes are not consistent");
    }

    const py::ssize_t *src_shape = src.get_shape_raw();
    const py::ssize_t *dst_shape = dst.get_shape_raw();
    const py::ssize_t *chc_shape = chc_rep.get_shape_raw();

    size_t nelems = src.get_size();
    bool shapes_equal = std::equal(src_shape, src_shape + nd, dst_shape);
    shapes_equal &= std::equal(src_shape, src_shape + nd, chc_shape);

    if (!shapes_equal) {
        throw py::value_error("Array shapes don't match.");
    }

    if (nelems == 0) {
        return std::make_pair(sycl::event{}, sycl::event{});
    }

    char *src_data = src.get_data();
    char *dst_data = dst.get_data();

    if (!dpctl::utils::queues_are_compatible(exec_q, {src, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(src, dst)) {
        throw py::value_error("Array memory overlap.");
    }

    // trivial offsets as choose does not apply stride
    // simplification, but may in the future
    constexpr py::ssize_t src_offset = py::ssize_t(0);
    constexpr py::ssize_t dst_offset = py::ssize_t(0);

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();
    int chc_typenum = chc_rep.get_typenum();

    auto array_types = td_ns::usm_ndarray_types();
    int src_type_id = array_types.typenum_to_lookup_id(src_typenum);
    int dst_type_id = array_types.typenum_to_lookup_id(dst_typenum);
    int chc_type_id = array_types.typenum_to_lookup_id(chc_typenum);

    if (chc_type_id != dst_type_id) {
        throw py::type_error("Output and choice data types are not the same.");
    }

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(dst, nelems);

    std::vector<char *> chc_ptrs;
    chc_ptrs.reserve(n_chcs);

    std::vector<py::ssize_t> chc_offsets;
    chc_offsets.reserve(n_chcs);

    auto sh_nelems = std::max<int>(nd, 1);
    std::vector<py::ssize_t> chc_strides(n_chcs * sh_nelems, 0);

    for (auto i = 0; i < n_chcs; ++i) {
        dpctl::tensor::usm_ndarray chc_ = chcs[i];

        // ndim, type, and shape are checked against the first array
        if (i > 0) {
            if (!(chc_.get_ndim() == nd)) {
                throw py::value_error(
                    "Choice array dimensions are not the same");
            }

            if (!(chc_type_id ==
                  array_types.typenum_to_lookup_id(chc_.get_typenum()))) {
                throw py::type_error(
                    "Choice array data types are not all the same.");
            }

            const py::ssize_t *chc_shape_ = chc_.get_shape_raw();
            if (!std::equal(chc_shape_, chc_shape_ + nd, chc_shape)) {
                throw py::value_error("Choice shapes are not all equal.");
            }
        }

        // check for overlap with destination
        if (overlap(dst, chc_)) {
            throw py::value_error(
                "Arrays index overlapping segments of memory");
        }

        char *chc_data = chc_.get_data();

        if (nd > 0) {
            auto chc_strides_ = chc_.get_strides_vector();
            std::copy(chc_strides_.begin(), chc_strides_.end(),
                      chc_strides.begin() + i * nd);
        }

        chc_ptrs.push_back(chc_data);
        chc_offsets.push_back(py::ssize_t(0));
    }

    auto fn = mode ? choose_clip_dispatch_table[src_type_id][chc_type_id]
                   : choose_wrap_dispatch_table[src_type_id][chc_type_id];

    if (fn == nullptr) {
        throw std::runtime_error("Indices must be integer type, got " +
                                 std::to_string(src_type_id));
    }

    auto packed_chc_ptrs =
        dpctl::tensor::alloc_utils::smart_malloc_device<char *>(n_chcs, exec_q);

    // packed_shapes_strides = [common shape,
    //                          src.strides,
    //                          dst.strides,
    //                          chcs[0].strides,
    //                          ...,
    //                          chcs[n_chcs].strides]
    auto packed_shapes_strides =
        dpctl::tensor::alloc_utils::smart_malloc_device<py::ssize_t>(
            (3 + n_chcs) * sh_nelems, exec_q);

    auto packed_chc_offsets =
        dpctl::tensor::alloc_utils::smart_malloc_device<py::ssize_t>(n_chcs,
                                                                     exec_q);

    std::vector<sycl::event> host_task_events;
    host_task_events.reserve(2);

    std::vector<sycl::event> pack_deps;
    std::vector<py::ssize_t> common_shape;
    std::vector<py::ssize_t> src_strides;
    std::vector<py::ssize_t> dst_strides;
    if (nd == 0) {
        // special case where all inputs are scalars
        // need to pass src, dst shape=1 and strides=0
        // chc_strides already initialized to 0 so ignore
        common_shape = {1};
        src_strides = {0};
        dst_strides = {0};
    }
    else {
        common_shape = src.get_shape_vector();
        src_strides = src.get_strides_vector();
        dst_strides = dst.get_strides_vector();
    }

    auto host_chc_ptrs = detail::make_host_ptrs(exec_q, chc_ptrs);
    auto host_chc_offsets = detail::make_host_offsets(exec_q, chc_offsets);
    auto host_shape_strides = detail::make_host_shape_strides(
        exec_q, n_chcs, common_shape, src_strides, dst_strides, chc_strides);

    pack_deps = detail::batched_copy(
        exec_q, std::make_pair(packed_chc_ptrs.get(), host_chc_ptrs),
        std::make_pair(packed_chc_offsets.get(), host_chc_offsets),
        std::make_pair(packed_shapes_strides.get(), host_shape_strides));

    host_task_events.push_back(
        detail::async_shp_free(exec_q, pack_deps, host_chc_ptrs,
                               host_chc_offsets, host_shape_strides));

    std::vector<sycl::event> all_deps;
    all_deps.reserve(depends.size() + pack_deps.size());
    all_deps.insert(std::end(all_deps), std::begin(pack_deps),
                    std::end(pack_deps));
    all_deps.insert(std::end(all_deps), std::begin(depends), std::end(depends));

    sycl::event choose_generic_ev =
        fn(exec_q, nelems, n_chcs, sh_nelems, packed_shapes_strides.get(),
           src_data, dst_data, packed_chc_ptrs.get(), src_offset, dst_offset,
           packed_chc_offsets.get(), all_deps);

    // async_smart_free releases owners
    sycl::event temporaries_cleanup_ev =
        dpctl::tensor::alloc_utils::async_smart_free(
            exec_q, {choose_generic_ev}, packed_chc_ptrs, packed_shapes_strides,
            packed_chc_offsets);

    host_task_events.push_back(temporaries_cleanup_ev);

    using dpctl::utils::keep_args_alive;
    sycl::event arg_cleanup_ev =
        keep_args_alive(exec_q, {src, py_chcs, dst}, host_task_events);

    return std::make_pair(arg_cleanup_ev, choose_generic_ev);
}

template <typename fnT, typename IndT, typename T, typename Index>
struct ChooseFactory
{
    fnT get()
    {
        if constexpr (std::is_integral<IndT>::value &&
                      !std::is_same<IndT, bool>::value) {
            fnT fn = kernels::choose_impl<Index, IndT, T>;
            return fn;
        }
        else {
            fnT fn = nullptr;
            return fn;
        }
    }
};

using dpctl::tensor::indexing_utils::ClipIndex;
using dpctl::tensor::indexing_utils::WrapIndex;

template <typename fnT, typename IndT, typename T>
using ChooseWrapFactory = ChooseFactory<fnT, IndT, T, WrapIndex<IndT>>;

template <typename fnT, typename IndT, typename T>
using ChooseClipFactory = ChooseFactory<fnT, IndT, T, ClipIndex<IndT>>;

void init_choose_dispatch_tables(void)
{
    using ext::common::init_dispatch_table;
    using kernels::choose_fn_ptr_t;

    init_dispatch_table<choose_fn_ptr_t, ChooseClipFactory>(
        choose_clip_dispatch_table);
    init_dispatch_table<choose_fn_ptr_t, ChooseWrapFactory>(
        choose_wrap_dispatch_table);
}

void init_choose(py::module_ m)
{
    dpnp::extensions::indexing::init_choose_dispatch_tables();

    m.def("_choose", &py_choose, "", py::arg("src"), py::arg("chcs"),
          py::arg("dst"), py::arg("mode"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());

    return;
}
} // namespace dpnp::extensions::indexing
