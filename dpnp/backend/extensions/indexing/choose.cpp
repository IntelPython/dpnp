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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sycl/sycl.hpp>
#include <type_traits>
#include <utility>
#include <vector>

#include "choose_kernel.hpp"
#include "dpctl4pybind11.hpp"
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

/*
 Returns an std::unique_ptr wrapping a USM allocation and deleter.

 Must still be manually freed by host_task when allocation is needed
 for duration of asynchronous kernel execution.
*/
template <typename T>
auto usm_unique_ptr(std::size_t sz, sycl::queue &q)
{
    using dpctl::tensor::alloc_utils::sycl_free_noexcept;
    auto deleter = [&q](T *usm) { sycl_free_noexcept(usm, q); };

    return std::unique_ptr<T, decltype(deleter)>(sycl::malloc_device<T>(sz, q),
                                                 deleter);
}

std::vector<sycl::event>
    _populate_choose_kernel_params(sycl::queue &exec_q,
                                   std::vector<sycl::event> &host_task_events,
                                   char **device_chc_ptrs,
                                   py::ssize_t *device_shape_strides,
                                   py::ssize_t *device_chc_offsets,
                                   const py::ssize_t *shape,
                                   int shape_len,
                                   std::vector<py::ssize_t> &inp_strides,
                                   std::vector<py::ssize_t> &dst_strides,
                                   std::vector<py::ssize_t> &chc_strides,
                                   std::vector<char *> &chc_ptrs,
                                   std::vector<py::ssize_t> &chc_offsets,
                                   py::ssize_t n_chcs)
{
    using ptr_host_allocator_T =
        dpctl::tensor::alloc_utils::usm_host_allocator<char *>;
    using ptrT = std::vector<char *, ptr_host_allocator_T>;

    ptr_host_allocator_T ptr_allocator(exec_q);
    std::shared_ptr<ptrT> host_chc_ptrs_shp =
        std::make_shared<ptrT>(n_chcs, ptr_allocator);

    using usm_host_allocatorT =
        dpctl::tensor::alloc_utils::usm_host_allocator<py::ssize_t>;
    using shT = std::vector<py::ssize_t, usm_host_allocatorT>;

    usm_host_allocatorT sz_allocator(exec_q);
    std::shared_ptr<shT> host_shape_strides_shp =
        std::make_shared<shT>(shape_len * (3 + n_chcs), sz_allocator);

    std::shared_ptr<shT> host_chc_offsets_shp =
        std::make_shared<shT>(n_chcs, sz_allocator);

    std::copy(shape, shape + shape_len, host_shape_strides_shp->begin());
    std::copy(inp_strides.begin(), inp_strides.end(),
              host_shape_strides_shp->begin() + shape_len);
    std::copy(dst_strides.begin(), dst_strides.end(),
              host_shape_strides_shp->begin() + 2 * shape_len);
    std::copy(chc_strides.begin(), chc_strides.end(),
              host_shape_strides_shp->begin() + 3 * shape_len);

    std::copy(chc_ptrs.begin(), chc_ptrs.end(), host_chc_ptrs_shp->begin());
    std::copy(chc_offsets.begin(), chc_offsets.end(),
              host_chc_offsets_shp->begin());

    const sycl::event &device_chc_ptrs_copy_ev = exec_q.copy<char *>(
        host_chc_ptrs_shp->data(), device_chc_ptrs, host_chc_ptrs_shp->size());

    const sycl::event &device_shape_strides_copy_ev = exec_q.copy<py::ssize_t>(
        host_shape_strides_shp->data(), device_shape_strides,
        host_shape_strides_shp->size());

    const sycl::event &device_chc_offsets_copy_ev = exec_q.copy<py::ssize_t>(
        host_chc_offsets_shp->data(), device_chc_offsets,
        host_chc_offsets_shp->size());

    const sycl::event &shared_ptr_cleanup_ev =
        exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on({device_chc_offsets_copy_ev,
                            device_shape_strides_copy_ev,
                            device_chc_ptrs_copy_ev});
            cgh.host_task([host_chc_offsets_shp, host_shape_strides_shp,
                           host_chc_ptrs_shp]() {});
        });
    host_task_events.push_back(shared_ptr_cleanup_ev);

    std::vector<sycl::event> param_pack_deps{device_chc_ptrs_copy_ev,
                                             device_shape_strides_copy_ev,
                                             device_chc_offsets_copy_ev};
    return param_pack_deps;
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

std::pair<sycl::event, sycl::event>
    py_choose(const dpctl::tensor::usm_ndarray &src,
              const py::object &py_chcs,
              const dpctl::tensor::usm_ndarray &dst,
              uint8_t mode,
              sycl::queue &exec_q,
              const std::vector<sycl::event> &depends)
{
    std::vector<dpctl::tensor::usm_ndarray> chcs =
        parse_py_chcs(exec_q, py_chcs);

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

    auto packed_chc_ptrs = usm_unique_ptr<char *>(n_chcs, exec_q);
    if (packed_chc_ptrs.get() == nullptr) {
        throw std::runtime_error(
            "Unable to allocate packed_chc_ptrs device memory");
    }

    // packed_shapes_strides = [common shape,
    //                          src.strides,
    //                          dst.strides,
    //                          chcs[0].strides,
    //                          ...,
    //                          chcs[n_chcs].strides]
    auto packed_shapes_strides =
        usm_unique_ptr<py::ssize_t>((3 + n_chcs) * sh_nelems, exec_q);
    if (packed_shapes_strides.get() == nullptr) {
        throw std::runtime_error(
            "Unable to allocate packed_shapes_strides device memory");
    }

    auto packed_chc_offsets = usm_unique_ptr<py::ssize_t>(n_chcs, exec_q);
    if (packed_chc_offsets.get() == nullptr) {
        throw std::runtime_error(
            "Unable to allocate packed_chc_offsets device memory");
    }

    std::vector<sycl::event> host_task_events;
    host_task_events.reserve(2);

    std::vector<sycl::event> pack_deps;
    if (nd == 0) {
        // special case where all inputs are scalars
        // need to pass src, dst shape=1 and strides=0
        // chc_strides already initialized to 0 so ignore
        std::array<py::ssize_t, 1> scalar_sh{1};
        std::vector<py::ssize_t> src_strides{0};
        std::vector<py::ssize_t> dst_strides{0};

        pack_deps = _populate_choose_kernel_params(
            exec_q, host_task_events, packed_chc_ptrs.get(),
            packed_shapes_strides.get(), packed_chc_offsets.get(),
            scalar_sh.data(), sh_nelems, src_strides, dst_strides, chc_strides,
            chc_ptrs, chc_offsets, n_chcs);
    }
    else {
        auto src_strides = src.get_strides_vector();
        auto dst_strides = dst.get_strides_vector();

        pack_deps = _populate_choose_kernel_params(
            exec_q, host_task_events, packed_chc_ptrs.get(),
            packed_shapes_strides.get(), packed_chc_offsets.get(), src_shape,
            sh_nelems, src_strides, dst_strides, chc_strides, chc_ptrs,
            chc_offsets, n_chcs);
    }

    std::vector<sycl::event> all_deps;
    all_deps.reserve(depends.size() + pack_deps.size());
    all_deps.insert(std::end(all_deps), std::begin(pack_deps),
                    std::end(pack_deps));
    all_deps.insert(std::end(all_deps), std::begin(depends), std::end(depends));

    sycl::event choose_generic_ev =
        fn(exec_q, nelems, n_chcs, sh_nelems, packed_shapes_strides.get(),
           src_data, dst_data, packed_chc_ptrs.get(), src_offset, dst_offset,
           packed_chc_offsets.get(), all_deps);

    // release usm_unique_ptrs
    auto chc_ptrs_ = packed_chc_ptrs.release();
    auto shapes_strides_ = packed_shapes_strides.release();
    auto chc_offsets_ = packed_chc_offsets.release();

    // free packed temporaries
    sycl::event temporaries_cleanup_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(choose_generic_ev);
        const auto &ctx = exec_q.get_context();

        using dpctl::tensor::alloc_utils::sycl_free_noexcept;
        cgh.host_task([chc_ptrs_, shapes_strides_, chc_offsets_, ctx]() {
            sycl_free_noexcept(chc_ptrs_, ctx);
            sycl_free_noexcept(shapes_strides_, ctx);
            sycl_free_noexcept(chc_offsets_, ctx);
        });
    });

    host_task_events.push_back(temporaries_cleanup_ev);

    using dpctl::utils::keep_args_alive;
    sycl::event arg_cleanup_ev =
        keep_args_alive(exec_q, {src, py_chcs, dst}, host_task_events);

    return std::make_pair(arg_cleanup_ev, choose_generic_ev);
}

template <typename fnT, typename IndT, typename T>
struct ChooseWrapFactory
{
    fnT get()
    {
        if constexpr (std::is_integral<IndT>::value &&
                      !std::is_same<IndT, bool>::value) {
            using dpctl::tensor::indexing_utils::WrapIndex;
            fnT fn = kernels::choose_impl<WrapIndex<IndT>, IndT, T>;
            return fn;
        }
        else {
            fnT fn = nullptr;
            return fn;
        }
    }
};

template <typename fnT, typename IndT, typename T>
struct ChooseClipFactory
{
    fnT get()
    {
        if constexpr (std::is_integral<IndT>::value &&
                      !std::is_same<IndT, bool>::value) {
            using dpctl::tensor::indexing_utils::ClipIndex;
            fnT fn = kernels::choose_impl<ClipIndex<IndT>, IndT, T>;
            return fn;
        }
        else {
            fnT fn = nullptr;
            return fn;
        }
    }
};

void init_choose_dispatch_tables(void)
{
    using namespace td_ns;
    using kernels::choose_fn_ptr_t;

    DispatchTableBuilder<choose_fn_ptr_t, ChooseClipFactory, num_types>
        dtb_choose_clip;
    dtb_choose_clip.populate_dispatch_table(choose_clip_dispatch_table);

    DispatchTableBuilder<choose_fn_ptr_t, ChooseWrapFactory, num_types>
        dtb_choose_wrap;
    dtb_choose_wrap.populate_dispatch_table(choose_wrap_dispatch_table);

    return;
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
