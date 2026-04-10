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
/// This file defines functions of dpnp.tensor._tensor_sorting_impl
/// extension.
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kernels/sorting/topk.hpp"
#include "utils/memory_overlap.hpp"
#include "utils/output_validation.hpp"
#include "utils/rich_comparisons.hpp"
#include "utils/type_dispatch.hpp"

#include "topk.hpp"

namespace dpctl::tensor::py_internal
{

namespace td_ns = dpctl::tensor::type_dispatch;

typedef sycl::event (*topk_impl_fn_ptr_t)(sycl::queue &,
                                          std::size_t,
                                          std::size_t,
                                          std::size_t,
                                          bool,
                                          const char *,
                                          char *,
                                          char *,
                                          const std::vector<sycl::event> &);

static topk_impl_fn_ptr_t topk_dispatch_vector[td_ns::num_types];

namespace
{

template <typename T, typename = void>
struct use_radix_sort : public std::false_type
{
};

template <typename T>
struct use_radix_sort<
    T,
    std::enable_if_t<std::disjunction<std::is_same<T, bool>,
                                      std::is_same<T, std::uint8_t>,
                                      std::is_same<T, std::int8_t>,
                                      std::is_same<T, std::uint16_t>,
                                      std::is_same<T, std::int16_t>>::value>>
    : public std::true_type
{
};

template <typename argTy, typename IndexTy>
sycl::event topk_caller(sycl::queue &exec_q,
                        std::size_t iter_nelems, // number of sub-arrays
                        std::size_t axis_nelems, // size of each sub-array
                        std::size_t k,
                        bool largest,
                        const char *arg_cp,
                        char *vals_cp,
                        char *inds_cp,
                        const std::vector<sycl::event> &depends)
{
    if constexpr (use_radix_sort<argTy>::value) {
        using dpctl::tensor::kernels::topk_radix_impl;
        auto ascending = !largest;
        return topk_radix_impl<argTy, IndexTy>(exec_q, iter_nelems, axis_nelems,
                                               k, ascending, arg_cp, vals_cp,
                                               inds_cp, depends);
    }
    else {
        using dpctl::tensor::kernels::topk_merge_impl;
        if (largest) {
            using CompTy =
                typename dpctl::tensor::rich_comparisons::DescendingSorter<
                    argTy>::type;
            return topk_merge_impl<argTy, IndexTy, CompTy>(
                exec_q, iter_nelems, axis_nelems, k, arg_cp, vals_cp, inds_cp,
                depends);
        }
        else {
            using CompTy =
                typename dpctl::tensor::rich_comparisons::AscendingSorter<
                    argTy>::type;
            return topk_merge_impl<argTy, IndexTy, CompTy>(
                exec_q, iter_nelems, axis_nelems, k, arg_cp, vals_cp, inds_cp,
                depends);
        }
    }
}

} // namespace

std::pair<sycl::event, sycl::event>
    py_topk(const dpctl::tensor::usm_ndarray &src,
            std::optional<const int> trailing_dims_to_search,
            const std::size_t k,
            const bool largest,
            const dpctl::tensor::usm_ndarray &vals,
            const dpctl::tensor::usm_ndarray &inds,
            sycl::queue &exec_q,
            const std::vector<sycl::event> &depends)
{
    int src_nd = src.get_ndim();
    int vals_nd = vals.get_ndim();
    int inds_nd = inds.get_ndim();

    const py::ssize_t *src_shape_ptr = src.get_shape_raw();
    const py::ssize_t *vals_shape_ptr = vals.get_shape_raw();
    const py::ssize_t *inds_shape_ptr = inds.get_shape_raw();

    std::size_t axis_nelems(1);
    std::size_t iter_nelems(1);
    if (trailing_dims_to_search.has_value()) {
        if (src_nd != vals_nd || src_nd != inds_nd) {
            throw py::value_error("The input and output arrays must have "
                                  "the same array ranks");
        }

        auto trailing_dims = trailing_dims_to_search.value();
        int iter_nd = src_nd - trailing_dims;
        if (trailing_dims <= 0 || iter_nd < 0) {
            throw py::value_error(
                "trailing_dims_to_search must be positive, but no "
                "greater than rank of the array being searched");
        }

        bool same_shapes = true;
        for (int i = 0; same_shapes && (i < iter_nd); ++i) {
            auto src_shape_i = src_shape_ptr[i];
            same_shapes = same_shapes && (src_shape_i == vals_shape_ptr[i] &&
                                          src_shape_i == inds_shape_ptr[i]);
            iter_nelems *= static_cast<std::size_t>(src_shape_i);
        }

        if (!same_shapes) {
            throw py::value_error(
                "Destination shape does not match the input shape");
        }

        std::size_t vals_k(1);
        std::size_t inds_k(1);
        for (int i = iter_nd; i < src_nd; ++i) {
            axis_nelems *= static_cast<std::size_t>(src_shape_ptr[i]);
            vals_k *= static_cast<std::size_t>(vals_shape_ptr[i]);
            inds_k *= static_cast<std::size_t>(inds_shape_ptr[i]);
        }

        bool valid_k = (vals_k == k && inds_k == k && axis_nelems >= k);
        if (!valid_k) {
            throw py::value_error("The value of k is invalid for the input and "
                                  "destination arrays");
        }
    }
    else {
        if (vals_nd != 1 || inds_nd != 1) {
            throw py::value_error("Output arrays must be one-dimensional");
        }

        for (int i = 0; i < src_nd; ++i) {
            axis_nelems *= static_cast<std::size_t>(src_shape_ptr[i]);
        }

        bool valid_k = (axis_nelems >= k &&
                        static_cast<std::size_t>(vals_shape_ptr[0]) == k &&
                        static_cast<std::size_t>(inds_shape_ptr[0]) == k);
        if (!valid_k) {
            throw py::value_error("The value of k is invalid for the input and "
                                  "destination arrays");
        }
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {src, vals, inds})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(vals);
    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(inds);

    if ((iter_nelems == 0) || (axis_nelems == 0)) {
        // Nothing to do
        return std::make_pair(sycl::event(), sycl::event());
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(src, vals) || overlap(src, inds)) {
        throw py::value_error("Arrays index overlapping segments of memory");
    }

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(vals,
                                                               k * iter_nelems);

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(inds,
                                                               k * iter_nelems);

    int src_typenum = src.get_typenum();
    int vals_typenum = vals.get_typenum();
    int inds_typenum = inds.get_typenum();

    const auto &array_types = td_ns::usm_ndarray_types();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);
    int vals_typeid = array_types.typenum_to_lookup_id(vals_typenum);
    int inds_typeid = array_types.typenum_to_lookup_id(inds_typenum);

    if (src_typeid != vals_typeid) {
        throw py::value_error("Input array and vals array must have "
                              "the same data type");
    }

    if (inds_typeid != static_cast<int>(td_ns::typenum_t::INT64)) {
        throw py::value_error("Inds array must have data type int64");
    }

    bool is_src_c_contig = src.is_c_contiguous();
    bool is_vals_c_contig = vals.is_c_contiguous();
    bool is_inds_c_contig = inds.is_c_contiguous();

    if (is_src_c_contig && is_vals_c_contig && is_inds_c_contig) {
        auto fn = topk_dispatch_vector[src_typeid];

        sycl::event comp_ev =
            fn(exec_q, iter_nelems, axis_nelems, k, largest, src.get_data(),
               vals.get_data(), inds.get_data(), depends);

        sycl::event keep_args_alive_ev =
            dpctl::utils::keep_args_alive(exec_q, {src, vals, inds}, {comp_ev});

        return std::make_pair(keep_args_alive_ev, comp_ev);
    }

    return std::make_pair(sycl::event(), sycl::event());
}

template <typename fnT, typename T>
struct TopKFactory
{
    fnT get()
    {
        using IdxT = std::int64_t;
        return topk_caller<T, IdxT>;
    }
};

void init_topk_dispatch_vectors(void)
{
    td_ns::DispatchVectorBuilder<topk_impl_fn_ptr_t, TopKFactory,
                                 td_ns::num_types>
        dvb;
    dvb.populate_dispatch_vector(topk_dispatch_vector);
}

void init_topk_functions(py::module_ m)
{
    init_topk_dispatch_vectors();

    m.def("_topk", &py_topk, py::arg("src"), py::arg("trailing_dims_to_search"),
          py::arg("k"), py::arg("largest"), py::arg("vals"), py::arg("inds"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());
}

} // namespace dpctl::tensor::py_internal
