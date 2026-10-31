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
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dpnp4pybind11.hpp"

#include "partition.hpp"

// dpnp tensor headers
#include "kernels/sorting/radix_select.hpp"
#include "utils/output_validation.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

// utils extension headers
#include "ext/common.hpp"
#include "ext/validation_utils.hpp"

namespace py = pybind11;
namespace td_ns = dpnp::tensor::type_dispatch;

using dpnp::tensor::usm_ndarray;

using ext::validation::array_names;
using ext::validation::check_c_contig;
using ext::validation::check_has_dtype;
using ext::validation::check_no_overlap;
using ext::validation::check_num_dims;
using ext::validation::check_queue;
using ext::validation::check_same_dtype;
using ext::validation::check_writable;

namespace dpnp::tensor::py_internal
{
using ext::common::init_dispatch_vector;

typedef sycl::event (*partition_fn_ptr_t)(sycl::queue &,
                                          const std::size_t, // n_iters
                                          const std::size_t, // n_values
                                          const std::size_t, // kth
                                          const char *,      // src
                                          char *,            // dst
                                          const std::vector<sycl::event> &);

static partition_fn_ptr_t partition_dispatch_vector[td_ns::num_types];
static partition_fn_ptr_t argpartition_dispatch_vector[td_ns::num_types];

namespace
{
namespace rs_ns = dpnp::tensor::kernels::radix_select_details;

// types with an order-preserving radix key
template <typename T>
constexpr bool use_radix_select = !dpnp::tensor::type_utils::is_complex_v<T>;

template <typename T>
sycl::event partition_call(sycl::queue &exec_q,
                           const std::size_t n_iters,
                           const std::size_t n_values,
                           const std::size_t kth,
                           const char *src_p,
                           char *dst_p,
                           const std::vector<sycl::event> &depends)
{
    return rs_ns::radix_partition_impl<T>(
        exec_q, n_iters, n_values, kth, reinterpret_cast<const T *>(src_p),
        reinterpret_cast<T *>(dst_p), depends);
}

template <typename T>
sycl::event argpartition_call(sycl::queue &exec_q,
                              const std::size_t n_iters,
                              const std::size_t n_values,
                              const std::size_t kth,
                              const char *src_p,
                              char *dst_p,
                              const std::vector<sycl::event> &depends)
{
    using IndexT = std::int64_t;
    return rs_ns::radix_argpartition_impl<T, IndexT>(
        exec_q, n_iters, n_values, kth, reinterpret_cast<const T *>(src_p),
        reinterpret_cast<IndexT *>(dst_p), depends);
}

/*! @brief Partitions the rows of `src` along its last axis into `dst` of the
 * same shape, the implementation being taken from `dispatch_vector` */
std::pair<sycl::event, sycl::event>
    partition_rows(const partition_fn_ptr_t *dispatch_vector,
                   const usm_ndarray &src,
                   const std::size_t kth,
                   const usm_ndarray &dst,
                   sycl::queue &exec_q,
                   const std::vector<sycl::event> &depends)
{
    array_names names = {{&src, "src"}, {&dst, "dst"}};

    const int nd = src.get_ndim();
    if (nd == 0) {
        throw py::value_error("The input array must have at least one "
                              "dimension");
    }
    check_num_dims(&dst, nd, names);

    const py::ssize_t *src_shape = src.get_shape_raw();
    const py::ssize_t *dst_shape = dst.get_shape_raw();
    std::size_t n_iters(1);
    for (int i = 0; i < nd; ++i) {
        if (src_shape[i] != dst_shape[i]) {
            throw py::value_error("`src` and `dst` shapes must match");
        }
        if (i + 1 < nd) {
            n_iters *= static_cast<std::size_t>(src_shape[i]);
        }
    }
    const std::size_t n_values = static_cast<std::size_t>(src_shape[nd - 1]);

    check_queue({&src, &dst}, names, exec_q);
    check_writable({&dst}, names);

    if (n_iters == 0 || n_values == 0) {
        return {sycl::event(), sycl::event()};
    }
    if (kth >= n_values) {
        throw py::value_error("`kth` is out of bounds for the last axis");
    }

    check_c_contig({&src, &dst}, names);
    check_no_overlap(&src, &dst, names);
    dpnp::tensor::validation::AmpleMemory::throw_if_not_ample(
        dst, n_iters * n_values);

    auto types = td_ns::usm_ndarray_types();
    const int src_typeid = types.typenum_to_lookup_id(src.get_typenum());

    auto fn = dispatch_vector[src_typeid];
    if (fn == nullptr) {
        throw py::value_error("No partition implementation is available for "
                              "the data type of the input array");
    }

    sycl::event comp_ev = fn(exec_q, n_iters, n_values, kth, src.get_data(),
                             dst.get_data(), depends);
    sycl::event ht_ev =
        dpnp::utils::keep_args_alive(exec_q, {src, dst}, {comp_ev});

    return std::make_pair(ht_ev, comp_ev);
}
} // namespace

std::pair<sycl::event, sycl::event>
    py_partition(const usm_ndarray &src,
                 const std::size_t kth,
                 const usm_ndarray &dst,
                 sycl::queue &exec_q,
                 const std::vector<sycl::event> &depends)
{
    array_names names = {{&src, "src"}, {&dst, "dst"}};
    check_same_dtype(&src, &dst, names);

    return partition_rows(partition_dispatch_vector, src, kth, dst, exec_q,
                          depends);
}

std::pair<sycl::event, sycl::event>
    py_argpartition(const usm_ndarray &src,
                    const std::size_t kth,
                    const usm_ndarray &dst,
                    sycl::queue &exec_q,
                    const std::vector<sycl::event> &depends)
{
    array_names names = {{&src, "src"}, {&dst, "dst"}};
    check_has_dtype(&dst, td_ns::typenum_t::INT64, names);

    return partition_rows(argpartition_dispatch_vector, src, kth, dst, exec_q,
                          depends);
}

template <typename fnT, typename T>
struct PartitionFactory
{
    fnT get()
    {
        if constexpr (use_radix_select<T>) {
            return partition_call<T>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename T>
struct ArgPartitionFactory
{
    fnT get()
    {
        if constexpr (use_radix_select<T>) {
            return argpartition_call<T>;
        }
        else {
            return nullptr;
        }
    }
};

static void populate_partition_dispatch_vectors()
{
    init_dispatch_vector<partition_fn_ptr_t, PartitionFactory>(
        partition_dispatch_vector);
    init_dispatch_vector<partition_fn_ptr_t, ArgPartitionFactory>(
        argpartition_dispatch_vector);
}

void init_partition(py::module_ &m)
{
    populate_partition_dispatch_vectors();

    m.def("_partition", &py_partition,
          "Partitions each row of C-contiguous `src` along its last axis "
          "about the element at `kth` of the sorted row, writing into `dst` "
          "of the same shape and data type",
          py::arg("src"), py::arg("kth"), py::arg("dst"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());

    m.def("_argpartition", &py_argpartition,
          "Writes into int64 `dst` of the shape of C-contiguous `src` the "
          "indices which partition each row of `src` along its last axis "
          "about the element at `kth` of the sorted row",
          py::arg("src"), py::arg("kth"), py::arg("dst"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());
}

} // namespace dpnp::tensor::py_internal
