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
//===---------------------------------------------------------------------===//
///
/// \file
/// This file defines functions of dpctl.tensor._tensor_elementwise_impl
/// extension, specifically functions for elementwise operations.
//===---------------------------------------------------------------------===//

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility> // for std::ignore
#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "elementwise_functions.hpp"
#include "simplify_iteration_space.hpp"
#include "true_divide.hpp"

#include "utils/memory_overlap.hpp"
#include "utils/offset_utils.hpp"
#include "utils/output_validation.hpp"
#include "utils/sycl_alloc_utils.hpp"
#include "utils/type_dispatch.hpp"

#include "kernels/elementwise_functions/common.hpp"
#include "kernels/elementwise_functions/common_inplace.hpp"
#include "kernels/elementwise_functions/true_divide.hpp"

namespace dpctl::tensor::py_internal
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

namespace ew_cmn_ns = dpctl::tensor::kernels::elementwise_common;
using ew_cmn_ns::binary_contig_impl_fn_ptr_t;
using ew_cmn_ns::binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t;
using ew_cmn_ns::binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t;
using ew_cmn_ns::binary_strided_impl_fn_ptr_t;

using ew_cmn_ns::binary_inplace_contig_impl_fn_ptr_t;
using ew_cmn_ns::binary_inplace_row_matrix_broadcast_impl_fn_ptr_t;
using ew_cmn_ns::binary_inplace_strided_impl_fn_ptr_t;

// B08: ===== DIVIDE (x1, x2)
namespace impl
{
namespace true_divide_fn_ns = dpctl::tensor::kernels::true_divide;

static binary_contig_impl_fn_ptr_t
    true_divide_contig_dispatch_table[td_ns::num_types][td_ns::num_types];
static int true_divide_output_id_table[td_ns::num_types][td_ns::num_types];
static int true_divide_inplace_output_id_table[td_ns::num_types]
                                              [td_ns::num_types];

static binary_strided_impl_fn_ptr_t
    true_divide_strided_dispatch_table[td_ns::num_types][td_ns::num_types];

// divide(matrix, row)
static binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t
    true_divide_contig_matrix_contig_row_broadcast_dispatch_table
        [td_ns::num_types][td_ns::num_types];

// divide(row, matrix)
static binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t
    true_divide_contig_row_contig_matrix_broadcast_dispatch_table
        [td_ns::num_types][td_ns::num_types];

static binary_inplace_contig_impl_fn_ptr_t
    true_divide_inplace_contig_dispatch_table[td_ns::num_types]
                                             [td_ns::num_types];
static binary_inplace_strided_impl_fn_ptr_t
    true_divide_inplace_strided_dispatch_table[td_ns::num_types]
                                              [td_ns::num_types];
static binary_inplace_row_matrix_broadcast_impl_fn_ptr_t
    true_divide_inplace_row_matrix_dispatch_table[td_ns::num_types]
                                                 [td_ns::num_types];

void populate_true_divide_dispatch_tables(void)
{
    using namespace td_ns;
    namespace fn_ns = true_divide_fn_ns;

    // which input types are supported, and what is the type of the result
    using fn_ns::TrueDivideTypeMapFactory;
    DispatchTableBuilder<int, TrueDivideTypeMapFactory, num_types> dtb1;
    dtb1.populate_dispatch_table(true_divide_output_id_table);

    // function pointers for operation on general strided arrays
    using fn_ns::TrueDivideStridedFactory;
    DispatchTableBuilder<binary_strided_impl_fn_ptr_t, TrueDivideStridedFactory,
                         num_types>
        dtb2;
    dtb2.populate_dispatch_table(true_divide_strided_dispatch_table);

    // function pointers for operation on contiguous inputs and output
    using fn_ns::TrueDivideContigFactory;
    DispatchTableBuilder<binary_contig_impl_fn_ptr_t, TrueDivideContigFactory,
                         num_types>
        dtb3;
    dtb3.populate_dispatch_table(true_divide_contig_dispatch_table);

    // function pointers for operation on contiguous matrix, contiguous row
    // with contiguous matrix output
    using fn_ns::TrueDivideContigMatrixContigRowBroadcastFactory;
    DispatchTableBuilder<
        binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t,
        TrueDivideContigMatrixContigRowBroadcastFactory, num_types>
        dtb4;
    dtb4.populate_dispatch_table(
        true_divide_contig_matrix_contig_row_broadcast_dispatch_table);

    // function pointers for operation on contiguous row, contiguous matrix
    // with contiguous matrix output
    using fn_ns::TrueDivideContigRowContigMatrixBroadcastFactory;
    DispatchTableBuilder<
        binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t,
        TrueDivideContigRowContigMatrixBroadcastFactory, num_types>
        dtb5;
    dtb5.populate_dispatch_table(
        true_divide_contig_row_contig_matrix_broadcast_dispatch_table);

    // which types are supported by the in-place kernels
    using fn_ns::TrueDivideInplaceTypeMapFactory;
    DispatchTableBuilder<int, TrueDivideInplaceTypeMapFactory, num_types> dtb6;
    dtb6.populate_dispatch_table(true_divide_inplace_output_id_table);

    // function pointers for inplace operation on general strided arrays
    using fn_ns::TrueDivideInplaceStridedFactory;
    DispatchTableBuilder<binary_inplace_strided_impl_fn_ptr_t,
                         TrueDivideInplaceStridedFactory, num_types>
        dtb7;
    dtb7.populate_dispatch_table(true_divide_inplace_strided_dispatch_table);

    // function pointers for inplace operation on contiguous inputs and output
    using fn_ns::TrueDivideInplaceContigFactory;
    DispatchTableBuilder<binary_inplace_contig_impl_fn_ptr_t,
                         TrueDivideInplaceContigFactory, num_types>
        dtb8;
    dtb8.populate_dispatch_table(true_divide_inplace_contig_dispatch_table);

    // function pointers for inplace operation on contiguous matrix
    // and contiguous row
    using fn_ns::TrueDivideInplaceRowMatrixBroadcastFactory;
    DispatchTableBuilder<binary_inplace_row_matrix_broadcast_impl_fn_ptr_t,
                         TrueDivideInplaceRowMatrixBroadcastFactory, num_types>
        dtb9;
    dtb9.populate_dispatch_table(true_divide_inplace_row_matrix_dispatch_table);
};

template <typename T>
class divide_by_scalar_krn;

typedef sycl::event (*divide_by_scalar_fn_ptr_t)(
    sycl::queue &,
    std::size_t,
    int,
    const ssize_t *,
    const char *,
    py::ssize_t,
    const char *,
    char *,
    py::ssize_t,
    const std::vector<sycl::event> &);

template <typename T, typename scalarT>
sycl::event divide_by_scalar(sycl::queue &exec_q,
                             std::size_t nelems,
                             int nd,
                             const ssize_t *shape_and_strides,
                             const char *arg_p,
                             py::ssize_t arg_offset,
                             const char *scalar_ptr,
                             char *res_p,
                             py::ssize_t res_offset,
                             const std::vector<sycl::event> &depends = {})
{
    const scalarT sc_v = *reinterpret_cast<const scalarT *>(scalar_ptr);

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using BinOpT =
            dpctl::tensor::kernels::true_divide::TrueDivideFunctor<T, scalarT,
                                                                   T>;

        auto op = BinOpT();

        using IndexerT =
            typename dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;

        const IndexerT two_offsets_indexer{nd, arg_offset, res_offset,
                                           shape_and_strides};

        const T *arg_tp = reinterpret_cast<const T *>(arg_p);
        T *res_tp = reinterpret_cast<T *>(res_p);

        cgh.parallel_for<divide_by_scalar_krn<T>>(
            {nelems}, [=](sycl::id<1> id) {
                const auto &two_offsets_ =
                    two_offsets_indexer(static_cast<ssize_t>(id.get(0)));

                const auto &arg_i = two_offsets_.get_first_offset();
                const auto &res_i = two_offsets_.get_second_offset();
                res_tp[res_i] = op(arg_tp[arg_i], sc_v);
            });
    });
    return comp_ev;
}

std::pair<sycl::event, sycl::event>
    py_divide_by_scalar(const dpctl::tensor::usm_ndarray &src,
                        double scalar,
                        const dpctl::tensor::usm_ndarray &dst,
                        sycl::queue &exec_q,
                        const std::vector<sycl::event> &depends = {})
{
    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    auto array_types = td_ns::usm_ndarray_types();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    if (src_typeid != dst_typeid) {
        throw py::value_error(
            "Destination array has unexpected elemental data type.");
    }

    // check that queues are compatible
    if (!dpctl::utils::queues_are_compatible(exec_q, {src, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);
    // check shapes, broadcasting is assumed done by caller
    // check that dimensions are the same
    int dst_nd = dst.get_ndim();
    if (dst_nd != src.get_ndim()) {
        throw py::value_error("Array dimensions are not the same.");
    }

    // check that shapes are the same
    const py::ssize_t *src_shape = src.get_shape_raw();
    const py::ssize_t *dst_shape = dst.get_shape_raw();
    bool shapes_equal(true);
    std::size_t src_nelems(1);

    for (int i = 0; i < dst_nd; ++i) {
        src_nelems *= static_cast<std::size_t>(src_shape[i]);
        shapes_equal = shapes_equal && (src_shape[i] == dst_shape[i]);
    }
    if (!shapes_equal) {
        throw py::value_error("Array shapes are not the same.");
    }

    // if nelems is zero, return
    if (src_nelems == 0) {
        return std::make_pair(sycl::event(), sycl::event());
    }

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(dst, src_nelems);

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    auto const &same_logical_tensors =
        dpctl::tensor::overlap::SameLogicalTensors();
    if ((overlap(src, dst) && !same_logical_tensors(src, dst))) {
        throw py::value_error("Arrays index overlapping segments of memory");
    }

    const char *src_data = src.get_data();
    char *dst_data = dst.get_data();

    static constexpr int float16_typeid =
        static_cast<int>(td_ns::typenum_t::HALF);
    static constexpr int float32_typeid =
        static_cast<int>(td_ns::typenum_t::FLOAT);
    static constexpr int float64_typeid =
        static_cast<int>(td_ns::typenum_t::DOUBLE);
    static constexpr int complex64_typeid =
        static_cast<int>(td_ns::typenum_t::CFLOAT);
    static constexpr int complex128_typeid =
        static_cast<int>(td_ns::typenum_t::CDOUBLE);

    // statically pre-allocated memory for scalar
    alignas(double) char scalar_alloc[sizeof(double)] = {0};

    divide_by_scalar_fn_ptr_t fn;
    // placement new into stack memory means no call to delete is necessary
    switch (src_typeid) {
    case float16_typeid:
    {
        fn = divide_by_scalar<sycl::half, sycl::half>;
        std::ignore =
            new (scalar_alloc) sycl::half(static_cast<sycl::half>(scalar));
        break;
    }
    case float32_typeid:
    {
        fn = divide_by_scalar<float, float>;
        std::ignore = new (scalar_alloc) float(scalar);
        break;
    }
    case float64_typeid:
    {
        fn = divide_by_scalar<double, double>;
        std::ignore = new (scalar_alloc) double(scalar);
        break;
    }
    case complex64_typeid:
    {
        fn = divide_by_scalar<std::complex<float>, float>;
        std::ignore = new (scalar_alloc) float(scalar);
        break;
    }
    case complex128_typeid:
    {
        fn = divide_by_scalar<std::complex<double>, double>;
        std::ignore = new (scalar_alloc) double(scalar);
        break;
    }
    default:
        throw std::runtime_error("Implementation is missing for typeid=" +
                                 std::to_string(src_typeid));
    }

    // simplify strides
    auto const &src_strides = src.get_strides_vector();
    auto const &dst_strides = dst.get_strides_vector();

    using shT = std::vector<py::ssize_t>;
    shT simplified_shape;
    shT simplified_src_strides;
    shT simplified_dst_strides;
    py::ssize_t src_offset(0);
    py::ssize_t dst_offset(0);

    int nd = dst_nd;
    const py::ssize_t *shape = src_shape;

    std::vector<sycl::event> host_tasks{};
    simplify_iteration_space(nd, shape, src_strides, dst_strides,
                             // outputs
                             simplified_shape, simplified_src_strides,
                             simplified_dst_strides, src_offset, dst_offset);

    if (nd == 0) {
        // handle 0d array as 1d array with 1 element
        static constexpr py::ssize_t one{1};
        simplified_shape.push_back(one);
        simplified_src_strides.push_back(one);
        simplified_dst_strides.push_back(one);
        src_offset = 0;
        dst_offset = 0;
    }

    using dpctl::tensor::offset_utils::device_allocate_and_pack;
    auto ptr_sz_event_triple_ = device_allocate_and_pack<py::ssize_t>(
        exec_q, host_tasks, simplified_shape, simplified_src_strides,
        simplified_dst_strides);
    auto shape_strides_owner = std::move(std::get<0>(ptr_sz_event_triple_));
    auto &copy_metadata_ev = std::get<2>(ptr_sz_event_triple_);

    const py::ssize_t *shape_strides = shape_strides_owner.get();

    std::vector<sycl::event> all_deps;
    all_deps.reserve(depends.size() + 1);
    all_deps.resize(depends.size());
    std::copy(depends.begin(), depends.end(), all_deps.begin());
    all_deps.push_back(copy_metadata_ev);

    sycl::event div_ev =
        fn(exec_q, src_nelems, nd, shape_strides, src_data, src_offset,
           scalar_alloc, dst_data, dst_offset, all_deps);

    // async free of shape_strides temporary
    sycl::event tmp_cleanup_ev = dpctl::tensor::alloc_utils::async_smart_free(
        exec_q, {div_ev}, shape_strides_owner);

    host_tasks.push_back(tmp_cleanup_ev);

    return std::make_pair(
        dpctl::utils::keep_args_alive(exec_q, {src, dst}, host_tasks), div_ev);
}

} // namespace impl

void init_divide(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;
    {
        impl::populate_true_divide_dispatch_tables();
        using impl::true_divide_contig_dispatch_table;
        using impl::
            true_divide_contig_matrix_contig_row_broadcast_dispatch_table;
        using impl::
            true_divide_contig_row_contig_matrix_broadcast_dispatch_table;
        using impl::true_divide_output_id_table;
        using impl::true_divide_strided_dispatch_table;

        auto divide_pyapi = [&](const arrayT &src1, const arrayT &src2,
                                const arrayT &dst, sycl::queue &exec_q,
                                const event_vecT &depends = {}) {
            return py_binary_ufunc(
                src1, src2, dst, exec_q, depends, true_divide_output_id_table,
                // function pointers to handle operation on contiguous arrays
                // (pointers may be nullptr)
                true_divide_contig_dispatch_table,
                // function pointers to handle operation on strided arrays (most
                // general case)
                true_divide_strided_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                true_divide_contig_matrix_contig_row_broadcast_dispatch_table,
                // function pointers to handle operation of c-contig matrix and
                // c-contig row with broadcasting (may be nullptr)
                true_divide_contig_row_contig_matrix_broadcast_dispatch_table);
        };
        auto divide_result_type_pyapi = [&](const py::dtype &dtype1,
                                            const py::dtype &dtype2) {
            return py_binary_ufunc_result_type(dtype1, dtype2,
                                               true_divide_output_id_table);
        };
        m.def("_divide", divide_pyapi, "", py::arg("src1"), py::arg("src2"),
              py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
        m.def("_divide_result_type", divide_result_type_pyapi, "");

        using impl::true_divide_inplace_contig_dispatch_table;
        using impl::true_divide_inplace_output_id_table;
        using impl::true_divide_inplace_row_matrix_dispatch_table;
        using impl::true_divide_inplace_strided_dispatch_table;

        auto divide_inplace_pyapi = [&](const arrayT &src, const arrayT &dst,
                                        sycl::queue &exec_q,
                                        const event_vecT &depends = {}) {
            return py_binary_inplace_ufunc(
                src, dst, exec_q, depends, true_divide_inplace_output_id_table,
                // function pointers to handle inplace operation on
                // contiguous arrays (pointers may be nullptr)
                true_divide_inplace_contig_dispatch_table,
                // function pointers to handle inplace operation on strided
                // arrays (most general case)
                true_divide_inplace_strided_dispatch_table,
                // function pointers to handle inplace operation on
                // c-contig matrix with c-contig row with broadcasting
                // (may be nullptr)
                true_divide_inplace_row_matrix_dispatch_table);
        };
        m.def("_divide_inplace", divide_inplace_pyapi, "", py::arg("lhs"),
              py::arg("rhs"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());

        using impl::py_divide_by_scalar;
        m.def("_divide_by_scalar", &py_divide_by_scalar, "", py::arg("src"),
              py::arg("scalar"), py::arg("dst"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());
    }
}

} // namespace dpctl::tensor::py_internal
