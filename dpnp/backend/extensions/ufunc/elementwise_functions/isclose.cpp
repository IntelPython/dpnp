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

using dpctl::tensor::usm_ndarray;
using event_vector = std::vector<sycl::event>;

using isclose_contig_fn_ptr_t =
    sycl::event (*)(sycl::queue &,
                    std::size_t,
                    const py::object &,
                    const py::object &,
                    const py::object &,
                    const char *,
                    const char *,
                    char *,
                    const std::vector<sycl::event> &);

static isclose_contig_fn_ptr_t isclose_contig_dispatch_table[td_ns::num_types]
                                                            [td_ns::num_types];

template <typename T>
using value_type_of_t = typename value_type_of<T>::type;

// // Template for checking if T is supported
// template <typename T>
// struct IsCloseOutputType
// {
//     using value_type = typename std::disjunction<
//         td_ns::TypeMapResultEntry<T, sycl::half>,
//         td_ns::TypeMapResultEntry<T, float>,
//         td_ns::TypeMapResultEntry<T, double>,
//         td_ns::TypeMapResultEntry<T, std::complex<float>>,
//         td_ns::TypeMapResultEntry<T, std::complex<double>>,
//         td_ns::DefaultResultEntry<void>>::result_type;
// };

// Supports only float and complex types
template <typename T1, typename T2>
struct IsCloseOutputType
{
    using value_type = typename std::disjunction<
        td_ns::BinaryTypeMapResultEntry<T1, float, T2, float, bool>,
        td_ns::BinaryTypeMapResultEntry<T1, double, T2, double, bool>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::complex<float>,
                                        T2,
                                        std::complex<float>,
                                        bool>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::complex<double>,
                                        T2,
                                        std::complex<double>,
                                        bool>,
        td_ns::DefaultResultEntry<void>>::result_type;

    static constexpr bool is_defined = !std::is_same_v<value_type, void>;
};

template <typename T1, typename T2>
sycl::event isclose_contig_call(sycl::queue &q,
                                std::size_t nelems,
                                const py::object &rtol_,
                                const py::object &atol_,
                                const py::object &equal_nan_,
                                const char *in1_p,
                                const char *in2_p,
                                char *out_p,
                                const event_vector &depends)
{
    using dpctl::tensor::type_utils::is_complex_v;
    using scT = std::conditional_t<is_complex_v<T1>,
                                   typename value_type_of<T1>::type, T1>;

    const scT rtol = py::cast<scT>(rtol_);
    const scT atol = py::cast<scT>(atol_);
    const bool equal_nan = py::cast<bool>(equal_nan_);

    return dpnp::kernels::isclose::isclose_contig_impl<T1, T2, scT>(
        q, nelems, rtol, atol, equal_nan, in1_p, 0, in2_p, 0, out_p, 0,
        depends);
}

template <typename fnT, typename T1, typename T2>
struct IsCloseContigFactory
{
    fnT get()
    {
        if constexpr (!IsCloseOutputType<T1, T2>::is_defined) {
            return nullptr;
        }
        else {
            return isclose_contig_call<T1, T2>;
        }
    }
};

void populate_isclose_dispatch_table()
{
    td_ns::DispatchTableBuilder<isclose_contig_fn_ptr_t, IsCloseContigFactory,
                                td_ns::num_types>
        dvb;
    dvb.populate_dispatch_table(isclose_contig_dispatch_table);
}

std::pair<sycl::event, sycl::event>
    py_isclose(const usm_ndarray &a,
               const usm_ndarray &b,
               const py::object &rtol,
               const py::object &atol,
               const py::object &equal_nan,
               const usm_ndarray &res,
               sycl::queue &exec_q,
               const std::vector<sycl::event> &depends)
{
    auto types = td_ns::usm_ndarray_types();
    int a_typeid = types.typenum_to_lookup_id(a.get_typenum());
    int b_typeid = types.typenum_to_lookup_id(b.get_typenum());

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
        auto contig_fn = isclose_contig_dispatch_table[a_typeid][b_typeid];

        if (contig_fn == nullptr) {
            throw std::runtime_error(
                "Contiguous implementation is missing for a_typeid=" +
                std::to_string(a_typeid) +
                " and b_typeid=" + std::to_string(b_typeid));
        }

        auto comp_ev = contig_fn(exec_q, nelems, rtol, atol, equal_nan, a_data,
                                 b_data, res_data, depends);
        sycl::event ht_ev =
            dpctl::utils::keep_args_alive(exec_q, {a, b, res}, {comp_ev});

        return std::make_pair(ht_ev, comp_ev);
    }
    else {
        throw py::value_error("Stride implementation is not implemented");
    }
}

} // namespace impl

void init_isclose(py::module_ m)
{
    impl::populate_isclose_dispatch_table();
    m.def("_isclose", &impl::py_isclose, "", py::arg("a"), py::arg("b"),
          py::arg("rtol"), py::arg("atol"), py::arg("equal_nan"),
          py::arg("res"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());
}

} // namespace dpnp::extensions::ufunc
