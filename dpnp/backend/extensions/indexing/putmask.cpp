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

#include <stdexcept>
#include <type_traits>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpctl4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "putmask_kernel.hpp"

#include "../elementwise_functions/simplify_iteration_space.hpp"

// dpctl tensor headers
#include "utils/offset_utils.hpp"
#include "utils/output_validation.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

// utils extension headers
#include "ext/common.hpp"
#include "ext/validation_utils.hpp"

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::usm_ndarray;

using ext::common::dtype_from_typenum;
using ext::validation::array_names;
using ext::validation::check_has_dtype;
using ext::validation::check_no_overlap;
using ext::validation::check_num_dims;
using ext::validation::check_queue;
using ext::validation::check_same_dtype;
using ext::validation::check_same_size;
using ext::validation::check_writable;

namespace dpnp::extensions::indexing
{
using ext::common::init_dispatch_vector;

typedef sycl::event (*putmask_contig_fn_ptr_t)(
    sycl::queue &,
    const std::size_t, // nelems
    char *,            // dst
    const char *,      // mask
    const char *,      // values
    const std::size_t, // values_size
    const std::vector<sycl::event> &);

static putmask_contig_fn_ptr_t putmask_contig_dispatch_vector[td_ns::num_types];

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
    check_num_dims(&mask, nd, names);

    check_queue({&dst, &mask, &values}, names, exec_q);
    check_no_overlap({&mask, &values}, {&dst}, names);
    check_writable({&dst}, names);

    // values must be 1D
    check_num_dims(&values, 1, names);

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

    // if nelems is zero, return
    if (nelems == 0) {
        return {sycl::event(), sycl::event()};
    }

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(dst, nelems);

    char *dst_p = dst.get_data();
    const char *mask_p = mask.get_data();
    const char *values_p = values.get_data();
    const std::size_t values_size = values.get_size();

    // handle C contiguous inputs
    const bool is_dst_c_contig = dst.is_c_contiguous();
    const bool is_mask_c_contig = mask.is_c_contiguous();
    const bool is_values_c_contig = values.is_c_contiguous();

    const bool all_c_contig =
        (is_dst_c_contig && is_mask_c_contig && is_values_c_contig);

    if (all_c_contig) {
        auto contig_fn = putmask_contig_dispatch_vector[dst_values_typeid];

        if (contig_fn == nullptr) {
            py::dtype dst_values_dtype_py =
                dtype_from_typenum(dst_values_typeid);
            throw std::runtime_error(
                "Contiguous implementation is missing for " +
                std::string(py::str(dst_values_dtype_py)) + "data type");
        }

        auto comp_ev = contig_fn(exec_q, nelems, dst_p, mask_p, values_p,
                                 values_size, depends);
        sycl::event ht_ev = dpctl::utils::keep_args_alive(
            exec_q, {dst, mask, values}, {comp_ev});

        return std::make_pair(ht_ev, comp_ev);
    }

    throw py::value_error("Stride implementation is not implemented yet");
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
struct PutMaskContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename PutMaskOutputType<T>::value_type,
                                     void>) {
            return nullptr;
        }
        else {
            return kernels::putmask_contig_impl<T>;
        }
    }
};

static void populate_putmask_dispatch_vectors()
{
    init_dispatch_vector<putmask_contig_fn_ptr_t, PutMaskContigFactory>(
        putmask_contig_dispatch_vector);
}

void init_putmask(py::module_ m)
{
    populate_putmask_dispatch_vectors();

    m.def("_putmask", &py_putmask, "", py::arg("dst"), py::arg("mask"),
          py::arg("values"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());
}

} // namespace dpnp::extensions::indexing
