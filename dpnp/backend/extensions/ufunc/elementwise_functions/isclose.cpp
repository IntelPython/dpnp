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

using isclose_contig_fn_ptr_t = sycl::event (*)(
    sycl::queue &, std::size_t,
    const py::object &, const py::object &, const py::object &,
    const char *, const char *, char *, const std::vector<sycl::event> &);

static isclose_contig_fn_ptr_t isclose_contig_dispatch_vector[td_ns::num_types];

template <typename T>
using value_type_of_t = typename value_type_of<T>::type;

// Template for checking if T is supported
template <typename T>
struct IsCloseOutputType
{
    using value_type = typename std::disjunction<
        td_ns::TypeMapResultEntry<T, sycl::half>,
        td_ns::TypeMapResultEntry<T, float>,
        td_ns::TypeMapResultEntry<T, double>,
        td_ns::TypeMapResultEntry<T, std::complex<float>>,
        td_ns::TypeMapResultEntry<T, std::complex<double>>,
        td_ns::DefaultResultEntry<void>>::result_type;
};

template <typename T>
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
    using scT = std::conditional_t<is_complex_v<T>, typename value_type_of<T>::type, T>;

    const scT rtol = py::cast<scT>(rtol_);
    const scT atol = py::cast<scT>(atol_);
    const bool equal_nan = py::cast<bool>(equal_nan_);

    return dpnp::kernels::isclose::isclose_contig_impl<T, scT>(
        q, nelems, rtol, atol, equal_nan, in1_p, 0, in2_p, 0, out_p, 0, depends);
}

template <typename fnT, typename T>
struct IsCloseContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename IsCloseOutputType<T>::value_type, void>) {
            return nullptr;
        } else {
            return isclose_contig_call<T>;
        }
    }
};

void populate_isclose_dispatch_vector()
{
    td_ns::DispatchVectorBuilder<isclose_contig_fn_ptr_t, IsCloseContigFactory, td_ns::num_types>
        dvb;
    dvb.populate_dispatch_vector(isclose_contig_dispatch_vector);
}

std::pair<sycl::event, sycl::event> py_isclose(
    const usm_ndarray &a,
    const usm_ndarray &b,
    const py::object &rtol,
    const py::object &atol,
    const py::object &equal_nan,
    const usm_ndarray &dst,
    sycl::queue &exec_q,
    const std::vector<sycl::event> &depends)
{
    auto types = td_ns::usm_ndarray_types();
    int typeid_ = types.typenum_to_lookup_id(a.get_typenum());

    auto fn = isclose_contig_dispatch_vector[typeid_];
    auto comp_ev = fn(exec_q, a.get_size(), rtol, atol, equal_nan,
                      a.get_data(), b.get_data(), dst.get_data(), depends);

    sycl::event ht_ev = dpctl::utils::keep_args_alive(exec_q, {a, b, dst}, {comp_ev});
    return {ht_ev, comp_ev};
}

} // namespace impl

void init_isclose(py::module_ m)
{
    impl::populate_isclose_dispatch_vector();
    m.def("_isclose", &impl::py_isclose, "",
          py::arg("a"), py::arg("b"), py::arg("rtol"), py::arg("atol"),
          py::arg("equal_nan"), py::arg("dst"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());
}

} // namespace dpnp::extensions::ufunc
