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

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kernels/distributions.hpp"
#include "kernels/engines.hpp"
#include "kernels/sample.hpp"

// utils extension header
#include "ext/common.hpp"

// dpnp tensor headers
#include "utils/output_validation.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

namespace dpnp::extensions::rng
{
namespace py = pybind11;
namespace td_ns = dpnp::tensor::type_dispatch;
namespace rng_ns = dpnp::kernels::rng;

// A distribution parameter as passed from Python: either a C-contiguous
// usm_ndarray with the data type and the size of the result, whose data
// pointer is `data`, or a Python scalar.
struct ParamArg
{
    const char *data = nullptr;
    py::object scalar;
};

typedef sycl::event (*sample_fn_ptr_t)(sycl::queue &,
                                       const rng_ns::EngineState &,
                                       const std::size_t,
                                       char *,
                                       const ParamArg &,
                                       const ParamArg &,
                                       const std::vector<sycl::event> &);

using sample_dispatch_table_t =
    sample_fn_ptr_t[rng_ns::num_engines][td_ns::num_types];

namespace detail
{
template <typename T>
rng_ns::Param<T> to_param(const ParamArg &arg)
{
    if (arg.data != nullptr) {
        return {reinterpret_cast<const T *>(arg.data), T{}};
    }
    if (arg.scalar.is_none()) {
        return {nullptr, T{}};
    }
    return {nullptr, py::cast<T>(arg.scalar)};
}
} // namespace detail

template <typename EngineTraitsT,
          typename T,
          template <typename> class SamplerT>
sycl::event sample_impl(sycl::queue &exec_q,
                        const rng_ns::EngineState &state,
                        const std::size_t nelems,
                        char *result,
                        const ParamArg &a,
                        const ParamArg &b,
                        const std::vector<sycl::event> &depends)
{
    dpnp::tensor::type_utils::validate_type_for_device<T>(exec_q);

    T *res = reinterpret_cast<T *>(result);
    const SamplerT<T> sampler(detail::to_param<T>(a), detail::to_param<T>(b));
    const std::size_t n_wi = rng_ns::num_work_items(nelems);

    sycl::event sample_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using SampleKernel =
            rng_ns::SampleFunctor<EngineTraitsT, T, SamplerT<T>>;
        cgh.parallel_for<SampleKernel>(
            sycl::range<1>(n_wi),
            SampleKernel(state, sampler, res, nelems, n_wi));
    });

    return sample_ev;
}

// Types the floating-point samplers support
template <typename T>
struct RealTypes
    : std::bool_constant<std::is_same_v<T, float> || std::is_same_v<T, double>>
{
};

// Types the integer samplers support
template <typename T>
struct IntegerTypes : std::bool_constant<std::is_integral_v<T>>
{
};

// The raw output of a bit generator is always 64 bits wide
template <typename T>
struct RawTypes : std::bool_constant<std::is_same_v<T, std::uint64_t>>
{
};

template <typename EngineTraitsT,
          template <typename> class SamplerT,
          template <typename> class SupportedT>
struct SampleFactoryFor
{
    template <typename fnT, typename T>
    struct type
    {
        fnT get()
        {
            if constexpr (SupportedT<T>::value) {
                return sample_impl<EngineTraitsT, T, SamplerT>;
            }
            else {
                return nullptr;
            }
        }
    };
};

template <template <typename> class SamplerT,
          template <typename> class SupportedT>
void init_sample_dispatch_table(sample_dispatch_table_t &table)
{
    using ext::common::init_dispatch_vector;

    static_assert(rng_ns::num_engines == 1,
                  "Every engine must have a row in the dispatch table");

    constexpr int philox = static_cast<int>(rng_ns::engine_id::philox4x32x10);
    init_dispatch_vector<sample_fn_ptr_t,
                         SampleFactoryFor<rng_ns::Philox4x32x10, SamplerT,
                                          SupportedT>::template type>(
        table[philox]);
}

extern std::pair<sycl::event, sycl::event>
    py_sample(sycl::queue &exec_q,
              const int engine,
              const std::vector<std::uint64_t> &seed,
              const std::uint64_t counter_lo,
              const std::uint64_t counter_hi,
              const py::object &a,
              const py::object &b,
              const dpnp::tensor::usm_ndarray &result,
              const std::vector<sycl::event> &depends,
              const sample_dispatch_table_t &dispatch_table);

// Fills `table` and binds it to the Python function `name`
template <template <typename> class SamplerT,
          template <typename> class SupportedT>
void def_sampler(py::module_ &m,
                 const char *name,
                 const char *doc,
                 sample_dispatch_table_t &table)
{
    init_sample_dispatch_table<SamplerT, SupportedT>(table);

    auto sample_pyapi = [&table](sycl::queue &exec_q, const int engine,
                                 const std::vector<std::uint64_t> &seed,
                                 const std::uint64_t counter_lo,
                                 const std::uint64_t counter_hi,
                                 const py::object &a, const py::object &b,
                                 const dpnp::tensor::usm_ndarray &result,
                                 const std::vector<sycl::event> &depends) {
        return py_sample(exec_q, engine, seed, counter_lo, counter_hi, a, b,
                         result, depends, table);
    };

    m.def(name, sample_pyapi, doc, py::arg("sycl_queue"), py::arg("engine"),
          py::arg("seed"), py::arg("counter_lo"), py::arg("counter_hi"),
          py::arg("a"), py::arg("b"), py::arg("result"),
          py::arg("depends") = py::list());
}
} // namespace dpnp::extensions::rng
