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
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kernels/engines.hpp"
#include "kernels/sample.hpp"
#include "kernels/samplers.hpp"

// utils extension header
#include "ext/common.hpp"

// dpnp tensor headers
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

namespace dpnp::extensions::rng
{
namespace py = pybind11;
namespace td_ns = dpnp::tensor::type_dispatch;
namespace rng_ns = dpnp::kernels::rng;

// A distribution parameter as passed from Python: either a C-contiguous
// usm_ndarray with the size of the result, whose data pointer is `data` and
// whose type lookup id is `type_id`, or a Python scalar.
struct ParamArg
{
    const char *data = nullptr;
    int type_id = -1;
    py::object scalar;
};

typedef sycl::event (*sample_fn_ptr_t)(sycl::queue &,
                                       const rng_ns::EngineState &,
                                       const std::size_t,
                                       const int,
                                       char *,
                                       const std::vector<ParamArg> &,
                                       const std::vector<sycl::event> &);

// Samplers of a result type, indexed by the engine and the result type
using sample_dispatch_table_t =
    sample_fn_ptr_t[rng_ns::num_engines][td_ns::num_types];

// Discrete samplers, whose result is always int64, indexed by the engine and
// by whether they compute in double (1) or in float (0)
using discrete_dispatch_table_t = sample_fn_ptr_t[rng_ns::num_engines][2];

namespace detail
{
template <typename T>
rng_ns::Param<T> to_param(const ParamArg &arg, const std::size_t pos)
{
    if (arg.data != nullptr) {
        if (arg.type_id != td_ns::GetTypeid<T>{}.get()) {
            throw py::value_error("Parameter array " + std::to_string(pos) +
                                  " has an unexpected data type.");
        }
        return {reinterpret_cast<const T *>(arg.data), T{}};
    }
    return {nullptr, py::cast<T>(arg.scalar)};
}

template <typename SamplerT, std::size_t... Is>
SamplerT make_sampler(const std::vector<ParamArg> &params,
                      std::index_sequence<Is...>)
{
    using param_types = typename SamplerT::param_types;
    return SamplerT(
        to_param<std::tuple_element_t<Is, param_types>>(params[Is], Is)...);
}

template <typename T>
void validate_type_for_device(sycl::queue &exec_q)
{
    dpnp::tensor::type_utils::validate_type_for_device<T>(exec_q);
}

template <typename Tuple, std::size_t... Is>
void validate_types_for_device(sycl::queue &exec_q, std::index_sequence<Is...>)
{
    (validate_type_for_device<std::tuple_element_t<Is, Tuple>>(exec_q), ...);
}
} // namespace detail

// Checks the parameters and the result against SamplerT, then submits the
// kernel, unless there is nothing to sample. ComputeT is the type the
// sampler computes in, which the device must support.
template <typename EngineTraitsT,
          typename SamplerT,
          typename ComputeT = typename SamplerT::result_type>
sycl::event sample_impl(sycl::queue &exec_q,
                        const rng_ns::EngineState &state,
                        const std::size_t nelems,
                        const int result_type_id,
                        char *result,
                        const std::vector<ParamArg> &params,
                        const std::vector<sycl::event> &depends)
{
    using T = typename SamplerT::result_type;
    using param_types = typename SamplerT::param_types;
    constexpr std::size_t n_params = std::tuple_size_v<param_types>;

    if (result_type_id != td_ns::GetTypeid<T>{}.get()) {
        throw py::value_error("The result array has an unexpected data type.");
    }
    if (params.size() != n_params) {
        throw py::value_error("Expected " + std::to_string(n_params) +
                              " parameters, got " +
                              std::to_string(params.size()) + ".");
    }

    detail::validate_type_for_device<T>(exec_q);
    detail::validate_type_for_device<ComputeT>(exec_q);
    detail::validate_types_for_device<param_types>(
        exec_q, std::make_index_sequence<n_params>{});

    const SamplerT sampler = detail::make_sampler<SamplerT>(
        params, std::make_index_sequence<n_params>{});

    if (nelems == 0) {
        return sycl::event{};
    }

    T *res = reinterpret_cast<T *>(result);
    const std::size_t n_wi = rng_ns::num_work_items(nelems);

    return exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using SampleKernel = rng_ns::SampleFunctor<EngineTraitsT, SamplerT>;
        cgh.parallel_for<SampleKernel>(
            sycl::range<1>(n_wi),
            SampleKernel(state, sampler, res, nelems, n_wi));
    });
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
                return sample_impl<EngineTraitsT, SamplerT<T>>;
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

template <template <typename> class SamplerT>
void init_discrete_dispatch_table(discrete_dispatch_table_t &table)
{
    static_assert(rng_ns::num_engines == 1,
                  "Every engine must have a row in the dispatch table");

    using Philox = rng_ns::Philox4x32x10;
    constexpr int philox = static_cast<int>(rng_ns::engine_id::philox4x32x10);
    table[philox][0] = sample_impl<Philox, SamplerT<float>, float>;
    table[philox][1] = sample_impl<Philox, SamplerT<double>, double>;
}

// Looks up the sampler of `result`'s type, and throws if there is none
extern sample_fn_ptr_t lookup_sampler(const sample_dispatch_table_t &table,
                                      const int engine,
                                      const dpnp::tensor::usm_ndarray &result);

extern sample_fn_ptr_t lookup_sampler(const discrete_dispatch_table_t &table,
                                      const int engine,
                                      const bool double_precision);

extern std::pair<sycl::event, sycl::event>
    py_sample(sycl::queue &exec_q,
              const std::vector<std::uint64_t> &seed,
              const std::uint64_t counter_lo,
              const std::uint64_t counter_hi,
              const py::sequence &params,
              const dpnp::tensor::usm_ndarray &result,
              const std::vector<sycl::event> &depends,
              const sample_fn_ptr_t fn);

// Fills `table` and binds it to the Python function `name`, which takes the
// distribution parameters as a sequence
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
                                 const py::sequence &params,
                                 const dpnp::tensor::usm_ndarray &result,
                                 const std::vector<sycl::event> &depends) {
        const sample_fn_ptr_t fn = lookup_sampler(table, engine, result);
        return py_sample(exec_q, seed, counter_lo, counter_hi, params, result,
                         depends, fn);
    };

    m.def(name, sample_pyapi, doc, py::arg("sycl_queue"), py::arg("engine"),
          py::arg("seed"), py::arg("counter_lo"), py::arg("counter_hi"),
          py::arg("params"), py::arg("result"),
          py::arg("depends") = py::list());
}

// As def_sampler, for a discrete sampler, whose Python function also takes
// whether to compute in double precision
template <template <typename> class SamplerT>
void def_discrete_sampler(py::module_ &m,
                          const char *name,
                          const char *doc,
                          discrete_dispatch_table_t &table)
{
    init_discrete_dispatch_table<SamplerT>(table);

    auto sample_pyapi = [&table](sycl::queue &exec_q, const int engine,
                                 const std::vector<std::uint64_t> &seed,
                                 const std::uint64_t counter_lo,
                                 const std::uint64_t counter_hi,
                                 const py::sequence &params,
                                 const dpnp::tensor::usm_ndarray &result,
                                 const bool double_precision,
                                 const std::vector<sycl::event> &depends) {
        const sample_fn_ptr_t fn =
            lookup_sampler(table, engine, double_precision);
        return py_sample(exec_q, seed, counter_lo, counter_hi, params, result,
                         depends, fn);
    };

    m.def(name, sample_pyapi, doc, py::arg("sycl_queue"), py::arg("engine"),
          py::arg("seed"), py::arg("counter_lo"), py::arg("counter_hi"),
          py::arg("params"), py::arg("result"), py::arg("double_precision"),
          py::arg("depends") = py::list());
}
} // namespace dpnp::extensions::rng
