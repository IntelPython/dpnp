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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"
#include <pybind11/pybind11.h>

#include "common.hpp"

// dpnp tensor headers
#include "utils/output_validation.hpp"
#include "utils/type_dispatch.hpp"

namespace dpnp::extensions::rng
{
namespace py = pybind11;
namespace td_ns = dpnp::tensor::type_dispatch;

namespace
{
void check_engine(const int engine)
{
    if (engine < 0 || engine >= rng_ns::num_engines) {
        throw py::value_error("Unknown engine " + std::to_string(engine) + ".");
    }
}

ParamArg to_param_arg(const std::size_t pos,
                      sycl::queue &exec_q,
                      const py::handle &param,
                      const dpnp::tensor::usm_ndarray &result)
{
    if (!py::isinstance<dpnp::tensor::usm_ndarray>(param)) {
        return {nullptr, -1, py::reinterpret_borrow<py::object>(param)};
    }

    auto arr = py::reinterpret_borrow<dpnp::tensor::usm_ndarray>(param);
    const std::string name = "Parameter array " + std::to_string(pos);

    if (arr.get_size() != result.get_size()) {
        throw py::value_error(name +
                              " and the result must have the same size.");
    }
    if (!arr.is_c_contiguous()) {
        throw py::value_error(name + " is not c-contiguous.");
    }
    if (!dpnp::utils::queues_are_compatible(exec_q, {arr.get_queue()})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queue.");
    }

    auto array_types = td_ns::usm_ndarray_types();
    const int type_id = array_types.typenum_to_lookup_id(arr.get_typenum());
    return {arr.get_data(), type_id, py::none()};
}
} // namespace

sample_fn_ptr_t lookup_sampler(const sample_dispatch_table_t &table,
                               const int engine,
                               const dpnp::tensor::usm_ndarray &result)
{
    check_engine(engine);

    auto array_types = td_ns::usm_ndarray_types();
    const int result_type_id =
        array_types.typenum_to_lookup_id(result.get_typenum());
    const sample_fn_ptr_t fn = table[engine][result_type_id];

    if (fn == nullptr) {
        throw std::runtime_error("Type of given array is not supported");
    }
    return fn;
}

sample_fn_ptr_t lookup_sampler(const discrete_dispatch_table_t &table,
                               const int engine,
                               const bool double_precision)
{
    check_engine(engine);
    return table[engine][double_precision ? 1 : 0];
}

std::pair<sycl::event, sycl::event>
    py_sample(sycl::queue &exec_q,
              const std::vector<std::uint64_t> &seed,
              const std::uint64_t counter_lo,
              const std::uint64_t counter_hi,
              const py::sequence &params,
              const dpnp::tensor::usm_ndarray &result,
              const std::vector<sycl::event> &depends,
              const sample_fn_ptr_t fn)
{
    dpnp::tensor::validation::CheckWritable::throw_if_not_writable(result);

    if (!dpnp::utils::queues_are_compatible(exec_q, {result.get_queue()})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queue.");
    }

    if (!result.is_c_contiguous()) {
        throw py::value_error("The result array is not c-contiguous.");
    }

    if (seed.size() > rng_ns::max_seed_words) {
        throw py::value_error("The seed has too many words.");
    }

    // hold the parameters in a tuple, to keep them alive with the result
    const py::tuple params_tuple(params);
    std::vector<ParamArg> param_args;
    param_args.reserve(params_tuple.size());
    for (std::size_t i = 0; i < params_tuple.size(); ++i) {
        param_args.push_back(to_param_arg(i, exec_q, params_tuple[i], result));
    }

    rng_ns::EngineState state{};
    for (std::size_t i = 0; i < seed.size(); ++i) {
        state.seed[i] = seed[i];
    }
    state.counter_lo = counter_lo;
    state.counter_hi = counter_hi;

    auto array_types = td_ns::usm_ndarray_types();
    const int result_type_id =
        array_types.typenum_to_lookup_id(result.get_typenum());
    const std::size_t nelems = result.get_size();

    // checks the parameters, and submits nothing when there is nothing to
    // sample
    sycl::event sample_ev = fn(exec_q, state, nelems, result_type_id,
                               result.get_data(), param_args, depends);
    if (nelems == 0) {
        return std::make_pair(sycl::event{}, sycl::event{});
    }

    sycl::event args_ev = dpnp::utils::keep_args_alive(
        exec_q, {result, params_tuple}, {sample_ev});

    return std::make_pair(args_ev, sample_ev);
}
} // namespace dpnp::extensions::rng
