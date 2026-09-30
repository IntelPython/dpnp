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
ParamArg to_param_arg(const char *name,
                      sycl::queue &exec_q,
                      const py::object &param,
                      const dpnp::tensor::usm_ndarray &result)
{
    if (!py::isinstance<dpnp::tensor::usm_ndarray>(param)) {
        return {nullptr, param};
    }

    auto arr = py::reinterpret_borrow<dpnp::tensor::usm_ndarray>(param);

    if (arr.get_typenum() != result.get_typenum()) {
        throw py::value_error(std::string("Parameter array `") + name +
                              "` and the result must have the same data type.");
    }
    if (arr.get_size() != result.get_size()) {
        throw py::value_error(std::string("Parameter array `") + name +
                              "` and the result must have the same size.");
    }
    if (!arr.is_c_contiguous()) {
        throw py::value_error(std::string("Parameter array `") + name +
                              "` is not c-contiguous.");
    }
    if (!dpnp::utils::queues_are_compatible(exec_q, {arr.get_queue()})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queue.");
    }

    return {arr.get_data(), py::none()};
}
} // namespace

std::pair<sycl::event, sycl::event>
    py_sample(sycl::queue &exec_q,
              const int engine,
              const std::vector<std::uint64_t> &seed,
              const std::uint64_t counter_lo,
              const std::uint64_t counter_hi,
              const py::object &a,
              const py::object &b,
              const dpnp::tensor::usm_ndarray &result,
              const std::vector<sycl::event> &depends,
              const sample_dispatch_table_t &dispatch_table)
{
    dpnp::tensor::validation::CheckWritable::throw_if_not_writable(result);

    if (!dpnp::utils::queues_are_compatible(exec_q, {result.get_queue()})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queue.");
    }

    if (!result.is_c_contiguous()) {
        throw py::value_error("The result array is not c-contiguous.");
    }

    if (engine < 0 || engine >= rng_ns::num_engines) {
        throw py::value_error("Unknown engine " + std::to_string(engine) + ".");
    }

    if (seed.size() > rng_ns::max_seed_words) {
        throw py::value_error("The seed has too many words.");
    }

    const int result_typenum = result.get_typenum();
    auto array_types = td_ns::usm_ndarray_types();
    const int result_type_id = array_types.typenum_to_lookup_id(result_typenum);
    sample_fn_ptr_t fn = dispatch_table[engine][result_type_id];

    if (fn == nullptr) {
        throw std::runtime_error("Type of given array is not supported");
    }

    const ParamArg a_arg = to_param_arg("a", exec_q, a, result);
    const ParamArg b_arg = to_param_arg("b", exec_q, b, result);

    const std::size_t nelems = result.get_size();
    if (nelems == 0) {
        return std::make_pair(sycl::event{}, sycl::event{});
    }

    rng_ns::EngineState state{};
    for (std::size_t i = 0; i < seed.size(); ++i) {
        state.seed[i] = seed[i];
    }
    state.counter_lo = counter_lo;
    state.counter_hi = counter_hi;

    sycl::event sample_ev =
        fn(exec_q, state, nelems, result.get_data(), a_arg, b_arg, depends);

    sycl::event args_ev =
        dpnp::utils::keep_args_alive(exec_q, {result, a, b}, {sample_ev});

    return std::make_pair(args_ev, sample_ev);
}
} // namespace dpnp::extensions::rng
