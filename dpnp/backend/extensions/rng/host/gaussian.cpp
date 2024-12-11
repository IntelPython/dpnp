//*****************************************************************************
// Copyright (c) 2023, Intel Corporation
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

#include <pybind11/pybind11.h>

#include <oneapi/mkl/rng.hpp>

// dpctl tensor headers
#include "utils/output_validation.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

#include "gaussian.hpp"

#include "dispatch/matrix.hpp"
#include "dispatch/table_builder.hpp"

namespace dpnp::backend::ext::rng::host
{
namespace dpctl_td_ns = dpctl::tensor::type_dispatch;
namespace dpctl_tu_ns = dpctl::tensor::type_utils;
namespace mkl_rng = oneapi::mkl::rng;
namespace py = pybind11;

constexpr auto no_of_methods = 2; // number of methods of gaussian distribution
constexpr auto no_of_engines = device::engine::no_of_engines;

typedef sycl::event (*gaussian_impl_fn_ptr_t)(
    device::engine::EngineBase *engine,
    const double,
    const double,
    const std::uint64_t,
    char *,
    const std::vector<sycl::event> &);

static gaussian_impl_fn_ptr_t gaussian_dispatch_table[no_of_engines]
                                                     [dpctl_td_ns::num_types]
                                                     [no_of_methods];

template <typename EngineT, typename DataT, typename Method>
static sycl::event gaussian_impl(device::engine::EngineBase *engine,
                                 const double mean_val,
                                 const double stddev_val,
                                 const std::uint64_t n,
                                 char *out_ptr,
                                 const std::vector<sycl::event> &depends)
{
    auto &exec_q = engine->get_queue();
    dpctl_tu_ns::validate_type_for_device<DataT>(exec_q);

    DataT *out = reinterpret_cast<DataT *>(out_ptr);
    DataT mean = static_cast<DataT>(mean_val);
    DataT stddev = static_cast<DataT>(stddev_val);

    auto seed_values = engine->get_seeds();
    auto no_of_seeds = seed_values.size();
    if (no_of_seeds > 1) {
        throw std::runtime_error("");
    }

    mkl_rng::gaussian<DataT, Method> distribution(mean, stddev);
    mkl_rng::mcg59 eng(exec_q, seed_values[0]);

    return mkl_rng::generate(distribution, eng, n, out, depends);
}

std::pair<sycl::event, sycl::event>
    gaussian(device::engine::EngineBase *engine,
             const std::uint8_t method_id,
             const double mean,
             const double stddev,
             const std::uint64_t n,
             dpctl::tensor::usm_ndarray res,
             const std::vector<sycl::event> &depends)
{
    auto &exec_q = engine->get_queue();

    const int res_nd = res.get_ndim();
    const py::ssize_t *res_shape = res.get_shape_raw();

    size_t res_nelems(1);
    for (int i = 0; i < res_nd; ++i) {
        res_nelems *= static_cast<size_t>(res_shape[i]);
    }

    if (res_nelems == 0) {
        // nothing to do
        return std::make_pair(sycl::event(), sycl::event());
    }

    // ensure that output is ample enough to accommodate all elements
    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(res, res_nelems);

    if (!dpctl::utils::queues_are_compatible(exec_q, {res})) {
        throw py::value_error(
            "Execution queue is not compatible with the allocation queue");
    }

    bool is_res_c_contig = res.is_c_contiguous();
    if (!is_res_c_contig) {
        throw std::runtime_error(
            "Only population of contiguous array is supported.");
    }

    auto enginge_id = engine->get_type().id();
    if (enginge_id >= device::engine::no_of_engines) {
        throw std::runtime_error(
            "Unknown engine type=" + std::to_string(enginge_id) +
            " for gaussian distribution.");
    }

    if (method_id >= no_of_methods) {
        throw std::runtime_error("Unknown method=" + std::to_string(method_id) +
                                 " for gaussian distribution.");
    }

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    int res_type_id = array_types.typenum_to_lookup_id(res.get_typenum());

    auto gaussian_fn =
        gaussian_dispatch_table[enginge_id][res_type_id][method_id];
    if (gaussian_fn == nullptr) {
        throw py::value_error(
            "No gaussian implementation defined for a required type");
    }

    char *res_data = res.get_data();
    sycl::event gaussian_ev =
        gaussian_fn(engine, mean, stddev, n, res_data, depends);

    sycl::event ht_ev =
        dpctl::utils::keep_args_alive(exec_q, {res}, {gaussian_ev});
    return std::make_pair(ht_ev, gaussian_ev);
}

template <typename fnT, typename E, typename T, typename M>
struct GaussianContigFactory
{
    fnT get()
    {
        if constexpr (dispatch::GaussianTypePairSupportFactory<T,
                                                               M>::is_defined) {
            return gaussian_impl<E, T, M>;
        }
        else {
            return nullptr;
        }
    }
};

void init_gaussian_dispatch_3d_table(void)
{
    dispatch::Dispatch3DTableBuilder<gaussian_impl_fn_ptr_t,
                                     GaussianContigFactory, no_of_engines,
                                     dpctl_td_ns::num_types, no_of_methods>
        contig;
    contig.populate(gaussian_dispatch_table);
}
} // namespace dpnp::backend::ext::rng::host
