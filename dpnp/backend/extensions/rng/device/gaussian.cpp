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

// dpctl tensor headers
#include "utils/memory_overlap.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

// dpctl tensor headers
#include "kernels/alignment.hpp"

using dpctl::tensor::kernels::alignment_utils::is_aligned;
using dpctl::tensor::kernels::alignment_utils::required_alignment;

#include "gaussian.hpp"

// #include "dpnp_utils.hpp"

namespace dpnp
{
namespace backend
{
namespace ext
{
namespace rng
{
namespace device
{
namespace dpctl_td_ns = dpctl::tensor::type_dispatch;
namespace mkl_rng_dev = oneapi::mkl::rng::device;
namespace py = pybind11;
namespace type_utils = dpctl::tensor::type_utils;

typedef sycl::event (*gaussian_impl_fn_ptr_t)(sycl::queue &,
                                           const std::uint32_t,
                                           const double,
                                           const double,
                                           const std::uint64_t,
                                           char *,
                                           std::vector<sycl::event> &,
                                           const std::vector<sycl::event> &);

static gaussian_impl_fn_ptr_t gaussian_dispatch_vector[dpctl_td_ns::num_types];

// template <typename DataType, typename Method = mkl_rng_dev::gaussian_method::by_default>
template <typename DataType>
class gaussian_kernel;

// template <typename DataType, typename Method = mkl_rng_dev::gaussian_method::by_default>
template <typename DataType>
static sycl::event gaussian_impl(sycl::queue& exec_q,
                                 const std::uint32_t seed,
                                 const double mean_val,
                                 const double stddev_val,
                                 const std::uint64_t n,
                                 char *out_ptr,
                                 std::vector<sycl::event> &host_task_events,
                                 const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<DataType>(exec_q);

    using Method = mkl_rng_dev::gaussian_method::by_default;

    const bool enable_sg_load = is_aligned<required_alignment>(out_ptr);
    DataType *out = reinterpret_cast<DataType *>(out_ptr);
    DataType mean = static_cast<DataType>(mean_val);
    DataType stddev = static_cast<DataType>(stddev_val);

    constexpr std::size_t vec_sz = 8;
    constexpr std::size_t items_per_wi = 4;
    constexpr std::size_t local_size = 256;
    const std::size_t wg_items = local_size * vec_sz * items_per_wi;
    const std::size_t global_size = ((n + wg_items - 1) / (wg_items)) * local_size;

    sycl::event distr_event;
    
    try {
        distr_event = exec_q.parallel_for<gaussian_kernel<DataType>>(
            sycl::nd_range<1>({global_size}, {local_size}), depends,
            [=](sycl::nd_item<1> nd_it)
            {
                auto global_id = nd_it.get_global_id();
                
                auto sg = nd_it.get_sub_group();
                const std::uint8_t sg_size = sg.get_local_range()[0];
                const std::uint8_t max_sg_size = sg.get_max_local_range()[0];
                const size_t base = items_per_wi * vec_sz * (nd_it.get_group(0) * nd_it.get_local_range(0) + sg.get_group_id()[0] * max_sg_size);

                mkl_rng_dev::gaussian<DataType, Method> distr(mean, stddev);

                if (enable_sg_load && (sg_size == max_sg_size) && (base + items_per_wi * vec_sz * sg_size < n)) {
                    auto engine = mkl_rng_dev::mrg32k3a<vec_sz>(seed, n * global_id);

#pragma unroll
                    for (std::uint16_t it = 0; it < items_per_wi * vec_sz; it += vec_sz) {
                        size_t offset = base + static_cast<size_t>(it) * static_cast<size_t>(sg_size);
                        auto out_multi_ptr = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(&out[offset]);

                        sycl::vec<DataType, vec_sz> rng_val_vec = mkl_rng_dev::generate(distr, engine);
                        sg.store<vec_sz>(out_multi_ptr, rng_val_vec);
                    }
                }
                else {
                    auto engine = mkl_rng_dev::mrg32k3a(seed, n * global_id);

                    for (size_t k = base + sg.get_local_id()[0]; k < n; k += sg_size) {
                        out[k] = mkl_rng_dev::generate(distr, engine);
                    }
                }
        });
    } catch (oneapi::mkl::exception const &e) {
        std::stringstream error_msg;

        error_msg << "Unexpected MKL exception caught during gaussian call:\nreason: " << e.what();
        throw std::runtime_error(error_msg.str());
    }
    return distr_event;
}

std::pair<sycl::event, sycl::event> gaussian(sycl::queue exec_q,
                                             const std::uint32_t seed,
                                             const double mean,
                                             const double stddev,
                                             const std::uint64_t n,
                                             dpctl::tensor::usm_ndarray res,
                                             const std::vector<sycl::event> &depends)
{
    const int res_nd = res.get_ndim();

    // if (eig_vecs_nd != 2) {
    //     throw py::value_error("Unexpected ndim=" + std::to_string(eig_vecs_nd) +
    //                           " of an output array with eigenvectors");
    // }
    // else if (eig_vals_nd != 1) {
    //     throw py::value_error("Unexpected ndim=" + std::to_string(eig_vals_nd) +
    //                           " of an output array with eigenvalues");
    // }

    const py::ssize_t *res_shape = res.get_shape_raw();

    // if (eig_vecs_shape[0] != eig_vecs_shape[1]) {
    //     throw py::value_error("Output array with eigenvectors with be square");
    // }
    // else if (eig_vecs_shape[0] != eig_vals_shape[0]) {
    //     throw py::value_error(
    //         "Eigenvectors and eigenvalues have different shapes");
    // }

    size_t src_nelems(1);

    for (int i = 0; i < res_nd; ++i) {
        src_nelems *= static_cast<size_t>(res_shape[i]);
    }

    if (src_nelems == 0) {
        // nothing to do
        return std::make_pair(sycl::event(), sycl::event());
    }

    // check compatibility of execution queue and allocation queue
    // if (!dpctl::utils::queues_are_compatible(exec_q, {eig_vecs, eig_vals})) {
    //     throw py::value_error(
    //         "Execution queue is not compatible with allocation queues");
    // }

    // auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    // if (overlap(eig_vecs, eig_vals)) {
    //     throw py::value_error("Arrays with eigenvectors and eigenvalues are "
    //                           "overlapping segments of memory");
    // }

    bool is_res_c_contig = res.is_c_contiguous();
    if (!is_res_c_contig) {
        throw py::value_error(
            "An array with input matrix must be C-contiguous");
    }

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    int res_type_id =
        array_types.typenum_to_lookup_id(res.get_typenum());

    gaussian_impl_fn_ptr_t gaussian_fn = gaussian_dispatch_vector[res_type_id];
    if (gaussian_fn == nullptr) {
        throw py::value_error("No gaussian implementation defined for a required type");
    }

    char *res_data = res.get_data();

    std::vector<sycl::event> host_task_events;
    sycl::event gaussian_ev =
        gaussian_fn(exec_q, seed, mean, stddev, n, res_data,
                 host_task_events, depends);

    sycl::event args_ev = dpctl::utils::keep_args_alive(
        exec_q, {res}, host_task_events);
    return std::make_pair(args_ev, gaussian_ev);
}

template <typename T>
struct GaussianTypePairSupportFactory
{
    static constexpr bool is_defined = std::disjunction<
        dpctl_td_ns::TypePairDefinedEntry<T, double, T, double>,
        dpctl_td_ns::TypePairDefinedEntry<T, float, T, float>,
        // fall-through
        dpctl_td_ns::NotDefinedEntry>::is_defined;
};

template <typename fnT, typename T>
struct GaussianContigFactory
{
    fnT get()
    {
        if constexpr (GaussianTypePairSupportFactory<T>::is_defined) {
            return gaussian_impl<T>;
        }
        else {
            return nullptr;
        }
    }
};

void init_gaussian_dispatch_vector(void)
{
    dpctl_td_ns::DispatchVectorBuilder<gaussian_impl_fn_ptr_t, GaussianContigFactory,
                                       dpctl_td_ns::num_types>
        contig;
    contig.populate_dispatch_vector(gaussian_dispatch_vector);
}
} // namespace device
} // namespace rng
} // namespace ext
} // namespace backend
} // namespace dpnp
