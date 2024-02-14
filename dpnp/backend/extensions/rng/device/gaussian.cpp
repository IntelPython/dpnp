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
// #include "utils/memory_overlap.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

// dpctl tensor headers
#include "kernels/alignment.hpp"

using dpctl::tensor::kernels::alignment_utils::disabled_sg_loadstore_wrapper_krn;
using dpctl::tensor::kernels::alignment_utils::is_aligned;
using dpctl::tensor::kernels::alignment_utils::required_alignment;

#include "common_impl.hpp"
#include "gaussian.hpp"

#include "engine/engine_base.hpp"
#include "engine/engine_builder.hpp"

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

constexpr int no_of_methods = 2; // number of methods of gaussian distribution

template <typename DataT, typename Method>
struct GaussianDistr
{
private:
    const DataT mean_;
    const DataT stddev_;

public:
    using method_type = Method;
    using result_type = DataT;
    using distr_type = typename mkl_rng_dev::gaussian<DataT, Method>;

    GaussianDistr(const DataT mean, const DataT stddev)
        : mean_(mean), stddev_(stddev)
    {
    }

    inline auto operator()(void) const
    {
        return distr_type(mean_, stddev_);
    }
};

typedef sycl::event (*gaussian_impl_fn_ptr_t)(engine::EngineBase *engine,
                                              const double,
                                              const double,
                                              const std::uint64_t,
                                              char *,
                                              const std::vector<sycl::event> &);

static gaussian_impl_fn_ptr_t gaussian_dispatch_table[engine::no_of_engines][dpctl_td_ns::num_types][no_of_methods];

template <typename EngineT, typename DataT,  typename Method, unsigned int items_per_wi>
class gaussian_kernel;

template <typename EngineT, typename DataT, typename Method>
static sycl::event gaussian_impl(engine::EngineBase *engine,
                                 const double mean_val,
                                 const double stddev_val,
                                 const std::uint64_t n,
                                 char *out_ptr,
                                 const std::vector<sycl::event> &depends)
{
    auto &exec_q = engine->get_queue();
    type_utils::validate_type_for_device<DataT>(exec_q);

    DataT *out = reinterpret_cast<DataT *>(out_ptr);
    DataT mean = static_cast<DataT>(mean_val);
    DataT stddev = static_cast<DataT>(stddev_val);

    constexpr std::size_t vec_sz = EngineT::vec_size;
    constexpr std::size_t items_per_wi = 4;
    constexpr std::size_t local_size = 256;
    const std::size_t wg_items = local_size * vec_sz * items_per_wi;
    const std::size_t global_size = ((n + wg_items - 1) / (wg_items)) * local_size;

    sycl::event distr_event;

    try {
        distr_event = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using EngineBuilderT = engine::Builder<EngineT>;
            EngineBuilderT eng_builder(engine);
            eng_builder.print(); // TODO: remove

            using GaussianDistrT = GaussianDistr<DataT, Method>;
            GaussianDistrT distr(mean, stddev);

            if (is_aligned<required_alignment>(out_ptr)) {
                constexpr bool enable_sg_load = true;
                using KernelName = gaussian_kernel<EngineT, DataT, Method, items_per_wi>;

                cgh.parallel_for<KernelName>(sycl::nd_range<1>({global_size}, {local_size}),
                    details::RngContigFunctor<EngineBuilderT, DataT, GaussianDistrT, items_per_wi, enable_sg_load>(eng_builder, distr, out, n));
            }
            else {
                constexpr bool disable_sg_load = false;
                using InnerKernelName = gaussian_kernel<EngineT, DataT, Method, items_per_wi>;
                using KernelName = disabled_sg_loadstore_wrapper_krn<InnerKernelName>;

                cgh.parallel_for<KernelName>(sycl::nd_range<1>({global_size}, {local_size}),
                    details::RngContigFunctor<EngineBuilderT, DataT, GaussianDistrT, items_per_wi, disable_sg_load>(eng_builder, distr, out, n));
            }
        });
    } catch (oneapi::mkl::exception const &e) {
        std::stringstream error_msg;

        error_msg << "Unexpected MKL exception caught during gaussian call:\nreason: " << e.what();
        throw std::runtime_error(error_msg.str());
    } catch (sycl::exception const &e) {
        std::stringstream error_msg;

        error_msg << "Unexpected SYCL exception caught during gaussian call:\n" << e.what();
        throw std::runtime_error(error_msg.str());
    }
    return distr_event;
}

std::pair<sycl::event, sycl::event> gaussian(engine::EngineBase *engine,
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
    auto res_offsets = res.get_minmax_offsets();
    // destination must be ample enough to accommodate all elements
    {
        size_t range =
            static_cast<size_t>(res_offsets.second - res_offsets.first);
        if (range + 1 < res_nelems) {
            throw py::value_error(
                "Destination array can not accommodate all the elements of source array.");
        }
    }

    bool is_res_c_contig = res.is_c_contiguous();
    if (!is_res_c_contig) {
        throw std::runtime_error("Only population of contiguous array is supported.");
    }

    auto enginge_id = engine->get_type().id();
    if (enginge_id >= engine::no_of_engines) {
        throw std::runtime_error("Unknown engine type=" + std::to_string(enginge_id) + " for gaussian distribution.");
    }

    if (method_id >= no_of_methods) {
        throw std::runtime_error("Unknown method=" + std::to_string(method_id) + " for gaussian distribution.");
    }

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    int res_type_id = array_types.typenum_to_lookup_id(res.get_typenum());

    auto gaussian_fn = gaussian_dispatch_table[enginge_id][res_type_id][method_id];
    if (gaussian_fn == nullptr) {
        throw py::value_error("No gaussian implementation defined for a required type");
    }

    char *res_data = res.get_data();
    sycl::event gaussian_ev = gaussian_fn(engine, mean, stddev, n, res_data, depends);

    sycl::event ht_ev = dpctl::utils::keep_args_alive(exec_q, {res}, {gaussian_ev});
     return std::make_pair(ht_ev, gaussian_ev);
}

template <typename funcPtrT,
          template <typename fnT, typename E, typename T, typename M> typename factory,
          int _no_of_engines,
          int _no_of_types,
          int _no_of_methods>
class Dispatch3DTableBuilder
{
private:
    template <typename E, typename T>
    const std::vector<funcPtrT> row_per_method() const
    {
        std::vector<funcPtrT> per_method = {
            factory<funcPtrT, E, T, mkl_rng_dev::gaussian_method::by_default>{}.get(),
            factory<funcPtrT, E, T, mkl_rng_dev::gaussian_method::box_muller2>{}.get(),
        };
        assert(per_method.size() == _no_of_methods);
        return per_method;
    }

    template <typename E>
    auto table_per_type_and_method() const
    {
        std::vector<std::vector<funcPtrT>>
                   table_by_type = {row_per_method<E, bool>(),
                                    row_per_method<E, int8_t>(),
                                    row_per_method<E, uint8_t>(),
                                    row_per_method<E, int16_t>(),
                                    row_per_method<E, uint16_t>(),
                                    row_per_method<E, int32_t>(),
                                    row_per_method<E, uint32_t>(),
                                    row_per_method<E, int64_t>(),
                                    row_per_method<E, uint64_t>(),
                                    row_per_method<E, sycl::half>(),
                                    row_per_method<E, float>(),
                                    row_per_method<E, double>(),
                                    row_per_method<E, std::complex<float>>(),
                                    row_per_method<E, std::complex<double>>()};
        assert(table_by_type.size() == _no_of_types);
        return table_by_type;
    }

public:
    Dispatch3DTableBuilder() = default;
    ~Dispatch3DTableBuilder() = default;

    void populate(funcPtrT table[][_no_of_types][_no_of_methods]) const
    {
        const auto map_by_engine = {table_per_type_and_method<mkl_rng_dev::mrg32k3a<8>>()};
        assert(map_by_engine.size() == _no_of_engines);

        std::uint16_t engine_id = 0;
        for (auto &table_by_type : map_by_engine) {
            std::uint16_t type_id = 0;
            for (auto &row_by_method : table_by_type) {
                std::uint16_t method_id = 0;
                for (auto &fn_ptr : row_by_method) {
                    table[engine_id][type_id][method_id] = fn_ptr;
                    ++method_id;
                }
                ++type_id;
            }
            ++engine_id;
        }
    }
};

template <typename Ty, typename ArgTy, typename Method, typename argMethod>
struct TypePairDefinedEntry : std::bool_constant<std::is_same_v<Ty, ArgTy> &&
                                                 std::is_same_v<Method, argMethod>>
{
    static constexpr bool is_defined = true;
};

template <typename T, typename M>
struct GaussianTypePairSupportFactory
{
    static constexpr bool is_defined = std::disjunction<
        TypePairDefinedEntry<T, double, M, mkl_rng_dev::gaussian_method::by_default>,
        TypePairDefinedEntry<T, double, M, mkl_rng_dev::gaussian_method::box_muller2>,
        TypePairDefinedEntry<T, float, M, mkl_rng_dev::gaussian_method::by_default>,
        TypePairDefinedEntry<T, float, M, mkl_rng_dev::gaussian_method::box_muller2>,
        // fall-through
        dpctl_td_ns::NotDefinedEntry>::is_defined;
};

template <typename fnT, typename E, typename T, typename M>
struct GaussianContigFactory
{
    fnT get()
    {
        if constexpr (GaussianTypePairSupportFactory<T, M>::is_defined) {
            return gaussian_impl<E, T, M>;
        }
        else {
            return nullptr;
        }
    }
};

void init_gaussian_dispatch_table(void)
{
    Dispatch3DTableBuilder<gaussian_impl_fn_ptr_t, GaussianContigFactory, engine::no_of_engines, dpctl_td_ns::num_types, no_of_methods> contig;
    contig.populate(gaussian_dispatch_table);
}
} // namespace device
} // namespace rng
} // namespace ext
} // namespace backend
} // namespace dpnp
