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

#include <algorithm>
#include <complex>
#include <memory>
#include <tuple>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// dpctl tensor headers
#include "dpctl4pybind11.hpp"
#include "utils/type_dispatch.hpp"

#include "histogram.hpp"
#include "histogram_common.hpp"

namespace dpctl_td_ns = dpctl::tensor::type_dispatch;
using dpctl::tensor::usm_ndarray;

using namespace statistics::histogram;
using namespace ext::common;

namespace
{

template <typename T, typename DataStorage>
struct HistogramEdges
{
    static constexpr bool const sync_after_init = DataStorage::sync_after_init;
    using boundsT = std::tuple<T, T>;

    HistogramEdges(const T *global_data, size_t size, sycl::handler &cgh)
        : data(global_data, sycl::range<1>(size), cgh)
    {
    }

    template <int _Dims>
    void init(const sycl::nd_item<_Dims> &item) const
    {
        data.init(item);
    }

    boundsT get_bounds() const
    {
        auto min = data.get_ptr()[0];
        auto max = data.get_ptr()[data.size() - 1];
        return {min, max};
    }

    template <int _Dims, typename dT>
    size_t get_bin(const sycl::nd_item<_Dims> &,
                   const dT *val,
                   const boundsT &) const
    {
        uint32_t edges_count = data.size();
        uint32_t bins_count = edges_count - 1;
        const auto *bins = data.get_ptr();

        uint32_t bin =
            std::upper_bound(bins, bins + edges_count, val[0], Less<dT>{}) -
            bins - 1;
        bin = std::min(bin, bins_count - 1);

        return bin;
    }

    template <typename dT>
    bool in_bounds(const dT *val, const boundsT &bounds) const
    {
        return check_in_bounds(val[0], std::get<0>(bounds),
                               std::get<1>(bounds));
    }

private:
    DataStorage data;
};

template <typename T>
using CachedEdges = HistogramEdges<T, CachedData<const T, 1>>;

template <typename T>
using UncachedEdges = HistogramEdges<T, UncachedData<const T, 1>>;

using DefaultHistType = int64_t;

template <typename T, typename BinsT, typename HistType = DefaultHistType>
struct HistogramF
{
    static sycl::event impl(sycl::queue &exec_q,
                            const void *vin,
                            const void *vbins_edges,
                            const void *vweights,
                            void *vout,
                            const size_t bins_count,
                            const size_t size,
                            const std::vector<sycl::event> &depends)
    {
        const T *in = static_cast<const T *>(vin);
        const BinsT *bins_edges = static_cast<const BinsT *>(vbins_edges);
        const HistType *weights = static_cast<const HistType *>(vweights);
        HistType *out = static_cast<HistType *>(vout);

        auto device = exec_q.get_device();
        const auto local_size = get_max_local_size(device);

        constexpr uint32_t WorkPI = 128; // empirically found number

        const auto nd_range = make_ndrange(size, local_size, WorkPI);

        return exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);
            constexpr uint32_t dims = 1;

            auto dispatch_edges = [&](uint32_t local_mem, const auto &weights,
                                      auto &hist) {
                if (device.is_gpu() && (local_mem >= bins_count + 1)) {
                    auto edges =
                        CachedEdges<BinsT>(bins_edges, bins_count + 1, cgh);
                    submit_histogram(in, size, dims, WorkPI, hist, edges,
                                     weights, nd_range, cgh);
                }
                else {
                    auto edges =
                        UncachedEdges<BinsT>(bins_edges, bins_count + 1, cgh);
                    submit_histogram(in, size, dims, WorkPI, hist, edges,
                                     weights, nd_range, cgh);
                }
            };

            auto dispatch_bins = [&](const auto &weights) {
                const auto local_mem_size =
                    get_local_mem_size_in_items<T>(device);
                if (local_mem_size >= bins_count) {
                    const auto local_hist_count = get_local_hist_copies_count(
                        local_mem_size, local_size, bins_count);

                    auto hist = HistWithLocalCopies<HistType>(
                        out, bins_count, local_hist_count, cgh);
                    const auto free_local_mem = local_mem_size - hist.size();

                    dispatch_edges(free_local_mem, weights, hist);
                }
                else {
                    auto hist = HistGlobalMemory<HistType>(out);
                    auto edges =
                        UncachedEdges<BinsT>(bins_edges, bins_count + 1, cgh);
                    submit_histogram(in, size, dims, WorkPI, hist, edges,
                                     weights, nd_range, cgh);
                }
            };

            if (weights) {
                auto _weights = Weights(weights);
                dispatch_bins(_weights);
            }
            else {
                auto _weights = NoWeights();
                dispatch_bins(_weights);
            }
        });
    }
};

template <typename SampleType, typename HistType>
using HistogramF_ = HistogramF<SampleType, SampleType, HistType>;

} // namespace

using SupportedTypes =
    std::tuple<std::tuple<uint64_t, DefaultHistType>,
               std::tuple<int64_t, DefaultHistType>,
               std::tuple<uint64_t, float>,
               std::tuple<int64_t, float>,
               std::tuple<uint64_t, double>,
               std::tuple<int64_t, double>,
               std::tuple<uint64_t, std::complex<float>>,
               std::tuple<int64_t, std::complex<float>>,
               std::tuple<uint64_t, std::complex<double>>,
               std::tuple<int64_t, std::complex<double>>,
               std::tuple<float, DefaultHistType>,
               std::tuple<double, DefaultHistType>,
               std::tuple<float, float>,
               std::tuple<double, double>,
               std::tuple<float, std::complex<float>>,
               std::tuple<double, std::complex<double>>,
               std::tuple<std::complex<float>, DefaultHistType>,
               std::tuple<std::complex<double>, DefaultHistType>,
               std::tuple<std::complex<float>, float>,
               std::tuple<std::complex<double>, double>>;

Histogram::Histogram() : dispatch_table("sample", "histogram")
{
    dispatch_table.populate_dispatch_table<SupportedTypes, HistogramF_>();
}

std::tuple<sycl::event, sycl::event>
    Histogram::call(const dpctl::tensor::usm_ndarray &sample,
                    const dpctl::tensor::usm_ndarray &bins,
                    std::optional<const dpctl::tensor::usm_ndarray> &weights,
                    dpctl::tensor::usm_ndarray &histogram,
                    const std::vector<sycl::event> &depends)
{
    validate(sample, bins, weights, histogram);

    if (sample.get_size() == 0) {
        return {sycl::event(), sycl::event()};
    }

    const int sample_typenum = sample.get_typenum();
    const int hist_typenum = histogram.get_typenum();

    auto histogram_func = dispatch_table.get(sample_typenum, hist_typenum);

    auto exec_q = sample.get_queue();

    void *weights_ptr =
        weights.has_value() ? weights.value().get_data() : nullptr;

    auto ev =
        histogram_func(exec_q, sample.get_data(), bins.get_data(), weights_ptr,
                       histogram.get_data(), histogram.get_shape(0),
                       sample.get_shape(0), depends);

    sycl::event args_ev;
    if (weights.has_value()) {
        args_ev = dpctl::utils::keep_args_alive(
            exec_q, {sample, bins, weights.value(), histogram}, {ev});
    }
    else {
        args_ev = dpctl::utils::keep_args_alive(
            exec_q, {sample, bins, histogram}, {ev});
    }

    return std::make_pair(args_ev, ev);
}

std::unique_ptr<Histogram> hist;

void statistics::histogram::populate_histogram(py::module_ m)
{
    using namespace std::placeholders;

    hist.reset(new Histogram());

    auto hist_func =
        [histp = hist.get()](
            const dpctl::tensor::usm_ndarray &sample,
            const dpctl::tensor::usm_ndarray &bins,
            std::optional<const dpctl::tensor::usm_ndarray> &weights,
            dpctl::tensor::usm_ndarray &histogram,
            const std::vector<sycl::event> &depends) {
            return histp->call(sample, bins, weights, histogram, depends);
        };

    m.def("histogram", hist_func, "Compute the histogram of a dataset.",
          py::arg("sample"), py::arg("bins"), py::arg("weights"),
          py::arg("histogram"), py::arg("depends") = py::list());

    auto histogram_dtypes = [histp = hist.get()]() {
        return histp->dispatch_table.get_all_supported_types();
    };

    m.def("histogram_dtypes", histogram_dtypes,
          "Get the supported data types for histogram.");
}
