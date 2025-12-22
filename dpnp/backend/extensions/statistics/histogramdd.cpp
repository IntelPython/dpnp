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
#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "histogram_common.hpp"
#include "histogramdd.hpp"

using dpctl::tensor::usm_ndarray;

using namespace statistics::histogram;
using namespace ext::common;

namespace
{

template <typename DataStorage, typename EdgesCountStorage>
struct EdgesDd
{
    static constexpr bool const sync_after_init = true;
    using EdgesT = typename DataStorage::value_type;
    using EdgesCountT = typename EdgesCountStorage::value_type;
    using ncT = typename std::remove_const<EdgesT>::type;
    using LocalData = sycl::local_accessor<ncT, 1>;
    using boundsT = std::tuple<EdgesT *, EdgesT *>;

    EdgesDd(const EdgesT *global_edges,
            const size_t total_size,
            const EdgesCountT *global_edges_sizes,
            const size_t dims,
            sycl::handler &cgh)
        : edges(global_edges, sycl::range<1>(total_size), cgh),
          edges_size(global_edges_sizes, sycl::range<1>(dims), cgh)
    {
        min = LocalData(dims, cgh);
        max = LocalData(dims, cgh);
    }

    template <int _Dims>
    void init(const sycl::nd_item<_Dims> &item) const
    {
        auto group = item.get_group();
        edges.init(item);
        edges_size.init(item);

        if constexpr (DataStorage::sync_after_init) {
            sycl::group_barrier(group, sycl::memory_scope::work_group);
        }

        const auto *edges_ptr = edges.get_ptr();
        auto *min_ptr = &min[0];
        auto *max_ptr = &max[0];

        if (group.leader()) {
            for (uint32_t i = 0; i < edges_size.size(); ++i) {
                const auto size = edges_size[i];
                min_ptr[i] = edges_ptr[0];
                max_ptr[i] = edges_ptr[size - 1];
                edges_ptr += size;
            }
        }
    }

    boundsT get_bounds() const
    {
        return {&min[0], &max[0]};
    }

    auto get_bin_for_dim(const EdgesT &val,
                         const EdgesT *edges_data,
                         const uint32_t edges_count) const
    {
        const uint32_t bins_count = edges_count - 1;

        uint32_t bin = std::upper_bound(edges_data, edges_data + edges_count,
                                        val, Less<EdgesT>{}) -
                       edges_data - 1;
        bin = std::min(bin, bins_count - 1);

        return bin;
    }

    template <int _Dims, typename dT>
    auto get_bin(const sycl::nd_item<_Dims> &,
                 const dT *val,
                 const boundsT &) const
    {
        uint32_t resulting_bin = 0;
        const auto *edges_ptr = &edges[0];
        const uint32_t dims = edges_size.size();

        for (uint32_t i = 0; i < dims; ++i) {
            const uint32_t curr_edges_count = edges_size[i];

            const auto bin_id =
                get_bin_for_dim(val[i], edges_ptr, curr_edges_count);
            resulting_bin = resulting_bin * (curr_edges_count - 1) + bin_id;
            edges_ptr += curr_edges_count;
        }

        return resulting_bin;
    }

    template <typename dT>
    bool in_bounds(const dT *val, const boundsT &bounds) const
    {
        const EdgesT *min = std::get<0>(bounds);
        const EdgesT *max = std::get<1>(bounds);
        const uint32_t dims = edges_size.size();

        auto in_bounds = true;
        for (uint32_t i = 0; i < dims; ++i) {
            in_bounds &= check_in_bounds(val[i], min[i], max[i]);
        }

        return in_bounds;
    }

private:
    DataStorage edges;
    EdgesCountStorage edges_size;
    LocalData min;
    LocalData max;
};

template <typename T, typename esT>
using CachedEdgesDd = EdgesDd<CachedData<const T, 1>, CachedData<const esT, 1>>;

template <typename T, typename esT>
using UncachedEdgesDd =
    EdgesDd<UncachedData<const T, 1>, CachedData<const esT, 1>>;

template <typename T,
          typename BinsT,
          typename HistType = size_t,
          typename EdgesCountT = size_t>
struct HistogramddF
{
    static sycl::event impl(sycl::queue &exec_q,
                            const void *vin,
                            const void *vbins_edges,
                            const void *vbins_edges_count,
                            const void *vweights,
                            void *vout,
                            const size_t bins_count,
                            const size_t size,
                            const size_t total_edges,
                            const size_t dims,
                            const std::vector<sycl::event> &depends)
    {
        const T *in = static_cast<const T *>(vin);
        const BinsT *bins_edges = static_cast<const BinsT *>(vbins_edges);
        const HistType *weights = static_cast<const HistType *>(vweights);
        const EdgesCountT *bins_edges_count =
            static_cast<const EdgesCountT *>(vbins_edges_count);
        HistType *out = static_cast<HistType *>(vout);

        auto device = exec_q.get_device();

        const uint32_t local_size = get_max_local_size(exec_q);
        constexpr uint32_t WorkPI = 128; // empirically found number

        const auto nd_range = make_ndrange(size, local_size, WorkPI);

        return exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            auto dispatch_edges = [&](const uint32_t local_mem,
                                      const auto &weights, auto &hist) {
                if (device.is_gpu() && (local_mem >= total_edges)) {
                    auto edges = CachedEdgesDd<BinsT, EdgesCountT>(
                        bins_edges, total_edges, bins_edges_count, dims, cgh);
                    submit_histogram(in, size, dims, WorkPI, hist, edges,
                                     weights, nd_range, cgh);
                }
                else {
                    auto edges = UncachedEdgesDd<BinsT, EdgesCountT>(
                        bins_edges, total_edges, bins_edges_count, dims, cgh);
                    submit_histogram(in, size, dims, WorkPI, hist, edges,
                                     weights, nd_range, cgh);
                }
            };

            auto dispatch_bins = [&](const auto &weights) {
                auto local_mem_size = get_local_mem_size_in_items<T>(device);
                local_mem_size -= 2 * dims; // for min-max values
                local_mem_size -= CeilDiv(dims * sizeof(EdgesCountT),
                                          sizeof(T)); // for edges count

                if (local_mem_size >= bins_count) {
                    const auto local_hist_count = get_local_hist_copies_count(
                        local_mem_size, local_size, bins_count);

                    auto hist = HistWithLocalCopies<HistType>(
                        out, bins_count, local_hist_count, cgh);
                    const uint32_t free_local_mem =
                        local_mem_size - hist.size();

                    dispatch_edges(free_local_mem, weights, hist);
                }
                else {
                    auto hist = HistGlobalMemory<HistType>(out);
                    auto edges = UncachedEdgesDd<BinsT, EdgesCountT>(
                        bins_edges, total_edges, bins_edges_count, dims, cgh);
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

template <typename T, typename HistType = size_t>
using HistogramddF_ = HistogramddF<T, T, HistType>;

using SupportedTypes =
    std::tuple<std::tuple<uint64_t, float>,
               std::tuple<int64_t, float>,
               std::tuple<uint64_t, double>,
               std::tuple<int64_t, double>,
               std::tuple<uint64_t, std::complex<float>>,
               std::tuple<int64_t, std::complex<float>>,
               std::tuple<uint64_t, std::complex<double>>,
               std::tuple<int64_t, std::complex<double>>,
               std::tuple<float, float>,
               std::tuple<double, double>,
               std::tuple<float, std::complex<float>>,
               std::tuple<double, std::complex<double>>,
               std::tuple<std::complex<float>, float>,
               std::tuple<std::complex<double>, double>,
               std::tuple<std::complex<float>, std::complex<float>>,
               std::tuple<std::complex<double>, std::complex<double>>>;
} // namespace

Histogramdd::Histogramdd() : dispatch_table("sample", "histogram")
{
    dispatch_table.populate_dispatch_table<SupportedTypes, HistogramddF_>();
}

std::tuple<sycl::event, sycl::event> Histogramdd::call(
    const dpctl::tensor::usm_ndarray &sample,
    const dpctl::tensor::usm_ndarray &bins_edges,
    const dpctl::tensor::usm_ndarray &bins_edges_count,
    const std::optional<const dpctl::tensor::usm_ndarray> &weights,
    dpctl::tensor::usm_ndarray &histogram,
    const std::vector<sycl::event> &depends)
{
    validate(sample, bins_edges, weights, histogram);

    if (sample.get_size() == 0) {
        return {sycl::event(), sycl::event()};
    }

    const int sample_typenum = sample.get_typenum();
    const int hist_typenum = histogram.get_typenum();

    auto histogram_func = dispatch_table.get(sample_typenum, hist_typenum);

    auto exec_q = sample.get_queue();

    void *weights_ptr =
        weights.has_value() ? weights.value().get_data() : nullptr;

    if (sample.get_shape(1) != bins_edges_count.get_size()) {
        throw py::value_error("'sample' parameter has shape (" +
                              std::to_string(sample.get_shape(0)) + ", " +
                              std::to_string(sample.get_shape(1)) + ")" +
                              " so array of bins edges must be of size " +
                              std::to_string(sample.get_shape(1)) +
                              " but actually is " +
                              std::to_string(bins_edges_count.get_shape(0)));
    }

    auto ev = histogram_func(exec_q, sample.get_data(), bins_edges.get_data(),
                             bins_edges_count.get_data(), weights_ptr,
                             histogram.get_data(), histogram.get_size(),
                             sample.get_shape(0), bins_edges.get_shape(0),
                             sample.get_shape(1), depends);

    sycl::event args_ev;
    if (weights.has_value()) {
        args_ev = dpctl::utils::keep_args_alive(
            exec_q,
            {sample, bins_edges, bins_edges_count, weights.value(), histogram},
            {ev});
    }
    else {
        args_ev = dpctl::utils::keep_args_alive(
            exec_q, {sample, bins_edges, bins_edges_count, histogram}, {ev});
    }

    return std::make_pair(args_ev, ev);
}

std::unique_ptr<Histogramdd> histdd;

void statistics::histogram::populate_histogramdd(py::module_ m)
{
    using namespace std::placeholders;

    histdd.reset(new Histogramdd());

    auto hist_func =
        [histp = histdd.get()](
            const dpctl::tensor::usm_ndarray &sample,
            const dpctl::tensor::usm_ndarray &bins,
            const dpctl::tensor::usm_ndarray &bins_count,
            const std::optional<const dpctl::tensor::usm_ndarray> &weights,
            dpctl::tensor::usm_ndarray &histogram,
            const std::vector<sycl::event> &depends) {
            return histp->call(sample, bins, bins_count, weights, histogram,
                               depends);
        };

    m.def("histogramdd", hist_func,
          "Compute the multidimensional histogram of some data.",
          py::arg("sample"), py::arg("bins"), py::arg("bins_count"),
          py::arg("weights"), py::arg("histogram"),
          py::arg("depends") = py::list());

    auto histogramdd_dtypes = [histp = histdd.get()]() {
        return histp->dispatch_table.get_all_supported_types();
    };

    m.def("histogramdd_dtypes", histogramdd_dtypes,
          "Get the supported data types for histogramdd.");
}
