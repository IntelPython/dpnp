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
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
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

using namespace histogram;

namespace
{

template <typename T, typename BinsT, typename HistType = size_t>
static sycl::event histogram_impl(sycl::queue &exec_q,
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

    uint32_t local_size =
        device.is_cpu()
            ? 256
            : device.get_info<sycl::info::device::max_work_group_size>();

    constexpr uint32_t WorkPI = 128; // empirically found number
    auto global_size = Align(CeilDiv(size, WorkPI), local_size);

    auto nd_range =
        sycl::nd_range(sycl::range<1>(global_size), sycl::range<1>(local_size));

    return exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        constexpr uint32_t dims = 1;

        auto dispatch_edges = [&](uint32_t local_mem, auto &weights,
                                  auto &hist) {
            if (device.is_gpu() && (local_mem >= bins_count + 1)) {
                auto edges = CachedEdges(bins_edges, bins_count + 1, cgh);
                submit_histogram(in, size, dims, WorkPI, hist, edges, weights,
                                 nd_range, cgh);
            }
            else {
                auto edges = UncachedEdges(bins_edges, bins_count + 1, cgh);
                submit_histogram(in, size, dims, WorkPI, hist, edges, weights,
                                 nd_range, cgh);
            }
        };

        auto dispatch_bins = [&](auto &weights) {
            auto local_mem_size =
                device.get_info<sycl::info::device::local_mem_size>() /
                sizeof(T);
            if (local_mem_size >= bins_count) {
                uint32_t max_local_copies = local_mem_size / bins_count;
                uint32_t local_hist_count = std::max(
                    std::min(
                        int(std::ceil((float(4 * local_size) / bins_count))),
                        16),
                    1);
                local_hist_count = std::min(local_hist_count, max_local_copies);

                auto hist = HistWithLocalCopies<HistType>(
                    out, bins_count, local_hist_count, cgh);
                uint32_t free_local_mem = local_mem_size - hist.size();

                dispatch_edges(free_local_mem, weights, hist);
            }
            else {
                auto hist = HistGlobalMemory<HistType>(out);
                auto edges = UncachedEdges(bins_edges, bins_count + 1, cgh);
                submit_histogram(in, size, dims, WorkPI, hist, edges, weights,
                                 nd_range, cgh);
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

template <typename fnT, typename dT, typename hT>
struct ContigFactory
{
    static constexpr bool is_defined = std::disjunction<
        dpctl_td_ns::TypePairDefinedEntry<dT, uint64_t, hT, int64_t>,
        dpctl_td_ns::TypePairDefinedEntry<dT, int64_t, hT, int64_t>,
        dpctl_td_ns::TypePairDefinedEntry<dT, uint64_t, hT, float>,
        dpctl_td_ns::TypePairDefinedEntry<dT, int64_t, hT, float>,
        dpctl_td_ns::TypePairDefinedEntry<dT, uint64_t, hT, double>,
        dpctl_td_ns::TypePairDefinedEntry<dT, int64_t, hT, double>,
        dpctl_td_ns::
            TypePairDefinedEntry<dT, uint64_t, hT, std::complex<float>>,
        dpctl_td_ns::TypePairDefinedEntry<dT, int64_t, hT, std::complex<float>>,
        dpctl_td_ns::
            TypePairDefinedEntry<dT, uint64_t, hT, std::complex<double>>,
        dpctl_td_ns::
            TypePairDefinedEntry<dT, int64_t, hT, std::complex<double>>,
        dpctl_td_ns::TypePairDefinedEntry<dT, float, hT, int64_t>,
        dpctl_td_ns::TypePairDefinedEntry<dT, double, hT, int64_t>,
        dpctl_td_ns::TypePairDefinedEntry<dT, float, hT, float>,
        dpctl_td_ns::TypePairDefinedEntry<dT, double, hT, double>,
        dpctl_td_ns::TypePairDefinedEntry<dT, float, hT, std::complex<float>>,
        dpctl_td_ns::TypePairDefinedEntry<dT, double, hT, std::complex<double>>,
        dpctl_td_ns::TypePairDefinedEntry<dT, std::complex<float>, hT, int64_t>,
        dpctl_td_ns::
            TypePairDefinedEntry<dT, std::complex<double>, hT, int64_t>,
        dpctl_td_ns::TypePairDefinedEntry<dT, std::complex<float>, hT, float>,
        dpctl_td_ns::TypePairDefinedEntry<dT, std::complex<double>, hT, double>,
        // fall-through
        dpctl_td_ns::NotDefinedEntry>::is_defined;

    fnT get()
    {
        if constexpr (is_defined) {
            return histogram_impl<dT, dT, hT>;
        }
        else {
            return nullptr;
        }
    }
};

using sycl_ext::histogram::Histogram;

Histogram::FnT
    dispatch(Histogram *hist, int data_typenum, int, int hist_typenum)
{
    auto array_types = dpctl_td_ns::usm_ndarray_types();
    const int data_type_id = array_types.typenum_to_lookup_id(data_typenum);
    const int hist_type_id = array_types.typenum_to_lookup_id(hist_typenum);

    auto histogram_fn = hist->dispatch_table[data_type_id][hist_type_id];

    if (histogram_fn == nullptr) {
        throw py::value_error("Unsupported data types"); // report types?
    }

    return histogram_fn;
}

} // namespace

Histogram::Histogram()
{
    dpctl_td_ns::DispatchTableBuilder<FnT, ContigFactory,
                                      dpctl_td_ns::num_types>
        contig;
    contig.populate_dispatch_table(dispatch_table);
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
    const int bins_typenum = bins.get_typenum();
    const int hist_typenum = histogram.get_typenum();

    auto histogram_func =
        dispatch(this, sample_typenum, bins_typenum, hist_typenum);

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

    return {args_ev, ev};
}

std::unique_ptr<Histogram> hist;

void sycl_ext::histogram::populate_histogram(py::module_ m)
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
}
