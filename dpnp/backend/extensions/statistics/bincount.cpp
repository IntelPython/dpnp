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

#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bincount.hpp"
#include "histogram_common.hpp"

using dpctl::tensor::usm_ndarray;

using namespace statistics::histogram;
using namespace ext::common;

namespace
{

template <typename T>
struct BincountEdges
{
    static constexpr bool const sync_after_init = false;
    using boundsT = std::tuple<T, T>;

    BincountEdges(const T &min, const T &max)
    {
        this->min = min;
        this->max = max;
    }

    template <int _Dims>
    void init(const sycl::nd_item<_Dims> &) const
    {
    }

    boundsT get_bounds() const
    {
        return {min, max};
    }

    template <int _Dims, typename dT>
    size_t get_bin(const sycl::nd_item<_Dims> &,
                   const dT *val,
                   const boundsT &) const
    {
        return val[0] - min;
    }

    template <typename dT>
    bool in_bounds(const dT *val, const boundsT &bounds) const
    {
        return check_in_bounds(static_cast<T>(val[0]), std::get<0>(bounds),
                               std::get<1>(bounds));
    }

private:
    T min;
    T max;
};

using DefaultHistType = int64_t;

template <typename T, typename HistType = DefaultHistType>
struct BincountF
{
    static sycl::event impl(sycl::queue &exec_q,
                            const void *vin,
                            const uint64_t min,
                            const uint64_t max,
                            const void *vweights,
                            void *vout,
                            const size_t size,
                            const std::vector<sycl::event> &depends)
    {
        const T *in = static_cast<const T *>(vin);
        const HistType *weights = static_cast<const HistType *>(vweights);
        // shift output pointer by min elements
        HistType *out = static_cast<HistType *>(vout) + min;

        const size_t needed_bins_count = (max - min) + 1;

        const uint32_t local_size = get_max_local_size(exec_q);

        constexpr uint32_t WorkPI = 128; // empirically found number
        const auto nd_range = make_ndrange(size, local_size, WorkPI);

        return exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);
            constexpr uint32_t dims = 1;

            auto dispatch_bins = [&](const auto &weights) {
                const auto local_mem_size =
                    get_local_mem_size_in_items<T>(exec_q);
                if (local_mem_size >= needed_bins_count) {
                    const uint32_t local_hist_count =
                        get_local_hist_copies_count(local_mem_size, local_size,
                                                    needed_bins_count);

                    auto hist = HistWithLocalCopies<HistType>(
                        out, needed_bins_count, local_hist_count, cgh);

                    auto edges = BincountEdges(min, max);
                    submit_histogram(in, size, dims, WorkPI, hist, edges,
                                     weights, nd_range, cgh);
                }
                else {
                    auto hist = HistGlobalMemory<HistType>(out);
                    auto edges = BincountEdges(min, max);
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

using SupportedTypes = std::tuple<std::tuple<int64_t, DefaultHistType>,
                                  std::tuple<uint64_t, DefaultHistType>,
                                  std::tuple<int64_t, float>,
                                  std::tuple<uint64_t, float>,
                                  std::tuple<int64_t, double>,
                                  std::tuple<uint64_t, double>>;

} // namespace

Bincount::Bincount() : dispatch_table("sample", "histogram")
{
    dispatch_table.populate_dispatch_table<SupportedTypes, BincountF>();
}

std::tuple<sycl::event, sycl::event> Bincount::call(
    const dpctl::tensor::usm_ndarray &sample,
    const uint64_t min,
    const uint64_t max,
    const std::optional<const dpctl::tensor::usm_ndarray> &weights,
    dpctl::tensor::usm_ndarray &histogram,
    const std::vector<sycl::event> &depends)
{
    validate(sample, std::optional<const dpctl::tensor::usm_ndarray>(), weights,
             histogram);

    if (sample.get_size() == 0) {
        return {sycl::event(), sycl::event()};
    }

    const int sample_typenum = sample.get_typenum();
    const int hist_typenum = histogram.get_typenum();

    auto bincount_func = dispatch_table.get(sample_typenum, hist_typenum);

    auto exec_q = sample.get_queue();

    void *weights_ptr =
        weights.has_value() ? weights.value().get_data() : nullptr;

    auto ev = bincount_func(exec_q, sample.get_data(), min, max, weights_ptr,
                            histogram.get_data(), sample.get_size(), depends);

    sycl::event args_ev;
    if (weights.has_value()) {
        args_ev = dpctl::utils::keep_args_alive(
            exec_q, {sample, weights.value(), histogram}, {ev});
    }
    else {
        args_ev =
            dpctl::utils::keep_args_alive(exec_q, {sample, histogram}, {ev});
    }

    return std::make_pair(args_ev, ev);
}

std::unique_ptr<Bincount> bincount;

void statistics::histogram::populate_bincount(py::module_ m)
{
    using namespace std::placeholders;

    bincount.reset(new Bincount());

    auto bincount_func =
        [bincountp = bincount.get()](
            const dpctl::tensor::usm_ndarray &sample, int64_t min, int64_t max,
            std::optional<const dpctl::tensor::usm_ndarray> &weights,
            dpctl::tensor::usm_ndarray &histogram,
            const std::vector<sycl::event> &depends) {
            return bincountp->call(sample, min, max, weights, histogram,
                                   depends);
        };

    m.def("bincount", bincount_func,
          "Count number of occurrences of each value in array of non-negative "
          "ints.",
          py::arg("sample"), py::arg("min"), py::arg("max"), py::arg("weights"),
          py::arg("histogram"), py::arg("depends") = py::list());

    auto bincount_dtypes = [bincountp = bincount.get()]() {
        return bincountp->dispatch_table.get_all_supported_types();
    };

    m.def("bincount_dtypes", bincount_dtypes,
          "Get the supported data types for bincount.");
}
