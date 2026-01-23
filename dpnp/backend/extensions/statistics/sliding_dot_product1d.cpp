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

#include <complex>
#include <memory>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dpnp4pybind11.hpp"

// utils extension header
#include "ext/common.hpp"

// dpctl tensor headers
#include "utils/type_dispatch.hpp"

#include "sliding_dot_product1d.hpp"
#include "sliding_window1d.hpp"

// #include <iostream>

namespace dpctl_td_ns = dpctl::tensor::type_dispatch;
using dpctl::tensor::usm_ndarray;

using namespace statistics::sliding_window1d;
using namespace ext::common;

namespace
{
template <typename T>
struct SlidingDotProductF
{
    static sycl::event impl(sycl::queue &exec_q,
                            const void *v_ain,
                            const void *v_vin,
                            void *v_out,
                            const size_t a_size,
                            const size_t v_size,
                            const size_t l_pad,
                            const size_t r_pad,
                            const std::vector<sycl::event> &depends)
    {
        const T *ain = static_cast<const T *>(v_ain);
        const T *vin = static_cast<const T *>(v_vin);
        T *out = static_cast<T *>(v_out);

        auto device = exec_q.get_device();
        const auto local_size = get_max_local_size(device);
        const auto work_size = l_pad + r_pad + a_size - v_size + 1;

        constexpr int32_t WorkPI = 4;

        return exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            auto input = make_padded_span(ain, a_size, l_pad);
            auto kernel = make_span(vin, v_size);
            auto result = make_span(out, work_size);

            if (v_size > 8) {
                const auto nd_range =
                    make_ndrange(work_size, local_size, WorkPI);
                submit_sliding_window1d<WorkPI>(
                    input, kernel, std::multiplies<>{}, std::plus<>{}, result,
                    nd_range, cgh);
            }
            else {
                const auto nd_range =
                    make_ndrange(work_size, local_size, WorkPI);
                submit_sliding_window1d_small_kernel<WorkPI>(
                    input, kernel, std::multiplies<>{}, std::plus<>{}, result,
                    nd_range, cgh);
            }
        });
    }
};

using SupportedTypes = std::tuple<uint32_t,
                                  int32_t,
                                  uint64_t,
                                  int64_t,
                                  float,
                                  double,
                                  std::complex<float>,
                                  std::complex<double>>;
} // namespace

SlidingDotProduct1d::SlidingDotProduct1d() : dispatch_table("a")
{
    dispatch_table
        .populate_dispatch_table<SupportedTypes, SlidingDotProductF>();
}

std::tuple<sycl::event, sycl::event>
    SlidingDotProduct1d::call(const dpctl::tensor::usm_ndarray &a,
                              const dpctl::tensor::usm_ndarray &v,
                              dpctl::tensor::usm_ndarray &out,
                              const size_t l_pad,
                              const size_t r_pad,
                              const std::vector<sycl::event> &depends)
{
    validate(a, v, out, l_pad, r_pad);

    const int a_typenum = a.get_typenum();

    auto corr_func = dispatch_table.get(a_typenum);

    auto exec_q = a.get_queue();

    auto ev = corr_func(exec_q, a.get_data(), v.get_data(), out.get_data(),
                        a.get_shape(0), v.get_shape(0), l_pad, r_pad, depends);

    sycl::event args_ev;
    args_ev = dpctl::utils::keep_args_alive(exec_q, {a, v, out}, {ev});

    return {args_ev, ev};
}

std::unique_ptr<SlidingDotProduct1d> sdp;

void statistics::sliding_window1d::populate_sliding_dot_product1d(py::module_ m)
{
    using namespace std::placeholders;

    sdp.reset(new SlidingDotProduct1d());

    auto sdp_func = [sdpp =
                         sdp.get()](const dpctl::tensor::usm_ndarray &a,
                                    const dpctl::tensor::usm_ndarray &v,
                                    dpctl::tensor::usm_ndarray &out,
                                    const size_t l_pad, const size_t r_pad,
                                    const std::vector<sycl::event> &depends) {
        return sdpp->call(a, v, out, l_pad, r_pad, depends);
    };

    m.def("sliding_dot_product1d", sdp_func, "1d sliding dot product.",
          py::arg("a"), py::arg("v"), py::arg("out"), py::arg("l_pad"),
          py::arg("r_pad"), py::arg("depends") = py::list());

    auto sdp_dtypes = [sdpp = sdp.get()]() {
        return sdpp->dispatch_table.get_all_supported_types();
    };

    m.def("sliding_dot_product1d_dtypes", sdp_dtypes,
          "Get the supported data types for sliding_dot_product1d_dtypes.");
}
