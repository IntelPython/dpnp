//*****************************************************************************
// Copyright (c) 2025, Intel Corporation
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
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// dpctl tensor headers
#include "dpctl4pybind11.hpp"
#include "utils/type_dispatch.hpp"

#include "interpolate.hpp"

namespace dpctl_td_ns = dpctl::tensor::type_dispatch;
using dpctl::tensor::usm_ndarray;

using namespace math::interpolate;

namespace
{

template <typename T, typename IndexT>
struct interpolate_kernel
{
    static sycl::event impl(sycl::queue &exec_q,
                            const void *vx,
                            const void *vidx,
                            const void *vxp,
                            const void *vfp,
                            void *vout,
                            const size_t xp_size,
                            const size_t n,
                            const std::vector<sycl::event> &depends)
    {
        const T *x = static_cast<const T *>(vx);
        const T *xp = static_cast<const T *>(vxp);
        const T *fp = static_cast<const T *>(vfp);
        const IndexT *idx = static_cast<const IndexT *>(vidx);
        T *out = static_cast<T *>(vout);

        // size_t n = x.get_size();

        // std::vector<float> result(n);
        // sycl::event copy_ev = exec_q.memcpy(result.data(), output.get_data(), n * sizeof(float), {ev});

        // for (size_t i = 0; i< n; i++){
        //     std::cout << result[i] << " ";
        // }

        return exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            //     T left = fp[0];
            //     T right = fp[xp_size - 1];

            //     // IndexT x_idx = idx[i] - 1;

            //     // if (sycl::isnan(x[i])) {
            //     //     out[i] = x[i];
            //     // }
            //     // else if (x_idx < 0) {
            //     //     out[i] = left;
            //     // }
            //     // //old version check
            //     // // else if (x[i] == xp[xp_size - 1]) {
            //     // //     out[i] = right;
            //     // // }
            //     // // else if (idx[i] >= xp_size - 1) {
            //     // //     out[i] = right;
            //     // // }
            //     // // new version check
            //     // else if (idx[i] == xp_size) {
            //     //     out[i] = right;
            //     // }
            //     // else if (idx[i] == xp_size - 1) {
            //     //     out[i] = fp[x_idx];
            //     // }
            //     // else if (x[i] == xp[x_idx]) {
            //     //     out[i] = fp[x_idx];
            //     // }

            //     IndexT j = idx[i];

            //     if (sycl::isnan(x[i])) {
            //         out[i] = x[i];
            //     }
            //     else if (j == 0) {
            //         out[i] = left;
            //     }
            //     else if (j == xp_size) {
            //         out[i] = right;
            //     }
            //     else {
            //         IndexT x_idx = j - 1;

            //         if (x[i] == xp[x_idx]) {
            //             out[i] = fp[x_idx];
            //         }
            //         else {
            //         T slope = (fp[x_idx + 1] - fp[x_idx]) / (xp[x_idx + 1] - xp[x_idx]);
            //         T res = slope * (x[i] - xp[x_idx]) + fp[x_idx];
            //         // T res = (x[i] - xp[x_idx]) + fp[x_idx];

            //         if (sycl::isnan(res)) {
            //             res = slope * (x[i] - xp[x_idx + 1]) + fp[x_idx + 1];
            //             if (sycl::isnan(res) && (fp[x_idx] == fp[x_idx + 1])) {
            //                 res = fp[x_idx];
            //             }
            //         }
            //         out[i] = res;
            //     }
            // }

            T left = fp[0];
            T right = fp[xp_size - 1];
            IndexT x_idx = idx[i] - 1;

            if (sycl::isnan(x[i])) {
                out[i] = x[i];
            }
            else if (x_idx < 0) {
                out[i] = left;
            }
            else if (x[i] == xp[xp_size - 1]) {
                out[i] = right;
            }
            else if (x_idx >= xp_size - 1) {
                out[i] = right;
            }
            else if (x[i] == xp[x_idx]) {
                out[i] = fp[x_idx];
            }
            else {
                T slope = (fp[x_idx + 1] - fp[x_idx]) / (xp[x_idx + 1] - xp[x_idx]);
                T res = slope * (x[i] - xp[x_idx]) + fp[x_idx];

                if (sycl::isnan(res)) {
                    res = slope * (x[i] - xp[x_idx + 1]) + fp[x_idx + 1];
                    if (sycl::isnan(res) && (fp[x_idx] == fp[x_idx + 1])) {
                        res = fp[x_idx];
                    }
                }

                out[i] = res;
            }

                // out[i] = x[i];
            });
        });
    }
};

// using SupportedTypes = std::tuple<std::tuple<float>, std::tuple<double>>;
// using SupportedTypes = std::tuple<std::tuple<uint64_t, int64_t>,
//                                   std::tuple<int64_t, int64_t>,
//                                   std::tuple<uint64_t, float>,
//                                   std::tuple<int64_t, float>,
//                                   std::tuple<uint64_t, double>,
//                                   std::tuple<int64_t, double>,
//                                   std::tuple<uint64_t, std::complex<float>>,
//                                   std::tuple<int64_t, std::complex<float>>,
//                                   std::tuple<uint64_t, std::complex<double>>,
//                                   std::tuple<int64_t, std::complex<double>>,
//                                   std::tuple<float, int64_t>,
//                                   std::tuple<double, int64_t>,
//                                   std::tuple<float, float>,
//                                   std::tuple<double, double>,
//                                   std::tuple<float, std::complex<float>>,
//                                   std::tuple<double, std::complex<double>>,
//                                   std::tuple<std::complex<float>, int64_t>,
//                                   std::tuple<std::complex<double>, int64_t>,
//                                   std::tuple<std::complex<float>, float>,
//                                   std::tuple<std::complex<double>, double>>;

using SupportedTypes = std::tuple<
    // std::tuple<float, int64_t>,
    // std::tuple<double, int64_t>,
    std::tuple<float, size_t>,
    std::tuple<double, size_t>>;

} // namespace

Interpolate::Interpolate() : dispatch_table("x", "idx")
{
    dispatch_table.populate_dispatch_table<SupportedTypes, interpolate_kernel>();
}

std::tuple<sycl::event, sycl::event>
Interpolate::call(const dpctl::tensor::usm_ndarray &x,
                  const dpctl::tensor::usm_ndarray &idx,
                  const dpctl::tensor::usm_ndarray &xp,
                  const dpctl::tensor::usm_ndarray &fp,
                  const size_t xp_size,
                  dpctl::tensor::usm_ndarray &output,
                  const std::vector<sycl::event> &depends)
{
    // validate(x, xp, fp, output);

    if (x.get_size() == 0) {
        return {sycl::event(), sycl::event()};
    }

    const int x_typenum = x.get_typenum();
    const int idx_typenum = idx.get_typenum();

    auto interp_func = dispatch_table.get(x_typenum, idx_typenum);

    auto exec_q = x.get_queue();

    // size_t n = x.get_size();
    // const size_t m = xp.get_size();

    // std::vector<float> x_h(n);
    // std::vector<float> xp_h(n);
    // std::vector<float> fp_h(n);
    // std::vector<float> output_h(n);

    // sycl::event copy_1 = exec_q.memcpy(x_h.data(), x.get_data(), n * sizeof(float));
    // sycl::event copy_2 = exec_q.memcpy(xp_h.data(), xp.get_data(), m * sizeof(float));
    // sycl::event copy_3 = exec_q.memcpy(fp_h.data(), fp.get_data(), m * sizeof(float));
    // sycl::event copy_4 = exec_q.memcpy(output_h.data(), output.get_data(), n * sizeof(float));

    // copy_1.wait();
    // copy_2.wait();
    // copy_3.wait();
    // copy_4.wait();

    // std::cout << "x: " << std::endl;
    // for (size_t i = 0; i< n; i++){
    //     std::cout << x_h[i] << " ";
    // }
    // std::cout << "\n";

    // std::cout << "xp: " << std::endl;
    // for (size_t i = 0; i< m; i++){
    //     std::cout << xp_h[i] << " ";
    // }
    // std::cout << "\n";


    // std::cout << "fp: " << std::endl;
    // for (size_t i = 0; i< m; i++){
    //     std::cout << fp_h[i] << " ";
    // }
    // std::cout << "\n";


    // std::cout << "out: " << std::endl;
    // for (size_t i = 0; i< n; i++){
    //     std::cout << output_h[i] << " ";
    // }
    // std::cout << "\n";

    auto ev =
        interp_func(exec_q, x.get_data(), idx.get_data(), xp.get_data(), fp.get_data(),
                    output.get_data(), xp.get_size(), x.get_size(), depends);

    ev.wait();

    auto args_ev = dpctl::utils::keep_args_alive(
        exec_q, {x, idx, xp, fp, output}, {ev});

    // size_t n = x.get_size();

    // std::vector<float> result(n);
    // sycl::event copy_ev = exec_q.memcpy(result.data(), output.get_data(), n * sizeof(float), {ev});

    // copy_ev.wait();

    // std::cout << "out_host: " << std::endl;
    // for (size_t i = 0; i< n; i++){
    //     std::cout << result[i] << " ";
    // }

    // std::cout << "\n";

    return {args_ev, ev};
}

std::unique_ptr<Interpolate> interp;

void math::interpolate::populate_interpolate(py::module_ m)
{
    using namespace std::placeholders;

    interp.reset(new Interpolate());

    auto interp_func =
        [interpp = interp.get()](
            const dpctl::tensor::usm_ndarray &x,
            const dpctl::tensor::usm_ndarray &idx,
            const dpctl::tensor::usm_ndarray &xp,
            const dpctl::tensor::usm_ndarray &fp,
            const size_t xp_size,
            dpctl::tensor::usm_ndarray &output,
            const std::vector<sycl::event> &depends) {
            return interpp->call(x, idx, xp, fp, xp_size, output, depends);
        };

    m.def("interpolate", interp_func, "Perform linear interpolation.",
          py::arg("x"), py::arg("idx"), py::arg("xp"), py::arg("fp"),
          py::arg("xp_size"), py::arg("output"), py::arg("depends") = py::list());

    auto interpolate_dtypes = [interpp = interp.get()]() {
        return interpp->dispatch_table.get_all_supported_types();
    };

    m.def("interpolate_dtypes", interpolate_dtypes,
          "Get the supported data types for interpolation.");
}
