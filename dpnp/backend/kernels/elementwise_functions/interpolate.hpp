#pragma once

#include <sycl/sycl.hpp>
#include <vector>

#include "utils/type_utils.hpp"

namespace dpnp::kernels::interpolate
{

template <typename TCoord, typename TValue>
sycl::event interpolate_impl(sycl::queue &q,
                             const void *vx,
                             const void *vidx,
                             const void *vxp,
                             const void *vfp,
                             void *vout,
                             std::size_t n,
                             std::size_t xp_size,
                             const std::vector<sycl::event> &depends)
{
    const TCoord *x = static_cast<const TCoord *>(vx);
    const std::size_t *idx = static_cast<const std::size_t *>(vidx);
    const TCoord *xp = static_cast<const TCoord *>(vxp);
    const TValue *fp = static_cast<const TValue *>(vfp);
    TValue *out = static_cast<TValue *>(vout);

    return q.submit([&](sycl::handler &h) {
        h.depends_on(depends);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            TValue left = fp[0];
            TValue right = fp[xp_size - 1];

            TCoord x_val = x[i];
            std::size_t x_idx = idx[i] - 1;

            if (sycl::isnan(x_val)) {
                out[i] = x_val;
            }
            else if (x_idx < 0) {
                out[i] = left;
            }
            else if (x_val == xp[xp_size - 1]) {
                out[i] = right;
            }
            else if (x_idx >= xp_size - 1) {
                out[i] = right;
            }
            else if (x_val == xp[x_idx]) {
                out[i] = fp[x_idx];
            }
            else {
                TValue slope =
                    (fp[x_idx + 1] - fp[x_idx]) / (xp[x_idx + 1] - xp[x_idx]);
                TValue res = slope * (x_val - xp[x_idx]) + fp[x_idx];

                if (sycl::isnan(res)) {
                    res = slope * (x_val - xp[x_idx + 1]) + fp[x_idx + 1];
                    if (sycl::isnan(res) && (fp[x_idx] == fp[x_idx + 1])) {
                        res = fp[x_idx];
                    }
                }
                out[i] = res;
            }
        });
    });
}

template <typename TCoord, typename TValue>
sycl::event interpolate_complex_impl(sycl::queue &q,
                                     const void *vx,
                                     const void *vidx,
                                     const void *vxp,
                                     const void *vfp,
                                     void *vout,
                                     std::size_t n,
                                     std::size_t xp_size,
                                     const std::vector<sycl::event> &depends)
{
    const TCoord *x = static_cast<const TCoord *>(vx);
    const std::size_t *idx = static_cast<const std::size_t *>(vidx);
    const TCoord *xp = static_cast<const TCoord *>(vxp);
    const TValue *fp = static_cast<const TValue *>(vfp);
    TValue *out = static_cast<TValue *>(vout);

    using realT = typename TValue::value_type;

    return q.submit([&](sycl::handler &h) {
        h.depends_on(depends);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            realT left_r = fp[0].real();
            realT right_r = fp[xp_size - 1].real();
            realT left_i = fp[0].imag();
            realT right_i = fp[xp_size - 1].imag();

            TCoord x_val = x[i];
            std::size_t x_idx = idx[i] - 1;

            realT res_r = 0.0;
            realT res_i = 0.0;

            if (sycl::isnan(x_val)) {
                res_r = x_val;
                res_i = 0.0;
            }
            else if (x_idx < 0) {
                res_r = left_r;
                res_i = left_i;
            }
            else if (x_val == xp[xp_size - 1]) {
                res_r = right_r;
                res_i = right_i;
            }
            else if (x_idx >= xp_size - 1) {
                res_r = right_r;
                res_i = right_i;
            }
            else if (x_val == xp[x_idx]) {
                res_r = fp[x_idx].real();
                res_i = fp[x_idx].imag();
            }
            else {
                realT dx = xp[x_idx + 1] - xp[x_idx];

                realT slope_r = (fp[x_idx + 1].real() - fp[x_idx].real()) / dx;
                res_r = slope_r * (x_val - xp[x_idx]) + fp[x_idx].real();
                if (sycl::isnan(res_r)) {
                    res_r = slope_r * (x_val - xp[x_idx + 1]) +
                            fp[x_idx + 1].real();
                    if (sycl::isnan(res_r) &&
                        fp[x_idx].real() == fp[x_idx + 1].real()) {
                        res_r = fp[x_idx].real();
                    }
                }

                realT slope_i = (fp[x_idx + 1].imag() - fp[x_idx].imag()) / dx;
                res_i = slope_i * (x_val - xp[x_idx]) + fp[x_idx].imag();
                if (sycl::isnan(res_i)) {
                    res_i = slope_i * (x_val - xp[x_idx + 1]) +
                            fp[x_idx + 1].imag();
                    if (sycl::isnan(res_i) &&
                        fp[x_idx].imag() == fp[x_idx + 1].imag()) {
                        res_i = fp[x_idx].imag();
                    }
                }
            }
            out[i] = TValue(res_r, res_i);
        });
    });
}

} // namespace dpnp::kernels::interpolate
