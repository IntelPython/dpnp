#pragma once

#include <sycl/sycl.hpp>
#include <vector>
#include <cstddef>
#include <type_traits>

#include <sycl/sycl.hpp>

#include "utils/type_utils.hpp"

namespace dpnp::extensions::math::kernels
{

using interpolate_fn_ptr_t = sycl::event (*)(sycl::queue &,
                                             const void *,  // x
                                             const void *,  // idx
                                             const void *,  // xp
                                             const void *,  // fp
                                             void *,        // out
                                             std::size_t,   // n
                                             std::size_t,   // xp_size
                                             const std::vector<sycl::event> &);

template <typename T>
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
    const T *x = static_cast<const T *>(vx);
    const std::size_t *idx = static_cast<const std::size_t *>(vidx);
    const T *xp = static_cast<const T *>(vxp);
    const T *fp = static_cast<const T *>(vfp);
    T *out = static_cast<T *>(vout);

    return q.submit([&](sycl::handler &h) {
        h.depends_on(depends);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            T left = fp[0];
            T right = fp[xp_size - 1];
            std::size_t x_idx = idx[i] - 1;

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
        });
    });
}

} // namespace dpnp::extensions::math::kernels
