#pragma once

#include <sycl/sycl.hpp>
#include <vector>

#include "utils/type_utils.hpp"

namespace type_utils = dpctl::tensor::type_utils;

namespace dpnp::kernels::interpolate
{

template <typename T>
struct IsNan
{
    static bool isnan(const T &v)
    {
        if constexpr (type_utils::is_complex_v<T>) {
            using vT = typename T::value_type;

            const vT real1 = std::real(v);
            const vT imag1 = std::imag(v);

            return IsNan<vT>::isnan(real1) || IsNan<vT>::isnan(imag1);
        }
        else if constexpr (std::is_floating_point_v<T> ||
                           std::is_same_v<T, sycl::half>) {
            return sycl::isnan(v);
        }

        return false;
    }
};

template <typename TCoord, typename TValue>
sycl::event interpolate_impl(sycl::queue &q,
                             const TCoord *x,
                             const std::int64_t *idx,
                             const TCoord *xp,
                             const TValue *fp,
                             const TValue *left,
                             const TValue *right,
                             TValue *out,
                             const std::size_t n,
                             const std::size_t xp_size,
                             const std::vector<sycl::event> &depends)
{
    return q.submit([&](sycl::handler &h) {
        h.depends_on(depends);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            TValue left_val = left ? *left : fp[0];
            TValue right_val = right ? *right : fp[xp_size - 1];

            TCoord x_val = x[i];
            std::int64_t x_idx = idx[i] - 1;

            if (IsNan<TCoord>::isnan(x_val)) {
                out[i] = x_val;
            }
            else if (x_idx < 0) {
                out[i] = left_val;
            }
            else if (x_val == xp[xp_size - 1]) {
                out[i] = right_val;
            }
            else if (x_idx >= static_cast<std::int64_t>(xp_size - 1)) {
                out[i] = right_val;
            }
            else {
                TValue slope =
                    (fp[x_idx + 1] - fp[x_idx]) / (xp[x_idx + 1] - xp[x_idx]);
                TValue res = slope * (x_val - xp[x_idx]) + fp[x_idx];

                if (IsNan<TValue>::isnan(res)) {
                    res = slope * (x_val - xp[x_idx + 1]) + fp[x_idx + 1];
                    if (IsNan<TValue>::isnan(res) &&
                        (fp[x_idx] == fp[x_idx + 1])) {
                        res = fp[x_idx];
                    }
                }
                out[i] = res;
            }
        });
    });
}

} // namespace dpnp::kernels::interpolate
