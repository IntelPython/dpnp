#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstdint>
#include <algorithm>

template <typename TCoord, typename TValue>
void interp_cpu(const std::vector<TCoord>& x,
                const std::vector<std::int64_t>& idx,
                const std::vector<TCoord>& xp,
                const std::vector<TValue>& fp,
                const TValue* left,
                const TValue* right,
                std::vector<TValue>& out)
{
    std::size_t n = x.size();
    std::size_t xp_size = xp.size();

    for (std::size_t i = 0; i < n; ++i)
    {
        TValue left_val = left ? *left : fp[0];
        TValue right_val = right ? *right : fp[xp_size - 1];
        TCoord x_val = x[i];
        std::int64_t insert_idx = idx[i];

        if (std::isnan(x_val)) {
            out[i] = x_val;
        }
        else if (insert_idx == 0) {
            out[i] = left_val;
        }
        else {
            std::int64_t x_idx = insert_idx - 1;

            if (x_val == xp[xp_size - 1]) {
                out[i] = right_val;
            }
            else if (x_idx >= static_cast<std::int64_t>(xp_size - 1)) {
                out[i] = right_val;
            }
            else if (x_val == xp[x_idx]) {
                out[i] = fp[x_idx];
            }
            else {
                TValue slope = (fp[x_idx + 1] - fp[x_idx]) / (xp[x_idx + 1] - xp[x_idx]);
                TValue res = slope * (x_val - xp[x_idx]) + fp[x_idx];

                if (std::isnan(res)) {
                    res = slope * (x_val - xp[x_idx + 1]) + fp[x_idx + 1];
                    if (std::isnan(res) && (fp[x_idx] == fp[x_idx + 1])) {
                        res = fp[x_idx];
                    }
                }
                out[i] = res;
            }
        }

        std::cout << "i=" << i << ", x=" << x[i]
                  << ", idx=" << idx[i]
                  << ", result=" << out[i] << std::endl;
    }
}

std::vector<std::int64_t> searchsorted(const std::vector<double>& xp, const std::vector<double>& x)
{
    std::vector<std::int64_t> result;
    for (const auto& val : x) {
        auto it = std::upper_bound(xp.begin(), xp.end(), val);
        result.push_back(static_cast<std::int64_t>(std::distance(xp.begin(), it)));
    }
    return result;
}

int main()
{
    std::vector<double> x   = {0, 1, 2, 4, 6, 8, 9, 10};
    std::vector<double> fx  = {1, 3, 5, 7, 9};
    std::vector<double> fy;
    for (double val : fx)
        fy.push_back(std::sin(val));

    std::vector<std::int64_t> idx = searchsorted(fx, x);
    std::vector<double> out(x.size());

    interp_cpu<double, double>(x, idx, fx, fy, nullptr, nullptr, out);

    std::cout << "\nFinal output:\n";
    for (double val : out)
        std::cout << std::setprecision(6) << val << ", ";
    std::cout << std::endl;

    return 0;
}

// 0.841471, 0.841471, 0.491295, -0.408902, -0.150969, 0.534553, 0.412118, 0.412118,
