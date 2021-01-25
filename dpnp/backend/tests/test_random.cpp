#include <dpnp_iface.hpp>
#include <dpnp_iface_fptr.hpp>

#include <math.h>
#include <vector>
  
#include "gtest/gtest.h"

template <typename _DataType>
bool check_statistics(_DataType* r, double tM, double tD, double tQ, size_t size)
{
    double tD2;
    double sM, sD;
    double sum, sum2;
    double n, s;
    double DeltaM, DeltaD;

    /***** Sample moments *****/
    sum = 0.0;
    sum2 = 0.0;
    for (size_t i = 0; i < size; i++) {
        sum += (double)r[i];
        sum2 += (double)r[i] * (double)r[i];
    }
    sM = sum / ((double)size);
    sD = sum2 / (double)size - (sM * sM);

    /***** Comparison of theoretical and sample moments *****/
    n = (double)size;
    tD2 = tD * tD;
    s = ((tQ - tD2) / n) - (2 * (tQ - 2 * tD2) / (n * n)) + ((tQ - 3 * tD2) / (n * n * n));

    DeltaM = (tM - sM) / sqrt(tD / n);
    DeltaD = (tD - sD) / sqrt(s);
    if (fabs(DeltaM) > 3.0 || fabs(DeltaD) > 3.0)
        return false;
    else
        return true;
}

TEST (TestBackendRandomBeta, test_seed) {
    const size_t size = 256;
    size_t seed = 10;
    double a = 0.4;
    double b = 0.5;

    auto QueueOptionsDevices = std::vector<QueueOptions>{ QueueOptions::CPU_SELECTOR,
        QueueOptions::GPU_SELECTOR };

    for (auto device_selector :  QueueOptionsDevices) {
        dpnp_queue_initialize_c(device_selector);
        double* result1 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
        double* result2 = (double*)dpnp_memory_alloc_c(size * sizeof(double));

        dpnp_rng_srand_c(seed);
        dpnp_rng_beta_c<double>(result1, a, b, size);

        dpnp_rng_srand_c(seed);
        dpnp_rng_beta_c<double>(result2, a, b, size);

        for (size_t i = 0; i < size; ++i)
        {
            EXPECT_NEAR (result1[i], result2[i], 0.004);
        }
    }
}

TEST (TestBackendRandomF, test_seed) {
    const size_t size = 256;
    size_t seed = 10;
    double dfnum = 10.4;
    double dfden = 12.5;

    auto QueueOptionsDevices = std::vector<QueueOptions>{ QueueOptions::CPU_SELECTOR,
        QueueOptions::GPU_SELECTOR };

    for (auto device_selector :  QueueOptionsDevices) {
        dpnp_queue_initialize_c(device_selector);
        double* result1 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
        double* result2 = (double*)dpnp_memory_alloc_c(size * sizeof(double));

        dpnp_rng_srand_c(seed);
        dpnp_rng_f_c<double>(result1, dfnum, dfden, size);

        dpnp_rng_srand_c(seed);
        dpnp_rng_f_c<double>(result2, dfnum, dfden, size);

        for (size_t i = 0; i < size; ++i)
        {
            EXPECT_NEAR (result1[i], result2[i], 0.004);
        }
    }
}

TEST (TestBackendRandomNormal, test_seed) {
    const size_t size = 256;
    size_t seed = 10;
    double loc = 2.56;
    double scale = 0.8;

    auto QueueOptionsDevices = std::vector<QueueOptions>{ QueueOptions::CPU_SELECTOR,
        QueueOptions::GPU_SELECTOR };

    for (auto device_selector :  QueueOptionsDevices) {
        dpnp_queue_initialize_c(device_selector);
        double* result1 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
        double* result2 = (double*)dpnp_memory_alloc_c(size * sizeof(double));

        dpnp_rng_srand_c(seed);
        dpnp_rng_normal_c<double>(result1, loc, scale, size);

        dpnp_rng_srand_c(seed);
        dpnp_rng_normal_c<double>(result2, loc, scale, size);

        for (size_t i = 0; i < size; ++i)
        {
            EXPECT_NEAR (result1[i], result2[i], 0.004);
        }
    }
}

TEST (TestBackendRandomUniform, test_seed) {
    const size_t size = 256;
    size_t seed = 10;
    long low = 1;
    long high = 120;

    auto QueueOptionsDevices = std::vector<QueueOptions>{ QueueOptions::CPU_SELECTOR,
        QueueOptions::GPU_SELECTOR };

    for (auto device_selector :  QueueOptionsDevices) {
        dpnp_queue_initialize_c(device_selector);
        double* result1 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
        double* result2 = (double*)dpnp_memory_alloc_c(size * sizeof(double));

        dpnp_rng_srand_c(seed);
        dpnp_rng_uniform_c<double>(result1, low, high, size);

        dpnp_rng_srand_c(seed);
        dpnp_rng_uniform_c<double>(result2, low, high, size);

        for (size_t i = 0; i < size; ++i)
        {
            EXPECT_NEAR (result1[i], result2[i], 0.004);
        }
    }
}

TEST (TestBackendRandomUniform, test_statistics) {
    const size_t size = 256;
    size_t seed = 10;
    long a = 1;
    long b = 120;
    bool check_statistics_res = true;

    /***** Theoretical moments *****/
    double tM = (b + a) / 2.0;
    double tD = ((b - a) * (b - a)) / 12.0;
    double tQ = ((b - a) * (b - a) * (b - a) * (b - a)) / 80.0;

    auto QueueOptionsDevices = std::vector<QueueOptions>{ QueueOptions::CPU_SELECTOR,
        QueueOptions::GPU_SELECTOR };

    for (auto device_selector : QueueOptionsDevices) {
        dpnp_queue_initialize_c(device_selector);
        double* result = (double*)dpnp_memory_alloc_c(size * sizeof(double));
        dpnp_rng_srand_c(seed);
        dpnp_rng_uniform_c<double>(result, a, b, size);
        check_statistics_res = check_statistics<double>(result, tM, tD, tQ, size);

        ASSERT_TRUE (check_statistics_res);
    }
}

TEST (TestBackendRandomSrand, test_func_ptr) {

    void * fptr = nullptr;
    DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNPFuncName::DPNP_FN_RNG_SRAND,
        DPNPFuncType::DPNP_FT_DOUBLE, DPNPFuncType::DPNP_FT_DOUBLE);

    fptr = get_dpnp_function_ptr1(kernel_data.return_type, DPNPFuncName::DPNP_FN_RNG_SRAND,
        DPNPFuncType::DPNP_FT_DOUBLE, DPNPFuncType::DPNP_FT_DOUBLE);

    EXPECT_TRUE(fptr != nullptr);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
