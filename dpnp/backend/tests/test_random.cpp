//*****************************************************************************
// Copyright (c) 2016-2023, Intel Corporation
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

#include <dpnp_iface.hpp>
#include <dpnp_iface_fptr.hpp>

#include <iostream>

#include <math.h>
#include <vector>

#include "gtest/gtest.h"

// TODO
// * data management will be redesigned: allocation as chars (and casting on teste suits)
// * this class will be generlized
class RandomTestCase : public ::testing::Test
{
public:
    static void SetUpTestCase()
    {
        _get_device_mem();
    }

    static void TearDownTestCase()
    {
        dpnp_memory_free_c(result1);
        result1 = result2 = nullptr;
    }

    void SetUp() override
    {
        for (size_t i = 0; i < size; ++i)
        {
            result1[i] = 1;
            result2[i] = 0;
        }
    }

    void TearDown() override
    {
    }

private:
    static void _get_device_mem()
    {
        result1 = (double*)dpnp_memory_alloc_c(size * 2 * sizeof(double));
        result2 = result1 + size;
    }

protected:
    static size_t size;
    static double* result1;
    static double* result2;
};

double* RandomTestCase::result1 = nullptr;
double* RandomTestCase::result2 = nullptr;
size_t RandomTestCase::size = 10;

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
    for (size_t i = 0; i < size; i++)
    {
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

TEST_F(RandomTestCase, rng_beta_test_seed)
{
    const size_t seed = 10;
    const double a = 0.4;
    const double b = 0.5;

    dpnp_rng_srand_c(seed);
    dpnp_rng_beta_c<double>(result1, a, b, size);

    dpnp_rng_srand_c(seed);
    dpnp_rng_beta_c<double>(result2, a, b, size);

    for (size_t i = 0; i < size; ++i)
    {
        EXPECT_NEAR(result1[i], result2[i], 0.004);
    }
}

TEST_F(RandomTestCase, rng_f_test_seed)
{
    const size_t seed = 10;
    const double dfnum = 10.4;
    const double dfden = 12.5;

    dpnp_rng_srand_c(seed);
    dpnp_rng_f_c<double>(result1, dfnum, dfden, size);

    dpnp_rng_srand_c(seed);
    dpnp_rng_f_c<double>(result2, dfnum, dfden, size);

    for (size_t i = 0; i < size; ++i)
    {
        EXPECT_NEAR(result1[i], result2[i], 0.004);
    }
}

TEST_F(RandomTestCase, rng_normal_test_seed)
{
    const size_t seed = 10;
    const double loc = 2.56;
    const double scale = 0.8;

    dpnp_rng_srand_c(seed);
    dpnp_rng_normal_c<double>(result1, loc, scale, size);

    dpnp_rng_srand_c(seed);
    dpnp_rng_normal_c<double>(result2, loc, scale, size);

    for (size_t i = 0; i < size; ++i)
    {
        EXPECT_NEAR(result1[i], result2[i], 0.004);
    }
}

TEST_F(RandomTestCase, rng_uniform_test_seed)
{
    const size_t seed = 10;
    const long low = 1;
    const long high = 120;

    dpnp_rng_srand_c(seed);
    dpnp_rng_uniform_c<double>(result1, low, high, size);

    dpnp_rng_srand_c(seed);
    dpnp_rng_uniform_c<double>(result2, low, high, size);

    for (size_t i = 0; i < size; ++i)
    {
        EXPECT_NEAR(result1[i], result2[i], 0.004);
    }
}

TEST(TestBackendRandomUniform, test_statistics)
{
    const size_t size = 256;
    size_t seed = 10;
    long a = 1;
    long b = 120;
    bool check_statistics_res = false;

    /***** Theoretical moments *****/
    double tM = (b + a) / 2.0;
    double tD = ((b - a) * (b - a)) / 12.0;
    double tQ = ((b - a) * (b - a) * (b - a) * (b - a)) / 80.0;

    double* result = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_rng_srand_c(seed);
    dpnp_rng_uniform_c<double>(result, a, b, size);
    check_statistics_res = check_statistics<double>(result, tM, tD, tQ, size);
    ASSERT_TRUE(check_statistics_res);
    dpnp_memory_free_c(result);
}

// TODO:
// Generalization for all DPNPFuncName
TEST(TestBackendRandomSrand, test_func_ptr)
{
    void* fptr = nullptr;
    DPNPFuncData kernel_data = get_dpnp_function_ptr(
        DPNPFuncName::DPNP_FN_RNG_SRAND, DPNPFuncType::DPNP_FT_DOUBLE, DPNPFuncType::DPNP_FT_DOUBLE);

    fptr = get_dpnp_function_ptr1(kernel_data.return_type,
                                  DPNPFuncName::DPNP_FN_RNG_SRAND,
                                  DPNPFuncType::DPNP_FT_DOUBLE,
                                  DPNPFuncType::DPNP_FT_DOUBLE);

    EXPECT_TRUE(fptr != nullptr);
}
