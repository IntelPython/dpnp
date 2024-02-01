//*****************************************************************************
// Copyright (c) 2016-2024, Intel Corporation
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

#include <gtest/gtest.h>
#include <numeric>
#include <vector>

#include "dpnp_iterator.hpp"
#include "dpnp_test_utils.hpp"

// TODO need to fix build procedure and remove this workaround. Issue #551
#define DPNP_LOCAL_QUEUE 1
#include "queue_sycl.hpp"

using namespace std;

TEST(TestUtilsIterator, begin_prefix_postfix)
{
    using test_it = dpnpc_it_t;

    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);

    vector<dpnpc_value_t> input_data = get_input_data<dpnpc_value_t>({2});
    DPNPC_id<dpnpc_value_t> result_obj(q_ref, input_data.data(), {2});

    test_it begin = result_obj.begin();
    test_it end = result_obj.end();

    EXPECT_NE(begin, end);

    test_it begin0 = begin;
    EXPECT_EQ(begin0, begin);

    test_it begin1 = begin0++;
    EXPECT_NE(begin1, begin0);
    EXPECT_EQ(begin1, begin);

    begin1++;
    EXPECT_EQ(begin1, begin0);

    test_it begin_1 = ++begin0;
    EXPECT_EQ(begin_1, begin0);
    EXPECT_EQ(begin0, end);
}

TEST(TestUtilsIterator, take_value)
{
    using test_it = dpnpc_it_t;

    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);

    // expected data 1, 2
    vector<dpnpc_value_t> input_data = get_input_data<dpnpc_value_t>({2});
    DPNPC_id<dpnpc_value_t> result_obj(q_ref, input_data.data(), {2});

    test_it begin = result_obj.begin();
    EXPECT_EQ(*begin, 1);

    ++begin;
    EXPECT_EQ(*begin, 2);

    EXPECT_EQ(result_obj[1], 2);
}

TEST(TestUtilsIterator, take_value_loop)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);

    // expected data 1, 2 ,3, 4
    vector<dpnpc_value_t> input_data = get_input_data<dpnpc_value_t>({4});
    DPNPC_id<dpnpc_value_t> result_obj(q_ref, input_data.data(), {4});

    dpnpc_it_t begin = result_obj.begin();
    for (size_t i = 0; i < input_data.size(); ++i, ++begin) {
        EXPECT_EQ(result_obj[i], i + 1);
    }
}

TEST(TestUtilsIterator, take_value_loop_3D)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);

    // expected input data 1, 2 ,3, 4...24
    vector<dpnpc_value_t> input_data = get_input_data<dpnpc_value_t>({2, 3, 4});
    DPNPC_id<dpnpc_value_t> result_obj(q_ref, input_data.data(), {2, 3, 4});

    dpnpc_it_t begin = result_obj.begin();
    for (size_t i = 0; i < input_data.size(); ++i, ++begin) {
        EXPECT_EQ(result_obj[i], i + 1);
    }
}

TEST(TestUtilsIterator, take_value_axes_loop_3D)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);

    vector<dpnpc_value_t> expected_data{1, 2, 3, 4, 13, 14, 15, 16};
    // expected input data 1, 2 ,3, 4...24
    vector<dpnpc_value_t> input_data = get_input_data<dpnpc_value_t>({2, 3, 4});
    DPNPC_id<dpnpc_value_t> result_obj(q_ref, input_data.data(), {2, 3, 4});
    result_obj.set_axes({0, 2});

    vector<dpnpc_value_t>::iterator expected_it = expected_data.begin();
    DPNPC_id<dpnpc_value_t>::iterator end = result_obj.end();
    for (DPNPC_id<dpnpc_value_t>::iterator it = result_obj.begin(); it != end;
         ++it, ++expected_it)
    {
        EXPECT_EQ(*it, *expected_it);
    }
}

TEST(TestUtilsIterator, take_value_axis_0_0)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);

    vector<dpnpc_value_t> input_data = get_input_data<dpnpc_value_t>({4});
    DPNPC_id<dpnpc_value_t> result_obj(q_ref, input_data.data(), {2, 2});
    result_obj.set_axis(0); // expected data {{1, 3}, {2 ,4}} with shape {2, 2}

    dpnpc_it_t begin = result_obj.begin();
    dpnpc_it_t end = result_obj.end();
    EXPECT_NE(begin, end);
    EXPECT_EQ(*begin, 1);

    ++begin;
    EXPECT_EQ(*begin, 3);
}

TEST(TestUtilsIterator, take_value_axis_0_1)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);

    vector<dpnpc_value_t> input_data = get_input_data<dpnpc_value_t>({4});
    DPNPC_id<dpnpc_value_t> result_obj(q_ref, input_data.data(), {2, 2});
    result_obj.set_axis(0); // expected data {{1, 3}, {2 ,4}} with shape {2, 2}

    dpnpc_it_t begin = result_obj.begin(1);
    dpnpc_it_t end = result_obj.end(1);
    EXPECT_NE(begin, end);
    EXPECT_EQ(*begin, 2);

    ++begin;
    EXPECT_EQ(*begin, 4);
}

TEST(TestUtilsIterator, take_value_axis_1)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);

    vector<dpnpc_value_t> input_data = get_input_data<dpnpc_value_t>({4});
    DPNPC_id<dpnpc_value_t> result_obj(q_ref, input_data.data(), {2, 2});
    result_obj.set_axis(1); // expected data {{1, 2}, {3 ,4}}

    dpnpc_it_t begin = result_obj.begin();
    dpnpc_it_t end = result_obj.end();
    EXPECT_NE(begin, end);
    EXPECT_EQ(*begin, 1);
    EXPECT_EQ(*end, 3); // linear data space

    ++begin;
    EXPECT_EQ(*begin, 2);
}

TEST(TestUtilsIterator, full_reduction_with_input_shape)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);

    vector<dpnpc_value_t> input_data = get_input_data<dpnpc_value_t>({2, 3});
    DPNPC_id<dpnpc_value_t> result_obj(q_ref, input_data.data(), {2, 3});

    dpnpc_value_t result = 0;
    for (dpnpc_it_t data_it = result_obj.begin(0); data_it != result_obj.end(0);
         ++data_it)
    {
        result += *data_it;
    }

    EXPECT_EQ(result, 21);
}

TEST(TestUtilsIterator, output_size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);

    // expected data 1, 2
    vector<dpnpc_value_t> input_data = get_input_data<dpnpc_value_t>({2, 3});
    DPNPC_id<dpnpc_value_t> result_obj(q_ref, input_data.data(), {2, 3});

    const dpnpc_index_t output_size = result_obj.get_output_size();

    EXPECT_EQ(output_size, 1);
}

TEST(TestUtilsIterator, output_size_empty)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);

    vector<dpnpc_value_t> input_data = get_input_data<dpnpc_value_t>({});
    DPNPC_id<dpnpc_value_t> result_obj(q_ref, input_data.data(), {});

    const dpnpc_index_t output_size = result_obj.get_output_size();

    EXPECT_EQ(output_size, 1);
}

TEST(TestUtilsIterator, output_size_nullptr)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);

    DPNPC_id<dpnpc_value_t> result_obj(q_ref, nullptr, {});

    const dpnpc_index_t output_size = result_obj.get_output_size();

    EXPECT_EQ(output_size, 0);
}

TEST(TestUtilsIterator, output_size_axis)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);

    // expected data 1, 2
    vector<dpnpc_value_t> input_data = get_input_data<dpnpc_value_t>({2, 3, 4});
    DPNPC_id<dpnpc_value_t> result_obj(q_ref, input_data.data(), {2, 3, 4});

    result_obj.set_axis(1);
    const dpnpc_index_t output_size = result_obj.get_output_size();
    EXPECT_EQ(output_size, 8);

    result_obj.set_axis(-2);
    const dpnpc_index_t output_size_1 = result_obj.get_output_size();
    EXPECT_EQ(output_size_1, 8);
}

TEST(TestUtilsIterator, output_size_axis_2D)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);

    // expected data 1, 2
    vector<dpnpc_value_t> input_data = get_input_data<dpnpc_value_t>({2, 3, 4});
    DPNPC_id<dpnpc_value_t> result_obj(q_ref, input_data.data(), {2, 3, 4});

    result_obj.set_axes({0, 2});
    const dpnpc_index_t output_size = result_obj.get_output_size();
    EXPECT_EQ(output_size, 3);

    result_obj.set_axes({-3, 2});
    const dpnpc_index_t output_size_1 = result_obj.get_output_size();
    EXPECT_EQ(output_size_1, 3);
}

TEST(TestUtilsIterator, iterator_loop)
{
    const dpnpc_it_t::size_type size = 10;

    vector<dpnpc_value_t> expected = get_input_data<dpnpc_value_t>({size});

    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);

    dpnpc_value_t input_data[size];
    DPNPC_id<dpnpc_value_t> result(q_ref, input_data, {size});
    iota(result.begin(), result.end(), 1);

    vector<dpnpc_value_t>::iterator it_expected = expected.begin();
    dpnpc_it_t it_result = result.begin();

    for (; it_expected != expected.end(); ++it_expected, ++it_result) {
        EXPECT_EQ(*it_expected, *it_result);
    }
}

TEST(TestUtilsIterator, operator_minus)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);

    vector<dpnpc_value_t> input_data = get_input_data<dpnpc_value_t>({3, 4});
    DPNPC_id<dpnpc_value_t> obj(q_ref, input_data.data(), {3, 4});

    EXPECT_EQ(obj.begin() - obj.end(), -12);
    EXPECT_EQ(obj.end() - obj.begin(), 12);

    obj.set_axis(0);
    EXPECT_EQ(obj.begin() - obj.end(), -3);
    EXPECT_EQ(obj.end() - obj.begin(), 3);

    EXPECT_EQ(obj.begin(1) - obj.end(1), -3);
    EXPECT_EQ(obj.end(1) - obj.begin(1), 3);

    obj.set_axis(1);
    EXPECT_EQ(obj.begin() - obj.end(), -4);
    EXPECT_EQ(obj.end() - obj.begin(), 4);

    EXPECT_EQ(obj.begin(1) - obj.end(1), -4);
    EXPECT_EQ(obj.end(1) - obj.begin(1), 4);
}

TEST(TestUtilsIterator, iterator_distance)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);

    vector<dpnpc_value_t> input_data = get_input_data<dpnpc_value_t>({3, 4});
    DPNPC_id<dpnpc_value_t> obj(q_ref, input_data.data(), {3, 4});

    dpnpc_it_t::difference_type default_diff_distance =
        std::distance(obj.begin(), obj.end());
    EXPECT_EQ(default_diff_distance, 12);

    obj.set_axis(0);
    dpnpc_it_t::difference_type axis_0_diff_distance =
        std::distance(obj.begin(), obj.end());
    EXPECT_EQ(axis_0_diff_distance, 3);

    dpnpc_it_t::difference_type axis_0_1_diff_distance =
        std::distance(obj.begin(1), obj.end(1));
    EXPECT_EQ(axis_0_1_diff_distance, 3);

    obj.set_axis(1);
    dpnpc_it_t::difference_type axis_1_diff_distance =
        std::distance(obj.begin(), obj.end());
    EXPECT_EQ(axis_1_diff_distance, 4);

    dpnpc_it_t::difference_type axis_1_1_diff_distance =
        std::distance(obj.begin(1), obj.end(1));
    EXPECT_EQ(axis_1_1_diff_distance, 4);
}

struct IteratorParameters
{
    vector<dpnpc_it_t::size_type> input_shape;
    vector<long> axes;
    vector<dpnpc_value_t> result;

    /// Operator needs to print this container in human readable form in error
    /// reporting
    friend std::ostream &operator<<(std::ostream &out,
                                    const IteratorParameters &data)
    {
        out << "IteratorParameters(input_shape=" << data.input_shape
            << ", axis=" << data.axes << ", result=" << data.result << ")";

        return out;
    }
};

class IteratorReduction : public ::testing::TestWithParam<IteratorParameters>
{
};

TEST_P(IteratorReduction, loop_reduce_axis)
{
    const IteratorParameters &param = GetParam();
    const dpnpc_index_t result_size = param.result.size();

    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);

    vector<dpnpc_value_t> input_data =
        get_input_data<dpnpc_value_t>(param.input_shape);
    DPNPC_id<dpnpc_value_t> input(q_ref, input_data.data(), param.input_shape);
    input.set_axes(param.axes);

    ASSERT_EQ(input.get_output_size(), result_size);

    vector<dpnpc_value_t> test_result(result_size, 42);
    for (dpnpc_index_t output_id = 0; output_id < result_size; ++output_id) {
        test_result[output_id] = 0;
        for (dpnpc_it_t data_it = input.begin(output_id);
             data_it != input.end(output_id); ++data_it)
        {
            test_result[output_id] += *data_it;
        }

        EXPECT_EQ(test_result[output_id], param.result.at(output_id));
    }
}

TEST_P(IteratorReduction, pstl_reduce_axis)
{
    using data_type = double;

    const IteratorParameters &param = GetParam();
    const dpnpc_index_t result_size = param.result.size();

    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);

    vector<data_type> input_data = get_input_data<data_type>(param.input_shape);
    DPNPC_id<data_type> input(q_ref, input_data.data(), param.input_shape);
    input.set_axes(param.axes);

    ASSERT_EQ(input.get_output_size(), result_size);

    vector<data_type> result(result_size, 42);
    for (dpnpc_index_t output_id = 0; output_id < result_size; ++output_id) {
        auto policy = oneapi::dpl::execution::make_device_policy<
            class test_pstl_reduce_axis_kernel>(DPNP_QUEUE);
        result[output_id] =
            std::reduce(policy, input.begin(output_id), input.end(output_id),
                        data_type(0), std::plus<data_type>());
        policy.queue().wait();

        EXPECT_EQ(result[output_id], param.result.at(output_id));
    }
}

TEST_P(IteratorReduction, sycl_reduce_axis)
{
    using data_type = double;

    const IteratorParameters &param = GetParam();
    const dpnpc_index_t result_size = param.result.size();
    vector<data_type> result(result_size, 42);
    data_type *result_ptr = result.data();

    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);

    vector<data_type> input_data = get_input_data<data_type>(param.input_shape);
    DPNPC_id<data_type> input(q_ref, input_data.data(), param.input_shape);
    input.set_axes(param.axes);

    ASSERT_EQ(input.get_output_size(), result_size);

    sycl::range<1> gws(result_size);
    const DPNPC_id<data_type> *input_it = &input;
    auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
        const size_t idx = global_id[0];

        data_type accumulator = 0;
        for (DPNPC_id<data_type>::iterator data_it = input_it->begin(idx);
             data_it != input_it->end(idx); ++data_it)
        {
            accumulator += *data_it;
        }
        result_ptr[idx] = accumulator;
    };

    auto kernel_func = [&](sycl::handler &cgh) {
        cgh.parallel_for<class test_sycl_reduce_axis_kernel>(
            gws, kernel_parallel_for_func);
    };

    sycl::event event = DPNP_QUEUE.submit(kernel_func);
    event.wait();

    for (dpnpc_index_t i = 0; i < result_size; ++i) {
        EXPECT_EQ(result.at(i), param.result.at(i));
    }
}

/**
 * Expected values produced by following script:
 *
 * import numpy as np
 *
 * shape = [2, 3, 4]
 * size = 24
 * axis=1
 * input = np.arange(1, size + 1).reshape(shape)
 * print(f"axis={axis}")
 * print(f"input.dtype={input.dtype}")
 * print(f"input shape={input.shape}")
 * print(f"input:\n{input}\n")
 *
 * result = np.sum(input, axis=axis)
 * print(f"result.dtype={result.dtype}")
 * print(f"result shape={result.shape}")
 *
 * print(f"result={np.array2string(result.reshape(result.size),
 * separator=',')}\n", sep=",")
 */
INSTANTIATE_TEST_SUITE_P(
    TestUtilsIterator,
    IteratorReduction,
    testing::Values(
        IteratorParameters{{2, 3, 4},
                           {0},
                           {14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36}},
        IteratorParameters{{2, 3, 4}, {1}, {15, 18, 21, 24, 51, 54, 57, 60}},
        IteratorParameters{{2, 3, 4}, {2}, {10, 26, 42, 58, 74, 90}},
        IteratorParameters{{1, 1, 1}, {0}, {1}},
        IteratorParameters{{1, 1, 1}, {1}, {1}},
        IteratorParameters{{1, 1, 1}, {2}, {1}},
        IteratorParameters{{2, 3, 4, 2}, {0}, {26, 28, 30, 32, 34, 36, 38, 40,
                                               42, 44, 46, 48, 50, 52, 54, 56,
                                               58, 60, 62, 64, 66, 68, 70, 72}},
        IteratorParameters{{2, 3, 4, 2},
                           {1},
                           {27, 30, 33, 36, 39, 42, 45, 48, 99, 102, 105, 108,
                            111, 114, 117, 120}},
        IteratorParameters{
            {2, 3, 4, 2},
            {2},
            {16, 20, 48, 52, 80, 84, 112, 116, 144, 148, 176, 180}},
        IteratorParameters{{2, 3, 4, 2}, {3}, {3,  7,  11, 15, 19, 23, 27, 31,
                                               35, 39, 43, 47, 51, 55, 59, 63,
                                               67, 71, 75, 79, 83, 87, 91, 95}},
        IteratorParameters{{2, 3, 4, 2},
                           {0, 1},
                           {126, 132, 138, 144, 150, 156, 162, 168}},
        IteratorParameters{{2, 3, 4, 2}, {2, 3}, {36, 100, 164, 228, 292, 356}},
        IteratorParameters{
            {2, 3, 4, 2},
            {0, 3},
            {54, 62, 70, 78, 86, 94, 102, 110, 118, 126, 134, 142}},
        IteratorParameters{{2, 3, 4, 2}, {0, 1, 2}, {576, 600}},
        IteratorParameters{{2, 3, 4, 2}, {0, 2, 3}, {264, 392, 520}},
        IteratorParameters{{2, 3, 4, 2}, {0, -2, -1}, {264, 392, 520}},
        IteratorParameters{{3, 4}, {0}, {15, 18, 21, 24}},
        IteratorParameters{{3, 4}, {1}, {10, 26, 42}},
        IteratorParameters{{2, 3, 4, 5, 6}, {0, 1, 2, 3, 4}, {259560}},
        IteratorParameters{{2, 3, 4, 5, 6},
                           {0, 1, 3, 4},
                           {56790, 62190, 67590, 72990}},
        IteratorParameters{{2, 3, 4, 5, 6},
                           {1, 2, 3},
                           {10680, 10740, 10800, 10860, 10920, 10980, 32280,
                            32340, 32400, 32460, 32520, 32580}},
        IteratorParameters{{2, 3, 4, 5, 6},
                           {3, 1, 2},
                           {10680, 10740, 10800, 10860, 10920, 10980, 32280,
                            32340, 32400, 32460, 32520, 32580}},
        IteratorParameters{{2, 3, 4, 5, 6},
                           {0, 3, 1, 2},
                           {42960, 43080, 43200, 43320, 43440, 43560}},
        IteratorParameters{{2, 3, 4, 5},
                           {1, 3},
                           {345, 420, 495, 570, 1245, 1320, 1395, 1470}},
        IteratorParameters{{2, 3, 0, 5}, {1, 3}, {}},
        IteratorParameters{{2, 0, 4, 5}, {1, 3}, {0, 0, 0, 0, 0, 0, 0, 0}},
        // IteratorParameters{{2, 3, -4, 5}, {1, 3}, {}},
        // IteratorParameters{{2, -3, 4, 5}, {1, 3}, {0,0,0,0,0,0,0,0}},
        IteratorParameters{{}, {}, {1}},
        IteratorParameters{{0}, {}, {}},
        IteratorParameters{{}, {0}, {1}},
        IteratorParameters{{0}, {0}, {0}},
        IteratorParameters{
            {1},
            {0},
            {1}}) /*TODO ,  testing::PrintToStringParamName() */);
