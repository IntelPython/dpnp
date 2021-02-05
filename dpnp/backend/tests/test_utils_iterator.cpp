//*****************************************************************************
// Copyright (c) 2016-2020, Intel Corporation
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

#define DPNP_LOCAL_QUEUE 1 // TODO need to fix build procedure and remove this workaround. Issue #551
#include "queue_sycl.hpp"

using namespace std;
using dpnpc_it_t = DPNPC_id<size_t>::iterator;
using dpnpc_value_t = dpnpc_it_t::value_type;
using dpnpc_index_t = dpnpc_it_t::size_type;

template <typename _DataType>
vector<_DataType> get_input_data(const vector<dpnpc_index_t>& shape)
{
    const dpnpc_index_t size = accumulate(shape.begin(), shape.end(), dpnpc_index_t(1), multiplies<dpnpc_index_t>());

    vector<_DataType> input_data(size, 42);
    iota(input_data.begin(), input_data.end(), 0);

    return input_data;
}

TEST(TestUtilsIterator, begin_prefix_postfix)
{
    using test_it = dpnpc_it_t;

    vector<dpnpc_value_t> input_data = get_input_data<dpnpc_value_t>({2});
    DPNPC_id<dpnpc_value_t> result_obj(input_data.data(), {2});

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

    // expected data 0, 1
    vector<dpnpc_value_t> input_data = get_input_data<dpnpc_value_t>({2});
    DPNPC_id<dpnpc_value_t> result_obj(input_data.data(), {2});

    test_it begin = result_obj.begin();
    EXPECT_EQ(*begin, 0);

    ++begin;
    EXPECT_EQ(*begin, 1);

    EXPECT_EQ(result_obj[1], 1);
}

TEST(TestUtilsIterator, take_value_loop)
{
    // expected data 0, 1, 2 ,3
    vector<dpnpc_value_t> input_data = get_input_data<dpnpc_value_t>({4});
    DPNPC_id<dpnpc_value_t> result_obj(input_data.data(), {4});

    dpnpc_it_t begin = result_obj.begin();
    for (size_t i = 0; i < input_data.size(); ++i, ++begin)
    {
        EXPECT_EQ(result_obj[i], i);
    }
}

TEST(TestUtilsIterator, take_value_axis_0_0)
{
    vector<dpnpc_value_t> input_data = get_input_data<dpnpc_value_t>({4});
    DPNPC_id<dpnpc_value_t> result_obj(input_data.data(), {2, 2});
    result_obj.set_axis(0); // expected data {{0, 2}, {1 ,3}} with shape {2, 2}

    dpnpc_it_t begin = result_obj.begin();
    dpnpc_it_t end = result_obj.end();
    EXPECT_NE(begin, end);
    EXPECT_EQ(*begin, 0);

    ++begin;
    EXPECT_EQ(*begin, 2);
}

TEST(TestUtilsIterator, take_value_axis_0_1)
{
    vector<dpnpc_value_t> input_data = get_input_data<dpnpc_value_t>({4});
    DPNPC_id<dpnpc_value_t> result_obj(input_data.data(), {2, 2});
    result_obj.set_axis(0); // expected data {{0, 2}, {1 ,3}} with shape {2, 2}

    dpnpc_it_t begin = result_obj.begin(1);
    dpnpc_it_t end = result_obj.end(1);
    EXPECT_NE(begin, end);
    EXPECT_EQ(*begin, 1);

    ++begin;
    EXPECT_EQ(*begin, 3);
}

TEST(TestUtilsIterator, take_value_axis_1)
{
    vector<dpnpc_value_t> input_data = get_input_data<dpnpc_value_t>({4});
    DPNPC_id<dpnpc_value_t> result_obj(input_data.data(), {2, 2});
    result_obj.set_axis(1); // expected data {{0, 1}, {2 ,3}}

    dpnpc_it_t begin = result_obj.begin();
    dpnpc_it_t end = result_obj.end();
    EXPECT_NE(begin, end);
    EXPECT_EQ(*begin, 0);
    EXPECT_EQ(*end, 2); // linear data space

    ++begin;
    EXPECT_EQ(*begin, 1);
}

TEST(TestUtilsIterator, iterator_loop)
{
    const dpnpc_it_t::size_type size = 10;

    vector<dpnpc_value_t> expected = get_input_data<dpnpc_value_t>({size});

    dpnpc_value_t input_data[size];
    DPNPC_id<dpnpc_value_t> result(input_data, {size});
    iota(result.begin(), result.end(), 0);

    vector<dpnpc_value_t>::iterator it_expected = expected.begin();
    dpnpc_it_t it_result = result.begin();

    for (; it_expected != expected.end(); ++it_expected, ++it_result)
    {
        EXPECT_EQ(*it_expected, *it_result);
    }
}

TEST(TestUtilsIterator, operator_minus)
{
    vector<dpnpc_value_t> input_data = get_input_data<dpnpc_value_t>({3, 4});
    DPNPC_id<dpnpc_value_t> obj(input_data.data(), {3, 4});

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
    vector<dpnpc_value_t> input_data = get_input_data<dpnpc_value_t>({3, 4});
    DPNPC_id<dpnpc_value_t> obj(input_data.data(), {3, 4});

    dpnpc_it_t::difference_type default_diff_distance = std::distance(obj.begin(), obj.end());
    EXPECT_EQ(default_diff_distance, 12);

    obj.set_axis(0);
    dpnpc_it_t::difference_type axis_0_diff_distance = std::distance(obj.begin(), obj.end());
    EXPECT_EQ(axis_0_diff_distance, 3);

    dpnpc_it_t::difference_type axis_0_1_diff_distance = std::distance(obj.begin(1), obj.end(1));
    EXPECT_EQ(axis_0_1_diff_distance, 3);

    obj.set_axis(1);
    dpnpc_it_t::difference_type axis_1_diff_distance = std::distance(obj.begin(), obj.end());
    EXPECT_EQ(axis_1_diff_distance, 4);

    dpnpc_it_t::difference_type axis_1_1_diff_distance = std::distance(obj.begin(1), obj.end(1));
    EXPECT_EQ(axis_1_1_diff_distance, 4);
}

struct IteratorParameters
{
    vector<dpnpc_it_t::size_type> input_shape;
    dpnpc_it_t::size_type axis;
    vector<dpnpc_value_t> result;

    /// Operator needs to print this container in human readable form in error reporting
    friend std::ostream& operator<<(std::ostream& out, const IteratorParameters& data)
    {
        out << "IteratorParameters(input_shape:" << data.input_shape << ", axis=" << data.axis
            << ", result=" << data.result << ")";

        return out;
    }
};

class IteratorReduction : public ::testing::TestWithParam<IteratorParameters>
{
};

TEST_P(IteratorReduction, loop_reduce_axis)
{
    const IteratorParameters& param = GetParam();
    const dpnpc_index_t result_size = param.result.size();

    vector<dpnpc_value_t> input_data = get_input_data<dpnpc_value_t>(param.input_shape);
    DPNPC_id<dpnpc_value_t> input(input_data.data(), param.input_shape);
    input.set_axis(param.axis);

    vector<dpnpc_value_t> test_result(result_size, 42);
    for (dpnpc_index_t output_id = 0; output_id < result_size; ++output_id)
    {
        test_result[output_id] = 0;
        for (dpnpc_it_t data_it = input.begin(output_id); data_it != input.end(output_id); ++data_it)
        {
            test_result[output_id] += *data_it;
        }

        EXPECT_EQ(test_result[output_id], param.result.at(output_id));
    }
}

TEST_P(IteratorReduction, pstl_reduce_axis)
{
    using data_type = size_t;

    const IteratorParameters& param = GetParam();
    const dpnpc_index_t result_size = param.result.size();

    vector<data_type> input_data = get_input_data<data_type>(param.input_shape);
    DPNPC_id<data_type> input(input_data.data(), param.input_shape);
    input.set_axis(param.axis);

    vector<dpnpc_value_t> result(result_size, 42);
    for (dpnpc_index_t output_id = 0; output_id < result_size; ++output_id)
    {
        auto policy = oneapi::dpl::execution::make_device_policy<class test_pstl_reduce_axis_kernel>(DPNP_QUEUE);
        result[output_id] =
            std::reduce(policy, input.begin(output_id), input.end(output_id), data_type(0), std::plus<data_type>());
        policy.queue().wait();

        EXPECT_EQ(result[output_id], param.result.at(output_id));
    }
}

INSTANTIATE_TEST_SUITE_P(
    TestUtilsIterator,
    IteratorReduction,
    testing::Values(IteratorParameters{{2, 3, 4}, 0, {12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34}},
                    IteratorParameters{{2, 3, 4}, 1, {12, 15, 18, 21, 48, 51, 54, 57}},
                    IteratorParameters{{2, 3, 4}, 2, {6, 22, 38, 54, 70, 86}},
                    IteratorParameters{{1, 1, 1}, 0, {0}},
                    IteratorParameters{{1, 1, 1}, 1, {0}},
                    IteratorParameters{{1, 1, 1}, 2, {0}},
                    IteratorParameters{{2, 3, 4, 2}, 0, {24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46,
                                                         48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70}},
                    IteratorParameters{
                        {2, 3, 4, 2}, 1, {24, 27, 30, 33, 36, 39, 42, 45, 96, 99, 102, 105, 108, 111, 114, 117}},
                    IteratorParameters{{2, 3, 4, 2}, 2, {12, 16, 44, 48, 76, 80, 108, 112, 140, 144, 172, 176}},
                    IteratorParameters{{2, 3, 4, 2}, 3, {1,  5,  9,  13, 17, 21, 25, 29, 33, 37, 41, 45,
                                                         49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93}},
                    IteratorParameters{{3, 4}, 0, {12, 15, 18, 21}},
                    IteratorParameters{{3, 4}, 1, {6, 22, 38}},
                    IteratorParameters{{1}, 0, {0}}) /*TODO ,  testing::PrintToStringParamName() */);
