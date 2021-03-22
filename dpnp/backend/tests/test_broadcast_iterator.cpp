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

    vector<_DataType> input_data(size);
    iota(input_data.begin(), input_data.end(), 1); // let's start from 1 to avoid cleaned memory comparison

    return input_data;
}

TEST(TestBroadcastIterator, take_value_broadcast_loop_2D)
{
    vector<dpnpc_value_t> expected_data{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    // expected input data 1, 2, 3
    vector<dpnpc_value_t> input_data = get_input_data<dpnpc_value_t>({3, 1});
    DPNPC_id<dpnpc_value_t> result_obj(input_data.data(), {3, 1});
    result_obj.broadcast_to_shape({3, 4});

    for (dpnpc_index_t output_id = 0; output_id < result_obj.get_output_size(); ++output_id)
    {
        EXPECT_EQ(result_obj[output_id], expected_data[output_id]);
    }
}

TEST(TestBroadcastIterator, take_value_broadcast_loop_3D)
{
    vector<dpnpc_value_t> expected_data{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
    // expected input data 1, 2, 3, 4, 5, 6
    vector<dpnpc_value_t> input_data = get_input_data<dpnpc_value_t>({2, 3});
    DPNPC_id<dpnpc_value_t> result_obj(input_data.data(), {2, 3});
    result_obj.broadcast_to_shape({2, 2, 3});

    for (dpnpc_index_t output_id = 0; output_id < result_obj.get_output_size(); ++output_id)
    {
        EXPECT_EQ(result_obj[output_id], expected_data[output_id]);
    }
}

TEST(TestBroadcastIterator, output_size_broadcast)
{
    vector<dpnpc_value_t> input_data = get_input_data<dpnpc_value_t>({3, 4});
    DPNPC_id<dpnpc_value_t> result_obj(input_data.data(), {3, 4});
    result_obj.broadcast_to_shape({2, 3, 4});

    const dpnpc_index_t output_size = result_obj.get_output_size();
    EXPECT_EQ(output_size, 24);
}

struct IteratorParameters
{
    vector<dpnpc_it_t::size_type> input_shape;
    vector<dpnpc_it_t::size_type> output_shape;
    vector<dpnpc_value_t> result;

    /// Operator needs to print this container in human readable form in error reporting
    friend std::ostream& operator<<(std::ostream& out, const IteratorParameters& data)
    {
        out << "IteratorParameters(input_shape=" << data.input_shape << ", output_shape=" << data.output_shape
            << ", result=" << data.result << ")";

        return out;
    }
};

class IteratorBroadcasting : public ::testing::TestWithParam<IteratorParameters>
{
};

TEST_P(IteratorBroadcasting, loop_broadcast)
{
    const IteratorParameters& param = GetParam();
    const dpnpc_index_t result_size = param.result.size();

    vector<dpnpc_value_t> input_data = get_input_data<dpnpc_value_t>(param.input_shape);
    DPNPC_id<dpnpc_value_t> input(input_data.data(), param.input_shape);
    input.broadcast_to_shape(param.output_shape);

    ASSERT_EQ(input.get_output_size(), result_size);

    vector<dpnpc_value_t> test_result(result_size, 42);
    for (dpnpc_index_t output_id = 0; output_id < result_size; ++output_id)
    {
        test_result[output_id] = input[output_id];
        std::cout << "test_result[" << output_id << "] = " << test_result[output_id] << std::endl;
        std::cout << "param.result.at(" << output_id << ") = " << param.result.at(output_id) << std::endl;
        EXPECT_EQ(test_result[output_id], param.result.at(output_id));
    }
}

TEST_P(IteratorBroadcasting, sycl_broadcast)
{
    using data_type = double;

    const IteratorParameters& param = GetParam();
    const dpnpc_index_t result_size = param.result.size();
    vector<data_type> result(result_size, 42);
    data_type* result_ptr = result.data();

    vector<data_type> input_data = get_input_data<data_type>(param.input_shape);
    DPNPC_id<data_type> input(input_data.data(), param.input_shape);
    input.broadcast_to_shape(param.output_shape);

    ASSERT_EQ(input.get_output_size(), result_size);

    cl::sycl::range<1> gws(result_size);
    const DPNPC_id<data_type>* input_it = &input;
    auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {
        const size_t idx = global_id[0];
        result_ptr[idx] = (*input_it)[idx];
    };

    auto kernel_func = [&](cl::sycl::handler& cgh) {
        cgh.parallel_for<class test_sycl_reduce_axis_kernel>(gws, kernel_parallel_for_func);
    };

    cl::sycl::event event = DPNP_QUEUE.submit(kernel_func);
    event.wait();

    for (dpnpc_index_t i = 0; i < result_size; ++i)
    {
        EXPECT_EQ(result.at(i), param.result.at(i));
    }
}

/**
 * Expected values produced by following script:
 *
 * import numpy as np
 *
 * input_size = 12
 * input_shape = [3, 4]
 * input = np.arange(1, input_size + 1, dtype=np.int64).reshape(input_shape)
 * print(f"input shape={input.shape}")
 * print(f"input:\n{input}\n")
 * 
 * output_shape = [2, 3, 4]
 * output = np.ones(output_shape, dtype=np.int64)
 * print(f"output shape={output.shape}")
 *
 * result = input * output
 * print(f"result={np.array2string(result.reshape(result.size), separator=', ')}\n", sep=", ")
 */
INSTANTIATE_TEST_SUITE_P(
    TestBroadcastIterator,
    IteratorBroadcasting,
    testing::Values(
        IteratorParameters{{1}, {1}, {1}},
        IteratorParameters{{1}, {4}, {1, 1, 1, 1}},
        IteratorParameters{{1}, {3, 4}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
        IteratorParameters{{1}, {2, 3, 4}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
        IteratorParameters{{4}, {4}, {1, 2, 3, 4}},
        IteratorParameters{{4}, {3, 4}, {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}},
        IteratorParameters{{4}, {2, 3, 4}, {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}},
        IteratorParameters{{3, 4}, {3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
        IteratorParameters{{3, 4}, {2, 3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                               1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
        IteratorParameters{{2, 3, 4}, {2, 3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                                  13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}},
        IteratorParameters{{2, 3, 1}, {2, 3, 4}, {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                                  4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6}},
        IteratorParameters{{2, 1, 4}, {2, 3, 4}, {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                                                  5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8}}));
