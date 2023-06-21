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

#include <gtest/gtest.h>
#include <numeric>
#include <vector>

#include "dpnp_iterator.hpp"
#include "dpnp_test_utils.hpp"

// TODO need to fix build procedure and remove this workaround. Issue #551
#define DPNP_LOCAL_QUEUE 1
#include "queue_sycl.hpp"

struct IteratorParameters
{
    vector<dpnpc_it_t::size_type> input_shape;
    vector<dpnpc_it_t::size_type> output_shape;
    vector<dpnpc_value_t> result;

    /// Operator needs to print this container in human readable form in error
    /// reporting
    friend std::ostream &operator<<(std::ostream &out,
                                    const IteratorParameters &data)
    {
        out << "IteratorParameters(input_shape=" << data.input_shape
            << ", output_shape=" << data.output_shape
            << ", result=" << data.result << ")";

        return out;
    }
};

class IteratorBroadcasting : public ::testing::TestWithParam<IteratorParameters>
{
};

TEST_P(IteratorBroadcasting, loop_broadcast)
{
    using data_type = double;

    const IteratorParameters &param = GetParam();
    std::vector<data_type> input_data =
        get_input_data<data_type>(param.input_shape);

    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);

    DPNPC_id<data_type> input(q_ref, input_data.data(), param.input_shape);
    input.broadcast_to_shape(param.output_shape);

    ASSERT_EQ(input.get_output_size(), param.result.size());

    for (dpnpc_index_t output_id = 0; output_id < input.get_output_size();
         ++output_id)
    {
        EXPECT_EQ(input[output_id], param.result.at(output_id));
    }
}

TEST_P(IteratorBroadcasting, sycl_broadcast)
{
    using data_type = double;

    const IteratorParameters &param = GetParam();
    const dpnpc_index_t result_size = param.result.size();
    data_type *result               = reinterpret_cast<data_type *>(
        dpnp_memory_alloc_c(result_size * sizeof(data_type)));

    std::vector<data_type> input_data =
        get_input_data<data_type>(param.input_shape);
    data_type *shared_data = get_shared_data<data_type>(input_data);

    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);

    DPNPC_id<data_type> *input_it;
    input_it = reinterpret_cast<DPNPC_id<data_type> *>(
        dpnp_memory_alloc_c(q_ref, sizeof(DPNPC_id<data_type>)));
    new (input_it) DPNPC_id<data_type>(q_ref, shared_data, param.input_shape);

    input_it->broadcast_to_shape(param.output_shape);

    ASSERT_EQ(input_it->get_output_size(), result_size);

    sycl::range<1> gws(result_size);
    auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
        const size_t idx = global_id[0];
        result[idx]      = (*input_it)[idx];
    };

    auto kernel_func = [&](sycl::handler &cgh) {
        cgh.parallel_for<class test_sycl_reduce_axis_kernel>(
            gws, kernel_parallel_for_func);
    };

    sycl::event event = DPNP_QUEUE.submit(kernel_func);
    event.wait();

    for (dpnpc_index_t i = 0; i < result_size; ++i) {
        EXPECT_EQ(result[i], param.result.at(i));
    }

    input_it->~DPNPC_id();
    dpnp_memory_free_c(shared_data);
    dpnp_memory_free_c(result);
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
 * print(f"result={np.array2string(result.reshape(result.size), separator=',
 * ')}\n", sep=", ")
 */
INSTANTIATE_TEST_SUITE_P(
    TestBroadcastIterator,
    IteratorBroadcasting,
    testing::Values(
        IteratorParameters{{1}, {1}, {1}},
        IteratorParameters{{1}, {4}, {1, 1, 1, 1}},
        IteratorParameters{{1}, {3, 4}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
        IteratorParameters{{1}, {2, 3, 4}, {1, 1, 1, 1, 1, 1, 1, 1,
                                            1, 1, 1, 1, 1, 1, 1, 1,
                                            1, 1, 1, 1, 1, 1, 1, 1}},
        IteratorParameters{{4}, {4}, {1, 2, 3, 4}},
        IteratorParameters{{4}, {3, 4}, {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}},
        IteratorParameters{{4}, {2, 3, 4}, {1, 2, 3, 4, 1, 2, 3, 4,
                                            1, 2, 3, 4, 1, 2, 3, 4,
                                            1, 2, 3, 4, 1, 2, 3, 4}},
        IteratorParameters{{3, 4},
                           {3, 4},
                           {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
        IteratorParameters{{3, 4}, {2, 3, 4}, {1, 2,  3,  4,  5, 6,  7,  8,
                                               9, 10, 11, 12, 1, 2,  3,  4,
                                               5, 6,  7,  8,  9, 10, 11, 12}},
        IteratorParameters{{2, 3, 4},
                           {2, 3, 4},
                           {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}},
        IteratorParameters{{2, 3, 1}, {2, 3, 4}, {1, 1, 1, 1, 2, 2, 2, 2,
                                                  3, 3, 3, 3, 4, 4, 4, 4,
                                                  5, 5, 5, 5, 6, 6, 6, 6}},
        IteratorParameters{{2, 1, 4}, {2, 3, 4}, {1, 2, 3, 4, 1, 2, 3, 4,
                                                  1, 2, 3, 4, 5, 6, 7, 8,
                                                  5, 6, 7, 8, 5, 6, 7, 8}}));
