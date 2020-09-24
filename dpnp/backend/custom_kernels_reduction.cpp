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

#include <cmath>
#include <iostream>

#include <backend_iface.hpp>

#include "backend_pstl.hpp"
#include "queue_sycl.hpp"

template <typename _KernelNameSpecialization>
class custom_sum_c_kernel;

template <typename _DataType>
void custom_sum_c(void* array1_in, void* result1, size_t size)
{
    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

#if 1 // naive algorithm
    // cl::sycl::range<1> gws(size);
    auto policy = oneapi::dpl::execution::make_device_policy<custom_sum_c_kernel<_DataType>>(DPNP_QUEUE);

    // sycl::buffer<_DataType, 1> array_1_buf(array_1, gws);
    // auto it_begin = oneapi::dpl::begin(array_1_buf);
    // auto it_end = oneapi::dpl::end(array_1_buf);

    _DataType accumulator = 0;
    accumulator = std::reduce(policy, array_1, array_1 + size, _DataType(0), std::plus<_DataType>());

    policy.queue().wait();

#if 0 // verification
    accumulator = 0;
    for (size_t i = 0; i < size; ++i)
    {
        accumulator += array_1[i];
    }
    // std::cout << "result: " << accumulator << std::endl;
#endif

    result[0] = accumulator;

    return;

#else // naive algorithm

    cl::sycl::event event;

    cl::sycl::buffer<_DataType, 1> work_buffer(array_1, cl::sycl::range<1>(size));
    work_buffer.set_final_data(nullptr);

    auto group_max_size = DPNP_QUEUE.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
    size_t local = std::min(size, group_max_size);
    size_t length = size;

    /**
     * Each iteration of the do loop applies one level of reduction until
     * the input is of length 1 (i.e. the reduction is complete).
     */
    do
    {
        std::cout << "====================================\n"
                  << "local=" << local << "\n"
                  << "length=" << length << "\n"
                  << std::flush;
        auto kernel_func = [length, local, &work_buffer](cl::sycl::handler& cgh) {
            /* Two accessors are used: one to the buffer that is being reduced,
             * and a second to local memory, used to store intermediate data. */
            auto work_buffer_ptr = work_buffer.template get_access<cl::sycl::access::mode::read_write>(cgh);
            // local, read/write memory to make barriers work
            cl::sycl::accessor<_DataType, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>
                local_mem_buf(cl::sycl::range<1>(local), cgh);

            auto kernel_parallel_for_func = [work_buffer_ptr, local_mem_buf, local, length](cl::sycl::nd_item<1> id) {
                size_t globalid = id.get_global_id(0);
                size_t localid = id.get_local_id(0);

                /* All threads collectively read from global memory into local.
                 * The barrier ensures all threads' IO is resolved before
                 * execution continues (strictly speaking, all threads within
                 * a single work-group - there is no co-ordination between
                 * work-groups, only work-items). */
                if (globalid < length)
                {
                    local_mem_buf[localid] = work_buffer_ptr[globalid];
                }
                id.barrier(cl::sycl::access::fence_space::local_space);

                /* Apply the reduction operation between the current local
                 * id and the one on the other half of the vector. */
                if (globalid < length)
                {
                    int min = (length < local) ? length : local;
                    for (size_t offset = min / 2; offset > 0; offset /= 2)
                    {
                        if (localid < offset)
                        {
                            local_mem_buf[localid] += local_mem_buf[localid + offset];
                        }
                        id.barrier(cl::sycl::access::fence_space::local_space);
                    }
                    /* The final result will be stored in local id 0. */
                    if (localid == 0)
                    {
                        work_buffer_ptr[id.get_group(0)] = local_mem_buf[localid];
                    }
                }
            };

            /**
             * The parallel_for invocation chosen is the variant with an nd_item
             * parameter, since the code requires barriers for correctness.
             */
            cl::sycl::nd_range<1> work_range{cl::sycl::range<1>{std::max(length, local)},
                                             cl::sycl::range<1>{std::min(length, local)}};

            cgh.parallel_for<custom_sum_c_kernel<_DataType>>(work_range, kernel_parallel_for_func);
        };

        DPNP_QUEUE.submit(kernel_func);
        /* At this point, you could queue::wait_and_throw() to ensure that
         * errors are caught quickly. However, this would likely impact
         * performance negatively. */
        length = length / local;
    } while (length > 1);

    /* It is always sensible to wrap host accessors in their own scope as
     * kernels using the buffers they access are blocked for the length
     * of the accessor's lifetime. */
    auto work_buffer_ro = work_buffer.template get_access<cl::sycl::access::mode::read>();
    result[0] = work_buffer_ro[0];

    DPNP_QUEUE.wait();

#endif // naive algorithm
}

template void custom_sum_c<double>(void* array1_in, void* result1, size_t size);
template void custom_sum_c<float>(void* array1_in, void* result1, size_t size);
template void custom_sum_c<long>(void* array1_in, void* result1, size_t size);
template void custom_sum_c<int>(void* array1_in, void* result1, size_t size);

template <typename _KernelNameSpecialization>
class custom_prod_c_kernel;

template <typename _DataType>
void custom_prod_c(void* array1_in, void* result1, size_t size)
{
    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    auto policy = oneapi::dpl::execution::make_device_policy<custom_prod_c_kernel<_DataType>>(DPNP_QUEUE);

    result[0] = std::reduce(policy, array_1, array_1 + size, _DataType(1), std::multiplies<_DataType>());

    policy.queue().wait();
}

template void custom_prod_c<double>(void* array1_in, void* result1, size_t size);
template void custom_prod_c<float>(void* array1_in, void* result1, size_t size);
template void custom_prod_c<long>(void* array1_in, void* result1, size_t size);
template void custom_prod_c<int>(void* array1_in, void* result1, size_t size);
