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

#pragma once
#ifndef DPNP_MEMORY_ADAPTER_H // Cython compatibility
#define DPNP_MEMORY_ADAPTER_H

#include "dpnp_utils.hpp"
#include "queue_sycl.hpp"

/**
 * @ingroup BACKEND_UTILS
 * @brief Adapter for the memory given by parameters in the DPNPC functions
 *
 * This type should be used to accommodate memory in the function. For example,
 * if the kernel must be executed in "queue_1" which is host based, but
 * input arrays are located on other "queue_2" or unknown place.
 *
 * Also, some functions completely host based and has no SYCL environment.
 *
 */
template <typename _DataType>
class DPNPC_ptr_adapter final
{
    DPCTLSyclQueueRef queue_ref; /**< reference to SYCL queue */
    sycl::queue queue;           /**< SYCL queue */
    void *aux_ptr = nullptr; /**< pointer to allocated memory by this adapter */
    void *orig_ptr =
        nullptr; /**< original pointer to memory given by parameters */
    size_t size_in_bytes = 0; /**< size of bytes of the memory */
    bool allocated = false; /**< True if the memory allocated by this procedure
                               and needs to be free */
    bool target_no_queue = false; /**< Indicates that original memory will be
                                     accessed from non SYCL environment */
    bool copy_back = false; /**< If the memory is 'result' it needs to be copied
                               back to original */
    const bool verbose = false;
    std::vector<sycl::event> deps;

public:
    DPNPC_ptr_adapter() = delete;

    DPNPC_ptr_adapter(DPCTLSyclQueueRef q_ref,
                      const void *src_ptr,
                      const size_t size,
                      bool target_no_sycl    = false,
                      bool copy_back_request = false)
    {
        queue_ref       = q_ref;
        queue           = *(reinterpret_cast<sycl::queue *>(queue_ref));
        target_no_queue = target_no_sycl;
        copy_back       = copy_back_request;
        orig_ptr        = const_cast<void *>(src_ptr);
        size_in_bytes   = size * sizeof(_DataType);
        deps            = std::vector<sycl::event>{};

        // enum class alloc { host = 0, device = 1, shared = 2, unknown = 3 };
        sycl::usm::alloc src_ptr_type = sycl::usm::alloc::unknown;
        src_ptr_type = sycl::get_pointer_type(src_ptr, queue.get_context());
        if (verbose) {
            std::cerr << "DPNPC_ptr_converter:";
            std::cerr << "\n\t target_no_queue=" << target_no_queue;
            std::cerr << "\n\t copy_back=" << copy_back;
            std::cerr << "\n\t pointer=" << src_ptr;
            std::cerr << "\n\t size=" << size;
            std::cerr << "\n\t size_in_bytes=" << size_in_bytes;
            std::cerr << "\n\t pointer type=" << (long)src_ptr_type;
            std::cerr << "\n\t queue inorder=" << queue.is_in_order();
            std::cerr << "\n\t queue device is_cpu="
                      << queue.get_device().is_cpu();
            std::cerr << "\n\t queue device is_gpu="
                      << queue.get_device().is_gpu();
            std::cerr << "\n\t queue device is_accelerator="
                      << queue.get_device().is_accelerator();
            std::cerr << std::endl;
        }

        if (is_memcpy_required(src_ptr_type)) {
            aux_ptr = dpnp_memory_alloc_c(queue_ref, size_in_bytes);
            dpnp_memory_memcpy_c(queue_ref, aux_ptr, src_ptr, size_in_bytes);
            allocated = true;
            if (verbose) {
                std::cerr << "DPNPC_ptr_converter::alloc and copy memory"
                          << " from=" << src_ptr << " to=" << aux_ptr
                          << " size_in_bytes=" << size_in_bytes << std::endl;
            }
        }
        else {
            aux_ptr = const_cast<void *>(src_ptr);
        }
    }

    ~DPNPC_ptr_adapter()
    {
        if (allocated) {
            if (verbose) {
                std::cerr << "DPNPC_ptr_converter::free_memory at=" << aux_ptr
                          << std::endl;
            }

            sycl::event::wait(deps);

            if (copy_back) {
                copy_data_back();
            }

            dpnp_memory_free_c(queue_ref, aux_ptr);
        }
    }

    bool is_memcpy_required(sycl::usm::alloc src_ptr_type)
    {
        if (target_no_queue || queue.get_device().is_gpu()) {
            if (src_ptr_type == sycl::usm::alloc::unknown) {
                return true;
            }
            else if (target_no_queue &&
                     src_ptr_type == sycl::usm::alloc::device) {
                return true;
            }
        }

        return false;
    }

    _DataType *get_ptr() const
    {
        return reinterpret_cast<_DataType *>(aux_ptr);
    }

    void copy_data_back() const
    {
        if (verbose) {
            std::cerr << "DPNPC_ptr_converter::copy_data_back:"
                      << " from=" << aux_ptr << " to=" << orig_ptr
                      << " size_in_bytes=" << size_in_bytes << std::endl;
        }

        dpnp_memory_memcpy_c(queue_ref, orig_ptr, aux_ptr, size_in_bytes);
    }

    void depends_on(const std::vector<sycl::event> &new_deps)
    {
        assert(allocated);
        deps.insert(std::end(deps), std::begin(new_deps), std::end(new_deps));
    }

    void depends_on(const sycl::event &new_dep)
    {
        assert(allocated);
        deps.push_back(new_dep);
    }
};

#endif // DPNP_MEMORY_ADAPTER_H
