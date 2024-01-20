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

#include "verbose.hpp"
#include <iostream>

bool _is_verbose_mode = false;
bool _is_verbose_mode_init = false;

bool is_verbose_mode()
{
    if (!_is_verbose_mode_init) {
        _is_verbose_mode = false;
        char *env_var = nullptr;

#ifdef _WIN32
        size_t env_var_size = 0;
        _dupenv_s(&env_var, &env_var_size, "DPNP_VERBOSE");
#else
        env_var = std::getenv("DPNP_VERBOSE");
#endif

        if (env_var && std::string(env_var) == "1") {
            _is_verbose_mode = true;
        }
        _is_verbose_mode_init = true;

#ifdef _WIN32
        if (env_var != nullptr)
            free(env_var);
#endif
    }
    return _is_verbose_mode;
}

class barrierKernelClass;

void set_barrier_event(sycl::queue queue, std::vector<sycl::event> &depends)
{
    if (is_verbose_mode()) {
        sycl::event barrier_event =
            queue.single_task<barrierKernelClass>(depends, [=] {});
        depends.clear();
        depends.push_back(barrier_event);
    }
}

void verbose_print(std::string header,
                   sycl::event first_event,
                   sycl::event last_event)
{
    if (is_verbose_mode()) {
        auto first_event_end =
            first_event
                .get_profiling_info<sycl::info::event_profiling::command_end>();
        auto last_event_end =
            last_event
                .get_profiling_info<sycl::info::event_profiling::command_end>();
        std::cout << "DPNP_VERBOSE " << header
                  << " Time: " << (last_event_end - first_event_end) / 1.0e9
                  << " s" << std::endl;
    }
}
