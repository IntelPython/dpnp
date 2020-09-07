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

/**
 * Example of experimental interface.
 *
 * This example shows how to get a runtime pointer from DPNP C++ Backend library
 *
 * Possible compile line:
 * clang++ examples/example_experimental_iface.cpp -o example_experimental_iface -Idpnp -Idpnp/backend -Ldpnp -Wl,-rpath='$ORIGIN'/dpnp -ldpnp_backend_c
 *
 */

#include <iostream>

#include <backend/backend_iface_fptr.hpp>

int main(int, char**)
{
    void* result = get_backend_function_name("dpnp_dot", "float");
    std::cout << "Result Dot() function pointer (by old interface): " << result << std::endl;

    try {
        DPNPFuncType func_dot_type = DPNPFuncType::DPNP_FT_FLOAT;
        void* dpnp_dot_fptr = get_dpnp_function_ptr(DPNPFuncName::DPNP_FN_DOT, {DPNPFuncType::DPNP_FT_FLOAT});
        std::cout << "Result Dot() function pointer: " << dpnp_dot_fptr << std::endl;
    }
    catch (std::runtime_error &e)
    {
        std::cout << "Function Dot is not implemented in the library yet." << std::endl;
    }

    void* dpnp_add_fptr = get_dpnp_function_ptr(DPNPFuncName::DPNP_FN_ADD, {DPNPFuncType::DPNP_FT_FLOAT, DPNPFuncType::DPNP_FT_FLOAT});
    std::cout << "Result Add() function pointer: " << dpnp_add_fptr << std::endl;

    return 0;
}
