//*****************************************************************************
// Copyright (c) 2016, Intel Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// - Neither the name of the copyright holder nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
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

/*
 * This header file is for interface Cython with C++.
 * It should not contains any backend specific headers (like SYCL or math
 * library) because all included headers will be exposed in Cython compilation
 * procedure
 *
 * Also, this file should contains documentation on functions and types
 * which are used in the interface
 */

#include <cstring>
#include <stdexcept>

#include "dpnp_fptr.hpp"

static func_map_t func_map_init();

func_map_t func_map = func_map_init();

DPNPFuncData_t get_dpnp_function_ptr(DPNPFuncName func_name,
                                     DPNPFuncType first_type,
                                     DPNPFuncType second_type)
{
    DPNPFuncType local_second_type =
        (second_type == DPNPFuncType::DPNP_FT_NONE) ? first_type : second_type;

    func_map_t::const_iterator func_it = func_map.find(func_name);
    if (func_it == func_map.cend()) {
        throw std::runtime_error(
            "DPNP Error: Unsupported function call."); // TODO print Function ID
    }

    const map_1p_t &type1_map = func_it->second;
    map_1p_t::const_iterator type1_map_it = type1_map.find(first_type);
    if (type1_map_it == type1_map.cend()) {
        throw std::runtime_error("DPNP Error: Function ID with unsupported "
                                 "first parameter type."); // TODO print
                                                           // Function ID
    }

    const map_2p_t &type2_map = type1_map_it->second;
    map_2p_t::const_iterator type2_map_it = type2_map.find(local_second_type);
    if (type2_map_it == type2_map.cend()) {
        throw std::runtime_error("DPNP Error: Function ID with unsupported "
                                 "second parameter type."); // TODO print
                                                            // Function ID
    }

    DPNPFuncData_t func_info = type2_map_it->second;

    return func_info;
}

/**
 * This operator is needed for compatibility with Cython 0.29 which has a bug in
 * Enum handling
 * TODO needs to be deleted in future
 */
size_t operator-(DPNPFuncType lhs, DPNPFuncType rhs)
{
    size_t lhs_base = static_cast<size_t>(lhs);
    size_t rhs_base = static_cast<size_t>(rhs);

    size_t result = lhs_base - rhs_base;

    return result;
}

static func_map_t func_map_init()
{
    func_map_t fmap;

    func_map_init_arraycreation(fmap);
    func_map_init_linalg(fmap);
    func_map_init_mathematical(fmap);
    func_map_init_random(fmap);
    func_map_init_sorting(fmap);

    return fmap;
};
