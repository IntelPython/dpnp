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

template <typename _DataType_output,
          typename _DataType_input1,
          typename _DataType_input2>
void (*dpnp_dot_default_c)(void *,
                           const size_t,
                           const size_t,
                           const shape_elem_type *,
                           const shape_elem_type *,
                           const void *,
                           const size_t,
                           const size_t,
                           const shape_elem_type *,
                           const shape_elem_type *,
                           const void *,
                           const size_t,
                           const size_t,
                           const shape_elem_type *,
                           const shape_elem_type *) =
    dpnp_dot_c<_DataType_output, _DataType_input1, _DataType_input2>;

void *get_backend_function_name(const char *func_name, const char *type_name)
{
    /** Implement it in this way to allow easier play with it */
    const char *supported_func_name = "dpnp_dot";
    const char *supported_type1_name = "double";
    const char *supported_type2_name = "float";
    const char *supported_type3_name = "long";
    const char *supported_type4_name = "int";

    /** of coerce it will be converted into std::map later */
    if (!strncmp(func_name, supported_func_name, strlen(supported_func_name))) {
        if (!strncmp(type_name, supported_type1_name,
                     strlen(supported_type1_name))) {
            return reinterpret_cast<void *>(
                dpnp_dot_default_c<double, double, double>);
        }
        else if (!strncmp(type_name, supported_type2_name,
                          strlen(supported_type2_name)))
        {
            return reinterpret_cast<void *>(
                dpnp_dot_default_c<float, float, float>);
        }
        else if (!strncmp(type_name, supported_type3_name,
                          strlen(supported_type3_name)))
        {
            return reinterpret_cast<void *>(
                dpnp_dot_default_c<int64_t, int64_t, int64_t>);
        }
        else if (!strncmp(type_name, supported_type4_name,
                          strlen(supported_type4_name)))
        {
            return reinterpret_cast<void *>(
                dpnp_dot_default_c<int32_t, int32_t, int32_t>);
        }
    }

    throw std::runtime_error("DPNP Error: Unsupported function call");
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

void *get_dpnp_function_ptr1(DPNPFuncType &result_type,
                             DPNPFuncName name,
                             DPNPFuncType first_type,
                             DPNPFuncType second_type)
{
    DPNPFuncData_t result =
        get_dpnp_function_ptr(name, first_type, second_type);

    result_type = result.return_type;
    return result.ptr;
}

static func_map_t func_map_init()
{
    func_map_t fmap;

    func_map_init_arraycreation(fmap);
    func_map_init_bitwise(fmap);
    func_map_init_elemwise(fmap);
    func_map_init_fft_func(fmap);
    func_map_init_indexing_func(fmap);
    func_map_init_linalg(fmap);
    func_map_init_linalg_func(fmap);
    func_map_init_logic(fmap);
    func_map_init_manipulation(fmap);
    func_map_init_mathematical(fmap);
    func_map_init_random(fmap);
    func_map_init_reduction(fmap);
    func_map_init_searching(fmap);
    func_map_init_sorting(fmap);
    func_map_init_statistics(fmap);

    return fmap;
};
