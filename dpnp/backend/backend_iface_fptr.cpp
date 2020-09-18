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

/*
 * This header file is for interface Cython with C++.
 * It should not contains any backend specific headers (like SYCL or MKL) because
 * all included headers will be exposed in Cython compilation procedure
 *
 * Also, this file should contains documentation on functions and types
 * which are used in the interface
 */

#include <cstring>
#include <map>
#include <stdexcept>

#include <backend/backend_iface_fptr.hpp>

typedef std::map<DPNPFuncType, DPNPFuncData_t> map_2p_t;
typedef std::map<DPNPFuncType, map_2p_t> map_1p_t;
typedef std::map<DPNPFuncName, map_1p_t> func_map_t;

static func_map_t func_map_init();

func_map_t func_map = func_map_init();

DPNPFuncData_t get_dpnp_function_ptr(DPNPFuncName func_name, DPNPFuncType first_type, DPNPFuncType second_type)
{
    DPNPFuncType local_second_type = (second_type == DPNPFuncType::DPNP_FT_NONE) ? first_type : second_type;

    func_map_t::const_iterator func_it = func_map.find(func_name);
    if (func_it == func_map.cend())
    {
        throw std::runtime_error("Intel NumPy Error: Unsupported function call."); // TODO print Function ID
    }

    const map_1p_t& type1_map = func_it->second;
    map_1p_t::const_iterator type1_map_it = type1_map.find(first_type);
    if (type1_map_it == type1_map.cend())
    {
        throw std::runtime_error(
            "Intel NumPy Error: Function ID with unsupported first parameter type."); // TODO print Function ID
    }

    const map_2p_t& type2_map = type1_map_it->second;
    map_2p_t::const_iterator type2_map_it = type2_map.find(local_second_type);
    if (type2_map_it == type2_map.cend())
    {
        throw std::runtime_error(
            "Intel NumPy Error: Function ID with unsupported second parameter type."); // TODO print Function ID
    }

    DPNPFuncData_t func_info = type2_map_it->second;

    return func_info;
}

void* get_backend_function_name(const char* func_name, const char* type_name)
{
    /** Implement it in this way to allow easier play with it */
    const char* supported_func_name = "dpnp_dot";
    const char* supported_type1_name = "double";
    const char* supported_type2_name = "float";
    const char* supported_type3_name = "long";
    const char* supported_type4_name = "int";

    /** of coerce it will be converted into std::map later */
    if (!strncmp(func_name, supported_func_name, strlen(supported_func_name)))
    {
        if (!strncmp(type_name, supported_type1_name, strlen(supported_type1_name)))
        {
            return reinterpret_cast<void*>(mkl_blas_dot_c<double>);
        }
        else if (!strncmp(type_name, supported_type2_name, strlen(supported_type2_name)))
        {
            return reinterpret_cast<void*>(mkl_blas_dot_c<float>);
        }
        else if (!strncmp(type_name, supported_type3_name, strlen(supported_type3_name)))
        {
            return reinterpret_cast<void*>(custom_blas_dot_c<long>);
        }
        else if (!strncmp(type_name, supported_type4_name, strlen(supported_type4_name)))
        {
            return reinterpret_cast<void*>(custom_blas_dot_c<int>);
        }
    }

    throw std::runtime_error("Intel NumPy Error: Unsupported function call");
}

/**
 * This operator is needed for compatibility with Cython 0.29 which has a bug in Enum handling
 * TODO needs to be deleted in future
 */
size_t operator-(DPNPFuncType lhs, DPNPFuncType rhs)
{
    size_t lhs_base = static_cast<size_t>(lhs);
    size_t rhs_base = static_cast<size_t>(rhs);

    size_t result = lhs_base - rhs_base;

    return result;
}

void* get_dpnp_function_ptr1(DPNPFuncType& result_type,
                             DPNPFuncName name,
                             DPNPFuncType first_type,
                             DPNPFuncType second_type)
{
    DPNPFuncData_t result = get_dpnp_function_ptr(name, first_type, second_type);

    result_type = result.return_type;
    return result.ptr;
}

static func_map_t func_map_init()
{
    const DPNPFuncType eft_INT = DPNPFuncType::DPNP_FT_INT;
    const DPNPFuncType eft_LNG = DPNPFuncType::DPNP_FT_LONG;
    const DPNPFuncType eft_FLT = DPNPFuncType::DPNP_FT_FLOAT;
    const DPNPFuncType eft_DBL = DPNPFuncType::DPNP_FT_DOUBLE;
    func_map_t fmap;

    fmap[DPNPFuncName::DPNP_FN_ADD][eft_INT][eft_INT] = {eft_INT, (void*)custom_elemwise_add_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_INT][eft_LNG] = {eft_LNG, (void*)custom_elemwise_add_c<int, long, long>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_INT][eft_FLT] = {eft_DBL, (void*)custom_elemwise_add_c<int, float, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_INT][eft_DBL] = {eft_DBL, (void*)custom_elemwise_add_c<int, double, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_LNG][eft_INT] = {eft_LNG, (void*)custom_elemwise_add_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_LNG][eft_LNG] = {eft_LNG, (void*)custom_elemwise_add_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_LNG][eft_FLT] = {eft_DBL, (void*)custom_elemwise_add_c<long, float, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_LNG][eft_DBL] = {eft_DBL, (void*)custom_elemwise_add_c<long, double, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_FLT][eft_INT] = {eft_DBL, (void*)custom_elemwise_add_c<float, int, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_FLT][eft_LNG] = {eft_DBL, (void*)custom_elemwise_add_c<float, long, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_add_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_FLT][eft_DBL] = {eft_DBL, (void*)custom_elemwise_add_c<float, double, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_DBL][eft_INT] = {eft_DBL, (void*)custom_elemwise_add_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_DBL][eft_LNG] = {eft_DBL, (void*)custom_elemwise_add_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_DBL][eft_FLT] = {eft_DBL, (void*)custom_elemwise_add_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_add_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_INT][eft_INT] = {eft_INT, (void*)custom_argmax_c<int, int>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_INT][eft_LNG] = {eft_LNG, (void*)custom_argmax_c<int, long>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_LNG][eft_INT] = {eft_INT, (void*)custom_argmax_c<long, int>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_LNG][eft_LNG] = {eft_LNG, (void*)custom_argmax_c<long, long>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_FLT][eft_INT] = {eft_INT, (void*)custom_argmax_c<float, int>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_FLT][eft_LNG] = {eft_LNG, (void*)custom_argmax_c<float, long>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_DBL][eft_INT] = {eft_INT, (void*)custom_argmax_c<double, int>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_DBL][eft_LNG] = {eft_LNG, (void*)custom_argmax_c<double, long>};

    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_INT][eft_INT] = {eft_INT, (void*)custom_argmin_c<int, int>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_INT][eft_LNG] = {eft_LNG, (void*)custom_argmin_c<int, long>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_LNG][eft_INT] = {eft_INT, (void*)custom_argmin_c<long, int>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_LNG][eft_LNG] = {eft_LNG, (void*)custom_argmin_c<long, long>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_FLT][eft_INT] = {eft_INT, (void*)custom_argmin_c<float, int>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_FLT][eft_LNG] = {eft_LNG, (void*)custom_argmin_c<float, long>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_DBL][eft_INT] = {eft_INT, (void*)custom_argmin_c<double, int>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_DBL][eft_LNG] = {eft_LNG, (void*)custom_argmin_c<double, long>};

    fmap[DPNPFuncName::DPNP_FN_DOT][eft_INT][eft_INT] = {eft_INT, (void*)custom_blas_dot_c<int>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_LNG][eft_LNG] = {eft_LNG, (void*)custom_blas_dot_c<long>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_FLT][eft_FLT] = {eft_FLT, (void*)mkl_blas_dot_c<float>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_DBL][eft_DBL] = {eft_DBL, (void*)mkl_blas_dot_c<double>};

    return fmap;
};
