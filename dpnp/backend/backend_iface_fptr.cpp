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
 * It should not contains any backend specific headers (like SYCL or math library) because
 * all included headers will be exposed in Cython compilation procedure
 *
 * Also, this file should contains documentation on functions and types
 * which are used in the interface
 */

#include <cstring>
#include <stdexcept>

#include "backend_fptr.hpp"


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
            return reinterpret_cast<void*>(custom_blas_dot_c<double>);
        }
        else if (!strncmp(type_name, supported_type2_name, strlen(supported_type2_name)))
        {
            return reinterpret_cast<void*>(custom_blas_dot_c<float>);
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
    func_map_t fmap;

    func_map_init_manipulation(fmap);

    fmap[DPNPFuncName::DPNP_FN_ABSOLUTE][eft_INT][eft_INT] = {eft_INT, (void*)custom_elemwise_absolute_c<int>};
    fmap[DPNPFuncName::DPNP_FN_ABSOLUTE][eft_LNG][eft_LNG] = {eft_LNG, (void*)custom_elemwise_absolute_c<long>};
    fmap[DPNPFuncName::DPNP_FN_ABSOLUTE][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_absolute_c<float>};
    fmap[DPNPFuncName::DPNP_FN_ABSOLUTE][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_absolute_c<double>};

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

    fmap[DPNPFuncName::DPNP_FN_ARCCOS][eft_INT][eft_INT] = {eft_DBL, (void*)custom_elemwise_acos_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOS][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_elemwise_acos_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOS][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_acos_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOS][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_acos_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCCOSH][eft_INT][eft_INT] = {eft_DBL, (void*)custom_elemwise_acosh_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOSH][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_elemwise_acosh_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOSH][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_acosh_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOSH][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_acosh_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCSIN][eft_INT][eft_INT] = {eft_DBL, (void*)custom_elemwise_asin_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCSIN][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_elemwise_asin_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCSIN][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_asin_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCSIN][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_asin_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCSINH][eft_INT][eft_INT] = {eft_DBL, (void*)custom_elemwise_asinh_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCSINH][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_elemwise_asinh_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCSINH][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_asinh_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCSINH][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_asinh_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCTAN][eft_INT][eft_INT] = {eft_DBL, (void*)custom_elemwise_atan_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_elemwise_atan_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_atan_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_atan_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_INT][eft_INT] = {eft_DBL,
                                                             (void*)custom_elemwise_arctan2_c<int, int, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_INT][eft_LNG] = {eft_DBL,
                                                             (void*)custom_elemwise_arctan2_c<int, long, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_INT][eft_FLT] = {eft_DBL,
                                                             (void*)custom_elemwise_arctan2_c<int, float, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_INT][eft_DBL] = {eft_DBL,
                                                             (void*)custom_elemwise_arctan2_c<int, double, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_LNG][eft_INT] = {eft_DBL,
                                                             (void*)custom_elemwise_arctan2_c<long, int, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_LNG][eft_LNG] = {eft_DBL,
                                                             (void*)custom_elemwise_arctan2_c<long, long, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_LNG][eft_FLT] = {eft_DBL,
                                                             (void*)custom_elemwise_arctan2_c<long, float, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_LNG][eft_DBL] = {eft_DBL,
                                                             (void*)custom_elemwise_arctan2_c<long, double, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_FLT][eft_INT] = {eft_DBL,
                                                             (void*)custom_elemwise_arctan2_c<float, int, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_FLT][eft_LNG] = {eft_DBL,
                                                             (void*)custom_elemwise_arctan2_c<float, long, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_FLT][eft_FLT] = {eft_FLT,
                                                             (void*)custom_elemwise_arctan2_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_FLT][eft_DBL] = {eft_DBL,
                                                             (void*)custom_elemwise_arctan2_c<float, double, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_DBL][eft_INT] = {eft_DBL,
                                                             (void*)custom_elemwise_arctan2_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_DBL][eft_LNG] = {eft_DBL,
                                                             (void*)custom_elemwise_arctan2_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_DBL][eft_FLT] = {eft_DBL,
                                                             (void*)custom_elemwise_arctan2_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_DBL][eft_DBL] = {eft_DBL,
                                                             (void*)custom_elemwise_arctan2_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCTANH][eft_INT][eft_INT] = {eft_DBL, (void*)custom_elemwise_atanh_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTANH][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_elemwise_atanh_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTANH][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_atanh_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTANH][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_atanh_c<double, double>};

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

    fmap[DPNPFuncName::DPNP_FN_ARGSORT][eft_INT][eft_INT] = {eft_LNG, (void*)custom_argsort_c<int, long>};
    fmap[DPNPFuncName::DPNP_FN_ARGSORT][eft_LNG][eft_LNG] = {eft_LNG, (void*)custom_argsort_c<long, long>};
    fmap[DPNPFuncName::DPNP_FN_ARGSORT][eft_FLT][eft_FLT] = {eft_LNG, (void*)custom_argsort_c<float, long>};
    fmap[DPNPFuncName::DPNP_FN_ARGSORT][eft_DBL][eft_DBL] = {eft_LNG, (void*)custom_argsort_c<double, long>};

    fmap[DPNPFuncName::DPNP_FN_CBRT][eft_INT][eft_INT] = {eft_DBL, (void*)custom_elemwise_cbrt_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_CBRT][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_elemwise_cbrt_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_CBRT][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_cbrt_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_CBRT][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_cbrt_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_CEIL][eft_INT][eft_INT] = {eft_DBL, (void*)custom_elemwise_ceil_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_CEIL][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_elemwise_ceil_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_CEIL][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_ceil_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_CEIL][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_ceil_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_COS][eft_INT][eft_INT] = {eft_DBL, (void*)custom_elemwise_cos_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_COS][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_elemwise_cos_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_COS][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_cos_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_COS][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_cos_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_COSH][eft_INT][eft_INT] = {eft_DBL, (void*)custom_elemwise_cosh_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_COSH][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_elemwise_cosh_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_COSH][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_cosh_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_COSH][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_cosh_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_COV][eft_INT][eft_INT] = {eft_DBL, (void*)custom_cov_c<double>};
    fmap[DPNPFuncName::DPNP_FN_COV][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_cov_c<double>};
    fmap[DPNPFuncName::DPNP_FN_COV][eft_FLT][eft_FLT] = {eft_DBL, (void*)custom_cov_c<double>};
    fmap[DPNPFuncName::DPNP_FN_COV][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_cov_c<double>};

    fmap[DPNPFuncName::DPNP_FN_DEGREES][eft_INT][eft_INT] = {eft_DBL, (void*)custom_elemwise_degrees_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_DEGREES][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_elemwise_degrees_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_DEGREES][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_degrees_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_DEGREES][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_degrees_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_INT][eft_INT] = {eft_INT, (void*)custom_elemwise_divide_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_INT][eft_LNG] = {eft_LNG, (void*)custom_elemwise_divide_c<int, long, long>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_INT][eft_FLT] = {eft_DBL,
                                                            (void*)custom_elemwise_divide_c<int, float, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_INT][eft_DBL] = {eft_DBL,
                                                            (void*)custom_elemwise_divide_c<int, double, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_LNG][eft_INT] = {eft_LNG, (void*)custom_elemwise_divide_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_LNG][eft_LNG] = {eft_LNG, (void*)custom_elemwise_divide_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_LNG][eft_FLT] = {eft_DBL,
                                                            (void*)custom_elemwise_divide_c<long, float, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_LNG][eft_DBL] = {eft_DBL,
                                                            (void*)custom_elemwise_divide_c<long, double, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_FLT][eft_INT] = {eft_DBL,
                                                            (void*)custom_elemwise_divide_c<float, int, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_FLT][eft_LNG] = {eft_DBL,
                                                            (void*)custom_elemwise_divide_c<float, long, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_FLT][eft_FLT] = {eft_FLT,
                                                            (void*)custom_elemwise_divide_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_FLT][eft_DBL] = {eft_DBL,
                                                            (void*)custom_elemwise_divide_c<float, double, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_DBL][eft_INT] = {eft_DBL,
                                                            (void*)custom_elemwise_divide_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_DBL][eft_LNG] = {eft_DBL,
                                                            (void*)custom_elemwise_divide_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_DBL][eft_FLT] = {eft_DBL,
                                                            (void*)custom_elemwise_divide_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_DBL][eft_DBL] = {eft_DBL,
                                                            (void*)custom_elemwise_divide_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_DOT][eft_INT][eft_INT] = {eft_INT, (void*)custom_blas_dot_c<int>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_LNG][eft_LNG] = {eft_LNG, (void*)custom_blas_dot_c<long>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_blas_dot_c<float>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_blas_dot_c<double>};

    fmap[DPNPFuncName::DPNP_FN_EIG][eft_INT][eft_INT] = {eft_DBL, (void*)custom_lapack_eig_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_EIG][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_lapack_eig_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_EIG][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_lapack_eig_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_EIG][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_lapack_eig_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_EXP][eft_INT][eft_INT] = {eft_DBL, (void*)custom_elemwise_exp_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_EXP][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_elemwise_exp_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_EXP][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_exp_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_EXP][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_exp_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_EXP2][eft_INT][eft_INT] = {eft_DBL, (void*)custom_elemwise_exp2_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_EXP2][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_elemwise_exp2_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_EXP2][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_exp2_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_EXP2][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_exp2_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_EXPM1][eft_INT][eft_INT] = {eft_DBL, (void*)custom_elemwise_expm1_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_EXPM1][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_elemwise_expm1_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_EXPM1][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_expm1_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_EXPM1][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_expm1_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_FABS][eft_INT][eft_INT] = {eft_DBL, (void*)custom_elemwise_fabs_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_FABS][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_elemwise_fabs_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_FABS][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_fabs_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_FABS][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_fabs_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_FLOOR][eft_INT][eft_INT] = {eft_DBL, (void*)custom_elemwise_floor_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_elemwise_floor_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_floor_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_floor_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_INT][eft_INT] = {eft_INT, (void*)custom_elemwise_fmod_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_INT][eft_LNG] = {eft_LNG, (void*)custom_elemwise_fmod_c<int, long, long>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_INT][eft_FLT] = {eft_DBL, (void*)custom_elemwise_fmod_c<int, float, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_INT][eft_DBL] = {eft_DBL, (void*)custom_elemwise_fmod_c<int, double, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_LNG][eft_INT] = {eft_LNG, (void*)custom_elemwise_fmod_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_LNG][eft_LNG] = {eft_LNG, (void*)custom_elemwise_fmod_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_LNG][eft_FLT] = {eft_DBL, (void*)custom_elemwise_fmod_c<long, float, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_LNG][eft_DBL] = {eft_DBL, (void*)custom_elemwise_fmod_c<long, double, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_FLT][eft_INT] = {eft_DBL, (void*)custom_elemwise_fmod_c<float, int, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_FLT][eft_LNG] = {eft_DBL, (void*)custom_elemwise_fmod_c<float, long, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_fmod_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_FLT][eft_DBL] = {eft_DBL,
                                                          (void*)custom_elemwise_fmod_c<float, double, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_DBL][eft_INT] = {eft_DBL, (void*)custom_elemwise_fmod_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_DBL][eft_LNG] = {eft_DBL, (void*)custom_elemwise_fmod_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_DBL][eft_FLT] = {eft_DBL,
                                                          (void*)custom_elemwise_fmod_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_DBL][eft_DBL] = {eft_DBL,
                                                          (void*)custom_elemwise_fmod_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_GAUSSIAN][eft_DBL][eft_DBL] = {eft_DBL, (void*)mkl_rng_gaussian<double>};

    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_INT][eft_INT] = {eft_DBL, (void*)custom_elemwise_hypot_c<int, int, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_INT][eft_LNG] = {eft_DBL, (void*)custom_elemwise_hypot_c<int, long, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_INT][eft_FLT] = {eft_DBL, (void*)custom_elemwise_hypot_c<int, float, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_INT][eft_DBL] = {eft_DBL,
                                                           (void*)custom_elemwise_hypot_c<int, double, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_LNG][eft_INT] = {eft_DBL, (void*)custom_elemwise_hypot_c<long, int, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_elemwise_hypot_c<long, long, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_LNG][eft_FLT] = {eft_DBL,
                                                           (void*)custom_elemwise_hypot_c<long, float, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_LNG][eft_DBL] = {eft_DBL,
                                                           (void*)custom_elemwise_hypot_c<long, double, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_FLT][eft_INT] = {eft_DBL, (void*)custom_elemwise_hypot_c<float, int, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_FLT][eft_LNG] = {eft_DBL,
                                                           (void*)custom_elemwise_hypot_c<float, long, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_FLT][eft_FLT] = {eft_FLT,
                                                           (void*)custom_elemwise_hypot_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_FLT][eft_DBL] = {eft_DBL,
                                                           (void*)custom_elemwise_hypot_c<float, double, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_DBL][eft_INT] = {eft_DBL,
                                                           (void*)custom_elemwise_hypot_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_DBL][eft_LNG] = {eft_DBL,
                                                           (void*)custom_elemwise_hypot_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_DBL][eft_FLT] = {eft_DBL,
                                                           (void*)custom_elemwise_hypot_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_DBL][eft_DBL] = {eft_DBL,
                                                           (void*)custom_elemwise_hypot_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_LOG][eft_INT][eft_INT] = {eft_DBL, (void*)custom_elemwise_log_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_elemwise_log_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_log_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_LOG][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_log_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_LOG10][eft_INT][eft_INT] = {eft_DBL, (void*)custom_elemwise_log10_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG10][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_elemwise_log10_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG10][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_log10_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_LOG10][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_log10_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_LOG1P][eft_INT][eft_INT] = {eft_DBL, (void*)custom_elemwise_log1p_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG1P][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_elemwise_log1p_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG1P][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_log1p_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_LOG1P][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_log1p_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_LOG2][eft_INT][eft_INT] = {eft_DBL, (void*)custom_elemwise_log2_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG2][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_elemwise_log2_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG2][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_log2_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_LOG2][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_log2_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_MATMUL][eft_INT][eft_INT] = {eft_INT, (void*)custom_blas_gemm_c<int>};
    fmap[DPNPFuncName::DPNP_FN_MATMUL][eft_LNG][eft_LNG] = {eft_LNG, (void*)custom_blas_gemm_c<long>};
    fmap[DPNPFuncName::DPNP_FN_MATMUL][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_blas_gemm_c<float>};
    fmap[DPNPFuncName::DPNP_FN_MATMUL][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_blas_gemm_c<double>};

    fmap[DPNPFuncName::DPNP_FN_MAX][eft_INT][eft_INT] = {eft_INT, (void*)custom_max_c<int>};
    fmap[DPNPFuncName::DPNP_FN_MAX][eft_LNG][eft_LNG] = {eft_LNG, (void*)custom_max_c<long>};
    fmap[DPNPFuncName::DPNP_FN_MAX][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_max_c<float>};
    fmap[DPNPFuncName::DPNP_FN_MAX][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_max_c<double>};

    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_INT][eft_INT] = {eft_INT, (void*)custom_elemwise_maximum_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_INT][eft_LNG] = {eft_LNG,
                                                             (void*)custom_elemwise_maximum_c<int, long, long>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_INT][eft_FLT] = {eft_DBL,
                                                             (void*)custom_elemwise_maximum_c<int, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_INT][eft_DBL] = {eft_DBL,
                                                             (void*)custom_elemwise_maximum_c<int, double, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_LNG][eft_INT] = {eft_LNG,
                                                             (void*)custom_elemwise_maximum_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_LNG][eft_LNG] = {eft_LNG,
                                                             (void*)custom_elemwise_maximum_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_LNG][eft_FLT] = {eft_DBL,
                                                             (void*)custom_elemwise_maximum_c<long, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_LNG][eft_DBL] = {eft_DBL,
                                                             (void*)custom_elemwise_maximum_c<long, double, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_FLT][eft_INT] = {eft_DBL,
                                                             (void*)custom_elemwise_maximum_c<float, int, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_FLT][eft_LNG] = {eft_DBL,
                                                             (void*)custom_elemwise_maximum_c<float, long, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_FLT][eft_FLT] = {eft_FLT,
                                                             (void*)custom_elemwise_maximum_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_FLT][eft_DBL] = {eft_DBL,
                                                             (void*)custom_elemwise_maximum_c<float, double, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_DBL][eft_INT] = {eft_DBL,
                                                             (void*)custom_elemwise_maximum_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_DBL][eft_LNG] = {eft_DBL,
                                                             (void*)custom_elemwise_maximum_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_DBL][eft_FLT] = {eft_DBL,
                                                             (void*)custom_elemwise_maximum_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_DBL][eft_DBL] = {eft_DBL,
                                                             (void*)custom_elemwise_maximum_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_MEAN][eft_INT][eft_INT] = {eft_DBL, (void*)custom_mean_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_MEAN][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_mean_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_MEAN][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_mean_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_MEAN][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_mean_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_MEDIAN][eft_INT][eft_INT] = {eft_DBL, (void*)custom_median_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_MEDIAN][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_median_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_MEDIAN][eft_FLT][eft_FLT] = {eft_DBL, (void*)custom_median_c<float, double>};
    fmap[DPNPFuncName::DPNP_FN_MEDIAN][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_median_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_MIN][eft_INT][eft_INT] = {eft_INT, (void*)custom_min_c<int>};
    fmap[DPNPFuncName::DPNP_FN_MIN][eft_LNG][eft_LNG] = {eft_LNG, (void*)custom_min_c<long>};
    fmap[DPNPFuncName::DPNP_FN_MIN][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_min_c<float>};
    fmap[DPNPFuncName::DPNP_FN_MIN][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_min_c<double>};

    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_INT][eft_INT] = {eft_INT, (void*)custom_elemwise_minimum_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_INT][eft_LNG] = {eft_LNG,
                                                             (void*)custom_elemwise_minimum_c<int, long, long>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_INT][eft_FLT] = {eft_DBL,
                                                             (void*)custom_elemwise_minimum_c<int, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_INT][eft_DBL] = {eft_DBL,
                                                             (void*)custom_elemwise_minimum_c<int, double, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_LNG][eft_INT] = {eft_LNG,
                                                             (void*)custom_elemwise_minimum_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_LNG][eft_LNG] = {eft_LNG,
                                                             (void*)custom_elemwise_minimum_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_LNG][eft_FLT] = {eft_DBL,
                                                             (void*)custom_elemwise_minimum_c<long, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_LNG][eft_DBL] = {eft_DBL,
                                                             (void*)custom_elemwise_minimum_c<long, double, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_FLT][eft_INT] = {eft_DBL,
                                                             (void*)custom_elemwise_minimum_c<float, int, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_FLT][eft_LNG] = {eft_DBL,
                                                             (void*)custom_elemwise_minimum_c<float, long, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_FLT][eft_FLT] = {eft_FLT,
                                                             (void*)custom_elemwise_minimum_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_FLT][eft_DBL] = {eft_DBL,
                                                             (void*)custom_elemwise_minimum_c<float, double, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_DBL][eft_INT] = {eft_DBL,
                                                             (void*)custom_elemwise_minimum_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_DBL][eft_LNG] = {eft_DBL,
                                                             (void*)custom_elemwise_minimum_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_DBL][eft_FLT] = {eft_DBL,
                                                             (void*)custom_elemwise_minimum_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_DBL][eft_DBL] = {eft_DBL,
                                                             (void*)custom_elemwise_minimum_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_INT][eft_INT] = {eft_INT,
                                                              (void*)custom_elemwise_multiply_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_INT][eft_LNG] = {eft_LNG,
                                                              (void*)custom_elemwise_multiply_c<int, long, long>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_INT][eft_FLT] = {eft_DBL,
                                                              (void*)custom_elemwise_multiply_c<int, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_INT][eft_DBL] = {eft_DBL,
                                                              (void*)custom_elemwise_multiply_c<int, double, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_LNG][eft_INT] = {eft_LNG,
                                                              (void*)custom_elemwise_multiply_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_LNG][eft_LNG] = {eft_LNG,
                                                              (void*)custom_elemwise_multiply_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_LNG][eft_FLT] = {eft_DBL,
                                                              (void*)custom_elemwise_multiply_c<long, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_LNG][eft_DBL] = {eft_DBL,
                                                              (void*)custom_elemwise_multiply_c<long, double, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_FLT][eft_INT] = {eft_DBL,
                                                              (void*)custom_elemwise_multiply_c<float, int, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_FLT][eft_LNG] = {eft_DBL,
                                                              (void*)custom_elemwise_multiply_c<float, long, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_FLT][eft_FLT] = {eft_FLT,
                                                              (void*)custom_elemwise_multiply_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_FLT][eft_DBL] = {eft_DBL,
                                                              (void*)custom_elemwise_multiply_c<float, double, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_DBL][eft_INT] = {eft_DBL,
                                                              (void*)custom_elemwise_multiply_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_DBL][eft_LNG] = {eft_DBL,
                                                              (void*)custom_elemwise_multiply_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_DBL][eft_FLT] = {eft_DBL,
                                                              (void*)custom_elemwise_multiply_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_DBL][eft_DBL] = {
        eft_DBL, (void*)custom_elemwise_multiply_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_POWER][eft_INT][eft_INT] = {eft_INT, (void*)custom_elemwise_power_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_INT][eft_LNG] = {eft_LNG, (void*)custom_elemwise_power_c<int, long, long>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_INT][eft_FLT] = {eft_DBL, (void*)custom_elemwise_power_c<int, float, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_INT][eft_DBL] = {eft_DBL,
                                                           (void*)custom_elemwise_power_c<int, double, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_LNG][eft_INT] = {eft_LNG, (void*)custom_elemwise_power_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_LNG][eft_LNG] = {eft_LNG, (void*)custom_elemwise_power_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_LNG][eft_FLT] = {eft_DBL,
                                                           (void*)custom_elemwise_power_c<long, float, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_LNG][eft_DBL] = {eft_DBL,
                                                           (void*)custom_elemwise_power_c<long, double, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_FLT][eft_INT] = {eft_DBL, (void*)custom_elemwise_power_c<float, int, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_FLT][eft_LNG] = {eft_DBL,
                                                           (void*)custom_elemwise_power_c<float, long, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_FLT][eft_FLT] = {eft_FLT,
                                                           (void*)custom_elemwise_power_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_FLT][eft_DBL] = {eft_DBL,
                                                           (void*)custom_elemwise_power_c<float, double, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_DBL][eft_INT] = {eft_DBL,
                                                           (void*)custom_elemwise_power_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_DBL][eft_LNG] = {eft_DBL,
                                                           (void*)custom_elemwise_power_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_DBL][eft_FLT] = {eft_DBL,
                                                           (void*)custom_elemwise_power_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_DBL][eft_DBL] = {eft_DBL,
                                                           (void*)custom_elemwise_power_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_PROD][eft_INT][eft_INT] = {eft_INT, (void*)custom_prod_c<int>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_LNG][eft_LNG] = {eft_LNG, (void*)custom_prod_c<long>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_prod_c<float>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_prod_c<double>};

    fmap[DPNPFuncName::DPNP_FN_UNIFORM][eft_INT][eft_INT] = {eft_INT, (void*)mkl_rng_uniform<int>};
    fmap[DPNPFuncName::DPNP_FN_UNIFORM][eft_FLT][eft_FLT] = {eft_FLT, (void*)mkl_rng_uniform<float>};
    fmap[DPNPFuncName::DPNP_FN_UNIFORM][eft_DBL][eft_DBL] = {eft_DBL, (void*)mkl_rng_uniform<double>};

    fmap[DPNPFuncName::DPNP_FN_RADIANS][eft_INT][eft_INT] = {eft_DBL, (void*)custom_elemwise_radians_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_RADIANS][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_elemwise_radians_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_RADIANS][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_radians_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_RADIANS][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_radians_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_RECIP][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_recip_c<float>};
    fmap[DPNPFuncName::DPNP_FN_RECIP][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_recip_c<double>};

    fmap[DPNPFuncName::DPNP_FN_SIGN][eft_INT][eft_INT] = {eft_INT, (void*)custom_elemwise_sign_c<int>};
    fmap[DPNPFuncName::DPNP_FN_SIGN][eft_LNG][eft_LNG] = {eft_LNG, (void*)custom_elemwise_sign_c<long>};
    fmap[DPNPFuncName::DPNP_FN_SIGN][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_sign_c<float>};
    fmap[DPNPFuncName::DPNP_FN_SIGN][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_sign_c<double>};

    fmap[DPNPFuncName::DPNP_FN_SIN][eft_INT][eft_INT] = {eft_DBL, (void*)custom_elemwise_sin_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_SIN][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_elemwise_sin_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_SIN][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_sin_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_SIN][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_sin_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_SINH][eft_INT][eft_INT] = {eft_DBL, (void*)custom_elemwise_sinh_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_SINH][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_elemwise_sinh_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_SINH][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_sinh_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_SINH][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_sinh_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_SORT][eft_INT][eft_INT] = {eft_INT, (void*)custom_sort_c<int>};
    fmap[DPNPFuncName::DPNP_FN_SORT][eft_LNG][eft_LNG] = {eft_LNG, (void*)custom_sort_c<long>};
    fmap[DPNPFuncName::DPNP_FN_SORT][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_sort_c<float>};
    fmap[DPNPFuncName::DPNP_FN_SORT][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_sort_c<double>};

    fmap[DPNPFuncName::DPNP_FN_SQRT][eft_INT][eft_INT] = {eft_DBL, (void*)custom_elemwise_sqrt_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_SQRT][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_elemwise_sqrt_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_SQRT][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_sqrt_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_SQRT][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_sqrt_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_SQUARE][eft_INT][eft_INT] = {eft_INT, (void*)custom_elemwise_square_c<int>};
    fmap[DPNPFuncName::DPNP_FN_SQUARE][eft_LNG][eft_LNG] = {eft_LNG, (void*)custom_elemwise_square_c<long>};
    fmap[DPNPFuncName::DPNP_FN_SQUARE][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_square_c<float>};
    fmap[DPNPFuncName::DPNP_FN_SQUARE][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_square_c<double>};

    fmap[DPNPFuncName::DPNP_FN_STD][eft_INT][eft_INT] = {eft_DBL, (void*)custom_std_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_STD][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_std_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_STD][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_std_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_STD][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_std_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_INT][eft_INT] = {eft_INT,
                                                              (void*)custom_elemwise_subtract_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_INT][eft_LNG] = {eft_LNG,
                                                              (void*)custom_elemwise_subtract_c<int, long, long>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_INT][eft_FLT] = {eft_DBL,
                                                              (void*)custom_elemwise_subtract_c<int, float, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_INT][eft_DBL] = {eft_DBL,
                                                              (void*)custom_elemwise_subtract_c<int, double, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_LNG][eft_INT] = {eft_LNG,
                                                              (void*)custom_elemwise_subtract_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_LNG][eft_LNG] = {eft_LNG,
                                                              (void*)custom_elemwise_subtract_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_LNG][eft_FLT] = {eft_DBL,
                                                              (void*)custom_elemwise_subtract_c<long, float, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_LNG][eft_DBL] = {eft_DBL,
                                                              (void*)custom_elemwise_subtract_c<long, double, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_FLT][eft_INT] = {eft_DBL,
                                                              (void*)custom_elemwise_subtract_c<float, int, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_FLT][eft_LNG] = {eft_DBL,
                                                              (void*)custom_elemwise_subtract_c<float, long, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_FLT][eft_FLT] = {eft_FLT,
                                                              (void*)custom_elemwise_subtract_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_FLT][eft_DBL] = {eft_DBL,
                                                              (void*)custom_elemwise_subtract_c<float, double, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_DBL][eft_INT] = {eft_DBL,
                                                              (void*)custom_elemwise_subtract_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_DBL][eft_LNG] = {eft_DBL,
                                                              (void*)custom_elemwise_subtract_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_DBL][eft_FLT] = {eft_DBL,
                                                              (void*)custom_elemwise_subtract_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_DBL][eft_DBL] = {
        eft_DBL, (void*)custom_elemwise_subtract_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_SUM][eft_INT][eft_INT] = {eft_INT, (void*)custom_sum_c<int>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_LNG][eft_LNG] = {eft_LNG, (void*)custom_sum_c<long>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_sum_c<float>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_sum_c<double>};

    fmap[DPNPFuncName::DPNP_FN_TAN][eft_INT][eft_INT] = {eft_DBL, (void*)custom_elemwise_tan_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_TAN][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_elemwise_tan_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_TAN][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_tan_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_TAN][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_tan_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_TANH][eft_INT][eft_INT] = {eft_DBL, (void*)custom_elemwise_tanh_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_TANH][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_elemwise_tanh_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_TANH][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_tanh_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_TANH][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_tanh_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_TRUNC][eft_INT][eft_INT] = {eft_DBL, (void*)custom_elemwise_trunc_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_TRUNC][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_elemwise_trunc_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_TRUNC][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_elemwise_trunc_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_TRUNC][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_elemwise_trunc_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_VAR][eft_INT][eft_INT] = {eft_DBL, (void*)custom_var_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_VAR][eft_LNG][eft_LNG] = {eft_DBL, (void*)custom_var_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_VAR][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_var_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_VAR][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_var_c<double, double>};

    return fmap;
};
