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
 * We would like to avoid backend specific things in higher level Cython modules.
 * Any backend interface functions and types should be defined here.
 *
 * Also, this file should contains documentation on functions and types
 * which are used in the interface
 */

#pragma once
#ifndef BACKEND_IFACE_FPTR_H // Cython compatibility
#define BACKEND_IFACE_FPTR_H

#include <vector>

#include <backend/backend_iface.hpp>

/**
 * @defgroup BACKEND_FUNC_PTR_API Backend C++ library runtime interface API
 * @{
 * This section describes Backend API for runtime function pointers
 * @}
 */

/**
 * @ingroup BACKEND_FUNC_PTR_API
 * @brief Function names to request via this interface
 *
 * The structure defines the parameters that are used
 * by @ref get_dpnp_function_ptr "get_dpnp_function_ptr".
 */
enum class DPNPFuncName : size_t
{
    DPNP_FN_NONE,    /**< Very first element of the enumeration */
    DPNP_FN_ADD,     /**< Used in numpy.add() implementation  */
    DPNP_FN_ARGMAX,  /**< Used in numpy.argmax() implementation  */
    DPNP_FN_ARGMIN,  /**< Used in numpy.argmin() implementation  */
    DPNP_FN_DOT,     /**< Used in numpy.dot() implementation  */
    DPNP_FN_MAXIMUM, /**< Used in numpy.maximum() implementation  */
    DPNP_FN_MINIMUM, /**< Used in numpy.minimum() implementation  */
    DPNP_FN_LAST     /**< The latest element of the enumeration */
};

/**
 * @ingroup BACKEND_FUNC_PTR_API
 * @brief Template types which are used in this interface
 *
 * The structure defines the types that are used
 * by @ref get_dpnp_function_ptr "get_dpnp_function_ptr".
 */
enum class DPNPFuncType : size_t
{
    DPNP_FT_NONE,  /**< Very first element of the enumeration */
    DPNP_FT_INT,   /**< analog of numpy.int32 or int */
    DPNP_FT_LONG,  /**< analog of numpy.int64 or long */
    DPNP_FT_FLOAT, /**< analog of numpy.float32 or float */
    DPNP_FT_DOUBLE /**< analog of numpy.float32 or double */
};

/**
 * This operator is needed for compatibility with Cython 0.29 which has a bug in Enum handling
 * TODO needs to be deleted in future
 */
size_t operator-(DPNPFuncType lhs, DPNPFuncType rhs);

/**
 * @ingroup BACKEND_FUNC_PTR_API
 * @brief Contains information about the C++ backend function
 *
 * The structure defines the types that are used
 * by @ref get_dpnp_function_ptr "get_dpnp_function_ptr".
 */
typedef struct DPNPFuncData
{
    DPNPFuncType return_type; /**< return type identifier which expected by the @ref ptr function */
    void* ptr;                /**< C++ backend function pointer */
} DPNPFuncData_t;

/**
 * @ingroup BACKEND_API
 * @brief get runtime pointer to selected function
 *
 * Runtime pointer to the backend API function from storage map<name, map<first_type, map<second_type, DPNPFuncData_t>>>
 *
 * @param [in]  name         Name of the function in storage
 * @param [in]  first_type   First type of the storage
 * @param [in]  second_type  Second type of the storage
 *
 * @return Struct @ref DPNPFuncData_t with information about the backend API function.
 */
INP_DLLEXPORT
DPNPFuncData_t get_dpnp_function_ptr(DPNPFuncName name,
                                     DPNPFuncType first_type,
                                     DPNPFuncType second_type = DPNPFuncType::DPNP_FT_NONE);

/**
 * @ingroup BACKEND_API
 * @brief get runtime pointer to selected function
 *
 * Same interface function as @ref get_dpnp_function_ptr with a bit diffrent interface
 *
 * @param [out] result_type  Type of the result provided by the backend API function
 * @param [in]  name         Name of the function in storage
 * @param [in]  first_type   First type of the storage
 * @param [in]  second_type  Second type of the storage
 *
 * @return pointer to the backend API function.
 */
INP_DLLEXPORT
void* get_dpnp_function_ptr1(DPNPFuncType& result_type,
                             DPNPFuncName name,
                             DPNPFuncType first_type,
                             DPNPFuncType second_type = DPNPFuncType::DPNP_FT_NONE);

/**
 * DEPRECATED.
 * Experimental interface. DO NOT USE IT!
 *
 * parameter @ref type_name will be converted into var_args or char *[] with extra length parameter
 */
INP_DLLEXPORT
void* get_backend_function_name(const char* func_name, const char* type_name) __attribute__((deprecated));

#endif // BACKEND_IFACE_FPTR_H
