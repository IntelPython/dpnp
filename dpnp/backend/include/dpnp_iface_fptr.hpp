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
 * We would like to avoid backend specific things in higher level Cython
 * modules. Any backend interface functions and types should be defined here.
 *
 * Also, this file should contains documentation on functions and types
 * which are used in the interface
 */

#pragma once
#ifndef BACKEND_IFACE_FPTR_H // Cython compatibility
#define BACKEND_IFACE_FPTR_H

#include <dpnp_iface.hpp>

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
    DPNP_FN_NONE,          /**< Very first element of the enumeration */
    DPNP_FN_PARTITION_EXT, /**< Used in numpy.partition() impl, requires extra
                              parameters */
    DPNP_FN_LAST,          /**< The latest element of the enumeration */
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
    DPNP_FT_NONE,    /**< Very first element of the enumeration */
    DPNP_FT_BOOL,    /**< analog of numpy.bool_ or bool */
    DPNP_FT_INT,     /**< analog of numpy.int32 or int */
    DPNP_FT_LONG,    /**< analog of numpy.int64 or long */
    DPNP_FT_FLOAT,   /**< analog of numpy.float32 or float */
    DPNP_FT_DOUBLE,  /**< analog of numpy.float32 or double */
    DPNP_FT_CMPLX64, /**< analog of numpy.complex64 or std::complex<float> */
    DPNP_FT_CMPLX128 /**< analog of numpy.complex128 or std::complex<double> */
};

/**
 * This operator is needed for compatibility with Cython 0.29 which has a bug in
 * Enum handling
 * TODO needs to be deleted in future
 */
INP_DLLEXPORT
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
    DPNPFuncData(const DPNPFuncType gen_type,
                 void *gen_ptr,
                 const DPNPFuncType type_no_fp64,
                 void *ptr_no_fp64)
        : return_type(gen_type), ptr(gen_ptr),
          return_type_no_fp64(type_no_fp64), ptr_no_fp64(ptr_no_fp64)
    {
    }
    DPNPFuncData(const DPNPFuncType gen_type, void *gen_ptr)
        : DPNPFuncData(gen_type, gen_ptr, DPNPFuncType::DPNP_FT_NONE, nullptr)
    {
    }
    DPNPFuncData() : DPNPFuncData(DPNPFuncType::DPNP_FT_NONE, nullptr) {}

    DPNPFuncType return_type; /**< return type identifier which expected by the
                                 @ref ptr function */
    void *ptr;                /**< C++ backend function pointer */
    DPNPFuncType return_type_no_fp64; /**< alternative return type identifier
                                         when no fp64 support by device */
    void *ptr_no_fp64; /**< alternative C++ backend function pointer when no
                          fp64 support by device */
} DPNPFuncData_t;

/**
 * @ingroup BACKEND_API
 * @brief get runtime pointer to selected function
 *
 * Runtime pointer to the backend API function from storage map<name,
 * map<first_type, map<second_type, DPNPFuncData_t>>>
 *
 * @param [in]  name         Name of the function in storage
 * @param [in]  first_type   First type of the storage
 * @param [in]  second_type  Second type of the storage
 *
 * @return Struct @ref DPNPFuncData_t with information about the backend API
 * function.
 */
INP_DLLEXPORT
DPNPFuncData_t get_dpnp_function_ptr(
    DPNPFuncName name,
    DPNPFuncType first_type,
    DPNPFuncType second_type = DPNPFuncType::DPNP_FT_NONE);

/**
 * @ingroup BACKEND_API
 * @brief get runtime pointer to selected function
 *
 * Same interface function as @ref get_dpnp_function_ptr with a bit different
 * interface
 *
 * @param [out] result_type  Type of the result provided by the backend API
 * function
 * @param [in]  name         Name of the function in storage
 * @param [in]  first_type   First type of the storage
 * @param [in]  second_type  Second type of the storage
 *
 * @return pointer to the backend API function.
 */
INP_DLLEXPORT
void *get_dpnp_function_ptr1(
    DPNPFuncType &result_type,
    DPNPFuncName name,
    DPNPFuncType first_type,
    DPNPFuncType second_type = DPNPFuncType::DPNP_FT_NONE);

#endif // BACKEND_IFACE_FPTR_H
