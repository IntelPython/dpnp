//*****************************************************************************
// Copyright (c) 2016-2025, Intel Corporation
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
 * This header file contains internal function declarations related to FPTR
 * interface. It should not contains public declarations
 */

#pragma once
#ifndef BACKEND_FPTR_H // Cython compatibility
#define BACKEND_FPTR_H

#include <complex>
#include <map>
#include <stdexcept>

#include <sycl/sycl.hpp>

#include <dpnp_iface_fptr.hpp>

/**
 * Data storage type of the FPTR interface
 *
 * map[FunctionName][InputType2][InputType2]
 *
 * Function name is enum DPNPFuncName
 * InputTypes are presented as enum DPNPFuncType
 *
 * contains structure with kernel information
 *
 * if the kernel requires only one input type - use same type for both
 * parameters
 *
 */
typedef std::map<DPNPFuncType, DPNPFuncData_t> map_2p_t;
typedef std::map<DPNPFuncType, map_2p_t> map_1p_t;
typedef std::map<DPNPFuncName, map_1p_t> func_map_t;

/**
 * Internal shortcuts for Data type enum values
 */
const DPNPFuncType eft_INT = DPNPFuncType::DPNP_FT_INT;
const DPNPFuncType eft_LNG = DPNPFuncType::DPNP_FT_LONG;
const DPNPFuncType eft_FLT = DPNPFuncType::DPNP_FT_FLOAT;
const DPNPFuncType eft_DBL = DPNPFuncType::DPNP_FT_DOUBLE;
const DPNPFuncType eft_C64 = DPNPFuncType::DPNP_FT_CMPLX64;
const DPNPFuncType eft_C128 = DPNPFuncType::DPNP_FT_CMPLX128;
const DPNPFuncType eft_BLN = DPNPFuncType::DPNP_FT_BOOL;

/**
 * Implements std::is_same<> with variadic number of types to compare with
 * and when type T has to match only one of types Ts.
 */
template <typename T, typename... Ts>
struct is_any : std::disjunction<std::is_same<T, Ts>...>
{
};

/**
 * Implements std::is_same<> with variadic number of types to compare with
 * and when type T has to match every type from Ts sequence.
 */
template <typename T, typename... Ts>
struct are_same : std::conjunction<std::is_same<T, Ts>...>
{
};

/**
 * A template constant to check if type T matches any type from Ts.
 */
template <typename T, typename... Ts>
constexpr auto is_any_v = is_any<T, Ts...>::value;

/**
 * A template constat to check if both types T1 and T2 match every type from Ts
 * sequence.
 */
template <typename T1, typename T2, typename... Ts>
constexpr auto both_types_are_same =
    std::conjunction_v<is_any<T1, Ts...>, are_same<T1, T2>>;

/**
 * @brief If the type _Tp is a reference type, provides the member typedef type
 * which is the type referred to by _Tp with its topmost cv-qualifiers removed.
 * Otherwise type is _Tp with its topmost cv-qualifiers removed.
 *
 * @note std::remove_cvref is only available since c++20
 */
template <typename _Tp>
using dpnp_remove_cvref_t =
    typename std::remove_cv_t<typename std::remove_reference_t<_Tp>>;

/**
 * @brief "<" comparison with complex types support.
 *
 * @note return a result of lexicographical "<" comparison for complex types.
 */
class dpnp_less_comp
{
public:
    template <typename _Xp, typename _Yp>
    bool operator()(_Xp &&__x, _Yp &&__y) const
    {
        if constexpr (both_types_are_same<
                          dpnp_remove_cvref_t<_Xp>, dpnp_remove_cvref_t<_Yp>,
                          std::complex<float>, std::complex<double>>)
        {
            bool ret = false;
            _Xp a = std::forward<_Xp>(__x);
            _Yp b = std::forward<_Yp>(__y);

            if (a.real() < b.real()) {
                ret = (a.imag() == a.imag() || b.imag() != b.imag());
            }
            else if (a.real() > b.real()) {
                ret = (b.imag() != b.imag() && a.imag() == a.imag());
            }
            else if (a.real() == b.real() ||
                     (a.real() != a.real() && b.real() != b.real()))
            {
                ret = (a.imag() < b.imag() ||
                       (b.imag() != b.imag() && a.imag() == a.imag()));
            }
            else {
                ret = (b.real() != b.real());
            }
            return ret;
        }
        else {
            return std::forward<_Xp>(__x) < std::forward<_Yp>(__y);
        }
    }
};

/**
 * FPTR interface initialization functions
 */
void func_map_init_arraycreation(func_map_t &fmap);
void func_map_init_elemwise(func_map_t &fmap);
void func_map_init_linalg(func_map_t &fmap);
void func_map_init_mathematical(func_map_t &fmap);
void func_map_init_random(func_map_t &fmap);
void func_map_init_sorting(func_map_t &fmap);

#endif // BACKEND_FPTR_H
