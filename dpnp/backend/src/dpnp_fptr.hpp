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
 * This header file contains internal function declarations related to FPTR
 * interface. It should not contains public declarations
 */

#pragma once
#ifndef BACKEND_FPTR_H // Cython compatibility
#define BACKEND_FPTR_H

#include <complex>
#include <map>

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
 * An internal structure to build a pair of Data type enum value with C++ type
 */
template <DPNPFuncType FuncType, typename T>
struct func_type_pair_t
{
    using type = T;

    static func_type_pair_t
        get_pair(std::integral_constant<DPNPFuncType, FuncType>)
    {
        return {};
    }
};

/**
 * An internal structure to create a map of Data type enum value associated with
 * C++ type
 */
template <typename... Ps>
struct func_type_map_factory_t : public Ps...
{
    using Ps::get_pair...;

    template <DPNPFuncType FuncType>
    using find_type = typename decltype(get_pair(
        std::integral_constant<DPNPFuncType, FuncType>{}))::type;
};

/**
 * A map of the FPTR interface to link Data type enum value with associated C++
 * type
 */
typedef func_type_map_factory_t<
    func_type_pair_t<eft_BLN, bool>,
    func_type_pair_t<eft_INT, std::int32_t>,
    func_type_pair_t<eft_LNG, std::int64_t>,
    func_type_pair_t<eft_FLT, float>,
    func_type_pair_t<eft_DBL, double>,
    func_type_pair_t<eft_C64, std::complex<float>>,
    func_type_pair_t<eft_C128, std::complex<double>>>
    func_type_map_t;

/**
 * Return an enum value of result type populated from input types.
 */
template <DPNPFuncType FT1, DPNPFuncType FT2>
static constexpr DPNPFuncType populate_func_types()
{
    if constexpr (FT1 == DPNPFuncType::DPNP_FT_NONE) {
        throw std::runtime_error("Templated enum value of FT1 is None");
    }
    else if constexpr (FT2 == DPNPFuncType::DPNP_FT_NONE) {
        throw std::runtime_error("Templated enum value of FT2 is None");
    }
    return (FT1 < FT2) ? FT2 : FT1;
}

/**
 * @brief A helper function to cast SYCL vector between types.
 */
template <typename Op, typename Vec, std::size_t... I>
static auto dpnp_vec_cast_impl(const Vec &v, std::index_sequence<I...>)
{
    return Op{v[I]...};
}

/**
 * @brief A casting function for SYCL vector.
 *
 * @tparam dstT A result type upon casting.
 * @tparam srcT An incoming type of the vector.
 * @tparam N A number of elements with the vector.
 * @tparam Indices A sequence of integers
 * @param s An incoming SYCL vector to cast.
 * @return SYCL vector casted to desctination type.
 */
template <typename dstT,
          typename srcT,
          std::size_t N,
          typename Indices = std::make_index_sequence<N>>
static auto dpnp_vec_cast(const sycl::vec<srcT, N> &s)
{
    return dpnp_vec_cast_impl<sycl::vec<dstT, N>, sycl::vec<srcT, N>>(
        s, Indices{});
}

/**
 * Removes parentheses for a passed list of types separated by comma.
 * It's intended to be used in operations macro.
 */
#define MACRO_UNPACK_TYPES(...) __VA_ARGS__

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
 * A template constat to check if both types T1 and T2 match any type from Ts.
 */
template <typename T1, typename T2, typename... Ts>
constexpr auto both_types_are_any_of =
    std::conjunction_v<is_any<T1, Ts...>, is_any<T2, Ts...>>;

/**
 * A template constat to check if both types T1 and T2 don't match any type from
 * Ts sequence.
 */
template <typename T1, typename T2, typename... Ts>
constexpr auto none_of_both_types =
    !std::disjunction_v<is_any<T1, Ts...>, is_any<T2, Ts...>>;

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
 * A helper alias template to return true value for complex types and false
 * otherwise.
 */
template <typename _Tp>
struct is_complex : public std::integral_constant<
                        bool,
                        std::is_same_v<_Tp, std::complex<float>> ||
                            std::is_same_v<_Tp, std::complex<double>>>
{
};

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
                     (a.real() != a.real() && b.real() != b.real())) {
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
 * A template function that determines the default floating-point type
 * based on the value of the template parameter has_fp64.
 */
template <typename has_fp64 = std::true_type>
static constexpr DPNPFuncType get_default_floating_type()
{
    return has_fp64::value ? DPNPFuncType::DPNP_FT_DOUBLE
                           : DPNPFuncType::DPNP_FT_FLOAT;
}

/**
 * A template function that determines the resulting floating-point type
 * based on the value of the template parameter has_fp64.
 */
template <DPNPFuncType FT1,
          DPNPFuncType FT2,
          typename has_fp64 = std::true_type,
          typename keep_int = std::false_type>
static constexpr DPNPFuncType get_floating_res_type()
{
    constexpr auto widest_type = populate_func_types<FT1, FT2>();
    constexpr auto shortes_type = (widest_type == FT1) ? FT2 : FT1;

    // Return integer result type if save_int is True
    if constexpr (keep_int::value) {
        if constexpr (widest_type == DPNPFuncType::DPNP_FT_INT ||
                      widest_type == DPNPFuncType::DPNP_FT_LONG)
        {
            return widest_type;
        }
    }

    // Check for double
    if constexpr (widest_type == DPNPFuncType::DPNP_FT_DOUBLE) {
        return widest_type;
    }

    // Check for float
    else if constexpr (widest_type == DPNPFuncType::DPNP_FT_FLOAT) {
        // Check if the shortest type is also float
        if constexpr (shortes_type == DPNPFuncType::DPNP_FT_FLOAT) {
            return widest_type;
        }
    }

    // Default case
    return get_default_floating_type<has_fp64>();
}

/**
 * FPTR interface initialization functions
 */
void func_map_init_arraycreation(func_map_t &fmap);
void func_map_init_elemwise(func_map_t &fmap);
void func_map_init_indexing_func(func_map_t &fmap);
void func_map_init_linalg(func_map_t &fmap);
void func_map_init_mathematical(func_map_t &fmap);
void func_map_init_random(func_map_t &fmap);
void func_map_init_reduction(func_map_t &fmap);
void func_map_init_searching(func_map_t &fmap);
void func_map_init_sorting(func_map_t &fmap);
void func_map_init_statistics(func_map_t &fmap);

#endif // BACKEND_FPTR_H
