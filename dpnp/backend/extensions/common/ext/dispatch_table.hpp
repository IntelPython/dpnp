//*****************************************************************************
// Copyright (c) 2024-2025, Intel Corporation
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

#pragma once

#include <unordered_set>
#include <vector>

#include "utils/type_dispatch.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sycl/sycl.hpp>

#include "ext/common.hpp"

namespace dpctl_td_ns = dpctl::tensor::type_dispatch;
namespace py = pybind11;

namespace ext::common
{
template <typename T, typename Rest>
struct one_of
{
    static_assert(std::is_same_v<Rest, std::tuple<>>,
                  "one_of: second parameter cannot be empty std::tuple");
    static_assert(false, "one_of: second parameter must be std::tuple");
};

template <typename T, typename Top, typename... Rest>
struct one_of<T, std::tuple<Top, Rest...>>
{
    static constexpr bool value =
        std::is_same_v<T, Top> || one_of<T, std::tuple<Rest...>>::value;
};

template <typename T, typename Top>
struct one_of<T, std::tuple<Top>>
{
    static constexpr bool value = std::is_same_v<T, Top>;
};

template <typename T, typename Rest>
constexpr bool one_of_v = one_of<T, Rest>::value;

template <typename FnT>
using Table = FnT[dpctl_td_ns::num_types];
template <typename FnT>
using Table2 = Table<FnT>[dpctl_td_ns::num_types];

using TypeId = int32_t;
using TypesPair = std::pair<TypeId, TypeId>;

struct int_pair_hash
{
    inline size_t operator()(const TypesPair &p) const
    {
        std::hash<size_t> hasher;
        return hasher(size_t(p.first) << (8 * sizeof(TypeId)) |
                      size_t(p.second));
    }
};

using SupportedTypesList = std::vector<TypeId>;
using SupportedTypesList2 = std::vector<TypesPair>;
using SupportedTypesSet = std::unordered_set<TypeId>;
using SupportedTypesSet2 = std::unordered_set<TypesPair, int_pair_hash>;

using DType = py::dtype;
using DTypePair = std::pair<DType, DType>;

using SupportedDTypeList = std::vector<DType>;
using SupportedDTypeList2 = std::vector<DTypePair>;

template <typename FnT,
          typename SupportedTypes,
          template <typename>
          typename Func>
struct TableBuilder
{
    template <typename _FnT, typename T>
    struct impl
    {
        static constexpr bool is_defined = one_of_v<T, SupportedTypes>;

        _FnT get()
        {
            if constexpr (is_defined) {
                return Func<T>::impl;
            }
            else {
                return nullptr;
            }
        }
    };

    using type =
        dpctl_td_ns::DispatchVectorBuilder<FnT, impl, dpctl_td_ns::num_types>;
};

template <typename FnT,
          typename SupportedTypes,
          template <typename, typename>
          typename Func>
struct TableBuilder2
{
    template <typename _FnT, typename T1, typename T2>
    struct impl
    {
        static constexpr bool is_defined =
            one_of_v<std::tuple<T1, T2>, SupportedTypes>;

        _FnT get()
        {
            if constexpr (is_defined) {
                return Func<T1, T2>::impl;
            }
            else {
                return nullptr;
            }
        }
    };

    using type =
        dpctl_td_ns::DispatchTableBuilder<FnT, impl, dpctl_td_ns::num_types>;
};

template <typename FnT>
class DispatchTable
{
public:
    DispatchTable(std::string name) : name(name) {}

    template <typename SupportedTypes, template <typename> typename Func>
    void populate_dispatch_table()
    {
        using TBulder = typename TableBuilder<FnT, SupportedTypes, Func>::type;
        TBulder builder;

        builder.populate_dispatch_vector(table);
        populate_supported_types();
    }

    FnT get_unsafe(int _typenum) const
    {
        auto array_types = dpctl_td_ns::usm_ndarray_types();
        const int type_id = array_types.typenum_to_lookup_id(_typenum);

        return table[type_id];
    }

    FnT get(int _typenum) const
    {
        auto fn = get_unsafe(_typenum);

        if (fn == nullptr) {
            auto array_types = dpctl_td_ns::usm_ndarray_types();
            const int _type_id = array_types.typenum_to_lookup_id(_typenum);

            py::dtype _dtype = dtype_from_typenum(_type_id);
            auto _type_pos = std::find(supported_types.begin(),
                                       supported_types.end(), _dtype);
            if (_type_pos == supported_types.end()) {
                py::str types = py::str(py::cast(supported_types));
                py::str dtype = py::str(_dtype);

                py::str err_msg =
                    py::str("'" + name + "' has unsupported type '") + dtype +
                    py::str("'."
                            " Supported types are: ") +
                    types;

                throw py::value_error(static_cast<std::string>(err_msg));
            }
        }

        return fn;
    }

    const SupportedDTypeList &get_all_supported_types() const
    {
        return supported_types;
    }

private:
    void populate_supported_types()
    {
        for (int i = 0; i < dpctl_td_ns::num_types; ++i) {
            if (table[i] != nullptr) {
                supported_types.emplace_back(dtype_from_typenum(i));
            }
        }
    }

    std::string name;
    SupportedDTypeList supported_types;
    Table<FnT> table;
};

template <typename FnT>
class DispatchTable2
{
public:
    DispatchTable2(std::string first_name, std::string second_name)
        : first_name(first_name), second_name(second_name)
    {
    }

    template <typename SupportedTypes,
              template <typename, typename>
              typename Func>
    void populate_dispatch_table()
    {
        using TBulder = typename TableBuilder2<FnT, SupportedTypes, Func>::type;
        TBulder builder;

        builder.populate_dispatch_table(table);
        populate_supported_types();
    }

    FnT get_unsafe(int first_typenum, int second_typenum) const
    {
        auto array_types = dpctl_td_ns::usm_ndarray_types();
        const int first_type_id =
            array_types.typenum_to_lookup_id(first_typenum);
        const int second_type_id =
            array_types.typenum_to_lookup_id(second_typenum);

        return table[first_type_id][second_type_id];
    }

    FnT get(int first_typenum, int second_typenum) const
    {
        auto fn = get_unsafe(first_typenum, second_typenum);

        if (fn == nullptr) {
            auto array_types = dpctl_td_ns::usm_ndarray_types();
            const int first_type_id =
                array_types.typenum_to_lookup_id(first_typenum);
            const int second_type_id =
                array_types.typenum_to_lookup_id(second_typenum);

            py::dtype first_dtype = dtype_from_typenum(first_type_id);
            auto first_type_pos =
                std::find(supported_first_type.begin(),
                          supported_first_type.end(), first_dtype);
            if (first_type_pos == supported_first_type.end()) {
                py::str types = py::str(py::cast(supported_first_type));
                py::str dtype = py::str(first_dtype);

                py::str err_msg =
                    py::str("'" + first_name + "' has unsupported type '") +
                    dtype +
                    py::str("'."
                            " Supported types are: ") +
                    types;

                throw py::value_error(static_cast<std::string>(err_msg));
            }

            py::dtype second_dtype = dtype_from_typenum(second_type_id);
            auto second_type_pos =
                std::find(supported_second_type.begin(),
                          supported_second_type.end(), second_dtype);
            if (second_type_pos == supported_second_type.end()) {
                py::str types = py::str(py::cast(supported_second_type));
                py::str dtype = py::str(second_dtype);

                py::str err_msg =
                    py::str("'" + second_name + "' has unsupported type '") +
                    dtype +
                    py::str("'."
                            " Supported types are: ") +
                    types;

                throw py::value_error(static_cast<std::string>(err_msg));
            }

            py::str first_dtype_str = py::str(first_dtype);
            py::str second_dtype_str = py::str(second_dtype);
            py::str types = py::str(py::cast(all_supported_types));

            py::str err_msg =
                py::str("'" + first_name + "' and '" + second_name +
                        "' has unsupported types combination: ('") +
                first_dtype_str + py::str("', '") + second_dtype_str +
                py::str("')."
                        " Supported types combinations are: ") +
                types;

            throw py::value_error(static_cast<std::string>(err_msg));
        }

        return fn;
    }

    const SupportedDTypeList &get_supported_first_type() const
    {
        return supported_first_type;
    }

    const SupportedDTypeList &get_supported_second_type() const
    {
        return supported_second_type;
    }

    const SupportedDTypeList2 &get_all_supported_types() const
    {
        return all_supported_types;
    }

private:
    void populate_supported_types()
    {
        SupportedTypesSet first_supported_types_set;
        SupportedTypesSet second_supported_types_set;
        SupportedTypesSet2 all_supported_types_set;

        for (int i = 0; i < dpctl_td_ns::num_types; ++i) {
            for (int j = 0; j < dpctl_td_ns::num_types; ++j) {
                if (table[i][j] != nullptr) {
                    all_supported_types_set.emplace(i, j);
                    first_supported_types_set.emplace(i);
                    second_supported_types_set.emplace(j);
                }
            }
        }

        auto to_supported_dtype_list = [](const auto &supported_set,
                                          auto &supported_list) {
            SupportedTypesList lst(supported_set.begin(), supported_set.end());
            std::sort(lst.begin(), lst.end());
            supported_list.resize(supported_set.size());
            std::transform(lst.begin(), lst.end(), supported_list.begin(),
                           [](TypeId i) { return dtype_from_typenum(i); });
        };

        to_supported_dtype_list(first_supported_types_set,
                                supported_first_type);
        to_supported_dtype_list(second_supported_types_set,
                                supported_second_type);

        SupportedTypesList2 lst(all_supported_types_set.begin(),
                                all_supported_types_set.end());
        std::sort(lst.begin(), lst.end());
        all_supported_types.resize(all_supported_types_set.size());
        std::transform(lst.begin(), lst.end(), all_supported_types.begin(),
                       [](TypesPair p) {
                           return DTypePair(dtype_from_typenum(p.first),
                                            dtype_from_typenum(p.second));
                       });
    }

    std::string first_name;
    std::string second_name;

    SupportedDTypeList supported_first_type;
    SupportedDTypeList supported_second_type;
    SupportedDTypeList2 all_supported_types;

    Table2<FnT> table;
};

} // namespace ext::common
