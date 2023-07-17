#include <array>
#include <tuple>

namespace dpnp
{
namespace backend
{
namespace ext
{
namespace sycl_ext
{

template <template <typename...> class C, typename Tuple>
struct resolve_type
{
};

template <template <typename...> class C, typename... Args>
struct resolve_type<C, std::tuple<Args...>>
{
    using type = C<Args...>;
};

template <typename Tuple>
struct pop_tuple_type
{
};

template <typename T, typename... Args>
struct pop_tuple_type<std::tuple<T, Args...>>
{
    using element = T;
    using rest = std::tuple<Args...>;
};

template <typename FirstTuple, typename SecondTuple>
struct merge_two_tuple_types
{
};

template <typename... FirstTupleArgs, typename... SecondTupleArgs>
struct merge_two_tuple_types<std::tuple<FirstTupleArgs...>,
                             std::tuple<SecondTupleArgs...>>
{
    using type = std::tuple<FirstTupleArgs..., SecondTupleArgs...>;
};

template <typename FirstTuple, typename SecondTuple>
struct merge_two_inner_tuple_types
{
};

template <typename... FirstTupleArgs, typename... SecondTupleArgs>
struct merge_two_inner_tuple_types<std::tuple<std::tuple<FirstTupleArgs...>>,
                                   std::tuple<std::tuple<SecondTupleArgs...>>>
{
    using type = std::tuple<std::tuple<FirstTupleArgs..., SecondTupleArgs...>>;
};

template <typename... FirstTupleArgs, typename... SecondTupleArgs>
struct merge_two_inner_tuple_types<std::tuple<FirstTupleArgs...>,
                                   std::tuple<SecondTupleArgs...>>
{
    using T1 = typename pop_tuple_type<std::tuple<FirstTupleArgs...>>::element;
    using rest1 = typename pop_tuple_type<std::tuple<FirstTupleArgs...>>::rest;

    using T2 = typename pop_tuple_type<std::tuple<SecondTupleArgs...>>::element;
    using rest2 = typename pop_tuple_type<std::tuple<SecondTupleArgs...>>::rest;

    using merged_first = typename merge_two_tuple_types<T1, T2>::type;
    using merged_rest =
        typename merge_two_inner_tuple_types<rest1, rest2>::type;

    using type = typename merge_two_tuple_types<std::tuple<merged_first>,
                                                merged_rest>::type;
};

template <typename Tuple, typename T>
struct extend_tuple_type
{
};

template <typename... Args, typename T>
struct extend_tuple_type<std::tuple<Args...>, T>
{
    using type = std::tuple<Args..., T>;
};

template <typename... Args>
struct tuple_of_tuples
{
};

template <typename T>
struct tuple_of_tuples<std::tuple<T>>
{
    using type = std::tuple<std::tuple<T>>;
};

template <typename... Args>
struct tuple_of_tuples<std::tuple<Args...>>
{
    using T = typename pop_tuple_type<std::tuple<Args...>>::element;
    using other = typename pop_tuple_type<std::tuple<Args...>>::rest;

    using tuple_t = std::tuple<T>;
    using tuple_other = typename tuple_of_tuples<other>::type;
    using type =
        typename merge_two_tuple_types<std::tuple<tuple_t>, tuple_other>::type;
};

template <typename T, int N>
struct tuple_of_n_elements
{
    using type =
        typename extend_tuple_type<typename tuple_of_n_elements<T, N - 1>::type,
                                   T>::type;
};

template <typename T>
struct tuple_of_n_elements<T, 1>
{
    using type = std::tuple<T>;
};

template <typename T>
struct tuple_of_n_elements<T, 0>
{
    using type = std::tuple<>;
};

template <typename... Args>
struct cartesian_product_impl
{
};

template <typename T, typename... SecondTupleArgs>
struct cartesian_product_impl<std::tuple<std::tuple<T>>,
                              std::tuple<SecondTupleArgs...>>
{
    using extended_T_tuple =
        typename tuple_of_n_elements<T, sizeof...(SecondTupleArgs)>::type;
    using extended_T_tuple_of_tuples =
        typename tuple_of_tuples<extended_T_tuple>::type;
    using type = typename merge_two_inner_tuple_types<
        extended_T_tuple_of_tuples,
        std::tuple<SecondTupleArgs...>>::type;
};

template <typename T, typename... FirstTupleArgs, typename... SecondTupleArgs>
struct cartesian_product_impl<std::tuple<std::tuple<T>, FirstTupleArgs...>,
                              std::tuple<SecondTupleArgs...>>
{
    using extended_T_tuple =
        typename tuple_of_n_elements<T, sizeof...(SecondTupleArgs)>::type;
    using extended_T_tuple_of_tuples =
        typename tuple_of_tuples<extended_T_tuple>::type;
    using cartesian_with_first = typename merge_two_inner_tuple_types<
        extended_T_tuple_of_tuples,
        std::tuple<SecondTupleArgs...>>::type;
    using cartesian_with_other =
        typename cartesian_product_impl<std::tuple<FirstTupleArgs...>,
                                        std::tuple<SecondTupleArgs...>>::type;
    using type = typename merge_two_tuple_types<cartesian_with_first,
                                                cartesian_with_other>::type;
};

template <typename... Args>
struct cartesian_product
{
};

template <typename... FirstTupleArgs>
struct cartesian_product<std::tuple<std::tuple<FirstTupleArgs...>>>
{
    using type = typename tuple_of_tuples<std::tuple<FirstTupleArgs...>>::type;
};

template <typename... Tuples>
struct cartesian_product<std::tuple<Tuples...>>
{
    using first_tuple = typename pop_tuple_type<std::tuple<Tuples...>>::element;
    using other = typename pop_tuple_type<std::tuple<Tuples...>>::rest;
    using cartesian_other = typename cartesian_product<other>::type;
    using first_tuple_of_tuples = typename tuple_of_tuples<first_tuple>::type;
    using type = typename cartesian_product_impl<first_tuple_of_tuples,
                                                 cartesian_other>::type;
};

template <template <typename...> class C, typename FnT, typename Tuple>
struct populate_table
{
};

template <template <typename...> class C, typename FnT, typename... Args>
struct populate_table<C, FnT, std::tuple<std::tuple<Args...>>>
{
    static void populate(FnT *data)
    {
        data[0] = resolve_type<C, std::tuple<Args...>>::type::call;
    }
};

template <template <typename...> class C, typename FnT, typename... Args>
struct populate_table<C, FnT, std::tuple<Args...>>
{
    static void populate(FnT *data)
    {
        using T = typename pop_tuple_type<std::tuple<Args...>>::element;
        using rest = typename pop_tuple_type<std::tuple<Args...>>::rest;

        data[0] = resolve_type<C, T>::type::call;
        populate_table<C, FnT, rest>::populate(data + 1);
    }
};

template <size_t idx, typename DispatchT, typename Matcher, typename Tuple>
struct index_on_axis
{
};

template <size_t idx, typename DispatchT, typename Matcher, typename T>
struct index_on_axis<idx, DispatchT, Matcher, std::tuple<T>>
{
    static int get_index(const DispatchT &val, int *result)
    {
        if (Matcher::template match<T>(val)) {
            result[0] = idx;
            return true;
        }

        result[0] = idx + 1;
        return false;
    }
};

template <size_t idx,
          typename DispatchT,
          typename Matcher,
          typename... AxisData>
struct index_on_axis<idx, DispatchT, Matcher, std::tuple<AxisData...>>
{
    using T = typename pop_tuple_type<std::tuple<AxisData...>>::element;
    using rest = typename pop_tuple_type<std::tuple<AxisData...>>::rest;

    static int get_index(const DispatchT &val, int *result)
    {
        if (Matcher::template match<T>(val)) {
            result[0] = idx;
            return true;
        }

        return index_on_axis<idx + 1, DispatchT, Matcher, rest>::get_index(
            val, result);
    }
};

template <typename DispatchT, typename Matcher, typename... AxisData>
struct coord_in_space
{
};

template <typename DispatchT, typename Matcher, typename CurrentAxis>
struct coord_in_space<DispatchT, Matcher, std::tuple<CurrentAxis>>
{
    static bool get(const DispatchT *data, int *result)
    {
        auto found =
            index_on_axis<0, DispatchT, Matcher, CurrentAxis>::get_index(
                data[0], result);

        return found;
    }
};

template <typename DispatchT, typename Matcher, typename... Axises>
struct coord_in_space<DispatchT, Matcher, std::tuple<Axises...>>
{
    using CurrentAxis = typename pop_tuple_type<std::tuple<Axises...>>::element;
    using OtherAxises = typename pop_tuple_type<std::tuple<Axises...>>::rest;

    static bool get(const DispatchT *data, int *result)
    {
        auto found =
            index_on_axis<0, DispatchT, Matcher, CurrentAxis>::get_index(
                data[0], result);
        if (found) {
            found = coord_in_space<DispatchT, Matcher, OtherAxises>::get(
                data + 1, result + 1);

            return found;
        }

        return found;
    }
};

template <typename Tuple>
struct elements_count
{
};

template <typename T>
struct elements_count<std::tuple<T>>
{
    static constexpr int count()
    {
        return std::tuple_size_v<T>;
    }
};

template <typename T, typename... Axis>
struct elements_count<std::tuple<T, Axis...>>
{
    static constexpr int count()
    {
        return std::tuple_size_v<T> *
               elements_count<std::tuple<Axis...>>::count();
    }
};

template <typename Tuple>
struct get_linear_id
{
};

template <typename T>
struct get_linear_id<std::tuple<T>>
{
    static int get(const int *data)
    {
        return data[0];
    }
};

template <typename T, typename... Axis>
struct get_linear_id<std::tuple<T, Axis...>>
{
    static int get(const int *data)
    {
        return data[0] * elements_count<std::tuple<Axis...>>::count() +
               get_linear_id<std::tuple<Axis...>>::get(data + 1);
    }
};

template <template <typename...> class C,
          typename FnT,
          typename DispatchT,
          typename Matcher,
          typename... Axis>
struct CartesianDispatcher
{
    using space = typename cartesian_product<std::tuple<Axis...>>::type;

    std::array<FnT, std::tuple_size_v<space>> dispatch_table;

    CartesianDispatcher()
    {
        populate_table<C, FnT, space>::populate(dispatch_table.data());
    }

    FnT operator()(std::initializer_list<DispatchT> values)
    {
        // though std::initializer_list::size is constexpr it doesnt work for
        // some reson constexpr auto const list_size = values.size();
        std::array<int, sizeof...(Axis)> idx;
        bool found =
            coord_in_space<DispatchT, Matcher, std::tuple<Axis...>>::get(
                std::data(values), idx.data());

        if (found) {
            int linear_idx =
                get_linear_id<std::tuple<Axis...>>::get(idx.data());
            return dispatch_table[linear_idx];
        }

        return nullptr;
    }
};

struct UsmArrayMatcher
{
    template <class T>
    static bool match(const dpctl::tensor::usm_ndarray &)
    {
        return false;
    }
};

// int typenum_to_lookup_id(int typenum) const
// {
//     using typenum_t = ::dpctl::tensor::type_dispatch::typenum_t;
//     auto const &api = ::dpctl::detail::dpctl_capi::get();

//     if (typenum == api.UAR_DOUBLE_) {
//         return static_cast<int>(typenum_t::DOUBLE);
//     }
//     else if (typenum == api.UAR_INT64_) {
//         return static_cast<int>(typenum_t::INT64);
//     }
//     else if (typenum == api.UAR_INT32_) {
//         return static_cast<int>(typenum_t::INT32);
//     }
//     else if (typenum == api.UAR_BOOL_) {
//         return static_cast<int>(typenum_t::BOOL);
//     }
//     else if (typenum == api.UAR_CDOUBLE_) {
//         return static_cast<int>(typenum_t::CDOUBLE);
//     }
//     else if (typenum == api.UAR_FLOAT_) {
//         return static_cast<int>(typenum_t::FLOAT);
//     }
//     else if (typenum == api.UAR_INT16_) {
//         return static_cast<int>(typenum_t::INT16);
//     }
//     else if (typenum == api.UAR_INT8_) {
//         return static_cast<int>(typenum_t::INT8);
//     }
//     else if (typenum == api.UAR_UINT64_) {
//         return static_cast<int>(typenum_t::UINT64);
//     }
//     else if (typenum == api.UAR_UINT32_) {
//         return static_cast<int>(typenum_t::UINT32);
//     }
//     else if (typenum == api.UAR_UINT16_) {
//         return static_cast<int>(typenum_t::UINT16);
//     }
//     else if (typenum == api.UAR_UINT8_) {
//         return static_cast<int>(typenum_t::UINT8);
//     }
//     else if (typenum == api.UAR_CFLOAT_) {
//         return static_cast<int>(typenum_t::CFLOAT);
//     }
//     else if (typenum == api.UAR_HALF_) {
//         return static_cast<int>(typenum_t::HALF);
//     }
//     else if (typenum == api.UAR_INT_ || typenum == api.UAR_UINT_) {
//         switch (sizeof(int)) {
//         case sizeof(std::int32_t):
//             return ((typenum == api.UAR_INT_)
//                         ? static_cast<int>(typenum_t::INT32)
//                         : static_cast<int>(typenum_t::UINT32));
//         case sizeof(std::int64_t):
//             return ((typenum == api.UAR_INT_)
//                         ? static_cast<int>(typenum_t::INT64)
//                         : static_cast<int>(typenum_t::UINT64));
//         default:
//             throw_unrecognized_typenum_error(typenum);
//         }
//     }
//     else if (typenum == api.UAR_LONGLONG_ || typenum == api.UAR_ULONGLONG_)
//     {
//         switch (sizeof(long long)) {
//         case sizeof(std::int64_t):
//             return ((typenum == api.UAR_LONGLONG_)
//                         ? static_cast<int>(typenum_t::INT64)
//                         : static_cast<int>(typenum_t::UINT64));
//         default:
//             throw_unrecognized_typenum_error(typenum);
//         }
//     }
//     else {
//         throw_unrecognized_typenum_error(typenum);
//     }
//     // return code signalling error, should never be reached
//     assert(false);
//     return -1;
// }

template <>
bool UsmArrayMatcher::match<int8_t>(const dpctl::tensor::usm_ndarray &arr)
{
    return arr.get_typenum() == UAR_BYTE;
}

template <>
bool UsmArrayMatcher::match<uint8_t>(const dpctl::tensor::usm_ndarray &arr)
{
    return arr.get_typenum() == UAR_UBYTE;
}

template <>
bool UsmArrayMatcher::match<int16_t>(const dpctl::tensor::usm_ndarray &arr)
{
    return arr.get_typenum() == UAR_SHORT;
}

template <>
bool UsmArrayMatcher::match<uint16_t>(const dpctl::tensor::usm_ndarray &arr)
{
    return arr.get_typenum() == UAR_USHORT;
}

template <>
bool UsmArrayMatcher::match<int32_t>(const dpctl::tensor::usm_ndarray &arr)
{
    return (arr.get_typenum() == UAR_INT and sizeof(int) == sizeof(int32_t)) or
           (arr.get_typenum() == UAR_LONG and sizeof(long) == sizeof(int32_t));
}

template <>
bool UsmArrayMatcher::match<uint32_t>(const dpctl::tensor::usm_ndarray &arr)
{
    return (arr.get_typenum() == UAR_UINT and
            sizeof(unsigned int) == sizeof(uint32_t)) or
           (arr.get_typenum() == UAR_ULONG and
            sizeof(unsigned long) == sizeof(uint32_t));
}

template <>
bool UsmArrayMatcher::match<int64_t>(const dpctl::tensor::usm_ndarray &arr)
{
    return (arr.get_typenum() == UAR_INT and sizeof(int) == sizeof(int64_t)) or
           (arr.get_typenum() == UAR_LONG and
            sizeof(long) == sizeof(int64_t)) or
           (arr.get_typenum() == UAR_LONGLONG and
            sizeof(long long) == sizeof(int64_t));
}

template <>
bool UsmArrayMatcher::match<uint64_t>(const dpctl::tensor::usm_ndarray &arr)
{
    return (arr.get_typenum() == UAR_UINT and
            sizeof(unsigned int) == sizeof(uint64_t)) or
           (arr.get_typenum() == UAR_ULONG and
            sizeof(unsigned long) == sizeof(uint64_t)) or
           (arr.get_typenum() == UAR_ULONGLONG and
            sizeof(unsigned long long) == sizeof(uint64_t));
}

template <>
bool UsmArrayMatcher::match<float>(const dpctl::tensor::usm_ndarray &arr)
{
    return arr.get_typenum() == UAR_FLOAT;
}

template <>
bool UsmArrayMatcher::match<double>(const dpctl::tensor::usm_ndarray &arr)
{
    return arr.get_typenum() == UAR_DOUBLE;
}

template <class NumT, class DenT>
NumT DivUp(NumT numerator, DenT denominator)
{
    return (numerator + denominator - 1) / denominator;
}

template <class VT, class BT>
VT Align(VT value, BT base)
{
    return base * DivUp(value, base);
}

} // namespace sycl_ext
} // namespace ext
} // namespace backend
} // namespace dpnp
