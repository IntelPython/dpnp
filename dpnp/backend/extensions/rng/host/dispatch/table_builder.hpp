//*****************************************************************************
// Copyright (c) 2024, Intel Corporation
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

#include <oneapi/mkl/rng.hpp>

namespace dpnp::backend::ext::rng::host::dispatch
{
namespace mkl_rng = oneapi::mkl::rng;

template <typename funcPtrT,
          template <typename fnT, typename E, typename T, typename M>
          typename factory,
          int _no_of_engines,
          int _no_of_types,
          int _no_of_methods>
class Dispatch3DTableBuilder
{
private:
    template <typename E, typename T>
    const std::vector<funcPtrT> row_per_method() const
    {
        std::vector<funcPtrT> per_method = {
            factory<funcPtrT, E, T, mkl_rng::gaussian_method::by_default>{}
                .get(),
            factory<funcPtrT, E, T, mkl_rng::gaussian_method::box_muller2>{}
                .get(),
        };
        assert(per_method.size() == _no_of_methods);
        return per_method;
    }

    template <typename E>
    auto table_per_type_and_method() const
    {
        std::vector<std::vector<funcPtrT>> table_by_type = {
            row_per_method<E, bool>(),
            row_per_method<E, int8_t>(),
            row_per_method<E, uint8_t>(),
            row_per_method<E, int16_t>(),
            row_per_method<E, uint16_t>(),
            row_per_method<E, int32_t>(),
            row_per_method<E, uint32_t>(),
            row_per_method<E, int64_t>(),
            row_per_method<E, uint64_t>(),
            row_per_method<E, sycl::half>(),
            row_per_method<E, float>(),
            row_per_method<E, double>(),
            row_per_method<E, std::complex<float>>(),
            row_per_method<E, std::complex<double>>()};
        assert(table_by_type.size() == _no_of_types);
        return table_by_type;
    }

public:
    Dispatch3DTableBuilder() = default;
    ~Dispatch3DTableBuilder() = default;

    void populate(funcPtrT table[][_no_of_types][_no_of_methods]) const
    {
        const auto map_by_engine = {
            table_per_type_and_method<mkl_rng::mrg32k3a>(),
            table_per_type_and_method<mkl_rng::philox4x32x10>(),
            table_per_type_and_method<mkl_rng::mcg31m1>(),
            table_per_type_and_method<mkl_rng::mcg59>()};
        assert(map_by_engine.size() == _no_of_engines);

        std::uint16_t engine_id = 0;
        for (auto &table_by_type : map_by_engine) {
            std::uint16_t type_id = 0;
            for (auto &row_by_method : table_by_type) {
                std::uint16_t method_id = 0;
                for (auto &fn_ptr : row_by_method) {
                    table[engine_id][type_id][method_id] = fn_ptr;
                    ++method_id;
                }
                ++type_id;
            }
            ++engine_id;
        }
    }
};
} // namespace dpnp::backend::ext::rng::host::dispatch
