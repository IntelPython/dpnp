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

#include "engine_base.hpp"


namespace dpnp::backend::ext::rng::device::engine
{
template <typename EngineT, typename SeedT, typename OffsetT>
class BaseBuilder {
private:
    static constexpr std::uint8_t max_n = 10;

    std::uint8_t no_of_seeds;
    std::uint8_t no_of_offsets;

    std::array<SeedT, max_n> seeds{};
    std::array<OffsetT, max_n> offsets{};

public:
    BaseBuilder(EngineBase *engine)
    {
        auto seed_values = engine->get_seeds();
        no_of_seeds = seed_values.size();
        if (no_of_seeds > max_n) {
            throw std::runtime_error("");
        }

        // TODO: implement a caster
        for (std::uint16_t i = 0; i < no_of_seeds; i++) {
            seeds[i] = static_cast<SeedT>(seed_values[i]);
        }

        auto offset_values = engine->get_offsets();
        no_of_offsets = offset_values.size();
        if (no_of_offsets > max_n) {
            throw std::runtime_error("");
        }

        // TODO: implement a caster
        for (std::uint16_t i = 0; i < no_of_seeds; i++) {
            offsets[i] = static_cast<OffsetT>(offset_values[i]);
        }
    }

    inline auto operator()() const
    {
        switch (no_of_seeds) {
            case 1: {
                return EngineT({seeds[0]}, {offsets[0]});
            }
            // TODO: implement full switch
            default:
                break;
        }
        return EngineT();
    }

    inline auto operator()(OffsetT offset) const
    {
        switch (no_of_seeds) {
            case 1: {
                return EngineT({seeds[0]}, offset);
            }
            // TODO: implement full switch
            default:
                break;
        }
        return EngineT();
    }

    // TODO: remove
    void print() {
        std::cout << "list_of_seeds: ";
        for (auto &val: seeds) {
            std::cout << std::to_string(val) << ", ";
        }
        std::cout << std::endl;
    }
};
} // dpnp::backend::ext::rng::device::engine
