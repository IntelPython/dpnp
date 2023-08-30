//*****************************************************************************
// Copyright (c) 2023, Intel Corporation
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

#include <CL/sycl.hpp>

namespace dpnp
{
namespace backend
{
namespace ext
{
namespace rng
{
class EngineBase {
public:
    EngineBase(sycl::queue queue) {
        q = std::make_unique<sycl::queue>(queue);
    };

    sycl::queue& get_queue() { return *q; }

private:
    std::unique_ptr<sycl::queue> q;
};


template <typename EngineT, typename SeedT>
class EngineProxy: public EngineBase {
public:
    using engine_t = EngineT;

    EngineProxy(sycl::queue queue, SeedT seed): EngineBase(queue) {
        engine = std::make_unique<engine_t>(queue, seed);
    };

    // template <int... N>
    EngineProxy(sycl::queue queue, std::vector<SeedT> vec_seed): EngineBase(queue) {
        switch (vec_seed.size()) {
        case 1:
            engine = std::make_unique<engine_t>(queue, std::initializer_list<SeedT>({vec_seed[0]}));
            break;
        case 2:
            engine = std::make_unique<engine_t>(queue, std::initializer_list<SeedT>({vec_seed[0], vec_seed[1]}));
            break;
        case 3:
            engine = std::make_unique<engine_t>(queue, std::initializer_list<SeedT>({vec_seed[0], vec_seed[1], vec_seed[2]}));
            break;
        default:
            // TODO need to get rid of the limitation for seed vector length
            throw std::runtime_error("Too long seed vector");
        }
    };

    // ~EngineProxy() = default;

private:
    std::unique_ptr<engine_t> engine;

};
} // namespace lapack
} // namespace ext
} // namespace backend
} // namespace rng
