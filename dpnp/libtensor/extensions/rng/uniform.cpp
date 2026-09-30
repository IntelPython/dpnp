//*****************************************************************************
// Copyright (c) 2026, Intel Corporation
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
#include <pybind11/pybind11.h>

#include "common.hpp"
#include "distributions.hpp"

namespace dpnp::extensions::rng
{
static sample_dispatch_table_t uniform_dispatch_table;
static sample_dispatch_table_t random_raw_dispatch_table;
static sample_dispatch_table_t laplace_dispatch_table;
static sample_dispatch_table_t gumbel_dispatch_table;
static sample_dispatch_table_t logistic_dispatch_table;
static sample_dispatch_table_t triangular_dispatch_table;
static sample_dispatch_table_t vonmises_dispatch_table;

void init_uniform(py::module_ m)
{
    def_sampler<rng_ns::UniformSampler, RealTypes>(
        m, "_uniform", "Fill the result with uniform samples on [low, high)",
        uniform_dispatch_table);

    def_sampler<rng_ns::RawSampler, RawTypes>(
        m, "_random_raw",
        "Fill the result with the raw 64-bit output of the engine, or with "
        "its 32-bit halves, low half first",
        random_raw_dispatch_table);

    def_sampler<rng_ns::LaplaceSampler, RealTypes>(
        m, "_laplace",
        "Fill the result with Laplace samples of location loc and scale "
        "scale",
        laplace_dispatch_table);

    def_sampler<rng_ns::GumbelSampler, RealTypes>(
        m, "_gumbel",
        "Fill the result with Gumbel samples of location loc and scale "
        "scale",
        gumbel_dispatch_table);

    def_sampler<rng_ns::LogisticSampler, RealTypes>(
        m, "_logistic",
        "Fill the result with logistic samples of location loc and scale "
        "scale",
        logistic_dispatch_table);

    def_sampler<rng_ns::TriangularSampler, RealTypes>(
        m, "_triangular",
        "Fill the result with triangular samples on [left, right] with mode "
        "mode",
        triangular_dispatch_table);

    def_sampler<rng_ns::VonmisesSampler, RealTypes>(
        m, "_vonmises",
        "Fill the result with von Mises samples of mode mu and "
        "concentration kappa",
        vonmises_dispatch_table);
}
} // namespace dpnp::extensions::rng
