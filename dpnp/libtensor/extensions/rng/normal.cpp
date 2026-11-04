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
static sample_dispatch_table_t normal_dispatch_table;
static sample_dispatch_table_t lognormal_dispatch_table;
static sample_dispatch_table_t standard_cauchy_dispatch_table;
static sample_dispatch_table_t wald_dispatch_table;

void init_normal(py::module_ m)
{
    def_sampler<rng_ns::NormalSampler, RealTypes>(
        m, "_normal",
        "Fill the result with normal samples of mean loc and standard "
        "deviation scale",
        normal_dispatch_table);

    def_sampler<rng_ns::LognormalSampler, RealTypes>(
        m, "_lognormal",
        "Fill the result with log-normal samples whose logarithm has mean "
        "mean and standard deviation sigma",
        lognormal_dispatch_table);

    def_sampler<rng_ns::StandardCauchySampler, RealTypes>(
        m, "_standard_cauchy", "Fill the result with standard Cauchy samples",
        standard_cauchy_dispatch_table);

    def_sampler<rng_ns::WaldSampler, RealTypes>(
        m, "_wald",
        "Fill the result with Wald samples of mean mean and shape scale",
        wald_dispatch_table);
}
} // namespace dpnp::extensions::rng
