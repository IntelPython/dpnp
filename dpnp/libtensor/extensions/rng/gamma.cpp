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
static sample_dispatch_table_t gamma_dispatch_table;
static sample_dispatch_table_t beta_dispatch_table;
static sample_dispatch_table_t chisquare_dispatch_table;
static sample_dispatch_table_t standard_t_dispatch_table;
static sample_dispatch_table_t f_dispatch_table;

void init_gamma(py::module_ m)
{
    def_sampler<rng_ns::GammaSampler, RealTypes>(
        m, "_gamma",
        "Fill the result with gamma samples of shape shape and scale scale",
        gamma_dispatch_table);

    def_sampler<rng_ns::BetaSampler, RealTypes>(
        m, "_beta",
        "Fill the result with beta samples of shape parameters a and b",
        beta_dispatch_table);

    def_sampler<rng_ns::ChisquareSampler, RealTypes>(
        m, "_chisquare",
        "Fill the result with chi-square samples of df degrees of freedom",
        chisquare_dispatch_table);

    def_sampler<rng_ns::StandardTSampler, RealTypes>(
        m, "_standard_t",
        "Fill the result with Student's t samples of df degrees of freedom",
        standard_t_dispatch_table);

    def_sampler<rng_ns::FSampler, RealTypes>(
        m, "_f",
        "Fill the result with F samples of dfnum and dfden degrees of "
        "freedom",
        f_dispatch_table);
}
} // namespace dpnp::extensions::rng
