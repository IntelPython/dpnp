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
//
// This file defines functions of dpnp.libtensor._rng_impl extensions
//
//*****************************************************************************

#include <pybind11/pybind11.h>

#include "kernels/engines.hpp"
#include "kernels/sample.hpp"

#include "distributions.hpp"

namespace rng_ns = dpnp::extensions::rng;
namespace py = pybind11;

PYBIND11_MODULE(_rng_impl, m, py::mod_gil_not_used())
{
    using dpnp::kernels::rng::engine_id;

    py::enum_<engine_id>(m, "engine_id")
        .value("philox4x32x10", engine_id::philox4x32x10)
        .value("pcg64dxsm", engine_id::pcg64dxsm);

    m.attr("items_per_work_item") =
        py::int_(dpnp::kernels::rng::items_per_work_item);

    rng_ns::init_uniform(m);
    rng_ns::init_normal(m);
    rng_ns::init_exponential(m);
    rng_ns::init_gamma(m);
    rng_ns::init_noncentral(m);
    rng_ns::init_integers(m);
    rng_ns::init_poisson(m);
    rng_ns::init_binomial(m);
    rng_ns::init_discrete(m);
}
