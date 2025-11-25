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

#include "bitwise_count.hpp"
#include "degrees.hpp"
#include "divmod.hpp"
#include "erf_funcs.hpp"
#include "fabs.hpp"
#include "fix.hpp"
#include "float_power.hpp"
#include "fmax.hpp"
#include "fmin.hpp"
#include "fmod.hpp"
#include "frexp.hpp"
#include "gcd.hpp"
#include "heaviside.hpp"
#include "i0.hpp"
#include "interpolate.hpp"
#include "isclose.hpp"
#include "lcm.hpp"
#include "ldexp.hpp"
#include "logaddexp2.hpp"
#include "modf.hpp"
#include "nan_to_num.hpp"
#include "radians.hpp"
#include "sinc.hpp"
#include "spacing.hpp"

namespace py = pybind11;

namespace dpnp::extensions::ufunc
{
/**
 * @brief Add elementwise functions to Python module
 */
void init_elementwise_functions(py::module_ m)
{
    init_bitwise_count(m);
    init_degrees(m);
    init_divmod(m);
    init_erf_funcs(m);
    init_fabs(m);
    init_fix(m);
    init_float_power(m);
    init_fmax(m);
    init_fmin(m);
    init_fmod(m);
    init_frexp(m);
    init_gcd(m);
    init_heaviside(m);
    init_i0(m);
    init_interpolate(m);
    init_isclose(m);
    init_lcm(m);
    init_ldexp(m);
    init_logaddexp2(m);
    init_modf(m);
    init_nan_to_num(m);
    init_radians(m);
    init_sinc(m);
    init_spacing(m);
}
} // namespace dpnp::extensions::ufunc
