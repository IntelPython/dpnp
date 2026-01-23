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

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#if __has_include(<dpnp4pybind11.hpp>)
#include "dpnp4pybind11.hpp"
#else
#include "dpctl4pybind11.hpp"
#endif

// dpctl tensor headers
#include "utils/type_dispatch.hpp"

namespace dpnp::extensions::py_internal::type_utils
{
namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

/*! @brief Produce dtype from a type number */
extern py::dtype _dtype_from_typenum(td_ns::typenum_t);

/*! @brief Lookup typeid of the result from typeid of
 *         argument and the mapping table */
template <typename output_idT>
output_idT _result_typeid(int arg_typeid, const output_idT *fn_output_id)
{
    if (arg_typeid < 0 || arg_typeid >= td_ns::num_types) {
        throw py::value_error("Input typeid " + std::to_string(arg_typeid) +
                              " is outside of expected bounds.");
    }

    return fn_output_id[arg_typeid];
}
} // namespace dpnp::extensions::py_internal::type_utils
