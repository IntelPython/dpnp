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

#include <string>
#include <vector>

#include <pybind11/pybind11.h>

#include "dpnp4pybind11.hpp"

// utils extension header
#include "ext/validation_utils.hpp"

// dpctl tensor headers
#include "utils/type_dispatch.hpp"

#include "sliding_window1d.hpp"

namespace dpctl_td_ns = dpctl::tensor::type_dispatch;
using dpctl::tensor::usm_ndarray;
using dpctl_td_ns::typenum_t;

using ext::validation::array_names;
using ext::validation::array_ptr;
using ext::validation::check_num_dims;
using ext::validation::common_checks;
using ext::validation::name_of;

namespace statistics::sliding_window1d
{
void validate(const usm_ndarray &a,
              const usm_ndarray &v,
              const usm_ndarray &out,
              const size_t l_pad,
              const size_t r_pad)
{
    std::vector<array_ptr> inputs = {&a, &v};
    std::vector<array_ptr> outputs = {&out};

    array_names names;
    names[&a] = "a";
    names[&v] = "v";
    names[&out] = "out";

    common_checks(inputs, outputs, names);

    check_num_dims(&a, 1, names);
    check_num_dims(&v, 1, names);
    check_num_dims(&out, 1, names);

    py::ssize_t padded_a_size = l_pad + r_pad + a.get_size();

    if (v.get_size() > padded_a_size) {
        throw py::value_error(name_of(&v, names) + " size (" +
                              std::to_string(v.get_size()) +
                              ") must be less than or equal to a size (" +
                              std::to_string(a.get_size()) + ") + l_pad (" +
                              std::to_string(l_pad) + ") + r_pad (" +
                              std::to_string(r_pad) + ")");
    }

    auto expected_output_size = padded_a_size - v.get_size() + 1;
    if (out.get_size() != expected_output_size) {
        throw py::value_error(
            name_of(&out, names) + " size (" + std::to_string(out.get_size()) +
            ") must be equal to " + name_of(&a, names) +
            " size + l_pad + r_pad " + name_of(&v, names) + " size + 1 (" +
            std::to_string(expected_output_size) + ")");
    }
}
} // namespace statistics::sliding_window1d
