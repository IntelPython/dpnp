//*****************************************************************************
// Copyright (c) 2024-2025, Intel Corporation
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

#include <string>
#include <vector>

#include "dpctl4pybind11.hpp"
#include "utils/type_dispatch.hpp"
#include <pybind11/pybind11.h>

#include "ext/common.hpp"
#include "ext/validation_utils.hpp"
#include "sliding_window1d.hpp"

namespace dpctl_td_ns = dpctl::tensor::type_dispatch;
using namespace ext::common;
using namespace ext::validation;

using dpctl::tensor::usm_ndarray;
using dpctl_td_ns::typenum_t;

namespace statistics::partitioning
{

void validate(const usm_ndarray &a,
              const usm_ndarray &partitioned,
              const size_t k)
{
    array_names names = {{&a, "a"}, {&partitioned, "partitioned"}};

    common_checks({&a}, {&partitioned}, names);
    check_same_size(&a, &partitioned, names);
    check_num_dims(&a, 1, names);
    check_num_dims(&partitioned, 1, names);
    check_same_dtype(&a, &partitioned, names);

    if (k > a.get_size() - 2) {
        throw py::value_error("'k' must be from 0 to a.size() - 2, "
                              "but got k = " +
                              std::to_string(k) + " and a.size() = " +
                              std::to_string(a.get_size()));
    }
}

} // namespace statistics::partitioning
