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
//===---------------------------------------------------------------------===//
///
/// \file
/// This file defines functions of dpctl.tensor._tensor_reductions_impl
/// extension.
//===---------------------------------------------------------------------===//

#include <pybind11/pybind11.h>

#include "all.hpp"
#include "any.hpp"
#include "argmax.hpp"
#include "argmin.hpp"
#include "logsumexp.hpp"
#include "max.hpp"
#include "min.hpp"
#include "prod.hpp"
#include "reduce_hypot.hpp"
#include "sum.hpp"

namespace py = pybind11;

namespace dpctl::tensor::py_internal
{

/*! @brief Add reduction functions to Python module */
void init_reduction_functions(py::module_ m)
{
    init_all(m);
    init_any(m);
    init_argmax(m);
    init_argmin(m);
    init_logsumexp(m);
    init_max(m);
    init_min(m);
    init_prod(m);
    init_reduce_hypot(m);
    init_sum(m);
}

} // namespace dpctl::tensor::py_internal
