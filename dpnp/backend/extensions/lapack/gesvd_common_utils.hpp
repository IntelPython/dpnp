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
#include <oneapi/mkl.hpp>

namespace dpnp::extensions::lapack::gesvd_utils
{
inline void handle_lapack_exc(std::int64_t scratchpad_size,
                              const oneapi::mkl::lapack::exception &e,
                              std::stringstream &error_msg)
{
    std::int64_t info = e.info();
    if (info < 0) {
        error_msg << "Parameter number " << -info << " had an illegal value.";
    }
    else if (info == scratchpad_size && e.detail() != 0) {
        error_msg << "Insufficient scratchpad size. Required size is at least "
                  << e.detail();
    }
    else if (info > 0) {
        error_msg << "The algorithm computing SVD failed to converge; " << info
                  << " off-diagonal elements of an intermediate "
                  << "bidiagonal form did not converge to zero.\n";
    }
    else {
        error_msg
            << "Unexpected MKL exception caught during gesv() call:\nreason: "
            << e.what() << "\ninfo: " << e.info();
    }
}
} // namespace dpnp::extensions::lapack::gesvd_utils
