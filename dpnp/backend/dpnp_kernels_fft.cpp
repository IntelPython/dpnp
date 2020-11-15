//*****************************************************************************
// Copyright (c) 2016-2020, Intel Corporation
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

#include <iostream>

#include <backend_iface.hpp>
#include "backend_fptr.hpp"
#include "backend_utils.hpp"
#include "queue_sycl.hpp"

namespace mkl_dft = oneapi::mkl::dft;

template <typename _DataType_input, typename _DataType_output>
void dpnp_fft_fft_c(void* array1_in, void* result1, size_t size)
{
    if (!size)
    {
        return;
    }

    cl::sycl::event event;

    _DataType_input* array_1 = reinterpret_cast<_DataType_input*>(array1_in);
    _DataType_output* result = reinterpret_cast<_DataType_output*>(result1);

    oneapi::mkl::dft::descriptor<mkl_dft::precision::DOUBLE, mkl_dft::domain::COMPLEX> desc(size);
    desc.set_value(mkl_dft::config_param::FORWARD_SCALE, static_cast<double>(size));
    desc.set_value(mkl_dft::config_param::PLACEMENT, DFTI_NOT_INPLACE); // enum value from MKL C interface
    desc.commit(DPNP_QUEUE);

    event = mkl_dft::compute_forward(desc, array_1, result);
    event.wait();

    return;
}

void func_map_init_fft_func(func_map_t& fmap)
{
    // fmap[DPNPFuncName::DPNP_FN_FFT_FFT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_fft_fft_c<float, std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_FFT_FFT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_fft_fft_c<double, double>};

    return;
}
