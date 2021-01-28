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
#include <list>

#include <dpnp_iface.hpp>
#include "dpnp_fptr.hpp"
#include "dpnp_utils.hpp"
#include "queue_sycl.hpp"


template <typename _DataType>
class dpnp_take_c_kernel;

template <typename _DataType>
void dpnp_take_c(void* array1_in, void* indices1, void* result1, size_t size)
{
    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);
    size_t* indices = reinterpret_cast<size_t*>(indices1);

    for (size_t i = 0; i < size; i++)
    {
        size_t ind = indices[i];
        result[i] = array_1[ind];
    }

    return;
}


void func_map_init_indexing_func(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_TAKE][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_take_c<int>};
    fmap[DPNPFuncName::DPNP_FN_TAKE][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_take_c<long>};
    fmap[DPNPFuncName::DPNP_FN_TAKE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_take_c<float>};
    fmap[DPNPFuncName::DPNP_FN_TAKE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_take_c<double>};

    return;
}
