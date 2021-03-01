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

#include "dpnp_fptr.hpp"
#include "dpnp_iface.hpp"
#include "queue_sycl.hpp"

template <typename _KernelNameSpecialization>
class dpnp_arange_c_kernel;

template <typename _DataType>
void dpnp_arange_c(size_t start, size_t step, void* result1, size_t size)
{
    // parameter `size` used instead `stop` to avoid dependency on array length calculation algorithm
    // TODO: floating point (and negatives) types from `start` and `step`

    if (!size)
    {
        return;
    }

    cl::sycl::event event;

    _DataType* result = reinterpret_cast<_DataType*>(result1);

    cl::sycl::range<1> gws(size);
    auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {
        size_t i = global_id[0];

        result[i] = start + i * step;
    };

    auto kernel_func = [&](cl::sycl::handler& cgh) {
        cgh.parallel_for<class dpnp_arange_c_kernel<_DataType>>(gws, kernel_parallel_for_func);
    };

    event = DPNP_QUEUE.submit(kernel_func);

    event.wait();
}

template <typename _DataType>
void dpnp_diag_c(
    void* v_in, void* result1, const int k, size_t* shape, size_t* res_shape, const size_t ndim, const size_t res_ndim)
{
    _DataType* v = reinterpret_cast<_DataType*>(v_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    size_t init0 = std::max(0, -k);
    size_t init1 = std::max(0, k);

    if (ndim == 1)
    {
        for (size_t i = 0; i < shape[0]; ++i)
        {
            size_t ind = (init0 + i) * res_shape[1] + init1 + i;
            result[ind] = v[i];
        }
    }
    else
    {
        for (size_t i = 0; i < res_shape[0]; ++i)
        {
            size_t ind = (init0 + i) * shape[1] + init1 + i;
            result[i] = v[ind];
        }
    }
    return;
}

template <typename _KernelNameSpecialization>
class dpnp_full_c_kernel;

template <typename _DataType>
void dpnp_full_c(void* array_in, void* result, const size_t size)
{
    dpnp_initval_c<_DataType>(result, array_in, size);
}

template <typename _DataType>
void dpnp_tril_c(void* array_in,
                 void* result1,
                 const int k,
                 size_t* shape,
                 size_t* res_shape,
                 const size_t ndim,
                 const size_t res_ndim)
{
    _DataType* array_m = reinterpret_cast<_DataType*>(array_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    size_t res_size = 1;
    for (size_t i = 0; i < res_ndim; ++i)
    {
        res_size *= res_shape[i];
    }

    if (ndim == 1)
    {
        for (size_t i = 0; i < res_size; ++i)
        {
            size_t n = res_size;
            size_t val = i;
            int ids[res_ndim];
            for (size_t j = 0; j < res_ndim; ++j)
            {
                n /= res_shape[j];
                size_t p = val / n;
                ids[j] = p;
                if (p != 0)
                {
                    val = val - p * n;
                }
            }

            int diag_idx_ = (ids[res_ndim - 2] + k > -1) ? (ids[res_ndim - 2] + k) : -1;
            int values = res_shape[res_ndim - 1];
            int diag_idx = (values < diag_idx_) ? values : diag_idx_;

            if (ids[res_ndim - 1] <= diag_idx)
            {
                result[i] = array_m[ids[res_ndim - 1]];
            }
            else
            {
                result[i] = 0;
            }
        }
    }
    else
    {
        for (size_t i = 0; i < res_size; ++i)
        {
            size_t n = res_size;
            size_t val = i;
            int ids[res_ndim];
            for (size_t j = 0; j < res_ndim; ++j)
            {
                n /= res_shape[j];
                size_t p = val / n;
                ids[j] = p;
                if (p != 0)
                {
                    val = val - p * n;
                }
            }

            int diag_idx_ = (ids[res_ndim - 2] + k > -1) ? (ids[res_ndim - 2] + k) : -1;
            int values = res_shape[res_ndim - 1];
            int diag_idx = (values < diag_idx_) ? values : diag_idx_;

            if (ids[res_ndim - 1] <= diag_idx)
            {
                result[i] = array_m[i];
            }
            else
            {
                result[i] = 0;
            }
        }
    }
    return;
}

template <typename _DataType>
void dpnp_triu_c(void* array_in, void* result1, const int k, size_t* shape, size_t* res_shape, const size_t ndim, const size_t res_ndim)
{
    _DataType* array_m = reinterpret_cast<_DataType*>(array_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    size_t res_size = 1;
    for (size_t i = 0; i < res_ndim; ++i)
    {
        res_size *= res_shape[i];
    }

    if (ndim == 1)
    {
        for (size_t i = 0; i < res_size; ++i)
        {
            size_t n = res_size;
            size_t val = i;
            int ids[res_ndim];
            for (size_t j = 0; j < res_ndim; ++j)
            {
                n /= res_shape[j];
                size_t p = val / n;
                ids[j] = p;
                if (p != 0)
                {
                    val = val - p * n;
                }
            }

            int diag_idx_ = (ids[res_ndim - 2] + k > -1) ? (ids[res_ndim - 2] + k) : -1;
            int values = res_shape[res_ndim - 1];
            int diag_idx = (values < diag_idx_) ? values : diag_idx_;

            if (ids[res_ndim - 1] >= diag_idx)
            {
                result[i] = array_m[ids[res_ndim - 1]];
            }
            else
            {
                result[i] = 0;
            }
        }
    }
    else
    {
        for (size_t i = 0; i < res_size; ++i)
        {
            size_t n = res_size;
            size_t val = i;
            int ids[res_ndim];
            for (size_t j = 0; j < res_ndim; ++j)
            {
                n /= res_shape[j];
                size_t p = val / n;
                ids[j] = p;
                if (p != 0)
                {
                    val = val - p * n;
                }
            }

            int diag_idx_ = (ids[res_ndim - 2] + k > -1) ? (ids[res_ndim - 2] + k) : -1;
            int values = res_shape[res_ndim - 1];
            int diag_idx = (values < diag_idx_) ? values : diag_idx_;

            if (ids[res_ndim - 1] >= diag_idx)
            {
                result[i] = array_m[i];
            }
            else
            {
                result[i] = 0;
            }
        }
    }
    return;
}

void func_map_init_arraycreation(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_ARANGE][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_arange_c<int>};
    fmap[DPNPFuncName::DPNP_FN_ARANGE][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_arange_c<long>};
    fmap[DPNPFuncName::DPNP_FN_ARANGE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_arange_c<float>};
    fmap[DPNPFuncName::DPNP_FN_ARANGE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_arange_c<double>};

    fmap[DPNPFuncName::DPNP_FN_DIAG][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_diag_c<int>};
    fmap[DPNPFuncName::DPNP_FN_DIAG][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_diag_c<long>};
    fmap[DPNPFuncName::DPNP_FN_DIAG][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_diag_c<float>};
    fmap[DPNPFuncName::DPNP_FN_DIAG][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_diag_c<double>};

    fmap[DPNPFuncName::DPNP_FN_FULL][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_full_c<int>};
    fmap[DPNPFuncName::DPNP_FN_FULL][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_full_c<long>};
    fmap[DPNPFuncName::DPNP_FN_FULL][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_full_c<float>};
    fmap[DPNPFuncName::DPNP_FN_FULL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_full_c<double>};

    fmap[DPNPFuncName::DPNP_FN_TRIL][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_tril_c<int>};
    fmap[DPNPFuncName::DPNP_FN_TRIL][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_tril_c<long>};
    fmap[DPNPFuncName::DPNP_FN_TRIL][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_tril_c<float>};
    fmap[DPNPFuncName::DPNP_FN_TRIL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_tril_c<double>};

    fmap[DPNPFuncName::DPNP_FN_TRIU][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_triu_c<int>};
    fmap[DPNPFuncName::DPNP_FN_TRIU][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_triu_c<long>};
    fmap[DPNPFuncName::DPNP_FN_TRIU][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_triu_c<float>};
    fmap[DPNPFuncName::DPNP_FN_TRIU][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_triu_c<double>};

    return;
}
