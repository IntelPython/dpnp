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

namespace mkl_blas = oneapi::mkl::blas::row_major;

template <typename _DataType>
class dpnp_cholesky_c_kernel;

template <typename _DataType>
void dpnp_cholesky_c(void* array1_in, void* result1, size_t* shape)
{
    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* l_result = reinterpret_cast<_DataType*>(result1);

    size_t n = shape[0];

    l_result[0] = sqrt(array_1[0]);

    for (size_t j = 1; j < n; j++)
    {
        l_result[j * n] = array_1[j * n] / l_result[0];
    }

    for (size_t i = 1; i < n; i++)
    {
        _DataType sum_val = 0;
        for (size_t p = 0; p < i - 1; p++)
        {
            sum_val += l_result[i * n + p - 1] * l_result[i * n + p - 1];
        }
        l_result[i * n + i - 1] = sqrt(array_1[i * n + i - 1] - sum_val);
    }

    for (size_t i = 1; i < n - 1; i++)
    {
        for (size_t j = i; j < n; j++)
        {
            _DataType sum_val = 0;
            for (size_t p = 0; p < i - 1; p++)
            {
                sum_val += l_result[i * n + p - 1] * l_result[j * n + p - 1];
            }
            l_result[j * n + i - 1] = (1 / l_result[i * n + i - 1]) * (array_1[j * n + i - 1] - sum_val);
        }
    }
    return;
}

template <typename _DataType>
class dpnp_det_c_kernel;

template <typename _DataType>
void dpnp_det_c(void* array1_in, void* result1, size_t* shape, size_t ndim)
{
    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    size_t n = shape[ndim - 1];
    size_t size_out = 1;
    if (ndim != 2)
    {
        for (size_t i = 0; i < ndim - 2; i++)
        {
            size_out *= shape[i];
        }
    }

    for (size_t i = 0; i < size_out; i++)
    {
        _DataType matrix[n][n];
        if (size_out > 1)
        {
            _DataType elems[n * n];
            for (size_t j = i * n * n; j < (i + 1) * n * n; j++)
            {
                elems[j - i * n * n] = array_1[j];
            }

            for (size_t j = 0; j < n; j++)
            {
                for (size_t k = 0; k < n; k++)
                {
                    matrix[j][k] = elems[j * n + k];
                }
            }
        }
        else
        {
            for (size_t j = 0; j < n; j++)
            {
                for (size_t k = 0; k < n; k++)
                {
                    matrix[j][k] = array_1[j * n + k];
                }
            }
        }

        _DataType det_val = 1;
        for (size_t l = 0; l < n; l++)
        {
            if (matrix[l][l] == 0)
            {
                for (size_t j = l; j < n; j++)
                {
                    if (matrix[j][l] != 0)
                    {
                        for (size_t k = l; k < n; k++)
                        {
                            _DataType c = matrix[l][k];
                            matrix[l][k] = -1 * matrix[j][k];
                            matrix[j][k] = c;
                        }
                        break;
                    }
                    if (j == n - 1 and matrix[j][l] == 0)
                    {
                        det_val = 0;
                    }
                }
            }
            if (det_val != 0)
            {
                for (size_t j = l + 1; j < n; j++)
                {
                    _DataType q = -(matrix[j][l] / matrix[l][l]);
                    for (size_t k = l + 1; k < n; k++)
                    {
                        matrix[j][k] += q * matrix[l][k];
                    }
                }
            }
        }

        if (det_val != 0)
        {
            for (size_t l = 0; l < n; l++)
            {
                det_val *= matrix[l][l];
            }
        }

        result[i] = det_val;
    }

    return;
}

template <typename _DataType>
class dpnp_inv_c_kernel;

template <typename _DataType>
void dpnp_inv_c(void* array1_in, void* result1, size_t* shape, size_t ndim)
{
    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    size_t n = shape[0];

    _DataType a_arr[n][n];
    _DataType e_arr[n][n];

    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            a_arr[i][j] = array_1[i * n + j];
            if (i == j)
            {
                e_arr[i][j] = 1;
            }
            else
            {
                e_arr[i][j] = 0;
            }
        }
    }

    for (size_t k = 0; k < n; ++k)
    {
        if (a_arr[k][k] == 0)
        {
            for (size_t i = k; i < n; ++i)
            {
                if (a_arr[i][k] != 0)
                {
                    for (size_t j = 0; j < n; ++j)
                    {
                        float c = a_arr[k][j];
                        a_arr[k][j] = a_arr[i][j];
                        a_arr[i][j] = c;
                        float c_e = e_arr[k][j];
                        e_arr[k][j] = e_arr[i][j];
                        e_arr[i][j] = c_e;
                    }
                    break;
                }
            }
        }

        float temp = a_arr[k][k];

        for (size_t j = 0; j < n; ++j)
        {
            a_arr[k][j] = a_arr[k][j] / temp;
            e_arr[k][j] = e_arr[k][j] / temp;
        }

        for (size_t i = k + 1; i < n; ++i)
        {
            temp = a_arr[i][k];
            for (size_t j = 0; j < n; j++)
            {
                a_arr[i][j] = a_arr[i][j] - a_arr[k][j] * temp;
                e_arr[i][j] = e_arr[i][j] - e_arr[k][j] * temp;
            }
        }
    }

    for (size_t k = 0; k < n - 1; ++k)
    {
        size_t ind_k = n - 1 - k;
        for (size_t i = 0; i < ind_k; ++i)
        {
            size_t ind_i = ind_k - 1 - i;

            float temp = a_arr[ind_i][ind_k];
            for (size_t j = 0; j < n; ++j)
            {
                a_arr[ind_i][j] = a_arr[ind_i][j] - a_arr[ind_k][j] * temp;
                e_arr[ind_i][j] = e_arr[ind_i][j] - e_arr[ind_k][j] * temp;
            }
        }
    }

    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            result[i * n + j] = e_arr[i][j];
        }
    }

    return;
}

template <typename _DataType>
class dpnp_matrix_rank_c_kernel;

template <typename _DataType>
void dpnp_matrix_rank_c(void* array1_in, void* result1, size_t* shape, size_t ndim)
{
    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    size_t elems = 1;
    result[0] = 0;
    if (ndim > 1)
    {
        elems = shape[0];
        for (size_t i = 1; i < ndim; i++)
        {
            if (shape[i] < elems)
            {
                elems = shape[i];
            }
        }
    }
    for (size_t i = 0; i < elems; i++)
    {
        size_t ind = 0;
        for (size_t j = 0; j < ndim; j++)
        {
            ind += (shape[j] - 1) * i;
        }
        result[0] += array_1[ind];
    }

    return;
}

void func_map_init_linalg_func(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_CHOLESKY][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_cholesky_c<int>};
    fmap[DPNPFuncName::DPNP_FN_CHOLESKY][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_cholesky_c<long>};
    fmap[DPNPFuncName::DPNP_FN_CHOLESKY][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_cholesky_c<float>};
    fmap[DPNPFuncName::DPNP_FN_CHOLESKY][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_cholesky_c<double>};

    fmap[DPNPFuncName::DPNP_FN_DET][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_det_c<int>};
    fmap[DPNPFuncName::DPNP_FN_DET][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_det_c<long>};
    fmap[DPNPFuncName::DPNP_FN_DET][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_det_c<float>};
    fmap[DPNPFuncName::DPNP_FN_DET][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_det_c<double>};

    fmap[DPNPFuncName::DPNP_FN_INV][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_inv_c<int>};
    fmap[DPNPFuncName::DPNP_FN_INV][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_inv_c<long>};
    fmap[DPNPFuncName::DPNP_FN_INV][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_inv_c<float>};
    fmap[DPNPFuncName::DPNP_FN_INV][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_inv_c<double>};

    fmap[DPNPFuncName::DPNP_FN_MATRIX_RANK][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_matrix_rank_c<int>};
    fmap[DPNPFuncName::DPNP_FN_MATRIX_RANK][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_matrix_rank_c<long>};
    fmap[DPNPFuncName::DPNP_FN_MATRIX_RANK][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_matrix_rank_c<float>};
    fmap[DPNPFuncName::DPNP_FN_MATRIX_RANK][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_matrix_rank_c<double>};

    return;
}
