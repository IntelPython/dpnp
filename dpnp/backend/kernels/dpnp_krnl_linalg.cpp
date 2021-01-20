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
namespace mkl_lapack = oneapi::mkl::lapack;

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
    (void)ndim; // avoid warning unused variable
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

template <typename _DataType1, typename _DataType2, typename _ResultType>
class dpnp_kron_c_kernel;

template <typename _DataType1, typename _DataType2, typename _ResultType>
void dpnp_kron_c(void* array1_in, void* array2_in, void* result1, size_t* in1_shape, size_t* in2_shape, size_t* res_shape, size_t ndim)
{
    _DataType1* array1 = reinterpret_cast<_DataType1*>(array1_in);
    _DataType2* array2 = reinterpret_cast<_DataType2*>(array2_in);
    _ResultType* result = reinterpret_cast<_ResultType*>(result1);

    size_t size = 1;
    for (size_t i = 0; i < ndim; ++i)
    {
        size *= res_shape[i];
    }

    size_t* _in1_shape = reinterpret_cast<size_t*>(dpnp_memory_alloc_c(ndim * sizeof(size_t)));
    size_t* _in2_shape = reinterpret_cast<size_t*>(dpnp_memory_alloc_c(ndim * sizeof(size_t)));

    dpnp_memory_memcpy_c(_in1_shape, in1_shape, ndim * sizeof(size_t));
    dpnp_memory_memcpy_c(_in2_shape, in2_shape, ndim * sizeof(size_t));

    size_t* in1_offsets = reinterpret_cast<size_t*>(dpnp_memory_alloc_c(ndim * sizeof(size_t)));
    size_t* in2_offsets = reinterpret_cast<size_t*>(dpnp_memory_alloc_c(ndim * sizeof(size_t)));
    size_t* res_offsets = reinterpret_cast<size_t*>(dpnp_memory_alloc_c(ndim * sizeof(size_t)));

    get_shape_offsets_inkernel<size_t>(in1_shape, ndim, in1_offsets);
    get_shape_offsets_inkernel<size_t>(in2_shape, ndim, in2_offsets);
    get_shape_offsets_inkernel<size_t>(res_shape, ndim, res_offsets);

    cl::sycl::range<1> gws(size);
    auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {
        const size_t idx = global_id[0];

        size_t idx1 = 0;
        size_t idx2 = 0;
        size_t reminder = idx;
        for (size_t axis = 0; axis < ndim; ++axis)
        {
            const size_t res_axis = reminder / res_offsets[axis];
            reminder = reminder - res_axis * res_offsets[axis];

            const size_t in1_axis = res_axis / _in2_shape[axis];
            const size_t in2_axis = res_axis - in1_axis * _in2_shape[axis];

            idx1 += in1_axis * in1_offsets[axis];
            idx2 += in2_axis * in2_offsets[axis];
        }

        result[idx] = array1[idx1] * array2[idx2];
    };

    auto kernel_func = [&](cl::sycl::handler& cgh) {
        cgh.parallel_for<class dpnp_kron_c_kernel<_DataType1, _DataType2, _ResultType>>(gws, kernel_parallel_for_func);
    };

    cl::sycl::event event = DPNP_QUEUE.submit(kernel_func);

    event.wait();
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

template <typename _InputDT, typename _ComputeDT, typename _SVDT>
void dpnp_svd_c(void* array1_in, void* result1, void* result2, void* result3, size_t size_m, size_t size_n)
{
    cl::sycl::event event;

    _InputDT* in_array = reinterpret_cast<_InputDT*>(array1_in);

    // math lib gesvd func overrides input
    _ComputeDT* in_a = reinterpret_cast<_ComputeDT*>(dpnp_memory_alloc_c(size_m * size_n * sizeof(_ComputeDT)));
    for (size_t it = 0; it < size_m * size_n; ++it)
    {
        in_a[it] = in_array[it];
    }

    _ComputeDT* res_u = reinterpret_cast<_ComputeDT*>(result1);
    _SVDT* res_s = reinterpret_cast<_SVDT*>(result2);
    _ComputeDT* res_vt = reinterpret_cast<_ComputeDT*>(result3);

    const std::int64_t m = size_m;
    const std::int64_t n = size_n;

    const std::int64_t lda = std::max<size_t>(1UL, n);
    const std::int64_t ldu = std::max<size_t>(1UL, m);
    const std::int64_t ldvt = std::max<size_t>(1UL, n);

    const std::int64_t scratchpad_size1 = mkl_lapack::gesvd_scratchpad_size<_ComputeDT>(
        DPNP_QUEUE, oneapi::mkl::jobsvd::vectors, oneapi::mkl::jobsvd::vectors, n, m, lda, ldvt, ldu);

    const std::int64_t scratchpad_size = scratchpad_size1;

    _ComputeDT* scratchpad = reinterpret_cast<_ComputeDT*>(dpnp_memory_alloc_c(scratchpad_size * sizeof(_ComputeDT)));

    event = mkl_lapack::gesvd(DPNP_QUEUE,
                              oneapi::mkl::jobsvd::vectors, // onemkl::job jobu,
                              oneapi::mkl::jobsvd::vectors, // onemkl::job jobvt,
                              n,
                              m,
                              in_a,
                              lda,
                              res_s,
                              res_vt,
                              ldvt,
                              res_u,
                              ldu,
                              scratchpad,
                              scratchpad_size);

    event.wait();

    dpnp_memory_free_c(scratchpad);
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

    fmap[DPNPFuncName::DPNP_FN_KRON][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_kron_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_kron_c<int, long, long>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_INT][eft_FLT] = {eft_FLT, (void*)dpnp_kron_c<int, float, float>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_kron_c<int, double, double>};
    // fmap[DPNPFuncName::DPNP_FN_KRON][eft_INT][eft_C128] = {
        // eft_C128, (void*)dpnp_kron_c<int, std::complex<double>, std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_kron_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_kron_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_LNG][eft_FLT] = {eft_FLT, (void*)dpnp_kron_c<long, float, float>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_kron_c<long, double, double>};
    // fmap[DPNPFuncName::DPNP_FN_KRON][eft_LNG][eft_C128] = {
        // eft_C128, (void*)dpnp_kron_c<long, std::complex<double>, std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_FLT][eft_INT] = {eft_FLT, (void*)dpnp_kron_c<float, int, float>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_FLT][eft_LNG] = {eft_FLT, (void*)dpnp_kron_c<float, long, float>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_kron_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_kron_c<float, double, double>};
    // fmap[DPNPFuncName::DPNP_FN_KRON][eft_FLT][eft_C128] = {
        // eft_C128, (void*)dpnp_kron_c<float, std::complex<double>, std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_kron_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_kron_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_kron_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_kron_c<double, double, double>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_DBL][eft_C128] = {
        eft_C128, (void*)dpnp_kron_c<double, std::complex<double>, std::complex<double>>};
    // fmap[DPNPFuncName::DPNP_FN_KRON][eft_C128][eft_INT] = {
        // eft_C128, (void*)dpnp_kron_c<std::complex<double>, int, std::complex<double>>};
    // fmap[DPNPFuncName::DPNP_FN_KRON][eft_C128][eft_LNG] = {
        // eft_C128, (void*)dpnp_kron_c<std::complex<double>, long, std::complex<double>>};
    // fmap[DPNPFuncName::DPNP_FN_KRON][eft_C128][eft_FLT] = {
        // eft_C128, (void*)dpnp_kron_c<std::complex<double>, float, std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_C128][eft_DBL] = {
        eft_C128, (void*)dpnp_kron_c<std::complex<double>, double, std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_C128][eft_C128] = {
        eft_C128, (void*)dpnp_kron_c<std::complex<double>, std::complex<double>, std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_MATRIX_RANK][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_matrix_rank_c<int>};
    fmap[DPNPFuncName::DPNP_FN_MATRIX_RANK][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_matrix_rank_c<long>};
    fmap[DPNPFuncName::DPNP_FN_MATRIX_RANK][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_matrix_rank_c<float>};
    fmap[DPNPFuncName::DPNP_FN_MATRIX_RANK][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_matrix_rank_c<double>};

    fmap[DPNPFuncName::DPNP_FN_SVD][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_svd_c<int, double, double>};
    fmap[DPNPFuncName::DPNP_FN_SVD][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_svd_c<long, double, double>};
    fmap[DPNPFuncName::DPNP_FN_SVD][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_svd_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_SVD][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_svd_c<double, double, double>};
    fmap[DPNPFuncName::DPNP_FN_SVD][eft_C128][eft_C128] = {
        eft_C128, (void*)dpnp_svd_c<std::complex<double>, std::complex<double>, double>};

    return;
}
