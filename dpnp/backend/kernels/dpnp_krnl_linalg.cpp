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
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"

namespace mkl_blas = oneapi::mkl::blas::row_major;
namespace mkl_lapack = oneapi::mkl::lapack;

template <typename _DataType>
void dpnp_cholesky_c(void* array1_in, void* result1, const size_t size, const size_t data_size)
{
    cl::sycl::event event;

    DPNPC_ptr_adapter<_DataType> input1_ptr(array1_in, size, true);
    DPNPC_ptr_adapter<_DataType> result_ptr(result1, size, true, true);
    _DataType* in_array = input1_ptr.get_ptr();
    _DataType* result = result_ptr.get_ptr();

    size_t iters = size / (data_size * data_size);

    // math lib func overrides input
    _DataType* in_a = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(data_size * data_size * sizeof(_DataType)));

    for (size_t k = 0; k < iters; ++k)
    {
        for (size_t it = 0; it < data_size * data_size; ++it)
        {
            in_a[it] = in_array[k * (data_size * data_size) + it];
        }

        const std::int64_t n = data_size;

        const std::int64_t lda = std::max<size_t>(1UL, n);

        const std::int64_t scratchpad_size =
            mkl_lapack::potrf_scratchpad_size<_DataType>(DPNP_QUEUE, oneapi::mkl::uplo::upper, n, lda);

        _DataType* scratchpad = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(scratchpad_size * sizeof(_DataType)));

        event = mkl_lapack::potrf(DPNP_QUEUE, oneapi::mkl::uplo::upper, n, in_a, lda, scratchpad, scratchpad_size);

        event.wait();

        for (size_t i = 0; i < data_size; i++)
        {
            bool arg = false;
            for (size_t j = 0; j < data_size; j++)
            {
                if (i == j - 1)
                {
                    arg = true;
                }
                if (arg)
                {
                    in_a[i * data_size + j] = 0;
                }
            }
        }

        dpnp_memory_free_c(scratchpad);

        for (size_t t = 0; t < data_size * data_size; ++t)
        {
            result[k * (data_size * data_size) + t] = in_a[t];
        }
    }

    dpnp_memory_free_c(in_a);
}

template <typename _DataType>
void dpnp_det_c(void* array1_in, void* result1, shape_elem_type* shape, size_t ndim)
{
    const size_t input_size = std::accumulate(shape, shape + ndim, 1, std::multiplies<shape_elem_type>());
    if (!input_size)
    {
        return;
    }

    size_t n = shape[ndim - 1];
    size_t size_out = 1;
    if (ndim != 2)
    {
        for (size_t i = 0; i < ndim - 2; i++)
        {
            size_out *= shape[i];
        }
    }

    DPNPC_ptr_adapter<_DataType> input1_ptr(array1_in, input_size, true);
    DPNPC_ptr_adapter<_DataType> result_ptr(result1, size_out, true, true);
    _DataType* array_1 = input1_ptr.get_ptr();
    _DataType* result = result_ptr.get_ptr();

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

template <typename _DataType, typename _ResultType>
void dpnp_inv_c(void* array1_in, void* result1, shape_elem_type* shape, size_t ndim)
{
    (void)ndim; // avoid warning unused variable

    const size_t input_size = std::accumulate(shape, shape + ndim, 1, std::multiplies<shape_elem_type>());
    if (!input_size)
    {
        return;
    }

    DPNPC_ptr_adapter<_DataType> input1_ptr(array1_in, input_size, true);
    DPNPC_ptr_adapter<_ResultType> result_ptr(result1, input_size, true, true);
    _DataType* array_1 = input1_ptr.get_ptr();
    _ResultType* result = result_ptr.get_ptr();

    size_t n = shape[0];

    _ResultType a_arr[n][n];
    _ResultType e_arr[n][n];

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
void dpnp_kron_c(void* array1_in,
                 void* array2_in,
                 void* result1,
                 shape_elem_type* in1_shape,
                 shape_elem_type* in2_shape,
                 shape_elem_type* res_shape,
                 size_t ndim)
{
    const size_t input1_size = std::accumulate(in1_shape, in1_shape + ndim, 1, std::multiplies<shape_elem_type>());
    const size_t input2_size = std::accumulate(in2_shape, in2_shape + ndim, 1, std::multiplies<shape_elem_type>());
    const size_t result_size = std::accumulate(res_shape, res_shape + ndim, 1, std::multiplies<shape_elem_type>());
    if (!(result_size && input1_size && input2_size))
    {
        return;
    }

    DPNPC_ptr_adapter<_DataType1> input1_ptr(array1_in, input1_size);
    DPNPC_ptr_adapter<_DataType2> input2_ptr(array2_in, input2_size);
    DPNPC_ptr_adapter<_ResultType> result_ptr(result1, result_size);
    _DataType1* array1 = input1_ptr.get_ptr();
    _DataType2* array2 = input2_ptr.get_ptr();
    _ResultType* result = result_ptr.get_ptr();

    shape_elem_type* _in1_shape =
        reinterpret_cast<shape_elem_type*>(dpnp_memory_alloc_c(ndim * sizeof(shape_elem_type)));
    shape_elem_type* _in2_shape =
        reinterpret_cast<shape_elem_type*>(dpnp_memory_alloc_c(ndim * sizeof(shape_elem_type)));

    dpnp_memory_memcpy_c(_in1_shape, in1_shape, ndim * sizeof(shape_elem_type));
    dpnp_memory_memcpy_c(_in2_shape, in2_shape, ndim * sizeof(shape_elem_type));

    shape_elem_type* in1_offsets =
        reinterpret_cast<shape_elem_type*>(dpnp_memory_alloc_c(ndim * sizeof(shape_elem_type)));
    shape_elem_type* in2_offsets =
        reinterpret_cast<shape_elem_type*>(dpnp_memory_alloc_c(ndim * sizeof(shape_elem_type)));
    shape_elem_type* res_offsets =
        reinterpret_cast<shape_elem_type*>(dpnp_memory_alloc_c(ndim * sizeof(shape_elem_type)));

    get_shape_offsets_inkernel(in1_shape, ndim, in1_offsets);
    get_shape_offsets_inkernel(in2_shape, ndim, in2_offsets);
    get_shape_offsets_inkernel(res_shape, ndim, res_offsets);

    cl::sycl::range<1> gws(result_size);
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
void dpnp_matrix_rank_c(void* array1_in, void* result1, shape_elem_type* shape, size_t ndim)
{
    const size_t input_size = std::accumulate(shape, shape + ndim, 1, std::multiplies<shape_elem_type>());
    if (!input_size)
    {
        return;
    }

    DPNPC_ptr_adapter<_DataType> input1_ptr(array1_in, input_size, true);
    DPNPC_ptr_adapter<_DataType> result_ptr(result1, 1, true, true);
    _DataType* array_1 = input1_ptr.get_ptr();
    _DataType* result = result_ptr.get_ptr();

    size_t elems = 1;
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

    _DataType acc = 0;
    for (size_t i = 0; i < elems; i++)
    {
        size_t ind = 0;
        for (size_t j = 0; j < ndim; j++)
        {
            ind += (shape[j] - 1) * i;
        }
        acc += array_1[ind];
    }
    result[0] = acc;

    return;
}

template <typename _InputDT, typename _ComputeDT>
void dpnp_qr_c(void* array1_in, void* result1, void* result2, void* result3, size_t size_m, size_t size_n)
{
    cl::sycl::event event;

    DPNPC_ptr_adapter<_InputDT> input1_ptr(array1_in, size_m * size_n, true);
    _InputDT* in_array = input1_ptr.get_ptr();

    // math lib func overrides input
    _ComputeDT* in_a = reinterpret_cast<_ComputeDT*>(dpnp_memory_alloc_c(size_m * size_n * sizeof(_ComputeDT)));

    for (size_t i = 0; i < size_m; ++i)
    {
        for (size_t j = 0; j < size_n; ++j)
        {
            // TODO transpose? use dpnp_transpose_c()
            in_a[j * size_m + i] = in_array[i * size_n + j];
        }
    }

    DPNPC_ptr_adapter<_ComputeDT> result1_ptr(result1, size_m * size_m, true, true);
    DPNPC_ptr_adapter<_ComputeDT> result2_ptr(result2, size_m * size_n, true, true);
    DPNPC_ptr_adapter<_ComputeDT> result3_ptr(result3, std::min(size_m, size_n), true, true);
    _ComputeDT* res_q = result1_ptr.get_ptr();
    _ComputeDT* res_r = result2_ptr.get_ptr();
    _ComputeDT* tau = result3_ptr.get_ptr();

    const std::int64_t lda = size_m;

    const std::int64_t geqrf_scratchpad_size =
        mkl_lapack::geqrf_scratchpad_size<_ComputeDT>(DPNP_QUEUE, size_m, size_n, lda);

    _ComputeDT* geqrf_scratchpad =
        reinterpret_cast<_ComputeDT*>(dpnp_memory_alloc_c(geqrf_scratchpad_size * sizeof(_ComputeDT)));

    std::vector<cl::sycl::event> depends(1);
    set_barrier_event(DPNP_QUEUE, depends);

    event =
        mkl_lapack::geqrf(DPNP_QUEUE, size_m, size_n, in_a, lda, tau, geqrf_scratchpad, geqrf_scratchpad_size, depends);

    event.wait();

    verbose_print("oneapi::mkl::lapack::geqrf", depends.front(), event);

    dpnp_memory_free_c(geqrf_scratchpad);

    // R
    for (size_t i = 0; i < size_m; ++i)
    {
        for (size_t j = 0; j < size_n; ++j)
        {
            if (j >= i)
            {
                res_r[i * size_n + j] = in_a[j * size_m + i];
            }
            else
            {
                res_r[i * size_n + j] = _ComputeDT(0);
            }
        }
    }

    // Q
    const size_t nrefl = std::min<size_t>(size_m, size_n);
    const std::int64_t orgqr_scratchpad_size =
        mkl_lapack::orgqr_scratchpad_size<_ComputeDT>(DPNP_QUEUE, size_m, size_m, nrefl, lda);

    _ComputeDT* orgqr_scratchpad =
        reinterpret_cast<_ComputeDT*>(dpnp_memory_alloc_c(orgqr_scratchpad_size * sizeof(_ComputeDT)));

    depends.clear();
    set_barrier_event(DPNP_QUEUE, depends);

    event = mkl_lapack::orgqr(
        DPNP_QUEUE, size_m, size_m, nrefl, in_a, lda, tau, orgqr_scratchpad, orgqr_scratchpad_size, depends);

    event.wait();

    verbose_print("oneapi::mkl::lapack::orgqr", depends.front(), event);

    dpnp_memory_free_c(orgqr_scratchpad);

    for (size_t i = 0; i < size_m; ++i)
    {
        for (size_t j = 0; j < size_m; ++j)
        {
            if (j < nrefl)
            {
                res_q[i * size_m + j] = in_a[j * size_m + i];
            }
            else
            {
                res_q[i * size_m + j] = _ComputeDT(0);
            }
        }
    }

    dpnp_memory_free_c(in_a);
}

template <typename _InputDT, typename _ComputeDT, typename _SVDT>
void dpnp_svd_c(void* array1_in, void* result1, void* result2, void* result3, size_t size_m, size_t size_n)
{
    cl::sycl::event event;

    DPNPC_ptr_adapter<_InputDT> input1_ptr(array1_in, size_m * size_n, true); // TODO no need this if use dpnp_copy_to()
    _InputDT* in_array = input1_ptr.get_ptr();

    // math lib gesvd func overrides input
    _ComputeDT* in_a = reinterpret_cast<_ComputeDT*>(dpnp_memory_alloc_c(size_m * size_n * sizeof(_ComputeDT)));
    for (size_t it = 0; it < size_m * size_n; ++it)
    {
        in_a[it] = in_array[it]; // TODO Type conversion. memcpy can not be used directly. dpnp_copy_to() ?
    }

    DPNPC_ptr_adapter<_ComputeDT> result1_ptr(result1, size_m * size_m, true, true);
    DPNPC_ptr_adapter<_SVDT> result2_ptr(result2, std::min(size_m, size_n), true, true);
    DPNPC_ptr_adapter<_ComputeDT> result3_ptr(result3, size_n * size_n, true, true);
    _ComputeDT* res_u = result1_ptr.get_ptr();
    _SVDT* res_s = result2_ptr.get_ptr();
    _ComputeDT* res_vt = result3_ptr.get_ptr();

    const std::int64_t m = size_m;
    const std::int64_t n = size_n;

    const std::int64_t lda = std::max<size_t>(1UL, n);
    const std::int64_t ldu = std::max<size_t>(1UL, m);
    const std::int64_t ldvt = std::max<size_t>(1UL, n);

    const std::int64_t scratchpad_size = mkl_lapack::gesvd_scratchpad_size<_ComputeDT>(
        DPNP_QUEUE, oneapi::mkl::jobsvd::vectors, oneapi::mkl::jobsvd::vectors, n, m, lda, ldvt, ldu);

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
    fmap[DPNPFuncName::DPNP_FN_CHOLESKY][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_cholesky_c<float>};
    fmap[DPNPFuncName::DPNP_FN_CHOLESKY][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_cholesky_c<double>};

    fmap[DPNPFuncName::DPNP_FN_DET][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_det_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_DET][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_det_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_DET][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_det_c<float>};
    fmap[DPNPFuncName::DPNP_FN_DET][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_det_c<double>};

    fmap[DPNPFuncName::DPNP_FN_INV][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_inv_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_INV][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_inv_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_INV][eft_FLT][eft_FLT] = {eft_DBL, (void*)dpnp_inv_c<float, double>};
    fmap[DPNPFuncName::DPNP_FN_INV][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_inv_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_KRON][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_kron_c<int32_t, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_kron_c<int32_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_INT][eft_FLT] = {eft_FLT, (void*)dpnp_kron_c<int32_t, float, float>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_kron_c<int32_t, double, double>};
    // fmap[DPNPFuncName::DPNP_FN_KRON][eft_INT][eft_C128] = {
    // eft_C128, (void*)dpnp_kron_c<int32_t, std::complex<double>, std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_kron_c<int64_t, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_kron_c<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_LNG][eft_FLT] = {eft_FLT, (void*)dpnp_kron_c<int64_t, float, float>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_kron_c<int64_t, double, double>};
    // fmap[DPNPFuncName::DPNP_FN_KRON][eft_LNG][eft_C128] = {
    // eft_C128, (void*)dpnp_kron_c<int64_t, std::complex<double>, std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_FLT][eft_INT] = {eft_FLT, (void*)dpnp_kron_c<float, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_FLT][eft_LNG] = {eft_FLT, (void*)dpnp_kron_c<float, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_kron_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_kron_c<float, double, double>};
    // fmap[DPNPFuncName::DPNP_FN_KRON][eft_FLT][eft_C128] = {
    // eft_C128, (void*)dpnp_kron_c<float, std::complex<double>, std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_kron_c<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_kron_c<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_kron_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_kron_c<double, double, double>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_DBL][eft_C128] = {
        eft_C128, (void*)dpnp_kron_c<double, std::complex<double>, std::complex<double>>};
    // fmap[DPNPFuncName::DPNP_FN_KRON][eft_C128][eft_INT] = {
    // eft_C128, (void*)dpnp_kron_c<std::complex<double>, int32_t, std::complex<double>>};
    // fmap[DPNPFuncName::DPNP_FN_KRON][eft_C128][eft_LNG] = {
    // eft_C128, (void*)dpnp_kron_c<std::complex<double>, int64_t, std::complex<double>>};
    // fmap[DPNPFuncName::DPNP_FN_KRON][eft_C128][eft_FLT] = {
    // eft_C128, (void*)dpnp_kron_c<std::complex<double>, float, std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_C128][eft_DBL] = {
        eft_C128, (void*)dpnp_kron_c<std::complex<double>, double, std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_C128][eft_C128] = {
        eft_C128, (void*)dpnp_kron_c<std::complex<double>, std::complex<double>, std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_MATRIX_RANK][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_matrix_rank_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MATRIX_RANK][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_matrix_rank_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MATRIX_RANK][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_matrix_rank_c<float>};
    fmap[DPNPFuncName::DPNP_FN_MATRIX_RANK][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_matrix_rank_c<double>};

    fmap[DPNPFuncName::DPNP_FN_QR][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_qr_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_QR][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_qr_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_QR][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_qr_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_QR][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_qr_c<double, double>};
    // fmap[DPNPFuncName::DPNP_FN_QR][eft_C128][eft_C128] = {
    // eft_C128, (void*)dpnp_qr_c<std::complex<double>, std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_SVD][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_svd_c<int32_t, double, double>};
    fmap[DPNPFuncName::DPNP_FN_SVD][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_svd_c<int64_t, double, double>};
    fmap[DPNPFuncName::DPNP_FN_SVD][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_svd_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_SVD][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_svd_c<double, double, double>};
    fmap[DPNPFuncName::DPNP_FN_SVD][eft_C128][eft_C128] = {
        eft_C128, (void*)dpnp_svd_c<std::complex<double>, std::complex<double>, double>};

    return;
}
