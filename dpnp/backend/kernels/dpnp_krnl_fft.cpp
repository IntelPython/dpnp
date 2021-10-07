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

#include <dpnp_iface.hpp>
#include "dpnp_fptr.hpp"
#include "dpnp_utils.hpp"
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"

namespace mkl_dft = oneapi::mkl::dft;

typedef mkl_dft::descriptor<mkl_dft::precision::DOUBLE, mkl_dft::domain::COMPLEX> desc_dp_cmplx_t;
typedef mkl_dft::descriptor<mkl_dft::precision::SINGLE, mkl_dft::domain::COMPLEX> desc_sp_cmplx_t;
typedef mkl_dft::descriptor<mkl_dft::precision::DOUBLE, mkl_dft::domain::REAL> desc_dp_real_t;
typedef mkl_dft::descriptor<mkl_dft::precision::SINGLE, mkl_dft::domain::REAL> desc_sp_real_t;

#ifdef _WIN32
#ifndef M_PI // Windows compatibility
#define M_PI 3.14159265358979323846
#endif
#endif

template <typename _KernelNameSpecialization1, typename _KernelNameSpecialization2>
class dpnp_fft_fft_c_kernel;

template <typename _DataType_input, typename _DataType_output>
void dpnp_fft_fft_sycl_c(const void* array1_in,
                         void* result1,
                         const long* input_shape,
                         const long* output_shape,
                         size_t shape_size,
                         const size_t result_size,
                         const size_t input_size,
                         long axis,
                         long input_boundarie,
                         size_t inverse)
{
    if (!(input_size && result_size && shape_size))
    {
        return;
    }

    cl::sycl::event event;

    const double kernel_pi = inverse ? -M_PI : M_PI;

    DPNPC_ptr_adapter<_DataType_input> input1_ptr(array1_in, input_size);
    const _DataType_input* array_1 = input1_ptr.get_ptr();
    _DataType_output* result = reinterpret_cast<_DataType_output*>(result1);

    // kernel specific temporal data
    long* output_shape_offsets = reinterpret_cast<long*>(dpnp_memory_alloc_c(shape_size * sizeof(long)));
    long* input_shape_offsets = reinterpret_cast<long*>(dpnp_memory_alloc_c(shape_size * sizeof(long)));
    // must be a thread local storage.
    long* axis_iterator = reinterpret_cast<long*>(dpnp_memory_alloc_c(result_size * shape_size * sizeof(long)));

    get_shape_offsets_inkernel<long>(output_shape, shape_size, output_shape_offsets);
    get_shape_offsets_inkernel<long>(input_shape, shape_size, input_shape_offsets);

    cl::sycl::range<1> gws(result_size);
    auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {
        size_t output_id = global_id[0];

        double sum_real = 0.0;
        double sum_imag = 0.0;
        // need to replace this array by thread local storage
        long* axis_iterator_thread = axis_iterator + (output_id * shape_size);

        size_t xyz_id;
        for (size_t i = 0; i < shape_size; ++i)
        {
            xyz_id = get_xyz_id_by_id_inkernel(output_id, output_shape_offsets, shape_size, i);
            axis_iterator_thread[i] = xyz_id;
        }

        const long axis_length = input_boundarie;
        for (long it = 0; it < axis_length; ++it)
        {
            double in_real = 0.0;
            double in_imag = 0.0;

            axis_iterator_thread[axis] = it;

            const size_t input_it = get_id_by_xyz_inkernel(axis_iterator_thread, shape_size, input_shape_offsets);

            if (it < input_shape[axis])
            {
                if constexpr (std::is_same<_DataType_input, std::complex<double>>::value)
                {
                    const _DataType_input* cmplx_ptr = array_1 + input_it;
                    const double* dbl_ptr = reinterpret_cast<const double*>(cmplx_ptr);
                    in_real = *dbl_ptr;
                    in_imag = *(dbl_ptr + 1);
                }
                else if constexpr (std::is_same<_DataType_input, std::complex<float>>::value)
                {
                    const _DataType_input* cmplx_ptr = array_1 + input_it;
                    const float* dbl_ptr = reinterpret_cast<const float*>(cmplx_ptr);
                    in_real = *dbl_ptr;
                    in_imag = *(dbl_ptr + 1);
                }
                else
                {
                    in_real = array_1[input_it];
                }
            }

            xyz_id = get_xyz_id_by_id_inkernel(output_id, output_shape_offsets, shape_size, axis);
            const size_t output_local_id = xyz_id;
            const double angle = 2.0 * kernel_pi * it * output_local_id / axis_length;

            const double angle_cos = cl::sycl::cos(angle);
            const double angle_sin = cl::sycl::sin(angle);

            sum_real += in_real * angle_cos + in_imag * angle_sin;
            sum_imag += -in_real * angle_sin + in_imag * angle_cos;
        }

        if (inverse)
        {
            sum_real = sum_real / input_boundarie;
            sum_imag = sum_imag / input_boundarie;
        }

        result[output_id] = _DataType_output(sum_real, sum_imag);
    };

    auto kernel_func = [&](cl::sycl::handler& cgh) {
        cgh.parallel_for<class dpnp_fft_fft_c_kernel<_DataType_input, _DataType_output>>(gws, kernel_parallel_for_func);
    };

    event = DPNP_QUEUE.submit(kernel_func);
    event.wait();

    dpnp_memory_free_c(input_shape_offsets);
    dpnp_memory_free_c(output_shape_offsets);
    dpnp_memory_free_c(axis_iterator);

    return;
}

// TODO
/* future interface */
template <typename _DataType_input, typename _DataType_output, typename _Descriptor_type>
void dpnp_fft_fft_mathlib_cmplx_to_cmplx_c(const void* array1_in,
                                           void* result1,
                                           const size_t shape_size,
                                           const size_t result_size,
                                           _Descriptor_type& desc,
                                           const size_t norm)
{
    cl::sycl::event event;

    DPNPC_ptr_adapter<_DataType_input> input1_ptr(array1_in, result_size);
    DPNPC_ptr_adapter<_DataType_output> result_ptr(result1, result_size);
    _DataType_input* array_1 = input1_ptr.get_ptr();
    _DataType_output* result = result_ptr.get_ptr();

    desc.set_value(mkl_dft::config_param::BACKWARD_SCALE, (1.0 / result_size));
    // enum value from math library C interface
    // instead of mkl_dft::config_value::NOT_INPLACE
    desc.set_value(mkl_dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
    desc.commit(DPNP_QUEUE);

    event = mkl_dft::compute_forward(desc, array_1, result);

    event.wait();

    return;
}

template <typename _DataType_input, typename _DataType_output, typename _Descriptor_type>
void dpnp_fft_fft_mathlib_real_to_cmplx_c(const void* array1_in,
                                          void* result1,
                                          const size_t shape_size,
                                          const size_t result_size,
                                          const long* input_strides,
                                          const long* output_strides,
                                          _Descriptor_type& desc,
                                          const size_t norm)
{
    cl::sycl::event event;

    DPNPC_ptr_adapter<_DataType_input> input1_ptr(array1_in, result_size);
    DPNPC_ptr_adapter<_DataType_output> result_ptr(result1, result_size);
    _DataType_input* array_1 = input1_ptr.get_ptr();
    _DataType_output* result = result_ptr.get_ptr();

    // works only for 1d double
    MKL_LONG input_strides_desc[2] = {0, 1};
    MKL_LONG output_strides_desc[2] = {0, 1};

    const std::int64_t* xin_strides = reinterpret_cast<const std::int64_t*>(input_strides);
    const std::int64_t* xout_strides = reinterpret_cast<const std::int64_t*>(output_strides);
    MKL_LONG input_number_of_transforms = 1; // hardcoded now. TBD

    char *tmp = (char *) array_1;
    input_strides_desc[1] = ((_DataType_input*) (tmp + (xin_strides[0] * sizeof(_DataType_input)))) - array_1;
    tmp = (char *) result;
    output_strides_desc[1] =
            ((_DataType_output*) (tmp + (xout_strides[0] * sizeof(_DataType_output)))) - result;

    desc.set_value(mkl_dft::config_param::BACKWARD_SCALE, (1.0 / std::sqrt(result_size)));     // TODO
                                                                                               // result_size is real results_size
                                                                                               // add bckward_scale / fw_scale params

    desc.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);

    desc.set_value(oneapi::mkl::dft::config_param::PACKED_FORMAT, DFTI_CCE_FORMAT);
    // enum value from math library C interface
    // instead of mkl_dft::config_value::NOT_INPLACE
    desc.set_value(mkl_dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
    // TODO lets find correct values
    desc.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_strides_desc);
    desc.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_strides_desc);
    desc.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, input_number_of_transforms);

    desc.commit(DPNP_QUEUE);

    event = mkl_dft::compute_forward(desc, array_1, result);

    event.wait();

    return;
}

/* norm: backward - 0, forward - 1 */
template <typename _DataType_input, typename _DataType_output>
void dpnp_fft_fft_mathlib_c(const void* array1_in,
                            void* result1,
                            const long* input_shape,
                            const long* input_strides,
                            const long* output_strides,
                            const size_t shape_size,
                            const size_t result_size,
                            const size_t norm)
{
    if (!shape_size || !result_size || !array1_in || !result1)
    {
        return;
    }
    std::vector<std::int64_t> dimensions(input_shape, input_shape + shape_size);

    /* complex-to-complex, double precision */
    if constexpr (std::is_same<_DataType_input, std::complex<double>>::value &&
                  std::is_same<_DataType_output, std::complex<double>>::value)
    {
        desc_dp_cmplx_t desc(dimensions);
        dpnp_fft_fft_mathlib_cmplx_to_cmplx_c<_DataType_input, _DataType_output, desc_dp_cmplx_t>(
            array1_in, result1, shape_size, result_size, desc, norm);

    }
    /* complex-to-complex, single precision */
    else if (std::is_same<_DataType_input, std::complex<float>>::value &&
             std::is_same<_DataType_output, std::complex<float>>::value)
    {

        desc_sp_cmplx_t desc(dimensions);
        dpnp_fft_fft_mathlib_cmplx_to_cmplx_c<_DataType_input, _DataType_output, desc_sp_cmplx_t>(
            array1_in, result1, shape_size, result_size, desc, norm);

    }
    /* real-to-complex, double precision */
    else if (std::is_same<_DataType_input, double>::value &&
             std::is_same<_DataType_output, std::complex<double>>::value)
    {

        // const result_size_cce_pack_format = result_size * 2;
        desc_dp_real_t desc(dimensions); // try: 2 * result_size
        dpnp_fft_fft_mathlib_real_to_cmplx_c<_DataType_input, double, desc_dp_real_t>(
            array1_in, result1, shape_size, result_size, input_strides, output_strides, desc, norm);

    }
    /* real-to-complex, single precision */
    else if (std::is_same<_DataType_input, float>::value &&
             std::is_same<_DataType_output, std::complex<float>>::value)
    {

        // const result_size_cce_pack_format = result_size * 2;
        desc_sp_real_t desc(dimensions); // try: 2 * result_size
        dpnp_fft_fft_mathlib_real_to_cmplx_c<_DataType_input, float, desc_sp_real_t>(
            array1_in, result1, shape_size, result_size, input_strides, output_strides, desc, norm);

    }
    return;
}
// TODO
// Will be deprecated
template <typename _DataType_input, typename _DataType_output>
void dpnp_fft_fft_c(const void* array1_in,
                    void* result1,
                    const long* input_shape,
                    const long* output_shape,
                    size_t shape_size,
                    const long* input_strides,
                    const long* output_strides,
                    long axis,
                    long input_boundarie,
                    size_t inverse,
                    const size_t norm)
{
    if (!shape_size)
    {
        return;
    }

    const size_t result_size = std::accumulate(output_shape, output_shape + shape_size, 1, std::multiplies<size_t>());
    const size_t input_size = std::accumulate(input_shape, input_shape + shape_size, 1, std::multiplies<size_t>());

    if (!input_size || !result_size || !array1_in || !result1)
    {
        return;
    }

    if (((std::is_same<_DataType_input, std::complex<double>>::value &&
          std::is_same<_DataType_output, std::complex<double>>::value) ||
         (std::is_same<_DataType_input, std::complex<float>>::value &&
          std::is_same<_DataType_output, std::complex<float>>::value) ||
         (std::is_same<_DataType_input, double>::value &&
          std::is_same<_DataType_output, std::complex<double>>::value)||
         (std::is_same<_DataType_input, float>::value &&
          std::is_same<_DataType_output, std::complex<float>>::value))
        && (shape_size <= 3))
    {
        dpnp_fft_fft_mathlib_c<_DataType_input, _DataType_output>(
            array1_in, result1, input_shape, input_strides, output_strides,shape_size, result_size, norm);
    }
    else
    {
        dpnp_fft_fft_sycl_c<_DataType_input, _DataType_output>(array1_in,
                                                               result1,
                                                               input_shape,
                                                               output_shape,
                                                               shape_size,
                                                               result_size,
                                                               input_size,
                                                               axis,
                                                               input_boundarie,
                                                               inverse);
    }

    return;
}

void func_map_init_fft_func(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_FFT_FFT][eft_INT][eft_INT] = {eft_C128,
                                                             (void*)dpnp_fft_fft_c<int, std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_FFT_FFT][eft_LNG][eft_LNG] = {eft_C128,
                                                             (void*)dpnp_fft_fft_c<long, std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_FFT_FFT][eft_FLT][eft_FLT] = {eft_C64,
                                                             (void*)dpnp_fft_fft_c<float, std::complex<float>>};
    fmap[DPNPFuncName::DPNP_FN_FFT_FFT][eft_DBL][eft_DBL] = {eft_C128,
                                                             (void*)dpnp_fft_fft_c<double, std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_FFT_FFT][eft_C64][eft_C64] = {
        eft_C64, (void*)dpnp_fft_fft_c<std::complex<float>, std::complex<float>>};
    fmap[DPNPFuncName::DPNP_FN_FFT_FFT][eft_C128][eft_C128] = {
        eft_C128, (void*)dpnp_fft_fft_c<std::complex<double>, std::complex<double>>};
    return;
}
