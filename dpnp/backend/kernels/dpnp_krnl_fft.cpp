//*****************************************************************************
// Copyright (c) 2016-2023, Intel Corporation
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
#include "dpnp_utils.hpp"
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"
#include <dpnp_iface.hpp>

namespace mkl_dft = oneapi::mkl::dft;

typedef mkl_dft::descriptor<mkl_dft::precision::DOUBLE,
                            mkl_dft::domain::COMPLEX>
    desc_dp_cmplx_t;
typedef mkl_dft::descriptor<mkl_dft::precision::SINGLE,
                            mkl_dft::domain::COMPLEX>
    desc_sp_cmplx_t;
typedef mkl_dft::descriptor<mkl_dft::precision::DOUBLE, mkl_dft::domain::REAL>
    desc_dp_real_t;
typedef mkl_dft::descriptor<mkl_dft::precision::SINGLE, mkl_dft::domain::REAL>
    desc_sp_real_t;

#ifdef _WIN32
#ifndef M_PI // Windows compatibility
#define M_PI 3.14159265358979323846
#endif
#endif

template <typename _KernelNameSpecialization1,
          typename _KernelNameSpecialization2>
class dpnp_fft_fft_c_kernel;

template <typename _DataType_input, typename _DataType_output>
static void dpnp_fft_fft_sycl_c(DPCTLSyclQueueRef q_ref,
                                const void *array1_in,
                                void *result_out,
                                const shape_elem_type *input_shape,
                                const shape_elem_type *output_shape,
                                size_t shape_size,
                                const size_t result_size,
                                const size_t input_size,
                                long axis,
                                long input_boundarie,
                                size_t inverse)
{
    if (!(input_size && result_size && shape_size)) {
        return;
    }

    sycl::event event;

    const double kernel_pi = inverse ? -M_PI : M_PI;

    sycl::queue queue = *(reinterpret_cast<sycl::queue *>(q_ref));

    _DataType_input *array_1 =
        static_cast<_DataType_input *>(const_cast<void *>(array1_in));
    _DataType_output *result = static_cast<_DataType_output *>(result_out);

    // kernel specific temporal data
    shape_elem_type *output_shape_offsets = reinterpret_cast<shape_elem_type *>(
        sycl::malloc_shared(shape_size * sizeof(shape_elem_type), queue));
    shape_elem_type *input_shape_offsets = reinterpret_cast<shape_elem_type *>(
        sycl::malloc_shared(shape_size * sizeof(shape_elem_type), queue));
    // must be a thread local storage.
    shape_elem_type *axis_iterator =
        reinterpret_cast<shape_elem_type *>(sycl::malloc_shared(
            result_size * shape_size * sizeof(shape_elem_type), queue));

    get_shape_offsets_inkernel(output_shape, shape_size, output_shape_offsets);
    get_shape_offsets_inkernel(input_shape, shape_size, input_shape_offsets);

    sycl::range<1> gws(result_size);
    auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
        size_t output_id = global_id[0];

        double sum_real = 0.0;
        double sum_imag = 0.0;
        // need to replace this array by thread local storage
        shape_elem_type *axis_iterator_thread =
            axis_iterator + (output_id * shape_size);

        size_t xyz_id;
        for (size_t i = 0; i < shape_size; ++i) {
            xyz_id = get_xyz_id_by_id_inkernel(output_id, output_shape_offsets,
                                               shape_size, i);
            axis_iterator_thread[i] = xyz_id;
        }

        const long axis_length = input_boundarie;
        for (long it = 0; it < axis_length; ++it) {
            double in_real = 0.0;
            double in_imag = 0.0;

            axis_iterator_thread[axis] = it;

            const size_t input_it = get_id_by_xyz_inkernel(
                axis_iterator_thread, shape_size, input_shape_offsets);

            if (it < input_shape[axis]) {
                if constexpr (std::is_same<_DataType_input,
                                           std::complex<double>>::value) {
                    const _DataType_input *cmplx_ptr = array_1 + input_it;
                    const double *dbl_ptr =
                        reinterpret_cast<const double *>(cmplx_ptr);
                    in_real = *dbl_ptr;
                    in_imag = *(dbl_ptr + 1);
                }
                else if constexpr (std::is_same<_DataType_input,
                                                std::complex<float>>::value) {
                    const _DataType_input *cmplx_ptr = array_1 + input_it;
                    const float *dbl_ptr =
                        reinterpret_cast<const float *>(cmplx_ptr);
                    in_real = *dbl_ptr;
                    in_imag = *(dbl_ptr + 1);
                }
                else {
                    in_real = array_1[input_it];
                }
            }

            xyz_id = get_xyz_id_by_id_inkernel(output_id, output_shape_offsets,
                                               shape_size, axis);
            const size_t output_local_id = xyz_id;
            const double angle =
                2.0 * kernel_pi * it * output_local_id / axis_length;

            const double angle_cos = sycl::cos(angle);
            const double angle_sin = sycl::sin(angle);

            sum_real += in_real * angle_cos + in_imag * angle_sin;
            sum_imag += -in_real * angle_sin + in_imag * angle_cos;
        }

        if (inverse) {
            sum_real = sum_real / input_boundarie;
            sum_imag = sum_imag / input_boundarie;
        }

        result[output_id] = _DataType_output(sum_real, sum_imag);
    };

    auto kernel_func = [&](sycl::handler &cgh) {
        cgh.parallel_for<
            class dpnp_fft_fft_c_kernel<_DataType_input, _DataType_output>>(
            gws, kernel_parallel_for_func);
    };

    event = queue.submit(kernel_func);
    event.wait();

    sycl::free(input_shape_offsets, queue);
    sycl::free(output_shape_offsets, queue);
    sycl::free(axis_iterator, queue);

    return;
}

template <typename _DataType_input,
          typename _DataType_output,
          typename _Descriptor_type>
static void
    dpnp_fft_fft_mathlib_cmplx_to_cmplx_c(DPCTLSyclQueueRef q_ref,
                                          const void *array1_in,
                                          void *result_out,
                                          const shape_elem_type *input_shape,
                                          const shape_elem_type *result_shape,
                                          const size_t shape_size,
                                          const size_t input_size,
                                          const size_t result_size,
                                          size_t inverse,
                                          const size_t norm)
{
    // avoid warning unused variable
    (void)result_shape;
    (void)input_size;
    (void)result_size;

    if (!shape_size) {
        return;
    }

    sycl::queue queue = *(reinterpret_cast<sycl::queue *>(q_ref));

    _DataType_input *array_1 =
        static_cast<_DataType_input *>(const_cast<void *>(array1_in));
    _DataType_output *result = static_cast<_DataType_output *>(result_out);

    const size_t n_iter =
        std::accumulate(input_shape, input_shape + shape_size - 1, 1,
                        std::multiplies<shape_elem_type>());

    const size_t shift = input_shape[shape_size - 1];

    double backward_scale = 1.;
    double forward_scale = 1.;

    if (norm == 0) // norm = "backward"
    {
        backward_scale = 1. / shift;
    }
    else if (norm == 1) // norm = "forward"
    {
        forward_scale = 1. / shift;
    }
    else // norm = "ortho"
    {
        if (inverse) {
            backward_scale = 1. / sqrt(shift);
        }
        else {
            forward_scale = 1. / sqrt(shift);
        }
    }

    std::vector<sycl::event> fft_events(n_iter);

    for (size_t i = 0; i < n_iter; ++i) {
        std::unique_ptr<_Descriptor_type> desc =
            std::make_unique<_Descriptor_type>(shift);
        desc->set_value(mkl_dft::config_param::BACKWARD_SCALE, backward_scale);
        desc->set_value(mkl_dft::config_param::FORWARD_SCALE, forward_scale);
        desc->set_value(mkl_dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
        desc->commit(queue);

        if (inverse) {
            fft_events[i] =
                mkl_dft::compute_backward<_Descriptor_type, _DataType_input,
                                          _DataType_output>(
                    *desc, array_1 + i * shift, result + i * shift);
        }
        else {
            fft_events[i] =
                mkl_dft::compute_forward<_Descriptor_type, _DataType_input,
                                         _DataType_output>(
                    *desc, array_1 + i * shift, result + i * shift);
        }
    }

    sycl::event::wait(fft_events);
}

template <typename _KernelNameSpecialization1,
          typename _KernelNameSpecialization2,
          typename _KernelNameSpecialization3>
class dpnp_fft_fft_mathlib_real_to_cmplx_c_kernel;

template <typename _DataType_input,
          typename _DataType_output,
          typename _Descriptor_type>
static DPCTLSyclEventRef
    dpnp_fft_fft_mathlib_real_to_cmplx_c(DPCTLSyclQueueRef q_ref,
                                         const void *array1_in,
                                         void *result_out,
                                         const shape_elem_type *input_shape,
                                         const shape_elem_type *result_shape,
                                         const size_t shape_size,
                                         const size_t input_size,
                                         const size_t result_size,
                                         size_t inverse,
                                         const size_t norm,
                                         const size_t real)
{
    // avoid warning unused variable
    (void)input_size;

    DPCTLSyclEventRef event_ref = nullptr;
    if (!shape_size) {
        return event_ref;
    }

    sycl::queue queue = *(reinterpret_cast<sycl::queue *>(q_ref));

    _DataType_input *array_1 =
        static_cast<_DataType_input *>(const_cast<void *>(array1_in));
    _DataType_output *result = static_cast<_DataType_output *>(result_out);

    const size_t n_iter =
        std::accumulate(input_shape, input_shape + shape_size - 1, 1,
                        std::multiplies<shape_elem_type>());

    const size_t input_shift = input_shape[shape_size - 1];
    const size_t result_shift = result_shape[shape_size - 1];

    double backward_scale = 1.;
    double forward_scale = 1.;

    if (norm == 0) // norm = "backward"
    {
        if (inverse) {
            forward_scale = 1. / result_shift;
        }
        else {
            backward_scale = 1. / result_shift;
        }
    }
    else if (norm == 1) // norm = "forward"
    {
        if (inverse) {
            backward_scale = 1. / result_shift;
        }
        else {
            forward_scale = 1. / result_shift;
        }
    }
    else // norm = "ortho"
    {
        forward_scale = 1. / sqrt(result_shift);
    }

    std::vector<sycl::event> fft_events(n_iter);

    for (size_t i = 0; i < n_iter; ++i) {
        std::unique_ptr<_Descriptor_type> desc =
            std::make_unique<_Descriptor_type>(input_shift);
        desc->set_value(mkl_dft::config_param::BACKWARD_SCALE, backward_scale);
        desc->set_value(mkl_dft::config_param::FORWARD_SCALE, forward_scale);
        desc->set_value(mkl_dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
        desc->commit(queue);

        // real result_size = 2 * result_size, because real type of "result" is
        // twice wider than '_DataType_output'
        fft_events[i] =
            mkl_dft::compute_forward<_Descriptor_type, _DataType_input,
                                     _DataType_output>(
                *desc, array_1 + i * input_shift,
                result + i * result_shift * 2);
    }

    sycl::event::wait(fft_events);

    if (real) // the output size of the rfft function is input_size/2 + 1 so we
              // don't need to fill the second half of the output
    {
        return event_ref;
    }

    size_t n_conj =
        result_shift % 2 == 0 ? result_shift / 2 - 1 : result_shift / 2;

    sycl::event event;

    sycl::range<2> gws(n_iter, n_conj);

    auto kernel_parallel_for_func = [=](sycl::id<2> global_id) {
        size_t i = global_id[0];
        {
            size_t j = global_id[1];
            {
                *(reinterpret_cast<std::complex<_DataType_output> *>(result) +
                  result_shift * (i + 1) - (j + 1)) =
                    std::conj(
                        *(reinterpret_cast<std::complex<_DataType_output> *>(
                              result) +
                          result_shift * i + (j + 1)));
            }
        }
    };

    auto kernel_func = [&](sycl::handler &cgh) {
        cgh.parallel_for<class dpnp_fft_fft_mathlib_real_to_cmplx_c_kernel<
            _DataType_input, _DataType_output, _Descriptor_type>>(
            gws, kernel_parallel_for_func);
    };

    event = queue.submit(kernel_func);

    if (inverse) {
        event.wait();
        event = oneapi::mkl::vm::conj(
            queue, result_size,
            reinterpret_cast<std::complex<_DataType_output> *>(result),
            reinterpret_cast<std::complex<_DataType_output> *>(result));
    }

    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);
    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType_input, typename _DataType_output>
DPCTLSyclEventRef dpnp_fft_fft_c(DPCTLSyclQueueRef q_ref,
                                 const void *array1_in,
                                 void *result_out,
                                 const shape_elem_type *input_shape,
                                 const shape_elem_type *result_shape,
                                 size_t shape_size,
                                 long axis,
                                 long input_boundarie,
                                 size_t inverse,
                                 const size_t norm,
                                 const DPCTLEventVectorRef dep_event_vec_ref)
{
    static_assert(sycl::detail::is_complex<_DataType_output>::value,
                  "Output data type must be a complex type.");

    DPCTLSyclEventRef event_ref = nullptr;

    if (!shape_size || !array1_in || !result_out) {
        return event_ref;
    }

    const size_t result_size =
        std::accumulate(result_shape, result_shape + shape_size, 1,
                        std::multiplies<shape_elem_type>());
    const size_t input_size =
        std::accumulate(input_shape, input_shape + shape_size, 1,
                        std::multiplies<shape_elem_type>());

    if constexpr (std::is_same<_DataType_output, std::complex<float>>::value ||
                  std::is_same<_DataType_output, std::complex<double>>::value)
    {
        if constexpr (std::is_same<_DataType_input,
                                   std::complex<double>>::value &&
                      std::is_same<_DataType_output,
                                   std::complex<double>>::value)
        {
            dpnp_fft_fft_mathlib_cmplx_to_cmplx_c<
                _DataType_input, _DataType_output, desc_dp_cmplx_t>(
                q_ref, array1_in, result_out, input_shape, result_shape,
                shape_size, input_size, result_size, inverse, norm);
        }
        /* complex-to-complex, single precision */
        else if constexpr (std::is_same<_DataType_input,
                                        std::complex<float>>::value &&
                           std::is_same<_DataType_output,
                                        std::complex<float>>::value)
        {
            dpnp_fft_fft_mathlib_cmplx_to_cmplx_c<
                _DataType_input, _DataType_output, desc_sp_cmplx_t>(
                q_ref, array1_in, result_out, input_shape, result_shape,
                shape_size, input_size, result_size, inverse, norm);
        }
        /* real-to-complex, double precision */
        else if constexpr (std::is_same<_DataType_input, double>::value &&
                           std::is_same<_DataType_output,
                                        std::complex<double>>::value)
        {
            event_ref =
                dpnp_fft_fft_mathlib_real_to_cmplx_c<_DataType_input, double,
                                                     desc_dp_real_t>(
                    q_ref, array1_in, result_out, input_shape, result_shape,
                    shape_size, input_size, result_size, inverse, norm, 0);
        }
        /* real-to-complex, single precision */
        else if constexpr (std::is_same<_DataType_input, float>::value &&
                           std::is_same<_DataType_output,
                                        std::complex<float>>::value)
        {
            event_ref =
                dpnp_fft_fft_mathlib_real_to_cmplx_c<_DataType_input, float,
                                                     desc_sp_real_t>(
                    q_ref, array1_in, result_out, input_shape, result_shape,
                    shape_size, input_size, result_size, inverse, norm, 0);
        }
        else if constexpr (std::is_same<_DataType_input, int32_t>::value ||
                           std::is_same<_DataType_input, int64_t>::value)
        {
            using CastType = typename _DataType_output::value_type;

            CastType *array1_copy = reinterpret_cast<CastType *>(
                dpnp_memory_alloc_c(q_ref, input_size * sizeof(CastType)));

            shape_elem_type *copy_strides = reinterpret_cast<shape_elem_type *>(
                dpnp_memory_alloc_c(q_ref, sizeof(shape_elem_type)));
            *copy_strides = 1;
            shape_elem_type *copy_shape = reinterpret_cast<shape_elem_type *>(
                dpnp_memory_alloc_c(q_ref, sizeof(shape_elem_type)));
            *copy_shape = input_size;
            shape_elem_type copy_shape_size = 1;
            event_ref = dpnp_copyto_c<_DataType_input, CastType>(
                q_ref, array1_copy, input_size, copy_shape_size, copy_shape,
                copy_strides, array1_in, input_size, copy_shape_size,
                copy_shape, copy_strides, NULL, dep_event_vec_ref);
            DPCTLEvent_WaitAndThrow(event_ref);
            DPCTLEvent_Delete(event_ref);

            event_ref = dpnp_fft_fft_mathlib_real_to_cmplx_c<
                CastType, CastType,
                std::conditional_t<std::is_same<CastType, double>::value,
                                   desc_dp_real_t, desc_sp_real_t>>(
                q_ref, array1_copy, result_out, input_shape, result_shape,
                shape_size, input_size, result_size, inverse, norm, 0);

            DPCTLEvent_WaitAndThrow(event_ref);
            DPCTLEvent_Delete(event_ref);
            event_ref = nullptr;

            dpnp_memory_free_c(q_ref, array1_copy);
            dpnp_memory_free_c(q_ref, copy_strides);
            dpnp_memory_free_c(q_ref, copy_shape);
        }
        else {
            dpnp_fft_fft_sycl_c<_DataType_input, _DataType_output>(
                q_ref, array1_in, result_out, input_shape, result_shape,
                shape_size, result_size, input_size, axis, input_boundarie,
                inverse);
        }
    }

    return event_ref;
}

template <typename _DataType_input, typename _DataType_output>
void dpnp_fft_fft_c(const void *array1_in,
                    void *result1,
                    const shape_elem_type *input_shape,
                    const shape_elem_type *output_shape,
                    size_t shape_size,
                    long axis,
                    long input_boundarie,
                    size_t inverse,
                    const size_t norm)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref =
        dpnp_fft_fft_c<_DataType_input, _DataType_output>(
            q_ref, array1_in, result1, input_shape, output_shape, shape_size,
            axis, input_boundarie, inverse, norm, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType_input, typename _DataType_output>
void (*dpnp_fft_fft_default_c)(const void *,
                               void *,
                               const shape_elem_type *,
                               const shape_elem_type *,
                               size_t,
                               long,
                               long,
                               size_t,
                               const size_t) =
    dpnp_fft_fft_c<_DataType_input, _DataType_output>;

template <typename _DataType_input, typename _DataType_output>
DPCTLSyclEventRef (*dpnp_fft_fft_ext_c)(DPCTLSyclQueueRef,
                                        const void *,
                                        void *,
                                        const shape_elem_type *,
                                        const shape_elem_type *,
                                        size_t,
                                        long,
                                        long,
                                        size_t,
                                        const size_t,
                                        const DPCTLEventVectorRef) =
    dpnp_fft_fft_c<_DataType_input, _DataType_output>;

template <typename _DataType_input, typename _DataType_output>
DPCTLSyclEventRef dpnp_fft_rfft_c(DPCTLSyclQueueRef q_ref,
                                  const void *array1_in,
                                  void *result_out,
                                  const shape_elem_type *input_shape,
                                  const shape_elem_type *result_shape,
                                  size_t shape_size,
                                  long, // axis
                                  long, // input_boundary
                                  size_t inverse,
                                  const size_t norm,
                                  const DPCTLEventVectorRef dep_event_vec_ref)
{
    static_assert(sycl::detail::is_complex<_DataType_output>::value,
                  "Output data type must be a complex type.");
    DPCTLSyclEventRef event_ref = nullptr;

    if (!shape_size || !array1_in || !result_out) {
        return event_ref;
    }

    const size_t result_size =
        std::accumulate(result_shape, result_shape + shape_size, 1,
                        std::multiplies<shape_elem_type>());
    const size_t input_size =
        std::accumulate(input_shape, input_shape + shape_size, 1,
                        std::multiplies<shape_elem_type>());

    if constexpr (std::is_same<_DataType_output, std::complex<float>>::value ||
                  std::is_same<_DataType_output, std::complex<double>>::value)
    {
        if constexpr (std::is_same<_DataType_input, double>::value &&
                      std::is_same<_DataType_output,
                                   std::complex<double>>::value)
        {
            event_ref =
                dpnp_fft_fft_mathlib_real_to_cmplx_c<_DataType_input, double,
                                                     desc_dp_real_t>(
                    q_ref, array1_in, result_out, input_shape, result_shape,
                    shape_size, input_size, result_size, inverse, norm, 1);
        }
        /* real-to-complex, single precision */
        else if constexpr (std::is_same<_DataType_input, float>::value &&
                           std::is_same<_DataType_output,
                                        std::complex<float>>::value)
        {
            event_ref =
                dpnp_fft_fft_mathlib_real_to_cmplx_c<_DataType_input, float,
                                                     desc_sp_real_t>(
                    q_ref, array1_in, result_out, input_shape, result_shape,
                    shape_size, input_size, result_size, inverse, norm, 1);
        }
        else if constexpr (std::is_same<_DataType_input, int32_t>::value ||
                           std::is_same<_DataType_input, int64_t>::value)
        {
            using CastType = typename _DataType_output::value_type;

            CastType *array1_copy = reinterpret_cast<CastType *>(
                dpnp_memory_alloc_c(q_ref, input_size * sizeof(CastType)));

            shape_elem_type *copy_strides = reinterpret_cast<shape_elem_type *>(
                dpnp_memory_alloc_c(q_ref, sizeof(shape_elem_type)));
            *copy_strides = 1;
            shape_elem_type *copy_shape = reinterpret_cast<shape_elem_type *>(
                dpnp_memory_alloc_c(q_ref, sizeof(shape_elem_type)));
            *copy_shape = input_size;
            shape_elem_type copy_shape_size = 1;
            event_ref = dpnp_copyto_c<_DataType_input, CastType>(
                q_ref, array1_copy, input_size, copy_shape_size, copy_shape,
                copy_strides, array1_in, input_size, copy_shape_size,
                copy_shape, copy_strides, NULL, dep_event_vec_ref);
            DPCTLEvent_WaitAndThrow(event_ref);
            DPCTLEvent_Delete(event_ref);

            event_ref = dpnp_fft_fft_mathlib_real_to_cmplx_c<
                CastType, CastType,
                std::conditional_t<std::is_same<CastType, double>::value,
                                   desc_dp_real_t, desc_sp_real_t>>(
                q_ref, array1_copy, result_out, input_shape, result_shape,
                shape_size, input_size, result_size, inverse, norm, 1);

            DPCTLEvent_WaitAndThrow(event_ref);
            DPCTLEvent_Delete(event_ref);
            event_ref = nullptr;

            dpnp_memory_free_c(q_ref, array1_copy);
            dpnp_memory_free_c(q_ref, copy_strides);
            dpnp_memory_free_c(q_ref, copy_shape);
        }
    }

    return event_ref;
}

template <typename _DataType_input, typename _DataType_output>
void dpnp_fft_rfft_c(const void *array1_in,
                     void *result1,
                     const shape_elem_type *input_shape,
                     const shape_elem_type *output_shape,
                     size_t shape_size,
                     long axis,
                     long input_boundarie,
                     size_t inverse,
                     const size_t norm)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref =
        dpnp_fft_rfft_c<_DataType_input, _DataType_output>(
            q_ref, array1_in, result1, input_shape, output_shape, shape_size,
            axis, input_boundarie, inverse, norm, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType_input, typename _DataType_output>
void (*dpnp_fft_rfft_default_c)(const void *,
                                void *,
                                const shape_elem_type *,
                                const shape_elem_type *,
                                size_t,
                                long,
                                long,
                                size_t,
                                const size_t) =
    dpnp_fft_rfft_c<_DataType_input, _DataType_output>;

template <typename _DataType_input, typename _DataType_output>
DPCTLSyclEventRef (*dpnp_fft_rfft_ext_c)(DPCTLSyclQueueRef,
                                         const void *,
                                         void *,
                                         const shape_elem_type *,
                                         const shape_elem_type *,
                                         size_t,
                                         long,
                                         long,
                                         size_t,
                                         const size_t,
                                         const DPCTLEventVectorRef) =
    dpnp_fft_rfft_c<_DataType_input, _DataType_output>;

void func_map_init_fft_func(func_map_t &fmap)
{
    fmap[DPNPFuncName::DPNP_FN_FFT_FFT][eft_INT][eft_INT] = {
        eft_C128,
        (void *)dpnp_fft_fft_default_c<int32_t, std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_FFT_FFT][eft_LNG][eft_LNG] = {
        eft_C128,
        (void *)dpnp_fft_fft_default_c<int64_t, std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_FFT_FFT][eft_FLT][eft_FLT] = {
        eft_C64, (void *)dpnp_fft_fft_default_c<float, std::complex<float>>};
    fmap[DPNPFuncName::DPNP_FN_FFT_FFT][eft_DBL][eft_DBL] = {
        eft_C128, (void *)dpnp_fft_fft_default_c<double, std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_FFT_FFT][eft_C64][eft_C64] = {
        eft_C64,
        (void *)
            dpnp_fft_fft_default_c<std::complex<float>, std::complex<float>>};
    fmap[DPNPFuncName::DPNP_FN_FFT_FFT][eft_C128][eft_C128] = {
        eft_C128,
        (void *)
            dpnp_fft_fft_default_c<std::complex<double>, std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_FFT_FFT_EXT][eft_INT][eft_INT] = {
        eft_C128, (void *)dpnp_fft_fft_ext_c<int32_t, std::complex<double>>,
        eft_C64, (void *)dpnp_fft_fft_ext_c<int32_t, std::complex<float>>};
    fmap[DPNPFuncName::DPNP_FN_FFT_FFT_EXT][eft_LNG][eft_LNG] = {
        eft_C128, (void *)dpnp_fft_fft_ext_c<int64_t, std::complex<double>>,
        eft_C64, (void *)dpnp_fft_fft_ext_c<int64_t, std::complex<float>>};
    fmap[DPNPFuncName::DPNP_FN_FFT_FFT_EXT][eft_FLT][eft_FLT] = {
        eft_C64, (void *)dpnp_fft_fft_ext_c<float, std::complex<float>>};
    fmap[DPNPFuncName::DPNP_FN_FFT_FFT_EXT][eft_DBL][eft_DBL] = {
        eft_C128, (void *)dpnp_fft_fft_ext_c<double, std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_FFT_FFT_EXT][eft_C64][eft_C64] = {
        eft_C64,
        (void *)dpnp_fft_fft_ext_c<std::complex<float>, std::complex<float>>};
    fmap[DPNPFuncName::DPNP_FN_FFT_FFT_EXT][eft_C128][eft_C128] = {
        eft_C128,
        (void *)dpnp_fft_fft_ext_c<std::complex<double>, std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_FFT_RFFT][eft_INT][eft_INT] = {
        eft_C128,
        (void *)dpnp_fft_rfft_default_c<int32_t, std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_FFT_RFFT][eft_LNG][eft_LNG] = {
        eft_C128,
        (void *)dpnp_fft_rfft_default_c<int64_t, std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_FFT_RFFT][eft_FLT][eft_FLT] = {
        eft_C64, (void *)dpnp_fft_rfft_default_c<float, std::complex<float>>};
    fmap[DPNPFuncName::DPNP_FN_FFT_RFFT][eft_DBL][eft_DBL] = {
        eft_C128,
        (void *)dpnp_fft_rfft_default_c<double, std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_FFT_RFFT_EXT][eft_INT][eft_INT] = {
        eft_C128, (void *)dpnp_fft_rfft_ext_c<int32_t, std::complex<double>>,
        eft_C64, (void *)dpnp_fft_rfft_ext_c<int32_t, std::complex<float>>};
    fmap[DPNPFuncName::DPNP_FN_FFT_RFFT_EXT][eft_LNG][eft_LNG] = {
        eft_C128, (void *)dpnp_fft_rfft_ext_c<int64_t, std::complex<double>>,
        eft_C64, (void *)dpnp_fft_rfft_ext_c<int64_t, std::complex<float>>};
    fmap[DPNPFuncName::DPNP_FN_FFT_RFFT_EXT][eft_FLT][eft_FLT] = {
        eft_C64, (void *)dpnp_fft_rfft_ext_c<float, std::complex<float>>};
    fmap[DPNPFuncName::DPNP_FN_FFT_RFFT_EXT][eft_DBL][eft_DBL] = {
        eft_C128, (void *)dpnp_fft_rfft_ext_c<double, std::complex<double>>};

    return;
}
