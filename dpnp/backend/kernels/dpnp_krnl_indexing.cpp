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
#include <list>
#include <vector>

#include "dpnp_fptr.hpp"
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"
#include <dpnp_iface.hpp>

template <typename _DataType1, typename _DataType2>
class dpnp_choose_c_kernel;

template <typename _DataType1, typename _DataType2>
DPCTLSyclEventRef dpnp_choose_c(DPCTLSyclQueueRef q_ref,
                                void *result1,
                                void *array1_in,
                                void **choices1,
                                size_t size,
                                size_t choices_size,
                                size_t choice_size,
                                const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if ((array1_in == nullptr) || (result1 == nullptr) || (choices1 == nullptr))
    {
        return event_ref;
    }
    if (!size || !choices_size || !choice_size) {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));

    DPNPC_ptr_adapter<_DataType1> input1_ptr(q_ref, array1_in, size);
    _DataType1 *array_in = input1_ptr.get_ptr();

    DPNPC_ptr_adapter<_DataType2 *> choices_ptr(q_ref, choices1, choices_size);
    _DataType2 **choices = choices_ptr.get_ptr();

    for (size_t i = 0; i < choices_size; ++i) {
        DPNPC_ptr_adapter<_DataType2> choice_ptr(q_ref, choices[i],
                                                 choice_size);
        choices[i] = choice_ptr.get_ptr();
    }

    DPNPC_ptr_adapter<_DataType2> result1_ptr(q_ref, result1, size, false,
                                              true);
    _DataType2 *result = result1_ptr.get_ptr();

    sycl::range<1> gws(size);
    auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
        const size_t idx = global_id[0];
        result[idx] = choices[array_in[idx]][idx];
    };

    auto kernel_func = [&](sycl::handler &cgh) {
        cgh.parallel_for<class dpnp_choose_c_kernel<_DataType1, _DataType2>>(
            gws, kernel_parallel_for_func);
    };

    sycl::event event = q.submit(kernel_func);

    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType1, typename _DataType2>
void dpnp_choose_c(void *result1,
                   void *array1_in,
                   void **choices1,
                   size_t size,
                   size_t choices_size,
                   size_t choice_size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_choose_c<_DataType1, _DataType2>(
        q_ref, result1, array1_in, choices1, size, choices_size, choice_size,
        dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType1, typename _DataType2>
void (*dpnp_choose_default_c)(void *, void *, void **, size_t, size_t, size_t) =
    dpnp_choose_c<_DataType1, _DataType2>;

template <typename _DataType1, typename _DataType2>
DPCTLSyclEventRef (*dpnp_choose_ext_c)(DPCTLSyclQueueRef,
                                       void *,
                                       void *,
                                       void **,
                                       size_t,
                                       size_t,
                                       size_t,
                                       const DPCTLEventVectorRef) =
    dpnp_choose_c<_DataType1, _DataType2>;

template <typename _DataType>
DPCTLSyclEventRef
    dpnp_diag_indices_c(DPCTLSyclQueueRef q_ref,
                        void *result1,
                        size_t size,
                        const DPCTLEventVectorRef dep_event_vec_ref)
{
    return dpnp_arange_c<_DataType>(q_ref, 0, 1, result1, size,
                                    dep_event_vec_ref);
}

template <typename _DataType>
void dpnp_diag_indices_c(void *result1, size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref =
        dpnp_diag_indices_c<_DataType>(q_ref, result1, size, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_diag_indices_default_c)(void *,
                                    size_t) = dpnp_diag_indices_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_diag_indices_ext_c)(DPCTLSyclQueueRef,
                                             void *,
                                             size_t,
                                             const DPCTLEventVectorRef) =
    dpnp_diag_indices_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_diagonal_c(DPCTLSyclQueueRef q_ref,
                                  void *array1_in,
                                  const size_t input1_size,
                                  void *result1,
                                  const size_t offset,
                                  shape_elem_type *shape,
                                  shape_elem_type *res_shape,
                                  const size_t res_ndim,
                                  const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    const size_t res_size = std::accumulate(res_shape, res_shape + res_ndim, 1,
                                            std::multiplies<shape_elem_type>());
    if (!(res_size && input1_size)) {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));

    DPNPC_ptr_adapter<_DataType> input1_ptr(q_ref, array1_in, input1_size,
                                            true);
    DPNPC_ptr_adapter<_DataType> result_ptr(q_ref, result1, res_size, true,
                                            true);
    _DataType *array_1 = input1_ptr.get_ptr();
    _DataType *result = result_ptr.get_ptr();

    if (res_ndim <= 1) {
        for (size_t i = 0; i < static_cast<size_t>(res_shape[res_ndim - 1]);
             ++i) {
            result[i] = array_1[i * shape[res_ndim] + i + offset];
        }
    }
    else {
        std::map<size_t, std::vector<size_t>> xyz;
        for (size_t i = 0; i < static_cast<size_t>(res_shape[0]); i++) {
            xyz[i] = {i};
        }

        size_t index = 1;
        while (index < res_ndim - 1) {
            size_t shape_element = res_shape[index];
            std::map<size_t, std::vector<size_t>> new_shape_array;
            size_t ind = 0;
            for (size_t i = 0; i < shape_element; i++) {
                for (size_t j = 0; j < xyz.size(); j++) {
                    std::vector<size_t> new_shape;
                    std::vector<size_t> list_ind = xyz[j];
                    for (size_t k = 0; k < list_ind.size(); k++) {
                        new_shape.push_back(list_ind.at(k));
                    }
                    new_shape.push_back(i);
                    new_shape_array[ind] = new_shape;
                    ind += 1;
                }
            }
            size_t len_new_shape_array = new_shape_array.size() * (index + 1);

            for (size_t k = 0; k < len_new_shape_array; k++) {
                xyz[k] = new_shape_array[k];
            }
            index += 1;
        }

        for (size_t i = 0; i < static_cast<size_t>(res_shape[res_ndim - 1]);
             i++) {
            for (size_t j = 0; j < xyz.size(); j++) {
                std::vector<size_t> ind_list = xyz[j];
                if (ind_list.size() == 0) {
                    continue;
                }
                else {
                    size_t ind_input_size = ind_list.size() + 2;
                    size_t ind_input_[ind_input_size];
                    ind_input_[0] = i;
                    ind_input_[1] = i + offset;
                    size_t ind_output_size = ind_list.size() + 1;
                    size_t ind_output_[ind_output_size];
                    for (size_t k = 0; k < ind_list.size(); k++) {
                        ind_input_[k + 2] = ind_list.at(k);
                        ind_output_[k] = ind_list.at(k);
                    }
                    ind_output_[ind_list.size()] = i;

                    size_t ind_output = 0;
                    size_t n = 1;
                    for (size_t k = 0; k < ind_output_size; k++) {
                        size_t ind = ind_output_size - 1 - k;
                        ind_output += n * ind_output_[ind];
                        n *= res_shape[ind];
                    }

                    size_t ind_input = 0;
                    size_t m = 1;
                    for (size_t k = 0; k < ind_input_size; k++) {
                        size_t ind = ind_input_size - 1 - k;
                        ind_input += m * ind_input_[ind];
                        m *= shape[ind];
                    }

                    result[ind_output] = array_1[ind_input];
                }
            }
        }
    }

    return event_ref;
}

template <typename _DataType>
void dpnp_diagonal_c(void *array1_in,
                     const size_t input1_size,
                     void *result1,
                     const size_t offset,
                     shape_elem_type *shape,
                     shape_elem_type *res_shape,
                     const size_t res_ndim)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_diagonal_c<_DataType>(
        q_ref, array1_in, input1_size, result1, offset, shape, res_shape,
        res_ndim, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_diagonal_default_c)(void *,
                                const size_t,
                                void *,
                                const size_t,
                                shape_elem_type *,
                                shape_elem_type *,
                                const size_t) = dpnp_diagonal_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_diagonal_ext_c)(DPCTLSyclQueueRef,
                                         void *,
                                         const size_t,
                                         void *,
                                         const size_t,
                                         shape_elem_type *,
                                         shape_elem_type *,
                                         const size_t,
                                         const DPCTLEventVectorRef) =
    dpnp_diagonal_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef
    dpnp_fill_diagonal_c(DPCTLSyclQueueRef q_ref,
                         void *array1_in,
                         void *val_in,
                         shape_elem_type *shape,
                         const size_t ndim,
                         const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    const size_t result_size = std::accumulate(
        shape, shape + ndim, 1, std::multiplies<shape_elem_type>());
    if (!(result_size && array1_in)) {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));

    DPNPC_ptr_adapter<_DataType> result_ptr(q_ref, array1_in, result_size, true,
                                            true);
    DPNPC_ptr_adapter<_DataType> val_ptr(q_ref, val_in, 1, true);
    _DataType *array_1 = result_ptr.get_ptr();
    _DataType *val_arr = val_ptr.get_ptr();

    shape_elem_type min_shape = shape[0];
    for (size_t i = 0; i < ndim; ++i) {
        if (shape[i] < min_shape) {
            min_shape = shape[i];
        }
    }

    _DataType val = val_arr[0];

    for (size_t i = 0; i < static_cast<size_t>(min_shape); ++i) {
        size_t ind = 0;
        size_t n = 1;
        for (size_t k = 0; k < ndim; k++) {
            size_t ind_ = ndim - 1 - k;
            ind += n * i;
            n *= shape[ind_];
        }
        array_1[ind] = val;
    }

    return event_ref;
}

template <typename _DataType>
void dpnp_fill_diagonal_c(void *array1_in,
                          void *val_in,
                          shape_elem_type *shape,
                          const size_t ndim)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_fill_diagonal_c<_DataType>(
        q_ref, array1_in, val_in, shape, ndim, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_fill_diagonal_default_c)(void *,
                                     void *,
                                     shape_elem_type *,
                                     const size_t) =
    dpnp_fill_diagonal_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_fill_diagonal_ext_c)(DPCTLSyclQueueRef,
                                              void *,
                                              void *,
                                              shape_elem_type *,
                                              const size_t,
                                              const DPCTLEventVectorRef) =
    dpnp_fill_diagonal_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_nonzero_c(DPCTLSyclQueueRef q_ref,
                                 const void *in_array1,
                                 void *result1,
                                 const size_t result_size,
                                 const shape_elem_type *shape,
                                 const size_t ndim,
                                 const size_t j,
                                 const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if ((in_array1 == nullptr) || (result1 == nullptr)) {
        return event_ref;
    }

    if (ndim == 0) {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));

    const size_t input1_size = std::accumulate(
        shape, shape + ndim, 1, std::multiplies<shape_elem_type>());

    DPNPC_ptr_adapter<_DataType> input1_ptr(q_ref, in_array1, input1_size,
                                            true);
    DPNPC_ptr_adapter<long> result_ptr(q_ref, result1, result_size, true, true);
    const _DataType *arr = input1_ptr.get_ptr();
    long *result = result_ptr.get_ptr();

    size_t idx = 0;
    for (size_t i = 0; i < input1_size; ++i) {
        if (arr[i] != 0) {
            size_t ids[ndim];
            size_t ind1 = input1_size;
            size_t ind2 = i;
            for (size_t k = 0; k < ndim; ++k) {
                ind1 = ind1 / shape[k];
                ids[k] = ind2 / ind1;
                ind2 = ind2 % ind1;
            }

            result[idx] = ids[j];
            idx += 1;
        }
    }

    return event_ref;
}

template <typename _DataType>
void dpnp_nonzero_c(const void *in_array1,
                    void *result1,
                    const size_t result_size,
                    const shape_elem_type *shape,
                    const size_t ndim,
                    const size_t j)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref =
        dpnp_nonzero_c<_DataType>(q_ref, in_array1, result1, result_size, shape,
                                  ndim, j, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType>
void (*dpnp_nonzero_default_c)(const void *,
                               void *,
                               const size_t,
                               const shape_elem_type *,
                               const size_t,
                               const size_t) = dpnp_nonzero_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_place_c(DPCTLSyclQueueRef q_ref,
                               void *arr_in,
                               long *mask_in,
                               void *vals_in,
                               const size_t arr_size,
                               const size_t vals_size,
                               const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!arr_size) {
        return event_ref;
    }

    if (!vals_size) {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));

    DPNPC_ptr_adapter<_DataType> input1_ptr(q_ref, vals_in, vals_size, true);
    DPNPC_ptr_adapter<_DataType> result_ptr(q_ref, arr_in, arr_size, true,
                                            true);
    _DataType *vals = input1_ptr.get_ptr();
    _DataType *arr = result_ptr.get_ptr();

    DPNPC_ptr_adapter<long> mask_ptr(q_ref, mask_in, arr_size, true);
    long *mask = mask_ptr.get_ptr();

    size_t counter = 0;
    for (size_t i = 0; i < arr_size; ++i) {
        if (mask[i]) {
            arr[i] = vals[counter % vals_size];
            counter += 1;
        }
    }

    return event_ref;
}

template <typename _DataType>
void dpnp_place_c(void *arr_in,
                  long *mask_in,
                  void *vals_in,
                  const size_t arr_size,
                  const size_t vals_size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref =
        dpnp_place_c<_DataType>(q_ref, arr_in, mask_in, vals_in, arr_size,
                                vals_size, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType>
void (*dpnp_place_default_c)(void *,
                             long *,
                             void *,
                             const size_t,
                             const size_t) = dpnp_place_c<_DataType>;

template <typename _DataType, typename _IndecesType, typename _ValueType>
DPCTLSyclEventRef dpnp_put_c(DPCTLSyclQueueRef q_ref,
                             void *array1_in,
                             void *ind_in,
                             void *v_in,
                             const size_t size,
                             const size_t size_ind,
                             const size_t size_v,
                             const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if ((array1_in == nullptr) || (ind_in == nullptr) || (v_in == nullptr)) {
        return event_ref;
    }

    if (size_v == 0) {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));
    DPNPC_ptr_adapter<size_t> input1_ptr(q_ref, ind_in, size_ind, true);
    DPNPC_ptr_adapter<_DataType> input2_ptr(q_ref, v_in, size_v, true);
    DPNPC_ptr_adapter<_DataType> result_ptr(q_ref, array1_in, size, true, true);
    size_t *ind = input1_ptr.get_ptr();
    _DataType *v = input2_ptr.get_ptr();
    _DataType *array_1 = result_ptr.get_ptr();

    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size_ind; ++j) {
            if (i == ind[j] || (i == (size + ind[j]))) {
                array_1[i] = v[j % size_v];
            }
        }
    }

    return event_ref;
}

template <typename _DataType, typename _IndecesType, typename _ValueType>
void dpnp_put_c(void *array1_in,
                void *ind_in,
                void *v_in,
                const size_t size,
                const size_t size_ind,
                const size_t size_v)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref =
        dpnp_put_c<_DataType, _IndecesType, _ValueType>(
            q_ref, array1_in, ind_in, v_in, size, size_ind, size_v,
            dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType, typename _IndecesType, typename _ValueType>
void (*dpnp_put_default_c)(void *,
                           void *,
                           void *,
                           const size_t,
                           const size_t,
                           const size_t) =
    dpnp_put_c<_DataType, _IndecesType, _ValueType>;

template <typename _DataType>
DPCTLSyclEventRef
    dpnp_put_along_axis_c(DPCTLSyclQueueRef q_ref,
                          void *arr_in,
                          long *indices_in,
                          void *values_in,
                          size_t axis,
                          const shape_elem_type *shape,
                          size_t ndim,
                          size_t size_indices,
                          size_t values_size,
                          const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;
    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));

    size_t res_ndim = ndim - 1;
    size_t res_shape[res_ndim];
    const size_t size_arr = std::accumulate(shape, shape + ndim, 1,
                                            std::multiplies<shape_elem_type>());

    DPNPC_ptr_adapter<size_t> input1_ptr(q_ref, indices_in, size_indices, true);
    DPNPC_ptr_adapter<_DataType> input2_ptr(q_ref, values_in, values_size,
                                            true);
    DPNPC_ptr_adapter<_DataType> result_ptr(q_ref, arr_in, size_arr, true,
                                            true);
    size_t *indices = input1_ptr.get_ptr();
    _DataType *values = input2_ptr.get_ptr();
    _DataType *arr = result_ptr.get_ptr();

    if (axis != res_ndim) {
        int ind = 0;
        for (size_t i = 0; i < ndim; i++) {
            if (axis != i) {
                res_shape[ind] = shape[i];
                ind++;
            }
        }

        size_t prod = 1;
        for (size_t i = 0; i < res_ndim; ++i) {
            if (res_shape[i] != 0) {
                prod *= res_shape[i];
            }
        }

        size_t ind_array[prod];
        bool bool_ind_array[prod];
        for (size_t i = 0; i < prod; ++i) {
            bool_ind_array[i] = true;
        }
        size_t arr_shape_offsets[ndim];
        size_t acc = 1;
        for (size_t i = ndim - 1; i > 0; --i) {
            arr_shape_offsets[i] = acc;
            acc *= shape[i];
        }
        arr_shape_offsets[0] = acc;

        size_t output_shape_offsets[res_ndim];
        acc = 1;
        if (res_ndim > 0) {
            for (size_t i = res_ndim - 1; i > 0; --i) {
                output_shape_offsets[i] = acc;
                acc *= res_shape[i];
            }
        }
        output_shape_offsets[0] = acc;

        size_t size_result = 1;
        for (size_t i = 0; i < res_ndim; ++i) {
            size_result *= res_shape[i];
        }

        // init result array
        for (size_t result_idx = 0; result_idx < size_result; ++result_idx) {
            size_t xyz[res_ndim];
            size_t remainder = result_idx;
            for (size_t i = 0; i < res_ndim; ++i) {
                xyz[i] = remainder / output_shape_offsets[i];
                remainder = remainder - xyz[i] * output_shape_offsets[i];
            }

            size_t source_axis[ndim];
            size_t result_axis_idx = 0;
            for (size_t idx = 0; idx < ndim; ++idx) {
                bool found = false;
                if (axis == idx) {
                    found = true;
                }
                if (found) {
                    source_axis[idx] = 0;
                }
                else {
                    source_axis[idx] = xyz[result_axis_idx];
                    result_axis_idx++;
                }
            }

            // FIXME: computed, but unused. Commented out per compiler warning
            // size_t source_idx = 0;
            // for (size_t i = 0; i < static_cast<size_t>(ndim); ++i)
            // {
            //   source_idx += arr_shape_offsets[i] * source_axis[i];
            // }
        }

        for (size_t source_idx = 0; source_idx < size_arr; ++source_idx) {
            // reconstruct x,y,z from linear source_idx
            size_t xyz[ndim];
            size_t remainder = source_idx;
            for (size_t i = 0; i < ndim; ++i) {
                xyz[i] = remainder / arr_shape_offsets[i];
                remainder = remainder - xyz[i] * arr_shape_offsets[i];
            }

            // extract result axis
            size_t result_axis[res_ndim];
            size_t result_idx = 0;
            for (size_t idx = 0; idx < ndim; ++idx) {
                // try to find current idx in axis array
                bool found = false;
                if (axis == idx) {
                    found = true;
                }
                if (!found) {
                    result_axis[result_idx] = xyz[idx];
                    result_idx++;
                }
            }

            // Construct result offset
            size_t result_offset = 0;
            for (size_t i = 0; i < res_ndim; ++i) {
                result_offset += output_shape_offsets[i] * result_axis[i];
            }

            if (bool_ind_array[result_offset]) {
                ind_array[result_offset] = 0;
                bool_ind_array[result_offset] = false;
            }
            else {
                ind_array[result_offset] += 1;
            }

            if ((ind_array[result_offset] % size_indices) ==
                indices[result_offset % size_indices])
            {
                arr[source_idx] = values[source_idx % values_size];
            }
        }
    }
    else {
        for (size_t i = 0; i < size_arr; ++i) {
            size_t ind =
                size_indices * (i / size_indices) + indices[i % size_indices];
            arr[ind] = values[i % values_size];
        }
    }
    return event_ref;
}

template <typename _DataType>
void dpnp_put_along_axis_c(void *arr_in,
                           long *indices_in,
                           void *values_in,
                           size_t axis,
                           const shape_elem_type *shape,
                           size_t ndim,
                           size_t size_indices,
                           size_t values_size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_put_along_axis_c<_DataType>(
        q_ref, arr_in, indices_in, values_in, axis, shape, ndim, size_indices,
        values_size, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_put_along_axis_default_c)(void *,
                                      long *,
                                      void *,
                                      size_t,
                                      const shape_elem_type *,
                                      size_t,
                                      size_t,
                                      size_t) =
    dpnp_put_along_axis_c<_DataType>;

template <typename _DataType, typename _IndecesType>
class dpnp_take_c_kernel;

template <typename _DataType, typename _IndecesType>
DPCTLSyclEventRef dpnp_take_c(DPCTLSyclQueueRef q_ref,
                              void *array1_in,
                              const size_t array1_size,
                              void *indices1,
                              void *result1,
                              size_t size,
                              const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)array1_size;
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;
    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));

    _DataType *array_1 = reinterpret_cast<_DataType *>(array1_in);
    _IndecesType *indices = reinterpret_cast<_IndecesType *>(indices1);
    _DataType *result = reinterpret_cast<_DataType *>(result1);

    sycl::range<1> gws(size);
    auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
        const size_t idx = global_id[0];
        result[idx] = array_1[indices[idx]];
    };

    auto kernel_func = [&](sycl::handler &cgh) {
        cgh.parallel_for<class dpnp_take_c_kernel<_DataType, _IndecesType>>(
            gws, kernel_parallel_for_func);
    };

    sycl::event event = q.submit(kernel_func);

    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);
    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType, typename _IndecesType>
void dpnp_take_c(void *array1_in,
                 const size_t array1_size,
                 void *indices1,
                 void *result1,
                 size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_take_c<_DataType, _IndecesType>(
        q_ref, array1_in, array1_size, indices1, result1, size,
        dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType, typename _IndecesType>
void (*dpnp_take_default_c)(void *, const size_t, void *, void *, size_t) =
    dpnp_take_c<_DataType, _IndecesType>;

template <typename _DataType, typename _IndecesType>
DPCTLSyclEventRef (*dpnp_take_ext_c)(DPCTLSyclQueueRef,
                                     void *,
                                     const size_t,
                                     void *,
                                     void *,
                                     size_t,
                                     const DPCTLEventVectorRef) =
    dpnp_take_c<_DataType, _IndecesType>;

void func_map_init_indexing_func(func_map_t &fmap)
{
    fmap[DPNPFuncName::DPNP_FN_CHOOSE][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_choose_default_c<int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_CHOOSE][eft_INT][eft_LNG] = {
        eft_LNG, (void *)dpnp_choose_default_c<int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CHOOSE][eft_INT][eft_FLT] = {
        eft_FLT, (void *)dpnp_choose_default_c<int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_CHOOSE][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_choose_default_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_CHOOSE][eft_LNG][eft_INT] = {
        eft_INT, (void *)dpnp_choose_default_c<int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_CHOOSE][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_choose_default_c<int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CHOOSE][eft_LNG][eft_FLT] = {
        eft_FLT, (void *)dpnp_choose_default_c<int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_CHOOSE][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_choose_default_c<int64_t, double>};

    fmap[DPNPFuncName::DPNP_FN_CHOOSE_EXT][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_choose_ext_c<int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_CHOOSE_EXT][eft_INT][eft_LNG] = {
        eft_LNG, (void *)dpnp_choose_ext_c<int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CHOOSE_EXT][eft_INT][eft_FLT] = {
        eft_FLT, (void *)dpnp_choose_ext_c<int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_CHOOSE_EXT][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_choose_ext_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_CHOOSE_EXT][eft_LNG][eft_INT] = {
        eft_INT, (void *)dpnp_choose_ext_c<int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_CHOOSE_EXT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_choose_ext_c<int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CHOOSE_EXT][eft_LNG][eft_FLT] = {
        eft_FLT, (void *)dpnp_choose_ext_c<int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_CHOOSE_EXT][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_choose_ext_c<int64_t, double>};

    fmap[DPNPFuncName::DPNP_FN_DIAG_INDICES][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_diag_indices_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_DIAG_INDICES][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_diag_indices_default_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_DIAG_INDICES][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_diag_indices_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_DIAG_INDICES][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_diag_indices_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_DIAG_INDICES_EXT][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_diag_indices_ext_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_DIAG_INDICES_EXT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_diag_indices_ext_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_DIAG_INDICES_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_diag_indices_ext_c<float>};
    fmap[DPNPFuncName::DPNP_FN_DIAG_INDICES_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_diag_indices_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_DIAGONAL][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_diagonal_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_DIAGONAL][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_diagonal_default_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_DIAGONAL][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_diagonal_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_DIAGONAL][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_diagonal_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_DIAGONAL_EXT][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_diagonal_ext_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_DIAGONAL_EXT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_diagonal_ext_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_DIAGONAL_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_diagonal_ext_c<float>};
    fmap[DPNPFuncName::DPNP_FN_DIAGONAL_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_diagonal_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_FILL_DIAGONAL][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_fill_diagonal_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_FILL_DIAGONAL][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_fill_diagonal_default_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_FILL_DIAGONAL][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_fill_diagonal_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_FILL_DIAGONAL][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_fill_diagonal_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_FILL_DIAGONAL_EXT][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_fill_diagonal_ext_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_FILL_DIAGONAL_EXT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_fill_diagonal_ext_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_FILL_DIAGONAL_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_fill_diagonal_ext_c<float>};
    fmap[DPNPFuncName::DPNP_FN_FILL_DIAGONAL_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_fill_diagonal_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_NONZERO][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_nonzero_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_NONZERO][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_nonzero_default_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_NONZERO][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_nonzero_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_NONZERO][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_nonzero_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_PLACE][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_place_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_PLACE][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_place_default_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_PLACE][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_place_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_PLACE][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_place_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_PUT][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_put_default_c<int32_t, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_PUT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_put_default_c<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_PUT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_put_default_c<float, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_PUT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_put_default_c<double, int64_t, double>};

    fmap[DPNPFuncName::DPNP_FN_PUT_ALONG_AXIS][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_put_along_axis_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_PUT_ALONG_AXIS][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_put_along_axis_default_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_PUT_ALONG_AXIS][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_put_along_axis_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_PUT_ALONG_AXIS][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_put_along_axis_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_TAKE][eft_BLN][eft_INT] = {
        eft_BLN, (void *)dpnp_take_default_c<bool, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_TAKE][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_take_default_c<int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_TAKE][eft_LNG][eft_INT] = {
        eft_LNG, (void *)dpnp_take_default_c<int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_TAKE][eft_FLT][eft_INT] = {
        eft_FLT, (void *)dpnp_take_default_c<float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_TAKE][eft_DBL][eft_INT] = {
        eft_DBL, (void *)dpnp_take_default_c<double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_TAKE][eft_C128][eft_INT] = {
        eft_C128, (void *)dpnp_take_default_c<std::complex<double>, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_TAKE][eft_BLN][eft_LNG] = {
        eft_BLN, (void *)dpnp_take_default_c<bool, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_TAKE][eft_INT][eft_LNG] = {
        eft_INT, (void *)dpnp_take_default_c<int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_TAKE][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_take_default_c<int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_TAKE][eft_FLT][eft_LNG] = {
        eft_FLT, (void *)dpnp_take_default_c<float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_TAKE][eft_DBL][eft_LNG] = {
        eft_DBL, (void *)dpnp_take_default_c<double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_TAKE][eft_C128][eft_LNG] = {
        eft_C128, (void *)dpnp_take_default_c<std::complex<double>, int64_t>};

    return;
}
