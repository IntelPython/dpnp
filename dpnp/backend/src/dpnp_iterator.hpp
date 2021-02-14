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

#pragma once
#ifndef DPNP_ITERATOR_H // Cython compatibility
#define DPNP_ITERATOR_H

#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>

#include <dpnp_utils.hpp>

/**
 * @ingroup BACKEND_UTILS
 * @brief Iterator for @ref DPNPC_id type
 *
 * This type should be used to simplify data iteraton over input with parameters "[axis|axes]"
 * It is designed to be used in SYCL environment
 *
 */
template <typename _Tp>
class DPNP_USM_iterator final
{
public:
    using value_type = _Tp;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;
    using pointer = value_type*;
    using reference = value_type&;
    using size_type = size_t;

    DPNP_USM_iterator(pointer __ptr, bool __axis_use = false, size_type __axis = 0, const size_type* __stride = nullptr)
        : data(__ptr)
    {
        if (__axis_use)
        {
            axis_use = __axis_use;
            axis = __axis;
            stride = __stride;
        }
    }

    DPNP_USM_iterator() = delete;

    inline reference operator*() const
    {
        return *data;
    }

    inline pointer operator->() const
    {
        return data;
    }

    /// prefix increment
    inline DPNP_USM_iterator& operator++()
    {
        size_type axis_stride = 1;
        if (axis_use)
        {
            axis_stride = stride[axis];
        }

        data += axis_stride;

        return *this;
    }

    /// postfix increment
    inline DPNP_USM_iterator operator++(int)
    {
        DPNP_USM_iterator tmp = *this;
        ++(*this); // call prefix increment

        return tmp;
    }

    inline bool operator==(const DPNP_USM_iterator& __rhs) const
    {
        return (data == __rhs.data);
    };

    inline bool operator!=(const DPNP_USM_iterator& __rhs) const
    {
        return (data != __rhs.data);
    };

    inline bool operator<(const DPNP_USM_iterator& __rhs) const
    {
        return (data < __rhs.data);
    };

    // TODO need more operators

    inline difference_type operator-(const DPNP_USM_iterator& __rhs) const
    {
        size_type axis_stride = 1;
        if (axis_use)
        {
            axis_stride = stride[axis];
        }

        difference_type linear_diff = data - __rhs.data;
        difference_type elements_in_stride = axis_stride; // potential issue with unsigned conversion to signed

        return linear_diff / elements_in_stride;
    }

    /// Operator needs to print this container in human readable form in error reporting
    friend std::ostream& operator<<(std::ostream& __out, const DPNP_USM_iterator& __it)
    {
        __out << "DPNP_USM_iterator(data:" << __it.data << ", shape_pitch=" << __it.stride
              << ", axis_use=" << __it.axis_use << ", axis=" << __it.axis << ")";

        return __out;
    }

private:
    pointer data = nullptr;
    const size_type* stride = nullptr;
    size_type axis = 0;    // TODO it should be a vector to support "axes" parameters
    bool axis_use = false; // TODO it looks like it should be eliminated
};

/**
 * @ingroup BACKEND_UTILS
 * @brief Type to keep USM array pointers used in kernels
 *
 * This type should be used in host part of the code to provide pre-calculated data. The @ref DPNP_USM_iterator
 * will be used later in SYCL environment
 *
 */
template <typename _Tp>
class DPNPC_id final
{
public:
    using value_type = _Tp;
    using iterator = DPNP_USM_iterator<value_type>;
    using pointer = value_type*;
    using reference = value_type&;
    using size_type = size_t;

    /// this function is designed for host execution
    DPNPC_id(void* __ptr, const std::vector<size_type>& __shape)
    {
        data = reinterpret_cast<pointer>(__ptr);
        if (!__shape.empty())
        {
            shape_size = __shape.size();
            shape = reinterpret_cast<size_type*>(dpnp_memory_alloc_c(shape_size * sizeof(size_type)));
            std::copy(__shape.begin(), __shape.end(), shape);

            shape_strides = reinterpret_cast<size_type*>(dpnp_memory_alloc_c(shape_size * sizeof(size_type)));
            get_shape_offsets_inkernel<size_type>(shape, shape_size, shape_strides);

            size = std::accumulate(__shape.begin(), __shape.end(), size_type(1), std::multiplies<size_type>());
            if (size)
            {
                output_size = 1; // if input size is not zero it means we will have at least scalar as output
            }
        }
    }

    DPNPC_id() = delete;

    ~DPNPC_id()
    {
        dpnp_memory_free_c(shape);
        dpnp_memory_free_c(shape_strides);
        dpnp_memory_free_c(output_shape);
        dpnp_memory_free_c(output_shape_strides);
        dpnp_memory_free_c(sycl_output_xyz);
        dpnp_memory_free_c(sycl_input_xyz);
    }

    /// this function return number of elements in output
    inline size_type get_output_size() const
    {
        // TODO if axis is not set need to return input array size
        return output_size;
    }

    /// this function is designed for host execution
    inline void set_axis(size_type __axis)
    {
        if (__axis < shape_size)
        {
            axis = __axis;
            axis_use = true;

            output_shape_size = shape_size - 1;
            const size_type output_shape_size_in_bytes = output_shape_size * sizeof(size_type);

            output_shape = reinterpret_cast<size_type*>(dpnp_memory_alloc_c(output_shape_size_in_bytes));
            size_type* output_shape_it = std::copy(shape, shape + axis, output_shape);
            std::copy(shape + axis + 1, shape + shape_size, output_shape_it);

            output_size = std::accumulate(
                output_shape, output_shape + output_shape_size, size_type(1), std::multiplies<size_type>());

            output_shape_strides = reinterpret_cast<size_type*>(dpnp_memory_alloc_c(output_shape_size_in_bytes));
            get_shape_offsets_inkernel<size_type>(output_shape, output_shape_size, output_shape_strides);

            // make thread private storage for each shape by multiplying memory
            sycl_output_xyz =
                reinterpret_cast<size_type*>(dpnp_memory_alloc_c(output_size * output_shape_size_in_bytes));
            sycl_input_xyz =
                reinterpret_cast<size_type*>(dpnp_memory_alloc_c(output_size * shape_size * sizeof(size_type)));
        }
        // TODO exception if wrong axis? need common function for throwing exceptions
        // TODO need conversion from negative axis to positive one
    }

    /// this function is designed for SYCL environment execution
    inline iterator begin(size_type output_global_id = 0) const
    {
        return iterator(data + get_input_begin_offset(output_global_id), axis_use, axis, shape_strides);
    }

    /// this function is designed for SYCL environment execution
    inline iterator end(size_type output_global_id = 0) const
    {
        // TODO it is better to get begin() iterator as a parameter

        return iterator(
            data + get_input_begin_offset(output_global_id) + get_input_end_length(), axis_use, axis, shape_strides);
    }

    /// this function is designed for SYCL environment execution
    inline reference operator[](size_type __n) const
    {
        return *(data + __n); // TODO take care about shape and axis
    }

private:
    /// this function is designed for SYCL environment execution
    size_type get_input_begin_offset(size_type output_global_id) const
    {
        size_type input_global_id = 0;
        if (axis_use)
        {
            assert(output_global_id < output_size);

            // use thread private storage
            size_type* sycl_output_xyz_thread = sycl_output_xyz + (output_global_id * output_shape_size);
            size_type* sycl_input_xyz_thread = sycl_input_xyz + (output_global_id * shape_size);

            get_xyz_by_id_inkernel(output_global_id, output_shape_strides, output_shape_size, sycl_output_xyz_thread);

            for (size_t iit = 0, oit = 0; iit < shape_size; ++iit)
            {
                if (iit == axis)
                {
                    sycl_input_xyz_thread[iit] = 0; // put begin point of the axis
                }
                else
                {
                    sycl_input_xyz_thread[iit] = sycl_output_xyz_thread[oit];
                    ++oit;
                }
            }

            input_global_id = get_id_by_xyz_inkernel(sycl_input_xyz_thread, shape_size, shape_strides);
        }

        return input_global_id;
    }

    /// this function is designed for SYCL environment execution
    size_type get_input_end_length() const
    {
        if (!axis_use)
        {
            return size;
        }

        const size_type dim_size = shape[axis];
        const size_type dim_pitch = shape_strides[axis];

        return (dim_size * dim_pitch);
    }

    pointer data = nullptr;       /**< array begin pointer */
    size_type size = size_type{}; /**< array size */

    size_type* shape = nullptr;         /**< array shape */
    size_type shape_size = size_type{}; /**< array shape size */
    size_type* shape_strides = nullptr; /**< array shape strides (same size as shape array) */

    size_type axis = 0; /**< reduction axis (negative unsupported) */
    bool axis_use = false;

    size_type output_size = size_type{}; /**< output array size. Expected is same as GWS */
    size_type* output_shape = nullptr;
    size_type output_shape_size = size_type{};
    size_type* output_shape_strides = nullptr;

    // data allocated to use inside SYCL kernels
    size_type* sycl_output_xyz = nullptr;
    size_type* sycl_input_xyz = nullptr;
};

#endif // DPNP_ITERATOR_H
