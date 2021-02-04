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
#include <iostream>
#include <iterator>
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

    DPNP_USM_iterator(pointer __ptr,
                      bool __axis_use = false,
                      size_type __axis = 0,
                      const std::vector<size_type>& __shape_pitch = {})
        : data(__ptr)
    {
        if (__axis_use)
        {
            axis_use = __axis_use;
            axis = __axis;
            shape_pitch = __shape_pitch;
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
        size_type stride = 1;
        if (axis_use)
        {
            stride = shape_pitch[axis];
        }

        data += stride;

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
        return data - __rhs.data;
    }

    /// Operator needs to print this container in human readable form in error reporting
    friend std::ostream& operator<<(std::ostream& __out, const DPNP_USM_iterator& __it)
    {
        __out << "DPNP_USM_iterator(data:" << __it.data << ", shape_pitch=" << __it.shape_pitch
              << ", axis_use=" << __it.axis_use << ", axis=" << __it.axis << ")";

        return __out;
    }

private:
    pointer data;
    std::vector<size_type> shape_pitch; // TODO needs to be replaced by sycl memory allocation
    size_type axis = 0;                 // TODO it should be a vector to support "axes" parameters
    bool axis_use = false;              // TODO it looks like it should be eliminated
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
            shape = __shape;
            shape_pitch.resize(__shape.size());
            get_shape_offsets_inkernel<size_type>(__shape.data(), __shape.size(), shape_pitch.data());
            size = std::accumulate(__shape.begin(), __shape.end(), size_type(1), std::multiplies<size_type>());
        }
    }

    DPNPC_id() = delete;

    /// this function is designed for host execution
    inline void set_axis(size_type __axis)
    {
        if (__axis < shape.size())
        {
            axis = __axis;
            axis_use = true;
        }
        // TODO exception if wrong axis? need common function for throwing exceptions
        // TODO need conversion from negative axis to positive one
    }

    /// this function is designed for SYCL environment execution
    inline iterator begin(size_type output_global_id = 0) const
    {
        return iterator(data + get_input_begin_offset(output_global_id), axis_use, axis, shape_pitch);
    }

    /// this function is designed for SYCL environment execution
    inline iterator end(size_type output_global_id = 0) const
    {
        // TODO it is better to get begin() iterator as a parameter

        return iterator(data + get_input_begin_offset(output_global_id) + get_input_end_length());
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
        if (!axis_use)
        {
            return 0;
        }

        std::vector<size_type> output_shape = shape;
        output_shape.erase(output_shape.begin() + axis);

        std::vector<size_type> output_strides = output_shape;
        output_strides.resize(output_shape.size());
        get_shape_offsets_inkernel<size_type>(output_shape.data(), output_shape.size(), output_strides.data());

        std::vector<size_type> output_xyz = output_shape;
        output_xyz.resize(output_shape.size());
        get_xyz_by_id_inkernel(output_global_id, output_strides.data(), output_strides.size(), output_xyz.data());

        std::vector<size_type> input_xyz = output_xyz;
        input_xyz.insert(input_xyz.begin() + axis, 0); // put begin point of the axis

        size_type input_global_id = get_id_by_xyz_inkernel(input_xyz.data(), input_xyz.size(), shape_pitch.data());

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
        const size_type dim_pitch = shape_pitch[axis];

        return (dim_size * dim_pitch);
    }

    pointer data = nullptr;
    size_type size = size_type{};
    std::vector<size_type> shape;
    std::vector<size_type> shape_pitch;

    size_type axis = 0;
    bool axis_use = false;
};

#endif // DPNP_ITERATOR_H
