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

    DPNP_USM_iterator(pointer __base_ptr,
                      size_type __id,
                      const size_type* __shape_stride = nullptr,
                      const size_type* __axes_stride = nullptr,
                      size_type __shape_size = 0)
        : base(__base_ptr)
        , iter_id(__id)
        , iteration_shape_size(__shape_size)
        , iteration_shape_strides(__shape_stride)
        , axes_shape_strides(__axes_stride)
    {
    }

    DPNP_USM_iterator() = delete;

    inline reference operator*() const
    {
        return *ptr();
    }

    inline pointer operator->() const
    {
        return ptr();
    }

    /// prefix increment
    inline DPNP_USM_iterator& operator++()
    {
        ++iter_id;

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
        assert(base == __rhs.base); // iterators are incomparable if base pointers are different
        return (iter_id == __rhs.iter_id);
    };

    inline bool operator!=(const DPNP_USM_iterator& __rhs) const
    {
        return !(*this == __rhs);
    };

    inline bool operator<(const DPNP_USM_iterator& __rhs) const
    {
        return iter_id < __rhs.iter_id;
    };

    // TODO need more operators

    // Random access iterator requirements
    inline reference operator[](size_type __n) const
    {
        return *ptr(__n);
    }

    inline difference_type operator-(const DPNP_USM_iterator& __rhs) const
    {
        difference_type diff = difference_type(iter_id) - difference_type(__rhs.iter_id);

        return diff;
    }

    /// Print this container in human readable form in error reporting
    friend std::ostream& operator<<(std::ostream& __out, const DPNP_USM_iterator& __it)
    {
        const std::vector<size_type> it_strides(__it.iteration_shape_strides,
                                                __it.iteration_shape_strides + __it.iteration_shape_size);
        const std::vector<size_type> it_axes_strides(__it.axes_shape_strides,
                                                     __it.axes_shape_strides + __it.iteration_shape_size);

        __out << "DPNP_USM_iterator(base=" << __it.base;
        __out << ", iter_id=" << __it.iter_id;
        __out << ", iteration_shape_size=" << __it.iteration_shape_size;
        __out << ", iteration_shape_strides=" << it_strides;
        __out << ", axes_shape_strides=" << it_axes_strides;
        __out << ")";

        return __out;
    }

private:
    inline pointer ptr() const
    {
        return ptr(iter_id);
    }

    inline pointer ptr(size_type iteration_id) const
    {
        size_type offset = 0;

        if (iteration_shape_size > 0)
        {
            long reminder = iteration_id;
            for (size_t it = 0; it < iteration_shape_size; ++it)
            {
                const size_type axis_val = iteration_shape_strides[it];
                size_type xyz_id = reminder / axis_val;
                offset += (xyz_id * axes_shape_strides[it]);

                reminder = reminder % axis_val;
            }
        }
        else
        {
            offset = iteration_id;
        }

        return base + offset;
    }

    const pointer base = nullptr;
    size_type iter_id = size_type{};                    /**< Iterator logical ID over iteration shape */
    const size_type iteration_shape_size = size_type{}; /**< Number of elements in @ref iteration_shape_strides array */
    const size_type* iteration_shape_strides = nullptr;
    const size_type* axes_shape_strides = nullptr;
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

    DPNPC_id(pointer __ptr, const size_type* __shape, const size_type __shape_size)
    {
        std::vector<size_type> shape(__shape, __shape + __shape_size);
        init_container(__ptr, shape);
    }

    /**
     * @ingroup BACKEND_UTILS
     * @brief Main container for reduction iterator
     *
     * Construct object to hold @ref __ptr data with shape @ref __shape.
     * It is needed to provide reduction iterator over the data.
     *
     * @note this function is designed for non-SYCL environment execution
     *
     * @param [in]  __ptr    Pointer to input data. Used to get values only.
     * @param [in]  __shape  Shape of data provided by @ref __ptr.
     *                       Empty container means scalar value pointed by @ref __ptr.
     */
    DPNPC_id(pointer __ptr, const std::vector<size_type>& __shape)
    {
        init_container(__ptr, __shape);
    }

    DPNPC_id() = delete;

    ~DPNPC_id()
    {
        free_memory();
    }

    /// this function return number of elements in output
    inline size_type get_output_size() const
    {
        return output_size;
    }

    /**
     * @ingroup BACKEND_UTILS
     * @brief Set axis for the data object to use in computation.
     *
     * Set axis of the shape of input array to use in iteration.
     * Axis might be negative to indicate right to left axes indexing in a shape
     *
     * Indexing goes from left to right.
     * Reduction example:
     *   Input shape A[6, 7, 8, 9]
     *   set_axis(1)                       // same as -3 in this example
     *   output shape will be C[6, 8, 9]
     *
     * @note this function is designed for non-SYCL environment execution
     *
     * @param [in]  __axis    Axis in a shape of input array.
     */
    inline void set_axis(long __axis)
    {
        set_axes({__axis});
    }

    inline void set_axes(const long* __axes, const size_t axes_ndim)
    {
        const std::vector<long> axes_vec(__axes, __axes + axes_ndim);
        set_axes(axes_vec);
    }

    /**
     * @ingroup BACKEND_UTILS
     * @brief Set axes for the data object to use in computation.
     *
     * Set axes of the shape of input array to use in iteration.
     * Axes might be negative to indicate axes as reverse iterator
     *
     * Indexing goes from left to right.
     * Reduction example:
     *   Input shape A[2, 3, 4, 5]
     *   set_axes({1, 2})                 // same as {-3, -2}
     *   output shape will be C[2, 5]
     *
     * @note this function is designed for non-SYCL environment execution
     *
     * @param [in]  __axes       Vector of axes of a shape of input array.
     */
    inline void set_axes(const std::vector<long>& __axes)
    {
        if (!__axes.empty() && input_shape_size)
        {
            axes = get_validated_axes(__axes, input_shape_size);
            axis_use = true;

            output_shape_size = input_shape_size - axes.size();
            const size_type output_shape_size_in_bytes = output_shape_size * sizeof(size_type);

            iteration_shape_size = axes.size();
            const size_type iteration_shape_size_in_bytes = iteration_shape_size * sizeof(size_type);
            std::vector<size_type> iteration_shape;

            output_shape = reinterpret_cast<size_type*>(dpnp_memory_alloc_c(output_shape_size_in_bytes));
            size_type* output_shape_it = output_shape;
            for (size_type i = 0; i < input_shape_size; ++i)
            {
                if (std::find(axes.begin(), axes.end(), i) == axes.end())
                {
                    *output_shape_it = input_shape[i];
                    ++output_shape_it;
                }
            }

            output_size = std::accumulate(
                output_shape, output_shape + output_shape_size, size_type(1), std::multiplies<size_type>());

            output_shape_strides = reinterpret_cast<size_type*>(dpnp_memory_alloc_c(output_shape_size_in_bytes));
            get_shape_offsets_inkernel<size_type>(output_shape, output_shape_size, output_shape_strides);

            iteration_size = 1;
            iteration_shape.reserve(iteration_shape_size);
            for (const auto& axis : axes)
            {
                const size_type axis_dim = input_shape[axis];
                iteration_shape.push_back(axis_dim);
                iteration_size *= axis_dim;
            }

            iteration_shape_strides = reinterpret_cast<size_type*>(dpnp_memory_alloc_c(iteration_shape_size_in_bytes));
            get_shape_offsets_inkernel<size_type>(
                iteration_shape.data(), iteration_shape.size(), iteration_shape_strides);

            axes_shape_strides = reinterpret_cast<size_type*>(dpnp_memory_alloc_c(iteration_shape_size_in_bytes));
            for (size_t i = 0; i < iteration_shape_size; ++i)
            {
                axes_shape_strides[i] = input_shape_strides[axes[i]];
            }

            // make thread private storage for each shape by multiplying memory
            sycl_output_xyz =
                reinterpret_cast<size_type*>(dpnp_memory_alloc_c(output_size * output_shape_size_in_bytes));
        }
    }

    /// this function is designed for SYCL environment execution
    inline iterator begin(size_type output_global_id = 0) const
    {
        return iterator(data + get_input_begin_offset(output_global_id),
                        0,
                        iteration_shape_strides,
                        axes_shape_strides,
                        iteration_shape_size);
    }

    /// this function is designed for SYCL environment execution
    inline iterator end(size_type output_global_id = 0) const
    {
        // TODO it is better to get begin() iterator as a parameter

        return iterator(data + get_input_begin_offset(output_global_id),
                        get_iteration_size(),
                        iteration_shape_strides,
                        axes_shape_strides,
                        iteration_shape_size);
    }

    /// this function is designed for SYCL environment execution
    inline reference operator[](size_type __n) const
    {
        const iterator it = begin();
        return it[__n];
    }

private:
    void init_container(pointer __ptr, const std::vector<size_type>& __shape)
    {
        // TODO needs to address negative values in __shape with exception
        if ((__ptr == nullptr) && __shape.empty())
        {
            return;
        }

        if (__ptr != nullptr)
        {
            data = __ptr;
            input_size = 1;  // means scalar at this stage
            output_size = 1; // if input size is not zero it means we have scalar as output
            iteration_size = 1;
        }

        if (!__shape.empty())
        {
            input_size = std::accumulate(__shape.begin(), __shape.end(), size_type(1), std::multiplies<size_type>());
            if (input_size == 0)
            {                    // shape might be shape[3, 4, 0, 6]. This means no input memory and no output expected
                output_size = 0; // depends on axes. zero at this stage only
            }

            input_shape_size = __shape.size();
            input_shape = reinterpret_cast<size_type*>(dpnp_memory_alloc_c(input_shape_size * sizeof(size_type)));
            std::copy(__shape.begin(), __shape.end(), input_shape);

            input_shape_strides =
                reinterpret_cast<size_type*>(dpnp_memory_alloc_c(input_shape_size * sizeof(size_type)));
            get_shape_offsets_inkernel<size_type>(input_shape, input_shape_size, input_shape_strides);
        }
        iteration_size = input_size;
    }

    /// this function is designed for SYCL environment execution
    size_type get_input_begin_offset(size_type output_global_id) const
    {
        size_type input_global_id = 0;
        if (axis_use)
        {
            assert(output_global_id < output_size);

            // use thread private storage
            size_type* sycl_output_xyz_thread = sycl_output_xyz + (output_global_id * output_shape_size);

            get_xyz_by_id_inkernel(output_global_id, output_shape_strides, output_shape_size, sycl_output_xyz_thread);

            for (size_t iit = 0, oit = 0; iit < input_shape_size; ++iit)
            {
                if (std::find(axes.begin(), axes.end(), iit) == axes.end())
                {
                    input_global_id += (sycl_output_xyz_thread[oit] * input_shape_strides[iit]);
                    ++oit;
                }
            }
        }

        return input_global_id;
    }

    /// this function is designed for SYCL environment execution
    size_type get_iteration_size() const
    {
        return iteration_size;
    }

    void free_memory()
    {
        dpnp_memory_free_c(input_shape);
        dpnp_memory_free_c(input_shape_strides);
        dpnp_memory_free_c(output_shape);
        dpnp_memory_free_c(output_shape_strides);
        dpnp_memory_free_c(iteration_shape_strides);
        dpnp_memory_free_c(axes_shape_strides);
        dpnp_memory_free_c(sycl_output_xyz);
    }

    pointer data = nullptr;                   /**< input array begin pointer */
    size_type input_size = size_type{};       /**< input array size */
    size_type* input_shape = nullptr;         /**< input array shape */
    size_type input_shape_size = size_type{}; /**< input array shape size */
    size_type* input_shape_strides = nullptr; /**< input array shape strides (same size as input_shape) */

    std::vector<size_type> axes; /**< input shape reduction axes */
    bool axis_use = false;

    size_type output_size = size_type{};       /**< output array size. Expected is same as GWS */
    size_type* output_shape = nullptr;         /**< output array shape */
    size_type output_shape_size = size_type{}; /**< output array shape size */
    size_type* output_shape_strides = nullptr; /**< output array shape strides (same size as output_shape) */

    size_type iteration_size = size_type{}; /**< iteration array size in elements */
    size_type iteration_shape_size = size_type{};
    size_type* iteration_shape_strides = nullptr;
    size_type* axes_shape_strides = nullptr;

    // data allocated to use inside SYCL kernels
    size_type* sycl_output_xyz = nullptr;
};

#endif // DPNP_ITERATOR_H
