//*****************************************************************************
// Copyright (c) 2016-2025, Intel Corporation
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
 * This type should be used to simplify data iteration over input with
 * parameters
 * "[axis|axes]" It is designed to be used in SYCL environment
 *
 */
template <typename _Tp>
class DPNP_USM_iterator final
{
public:
    using value_type = _Tp;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;
    using pointer = value_type *;
    using reference = value_type &;
    using size_type = shape_elem_type;

    DPNP_USM_iterator(pointer __base_ptr,
                      size_type __id,
                      const size_type *__shape_stride = nullptr,
                      const size_type *__axes_stride = nullptr,
                      size_type __shape_size = 0)
        : base(__base_ptr), iter_id(__id), iteration_shape_size(__shape_size),
          iteration_shape_strides(__shape_stride),
          axes_shape_strides(__axes_stride)
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
    inline DPNP_USM_iterator &operator++()
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

    inline bool operator==(const DPNP_USM_iterator &__rhs) const
    {
        assert(base == __rhs.base); // iterators are incomparable if base
                                    // pointers are different
        return (iter_id == __rhs.iter_id);
    };

    inline bool operator!=(const DPNP_USM_iterator &__rhs) const
    {
        return !(*this == __rhs);
    };

    inline bool operator<(const DPNP_USM_iterator &__rhs) const
    {
        return iter_id < __rhs.iter_id;
    };

    // TODO need more operators

    // Random access iterator requirements
    inline reference operator[](size_type __n) const
    {
        return *ptr(__n);
    }

    inline difference_type operator-(const DPNP_USM_iterator &__rhs) const
    {
        difference_type diff =
            difference_type(iter_id) - difference_type(__rhs.iter_id);

        return diff;
    }

    /// Print this container in human readable form in error reporting
    friend std::ostream &operator<<(std::ostream &__out,
                                    const DPNP_USM_iterator &__it)
    {
        const std::vector<size_type> it_strides(__it.iteration_shape_strides,
                                                __it.iteration_shape_strides +
                                                    __it.iteration_shape_size);
        const std::vector<size_type> it_axes_strides(
            __it.axes_shape_strides,
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

        if (iteration_shape_size > 0) {
            long reminder = iteration_id;
            for (size_t it = 0; it < static_cast<size_t>(iteration_shape_size);
                 ++it) {
                const size_type axis_val = iteration_shape_strides[it];
                size_type xyz_id = reminder / axis_val;
                offset += (xyz_id * axes_shape_strides[it]);

                reminder = reminder % axis_val;
            }
        }
        else {
            offset = iteration_id;
        }

        return base + offset;
    }

    const pointer base = nullptr;
    size_type iter_id =
        size_type{}; /**< Iterator logical ID over iteration shape */
    const size_type iteration_shape_size =
        size_type{}; /**< Number of elements in @ref iteration_shape_strides
                        array */
    const size_type *iteration_shape_strides = nullptr;
    const size_type *axes_shape_strides = nullptr;
};

#endif // DPNP_ITERATOR_H
