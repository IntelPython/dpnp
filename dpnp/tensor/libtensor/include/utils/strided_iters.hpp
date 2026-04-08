//*****************************************************************************
// Copyright (c) 2026, Intel Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// - Neither the name of the copyright holder nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
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
///
/// \file
/// This file defines CIndexer_array, and CIndexer_vector classes, as well
/// iteration space simplifiers.
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <vector>

namespace dpctl::tensor::strides
{
/* An N-dimensional array can be stored in a single
 * contiguous chunk of memory by contiguously laying
 * array elements in lexicographinc order of their
 * array indices. Such a layout is called C-contiguous.
 *
 * E.g. for (2, 3, 2) array `a` with zero-based indexing convention
 * the C-array's elements are
 * { a[0,0,0], a[0,0,1], a[0,1,0], a[0,1,1], a[0,2,0], a[0,2,1],
 *   a[1,0,0], a[1,0,1], a[1,1,0], a[1,1,1], a[1,2,0], a[1,2,1] }
 *
 * Indexer maps zero-based index in C-array to a multi-index
 * for the purpose of computing element displacement in the
 * strided array, i.e. in the above example for k = 5, the displacement
 * is (s0*0 + s1*2 + s2*1), and for k = 7 it is (s0*1 + s1*0 + s2*1)
 * for N-dimensional array with strides (s0, s1, s2).
 *
 * Cindexer_vector need not know array rank `dim` at compile time.
 * Shape and strides are stored in std::vector, which are not trivially
 * copyable.
 *
 * For the class to be trivially copyable for offloading displacement
 * computation methods take accessor/pointer arguments of type T for
 * shape and stride and modify displacement argument passed by reference.
 */
template <typename indT = std::ptrdiff_t>
class CIndexer_vector
{
    static_assert(std::is_integral<indT>::value, "Integral type is required");
    static_assert(std::is_signed<indT>::value,
                  "Signed integral type is required");
    int nd;

public:
    CIndexer_vector(int dim) : nd(dim) {}

    template <class ShapeTy>
    indT size(const ShapeTy &shape) const
    {
        indT s = static_cast<indT>(1);
        for (int i = 0; i < nd; ++i) {
            s *= shape[i];
        }
        return s;
    }

    template <class ShapeTy, class StridesTy>
    void get_displacement(const indT i,
                          const ShapeTy &shape,
                          const StridesTy &stride,
                          indT &disp) const
    {
        if (nd == 1) {
            disp = i * stride[0];
            return;
        }

        indT i_ = i;
        indT d = 0;
        for (int dim = nd; --dim > 0;) {
            const indT si = shape[dim];
            const indT q = i_ / si;
            const indT r = (i_ - q * si);
            d += r * stride[dim];
            i_ = q;
        }
        disp = d + i_ * stride[0];
    }

    template <class ShapeTy, class StridesTy>
    void get_displacement(const indT i,
                          const ShapeTy &shape,
                          const StridesTy &stride1,
                          const StridesTy &stride2,
                          indT &disp1,
                          indT &disp2) const
    {
        if (nd == 1) {
            disp1 = i * stride1[0];
            disp2 = i * stride2[0];
            return;
        }

        indT i_ = i;
        indT d1 = 0, d2 = 0;
        for (int dim = nd; --dim > 0;) {
            const indT si = shape[dim];
            const indT q = i_ / si;
            const indT r = (i_ - q * si);
            i_ = q;
            d1 += r * stride1[dim];
            d2 += r * stride2[dim];
        }
        disp1 = d1 + i_ * stride1[0];
        disp2 = d2 + i_ * stride2[0];
        return;
    }

    template <class ShapeTy, class StridesTy>
    void get_displacement(const indT i,
                          const ShapeTy &shape,
                          const StridesTy &stride1,
                          const StridesTy &stride2,
                          const StridesTy &stride3,
                          indT &disp1,
                          indT &disp2,
                          indT &disp3) const
    {
        if (nd == 1) {
            disp1 = i * stride1[0];
            disp2 = i * stride2[0];
            disp3 = i * stride3[0];
            return;
        }

        indT i_ = i;
        indT d1 = 0, d2 = 0, d3 = 0;
        for (int dim = nd; --dim > 0;) {
            const indT si = shape[dim];
            const indT q = i_ / si;
            const indT r = (i_ - q * si);
            i_ = q;
            d1 += r * stride1[dim];
            d2 += r * stride2[dim];
            d3 += r * stride3[dim];
        };
        disp1 = d1 + i_ * stride1[0];
        disp2 = d2 + i_ * stride2[0];
        disp3 = d3 + i_ * stride3[0];
        return;
    }

    template <class ShapeTy, class StridesTy>
    void get_displacement(const indT i,
                          const ShapeTy &shape,
                          const StridesTy &stride1,
                          const StridesTy &stride2,
                          const StridesTy &stride3,
                          const StridesTy &stride4,
                          indT &disp1,
                          indT &disp2,
                          indT &disp3,
                          indT &disp4) const
    {
        if (nd == 1) {
            disp1 = i * stride1[0];
            disp2 = i * stride2[0];
            disp3 = i * stride3[0];
            disp4 = i * stride4[0];
            return;
        }

        indT i_ = i;
        indT d1 = 0, d2 = 0, d3 = 0, d4 = 0;
        for (int dim = nd; --dim > 0;) {
            const indT si = shape[dim];
            const indT q = i_ / si;
            const indT r = (i_ - q * si);
            i_ = q;
            d1 += r * stride1[dim];
            d2 += r * stride2[dim];
            d3 += r * stride3[dim];
            d4 += r * stride4[dim];
        }
        disp1 = d1 + i_ * stride1[0];
        disp2 = d2 + i_ * stride2[0];
        disp3 = d3 + i_ * stride3[0];
        disp4 = d4 + i_ * stride4[0];
        return;
    }

    template <class ShapeTy, class StridesTy, int nstrides>
    void get_displacement(const indT i,
                          const ShapeTy &shape,
                          const std::array<StridesTy, nstrides> &strides,
                          std::array<indT, nstrides> &disps) const
    {
        if (nd == 1) {
            for (int k = 0; k < nstrides; ++k) {
                disps[k] = i * strides[k][0];
            }
            return;
        }

        indT i_ = i;
        std::array<indT, nstrides> ds;
        for (int k = 0; k < nstrides; ++k) {
            ds[k] = 0;
        }

        for (int dim = nd; --dim > 0;) {
            const indT si = shape[dim];
            const indT q = i_ / si;
            const indT r = (i_ - q * si);
            for (int k = 0; k < nstrides; ++k) {
                ds[k] += r * strides[k][dim];
            }
            i_ = q;
        };
        for (int k = 0; k < nstrides; ++k) {
            disps[k] = ds[k] + i_ * strides[k][0];
        }
        return;
    }

    template <class ShapeTy, class StridesTy>
    void get_left_rolled_displacement(const indT i,
                                      const ShapeTy &shape,
                                      const StridesTy &stride,
                                      const StridesTy &shifts,
                                      indT &disp) const
    {
        indT i_ = i;
        indT d(0);
        for (int dim = nd; --dim > 0;) {
            const indT si = shape[dim];
            const indT q = i_ / si;
            const indT r = (i_ - q * si);
            // assumes si > shifts[dim] >= 0
            const indT shifted_r =
                (r < shifts[dim] ? r + si - shifts[dim] : r - shifts[dim]);
            d += shifted_r * stride[dim];
            i_ = q;
        }
        const indT shifted_r =
            (i_ < shifts[0] ? i_ + shape[0] - shifts[0] : i_ - shifts[0]);
        disp = d + shifted_r * stride[0];
    }
};

/*
 * CIndexer is for arrays whose array-rank is known at compile time.
 * Statically allocated shape and multi_index arrays are members of
 * the class instance, and it remains trivially copyable.
 *
 * Method `set(k)` populates work-item private array multi_index, which
 * can be accessed using `get()` to compute the displacement as needed.
 */

template <int _ndim, typename indT = std::ptrdiff_t>
class CIndexer_array
{
    static constexpr int ndim = _ndim;

    static_assert(std::is_integral<indT>::value, "Integral type is required");
    static_assert(std::is_signed<indT>::value,
                  "Signed integral type is required");
    static_assert(ndim > 0, "Dimensionality must be positive");

private:
    typedef std::array<indT, ndim> index_t;

    indT elem_count;
    index_t shape;
    index_t multi_index;

public:
    CIndexer_array() : elem_count(0), shape{}, multi_index{} {}

    explicit CIndexer_array(const index_t &input_shape)
        : elem_count(0), shape{}, multi_index{}
    {
        indT s(1);
        for (int i = 0; i < ndim; ++i) {
            shape[i] = input_shape[i];
            s *= input_shape[i];
        }
        elem_count = s;
    }

    indT size() const
    {
        return elem_count;
    }
    indT rank() const
    {
        return ndim;
    }

    void set(const indT i)
    {
        if (ndim == 1) {
            multi_index[0] = i;
            return;
        }

        indT i_ = i;
#pragma unroll
        for (int dim = ndim; --dim > 0;) {
            indT si = shape[dim];
            indT q = i_ / si;
            multi_index[dim] = i_ - q * si;
            i_ = q;
        }
        multi_index[0] = i_;
    }

    const index_t &get() const
    {
        return multi_index;
    }
};

/*
    For purposes of iterating over elements of array with
    `shape` and `strides` given as pointers
    `simplify_iteration_strides(nd, shape_ptr, strides_ptr, disp)`
    may modify memory and returns new length of these arrays.

    The new shape and new strides, as well as the offset
    `(new_shape, new_strides, disp)` are such that iterating over
    them will traverse the same elements, possibly in
    different order.

    ..Example: python
        import itertools
        # for some array Y over whose elements we iterate
        csh, cst, cp = contract_iter(Y.shape, Y.strides)
        def pointers_set(sh, st, p):
            citers = itertools.product(*map(lambda s: range(s), sh))
            dot = lambda st, it: sum(st[k]*it[k] for k in range(len(st)))
            return set(p + dot(st, it) for it in citers)
        ps1 = pointers_set(csh, cst, cp)
        ps2 = pointers_set(Y.shape, Y.strides, 0)
        assert ps1 == ps2

 */
template <class ShapeTy, class StridesTy>
int simplify_iteration_stride(const int nd,
                              ShapeTy *shape,
                              StridesTy *strides,
                              StridesTy &disp)
{
    disp = StridesTy(0);
    if (nd < 2)
        return nd;

    std::vector<int> pos(nd);
    std::iota(pos.begin(), pos.end(), 0);

    std::stable_sort(
        pos.begin(), pos.end(), [&strides, &shape](int i1, int i2) {
            auto abs_str1 = (strides[i1] < 0) ? -strides[i1] : strides[i1];
            auto abs_str2 = (strides[i2] < 0) ? -strides[i2] : strides[i2];
            return (abs_str1 > abs_str2) ||
                   (abs_str1 == abs_str2 && shape[i1] > shape[i2]);
        });

    std::vector<ShapeTy> shape_w;
    std::vector<StridesTy> strides_w;
    int nd_ = nd;
    shape_w.reserve(nd_);
    strides_w.reserve(nd_);

    for (int i = 0; i < nd; ++i) {
        auto p = pos[i];
        auto sh_p = shape[p];
        auto str_p = strides[p];
        shape_w.push_back(sh_p);
        if (str_p < 0) {
            disp += str_p * (sh_p - 1);
            str_p = -str_p;
        }
        strides_w.push_back(str_p);
    }

    {
        bool changed;
        do {
            changed = false;
            for (int i = 0; i + 1 < nd_; ++i) {
                StridesTy step = strides_w[i + 1];
                StridesTy jump = strides_w[i] - (shape_w[i + 1] - 1) * step;
                if (jump == step) {
                    changed = true;
                    for (int k = i; k + 1 < nd_; ++k) {
                        strides_w[k] = strides_w[k + 1];
                    }
                    shape_w[i] *= shape_w[i + 1];
                    for (int k = i + 1; k + 1 < nd_; ++k) {
                        shape_w[k] = shape_w[k + 1];
                    }
                    --nd_;
                }
            }
        } while (changed);
    }

    for (int i = 0; i < nd_; ++i) {
        shape[i] = shape_w[i];
    }
    for (int i = 0; i < nd_; ++i) {
        strides[i] = strides_w[i];
    }

    return nd_;
}

/*
    For purposes of iterating over pairs of elements of two arrays
    with  `shape` and strides `strides1`, `strides2` given as pointers
    `simplify_iteration_two_strides(nd, shape_ptr, strides1_ptr,
    strides2_ptr, disp1, disp2)`
    may modify memory and returns new length of these arrays.

    The new shape and new strides, as well as the offset
    `(new_shape, new_strides1, disp1, new_stride2, disp2)` are such that
    iterating over them will traverse the same set of pairs of elements,
    possibly in a different order.
 */
template <class ShapeTy, class StridesTy>
int simplify_iteration_two_strides(const int nd,
                                   ShapeTy *shape,
                                   StridesTy *strides1,
                                   StridesTy *strides2,
                                   StridesTy &disp1,
                                   StridesTy &disp2)
{
    disp1 = StridesTy(0);
    disp2 = StridesTy(0);
    if (nd < 2)
        return nd;

    std::vector<int> pos(nd);
    std::iota(pos.begin(), pos.end(), 0);

    std::stable_sort(
        pos.begin(), pos.end(), [&strides1, &strides2, &shape](int i1, int i2) {
            auto abs_str1_i1 =
                (strides1[i1] < 0) ? -strides1[i1] : strides1[i1];
            auto abs_str1_i2 =
                (strides1[i2] < 0) ? -strides1[i2] : strides1[i2];
            auto abs_str2_i1 =
                (strides2[i1] < 0) ? -strides2[i1] : strides2[i1];
            auto abs_str2_i2 =
                (strides2[i2] < 0) ? -strides2[i2] : strides2[i2];
            return (abs_str2_i1 > abs_str2_i2) ||
                   (abs_str2_i1 == abs_str2_i2 &&
                    (abs_str1_i1 > abs_str1_i2 ||
                     (abs_str1_i1 == abs_str1_i2 && shape[i1] > shape[i2])));
        });

    std::vector<ShapeTy> shape_w;
    std::vector<StridesTy> strides1_w;
    std::vector<StridesTy> strides2_w;

    bool contractable = true;
    for (int i = 0; i < nd; ++i) {
        auto p = pos[i];
        auto sh_p = shape[p];
        auto str1_p = strides1[p];
        auto str2_p = strides2[p];
        shape_w.push_back(sh_p);
        if (str1_p <= 0 && str2_p <= 0 && std::min(str1_p, str2_p) < 0) {
            disp1 += str1_p * (sh_p - 1);
            str1_p = -str1_p;
            disp2 += str2_p * (sh_p - 1);
            str2_p = -str2_p;
        }
        if (str1_p < 0 || str2_p < 0) {
            contractable = false;
        }
        strides1_w.push_back(str1_p);
        strides2_w.push_back(str2_p);
    }

    int nd_ = nd;
    while (contractable) {
        bool changed = false;
        for (int i = 0; i + 1 < nd_; ++i) {
            StridesTy str1 = strides1_w[i + 1];
            StridesTy str2 = strides2_w[i + 1];
            StridesTy jump1 = strides1_w[i] - (shape_w[i + 1] - 1) * str1;
            StridesTy jump2 = strides2_w[i] - (shape_w[i + 1] - 1) * str2;

            if (jump1 == str1 && jump2 == str2) {
                changed = true;
                shape_w[i] *= shape_w[i + 1];
                for (int j = i; j < nd_; ++j) {
                    strides1_w[j] = strides1_w[j + 1];
                }
                for (int j = i; j < nd_; ++j) {
                    strides2_w[j] = strides2_w[j + 1];
                }
                for (int j = i + 1; j + 1 < nd_; ++j) {
                    shape_w[j] = shape_w[j + 1];
                }
                --nd_;
                break;
            }
        }
        if (!changed)
            break;
    }
    for (int i = 0; i < nd_; ++i) {
        shape[i] = shape_w[i];
    }
    for (int i = 0; i < nd_; ++i) {
        strides1[i] = strides1_w[i];
    }
    for (int i = 0; i < nd_; ++i) {
        strides2[i] = strides2_w[i];
    }

    return nd_;
}

template <typename T, class Error, typename vecT = std::vector<T>>
std::tuple<vecT, vecT, T> contract_iter(const vecT &shape, const vecT &strides)
{
    const std::size_t dim = shape.size();
    if (dim != strides.size()) {
        throw Error("Shape and strides must be of equal size.");
    }
    vecT out_shape = shape;
    vecT out_strides = strides;
    T disp(0);

    int nd = simplify_iteration_stride(dim, out_shape.data(),
                                       out_strides.data(), disp);
    out_shape.resize(nd);
    out_strides.resize(nd);
    return std::make_tuple(out_shape, out_strides, disp);
}

template <typename T, class Error, typename vecT = std::vector<T>>
std::tuple<vecT, vecT, T, vecT, T> contract_iter2(const vecT &shape,
                                                  const vecT &strides1,
                                                  const vecT &strides2)
{
    const std::size_t dim = shape.size();
    if (dim != strides1.size() || dim != strides2.size()) {
        throw Error("Shape and strides must be of equal size.");
    }
    vecT out_shape = shape;
    vecT out_strides1 = strides1;
    vecT out_strides2 = strides2;
    T disp1(0);
    T disp2(0);

    int nd = simplify_iteration_two_strides(dim, out_shape.data(),
                                            out_strides1.data(),
                                            out_strides2.data(), disp1, disp2);
    out_shape.resize(nd);
    out_strides1.resize(nd);
    out_strides2.resize(nd);
    return std::make_tuple(out_shape, out_strides1, disp1, out_strides2, disp2);
}

/*
    For purposes of iterating over pairs of elements of three arrays
    with  `shape` and strides `strides1`, `strides2`, `strides3` given as
    pointers `simplify_iteration_three_strides(nd, shape_ptr, strides1_ptr,
    strides2_ptr, strides3_ptr, disp1, disp2, disp3)`
    may modify memory and returns new length of these arrays.

    The new shape and new strides, as well as the offset
    `(new_shape, new_strides1, disp1, new_stride2, disp2, new_stride3, disp3)`
    are such that iterating over them will traverse the same set of tuples of
    elements, possibly in a different order.
 */
template <class ShapeTy, class StridesTy>
int simplify_iteration_three_strides(const int nd,
                                     ShapeTy *shape,
                                     StridesTy *strides1,
                                     StridesTy *strides2,
                                     StridesTy *strides3,
                                     StridesTy &disp1,
                                     StridesTy &disp2,
                                     StridesTy &disp3)
{
    disp1 = StridesTy(0);
    disp2 = StridesTy(0);
    if (nd < 2)
        return nd;

    std::vector<int> pos(nd);
    std::iota(pos.begin(), pos.end(), 0);

    std::stable_sort(pos.begin(), pos.end(),
                     [&strides1, &strides2, &strides3, &shape](int i1, int i2) {
                         auto abs_str1_i1 =
                             (strides1[i1] < 0) ? -strides1[i1] : strides1[i1];
                         auto abs_str1_i2 =
                             (strides1[i2] < 0) ? -strides1[i2] : strides1[i2];
                         auto abs_str2_i1 =
                             (strides2[i1] < 0) ? -strides2[i1] : strides2[i1];
                         auto abs_str2_i2 =
                             (strides2[i2] < 0) ? -strides2[i2] : strides2[i2];
                         auto abs_str3_i1 =
                             (strides3[i1] < 0) ? -strides3[i1] : strides3[i1];
                         auto abs_str3_i2 =
                             (strides3[i2] < 0) ? -strides3[i2] : strides3[i2];
                         return (abs_str3_i1 > abs_str3_i2) ||
                                ((abs_str3_i1 == abs_str3_i2) &&
                                 ((abs_str2_i1 > abs_str2_i2) ||
                                  ((abs_str2_i1 == abs_str2_i2) &&
                                   ((abs_str1_i1 > abs_str1_i2) ||
                                    ((abs_str1_i1 == abs_str1_i2) &&
                                     (shape[i1] > shape[i2]))))));
                     });

    std::vector<ShapeTy> shape_w;
    std::vector<StridesTy> strides1_w;
    std::vector<StridesTy> strides2_w;
    std::vector<StridesTy> strides3_w;

    bool contractable = true;
    for (int i = 0; i < nd; ++i) {
        auto p = pos[i];
        auto sh_p = shape[p];
        auto str1_p = strides1[p];
        auto str2_p = strides2[p];
        auto str3_p = strides3[p];
        shape_w.push_back(sh_p);
        if (str1_p <= 0 && str2_p <= 0 && str3_p <= 0 &&
            std::min({str1_p, str2_p, str3_p}) < 0)
        {
            disp1 += str1_p * (sh_p - 1);
            str1_p = -str1_p;
            disp2 += str2_p * (sh_p - 1);
            str2_p = -str2_p;
            disp3 += str3_p * (sh_p - 1);
            str3_p = -str3_p;
        }
        if (str1_p < 0 || str2_p < 0 || str3_p < 0) {
            contractable = false;
        }
        strides1_w.push_back(str1_p);
        strides2_w.push_back(str2_p);
        strides3_w.push_back(str3_p);
    }
    int nd_ = nd;
    while (contractable) {
        bool changed = false;
        for (int i = 0; i + 1 < nd_; ++i) {
            StridesTy str1 = strides1_w[i + 1];
            StridesTy str2 = strides2_w[i + 1];
            StridesTy str3 = strides3_w[i + 1];
            StridesTy jump1 = strides1_w[i] - (shape_w[i + 1] - 1) * str1;
            StridesTy jump2 = strides2_w[i] - (shape_w[i + 1] - 1) * str2;
            StridesTy jump3 = strides3_w[i] - (shape_w[i + 1] - 1) * str3;

            if (jump1 == str1 && jump2 == str2 && jump3 == str3) {
                changed = true;
                shape_w[i] *= shape_w[i + 1];
                for (int j = i; j < nd_; ++j) {
                    strides1_w[j] = strides1_w[j + 1];
                }
                for (int j = i; j < nd_; ++j) {
                    strides2_w[j] = strides2_w[j + 1];
                }
                for (int j = i; j < nd_; ++j) {
                    strides3_w[j] = strides3_w[j + 1];
                }
                for (int j = i + 1; j + 1 < nd_; ++j) {
                    shape_w[j] = shape_w[j + 1];
                }
                --nd_;
                break;
            }
        }
        if (!changed)
            break;
    }
    for (int i = 0; i < nd_; ++i) {
        shape[i] = shape_w[i];
    }
    for (int i = 0; i < nd_; ++i) {
        strides1[i] = strides1_w[i];
    }
    for (int i = 0; i < nd_; ++i) {
        strides2[i] = strides2_w[i];
    }
    for (int i = 0; i < nd_; ++i) {
        strides3[i] = strides3_w[i];
    }

    return nd_;
}

template <typename T, class Error, typename vecT = std::vector<T>>
std::tuple<vecT, vecT, T, vecT, T, vecT, T> contract_iter3(const vecT &shape,
                                                           const vecT &strides1,
                                                           const vecT &strides2,
                                                           const vecT &strides3)
{
    const std::size_t dim = shape.size();
    if (dim != strides1.size() || dim != strides2.size() ||
        dim != strides3.size()) {
        throw Error("Shape and strides must be of equal size.");
    }
    vecT out_shape = shape;
    vecT out_strides1 = strides1;
    vecT out_strides2 = strides2;
    vecT out_strides3 = strides3;
    T disp1(0);
    T disp2(0);
    T disp3(0);

    int nd = simplify_iteration_three_strides(
        dim, out_shape.data(), out_strides1.data(), out_strides2.data(),
        out_strides3.data(), disp1, disp2, disp3);
    out_shape.resize(nd);
    out_strides1.resize(nd);
    out_strides2.resize(nd);
    out_strides3.resize(nd);
    return std::make_tuple(out_shape, out_strides1, disp1, out_strides2, disp2,
                           out_strides3, disp3);
}

/*
    For purposes of iterating over pairs of elements of four arrays
    with  `shape` and strides `strides1`, `strides2`, `strides3`,
    `strides4` given as pointers `simplify_iteration_four_strides(nd,
    shape_ptr, strides1_ptr, strides2_ptr, strides3_ptr, strides4_ptr,
    disp1, disp2, disp3, disp4)` may modify memory and returns new
    length of these arrays.

    The new shape and new strides, as well as the offset
    `(new_shape, new_strides1, disp1, new_stride2, disp2, new_stride3, disp3,
    new_stride4, disp4)` are such that iterating over them will traverse the
    same set of tuples of elements, possibly in a different order.
 */
template <class ShapeTy, class StridesTy>
int simplify_iteration_four_strides(const int nd,
                                    ShapeTy *shape,
                                    StridesTy *strides1,
                                    StridesTy *strides2,
                                    StridesTy *strides3,
                                    StridesTy *strides4,
                                    StridesTy &disp1,
                                    StridesTy &disp2,
                                    StridesTy &disp3,
                                    StridesTy &disp4)
{
    disp1 = StridesTy(0);
    disp2 = StridesTy(0);
    if (nd < 2)
        return nd;

    std::vector<int> pos(nd);
    std::iota(pos.begin(), pos.end(), 0);

    std::stable_sort(
        pos.begin(), pos.end(),
        [&strides1, &strides2, &strides3, &strides4, &shape](int i1, int i2) {
            auto abs_str1_i1 =
                (strides1[i1] < 0) ? -strides1[i1] : strides1[i1];
            auto abs_str1_i2 =
                (strides1[i2] < 0) ? -strides1[i2] : strides1[i2];
            auto abs_str2_i1 =
                (strides2[i1] < 0) ? -strides2[i1] : strides2[i1];
            auto abs_str2_i2 =
                (strides2[i2] < 0) ? -strides2[i2] : strides2[i2];
            auto abs_str3_i1 =
                (strides3[i1] < 0) ? -strides3[i1] : strides3[i1];
            auto abs_str3_i2 =
                (strides3[i2] < 0) ? -strides3[i2] : strides3[i2];
            auto abs_str4_i1 =
                (strides4[i1] < 0) ? -strides4[i1] : strides4[i1];
            auto abs_str4_i2 =
                (strides4[i2] < 0) ? -strides4[i2] : strides4[i2];
            return (abs_str4_i1 > abs_str4_i2) ||
                   ((abs_str4_i1 == abs_str4_i2) &&
                    ((abs_str3_i1 > abs_str3_i2) ||
                     ((abs_str3_i1 == abs_str3_i2) &&
                      ((abs_str2_i1 > abs_str2_i2) ||
                       ((abs_str2_i1 == abs_str2_i2) &&
                        ((abs_str1_i1 > abs_str1_i2) ||
                         ((abs_str1_i1 == abs_str1_i2) &&
                          (shape[i1] > shape[i2]))))))));
        });

    std::vector<ShapeTy> shape_w;
    std::vector<StridesTy> strides1_w;
    std::vector<StridesTy> strides2_w;
    std::vector<StridesTy> strides3_w;
    std::vector<StridesTy> strides4_w;

    bool contractable = true;
    for (int i = 0; i < nd; ++i) {
        auto p = pos[i];
        auto sh_p = shape[p];
        auto str1_p = strides1[p];
        auto str2_p = strides2[p];
        auto str3_p = strides3[p];
        auto str4_p = strides4[p];
        shape_w.push_back(sh_p);
        if (str1_p <= 0 && str2_p <= 0 && str3_p <= 0 && str4_p <= 0 &&
            std::min({str1_p, str2_p, str3_p, str4_p}) < 0)
        {
            disp1 += str1_p * (sh_p - 1);
            str1_p = -str1_p;
            disp2 += str2_p * (sh_p - 1);
            str2_p = -str2_p;
            disp3 += str3_p * (sh_p - 1);
            str3_p = -str3_p;
            disp4 += str4_p * (sh_p - 1);
            str4_p = -str4_p;
        }
        if (str1_p < 0 || str2_p < 0 || str3_p < 0 || str4_p < 0) {
            contractable = false;
        }
        strides1_w.push_back(str1_p);
        strides2_w.push_back(str2_p);
        strides3_w.push_back(str3_p);
        strides4_w.push_back(str4_p);
    }
    int nd_ = nd;
    while (contractable) {
        bool changed = false;
        for (int i = 0; i + 1 < nd_; ++i) {
            StridesTy str1 = strides1_w[i + 1];
            StridesTy str2 = strides2_w[i + 1];
            StridesTy str3 = strides3_w[i + 1];
            StridesTy str4 = strides4_w[i + 1];
            StridesTy jump1 = strides1_w[i] - (shape_w[i + 1] - 1) * str1;
            StridesTy jump2 = strides2_w[i] - (shape_w[i + 1] - 1) * str2;
            StridesTy jump3 = strides3_w[i] - (shape_w[i + 1] - 1) * str3;
            StridesTy jump4 = strides4_w[i] - (shape_w[i + 1] - 1) * str4;

            if (jump1 == str1 && jump2 == str2 && jump3 == str3 &&
                jump4 == str4) {
                changed = true;
                shape_w[i] *= shape_w[i + 1];
                for (int j = i; j < nd_; ++j) {
                    strides1_w[j] = strides1_w[j + 1];
                }
                for (int j = i; j < nd_; ++j) {
                    strides2_w[j] = strides2_w[j + 1];
                }
                for (int j = i; j < nd_; ++j) {
                    strides3_w[j] = strides3_w[j + 1];
                }
                for (int j = i; j < nd_; ++j) {
                    strides4_w[j] = strides4_w[j + 1];
                }
                for (int j = i + 1; j + 1 < nd_; ++j) {
                    shape_w[j] = shape_w[j + 1];
                }
                --nd_;
                break;
            }
        }
        if (!changed)
            break;
    }
    for (int i = 0; i < nd_; ++i) {
        shape[i] = shape_w[i];
    }
    for (int i = 0; i < nd_; ++i) {
        strides1[i] = strides1_w[i];
    }
    for (int i = 0; i < nd_; ++i) {
        strides2[i] = strides2_w[i];
    }
    for (int i = 0; i < nd_; ++i) {
        strides3[i] = strides3_w[i];
    }
    for (int i = 0; i < nd_; ++i) {
        strides4[i] = strides4_w[i];
    }

    return nd_;
}

template <typename T, class Error, typename vecT = std::vector<T>>
std::tuple<vecT, vecT, T, vecT, T, vecT, T, vecT, T>
    contract_iter4(const vecT &shape,
                   const vecT &strides1,
                   const vecT &strides2,
                   const vecT &strides3,
                   const vecT &strides4)
{
    const std::size_t dim = shape.size();
    if (dim != strides1.size() || dim != strides2.size() ||
        dim != strides3.size() || dim != strides4.size())
    {
        throw Error("Shape and strides must be of equal size.");
    }
    vecT out_shape = shape;
    vecT out_strides1 = strides1;
    vecT out_strides2 = strides2;
    vecT out_strides3 = strides3;
    vecT out_strides4 = strides4;
    T disp1(0);
    T disp2(0);
    T disp3(0);
    T disp4(0);

    int nd = simplify_iteration_four_strides(
        dim, out_shape.data(), out_strides1.data(), out_strides2.data(),
        out_strides3.data(), out_strides4.data(), disp1, disp2, disp3, disp4);
    out_shape.resize(nd);
    out_strides1.resize(nd);
    out_strides2.resize(nd);
    out_strides3.resize(nd);
    out_strides4.resize(nd);
    return std::make_tuple(out_shape, out_strides1, disp1, out_strides2, disp2,
                           out_strides3, disp3, out_strides4, disp4);
}

/*
    For purposes of iterating over elements of an array with  `shape` and
    strides `strides` given as pointers `compact_iteration(nd, shape, strides)`
    may modify memory and returns the new length of the array.

    The new shape and new strides `(new_shape, new_strides)` are such that
    iterating over them will traverse the same elements in the same order,
    possibly with reduced dimensionality.
 */
template <class ShapeTy, class StridesTy>
int compact_iteration(const int nd, ShapeTy *shape, StridesTy *strides)
{
    if (nd < 2)
        return nd;

    bool contractable = true;
    for (int i = 0; i < nd; ++i) {
        if (strides[i] < 0) {
            contractable = false;
        }
    }

    int nd_ = nd;
    while (contractable) {
        bool changed = false;
        for (int i = 0; i + 1 < nd_; ++i) {
            StridesTy str = strides[i + 1];
            StridesTy jump = strides[i] - (shape[i + 1] - 1) * str;

            if (jump == str) {
                changed = true;
                shape[i] *= shape[i + 1];
                for (int j = i; j < nd_; ++j) {
                    strides[j] = strides[j + 1];
                }
                for (int j = i + 1; j + 1 < nd_; ++j) {
                    shape[j] = shape[j + 1];
                }
                --nd_;
                break;
            }
        }
        if (!changed)
            break;
    }

    return nd_;
}
} // namespace dpctl::tensor::strides
