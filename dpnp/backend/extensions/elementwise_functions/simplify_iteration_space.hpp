//*****************************************************************************
// Copyright (c) 2024, Intel Corporation
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

#include <vector>

#include "dpctl4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace dpnp::extensions::py_internal
{

namespace py = pybind11;

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

void simplify_iteration_space(int &nd,
                              const py::ssize_t *const &shape,
                              std::vector<py::ssize_t> const &src_strides,
                              std::vector<py::ssize_t> const &dst_strides,
                              // output
                              std::vector<py::ssize_t> &simplified_shape,
                              std::vector<py::ssize_t> &simplified_src_strides,
                              std::vector<py::ssize_t> &simplified_dst_strides,
                              py::ssize_t &src_offset,
                              py::ssize_t &dst_offset)
{
    if (nd > 1) {
        // Simplify iteration space to reduce dimensionality
        // and improve access pattern
        simplified_shape.reserve(nd);
        simplified_shape.insert(std::begin(simplified_shape), shape,
                                shape + nd);
        assert(simplified_shape.size() == static_cast<size_t>(nd));

        simplified_src_strides.reserve(nd);
        simplified_src_strides.insert(std::end(simplified_src_strides),
                                      std::begin(src_strides),
                                      std::end(src_strides));
        assert(simplified_src_strides.size() == static_cast<size_t>(nd));

        simplified_dst_strides.reserve(nd);
        simplified_dst_strides.insert(std::end(simplified_dst_strides),
                                      std::begin(dst_strides),
                                      std::end(dst_strides));
        assert(simplified_dst_strides.size() == static_cast<size_t>(nd));

        int contracted_nd = simplify_iteration_two_strides(
            nd, simplified_shape.data(), simplified_src_strides.data(),
            simplified_dst_strides.data(),
            src_offset, // modified by reference
            dst_offset  // modified by reference
        );
        simplified_shape.resize(contracted_nd);
        simplified_src_strides.resize(contracted_nd);
        simplified_dst_strides.resize(contracted_nd);

        nd = contracted_nd;
    }
    else if (nd == 1) {
        src_offset = 0;
        dst_offset = 0;
        // Populate vectors
        simplified_shape.reserve(nd);
        simplified_shape.push_back(shape[0]);
        assert(simplified_shape.size() == static_cast<size_t>(nd));

        simplified_src_strides.reserve(nd);
        simplified_dst_strides.reserve(nd);

        if (src_strides[0] < 0 && dst_strides[0] < 0) {
            simplified_src_strides.push_back(-src_strides[0]);
            simplified_dst_strides.push_back(-dst_strides[0]);
            if (shape[0] > 1) {
                src_offset += (shape[0] - 1) * src_strides[0];
                dst_offset += (shape[0] - 1) * dst_strides[0];
            }
        }
        else {
            simplified_src_strides.push_back(src_strides[0]);
            simplified_dst_strides.push_back(dst_strides[0]);
        }

        assert(simplified_src_strides.size() == static_cast<size_t>(nd));
        assert(simplified_dst_strides.size() == static_cast<size_t>(nd));
    }
}

void simplify_iteration_space_3(
    int &nd,
    const py::ssize_t *const &shape,
    // src1
    std::vector<py::ssize_t> const &src1_strides,
    // src2
    std::vector<py::ssize_t> const &src2_strides,
    // dst
    std::vector<py::ssize_t> const &dst_strides,
    // output
    std::vector<py::ssize_t> &simplified_shape,
    std::vector<py::ssize_t> &simplified_src1_strides,
    std::vector<py::ssize_t> &simplified_src2_strides,
    std::vector<py::ssize_t> &simplified_dst_strides,
    py::ssize_t &src1_offset,
    py::ssize_t &src2_offset,
    py::ssize_t &dst_offset)
{
    if (nd > 1) {
        // Simplify iteration space to reduce dimensionality
        // and improve access pattern
        simplified_shape.reserve(nd);
        simplified_shape.insert(std::end(simplified_shape), shape, shape + nd);
        assert(simplified_shape.size() == static_cast<size_t>(nd));

        simplified_src1_strides.reserve(nd);
        simplified_src1_strides.insert(std::end(simplified_src1_strides),
                                       std::begin(src1_strides),
                                       std::end(src1_strides));
        assert(simplified_src1_strides.size() == static_cast<size_t>(nd));

        simplified_src2_strides.reserve(nd);
        simplified_src2_strides.insert(std::end(simplified_src2_strides),
                                       std::begin(src2_strides),
                                       std::end(src2_strides));
        assert(simplified_src2_strides.size() == static_cast<size_t>(nd));

        simplified_dst_strides.reserve(nd);
        simplified_dst_strides.insert(std::end(simplified_dst_strides),
                                      std::begin(dst_strides),
                                      std::end(dst_strides));
        assert(simplified_dst_strides.size() == static_cast<size_t>(nd));

        int contracted_nd = simplify_iteration_three_strides(
            nd, simplified_shape.data(), simplified_src1_strides.data(),
            simplified_src2_strides.data(), simplified_dst_strides.data(),
            src1_offset, // modified by reference
            src2_offset, // modified by reference
            dst_offset   // modified by reference
        );
        simplified_shape.resize(contracted_nd);
        simplified_src1_strides.resize(contracted_nd);
        simplified_src2_strides.resize(contracted_nd);
        simplified_dst_strides.resize(contracted_nd);

        nd = contracted_nd;
    }
    else if (nd == 1) {
        src1_offset = 0;
        src2_offset = 0;
        dst_offset = 0;
        // Populate vectors
        simplified_shape.reserve(nd);
        simplified_shape.push_back(shape[0]);
        assert(simplified_shape.size() == static_cast<size_t>(nd));

        simplified_src1_strides.reserve(nd);
        simplified_src2_strides.reserve(nd);
        simplified_dst_strides.reserve(nd);

        if ((src1_strides[0] < 0) && (src2_strides[0] < 0) &&
            (dst_strides[0] < 0)) {
            simplified_src1_strides.push_back(-src1_strides[0]);
            simplified_src2_strides.push_back(-src2_strides[0]);
            simplified_dst_strides.push_back(-dst_strides[0]);
            if (shape[0] > 1) {
                src1_offset += src1_strides[0] * (shape[0] - 1);
                src2_offset += src2_strides[0] * (shape[0] - 1);
                dst_offset += dst_strides[0] * (shape[0] - 1);
            }
        }
        else {
            simplified_src1_strides.push_back(src1_strides[0]);
            simplified_src2_strides.push_back(src2_strides[0]);
            simplified_dst_strides.push_back(dst_strides[0]);
        }

        assert(simplified_src1_strides.size() == static_cast<size_t>(nd));
        assert(simplified_src2_strides.size() == static_cast<size_t>(nd));
        assert(simplified_dst_strides.size() == static_cast<size_t>(nd));
    }
}
} // namespace dpnp::extensions::py_internal
