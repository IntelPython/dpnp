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

#include <cstddef>
#include <pybind11/pybind11.h>
#include <vector>

#include "simplify_iteration_space.hpp"

// dpctl tensor headers
#include "utils/strided_iters.hpp"

namespace dpnp::extensions::py_internal
{
namespace py = pybind11;
namespace st_ns = dpctl::tensor::strides;

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
        assert(simplified_shape.size() == static_cast<std::size_t>(nd));

        simplified_src_strides.reserve(nd);
        simplified_src_strides.insert(std::end(simplified_src_strides),
                                      std::begin(src_strides),
                                      std::end(src_strides));
        assert(simplified_src_strides.size() == static_cast<std::size_t>(nd));

        simplified_dst_strides.reserve(nd);
        simplified_dst_strides.insert(std::end(simplified_dst_strides),
                                      std::begin(dst_strides),
                                      std::end(dst_strides));
        assert(simplified_dst_strides.size() == static_cast<std::size_t>(nd));

        int contracted_nd = st_ns::simplify_iteration_two_strides(
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
        assert(simplified_shape.size() == static_cast<std::size_t>(nd));

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

        assert(simplified_src_strides.size() == static_cast<std::size_t>(nd));
        assert(simplified_dst_strides.size() == static_cast<std::size_t>(nd));
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
        assert(simplified_shape.size() == static_cast<std::size_t>(nd));

        simplified_src1_strides.reserve(nd);
        simplified_src1_strides.insert(std::end(simplified_src1_strides),
                                       std::begin(src1_strides),
                                       std::end(src1_strides));
        assert(simplified_src1_strides.size() == static_cast<std::size_t>(nd));

        simplified_src2_strides.reserve(nd);
        simplified_src2_strides.insert(std::end(simplified_src2_strides),
                                       std::begin(src2_strides),
                                       std::end(src2_strides));
        assert(simplified_src2_strides.size() == static_cast<std::size_t>(nd));

        simplified_dst_strides.reserve(nd);
        simplified_dst_strides.insert(std::end(simplified_dst_strides),
                                      std::begin(dst_strides),
                                      std::end(dst_strides));
        assert(simplified_dst_strides.size() == static_cast<std::size_t>(nd));

        int contracted_nd = st_ns::simplify_iteration_three_strides(
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
        assert(simplified_shape.size() == static_cast<std::size_t>(nd));

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

        assert(simplified_src1_strides.size() == static_cast<std::size_t>(nd));
        assert(simplified_src2_strides.size() == static_cast<std::size_t>(nd));
        assert(simplified_dst_strides.size() == static_cast<std::size_t>(nd));
    }
}

void simplify_iteration_space_4(
    int &nd,
    const py::ssize_t *const &shape,
    // src1
    std::vector<py::ssize_t> const &src1_strides,
    // src2
    std::vector<py::ssize_t> const &src2_strides,
    // src3
    std::vector<py::ssize_t> const &src3_strides,
    // dst
    std::vector<py::ssize_t> const &dst_strides,
    // output
    std::vector<py::ssize_t> &simplified_shape,
    std::vector<py::ssize_t> &simplified_src1_strides,
    std::vector<py::ssize_t> &simplified_src2_strides,
    std::vector<py::ssize_t> &simplified_src3_strides,
    std::vector<py::ssize_t> &simplified_dst_strides,
    py::ssize_t &src1_offset,
    py::ssize_t &src2_offset,
    py::ssize_t &src3_offset,
    py::ssize_t &dst_offset)
{
    using dpctl::tensor::strides::simplify_iteration_four_strides;
    if (nd > 1) {
        // Simplify iteration space to reduce dimensionality
        // and improve access pattern
        simplified_shape.reserve(nd);
        simplified_shape.insert(std::end(simplified_shape), shape, shape + nd);
        assert(simplified_shape.size() == static_cast<std::size_t>(nd));

        simplified_src1_strides.reserve(nd);
        simplified_src1_strides.insert(std::end(simplified_src1_strides),
                                       std::begin(src1_strides),
                                       std::end(src1_strides));
        assert(simplified_src1_strides.size() == static_cast<std::size_t>(nd));

        simplified_src2_strides.reserve(nd);
        simplified_src2_strides.insert(std::end(simplified_src2_strides),
                                       std::begin(src2_strides),
                                       std::end(src2_strides));
        assert(simplified_src2_strides.size() == static_cast<std::size_t>(nd));

        simplified_src3_strides.reserve(nd);
        simplified_src3_strides.insert(std::end(simplified_src3_strides),
                                       std::begin(src3_strides),
                                       std::end(src3_strides));
        assert(simplified_src3_strides.size() == static_cast<std::size_t>(nd));

        simplified_dst_strides.reserve(nd);
        simplified_dst_strides.insert(std::end(simplified_dst_strides),
                                      std::begin(dst_strides),
                                      std::end(dst_strides));
        assert(simplified_dst_strides.size() == static_cast<std::size_t>(nd));

        int contracted_nd = simplify_iteration_four_strides(
            nd, simplified_shape.data(), simplified_src1_strides.data(),
            simplified_src2_strides.data(), simplified_src3_strides.data(),
            simplified_dst_strides.data(),
            src1_offset, // modified by reference
            src2_offset, // modified by reference
            src3_offset, // modified by reference
            dst_offset   // modified by reference
        );
        simplified_shape.resize(contracted_nd);
        simplified_src1_strides.resize(contracted_nd);
        simplified_src2_strides.resize(contracted_nd);
        simplified_src3_strides.resize(contracted_nd);
        simplified_dst_strides.resize(contracted_nd);

        nd = contracted_nd;
    }
    else if (nd == 1) {
        src1_offset = 0;
        src2_offset = 0;
        src3_offset = 0;
        dst_offset = 0;
        // Populate vectors
        simplified_shape.reserve(nd);
        simplified_shape.push_back(shape[0]);
        assert(simplified_shape.size() == static_cast<std::size_t>(nd));

        simplified_src1_strides.reserve(nd);
        simplified_src2_strides.reserve(nd);
        simplified_src3_strides.reserve(nd);
        simplified_dst_strides.reserve(nd);

        if ((src1_strides[0] < 0) && (src2_strides[0] < 0) &&
            (src3_strides[0] < 0) && (dst_strides[0] < 0))
        {
            simplified_src1_strides.push_back(-src1_strides[0]);
            simplified_src2_strides.push_back(-src2_strides[0]);
            simplified_src3_strides.push_back(-src3_strides[0]);
            simplified_dst_strides.push_back(-dst_strides[0]);
            if (shape[0] > 1) {
                src1_offset += src1_strides[0] * (shape[0] - 1);
                src2_offset += src2_strides[0] * (shape[0] - 1);
                src3_offset += src3_strides[0] * (shape[0] - 1);
                dst_offset += dst_strides[0] * (shape[0] - 1);
            }
        }
        else {
            simplified_src1_strides.push_back(src1_strides[0]);
            simplified_src2_strides.push_back(src2_strides[0]);
            simplified_src3_strides.push_back(src3_strides[0]);
            simplified_dst_strides.push_back(dst_strides[0]);
        }

        assert(simplified_src1_strides.size() == static_cast<std::size_t>(nd));
        assert(simplified_src2_strides.size() == static_cast<std::size_t>(nd));
        assert(simplified_src3_strides.size() == static_cast<std::size_t>(nd));
        assert(simplified_dst_strides.size() == static_cast<std::size_t>(nd));
    }
}
} // namespace dpnp::extensions::py_internal
