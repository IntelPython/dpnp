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

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>

#include <pybind11/pybind11.h>

#include "dpnp4pybind11.hpp"

#include "common.hpp"
#include "fft_utils.hpp"
#include "out_of_place.hpp"

// dpctl tensor headers
#include "utils/memory_overlap.hpp"
#include "utils/output_validation.hpp"

namespace dpnp::extensions::fft
{
namespace mkl_dft = oneapi::mkl::dft;
namespace py = pybind11;

// out-of-place FFT computation
template <mkl_dft::precision prec, mkl_dft::domain dom>
std::pair<sycl::event, sycl::event>
    compute_fft_out_of_place(DescriptorWrapper<prec, dom> &descr,
                             const dpctl::tensor::usm_ndarray &in,
                             const dpctl::tensor::usm_ndarray &out,
                             const bool is_forward,
                             const std::vector<sycl::event> &depends)
{
    const bool committed = descr.is_committed();
    if (!committed) {
        throw py::value_error("Descriptor is not committed");
    }

    const bool in_place = descr.get_in_place();
    if (in_place) {
        throw py::value_error(
            "Descriptor is defined for in-place FFT while this function is set "
            "to compute out-of-place FFT.");
    }

    const int in_nd = in.get_ndim();
    const int out_nd = out.get_ndim();
    if (in_nd != out_nd) {
        throw py::value_error(
            "The input and output arrays must have the same dimension.");
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    auto const &same_logical_tensors =
        dpctl::tensor::overlap::SameLogicalTensors();
    if (overlap(in, out) && !same_logical_tensors(in, out)) {
        throw py::value_error(
            "The input and output arrays are overlapping segments of memory");
    }

    sycl::queue exec_q = descr.get_queue();
    if (!dpctl::utils::queues_are_compatible(exec_q,
                                             {in.get_queue(), out.get_queue()}))
    {
        throw py::value_error("USM allocations are not compatible with the "
                              "execution queue of the descriptor.");
    }

    const py::ssize_t *in_shape = in.get_shape_raw();
    const py::ssize_t *out_shape = out.get_shape_raw();
    const std::int64_t m = in_shape[in_nd - 1];
    const std::int64_t n = out_shape[out_nd - 1];

    std::int64_t in_size = 1;
    if (in_nd > 1) {
        for (int i = 0; i < in_nd - 1; ++i) {
            if (in_shape[i] != out_shape[i]) {
                throw py::value_error("The shape of the output array is not "
                                      "correct for the given input array.");
            }
            in_size *= in_shape[i];
        }
    }

    std::int64_t N;
    if (dom == mkl_dft::domain::REAL && is_forward) {
        // r2c FFT
        N = m / 2 + 1; // integer divide
        if (n != N) {
            throw py::value_error(
                "The shape of the output array is not correct for the given "
                "input array in real to complex FFT transform.");
        }
    }
    else {
        // c2c and c2r FFT. For c2r FFT, input is zero-padded in python side to
        // have the same size as output before calling this function
        N = m;
        if (n != N) {
            throw py::value_error("The shape of the output array is not "
                                  "correct for the given input array.");
        }
    }

    const std::size_t n_elems = in_size * N;
    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(out);
    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(out, n_elems);

    sycl::event fft_event = {};
    std::stringstream error_msg;
    bool is_exception_caught = false;

    try {
        if (is_forward) {
            using ScaleT_in = typename ScaleType<prec, dom, true>::type_in;
            using ScaleT_out = typename ScaleType<prec, dom, true>::type_out;
            ScaleT_in *in_ptr = in.get_data<ScaleT_in>();
            ScaleT_out *out_ptr = out.get_data<ScaleT_out>();
            fft_event = mkl_dft::compute_forward(descr.get_descriptor(), in_ptr,
                                                 out_ptr, depends);
        }
        else {
            using ScaleT_in = typename ScaleType<prec, dom, false>::type_in;
            using ScaleT_out = typename ScaleType<prec, dom, false>::type_out;
            ScaleT_in *in_ptr = in.get_data<ScaleT_in>();
            ScaleT_out *out_ptr = out.get_data<ScaleT_out>();
            fft_event = mkl_dft::compute_backward(descr.get_descriptor(),
                                                  in_ptr, out_ptr, depends);
        }
    } catch (oneapi::mkl::exception const &e) {
        error_msg
            << "Unexpected MKL exception caught during FFT() call:\nreason: "
            << e.what();
        is_exception_caught = true;
    } catch (sycl::exception const &e) {
        error_msg << "Unexpected SYCL exception caught during FFT() call:\n"
                  << e.what();
        is_exception_caught = true;
    }
    if (is_exception_caught) {
        throw std::runtime_error(error_msg.str());
    }

    sycl::event args_ev =
        dpctl::utils::keep_args_alive(exec_q, {in, out}, {fft_event});

    return std::make_pair(fft_event, args_ev);
}

} // namespace dpnp::extensions::fft
