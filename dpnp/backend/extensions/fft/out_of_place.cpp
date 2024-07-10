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

#include <oneapi/mkl/dfti.hpp>
#include <sycl/sycl.hpp>

#include <dpctl4pybind11.hpp>

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
    bool committed = descr.is_committed();
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

    py::ssize_t in_size = in.get_size();
    py::ssize_t out_size = out.get_size();
    if (in_size != out_size) {
        throw py::value_error("The size of the input vector must be "
                              "equal to the size of the output vector.");
    }

    size_t src_nelems = in_size;
    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(out);
    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(out, src_nelems);

    using ScaleT = typename ScaleType<prec>::value_type;
    std::complex<ScaleT> *in_ptr = in.get_data<std::complex<ScaleT>>();
    std::complex<ScaleT> *out_ptr = out.get_data<std::complex<ScaleT>>();

    sycl::event fft_event = {};
    std::stringstream error_msg;
    bool is_exception_caught = false;

    try {
        if (is_forward) {
            fft_event = mkl_dft::compute_forward(descr.get_descriptor(), in_ptr,
                                                 out_ptr, depends);
        }
        else {
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

// Explicit instantiations
template std::pair<sycl::event, sycl::event> compute_fft_out_of_place(
    DescriptorWrapper<mkl_dft::precision::SINGLE, mkl_dft::domain::COMPLEX>
        &descr,
    const dpctl::tensor::usm_ndarray &in,
    const dpctl::tensor::usm_ndarray &out,
    const bool is_forward,
    const std::vector<sycl::event> &depends);

template std::pair<sycl::event, sycl::event> compute_fft_out_of_place(
    DescriptorWrapper<mkl_dft::precision::DOUBLE, mkl_dft::domain::COMPLEX>
        &descr,
    const dpctl::tensor::usm_ndarray &in,
    const dpctl::tensor::usm_ndarray &out,
    const bool is_forward,
    const std::vector<sycl::event> &depends);

} // namespace dpnp::extensions::fft
