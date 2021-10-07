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

/*
 * This header file is for interface Cython with C++.
 * It should not contains any backend specific headers (like SYCL or math library) because
 * all included headers will be exposed in Cython compilation procedure
 *
 * We would like to avoid backend specific things in higher level Cython modules.
 * Any backend interface functions and types should be defined here.
 *
 * Also, this file should contains documentation on functions and types
 * which are used in the interface
 */

#pragma once
#ifndef BACKEND_IFACE_FFT_H // Cython compatibility
#define BACKEND_IFACE_FFT_H

// #include <cstdint>

// TODO needs to move into common place like backend_defs.hpp
#ifdef _WIN32
#define INP_DLLEXPORT __declspec(dllexport)
#else
#define INP_DLLEXPORT
#endif

// TODO needs to move into common place like backend_defs.hpp
/**
 * @defgroup BACKEND_FFT_API Backend C++ library interface FFT API
 * @{
 * This section describes Backend API for Discrete Fourier Transform (DFT) part.
 * @}
 */

/**
 * @ingroup BACKEND_FFT_API
 * @brief 1D discrete Fourier Transform.
 *
 * Compute the one-dimensional discrete Fourier Transform.
 *
 * @param[in]  array_in        Input array.
 * @param[out] result          Output array.
 * @param[in]  input_shape     Array with shape information for input array.
 * @param[in]  output_shape    Array with shape information for output array.
 * @param[in]  shape_size      Number of elements in @ref input_shape or @ref output_shape arrays.
 * @param[in]  axis            Axis ID to compute by.
 * @param[in]  input_boundarie Limit number of elements for @ref axis.
 * @param[in]  inverse         Using inverse algorithm.
 * @param[in]  norm            Normalization mode. 0 - backward, 1 - forward.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_fft_fft_c(const void* array_in,
                                  void* result,
                                  const long* input_shape,
                                  const long* output_shape,
                                  size_t shape_size,
                                  long axis,
                                  long input_boundarie,
                                  size_t inverse,
                                  const size_t norm);
#endif // BACKEND_IFACE_FFT_H
