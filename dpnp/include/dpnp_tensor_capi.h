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
//
//===---------------------------------------------------------------------===//
///
/// \file
/// This file provides access to dpnp tensor's C-API, including:
/// - dpctl C-API (from external dpctl package - SYCL interface)
/// - dpnp tensor C-API (usm_ndarray)
//===---------------------------------------------------------------------===//

#pragma once

// Include dpctl C-API headers explicitly from external dpctl package (SYCL
// interface)
// TODO: Once dpctl removes its tensor module and stabilizes dpctl_capi.h,
// we can simplify to just: #include "dpctl_capi.h"
// For now, explicit includes ensure we only get SYCL interface without tensor.

#include "syclinterface/dpctl_sycl_extension_interface.h"
#include "syclinterface/dpctl_sycl_types.h"

#ifdef __cplusplus
#define CYTHON_EXTERN_C extern "C"
#else
#define CYTHON_EXTERN_C
#endif

#include "dpctl/_sycl_context.h"
#include "dpctl/_sycl_context_api.h"
#include "dpctl/_sycl_device.h"
#include "dpctl/_sycl_device_api.h"
#include "dpctl/_sycl_event.h"
#include "dpctl/_sycl_event_api.h"
#include "dpctl/_sycl_queue.h"
#include "dpctl/_sycl_queue_api.h"
#include "dpctl/memory/_memory.h"
#include "dpctl/memory/_memory_api.h"
#include "dpctl/program/_program.h"
#include "dpctl/program/_program_api.h"

// Include the generated Cython C-API headers for usm_ndarray
// These headers are generated during build and placed in the build directory
#include "dpnp/tensor/_usmarray.h"
#include "dpnp/tensor/_usmarray_api.h"

/*
 * Function to import dpnp tensor C-API and make it available.
 * This imports both:
 * - dpctl C-API (from external dpctl package - SYCL interface)
 * - dpnp tensor C-API (tensor interface - usm_ndarray)
 *
 * C functions can use dpnp tensor's C-API functions without linking to
 * shared objects defining these symbols, if they call `import_dpnp_tensor()`
 * prior to using those symbols.
 *
 * It is declared inline to allow multiple definitions in
 * different translation units.
 */
static inline void import_dpnp_tensor(void)
{
    // Import dpctl SYCL interface
    // TODO: Once dpctl removes its tensor module and stabilizes dpctl_capi.h,
    // we can simplify to just: import_dpctl()
    import_dpctl___sycl_device();
    import_dpctl___sycl_context();
    import_dpctl___sycl_event();
    import_dpctl___sycl_queue();
    import_dpctl__memory___memory();
    import_dpctl__program___program();
    // Import dpnp tensor interface
    import_dpnp__tensor___usmarray();
    return;
}
