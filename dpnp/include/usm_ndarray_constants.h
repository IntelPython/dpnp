/* *****************************************************************************
 * Copyright (c) 2026, Intel Corporation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of the copyright holder nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 * ****************************************************************************/

#ifndef DPNP_TENSOR_USM_NDARRAY_CONSTANTS_H
#define DPNP_TENSOR_USM_NDARRAY_CONSTANTS_H

/* Array contiguity flags */
enum
{
    USM_ARRAY_C_CONTIGUOUS_VALUE = 1,
    USM_ARRAY_F_CONTIGUOUS_VALUE = 2,
    USM_ARRAY_WRITABLE_VALUE = 4
};

/* These typenum values are aligned to values in NumPy */
enum
{
    UAR_BOOL_VALUE = 0,
    UAR_BYTE_VALUE = 1,
    UAR_UBYTE_VALUE = 2,
    UAR_SHORT_VALUE = 3,
    UAR_USHORT_VALUE = 4,
    UAR_INT_VALUE = 5,
    UAR_UINT_VALUE = 6,
    UAR_LONG_VALUE = 7,
    UAR_ULONG_VALUE = 8,
    UAR_LONGLONG_VALUE = 9,
    UAR_ULONGLONG_VALUE = 10,
    UAR_FLOAT_VALUE = 11,
    UAR_DOUBLE_VALUE = 12,
    UAR_CFLOAT_VALUE = 14,
    UAR_CDOUBLE_VALUE = 15,
    UAR_TYPE_SENTINEL_VALUE = 17,
    UAR_HALF_VALUE = 23
};

#endif /* DPNP_TENSOR_USM_NDARRAY_CONSTANTS_H */
