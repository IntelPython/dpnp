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
 * This header file contains internal function declarations related to FPTR interface.
 * It should not contains public declarations
 */

#pragma once
#ifndef BACKEND_FPTR_H // Cython compatibility
#define BACKEND_FPTR_H

#include <map>

#include <backend_iface_fptr.hpp>

/**
 * Data storage type of the FPTR interface
 *
 * map[FunctionName][InputType2][InputType2]
 *
 * Function name is enum DPNPFuncName
 * InputTypes are presented as enum DPNPFuncType
 *
 * contains structure with kernel information
 *
 * if the kernel requires only one input type - use same type for both parameters
 *
 */
typedef std::map<DPNPFuncType, DPNPFuncData_t> map_2p_t;
typedef std::map<DPNPFuncType, map_2p_t> map_1p_t;
typedef std::map<DPNPFuncName, map_1p_t> func_map_t;

/**
 * Internal shortcuts for Data type enum values
 */
const DPNPFuncType eft_INT = DPNPFuncType::DPNP_FT_INT;
const DPNPFuncType eft_LNG = DPNPFuncType::DPNP_FT_LONG;
const DPNPFuncType eft_FLT = DPNPFuncType::DPNP_FT_FLOAT;
const DPNPFuncType eft_DBL = DPNPFuncType::DPNP_FT_DOUBLE;

/**
 * FPTR interface initialization functions
 */
void func_map_init_elemwise(func_map_t& fmap);
void func_map_init_linalg(func_map_t& fmap);
void func_map_init_manipulation(func_map_t& fmap);
void func_map_init_mathematical(func_map_t& fmap);
void func_map_init_reduction(func_map_t& fmap);
void func_map_init_searching(func_map_t& fmap);
void func_map_init_sorting(func_map_t& fmap);
void func_map_init_statistics(func_map_t& fmap);

#endif // BACKEND_FPTR_H
