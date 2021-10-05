# *****************************************************************************
# Copyright (c) 2016-2020, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

# The following variables are optionally searched for defaults
#  DPLROOT:         Environment variable to specify custom search place
#  ONEAPI_ROOT:     Environment variable to specify search place from oneAPI
#
# The following are set after configuration is done:
#  DPL_FOUND
#  DPL_INCLUDE_DIR

include(FindPackageHandleStandardArgs)

set(DPNP_ONEAPI_DPL "$ENV{DPNP_ONEAPI_ROOT}/dpl/latest" CACHE PATH "Folder contains DPL files from ONEAPI_ROOT")

if(DEFINED ENV{DPLROOT})
  set(DPNP_DPLROOT "$ENV{DPLROOT}" CACHE PATH "Folder contains DPL files from DPLROOT")
endif()

find_path(
  DPL_INCLUDE_DIR oneapi/dpl/algorithm
  HINTS ${DPNP_DPLROOT} ${DPNP_ONEAPI_DPL} ENV CONDA_PREFIX ENV PREFIX # search order is important
  PATH_SUFFIXES include linux/include
  DOC "Path to DPL include files")

find_package_handle_standard_args(DPL DEFAULT_MSG DPL_INCLUDE_DIR)

if(DPL_FOUND)
  message(STATUS "Found DPL:                       (include: ${DPL_INCLUDE_DIR})")
  # mark_as_advanced(DPNP_DPLROOT DPL_INCLUDE_DIR)
endif()
