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
#  PSTL_ROOT_DIR:     Base directory where all components are found
#
# The following are set after configuration is done:
#  PSTL_FOUND
#  PSTL_INCLUDE_DIR

include(FindPackageHandleStandardArgs)

set(PSTL_ROOT_DIR
    "${DPNP_ONEAPI_ROOT}/dpl"
    CACHE PATH "Folder contains PSTL headers")

find_path(
  PSTL_INCLUDE_DIR oneapi/dpl/algorithm
  HINTS ENV CONDA_PREFIX ${PSTL_ROOT_DIR} # search order is important
  PATH_SUFFIXES include latest/linux/include
  DOC "Path to PSTL include files")

find_package_handle_standard_args(PSTL DEFAULT_MSG PSTL_INCLUDE_DIR)

if(PSTL_FOUND)
  message(STATUS "Found PSTL:                      (include: ${PSTL_INCLUDE_DIR})")
  # mark_as_advanced(PSTL_ROOT_DIR PSTL_INCLUDE_DIR)
endif()
