# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__WINDOWS_INTEL)
  return()
endif()
set(__WINDOWS_INTEL 1)

include(Platform/Windows-MSVC)
macro(__windows_compiler_intel lang)
  __windows_compiler_msvc(${lang})

  set(CMAKE_${lang}_LINK_EXECUTABLE    "<CMAKE_${lang}_COMPILER> ${CMAKE_CL_NOLOGO} <CMAKE_${lang}_LINK_FLAGS> <OBJECTS> ${CMAKE_START_TEMP_FILE} -link -out:<TARGET> -implib:<TARGET_IMPLIB> -pdb:<TARGET_PDB> -version:<TARGET_VERSION_MAJOR>.<TARGET_VERSION_MINOR>${_PLATFORM_LINK_FLAGS} <LINK_FLAGS> <LINK_LIBRARIES>${CMAKE_END_TEMP_FILE}")
  set(CMAKE_${lang}_CREATE_SHARED_LIBRARY    "<CMAKE_${lang}_COMPILER> ${CMAKE_CL_NOLOGO} <CMAKE_${lang}_LINK_FLAGS> <OBJECTS> ${CMAKE_START_TEMP_FILE} -LD -link -out:<TARGET> -implib:<TARGET_IMPLIB> -pdb:<TARGET_PDB> -version:<TARGET_VERSION_MAJOR>.<TARGET_VERSION_MINOR>${_PLATFORM_LINK_FLAGS} <LINK_FLAGS> <LINK_LIBRARIES> ${CMAKE_END_TEMP_FILE}")
  set(CMAKE_${lang}_CREATE_SHARED_MODULE ${CMAKE_${lang}_CREATE_SHARED_LIBRARY})
  if (NOT "${lang}" STREQUAL "Fortran")    # Fortran driver does not support -fuse-ld, yet
    set(CMAKE_${lang}_CREATE_STATIC_LIBRARY      "<CMAKE_${lang}_COMPILER> ${CMAKE_CL_NOLOGO} <CMAKE_${lang}_LINK_FLAGS> <OBJECTS> ${CMAKE_START_TEMP_FILE} -fuse-ld=llvm-lib -o <TARGET> -link <LINK_FLAGS> <LINK_LIBRARIES> ${CMAKE_END_TEMP_FILE}")
  endif()
  set(CMAKE_DEPFILE_FLAGS_${lang} "-QMMD -QMT <DEP_TARGET> -QMF <DEP_FILE>")
  set(CMAKE_${lang}_DEPFILE_FORMAT gcc)

endmacro()
