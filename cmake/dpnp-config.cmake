#.rst:
#
# Find the include directory for ``dpnp4pybind11.hpp`` and dpnp tensor kernels.
#
# This module sets the following variables:
#
# ``Dpnp_FOUND``
#   True if DPNP was found.
# ``Dpnp_INCLUDE_DIR``
#   The include directory needed to use dpnp.
# ``Dpnp_TENSOR_INCLUDE_DIR``
#   The include directory for tensor kernels implementation.
# ``Dpnp_VERSION``
#   The version of dpnp found.
#
# The module will also explicitly define two cache variables:
#
# ``Dpnp_INCLUDE_DIR``
# ``Dpnp_TENSOR_INCLUDE_DIR``
#

if(NOT Dpnp_FOUND)
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

    if(Python_EXECUTABLE)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -m dpnp --include-dir
            OUTPUT_VARIABLE _dpnp_include_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c "import dpnp; print(dpnp.__version__)"
            OUTPUT_VARIABLE Dpnp_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()
endif()

find_path(
    Dpnp_INCLUDE_DIR
    dpnp4pybind11.hpp
    PATHS "${_dpnp_include_dir}" "${Python_INCLUDE_DIRS}"
    PATH_SUFFIXES dpnp/include
)
get_filename_component(_dpnp_dir "${Dpnp_INCLUDE_DIR}" DIRECTORY)

find_path(
    Dpnp_TENSOR_INCLUDE_DIR
    kernels
    utils
    PATHS "${_dpnp_dir}/tensor/libtensor/include"
)

set(Dpnp_INCLUDE_DIRS ${Dpnp_INCLUDE_DIR})

# handle the QUIETLY and REQUIRED arguments and set Dpnp_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    Dpnp
    REQUIRED_VARS Dpnp_INCLUDE_DIR Dpnp_TENSOR_INCLUDE_DIR
    VERSION_VAR Dpnp_VERSION
)

mark_as_advanced(Dpnp_INCLUDE_DIR)
mark_as_advanced(Dpnp_TENSOR_INCLUDE_DIR)
