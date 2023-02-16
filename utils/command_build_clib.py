# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2023, Intel Corporation
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

"""Module to customize build_clib command
Originally, 'build_clib' command produce static C library only.
This modification add:
 - build shared C library
 - copy this library to the project tree
 - a check if source needs to be rebuilt based on time stamp
 - a check if librayr needs to be rebuilt based on time stamp
"""

import os
import sys

from ctypes.util import find_library as find_shared_lib
from setuptools.command import build_clib
from distutils import log
from distutils.dep_util import newer_group
from distutils.file_util import copy_file

from utils.dpnp_build_utils import find_cmplr, find_dpl, find_mathlib, find_python_env

IS_WIN = False
IS_MAC = False
IS_LIN = False

if 'linux' in sys.platform:
    IS_LIN = True
elif sys.platform == 'darwin':
    IS_MAC = True
elif sys.platform in ['win32', 'cygwin']:
    IS_WIN = True
else:
    raise EnvironmentError("DPNP: " + sys.platform + " not supported")

"""
Set compiler for the project
"""
# default variables (for Linux)
_project_compiler = "icpx"
_project_linker = "icpx"
_project_cmplr_flag_sycl_devel = ["-fsycl-device-code-split=per_kernel", "-fno-approx-func", "-fno-finite-math-only"]
_project_cmplr_flag_sycl = ["-fsycl"]
_project_cmplr_flag_stdcpp_static = []  # This brakes TBB ["-static-libstdc++", "-static-libgcc"]
_project_cmplr_flag_compatibility = ["-Wl,--enable-new-dtags"]
_project_cmplr_flag_lib = ["-shared"]
_project_cmplr_flag_release_build = ["-O3", "-DNDEBUG", "-fPIC"]
_project_cmplr_flag_debug_build = ["-g", "-O1", "-W", "-Wextra", "-Wshadow", "-Wall", "-Wstrict-prototypes", "-fPIC"]
_project_cmplr_flag_default_build = []
_project_cmplr_macro = []
_project_force_build = False
_project_sycl_queue_control_macro = [("DPNP_LOCAL_QUEUE", "1")]
_project_rpath = ["$ORIGIN", os.path.join("$ORIGIN", "..")]
_dpctrl_include = []
_dpctrl_libpath = []
_dpctrl_lib = []
_sdl_cflags = ["-fstack-protector-strong",
               "-fPIC", "-D_FORTIFY_SOURCE=2",
               "-Wformat",
               "-Wformat-security",
               "-fno-strict-overflow",
               "-fno-delete-null-pointer-checks"]
_sdl_ldflags = ["-Wl,-z,noexecstack,-z,relro,-z,now"]

# TODO remove when it will be fixed on TBB side. Details:
# In GCC versions 9 and 10 the application that uses Parallel STL algorithms may fail to compile due to incompatible
# interface changes between earlier versions of Intel TBB and oneTBB. Disable support for Parallel STL algorithms
# by defining PSTL_USE_PARALLEL_POLICIES (in GCC 9), _GLIBCXX_USE_TBB_PAR_BACKEND (in GCC 10) macro to zero
# before inclusion of the first standard header file in each translation unit.
_project_cmplr_macro += [("PSTL_USE_PARALLEL_POLICIES", "0"), ("_GLIBCXX_USE_TBB_PAR_BACKEND", "0")]

# disable PSTL predefined policies objects (global queues, prevent fail on Windows)
_project_cmplr_macro += [("ONEDPL_USE_PREDEFINED_POLICIES", "0")]

try:
    """
    Detect external SYCL queue handling library
    """
    import dpctl

    _dpctrl_include += [dpctl.get_include()]
    # _dpctrl_libpath = for package build + for local build
    _dpctrl_libpath = ["$ORIGIN/../dpctl"] + [os.path.join(dpctl.get_include(), '..')]
    _dpctrl_lib = ["DPCTLSyclInterface"]
except ImportError:
    """
    Set local SYCL queue handler
    """
    _project_cmplr_macro += _project_sycl_queue_control_macro

# other OS specific
if IS_WIN:
    _project_compiler = "dpcpp"
    _project_linker = "lld-link"
    _project_cmplr_flag_sycl = []
    _project_cmplr_flag_stdcpp_static = []
    _project_cmplr_flag_compatibility = []
    _project_cmplr_flag_lib = ["/DLL"]
    _project_cmplr_flag_release_build += _project_cmplr_flag_sycl_devel
    _project_cmplr_macro += [("_WIN", "1")]
    _project_rpath = []
    # TODO this flag creates unexpected behavior during compilation, need to be fixed
    # _sdl_cflags = ["-GS"]
    _sdl_cflags = []
    _sdl_ldflags = ["-NXCompat", "-DynamicBase"]

"""
Get the project build type
"""
__dpnp_debug__ = os.environ.get('DPNP_DEBUG', None)
if __dpnp_debug__ is not None:
    """
    Debug configuration
    """
    _project_cmplr_flag_default_build = _project_cmplr_flag_debug_build
else:
    """
    Release configuration
    """
    _project_cmplr_flag_sycl += _project_cmplr_flag_sycl_devel
    _project_cmplr_flag_default_build = _project_cmplr_flag_release_build

"""
Get the math library environemnt
"""
_project_cmplr_macro += [("MKL_ILP64", "1")]  # using 64bit integers in MKL interface (long)
if IS_LIN:
    _mathlibs = ["mkl_sycl", "mkl_intel_ilp64", "mkl_sequential",
                 "mkl_core", "sycl", "OpenCL", "pthread", "m", "dl"]
elif IS_WIN:
    _sycl_lib = None
    for lib in {"sycl", "sycl6", "sycl7"}:
        if find_shared_lib(lib):
            _sycl_lib = lib
    if not _sycl_lib:
        raise EnvironmentError("DPNP: sycl library is not found")

    _mathlibs = ["mkl_sycl_dll", "mkl_intel_ilp64_dll", "mkl_tbb_thread_dll", "mkl_core_dll", _sycl_lib, "OpenCL", "tbb"]

"""
Final set of arguments for extentions
"""
_project_extra_link_args = _project_cmplr_flag_compatibility + _project_cmplr_flag_stdcpp_static + \
    ["-Wl,-rpath," + x for x in _project_rpath] + _sdl_ldflags
_project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_project_backend_dir = [os.path.join(_project_dir, "dpnp", "backend", "include"),
                        os.path.join(_project_dir, "dpnp", "backend", "src")  # not a public headers location
                        ]

dpnp_backend_c_description = [
    ["dpnp_backend_c",
        {
            "sources": [
                "dpnp/backend/kernels/dpnp_krnl_arraycreation.cpp",
                "dpnp/backend/kernels/dpnp_krnl_bitwise.cpp",
                "dpnp/backend/kernels/dpnp_krnl_common.cpp",
                "dpnp/backend/kernels/dpnp_krnl_elemwise.cpp",
                "dpnp/backend/kernels/dpnp_krnl_fft.cpp",
                "dpnp/backend/kernels/dpnp_krnl_indexing.cpp",
                "dpnp/backend/kernels/dpnp_krnl_linalg.cpp",
                "dpnp/backend/kernels/dpnp_krnl_logic.cpp",
                "dpnp/backend/kernels/dpnp_krnl_manipulation.cpp",
                "dpnp/backend/kernels/dpnp_krnl_mathematical.cpp",
                "dpnp/backend/kernels/dpnp_krnl_random.cpp",
                "dpnp/backend/kernels/dpnp_krnl_reduction.cpp",
                "dpnp/backend/kernels/dpnp_krnl_searching.cpp",
                "dpnp/backend/kernels/dpnp_krnl_sorting.cpp",
                "dpnp/backend/kernels/dpnp_krnl_statistics.cpp",
                "dpnp/backend/src/dpnp_iface_fptr.cpp",
                "dpnp/backend/src/memory_sycl.cpp",
                "dpnp/backend/src/constants.cpp",
                "dpnp/backend/src/queue_sycl.cpp",
                "dpnp/backend/src/verbose.cpp",
                "dpnp/backend/src/dpnp_random_state.cpp"
            ],
        }
     ]
]


def _compiler_compile(self, sources,
                      output_dir=None, macros=None, include_dirs=None, debug=0,
                      extra_preargs=None, extra_postargs=None, depends=None):

    if not self.initialized:
        self.initialize()
    compile_info = self._setup_compile(output_dir, macros, include_dirs,
                                       sources, depends, extra_postargs)
    macros, objects, extra_postargs, pp_opts, build = compile_info

    compile_opts = extra_preargs or []
    compile_opts.append('/c')
    if debug:
        compile_opts.extend(self.compile_options_debug)
    else:
        compile_opts.extend(self.compile_options)

    add_cpp_opts = False

    for obj in objects:
        try:
            src, ext = build[obj]
        except KeyError:
            continue
        if debug:
            # pass the full pathname to MSVC in debug mode,
            # this allows the debugger to find the source file
            # without asking the user to browse for it
            src = os.path.abspath(src)

        # Anaconda/conda-forge customisation, we want our pdbs to be
        # relocatable:
        # https://developercommunity.visualstudio.com/comments/623156/view.html
        d1trimfile_opts = []
        # if 'SRC_DIR' in os.environ:
        # d1trimfile_opts.append("/d1trimfile:" + os.environ['SRC_DIR'])

        if ext in self._c_extensions:
            input_opt = "/Tc" + src
        elif ext in self._cpp_extensions:
            input_opt = "/Tp" + src
            add_cpp_opts = True
        elif ext in self._rc_extensions:
            # compile .RC to .RES file
            input_opt = src
            output_opt = "/fo" + obj
            try:
                self.spawn([self.rc] + pp_opts + [output_opt, input_opt])
            except DistutilsExecError as msg:
                raise CompileError(msg)
            continue
        elif ext in self._mc_extensions:
            # Compile .MC to .RC file to .RES file.
            #   * '-h dir' specifies the directory for the
            #     generated include file
            #   * '-r dir' specifies the target directory of the
            #     generated RC file and the binary message resource
            #     it includes
            #
            # For now (since there are no options to change this),
            # we use the source-directory for the include file and
            # the build directory for the RC file and message
            # resources. This works at least for win32all.
            h_dir = os.path.dirname(src)
            rc_dir = os.path.dirname(obj)
            try:
                # first compile .MC to .RC and .H file
                self.spawn([self.mc, '-h', h_dir, '-r', rc_dir, src])
                base, _ = os.path.splitext(os.path.basename(src))
                rc_file = os.path.join(rc_dir, base + '.rc')
                # then compile .RC to .RES file
                self.spawn([self.rc, "/fo" + obj, rc_file])

            except DistutilsExecError as msg:
                raise CompileError(msg)
            continue
        else:
            # how to handle this file?
            raise CompileError("Don't know how to compile {} to {}"
                               .format(src, obj))

        args = [self.cc] + compile_opts + pp_opts + d1trimfile_opts
        if add_cpp_opts:
            args.append('/EHsc')
        args.append(input_opt)
        args.append("/Fo" + obj)
        args.extend(extra_postargs)

        try:
            self.spawn(args)
        except DistutilsExecError as msg:
            raise CompileError(msg)

    return objects


class custom_build_clib(build_clib.build_clib):

    def build_libraries(self, libraries):
        """
        This function is overloaded to the original function in build_clib.py file
        """

        for (lib_name, build_info) in libraries:
            c_library_name = self.compiler.library_filename(lib_name, lib_type='shared')
            c_library_filename = os.path.join(self.build_clib, c_library_name)
            dest_filename = "dpnp"  # TODO need to fix destination directory

            sources = build_info.get('sources')
            if sources is None or not isinstance(sources, (list, tuple)):
                err_msg = f"in 'libraries' option (library '{lib_name}'),"
                err_msg += f" 'sources' must be present and must be a list of source filenames"
                raise DistutilsSetupError(err_msg)

            sources = list(sources)

            log.info(f"DPNP: building {lib_name} library")

            """
            Get the compiler environemnt
            """
            _cmplr_include, _cmplr_libpath = find_cmplr(verbose=True)
            _mathlib_include, _mathlib_path = find_mathlib(verbose=True)
            # _, _omp_libpath = find_omp(verbose=True)
            _dpl_include, _ = find_dpl(verbose=True)
            _py_env_include, _py_env_lib = find_python_env(verbose=True)

            macros = _project_cmplr_macro
            include_dirs = _cmplr_include + _dpl_include + _mathlib_include + _project_backend_dir + _dpctrl_include + _py_env_include
            libraries = _mathlibs + _dpctrl_lib
            library_dirs = _mathlib_path + _dpctrl_libpath + _py_env_lib  # + _omp_libpath
            runtime_library_dirs = _project_rpath + _dpctrl_libpath
            extra_preargs = _project_cmplr_flag_sycl + _sdl_cflags
            extra_link_postargs = _project_cmplr_flag_lib
            extra_link_preargs = _project_cmplr_flag_compatibility + _sdl_ldflags
            force_build = _project_force_build
            compiler = [_project_compiler]
            linker = [_project_linker]
            default_flags = _project_cmplr_flag_default_build
            language = "c++"

            # set compiler and options
            self.compiler.compiler_so = compiler + default_flags
            self.compiler.compiler = self.compiler.compiler_so
            self.compiler.compiler_cxx = self.compiler.compiler_so
            self.compiler.linker_so = linker + default_flags
            self.compiler.linker_exe = self.compiler.linker_so

            os.environ["CC"] = _project_compiler

            objects = []
            """
            Build object files from sources
            """
            if IS_WIN:
                self.compiler.compile = _compiler_compile

            for source_it in sources:
                obj_file_list = self.compiler.object_filenames([source_it], strip_dir=0, output_dir=self.build_temp)
                obj_file = "".join(obj_file_list)  # convert from list to file name

                newer_than_obj = newer_group([source_it], obj_file, missing="newer")
                if force_build or newer_than_obj:
                    if IS_WIN:
                        obj_file_list = self.compiler.compile(self.compiler,
                                                              [source_it],
                                                              output_dir=self.build_temp,
                                                              macros=macros,
                                                              include_dirs=include_dirs,
                                                              extra_preargs=extra_preargs,
                                                              debug=self.debug)
                    else:
                        obj_file_list = self.compiler.compile([source_it],
                                                              output_dir=self.build_temp,
                                                              macros=macros,
                                                              include_dirs=include_dirs,
                                                              extra_preargs=extra_preargs,
                                                              debug=self.debug)
                    objects.extend(obj_file_list)
                else:
                    objects.append(obj_file)

            """
            Build library file from objects
            """
            newer_than_lib = newer_group(objects, c_library_filename, missing="newer")
            if force_build or newer_than_lib:
                # TODO very brute way, need to refactor
                if IS_WIN:
                    link_command = " ".join(compiler)
                    link_command += " " + " ".join(default_flags)
                    link_command += " " + " ".join(objects)  # specify *.obj files
                    link_command += " /link"  # start linker options
                    link_command += " " + " ".join(extra_link_preargs)
                    link_command += " " + ".lib ".join(libraries) + ".lib"  # libraries
                    link_command += " /LIBPATH:" + " /LIBPATH:".join(library_dirs)
                    link_command += " /OUT:" + c_library_filename  # output file name
                    link_command += " " + " ".join(extra_link_postargs)
                    print(link_command)
                    os.system(link_command)
                else:
                    self.compiler.link_shared_lib(objects,
                                                  lib_name,
                                                  output_dir=self.build_clib,
                                                  libraries=libraries,
                                                  library_dirs=library_dirs,
                                                  runtime_library_dirs=runtime_library_dirs,
                                                  extra_preargs=extra_preargs + extra_link_preargs,
                                                  extra_postargs=extra_link_postargs,
                                                  debug=self.debug,
                                                  build_temp=self.build_temp,
                                                  target_lang=language)

            """
            Copy library to the destination path
            """
            copy_file(c_library_filename, dest_filename, verbose=self.verbose, dry_run=self.dry_run)
            # TODO very brute way, need to refactor
            if c_library_filename.endswith(".dll"):
                copy_file(c_library_filename.replace(".dll", ".lib"),
                          dest_filename, verbose=self.verbose, dry_run=self.dry_run)

            log.info(f"DPNP: building {lib_name} library finished")
