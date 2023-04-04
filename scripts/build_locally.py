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

import os
import subprocess
import sys
import dpctl


def run(
    use_oneapi=True,
    build_type="Release",
    c_compiler=None,
    cxx_compiler=None,
    compiler_root=None,
    cmake_executable=None,
    verbose=False,
    cmake_opts="",
):
    build_system = None

    if "linux" in sys.platform:
        build_system = "Ninja"
    elif sys.platform in ["win32", "cygwin"]:
        build_system = "Ninja"
    else:
        assert False, sys.platform + " not supported"

    setup_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmake_args = [
        sys.executable,
        "setup.py",
        "develop",
    ]
    if cmake_executable:
        cmake_args += [
            "--cmake-executable=" + cmake_executable,
        ]
    dpctl_module_path = os.path.join(dpctl.get_include(), "..", "resources", "cmake")
    cmake_args += [
        "--build-type=" + build_type,
        "--generator=" + build_system,
        "--",
        "-DCMAKE_C_COMPILER:PATH=" + c_compiler,
        "-DCMAKE_CXX_COMPILER:PATH=" + cxx_compiler,
        "-DDPCTL_MODULE_PATH:PATH=" + dpctl_module_path,
    ]
    if verbose:
        cmake_args += [
            "-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON",
        ]
    if cmake_opts:
        cmake_args += cmake_opts.split()
    if use_oneapi:
        if "DPL_ROOT" in os.environ:
            os.environ["DPL_ROOT_HINT"] = os.environ["DPL_ROOT"]
    subprocess.check_call(
        cmake_args, shell=False, cwd=setup_dir, env=os.environ
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Driver to build dpnp for in-place installation"
    )
    driver = parser.add_argument_group(title="Coverage driver arguments")
    driver.add_argument("--c-compiler", help="Name of C compiler", default=None)
    driver.add_argument(
        "--cxx-compiler", help="Name of C++ compiler", default=None
    )
    driver.add_argument(
        "--oneapi",
        help="Set if using one-API installation",
        dest="oneapi",
        action="store_true",
    )
    driver.add_argument(
        "--debug",
        default="Release",
        const="Debug",
        action="store_const",
        help="Set the compilation mode to debugging",
    )
    driver.add_argument(
        "--compiler-root",
        type=str,
        help="Path to compiler home directory",
        default=None,
    )
    driver.add_argument(
        "--cmake-executable",
        type=str,
        help="Path to cmake executable",
        default=None,
    )
    driver.add_argument(
        "--verbose",
        help="Build using vebose makefile mode",
        dest="verbose",
        action="store_true",
    )
    driver.add_argument(
        "--cmake-opts",
        help="DPCTLSyclInterface uses Google logger",
        dest="cmake_opts",
        default="",
        type=str,
    )
    args = parser.parse_args()

    args_to_validate = [
        "c_compiler",
        "cxx_compiler",
        "compiler_root",
    ]

    if args.oneapi or (
        args.c_compiler is None
        and args.cxx_compiler is None
        and args.compiler_root is None
    ):
        args.c_compiler = "icx"
        args.cxx_compiler = "icpx" if "linux" in sys.platform else "icx"
        args.compiler_root = None
    else:
        cr = args.compiler_root
        if isinstance(cr, str) and os.path.exists(cr):
            if args.c_compiler is None:
                args.c_compiler = "icx"
            if args.cxx_compiler is None:
                args.cxx_compiler = "icpx" if "linux" in sys.platform else "icx"
        else:
            raise RuntimeError(
                "Option 'compiler-root' must be provided when "
                "using non-default DPC++ layout."
            )
        args_to_validate = [
            "c_compiler",
            "cxx_compiler",
        ]
        for p in args_to_validate:
            arg = getattr(args, p)
            assert isinstance(arg, str)
            if not os.path.exists(arg):
                arg2 = os.path.join(cr, arg)
                if os.path.exists(arg2):
                    arg = arg2
                    setattr(args, p, arg)
            if not os.path.exists(arg):
                opt_name = p.replace("_", "-")
                raise RuntimeError(f"Option {opt_name} value {arg} must exist.")

    run(
        use_oneapi=args.oneapi,
        build_type=args.debug,
        c_compiler=args.c_compiler,
        cxx_compiler=args.cxx_compiler,
        compiler_root=args.compiler_root,
        cmake_executable=args.cmake_executable,
        verbose=args.verbose,
        cmake_opts=args.cmake_opts,
    )
