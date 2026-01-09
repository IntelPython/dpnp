# *****************************************************************************
# Copyright (c) 2016, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
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

import argparse
import os
import sys

from _build_helper import (
    build_extension,
    clean_build_dir,
    err,
    get_dpctl_cmake_dir,
    install_editable,
    log_cmake_args,
    make_cmake_args,
    resolve_compilers,
    resolve_onemath,
)


def parse_args():
    p = argparse.ArgumentParser(description="Local dpnp build driver")

    # compiler and oneAPI relating options
    p.add_argument(
        "--c-compiler",
        type=str,
        default=None,
        help="Path or name of C compiler",
    )
    p.add_argument(
        "--cxx-compiler",
        type=str,
        default=None,
        help="Path or name of C++ compiler",
    )
    p.add_argument(
        "--compiler-root",
        type=str,
        default=None,
        help="Path to compiler installation root",
    )
    p.add_argument(
        "--oneapi",
        dest="oneapi",
        action="store_true",
        help="Use default oneAPI compiler layout",
    )

    # CMake relating options
    p.add_argument(
        "--generator", type=str, default="Ninja", help="CMake generator"
    )
    p.add_argument(
        "--cmake-executable",
        type=str,
        default=None,
        help="Path to CMake executable used by build",
    )
    p.add_argument(
        "--cmake-opts",
        type=str,
        default="",
        help="Additional options to pass directly to CMake",
    )
    p.add_argument(
        "--debug",
        dest="build_type",
        const="Debug",
        action="store_const",
        default="Release",
        help="Set build type to Debug (defaults to Release)",
    )
    p.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Enable verbose makefile output",
    )

    # platform target relating options
    p.add_argument(
        "--target-cuda",
        nargs="?",
        const="ON",
        default=None,
        help="Enable CUDA build. Architecture is optional to specify (e.g., --target-cuda=sm_80).",
    )
    p.add_argument(
        "--target-hip",
        required=False,
        type=str,
        help="Enable HIP backend. Architecture required to be specified  (e.g., --target-hip=gfx90a).",
    )

    # oneMath relating options
    p.add_argument(
        "--onemkl_interfaces",
        help="(DEPRECATED) Build using oneMath",
        dest="onemkl_interfaces",
        action="store_true",
    )
    p.add_argument(
        "--onemkl_interfaces_dir",
        help="(DEPRECATED) Local directory with source of oneMath",
        dest="onemkl_interfaces_dir",
        default=None,
        type=str,
    )
    p.add_argument(
        "--onemath",
        help="Build using oneMath",
        dest="onemath",
        action="store_true",
    )
    p.add_argument(
        "--onemath-dir",
        help="Local directory with source of oneMath",
        dest="onemath_dir",
        default=None,
        type=str,
    )

    # build relating options
    p.add_argument(
        "--clean",
        action="store_true",
        help="Remove build dir before rebuild",
    )
    p.add_argument(
        "--skip-editable",
        action="store_true",
        help="Skip pip editable install step",
    )

    return p.parse_args()


def main():
    if sys.platform not in ["cygwin", "win32", "linux"]:
        err(f"{sys.platform} not supported", "build_locally")

    args = parse_args()
    setup_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    c_compiler, cxx_compiler = resolve_compilers(
        args.oneapi, args.c_compiler, args.cxx_compiler, args.compiler_root
    )

    dpctl_cmake_dir = get_dpctl_cmake_dir()

    onemath, onemath_dir = resolve_onemath(
        args.onemath,
        args.onemath_dir,
        args.target_cuda,
        args.target_hip,
        args.onemkl_interfaces,
        args.onemkl_interfaces_dir,
    )

    # clean build dir if --clean set
    if args.clean:
        clean_build_dir(setup_dir)

    cmake_args = make_cmake_args(
        c_compiler=c_compiler,
        cxx_compiler=cxx_compiler,
        dpctl_cmake_dir=dpctl_cmake_dir,
        onemath=onemath,
        onemath_dir=onemath_dir,
        verbose=args.verbose,
        other_opts=args.cmake_opts,
    )

    # handle architecture conflicts
    if args.target_hip is not None and not args.target_hip.strip():
        err("--target-hip requires an explicit architecture", "build_locally")

    # CUDA/HIP targets
    if args.target_cuda:
        cmake_args += [f"-DDPNP_TARGET_CUDA={args.target_cuda}"]
    if args.target_hip:
        cmake_args += [f"-DDPNP_TARGET_HIP={args.target_hip}"]

    log_cmake_args(cmake_args, "build_locally")

    print("[build_locally] Building extensions in-place...")

    env = os.environ.copy()
    if args.oneapi and "DPL_ROOT" in env:
        env["DPL_ROOT_HINT"] = env["DPL_ROOT"]

    build_extension(
        setup_dir,
        env,
        cmake_args,
        cmake_executable=args.cmake_executable,
        generator=args.generator,
        build_type=args.build_type,
    )
    if not args.skip_editable:
        install_editable(setup_dir, env)
    else:
        print("[build_locally] Skipping editable install (--skip-editable)")

    print("[build_locally] Build complete")


if __name__ == "__main__":
    main()
