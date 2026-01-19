# *****************************************************************************
# Copyright (c) 2026, Intel Corporation
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

import os
import shutil
import subprocess
import sys
import warnings


def get_dpctl_cmake_dir():
    """
    If dpctl is locally built using `script/build_locally.py`, it is needed
    to pass the -DDpctl_ROOT=$(python -m dpctl --cmakedir) during the build.
    If dpctl is conda installed, it is optional to pass this parameter.

    """

    process = subprocess.Popen(
        [sys.executable, "-m", "dpctl", "--cmakedir"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output, error = process.communicate()
    if process.returncode == 0:
        return output.decode("utf-8").strip()

    raise RuntimeError(
        "Failed to retrieve dpctl cmake directory: "
        + error.decode("utf-8").strip()
    )


def resolve_compilers(
    oneapi: bool,
    c_compiler: str,
    cxx_compiler: str,
    compiler_root: str,
):
    is_linux = "linux" in sys.platform

    if oneapi or (
        c_compiler is None and cxx_compiler is None and compiler_root is None
    ):
        return "icx", ("icpx" if is_linux else "icx")

    if (
        (c_compiler is None or not os.path.isabs(c_compiler))
        and (cxx_compiler is None or not os.path.isabs(cxx_compiler))
        and (not compiler_root or not os.path.exists(compiler_root))
    ):
        raise RuntimeError(
            "--compiler-root option must be set when using non-default DPC++ "
            "layout unless absolute paths are provided for both compilers"
        )

    # default values
    if c_compiler is None:
        c_compiler = "icx"
    if cxx_compiler is None:
        cxx_compiler = "icpx" if is_linux else "icx"

    compiler_paths = []
    for name, opt_name in (
        (c_compiler, "--c-compiler"),
        (cxx_compiler, "--cxx-compiler"),
    ):
        if os.path.isabs(name):
            path = name
        else:
            path = os.path.join(compiler_root, name)

        if not os.path.exists(path):
            raise RuntimeError(
                f"{opt_name} value {name} not found and {path} not exist"
            )

        compiler_paths.append(path)
    return tuple(compiler_paths)


def resolve_onemath(
    onemath: bool,
    onemath_dir: str,
    target_cuda: str = None,
    target_hip: str = None,
    onemkl_interfaces: bool = False,
    onemkl_interfaces_dir: str = None,
):
    # always enable build with oneMath i/f when oneMath path is passed
    if onemath_dir:
        onemath = True

    # always enable build with oneMath i/f for CUDA or HIP target
    if target_cuda or target_hip:
        onemath = True

    # TODO: onemkl_interfaces and onemkl_interfaces_dir are deprecated in
    # dpnp-0.19.0 and should be removed in dpnp-0.20.0.
    if onemkl_interfaces:
        warnings.warn(
            "Using 'onemkl_interfaces' is deprecated. Please use 'onemath' instead.",
            DeprecationWarning,
            stacklevel=1,
        )
        onemath = True
    if onemkl_interfaces_dir is not None:
        warnings.warn(
            "Using 'onemkl_interfaces_dir' is deprecated. Please use 'onemath_dir' instead.",
            DeprecationWarning,
            stacklevel=1,
        )
        onemath_dir = onemkl_interfaces_dir
    return onemath, onemath_dir


def run(cmd: list[str], env: dict[str, str] = None, cwd: str = None):
    print("+", " ".join(cmd))
    subprocess.check_call(
        cmd, env=env or os.environ.copy(), cwd=cwd or os.getcwd()
    )


def capture_cmd_output(cmd: list[str], cwd: str = None):
    print("+", " ".join(cmd))
    return (
        subprocess.check_output(cmd, cwd=cwd or os.getcwd())
        .decode("utf-8")
        .strip("\n")
    )


def err(msg: str, script: str):
    raise RuntimeError(f"[{script}] error: {msg}")


def log_cmake_args(cmake_args: list[str], script: str):
    print(f"[{script}] Using CMake args:\n{' '.join(cmake_args)}")


def make_cmake_args(
    c_compiler: str = None,
    cxx_compiler: str = None,
    dpctl_cmake_dir: str = None,
    onemath: bool = False,
    onemath_dir: str = None,
    verbose: bool = False,
    other_opts: str = None,
):
    args = [
        f"-DCMAKE_C_COMPILER:PATH={c_compiler}" if c_compiler else "",
        f"-DCMAKE_CXX_COMPILER:PATH={cxx_compiler}" if cxx_compiler else "",
        f"-DDpctl_ROOT={dpctl_cmake_dir}" if dpctl_cmake_dir else "",
    ]

    if onemath:
        args.append("-DDPNP_USE_ONEMATH=ON")
        if onemath_dir:
            args.append(f"-DDPNP_ONEMATH_DIR={onemath_dir}")

    if verbose:
        args.append("-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON")
    if other_opts:
        args.extend(other_opts.split())

    return args


def build_extension(
    setup_dir: str,
    env: dict[str, str],
    cmake_args: list[str],
    cmake_executable: str = None,
    generator: str = None,
    build_type: str = None,
):
    cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
    if cmake_executable:
        cmd.append(f"--cmake-executable={cmake_executable}")
    if generator:
        cmd.append(f"--generator={generator}")
    if build_type:
        cmd.append(f"--build-type={build_type}")
    if cmake_args:
        cmd.append("--")
        cmd += cmake_args
    run(
        cmd,
        env=env,
        cwd=setup_dir,
    )


def install_editable(setup_dir: str, env: dict[str, str]):
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            ".",
            "--no-build-isolation",
        ],
        env=env,
        cwd=setup_dir,
    )


def clean_build_dir(setup_dir: str):
    if (
        not isinstance(setup_dir, str)
        or not setup_dir
        or not os.path.isdir(setup_dir)
    ):
        raise RuntimeError(f"Invalid setup directory provided: '{setup_dir}'")
    target = os.path.join(setup_dir, "_skbuild")
    if os.path.exists(target):
        print(f"Cleaning build directory: {target}")
        try:
            shutil.rmtree(target)
        except Exception as e:
            print(f"Failed to remove build directory: '{target}'")
            raise e
