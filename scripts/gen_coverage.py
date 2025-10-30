# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2023, Intel Corporation
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
import subprocess
import sys


def run(
    use_oneapi=True,
    c_compiler=None,
    cxx_compiler=None,
    compiler_root=None,
    bin_llvm=None,
    pytest_opts="",
    verbose=False,
):
    IS_LIN = False

    if "linux" in sys.platform:
        IS_LIN = True
    elif sys.platform in ["win32", "cygwin"]:
        pass
    else:
        raise AssertionError(sys.platform + " not supported")

    if not IS_LIN:
        raise RuntimeError(
            "This scripts only supports coverage collection on Linux"
        )

    setup_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    cmake_args = [
        sys.executable,
        "setup.py",
        "develop",
        "--generator=Ninja",
        "--",
        "-DCMAKE_C_COMPILER:PATH=" + c_compiler,
        "-DCMAKE_CXX_COMPILER:PATH=" + cxx_compiler,
        "-DDPNP_GENERATE_COVERAGE=ON",
    ]

    env = {}
    if bin_llvm:
        env = {
            "PATH": ":".join((os.environ.get("PATH", ""), bin_llvm)),
            "LLVM_TOOLS_HOME": bin_llvm,
        }

    # extend with global environment variables
    env.update({k: v for k, v in os.environ.items() if k != "PATH"})

    if verbose:
        cmake_args += [
            "-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON",
        ]

    subprocess.check_call(cmake_args, shell=False, cwd=setup_dir, env=env)

    env["LLVM_PROFILE_FILE"] = "dpnp_pytest.profraw"
    subprocess.check_call(
        [
            "pytest",
            "-q",
            "-ra",
            "--disable-warnings",
            "--cov-config",
            "pyproject.toml",
            "--cov",
            "dpnp",
            "--cov-report=lcov:coverage-python.lcov",
            "--pyargs",
            "dpnp",
            *pytest_opts.split(),
        ],
        cwd=setup_dir,
        shell=False,
        env=env,
    )

    def find_objects():
        objects = []
        dpnp_path = os.getcwd()
        search_path = os.path.join(dpnp_path, "dpnp")
        for root, _, files in os.walk(search_path):
            for file in files:
                if (
                    file.endswith("_c.so")
                    or root.find("extensions") != -1
                    and file.find("_impl.cpython") != -1
                ):
                    objects.extend(["-object", os.path.join(root, file)])
        return objects

    objects = find_objects()
    instr_profile_fn = "dpnp_pytest.profdata"
    # generate instrumentation profile data
    subprocess.check_call(
        [
            os.path.join(bin_llvm, "llvm-profdata"),
            "merge",
            "-sparse",
            env["LLVM_PROFILE_FILE"],
            "-o",
            instr_profile_fn,
        ]
    )

    # export lcov
    with open("coverage-cpp.lcov", "w") as fh:
        subprocess.check_call(
            [
                os.path.join(bin_llvm, "llvm-cov"),
                "export",
                "-format=lcov",
                "-ignore-filename-regex=/tmp/icpx*",
                r"-ignore-filename-regex=.*/backend/kernels/elementwise_functions/.*\.hpp$",
                "-instr-profile=" + instr_profile_fn,
            ]
            + objects
            + ["-sources", "dpnp"],
            stdout=fh,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Driver to build dpnp and generate coverage"
    )
    driver = parser.add_argument_group(title="Coverage driver arguments")
    driver.add_argument("--c-compiler", help="Name of C compiler", default=None)
    driver.add_argument(
        "--cxx-compiler", help="Name of C++ compiler", default=None
    )
    driver.add_argument(
        "--not-oneapi",
        help="Is one-API installation",
        dest="oneapi",
        action="store_false",
    )
    driver.add_argument(
        "--compiler-root", type=str, help="Path to compiler home directory"
    )
    driver.add_argument(
        "--bin-llvm", help="Path to folder where llvm-cov can be found"
    )
    driver.add_argument(
        "--pytest-opts",
        help="Channels through additional pytest options",
        dest="pytest_opts",
        default="",
        type=str,
    )
    driver.add_argument(
        "--verbose",
        help="Build using vebose makefile mode",
        dest="verbose",
        action="store_true",
    )
    args = parser.parse_args()

    if args.oneapi:
        args.c_compiler = "icx"
        args.cxx_compiler = "icpx"
        args.compiler_root = None
        icx_path = subprocess.check_output(["which", "icx"])
        bin_dir = os.path.dirname(icx_path)
        compiler_dir = os.path.join(bin_dir.decode("utf-8"), "compiler")
        if os.path.exists(compiler_dir):
            args.bin_llvm = os.path.join(bin_dir.decode("utf-8"), "compiler")
        else:
            bin_dir = os.path.dirname(bin_dir)
            args.bin_llvm = os.path.join(bin_dir.decode("utf-8"), "bin-llvm")
        assert os.path.exists(args.bin_llvm)
    else:
        args_to_validate = [
            "c_compiler",
            "cxx_compiler",
            "compiler_root",
            "bin_llvm",
        ]
        for p in args_to_validate:
            arg = getattr(args, p, None)
            if not isinstance(arg, str):
                opt_name = p.replace("_", "-")
                raise RuntimeError(
                    f"Option {opt_name} must be provided is "
                    "using non-default DPC++ layout"
                )
            if not os.path.exists(arg):
                raise RuntimeError(f"Path {arg} must exist")

    run(
        use_oneapi=args.oneapi,
        c_compiler=args.c_compiler,
        cxx_compiler=args.cxx_compiler,
        compiler_root=args.compiler_root,
        bin_llvm=args.bin_llvm,
        pytest_opts=args.pytest_opts,
        verbose=args.verbose,
    )
