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

import argparse
import os
import subprocess
import sys

from _build_helper import (
    build_extension,
    capture_cmd_output,
    clean_build_dir,
    err,
    get_dpctl_cmake_dir,
    install_editable,
    log_cmake_args,
    make_cmake_args,
    resolve_compilers,
    run,
)


def parse_args():
    p = argparse.ArgumentParser(description="Build dpnp and generate coverage")

    # compiler and oneAPI relating options
    p.add_argument(
        "--c-compiler", default=None, help="Path or name of C compiler"
    )
    p.add_argument(
        "--cxx-compiler", default=None, help="Path or name of C++ compiler"
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
    p.add_argument(
        "--bin-llvm",
        type=str,
        default=None,
        help="Path to folder where llvm-cov/llvm-profdata can be found",
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
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Enable verbose makefile output",
    )

    # test relating options
    p.add_argument(
        "--skip-pytest",
        dest="run_pytest",
        action="store_false",
        help="Skip running pytest and coverage generation",
    )
    p.add_argument(
        "--pytest-opts",
        help="Channels through additional pytest options",
        dest="pytest_opts",
        default="",
        type=str,
    )
    p.add_argument(
        "--skip-tensor-tests",
        dest="run_tensor_tests",
        action="store_false",
        help="Skip running dpnp.tensor tests (run in a separate pytest session)",
    )
    p.add_argument(
        "--tensor-cpp-coverage",
        dest="tensor_cpp_coverage",
        action="store_true",
        help="Build dpnp.tensor pybind11 extensions with coverage "
        "instrumentation and include them into C++ coverage report",
    )

    # build relating options
    p.add_argument(
        "--clean",
        action="store_true",
        help="Remove build dir before rebuild (default: False)",
    )

    return p.parse_args()


def find_bin_llvm(compiler):
    if os.path.isabs(compiler):
        bin_dir = os.path.dirname(compiler)
    else:
        compiler_path = capture_cmd_output(["which", compiler])
        if not compiler_path:
            raise RuntimeError(f"Compiler {compiler} not found in PATH")
        bin_dir = os.path.dirname(compiler_path)

    compiler_dir = os.path.join(bin_dir, "compiler")
    if os.path.exists(compiler_dir):
        bin_llvm = compiler_dir
    else:
        bin_dir = os.path.dirname(bin_dir)
        bin_llvm = os.path.join(bin_dir, "bin-llvm")

    if not os.path.exists(bin_llvm):
        raise RuntimeError(
            f"Path to folder with llvm-cov/llvm-profdata={bin_llvm} "
            "seems to not exist"
        )
    return bin_llvm


def main():
    is_linux = "linux" in sys.platform
    if not is_linux:
        err(f"{sys.platform} not supported", "gen_coverage")

    args = parse_args()
    setup_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    c_compiler, cxx_compiler = resolve_compilers(
        args.oneapi,
        args.c_compiler,
        args.cxx_compiler,
        args.compiler_root,
    )

    dpctl_cmake_dir = get_dpctl_cmake_dir()
    print(f"[gen_coverage] Found DPCTL CMake dir: {dpctl_cmake_dir}")

    if args.clean:
        clean_build_dir(setup_dir)

    cmake_args = make_cmake_args(
        c_compiler=c_compiler,
        cxx_compiler=cxx_compiler,
        dpctl_cmake_dir=dpctl_cmake_dir,
        verbose=args.verbose,
    )
    cmake_args.append("-DDPNP_GENERATE_COVERAGE=ON")
    if args.tensor_cpp_coverage:
        cmake_args.append(
            "-DDPNP_TENSOR_GENERATE_COVERAGE_FOR_PYBIND11_EXTENSIONS=ON"
        )

    env = os.environ.copy()

    if args.bin_llvm:
        bin_llvm = args.bin_llvm
    else:
        bin_llvm = find_bin_llvm(c_compiler)
    print(
        f"[gen_coverage] Path to folder with llvm-cov/llvm-profdata: {bin_llvm}"
    )

    if bin_llvm:
        env["PATH"] = ":".join((env.get("PATH", ""), bin_llvm))
        env["LLVM_TOOLS_HOME"] = bin_llvm

    log_cmake_args(cmake_args, "gen_coverage")

    build_extension(
        setup_dir,
        env,
        cmake_args,
        cmake_executable=args.cmake_executable,
        generator=args.generator,
        build_type="Coverage",
    )
    install_editable(setup_dir, env)

    if args.run_pytest:
        pytest_cmd = [
            "pytest",
            "-q",
            "-ra",
            "--disable-warnings",
            "--cov-config",
            "pyproject.toml",
            "--cov",
            "dpnp",
        ]
        cov_report = [
            "--cov-report=term-missing",
            "--cov-report=lcov:coverage-python.lcov",
        ]

        # dpnp tests w/o tensor ones, those are run in a separate session
        env["SKIP_TENSOR_TESTS"] = "1"
        env["LLVM_PROFILE_FILE"] = "dpnp_pytest.profraw"
        profraw_files = [env["LLVM_PROFILE_FILE"]]
        run(
            pytest_cmd
            # report once, on the combined data of the last session
            + (["--cov-report="] if args.run_tensor_tests else cov_report)
            + ["--pyargs", "dpnp", *args.pytest_opts.split()],
            env=env,
            cwd=setup_dir,
        )

        if args.run_tensor_tests:
            # separate session to limit peak memory, appends to coverage data
            env["SKIP_TENSOR_TESTS"] = "0"
            env["LLVM_PROFILE_FILE"] = "dpnp_tensor_pytest.profraw"
            profraw_files.append(env["LLVM_PROFILE_FILE"])
            run(
                pytest_cmd
                + ["--cov-append"]
                + cov_report
                + ["--pyargs", "dpnp.tests.tensor", *args.pytest_opts.split()],
                env=env,
                cwd=setup_dir,
            )

        def find_objects():
            objects = []
            dpnp_path = os.getcwd()
            search_path = os.path.join(dpnp_path, "dpnp")
            for root, _, files in os.walk(search_path):
                for file in files:
                    is_backend_obj = file.endswith("_c.so") or (
                        root.find("extensions") != -1
                        and file.find("_impl.cpython") != -1
                    )
                    # tensor pybind11 extensions, e.g. _tensor_impl
                    is_tensor_obj = (
                        args.tensor_cpp_coverage
                        and os.path.basename(root) == "tensor"
                        and file.startswith("_tensor_")
                        and file.find("_impl.cpython") != -1
                    )
                    if is_backend_obj or is_tensor_obj:
                        objects.extend(["-object", os.path.join(root, file)])
            return objects

        objects = find_objects()
        instr_profile_fn = "dpnp_pytest.profdata"

        # generate instrumentation profile data
        run(
            [
                os.path.join(bin_llvm, "llvm-profdata"),
                "merge",
                "-sparse",
                *profraw_files,
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
                    r"-ignore-filename-regex=.*/backend/kernels/indexing/.*\.hpp$",
                    r"-ignore-filename-regex=.*/backend/kernels/statistics/.*\.hpp$",
                    r"-ignore-filename-regex=.*/backend/kernels/window/.*\.hpp$",
                    "-instr-profile=" + instr_profile_fn,
                ]
                + objects
                + ["-sources", "dpnp"],
                cwd=setup_dir,
                env=env,
                stdout=fh,
            )

        print("[gen_coverage] Coverage export is completed")
    else:
        print(
            "[gen_coverage] Skipping pytest and coverage collection "
            "(--skip-pytest)"
        )

    print("[gen_coverage] Done")


if __name__ == "__main__":
    main()
