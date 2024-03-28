import os
import subprocess
import sys


def run(
    c_compiler=None,
    cxx_compiler=None,
    bin_llvm=None,
    pytest_opts="",
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
        "-G=Ninja",
        "--",
        "-DCMAKE_C_COMPILER:PATH=" + c_compiler,
        "-DCMAKE_CXX_COMPILER:PATH=" + cxx_compiler,
        "-DCMAKE_VERBOSE_MAKEFILE=ON",
        "-DDPNP_GENERATE_COVERAGE=ON",
    ]

    env = {}
    if bin_llvm:
        env = {
            "PATH": ":".join((os.environ.get("PATH", ""), bin_llvm)),
            "LLVM_TOOLS_HOME": bin_llvm,
        }

    # extend with global enviroment variables
    env.update({k: v for k, v in os.environ.items() if k != "PATH"})

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
            "--cov-report",
            "term-missing",
            "--pyargs",
            "tests",
            "-vv",
            *pytest_opts.split(),
        ],
        cwd=setup_dir,
        shell=False,
        env=env,
    )

    def find_objects():
        import os

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
    with open("dpnp_pytest.lcov", "w") as fh:
        subprocess.check_call(
            [
                os.path.join(bin_llvm, "llvm-cov"),
                "export",
                "-format=lcov",
                "-ignore-filename-regex=/tmp/icpx*",
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
    driver.add_argument(
        "--pytest-opts",
        help="Channels through additional pytest options",
        dest="pytest_opts",
        default="",
        type=str,
    )

    args = parser.parse_args()

    c_compiler = "icx"
    cxx_compiler = "icpx"
    icx_path = subprocess.check_output(["which", "icx"])
    bin_dir = os.path.dirname(os.path.dirname(icx_path))
    bin_llvm = os.path.join(bin_dir.decode("utf-8"), "bin-llvm")

    run(
        c_compiler=c_compiler,
        cxx_compiler=cxx_compiler,
        bin_llvm=bin_llvm,
        pytest_opts=args.pytest_opts,
    )
