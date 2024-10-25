import importlib.machinery as imm
import os

from skbuild import setup

import versioneer

"""
Get the project version
"""
thefile_path = os.path.abspath(os.path.dirname(__file__))
version_mod = imm.SourceFileLoader(
    "version", os.path.join(thefile_path, "dpnp", "_version.py")
).load_module()
__version__ = version_mod.get_versions()["version"]

"""
Set project auxiliary data like readme and licence files
"""
with open("README.md") as f:
    __readme_file__ = f.read()


def _get_cmdclass():
    return versioneer.get_cmdclass()


CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: Apache Software License
Programming Language :: C++
Programming Language :: Cython
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Programming Language :: Python :: 3.11
Programming Language :: Python :: 3.12
Programming Language :: Python :: Implementation :: CPython
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
"""

EXCLUDED_DIRS = ["tests_perf"]


def find_tests_files():
    files_by_destination = {}
    for root, dirs, files in os.walk("tests"):
        # Exclude specified directories
        dirs[:] = [dir for dir in dirs if dir not in EXCLUDED_DIRS]

        for file in files:
            file_path = os.path.join(root, file)
            # Get the path relative to `tests`` to keep folder structure
            relative_path = os.path.relpath(file_path, "tests")
            destination = os.path.join(
                "dpnp/tests", os.path.dirname(relative_path)
            )

            # Add the file to the correct destination folder in the dictionary
            if destination not in files_by_destination:
                files_by_destination[destination] = []
            files_by_destination[destination].append(file_path)

    # Convert the dictionary to the format expected by data_files:
    # [(destination_folder, [list_of_files])]
    tests_files = [
        (dest, files) for dest, files in files_by_destination.items()
    ]

    return tests_files


setup(
    name="dpnp",
    version=__version__,
    cmdclass=_get_cmdclass(),
    description="Data Parallel Extension for NumPy",
    long_description=__readme_file__,
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    classifiers=[_f for _f in CLASSIFIERS.split("\n") if _f],
    keywords="sycl numpy python3 intel mkl oneapi gpu dpcpp",
    platforms=["Linux", "Windows"],
    author="Intel Corporation",
    url="https://github.com/IntelPython/dpnp",
    packages=[
        "dpnp",
        "dpnp.dpnp_algo",
        "dpnp.dpnp_utils",
        "dpnp.fft",
        "dpnp.linalg",
        "dpnp.random",
    ],
    data_files=find_tests_files(),
    package_data={
        "dpnp": [
            "backend/include/*.hpp",
            "libdpnp_backend_c.so",
            "dpnp_backend_c.lib",
            "dpnp_backend_c.dll",
        ]
    },
    include_package_data=False,
)
