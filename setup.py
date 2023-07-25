import importlib.machinery as imm
import os

from skbuild import setup

"""
Get the project version
"""
thefile_path = os.path.abspath(os.path.dirname(__file__))
version_mod = imm.SourceFileLoader(
    "version", os.path.join(thefile_path, "dpnp", "version.py")
).load_module()
__version__ = version_mod.__version__

"""
Set project auxilary data like readme and licence files
"""
with open("README.md") as f:
    __readme_file__ = f.read()

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: Apache Software License
Programming Language :: C++
Programming Language :: Cython
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Programming Language :: Python :: 3.11
Programming Language :: Python :: Implementation :: CPython
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
"""

setup(
    name="dpnp",
    version=__version__,
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
    package_data={
        "dpnp": [
            "libdpnp_backend_c.so",
            "dpnp_backend_c.lib",
            "dpnp_backend_c.dll",
        ]
    },
    include_package_data=True,
)
