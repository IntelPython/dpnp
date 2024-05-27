import importlib.machinery as imm
import os
import shutil

import skbuild

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

def _patched_copy_file(
    src_file, dest_file, hide_listing=True, preserve_mode=True
):
    """Copy ``src_file`` to ``dest_file`` ensuring parent directory exists.

    By default, message like `creating directory /path/to/package` and
    `copying directory /src/path/to/package -> path/to/package` are displayed
    on standard output. Setting ``hide_listing`` to False avoids message from
    being displayed.

    NB: Patched here to not follows symbolic links
    """
    # Create directory if needed
    dest_dir = os.path.dirname(dest_file)
    if dest_dir != "" and not os.path.exists(dest_dir):
        if not hide_listing:
            print("creating directory {}".format(dest_dir))
        skbuild.utils.mkdir_p(dest_dir)

    # Copy file
    if not hide_listing:
        print("copying {} -> {}".format(src_file, dest_file))
    shutil.copyfile(src_file, dest_file, follow_symlinks=False)
    shutil.copymode(src_file, dest_file, follow_symlinks=False)


skbuild.setuptools_wrap._copy_file = _patched_copy_file

skbuild.setup(
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
    python_requires=">=3.9",
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
            "../tests/*.*",
            "../tests/third_party/cupy/*.py",
            "dpnp/backend/include/*.hpp",
            "libdpnp_backend_c.so",
            "dpnp_backend_c.lib",
            "dpnp_backend_c.dll",
        ]
    },
    include_package_data=False,
)
