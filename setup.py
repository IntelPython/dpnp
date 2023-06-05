from skbuild import setup
import os
import importlib.machinery as imm


"""
Get the project version
"""
thefile_path = os.path.abspath(os.path.dirname(__file__))
version_mod = imm.SourceFileLoader('version', os.path.join(thefile_path, 'dpnp', 'version.py')).load_module()
__version__ = version_mod.__version__

setup(
    name="dpnp",
    version=__version__,
    description="",
    long_description="",
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    author="Intel Corporation",
    url="https://github.com/IntelPython/dpnp",
    packages=['dpnp',
              'dpnp.dpnp_algo',
              'dpnp.dpnp_utils',
              'dpnp.fft',
              'dpnp.linalg',
              'dpnp.random'
    ],
    package_data={'dpnp': ['libdpnp_backend_c.so', 'dpnp_backend_c.lib', 'dpnp_backend_c.dll']},
    include_package_data=True,
)
