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

import argparse
import importlib
import os
import os.path
import sys


def _dpnp_dir() -> str:
    dpnp_dir = importlib.util.find_spec("dpnp").submodule_search_locations[0]
    abs_dpnp_dir = os.path.abspath(dpnp_dir)
    return abs_dpnp_dir


def get_tensor_include_dir() -> str:
    """Prints path to dpnp libtensor include directory"""
    dpnp_dir = _dpnp_dir()
    libtensor_dir = os.path.join(dpnp_dir, "tensor", "libtensor", "include")
    return libtensor_dir


def print_tensor_include_flags() -> None:
    """Prints include flags for dpnp tensor library"""
    libtensor_dir = get_tensor_include_dir()
    print("-I " + libtensor_dir)


def main() -> None:
    """Main entry-point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tensor-includes",
        action="store_true",
        help="Include flags for dpnp libtensor headers.",
    )
    parser.add_argument(
        "--tensor-include-dir",
        action="store_true",
        help="Path to dpnp libtensor include directory.",
    )
    args = parser.parse_args()
    if not sys.argv[1:]:
        parser.print_help()
    if args.tensor_includes:
        print_tensor_include_flags()
    if args.tensor_include_dir:
        print(get_tensor_include_dir())


if __name__ == "__main__":
    main()
