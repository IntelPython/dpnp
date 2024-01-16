//*****************************************************************************
// Copyright (c) 2023-2024, Intel Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
// THE POSSIBILITY OF SUCH DAMAGE.
//*****************************************************************************
//
// This file defines functions of dpnp.backend._lapack_impl extensions
//
//*****************************************************************************

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gesv.hpp"
#include "getrf.hpp"
#include "heevd.hpp"
#include "linalg_exceptions.hpp"
#include "potrf.hpp"
#include "syevd.hpp"

namespace lapack_ext = dpnp::backend::ext::lapack;
namespace py = pybind11;

// populate dispatch vectors
void init_dispatch_vectors(void)
{
    lapack_ext::init_gesv_dispatch_vector();
    lapack_ext::init_getrf_batch_dispatch_vector();
    lapack_ext::init_getrf_dispatch_vector();
    lapack_ext::init_potrf_dispatch_vector();
    lapack_ext::init_potrf_batch_dispatch_vector();
    lapack_ext::init_syevd_dispatch_vector();
}

// populate dispatch tables
void init_dispatch_tables(void)
{
    lapack_ext::init_heevd_dispatch_table();
}

PYBIND11_MODULE(_lapack_impl, m)
{
    // Register a custom LinAlgError exception in the dpnp.linalg submodule
    py::module_ linalg_module = py::module_::import("dpnp.linalg");
    py::register_exception<lapack_ext::LinAlgError>(
        linalg_module, "LinAlgError", PyExc_ValueError);

    init_dispatch_vectors();
    init_dispatch_tables();

    m.def("_gesv", &lapack_ext::gesv,
          "Call `gesv` from OneMKL LAPACK library to return "
          "the solution of a system of linear equations with "
          "a square coefficient matrix A and multiple dependent variables",
          py::arg("sycl_queue"), py::arg("coeff_matrix"),
          py::arg("dependent_vals"), py::arg("depends") = py::list());

    m.def("_getrf", &lapack_ext::getrf,
          "Call `getrf` from OneMKL LAPACK library to return "
          "the LU factorization of a general n x n matrix",
          py::arg("sycl_queue"), py::arg("a_array"), py::arg("ipiv_array"),
          py::arg("dev_info"), py::arg("depends") = py::list());

    m.def("_getrf_batch", &lapack_ext::getrf_batch,
          "Call `getrf_batch` from OneMKL LAPACK library to return "
          "the LU factorization of a batch of general n x n matrices",
          py::arg("sycl_queue"), py::arg("a_array"), py::arg("ipiv_array"),
          py::arg("dev_info_array"), py::arg("n"), py::arg("stride_a"),
          py::arg("stride_ipiv"), py::arg("batch_size"),
          py::arg("depends") = py::list());

    m.def("_heevd", &lapack_ext::heevd,
          "Call `heevd` from OneMKL LAPACK library to return "
          "the eigenvalues and eigenvectors of a complex Hermitian matrix",
          py::arg("sycl_queue"), py::arg("jobz"), py::arg("upper_lower"),
          py::arg("eig_vecs"), py::arg("eig_vals"),
          py::arg("depends") = py::list());

    m.def("_potrf", &lapack_ext::potrf,
          "Call `potrf` from OneMKL LAPACK library to return "
          "the Cholesky factorization of a symmetric positive-definite matrix",
          py::arg("sycl_queue"), py::arg("n"), py::arg("a_array"),
          py::arg("depends") = py::list());

    m.def("_potrf_batch", &lapack_ext::potrf_batch,
          "Call `potrf_batch` from OneMKL LAPACK library to return "
          "the Cholesky factorization of a batch of symmetric "
          "positive-definite matrix",
          py::arg("sycl_queue"), py::arg("a_array"), py::arg("n"),
          py::arg("stride_a"), py::arg("batch_size"),
          py::arg("depends") = py::list());

    m.def("_syevd", &lapack_ext::syevd,
          "Call `syevd` from OneMKL LAPACK library to return "
          "the eigenvalues and eigenvectors of a real symmetric matrix",
          py::arg("sycl_queue"), py::arg("jobz"), py::arg("upper_lower"),
          py::arg("eig_vecs"), py::arg("eig_vals"),
          py::arg("depends") = py::list());
}
