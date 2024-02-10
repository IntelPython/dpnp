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

#include "geqrf.hpp"
#include "gesv.hpp"
#include "gesvd.hpp"
#include "getrf.hpp"
#include "getri.hpp"
#include "heevd.hpp"
#include "linalg_exceptions.hpp"
#include "orgqr.hpp"
#include "potrf.hpp"
#include "syevd.hpp"
#include "ungqr.hpp"

namespace lapack_ext = dpnp::backend::ext::lapack;
namespace py = pybind11;

// populate dispatch vectors
void init_dispatch_vectors(void)
{
    lapack_ext::init_geqrf_batch_dispatch_vector();
    lapack_ext::init_geqrf_dispatch_vector();
    lapack_ext::init_gesv_dispatch_vector();
    lapack_ext::init_getrf_batch_dispatch_vector();
    lapack_ext::init_getrf_dispatch_vector();
    lapack_ext::init_getri_batch_dispatch_vector();
    lapack_ext::init_orgqr_batch_dispatch_vector();
    lapack_ext::init_orgqr_dispatch_vector();
    lapack_ext::init_potrf_batch_dispatch_vector();
    lapack_ext::init_potrf_dispatch_vector();
    lapack_ext::init_syevd_dispatch_vector();
    lapack_ext::init_ungqr_batch_dispatch_vector();
    lapack_ext::init_ungqr_dispatch_vector();
}

// populate dispatch tables
void init_dispatch_tables(void)
{
    lapack_ext::init_gesvd_dispatch_table();
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

    m.def("_geqrf_batch", &lapack_ext::geqrf_batch,
          "Call `geqrf_batch` from OneMKL LAPACK library to return "
          "the QR factorization of a batch general matrix ",
          py::arg("sycl_queue"), py::arg("a_array"), py::arg("tau_array"),
          py::arg("m"), py::arg("n"), py::arg("stride_a"),
          py::arg("stride_tau"), py::arg("batch_size"),
          py::arg("depends") = py::list());

    m.def("_geqrf", &lapack_ext::geqrf,
          "Call `geqrf` from OneMKL LAPACK library to return "
          "the QR factorization of a general m x n matrix ",
          py::arg("sycl_queue"), py::arg("a_array"), py::arg("tau_array"),
          py::arg("depends") = py::list());

    m.def("_gesv", &lapack_ext::gesv,
          "Call `gesv` from OneMKL LAPACK library to return "
          "the solution of a system of linear equations with "
          "a square coefficient matrix A and multiple dependent variables",
          py::arg("sycl_queue"), py::arg("coeff_matrix"),
          py::arg("dependent_vals"), py::arg("depends") = py::list());

    m.def("_gesvd", &lapack_ext::gesvd,
          "Call `gesvd` from OneMKL LAPACK library to return "
          "the singular value decomposition of a general rectangular matrix",
          py::arg("sycl_queue"), py::arg("jobu_val"), py::arg("jobvt_val"),
          py::arg("a_array"), py::arg("res_s"), py::arg("res_u"),
          py::arg("res_vt"), py::arg("depends") = py::list());

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

    m.def("_getri_batch", &lapack_ext::getri_batch,
          "Call `getri_batch` from OneMKL LAPACK library to return "
          "the inverses of a batch of LU-factored matrices",
          py::arg("sycl_queue"), py::arg("a_array"), py::arg("ipiv_array"),
          py::arg("dev_info"), py::arg("n"), py::arg("stride_a"),
          py::arg("stride_ipiv"), py::arg("batch_size"),
          py::arg("depends") = py::list());

    m.def("_heevd", &lapack_ext::heevd,
          "Call `heevd` from OneMKL LAPACK library to return "
          "the eigenvalues and eigenvectors of a complex Hermitian matrix",
          py::arg("sycl_queue"), py::arg("jobz"), py::arg("upper_lower"),
          py::arg("eig_vecs"), py::arg("eig_vals"),
          py::arg("depends") = py::list());

    m.def("_orgqr_batch", &lapack_ext::orgqr_batch,
          "Call `_orgqr_batch` from OneMKL LAPACK library to return "
          "the real orthogonal matrix Qi of the QR factorization "
          "for a batch of general matrices",
          py::arg("sycl_queue"), py::arg("a_array"), py::arg("tau_array"),
          py::arg("m"), py::arg("n"), py::arg("k"), py::arg("stride_a"),
          py::arg("stride_tau"), py::arg("batch_size"),
          py::arg("depends") = py::list());

    m.def("_orgqr", &lapack_ext::orgqr,
          "Call `orgqr` from OneMKL LAPACK library to return "
          "the real orthogonal matrix Q of the QR factorization",
          py::arg("sycl_queue"), py::arg("m"), py::arg("n"), py::arg("k"),
          py::arg("a_array"), py::arg("tau_array"),
          py::arg("depends") = py::list());

    m.def("_potrf", &lapack_ext::potrf,
          "Call `potrf` from OneMKL LAPACK library to return "
          "the Cholesky factorization of a symmetric positive-definite matrix",
          py::arg("sycl_queue"), py::arg("a_array"), py::arg("upper_lower"),
          py::arg("depends") = py::list());

    m.def("_potrf_batch", &lapack_ext::potrf_batch,
          "Call `potrf_batch` from OneMKL LAPACK library to return "
          "the Cholesky factorization of a batch of symmetric "
          "positive-definite matrix",
          py::arg("sycl_queue"), py::arg("a_array"), py::arg("upper_lower"),
          py::arg("n"), py::arg("stride_a"), py::arg("batch_size"),
          py::arg("depends") = py::list());

    m.def("_syevd", &lapack_ext::syevd,
          "Call `syevd` from OneMKL LAPACK library to return "
          "the eigenvalues and eigenvectors of a real symmetric matrix",
          py::arg("sycl_queue"), py::arg("jobz"), py::arg("upper_lower"),
          py::arg("eig_vecs"), py::arg("eig_vals"),
          py::arg("depends") = py::list());

    m.def("_ungqr_batch", &lapack_ext::ungqr_batch,
          "Call `_ungqr_batch` from OneMKL LAPACK library to return "
          "the complex unitary matrices matrix Qi of the QR factorization "
          "for a batch of general matrices",
          py::arg("sycl_queue"), py::arg("a_array"), py::arg("tau_array"),
          py::arg("m"), py::arg("n"), py::arg("k"), py::arg("stride_a"),
          py::arg("stride_tau"), py::arg("batch_size"),
          py::arg("depends") = py::list());

    m.def("_ungqr", &lapack_ext::ungqr,
          "Call `ungqr` from OneMKL LAPACK library to return "
          "the complex unitary matrix Q of the QR factorization",
          py::arg("sycl_queue"), py::arg("m"), py::arg("n"), py::arg("k"),
          py::arg("a_array"), py::arg("tau_array"),
          py::arg("depends") = py::list());
}
