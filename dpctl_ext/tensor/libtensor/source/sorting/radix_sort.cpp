//*****************************************************************************
// Copyright (c) 2026, Intel Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// - Neither the name of the copyright holder nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
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
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines functions of dpctl.tensor._tensor_sorting_impl
/// extension.
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <cstdint>
#include <exception>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils/type_dispatch.hpp"

#include "kernels/dpctl_tensor_types.hpp"
#include "kernels/sorting/radix_sort.hpp"
#include "kernels/sorting/sort_impl_fn_ptr_t.hpp"

#include "py_sort_common.hpp"
#include "radix_sort.hpp"
#include "radix_sort_support.hpp"

namespace dpctl::tensor::py_internal
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;
namespace impl_ns = dpctl::tensor::kernels::radix_sort_details;

using dpctl::tensor::kernels::sort_contig_fn_ptr_t;
static sort_contig_fn_ptr_t
    ascending_radix_sort_contig_dispatch_vector[td_ns::num_types];
static sort_contig_fn_ptr_t
    descending_radix_sort_contig_dispatch_vector[td_ns::num_types];

namespace
{

template <bool is_ascending, typename T>
sycl::event sort_axis1_contig_caller(sycl::queue &q,
                                     std::size_t iter_nelems,
                                     std::size_t sort_nelems,
                                     const char *arg_cp,
                                     char *res_cp,
                                     ssize_t iter_arg_offset,
                                     ssize_t iter_res_offset,
                                     ssize_t sort_arg_offset,
                                     ssize_t sort_res_offset,
                                     const std::vector<sycl::event> &depends)
{
    using dpctl::tensor::kernels::radix_sort_axis1_contig_impl;

    return radix_sort_axis1_contig_impl<T>(
        q, is_ascending, iter_nelems, sort_nelems, arg_cp, res_cp,
        iter_arg_offset, iter_res_offset, sort_arg_offset, sort_res_offset,
        depends);
}

} // end of anonymous namespace

template <typename fnT, typename argTy>
struct AscendingRadixSortContigFactory
{
    fnT get()
    {
        if constexpr (RadixSortSupportVector<argTy>::is_defined) {
            return sort_axis1_contig_caller</*ascending*/ true, argTy>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename argTy>
struct DescendingRadixSortContigFactory
{
    fnT get()
    {
        if constexpr (RadixSortSupportVector<argTy>::is_defined) {
            return sort_axis1_contig_caller</*ascending*/ false, argTy>;
        }
        else {
            return nullptr;
        }
    }
};

void init_radix_sort_dispatch_vectors(void)
{
    using dpctl::tensor::kernels::sort_contig_fn_ptr_t;

    td_ns::DispatchVectorBuilder<
        sort_contig_fn_ptr_t, AscendingRadixSortContigFactory, td_ns::num_types>
        dtv1;
    dtv1.populate_dispatch_vector(ascending_radix_sort_contig_dispatch_vector);

    td_ns::DispatchVectorBuilder<sort_contig_fn_ptr_t,
                                 DescendingRadixSortContigFactory,
                                 td_ns::num_types>
        dtv2;
    dtv2.populate_dispatch_vector(descending_radix_sort_contig_dispatch_vector);
}

bool py_radix_sort_defined(int typenum)
{
    const auto &array_types = td_ns::usm_ndarray_types();

    try {
        int type_id = array_types.typenum_to_lookup_id(typenum);
        return (nullptr !=
                ascending_radix_sort_contig_dispatch_vector[type_id]);
    } catch (const std::exception &e) {
        return false;
    }
}

void init_radix_sort_functions(py::module_ m)
{
    dpctl::tensor::py_internal::init_radix_sort_dispatch_vectors();

    auto py_radix_sort_ascending = [](const dpctl::tensor::usm_ndarray &src,
                                      const int trailing_dims_to_sort,
                                      const dpctl::tensor::usm_ndarray &dst,
                                      sycl::queue &exec_q,
                                      const std::vector<sycl::event> &depends)
        -> std::pair<sycl::event, sycl::event> {
        return dpctl::tensor::py_internal::py_sort(
            src, trailing_dims_to_sort, dst, exec_q, depends,
            dpctl::tensor::py_internal::
                ascending_radix_sort_contig_dispatch_vector);
    };
    m.def("_radix_sort_ascending", py_radix_sort_ascending, py::arg("src"),
          py::arg("trailing_dims_to_sort"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    auto py_radix_sort_descending = [](const dpctl::tensor::usm_ndarray &src,
                                       const int trailing_dims_to_sort,
                                       const dpctl::tensor::usm_ndarray &dst,
                                       sycl::queue &exec_q,
                                       const std::vector<sycl::event> &depends)
        -> std::pair<sycl::event, sycl::event> {
        return dpctl::tensor::py_internal::py_sort(
            src, trailing_dims_to_sort, dst, exec_q, depends,
            dpctl::tensor::py_internal::
                descending_radix_sort_contig_dispatch_vector);
    };
    m.def("_radix_sort_descending", py_radix_sort_descending, py::arg("src"),
          py::arg("trailing_dims_to_sort"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    m.def("_radix_sort_dtype_supported", py_radix_sort_defined);

    return;
}

} // namespace dpctl::tensor::py_internal
