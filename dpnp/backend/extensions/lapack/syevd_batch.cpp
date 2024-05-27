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

#include <pybind11/pybind11.h>

// dpctl tensor headers
#include "utils/memory_overlap.hpp"
#include "utils/type_utils.hpp"
// #include "copy_and_cast_usm_to_usm.hpp"

#include "syevd.hpp"
#include "types_matrix.hpp"

#include "dpnp_utils.hpp"

namespace dpnp
{
namespace backend
{
namespace ext
{
namespace lapack
{
namespace mkl_lapack = oneapi::mkl::lapack;
namespace py = pybind11;
namespace type_utils = dpctl::tensor::type_utils;


std::pair<sycl::event, sycl::event>
    syevd_batch(sycl::queue exec_q,
                const std::int8_t jobz,
                const std::int8_t upper_lower,
                dpctl::tensor::usm_ndarray eig_vecs,
                dpctl::tensor::usm_ndarray eig_vals,
                dpctl::tensor::usm_ndarray eig_vecs_out,
                const std::vector<sycl::event> &depends)
{
    const int eig_vecs_nd = eig_vecs.get_ndim();
    const int eig_vals_nd = eig_vals.get_ndim();
    const int eig_vecs_out_nd = eig_vecs_out.get_ndim();

    if (eig_vecs_nd != 3 || eig_vecs_out_nd != 3) {
        throw py::value_error("Unexpected ndim=" + std::to_string(eig_vecs_nd) +
                              " of an output array with eigenvectors");
    }
    else if (eig_vals_nd != 2) {
        throw py::value_error("Unexpected ndim=" + std::to_string(eig_vals_nd) +
                              " of an output array with eigenvalues");
    }

    const py::ssize_t *eig_vecs_shape = eig_vecs.get_shape_raw();
    const py::ssize_t *eig_vals_shape = eig_vals.get_shape_raw();
    const py::ssize_t *eig_vecs_out_shape = eig_vecs_out.get_shape_raw();

    const std::int64_t batch_size = eig_vecs_shape[0];
    const std::int64_t n = eig_vecs_shape[1];

    if (eig_vecs_shape[1] != eig_vecs_shape[2] || eig_vecs_out_shape[1] != eig_vecs_out_shape[2]) {
        throw py::value_error("The last two dimensions of 'eig_vecs' and 'eig_vecs_out' must be the same.");
    }

    if (eig_vals_shape[0] != batch_size || eig_vals_shape[1] != n ||
        eig_vecs_out_shape[0] != batch_size || eig_vecs_out_shape[1] != n) {
        throw py::value_error("The shape of 'eig_vals' must be (batch_size, n), and 'eig_vecs_out' must match 'eig_vecs'.");
    }


    // check compatibility of execution queue and allocation queue
    if (!dpctl::utils::queues_are_compatible(exec_q, {eig_vecs, eig_vals, eig_vecs_out})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(eig_vecs, eig_vals) || overlap(eig_vecs, eig_vecs_out) || overlap(eig_vals, eig_vecs_out)) {
        throw py::value_error("Arrays 'eig_vecs', 'eig_vals', and 'eig_vecs_out' are overlapping segments of memory");
    }

    // bool is_eig_vecs_f_contig = eig_vecs.is_f_contiguous();
    // bool is_eig_vals_c_contig = eig_vals.is_c_contiguous();
    // if (!is_eig_vecs_f_contig) {
    //     throw py::value_error(
    //         "An array with input matrix / output eigenvectors "
    //         "must be F-contiguous");
    // }
    // else if (!is_eig_vals_c_contig) {
    //     throw py::value_error(
    //         "An array with output eigenvalues must be C-contiguous");
    // }

    // auto array_types = dpctl_td_ns::usm_ndarray_types();
    // int eig_vecs_type_id =
    //     array_types.typenum_to_lookup_id(eig_vecs.get_typenum());
    // int eig_vals_type_id =
    //     array_types.typenum_to_lookup_id(eig_vals.get_typenum());

    // if (eig_vecs_type_id != eig_vals_type_id) {
    //     throw py::value_error(
    //         "Types of eigenvectors and eigenvalues are mismatched");
    // }

    // syevd_impl_fn_ptr_t syevd_fn = syevd_dispatch_vector[eig_vecs_type_id];
    // if (syevd_fn == nullptr) {
    //     throw py::value_error("No syevd implementation defined for a type of "
    //                           "eigenvectors and eigenvalues");
    // }

    char *eig_vecs_data = eig_vecs.get_data();
    char *eig_vals_data = eig_vals.get_data();
    char *eig_vecs_out_data = eig_vecs_out.get_data();

    std::vector<sycl::event> host_task_events;

    for (size_t i = 0; i < batch_size; ++i) {
        char *eig_vecs_batch = eig_vecs_data + i * n * n * sizeof(float);
        char *eig_vals_batch = eig_vals_data + i * n * sizeof(float);
        char *eig_vecs_out_batch = eig_vecs_out_data + i * n * n * sizeof(float);

        // get slice usm_ndarray
        // eig_vecs_slice, eig_vecs_out_slice, eig_vals_slice

        // copy from eig_vecs_slice to eig_vecs_out_slice
        // like
        // ht_list_ev[2 * i], copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
        //     src=a_usm_arr[i],
        //     dst=eig_vecs[i].get_array(),
        //     sycl_queue=a_sycl_queue,
        // )

        // call syevd for slice usm_ndarray
        // like
        // ht_list_ev[2 * i + 1], _ = getattr(li, lapack_func)(
        //     a_sycl_queue,
        //     jobz,
        //     uplo,
        //     eig_vecs[i].get_array(),
        //     w[i].get_array(),
        //     depends=[copy_ev],
        // )

        // push_back host_task_events

        }


    sycl::event args_ev = dpctl::utils::keep_args_alive(
        exec_q, {eig_vecs, eig_vals, eig_vecs_out}, host_task_events);

    return std::make_pair(args_ev, syevd_ev);
}

// template <typename fnT, typename T>
// struct SyevdContigFactory
// {
//     fnT get()
//     {
//         if constexpr (types::SyevdTypePairSupportFactory<T>::is_defined) {
//             return syevd_impl<T>;
//         }
//         else {
//             return nullptr;
//         }
//     }
// };

// void init_syevd_dispatch_vector(void)
// {
//     dpctl_td_ns::DispatchVectorBuilder<syevd_impl_fn_ptr_t, SyevdContigFactory,
//                                        dpctl_td_ns::num_types>
//         contig;
//     contig.populate_dispatch_vector(syevd_dispatch_vector);
// }
} // namespace lapack
} // namespace ext
} // namespace backend
} // namespace dpnp
