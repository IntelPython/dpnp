//*****************************************************************************
// Copyright (c) 2025, Intel Corporation
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

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// dpctl tensor headers
#include "dpctl4pybind11.hpp"
#include "utils/type_dispatch.hpp"

#include "kernels/elementwise_functions/interpolate.hpp"

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

namespace dpnp::extensions::ufunc
{

namespace impl
{

typedef sycl::event (*interpolate_fn_ptr_t)(sycl::queue &,
                                            const void *, // x
                                            const void *, // idx
                                            const void *, // xp
                                            const void *, // fp
                                            void *,       // out
                                            std::size_t,  // n
                                            std::size_t,  // xp_size
                                            const std::vector<sycl::event> &);

interpolate_fn_ptr_t interpolate_dispatch_table[td_ns::num_types]
                                               [td_ns::num_types];

std::pair<sycl::event, sycl::event>
    py_interpolate(const dpctl::tensor::usm_ndarray &x,
                   const dpctl::tensor::usm_ndarray &idx,
                   const dpctl::tensor::usm_ndarray &xp,
                   const dpctl::tensor::usm_ndarray &fp,
                   dpctl::tensor::usm_ndarray &out,
                   sycl::queue &exec_q,
                   const std::vector<sycl::event> &depends)
{
    int xp_typenum = xp.get_typenum();
    int fp_typenum = fp.get_typenum();

    auto array_types = td_ns::usm_ndarray_types();
    int xp_type_id = array_types.typenum_to_lookup_id(xp_typenum);
    int fp_type_id = array_types.typenum_to_lookup_id(fp_typenum);

    auto fn = interpolate_dispatch_table[xp_type_id][fp_type_id];
    if (!fn) {
        throw py::type_error("Unsupported dtype.");
    }

    std::size_t n = x.get_size();
    std::size_t xp_size = xp.get_size();

    sycl::event ev = fn(exec_q, x.get_data(), idx.get_data(), xp.get_data(),
                        fp.get_data(), out.get_data(), n, xp_size, depends);

    sycl::event keep =
        dpctl::utils::keep_args_alive(exec_q, {x, idx, xp, fp, out}, {ev});

    return std::make_pair(keep, ev);
}

template <typename fnT, typename TCoord, typename TValue>
struct InterpolateFactory
{
    fnT get()
    {
        if constexpr (std::is_floating_point_v<TCoord> &&
                      std::is_floating_point_v<TValue>)
        {
            return dpnp::kernels::interpolate::interpolate_impl<TCoord, TValue>;
        }
        else if constexpr (std::is_floating_point_v<TCoord> &&
                           (std::is_same_v<TValue, std::complex<float>> ||
                            std::is_same_v<TValue, std::complex<double>>))
        {
            return dpnp::kernels::interpolate::interpolate_complex_impl<TCoord,
                                                                        TValue>;
        }
        else {
            return nullptr;
        }
    }
};

void init_interpolate_dispatch_table()
{
    using namespace td_ns;

    DispatchTableBuilder<interpolate_fn_ptr_t, InterpolateFactory, num_types>
        dtb_interpolate;
    dtb_interpolate.populate_dispatch_table(interpolate_dispatch_table);
}

} // namespace impl

void init_interpolate(py::module_ m)
{
    impl::init_interpolate_dispatch_table();

    using impl::py_interpolate;
    m.def("_interpolate", &py_interpolate, "", py::arg("x"), py::arg("idx"),
          py::arg("xp"), py::arg("fp"), py::arg("out"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());
}

} // namespace dpnp::extensions::ufunc
