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

#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// dpctl tensor headers
#include "dpctl4pybind11.hpp"
#include "utils/type_dispatch.hpp"

#include "interpolate.hpp"
#include "interpolate_kernel.hpp"

namespace dpnp::extensions::math
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

static kernels::interpolate_fn_ptr_t interpolate_dispatch_table[td_ns::num_types];

template <typename fnT, typename T>
struct InterpolateFactory
{
    fnT get()
    {
        if constexpr (std::is_floating_point_v<T>) {
            return kernels::interpolate_impl<T>;
        }
        else {
            return nullptr;
        }
    }
};


std::pair<sycl::event, sycl::event>
py_interpolate(const dpctl::tensor::usm_ndarray &x,
               const dpctl::tensor::usm_ndarray &idx,
               const dpctl::tensor::usm_ndarray &xp,
               const dpctl::tensor::usm_ndarray &fp,
               dpctl::tensor::usm_ndarray &out,
               sycl::queue &exec_q,
               const std::vector<sycl::event> &depends)
{
    int typenum = x.get_typenum();
    auto array_types = td_ns::usm_ndarray_types();
    int type_id = array_types.typenum_to_lookup_id(typenum);

    auto fn = interpolate_dispatch_table[type_id];
    if (!fn) {
        throw py::type_error("Unsupported dtype.");
    }

    std::size_t n = x.get_size();
    std::size_t xp_size = xp.get_size();

    sycl::event ev = fn(exec_q, x.get_data(), idx.get_data(), xp.get_data(),
                        fp.get_data(), out.get_data(), n, xp_size, depends);

    sycl::event keep = dpctl::utils::keep_args_alive(exec_q, {x, idx, xp, fp, out}, {ev});

    return std::make_pair(keep, ev);
}


void init_interpolate_dispatch_table()
{
    using namespace td_ns;
    using kernels::interpolate_fn_ptr_t;

    DispatchVectorBuilder<interpolate_fn_ptr_t, InterpolateFactory, num_types>
        dtb_interpolate;
    dtb_interpolate.populate_dispatch_vector(interpolate_dispatch_table);
}

void init_interpolate(py::module_ m)
{
    dpnp::extensions::math::init_interpolate_dispatch_table();

    m.def("_interpolate", &py_interpolate, "",
          py::arg("x"), py::arg("idx"), py::arg("xp"), py::arg("fp"),
          py::arg("out"), py::arg("sycl_queue"), py::arg("depends") = py::list());
}

} // namespace dpnp::extensions::math
