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

#include "kaiser.hpp"
#include "common.hpp"

#include "utils/output_validation.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

#include <sycl/sycl.hpp>

#include "../kernels/elementwise_functions/i0.hpp"

namespace dpnp::extensions::window
{
namespace dpctl_td_ns = dpctl::tensor::type_dispatch;

typedef sycl::event (*kaiser_fn_ptr_t)(sycl::queue &,
                                       char *,
                                       const std::size_t,
                                       const py::object &,
                                       const std::vector<sycl::event> &);

static kaiser_fn_ptr_t kaiser_dispatch_vector[dpctl_td_ns::num_types];

template <typename T>
class KaiserFunctor
{
private:
    T *data = nullptr;
    const std::size_t N;
    const T beta;

public:
    KaiserFunctor(T *data, const std::size_t N, const T beta)
        : data(data), N(N), beta(beta)
    {
    }

    void operator()(sycl::id<1> id) const
    {
        using dpnp::kernels::i0::cyl_bessel_i0;

        const auto i = id.get(0);
        const T alpha = (N - 1) / T(2);
        const T tmp = (i - alpha) / alpha;
        data[i] = cyl_bessel_i0(beta * sycl::sqrt(1 - tmp * tmp)) /
                  cyl_bessel_i0(beta);
    }
};

template <typename T, template <typename> class Functor>
sycl::event kaiser_impl(sycl::queue &q,
                        char *result,
                        const std::size_t nelems,
                        const py::object &py_beta,
                        const std::vector<sycl::event> &depends)
{
    dpctl::tensor::type_utils::validate_type_for_device<T>(q);

    T *res = reinterpret_cast<T *>(result);
    const T beta = py::cast<const T>(py_beta);

    sycl::event kaiser_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using KaiserKernel = Functor<T>;
        cgh.parallel_for<KaiserKernel>(sycl::range<1>(nelems),
                                       KaiserKernel(res, nelems, beta));
    });

    return kaiser_ev;
}

template <typename fnT, typename T>
struct KaiserFactory
{
    fnT get()
    {
        if constexpr (std::is_floating_point_v<T>) {
            return kaiser_impl<T, KaiserFunctor>;
        }
        else {
            return nullptr;
        }
    }
};

std::pair<sycl::event, sycl::event>
    py_kaiser(sycl::queue &exec_q,
              const py::object &py_beta,
              const dpctl::tensor::usm_ndarray &result,
              const std::vector<sycl::event> &depends)
{
    auto [nelems, result_typeless_ptr, fn] =
        window_fn<kaiser_fn_ptr_t>(exec_q, result, kaiser_dispatch_vector);

    if (nelems == 0) {
        return std::make_pair(sycl::event{}, sycl::event{});
    }

    sycl::event kaiser_ev =
        fn(exec_q, result_typeless_ptr, nelems, py_beta, depends);
    sycl::event args_ev =
        dpctl::utils::keep_args_alive(exec_q, {result}, {kaiser_ev});

    return std::make_pair(args_ev, kaiser_ev);
}

void init_kaiser_dispatch_vectors()
{
    init_window_dispatch_vectors<kaiser_fn_ptr_t, KaiserFactory>(
        kaiser_dispatch_vector);
}

} // namespace dpnp::extensions::window
