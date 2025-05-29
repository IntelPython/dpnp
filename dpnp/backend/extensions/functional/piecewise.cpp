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

#include "piecewise.hpp"

#include "utils/output_validation.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

#include <pybind11/numpy.h>
#include <sycl/sycl.hpp>

namespace dpnp::extensions::functional
{
namespace dpctl_td_ns = dpctl::tensor::type_dispatch;

typedef sycl::event (*piecewise_fn_ptr_t)(sycl::queue &,
                                          const py::object &,
                                          const std::size_t,
                                          const char *,
                                          char *,
                                          const std::vector<sycl::event> &);

static piecewise_fn_ptr_t piecewise_dispatch_vector[dpctl_td_ns::num_types];

template <typename T>
class PiecewiseFunctor
{
private:
    const T val;
    const bool *cond = nullptr;
    T *res = nullptr;

public:
    PiecewiseFunctor(const T val, const bool *cond, T *res)
        : val(val), cond(cond), res(res)
    {
    }

    void operator()(sycl::id<1> id) const
    {
        const auto i = id.get(0);
        if (cond[i]) {
            res[i] = val;
        }
    }
};

template <typename T>
sycl::event piecewise_impl(sycl::queue &exec_q,
                           const py::object &value,
                           const std::size_t nelems,
                           const char *condition,
                           char *result,
                           const std::vector<sycl::event> &depends)
{
    dpctl::tensor::type_utils::validate_type_for_device<T>(exec_q);

    py::object type_obj = py::type::of(value);
    std::string type_name = py::str(type_obj.attr("__name__"));

    T *res = reinterpret_cast<T *>(result);
    const bool *cond = reinterpret_cast<const bool *>(condition);
    T val = py::cast<const T>(value);

    sycl::event piecewise_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using PiecewiseKernel = PiecewiseFunctor<T>;
        cgh.parallel_for<PiecewiseKernel>(sycl::range<1>(nelems),
                                          PiecewiseKernel(val, cond, res));
    });

    return piecewise_ev;
}

/**
 * @brief A factory to define pairs of supported types for which
 * piecewise function is available.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct PiecewiseOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, bool>,
        dpctl_td_ns::TypeMapResultEntry<T, std::uint8_t>,
        dpctl_td_ns::TypeMapResultEntry<T, std::int8_t>,
        dpctl_td_ns::TypeMapResultEntry<T, std::uint16_t>,
        dpctl_td_ns::TypeMapResultEntry<T, std::int16_t>,
        dpctl_td_ns::TypeMapResultEntry<T, std::uint32_t>,
        dpctl_td_ns::TypeMapResultEntry<T, std::int32_t>,
        dpctl_td_ns::TypeMapResultEntry<T, std::uint64_t>,
        dpctl_td_ns::TypeMapResultEntry<T, std::int64_t>,
        dpctl_td_ns::TypeMapResultEntry<T, sycl::half>,
        dpctl_td_ns::TypeMapResultEntry<T, float>,
        dpctl_td_ns::TypeMapResultEntry<T, double>,
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<float>>,
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<double>>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

template <typename fnT, typename T>
struct PiecewiseFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename PiecewiseOutputType<T>::value_type, void>) {
            return nullptr;
        }
        else {
            return piecewise_impl<T>;
        }
    }
};

std::pair<sycl::event, sycl::event>
    py_piecewise(sycl::queue &exec_q,
                 const py::object &value,
                 const dpctl::tensor::usm_ndarray &condition,
                 const dpctl::tensor::usm_ndarray &result,
                 const std::vector<sycl::event> &depends)
{
    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(result);

    const int res_nd = result.get_ndim();
    const int cond_nd = condition.get_ndim();
    if (res_nd != cond_nd) {
        throw py::value_error(
            "Condition and result arrays must have the same dimension.");
    }

    if (!dpctl::utils::queues_are_compatible(
            exec_q, {condition.get_queue(), result.get_queue()}))
    {
        throw py::value_error(
            "Execution queue is not compatible with allocation queue.");
    }

    const bool is_result_c_contig = result.is_c_contiguous();
    if (!is_result_c_contig) {
        throw py::value_error("The result array is not c-contiguous.");
    }

    const py::ssize_t *res_shape = result.get_shape_raw();
    const py::ssize_t *cond_shape = condition.get_shape_raw();

    const bool shapes_equal =
        std::equal(res_shape, res_shape + res_nd, cond_shape);
    if (!shapes_equal) {
        throw py::value_error(
            "Condition and result arrays must have the same shape.");
    }

    const std::size_t nelems = result.get_size();
    if (nelems == 0) {
        return std::make_pair(sycl::event{}, sycl::event{});
    }

    const int result_typenum = result.get_typenum();
    auto array_types = dpctl_td_ns::usm_ndarray_types();
    const int result_type_id = array_types.typenum_to_lookup_id(result_typenum);
    auto piecewise_fn = piecewise_dispatch_vector[result_type_id];

    if (piecewise_fn == nullptr) {
        throw std::runtime_error("Type of given array is not supported");
    }

    const char *condition_typeless_ptr = condition.get_data();
    char *result_typeless_ptr = result.get_data();

    sycl::event piecewise_ev =
        piecewise_fn(exec_q, value, nelems, condition_typeless_ptr,
                     result_typeless_ptr, depends);
    sycl::event args_ev =
        dpctl::utils::keep_args_alive(exec_q, {result}, {piecewise_ev});

    return std::make_pair(args_ev, piecewise_ev);
}

void init_piecewise_dispatch_vectors(void)
{
    dpctl_td_ns::DispatchVectorBuilder<piecewise_fn_ptr_t, PiecewiseFactory,
                                       dpctl_td_ns::num_types>
        contig;
    contig.populate_dispatch_vector(piecewise_dispatch_vector);

    return;
}

} // namespace dpnp::extensions::functional
