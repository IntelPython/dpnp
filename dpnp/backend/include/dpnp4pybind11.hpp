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

#pragma once

// Include for dpctl_capi struct and casters
#include "dpctl4pybind11.hpp"

// Include generated Cython headers for usm_ndarray
// (struct definition and constants only)
#include "dpnp/tensor/_usmarray.h"
#include "dpnp/tensor/_usmarray_api.h"

#include <cassert>
#include <complex>
#include <cstddef> // for std::size_t for C++ linkage
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>

#include <sycl/sycl.hpp>

namespace py = pybind11;

namespace dpctl
{
namespace detail
{
// Lookup a type according to its size, and return a value corresponding to the
// NumPy typenum.

// TODO: uncomment when these are removed from dpctl4pybind11 or when dpctl
// namespace is changed to dpnp
// template <typename Concrete>
// constexpr int platform_typeid_lookup()
// {
//     return -1;
// }

// template <typename Concrete, typename T, typename... Ts, typename... Ints>
// constexpr int platform_typeid_lookup(int I, Ints... Is)
// {
//     return sizeof(Concrete) == sizeof(T)
//                ? I
//                : platform_typeid_lookup<Concrete, Ts...>(Is...);
// }

class dpnp_capi
{
public:
    PyTypeObject *PyUSMArrayType_;

    int USM_ARRAY_C_CONTIGUOUS_;
    int USM_ARRAY_F_CONTIGUOUS_;
    int USM_ARRAY_WRITABLE_;
    int UAR_BOOL_, UAR_BYTE_, UAR_UBYTE_, UAR_SHORT_, UAR_USHORT_, UAR_INT_,
        UAR_UINT_, UAR_LONG_, UAR_ULONG_, UAR_LONGLONG_, UAR_ULONGLONG_,
        UAR_FLOAT_, UAR_DOUBLE_, UAR_CFLOAT_, UAR_CDOUBLE_, UAR_TYPE_SENTINEL_,
        UAR_HALF_;
    int UAR_INT8_, UAR_UINT8_, UAR_INT16_, UAR_UINT16_, UAR_INT32_, UAR_UINT32_,
        UAR_INT64_, UAR_UINT64_;

    ~dpnp_capi() { default_usm_ndarray_.reset(); };

    static auto &get()
    {
        static dpnp_capi api{};
        return api;
    }

    py::object default_usm_ndarray_pyobj() { return *default_usm_ndarray_; }

private:
    struct Deleter
    {
        void operator()(py::object *p) const
        {
            const bool initialized = Py_IsInitialized();
#if PY_VERSION_HEX < 0x30d0000
            const bool finalizing = _Py_IsFinalizing();
#else
            const bool finalizing = Py_IsFinalizing();
#endif
            const bool guard = initialized && !finalizing;

            if (guard) {
                delete p;
            }
        }
    };

    std::shared_ptr<py::object> default_usm_ndarray_;

    dpnp_capi()
        : PyUSMArrayType_(nullptr), USM_ARRAY_C_CONTIGUOUS_(0),
          USM_ARRAY_F_CONTIGUOUS_(0), USM_ARRAY_WRITABLE_(0), UAR_BOOL_(-1),
          UAR_BYTE_(-1), UAR_UBYTE_(-1), UAR_SHORT_(-1), UAR_USHORT_(-1),
          UAR_INT_(-1), UAR_UINT_(-1), UAR_LONG_(-1), UAR_ULONG_(-1),
          UAR_LONGLONG_(-1), UAR_ULONGLONG_(-1), UAR_FLOAT_(-1),
          UAR_DOUBLE_(-1), UAR_CFLOAT_(-1), UAR_CDOUBLE_(-1),
          UAR_TYPE_SENTINEL_(-1), UAR_HALF_(-1), UAR_INT8_(-1), UAR_UINT8_(-1),
          UAR_INT16_(-1), UAR_UINT16_(-1), UAR_INT32_(-1), UAR_UINT32_(-1),
          UAR_INT64_(-1), UAR_UINT64_(-1), default_usm_ndarray_{}

    {
        // Import dpnp tensor module for PyUSMArrayType
        import_dpnp__tensor___usmarray();

        this->PyUSMArrayType_ = &PyUSMArrayType;

        // constants
        this->USM_ARRAY_C_CONTIGUOUS_ = USM_ARRAY_C_CONTIGUOUS;
        this->USM_ARRAY_F_CONTIGUOUS_ = USM_ARRAY_F_CONTIGUOUS;
        this->USM_ARRAY_WRITABLE_ = USM_ARRAY_WRITABLE;
        this->UAR_BOOL_ = UAR_BOOL;
        this->UAR_BYTE_ = UAR_BYTE;
        this->UAR_UBYTE_ = UAR_UBYTE;
        this->UAR_SHORT_ = UAR_SHORT;
        this->UAR_USHORT_ = UAR_USHORT;
        this->UAR_INT_ = UAR_INT;
        this->UAR_UINT_ = UAR_UINT;
        this->UAR_LONG_ = UAR_LONG;
        this->UAR_ULONG_ = UAR_ULONG;
        this->UAR_LONGLONG_ = UAR_LONGLONG;
        this->UAR_ULONGLONG_ = UAR_ULONGLONG;
        this->UAR_FLOAT_ = UAR_FLOAT;
        this->UAR_DOUBLE_ = UAR_DOUBLE;
        this->UAR_CFLOAT_ = UAR_CFLOAT;
        this->UAR_CDOUBLE_ = UAR_CDOUBLE;
        this->UAR_TYPE_SENTINEL_ = UAR_TYPE_SENTINEL;
        this->UAR_HALF_ = UAR_HALF;

        // deduced disjoint types
        this->UAR_INT8_ = UAR_BYTE;
        this->UAR_UINT8_ = UAR_UBYTE;
        this->UAR_INT16_ = UAR_SHORT;
        this->UAR_UINT16_ = UAR_USHORT;
        this->UAR_INT32_ =
            platform_typeid_lookup<std::int32_t, long, int, short>(
                UAR_LONG, UAR_INT, UAR_SHORT);
        this->UAR_UINT32_ =
            platform_typeid_lookup<std::uint32_t, unsigned long, unsigned int,
                                   unsigned short>(UAR_ULONG, UAR_UINT,
                                                   UAR_USHORT);
        this->UAR_INT64_ =
            platform_typeid_lookup<std::int64_t, long, long long, int>(
                UAR_LONG, UAR_LONGLONG, UAR_INT);
        this->UAR_UINT64_ =
            platform_typeid_lookup<std::uint64_t, unsigned long,
                                   unsigned long long, unsigned int>(
                UAR_ULONG, UAR_ULONGLONG, UAR_UINT);

        py::object py_default_usm_memory =
            ::dpctl::detail::dpctl_capi::get().default_usm_memory_pyobj();

        py::module_ mod_usmarray = py::module_::import("dpnp.tensor._usmarray");
        auto tensor_kl = mod_usmarray.attr("usm_ndarray");

        const py::object &py_default_usm_ndarray =
            tensor_kl(py::tuple(), py::arg("dtype") = py::str("u1"),
                      py::arg("buffer") = py_default_usm_memory);

        default_usm_ndarray_ = std::shared_ptr<py::object>(
            new py::object{py_default_usm_ndarray}, Deleter{});
    }

    dpnp_capi(dpnp_capi const &) = default;
    dpnp_capi &operator=(dpnp_capi const &) = default;
    dpnp_capi &operator=(dpnp_capi &&) = default;

}; // struct dpnp_capi
} // namespace detail

namespace tensor
{
inline std::vector<py::ssize_t>
    c_contiguous_strides(int nd,
                         const py::ssize_t *shape,
                         py::ssize_t element_size = 1)
{
    if (nd > 0) {
        std::vector<py::ssize_t> c_strides(nd, element_size);
        for (int ic = nd - 1; ic > 0;) {
            py::ssize_t next_v = c_strides[ic] * shape[ic];
            c_strides[--ic] = next_v;
        }
        return c_strides;
    }
    else {
        return std::vector<py::ssize_t>();
    }
}

inline std::vector<py::ssize_t>
    f_contiguous_strides(int nd,
                         const py::ssize_t *shape,
                         py::ssize_t element_size = 1)
{
    if (nd > 0) {
        std::vector<py::ssize_t> f_strides(nd, element_size);
        for (int i = 0; i < nd - 1;) {
            py::ssize_t next_v = f_strides[i] * shape[i];
            f_strides[++i] = next_v;
        }
        return f_strides;
    }
    else {
        return std::vector<py::ssize_t>();
    }
}

inline std::vector<py::ssize_t>
    c_contiguous_strides(const std::vector<py::ssize_t> &shape,
                         py::ssize_t element_size = 1)
{
    return c_contiguous_strides(shape.size(), shape.data(), element_size);
}

inline std::vector<py::ssize_t>
    f_contiguous_strides(const std::vector<py::ssize_t> &shape,
                         py::ssize_t element_size = 1)
{
    return f_contiguous_strides(shape.size(), shape.data(), element_size);
}

class usm_ndarray : public py::object
{
public:
    PYBIND11_OBJECT(usm_ndarray, py::object, [](PyObject *o) -> bool {
        return PyObject_TypeCheck(
                   o, ::dpctl::detail::dpnp_capi::get().PyUSMArrayType_) != 0;
    })

    usm_ndarray()
        : py::object(
              ::dpctl::detail::dpnp_capi::get().default_usm_ndarray_pyobj(),
              borrowed_t{})
    {
        if (!m_ptr)
            throw py::error_already_set();
    }

    char *get_data() const
    {
        PyUSMArrayObject *raw_ar = usm_array_ptr();
        return raw_ar->data_;
    }

    template <typename T>
    T *get_data() const
    {
        return reinterpret_cast<T *>(get_data());
    }

    int get_ndim() const
    {
        PyUSMArrayObject *raw_ar = usm_array_ptr();
        return raw_ar->nd_;
    }

    const py::ssize_t *get_shape_raw() const
    {
        PyUSMArrayObject *raw_ar = usm_array_ptr();
        return raw_ar->shape_;
    }

    std::vector<py::ssize_t> get_shape_vector() const
    {
        auto raw_sh = get_shape_raw();
        auto nd = get_ndim();

        std::vector<py::ssize_t> shape_vector(raw_sh, raw_sh + nd);
        return shape_vector;
    }

    py::ssize_t get_shape(int i) const
    {
        auto shape_ptr = get_shape_raw();
        return shape_ptr[i];
    }

    const py::ssize_t *get_strides_raw() const
    {
        PyUSMArrayObject *raw_ar = usm_array_ptr();
        return raw_ar->strides_;
    }

    std::vector<py::ssize_t> get_strides_vector() const
    {
        auto raw_st = get_strides_raw();
        auto nd = get_ndim();

        if (raw_st == nullptr) {
            auto is_c_contig = is_c_contiguous();
            auto is_f_contig = is_f_contiguous();
            auto raw_sh = get_shape_raw();
            if (is_c_contig) {
                const auto &contig_strides = c_contiguous_strides(nd, raw_sh);
                return contig_strides;
            }
            else if (is_f_contig) {
                const auto &contig_strides = f_contiguous_strides(nd, raw_sh);
                return contig_strides;
            }
            else {
                throw std::runtime_error("Invalid array encountered when "
                                         "building strides");
            }
        }
        else {
            std::vector<py::ssize_t> st_vec(raw_st, raw_st + nd);
            return st_vec;
        }
    }

    py::ssize_t get_size() const
    {
        PyUSMArrayObject *raw_ar = usm_array_ptr();

        int ndim = raw_ar->nd_;
        const py::ssize_t *shape = raw_ar->shape_;

        py::ssize_t nelems = 1;
        for (int i = 0; i < ndim; ++i) {
            nelems *= shape[i];
        }

        assert(nelems >= 0);
        return nelems;
    }

    std::pair<py::ssize_t, py::ssize_t> get_minmax_offsets() const
    {
        PyUSMArrayObject *raw_ar = usm_array_ptr();

        int nd = raw_ar->nd_;
        const py::ssize_t *shape = raw_ar->shape_;
        const py::ssize_t *strides = raw_ar->strides_;

        py::ssize_t offset_min = 0;
        py::ssize_t offset_max = 0;
        if (strides == nullptr) {
            py::ssize_t stride(1);
            for (int i = 0; i < nd; ++i) {
                offset_max += stride * (shape[i] - 1);
                stride *= shape[i];
            }
        }
        else {
            for (int i = 0; i < nd; ++i) {
                py::ssize_t delta = strides[i] * (shape[i] - 1);
                if (strides[i] > 0) {
                    offset_max += delta;
                }
                else {
                    offset_min += delta;
                }
            }
        }
        return std::make_pair(offset_min, offset_max);
    }

    sycl::queue get_queue() const
    {
        PyUSMArrayObject *raw_ar = usm_array_ptr();
        Py_MemoryObject *mem_obj =
            reinterpret_cast<Py_MemoryObject *>(raw_ar->base_);

        auto const &dpctl_api = ::dpctl::detail::dpctl_capi::get();
        DPCTLSyclQueueRef QRef = dpctl_api.Memory_GetQueueRef_(mem_obj);
        return *(reinterpret_cast<sycl::queue *>(QRef));
    }

    sycl::device get_device() const
    {
        PyUSMArrayObject *raw_ar = usm_array_ptr();
        Py_MemoryObject *mem_obj =
            reinterpret_cast<Py_MemoryObject *>(raw_ar->base_);

        auto const &dpctl_api = ::dpctl::detail::dpctl_capi::get();
        DPCTLSyclQueueRef QRef = dpctl_api.Memory_GetQueueRef_(mem_obj);
        return reinterpret_cast<sycl::queue *>(QRef)->get_device();
    }

    int get_typenum() const
    {
        PyUSMArrayObject *raw_ar = usm_array_ptr();
        return raw_ar->typenum_;
    }

    int get_flags() const
    {
        PyUSMArrayObject *raw_ar = usm_array_ptr();
        return raw_ar->flags_;
    }

    int get_elemsize() const
    {
        int typenum = get_typenum();
        auto const &api = ::dpctl::detail::dpnp_capi::get();

        // Lookup table for element sizes based on typenum
        if (typenum == api.UAR_BOOL_)
            return 1;
        if (typenum == api.UAR_BYTE_)
            return 1;
        if (typenum == api.UAR_UBYTE_)
            return 1;
        if (typenum == api.UAR_SHORT_)
            return 2;
        if (typenum == api.UAR_USHORT_)
            return 2;
        if (typenum == api.UAR_INT_)
            return 4;
        if (typenum == api.UAR_UINT_)
            return 4;
        if (typenum == api.UAR_LONG_)
            return sizeof(long);
        if (typenum == api.UAR_ULONG_)
            return sizeof(unsigned long);
        if (typenum == api.UAR_LONGLONG_)
            return 8;
        if (typenum == api.UAR_ULONGLONG_)
            return 8;
        if (typenum == api.UAR_FLOAT_)
            return 4;
        if (typenum == api.UAR_DOUBLE_)
            return 8;
        if (typenum == api.UAR_CFLOAT_)
            return 8;
        if (typenum == api.UAR_CDOUBLE_)
            return 16;
        if (typenum == api.UAR_HALF_)
            return 2;

        return 0; // Unknown type
    }

    bool is_c_contiguous() const
    {
        int flags = get_flags();
        auto const &api = ::dpctl::detail::dpnp_capi::get();
        return static_cast<bool>(flags & api.USM_ARRAY_C_CONTIGUOUS_);
    }

    bool is_f_contiguous() const
    {
        int flags = get_flags();
        auto const &api = ::dpctl::detail::dpnp_capi::get();
        return static_cast<bool>(flags & api.USM_ARRAY_F_CONTIGUOUS_);
    }

    bool is_writable() const
    {
        int flags = get_flags();
        auto const &api = ::dpctl::detail::dpnp_capi::get();
        return static_cast<bool>(flags & api.USM_ARRAY_WRITABLE_);
    }

    /*! @brief Get usm_data property of array */
    py::object get_usm_data() const
    {
        PyUSMArrayObject *raw_ar = usm_array_ptr();
        // base_ is the Memory object - return new reference
        PyObject *usm_data = raw_ar->base_;
        Py_XINCREF(usm_data);

        // pass reference ownership to py::object
        return py::reinterpret_steal<py::object>(usm_data);
    }

    bool is_managed_by_smart_ptr() const
    {
        PyUSMArrayObject *raw_ar = usm_array_ptr();
        PyObject *usm_data = raw_ar->base_;

        auto const &dpctl_api = ::dpctl::detail::dpctl_capi::get();
        if (!PyObject_TypeCheck(usm_data, dpctl_api.Py_MemoryType_)) {
            return false;
        }

        Py_MemoryObject *mem_obj =
            reinterpret_cast<Py_MemoryObject *>(usm_data);
        const void *opaque_ptr = dpctl_api.Memory_GetOpaquePointer_(mem_obj);

        return bool(opaque_ptr);
    }

    const std::shared_ptr<void> &get_smart_ptr_owner() const
    {
        PyUSMArrayObject *raw_ar = usm_array_ptr();
        PyObject *usm_data = raw_ar->base_;

        auto const &dpctl_api = ::dpctl::detail::dpctl_capi::get();

        if (!PyObject_TypeCheck(usm_data, dpctl_api.Py_MemoryType_)) {
            throw std::runtime_error(
                "usm_ndarray object does not have Memory object "
                "managing lifetime of USM allocation");
        }

        Py_MemoryObject *mem_obj =
            reinterpret_cast<Py_MemoryObject *>(usm_data);
        void *opaque_ptr = dpctl_api.Memory_GetOpaquePointer_(mem_obj);

        if (opaque_ptr) {
            auto shptr_ptr =
                reinterpret_cast<std::shared_ptr<void> *>(opaque_ptr);
            return *shptr_ptr;
        }
        else {
            throw std::runtime_error(
                "Memory object underlying usm_ndarray does not have "
                "smart pointer managing lifetime of USM allocation");
        }
    }

private:
    PyUSMArrayObject *usm_array_ptr() const
    {
        return reinterpret_cast<PyUSMArrayObject *>(m_ptr);
    }
};
} // end namespace tensor

namespace utils
{

/*! @brief Check if all allocation queues of usm_ndarrays are the same as
    the execution queue */
template <std::size_t num>
bool queues_are_compatible(const sycl::queue &exec_q,
                           const ::dpctl::tensor::usm_ndarray (&arrs)[num])
{
    for (std::size_t i = 0; i < num; ++i) {

        if (exec_q != arrs[i].get_queue()) {
            return false;
        }
    }
    return true;
}
} // end namespace utils
} // end namespace dpctl
