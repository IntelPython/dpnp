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
#include "dpnp/tensor/_usmarray.h"
#include "dpnp/tensor/_usmarray_api.h"
// Include usm_ndarray constants (flags, type numbers)
#include "usm_ndarray_constants.h"

#include <array>
#include <cassert>
#include <cstddef> // for std::size_t for C++ linkage
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>

#include <sycl/sycl.hpp>

namespace py = pybind11;

namespace dpnp
{
namespace detail
{
// Lookup a type according to its size, and return a value corresponding to the
// NumPy typenum.

template <typename Concrete>
constexpr int platform_typeid_lookup()
{
    return -1;
}

template <typename Concrete, typename T, typename... Ts, typename... Ints>
constexpr int platform_typeid_lookup(int I, Ints... Is)
{
    return sizeof(Concrete) == sizeof(T)
               ? I
               : platform_typeid_lookup<Concrete, Ts...>(Is...);
}

class dpnp_capi
{
public:
    PyTypeObject *PyUSMArrayType_;

    char *(*UsmNDArray_GetData_)(PyUSMArrayObject *);
    int (*UsmNDArray_GetNDim_)(PyUSMArrayObject *);
    py::ssize_t *(*UsmNDArray_GetShape_)(PyUSMArrayObject *);
    py::ssize_t *(*UsmNDArray_GetStrides_)(PyUSMArrayObject *);
    int (*UsmNDArray_GetTypenum_)(PyUSMArrayObject *);
    int (*UsmNDArray_GetElementSize_)(PyUSMArrayObject *);
    int (*UsmNDArray_GetFlags_)(PyUSMArrayObject *);
    DPCTLSyclQueueRef (*UsmNDArray_GetQueueRef_)(PyUSMArrayObject *);
    py::ssize_t (*UsmNDArray_GetOffset_)(PyUSMArrayObject *);
    PyObject *(*UsmNDArray_GetUSMData_)(PyUSMArrayObject *);
    void (*UsmNDArray_SetWritableFlag_)(PyUSMArrayObject *, int);
    PyObject *(*UsmNDArray_MakeSimpleFromMemory_)(int,
                                                  const py::ssize_t *,
                                                  int,
                                                  Py_MemoryObject *,
                                                  py::ssize_t,
                                                  char);
    PyObject *(*UsmNDArray_MakeSimpleFromPtr_)(size_t,
                                               int,
                                               DPCTLSyclUSMRef,
                                               DPCTLSyclQueueRef,
                                               PyObject *);
    PyObject *(*UsmNDArray_MakeFromPtr_)(int,
                                         const py::ssize_t *,
                                         int,
                                         const py::ssize_t *,
                                         DPCTLSyclUSMRef,
                                         DPCTLSyclQueueRef,
                                         py::ssize_t,
                                         PyObject *);

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
        : PyUSMArrayType_(nullptr), UsmNDArray_GetData_(nullptr),
          UsmNDArray_GetNDim_(nullptr), UsmNDArray_GetShape_(nullptr),
          UsmNDArray_GetStrides_(nullptr), UsmNDArray_GetTypenum_(nullptr),
          UsmNDArray_GetElementSize_(nullptr), UsmNDArray_GetFlags_(nullptr),
          UsmNDArray_GetQueueRef_(nullptr), UsmNDArray_GetOffset_(nullptr),
          UsmNDArray_GetUSMData_(nullptr), UsmNDArray_SetWritableFlag_(nullptr),
          UsmNDArray_MakeSimpleFromMemory_(nullptr),
          UsmNDArray_MakeSimpleFromPtr_(nullptr),
          UsmNDArray_MakeFromPtr_(nullptr), USM_ARRAY_C_CONTIGUOUS_(0),
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

        // dpnp.tensor.usm_ndarray API
        this->UsmNDArray_GetData_ = UsmNDArray_GetData;
        this->UsmNDArray_GetNDim_ = UsmNDArray_GetNDim;
        this->UsmNDArray_GetShape_ = UsmNDArray_GetShape;
        this->UsmNDArray_GetStrides_ = UsmNDArray_GetStrides;
        this->UsmNDArray_GetTypenum_ = UsmNDArray_GetTypenum;
        this->UsmNDArray_GetElementSize_ = UsmNDArray_GetElementSize;
        this->UsmNDArray_GetFlags_ = UsmNDArray_GetFlags;
        this->UsmNDArray_GetQueueRef_ = UsmNDArray_GetQueueRef;
        this->UsmNDArray_GetOffset_ = UsmNDArray_GetOffset;
        this->UsmNDArray_GetUSMData_ = UsmNDArray_GetUSMData;
        this->UsmNDArray_SetWritableFlag_ = UsmNDArray_SetWritableFlag;
        this->UsmNDArray_MakeSimpleFromMemory_ =
            UsmNDArray_MakeSimpleFromMemory;
        this->UsmNDArray_MakeSimpleFromPtr_ = UsmNDArray_MakeSimpleFromPtr;
        this->UsmNDArray_MakeFromPtr_ = UsmNDArray_MakeFromPtr;

        // constants from usm_ndarray_constants.h
        this->USM_ARRAY_C_CONTIGUOUS_ = USM_ARRAY_C_CONTIGUOUS_VALUE;
        this->USM_ARRAY_F_CONTIGUOUS_ = USM_ARRAY_F_CONTIGUOUS_VALUE;
        this->USM_ARRAY_WRITABLE_ = USM_ARRAY_WRITABLE_VALUE;
        this->UAR_BOOL_ = UAR_BOOL_VALUE;
        this->UAR_BYTE_ = UAR_BYTE_VALUE;
        this->UAR_UBYTE_ = UAR_UBYTE_VALUE;
        this->UAR_SHORT_ = UAR_SHORT_VALUE;
        this->UAR_USHORT_ = UAR_USHORT_VALUE;
        this->UAR_INT_ = UAR_INT_VALUE;
        this->UAR_UINT_ = UAR_UINT_VALUE;
        this->UAR_LONG_ = UAR_LONG_VALUE;
        this->UAR_ULONG_ = UAR_ULONG_VALUE;
        this->UAR_LONGLONG_ = UAR_LONGLONG_VALUE;
        this->UAR_ULONGLONG_ = UAR_ULONGLONG_VALUE;
        this->UAR_FLOAT_ = UAR_FLOAT_VALUE;
        this->UAR_DOUBLE_ = UAR_DOUBLE_VALUE;
        this->UAR_CFLOAT_ = UAR_CFLOAT_VALUE;
        this->UAR_CDOUBLE_ = UAR_CDOUBLE_VALUE;
        this->UAR_TYPE_SENTINEL_ = UAR_TYPE_SENTINEL_VALUE;
        this->UAR_HALF_ = UAR_HALF_VALUE;

        // deduced disjoint types
        this->UAR_INT8_ = UAR_BYTE_VALUE;
        this->UAR_UINT8_ = UAR_UBYTE_VALUE;
        this->UAR_INT16_ = UAR_SHORT_VALUE;
        this->UAR_UINT16_ = UAR_USHORT_VALUE;
        this->UAR_INT32_ =
            platform_typeid_lookup<std::int32_t, long, int, short>(
                UAR_LONG_VALUE, UAR_INT_VALUE, UAR_SHORT_VALUE);
        this->UAR_UINT32_ =
            platform_typeid_lookup<std::uint32_t, unsigned long, unsigned int,
                                   unsigned short>(
                UAR_ULONG_VALUE, UAR_UINT_VALUE, UAR_USHORT_VALUE);
        this->UAR_INT64_ =
            platform_typeid_lookup<std::int64_t, long, long long, int>(
                UAR_LONG_VALUE, UAR_LONGLONG_VALUE, UAR_INT_VALUE);
        this->UAR_UINT64_ =
            platform_typeid_lookup<std::uint64_t, unsigned long,
                                   unsigned long long, unsigned int>(
                UAR_ULONG_VALUE, UAR_ULONGLONG_VALUE, UAR_UINT_VALUE);

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
                   o, detail::dpnp_capi::get().PyUSMArrayType_) != 0;
    })

    usm_ndarray()
        : py::object(detail::dpnp_capi::get().default_usm_ndarray_pyobj(),
                     borrowed_t{})
    {
        if (!m_ptr)
            throw py::error_already_set();
    }

    char *get_data() const
    {
        PyUSMArrayObject *raw_ar = usm_array_ptr();

        auto const &api = detail::dpnp_capi::get();
        return api.UsmNDArray_GetData_(raw_ar);
    }

    template <typename T>
    T *get_data() const
    {
        return reinterpret_cast<T *>(get_data());
    }

    int get_ndim() const
    {
        PyUSMArrayObject *raw_ar = usm_array_ptr();

        auto const &api = detail::dpnp_capi::get();
        return api.UsmNDArray_GetNDim_(raw_ar);
    }

    const py::ssize_t *get_shape_raw() const
    {
        PyUSMArrayObject *raw_ar = usm_array_ptr();

        auto const &api = detail::dpnp_capi::get();
        return api.UsmNDArray_GetShape_(raw_ar);
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

        auto const &api = detail::dpnp_capi::get();
        return api.UsmNDArray_GetStrides_(raw_ar);
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

        auto const &api = detail::dpnp_capi::get();
        int ndim = api.UsmNDArray_GetNDim_(raw_ar);
        const py::ssize_t *shape = api.UsmNDArray_GetShape_(raw_ar);

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

        auto const &api = detail::dpnp_capi::get();
        int nd = api.UsmNDArray_GetNDim_(raw_ar);
        const py::ssize_t *shape = api.UsmNDArray_GetShape_(raw_ar);
        const py::ssize_t *strides = api.UsmNDArray_GetStrides_(raw_ar);

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

        auto const &api = detail::dpnp_capi::get();
        DPCTLSyclQueueRef QRef = api.UsmNDArray_GetQueueRef_(raw_ar);
        return *(reinterpret_cast<sycl::queue *>(QRef));
    }

    sycl::device get_device() const
    {
        PyUSMArrayObject *raw_ar = usm_array_ptr();

        auto const &api = detail::dpnp_capi::get();
        DPCTLSyclQueueRef QRef = api.UsmNDArray_GetQueueRef_(raw_ar);
        return reinterpret_cast<sycl::queue *>(QRef)->get_device();
    }

    int get_typenum() const
    {
        PyUSMArrayObject *raw_ar = usm_array_ptr();

        auto const &api = detail::dpnp_capi::get();
        return api.UsmNDArray_GetTypenum_(raw_ar);
    }

    int get_flags() const
    {
        PyUSMArrayObject *raw_ar = usm_array_ptr();

        auto const &api = detail::dpnp_capi::get();
        return api.UsmNDArray_GetFlags_(raw_ar);
    }

    int get_elemsize() const
    {
        PyUSMArrayObject *raw_ar = usm_array_ptr();

        auto const &api = detail::dpnp_capi::get();
        return api.UsmNDArray_GetElementSize_(raw_ar);
    }

    bool is_c_contiguous() const
    {
        int flags = get_flags();
        auto const &api = detail::dpnp_capi::get();
        return static_cast<bool>(flags & api.USM_ARRAY_C_CONTIGUOUS_);
    }

    bool is_f_contiguous() const
    {
        int flags = get_flags();
        auto const &api = detail::dpnp_capi::get();
        return static_cast<bool>(flags & api.USM_ARRAY_F_CONTIGUOUS_);
    }

    bool is_writable() const
    {
        int flags = get_flags();
        auto const &api = detail::dpnp_capi::get();
        return static_cast<bool>(flags & api.USM_ARRAY_WRITABLE_);
    }

    /*! @brief Get usm_data property of array */
    py::object get_usm_data() const
    {
        PyUSMArrayObject *raw_ar = usm_array_ptr();

        auto const &api = detail::dpnp_capi::get();
        // base_ is the Memory object - return new reference
        PyObject *usm_data = api.UsmNDArray_GetUSMData_(raw_ar);

        // pass reference ownership to py::object
        return py::reinterpret_steal<py::object>(usm_data);
    }

    bool is_managed_by_smart_ptr() const
    {
        PyUSMArrayObject *raw_ar = usm_array_ptr();

        auto const &api = detail::dpnp_capi::get();
        PyObject *usm_data = api.UsmNDArray_GetUSMData_(raw_ar);

        auto const &dpctl_api = ::dpctl::detail::dpctl_capi::get();
        if (!PyObject_TypeCheck(usm_data, dpctl_api.Py_MemoryType_)) {
            Py_DECREF(usm_data);
            return false;
        }

        Py_MemoryObject *mem_obj =
            reinterpret_cast<Py_MemoryObject *>(usm_data);
        const void *opaque_ptr = dpctl_api.Memory_GetOpaquePointer_(mem_obj);

        Py_DECREF(usm_data);
        return bool(opaque_ptr);
    }

    const std::shared_ptr<void> &get_smart_ptr_owner() const
    {
        PyUSMArrayObject *raw_ar = usm_array_ptr();

        auto const &api = detail::dpnp_capi::get();
        PyObject *usm_data = api.UsmNDArray_GetUSMData_(raw_ar);

        auto const &dpctl_api = ::dpctl::detail::dpctl_capi::get();
        if (!PyObject_TypeCheck(usm_data, dpctl_api.Py_MemoryType_)) {
            Py_DECREF(usm_data);
            throw std::runtime_error(
                "usm_ndarray object does not have Memory object "
                "managing lifetime of USM allocation");
        }

        Py_MemoryObject *mem_obj =
            reinterpret_cast<Py_MemoryObject *>(usm_data);
        void *opaque_ptr = dpctl_api.Memory_GetOpaquePointer_(mem_obj);
        Py_DECREF(usm_data);

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
namespace detail
{
// TODO: future version of dpctl will include a more general way of passing
// shared_ptrs to keep_args_alive, so that future overload can be used here
// instead of reimplementing keep_args_alive

struct ManagedMemory
{
    // TODO: do we need to check for memory here? Or can we assume only
    // dpnp::tensor::usm_ndarray will be passed?
    static bool is_usm_managed_by_shared_ptr(const py::object &h)
    {

        if (py::isinstance<::dpctl::memory::usm_memory>(h)) {
            const auto &usm_memory_inst =
                py::cast<::dpctl::memory::usm_memory>(h);
            return usm_memory_inst.is_managed_by_smart_ptr();
        }
        else if (py::isinstance<tensor::usm_ndarray>(h)) {
            const auto &usm_array_inst = py::cast<tensor::usm_ndarray>(h);
            return usm_array_inst.is_managed_by_smart_ptr();
        }

        return false;
    }

    static const std::shared_ptr<void> &extract_shared_ptr(const py::object &h)
    {
        if (py::isinstance<dpctl::memory::usm_memory>(h)) {
            const auto &usm_memory_inst =
                py::cast<dpctl::memory::usm_memory>(h);
            return usm_memory_inst.get_smart_ptr_owner();
        }
        else if (py::isinstance<tensor::usm_ndarray>(h)) {
            const auto &usm_array_inst = py::cast<tensor::usm_ndarray>(h);
            return usm_array_inst.get_smart_ptr_owner();
        }

        throw std::runtime_error(
            "Attempted extraction of shared_ptr on an unrecognized type");
    }
};
} // end of namespace detail

template <std::size_t num>
sycl::event keep_args_alive(sycl::queue &q,
                            const py::object (&py_objs)[num],
                            const std::vector<sycl::event> &depends = {})
{
    std::size_t n_objects_held = 0;
    std::array<std::shared_ptr<py::handle>, num> shp_arr{};

    std::size_t n_usm_owners_held = 0;
    std::array<std::shared_ptr<void>, num> shp_usm{};

    for (std::size_t i = 0; i < num; ++i) {
        const auto &py_obj_i = py_objs[i];
        if (detail::ManagedMemory::is_usm_managed_by_shared_ptr(py_obj_i)) {
            const auto &shp =
                detail::ManagedMemory::extract_shared_ptr(py_obj_i);
            shp_usm[n_usm_owners_held] = shp;
            ++n_usm_owners_held;
        }
        else {
            shp_arr[n_objects_held] = std::make_shared<py::handle>(py_obj_i);
            shp_arr[n_objects_held]->inc_ref();
            ++n_objects_held;
        }
    }

    bool use_depends = true;
    sycl::event host_task_ev;

    if (n_usm_owners_held > 0) {
        host_task_ev = q.submit([&](sycl::handler &cgh) {
            if (use_depends) {
                cgh.depends_on(depends);
                use_depends = false;
            }
            else {
                cgh.depends_on(host_task_ev);
            }
            cgh.host_task([shp_usm = std::move(shp_usm)]() {
                // no body, but shared pointers are captured in
                // the lambda, ensuring that USM allocation is
                // kept alive
            });
        });
    }

    if (n_objects_held > 0) {
        host_task_ev = q.submit([&](sycl::handler &cgh) {
            if (use_depends) {
                cgh.depends_on(depends);
                use_depends = false;
            }
            else {
                cgh.depends_on(host_task_ev);
            }
            cgh.host_task([n_objects_held, shp_arr = std::move(shp_arr)]() {
                py::gil_scoped_acquire acquire;

                for (std::size_t i = 0; i < n_objects_held; ++i) {
                    shp_arr[i]->dec_ref();
                }
            });
        });
    }

    return host_task_ev;
}

// add to namespace for convenience
using ::dpctl::utils::queues_are_compatible;

/*! @brief Check if all allocation queues of usm_ndarrays are the same as
    the execution queue */
template <std::size_t num>
bool queues_are_compatible(const sycl::queue &exec_q,
                           const tensor::usm_ndarray (&arrs)[num])
{
    for (std::size_t i = 0; i < num; ++i) {

        if (exec_q != arrs[i].get_queue()) {
            return false;
        }
    }
    return true;
}
} // end namespace utils
} // end namespace dpnp
