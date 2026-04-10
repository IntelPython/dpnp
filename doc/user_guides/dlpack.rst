.. _dpnp_tensor_dlpack_support:

DLPack exchange of USM allocated arrays
=======================================

DLPack overview
---------------

`DLPack <dlpack_docs_>`_ is a commonly used C-ABI compatible data structure that allows data exchange
between major frameworks. DLPack strives to be minimal, intentionally leaves allocators API and
device API out of scope.

Data shared via DLPack are owned by the producer who provides a deleter function stored in the
`DLManagedTensor <dlpack_managed_tensor_>`_, and are only accessed by consumer.
Python semantics of using the structure is `explained in dlpack docs <dlpack_python_spec_>`_.

DLPack specifies data location in memory via ``void * data`` field of `DLTensor <dlpack_dltensor_>`_ struct, and via ``DLDevice device`` field.
The `DLDevice <dlpack_dldevice_>`_ struct has two members: an enumeration ``device_type`` and an integer ``device_id``.

DLPack recognizes enumeration value ``DLDeviceType::kDLOneAPI`` reserved for sharing SYCL USM allocations.
It is not ``kDLSycl`` since importing USM-allocated tensor with this device type relies on oneAPI SYCL extensions
``sycl_ext_oneapi_filter_selector`` and ``sycl_ext_oneapi_default_platform_context`` to operate.

.. _dlpack_docs: https://dmlc.github.io/dlpack/latest/
.. _dlpack_managed_tensor: https://dmlc.github.io/dlpack/latest/c_api.html#c.DLManagedTensor
.. _dlpack_dltensor: https://dmlc.github.io/dlpack/latest/c_api.html#c.DLTensor
.. _dlpack_dldevice: https://dmlc.github.io/dlpack/latest/c_api.html#c.DLDevice
.. _dlpack_python_spec: https://dmlc.github.io/dlpack/latest/python_spec.html

Exporting USM allocation to DLPack
-----------------------------------

When sharing USM allocation (of any ``sycl::usm::kind``) with ``void * ptr`` bound to ``sycl::context ctx``:

.. code-block:: cpp
    :caption: Protocol for exporting USM allocation as DLPack

    // Input: void *ptr:
    //             USM allocation pointer
    //        sycl::context ctx:
    //             context the pointer is bound to

    // Get device where allocation was originally made
    // Keep in mind, the device may be a sub-device
    const sycl::device &ptr_dev = sycl::get_pointer_device(ptr, ctx);

    #if SYCL_KHR_DEFAULT_CONTEXT
    const sycl::context &default_ctx = ptr_dev.get_platform().khr_get_default_context();
    #else
    static_assert(false, "ext_oneapi_default_context extension is required");
    #endif

    // Assert that ctx is the default platform context, or throw
    if (ctx != default_ctx) {
        throw pybind11::type_error(
            "Can not export USM allocations not "
            "bound to default platform context."
        );
    }

    // Find parent root device if ptr_dev is a sub-device
    const sycl::device &parent_root_device = get_parent_root_device(ptr_dev);

    // find position of parent_root_device in sycl::get_devices
    const auto &all_root_devs = sycl::device::get_devices();
    auto beg = std::begin(all_root_devs);
    auto end = std::end(all_root_devs);
    auto selectot_fn = [parent_root_device](const sycl::device &root_d) -> bool {
        return parent_root_device == root_d;
    };
    auto pos = find_if(beg, end, selector_fn);

    if (pos == end) {
        throw pybind11::type_error("Could not produce DLPack: failed finding device_id");
    }
    std::ptrdiff_t dev_idx = std::distance(beg, pos);

    // check that dev_idx can fit into int32_t if needed
    int32_t device_id = static_cast<int32_t>(dev_idx);

    // populate DLTensor with DLDeviceType::kDLOneAPI and computed device_id


Importing DLPack with ``device_type == kDLOneAPI``
--------------------------------------------------

.. code-block:: cpp
    :caption: Protocol for recognizing DLPack as a valid USM allocation

    // Input: ptr = dlm_tensor->dl_tensor.data
    //        device_id = dlm_tensor->dl_tensor.device.device_id

    // Get root_device from device_id
    const auto &device_vector = sycl::get_device();
    const sycl::device &root_device = device_vector.at(device_id);

    // Check if the backend of the device is supported by consumer
    //    Perhaps for certain backends (CUDA, hip, etc.) we should dispatch
    //    different dlpack importers

    // alternatively
    // sycl::device root_device = sycl::device(
    //       sycl::ext::oneapi::filter_selector{ std::to_string(device_id)}
    // );

    // Get default platform context
    #if SYCL_KHR_DEFAULT_CONTEXT
    const sycl::context &default_ctx = root_device.get_platform().khr_get_default_context();
    #else
    static_assert(false, "ext_oneapi_default_context extension is required");
    #endif

    // Check that pointer is known in the context
    const sycl::usm::kind &alloc_type = sycl::get_pointer_type(ptr, ctx);

    if (alloc_type == sycl::usm::kind::unknown) {
        throw pybind11::type_error(
            "Data pointer in DLPack is not bound to the "
            "default platform context of specified device"
        );
    }

    // Perform check that USM allocation type is supported by consumer if needed

    // Get sycl::device where the data was allocated
    const sycl::device &ptr_dev = sycl::get_pointer_device(ptr, ctx);

    // Create object of consumer's library from ptr, ptr_dev, ctx

Support of DLPack with ``kDLOneAPI`` device type
------------------------------------------------

:py:mod:`dpnp.tensor` supports DLPack v0.8. Exchange of USM allocations made using Level-Zero backend
is supported with ``torch.Tensor(device='xpu')`` for PyTorch when using `intel-extension-for-pytorch <intel_ext_for_torch_>`_,
as well as for TensorFlow when `intel-extension-for-tensorflow <intel_ext_for_tf_>`_ is used.

.. _intel_ext_for_torch: https://github.com/intel/intel-extension-for-pytorch
.. _intel_ext_for_tf: https://github.com/intel/intel-extension-for-tensorflow
