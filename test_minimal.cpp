// Minimal reproducer
//
// Build: icpx -fsycl --gcc-install-dir=$CONDA_PREFIX/lib/gcc/x86_64-conda-linux-gnu/14.3.0 --sysroot=$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot test_minimal.cpp -o test_minimal
// Run: ./test_minimal

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <iomanip>

using namespace sycl;

// Print detailed device information
void print_device_info(const device& dev) {
    std::cout << "========================================" << std::endl;
    std::cout << "DEVICE INFORMATION" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    std::cout << "Device name:              " << dev.get_info<info::device::name>() << std::endl;
    std::cout << "Vendor:                   " << dev.get_info<info::device::vendor>() << std::endl;
    std::cout << "Driver version:           " << dev.get_info<info::device::driver_version>() << std::endl;
    std::cout << "Device version:           " << dev.get_info<info::device::version>() << std::endl;

    std::cout << std::endl;
    std::cout << "Device type:              ";
    if (dev.is_cpu()) std::cout << "CPU";
    else if (dev.is_gpu()) std::cout << "GPU";
    else if (dev.is_accelerator()) std::cout << "Accelerator";
    else std::cout << "Unknown";
    std::cout << std::endl;

    std::cout << std::endl;
    std::cout << "Max compute units:        " << dev.get_info<info::device::max_compute_units>() << std::endl;
    std::cout << "Max work group size:      " << dev.get_info<info::device::max_work_group_size>() << std::endl;
    std::cout << "Max work item dimensions: " << dev.get_info<info::device::max_work_item_dimensions>() << std::endl;

    auto max_work_item_sizes = dev.get_info<info::device::max_work_item_sizes<3>>();
    std::cout << "Max work item sizes:      ["
              << max_work_item_sizes[0] << ", "
              << max_work_item_sizes[1] << ", "
              << max_work_item_sizes[2] << "]" << std::endl;

    std::cout << std::endl;
    std::cout << "Global mem size:          "
              << (dev.get_info<info::device::global_mem_size>() / (1024*1024)) << " MB" << std::endl;
    std::cout << "Local mem size:           "
              << (dev.get_info<info::device::local_mem_size>() / 1024) << " KB" << std::endl;
    std::cout << "Max mem alloc size:       "
              << (dev.get_info<info::device::max_mem_alloc_size>() / (1024*1024)) << " MB" << std::endl;

    std::cout << std::endl;
    std::cout << "Supports USM device:      "
              << (dev.has(aspect::usm_device_allocations) ? "YES" : "NO") << std::endl;
    std::cout << "Supports USM host:        "
              << (dev.has(aspect::usm_host_allocations) ? "YES" : "NO") << std::endl;
    std::cout << "Supports USM shared:      "
              << (dev.has(aspect::usm_shared_allocations) ? "YES" : "NO") << std::endl;

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
}

// Kernel with backward dimension writes
template <typename cumsumT, typename indexT>
class NonzeroIndexKernel;

template <typename cumsumT, typename indexT>
sycl::event extract_nonzero_indices(
    queue &q,
    size_t n_elems,
    size_t nz_count,
    int ndim,
    const cumsumT* cumsum_data,
    indexT* indices_data,
    const size_t* shape
)
{
    constexpr size_t lws = 256;
    const size_t n_groups = (n_elems + lws - 1) / lws;

    return q.submit([&](handler &cgh) {
        local_accessor<cumsumT, 1> local_cumsum(lws + 1, cgh);

        cgh.parallel_for<NonzeroIndexKernel<cumsumT, indexT>>(
            nd_range<1>(n_groups * lws, lws),
            [=](nd_item<1> ndit) {
                const size_t gid = ndit.get_global_id(0);
                const size_t lid = ndit.get_local_id(0);
                const size_t group_id = ndit.get_group(0);
                const size_t group_start = group_id * lws;

                // Load cumsum with halo
                if (lid == 0) {
                    local_cumsum[0] = (group_start == 0) ? 0 : cumsum_data[group_start - 1];
                }
                if (group_start + lid < n_elems) {
                    local_cumsum[lid + 1] = cumsum_data[group_start + lid];
                }

                group_barrier(ndit.get_group());

                if (gid < n_elems) {
                    bool is_nonzero = (local_cumsum[lid + 1] != local_cumsum[lid]);

                    if (is_nonzero) {
                        cumsumT output_pos = local_cumsum[lid + 1] - 1;
                        size_t flat_idx = gid;

                        for (int dim = ndim - 1; dim >= 0; dim--) {
                            indices_data[output_pos * ndim + dim] = flat_idx % shape[dim];
                            flat_idx /= shape[dim];
                        }
                    }
                }
            }
        );
    });
}

int main() {
    queue q;
    int64_t *cumsum_device = nullptr;
    size_t *indices_device = nullptr;
    size_t *shape_device = nullptr;
    size_t *indices_host = nullptr;

    try {
        q = queue(default_selector_v);

        auto device = q.get_device();
        print_device_info(device);

        std::cout << "========================================" << std::endl;
        std::cout << "TEST CONFIGURATION" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << std::endl;

        // Test parameters
        const size_t n_elems = 6;
        const int ndim = 2;
        const size_t nz_count = 3;
        const std::vector<size_t> shape = {2, 3};

        std::cout << "Input array (flat):  [1, 0, 0, 4, 0, 6]" << std::endl;
        std::cout << "Input array (2D):    [[1, 0, 0]," << std::endl;
        std::cout << "                      [4, 0, 6]]" << std::endl;
        std::cout << "Shape:               [" << shape[0] << ", " << shape[1] << "]" << std::endl;
        std::cout << std::endl;

        std::cout << "Cumsum (precomputed): [1, 1, 1, 2, 2, 3]" << std::endl;
        std::cout << "Nonzero elements:     3" << std::endl;
        std::cout << "Nonzero positions:" << std::endl;
        std::cout << "  gid=0 → output[0] → row=0, col=0" << std::endl;
        std::cout << "  gid=3 → output[1] → row=1, col=0" << std::endl;
        std::cout << "  gid=5 → output[2] → row=1, col=2" << std::endl;
        std::cout << std::endl;

        std::cout << "Kernel configuration:" << std::endl;
        std::cout << "  Work group size:     256" << std::endl;
        std::cout << "  Number of groups:    1" << std::endl;
        std::cout << "  Total work items:    256" << std::endl;
        std::cout << "  Active work items:   6 (processing 6 elements)" << std::endl;
        std::cout << "  Local memory:        (256 + 1) * 8 bytes = 2056 bytes" << std::endl;
        std::cout << std::endl;

        std::cout << "========================================" << std::endl;
        std::cout << std::endl;

        // Hardcoded cumsum values for input [[1, 0, 0], [4, 0, 6]]
        int64_t cumsum_values[] = {1, 1, 1, 2, 2, 3};

        // Allocate device memory
        cumsum_device = malloc_device<int64_t>(n_elems, q);
        indices_device = malloc_device<size_t>(nz_count * ndim, q);
        shape_device = malloc_device<size_t>(ndim, q);

        if (!cumsum_device || !indices_device || !shape_device) {
            throw std::runtime_error("Device allocation failed");
        }

        // Copy data to device
        q.copy<int64_t>(cumsum_values, cumsum_device, n_elems).wait();
        q.copy<size_t>(shape.data(), shape_device, ndim).wait();

        std::cout << "Running kernel..." << std::endl;
        std::cout << "(writes dim 1 first, then dim 0)" << std::endl;
        std::cout << std::endl;

        // Run the kernel
        auto kernel_ev = extract_nonzero_indices<int64_t, size_t>(
            q, n_elems, nz_count, ndim,
            cumsum_device, indices_device, shape_device
        );
        kernel_ev.wait();

        // Read results
        indices_host = malloc_host<size_t>(nz_count * ndim, q);
        if (!indices_host) {
            throw std::runtime_error("Host allocation failed");
        }
        q.copy<size_t>(indices_device, indices_host, nz_count * ndim).wait();

        std::cout << "========================================" << std::endl;
        std::cout << "RESULTS" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << std::endl;

        // Print raw packed output
        std::cout << "Raw packed output: [";
        for (size_t i = 0; i < nz_count * ndim; i++) {
            std::cout << indices_host[i];
            if (i < nz_count * ndim - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "Expected output:   [0, 0, 1, 0, 1, 2]" << std::endl;
        std::cout << "Format:            [row0, col0, row1, col1, row2, col2]" << std::endl;
        std::cout << std::endl;

        // Unpack
        std::vector<size_t> rows(nz_count), cols(nz_count);
        for (size_t i = 0; i < nz_count; i++) {
            rows[i] = indices_host[i * ndim + 0];
            cols[i] = indices_host[i * ndim + 1];
        }

        std::cout << "Row indices:       [";
        for (auto v : rows) std::cout << v << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected rows:     [0 1 1]" << std::endl;
        std::cout << std::endl;

        std::cout << "Col indices:       [";
        for (auto v : cols) std::cout << v << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected cols:     [0 0 2]" << std::endl;
        std::cout << std::endl;

        // Verify
        std::vector<size_t> expected_rows = {0, 1, 1};
        std::vector<size_t> expected_cols = {0, 0, 2};
        bool correct = (rows == expected_rows) && (cols == expected_cols);

        std::cout << "========================================" << std::endl;
        if (correct) {
            std::cout << "✓ Test PASSED!" << std::endl;
            return 0;
        } else {
            std::cout << "✗ Test FAILED!" << std::endl;
            std::cout << std::endl;
            std::cout << "Analysis:" << std::endl;

            // Detailed analysis
            bool rows_match = (rows == expected_rows);
            bool cols_match = (cols == expected_cols);

            if (!rows_match) {
                std::cout << "  - Row indices are WRONG" << std::endl;
                std::cout << "    Expected: [0 1 1]" << std::endl;
                std::cout << "    Got:      [";
                for (auto v : rows) std::cout << v << " ";
                std::cout << "]" << std::endl;
            } else {
                std::cout << "  - Row indices are correct" << std::endl;
            }

            if (!cols_match) {
                std::cout << "  - Column indices are WRONG" << std::endl;
                std::cout << "    Expected: [0 0 2]" << std::endl;
                std::cout << "    Got:      [";
                for (auto v : cols) std::cout << v << " ";
                std::cout << "]" << std::endl;
            } else {
                std::cout << "  - Column indices are correct" << std::endl;
            }

            std::cout << std::endl;

            // Cleanup
            if (cumsum_device) free(cumsum_device, q);
            if (indices_device) free(indices_device, q);
            if (shape_device) free(shape_device, q);
            if (indices_host) free(indices_host, q);

            return 1;
        }

        // Cleanup
        if (cumsum_device) free(cumsum_device, q);
        if (indices_device) free(indices_device, q);
        if (shape_device) free(shape_device, q);
        if (indices_host) free(indices_host, q);

        return 0;

    } catch (exception const& e) {
        std::cerr << std::endl;
        std::cerr << "========================================" << std::endl;
        std::cerr << "SYCL EXCEPTION" << std::endl;
        std::cerr << "========================================" << std::endl;
        std::cerr << e.what() << std::endl;

        // Cleanup on error
        if (cumsum_device) free(cumsum_device, q);
        if (indices_device) free(indices_device, q);
        if (shape_device) free(shape_device, q);
        if (indices_host) free(indices_host, q);

        return 1;
    } catch (std::exception const& e) {
        std::cerr << std::endl;
        std::cerr << "========================================" << std::endl;
        std::cerr << "STANDARD EXCEPTION" << std::endl;
        std::cerr << "========================================" << std::endl;
        std::cerr << e.what() << std::endl;

        // Note: Can't cleanup here as we don't have queue reference
        return 1;
    }
}
