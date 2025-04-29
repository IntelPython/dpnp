#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

using namespace sycl;

int main() {
    constexpr size_t size = 1024;
    constexpr int num_iters = 5000;

    for (int iter = 0; iter < num_iters; ++iter) {
        queue q{default_selector{}};

        if (iter % 100 == 0) {
            std::cout << "Using device: " << q.get_device().get_info<info::device::name>()
                      << " | Iteration: " << iter << "\n";
        }

        float* device_data = malloc_device<float>(size, q);

        std::vector<float> host_input(size, 1.0f);
        std::vector<float> host_output1(size, 0.0f);
        std::vector<float> host_output2(size, 0.0f);


        q.memcpy(device_data, host_input.data(), size * sizeof(float)).wait();

        q.memcpy(host_output1.data(), device_data, size * sizeof(float)).wait();
        q.memcpy(host_output2.data(), device_data, size * sizeof(float)).wait();

        for (size_t i = 0; i < size; ++i) {
            if (host_output1[i] != 1.0f || host_output2[i] != 1.0f) {
                std::cerr << "Mismatch at index " << i << " in iteration " << iter << "\n";
                break;
            }
        }

        free(device_data, q);
    }

    return 0;
}



#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>

using namespace sycl;

int main() {
    constexpr size_t size = 1024;
    constexpr int num_iters = 5000;

    for (int iter = 0; iter < num_iters; ++iter) {
        queue q{default_selector{}};

        float* device_data = malloc_device<float>(size, q);

        std::vector<float> host_input(size, 1.0f);
        std::vector<float> host_output1(size, 0.0f);
        std::vector<float> host_output2(size, 0.0f);

        event e1 = q.memcpy(device_data, host_input.data(), size * sizeof(float));

        event e2 = q.memcpy(host_output1.data(), device_data, size * sizeof(float), {e1});

        event kernel_event = q.submit([&](handler& h) {
            h.depends_on(e2);
            h.single_task<class empty_kernel>([]() {});
        });

        event e3 = q.memcpy(host_output2.data(), device_data, size * sizeof(float), {kernel_event});

        e3.wait();

        for (size_t i = 0; i < size; ++i) {
            if (host_output1[i] != 1.0f || host_output2[i] != 1.0f) {
                std::cerr << "Mismatch at index " << i << " in iteration " << iter << "\n";
                break;
            }
        }

        if (iter % 100 == 0) {
            std::cout << "Iteration " << iter << " completed\n";
        }

        free(device_data, q);
    }

    return 0;
}

// dpcpp -fsycl -I"%CONDA_PREFIX%\include" -o repro.exe .\rep.cpp



#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>

using namespace sycl;

int main() {
    constexpr size_t size = 1024;
    constexpr int num_iters = 5000;

    for (int iter = 0; iter < num_iters; ++iter) {
        queue q{default_selector{}};

        float* device_data = malloc_device<float>(size, q);
        int* check_flags = malloc_device<int>(size, q);

        std::vector<float> host_input(size, 1.0f);
        std::vector<float> host_output1(size, 0.0f);
        std::vector<float> host_output2(size, 0.0f);

        event e1 = q.memcpy(device_data, host_input.data(), size * sizeof(float));

        event e2 = q.memcpy(host_output1.data(), device_data, size * sizeof(float), {e1});

        event kernel_event = q.submit([&](handler& h) {
            h.depends_on(e2);
            h.parallel_for<class compare_kernel>(range<1>(size), [=](id<1> i) {
                check_flags[i] = (device_data[i] == 1.0f) ? 1 : 0;
            });
        });

        event e3 = q.memcpy(host_output2.data(), device_data, size * sizeof(float), {kernel_event});

        e3.wait();

        for (size_t i = 0; i < size; ++i) {
            if (host_output1[i] != 1.0f || host_output2[i] != 1.0f) {
                std::cerr << "Mismatch at index " << i << " in iteration " << iter << "\n";
                break;
            }
        }

        if (iter % 100 == 0) {
            std::cout << "Iteration " << iter << " completed\n";
        }

        free(device_data, q);
        free(check_flags, q);
    }

    return 0;
}



#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>

using namespace sycl;

int main() {
    constexpr size_t size = 1024;
    constexpr int num_iters = 5000;

    for (int iter = 0; iter < num_iters; ++iter) {
        try {
            queue q{default_selector{}};

            device dev = q.get_device();
            if (!dev.has(aspect::usm_device_allocations)) {
                std::cerr << "Device does not support USM device allocations. Iteration: " << iter << std::endl;
                break;
            }

            float* device_data = malloc_device<float>(size, q);
            int* check_flags = malloc_device<int>(size, q);

            if (!device_data || !check_flags) {
                std::cerr << "Memory allocation failed on iteration " << iter << std::endl;
                break;
            }

            std::vector<float> host_input(size, 1.0f);
            std::vector<float> host_output1(size, 0.0f);
            std::vector<float> host_output2(size, 0.0f);

            event e1 = q.memcpy(device_data, host_input.data(), size * sizeof(float));

            event e2 = q.memcpy(host_output1.data(), device_data, size * sizeof(float), {e1});

            event kernel_event = q.submit([&](handler& h) {
                h.depends_on(e2);
                h.parallel_for<class compare_kernel>(range<1>(size), [=](id<1> i) {
                    check_flags[i] = (device_data[i] == 1.0f) ? 1 : 0;
                });
            });

            event e3 = q.memcpy(host_output2.data(), device_data, size * sizeof(float), {kernel_event});

            e3.wait();
            q.wait_and_throw();

            for (size_t i = 0; i < size; ++i) {
                if (host_output1[i] != 1.0f || host_output2[i] != 1.0f) {
                    std::cerr << "Mismatch at index " << i << " in iteration " << iter << std::endl;
                    break;
                }
            }

            if (iter % 100 == 0) {
                std::cout << "Iteration " << iter << " completed" << std::endl;
            }

            free(device_data, q);
            free(check_flags, q);
        }
        catch (sycl::exception const& e) {
            std::cerr << "SYCL exception on iteration " << iter << ": " << e.what() << std::endl;
            break;
        }
        catch (std::exception const& e) {
            std::cerr << "Standard exception on iteration " << iter << ": " << e.what() << std::endl;
            break;
        }
        catch (...) {
            std::cerr << "Unknown error on iteration " << iter << std::endl;
            break;
        }
    }

    return 0;
}
