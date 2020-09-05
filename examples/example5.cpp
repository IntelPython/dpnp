#include <iostream>

#include <backend_iface.hpp>

int main(int, char**)
{
    const size_t size = 256;

    dpnp_queue_initialize_c(QueueOptions::CPU_SELECTOR);

    double* result = (double*)dpnp_memory_alloc_c(size * sizeof(double));

    //for (size_t i = 0; i < 10; ++i)
    //{
    //    result[i] = 0;
    //}
    //std::cout << std::endl;
    mkl_rng_gaussian<double>(result, size);

    for (size_t i = 0; i < 10; ++i)
    {
        std::cout << result[i] << ", ";
    }
    std::cout << std::endl;

    dpnp_memory_free_c(result);

    return 0;
}
