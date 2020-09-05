#include <iostream>

#include "backend_iface.hpp"

int main(int, char**)
{
    const size_t size = 256;

    dpnp_queue_initialize_c(QueueOptions::CPU_SELECTOR);

    int* array1 = (int*)dpnp_memory_alloc_c(size * sizeof(int));
    double* result = (double*)dpnp_memory_alloc_c(size * sizeof(double));

    for (size_t i = 0; i < 10; ++i)
    {
        array1[i] = i;
        result[i] = 0;
        std::cout << ", " << array1[i];
    }
    std::cout << std::endl;

    custom_elemwise_cos_c<int, double>(array1, result, size);

    for (size_t i = 0; i < 10; ++i)
    {
        std::cout << ", " << result[i];
    }
    std::cout << std::endl;

    dpnp_memory_free_c(result);
    dpnp_memory_free_c(array1);

    return 0;
}
