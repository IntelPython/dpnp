// clang++ -g -fPIC examples/example8.cpp -Idpnp -Idpnp/backend -Ldpnp -Wl,-rpath='$ORIGIN'/dpnp -ldpnp_backend_c -o example8
#include <iostream>

#include "backend_iface.hpp"

int main(int, char**)
{
    const size_t size = 16;

    dpnp_queue_initialize_c(QueueOptions::GPU_SELECTOR);

    double* array = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    int* result = (int*)dpnp_memory_alloc_c(size * sizeof(int));

    std::cout << "array" << std::endl;
    for (size_t i = 0; i < size; ++i)
    {
        array[i] = (double)(size - i) / 2;
        std::cout << array[i] << ", ";
    }
    std::cout << std::endl;

    custom_argsort_c<double, int>(array, result, size);

    std::cout << "array with 'sorted' indeces" << std::endl;
    for (size_t i = 0; i < size; ++i)
    {
        std::cout << result[i] << ", ";
    }
    std::cout << std::endl;

    dpnp_memory_free_c(result);
    dpnp_memory_free_c(array);

    return 0;
}
