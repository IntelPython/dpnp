// clang++ -g -fPIC examples/example7.cpp -Idpnp -Idpnp/backend -Ldpnp -Wl,-rpath='$ORIGIN'/dpnp -ldpnp_backend_c -o example7
// ./example7
#include <iostream>

#include "backend_iface.hpp"

int main(int, char**)
{
    const size_t size = 2;
    size_t len = size * size;

    dpnp_queue_initialize_c(QueueOptions::CPU_SELECTOR);

    double* array = (double*)dpnp_memory_alloc_c(len * sizeof(double));
    double* result = (double*)dpnp_memory_alloc_c(size * sizeof(double));

    /* init input diagonal array like:
    1, 0, 0,
    0, 2, 0,
    0, 0, 3
    */
    for (size_t i = 0; i < len; ++i)
    {
        array[i] = 0;
    }
    for (size_t i = 0; i < size; ++i)
    {
        array[size * i + i] = i + 1;
    }

    mkl_lapack_syevd_c<double>(array, result, size);

    std::cout << "eigen values" << std::endl;
    for (size_t i = 0; i < size; ++i)
    {
        std::cout << result[i] << ", ";
    }
    std::cout << std::endl;

    dpnp_memory_free_c(result);
    dpnp_memory_free_c(array);

    return 0;
}
