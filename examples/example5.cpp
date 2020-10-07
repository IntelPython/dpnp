#include <iostream>

#include <backend_iface.hpp>

void print_dpnp_array(double* arr, size_t size)
{
    std::cout << std::endl;
    for (size_t i = 0; i < size; ++i)
    {
        std::cout << arr[i] << ", ";
    }
    std::cout << std::endl;
}

int main(int, char**)
{
    const size_t size = 256;

    dpnp_queue_initialize_c(QueueOptions::CPU_SELECTOR);

    double* result = (double*)dpnp_memory_alloc_c(size * sizeof(double));

    size_t seed = 10;
    long low = 1;
    long high = 120;

    std::cout << "Uniform distr. params:\nlow is " << low << ", high is " << high << std::endl;

    std::cout << "Results, when seed is the same (10) for all random number generations:";
    for (size_t i = 0; i < 4; ++i)
    {
        dpnp_engine_rng_initialize(seed);
        mkl_rng_uniform<double>(result, low, high, size);
        print_dpnp_array(result, 10);
    }

    std::cout << std::endl << "Results, when seed is random:";
    dpnp_engine_rng_initialize();
    for (size_t i = 0; i < 4; ++i)
    {
        mkl_rng_uniform<double>(result, low, high, size);
        print_dpnp_array(result, 10);
    }

    dpnp_memory_free_c(result);

    return 0;
}
