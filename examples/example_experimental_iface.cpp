#include <iostream>

#include "backend_iface.hpp"

int main(int, char**)
{
    void* result = get_backend_function_name("dpnp_dot", "float");
    std::cout << "Result Dot() function pointer: " << result << std::endl;

    return 0;
}
