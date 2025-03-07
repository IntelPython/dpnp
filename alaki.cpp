#include <iostream>

int main()
{

    const double num = 65.78f;
    const char *bytes = reinterpret_cast<const char *>(&num);

    for (size_t i = 0; i < sizeof(num); ++i) {
        std::cout << static_cast<float>(bytes[i]) << " ";
    }
    std::cout << std::endl;

    double restoredNum = *reinterpret_cast<const double *>(bytes);

    std::cout << "restoredNum:" << restoredNum << std::endl;

    return 0;
}
