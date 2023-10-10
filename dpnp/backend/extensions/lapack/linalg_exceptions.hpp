#pragma once
#include <cstring>
#include <stdexcept>

namespace dpnp
{
namespace backend
{
namespace ext
{
namespace lapack
{
class LinAlgError : public std::exception
{
public:
    explicit LinAlgError(const char *message) : msg_(message) {}

    const char *what() const noexcept override
    {
        return msg_.c_str();
    }

private:
    std::string msg_;
};
} // namespace lapack
} // namespace ext
} // namespace backend
} // namespace dpnp
