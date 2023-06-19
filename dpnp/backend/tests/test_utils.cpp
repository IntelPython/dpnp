//*****************************************************************************
// Copyright (c) 2016-2023, Intel Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
// THE POSSIBILITY OF SUCH DAMAGE.
//*****************************************************************************

#include <gtest/gtest.h>
#include <vector>

#include "dpnp_utils.hpp"

using namespace std;

struct AxesParameters
{
    vector<long> axes;
    size_t shape_size = size_t{};
    bool duplications = bool{};
    vector<size_t> result;

    /// Operator needs to print this container in human readable form in error reporting
    friend std::ostream& operator<<(std::ostream& out, const AxesParameters& data)
    {
        out << "AxesParameters(axes=" << data.axes << ", shape_size=" << data.shape_size
            << ", duplications=" << data.duplications << ", result=" << data.result << ")";

        return out;
    }
};

struct AxesNormalization : public ::testing::TestWithParam<AxesParameters>
{
    // need to change test name string
    struct PrintToStringParamName
    {
        template <class ParamType>
        string operator()(const testing::TestParamInfo<ParamType>& info) const
        {
            const AxesParameters& param = static_cast<AxesParameters>(info.param);
            stringstream ss;
            ss << "axes=" << param.axes << ", shape_size=" << param.shape_size
               << ", duplications=" << param.duplications << ", result=" << param.result;
            return ss.str();
        }
    };
};

TEST_P(AxesNormalization, get_validated_axes)
{
    const AxesParameters& param = GetParam();
    vector<size_t> result = get_validated_axes(param.axes, param.shape_size, param.duplications);
    EXPECT_EQ(result, param.result);
}

INSTANTIATE_TEST_SUITE_P(TestUtilsAxesNormalization,
                         AxesNormalization,
                         testing::Values(AxesParameters{{0}, 1, true, {0}},
                                         AxesParameters{{1}, 4, false, {1}},
                                         AxesParameters{{-1}, 4, false, {3}},
                                         AxesParameters{{0, 1, 2, 3}, 4, false, {0, 1, 2, 3}},
                                         /* AxesParameters{{0, 1, 1, 3}, 4, false, {0, 1, 3}}, */
                                         AxesParameters{{0, 1, 1, 3}, 4, true, {0, 1, 1, 3}},
                                         AxesParameters{{-2, 1, -4, 3}, 4, false, {2, 1, 0, 3}},
                                         AxesParameters{{-4, -3, -2, -1}, 4, false, {0, 1, 2, 3}},
                                         AxesParameters{{-1, -2, -3, -4}, 4, false, {3, 2, 1, 0}},
                                         AxesParameters{{}, 0, true, {}},
                                         AxesParameters{{}, 0, false, {}},
                                         AxesParameters{{}, 2, true, {}})
                         /*, AxesNormalization::PrintToStringParamName()*/
);
