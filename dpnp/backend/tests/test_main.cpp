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

#include "gtest/gtest.h"

#include <dpnp_iface.hpp>

// TODO add namespace
// will added for test_commons
class DPNPCTestEnvironment : public testing::Environment
{
public:
    void SetUp() override
    {
        // TODO update print
        std::cout << "starting new env" << std::endl << std::endl;
    }
    void TearDown() override {}
};

int RunAllTests(DPNPCTestEnvironment *env)
{
    // testing::internal::GetUnitTestImpl()->ClearAdHocTestResult();
    (void)env;

    // It returns 0 if all tests are successful, or 1 otherwise.
    return RUN_ALL_TESTS();
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // currently using global queue

    // It returns 0 if all tests are successful, or 1 otherwise.
    return RUN_ALL_TESTS();
}
