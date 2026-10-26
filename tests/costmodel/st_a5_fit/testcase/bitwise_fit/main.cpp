/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>

#include <pto/costmodel/lightweight_costmodel.hpp>

#include "a5_cost_check.hpp"

namespace {

using namespace ::pto::mocker::lightweight;

struct BitwiseFitCase {
    PtoOpcode op;
    const char* name;
    DType dtype;
    int64_t rows;
    int64_t cols;
    double profilingCycles;
};

constexpr BitwiseFitCase kCases[] = {
    {PtoOpcode::TAND, "TAND", DType::Int16, 16, 16, 114.0},
    {PtoOpcode::TAND, "TAND", DType::Int16, 16, 256, 130.0},
    {PtoOpcode::TAND, "TAND", DType::Int32, 64, 128, 466.0},
    {PtoOpcode::TOR, "TOR", DType::Int16, 128, 128, 786.0},
    {PtoOpcode::TOR, "TOR", DType::Int32, 16, 256, 162.0},
    {PtoOpcode::TNOT, "TNOT", DType::Int16, 32, 32, 210.0},
    {PtoOpcode::TNOT, "TNOT", DType::Int16, 64, 128, 402.0},
    {PtoOpcode::TANDS, "TANDS", DType::Int16, 1, 1024, 43.0},
    {PtoOpcode::TANDS, "TANDS", DType::Int16, 24, 256, 186.0},
    {PtoOpcode::TORS, "TORS", DType::Int16, 1, 16, 36.0},
    {PtoOpcode::TORS, "TORS", DType::Int16, 48, 64, 306.0},
    {PtoOpcode::TXOR, "TXOR", DType::Int16, 2, 64, 39.0},
    {PtoOpcode::TXOR, "TXOR", DType::Int16, 64, 128, 402.0},
    {PtoOpcode::TXORS, "TXORS", DType::Int16, 16, 128, 114.0},
    {PtoOpcode::TXORS, "TXORS", DType::Int16, 24, 256, 186.0},
};

TEST(A5BitwiseFit, HardwareProfiles)
{
    constexpr double kRequiredPrecision = 0.90;
    for (const auto& testCase : kCases) {
        CostModelInput input{
            .op = testCase.op,
            .dtype = testCase.dtype,
            .rows = testCase.rows,
            .cols = testCase.cols,
            .arch = CostModelArch::A5,
            .valid_rows = testCase.rows,
            .valid_cols = testCase.cols,
        };
        pto::test::ExpectA5CycleNear(testCase.name, input, testCase.profilingCycles, kRequiredPrecision);
    }
}

} // namespace
