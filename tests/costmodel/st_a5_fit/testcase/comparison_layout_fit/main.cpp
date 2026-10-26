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

struct VfFitCase {
    PtoOpcode op;
    const char* name;
    int64_t rows;
    int64_t cols;
    double profilingCycles;
};

constexpr VfFitCase kCases[] = {
    {PtoOpcode::TCMP, "TCMP", 8, 128, 66.0},          {PtoOpcode::TCMPS, "TCMPS", 16, 128, 51.0},
    {PtoOpcode::TSELS, "TSELS", 8, 128, 66.0},        {PtoOpcode::TCONCAT, "TCONCAT", 128, 128, 3864.0},
    {PtoOpcode::TGATHER, "TGATHER", 16, 128, 136.0},  {PtoOpcode::TGATHERB, "TGATHERB", 8, 128, 67.0},
    {PtoOpcode::TSCATTER, "TSCATTER", 16, 128, 49.1}, {PtoOpcode::TMOV, "TMOV", 8, 128, 41.0},
    {PtoOpcode::TFILLPAD, "TFILLPAD", 8, 128, 54.0},  {PtoOpcode::TFILLPAD_EXPAND, "TFILLPAD_EXPAND", 16, 128, 89.0},
    {PtoOpcode::TTRI, "TTRI", 8, 128, 60.0},
};

TEST(A5ComparisonLayoutFit, HardwareProfiles)
{
    constexpr double kRequiredPrecision = 0.90;
    for (const auto& testCase : kCases) {
        CostModelInput input{
            .op = testCase.op,
            .dtype = DType::Half,
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
