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

void ExpectCycleNear(const char* name, const CostModelInput& input, double profilingCycles)
{
    constexpr double kRequiredPrecision = 0.90;
    pto::test::ExpectA5CycleNear(name, input, profilingCycles, kRequiredPrecision);
}

TEST(A5ReductionFit, RowSumFp16)
{
    ExpectCycleNear(
        "TROWSUM",
        CostModelInput{
            .op = PtoOpcode::TROWSUM, .dtype = DType::Half, .rows = 128, .cols = 64, .arch = CostModelArch::A5},
        810.5);
    ExpectCycleNear(
        "TROWSUM",
        CostModelInput{
            .op = PtoOpcode::TROWSUM, .dtype = DType::Half, .rows = 12, .cols = 320, .arch = CostModelArch::A5},
        255.0);
}

TEST(A5ReductionFit, RemainingReductionsFp16)
{
    struct ReductionCase {
        PtoOpcode op;
        const char* name;
        int64_t rows;
        int64_t cols;
        double profilingCycles;
    };
    constexpr ReductionCase kCases[] = {
        {PtoOpcode::TROWMAX, "TROWMAX", 7, 1024, 197.0},   {PtoOpcode::TROWMIN, "TROWMIN", 128, 128, 786.0},
        {PtoOpcode::TROWPROD, "TROWPROD", 33, 128, 907.5}, {PtoOpcode::TCOLSUM, "TCOLSUM", 12, 320, 131.5},
        {PtoOpcode::TCOLMAX, "TCOLMAX", 12, 320, 65.0},    {PtoOpcode::TCOLMIN, "TCOLMIN", 128, 128, 165.0},
        {PtoOpcode::TCOLPROD, "TCOLPROD", 7, 1024, 230.0},
    };
    for (const auto& testCase : kCases) {
        ExpectCycleNear(
            testCase.name,
            CostModelInput{
                .op = testCase.op,
                .dtype = DType::Half,
                .rows = testCase.rows,
                .cols = testCase.cols,
                .arch = CostModelArch::A5},
            testCase.profilingCycles);
    }
}

TEST(A5ReductionFit, IndexReductionHardwareProfiles)
{
    struct IndexReductionCase {
        PtoOpcode op;
        const char* name;
        DType dtype;
        int64_t rows;
        int64_t cols;
        double profilingCycles;
    };
    constexpr IndexReductionCase kCases[] = {
        {PtoOpcode::TROWARGMAX, "TROWARGMAX", DType::Half, 8, 256, 153.0},
        {PtoOpcode::TROWARGMIN, "TROWARGMIN", DType::Half, 16, 128, 159.0},
        {PtoOpcode::TCOLARGMAX, "TCOLARGMAX", DType::Half, 8, 256, 96.0},
        {PtoOpcode::TCOLARGMIN, "TCOLARGMIN", DType::Half, 16, 128, 122.0},
    };
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
