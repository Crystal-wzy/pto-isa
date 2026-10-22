/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>

#include <gtest/gtest.h>

#include <pto/costmodel/lightweight_costmodel.hpp>

#include "a5_cost_check.hpp"

namespace {

using namespace ::pto::mocker::lightweight;

struct FitCase {
    PtoOpcode op;
    const char* name;
    DType dtype;
    int64_t rows;
    int64_t cols;
    double profilingCycles;
};

constexpr FitCase kCases[] = {
    {PtoOpcode::TPARTADD, "TPARTADD", DType::Half, 1, 16, 38.0},
    {PtoOpcode::TPARTADD, "TPARTADD", DType::Half, 24, 256, 186.0},
    {PtoOpcode::TPARTMUL, "TPARTMUL", DType::Half, 1, 1024, 46.0},
    {PtoOpcode::TPARTMUL, "TPARTMUL", DType::Half, 48, 64, 306.0},
    {PtoOpcode::TPARTMAX, "TPARTMAX", DType::Half, 2, 64, 39.0},
    {PtoOpcode::TPARTMAX, "TPARTMAX", DType::Half, 64, 128, 402.0},
    {PtoOpcode::TPARTMIN, "TPARTMIN", DType::Half, 1, 512, 41.0},
    {PtoOpcode::TPARTMIN, "TPARTMIN", DType::Half, 16, 256, 130.0},
    {PtoOpcode::TROWEXPAND, "TROWEXPAND", DType::Half, 8, 128, 66.0},
    {PtoOpcode::TROWEXPANDADD, "TROWEXPANDADD", DType::Half, 16, 128, 114.0},
    {PtoOpcode::TROWEXPANDSUB, "TROWEXPANDSUB", DType::Half, 8, 128, 66.0},
    {PtoOpcode::TROWEXPANDMUL, "TROWEXPANDMUL", DType::Half, 16, 128, 114.0},
    {PtoOpcode::TROWEXPANDDIV, "TROWEXPANDDIV", DType::Half, 8, 128, 78.0},
    {PtoOpcode::TROWEXPANDMAX, "TROWEXPANDMAX", DType::Half, 16, 128, 114.0},
    {PtoOpcode::TROWEXPANDMIN, "TROWEXPANDMIN", DType::Half, 8, 128, 66.0},
    {PtoOpcode::TROWEXPANDEXPDIF, "TROWEXPANDEXPDIF", DType::Half, 8, 128, 81.0},
    {PtoOpcode::TCOLEXPAND, "TCOLEXPAND", DType::Half, 16, 128, 49.0},
    {PtoOpcode::TCOLEXPANDADD, "TCOLEXPANDADD", DType::Half, 8, 128, 66.0},
    {PtoOpcode::TCOLEXPANDSUB, "TCOLEXPANDSUB", DType::Half, 16, 128, 114.0},
    {PtoOpcode::TCOLEXPANDMUL, "TCOLEXPANDMUL", DType::Half, 8, 128, 66.0},
    {PtoOpcode::TCOLEXPANDDIV, "TCOLEXPANDDIV", DType::Half, 8, 128, 78.0},
    {PtoOpcode::TCOLEXPANDMAX, "TCOLEXPANDMAX", DType::Half, 16, 128, 114.0},
    {PtoOpcode::TCOLEXPANDMIN, "TCOLEXPANDMIN", DType::Half, 8, 128, 66.0},
    {PtoOpcode::TCOLEXPANDEXPDIF, "TCOLEXPANDEXPDIF", DType::Half, 8, 128, 81.0},
    {PtoOpcode::TEXPANDS, "TEXPANDS", DType::Half, 16, 128, 114.0},
};

TEST(A5LocalExpandFit, HardwareProfiles)
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
