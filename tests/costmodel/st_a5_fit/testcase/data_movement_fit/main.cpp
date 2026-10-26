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

#include <pto/pto-inst.hpp>

#include "a5_cost_check.hpp"

namespace {

using namespace ::pto::mocker::lightweight;

struct DequantFitCase {
    int64_t rows;
    int64_t cols;
    double profilingCycles;
};

constexpr DequantFitCase kDequantCases[] = {
    {1, 64, 50.1}, {1, 128, 51.0}, {1, 256, 57.0}, {1, 512, 67.0},
    {2, 64, 55.0}, {4, 128, 69.0}, {8, 128, 92.0}, {16, 128, 164.0},
};

TEST(A5DataMovementFit, DequantHardwareProfiles)
{
    constexpr double kRequiredPrecision = 0.90;
    for (const auto& testCase : kDequantCases) {
        CostModelInput input{
            .op = PtoOpcode::TDEQUANT,
            .dtype = DType::Int8,
            .rows = testCase.rows,
            .cols = testCase.cols,
            .arch = CostModelArch::A5,
            .dst_dtype = DType::Float,
            .valid_rows = testCase.rows,
            .valid_cols = testCase.cols,
        };
        pto::test::ExpectA5CycleNear("TDEQUANT", input, testCase.profilingCycles, kRequiredPrecision);
    }
}

TEST(A5DataMovementFit, DequantAdapterPreservesArchitectureAndDTypes)
{
    constexpr uint64_t kExpectedCycles = 98;
    auto& runtimeContext = ::pto::perf_sim::GetCostModelCtx();
    const uint32_t originalArch = runtimeContext.npu_arch;
    runtimeContext.npu_arch = ::pto::perf_sim::kAscend950Arch;
    EXPECT_EQ(::pto::perf_sim::EstimateLightweightCycles("TDEQUANT", 8, 128, "int8", "fp32"), kExpectedCycles);
    runtimeContext.npu_arch = originalArch;
}

TEST(A5DataMovementFit, ReshapeUsesScalarAliasCost)
{
    CostModelInput input{
        .op = PtoOpcode::TRESHAPE,
        .dtype = DType::Half,
        .rows = 1,
        .cols = 128,
        .arch = CostModelArch::A5,
    };
    const CostModelResult result = EstimateCycles(input);
    EXPECT_EQ(result.cycles, 1.0);
}

struct StoreFitCase {
    DType dtype;
    int64_t rows;
    int64_t cols;
    double profilingCycles;
};

constexpr StoreFitCase kStoreCases[] = {
    {DType::Half, 16, 16, 38.4},    {DType::Half, 16, 64, 50.0},    {DType::Half, 16, 256, 98.0},
    {DType::Half, 64, 128, 162.4},  {DType::Half, 128, 128, 290.1}, {DType::Float, 16, 16, 42.5},
    {DType::Float, 16, 64, 66.0},   {DType::Float, 64, 128, 290.1}, {DType::Int32, 16, 16, 42.5},
    {DType::Int32, 16, 256, 162.1},
};

TEST(A5DataMovementFit, StoreStreamingHardwareProfiles)
{
    constexpr double kRequiredPrecision = 0.90;
    for (const auto& testCase : kStoreCases) {
        CostModelInput input{
            .op = PtoOpcode::TSTORE,
            .dtype = testCase.dtype,
            .rows = testCase.rows,
            .cols = testCase.cols,
            .arch = CostModelArch::A5,
            .layout = MemLayout::ND,
            .tile_type = TransferTileType::VecTile,
            .data_size = testCase.rows * testCase.cols,
        };
        pto::test::ExpectA5CycleNear("TSTORE", input, testCase.profilingCycles, kRequiredPrecision);
    }
}

} // namespace
