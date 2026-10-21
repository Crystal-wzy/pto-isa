/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MOCKER_A5_REDUCTION_COSTMODEL_HPP
#define PTO_MOCKER_A5_REDUCTION_COSTMODEL_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace pto::mocker::lightweight::a5 {

inline constexpr uint64_t kRowSumElementsPerRepeatFp16 = 64;
inline constexpr double kRowSumSingleRepeatFloorCycles = 63.0;
inline constexpr double kRowSumSingleRepeatBaseCycles = 42.5;
inline constexpr double kRowSumSingleRepeatCyclesPerRow = 6.0;
inline constexpr double kRowSumMultiRepeatBaseCycles = 58.0;
inline constexpr double kRowSumMultiRepeatSetupCycles = 60.0;
inline constexpr double kRowSumMultiRepeatCyclesPerRepeat = 3.0;
inline constexpr double kRowSumMultiRepeatRowBias = 1.0;

struct DualReductionFitParams {
    double flatRepeatSlope;
    double flatBase;
    double rowBaseSlope;
    double rowRepeatSlope;
    double rowOffset;
};

inline constexpr uint64_t kRowReductionElementsPerRepeatFp16 = 128;
inline constexpr DualReductionFitParams kRowExtremaFit = {2.6345, 46.826, 3.0638, 2.5, 26.184};
inline constexpr DualReductionFitParams kRowProdFit = {9.9344, 124.14, 17.277, 6.5745, 83.493};

struct ColReductionFitParams {
    double flatRepeatSlope;
    double flatBase;
    double rowBaseSlope;
    double rowRepeatSlope;
    double rowRepeatOffset;
    double rowOffset;
};

inline constexpr uint64_t kColArithmeticElementsPerRepeatFp16 = 64;
inline constexpr ColReductionFitParams kColSumFit = {4.5919, 43.277, 1.0, 1.2233, 2.2869, 35.158};
inline constexpr ColReductionFitParams kColProdFit = {4.2774, 43.521, 2.5291, 2.0, -4.0839, 38.838};
inline constexpr double kColExtremaElementsPerRepeatFp16 = 128.0;
inline constexpr double kColExtremaSingleRowBaseCycles = 33.0;
inline constexpr double kColExtremaMultiRowBaseCycles = 37.0;

inline uint64_t RoundA5SpecialCycles(double cycles)
{
    return cycles <= 0.0 ? 0 : static_cast<uint64_t>(std::llround(cycles));
}

inline bool TryEstimateA5RowSumCycles(const CostModelInput& input, uint64_t& cycles)
{
    if (input.op != PtoOpcode::TROWSUM || input.dtype != DType::Half) {
        return false;
    }
    const int64_t rows_i = input.valid_rows > 0 ? input.valid_rows : input.rows;
    const int64_t cols_i = input.valid_cols > 0 ? input.valid_cols : input.cols;
    if (rows_i <= 0 || cols_i <= 0) {
        return false;
    }

    const double rows = static_cast<double>(rows_i);
    const uint64_t repeats = CeilDivU64(static_cast<uint64_t>(cols_i), kRowSumElementsPerRepeatFp16);
    double value = 0.0;
    if (repeats == 1) {
        value = std::max(
            kRowSumSingleRepeatFloorCycles, kRowSumSingleRepeatBaseCycles + kRowSumSingleRepeatCyclesPerRow * rows);
    } else {
        const double repeat_count = static_cast<double>(repeats);
        const double setup = kRowSumMultiRepeatSetupCycles + kRowSumMultiRepeatCyclesPerRepeat * repeat_count;
        const double row_path = kRowSumMultiRepeatBaseCycles +
                                rows * (kRowSumMultiRepeatCyclesPerRepeat * repeat_count + kRowSumMultiRepeatRowBias);
        value = std::max(setup, row_path);
    }
    cycles = RoundA5SpecialCycles(value);
    return true;
}

inline bool TryResolveA5ReductionShape(const CostModelInput& input, int64_t& rows, int64_t& cols)
{
    rows = input.valid_rows > 0 ? input.valid_rows : input.rows;
    cols = input.valid_cols > 0 ? input.valid_cols : input.cols;
    return input.dtype == DType::Half && rows > 0 && cols > 0;
}

inline double EvalA5DualReduction(const DualReductionFitParams& params, double rows, double repeats)
{
    const double flatPath = params.flatRepeatSlope * rows * repeats + params.flatBase;
    const double rowPath = (params.rowBaseSlope + params.rowRepeatSlope * repeats) * rows + params.rowOffset;
    return std::max(flatPath, rowPath);
}

inline bool TryEstimateA5RowReductionCycles(const CostModelInput& input, uint64_t& cycles)
{
    const DualReductionFitParams* params = nullptr;
    if (input.op == PtoOpcode::TROWMAX || input.op == PtoOpcode::TROWMIN) {
        params = &kRowExtremaFit;
    } else if (input.op == PtoOpcode::TROWPROD) {
        params = &kRowProdFit;
    } else {
        return false;
    }
    int64_t rows_i = 0;
    int64_t cols_i = 0;
    if (!TryResolveA5ReductionShape(input, rows_i, cols_i)) {
        return false;
    }
    const double rows = static_cast<double>(rows_i);
    const double repeats =
        static_cast<double>(CeilDivU64(static_cast<uint64_t>(cols_i), kRowReductionElementsPerRepeatFp16));
    cycles = RoundA5SpecialCycles(EvalA5DualReduction(*params, rows, repeats));
    return true;
}

inline double EvalA5ColArithmeticReduction(const ColReductionFitParams& params, double rows, double repeats)
{
    const double flatPath = params.flatRepeatSlope * repeats + params.flatBase;
    const double rowPath = (params.rowBaseSlope + params.rowRepeatSlope * repeats) * rows +
                           params.rowRepeatOffset * repeats + params.rowOffset;
    return std::max(flatPath, rowPath);
}

inline bool TryEstimateA5ColReductionCycles(const CostModelInput& input, uint64_t& cycles)
{
    int64_t rows_i = 0;
    int64_t cols_i = 0;
    if (!TryResolveA5ReductionShape(input, rows_i, cols_i)) {
        return false;
    }
    const double rows = static_cast<double>(rows_i);
    if (input.op == PtoOpcode::TCOLMAX || input.op == PtoOpcode::TCOLMIN) {
        const double repeats = std::max(1.0, static_cast<double>(cols_i) / kColExtremaElementsPerRepeatFp16);
        const double value =
            rows_i == 1 ? kColExtremaSingleRowBaseCycles + repeats : kColExtremaMultiRowBaseCycles + rows * repeats;
        cycles = RoundA5SpecialCycles(value);
        return true;
    }

    const ColReductionFitParams* params = nullptr;
    if (input.op == PtoOpcode::TCOLSUM) {
        params = &kColSumFit;
    } else if (input.op == PtoOpcode::TCOLPROD) {
        params = &kColProdFit;
    } else {
        return false;
    }
    const double repeats =
        static_cast<double>(CeilDivU64(static_cast<uint64_t>(cols_i), kColArithmeticElementsPerRepeatFp16));
    cycles = RoundA5SpecialCycles(EvalA5ColArithmeticReduction(*params, rows, repeats));
    return true;
}

inline bool TryEstimateA5SpecialCycles(const CostModelInput& input, uint64_t& cycles)
{
    return TryEstimateA5RowSumCycles(input, cycles) || TryEstimateA5RowReductionCycles(input, cycles) ||
           TryEstimateA5ColReductionCycles(input, cycles);
}

} // namespace pto::mocker::lightweight::a5

#endif // PTO_MOCKER_A5_REDUCTION_COSTMODEL_HPP
