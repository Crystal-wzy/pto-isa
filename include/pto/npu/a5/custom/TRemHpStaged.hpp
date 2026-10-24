/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TREMHPSTAGED_HPP
#define TREMHPSTAGED_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>
#include "TFmodRemHp.hpp"

namespace pto {

// Loop-inverted high precision remainder: every full vector advances one
// SolveScale iteration per pass, with the per-vector state staged in dst.
// Measured break-even on Ascend950PR sim: 48 full vectors (issue #2).
constexpr uint32_t TREM_HP_STAGED_MIN_VECS = 48;

template <typename T>
PTO_INTERNAL bool TRemHpStagedDisjoint(const __ubuf__ T* a, uint32_t aElems, const __ubuf__ T* b, uint32_t bElems)
{
    return a + aElems <= b || b + bElems <= a;
}

// Staging reuses dst as scratch, so it needs enough full vectors to pay for
// the extra UB round trips and must not alias either input.
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL bool TRemHpStagedEligible(
    const __ubuf__ float* dst, const __ubuf__ float* src0, const __ubuf__ float* src1, uint32_t validRows,
    uint32_t validCols)
{
    constexpr uint32_t elemsPerRepeat = CCE_VL / sizeof(float);
    if (validRows * (validCols / elemsPerRepeat) < TREM_HP_STAGED_MIN_VECS) {
        return false;
    }
    constexpr uint32_t dstElems = TileDataDst::Rows * TileDataDst::Cols;
    constexpr uint32_t src0Elems = TileDataSrc0::Rows * TileDataSrc0::Cols;
    constexpr uint32_t src1Elems = TileDataSrc1::Rows * TileDataSrc1::Cols;
    return TRemHpStagedDisjoint<float>(dst, dstElems, src0, src0Elems) &&
           TRemHpStagedDisjoint<float>(dst, dstElems, src1, src1Elems);
}

template <typename T>
PTO_INTERNAL void TRemHpStagedVecs(uint32_t validRows, uint32_t validCols, uint32_t& fullPerRow, uint32_t& tailCount)
{
    constexpr uint32_t elemsPerRepeat = CCE_VL / sizeof(T);
    fullPerRow = validCols / elemsPerRepeat;
    tailCount = validCols - fullPerRow * elemsPerRepeat;
}

// One SolveScale iteration over every full vector; iteration == FMOD_ITERATION_NUM_MAX
// also seeds the state from the original dividend. The divisor is reloaded from
// the original src1 each pass because a scaled divisor must not feed the next pass.
template <typename RoundMode, int32_t iterationNum, unsigned DstStride, unsigned Src0Stride, unsigned Src1Stride>
PTO_INTERNAL void TRemHpStagePass(
    __ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t validRows, uint32_t fullPerRow)
{
    constexpr uint32_t elemsPerRepeat = CCE_VL / sizeof(float);
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, DistVST::DIST_NORM>())>();
    RegTensor<float> stateReg, divReg, src0Reg;
    MaskReg preg = PSetWithType<float>(PAT_ALL);
    for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
        for (uint16_t j = 0; j < (uint16_t)fullPerRow; ++j) {
            uint32_t col = j * elemsPerRepeat;
            if constexpr (iterationNum == FMOD_ITERATION_NUM_MAX) {
                vlds(src0Reg, src0, row * Src0Stride + col, NORM);
                vabs(stateReg, src0Reg, preg, MODE_ZEROING);
            } else {
                vlds(stateReg, dst, row * DstStride + col, NORM);
            }
            vlds(divReg, src1, row * Src1Stride + col, NORM);
            vabs(divReg, divReg, preg, MODE_ZEROING);
            SolveScale<RoundMode, iterationNum>(
                stateReg, divReg, scaleList1[iterationNum - 1].f, scaleList2[iterationNum - 1].f, preg);
            vsts(stateReg, dst, row * DstStride + col, distValue, preg);
        }
    }
    mem_bar(VST_VLD);
}

// Runs passes [first, first+count) as plain loop bodies (no scopes); the
// caller groups them into a scoped chunk.
template <
    typename RoundMode, int32_t first, int32_t count, unsigned DstStride, unsigned Src0Stride, unsigned Src1Stride>
PTO_INTERNAL void TRemHpStageSeq(
    __ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t validRows, uint32_t fullPerRow)
{
    TRemHpStagePass<RoundMode, first, DstStride, Src0Stride, Src1Stride>(dst, src0, src1, validRows, fullPerRow);
    if constexpr (count > 1) {
        TRemHpStageSeq<RoundMode, first - 1, count - 1, DstStride, Src0Stride, Src1Stride>(
            dst, src0, src1, validRows, fullPerRow);
    }
}

// Sign restore, subnormal rescale, exceptions and rem sign correction over the
// staged state; reloads the original src0/src1 needed by the fixups.
template <bool isFmod, unsigned DstStride, unsigned Src0Stride, unsigned Src1Stride>
PTO_INTERNAL void TRemHpPostPass(
    __ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t validRows, uint32_t fullPerRow)
{
    constexpr uint32_t elemsPerRepeat = CCE_VL / sizeof(float);
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, DistVST::DIST_NORM>())>();
    RegTensor<float> stateReg, src0Reg, src1Reg, nanReg;
    MaskReg preg = PSetWithType<float>(PAT_ALL);
    vdup(nanReg, nan.f, preg, MODE_ZEROING);
    for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
        for (uint16_t j = 0; j < (uint16_t)fullPerRow; ++j) {
            uint32_t col = j * elemsPerRepeat;
            vlds(stateReg, dst, row * DstStride + col, NORM);
            vlds(src0Reg, src0, row * Src0Stride + col, NORM);
            vlds(src1Reg, src1, row * Src1Stride + col, NORM);
            TFmodRemHPPost<isFmod>(stateReg, src0Reg, src1Reg, nanReg, preg);
            vsts(stateReg, dst, row * DstStride + col, distValue, preg);
        }
    }
}

// Tail vectors keep the original register path: 11 chained iterations in
// registers with predicated loads/stores.
template <bool isFmod, unsigned DstStride, unsigned Src0Stride, unsigned Src1Stride>
PTO_INTERNAL void TRemHpTailPass(
    __ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t validRows, uint32_t validCols)
{
    constexpr uint32_t elemsPerRepeat = CCE_VL / sizeof(float);
    uint32_t fullPerRow, tailCount;
    TRemHpStagedVecs<float>(validRows, validCols, fullPerRow, tailCount);
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, DistVST::DIST_NORM>())>();
    RegTensor<float> vDst, vSrc0, vSrc1;
    MaskReg preg = CreatePredicate<float>(tailCount);
    uint32_t col = fullPerRow * elemsPerRepeat;
    for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
        vlds(vSrc0, src0, row * Src0Stride + col, NORM);
        vlds(vSrc1, src1, row * Src1Stride + col, NORM);
        TFmodRemHP<isFmod>(vDst, vSrc0, vSrc1, preg);
        vsts(vDst, dst, row * DstStride + col, distValue, preg);
    }
}

template <bool isFmod, unsigned DstStride, unsigned Src0Stride, unsigned Src1Stride>
PTO_INTERNAL void TRemHpStaged(
    __ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t validRows, uint32_t validCols)
{
    using RoundMode = std::conditional_t<isFmod, decltype(ROUND_Z), decltype(ROUND_F)>;
    uint32_t fullPerRow, tailCount;
    TRemHpStagedVecs<float>(validRows, validCols, fullPerRow, tailCount);
    // Passes are VF-fused in small chunks: mem_bar(VST_VLD) between passes
    // inside a chunk, a full barrier between chunks. One VF for everything
    // regresses 2D shapes (queue/register pressure); 13 separate VFs pay
    // too much launch overhead (npu_skills vf-fusion stop-conditions). TREMS
    // keeps one VF per pass (see the S variants below).
    __VEC_SCOPE__
    {
        TRemHpStageSeq<RoundMode, FMOD_ITERATION_NUM_MAX, 3, DstStride, Src0Stride, Src1Stride>(
            dst, src0, src1, validRows, fullPerRow);
    }
    pipe_barrier(PIPE_ALL);
    __VEC_SCOPE__
    {
        TRemHpStageSeq<RoundMode, FMOD_ITERATION_NUM_MAX - 3, 3, DstStride, Src0Stride, Src1Stride>(
            dst, src0, src1, validRows, fullPerRow);
    }
    pipe_barrier(PIPE_ALL);
    __VEC_SCOPE__
    {
        TRemHpStageSeq<RoundMode, FMOD_ITERATION_NUM_MAX - 6, 3, DstStride, Src0Stride, Src1Stride>(
            dst, src0, src1, validRows, fullPerRow);
    }
    pipe_barrier(PIPE_ALL);
    __VEC_SCOPE__
    {
        TRemHpStageSeq<RoundMode, 2, 2, DstStride, Src0Stride, Src1Stride>(dst, src0, src1, validRows, fullPerRow);
        TRemHpPostPass<isFmod, DstStride, Src0Stride, Src1Stride>(dst, src0, src1, validRows, fullPerRow);
    }
    if (tailCount > 0) {
        __VEC_SCOPE__
        {
            TRemHpTailPass<isFmod, DstStride, Src0Stride, Src1Stride>(dst, src0, src1, validRows, validCols);
        }
    }
}

// Scalar-divisor variant: the divisor is a broadcast register rebuilt once per
// pass, so no src1 reload is staged.
template <typename RoundMode, int32_t iterationNum, unsigned DstStride, unsigned Src0Stride>
PTO_INTERNAL void TRemHpStagePassS(
    __ubuf__ float* dst, __ubuf__ float* src0, float scalar, uint32_t validRows, uint32_t fullPerRow)
{
    constexpr uint32_t elemsPerRepeat = CCE_VL / sizeof(float);
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, DistVST::DIST_NORM>())>();
    RegTensor<float> stateReg, divReg, src0Reg;
    MaskReg preg = PSetWithType<float>(PAT_ALL);
    vdup(divReg, scalar, preg, MODE_ZEROING);
    vabs(divReg, divReg, preg, MODE_ZEROING);
    for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
        for (uint16_t j = 0; j < (uint16_t)fullPerRow; ++j) {
            uint32_t col = j * elemsPerRepeat;
            if constexpr (iterationNum == FMOD_ITERATION_NUM_MAX) {
                vlds(src0Reg, src0, row * Src0Stride + col, NORM);
                vabs(stateReg, src0Reg, preg, MODE_ZEROING);
            } else {
                vlds(stateReg, dst, row * DstStride + col, NORM);
            }
            SolveScale<RoundMode, iterationNum>(
                stateReg, divReg, scaleList1[iterationNum - 1].f, scaleList2[iterationNum - 1].f, preg);
            vsts(stateReg, dst, row * DstStride + col, distValue, preg);
        }
    }
    mem_bar(VST_VLD);
}

template <typename RoundMode, int32_t iterationNum, unsigned DstStride, unsigned Src0Stride>
PTO_INTERNAL void TRemHpStagePassScopedS(
    __ubuf__ float* dst, __ubuf__ float* src0, float scalar, uint32_t validRows, uint32_t fullPerRow)
{
    __VEC_SCOPE__
    {
        TRemHpStagePassS<RoundMode, iterationNum, DstStride, Src0Stride>(dst, src0, scalar, validRows, fullPerRow);
    }
    pipe_barrier(PIPE_ALL);
}

template <typename RoundMode, int32_t iterationNum, unsigned DstStride, unsigned Src0Stride>
PTO_INTERNAL void TRemHpStageIterS(
    __ubuf__ float* dst, __ubuf__ float* src0, float scalar, uint32_t validRows, uint32_t fullPerRow)
{
    TRemHpStagePassScopedS<RoundMode, iterationNum, DstStride, Src0Stride>(dst, src0, scalar, validRows, fullPerRow);
    if constexpr (iterationNum > 1) {
        TRemHpStageIterS<RoundMode, iterationNum - 1, DstStride, Src0Stride>(dst, src0, scalar, validRows, fullPerRow);
    }
}

template <bool isFmod, unsigned DstStride, unsigned Src0Stride>
PTO_INTERNAL void TRemHpPostPassS(
    __ubuf__ float* dst, __ubuf__ float* src0, float scalar, uint32_t validRows, uint32_t fullPerRow)
{
    constexpr uint32_t elemsPerRepeat = CCE_VL / sizeof(float);
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, DistVST::DIST_NORM>())>();
    RegTensor<float> stateReg, src0Reg, src1Reg, nanReg;
    MaskReg preg = PSetWithType<float>(PAT_ALL);
    vdup(nanReg, nan.f, preg, MODE_ZEROING);
    vdup(src1Reg, scalar, preg, MODE_ZEROING);
    for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
        for (uint16_t j = 0; j < (uint16_t)fullPerRow; ++j) {
            uint32_t col = j * elemsPerRepeat;
            vlds(stateReg, dst, row * DstStride + col, NORM);
            vlds(src0Reg, src0, row * Src0Stride + col, NORM);
            TFmodRemHPPost<isFmod>(stateReg, src0Reg, src1Reg, nanReg, preg);
            vsts(stateReg, dst, row * DstStride + col, distValue, preg);
        }
    }
}

template <bool isFmod, unsigned DstStride, unsigned Src0Stride>
PTO_INTERNAL void TRemHpTailPassS(
    __ubuf__ float* dst, __ubuf__ float* src0, float scalar, uint32_t validRows, uint32_t validCols)
{
    constexpr uint32_t elemsPerRepeat = CCE_VL / sizeof(float);
    uint32_t fullPerRow, tailCount;
    TRemHpStagedVecs<float>(validRows, validCols, fullPerRow, tailCount);
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, DistVST::DIST_NORM>())>();
    RegTensor<float> vDst, vSrc0, vSrc1;
    MaskReg preg = CreatePredicate<float>(tailCount);
    uint32_t col = fullPerRow * elemsPerRepeat;
    for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
        vlds(vSrc0, src0, row * Src0Stride + col, NORM);
        vdup(vSrc1, scalar, preg, MODE_ZEROING);
        TFmodRemHP<isFmod>(vDst, vSrc0, vSrc1, preg);
        vsts(vDst, dst, row * DstStride + col, distValue, preg);
    }
}

template <bool isFmod, unsigned DstStride, unsigned Src0Stride>
PTO_INTERNAL void TRemSStaged(
    __ubuf__ float* dst, __ubuf__ float* src0, float scalar, uint32_t validRows, uint32_t validCols)
{
    using RoundMode = std::conditional_t<isFmod, decltype(ROUND_Z), decltype(ROUND_F)>;
    uint32_t fullPerRow, tailCount;
    TRemHpStagedVecs<float>(validRows, validCols, fullPerRow, tailCount);
    // Per-pass VFs with full drains: measured faster than chunked fusion for
    // the scalar variant (load-light passes lose to bar_all drain stalls).
    TRemHpStageIterS<RoundMode, FMOD_ITERATION_NUM_MAX, DstStride, Src0Stride>(
        dst, src0, scalar, validRows, fullPerRow);
    __VEC_SCOPE__ { TRemHpPostPassS<isFmod, DstStride, Src0Stride>(dst, src0, scalar, validRows, fullPerRow); }
    if (tailCount > 0) {
        __VEC_SCOPE__ { TRemHpTailPassS<isFmod, DstStride, Src0Stride>(dst, src0, scalar, validRows, validCols); }
    }
}

} // namespace pto
#endif // TREMHPSTAGED_HPP
