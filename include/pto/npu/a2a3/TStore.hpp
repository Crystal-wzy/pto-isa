/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSTORE_HPP
#define TSTORE_HPP
#include "common.hpp"
#include <pto/common/pto_tile.hpp>

namespace pto {

template <typename GlobalData, typename TileData>
PTO_INTERNAL void TStoreUb2gmInstr(
    typename GlobalData::DType* dst, __ubuf__ typename TileData::DType* src, uint16_t nBurst, uint32_t lenBurst,
    uint32_t gmGap, uint32_t ubGap)
{
    if constexpr (sizeof(typename TileData::DType) == 1) {
        copy_ubuf_to_gm_align_b8(dst, src, 0, nBurst, lenBurst, 0, 0, ubGap, gmGap);
    } else if constexpr (sizeof(typename TileData::DType) == 2) {
        copy_ubuf_to_gm_align_b16(dst, src, 0, nBurst, lenBurst, 0, 0, ubGap, gmGap);
    } else if constexpr (sizeof(typename TileData::DType) == 4 || sizeof(typename TileData::DType) == 8) {
        copy_ubuf_to_gm_align_b32(dst, src, 0, nBurst, lenBurst, 0, 0, ubGap, gmGap);
    }
}

template <
    typename GlobalData, typename TileData, QuantMode_t quantizationMode = QuantMode_t::NoQuant,
    ReluPreMode reluPreMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TStoreAccNz2nz(
    typename GlobalData::DType* dstAddr, __cc__ typename TileData::DType* srcAddr, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
    int validRow, int validCol)
{
    PTO_ASSERT(validRow == gShape2 * gShape3, "The validRow of TileData must be equal to Shape2 * Shape3 of NZ shape!");
    PTO_ASSERT(
        validCol == gShape0 * gShape1 * gShape4,
        "The validCol of TileData must be equal to Shape0 * Shape1 * Shape4 of NZ shape!");
    PTO_ASSERT(
        validRow >= 1 && validRow <= 65535 && validCol % 16 == 0,
        "When GlobalData is NZ format, the range of validRow is [1, 65535] and validCol must be an integer multiple of "
        "16.");

    static_assert(
        GlobalData::staticShape[3] == FRACTAL_NZ_ROW,
        "When GlobalData is NZ format, the second-to-last dimension shall be 16.");
    static_assert(
        (std::is_same_v<typename GlobalData::DType, __gm__ float> &&
         (GlobalData::staticShape[4] == 8 || GlobalData::staticShape[4] == 16)) ||
            (std::is_same_v<typename GlobalData::DType, __gm__ int32_t> && GlobalData::staticShape[4] == 16) ||
            (GlobalData::staticShape[4] == BLOCK_BYTE_SIZE / sizeof(typename GlobalData::DType)),
        "When GlobalData is in NZ format: if DstType is float, the last dimension must be either 8 or 16, "
        "and the dimension value is 8 if and only if Channel Split is enabled; if DstType is int32_t, the "
        "last dimension must be exactly 16. In addition, the last dimension must be static and satisfy 32 / "
        "sizeof(DstType).");

    uint16_t mSize = validRow;
    uint16_t nSize = validCol;

    uint32_t c0Size = sizeof(typename GlobalData::DType) * gShape4;
    uint16_t srcStride = TileData::Rows;
    if constexpr (CompactMode::Normal == TileData::Compact) {
        srcStride = (FRACTAL_NZ_ROW + validRow - 1) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW;
    }
    uint32_t dstStride = gShape2 * gShape3 * c0Size >> SHIFT_BLOCK_BYTE;

    constexpr uint8_t unitFlagCtrl = static_cast<uint8_t>(Phase);
    uint8_t channelSplitEn = 0;
    if (std::is_same_v<typename TileData::DType, float> && std::is_same_v<typename GlobalData::DType, __gm__ float>) {
        if (gShape4 == 8) {
            channelSplitEn = 1;
        }
    }

    uint64_t xmReg =

        (static_cast<uint64_t>(nSize & 0xfff) << 4) |          // Xm[15:4] nSize
        (static_cast<uint64_t>(mSize & 0xffff) << 16) |        // Xm[31:16] mSize
        (static_cast<uint64_t>(dstStride & 0xffffffff) << 32); // Xm[63:32] destination stride between the start addr

    uint64_t xtReg = srcStride | // Xt[15:0] the source stride between the start addr
                     (static_cast<uint64_t>(unitFlagCtrl & 0x3) << 32) |      // Xt[33:32] unit flag control bit
                     (static_cast<uint64_t>(quantizationMode & 0x1f) << 34) | // Xt[38:34] pre-stage quantization mode
                     ((static_cast<uint64_t>(reluPreMode) & 0x7) << 39) |     //  Xt[41:39] relu pre mode
                     (static_cast<uint64_t>(channelSplitEn & 0x1) << 42);     // Xt[42] channel split control bit

    copy_matrix_cc_to_gm(dstAddr, srcAddr, xmReg, xtReg);
}

#include "pto/common/arch/memory/tstore_common.hpp"

template <
    typename GlobalData, typename TileData, QuantMode_t quantizationMode = QuantMode_t::NoQuant,
    ReluPreMode reluPreMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified>
__tf__ AICORE void TStoreAcc(
    typename GlobalData::DType __out__* dst, typename TileData::TileDType __in__ src, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
    int validRow, int validCol)
{
    __cc__ typename TileData::DType* srcAddr = (__cc__ typename TileData::DType*)__cce_get_tile_ptr(src);
    typename GlobalData::DType* dstAddr = dst;

    if constexpr (GlobalData::layout == Layout::ND) {
        TStoreAccNz2nd<GlobalData, TileData, quantizationMode, reluPreMode, Phase>(
            dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2, gStride3,
            gStride4, validRow, validCol);
    } else if constexpr (GlobalData::layout == Layout::NZ) {
        TStoreAccNz2nz<GlobalData, TileData, quantizationMode, reluPreMode, Phase>(
            dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2, gStride3,
            gStride4, validRow, validCol);
    } else if constexpr (GlobalData::layout == Layout::NC1HWC0) {
        TStoreAccNz2NC1HWC0<GlobalData, TileData, quantizationMode, reluPreMode, Phase>(
            dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride1, gStride3, validRow, validCol);
    } else if constexpr (GlobalData::layout == Layout::NDC1HWC0) {
        TStoreAccNz2NDC1HWC0<GlobalData, TileData, quantizationMode, reluPreMode, Phase>(
            dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride2, gStride4, validRow, validCol);
    }
}

template <typename TileData, typename GlobalData, bool isQuant>
PTO_INTERNAL void CheckAcc2gm(GlobalData& dst, TileData& src)
{
    static_assert(
        (GlobalData::layout == Layout::ND || GlobalData::layout == Layout::NZ ||
         GlobalData::layout == Layout::NC1HWC0 || GlobalData::layout == Layout::NDC1HWC0),
        "The output data layout must be ND, NZ, NC1HWC0 or NDC1HWC0.");
    static_assert(
        std::is_same_v<typename TileData::DType, int32_t> || std::is_same_v<typename TileData::DType, float>,
        "The input data type must be restricted to int32_t/float!");
    if constexpr (!isQuant) {
        static_assert(
            std::is_same_v<typename GlobalData::DType, __gm__ int32_t> ||
                std::is_same_v<typename GlobalData::DType, __gm__ float> ||
                std::is_same_v<typename GlobalData::DType, __gm__ half> ||
                std::is_same_v<typename GlobalData::DType, __gm__ bfloat16_t>,
            "The output data type must be restricted to int32_t/float/half/bfloat16_t!");
    } else if constexpr (isQuant) {
        if constexpr (std::is_same_v<typename TileData::DType, float>) {
            static_assert(
                std::is_same<typename GlobalData::DType, __gm__ int8_t>::value ||
                    std::is_same<typename GlobalData::DType, __gm__ uint8_t>::value,
                "The output data type must be restricted to int8_t/uint8_t when the source data type is float.");
        } else if constexpr (std::is_same_v<typename TileData::DType, int32_t>) {
            static_assert(
                std::is_same<typename GlobalData::DType, __gm__ int8_t>::value ||
                    std::is_same<typename GlobalData::DType, __gm__ uint8_t>::value ||
                    std::is_same<typename GlobalData::DType, __gm__ half>::value,
                "The output data type must be restricted to half/int8_t/uint8_t when the source data type is int32_t.");
        }
    }
    static_assert(TileData::Cols >= 1 && TileData::Cols <= 4095, "The range of Cols is [1, 4095].");
    static_assert(
        (GlobalData::layout == Layout::ND && TileData::Rows >= 1 && TileData::Rows <= 8192) ||
            ((GlobalData::layout == Layout::NZ || GlobalData::layout == Layout::NC1HWC0 ||
              GlobalData::layout == Layout::NDC1HWC0) &&
             TileData::Rows >= 1 && TileData::Rows <= 65535 && TileData::Cols % 16 == 0),
        "When GlobalData is ND format, the range of Rows is [1, 8192]."
        "When GlobalData is NZ or NC1HWC0 or NDC1HWC0 format, the range of Rows is [1, 65535] and Cols "
        "must be an integer multiple of 16.");
    PTO_ASSERT(src.GetValidCol() >= 1 && src.GetValidCol() <= 4095, "The range of validCol is [1, 4095].");
    PTO_ASSERT(
        dst.GetShape(GlobalTensorDim::DIM_0) > 0 && dst.GetShape(GlobalTensorDim::DIM_1) > 0 &&
            dst.GetShape(GlobalTensorDim::DIM_2) > 0 && dst.GetShape(GlobalTensorDim::DIM_3) > 0 &&
            dst.GetShape(GlobalTensorDim::DIM_4) > 0 && src.GetValidRow() > 0 && src.GetValidCol() > 0,
        "The shape of src and dst must be greater than 0!");
}

// -----------------------------------------------------------------------------
// ConvTile TSTORE: only the source valid box (anchored at the logical origin) is copied to GM; GM outside
// the box is left untouched. DMA bursts move whole 32-byte blocks, so the valid C0 extent must stay the
// full C0 block. Valid extents drive the loop counts / burst lengths, the declared tile extents provide
// every L1 pitch, and the GlobalTensor strides provide every GM address.
// -----------------------------------------------------------------------------
template <typename GlobalData, typename TileData>
PTO_INTERNAL void TStoreInstrL1ToGm(
    typename GlobalData::DType* dst, __cbuf__ typename TileData::DType* src, uint16_t nBurst, uint16_t lenBurst,
    uint16_t l1Gap, uint64_t gmGap)
{
    if (gmGap <= UINT16_MAX) {
        copy_cbuf_to_gm(dst, src, (uint8_t)0, nBurst, lenBurst, l1Gap, static_cast<uint16_t>(gmGap));
        return;
    }
    // Advance full-width addresses when the gap cannot be encoded by the DMA instruction.
    constexpr uint32_t BLOCK_ELEMENTS = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    for (uint16_t i = 0; i < nBurst; i++) {
        copy_cbuf_to_gm(
            dst + uint64_t(i) * (lenBurst + gmGap) * BLOCK_ELEMENTS,
            src + uint64_t(i) * (lenBurst + l1Gap) * BLOCK_ELEMENTS, (uint8_t)0, 1, lenBurst, 0, 0);
    }
}

template <typename GlobalData, typename TileData>
PTO_INTERNAL void CheckConvTileDataStore(GlobalData& dst, TileData& src)
{
    static_assert(
        std::is_same_v<typename TileData::DType, int8_t> || std::is_same_v<typename TileData::DType, uint8_t> ||
            std::is_same_v<typename TileData::DType, int16_t> || std::is_same_v<typename TileData::DType, uint16_t> ||
            std::is_same_v<typename TileData::DType, int32_t> || std::is_same_v<typename TileData::DType, uint32_t> ||
            std::is_same_v<typename TileData::DType, half> || std::is_same_v<typename TileData::DType, bfloat16_t> ||
            std::is_same_v<typename TileData::DType, float>,
        "Fix: Data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/half/bfloat16_t/float!");
    static_assert(TileData::Loc == pto::TileType::Mat, "Fix: Source TileType must be Mat!");
    static_assert(
        sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
        "Fix: Source dtype must be same with dst dtype!");

    constexpr bool isSameLayout =
        (GlobalData::layout == pto::Layout::NC1HWC0 && TileData::layout == pto::Layout::NC1HWC0) ||
        (GlobalData::layout == pto::Layout::FRACTAL_Z && TileData::layout == pto::Layout::FRACTAL_Z) ||
        (GlobalData::layout == pto::Layout::FRACTAL_Z_3D && TileData::layout == pto::Layout::FRACTAL_Z_3D) ||
        (GlobalData::layout == pto::Layout::NDC1HWC0 && TileData::layout == pto::Layout::NDC1HWC0);
    static_assert(
        isSameLayout == true, "Fix: Src Dst layout must be NC1HWC0 or FRACTAL_Z or FRACTAL_Z_3D or NDC1HWC0!");

    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    if constexpr (GlobalData::layout == pto::Layout::FRACTAL_Z || GlobalData::layout == pto::Layout::FRACTAL_Z_3D) {
        if constexpr (TileData::totalDimCount == 4) { // ConvTile layout is [C1HW,N/16,16,C0]
            static_assert(
                TileData::staticShape[2] == FRACTAL_NZ_ROW && TileData::staticShape[3] == c0ElemCount,
                "Fix: The TileData last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");
            static_assert(
                GlobalData::staticShape[3] == FRACTAL_NZ_ROW && GlobalData::staticShape[4] == c0ElemCount,
                "Fix: The GlobalTensor last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");
        } else { // [C1,H,W,N,C0]
            PTO_ASSERT(
                src.GetShape(GlobalTensorDim::DIM_1) == dst.GetShape(GlobalTensorDim::DIM_1) &&
                    src.GetShape(GlobalTensorDim::DIM_2) == dst.GetShape(GlobalTensorDim::DIM_2),
                "Fix: layout is Fractal_Z, [srcH,srcW] && [dstH,dstW] should be same!");
            PTO_ASSERT(dst.GetShape(GlobalTensorDim::DIM_3) <= UINT16_MAX, "Fix: max support dstN is UINT16_MAX!");
        }
    }
}

template <typename GlobalData, typename TileData>
__tf__ PTO_INTERNAL void TStore5HDValid(
    typename GlobalData::DType __out__* dst, typename TileData::TileDType __in__ src, int64_t gStride0,
    int64_t gStride1, int64_t gStride2, int srcC1, int srcH, int srcW, int vN, int vC1, int vH, int vW)
{
    __cbuf__ typename TileData::DType* srcAddr = (__cbuf__ typename TileData::DType*)__cce_get_tile_ptr(src);
    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    constexpr uint32_t elemBytes = sizeof(typename TileData::DType);

    PTO_ASSERT(vN > 0 && vC1 > 0 && vH > 0 && vW > 0, "Fix: ConvTile valid extents must be positive!");
    PTO_ASSERT(
        vW <= srcW && vH <= srcH && vC1 <= srcC1, "Fix: ConvTile valid W/H/C1 must not exceed the declared extent!");
    PTO_ASSERT(
        gStride1 >= int64_t(vH) * vW * c0ElemCount && gStride2 >= int64_t(vW) * c0ElemCount,
        "Fix: destination stride is smaller than the valid H*W box!");

    const int64_t gStride1Blocks = (gStride1 * elemBytes) >> SHIFT_BLOCK_BYTE;
    const int64_t gStride2Blocks = (gStride2 * elemBytes) >> SHIFT_BLOCK_BYTE;
    const int64_t declPlane = int64_t(srcH) * srcW;         // 32B blocks in one declared H*W plane
    const int64_t validPlane = int64_t(vH) * vW;            // 32B blocks in one valid H*W plane
    const int64_t declPlaneElems = declPlane * c0ElemCount; // elements in one declared H*W plane
    const int64_t declNPitch = int64_t(srcC1) * declPlaneElems;

    typename GlobalData::DType* dstAddrP = dst;
    __cbuf__ typename TileData::DType* srcAddrP = srcAddr;
    const bool fastPath = (vH == 1 || (vW == srcW && gStride2 == int64_t(vW) * c0ElemCount)) && vC1 <= 4095 &&
                          validPlane <= UINT16_MAX && declPlane <= UINT16_MAX;
    if (fastPath) {
        const uint16_t nBurst = static_cast<uint16_t>(vC1);
        const uint16_t lenBurst = static_cast<uint16_t>(validPlane);
        const uint16_t l1Gap = static_cast<uint16_t>(declPlane - validPlane);
        const uint64_t gmGap = static_cast<uint64_t>(gStride1Blocks - validPlane);
        for (int n = 0; n < vN; ++n) {
            TStoreInstrL1ToGm<GlobalData, TileData>(
                dstAddrP + int64_t(n) * gStride0, srcAddrP + int64_t(n) * declNPitch, nBurst, lenBurst, l1Gap, gmGap);
        }
        return;
    }

    PTO_ASSERT(vH <= 4095 && vW <= UINT16_MAX && srcW <= UINT16_MAX, "Fix: valid H/W exceeds the DMA range!");
    const uint16_t nBurst = static_cast<uint16_t>(vH);
    const uint16_t lenBurst = static_cast<uint16_t>(vW);
    const uint16_t l1Gap = static_cast<uint16_t>(srcW - vW);
    const uint64_t gmGap = static_cast<uint64_t>(gStride2Blocks - vW);
    for (int n = 0; n < vN; ++n) {
        for (int c1 = 0; c1 < vC1; ++c1) {
            TStoreInstrL1ToGm<GlobalData, TileData>(
                dstAddrP + int64_t(n) * gStride0 + int64_t(c1) * gStride1,
                srcAddrP + int64_t(n) * declNPitch + int64_t(c1) * declPlaneElems, nBurst, lenBurst, l1Gap, gmGap);
        }
    }
}

template <typename GlobalData, typename TileData>
__tf__ PTO_INTERNAL void TStoreNDC1HWC0Valid(
    typename GlobalData::DType __out__* dst, typename TileData::TileDType __in__ src, int64_t gStride0,
    int64_t gStride1, int64_t gStride2, int64_t gStride3, int srcD, int srcC1, int srcH, int srcW, int vN, int vD,
    int vC1, int vH, int vW)
{
    __cbuf__ typename TileData::DType* srcAddr = (__cbuf__ typename TileData::DType*)__cce_get_tile_ptr(src);
    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    constexpr uint32_t elemBytes = sizeof(typename TileData::DType);

    PTO_ASSERT(vN > 0 && vD > 0 && vC1 > 0 && vH > 0 && vW > 0, "Fix: ConvTile valid extents must be positive!");
    PTO_ASSERT(vW <= srcW && vH <= srcH && vD <= srcD && vC1 <= srcC1, "Fix: valid extent exceeds declared!");
    PTO_ASSERT(
        gStride2 >= int64_t(vH) * vW * c0ElemCount && gStride3 >= int64_t(vW) * c0ElemCount,
        "Fix: destination stride is smaller than the valid H*W box!");

    const int64_t gStride2Blocks = (gStride2 * elemBytes) >> SHIFT_BLOCK_BYTE;
    const int64_t gStride3Blocks = (gStride3 * elemBytes) >> SHIFT_BLOCK_BYTE;
    const int64_t declPlane = int64_t(srcH) * srcW;
    const int64_t validPlane = int64_t(vH) * vW;
    const int64_t declPlaneElems = declPlane * c0ElemCount;
    const int64_t declDpitch = int64_t(srcC1) * declPlaneElems;
    const int64_t declNPitch = int64_t(srcD) * declDpitch;

    typename GlobalData::DType* dstAddrP = dst;
    __cbuf__ typename TileData::DType* srcAddrP = srcAddr;
    const bool fastPath = (vH == 1 || (vW == srcW && gStride3 == int64_t(vW) * c0ElemCount)) && vC1 <= 4095 &&
                          validPlane <= UINT16_MAX && declPlane <= UINT16_MAX;
    if (fastPath) {
        const uint16_t nBurst = static_cast<uint16_t>(vC1);
        const uint16_t lenBurst = static_cast<uint16_t>(validPlane);
        const uint16_t l1Gap = static_cast<uint16_t>(declPlane - validPlane);
        const uint64_t gmGap = static_cast<uint64_t>(gStride2Blocks - validPlane);
        for (int n = 0; n < vN; ++n) {
            for (int d = 0; d < vD; ++d) {
                TStoreInstrL1ToGm<GlobalData, TileData>(
                    dstAddrP + int64_t(n) * gStride0 + int64_t(d) * gStride1,
                    srcAddrP + int64_t(n) * declNPitch + int64_t(d) * declDpitch, nBurst, lenBurst, l1Gap, gmGap);
            }
        }
        return;
    }

    PTO_ASSERT(vH <= 4095 && vW <= UINT16_MAX && srcW <= UINT16_MAX, "Fix: valid H/W exceeds the DMA range!");
    const uint16_t nBurst = static_cast<uint16_t>(vH);
    const uint16_t lenBurst = static_cast<uint16_t>(vW);
    const uint16_t l1Gap = static_cast<uint16_t>(srcW - vW);
    const uint64_t gmGap = static_cast<uint64_t>(gStride3Blocks - vW);
    for (int n = 0; n < vN; ++n) {
        for (int d = 0; d < vD; ++d) {
            for (int c1 = 0; c1 < vC1; ++c1) {
                TStoreInstrL1ToGm<GlobalData, TileData>(
                    dstAddrP + int64_t(n) * gStride0 + int64_t(d) * gStride1 + int64_t(c1) * gStride2,
                    srcAddrP + int64_t(n) * declNPitch + int64_t(d) * declDpitch + int64_t(c1) * declPlaneElems, nBurst,
                    lenBurst, l1Gap, gmGap);
            }
        }
    }
}

template <typename GlobalData, typename TileData>
__tf__ PTO_INTERNAL void TStoreFractalZ4DValid(
    typename GlobalData::DType __out__* dst, typename TileData::TileDType __in__ src, int64_t gStride1,
    int64_t gStride2, int srcD1, int srcD2, int srcD3, int v0, int v1, int v2, int v3)
{
    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    constexpr uint32_t elemBytes = sizeof(typename TileData::DType);
    static_assert(
        TileData::staticShape[2] == FRACTAL_NZ_ROW && TileData::staticShape[3] == c0ElemCount,
        "Fix: The TileData last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");
    static_assert(
        GlobalData::staticShape[3] == FRACTAL_NZ_ROW && GlobalData::staticShape[4] == c0ElemCount,
        "Fix: The GlobalTensor last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");

    __cbuf__ typename TileData::DType* srcAddr = (__cbuf__ typename TileData::DType*)__cce_get_tile_ptr(src);

    PTO_ASSERT(v0 > 0 && v1 > 0 && v2 > 0, "Fix: ConvTile valid extents must be positive!");
    PTO_ASSERT(v1 <= srcD1 && v2 <= srcD2, "Fix: ConvTile valid N1/N0 must not exceed the declared extent!");
    PTO_ASSERT(v3 == srcD3 && srcD3 == int(c0ElemCount), "Fix: valid C0 must stay the full 32-byte C0 block!");
    PTO_ASSERT(v1 <= UINT16_MAX && v2 <= UINT16_MAX, "Fix: valid N1/N0 exceeds the DMA range!");
    PTO_ASSERT(gStride2 >= int64_t(v2) * c0ElemCount, "Fix: destination stride is smaller than the valid N0 box!");

    const int64_t gStride2Blocks = (gStride2 * elemBytes) >> SHIFT_BLOCK_BYTE;
    const int64_t declPlaneElems = int64_t(srcD1) * srcD2 * c0ElemCount;
    const uint16_t nBurst = static_cast<uint16_t>(v1);
    const uint16_t lenBurst = static_cast<uint16_t>(v2);
    const uint16_t l1Gap = static_cast<uint16_t>(srcD2 - v2);
    const uint64_t gmGap = static_cast<uint64_t>(gStride2Blocks - v2);

    for (int i0 = 0; i0 < v0; ++i0) {
        TStoreInstrL1ToGm<GlobalData, TileData>(
            dst + int64_t(i0) * gStride1, srcAddr + int64_t(i0) * declPlaneElems, nBurst, lenBurst, l1Gap, gmGap);
    }
}

template <typename GlobalData, typename TileData>
__tf__ PTO_INTERNAL void TStoreFractalZ5DValid(
    typename GlobalData::DType __out__* dst, typename TileData::TileDType __in__ src, int64_t gStride0,
    int64_t gStride1, int64_t gStride2, int srcC1, int srcH, int srcW, int srcN, int srcC0, int vC1, int vH, int vW,
    int vN, int vC0)
{
    __cbuf__ typename TileData::DType* srcAddr = (__cbuf__ typename TileData::DType*)__cce_get_tile_ptr(src);
    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    constexpr uint32_t elemBytes = sizeof(typename TileData::DType);

    PTO_ASSERT(vC1 > 0 && vH > 0 && vW > 0 && vN > 0, "Fix: ConvTile valid extents must be positive!");
    PTO_ASSERT(vC1 <= srcC1 && vH <= srcH && vW <= srcW && vN <= srcN, "Fix: ConvTile valid extents exceed declared!");
    PTO_ASSERT(vC0 == srcC0 && srcC0 == int(c0ElemCount), "Fix: valid C0 must stay the full 32-byte C0 block!");
    PTO_ASSERT(vW <= UINT16_MAX && vN <= UINT16_MAX && srcN <= UINT16_MAX, "Fix: valid W/N exceeds the DMA range!");
    PTO_ASSERT(gStride2 >= int64_t(vN) * c0ElemCount, "Fix: destination stride is smaller than the valid N box!");

    const int64_t gStride2Blocks = (gStride2 * elemBytes) >> SHIFT_BLOCK_BYTE;
    const int64_t declHPitch = int64_t(srcW) * srcN * srcC0;
    const uint16_t lenBurst = static_cast<uint16_t>(vN);
    const uint16_t l1Gap = static_cast<uint16_t>(srcN - vN);
    const uint64_t gmGap = static_cast<uint64_t>(gStride2Blocks - vN);

    constexpr int maxBurst = 4095;
    for (int c1 = 0; c1 < vC1; ++c1) {
        for (int h = 0; h < vH; ++h) {
            const int64_t dstBase = int64_t(c1) * gStride0 + int64_t(h) * gStride1;
            const int64_t srcBase = (int64_t(c1) * srcH + h) * declHPitch;
            for (int w0 = 0; w0 < vW; w0 += maxBurst) {
                const int wCount = (vW - w0 > maxBurst) ? maxBurst : (vW - w0);
                TStoreInstrL1ToGm<GlobalData, TileData>(
                    dst + dstBase + int64_t(w0) * gStride2, srcAddr + srcBase + int64_t(w0) * srcN * srcC0,
                    static_cast<uint16_t>(wCount), lenBurst, l1Gap, gmGap);
            }
        }
    }
}

template <typename GlobalData, typename TileData, AtomicType atomicType = AtomicType::AtomicNone>
PTO_INTERNAL void TSTORE_CONVTILE_IMPL(GlobalData& dst, TileData& src)
{
    CheckConvTileDataStore<GlobalData, TileData>(dst, src);
    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);

    if constexpr (atomicType == AtomicType::AtomicAdd) {
        SetAtomicAdd<typename GlobalData::DType>();
    }
    if constexpr (GlobalData::layout == pto::Layout::NC1HWC0) {
        if constexpr (TileData::totalDimCount >= 5) {
            PTO_ASSERT(src.GetValidShape(4) == int(c0ElemCount), "Fix: valid C0 must stay the full 32-byte C0 block!");
        }
        TStore5HDValid<GlobalData, TileData>(
            dst.data(), src.data(), dst.GetStride(0), dst.GetStride(1), dst.GetStride(2), src.GetShape(1),
            src.GetShape(2), src.GetShape(3), src.GetValidShape(0), src.GetValidShape(1), src.GetValidShape(2),
            src.GetValidShape(3));
    } else if constexpr (
        GlobalData::layout == pto::Layout::FRACTAL_Z || GlobalData::layout == pto::Layout::FRACTAL_Z_3D) {
        if constexpr (TileData::totalDimCount == 4) {
            TStoreFractalZ4DValid<GlobalData, TileData>(
                dst.data(), src.data(), dst.GetStride(1), dst.GetStride(2), src.GetShape(1), src.GetShape(2),
                src.GetShape(3), src.GetValidShape(0), src.GetValidShape(1), src.GetValidShape(2),
                src.GetValidShape(3));
        } else {
            TStoreFractalZ5DValid<GlobalData, TileData>(
                dst.data(), src.data(), dst.GetStride(0), dst.GetStride(1), dst.GetStride(2), src.GetShape(0),
                src.GetShape(1), src.GetShape(2), src.GetShape(3), src.GetShape(4), src.GetValidShape(0),
                src.GetValidShape(1), src.GetValidShape(2), src.GetValidShape(3), src.GetValidShape(4));
        }
    } else {
        if constexpr (TileData::totalDimCount >= 6) {
            PTO_ASSERT(src.GetValidShape(5) == int(c0ElemCount), "Fix: valid C0 must stay the full 32-byte C0 block!");
        }
        TStoreNDC1HWC0Valid<GlobalData, TileData>(
            dst.data(), src.data(), dst.GetStride(0), dst.GetStride(1), dst.GetStride(2), dst.GetStride(3),
            src.GetShape(1), src.GetShape(2), src.GetShape(3), src.GetShape(4), src.GetValidShape(0),
            src.GetValidShape(1), src.GetValidShape(2), src.GetValidShape(3), src.GetValidShape(4));
    }
    if constexpr (atomicType == AtomicType::AtomicAdd) {
        SetAtomicNone();
    }
}

template <
    typename TileData, typename GlobalData, AtomicType currentAtomicType = AtomicType::AtomicNone,
    STPhase Phase = STPhase::Unspecified, TStoreL2Hint l2Control = TStoreL2Hint::NormalFirstVictim>
PTO_INTERNAL void TSTORE_IMPL(GlobalData& dst, TileData& src)
{
    if constexpr (is_conv_tile_v<TileData>) {
        TSTORE_CONVTILE_IMPL<GlobalData, TileData, currentAtomicType>(dst, src);
    } else {
        static_assert(
            TileData::Loc == TileType::Vec || TileData::Loc == TileType::Acc || TileData::Loc == TileType::Mat,
            "Source TileType only support Vec/Acc/Mat!");
        PTO_ASSERT(
            dst.GetShape(GlobalTensorDim::DIM_0) > 0 && dst.GetShape(GlobalTensorDim::DIM_1) > 0 &&
                dst.GetShape(GlobalTensorDim::DIM_2) > 0 && dst.GetShape(GlobalTensorDim::DIM_3) > 0 &&
                dst.GetShape(GlobalTensorDim::DIM_4) > 0 && src.GetValidRow() > 0 && src.GetValidCol() > 0,
            "The shape of src and dst must be greater than 0!");
        if constexpr (currentAtomicType == AtomicType::AtomicAdd) {
            SetAtomicAdd<typename GlobalData::DType>();
        }
        if constexpr (TileData::Loc == TileType::Vec) {
            CheckStaticForVecAndMat<TileData, GlobalData>();
            TStore<GlobalData, TileData, l2Control>(
                dst.data(), src.data(), dst.GetShape(GlobalTensorDim::DIM_0), dst.GetShape(GlobalTensorDim::DIM_1),
                dst.GetShape(GlobalTensorDim::DIM_2), dst.GetShape(GlobalTensorDim::DIM_3),
                dst.GetShape(GlobalTensorDim::DIM_4), dst.GetStride(GlobalTensorDim::DIM_0),
                dst.GetStride(GlobalTensorDim::DIM_1), dst.GetStride(GlobalTensorDim::DIM_2),
                dst.GetStride(GlobalTensorDim::DIM_3), dst.GetStride(GlobalTensorDim::DIM_4), src.GetValidRow(),
                src.GetValidCol());
        } else if constexpr (TileData::Loc == TileType::Acc) {
            CheckAcc2gm<TileData, GlobalData, false>(dst, src);
            constexpr QuantMode_t quantMode =
                GetCastPreQuantMode<typename TileData::DType, typename GlobalData::DType>();
            TStoreAcc<GlobalData, TileData, quantMode, ReluPreMode::NoRelu, Phase>(
                dst.data(), src.data(), dst.GetShape(GlobalTensorDim::DIM_0), dst.GetShape(GlobalTensorDim::DIM_1),
                dst.GetShape(GlobalTensorDim::DIM_2), dst.GetShape(GlobalTensorDim::DIM_3),
                dst.GetShape(GlobalTensorDim::DIM_4), dst.GetStride(GlobalTensorDim::DIM_0),
                dst.GetStride(GlobalTensorDim::DIM_1), dst.GetStride(GlobalTensorDim::DIM_2),
                dst.GetStride(GlobalTensorDim::DIM_3), dst.GetStride(GlobalTensorDim::DIM_4), src.GetValidRow(),
                src.GetValidCol());
        } else if constexpr (TileData::Loc == TileType::Mat) {
            CheckStaticForVecAndMat<TileData, GlobalData>();
            TStoreMat<GlobalData, TileData>(
                dst.data(), src.data(), dst.GetShape(GlobalTensorDim::DIM_0), dst.GetShape(GlobalTensorDim::DIM_1),
                dst.GetShape(GlobalTensorDim::DIM_2), dst.GetShape(GlobalTensorDim::DIM_3),
                dst.GetShape(GlobalTensorDim::DIM_4), dst.GetStride(GlobalTensorDim::DIM_0),
                dst.GetStride(GlobalTensorDim::DIM_1), dst.GetStride(GlobalTensorDim::DIM_2),
                dst.GetStride(GlobalTensorDim::DIM_3), dst.GetStride(GlobalTensorDim::DIM_4), src.GetValidRow(),
                src.GetValidCol());
        }
        if constexpr (currentAtomicType == AtomicType::AtomicAdd) {
            SetAtomicNone();
        }
    }
}

template <
    typename TileData, typename GlobalData, AtomicType currentAtomicType = AtomicType::AtomicNone,
    ReluPreMode reluPreMode, STPhase Phase = STPhase::Unspecified,
    TStoreL2Hint l2Control = TStoreL2Hint::NormalFirstVictim>
PTO_INTERNAL void TSTORE_IMPL(GlobalData& dst, TileData& src)
{
    static_assert(TileData::Loc == TileType::Acc, "Source TileType only support Acc!");
    CheckAcc2gm<TileData, GlobalData, false>(dst, src);
    if constexpr (currentAtomicType == AtomicType::AtomicAdd) {
        SetAtomicAdd<typename GlobalData::DType>();
    }
    constexpr QuantMode_t quantMode = GetCastPreQuantMode<typename TileData::DType, typename GlobalData::DType>();
    TStoreAcc<GlobalData, TileData, quantMode, reluPreMode, Phase>(
        dst.data(), src.data(), dst.GetShape(GlobalTensorDim::DIM_0), dst.GetShape(GlobalTensorDim::DIM_1),
        dst.GetShape(GlobalTensorDim::DIM_2), dst.GetShape(GlobalTensorDim::DIM_3),
        dst.GetShape(GlobalTensorDim::DIM_4), dst.GetStride(GlobalTensorDim::DIM_0),
        dst.GetStride(GlobalTensorDim::DIM_1), dst.GetStride(GlobalTensorDim::DIM_2),
        dst.GetStride(GlobalTensorDim::DIM_3), dst.GetStride(GlobalTensorDim::DIM_4), src.GetValidRow(),
        src.GetValidCol());
    if constexpr (currentAtomicType == AtomicType::AtomicAdd) {
        SetAtomicNone();
    }
}

template <
    typename TileData, typename GlobalData, AtomicType currentAtomicType = AtomicType::AtomicNone,
    ReluPreMode reluPreMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified,
    TStoreL2Hint l2Control = TStoreL2Hint::NormalFirstVictim>
PTO_INTERNAL void TSTORE_IMPL(GlobalData& dst, TileData& src, uint64_t preQuantScalar)
{
    static_assert(TileData::Loc == TileType::Acc, "Source TileType only support Acc!");
    CheckAcc2gm<TileData, GlobalData, true>(dst, src);
    if constexpr (currentAtomicType == AtomicType::AtomicAdd) {
        SetAtomicAdd<typename GlobalData::DType>();
    }
    set_quant_pre(preQuantScalar);
    constexpr QuantMode_t quantMode = GetScalarPreQuantMode<typename TileData::DType, typename GlobalData::DType>();
    TStoreAcc<GlobalData, TileData, quantMode, reluPreMode, Phase>(
        dst.data(), src.data(), dst.GetShape(GlobalTensorDim::DIM_0), dst.GetShape(GlobalTensorDim::DIM_1),
        dst.GetShape(GlobalTensorDim::DIM_2), dst.GetShape(GlobalTensorDim::DIM_3),
        dst.GetShape(GlobalTensorDim::DIM_4), dst.GetStride(GlobalTensorDim::DIM_0),
        dst.GetStride(GlobalTensorDim::DIM_1), dst.GetStride(GlobalTensorDim::DIM_2),
        dst.GetStride(GlobalTensorDim::DIM_3), dst.GetStride(GlobalTensorDim::DIM_4), src.GetValidRow(),
        src.GetValidCol());
    if constexpr (AtomicType::AtomicAdd == currentAtomicType) {
        SetAtomicNone();
    }
}

template <
    typename TileData, typename GlobalData, typename FpTileData, AtomicType currentAtomicType = AtomicType::AtomicNone,
    ReluPreMode reluPreMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified,
    TStoreL2Hint l2Control = TStoreL2Hint::NormalFirstVictim>
PTO_INTERNAL void TSTORE_IMPL(GlobalData& dst, TileData& src, FpTileData& fp)
{
    static_assert(TileData::Loc == TileType::Acc, "Source TileType only support Acc!");
    static_assert(FpTileData::Loc == TileType::Scaling, "Fp only support Scaling.");
    CheckAcc2gm<TileData, GlobalData, true>(dst, src);
    if constexpr (AtomicType::AtomicAdd == currentAtomicType) {
        SetAtomicAdd<typename GlobalData::DType>();
    }
    constexpr QuantMode_t quantMode = GetVectorPreQuantMode<typename TileData::DType, typename GlobalData::DType>();
    TStoreAccFp<GlobalData, TileData, FpTileData, quantMode, reluPreMode, Phase>(
        dst.data(), src.data(), fp.data(), dst.GetShape(GlobalTensorDim::DIM_0), dst.GetShape(GlobalTensorDim::DIM_1),
        dst.GetShape(GlobalTensorDim::DIM_2), dst.GetShape(GlobalTensorDim::DIM_3),
        dst.GetShape(GlobalTensorDim::DIM_4), dst.GetStride(GlobalTensorDim::DIM_0),
        dst.GetStride(GlobalTensorDim::DIM_1), dst.GetStride(GlobalTensorDim::DIM_2),
        dst.GetStride(GlobalTensorDim::DIM_3), dst.GetStride(GlobalTensorDim::DIM_4), src.GetValidRow(),
        src.GetValidCol());
    if constexpr (currentAtomicType == AtomicType::AtomicAdd) {
        SetAtomicNone();
    }
}
} // namespace pto
#endif
