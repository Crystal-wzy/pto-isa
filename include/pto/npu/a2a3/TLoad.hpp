/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TLOAD_HPP
#define TLOAD_HPP

#include <pto/npu/a2a3/l2_cache_hint.hpp>

namespace pto {

template <typename TileData, typename GlobalData>
PTO_INTERNAL void CheckNzFormat(
    int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int validRow, int validCol);

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadInstrGm2ubNative(
    __ubuf__ typename TileData::DType* dst, typename GlobalData::DType* src, uint16_t nBurst, uint32_t lenBurst,
    uint32_t gmGap, uint32_t ubGap, uint32_t ubPad)
{
    if constexpr (sizeof(typename TileData::DType) == 1) {
        copy_gm_to_ubuf_align_b8(dst, src, 0, nBurst, lenBurst, 0, ubPad, gmGap, ubGap);
    } else if constexpr (sizeof(typename TileData::DType) == 2) {
        copy_gm_to_ubuf_align_b16(dst, src, 0, nBurst, lenBurst, 0, ubPad, gmGap, ubGap);
    } else if constexpr (sizeof(typename TileData::DType) == 4) {
        copy_gm_to_ubuf_align_b32(dst, src, 0, nBurst, lenBurst, 0, ubPad, gmGap, ubGap);
    } else if constexpr (sizeof(typename TileData::DType) == 8) {
        copy_gm_to_ubuf_align_b32(dst, src, 0, nBurst, lenBurst, 0, ubPad * 2, gmGap, ubGap);
    }
}

#include <pto/common/utils.hpp>

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadInstrGm2L1(
    __cbuf__ typename TileData::DType* dst, typename GlobalData::DType* src, uint16_t nBurst, uint16_t lenBurst,
    uint64_t gmGap, uint16_t l1Gap)
{
    if (gmGap <= UINT16_MAX) {
        pto_copy_gm_to_cbuf(dst, src, (uint8_t)0, nBurst, lenBurst, static_cast<uint16_t>(gmGap), l1Gap);
        return;
    }
    // Advance full-width addresses when the gap cannot be encoded by the DMA instruction.
    constexpr uint32_t BLOCK_ELEMENTS = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    for (uint16_t i = 0; i < nBurst; i++) {
        pto_copy_gm_to_cbuf(
            dst + uint64_t(i) * (lenBurst + l1Gap) * BLOCK_ELEMENTS,
            src + uint64_t(i) * (lenBurst + gmGap) * BLOCK_ELEMENTS, (uint8_t)0, 1, lenBurst, 0, 0);
    }
}

template <typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadNDC1HWC0(
    typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__* src, int srcN, int srcD, int srcC1,
    int srcH, int srcW, int64_t gStride0, int64_t gStride1, int64_t gStride2, int64_t gStride3, int64_t gStride4,
    int dstN, int dstD, int dstC1, int dstH, int dstW)
{
    __cbuf__ typename TileData::DType* dstAddr = (__cbuf__ typename TileData::DType*)__cce_get_tile_ptr(dst);
    typename GlobalData::DType* srcAddr = src;

    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    typename GlobalData::DType* srcAddrP = srcAddr;
    __cbuf__ typename TileData::DType* dstAddrP = dstAddr;
    constexpr uint32_t maxSupportBurst = 4095;
    uint64_t gmGap = ((gStride2 - dstH * dstW * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;

    if ((gStride3 == dstW * c0ElemCount || dstH == 1) && gmGap <= UINT16_MAX && dstC1 <= maxSupportBurst &&
        dstH * dstW <= UINT16_MAX) {
        uint16_t nBurst = dstC1;
        uint64_t srcGap = gmGap;
        uint16_t lenBurst = dstH * dstW;
        for (uint32_t i = 0; i < dstN; i++) {
            int64_t srcAddr1 = i * gStride0;
            int64_t dstAddr1 = i * dstD * dstH * dstW * dstC1 * c0ElemCount;
            for (uint32_t j = 0; j < dstD; j++) {
                srcAddrP = srcAddr + srcAddr1 + j * gStride1;
                dstAddrP = dstAddr + dstAddr1 + j * dstH * dstW * dstC1 * c0ElemCount;
                TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, srcGap, 0);
            }
        }
    } else {
        PTO_ASSERT(dstH <= maxSupportBurst, "Fix: max support dstH is 4095!");
        PTO_ASSERT(dstW <= UINT16_MAX, "Fix: max support dstW is UINT16_MAX!");

        uint16_t nBurst = dstH;
        uint16_t lenBurst = dstW;
        uint64_t srcGap = ((gStride3 - srcW * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
        uint16_t l1Gap = 0;
        for (uint32_t i = 0; i < dstN; i++) {
            int64_t srcAddr1 = i * gStride0;
            int64_t dstAddr1 = i * dstD * dstH * dstW * dstC1 * c0ElemCount;
            for (uint32_t j = 0; j < dstD; j++) {
                int64_t srcAddr2 = j * gStride1;
                int64_t dstAddr2 = j * dstH * dstW * dstC1 * c0ElemCount;
                for (uint32_t k = 0; k < dstC1; k++) {
                    srcAddrP = srcAddr + srcAddr1 + srcAddr2 + k * gStride2;
                    dstAddrP = dstAddr + dstAddr1 + dstAddr2 + k * dstH * dstW * c0ElemCount;
                    TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, srcGap, l1Gap);
                }
            }
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2L1Nd2nd(
    __cbuf__ typename TileData::DType* dstAddr, typename GlobalData::DType* srcAddr, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int64_t gStride0, int64_t gStride1, int64_t gStride2, int64_t gStride3,
    int64_t gStride4, int validRow, int validCol)
{
    PTO_ASSERT(
        gShape4 * sizeof(typename TileData::DType) % BLOCK_BYTE_SIZE == 0,
        "The 5th dim of ND shape must be 32 bytes aligned!");
    PTO_ASSERT(validCol == gShape4, "The validCol of TileData must be equal to the 5th dim(Shape4) of ND shape!");
    PTO_ASSERT(
        validRow == gShape0 * gShape1 * gShape2 * gShape3,
        "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape3) of ND shape!");
    PTO_ASSERT(gShape3 < 4096, "The gshape3 (which equals nBurst) must be less than 4096 for A2/A3");
    uint16_t nBurst = gShape3;
    uint16_t lenBurst = (validCol * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint64_t gmGap = ((gStride3 - gShape4) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint16_t l1Gap = ((TileData::Cols - validCol) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;

    int64_t dstStride2 = gShape3 * TileData::Cols;
    int64_t dstStride1 = gShape2 * dstStride2;
    int64_t dstStride0 = gShape1 * dstStride1;
    typename GlobalData::DType* srcAddrP = srcAddr;
    __cbuf__ typename TileData::DType* dstAddrP = dstAddr;
    for (uint32_t i = 0; i < gShape0; i++) {
        int64_t srcAddr0 = i * gStride0;
        int64_t dstAddr0 = i * dstStride0;
        for (uint32_t j = 0; j < gShape1; j++) {
            int64_t dstAddr1 = j * dstStride1;
            int64_t srcAddr1 = j * gStride1;
            for (uint32_t k = 0; k < gShape2; k++) {
                srcAddrP = srcAddr + srcAddr0 + srcAddr1 + k * gStride2;
                dstAddrP = dstAddr + dstAddr0 + dstAddr1 + k * dstStride2;
                TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, l1Gap);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2L1Dn2dn(
    __cbuf__ typename TileData::DType* dstAddr, typename GlobalData::DType* srcAddr, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int64_t gStride0, int64_t gStride1, int64_t gStride2, int64_t gStride3,
    int64_t gStride4, int validRow, int validCol)
{
    PTO_ASSERT(
        gShape3 * sizeof(typename TileData::DType) % BLOCK_BYTE_SIZE == 0,
        "The 4th dim of DN shape must be 32 bytes aligned!");
    PTO_ASSERT(validRow == gShape3, "The validCol of TileData must be equal to the 4th dim(Shape3) of DN shape!");
    PTO_ASSERT(
        validCol == gShape0 * gShape1 * gShape2 * gShape4,
        "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape4) of DN shape!");
    PTO_ASSERT(gShape4 < 4096, "The gshape4 (which equals nBurst) must be less than 4096 for A2/A3");
    uint16_t nBurst = gShape4;
    uint16_t lenBurst = (validRow * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint64_t gmGap = ((gStride4 - gShape3) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint16_t l1Gap = ((TileData::Rows - gShape3) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    __cbuf__ typename TileData::DType* dstAddrP = dstAddr;
    typename GlobalData::DType* srcAddrP = srcAddr;

    int64_t dstStride2 = gShape4 * TileData::Rows;
    int64_t dstStride1 = gShape2 * dstStride2;
    int64_t dstStride0 = gShape1 * dstStride1;
    for (uint32_t i = 0; i < gShape0; i++) {
        int64_t dstAddr0 = i * dstStride0;
        int64_t srcAddr0 = i * gStride0;
        for (uint32_t j = 0; j < gShape1; j++) {
            int64_t srcAddr1 = j * gStride1;
            int64_t dstAddr1 = j * dstStride1;
            for (uint32_t k = 0; k < gShape2; k++) {
                srcAddrP = srcAddr + srcAddr0 + srcAddr1 + k * gStride2;
                dstAddrP = dstAddr + dstAddr0 + dstAddr1 + k * dstStride2;
                TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, l1Gap);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2L1Nz2nz(
    __cbuf__ typename TileData::DType* dstAddr, typename GlobalData::DType* srcAddr, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int64_t gStride0, int64_t gStride1, int64_t gStride2, int64_t gStride3,
    int64_t gStride4, int validRow, int validCol)
{
    CheckNzFormat<TileData, GlobalData>(gShape0, gShape1, gShape2, gShape3, gShape4, validRow, validCol);
    uint16_t nBurst = gShape1;
    uint32_t lenBurst = validRow;
    uint64_t gmGap = ((int64_t(gStride1) - int64_t(gShape2) * gShape3 * gShape4) * sizeof(typename TileData::DType)) >>
                     SHIFT_BLOCK_BYTE;
    uint32_t l1Gap = TileData::Rows - validRow;
    typename GlobalData::DType* srcAddrP = srcAddr;
    __cbuf__ typename TileData::DType* dstAddrP = dstAddr;
    int64_t burstStride = int64_t(TileData::Rows) * gShape4;
    int64_t tileStride = burstStride * gShape1;

    for (uint32_t i = 0; i < gShape0; i++) {
        srcAddrP = srcAddr + int64_t(i) * gStride0;
        dstAddrP = dstAddr + i * tileStride;
        TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, l1Gap);
    }
}

template <typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoad5HD(
    typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__* src, int srcN, int srcC1, int srcH,
    int srcW, int64_t gStride0, int64_t gStride1, int64_t gStride2, int64_t gStride3, int64_t gStride4, int dstN,
    int dstC1, int dstH, int dstW)
{
    __cbuf__ typename TileData::DType* dstAddr = (__cbuf__ typename TileData::DType*)__cce_get_tile_ptr(dst);
    typename GlobalData::DType* srcAddr = src;

    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    typename GlobalData::DType* srcAddrP = srcAddr;
    __cbuf__ typename TileData::DType* dstAddrP = dstAddr;
    constexpr uint32_t maxSupportBurst = 4095;
    uint64_t gmGap = ((gStride1 - dstH * dstW * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;

    if ((gStride2 == dstW * c0ElemCount || dstH == 1) && gmGap <= UINT16_MAX && dstC1 <= maxSupportBurst &&
        dstH * dstW <= UINT16_MAX) {
        uint16_t nBurst = dstC1;
        uint64_t srcGap = gmGap;
        uint16_t lenBurst = dstH * dstW;
        for (uint32_t i = 0; i < dstN; i++) {
            srcAddrP = srcAddr + i * gStride0;
            dstAddrP = dstAddr + i * dstH * dstW * dstC1 * c0ElemCount;
            TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, srcGap, 0);
        }
    } else {
        PTO_ASSERT(dstH <= maxSupportBurst, "Fix: max support dstH is 4095!");
        PTO_ASSERT(dstW <= UINT16_MAX, "Fix: max support dstW is UINT16_MAX!");

        uint16_t nBurst = dstH;
        uint16_t lenBurst = dstW;
        uint64_t srcGap = ((gStride2 - srcW * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
        uint16_t l1Gap = 0;
        for (uint32_t i = 0; i < dstN; i++) {
            int64_t dstAddr1 = i * dstH * dstW * dstC1 * c0ElemCount;
            int64_t srcAddr1 = i * gStride0;
            for (uint32_t j = 0; j < dstC1; j++) {
                srcAddrP = srcAddr + srcAddr1 + j * gStride1;
                dstAddrP = dstAddr + dstAddr1 + j * dstH * dstW * c0ElemCount;
                TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, srcGap, l1Gap);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadFractalZ(
    typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__* src, int srcShape0, int srcShape1,
    int srcShape2, int srcShape3, int srcShape4, int64_t gStride0, int64_t gStride1, int64_t gStride2, int64_t gStride3,
    int64_t gStride4, int dstShape0, int dstShape1, int dstShape2, int dstShape3)
{
    __cbuf__ typename TileData::DType* dstAddr = (__cbuf__ typename TileData::DType*)__cce_get_tile_ptr(dst);
    typename GlobalData::DType* srcAddr = src;

    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    typename GlobalData::DType* srcAddrP = srcAddr;
    __cbuf__ typename TileData::DType* dstAddrP = dstAddr;

    if constexpr (TileData::totalDimCount == 4) {
        static_assert(
            TileData::staticShape[2] == FRACTAL_NZ_ROW && TileData::staticShape[3] == c0ElemCount,
            "Fix: The TileData last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");
        static_assert(
            GlobalData::staticShape[3] == FRACTAL_NZ_ROW && GlobalData::staticShape[4] == c0ElemCount,
            "Fix: The GlobalTensor last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");

        uint16_t nBurst = dstShape0;
        uint16_t lenBurst = dstShape1 * dstShape2;
        uint64_t gmGap =
            ((gStride1 - srcShape2 * srcShape3 * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
        TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, 0);

    } else {
        PTO_ASSERT(
            srcShape1 == dstShape1 && srcShape2 == dstShape2,
            "Fix: layout is Fractal_Z, [srcH,srcW] && [dstH,dstW] should be same!");
        PTO_ASSERT(dstShape3 <= UINT16_MAX, "Fix: max support dstN is UINT16_MAX!");

        uint16_t lenBurst = dstShape3;
        uint64_t gmGap = ((gStride2 - srcShape3 * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
        constexpr uint32_t maxSupportBurst = 4095;

        if (dstShape0 * dstShape1 * dstShape2 <= maxSupportBurst) {
            uint16_t nBurst = dstShape0 * dstShape1 * dstShape2;
            TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, 0);
        } else {
            uint16_t nBurst = dstShape1 * dstShape2;
            for (uint32_t i = 0; i < dstShape0; i++) {
                srcAddrP = srcAddr + i * gStride0;
                dstAddrP = dstAddr + i * dstShape1 * dstShape2 * dstShape3 * c0ElemCount;
                TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, 0);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void CheckConvTileData(TileData& dst, GlobalData& src)
{
    static_assert(
        std::is_same_v<typename TileData::DType, int8_t> || std::is_same_v<typename TileData::DType, uint8_t> ||
            std::is_same_v<typename TileData::DType, int16_t> || std::is_same_v<typename TileData::DType, uint16_t> ||
            std::is_same_v<typename TileData::DType, int32_t> || std::is_same_v<typename TileData::DType, uint32_t> ||
            std::is_same_v<typename TileData::DType, half> || std::is_same_v<typename TileData::DType, bfloat16_t> ||
            std::is_same_v<typename TileData::DType, float>,
        "Fix: Data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/half/bfloat16_t/float!");
    static_assert(TileData::Loc == pto::TileType::Mat, "Fix: Dst TileType must be Mat!");
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
}

// -----------------------------------------------------------------------------
// ConvTile TLOAD with a cropped ValidShape.
//
// Valid extents bound the transferred sub-box (anchored at the logical origin) and drive the DMA loop
// counts / burst lengths; the declared Shape still provides every destination base and pitch (l1Gap) and
// the source strides still provide every GM address (gmGap). The bytes outside the valid box are left
// untouched. DMA transfers move whole 32-byte blocks, so the valid C0 extent must stay the full C0.
// -----------------------------------------------------------------------------
// Runtime comparison, needed only when a declared or valid extent is dynamic.
template <typename TileData>
PTO_INTERNAL bool ConvTileValidEqualsDeclared(const TileData& dst)
{
    for (int i = 0; i < TileData::totalDimCount; ++i) {
        if (dst.GetValidShape(i) != dst.GetShape(i)) {
            return false;
        }
    }
    return true;
}

// Compile-time classification of the valid box relative to the declared shape. Static extents are read
// from the type (staticValidShape/staticShape), so fully static tiles never pay a runtime comparison.
enum class ConvTileValidStaticKind {
    Full,    // every valid extent statically equals its declared extent -> original path
    Cropped, // every extent is static and at least one differs -> valid path, no runtime check
    Runtime, // at least one declared or valid extent is DYNAMIC -> compare at runtime
};

template <typename TileData>
PTO_INTERNAL constexpr ConvTileValidStaticKind GetConvTileValidStaticKind()
{
    if constexpr (TileData::dynamicDimCount != 0 || TileData::validDynamicDimCount != 0) {
        return ConvTileValidStaticKind::Runtime;
    }
    for (int i = 0; i < TileData::totalDimCount; ++i) {
        if (TileData::staticValidShape[i] != TileData::staticShape[i]) {
            return ConvTileValidStaticKind::Cropped;
        }
    }
    return ConvTileValidStaticKind::Full;
}

template <typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoad5HDValid(
    typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__* src, int64_t gStride0,
    int64_t gStride1, int64_t gStride2, int dstC1, int dstH, int dstW, int vN, int vC1, int vH, int vW)
{
    __cbuf__ typename TileData::DType* dstAddr = (__cbuf__ typename TileData::DType*)__cce_get_tile_ptr(dst);
    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    constexpr uint32_t elemBytes = sizeof(typename TileData::DType);

    PTO_ASSERT(vN > 0 && vC1 > 0 && vH > 0 && vW > 0, "Fix: ConvTile valid extents must be positive!");
    PTO_ASSERT(vW <= dstW && vH <= dstH, "Fix: ConvTile valid W/H must not exceed the declared extent!");
    PTO_ASSERT(
        gStride1 >= int64_t(vH) * vW * c0ElemCount && gStride2 >= int64_t(vW) * c0ElemCount,
        "Fix: source stride is smaller than the valid H*W box!");

    const int64_t gStride1Blocks = (gStride1 * elemBytes) >> SHIFT_BLOCK_BYTE;
    const int64_t gStride2Blocks = (gStride2 * elemBytes) >> SHIFT_BLOCK_BYTE;
    const int64_t declPlane = int64_t(dstH) * dstW;         // 32B blocks in one declared H*W plane
    const int64_t validPlane = int64_t(vH) * vW;            // 32B blocks in one valid H*W plane
    const int64_t declPlaneElems = declPlane * c0ElemCount; // elements in one declared H*W plane
    const int64_t declNPitch = int64_t(dstC1) * declPlaneElems;

    typename GlobalData::DType* srcAddrP = src;
    __cbuf__ typename TileData::DType* dstAddrP = dstAddr;
    const bool fastPath = (vH == 1 || (vW == dstW && gStride2 == int64_t(vW) * c0ElemCount)) && vC1 <= 4095 &&
                          validPlane <= UINT16_MAX && declPlane <= UINT16_MAX;
    if (fastPath) {
        const uint16_t nBurst = static_cast<uint16_t>(vC1);
        const uint16_t lenBurst = static_cast<uint16_t>(validPlane);
        const uint64_t gmGap = static_cast<uint64_t>(gStride1Blocks - validPlane);
        const uint16_t l1Gap = static_cast<uint16_t>(declPlane - validPlane);
        for (int n = 0; n < vN; ++n) {
            TLoadInstrGm2L1<TileData, GlobalData>(
                dstAddrP + int64_t(n) * declNPitch, srcAddrP + int64_t(n) * gStride0, nBurst, lenBurst, gmGap, l1Gap);
        }
        return;
    }

    PTO_ASSERT(vH <= 4095 && vW <= UINT16_MAX && dstW <= UINT16_MAX, "Fix: valid H/W exceeds the DMA range!");
    const uint16_t nBurst = static_cast<uint16_t>(vH);
    const uint16_t lenBurst = static_cast<uint16_t>(vW);
    const uint64_t gmGap = static_cast<uint64_t>(gStride2Blocks - vW);
    const uint16_t l1Gap = static_cast<uint16_t>(dstW - vW);
    for (int n = 0; n < vN; ++n) {
        for (int c1 = 0; c1 < vC1; ++c1) {
            TLoadInstrGm2L1<TileData, GlobalData>(
                dstAddrP + int64_t(n) * declNPitch + int64_t(c1) * declPlaneElems,
                srcAddrP + int64_t(n) * gStride0 + int64_t(c1) * gStride1, nBurst, lenBurst, gmGap, l1Gap);
        }
    }
}

template <typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadNDC1HWC0Valid(
    typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__* src, int64_t gStride0,
    int64_t gStride1, int64_t gStride2, int64_t gStride3, int dstD, int dstC1, int dstH, int dstW, int vN, int vD,
    int vC1, int vH, int vW)
{
    __cbuf__ typename TileData::DType* dstAddr = (__cbuf__ typename TileData::DType*)__cce_get_tile_ptr(dst);
    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    constexpr uint32_t elemBytes = sizeof(typename TileData::DType);

    PTO_ASSERT(vN > 0 && vD > 0 && vC1 > 0 && vH > 0 && vW > 0, "Fix: ConvTile valid extents must be positive!");
    PTO_ASSERT(vW <= dstW && vH <= dstH && vD <= dstD && vC1 <= dstC1, "Fix: valid extent exceeds declared!");
    PTO_ASSERT(
        gStride2 >= int64_t(vH) * vW * c0ElemCount && gStride3 >= int64_t(vW) * c0ElemCount,
        "Fix: source stride is smaller than the valid H*W box!");

    const int64_t gStride2Blocks = (gStride2 * elemBytes) >> SHIFT_BLOCK_BYTE;
    const int64_t gStride3Blocks = (gStride3 * elemBytes) >> SHIFT_BLOCK_BYTE;
    const int64_t declPlane = int64_t(dstH) * dstW;
    const int64_t validPlane = int64_t(vH) * vW;
    const int64_t declPlaneElems = declPlane * c0ElemCount;
    const int64_t declDpitch = int64_t(dstC1) * declPlaneElems;
    const int64_t declNPitch = int64_t(dstD) * declDpitch;

    typename GlobalData::DType* srcAddrP = src;
    __cbuf__ typename TileData::DType* dstAddrP = dstAddr;
    const bool fastPath = (vH == 1 || (vW == dstW && gStride3 == int64_t(vW) * c0ElemCount)) && vC1 <= 4095 &&
                          validPlane <= UINT16_MAX && declPlane <= UINT16_MAX;
    if (fastPath) {
        const uint16_t nBurst = static_cast<uint16_t>(vC1);
        const uint16_t lenBurst = static_cast<uint16_t>(validPlane);
        const uint64_t gmGap = static_cast<uint64_t>(gStride2Blocks - validPlane);
        const uint16_t l1Gap = static_cast<uint16_t>(declPlane - validPlane);
        for (int n = 0; n < vN; ++n) {
            for (int d = 0; d < vD; ++d) {
                TLoadInstrGm2L1<TileData, GlobalData>(
                    dstAddrP + int64_t(n) * declNPitch + int64_t(d) * declDpitch,
                    srcAddrP + int64_t(n) * gStride0 + int64_t(d) * gStride1, nBurst, lenBurst, gmGap, l1Gap);
            }
        }
        return;
    }

    PTO_ASSERT(vH <= 4095 && vW <= UINT16_MAX && dstW <= UINT16_MAX, "Fix: valid H/W exceeds the DMA range!");
    const uint16_t nBurst = static_cast<uint16_t>(vH);
    const uint16_t lenBurst = static_cast<uint16_t>(vW);
    const uint64_t gmGap = static_cast<uint64_t>(gStride3Blocks - vW);
    const uint16_t l1Gap = static_cast<uint16_t>(dstW - vW);
    for (int n = 0; n < vN; ++n) {
        for (int d = 0; d < vD; ++d) {
            for (int c1 = 0; c1 < vC1; ++c1) {
                TLoadInstrGm2L1<TileData, GlobalData>(
                    dstAddrP + int64_t(n) * declNPitch + int64_t(d) * declDpitch + int64_t(c1) * declPlaneElems,
                    srcAddrP + int64_t(n) * gStride0 + int64_t(d) * gStride1 + int64_t(c1) * gStride2, nBurst, lenBurst,
                    gmGap, l1Gap);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadFractalZ4DValid(
    typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__* src, int64_t gStride1,
    int64_t gStride2, int dstD1, int dstD2, int dstD3, int v0, int v1, int v2, int v3)
{
    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    constexpr uint32_t elemBytes = sizeof(typename TileData::DType);
    static_assert(
        TileData::staticShape[2] == FRACTAL_NZ_ROW && TileData::staticShape[3] == c0ElemCount,
        "Fix: The TileData last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");
    static_assert(
        GlobalData::staticShape[3] == FRACTAL_NZ_ROW && GlobalData::staticShape[4] == c0ElemCount,
        "Fix: The GlobalTensor last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");

    __cbuf__ typename TileData::DType* dstAddr = (__cbuf__ typename TileData::DType*)__cce_get_tile_ptr(dst);

    PTO_ASSERT(v0 > 0 && v1 > 0 && v2 > 0, "Fix: ConvTile valid extents must be positive!");
    PTO_ASSERT(v1 <= dstD1 && v2 <= dstD2, "Fix: ConvTile valid N1/N0 must not exceed the declared extent!");
    PTO_ASSERT(v3 == dstD3 && dstD3 == int(c0ElemCount), "Fix: valid C0 must stay the full 32-byte C0 block!");
    PTO_ASSERT(v1 <= UINT16_MAX && v2 <= UINT16_MAX, "Fix: valid N1/N0 exceeds the DMA range!");
    PTO_ASSERT(gStride2 >= int64_t(v2) * c0ElemCount, "Fix: source stride is smaller than the valid N0 box!");

    const int64_t gStride2Blocks = (gStride2 * elemBytes) >> SHIFT_BLOCK_BYTE;
    const int64_t declPlaneElems = int64_t(dstD1) * dstD2 * c0ElemCount;
    const uint16_t nBurst = static_cast<uint16_t>(v1);
    const uint16_t lenBurst = static_cast<uint16_t>(v2);
    const uint64_t gmGap = static_cast<uint64_t>(gStride2Blocks - v2);
    const uint16_t l1Gap = static_cast<uint16_t>(dstD2 - v2);

    for (int i0 = 0; i0 < v0; ++i0) {
        TLoadInstrGm2L1<TileData, GlobalData>(
            dstAddr + int64_t(i0) * declPlaneElems, src + int64_t(i0) * gStride1, nBurst, lenBurst, gmGap, l1Gap);
    }
}

template <typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadFractalZ5DValid(
    typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__* src, int64_t gStride0,
    int64_t gStride1, int64_t gStride2, int dstH, int dstW, int dstN, int dstC0, int vC1, int vH, int vW, int vN,
    int vC0)
{
    __cbuf__ typename TileData::DType* dstAddr = (__cbuf__ typename TileData::DType*)__cce_get_tile_ptr(dst);
    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    constexpr uint32_t elemBytes = sizeof(typename TileData::DType);

    PTO_ASSERT(vC1 > 0 && vH > 0 && vW > 0 && vN > 0, "Fix: ConvTile valid extents must be positive!");
    PTO_ASSERT(vH <= dstH && vW <= dstW && vN <= dstN, "Fix: ConvTile valid H/W/N must not exceed declared!");
    PTO_ASSERT(vC0 == dstC0 && dstC0 == int(c0ElemCount), "Fix: valid C0 must stay the full 32-byte C0 block!");
    PTO_ASSERT(vW <= UINT16_MAX && vN <= UINT16_MAX && dstN <= UINT16_MAX, "Fix: valid W/N exceeds the DMA range!");
    PTO_ASSERT(gStride2 >= int64_t(vN) * c0ElemCount, "Fix: source stride is smaller than the valid N box!");

    const int64_t gStride2Blocks = (gStride2 * elemBytes) >> SHIFT_BLOCK_BYTE;
    const int64_t declHPitch = int64_t(dstW) * dstN * dstC0;
    const uint16_t lenBurst = static_cast<uint16_t>(vN);
    const uint64_t gmGap = static_cast<uint64_t>(gStride2Blocks - vN);
    const uint16_t l1Gap = static_cast<uint16_t>(dstN - vN);

    constexpr int maxBurst = 4095;
    for (int c1 = 0; c1 < vC1; ++c1) {
        for (int h = 0; h < vH; ++h) {
            const int64_t srcBase = int64_t(c1) * gStride0 + int64_t(h) * gStride1;
            const int64_t dstBase = int64_t(c1) * dstH * declHPitch + int64_t(h) * declHPitch;
            for (int w0 = 0; w0 < vW; w0 += maxBurst) {
                const int wCount = (vW - w0 > maxBurst) ? maxBurst : (vW - w0);
                TLoadInstrGm2L1<TileData, GlobalData>(
                    dstAddr + dstBase + int64_t(w0) * dstN * dstC0, src + srcBase + int64_t(w0) * gStride2,
                    static_cast<uint16_t>(wCount), lenBurst, gmGap, l1Gap);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLOAD_CONVTILE_VALID_IMPL(TileData& dst, typename GlobalData::DType __in__* srcAddr, GlobalData& src)
{
    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    if constexpr (GlobalData::layout == pto::Layout::NC1HWC0) {
        if constexpr (TileData::totalDimCount >= 5) {
            PTO_ASSERT(dst.GetValidShape(4) == int(c0ElemCount), "Fix: valid C0 must stay the full 32-byte C0 block!");
        }
        TLoad5HDValid<TileData, GlobalData>(
            dst.data(), srcAddr, src.GetStride(0), src.GetStride(1), src.GetStride(2), dst.GetShape(1), dst.GetShape(2),
            dst.GetShape(3), dst.GetValidShape(0), dst.GetValidShape(1), dst.GetValidShape(2), dst.GetValidShape(3));
    } else if constexpr (
        GlobalData::layout == pto::Layout::FRACTAL_Z || GlobalData::layout == pto::Layout::FRACTAL_Z_3D) {
        if constexpr (TileData::totalDimCount == 4) {
            TLoadFractalZ4DValid<TileData, GlobalData>(
                dst.data(), srcAddr, src.GetStride(1), src.GetStride(2), dst.GetShape(1), dst.GetShape(2),
                dst.GetShape(3), dst.GetValidShape(0), dst.GetValidShape(1), dst.GetValidShape(2),
                dst.GetValidShape(3));
        } else {
            TLoadFractalZ5DValid<TileData, GlobalData>(
                dst.data(), srcAddr, src.GetStride(0), src.GetStride(1), src.GetStride(2), dst.GetShape(1),
                dst.GetShape(2), dst.GetShape(3), dst.GetShape(4), dst.GetValidShape(0), dst.GetValidShape(1),
                dst.GetValidShape(2), dst.GetValidShape(3), dst.GetValidShape(4));
        }
    } else if constexpr (GlobalData::layout == pto::Layout::NDC1HWC0) {
        if constexpr (TileData::totalDimCount >= 6) {
            PTO_ASSERT(dst.GetValidShape(5) == int(c0ElemCount), "Fix: valid C0 must stay the full 32-byte C0 block!");
        }
        TLoadNDC1HWC0Valid<TileData, GlobalData>(
            dst.data(), srcAddr, src.GetStride(0), src.GetStride(1), src.GetStride(2), src.GetStride(3),
            dst.GetShape(1), dst.GetShape(2), dst.GetShape(3), dst.GetShape(4), dst.GetValidShape(0),
            dst.GetValidShape(1), dst.GetValidShape(2), dst.GetValidShape(3), dst.GetValidShape(4));
    }
}

#include "pto/common/arch/memory/tload_common.hpp"

template <TLoadL2Hint l2Control = TLoadL2Hint::NormalFirstVictim, typename TileData, typename GlobalData>
PTO_INTERNAL void TLOAD_IMPL(TileData& dst, GlobalData& src)
{
    if constexpr (is_conv_tile_v<TileData>) {
        TLOAD_CONVTILE_IMPL<TileData, GlobalData, l2Control>(dst, src);
    } else {
        TLOAD_TILE_IMPL<TileData, GlobalData, l2Control>(dst, src);
    }
}

} // namespace pto
#endif // TLOAD_HPP
