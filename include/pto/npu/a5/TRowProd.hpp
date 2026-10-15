/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#ifndef TROWPROD_HPP
#define TROWPROD_HPP

#include "common.hpp"
#include "pto/common/pto_tile.hpp"
#include "pto/common/constants.hpp"
#include <type_traits>

namespace pto {
template <typename TileDataOut, typename TileDataIn>
PTO_INTERNAL void TRowProdCheck(uint32_t srcValidRows, uint32_t srcValidCols, uint32_t dstValidRow)
{
    using T = typename TileDataIn::DType;
    static_assert(
        std::is_same_v<T, half> || std::is_same_v<T, float> || std::is_same_v<T, int32_t> || std::is_same_v<T, int16_t>,
        "TRowProd only supports 'half', 'float', 'int32', or 'int16' data types. "
        "Fix: Define TileDataIn with DType = half, float, int32, or int16.");
    static_assert(
        std::is_same_v<T, typename TileDataOut::DType>, "Input and output tile data types must match. "
                                                        "Fix: Ensure TileDataOut uses the same DType as TileDataIn.");
    static_assert(
        TileDataOut::Loc == pto::TileType::Vec && TileDataIn::Loc == pto::TileType::Vec,
        "TRowProd only works on vector tiles (TileType::Vec). "
        "Fix: Instantiate TileDataIn and TileDataOut with Loc_ = TileType::Vec.");
    static_assert(
        TileDataIn::isRowMajor && !TileDataIn::isBoxedLayout,
        "TRowProd input tile must use standard ND layout (row-major, non-fractal). "
        "Fix: Define TileDataIn with BFractal_ = BLayout::RowMajor and SFractal_ "
        "= SLayout::NoneBox, e.g.,\n"
        "     Tile<TileType::Vec, T, ROWS, COLS, BLayout::RowMajor, ..., "
        "SLayout::NoneBox>");
    static_assert(
        (!TileDataOut::isBoxedLayout &&
         (TileDataOut::isRowMajor || (!TileDataOut::isRowMajor && TileDataOut::Cols == 1))),
        "TRowProd output tile layout must be either:\n"
        "  (a) ND layout: BLayout::RowMajor + SLayout::NoneBox, OR\n"
        "  (b) DN layout with exactly one column: BLayout::ColMajor + "
        "SLayout::NoneBox + Cols=1.\n"
        "Fix: Choose one of the following for TileDataOut:\n"
        "     - Tile<..., ROWS, COLS, BLayout::RowMajor, ValidRows, 1>   // ND\n"
        "     - Tile<..., ROWS, 1, BLayout::ColMajor, ValidRows, 1>  // DN with Cols=1");
    // runtime checks
    PTO_ASSERT(
        srcValidRows != 0 && srcValidCols != 0,
        "TRowProd input source valid rows or columns is zero — TRowProd requires at "
        "least one element per row. "
        "Fix: Ensure srcValidRows > 0 and srcValidCols > 0.");
    PTO_ASSERT(
        srcValidRows == dstValidRow, "TRowProd input and output valid row counts must be equal in TRowProd "
                                     "(row count is preserved). "
                                     "Fix: Pass dstValidRow = srcValidRows.");
}

template <typename TileDataOut, typename TileDataIn>
__tf__ PTO_INTERNAL OP_NAME(TROWPROD) OP_TYPE(reduce) void TRowProd(
    typename TileDataOut::TileDType __out__ dstData, typename TileDataIn::TileDType __in__ srcData, unsigned rows,
    unsigned cols, unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename TileDataIn::DType;
    __ubuf__ T* dst = (__ubuf__ T*)__cce_get_tile_ptr(dstData);
    __ubuf__ T* src = (__ubuf__ T*)__cce_get_tile_ptr(srcData);

    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    uint16_t repeatTimes = CeilDivision(cols, elementsPerRepeat);
    __VEC_SCOPE__
    {
        RegTensor<T> srcReg;
        RegTensor<T> dstReg;
        RegTensor<T> oneReg;
        RegTensor<T> intlvReg1;
        RegTensor<T> intlvReg2;
        vbr(oneReg, 1);

        constexpr auto oneElemMask =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_ONEPT>())>();
        constexpr uint16_t nLoop = (sizeof(T) == 2) ? TROW_PROD_LOOP_B16 : TROW_PROD_LOOP_B32;
        MaskReg pReg;
        uint32_t sreg = cols;
        uint32_t tmp = elementsPerRepeat;
        MaskReg reducePReg = CreatePredicate<T>(tmp);

        for (uint16_t i = 0; i < (uint16_t)rows; ++i) {
            sreg = cols;
            vbr(dstReg, 1);
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; j++) {
                pReg = CreatePredicate<T>(sreg);
                vlds(srcReg, src, i * TileDataIn::RowStride + j * elementsPerRepeat, NORM);
                vmul(dstReg, dstReg, srcReg, pReg, MODE_MERGING);
            }

            for (uint16_t k = 0; k < nLoop; k++) {
                vintlv(intlvReg1, intlvReg2, dstReg, oneReg);
                vmul(dstReg, intlvReg1, intlvReg2, reducePReg, MODE_ZEROING);
            }
            // reducePReg is neglected when oneElemMask mode set
            vsts(dstReg, dst, i * TileDataOut::RowStride, oneElemMask, reducePReg);
        }
    } // end VF
}

// High-precision row product for 16-bit float types: unpack to fp32, multiply
// in fp32 (partial lanes keep their value via MODE_MERGING), reduce the 64 fp32
// lanes with the interleave tree, then round back to the source type once.
template <typename T, unsigned DstRowStride, unsigned SrcRowStride>
PTO_INTERNAL void TRowProd_Fp32Acc(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols)
{
    constexpr unsigned elmPerRpt = CCE_VL / sizeof(float);
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_ONEPT>())>();
    uint16_t repeatTimes = CeilDivision(validCols, elmPerRpt);

    __VEC_SCOPE__
    {
        RegTensor<T> srcB16VReg;
        RegTensor<T> dstB16VReg;
        RegTensor<float> srcF32VReg;
        RegTensor<float> accF32VReg;
        RegTensor<float> oneF32VReg;
        RegTensor<float> intlvReg1;
        RegTensor<float> intlvReg2;
        vbr(oneF32VReg, 1.0f);
        constexpr uint16_t prodLoop = TROW_PROD_LOOP_B32;
        uint32_t reduceItems = elmPerRpt;
        MaskReg reducePReg = CreatePredicate<float>(reduceItems);
        MaskReg pRegOne = pset_b32(PAT_VL1);
        uint32_t dstItems = 1;
        MaskReg pRegDst = CreatePredicate<T>(dstItems);
        for (uint16_t i = 0; i < (uint16_t)validRows; ++i) {
            uint32_t sreg = validCols;
            vbr(accF32VReg, 1.0f);
            for (uint16_t j = 0; j < repeatTimes; ++j) {
                MaskReg pRegB32 = CreatePredicate<float>(sreg);
                vlds(srcB16VReg, src, i * SrcRowStride + j * elmPerRpt, UNPK_B16);
                vcvt(srcF32VReg, srcB16VReg, pRegB32, PART_EVEN);
                vmul(accF32VReg, accF32VReg, srcF32VReg, pRegB32, MODE_MERGING);
            }
            for (uint16_t k = 0; k < prodLoop; ++k) {
                vintlv(intlvReg1, intlvReg2, accF32VReg, oneF32VReg);
                vmul(accF32VReg, intlvReg1, intlvReg2, reducePReg, MODE_ZEROING);
            }
            vcvt(dstB16VReg, accF32VReg, pRegOne, ROUND_R, RS_DISABLE, PART_EVEN);
            vsts(dstB16VReg, dst, i * DstRowStride, distValue, pRegDst);
        }
    } // end VF
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWPROD_IMPL(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp)
{
    unsigned rows = src.GetValidRow();
    unsigned cols = src.GetValidCol();
    TRowProdCheck<TileDataOut, TileDataIn>(rows, cols, dst.GetValidRow());
    using T = typename TileDataIn::DType;
    if constexpr (std::is_same_v<T, half>) {
        // High-precision path: convert fp16 to fp32 and multiply in fp32.
        TRowProd_Fp32Acc<half, TileDataOut::RowStride, TileDataIn::RowStride>(
            (__ubuf__ half*)dst.data(), (__ubuf__ half*)src.data(), rows, cols);
    } else {
        TRowProd<TileDataOut, TileDataIn>(dst.data(), src.data(), rows, cols);
    }
}
} // namespace pto

#endif
