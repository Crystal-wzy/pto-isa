/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCOLPROD_HPP
#define TCOLPROD_HPP

#include "TColReduceOps.hpp"

namespace pto {
template <typename T>
struct TColProdOp {
    using PadType = T;
    static constexpr T InitVal = (T)1;
    PTO_INTERNAL static void ReduceInstr(RegTensor<T>& dst, RegTensor<T>& src0, RegTensor<T>& src1, MaskReg& pReg)
    {
        vmul(dst, src0, src1, pReg, MODE_ZEROING);
    }
};

// High-precision column product for 16-bit float types: unpack each row to
// fp32, multiply in fp32, then round back to the source type once.
template <typename T, unsigned SrcStride>
PTO_INTERNAL void TColProd_Fp32Acc(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRow, unsigned validCol)
{
    constexpr unsigned elmPerRpt = CCE_VL / sizeof(float);
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    uint16_t repeatTimes = CeilDivision(validCol, elmPerRpt);
    uint16_t rows = validRow;
    uint32_t sreg = validCol;

    __VEC_SCOPE__
    {
        RegTensor<T> srcB16VReg;
        RegTensor<T> dstB16VReg;
        RegTensor<float> srcF32VReg;
        RegTensor<float> dstF32VReg;
        MaskReg pRegB32;
        MaskReg pRegB16;

        for (uint16_t i = 0; i < repeatTimes; ++i) {
            uint16_t colOffset = i * elmPerRpt;
            pRegB32 = CreatePredicate<float>(sreg);
            vbr(dstF32VReg, 1.0f);

            for (uint16_t j = 0; j < rows; ++j) {
                vlds(srcB16VReg, src + j * SrcStride, colOffset, UNPK_B16);
                vcvt(srcF32VReg, srcB16VReg, pRegB32, PART_EVEN);
                vmul(dstF32VReg, dstF32VReg, srcF32VReg, pRegB32, MODE_ZEROING);
            }

            vcvt(dstB16VReg, dstF32VReg, pRegB32, ROUND_R, RS_DISABLE, PART_EVEN);
            vpack((RegTensor<uint16_t>&)dstB16VReg, (RegTensor<uint32_t>&)dstB16VReg, LOWER);
            ppack(pRegB16, pRegB32, LOWER);
            vsts(dstB16VReg, dst, colOffset, distValue, pRegB16);
        }
    } // end VF
}

template <typename T, typename TileDataOut, typename TileDataIn>
__tf__ PTO_INTERNAL void TColProd(
    typename TileDataOut::TileDType __out__ dstData, typename TileDataIn::TileDType __in__ srcData, unsigned validRow,
    unsigned validCol, unsigned version)
{
    __ubuf__ T* dst = (__ubuf__ T*)__cce_get_tile_ptr(dstData);
    __ubuf__ T* src = (__ubuf__ T*)__cce_get_tile_ptr(srcData);

    TColReduceInstr<TColProdOp<T>, T, TileDataIn>(dst, src, validRow, validCol, version);
}

template <typename TileDataOut, typename TileDataIn>
PTO_INTERNAL void TCOLPROD_IMPL(TileDataOut& dst, TileDataIn& src)
{
    using T = typename TileDataIn::DType;
    static_assert(
        std::is_same_v<T, float> || std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t> ||
            std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, int32_t> ||
            std::is_same_v<T, uint32_t>,
        "Fix: TCOLPROD input data type supports only int16_t/uint16_t/half/bfloat16_t/int32_t/uint32_t/float.");

    unsigned validCol = src.GetValidCol();
    unsigned validRow = src.GetValidRow();
    if (validCol == 0 || validRow == 0) {
        return;
    }

    TColReduceCheck<TileDataOut, TileDataIn>(validRow, validCol, dst.GetValidCol());

    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
        // High-precision path: convert fp16/bf16 to fp32 and multiply in fp32.
        TColProd_Fp32Acc<T, TileDataIn::Cols>(dst.data(), src.data(), validRow, validCol);
    } else {
        TColProd<T, TileDataOut, TileDataIn>(dst.data(), src.data(), validRow, validCol, VFImplKind::VFIMPL_DEFAULT);
    }
}
} // namespace pto
#endif
