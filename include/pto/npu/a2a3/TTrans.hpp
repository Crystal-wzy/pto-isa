/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TTRANS_HPP
#define TTRANS_HPP

#include "pto/common/constants.hpp"
#include "pto/common/utils.hpp"

namespace pto {

constexpr int ADDR_NUM = 16;
constexpr int HALF_ADDR_NUM = 8;
// for b8,  [32, 32] -> [32, 32]
constexpr int X_ELEM_B8 = 32;
constexpr int Y_ELEM_B8 = 32;
// for b16,  [16, 16] -> [16, 16]
// for b32,  [16, 8] -> [8,16]
constexpr int X_ELEM_B32 = 8;
constexpr int Y_ELEM_OTHER = 16;

template <typename T>
struct TransOp {
    PTO_INTERNAL static void TransB8Instr(uint8_t repeat, uint16_t dstStride, uint16_t srcStride)
    {
        // [32, 8] -> [8, 32]
        scatter_vnchwconv_b8(VA0, VA2, repeat, dstStride, srcStride, false, false);
        scatter_vnchwconv_b8(VA6, VA2, repeat, dstStride, srcStride, false, true);
        scatter_vnchwconv_b8(VA0, VA4, repeat, dstStride, srcStride, true, false);
        scatter_vnchwconv_b8(VA6, VA4, repeat, dstStride, srcStride, true, true);
    }

    PTO_INTERNAL static void TransB16Instr(uint8_t repeat, uint16_t dstStride, uint16_t srcStride)
    {
        // [16, 16] -> [16, 16]
        scatter_vnchwconv_b16(VA0, VA2, repeat, dstStride, srcStride);
    }

    PTO_INTERNAL static void TransB32Instr(uint8_t repeat, uint16_t dstStride, uint16_t srcStride)
    {
        // [16,8] -> [8,16]
        scatter_vnchwconv_b32(VA0, VA2, repeat, dstStride, srcStride);
    }

    PTO_INTERNAL static void CopyInstr(
        __ubuf__ uint32_t* dstPtr, __ubuf__ uint32_t* srcPtr, uint8_t repeat, uint16_t dstRepeatStride,
        uint16_t srcRepeatStride)
    {
        vcopy(dstPtr, srcPtr, repeat, 1, 1, dstRepeatStride, srcRepeatStride);
    }

    PTO_INTERNAL static void CopyInstr(
        __ubuf__ uint16_t* dstPtr, __ubuf__ uint16_t* srcPtr, uint8_t repeat, uint16_t dstRepeatStride,
        uint16_t srcRepeatStride)
    {
        vcopy(dstPtr, srcPtr, repeat, 1, 1, dstRepeatStride, srcRepeatStride);
    }
};

template <typename Op, typename T, unsigned blockSizeElem>
PTO_INTERNAL void TransFullSubTiles(
    __ubuf__ T* tmpPtr, __ubuf__ T* srcPtr, unsigned tmpStride, unsigned numSubTileX, unsigned numSubTileY,
    unsigned srcStride)
{
    for (int i = 0; i < numSubTileX; i++) {
        uint64_t srcUb[ADDR_NUM] = {0}, tmpUb[ADDR_NUM] = {0};
        uint64_t offset = i * blockSizeElem;
        uint16_t vconvSrcStride = Y_ELEM_OTHER * srcStride * sizeof(T) / BLOCK_BYTE_SIZE;
        for (int j = 0; j < ADDR_NUM; j++) {
            srcUb[j] = (uint64_t)(srcPtr + offset + j * srcStride);
            tmpUb[j] = (sizeof(T) == 2) ?
                           (uint64_t)(tmpPtr + (j + offset) * tmpStride) :
                           (uint64_t)(tmpPtr + ((j >> 1) + offset) * tmpStride + (j & 1) * blockSizeElem);
        }
        set_va_reg_sb(VA2, srcUb);
        set_va_reg_sb(VA3, &srcUb[HALF_ADDR_NUM]);
        set_va_reg_sb(VA0, tmpUb);
        set_va_reg_sb(VA1, &tmpUb[HALF_ADDR_NUM]);
        if constexpr (sizeof(T) == 2) {
            if (numSubTileY == 1) {
                Op::TransB16Instr(1, 0, 0);
            } else {
                Op::TransB16Instr(numSubTileY, 1, vconvSrcStride);
            }
        } else {
            if (numSubTileY == 1) {
                Op::TransB32Instr(1, 0, 0);
            } else {
                Op::TransB32Instr(numSubTileY, 2, vconvSrcStride);
            }
        }
    }
}

template <typename Op, typename T, unsigned blockSizeElem>
PTO_INTERNAL void TransB8FullSubTiles(
    __ubuf__ T* dstPtr, __ubuf__ T* srcPtr, unsigned tmpStride, unsigned numSubTileX, unsigned numSubTileY,
    unsigned srcStride)
{
    if (numSubTileX > 0) { // [32, 32] aligned
        uint64_t srcUb[ADDR_NUM] = {0}, srcUb1[ADDR_NUM] = {0}, tmpUb[ADDR_NUM] = {0}, tmpUb1[ADDR_NUM] = {0};
        uint16_t vconvSrcStride = Y_ELEM_B8 * srcStride * sizeof(T) / BLOCK_BYTE_SIZE;
        for (int i = 0; i < numSubTileX; i++) {
            uint64_t offset = i * blockSizeElem;
            for (int j = 0; j < ADDR_NUM; j++) {
                srcUb[j] = (uint64_t)(srcPtr + offset + j * srcStride);
                srcUb1[j] = srcUb[j] + ADDR_NUM * srcStride;
                tmpUb[j] = (uint64_t)(dstPtr + (j + offset) * tmpStride);
                tmpUb1[j] = tmpUb[j] + ADDR_NUM * tmpStride;
            }
            set_va_reg_sb(VA2, srcUb);
            set_va_reg_sb(VA3, &srcUb[HALF_ADDR_NUM]);
            set_va_reg_sb(VA0, tmpUb);
            set_va_reg_sb(VA1, &tmpUb[HALF_ADDR_NUM]);

            set_va_reg_sb(VA4, srcUb1);
            set_va_reg_sb(VA5, &srcUb1[HALF_ADDR_NUM]);
            set_va_reg_sb(VA6, tmpUb1);
            set_va_reg_sb(VA7, &tmpUb1[HALF_ADDR_NUM]);
            if (numSubTileY == 1) { // [32, 32]
                Op::TransB8Instr(1, 0, 0);
            } else { // larger then [32, 32], e.g, [32, 64]
                Op::TransB8Instr(numSubTileY, 1, vconvSrcStride);
            }
        } // end of numSubTileX
    }
}

template <typename Op, typename T, unsigned blockSizeElem>
PTO_INTERNAL void TransYTailTiles(
    __ubuf__ T* tmpPtr, __ubuf__ T* srcPtr, unsigned tmpStride, unsigned numSubTileX, unsigned numSubTileY,
    unsigned remainY, unsigned srcStride)
{
    uint64_t srcUb[ADDR_NUM] = {0}, tmpUb[ADDR_NUM] = {0};
    uint64_t offset = numSubTileY * Y_ELEM_OTHER;
    for (int i = 0; i < remainY; i++) {
        srcUb[i] = (uint64_t)(srcPtr + (offset + i) * srcStride);
    }
    for (int i = 0; i < ADDR_NUM; i++) {
        tmpUb[i] = (sizeof(T) == 2) ? (uint64_t)(tmpPtr + offset + i * tmpStride) :
                                      (uint64_t)(tmpPtr + offset + (i & 1) * blockSizeElem + (i >> 1) * tmpStride);
    }
    set_va_reg_sb(VA2, srcUb);
    set_va_reg_sb(VA3, &srcUb[HALF_ADDR_NUM]);
    set_va_reg_sb(VA0, tmpUb);
    set_va_reg_sb(VA1, &tmpUb[HALF_ADDR_NUM]);
    if constexpr (sizeof(T) == 2) {
        if (numSubTileX == 1) {
            Op::TransB16Instr(1, 0, 0);
        } else {
            Op::TransB16Instr(numSubTileX, blockSizeElem * tmpStride * sizeof(T) / BLOCK_BYTE_SIZE, 1);
        }
    } else {
        if (numSubTileX == 1) {
            Op::TransB32Instr(1, 0, 0);
        } else {
            Op::TransB32Instr(numSubTileX, blockSizeElem * tmpStride * sizeof(T) / BLOCK_BYTE_SIZE, 1);
        }
    }
}

template <typename Op, typename T, unsigned blockSizeEleme>
PTO_INTERNAL void TransB8YTailTiles(
    __ubuf__ T* dstPtr, __ubuf__ T* srcPtr, unsigned tmpStride, unsigned numSubTileX, unsigned numSubTileY,
    unsigned remainY, unsigned srcStride)
{
    uint64_t srcUb[ADDR_NUM] = {0}, tmpUb[ADDR_NUM] = {0};
    uint64_t srcUb1[ADDR_NUM] = {0}, tmpUb1[ADDR_NUM] = {0};
    uint64_t offset = numSubTileY * Y_ELEM_B8;
    for (int i = 0; i < remainY; i++) {
        if (i < ADDR_NUM) {
            srcUb[i] = (uint64_t)(srcPtr + (offset + i) * srcStride);
        } else {
            srcUb1[i - ADDR_NUM] = (uint64_t)(srcPtr + (offset + i) * srcStride);
        }
    }
    for (int i = 0; i < ADDR_NUM; i++) {
        tmpUb[i] = (uint64_t)(dstPtr + offset + i * tmpStride);
        tmpUb1[i] = tmpUb[i] + ADDR_NUM * tmpStride;
    }
    set_va_reg_sb(VA2, srcUb);
    set_va_reg_sb(VA3, &srcUb[HALF_ADDR_NUM]);
    set_va_reg_sb(VA0, tmpUb);
    set_va_reg_sb(VA1, &tmpUb[HALF_ADDR_NUM]);

    set_va_reg_sb(VA4, srcUb1);
    set_va_reg_sb(VA5, &srcUb1[HALF_ADDR_NUM]);
    set_va_reg_sb(VA6, tmpUb1);
    set_va_reg_sb(VA7, &tmpUb1[HALF_ADDR_NUM]);
    if (numSubTileX == 1) {
        Op::TransB8Instr(1, 0, 0);
    } else {
        Op::TransB8Instr(numSubTileX, Y_ELEM_B8 * tmpStride * sizeof(T) / BLOCK_BYTE_SIZE, 1);
    }
}

template <typename T, unsigned blockSizeElem, unsigned yTileSizeElem>
PTO_INTERNAL void TransTailTiles(
    __ubuf__ T* dstPtr, __ubuf__ T* srcPtr, unsigned tmpStride, unsigned validRow, unsigned validCol,
    unsigned dstStride, unsigned srcStride)
{
    // we can use constexpr if tmpStride is known in static way
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
#else
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
#endif
    for (int i = 0; i < validRow; i++) {
        for (int j = 0; j < validCol; j++) {
            dstPtr[j * dstStride + i] = srcPtr[i * srcStride + j];
        }
    }
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
#else
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
#endif
    return;
}

template <typename T, unsigned blockSizeElem>
PTO_INTERNAL void TTransOperation(
    __ubuf__ T* dstPtr, __ubuf__ T* srcPtr, __ubuf__ T* tmpPtr, unsigned validRow, unsigned validCol,
    unsigned dstStride, unsigned srcStride)
{
    constexpr unsigned yTileSizeElem = (sizeof(T) == 1) ? Y_ELEM_B8 : Y_ELEM_OTHER;
    // tmpStride should computed in static way
    unsigned tmpStride = (validRow + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem;
    if (((dstStride % yTileSizeElem) != 0) || ((srcStride % blockSizeElem) != 0) ||
        ((tmpStride % yTileSizeElem) != 0)) {
        TransTailTiles<T, blockSizeElem, yTileSizeElem>(
            dstPtr, srcPtr, tmpStride, validRow, validCol, dstStride, srcStride);
        return;
    }
    // go by subtile column, a.k.a. iter in row direction
    int numSubTileX = (validCol + blockSizeElem - 1) / blockSizeElem;
    int numSubTileY = validRow / yTileSizeElem;
    if (numSubTileY > 0) {
        if constexpr (sizeof(T) == 1) {
            TransB8FullSubTiles<TransOp<T>, T, blockSizeElem>(
                tmpPtr, srcPtr, tmpStride, numSubTileX, numSubTileY, srcStride);
        } else {
            TransFullSubTiles<TransOp<T>, T, blockSizeElem>(
                tmpPtr, srcPtr, tmpStride, numSubTileX, numSubTileY, srcStride);
        }
    }
    // tail
    int remainY = validRow % yTileSizeElem;
    if (remainY > 0) {
        if constexpr (sizeof(T) == 1) {
            TransB8YTailTiles<TransOp<T>, T, blockSizeElem>(
                tmpPtr, srcPtr, tmpStride, numSubTileX, numSubTileY, remainY, srcStride);
        } else {
            TransYTailTiles<TransOp<T>, T, blockSizeElem>(
                tmpPtr, srcPtr, tmpStride, numSubTileX, numSubTileY, remainY, srcStride);
        }
    }
    // copy to dst
    uint16_t lenBurst = (validRow * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;
    uint16_t srcGap = tmpStride * sizeof(T) / BLOCK_BYTE_SIZE - lenBurst;
    uint16_t dstGap = dstStride * sizeof(T) / BLOCK_BYTE_SIZE - lenBurst;
    pipe_barrier(PIPE_V);
    pto_copy_ubuf_to_ubuf(dstPtr, tmpPtr, validCol, lenBurst, srcGap, dstGap);
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, unsigned blockSizeElem>
__tf__ PTO_INTERNAL void TTrans(
    typename TileDataDst::TileDType __out__ dst, typename TileDataSrc::TileDType __in__ src,
    typename TileDataTmp::TileDType __in__ tmp, unsigned validRow, unsigned validCol, unsigned dstStride,
    unsigned srcStride)
{
    using T = typename TileDataSrc::DType;
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    __ubuf__ T* tmpPtr = (__ubuf__ T*)__cce_get_tile_ptr(tmp);
    TTransOperation<T, blockSizeElem>(dstPtr, srcPtr, tmpPtr, validRow, validCol, dstStride, srcStride);
}

////////////// TTrans Repeat X/////

template <typename Op, typename T, unsigned blockSizeElem>
PTO_INTERNAL void TransRepeatXFullSubTiles(
    __ubuf__ T* tmpPtr, __ubuf__ T* srcPtr, unsigned tmpStride, unsigned numSubTileX, unsigned numSubTileY,
    unsigned srcStride)
{
    constexpr unsigned yTileSizeElem = (sizeof(T) == 1) ? Y_ELEM_B8 : Y_ELEM_OTHER;
    for (int i = 0; i < numSubTileY; i++) {
        uint64_t srcUb[ADDR_NUM] = {0}, tmpUb[ADDR_NUM] = {0};
        uint64_t offset = i * srcStride * yTileSizeElem;
        uint64_t vconvDstStride = X_ELEM_B32 * tmpStride * sizeof(T) / BLOCK_BYTE_SIZE; // for float
        uint64_t dstOffset = Y_ELEM_OTHER * i;
        // for float, src: 16 * 8, dst: 8 * 16;
        // for half, src: 16 * 16, dst: 16 * 16;
        for (int j = 0; j < ADDR_NUM; j++) {
            srcUb[j] = (uint64_t)(srcPtr + offset + j * srcStride);
            tmpUb[j] = (sizeof(T) == 2) ?
                           (uint64_t)(tmpPtr + j * tmpStride + dstOffset) :
                           (uint64_t)(tmpPtr + (j >> 1) * tmpStride + dstOffset + (j & 1) * blockSizeElem);
        }
        set_va_reg_sb(VA2, srcUb);
        set_va_reg_sb(VA3, &srcUb[HALF_ADDR_NUM]);
        set_va_reg_sb(VA0, tmpUb);
        set_va_reg_sb(VA1, &tmpUb[HALF_ADDR_NUM]);
        if constexpr (sizeof(T) == 2) {
            if (numSubTileX == 1) {
                Op::TransB16Instr(1, 0, 0);
            } else {
                Op::TransB16Instr(numSubTileX, vconvDstStride * 2, 1);
            }
        } else {
            if (numSubTileX == 1) {
                Op::TransB32Instr(1, 0, 0);
            } else {
                Op::TransB32Instr(numSubTileX, vconvDstStride, 1);
            }
        }
    }
}

template <typename Op, typename T, unsigned blockSizeElem>
PTO_INTERNAL void TransRepeatXB8FullSubTiles(
    __ubuf__ T* dstPtr, __ubuf__ T* srcPtr, unsigned tmpStride, unsigned numSubTileX, unsigned numSubTileY,
    unsigned srcStride)
{
    constexpr unsigned yTileSizeElem = (sizeof(T) == 1) ? Y_ELEM_B8 : Y_ELEM_OTHER;
    if (numSubTileY > 0) { // [32, 32] aligned
        uint64_t srcUb[ADDR_NUM] = {0}, srcUb1[ADDR_NUM] = {0}, tmpUb[ADDR_NUM] = {0}, tmpUb1[ADDR_NUM] = {0};
        uint64_t vconvDstStride = X_ELEM_B8 * tmpStride * sizeof(T) / BLOCK_BYTE_SIZE;
        for (int i = 0; i < numSubTileY; i++) {
            uint64_t offset = i * yTileSizeElem * srcStride;
            for (int j = 0; j < ADDR_NUM; j++) {
                srcUb[j] = (uint64_t)(srcPtr + offset + j * srcStride);
                srcUb1[j] = srcUb[j] + ADDR_NUM * srcStride;
                tmpUb[j] = (uint64_t)(dstPtr + j * tmpStride + 32 * i);
                tmpUb1[j] = tmpUb[j] + ADDR_NUM * tmpStride;
            }
            set_va_reg_sb(VA4, srcUb1);
            set_va_reg_sb(VA5, &srcUb1[HALF_ADDR_NUM]);
            set_va_reg_sb(VA6, tmpUb1);
            set_va_reg_sb(VA7, &tmpUb1[HALF_ADDR_NUM]);

            set_va_reg_sb(VA2, srcUb);
            set_va_reg_sb(VA3, &srcUb[HALF_ADDR_NUM]);
            set_va_reg_sb(VA0, tmpUb);
            set_va_reg_sb(VA1, &tmpUb[HALF_ADDR_NUM]);
            if (numSubTileX == 1) { // [32, 32]
                Op::TransB8Instr(1, 0, 0);
            } else { // larger than [32, 32], e.g, [32, 64]
                Op::TransB8Instr(numSubTileX, vconvDstStride, 1);
            }
        } // end of numSubTileY
    }
}

template <typename T, unsigned blockSizeElem>
PTO_INTERNAL void TTransRepeatXOperation(
    __ubuf__ T* dstPtr, __ubuf__ T* srcPtr, __ubuf__ T* tmpPtr, unsigned validRow, unsigned validCol,
    unsigned dstStride, unsigned srcStride)
{
    constexpr unsigned yTileSizeElem = (sizeof(T) == 1) ? Y_ELEM_B8 : Y_ELEM_OTHER;
    // tmpStride should computed in static way
    unsigned tmpStride = (validRow + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem;
    // go by subtile column, a.k.a. iter in row direction
    int numSubTileX = validCol / blockSizeElem;
    int numSubTileY = validRow / yTileSizeElem;
    if (numSubTileX > 0) {
        if constexpr (sizeof(T) == 1) {
            TransRepeatXB8FullSubTiles<TransOp<T>, T, blockSizeElem>(
                tmpPtr, srcPtr, tmpStride, numSubTileX, numSubTileY, srcStride);
        } else {
            TransRepeatXFullSubTiles<TransOp<T>, T, blockSizeElem>(
                tmpPtr, srcPtr, tmpStride, numSubTileX, numSubTileY, srcStride);
        }
    }
    // tail
    int remainY = validRow % yTileSizeElem;
    if (remainY > 0) {
        if constexpr (sizeof(T) != 1) {
            TransYTailTiles<TransOp<T>, T, blockSizeElem>(
                tmpPtr, srcPtr, tmpStride, numSubTileX, numSubTileY, remainY, srcStride);
        } else {
            TransB8YTailTiles<TransOp<T>, T, blockSizeElem>(
                tmpPtr, srcPtr, tmpStride, numSubTileX, numSubTileY, remainY, srcStride);
        }
    }
    // copy to dst
    uint16_t lenBurst = (validRow * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;
    uint16_t dstGap = dstStride * sizeof(T) / BLOCK_BYTE_SIZE - lenBurst;
    uint16_t srcGap = tmpStride * sizeof(T) / BLOCK_BYTE_SIZE - lenBurst;
    pipe_barrier(PIPE_V);
    pto_copy_ubuf_to_ubuf(dstPtr, tmpPtr, validCol, lenBurst, srcGap, dstGap);
}

/////////////////// ConvTile format conversions ///////////////////
// Raw (declared) shape only supplies strides and capacity; the valid shape supplies the processing
// domain:
//   * read only the src valid box, write only the dst valid box;
//   * positions inside the box without a source counterpart (the C0 tail of the last C1 block, the
//     N0 tail of the last N1 block, the HW pad segment of a padded FRACTAL_Z dim0) are filled with +0;
//   * with valid == raw the result is byte-for-byte the historical one.
// Geometry the vector paths cannot express (32B block alignment, tmp capacity, a C tail block) is
// handled block by block with scalar code of identical semantics.

// Scalar phase brackets (PIPE_V <-> PIPE_S)
PTO_INTERNAL void TTransConvEnterScalar()
{
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
#else
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
#endif
}

PTO_INTERNAL void TTransConvExitScalar()
{
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
#else
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
#endif
}

// 暂存区向量清零：整 repeat（256B）写 +0，不足一个 repeat 的尾巴也按整 repeat 写满，
// 因此不使用连续掩码（掩码写法在只有部分 repeat 时不可靠），调用方需为多写的至多 255B 留余量。
// 不用 vmuls(x, 0)：那是真乘法，x 为负时写出 -0，与参考实现的 +0 差一个 bit。
template <typename T>
PTO_INTERNAL void TTransConvZeroScope(__ubuf__ T* ptr, unsigned numElems)
{
    __ubuf__ int32_t* wptr = reinterpret_cast<__ubuf__ int32_t*>(ptr);
    unsigned words = (numElems * static_cast<unsigned>(sizeof(T)) + 3U) / 4U;
    unsigned numRepeats = (words + 63U) / 64U; // 向上取整到整 repeat
    while (numRepeats > 0) {
        uint8_t repeat = (numRepeats > REPEAT_MAX) ? REPEAT_MAX : static_cast<uint8_t>(numRepeats);
        vector_dup(wptr, (int32_t)0, repeat, 1, 1, 8, 0);
        wptr += static_cast<unsigned>(repeat) * 64U;
        numRepeats -= repeat;
    }
}

// 暂存区需要多大（字节）：按 256B 向上取整
PTO_INTERNAL unsigned TTransConvZeroScopeBytes(unsigned bytes) { return ((bytes + 255U) / 256U) * 256U; }

template <typename T>
PTO_INTERNAL void TTransConvZeroScalar(__ubuf__ T* ptr, unsigned numElems)
{
    for (unsigned i = 0; i < numElems; ++i) {
        ptr[i] = static_cast<T>(0);
    }
}

// One transpose: dst[x][r] = src[r][x], x < cols, r < rows.
//   src has rows rows with row stride srcStride; dst has cols rows with row stride dstStride.
//   cols must be a multiple of blockSizeElem (the scatter X only moves whole blocks) and dstStride
//   must be a multiple of blockSizeElem with a 32B-block gap that fits a DMA descriptor.
// rowsConst != 0 时 rows 是编译期常量：scatter 内层的地址算术能全部折叠（实测这是热路径的关键，
// 传运行期 rows 会让每个子块多花上百 cycle）。rowsConst == 0 时用运行期 rows。
template <typename T, unsigned blockSizeElem, unsigned rowsConst = 0>
PTO_INTERNAL void TTransConvTranspose(
    __ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmp, unsigned rows, unsigned cols, unsigned dstStride,
    unsigned srcStride)
{
    const unsigned r = (rowsConst != 0) ? rowsConst : rows;
    constexpr unsigned kMaxCols = 255U * blockSizeElem; // scatter repeat limit
    if (r == 0) {
        return;
    }
    while (cols > 0) {
        unsigned run = (cols > kMaxCols) ? kMaxCols : cols;
        TTransRepeatXOperation<T, blockSizeElem>(dst, src, tmp, r, run, dstStride, srcStride);
        dst += run * dstStride;
        src += run;
        cols -= run;
    }
}

// 常见情形 C0 == blockSizeElem 时让 rows 走编译期常量，否则退回运行期 rows
template <typename T, unsigned blockSizeElem>
PTO_INTERNAL void TTransConvTransposeRows(
    __ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* tmp, unsigned rows, unsigned cols, unsigned dstStride,
    unsigned srcStride)
{
    if (rows == blockSizeElem) {
        TTransConvTranspose<T, blockSizeElem, blockSizeElem>(dst, src, tmp, rows, cols, dstStride, srcStride);
    } else {
        TTransConvTranspose<T, blockSizeElem>(dst, src, tmp, rows, cols, dstStride, srcStride);
    }
}

// Single column of a transpose (scalar filler when the X extent is not a whole block).
template <typename T>
PTO_INTERNAL void TTransConvTransposeColumn(
    __ubuf__ T* dstCol, __ubuf__ T* srcCol, unsigned rows, unsigned dstStride, unsigned srcStride)
{
    for (unsigned r = 0; r < rows; ++r) {
        dstCol[r * dstStride] = srcCol[r * srcStride];
    }
}

PTO_INTERNAL bool TTransConvBlockAligned(unsigned elems, unsigned blockSizeElem)
{
    return (elems % blockSizeElem) == 0;
}

// FRACTAL_Z address of logical n = n1 * n0Sem + n0；n0Sem 是 dst 的有效 N0（语义），
// dstN0Raw 是该维声明的 stride（raw 只表达内存）。
template <typename T>
PTO_INTERNAL unsigned TTransConvFractalZNOffset(unsigned n, unsigned n0Sem, unsigned dstN0Raw, unsigned dstC0)
{
    return (n / n0Sem) * dstN0Raw * dstC0 + (n % n0Sem) * dstC0;
}

// 从 n 推进到 n+1 的偏移增量（n0Idx 是当前的 n%N0，随之更新）。
// 逐 n 调用它可以把上面的除法换成比较+加法：N0 是运行期值，n/N0、n%N0 是真除法，热路径里很贵。
PTO_INTERNAL void TTransConvNAdvance(unsigned& nOff, unsigned& n0Idx, unsigned n0Sem, unsigned dstN0Raw, unsigned dstC0)
{
    if (++n0Idx == n0Sem) {
        n0Idx = 0;
        nOff += dstN0Raw * dstC0 - (n0Sem - 1) * dstC0;
    } else {
        nOff += dstC0;
    }
}

// ---------- NCHW -> NC1HWC0 ----------
// One (n, c1) group: src is NCHW with C row stride srcCStride, dst is NC1HWC0 with hw-neighbour
// stride dstC0Stride and H row stride dstHStride. rows (= blockSizeElem) is the group's valid C rows.
template <typename T, unsigned blockSizeElem>
PTO_INTERNAL void TTransConvNCHW2NC1HWC0Group(
    __ubuf__ T* dstC, __ubuf__ T* srcC, __ubuf__ T* tmp, unsigned srcCStride, unsigned srcW, unsigned dstC0Stride,
    unsigned dstHStride, unsigned vH, unsigned vW, unsigned c0Rows, bool wContiguous)
{
    if (wContiguous) { // W not shrunk: the source H rows are contiguous, one transpose does it all
        unsigned xVec = ((vH * srcW) / blockSizeElem) * blockSizeElem;
        TTransConvTransposeRows<T, blockSizeElem>(dstC, srcC, tmp, c0Rows, xVec, dstC0Stride, srcCStride);
    } else { // per H row (source rows are not contiguous)
        unsigned xVec = (vW / blockSizeElem) * blockSizeElem;
        for (unsigned h = 0; h < vH; ++h) {
            TTransConvTransposeRows<T, blockSizeElem>(
                dstC + h * dstHStride, srcC + h * srcW, tmp, c0Rows, xVec, dstC0Stride, srcCStride);
        }
    }
}

// X columns shorter than one block (scalar, PIPE_S)
template <typename T, unsigned blockSizeElem>
PTO_INTERNAL void TTransConvNCHW2NC1HWC0GroupTail(
    __ubuf__ T* dstC, __ubuf__ T* srcC, unsigned srcCStride, unsigned srcW, unsigned dstC0Stride, unsigned dstHStride,
    unsigned vH, unsigned vW, unsigned c0Rows, bool wContiguous)
{
    if (wContiguous) {
        unsigned xTotal = vH * srcW;
        unsigned xVec = (xTotal / blockSizeElem) * blockSizeElem;
        for (unsigned x = xVec; x < xTotal; ++x) {
            TTransConvTransposeColumn<T>(dstC + x * dstC0Stride, srcC + x, c0Rows, 1, srcCStride);
        }
    } else {
        unsigned xVec = (vW / blockSizeElem) * blockSizeElem;
        for (unsigned h = 0; h < vH; ++h) {
            __ubuf__ T* const d = dstC + h * dstHStride;
            __ubuf__ T* const s = srcC + h * srcW;
            for (unsigned w = xVec; w < vW; ++w) {
                TTransConvTransposeColumn<T>(d + w * dstC0Stride, s + w, c0Rows, 1, srcCStride);
            }
        }
    }
}

// Whole-group scalar: rows valid C rows copied, the remaining C0 slots filled with +0.
template <typename T>
PTO_INTERNAL void TTransConvNCHW2NC1HWC0ScalarBlock(
    __ubuf__ T* dstC, __ubuf__ T* srcC, unsigned srcCStride, unsigned srcW, unsigned dstC0Stride, unsigned dstHStride,
    unsigned c0, unsigned rows, unsigned vH, unsigned vW)
{
    for (unsigned h = 0; h < vH; ++h) {
        for (unsigned w = 0; w < vW; ++w) {
            __ubuf__ T* const d = dstC + h * dstHStride + w * dstC0Stride;
            __ubuf__ T* const s = srcC + h * srcW + w;
            for (unsigned j = 0; j < c0; ++j) {
                d[j] = (j < rows) ? s[j * srcCStride] : static_cast<T>(0);
            }
        }
    }
}

// NCHW -> NC1HWC0
template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, unsigned blockSizeElem>
__tf__ PTO_INTERNAL void TTransConvNCHW2NC1HWC0(TileDataDst& dst, TileDataSrc& src, TileDataTmp& tmp)
{
    using T = typename TileDataSrc::DType;

    __ubuf__ T* const dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst.data());
    __ubuf__ T* const srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src.data());
    __ubuf__ T* const tmpPtr = (__ubuf__ T*)__cce_get_tile_ptr(tmp.data());

    // src: NCHW [N][C][H][W]
    const unsigned rawC = (unsigned)src.GetShape(1);
    const unsigned rawH = (unsigned)src.GetShape(2);
    const unsigned rawW = (unsigned)src.GetShape(3);
    const unsigned srcC = rawH * rawW;
    const unsigned srcN = rawC * srcC;
    // dst: NC1HWC0 [N][C1][H][W][C0]；语义块宽取 dst 的有效 C0，地址 stride 取声明值
    const unsigned C0 = (unsigned)dst.GetValidShape(4);
    const unsigned dstC0 = (unsigned)dst.GetShape(4);
    const unsigned dstH = (unsigned)dst.GetShape(3) * dstC0;
    const unsigned dstC1 = (unsigned)dst.GetShape(2) * dstH;
    const unsigned dstN = (unsigned)dst.GetShape(1) * dstC1;

    const unsigned vN = (unsigned)src.GetValidShape(0);
    const unsigned vC = (unsigned)src.GetValidShape(1);
    const unsigned vH = (unsigned)src.GetValidShape(2);
    const unsigned vW = (unsigned)src.GetValidShape(3);
    const unsigned fullC1 = vC / C0;         // whole C0 blocks
    const unsigned vC1 = (vC + C0 - 1) / C0; // valid C1 blocks

    const bool vecOk = TTransConvBlockAligned(dstC0, blockSizeElem) && TTransConvBlockAligned(dstH, blockSizeElem) &&
                       TTransConvBlockAligned(dstC1, blockSizeElem) && TTransConvBlockAligned(dstN, blockSizeElem);
    const bool wContiguous = (vW == rawW) && ((unsigned)dst.GetShape(3) == rawW);
    const unsigned xMax = wContiguous ? (vH * rawW) : vW;
    const unsigned xRun = (xMax > 255U * blockSizeElem) ? (255U * blockSizeElem) : xMax;
    const unsigned tmpNeed = xRun * (((C0 + Y_ELEM_OTHER - 1) / Y_ELEM_OTHER) * Y_ELEM_OTHER);
    const bool tmpOk = (unsigned)(TileDataTmp::Rows * TileDataTmp::Cols) >= tmpNeed;
    const bool vec = vecOk && tmpOk;
    const bool needScalar = !vec || (vC % C0 != 0) || (xMax % blockSizeElem != 0);

    if (vec) {
        for (unsigned n = 0; n < vN; ++n) {
            for (unsigned c1 = 0; c1 < fullC1; ++c1) {
                TTransConvNCHW2NC1HWC0Group<T, blockSizeElem>(
                    dstPtr + n * dstN + c1 * dstC1, srcPtr + n * srcN + c1 * C0 * srcC, tmpPtr, srcC, rawW, dstC0, dstH,
                    vH, vW, C0, wContiguous);
            }
        }
    }

    if (needScalar) {
        TTransConvEnterScalar();
        for (unsigned n = 0; n < vN; ++n) {
            for (unsigned c1 = 0; c1 < vC1; ++c1) {
                const unsigned rows = (c1 < fullC1) ? C0 : (vC - fullC1 * C0);
                __ubuf__ T* const d = dstPtr + n * dstN + c1 * dstC1;
                __ubuf__ T* const s = srcPtr + n * srcN + c1 * C0 * srcC;
                if (vec && rows == C0) {
                    TTransConvNCHW2NC1HWC0GroupTail<T, blockSizeElem>(
                        d, s, srcC, rawW, dstC0, dstH, vH, vW, C0, wContiguous);
                } else {
                    TTransConvNCHW2NC1HWC0ScalarBlock<T>(d, s, srcC, rawW, dstC0, dstH, C0, rows, vH, vW);
                }
            }
        }
        TTransConvExitScalar();
    }
}

// ---------- NC1HWC0 -> NCHW ----------
template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, unsigned blockSizeElem>
__tf__ PTO_INTERNAL void TTransConvNC1HWC02NCHW(TileDataDst& dst, TileDataSrc& src, TileDataTmp& tmp)
{
    using T = typename TileDataSrc::DType;

    __ubuf__ T* const dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst.data());
    __ubuf__ T* const srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src.data());
    __ubuf__ T* const tmpPtr = (__ubuf__ T*)__cce_get_tile_ptr(tmp.data());

    // src: NC1HWC0 [N][C1][H][W][C0]；C0 是转换的语义块宽（由 tile 声明）
    const unsigned rawC1 = (unsigned)src.GetShape(1);
    const unsigned rawH = (unsigned)src.GetShape(2);
    const unsigned rawW = (unsigned)src.GetShape(3);
    const unsigned C0 = (unsigned)src.GetValidShape(4);
    const unsigned srcC0 = (unsigned)src.GetShape(4);
    const unsigned srcH = rawW * srcC0;
    const unsigned srcC1 = rawH * srcH;
    const unsigned srcN = rawC1 * srcC1;
    // dst: NCHW [N][C][H][W]
    const unsigned dstW = (unsigned)dst.GetShape(3);
    const unsigned dstC = (unsigned)dst.GetShape(2) * dstW;
    const unsigned dstN = (unsigned)dst.GetShape(1) * dstC;

    const unsigned vN = (unsigned)src.GetValidShape(0);
    const unsigned vC1 = (unsigned)src.GetValidShape(1);
    const unsigned vH = (unsigned)src.GetValidShape(2);
    const unsigned vW = (unsigned)src.GetValidShape(3);
    const unsigned vC0 = (unsigned)src.GetValidShape(4);

    const bool vecOk = (vC0 == C0) && TTransConvBlockAligned(dstC, blockSizeElem) &&
                       TTransConvBlockAligned(dstW, blockSizeElem) && TTransConvBlockAligned(srcC0, blockSizeElem);
    const bool wContiguous = (vW == rawW) && (dstW == rawW);
    const unsigned xMax = wContiguous ? (vH * dstW) : vW;
    const unsigned tmpNeed = C0 * (((xMax + Y_ELEM_OTHER - 1) / Y_ELEM_OTHER) * Y_ELEM_OTHER);
    const bool vec = vecOk && ((unsigned)(TileDataTmp::Rows * TileDataTmp::Cols) >= tmpNeed);
    const bool needScalar = !vec || (xMax % blockSizeElem != 0);

    if (vec) {
        for (unsigned n = 0; n < vN; ++n) {
            for (unsigned c1 = 0; c1 < vC1; ++c1) {
                __ubuf__ T* const s = srcPtr + n * srcN + c1 * srcC1;
                __ubuf__ T* const d = dstPtr + n * dstN + c1 * C0 * dstC;
                if (wContiguous) {
                    TTransConvTranspose<T, blockSizeElem>(
                        d, s, tmpPtr, ((vH * dstW) / blockSizeElem) * blockSizeElem, C0, dstC, srcC0);
                } else {
                    unsigned xVec = (vW / blockSizeElem) * blockSizeElem;
                    for (unsigned h = 0; h < vH; ++h) {
                        TTransConvTranspose<T, blockSizeElem>(
                            d + h * dstW, s + h * srcH, tmpPtr, xVec, C0, dstC, srcC0);
                    }
                }
            }
        }
    }

    if (needScalar) {
        TTransConvEnterScalar();
        for (unsigned n = 0; n < vN; ++n) {
            for (unsigned c1 = 0; c1 < vC1; ++c1) {
                __ubuf__ T* const s = srcPtr + n * srcN + c1 * srcC1;
                __ubuf__ T* const d = dstPtr + n * dstN + c1 * C0 * dstC;
                const unsigned xVec = wContiguous ? (((vH * dstW) / blockSizeElem) * blockSizeElem) :
                                                    ((vW / blockSizeElem) * blockSizeElem);
                if (vec) {
                    if (wContiguous) {
                        unsigned xTotal = vH * dstW;
                        for (unsigned x = xVec; x < xTotal; ++x) {
                            TTransConvTransposeColumn<T>(d + x, s + x * srcC0, C0, dstC, 1);
                        }
                    } else {
                        for (unsigned h = 0; h < vH; ++h) {
                            for (unsigned w = xVec; w < vW; ++w) {
                                TTransConvTransposeColumn<T>(d + h * dstW + w, s + h * srcH + w * srcC0, C0, dstC, 1);
                            }
                        }
                    }
                } else {
                    for (unsigned h = 0; h < vH; ++h) {
                        for (unsigned w = 0; w < vW; ++w) {
                            __ubuf__ T* const dd = d + h * dstW + w;
                            __ubuf__ T* const ss = s + h * srcH + w * srcC0;
                            for (unsigned c0 = 0; c0 < vC0; ++c0) {
                                dd[c0 * dstC] = ss[c0];
                            }
                        }
                    }
                }
            }
        }
        TTransConvExitScalar();
    }
}

// ---------- GNCHW -> GNC1HWC0 ----------
template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, unsigned blockSizeElem>
__tf__ PTO_INTERNAL void TTransConvGNCHW2GNC1HWC0(TileDataDst& dst, TileDataSrc& src, TileDataTmp& tmp)
{
    using T = typename TileDataSrc::DType;

    __ubuf__ T* const dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst.data());
    __ubuf__ T* const srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src.data());
    __ubuf__ T* const tmpPtr = (__ubuf__ T*)__cce_get_tile_ptr(tmp.data());

    // src: GNCHW [G][N][C][H][W]
    const unsigned rawN = (unsigned)src.GetShape(1);
    const unsigned rawC = (unsigned)src.GetShape(2);
    const unsigned rawH = (unsigned)src.GetShape(3);
    const unsigned rawW = (unsigned)src.GetShape(4);
    const unsigned srcC = rawH * rawW;
    const unsigned srcN = rawC * srcC;
    const unsigned srcG = rawN * srcN;
    // dst: GNC1HWC0 [G][N][C1][H][W][C0]；语义块宽取 dst 的有效 C0，地址 stride 取声明值
    const unsigned C0 = (unsigned)dst.GetValidShape(5);
    const unsigned dstC0 = (unsigned)dst.GetShape(5);
    const unsigned dstH = (unsigned)dst.GetShape(4) * dstC0;
    const unsigned dstC1 = (unsigned)dst.GetShape(3) * dstH;
    const unsigned dstN = (unsigned)dst.GetShape(2) * dstC1;
    const unsigned dstG = (unsigned)dst.GetShape(1) * dstN;
    const unsigned dstW = (unsigned)dst.GetShape(4);

    const unsigned vG = (unsigned)src.GetValidShape(0);
    const unsigned vN = (unsigned)src.GetValidShape(1);
    const unsigned vC = (unsigned)src.GetValidShape(2);
    const unsigned vH = (unsigned)src.GetValidShape(3);
    const unsigned vW = (unsigned)src.GetValidShape(4);
    const unsigned fullC1 = vC / C0;
    const unsigned vC1 = (vC + C0 - 1) / C0;

    const bool vecOk = TTransConvBlockAligned(dstC0, blockSizeElem) && TTransConvBlockAligned(dstH, blockSizeElem) &&
                       TTransConvBlockAligned(dstC1, blockSizeElem) && TTransConvBlockAligned(dstN, blockSizeElem) &&
                       TTransConvBlockAligned(dstG, blockSizeElem);
    const bool wContiguous = (vW == rawW) && (dstW == rawW);
    const unsigned xMax = wContiguous ? (vH * rawW) : vW;
    const unsigned xRun = (xMax > 255U * blockSizeElem) ? (255U * blockSizeElem) : xMax;
    const unsigned tmpNeed = xRun * (((C0 + Y_ELEM_OTHER - 1) / Y_ELEM_OTHER) * Y_ELEM_OTHER);
    const bool vec = vecOk && ((unsigned)(TileDataTmp::Rows * TileDataTmp::Cols) >= tmpNeed);
    const bool needScalar = !vec || (vC % C0 != 0) || (xMax % blockSizeElem != 0);

    if (vec) {
        for (unsigned g = 0; g < vG; ++g) {
            for (unsigned n = 0; n < vN; ++n) {
                for (unsigned c1 = 0; c1 < fullC1; ++c1) {
                    TTransConvNCHW2NC1HWC0Group<T, blockSizeElem>(
                        dstPtr + g * dstG + n * dstN + c1 * dstC1, srcPtr + g * srcG + n * srcN + c1 * C0 * srcC,
                        tmpPtr, srcC, rawW, dstC0, dstH, vH, vW, C0, wContiguous);
                }
            }
        }
    }

    if (needScalar) {
        TTransConvEnterScalar();
        for (unsigned g = 0; g < vG; ++g) {
            for (unsigned n = 0; n < vN; ++n) {
                for (unsigned c1 = 0; c1 < vC1; ++c1) {
                    const unsigned rows = (c1 < fullC1) ? C0 : (vC - fullC1 * C0);
                    __ubuf__ T* const d = dstPtr + g * dstG + n * dstN + c1 * dstC1;
                    __ubuf__ T* const s = srcPtr + g * srcG + n * srcN + c1 * C0 * srcC;
                    if (vec && rows == C0) {
                        TTransConvNCHW2NC1HWC0GroupTail<T, blockSizeElem>(
                            d, s, srcC, rawW, dstC0, dstH, vH, vW, C0, wContiguous);
                    } else {
                        TTransConvNCHW2NC1HWC0ScalarBlock<T>(d, s, srcC, rawW, dstC0, dstH, C0, rows, vH, vW);
                    }
                }
            }
        }
        TTransConvExitScalar();
    }
}

// ---------- GNC1HWC0 -> GNCHW ----------
template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, unsigned blockSizeElem>
__tf__ PTO_INTERNAL void TTransConvGNC1HWC02GNCHW(TileDataDst& dst, TileDataSrc& src, TileDataTmp& tmp)
{
    using T = typename TileDataSrc::DType;

    __ubuf__ T* const dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst.data());
    __ubuf__ T* const srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src.data());
    __ubuf__ T* const tmpPtr = (__ubuf__ T*)__cce_get_tile_ptr(tmp.data());

    // src: GNC1HWC0 [G][N][C1][H][W][C0]；语义块宽取 src 的有效 C0，地址 stride 取声明值
    const unsigned rawN = (unsigned)src.GetShape(1);
    const unsigned rawC1 = (unsigned)src.GetShape(2);
    const unsigned rawH = (unsigned)src.GetShape(3);
    const unsigned rawW = (unsigned)src.GetShape(4);
    const unsigned C0 = (unsigned)src.GetValidShape(5);
    const unsigned srcC0 = (unsigned)src.GetShape(5);
    const unsigned srcH = rawW * srcC0;
    const unsigned srcC1 = rawH * srcH;
    const unsigned srcN = rawC1 * srcC1;
    const unsigned srcG = rawN * srcN;
    // dst: GNCHW [G][N][C][H][W]
    const unsigned dstW = (unsigned)dst.GetShape(4);
    const unsigned dstC = (unsigned)dst.GetShape(3) * dstW;
    const unsigned dstN = (unsigned)dst.GetShape(2) * dstC;
    const unsigned dstG = (unsigned)dst.GetShape(1) * dstN;

    const unsigned vG = (unsigned)src.GetValidShape(0);
    const unsigned vN = (unsigned)src.GetValidShape(1);
    const unsigned vC1 = (unsigned)src.GetValidShape(2);
    const unsigned vH = (unsigned)src.GetValidShape(3);
    const unsigned vW = (unsigned)src.GetValidShape(4);
    const unsigned vC0 = (unsigned)src.GetValidShape(5);

    const bool vecOk = (vC0 == C0) && TTransConvBlockAligned(dstC, blockSizeElem) &&
                       TTransConvBlockAligned(dstW, blockSizeElem) && TTransConvBlockAligned(srcC0, blockSizeElem);
    const bool wContiguous = (vW == rawW) && (dstW == rawW);
    const unsigned xMax = wContiguous ? (vH * dstW) : vW;
    const unsigned tmpNeed = C0 * (((xMax + Y_ELEM_OTHER - 1) / Y_ELEM_OTHER) * Y_ELEM_OTHER);
    const bool vec = vecOk && ((unsigned)(TileDataTmp::Rows * TileDataTmp::Cols) >= tmpNeed);
    const bool needScalar = !vec || (xMax % blockSizeElem != 0);

    if (vec) {
        for (unsigned g = 0; g < vG; ++g) {
            for (unsigned n = 0; n < vN; ++n) {
                for (unsigned c1 = 0; c1 < vC1; ++c1) {
                    __ubuf__ T* const s = srcPtr + g * srcG + n * srcN + c1 * srcC1;
                    __ubuf__ T* const d = dstPtr + g * dstG + n * dstN + c1 * C0 * dstC;
                    if (wContiguous) {
                        TTransConvTranspose<T, blockSizeElem>(
                            d, s, tmpPtr, ((vH * dstW) / blockSizeElem) * blockSizeElem, C0, dstC, srcC0);
                    } else {
                        unsigned xVec = (vW / blockSizeElem) * blockSizeElem;
                        for (unsigned h = 0; h < vH; ++h) {
                            TTransConvTranspose<T, blockSizeElem>(
                                d + h * dstW, s + h * srcH, tmpPtr, xVec, C0, dstC, srcC0);
                        }
                    }
                }
            }
        }
    }

    if (needScalar) {
        TTransConvEnterScalar();
        for (unsigned g = 0; g < vG; ++g) {
            for (unsigned n = 0; n < vN; ++n) {
                for (unsigned c1 = 0; c1 < vC1; ++c1) {
                    __ubuf__ T* const s = srcPtr + g * srcG + n * srcN + c1 * srcC1;
                    __ubuf__ T* const d = dstPtr + g * dstG + n * dstN + c1 * C0 * dstC;
                    const unsigned xVec = wContiguous ? (((vH * dstW) / blockSizeElem) * blockSizeElem) :
                                                        ((vW / blockSizeElem) * blockSizeElem);
                    if (vec) {
                        if (wContiguous) {
                            unsigned xTotal = vH * dstW;
                            for (unsigned x = xVec; x < xTotal; ++x) {
                                TTransConvTransposeColumn<T>(d + x, s + x * srcC0, C0, dstC, 1);
                            }
                        } else {
                            for (unsigned h = 0; h < vH; ++h) {
                                for (unsigned w = xVec; w < vW; ++w) {
                                    TTransConvTransposeColumn<T>(
                                        d + h * dstW + w, s + h * srcH + w * srcC0, C0, dstC, 1);
                                }
                            }
                        }
                    } else {
                        for (unsigned h = 0; h < vH; ++h) {
                            for (unsigned w = 0; w < vW; ++w) {
                                __ubuf__ T* const dd = d + h * dstW + w;
                                __ubuf__ T* const ss = s + h * srcH + w * srcC0;
                                for (unsigned c0 = 0; c0 < vC0; ++c0) {
                                    dd[c0 * dstC] = ss[c0];
                                }
                            }
                        }
                    }
                }
            }
        }
        TTransConvExitScalar();
    }
}

// ---------- NC1HWC0 / GNC1HWC0 -> FRACTAL_Z ----------
// Every element is exactly one 32B block, so the copy is a pure block move from the src (h, w) rows
// to dst dim0 d = flat(g, c1, h, w); the N0 tail of the last N1 block is filled with +0.
// With hasGroup == false the source is a 4D NC1HWC0 (implicit G = 1).
template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, unsigned blockSizeElem, bool hasGroup>
__tf__ PTO_INTERNAL void TTransConvC2FractalZ(TileDataDst& dst, TileDataSrc& src, TileDataTmp& tmp)
{
    using T = typename TileDataSrc::DType;
    constexpr unsigned kOff = hasGroup ? 1U : 0U;

    __ubuf__ T* const dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst.data());
    __ubuf__ T* const srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src.data());
    __ubuf__ T* const tmpPtr = (__ubuf__ T*)__cce_get_tile_ptr(tmp.data());

    // src: [G][N][C1][H][W][C0]
    const unsigned rawN = (unsigned)src.GetShape(kOff);
    const unsigned rawC1 = (unsigned)src.GetShape(kOff + 1);
    const unsigned rawH = (unsigned)src.GetShape(kOff + 2);
    const unsigned rawW = (unsigned)src.GetShape(kOff + 3);
    const unsigned C0 = (unsigned)src.GetValidShape(kOff + 4); // 语义块宽
    const unsigned srcC0 = (unsigned)src.GetShape(kOff + 4);
    const unsigned srcH = rawW * srcC0;
    const unsigned srcC1 = rawH * srcH;
    const unsigned srcN = rawC1 * srcC1;
    const unsigned srcG = hasGroup ? ((unsigned)src.GetShape(kOff) * srcN) : 0;

    const unsigned vG = hasGroup ? (unsigned)src.GetValidShape(0) : 1;
    const unsigned vN = (unsigned)src.GetValidShape(kOff);
    const unsigned vC1 = (unsigned)src.GetValidShape(kOff + 1);
    const unsigned vH = (unsigned)src.GetValidShape(kOff + 2);
    const unsigned vW = (unsigned)src.GetValidShape(kOff + 3);
    const unsigned vC0 = (unsigned)src.GetValidShape(kOff + 4);

    // dst: FRACTAL_Z [flat][N1][N0][C0]；N0/C0 的语义取有效值，地址 stride 取声明值
    const unsigned dstN0Raw = (unsigned)dst.GetShape(2);
    const unsigned dstC0 = (unsigned)dst.GetShape(3);
    const unsigned N0 = (unsigned)dst.GetValidShape(2);
    const unsigned dstD = (unsigned)dst.GetShape(1) * dstN0Raw * dstC0;
    const unsigned vN1 = (unsigned)dst.GetValidShape(1);
    const unsigned hullN = vN1 * N0; // the last N1 block is padded to N0
    const unsigned dPerG = vC1 * vH * vW;

    const unsigned lenBurst = C0 * sizeof(T) / BLOCK_BYTE_SIZE; // exactly one block
    const bool blockOk = (vC0 == C0) && TTransConvBlockAligned(srcC0, blockSizeElem) &&
                         TTransConvBlockAligned(srcH, blockSizeElem) && TTransConvBlockAligned(srcC1, blockSizeElem) &&
                         TTransConvBlockAligned(srcN, blockSizeElem) &&
                         (!hasGroup || TTransConvBlockAligned(srcG, blockSizeElem)) &&
                         TTransConvBlockAligned(dstC0, blockSizeElem) && TTransConvBlockAligned(dstD, blockSizeElem);
    const unsigned dstGapBlocks = dstD * sizeof(T) / BLOCK_BYTE_SIZE - lenBurst;
    const unsigned srcGapBlocks = (srcC0 - C0) * sizeof(T) / BLOCK_BYTE_SIZE;
    const bool dmaOk = blockOk && dstGapBlocks <= 0xFFFFU && srcGapBlocks <= 0xFFFFU;
    // 源在 (c1, h, w) 上稠密时可整段搬（W、H 均未收缩），否则逐 h 行搬
    const bool coarse = dmaOk && (vW == rawW) && (vH == rawH) && (dPerG <= 0xFFFFU);
    const unsigned scratchBytes = (unsigned)(TileDataTmp::Rows * TileDataTmp::Cols) * sizeof(T);
    const bool scratchOk = scratchBytes >= TTransConvZeroScopeBytes(vW * C0 * sizeof(T));

    if (dmaOk) {
        // n 的地址偏移用增量维护：nOff(n) = (n/N0)*dstN0Raw*C0 + (n%N0)*C0。不能直接调
        // TTransConvFractalZNOffset——N0 是运行期值，n/N0 会变成真除法（实测这一项就要 ~0.3us）。
        // 逐 (c1,h) 把 srcRow 提到 n 循环外面（每次拷贝只做一次乘法），与原实现一致。
        for (unsigned g = 0; g < vG; ++g) {
            if (coarse) {
                unsigned nOff = 0;
                unsigned n0Idx = 0;
                for (unsigned n = 0; n < vN; ++n) {
                    pto_copy_ubuf_to_ubuf(
                        dstPtr + g * dPerG * dstD + nOff, srcPtr + g * srcG + n * srcN, (uint16_t)dPerG,
                        (uint16_t)lenBurst, 0, (uint16_t)dstGapBlocks);
                    TTransConvNAdvance(nOff, n0Idx, N0, dstN0Raw, dstC0);
                }
                continue;
            }
            for (unsigned c1 = 0; c1 < vC1; ++c1) {
                for (unsigned h = 0; h < vH; ++h) {
                    const unsigned d = (hasGroup ? ((g * vC1 + c1) * vH + h) : ((c1 * vH) + h)) * vW;
                    __ubuf__ T* const srcRow = srcPtr + g * srcG + c1 * srcC1 + h * srcH;
                    unsigned nOff = 0;
                    unsigned n0Idx = 0;
                    for (unsigned n = 0; n < vN; ++n) {
                        pto_copy_ubuf_to_ubuf(
                            dstPtr + d * dstD + nOff, srcRow + n * srcN, (uint16_t)vW, (uint16_t)lenBurst,
                            (uint16_t)srcGapBlocks, (uint16_t)dstGapBlocks);
                        TTransConvNAdvance(nOff, n0Idx, N0, dstN0Raw, dstC0);
                    }
                }
            }
        }
    } else {
        TTransConvEnterScalar();
        for (unsigned g = 0; g < vG; ++g) {
            for (unsigned n = 0; n < vN; ++n) {
                for (unsigned c1 = 0; c1 < vC1; ++c1) {
                    for (unsigned h = 0; h < vH; ++h) {
                        const unsigned d = (hasGroup ? ((g * vC1 + c1) * vH + h) : ((c1 * vH) + h)) * vW;
                        __ubuf__ T* const dd = dstPtr + d * dstD + TTransConvFractalZNOffset<T>(n, N0, dstN0Raw, dstC0);
                        __ubuf__ T* const ss = srcPtr + g * srcG + n * srcN + c1 * srcC1 + h * srcH;
                        for (unsigned w = 0; w < vW; ++w) {
                            for (unsigned c0 = 0; c0 < C0; ++c0) {
                                dd[w * dstD + c0] = ss[w * srcC0 + c0];
                            }
                        }
                    }
                }
            }
        }
        TTransConvExitScalar();
    }

    // N0 tail of the last N1 block
    if (hullN > vN) {
        if (dmaOk && scratchOk) {
            TTransConvZeroScope<T>(tmpPtr, vW * C0);
            pipe_barrier(PIPE_V); // 暂存区清零与随后的 UB 拷贝之间必须有屏障（与 TTransRepeatXOperation 一致）
            // 尾部起始偏移只依赖 (N0, vN)：算一次，之后用增量推进（同数据段，避免运行期除法）
            const unsigned tailOff0 = TTransConvFractalZNOffset<T>(vN, N0, dstN0Raw, dstC0);
            const unsigned tailIdx0 = vN % N0;
            for (unsigned g = 0; g < vG; ++g) {
                for (unsigned c1 = 0; c1 < vC1; ++c1) {
                    for (unsigned h = 0; h < vH; ++h) {
                        const unsigned d = (hasGroup ? ((g * vC1 + c1) * vH + h) : ((c1 * vH) + h)) * vW;
                        unsigned off = tailOff0;
                        unsigned idx = tailIdx0;
                        for (unsigned r = vN; r < hullN; ++r) {
                            pto_copy_ubuf_to_ubuf(
                                dstPtr + d * dstD + off, tmpPtr, (uint16_t)vW, (uint16_t)lenBurst, 0,
                                (uint16_t)dstGapBlocks);
                            if (++idx == N0) {
                                idx = 0;
                                off += dstN0Raw * dstC0 - (N0 - 1) * dstC0;
                            } else {
                                off += dstC0;
                            }
                        }
                    }
                }
            }
        } else {
            TTransConvEnterScalar();
            for (unsigned g = 0; g < vG; ++g) {
                for (unsigned c1 = 0; c1 < vC1; ++c1) {
                    for (unsigned h = 0; h < vH; ++h) {
                        const unsigned d = (hasGroup ? ((g * vC1 + c1) * vH + h) : ((c1 * vH) + h)) * vW;
                        for (unsigned r = vN; r < hullN; ++r) {
                            __ubuf__ T* const dd =
                                dstPtr + d * dstD + TTransConvFractalZNOffset<T>(r, N0, dstN0Raw, dstC0);
                            for (unsigned w = 0; w < vW; ++w) {
                                TTransConvZeroScalar<T>(dd + w * dstD, C0);
                            }
                        }
                    }
                }
            }
            TTransConvExitScalar();
        }
    }
}

// ---------- NCHW -> FRACTAL_Z ----------
// Writes the NCHW (C0, hw) transpose straight into dim0 of FRACTAL_Z (one 32B block per d).
// Flat mapping: the valid layout is d = (c1 * vH + h) * vW + w; the raw layout (dst dim0 declared as
// C1 * HW_padded) is d = c1 * HW_padded + h * rawW + w with each c1's HW pad segment zeroed.
template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, unsigned blockSizeElem>
__tf__ PTO_INTERNAL void TTransConvNCHW2FractalZ(TileDataDst& dst, TileDataSrc& src, TileDataTmp& tmp)
{
    using T = typename TileDataSrc::DType;

    __ubuf__ T* const dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst.data());
    __ubuf__ T* const srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src.data());
    __ubuf__ T* const tmpPtr = (__ubuf__ T*)__cce_get_tile_ptr(tmp.data());

    // src: NCHW [N][C][H][W]
    const unsigned rawC = (unsigned)src.GetShape(1);
    const unsigned rawH = (unsigned)src.GetShape(2);
    const unsigned rawW = (unsigned)src.GetShape(3);
    const unsigned srcC = rawH * rawW;
    const unsigned srcN = rawC * srcC;
    const unsigned rawHW = srcC;
    const unsigned rawHWPad = ((rawHW + blockSizeElem - 1) / blockSizeElem) * blockSizeElem;

    const unsigned vN = (unsigned)src.GetValidShape(0);
    const unsigned vC = (unsigned)src.GetValidShape(1);
    const unsigned vH = (unsigned)src.GetValidShape(2);
    const unsigned vW = (unsigned)src.GetValidShape(3);

    // dst: FRACTAL_Z [flat][N1][N0][C0]；N0/C0 的语义取有效值，地址 stride 取声明值
    const unsigned dstN0Raw = (unsigned)dst.GetShape(2);
    const unsigned dstC0 = (unsigned)dst.GetShape(3);
    const unsigned N0 = (unsigned)dst.GetValidShape(2);
    const unsigned C0 = (unsigned)dst.GetValidShape(3);
    const unsigned rawC1 = (rawC + C0 - 1) / C0;
    const unsigned vC1 = (vC + C0 - 1) / C0;
    const unsigned fullC1 = vC / C0;
    const unsigned dstD = (unsigned)dst.GetShape(1) * dstN0Raw * dstC0;
    const unsigned vN1 = (unsigned)dst.GetValidShape(1);
    const unsigned d0 = (unsigned)dst.GetValidShape(0);
    const unsigned hullN = vN1 * N0;

    // dst dim0 declares which flat layout it wants: the dense valid one, or the raw padded one
    const bool validFlat = (d0 == vC1 * vH * vW);
    const unsigned flatW = validFlat ? vW : rawW;
    const unsigned flatHW = validFlat ? (vH * vW) : rawHWPad;
    const unsigned flatC1Max = validFlat ? vC1 : rawC1;
    const unsigned flatC1 = (flatC1Max < vC1) ? flatC1Max : vC1; // never read past the valid C
    const unsigned flatPad = validFlat ? 0 : (rawHWPad - rawHW);

    const bool vecOk = TTransConvBlockAligned(dstD, blockSizeElem) && TTransConvBlockAligned(dstC0, blockSizeElem) &&
                       TTransConvBlockAligned(dstC0 * sizeof(T) / BLOCK_BYTE_SIZE * BLOCK_BYTE_SIZE, blockSizeElem) &&
                       (dstD * sizeof(T) / BLOCK_BYTE_SIZE) <= 0xFFFFU;
    const bool wContiguous = (vW == rawW);
    const unsigned xMax = wContiguous ? (vH * rawW) : vW;
    const unsigned xRun = (xMax > 255U * blockSizeElem) ? (255U * blockSizeElem) : xMax;
    const unsigned tmpNeed = xRun * (((C0 + Y_ELEM_OTHER - 1) / Y_ELEM_OTHER) * Y_ELEM_OTHER);
    const bool vec = vecOk && ((unsigned)(TileDataTmp::Rows * TileDataTmp::Cols) >= tmpNeed);
    const bool needScalar = !vec || (vC % C0 != 0) || (xMax % blockSizeElem != 0);

    if (vec) {
        const unsigned xVec =
            wContiguous ? (((vH * rawW) / blockSizeElem) * blockSizeElem) : ((vW / blockSizeElem) * blockSizeElem);
        for (unsigned c1 = 0; c1 < fullC1; ++c1) {
            __ubuf__ T* const dstC1 = dstPtr + (c1 * flatHW) * dstD;
            __ubuf__ T* const srcC1 = srcPtr + c1 * C0 * srcC;
            unsigned nOff = 0; // (n/N0)*dstN0Raw*C0 + (n%N0)*C0，增量维护（N0 是运行期值，除法很贵）
            unsigned n0Idx = 0;
            for (unsigned n = 0; n < vN; ++n) {
                __ubuf__ T* const dstD0 = dstC1 + nOff;
                __ubuf__ T* const srcC0 = srcC1 + n * srcN;
                if (wContiguous) {
                    TTransConvTransposeRows<T, blockSizeElem>(dstD0, srcC0, tmpPtr, C0, xVec, dstD, srcC);
                } else {
                    for (unsigned h = 0; h < vH; ++h) {
                        TTransConvTransposeRows<T, blockSizeElem>(
                            dstD0 + (h * flatW) * dstD, srcC0 + h * rawW, tmpPtr, C0, xVec, dstD, srcC);
                    }
                }
                TTransConvNAdvance(nOff, n0Idx, N0, dstN0Raw, dstC0);
            }
        }
    }

    if (needScalar || flatPad > 0) {
        TTransConvEnterScalar();
        for (unsigned c1 = 0; c1 < flatC1; ++c1) {
            const unsigned cRows = (c1 < fullC1) ? C0 : (vC - fullC1 * C0);
            for (unsigned n = 0; n < vN; ++n) {
                __ubuf__ T* const dstD0 =
                    dstPtr + (c1 * flatHW) * dstD + TTransConvFractalZNOffset<T>(n, N0, dstN0Raw, dstC0);
                __ubuf__ T* const srcC0 = srcPtr + n * srcN + c1 * C0 * srcC;
                if (vec && cRows == C0) {
                    if (wContiguous) {
                        unsigned xTotal = vH * rawW;
                        unsigned xVec = (xTotal / blockSizeElem) * blockSizeElem;
                        for (unsigned x = xVec; x < xTotal; ++x) {
                            TTransConvTransposeColumn<T>(dstD0 + x * dstD, srcC0 + x, C0, 1, srcC);
                        }
                    } else {
                        unsigned xVec = (vW / blockSizeElem) * blockSizeElem;
                        for (unsigned h = 0; h < vH; ++h) {
                            for (unsigned w = xVec; w < vW; ++w) {
                                TTransConvTransposeColumn<T>(
                                    dstD0 + (h * flatW + w) * dstD, srcC0 + h * rawW + w, C0, 1, srcC);
                            }
                        }
                    }
                } else {
                    for (unsigned h = 0; h < vH; ++h) {
                        for (unsigned w = 0; w < vW; ++w) {
                            __ubuf__ T* const dd = dstD0 + (h * flatW + w) * dstD;
                            __ubuf__ T* const ss = srcC0 + h * rawW + w;
                            for (unsigned c0 = 0; c0 < C0; ++c0) {
                                dd[c0] = (c0 < cRows) ? ss[c0 * srcC] : static_cast<T>(0);
                            }
                        }
                    }
                }
            }
        }
        // raw 布局下每个 c1 的 HW 补齐段清零
        for (unsigned c1 = 0; c1 < flatC1 && flatPad > 0; ++c1) {
            for (unsigned n = 0; n < vN; ++n) {
                __ubuf__ T* const dstPad =
                    dstPtr + (c1 * flatHW + rawHW) * dstD + TTransConvFractalZNOffset<T>(n, N0, dstN0Raw, dstC0);
                for (unsigned p = 0; p < flatPad; ++p) {
                    TTransConvZeroScalar<T>(dstPad + p * dstD, C0);
                }
            }
        }
        TTransConvExitScalar();
    }

    // N0 tail of the last N1 block：每个 d 一个 C0 块，整块清零
    if (hullN > vN) {
        const unsigned lenBurst = C0 * sizeof(T) / BLOCK_BYTE_SIZE;
        const unsigned gapBlocks = dstD * sizeof(T) / BLOCK_BYTE_SIZE - lenBurst;
        const bool tailVec = vec && (lenBurst > 0) && (C0 * sizeof(T) % BLOCK_BYTE_SIZE == 0) &&
                             ((unsigned)(TileDataTmp::Rows * TileDataTmp::Cols) * sizeof(T) >=
                              TTransConvZeroScopeBytes(d0 * C0 * sizeof(T))) &&
                             (d0 <= 0xFFFFU) && (gapBlocks <= 0xFFFFU);
        if (tailVec) {
            // 拷贝源每个 burst 前进一个块（srcGap=0），所以暂存区里 d0 个块都要是 +0
            TTransConvZeroScope<T>(tmpPtr, d0 * C0);
            pipe_barrier(PIPE_V);
            for (unsigned r = vN; r < hullN; ++r) {
                __ubuf__ T* const dstN = dstPtr + TTransConvFractalZNOffset<T>(r, N0, dstN0Raw, dstC0);
                pto_copy_ubuf_to_ubuf(dstN, tmpPtr, (uint16_t)d0, (uint16_t)lenBurst, 0, (uint16_t)gapBlocks);
            }
        } else {
            TTransConvEnterScalar();
            for (unsigned r = vN; r < hullN; ++r) {
                __ubuf__ T* const dstN = dstPtr + TTransConvFractalZNOffset<T>(r, N0, dstN0Raw, dstC0);
                for (unsigned d = 0; d < d0; ++d) {
                    TTransConvZeroScalar<T>(dstN + d * dstD, C0);
                }
            }
            TTransConvExitScalar();
        }
    }
}

// ---------- NCDHW -> FRACTAL_Z_3D ----------
// Per D plane, writes the NCHW (C0, hw) transpose straight into dim0 of FRACTAL_Z_3D, with the flat
// index (d * vC1 + c1) * vH * vW + h * vW + w.
template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, unsigned blockSizeElem>
__tf__ PTO_INTERNAL void TTransConvNCDHW2FractalZ3D(TileDataDst& dst, TileDataSrc& src, TileDataTmp& tmp)
{
    using T = typename TileDataSrc::DType;

    __ubuf__ T* const dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst.data());
    __ubuf__ T* const srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src.data());
    __ubuf__ T* const tmpPtr = (__ubuf__ T*)__cce_get_tile_ptr(tmp.data());

    // src: NCDHW [N][C][D][H][W]
    const unsigned rawC = (unsigned)src.GetShape(1);
    const unsigned rawD = (unsigned)src.GetShape(2);
    const unsigned rawH = (unsigned)src.GetShape(3);
    const unsigned rawW = (unsigned)src.GetShape(4);
    const unsigned rawHW = rawH * rawW;
    const unsigned srcC = rawD * rawHW; // C row stride inside one plane
    const unsigned srcN = rawC * srcC;

    const unsigned vN = (unsigned)src.GetValidShape(0);
    const unsigned vC = (unsigned)src.GetValidShape(1);
    const unsigned vD = (unsigned)src.GetValidShape(2);
    const unsigned vH = (unsigned)src.GetValidShape(3);
    const unsigned vW = (unsigned)src.GetValidShape(4);

    // dst: FRACTAL_Z_3D [flat][N1][N0][C0]；N0/C0 的语义取有效值，地址 stride 取声明值
    const unsigned dstN0Raw = (unsigned)dst.GetShape(2);
    const unsigned dstC0 = (unsigned)dst.GetShape(3);
    const unsigned N0 = (unsigned)dst.GetValidShape(2);
    const unsigned C0 = (unsigned)dst.GetValidShape(3);
    const unsigned vC1 = (vC + C0 - 1) / C0;
    const unsigned fullC1 = vC / C0;
    const unsigned dstD = (unsigned)dst.GetShape(1) * dstN0Raw * dstC0;
    const unsigned vN1 = (unsigned)dst.GetValidShape(1);
    const unsigned d0 = (unsigned)dst.GetValidShape(0);
    const unsigned hullN = vN1 * N0;

    const bool vecOk = TTransConvBlockAligned(dstD, blockSizeElem) && TTransConvBlockAligned(dstC0, blockSizeElem) &&
                       (dstD * sizeof(T) / BLOCK_BYTE_SIZE) <= 0xFFFFU;
    const bool wContiguous = (vW == rawW);
    const unsigned xMax = wContiguous ? (vH * rawW) : vW;
    const unsigned xRun = (xMax > 255U * blockSizeElem) ? (255U * blockSizeElem) : xMax;
    const unsigned tmpNeed = xRun * (((C0 + Y_ELEM_OTHER - 1) / Y_ELEM_OTHER) * Y_ELEM_OTHER);
    const bool vec = vecOk && ((unsigned)(TileDataTmp::Rows * TileDataTmp::Cols) >= tmpNeed);
    const bool needScalar = !vec || (vC % C0 != 0) || (xMax % blockSizeElem != 0);

    if (vec) {
        const unsigned xVec =
            wContiguous ? (((vH * rawW) / blockSizeElem) * blockSizeElem) : ((vW / blockSizeElem) * blockSizeElem);
        for (unsigned d = 0; d < vD; ++d) {
            for (unsigned c1 = 0; c1 < fullC1; ++c1) {
                __ubuf__ T* const dstDC1 = dstPtr + ((d * vC1 + c1) * vH * vW) * dstD;
                __ubuf__ T* const srcDC1 = srcPtr + c1 * C0 * srcC + d * rawHW;
                unsigned nOff = 0; // 增量维护，避免运行期除法（同 FRACTAL_Z 重排）
                unsigned n0Idx = 0;
                for (unsigned n = 0; n < vN; ++n) {
                    __ubuf__ T* const dstD0 = dstDC1 + nOff;
                    __ubuf__ T* const srcC0 = srcDC1 + n * srcN;
                    if (wContiguous) {
                        TTransConvTransposeRows<T, blockSizeElem>(dstD0, srcC0, tmpPtr, C0, xVec, dstD, srcC);
                    } else {
                        for (unsigned h = 0; h < vH; ++h) {
                            TTransConvTransposeRows<T, blockSizeElem>(
                                dstD0 + h * vW * dstD, srcC0 + h * rawW, tmpPtr, C0, xVec, dstD, srcC);
                        }
                    }
                    TTransConvNAdvance(nOff, n0Idx, N0, dstN0Raw, dstC0);
                }
            }
        }
    }

    if (needScalar) {
        TTransConvEnterScalar();
        for (unsigned d = 0; d < vD; ++d) {
            for (unsigned c1 = 0; c1 < vC1; ++c1) {
                const unsigned cRows = (c1 < fullC1) ? C0 : (vC - fullC1 * C0);
                for (unsigned n = 0; n < vN; ++n) {
                    __ubuf__ T* const dstD0 = dstPtr + ((d * vC1 + c1) * vH * vW) * dstD +
                                              TTransConvFractalZNOffset<T>(n, N0, dstN0Raw, dstC0);
                    __ubuf__ T* const srcC0 = srcPtr + n * srcN + c1 * C0 * srcC + d * rawHW;
                    if (vec && cRows == C0) {
                        if (wContiguous) {
                            unsigned xTotal = vH * rawW;
                            unsigned xVec = (xTotal / blockSizeElem) * blockSizeElem;
                            for (unsigned x = xVec; x < xTotal; ++x) {
                                TTransConvTransposeColumn<T>(dstD0 + x * dstD, srcC0 + x, C0, 1, srcC);
                            }
                        } else {
                            unsigned xVec = (vW / blockSizeElem) * blockSizeElem;
                            for (unsigned h = 0; h < vH; ++h) {
                                for (unsigned w = xVec; w < vW; ++w) {
                                    TTransConvTransposeColumn<T>(
                                        dstD0 + (h * vW + w) * dstD, srcC0 + h * rawW + w, C0, 1, srcC);
                                }
                            }
                        }
                    } else {
                        for (unsigned h = 0; h < vH; ++h) {
                            for (unsigned w = 0; w < vW; ++w) {
                                __ubuf__ T* const dd = dstD0 + (h * vW + w) * dstD;
                                __ubuf__ T* const ss = srcC0 + h * rawW + w;
                                for (unsigned c0 = 0; c0 < C0; ++c0) {
                                    dd[c0] = (c0 < cRows) ? ss[c0 * srcC] : static_cast<T>(0);
                                }
                            }
                        }
                    }
                }
            }
        }
        TTransConvExitScalar();
    }

    // N0 tail of the last N1 block：每个 d 一个 C0 块，整块清零
    if (hullN > vN) {
        const unsigned lenBurst = C0 * sizeof(T) / BLOCK_BYTE_SIZE;
        const unsigned gapBlocks = dstD * sizeof(T) / BLOCK_BYTE_SIZE - lenBurst;
        const bool tailVec = vec && (lenBurst > 0) && (C0 * sizeof(T) % BLOCK_BYTE_SIZE == 0) &&
                             ((unsigned)(TileDataTmp::Rows * TileDataTmp::Cols) * sizeof(T) >=
                              TTransConvZeroScopeBytes(d0 * C0 * sizeof(T))) &&
                             (d0 <= 0xFFFFU) && (gapBlocks <= 0xFFFFU);
        if (tailVec) {
            // 拷贝源每个 burst 前进一个块（srcGap=0），所以暂存区里 d0 个块都要是 +0
            TTransConvZeroScope<T>(tmpPtr, d0 * C0);
            pipe_barrier(PIPE_V);
            for (unsigned r = vN; r < hullN; ++r) {
                __ubuf__ T* const dstN = dstPtr + TTransConvFractalZNOffset<T>(r, N0, dstN0Raw, dstC0);
                pto_copy_ubuf_to_ubuf(dstN, tmpPtr, (uint16_t)d0, (uint16_t)lenBurst, 0, (uint16_t)gapBlocks);
            }
        } else {
            TTransConvEnterScalar();
            for (unsigned r = vN; r < hullN; ++r) {
                __ubuf__ T* const dstN = dstPtr + TTransConvFractalZNOffset<T>(r, N0, dstN0Raw, dstC0);
                for (unsigned d = 0; d < d0; ++d) {
                    TTransConvZeroScalar<T>(dstN + d * dstD, C0);
                }
            }
            TTransConvExitScalar();
        }
    }
}

// _DEBUG 校验：raw 形状只提供容量（UB 预算按 raw 算），各维关系按 valid 形状断言。
template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
PTO_INTERNAL void CheckConv3DTile(TileDataDst& dst, TileDataSrc& src, TileDataTmp& tmp)
{
#ifdef _DEBUG
    using T = typename TileDataSrc::DType;
    constexpr const int UB_SIZE = 196608; // 192*1024 B
    if constexpr (TileDataSrc::layout == Layout::NCDHW && TileDataDst::layout == Layout::FRACTAL_Z_3D) {
        const unsigned vN = src.GetValidShape(GlobalTensorDim::DIM_0);
        const unsigned vC = src.GetValidShape(GlobalTensorDim::DIM_1);
        const unsigned vD = src.GetValidShape(GlobalTensorDim::DIM_2);
        const unsigned vH = src.GetValidShape(GlobalTensorDim::DIM_3);
        const unsigned vW = src.GetValidShape(GlobalTensorDim::DIM_4);
        const unsigned dstD = dst.GetValidShape(GlobalTensorDim::DIM_0);
        const unsigned dstN1 = dst.GetValidShape(GlobalTensorDim::DIM_1);
        const unsigned dstN0 = dst.GetValidShape(GlobalTensorDim::DIM_2);
        const unsigned dstC0 = dst.GetValidShape(GlobalTensorDim::DIM_3);
        const unsigned vC1 = (vC + dstC0 - 1) / dstC0;
        unsigned srcSize = src.GetShape(GlobalTensorDim::DIM_0) * src.GetShape(GlobalTensorDim::DIM_1) *
                           src.GetShape(GlobalTensorDim::DIM_2) * src.GetShape(GlobalTensorDim::DIM_3) *
                           src.GetShape(GlobalTensorDim::DIM_4);
        unsigned dstSize = dst.GetShape(GlobalTensorDim::DIM_0) * dst.GetShape(GlobalTensorDim::DIM_1) *
                           dst.GetShape(GlobalTensorDim::DIM_2) * dst.GetShape(GlobalTensorDim::DIM_3);
        unsigned tmpSize = TileDataTmp::Rows * TileDataTmp::Cols;
        PTO_ASSERT(
            vD * vC1 * vH * vW == dstD && dstN1 == (vN + dstN0 - 1) / dstN0, "expect same size for src and dst.");
        PTO_ASSERT((srcSize + dstSize + tmpSize) * sizeof(T) < UB_SIZE, "ERROR: memory usage exceeds UB limit!");
    }
#endif
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
PTO_INTERNAL void CheckConvTile(TileDataDst& dst, TileDataSrc& src, TileDataTmp& tmp)
{
#ifdef _DEBUG
    using T = typename TileDataSrc::DType;
    constexpr const int UB_SIZE = 196608; // 192*1024 B
    if (TileDataSrc::layout == Layout::NCHW && TileDataDst::layout == Layout::NC1HWC0) {
        const unsigned vN = src.GetValidShape(GlobalTensorDim::DIM_0);
        const unsigned vC = src.GetValidShape(GlobalTensorDim::DIM_1);
        const unsigned vH = src.GetValidShape(GlobalTensorDim::DIM_2);
        const unsigned vW = src.GetValidShape(GlobalTensorDim::DIM_3);
        const unsigned dstC0 = dst.GetValidShape(GlobalTensorDim::DIM_4);
        unsigned srcSize = src.GetShape(GlobalTensorDim::DIM_0) * src.GetShape(GlobalTensorDim::DIM_1) *
                           src.GetShape(GlobalTensorDim::DIM_2) * src.GetShape(GlobalTensorDim::DIM_3);
        unsigned dstSize = dst.GetShape(GlobalTensorDim::DIM_0) * dst.GetShape(GlobalTensorDim::DIM_1) *
                           dst.GetShape(GlobalTensorDim::DIM_2) * dst.GetShape(GlobalTensorDim::DIM_3) *
                           dst.GetShape(GlobalTensorDim::DIM_4);
        unsigned tmpSize = TileDataTmp::Rows * TileDataTmp::Cols;
        PTO_ASSERT(
            vN == dst.GetValidShape(GlobalTensorDim::DIM_0) && vH == dst.GetValidShape(GlobalTensorDim::DIM_2) &&
                vW == dst.GetValidShape(GlobalTensorDim::DIM_3) &&
                dst.GetValidShape(GlobalTensorDim::DIM_1) == (vC + dstC0 - 1) / dstC0,
            "expect dst valid shape derived from src valid shape.");
        PTO_ASSERT((srcSize + dstSize + tmpSize) * sizeof(T) < UB_SIZE, "ERROR: memory usage exceeds UB limit!");
    } else if (TileDataSrc::layout == Layout::NC1HWC0 && TileDataDst::layout == Layout::NCHW) {
        const unsigned vN = src.GetValidShape(GlobalTensorDim::DIM_0);
        const unsigned vC1 = src.GetValidShape(GlobalTensorDim::DIM_1);
        const unsigned vH = src.GetValidShape(GlobalTensorDim::DIM_2);
        const unsigned vW = src.GetValidShape(GlobalTensorDim::DIM_3);
        const unsigned vC0 = src.GetValidShape(GlobalTensorDim::DIM_4);
        unsigned srcSize = src.GetShape(GlobalTensorDim::DIM_0) * src.GetShape(GlobalTensorDim::DIM_1) *
                           src.GetShape(GlobalTensorDim::DIM_2) * src.GetShape(GlobalTensorDim::DIM_3) *
                           src.GetShape(GlobalTensorDim::DIM_4);
        unsigned dstSize = dst.GetShape(GlobalTensorDim::DIM_0) * dst.GetShape(GlobalTensorDim::DIM_1) *
                           dst.GetShape(GlobalTensorDim::DIM_2) * dst.GetShape(GlobalTensorDim::DIM_3);
        unsigned tmpSize = TileDataTmp::Rows * TileDataTmp::Cols;
        PTO_ASSERT(
            vN == dst.GetValidShape(GlobalTensorDim::DIM_0) && vC1 * vC0 == dst.GetValidShape(GlobalTensorDim::DIM_1) &&
                vH == dst.GetValidShape(GlobalTensorDim::DIM_2) && vW == dst.GetValidShape(GlobalTensorDim::DIM_3),
            "expect dst valid shape derived from src valid shape.");
        PTO_ASSERT((srcSize + dstSize + tmpSize) * sizeof(T) < UB_SIZE, "ERROR: memory usage exceeds UB limit!");
    } else if (TileDataSrc::layout == Layout::NC1HWC0 && TileDataDst::layout == Layout::FRACTAL_Z) {
        const unsigned vN = src.GetValidShape(GlobalTensorDim::DIM_0);
        const unsigned vC1 = src.GetValidShape(GlobalTensorDim::DIM_1);
        const unsigned vH = src.GetValidShape(GlobalTensorDim::DIM_2);
        const unsigned vW = src.GetValidShape(GlobalTensorDim::DIM_3);
        const unsigned srcC0 = src.GetValidShape(GlobalTensorDim::DIM_4);
        unsigned srcSize = src.GetShape(GlobalTensorDim::DIM_0) * src.GetShape(GlobalTensorDim::DIM_1) *
                           src.GetShape(GlobalTensorDim::DIM_2) * src.GetShape(GlobalTensorDim::DIM_3) *
                           src.GetShape(GlobalTensorDim::DIM_4);
        unsigned dstSize = dst.GetShape(GlobalTensorDim::DIM_0) * dst.GetShape(GlobalTensorDim::DIM_1) *
                           dst.GetShape(GlobalTensorDim::DIM_2) * dst.GetShape(GlobalTensorDim::DIM_3);
        unsigned tmpSize = TileDataTmp::Rows * TileDataTmp::Cols;
        PTO_ASSERT((srcSize + dstSize + tmpSize) * sizeof(T) < UB_SIZE, "ERROR: memory usage exceeds UB limit!");
        PTO_ASSERT(
            vC1 * vH * vW == dst.GetValidShape(GlobalTensorDim::DIM_0) &&
                srcC0 == dst.GetValidShape(GlobalTensorDim::DIM_3) &&
                dst.GetValidShape(GlobalTensorDim::DIM_1) ==
                    (vN + dst.GetValidShape(GlobalTensorDim::DIM_2) - 1) / dst.GetValidShape(GlobalTensorDim::DIM_2),
            "expect dst valid shape derived from src valid shape.");
    } else if (TileDataSrc::layout == Layout::NCHW && TileDataDst::layout == Layout::FRACTAL_Z) {
        const unsigned vN = src.GetValidShape(GlobalTensorDim::DIM_0);
        const unsigned vC = src.GetValidShape(GlobalTensorDim::DIM_1);
        const unsigned vH = src.GetValidShape(GlobalTensorDim::DIM_2);
        const unsigned vW = src.GetValidShape(GlobalTensorDim::DIM_3);
        const unsigned dstC0 = dst.GetValidShape(GlobalTensorDim::DIM_3);
        const unsigned vC1 = (vC + dstC0 - 1) / dstC0;
        unsigned srcSize = src.GetShape(GlobalTensorDim::DIM_0) * src.GetShape(GlobalTensorDim::DIM_1) *
                           src.GetShape(GlobalTensorDim::DIM_2) * src.GetShape(GlobalTensorDim::DIM_3);
        unsigned dstSize = dst.GetShape(GlobalTensorDim::DIM_0) * dst.GetShape(GlobalTensorDim::DIM_1) *
                           dst.GetShape(GlobalTensorDim::DIM_2) * dst.GetShape(GlobalTensorDim::DIM_3);
        unsigned tmpSize = TileDataTmp::Rows * TileDataTmp::Cols;
        PTO_ASSERT((srcSize + dstSize + tmpSize) * sizeof(T) < UB_SIZE, "ERROR: memory usage exceeds UB limit!");
        // dim0 允许两种声明：valid 稠密（vC1 * vH * vW）或 raw 补齐（vC1 * HW_padded）
        constexpr unsigned kBlk = BLOCK_BYTE_SIZE / sizeof(T);
        const unsigned hwPadded = ((vH * vW + kBlk - 1) / kBlk) * kBlk;
        const unsigned flatLen = dst.GetValidShape(GlobalTensorDim::DIM_0);
        const unsigned dn0 = dst.GetValidShape(GlobalTensorDim::DIM_2);
        PTO_ASSERT(
            (flatLen == vC1 * vH * vW || flatLen == vC1 * hwPadded) &&
                dst.GetValidShape(GlobalTensorDim::DIM_1) == (vN + dn0 - 1) / dn0,
            "expect dst valid shape derived from src valid shape.");
    }
#endif
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
PTO_INTERNAL void CheckGroupConvTile(TileDataDst& dst, TileDataSrc& src, TileDataTmp& tmp)
{
#ifdef _DEBUG
    using T = typename TileDataSrc::DType;
    constexpr const int UB_SIZE = 196608; // 192*1024 B
    if (TileDataSrc::layout == Layout::GNCHW && TileDataDst::layout == Layout::GNC1HWC0) {
        const unsigned vG = src.GetValidShape(GlobalTensorDim::DIM_0);
        const unsigned vN = src.GetValidShape(GlobalTensorDim::DIM_1);
        const unsigned vC = src.GetValidShape(GlobalTensorDim::DIM_2);
        const unsigned vH = src.GetValidShape(GlobalTensorDim::DIM_3);
        const unsigned vW = src.GetValidShape(GlobalTensorDim::DIM_4);
        const unsigned dstC0 = dst.GetValidShape(GlobalTensorDim::TOTAL_DIM);
        unsigned srcSize = src.GetShape(GlobalTensorDim::DIM_0) * src.GetShape(GlobalTensorDim::DIM_1) *
                           src.GetShape(GlobalTensorDim::DIM_2) * src.GetShape(GlobalTensorDim::DIM_3) *
                           src.GetShape(GlobalTensorDim::DIM_4);
        unsigned dstSize = dst.GetShape(GlobalTensorDim::DIM_0) * dst.GetShape(GlobalTensorDim::DIM_1) *
                           dst.GetShape(GlobalTensorDim::DIM_2) * dst.GetShape(GlobalTensorDim::DIM_3) *
                           dst.GetShape(GlobalTensorDim::DIM_4) * dst.GetShape(GlobalTensorDim::TOTAL_DIM);
        unsigned tmpSize = TileDataTmp::Rows * TileDataTmp::Cols;
        PTO_ASSERT(
            vG == dst.GetValidShape(GlobalTensorDim::DIM_0) && vN == dst.GetValidShape(GlobalTensorDim::DIM_1) &&
                vH == dst.GetValidShape(GlobalTensorDim::DIM_3) && vW == dst.GetValidShape(GlobalTensorDim::DIM_4) &&
                dst.GetValidShape(GlobalTensorDim::DIM_2) == (vC + dstC0 - 1) / dstC0,
            "expect dst valid shape derived from src valid shape.");
        PTO_ASSERT((srcSize + dstSize + tmpSize) * sizeof(T) < UB_SIZE, "ERROR: memory usage exceeds UB limit!");
    } else if (TileDataSrc::layout == Layout::GNC1HWC0 && TileDataDst::layout == Layout::FRACTAL_Z) {
        const unsigned vG = src.GetValidShape(GlobalTensorDim::DIM_0);
        const unsigned vN = src.GetValidShape(GlobalTensorDim::DIM_1);
        const unsigned vC1 = src.GetValidShape(GlobalTensorDim::DIM_2);
        const unsigned vH = src.GetValidShape(GlobalTensorDim::DIM_3);
        const unsigned vW = src.GetValidShape(GlobalTensorDim::DIM_4);
        const unsigned srcC0 = src.GetValidShape(GlobalTensorDim::TOTAL_DIM);
        unsigned srcSize = src.GetShape(GlobalTensorDim::DIM_0) * src.GetShape(GlobalTensorDim::DIM_1) *
                           src.GetShape(GlobalTensorDim::DIM_2) * src.GetShape(GlobalTensorDim::DIM_3) *
                           src.GetShape(GlobalTensorDim::DIM_4) * src.GetShape(GlobalTensorDim::TOTAL_DIM);
        unsigned dstSize = dst.GetShape(GlobalTensorDim::DIM_0) * dst.GetShape(GlobalTensorDim::DIM_1) *
                           dst.GetShape(GlobalTensorDim::DIM_2) * dst.GetShape(GlobalTensorDim::DIM_3);
        unsigned tmpSize = TileDataTmp::Rows * TileDataTmp::Cols;
        PTO_ASSERT((srcSize + dstSize + tmpSize) * sizeof(T) < UB_SIZE, "ERROR: memory usage exceeds UB limit!");
        PTO_ASSERT(
            vG * vC1 * vH * vW == dst.GetValidShape(GlobalTensorDim::DIM_0) &&
                srcC0 == dst.GetValidShape(GlobalTensorDim::DIM_3) &&
                dst.GetValidShape(GlobalTensorDim::DIM_1) ==
                    (vN + dst.GetValidShape(GlobalTensorDim::DIM_2) - 1) / dst.GetValidShape(GlobalTensorDim::DIM_2),
            "expect dst valid shape derived from src valid shape.");
    }
#endif
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
PTO_INTERNAL void TTransImplConvTile(TileDataDst& dst, TileDataSrc& src, TileDataTmp& tmp)
{
    using T = typename TileDataSrc::DType;
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    // 每个转换只有一份实现：raw 形状供 stride/容量，valid 形状供处理域（见文件头说明）
    if constexpr (TileDataSrc::layout == Layout::NCHW && TileDataDst::layout == Layout::NC1HWC0) {
        CheckConvTile<TileDataDst, TileDataSrc, TileDataTmp>(dst, src, tmp);
        TTransConvNCHW2NC1HWC0<TileDataDst, TileDataSrc, TileDataTmp, blockSizeElem>(dst, src, tmp);
    } else if constexpr (TileDataSrc::layout == Layout::NC1HWC0 && TileDataDst::layout == Layout::NCHW) {
        CheckConvTile<TileDataDst, TileDataSrc, TileDataTmp>(dst, src, tmp);
        TTransConvNC1HWC02NCHW<TileDataDst, TileDataSrc, TileDataTmp, blockSizeElem>(dst, src, tmp);
    } else if constexpr (TileDataSrc::layout == Layout::GNCHW && TileDataDst::layout == Layout::GNC1HWC0) {
        CheckGroupConvTile<TileDataDst, TileDataSrc, TileDataTmp>(dst, src, tmp);
        TTransConvGNCHW2GNC1HWC0<TileDataDst, TileDataSrc, TileDataTmp, blockSizeElem>(dst, src, tmp);
    } else if constexpr (TileDataSrc::layout == Layout::NC1HWC0 && TileDataDst::layout == Layout::FRACTAL_Z) {
        CheckConvTile<TileDataDst, TileDataSrc, TileDataTmp>(dst, src, tmp);
        TTransConvC2FractalZ<TileDataDst, TileDataSrc, TileDataTmp, blockSizeElem, false>(dst, src, tmp);
    } else if constexpr (TileDataSrc::layout == Layout::GNC1HWC0 && TileDataDst::layout == Layout::FRACTAL_Z) {
        CheckGroupConvTile<TileDataDst, TileDataSrc, TileDataTmp>(dst, src, tmp);
        TTransConvC2FractalZ<TileDataDst, TileDataSrc, TileDataTmp, blockSizeElem, true>(dst, src, tmp);
    } else if constexpr (
        TileDataSrc::layout == Layout::NCHW &&
        (TileDataDst::layout == Layout::FRACTAL_Z || TileDataDst::layout == Layout::FRACTAL_Z_3D)) {
        // 4D NCHW 源产出 FRACTAL_Z_3D 等价于隐式 D = 1，与 FRACTAL_Z 布局相同，共用同一实现
        CheckConvTile<TileDataDst, TileDataSrc, TileDataTmp>(dst, src, tmp);
        TTransConvNCHW2FractalZ<TileDataDst, TileDataSrc, TileDataTmp, blockSizeElem>(dst, src, tmp);
    } else if constexpr (TileDataSrc::layout == Layout::NCDHW && TileDataDst::layout == Layout::FRACTAL_Z_3D) {
        CheckConv3DTile<TileDataDst, TileDataSrc, TileDataTmp>(dst, src, tmp);
        TTransConvNCDHW2FractalZ3D<TileDataDst, TileDataSrc, TileDataTmp, blockSizeElem>(dst, src, tmp);
    } else if constexpr (TileDataSrc::layout == Layout::GNC1HWC0 && TileDataDst::layout == Layout::GNCHW) {
        CheckGroupConvTile<TileDataDst, TileDataSrc, TileDataTmp>(dst, src, tmp);
        TTransConvGNC1HWC02GNCHW<TileDataDst, TileDataSrc, TileDataTmp, blockSizeElem>(dst, src, tmp);
    }
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
PTO_INTERNAL void TTRANS_IMPL(TileDataDst& dst, TileDataSrc& src, TileDataTmp& tmp)
{
    using T = typename TileDataSrc::DType;
    using U = typename TileDataDst::DType;
    static_assert(sizeof(T) == 4 || sizeof(T) == 2 || sizeof(T) == 1, "Fix: TTRANS has unsupported data type.");
    static_assert(sizeof(T) == sizeof(U), "Fix: TTRANS has inconsistent input and output data type.");
    if constexpr (is_conv_tile_v<TileDataSrc>) {
        TTransImplConvTile<TileDataDst, TileDataSrc, TileDataTmp>(dst, src, tmp);
        return;
    } else {
        static_assert(TileDataSrc::isRowMajor, "Fix: TTRANS has not supported layout type.");
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
        constexpr unsigned dstStride = TileDataDst::RowStride;
        constexpr unsigned srcStride = TileDataSrc::RowStride;

        unsigned validRow = src.GetValidRow();
        unsigned validCol = src.GetValidCol();
        TTrans<TileDataDst, TileDataSrc, TileDataTmp, blockSizeElem>(
            dst.data(), src.data(), tmp.data(), validRow, validCol, dstStride, srcStride);
    }
}
} // namespace pto
#endif
