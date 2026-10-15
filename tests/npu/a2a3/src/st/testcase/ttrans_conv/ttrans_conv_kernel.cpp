/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/common/debug.h>
#include "acl/acl.h"

using namespace pto;

// NCHW -> NC1HWC0
template <
    typename T, int dstN, int dstC1, int dstH, int dstW, int dstC0, int gWholeShape0, int gWholeShape1,
    int gWholeShape2, int gWholeShape3, int gWholeShape4>
__global__ AICORE void runTTRANSConv1(__gm__ T __out__* out, __gm__ T __in__* src)
{
    constexpr int bufferSize = dstN * dstC1 * dstH * dstW * dstC0 * sizeof(T);
    constexpr int validRow = dstN * dstC1 * dstH * dstW;
    constexpr int validCol = dstC0;
    static_assert(gWholeShape0 == 1, "");
    static_assert(gWholeShape1 == dstN, "");
    static_assert(dstC1 == (gWholeShape2 + dstC0 - 1) / dstC0, "");
    static_assert(gWholeShape3 == dstH, "");
    static_assert(gWholeShape4 == dstW, "");
    static_assert(dstH * dstW * sizeof(T) % 32 == 0, "");

    constexpr int elemNum = dstN * dstC1 * dstH * dstW * dstC0;

    using ShapeDim5 = Shape<1, 1, 1, 1, elemNum>;
    using StrideDim5 = pto::Stride<elemNum, elemNum, elemNum, elemNum, 1>;
    using GlobalDataIn = GlobalTensor<T, ShapeDim5, StrideDim5>;

    using SrcTileData = Tile<TileType::Vec, T, 1, elemNum, BLayout::RowMajor, 1, elemNum>;
    SrcTileData src0Tile;
    TASSIGN(src0Tile, 0x0);
    using TileData = ConvTile<TileType::Vec, T, elemNum, Layout::NCHW, ConvTileShape<dstN, dstC0 * dstC1, dstH, dstW>>;
    TileData srcTile;
    static_assert(srcTile.totalDimCount == 4);
    TASSIGN(srcTile, 0x0);
#ifdef __PTO_AUTO__
    TRESHAPE(src0Tile, srcTile);
#endif

    using DstTileData =
        ConvTile<TileType::Vec, T, elemNum, Layout::NC1HWC0, ConvTileShape<dstN, dstC1, dstH, dstW, dstC0>>;
    DstTileData dstTile;
    static_assert(dstTile.totalDimCount == 5);
    TASSIGN(dstTile, 0x0 + dstN * dstC1 * dstH * dstW * dstC0 * sizeof(T));
    SrcTileData dst0Tile;
#ifdef __PTO_AUTO__
    TRESHAPE(dst0Tile, dstTile);
#endif
    TASSIGN(dst0Tile, 0x0 + dstN * dstC1 * dstH * dstW * dstC0 * sizeof(T));

    constexpr int tmpTileH = dstH * dstW;
    constexpr unsigned yTileSizeElem = (sizeof(T) == 1) ? 32 : 16;
    constexpr int tmpTileW = (dstC0 + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem;
    using TmpTileData = Tile<TileType::Vec, T, tmpTileH, tmpTileW, BLayout::RowMajor, tmpTileH, tmpTileW>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + dstN * dstC1 * dstH * dstW * dstC0 * sizeof(T) * 2);

    GlobalDataIn srcGlobal(src);
    GlobalDataIn dstGlobal(out);
    TLOAD(src0Tile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
#endif
    TTRANS(dstTile, srcTile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dst0Tile);
}

template <
    typename T, int dstC1, int dstH, int dstW, int dstN1, int dstN0, int dstC0, int srcN, int srcC1, int srcH, int srcW,
    int srcC0>
__global__ AICORE void runTTRANSConv2(__gm__ T __out__* out, __gm__ T __in__* src)
{
    static_assert(srcC0 == dstC0);
    static_assert(srcW == dstW);
    static_assert(srcH == dstH);
    static_assert(dstN1 == (srcN + dstN0 - 1) / dstN0);

    constexpr int bufferSize = dstN1 * dstN0 * dstC1 * dstH * dstW * dstC0 * sizeof(T);
    constexpr int validRow = dstN1 * dstN0 * dstC1 * dstH * dstW;
    constexpr int validCol = dstC0;

    constexpr int elemNum = dstN1 * dstN0 * dstC1 * dstH * dstW * dstC0;

    using ShapeDim5 = Shape<1, 1, 1, 1, elemNum>;
    using StrideDim5 = pto::Stride<elemNum, elemNum, elemNum, elemNum, 1>;
    using GlobalDataIn = GlobalTensor<T, ShapeDim5, StrideDim5>;

    using SrcTileData = Tile<TileType::Vec, T, 1, elemNum, BLayout::RowMajor, 1, elemNum>;
    SrcTileData src0Tile;
    TASSIGN(src0Tile, 0x0);
    using TileData =
        ConvTile<TileType::Vec, T, elemNum, Layout::NC1HWC0, ConvTileShape<srcN, dstC1, dstH, dstW, dstC0>>;
    TileData srcTile;
    static_assert(srcTile.totalDimCount == 5);
    TASSIGN(srcTile, 0x0);
#ifdef __PTO_AUTO__
    TRESHAPE(src0Tile, srcTile);
#endif

    using DstTileData =
        ConvTile<TileType::Vec, T, elemNum, Layout::FRACTAL_Z, ConvTileShape<dstC1 * dstH * dstW, dstN1, dstN0, dstC0>>;
    using TmpTileData = Tile<TileType::Vec, T, 16, 32, BLayout::RowMajor, 16, 32>;
    DstTileData dstTile;
    static_assert(dstTile.totalDimCount == 4);
    SrcTileData dst0Tile;
#ifdef __PTO_AUTO__
    TRESHAPE(dst0Tile, dstTile);
#endif
    TASSIGN(dstTile, 0x0 + bufferSize);
    TASSIGN(dst0Tile, 0x0 + bufferSize);
    using ZeroTileData =
        Tile<TileType::Vec, int32_t, 1, elemNum * sizeof(T) / 4, BLayout::RowMajor, 1, elemNum * sizeof(T) / 4>;
    ZeroTileData dst1Tile;
#ifdef __PTO_AUTO__
    TRESHAPE(dst1Tile, dstTile);
#endif
    TASSIGN(dst1Tile, 0x0 + bufferSize);

    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + bufferSize * 2);

    GlobalDataIn srcGlobal(src);
    GlobalDataIn dstGlobal(out);
    TLOAD(src0Tile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TSUB(dst1Tile, dst1Tile, dst1Tile);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
#endif
    TTRANS(dstTile, srcTile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dst0Tile);
}

// NC1HWC0 -> NCHW
template <
    typename T, int srcN, int srcC1, int srcH, int srcW, int srcC0, int gWholeShape0, int gWholeShape1,
    int gWholeShape2, int gWholeShape3, int gWholeShape4>
__global__ AICORE void runTTRANSConv3(__gm__ T __out__* out, __gm__ T __in__* src)
{
    constexpr int elemNum = srcN * srcC1 * srcH * srcW * srcC0;
    constexpr int bufferSize = elemNum * sizeof(T);

    using ShapeDim5 = Shape<1, 1, 1, 1, elemNum>;
    using StrideDim5 = pto::Stride<elemNum, elemNum, elemNum, elemNum, 1>;
    using GlobalDataIn = GlobalTensor<T, ShapeDim5, StrideDim5>;

    using SrcTileData = Tile<TileType::Vec, T, 1, elemNum, BLayout::RowMajor, 1, elemNum>;
    SrcTileData src0Tile;
    TASSIGN(src0Tile, 0x0);
    using TileData =
        ConvTile<TileType::Vec, T, elemNum, Layout::NC1HWC0, ConvTileShape<srcN, srcC1, srcH, srcW, srcC0>>;
    TileData srcTile;
    static_assert(srcTile.totalDimCount == 5);
    TASSIGN(srcTile, 0x0);
#ifdef __PTO_AUTO__
    TRESHAPE(src0Tile, srcTile);
#endif

    using DstTileData =
        ConvTile<TileType::Vec, T, elemNum, Layout::NCHW, ConvTileShape<srcN, srcC1 * srcC0, srcH, srcW>>;
    DstTileData dstTile;
    static_assert(dstTile.totalDimCount == 4);
    TASSIGN(dstTile, 0x0 + bufferSize);
    SrcTileData dst0Tile;
#ifdef __PTO_AUTO__
    TRESHAPE(dst0Tile, dstTile);
#endif
    TASSIGN(dst0Tile, 0x0 + bufferSize);

    constexpr int tmpTileH = srcH * srcW;
    constexpr unsigned yTileSizeElem = (sizeof(T) == 1) ? 32 : 16;
    constexpr int tmpTileW = (srcC0 + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem;
    using TmpTileData = Tile<TileType::Vec, T, tmpTileH, tmpTileW, BLayout::RowMajor, tmpTileH, tmpTileW>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + bufferSize * 2);

    GlobalDataIn srcGlobal(src);
    GlobalDataIn dstGlobal(out);
    TLOAD(src0Tile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
#endif
    TTRANS(dstTile, srcTile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dst0Tile);
}

// NCHW -> FRACTAL_Z (three-stage direct conversion)
template <typename T, int srcN, int srcC, int srcH, int srcW, int dstC1HW, int dstN1, int dstN0, int dstC0>
__global__ AICORE void runTTRANSConv4(__gm__ T __out__* out, __gm__ T __in__* src)
{
    constexpr int DTypeSize = sizeof(T);
    constexpr int blockSize = 32 / DTypeSize;
    constexpr int HW = srcH * srcW;
    constexpr int HW_padded = ((HW + blockSize - 1) / blockSize) * blockSize;
    constexpr int C1 = (srcC + dstC0 - 1) / dstC0;
    constexpr int N_padded = dstN1 * dstN0;
    static_assert(dstC1HW == C1 * HW_padded, "");
    static_assert(dstN1 == (srcN + dstN0 - 1) / dstN0, "");
    static_assert(dstC0 == 32 / DTypeSize, "");

    constexpr int srcElemNum = srcN * srcC * HW;
    constexpr int srcBufferSize = srcElemNum * DTypeSize;

    constexpr int dstElemNum = dstC1HW * dstN1 * dstN0 * dstC0;
    constexpr int dstBufferSize = dstElemNum * DTypeSize;

    constexpr int tmpElemNum = N_padded * C1 * dstC0 * HW_padded;
    constexpr int tmpBufferSize = tmpElemNum * DTypeSize;

    using SrcShapeDim5 = Shape<1, 1, 1, 1, srcElemNum>;
    using SrcStrideDim5 = pto::Stride<srcElemNum, srcElemNum, srcElemNum, srcElemNum, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStrideDim5>;

    using DstShapeDim5 = Shape<1, 1, 1, 1, dstElemNum>;
    using DstStrideDim5 = pto::Stride<dstElemNum, dstElemNum, dstElemNum, dstElemNum, 1>;
    using DstGlobalData = GlobalTensor<T, DstShapeDim5, DstStrideDim5>;

    using SrcTileData = Tile<TileType::Vec, T, 1, srcElemNum, BLayout::RowMajor, 1, srcElemNum>;
    SrcTileData src0Tile;
    TASSIGN(src0Tile, 0x0);
    using SrcConvTile = ConvTile<TileType::Vec, T, srcBufferSize, Layout::NCHW, ConvTileShape<srcN, srcC, srcH, srcW>>;
    SrcConvTile srcTile;
    static_assert(srcTile.totalDimCount == 4);
    TASSIGN(srcTile, 0x0);
#ifdef __PTO_AUTO__
    TRESHAPE(src0Tile, srcTile);
#endif

    using DstTileData = Tile<TileType::Vec, T, 1, dstElemNum, BLayout::RowMajor, 1, dstElemNum>;
    DstTileData dst0Tile;
    TASSIGN(dst0Tile, 0x0 + srcBufferSize);
    using DstConvTile =
        ConvTile<TileType::Vec, T, dstBufferSize, Layout::FRACTAL_Z, ConvTileShape<dstC1HW, dstN1, dstN0, dstC0>>;
    DstConvTile dstTile;
    static_assert(dstTile.totalDimCount == 4);
    TASSIGN(dstTile, 0x0 + srcBufferSize);
#ifdef __PTO_AUTO__
    TRESHAPE(dst0Tile, dstTile);
#endif

    using TmpTileData = Tile<TileType::Vec, T, 1, tmpElemNum, BLayout::RowMajor, 1, tmpElemNum>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + srcBufferSize + dstBufferSize);

    SrcGlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    TLOAD(src0Tile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TTRANS(dstTile, srcTile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dst0Tile);
}

template <
    typename T, int format, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gShape5,
    int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
void LaunchTTRANSConv(T* out, T* src, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        if constexpr (format == 0) {
            runTTRANSConv1<
                half, gShape0, gShape1, gShape2, gShape3, gShape4, gWholeShape0, gWholeShape1, gWholeShape2,
                gWholeShape3, gWholeShape4><<<1, nullptr, stream>>>((half*)(out), (half*)(src));
        } else if constexpr (format == 1) {
            runTTRANSConv2<
                half, gShape0, gShape1, gShape2, gShape3, gShape4, gShape5, gWholeShape0, gWholeShape1, gWholeShape2,
                gWholeShape3, gWholeShape4><<<1, nullptr, stream>>>((half*)(out), (half*)(src));
        } else if constexpr (format == 2) {
            runTTRANSConv3<
                half, gShape0, gShape1, gShape2, gShape3, gShape4, gWholeShape0, gWholeShape1, gWholeShape2,
                gWholeShape3, gWholeShape4><<<1, nullptr, stream>>>((half*)(out), (half*)(src));
        }
    } else {
        if constexpr (format == 0) {
            runTTRANSConv1<
                T, gShape0, gShape1, gShape2, gShape3, gShape4, gWholeShape0, gWholeShape1, gWholeShape2, gWholeShape3,
                gWholeShape4><<<1, nullptr, stream>>>(out, src);
        } else if constexpr (format == 1) {
            runTTRANSConv2<
                T, gShape0, gShape1, gShape2, gShape3, gShape4, gShape5, gWholeShape0, gWholeShape1, gWholeShape2,
                gWholeShape3, gWholeShape4><<<1, nullptr, stream>>>(out, src);
        } else if constexpr (format == 2) {
            runTTRANSConv3<
                T, gShape0, gShape1, gShape2, gShape3, gShape4, gWholeShape0, gWholeShape1, gWholeShape2, gWholeShape3,
                gWholeShape4><<<1, nullptr, stream>>>(out, src);
        }
    }
}

// NCHW -> NC1HWC0
template void LaunchTTRANSConv<float, 0, 1, 4, 6, 56, 8, 1, 1, 1, 32, 6, 56>(float* out, float* src, void* stream);
template void LaunchTTRANSConv<int32_t, 0, 1, 1, 1, 8, 8, 1, 1, 1, 8, 1, 8>(int32_t* out, int32_t* src, void* stream);
template void LaunchTTRANSConv<float, 0, 5, 4, 4, 16, 16, 1, 1, 5, 57, 4, 16>(float* out, float* src, void* stream);
template void LaunchTTRANSConv<aclFloat16, 0, 1, 2, 2, 16, 16, 1, 1, 1, 30, 2, 16>(
    aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTTRANSConv<int16_t, 0, 7, 4, 6, 16, 16, 1, 1, 7, 53, 6, 16>(
    int16_t* out, int16_t* src, void* stream);
template void LaunchTTRANSConv<int8_t, 0, 3, 2, 2, 64, 32, 1, 1, 3, 64, 2, 64>(int8_t* out, int8_t* src, void* stream);
template void LaunchTTRANSConv<int8_t, 0, 1, 2, 2, 128, 32, 1, 1, 1, 63, 2, 128>(
    int8_t* out, int8_t* src, void* stream);
template void LaunchTTRANSConv<int8_t, 0, 5, 2, 2, 16, 32, 1, 1, 5, 58, 2, 16>(int8_t* out, int8_t* src, void* stream);
template void LaunchTTRANSConv<uint8_t, 0, 9, 3, 6, 16, 32, 1, 1, 9, 87, 6, 16>(
    uint8_t* out, uint8_t* src, void* stream);

template void LaunchTTRANSConv<float, 0, 1, 8, 6, 48, 4, 1, 1, 1, 32, 6, 48>(float* out, float* src, void* stream);
template void LaunchTTRANSConv<uint16_t, 0, 1, 7, 2, 16, 4, 1, 1, 1, 26, 2, 16>(
    uint16_t* out, uint16_t* src, void* stream);
template void LaunchTTRANSConv<int8_t, 0, 5, 5, 2, 16, 4, 1, 1, 5, 18, 2, 16>(int8_t* out, int8_t* src, void* stream);

// NC1HWC0 -> C1HWN1N0C0
template void LaunchTTRANSConv<float, 1, 2, 2, 16, 2, 2, 4, 3, 2, 2, 16, 4>(float* out, float* src, void* stream);
template void LaunchTTRANSConv<int32_t, 1, 2, 3, 10, 3, 16, 8, 37, 2, 3, 10, 8>(
    int32_t* out, int32_t* src, void* stream);
template void LaunchTTRANSConv<aclFloat16, 1, 2, 1, 8, 1, 16, 16, 7, 2, 1, 8, 16>(
    aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTTRANSConv<aclFloat16, 1, 2, 1, 8, 1, 16, 4, 7, 2, 1, 8, 4>(
    aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTTRANSConv<uint16_t, 1, 3, 2, 7, 3, 16, 16, 45, 3, 2, 7, 16>(
    uint16_t* out, uint16_t* src, void* stream);
template void LaunchTTRANSConv<int8_t, 1, 5, 1, 6, 2, 16, 32, 25, 5, 1, 6, 32>(int8_t* out, int8_t* src, void* stream);
template void LaunchTTRANSConv<uint8_t, 1, 2, 7, 7, 1, 16, 32, 11, 2, 7, 7, 32>(
    uint8_t* out, uint8_t* src, void* stream);

// NC1HWC0 -> NCHW
template void LaunchTTRANSConv<float, 2, 1, 1, 2, 4, 8, 1, 1, 1, 2, 4, 8>(float* out, float* src, void* stream);
template void LaunchTTRANSConv<float, 2, 2, 2, 2, 4, 8, 1, 2, 2, 2, 4, 8>(float* out, float* src, void* stream);
template void LaunchTTRANSConv<float, 2, 2, 2, 3, 4, 8, 1, 2, 2, 3, 4, 8>(float* out, float* src, void* stream);

// NCHW -> FRACTAL_Z
template <typename T, int srcN, int srcC, int srcH, int srcW, int dstC1HW, int dstN1, int dstN0, int dstC0>
void LaunchTTRANSConvNCHW2FZ(T* out, T* src, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTTRANSConv4<half, srcN, srcC, srcH, srcW, dstC1HW, dstN1, dstN0, dstC0>
            <<<1, nullptr, stream>>>((half*)(out), (half*)(src));
    } else {
        runTTRANSConv4<T, srcN, srcC, srcH, srcW, dstC1HW, dstN1, dstN0, dstC0><<<1, nullptr, stream>>>(out, src);
    }
}

template void LaunchTTRANSConvNCHW2FZ<aclFloat16, 7, 32, 2, 8, 32, 1, 16, 16>(
    aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTTRANSConvNCHW2FZ<aclFloat16, 4, 17, 4, 1, 32, 1, 16, 16>(
    aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTTRANSConvNCHW2FZ<float, 5, 4, 3, 8, 24, 1, 16, 8>(float* out, float* src, void* stream);
template void LaunchTTRANSConvNCHW2FZ<int8_t, 4, 32, 3, 7, 32, 1, 16, 32>(int8_t* out, int8_t* src, void* stream);

// GNCHW -> GNC1HWC0
template <
    typename T, int dstG, int dstN, int dstC1, int dstH, int dstW, int dstC0, int gWholeShape0, int gWholeShape1,
    int gWholeShape2, int gWholeShape3, int gWholeShape4, int gWholeShape5>
__global__ AICORE void runTTRANSGroupConv1(__gm__ T __out__* out, __gm__ T __in__* src)
{
    constexpr int bufferSize = dstG * dstN * dstC1 * dstH * dstW * dstC0 * sizeof(T);
    static_assert(gWholeShape0 == 1, "");
    static_assert(gWholeShape1 == dstG, "");
    static_assert(gWholeShape2 == dstN, "");
    static_assert(dstC1 == (gWholeShape3 + dstC0 - 1) / dstC0, "");
    static_assert(gWholeShape4 == dstH, "");
    static_assert(gWholeShape5 == dstW, "");
    static_assert(dstH * dstW * sizeof(T) % 32 == 0, "");

    constexpr int elemNum = dstG * dstN * dstC1 * dstH * dstW * dstC0;

    using ShapeDim5 = Shape<1, 1, 1, 1, elemNum>;
    using StrideDim5 = pto::Stride<elemNum, elemNum, elemNum, elemNum, 1>;
    using GlobalDataIn = GlobalTensor<T, ShapeDim5, StrideDim5>;

    using SrcTileData = Tile<TileType::Vec, T, 1, elemNum, BLayout::RowMajor, 1, elemNum>;
    SrcTileData src0Tile;
    TASSIGN(src0Tile, 0x0);
    using TileData =
        ConvTile<TileType::Vec, T, elemNum, Layout::GNCHW, ConvTileShape<dstG, dstN, dstC0 * dstC1, dstH, dstW>>;
    TileData srcTile;
    static_assert(srcTile.totalDimCount == 5);
    TASSIGN(srcTile, 0x0);
#ifdef __PTO_AUTO__
    TRESHAPE(src0Tile, srcTile);
#endif

    using DstTileData =
        ConvTile<TileType::Vec, T, elemNum, Layout::GNC1HWC0, ConvTileShape<dstG, dstN, dstC1, dstH, dstW, dstC0>>;
    DstTileData dstTile;
    static_assert(dstTile.totalDimCount == 6);
    TASSIGN(dstTile, 0x0 + dstG * dstN * dstC1 * dstH * dstW * dstC0 * sizeof(T));
    SrcTileData dst0Tile;
#ifdef __PTO_AUTO__
    TRESHAPE(dst0Tile, dstTile);
#endif
    TASSIGN(dst0Tile, 0x0 + dstG * dstN * dstC1 * dstH * dstW * dstC0 * sizeof(T));

    constexpr int tmpTileH = dstH * dstW;
    constexpr unsigned yTileSizeElem = (sizeof(T) == 1) ? 32 : 16;
    constexpr int tmpTileW = (dstC0 + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem;
    using TmpTileData = Tile<TileType::Vec, T, tmpTileH, tmpTileW, BLayout::RowMajor, tmpTileH, tmpTileW>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + dstG * dstN * dstC1 * dstH * dstW * dstC0 * sizeof(T) * 2);

    GlobalDataIn srcGlobal(src);
    GlobalDataIn dstGlobal(out);
    TLOAD(src0Tile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
#endif
    TTRANS(dstTile, srcTile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dst0Tile);
}

template <
    typename T, int dstG, int dstC1, int dstH, int dstW, int dstN1, int dstN0, int dstC0, int srcG, int srcN, int srcC1,
    int srcH, int srcW, int srcC0>
__global__ AICORE void runTTRANSGroupConv2(__gm__ T __out__* out, __gm__ T __in__* src)
{
    static_assert(srcG == dstG);
    static_assert(srcC0 == dstC0);
    static_assert(srcW == dstW);
    static_assert(srcH == dstH);
    static_assert(dstN1 == (srcN + dstN0 - 1) / dstN0);

    constexpr int bufferSize = dstG * dstN1 * dstN0 * dstC1 * dstH * dstW * dstC0 * sizeof(T);

    constexpr int elemNum = dstG * dstN1 * dstN0 * dstC1 * dstH * dstW * dstC0;

    using ShapeDim5 = Shape<1, 1, 1, 1, elemNum>;
    using StrideDim5 = pto::Stride<elemNum, elemNum, elemNum, elemNum, 1>;
    using GlobalDataIn = GlobalTensor<T, ShapeDim5, StrideDim5>;

    using SrcTileData = Tile<TileType::Vec, T, 1, elemNum, BLayout::RowMajor, 1, elemNum>;
    SrcTileData src0Tile;
    TASSIGN(src0Tile, 0x0);
    using TileData =
        // 源数据按 N 补齐到 N1*N0 的布局喂入（见 gen_data 的 _golden_gnc1hwc0_to_c1hwn1n0c0），
        // 因此 tile 声明的 N 维就是补齐后的长度：G 的 stride = N_padded * (C1*H*W*C0)。
        ConvTile<
            TileType::Vec, T, elemNum, Layout::GNC1HWC0, ConvTileShape<srcG, dstN1 * dstN0, dstC1, dstH, dstW, dstC0>>;
    TileData srcTile;
    static_assert(srcTile.totalDimCount == 6);
    TASSIGN(srcTile, 0x0);
#ifdef __PTO_AUTO__
    TRESHAPE(src0Tile, srcTile);
#endif
    using DstTileData = ConvTile<
        TileType::Vec, T, elemNum, Layout::FRACTAL_Z, ConvTileShape<dstG * dstC1 * dstH * dstW, dstN1, dstN0, dstC0>>;
    using TmpTileData = Tile<TileType::Vec, T, 16, 32, BLayout::RowMajor, 16, 32>;
    DstTileData dstTile;
    static_assert(dstTile.totalDimCount == 4);
    SrcTileData dst0Tile;
#ifdef __PTO_AUTO__
    TRESHAPE(dst0Tile, dstTile);
#endif
    TASSIGN(dstTile, 0x0 + bufferSize);
    TASSIGN(dst0Tile, 0x0 + bufferSize);
    using ZeroTileData =
        Tile<TileType::Vec, int32_t, 1, elemNum * sizeof(T) / 4, BLayout::RowMajor, 1, elemNum * sizeof(T) / 4>;
    ZeroTileData dst1Tile;
    TASSIGN(dst1Tile, 0x0 + bufferSize);

    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + bufferSize * 2);

    GlobalDataIn srcGlobal(src);
    GlobalDataIn dstGlobal(out);
    TLOAD(src0Tile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TSUB(dst1Tile, dst1Tile, dst1Tile);
    pipe_barrier(PIPE_ALL);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
#endif
    TTRANS(dstTile, srcTile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dst0Tile);
}

// GNC1HWC0 -> GNCHW
template <
    typename T, int dstG, int dstC1, int dstH, int dstW, int dstN1, int dstN0, int dstC0, int srcG, int srcN, int srcC1,
    int srcH, int srcW, int srcC0>
__global__ AICORE void runTTRANSGroupConv3(__gm__ T __out__* out, __gm__ T __in__* src)
{
    constexpr int elemNum = srcG * srcN * srcC1 * srcH * srcW * srcC0;
    constexpr int bufferSize = elemNum * sizeof(T);

    using ShapeDim5 = Shape<1, 1, 1, 1, elemNum>;
    using StrideDim5 = pto::Stride<elemNum, elemNum, elemNum, elemNum, 1>;
    using GlobalDataIn = GlobalTensor<T, ShapeDim5, StrideDim5>;

    using SrcTileData = Tile<TileType::Vec, T, 1, elemNum, BLayout::RowMajor, 1, elemNum>;
    SrcTileData src0Tile;
    TASSIGN(src0Tile, 0x0);
    using TileData =
        ConvTile<TileType::Vec, T, elemNum, Layout::GNC1HWC0, ConvTileShape<srcG, srcN, srcC1, srcH, srcW, srcC0>>;
    TileData srcTile;
    static_assert(srcTile.totalDimCount == 6);
    TASSIGN(srcTile, 0x0);
#ifdef __PTO_AUTO__
    TRESHAPE(src0Tile, srcTile);
#endif

    using DstTileData =
        ConvTile<TileType::Vec, T, elemNum, Layout::GNCHW, ConvTileShape<srcG, srcN, srcC1 * srcC0, srcH, srcW>>;
    DstTileData dstTile;
    static_assert(dstTile.totalDimCount == 5);
    TASSIGN(dstTile, 0x0 + elemNum * sizeof(T));
    SrcTileData dst0Tile;
#ifdef __PTO_AUTO__
    TRESHAPE(dst0Tile, dstTile);
#endif
    TASSIGN(dst0Tile, 0x0 + elemNum * sizeof(T));

    constexpr int tmpTileH = srcH * srcW;
    constexpr unsigned yTileSizeElem = (sizeof(T) == 1) ? 32 : 16;
    constexpr int tmpTileW = (srcC0 + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem;
    using TmpTileData = Tile<TileType::Vec, T, tmpTileH, tmpTileW, BLayout::RowMajor, tmpTileH, tmpTileW>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + elemNum * sizeof(T) * 2);

    GlobalDataIn srcGlobal(src);
    GlobalDataIn dstGlobal(out);
    TLOAD(src0Tile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
#endif
    TTRANS(dstTile, srcTile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dst0Tile);
}

template <
    typename T, int format, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gShape5, int gShape6,
    int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4, int gWholeShape5>
void LaunchTTRANSGroupConv(T* out, T* src, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        if constexpr (format == 0) {
            runTTRANSGroupConv1<
                half, gShape0, gShape1, gShape2, gShape3, gShape4, gShape5, gWholeShape0, gWholeShape1, gWholeShape2,
                gWholeShape3, gWholeShape4, gWholeShape5><<<1, nullptr, stream>>>((half*)(out), (half*)(src));
        } else if constexpr (format == 1) {
            runTTRANSGroupConv2<
                half, gShape0, gShape1, gShape2, gShape3, gShape4, gShape5, gShape6, gWholeShape0, gWholeShape1,
                gWholeShape2, gWholeShape3, gWholeShape4, gWholeShape5>
                <<<1, nullptr, stream>>>((half*)(out), (half*)(src));
        } else if constexpr (format == 2) {
            runTTRANSGroupConv3<
                half, gShape0, gShape1, gShape2, gShape3, gShape4, gShape5, gShape6, gWholeShape0, gWholeShape1,
                gWholeShape2, gWholeShape3, gWholeShape4, gWholeShape5>
                <<<1, nullptr, stream>>>((half*)(out), (half*)(src));
        }
    } else {
        if constexpr (format == 0) {
            runTTRANSGroupConv1<
                T, gShape0, gShape1, gShape2, gShape3, gShape4, gShape5, gWholeShape0, gWholeShape1, gWholeShape2,
                gWholeShape3, gWholeShape4, gWholeShape5><<<1, nullptr, stream>>>(out, src);
        } else if constexpr (format == 1) {
            runTTRANSGroupConv2<
                T, gShape0, gShape1, gShape2, gShape3, gShape4, gShape5, gShape6, gWholeShape0, gWholeShape1,
                gWholeShape2, gWholeShape3, gWholeShape4, gWholeShape5><<<1, nullptr, stream>>>(out, src);
        } else if constexpr (format == 2) {
            runTTRANSGroupConv3<
                T, gShape0, gShape1, gShape2, gShape3, gShape4, gShape5, gShape6, gWholeShape0, gWholeShape1,
                gWholeShape2, gWholeShape3, gWholeShape4, gWholeShape5><<<1, nullptr, stream>>>(out, src);
        }
    }
}

// GNCHW -> GNC1HWC0
template void LaunchTTRANSGroupConv<float, 0, 1, 1, 4, 6, 56, 8, 1, 1, 1, 1, 32, 6, 56>(
    float* out, float* src, void* stream);
template void LaunchTTRANSGroupConv<int32_t, 0, 4, 1, 1, 1, 8, 8, 1, 1, 4, 1, 8, 1, 8>(
    int32_t* out, int32_t* src, void* stream);
template void LaunchTTRANSGroupConv<float, 0, 2, 5, 2, 4, 16, 16, 1, 1, 2, 5, 30, 4, 16>(
    float* out, float* src, void* stream);
template void LaunchTTRANSGroupConv<aclFloat16, 0, 1, 1, 2, 2, 16, 16, 1, 1, 1, 1, 30, 2, 16>(
    aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTTRANSGroupConv<float, 0, 2, 1, 8, 6, 12, 4, 1, 1, 2, 1, 32, 6, 12>(
    float* out, float* src, void* stream);

// GNC1HWC0 -> GC1HWN1N0C0
template void LaunchTTRANSGroupConv<float, 1, 1, 2, 2, 16, 2, 2, 4, 1, 3, 2, 2, 16, 4>(
    float* out, float* src, void* stream);
template void LaunchTTRANSGroupConv<float, 1, 2, 2, 2, 16, 2, 2, 4, 2, 3, 2, 2, 16, 4>(
    float* out, float* src, void* stream);
template void LaunchTTRANSGroupConv<float, 1, 2, 2, 2, 16, 2, 2, 4, 2, 4, 2, 2, 16, 4>(
    float* out, float* src, void* stream);
template void LaunchTTRANSGroupConv<aclFloat16, 1, 1, 2, 1, 8, 1, 16, 16, 1, 7, 2, 1, 8, 16>(
    aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTTRANSGroupConv<aclFloat16, 1, 4, 2, 1, 8, 1, 16, 4, 4, 7, 2, 1, 8, 4>(
    aclFloat16* out, aclFloat16* src, void* stream);

// GNC1HWC0 -> GNCHW
template void LaunchTTRANSGroupConv<float, 2, 1, 1, 1, 2, 4, 8, 1, 1, 1, 1, 2, 4, 8>(
    float* out, float* src, void* stream);
template void LaunchTTRANSGroupConv<float, 2, 2, 2, 2, 2, 4, 8, 1, 2, 2, 2, 2, 4, 8>(
    float* out, float* src, void* stream);
template void LaunchTTRANSGroupConv<float, 2, 2, 2, 2, 3, 4, 8, 1, 2, 2, 2, 3, 4, 8>(
    float* out, float* src, void* stream);

/*=========================================================================================*/
/* Valid shape 用例：raw 只表达每维 stride 与容量，valid 决定处理域（与 CPU 侧用例同参数同语义）*/
/*   dst 由 host 预置哨兵并经 TLOAD 载入 UB，TTRANS 只允许改写 dst valid 盒，盒外必须原样写回。 */
/*=========================================================================================*/

template <typename SrcConvTile, typename DstConvTile, typename T>
AICORE void TTRANSConvValidBody(__gm__ T __out__* out, __gm__ T __in__* src, SrcConvTile& srcTile, DstConvTile& dstTile)
{
    constexpr int srcElemNum = SrcConvTile::bufferSize / sizeof(T);
    constexpr int dstElemNum = DstConvTile::bufferSize / sizeof(T);

    using SrcGlobalData =
        GlobalTensor<T, Shape<1, 1, 1, 1, srcElemNum>, pto::Stride<srcElemNum, srcElemNum, srcElemNum, srcElemNum, 1>>;
    using DstGlobalData =
        GlobalTensor<T, Shape<1, 1, 1, 1, dstElemNum>, pto::Stride<dstElemNum, dstElemNum, dstElemNum, dstElemNum, 1>>;

    using SrcTileData = Tile<TileType::Vec, T, 1, srcElemNum, BLayout::RowMajor, 1, srcElemNum>;
    using DstTileData = Tile<TileType::Vec, T, 1, dstElemNum, BLayout::RowMajor, 1, dstElemNum>;

    SrcTileData src0Tile;
    DstTileData dst0Tile;
    TASSIGN(src0Tile, 0x0);
    TASSIGN(dst0Tile, 0x0 + SrcConvTile::bufferSize);
#ifdef __PTO_AUTO__
    TRESHAPE(src0Tile, srcTile);
    TRESHAPE(dst0Tile, dstTile);
#endif

    // 快路径暂存区：转置收尾需要 cols * roundUp(C0, 16) 个元素（本套用例最大 1024）
    using TmpTileData = Tile<TileType::Vec, T, 1, 2048, BLayout::RowMajor, 1, 2048>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + SrcConvTile::bufferSize + DstConvTile::bufferSize);

    SrcGlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    TLOAD(src0Tile, srcGlobal);
    TLOAD(dst0Tile, dstGlobal); // 载入 host 预置的哨兵：valid 之外的字节必须原样保留
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
#endif
    TTRANS(dstTile, srcTile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dst0Tile);
}

// 场景 NCHW2NC1HWC0：NCHW -> NC1HWC0
__global__ AICORE void runTTRANSConvValid_NCHW2NC1HWC0(
    __gm__ float __out__* out, __gm__ float __in__* src, int64_t s0, int64_t s1, int64_t s2, int64_t s3, int64_t d0,
    int64_t d1, int64_t d2, int64_t d3, int64_t d4)
{
    using SrcValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using DstValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcConvTile =
        ConvTile<TileType::Vec, float, 1 * 16 * 8 * 16 * 4, Layout::NCHW, ConvTileShape<1, 16, 8, 16>, SrcValid>;
    using DstConvTile = ConvTile<
        TileType::Vec, float, 2 * 4 * 8 * 32 * 16 * 4, Layout::NC1HWC0, ConvTileShape<2, 4, 8, 32, 16>, DstValid>;

    SrcConvTile srcTile;
    DstConvTile dstTile;
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0 + SrcConvTile::bufferSize);

    srcTile.SetValidShape(s0, s1, s2, s3);
    dstTile.SetValidShape(d0, d1, d2, d3, d4);

    TTRANSConvValidBody<SrcConvTile, DstConvTile, float>(out, src, srcTile, dstTile);
}

// 场景 NC1HWC02NCHW：NC1HWC0 -> NCHW
__global__ AICORE void runTTRANSConvValid_NC1HWC02NCHW(
    __gm__ float __out__* out, __gm__ float __in__* src, int64_t s0, int64_t s1, int64_t s2, int64_t s3, int64_t s4,
    int64_t d0, int64_t d1, int64_t d2, int64_t d3)
{
    using SrcValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using DstValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcConvTile = ConvTile<
        TileType::Vec, float, 3 * 5 * 7 * 9 * 16 * 4, Layout::NC1HWC0, ConvTileShape<3, 5, 7, 9, 16>, SrcValid>;
    using DstConvTile =
        ConvTile<TileType::Vec, float, 3 * 32 * 7 * 9 * 4, Layout::NCHW, ConvTileShape<3, 32, 7, 9>, DstValid>;

    SrcConvTile srcTile;
    DstConvTile dstTile;
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0 + SrcConvTile::bufferSize);

    srcTile.SetValidShape(s0, s1, s2, s3, s4);
    dstTile.SetValidShape(d0, d1, d2, d3);

    TTRANSConvValidBody<SrcConvTile, DstConvTile, float>(out, src, srcTile, dstTile);
}

// 场景 NC1HWC02C1HWN1N0C0：NC1HWC0 -> FRACTAL_Z
__global__ AICORE void runTTRANSConvValid_NC1HWC02C1HWN1N0C0(
    __gm__ float __out__* out, __gm__ float __in__* src, int64_t s0, int64_t s1, int64_t s2, int64_t s3, int64_t s4,
    int64_t d0, int64_t d1, int64_t d2, int64_t d3)
{
    using SrcValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using DstValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcConvTile = ConvTile<
        TileType::Vec, float, 6 * 3 * 4 * 5 * 16 * 4, Layout::NC1HWC0, ConvTileShape<6, 3, 4, 5, 16>, SrcValid>;
    using DstConvTile =
        ConvTile<TileType::Vec, float, 16 * 2 * 24 * 16 * 4, Layout::FRACTAL_Z, ConvTileShape<16, 2, 24, 16>, DstValid>;

    SrcConvTile srcTile;
    DstConvTile dstTile;
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0 + SrcConvTile::bufferSize);

    srcTile.SetValidShape(s0, s1, s2, s3, s4);
    dstTile.SetValidShape(d0, d1, d2, d3);

    TTRANSConvValidBody<SrcConvTile, DstConvTile, float>(out, src, srcTile, dstTile);
}

// 场景 GNCHW2GNC1HWC0：GNCHW -> GNC1HWC0
__global__ AICORE void runTTRANSConvValid_GNCHW2GNC1HWC0(
    __gm__ float __out__* out, __gm__ float __in__* src, int64_t s0, int64_t s1, int64_t s2, int64_t s3, int64_t s4,
    int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5)
{
    using SrcValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using DstValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcConvTile =
        ConvTile<TileType::Vec, float, 3 * 4 * 10 * 5 * 6 * 4, Layout::GNCHW, ConvTileShape<3, 4, 10, 5, 6>, SrcValid>;
    using DstConvTile = ConvTile<
        TileType::Vec, float, 3 * 4 * 3 * 5 * 6 * 16 * 4, Layout::GNC1HWC0, ConvTileShape<3, 4, 3, 5, 6, 16>, DstValid>;

    SrcConvTile srcTile;
    DstConvTile dstTile;
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0 + SrcConvTile::bufferSize);

    srcTile.SetValidShape(s0, s1, s2, s3, s4);
    dstTile.SetValidShape(d0, d1, d2, d3, d4, d5);

    TTRANSConvValidBody<SrcConvTile, DstConvTile, float>(out, src, srcTile, dstTile);
}

// 场景 GNC1HWC02C1HWN1N0C0：GNC1HWC0 -> FRACTAL_Z
__global__ AICORE void runTTRANSConvValid_GNC1HWC02C1HWN1N0C0(
    __gm__ float __out__* out, __gm__ float __in__* src, int64_t s0, int64_t s1, int64_t s2, int64_t s3, int64_t s4,
    int64_t s5, int64_t d0, int64_t d1, int64_t d2, int64_t d3)
{
    using SrcValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using DstValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcConvTile = ConvTile<
        TileType::Vec, float, 3 * 4 * 3 * 4 * 5 * 16 * 4, Layout::GNC1HWC0, ConvTileShape<3, 4, 3, 4, 5, 16>, SrcValid>;
    using DstConvTile =
        ConvTile<TileType::Vec, float, 28 * 2 * 24 * 16 * 4, Layout::FRACTAL_Z, ConvTileShape<28, 2, 24, 16>, DstValid>;

    SrcConvTile srcTile;
    DstConvTile dstTile;
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0 + SrcConvTile::bufferSize);

    srcTile.SetValidShape(s0, s1, s2, s3, s4, s5);
    dstTile.SetValidShape(d0, d1, d2, d3);

    TTRANSConvValidBody<SrcConvTile, DstConvTile, float>(out, src, srcTile, dstTile);
}

// 场景 NCDHW2C1DHWN1N0C0：NCDHW -> FRACTAL_Z_3D
__global__ AICORE void runTTRANSConvValid_NCDHW2C1DHWN1N0C0(
    __gm__ float __out__* out, __gm__ float __in__* src, int64_t s0, int64_t s1, int64_t s2, int64_t s3, int64_t s4,
    int64_t d0, int64_t d1, int64_t d2, int64_t d3)
{
    using SrcValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using DstValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcConvTile =
        ConvTile<TileType::Vec, float, 3 * 10 * 3 * 4 * 5 * 4, Layout::NCDHW, ConvTileShape<3, 10, 3, 4, 5>, SrcValid>;
    using DstConvTile = ConvTile<
        TileType::Vec, float, 32 * 2 * 24 * 16 * 4, Layout::FRACTAL_Z_3D, ConvTileShape<32, 2, 24, 16>, DstValid>;

    SrcConvTile srcTile;
    DstConvTile dstTile;
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0 + SrcConvTile::bufferSize);

    srcTile.SetValidShape(s0, s1, s2, s3, s4);
    dstTile.SetValidShape(d0, d1, d2, d3);

    TTRANSConvValidBody<SrcConvTile, DstConvTile, float>(out, src, srcTile, dstTile);
}

// 场景 NCHW2C1HWN1N0C0：NCHW -> FRACTAL_Z
__global__ AICORE void runTTRANSConvValid_NCHW2C1HWN1N0C0(
    __gm__ float __out__* out, __gm__ float __in__* src, int64_t s0, int64_t s1, int64_t s2, int64_t s3, int64_t d0,
    int64_t d1, int64_t d2, int64_t d3)
{
    using SrcValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using DstValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcConvTile =
        ConvTile<TileType::Vec, float, 8 * 10 * 4 * 6 * 4, Layout::NCHW, ConvTileShape<8, 10, 4, 6>, SrcValid>;
    using DstConvTile =
        ConvTile<TileType::Vec, float, 16 * 2 * 24 * 16 * 4, Layout::FRACTAL_Z, ConvTileShape<16, 2, 24, 16>, DstValid>;

    SrcConvTile srcTile;
    DstConvTile dstTile;
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0 + SrcConvTile::bufferSize);

    srcTile.SetValidShape(s0, s1, s2, s3);
    dstTile.SetValidShape(d0, d1, d2, d3);

    TTRANSConvValidBody<SrcConvTile, DstConvTile, float>(out, src, srcTile, dstTile);
}

// 场景 NCHW2C1DHWN1N0C0：NCHW -> FRACTAL_Z_3D
__global__ AICORE void runTTRANSConvValid_NCHW2C1DHWN1N0C0(
    __gm__ float __out__* out, __gm__ float __in__* src, int64_t s0, int64_t s1, int64_t s2, int64_t s3, int64_t d0,
    int64_t d1, int64_t d2, int64_t d3)
{
    using SrcValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using DstValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcConvTile =
        ConvTile<TileType::Vec, float, 6 * 18 * 6 * 5 * 4, Layout::NCHW, ConvTileShape<6, 18, 6, 5>, SrcValid>;
    using DstConvTile = ConvTile<
        TileType::Vec, float, 32 * 2 * 24 * 16 * 4, Layout::FRACTAL_Z_3D, ConvTileShape<32, 2, 24, 16>, DstValid>;

    SrcConvTile srcTile;
    DstConvTile dstTile;
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0 + SrcConvTile::bufferSize);

    srcTile.SetValidShape(s0, s1, s2, s3);
    dstTile.SetValidShape(d0, d1, d2, d3);

    TTRANSConvValidBody<SrcConvTile, DstConvTile, float>(out, src, srcTile, dstTile);
}

/*--------------------------- valid 用例启动器 ---------------------------*/

void LaunchTTRANSConvValid_NCHW2NC1HWC0(
    float* out, float* src, void* stream, const int64_t (&srcValid)[4], const int64_t (&dstValid)[5])
{
    runTTRANSConvValid_NCHW2NC1HWC0<<<1, nullptr, stream>>>(
        out, src, srcValid[0], srcValid[1], srcValid[2], srcValid[3], dstValid[0], dstValid[1], dstValid[2],
        dstValid[3], dstValid[4]);
}

void LaunchTTRANSConvValid_NC1HWC02NCHW(
    float* out, float* src, void* stream, const int64_t (&srcValid)[5], const int64_t (&dstValid)[4])
{
    runTTRANSConvValid_NC1HWC02NCHW<<<1, nullptr, stream>>>(
        out, src, srcValid[0], srcValid[1], srcValid[2], srcValid[3], srcValid[4], dstValid[0], dstValid[1],
        dstValid[2], dstValid[3]);
}

void LaunchTTRANSConvValid_NC1HWC02C1HWN1N0C0(
    float* out, float* src, void* stream, const int64_t (&srcValid)[5], const int64_t (&dstValid)[4])
{
    runTTRANSConvValid_NC1HWC02C1HWN1N0C0<<<1, nullptr, stream>>>(
        out, src, srcValid[0], srcValid[1], srcValid[2], srcValid[3], srcValid[4], dstValid[0], dstValid[1],
        dstValid[2], dstValid[3]);
}

void LaunchTTRANSConvValid_GNCHW2GNC1HWC0(
    float* out, float* src, void* stream, const int64_t (&srcValid)[5], const int64_t (&dstValid)[6])
{
    runTTRANSConvValid_GNCHW2GNC1HWC0<<<1, nullptr, stream>>>(
        out, src, srcValid[0], srcValid[1], srcValid[2], srcValid[3], srcValid[4], dstValid[0], dstValid[1],
        dstValid[2], dstValid[3], dstValid[4], dstValid[5]);
}

void LaunchTTRANSConvValid_GNC1HWC02C1HWN1N0C0(
    float* out, float* src, void* stream, const int64_t (&srcValid)[6], const int64_t (&dstValid)[4])
{
    runTTRANSConvValid_GNC1HWC02C1HWN1N0C0<<<1, nullptr, stream>>>(
        out, src, srcValid[0], srcValid[1], srcValid[2], srcValid[3], srcValid[4], srcValid[5], dstValid[0],
        dstValid[1], dstValid[2], dstValid[3]);
}

void LaunchTTRANSConvValid_NCDHW2C1DHWN1N0C0(
    float* out, float* src, void* stream, const int64_t (&srcValid)[5], const int64_t (&dstValid)[4])
{
    runTTRANSConvValid_NCDHW2C1DHWN1N0C0<<<1, nullptr, stream>>>(
        out, src, srcValid[0], srcValid[1], srcValid[2], srcValid[3], srcValid[4], dstValid[0], dstValid[1],
        dstValid[2], dstValid[3]);
}

void LaunchTTRANSConvValid_NCHW2C1HWN1N0C0(
    float* out, float* src, void* stream, const int64_t (&srcValid)[4], const int64_t (&dstValid)[4])
{
    runTTRANSConvValid_NCHW2C1HWN1N0C0<<<1, nullptr, stream>>>(
        out, src, srcValid[0], srcValid[1], srcValid[2], srcValid[3], dstValid[0], dstValid[1], dstValid[2],
        dstValid[3]);
}

void LaunchTTRANSConvValid_NCHW2C1DHWN1N0C0(
    float* out, float* src, void* stream, const int64_t (&srcValid)[4], const int64_t (&dstValid)[4])
{
    runTTRANSConvValid_NCHW2C1DHWN1N0C0<<<1, nullptr, stream>>>(
        out, src, srcValid[0], srcValid[1], srcValid[2], srcValid[3], dstValid[0], dstValid[1], dstValid[2],
        dstValid[3]);
}

/*=========================================================================================*/
/* 性能基准：单个 kernel 内重复 iters 次 TTRANS（UB 常驻 tile），host 侧计时。                */
/*   收缩 valid 与 legacy（valid==raw）各一组，用于对比路径代价与合并前后的差异。             */
/*=========================================================================================*/

template <typename SrcConvTile, typename DstConvTile, typename T, int tmpElemNum>
AICORE void TTRANSConvBenchBody(
    __gm__ T __out__* out, __gm__ T __in__* src, SrcConvTile& srcTile, DstConvTile& dstTile, int64_t iters)
{
    constexpr int srcElemNum = SrcConvTile::bufferSize / sizeof(T);
    constexpr int dstElemNum = DstConvTile::bufferSize / sizeof(T);

    using SrcGlobalData =
        GlobalTensor<T, Shape<1, 1, 1, 1, srcElemNum>, pto::Stride<srcElemNum, srcElemNum, srcElemNum, srcElemNum, 1>>;
    using DstGlobalData =
        GlobalTensor<T, Shape<1, 1, 1, 1, dstElemNum>, pto::Stride<dstElemNum, dstElemNum, dstElemNum, dstElemNum, 1>>;

    using SrcTileData = Tile<TileType::Vec, T, 1, srcElemNum, BLayout::RowMajor, 1, srcElemNum>;
    using DstTileData = Tile<TileType::Vec, T, 1, dstElemNum, BLayout::RowMajor, 1, dstElemNum>;

    SrcTileData src0Tile;
    DstTileData dst0Tile;
    TASSIGN(src0Tile, 0x0);
    TASSIGN(dst0Tile, 0x0 + SrcConvTile::bufferSize);
#ifdef __PTO_AUTO__
    TRESHAPE(src0Tile, srcTile);
    TRESHAPE(dst0Tile, dstTile);
#endif

    using TmpTileData = Tile<TileType::Vec, T, 1, tmpElemNum, BLayout::RowMajor, 1, tmpElemNum>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + SrcConvTile::bufferSize + DstConvTile::bufferSize);

    SrcGlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    TLOAD(src0Tile, srcGlobal);
    TLOAD(dst0Tile, dstGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
#endif
    for (int64_t i = 0; i < iters; ++i) {
        TTRANS(dstTile, srcTile, tmpTile);
    }
#ifndef __PTO_AUTO__
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dst0Tile);
}

// B1: NCHW -> NC1HWC0，H 收缩（W 全量 -> 连续 X 快路径）
__global__ AICORE void runTTRANSConvBench_NCHW2NC1HWC0(
    __gm__ float __out__* out, __gm__ float __in__* src, int64_t iters)
{
    using SrcValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using DstValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcConvTile =
        ConvTile<TileType::Vec, float, 8 * 16 * 8 * 16 * 4, Layout::NCHW, ConvTileShape<8, 16, 8, 16>, SrcValid>;
    using DstConvTile = ConvTile<
        TileType::Vec, float, 8 * 2 * 8 * 16 * 8 * 4, Layout::NC1HWC0, ConvTileShape<8, 2, 8, 16, 8>, DstValid>;

    SrcConvTile srcTile;
    DstConvTile dstTile;
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0 + SrcConvTile::bufferSize);
    srcTile.SetValidShape((int64_t)8, (int64_t)16, (int64_t)6, (int64_t)16);
    dstTile.SetValidShape((int64_t)8, (int64_t)2, (int64_t)6, (int64_t)16, (int64_t)8);
    TTRANSConvBenchBody<SrcConvTile, DstConvTile, float, 2048>(out, src, srcTile, dstTile, iters);
}

// B2: NC1HWC0 -> NCHW，H 收缩
__global__ AICORE void runTTRANSConvBench_NC1HWC02NCHW(
    __gm__ float __out__* out, __gm__ float __in__* src, int64_t iters)
{
    using SrcValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using DstValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcConvTile = ConvTile<
        TileType::Vec, float, 8 * 2 * 8 * 16 * 8 * 4, Layout::NC1HWC0, ConvTileShape<8, 2, 8, 16, 8>, SrcValid>;
    using DstConvTile =
        ConvTile<TileType::Vec, float, 8 * 16 * 8 * 16 * 4, Layout::NCHW, ConvTileShape<8, 16, 8, 16>, DstValid>;

    SrcConvTile srcTile;
    DstConvTile dstTile;
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0 + SrcConvTile::bufferSize);
    srcTile.SetValidShape((int64_t)8, (int64_t)2, (int64_t)6, (int64_t)16, (int64_t)8);
    dstTile.SetValidShape((int64_t)8, (int64_t)16, (int64_t)6, (int64_t)16);
    TTRANSConvBenchBody<SrcConvTile, DstConvTile, float, 2048>(out, src, srcTile, dstTile, iters);
}

// B3: NC1HWC0 -> FRACTAL_Z，N/C1/H/W 全部收缩（含 N0 尾巴补零）
__global__ AICORE void runTTRANSConvBench_NC1HWC02FZ(__gm__ float __out__* out, __gm__ float __in__* src, int64_t iters)
{
    using SrcValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using DstValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcConvTile = ConvTile<
        TileType::Vec, float, 16 * 2 * 8 * 10 * 8 * 4, Layout::NC1HWC0, ConvTileShape<16, 2, 8, 10, 8>, SrcValid>;
    using DstConvTile =
        ConvTile<TileType::Vec, float, 160 * 1 * 16 * 8 * 4, Layout::FRACTAL_Z, ConvTileShape<160, 1, 16, 8>, DstValid>;

    SrcConvTile srcTile;
    DstConvTile dstTile;
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0 + SrcConvTile::bufferSize);
    srcTile.SetValidShape((int64_t)13, (int64_t)2, (int64_t)6, (int64_t)9, (int64_t)8);
    dstTile.SetValidShape((int64_t)108, (int64_t)1, (int64_t)16, (int64_t)8);
    TTRANSConvBenchBody<SrcConvTile, DstConvTile, float, 2048>(out, src, srcTile, dstTile, iters);
}

// B4: NCHW -> FRACTAL_Z，N/H 收缩（含 N0 尾巴补零）
__global__ AICORE void runTTRANSConvBench_NCHW2FZ(__gm__ float __out__* out, __gm__ float __in__* src, int64_t iters)
{
    using SrcValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using DstValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcConvTile =
        ConvTile<TileType::Vec, float, 16 * 8 * 8 * 16 * 4, Layout::NCHW, ConvTileShape<16, 8, 8, 16>, SrcValid>;
    using DstConvTile =
        ConvTile<TileType::Vec, float, 96 * 1 * 16 * 8 * 4, Layout::FRACTAL_Z, ConvTileShape<96, 1, 16, 8>, DstValid>;

    SrcConvTile srcTile;
    DstConvTile dstTile;
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0 + SrcConvTile::bufferSize);
    srcTile.SetValidShape((int64_t)13, (int64_t)8, (int64_t)6, (int64_t)16);
    dstTile.SetValidShape((int64_t)96, (int64_t)1, (int64_t)16, (int64_t)8);
    TTRANSConvBenchBody<SrcConvTile, DstConvTile, float, 2048>(out, src, srcTile, dstTile, iters);
}

// B5: NCHW -> NC1HWC0，valid == raw（legacy 几何；用来对比合并前后 legacy 路径的性能）
__global__ AICORE void runTTRANSConvBenchLegacy_NCHW2NC1HWC0(
    __gm__ float __out__* out, __gm__ float __in__* src, int64_t iters)
{
    using SrcValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using DstValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcConvTile =
        ConvTile<TileType::Vec, float, 8 * 16 * 8 * 16 * 4, Layout::NCHW, ConvTileShape<8, 16, 8, 16>, SrcValid>;
    using DstConvTile = ConvTile<
        TileType::Vec, float, 8 * 2 * 8 * 16 * 8 * 4, Layout::NC1HWC0, ConvTileShape<8, 2, 8, 16, 8>, DstValid>;

    SrcConvTile srcTile;
    DstConvTile dstTile;
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0 + SrcConvTile::bufferSize);
    srcTile.SetValidShape((int64_t)8, (int64_t)16, (int64_t)8, (int64_t)16);
    dstTile.SetValidShape((int64_t)8, (int64_t)2, (int64_t)8, (int64_t)16, (int64_t)8);
    TTRANSConvBenchBody<SrcConvTile, DstConvTile, float, 2048>(out, src, srcTile, dstTile, iters);
}

void LaunchTTRANSConvBench_NCHW2NC1HWC0(float* out, float* src, void* stream, int64_t iters)
{
    runTTRANSConvBench_NCHW2NC1HWC0<<<1, nullptr, stream>>>(out, src, iters);
}

void LaunchTTRANSConvBenchLegacy_NCHW2NC1HWC0(float* out, float* src, void* stream, int64_t iters)
{
    runTTRANSConvBenchLegacy_NCHW2NC1HWC0<<<1, nullptr, stream>>>(out, src, iters);
}

void LaunchTTRANSConvBench_NC1HWC02NCHW(float* out, float* src, void* stream, int64_t iters)
{
    runTTRANSConvBench_NC1HWC02NCHW<<<1, nullptr, stream>>>(out, src, iters);
}

void LaunchTTRANSConvBench_NC1HWC02FZ(float* out, float* src, void* stream, int64_t iters)
{
    runTTRANSConvBench_NC1HWC02FZ<<<1, nullptr, stream>>>(out, src, iters);
}

void LaunchTTRANSConvBench_NCHW2FZ(float* out, float* src, void* stream, int64_t iters)
{
    runTTRANSConvBench_NCHW2FZ<<<1, nullptr, stream>>>(out, src, iters);
}
