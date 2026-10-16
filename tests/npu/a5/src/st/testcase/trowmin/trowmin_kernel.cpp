/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <acl/acl.h>
#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

using namespace std;
using namespace pto;

template <typename T, int row, int validRow, int srcCol, int srcValidCol, int dstCol, int dstTileCol = 16>
PTO_INTERNAL void runTRowMin(__gm__ T* out, __gm__ T* src)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2StrideSrc = pto::Stride<1, 1, -1, -1, 1>;
    using DynDim2StrideDst = pto::Stride<1, 1, 1, -1, -1>;

    using GlobalDataSrc = GlobalTensor<T, DynDim2Shape, DynDim2StrideSrc>;
    using GlobalDataDst = GlobalTensor<T, DynDim2Shape, DynDim2StrideDst>;
    GlobalDataSrc srcGlobal(src, DynDim2Shape(validRow, srcValidCol), DynDim2StrideSrc(row, srcCol));
    GlobalDataDst dstGlobal(out, DynDim2Shape(validRow, dstCol), DynDim2StrideDst(dstCol, row));
    using srcTileData = Tile<TileType::Vec, T, row, srcCol, BLayout::RowMajor, -1, -1>;
    using dstTileData = Tile<TileType::Vec, T, row, dstTileCol, BLayout::RowMajor, -1, -1>;
    srcTileData srcTile(validRow, srcValidCol);
    srcTileData tmpTile(validRow, srcValidCol);
    dstTileData dstTile(validRow, dstCol);
    TASSIGN<0x0>(srcTile);
    TASSIGN<row * srcCol * sizeof(T)>(tmpTile);
    TASSIGN<2 * row * srcCol * sizeof(T)>(dstTile);

    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TROWMIN(dstTile, srcTile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
}

template <typename T, int row, int validRow, int srcCol, int srcValidCol, int dstCol, int globalSrcCol = srcCol>
PTO_INTERNAL void runTRowMinDNDst(__gm__ T* out, __gm__ T* src)
{
    using ValidSrcShape = TileShape2D<T, validRow, srcValidCol>;
    using NDSrcShape = BaseShape2D<T, row, globalSrcCol>;
    using GlobalDataSrc = GlobalTensor<T, ValidSrcShape, NDSrcShape>;
    GlobalDataSrc srcGlobal(src);

    using ValidDstShape = TileShape2D<T, dstCol, validRow>;
    using NDDstShape = BaseShape2D<T, row, dstCol>;
    using GlobalDataDst = GlobalTensor<T, ValidDstShape, NDDstShape>;
    GlobalDataDst dstGlobal(out);

    using srcTileData = Tile<TileType::Vec, T, row, srcCol, BLayout::RowMajor, validRow, srcValidCol>;
    using dstTileDataDN = Tile<TileType::Vec, T, row, 1, BLayout::ColMajor, validRow, 1>;
    srcTileData srcTile;
    srcTileData tmpTile;
    dstTileDataDN dstTile;
    TASSIGN<0x0>(srcTile);
    TASSIGN<row * srcCol * sizeof(T)>(tmpTile);
    TASSIGN<2 * row * srcCol * sizeof(T)>(dstTile);

    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TROWMIN(dstTile, srcTile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    using dstTileDataND = Tile<TileType::Vec, T, 1, row, BLayout::RowMajor, 1, validRow>;
    dstTileDataND dstTileND;
    TRESHAPE(dstTileND, dstTile);
    TSTORE(dstGlobal, dstTileND);
}

extern "C" __global__ AICORE void launchTROWMINCase1(__gm__ float* out, __gm__ float* src)
{
    runTRowMin<float, 127, 127, 64, 63, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase2(__gm__ float* out, __gm__ float* src)
{
    runTRowMin<float, 63, 63, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase3(__gm__ float* out, __gm__ float* src)
{
    runTRowMin<float, 31, 31, 128, 127, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase4(__gm__ float* out, __gm__ float* src)
{
    runTRowMin<float, 15, 15, 192, 192, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase5(__gm__ float* out, __gm__ float* src)
{
    runTRowMin<float, 7, 7, 448, 447, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase6(__gm__ half* out, __gm__ half* src)
{
    runTRowMin<half, 256, 256, 16, 15, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase7(__gm__ float* out, __gm__ float* src)
{
    runTRowMin<float, 30, 30, 216, 216, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase8(__gm__ float* out, __gm__ float* src)
{
    runTRowMin<float, 30, 30, 216, 24, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase9(__gm__ float* out, __gm__ float* src)
{
    runTRowMin<float, 30, 11, 216, 216, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase10(__gm__ float* out, __gm__ float* src)
{
    runTRowMin<float, 30, 11, 216, 24, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase11(__gm__ float* out, __gm__ float* src)
{
    runTRowMin<float, 238, 238, 40, 40, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase12(__gm__ float* out, __gm__ float* src)
{
    runTRowMin<float, 238, 238, 40, 16, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase13(__gm__ float* out, __gm__ float* src)
{
    runTRowMin<float, 238, 121, 40, 40, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase14(__gm__ float* out, __gm__ float* src)
{
    runTRowMin<float, 238, 121, 40, 16, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase15(__gm__ float* out, __gm__ float* src)
{
    runTRowMinDNDst<float, 64, 64, 128, 128, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase16(__gm__ float* out, __gm__ float* src)
{
    runTRowMinDNDst<float, 32, 32, 256, 256, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase17(__gm__ float* out, __gm__ float* src)
{
    runTRowMinDNDst<float, 16, 16, 512, 512, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase18(__gm__ float* out, __gm__ float* src)
{
    runTRowMinDNDst<float, 8, 8, 1024, 1024, 1>(out, src);
}

extern "C" __global__ AICORE void launchTROWMINCase19(__gm__ int32_t* out, __gm__ int32_t* src)
{
    runTRowMin<int32_t, 127, 127, 64, 63, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase20(__gm__ int32_t* out, __gm__ int32_t* src)
{
    runTRowMin<int32_t, 63, 63, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase21(__gm__ int32_t* out, __gm__ int32_t* src)
{
    runTRowMin<int32_t, 31, 31, 128, 127, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase22(__gm__ int32_t* out, __gm__ int32_t* src)
{
    runTRowMin<int32_t, 15, 15, 192, 192, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase23(__gm__ int32_t* out, __gm__ int32_t* src)
{
    runTRowMin<int32_t, 7, 7, 448, 447, 1>(out, src);
}

// Keep int16_t row strides 32-byte aligned.
extern "C" __global__ AICORE void launchTROWMINCase24(__gm__ int16_t* out, __gm__ int16_t* src)
{
    runTRowMin<int16_t, 128, 128, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase25(__gm__ int16_t* out, __gm__ int16_t* src)
{
    runTRowMin<int16_t, 64, 64, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase26(__gm__ int16_t* out, __gm__ int16_t* src)
{
    runTRowMin<int16_t, 32, 32, 128, 128, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase27(__gm__ int16_t* out, __gm__ int16_t* src)
{
    runTRowMin<int16_t, 16, 16, 192, 192, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase28(__gm__ int16_t* out, __gm__ int16_t* src)
{
    runTRowMin<int16_t, 8, 8, 448, 448, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase29(__gm__ int64_t* out, __gm__ int64_t* src)
{
    runTRowMin<int64_t, 4, 4, 16, 15, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase30(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTRowMin<uint64_t, 4, 4, 16, 15, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase31(__gm__ int64_t* out, __gm__ int64_t* src)
{
    runTRowMin<int64_t, 4, 4, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase32(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTRowMin<uint64_t, 4, 4, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase33(__gm__ int64_t* out, __gm__ int64_t* src)
{
    runTRowMin<int64_t, 32, 32, 32, 32, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase34(__gm__ int64_t* out, __gm__ int64_t* src)
{
    runTRowMin<int64_t, 1, 1, 10912, 10912, 1>(out, src);
}

extern "C" __global__ AICORE void launchTROWMINCase35(__gm__ int64_t* out, __gm__ int64_t* src)
{
    runTRowMinDNDst<int64_t, 32, 32, 32, 32, 1>(out, src);
}

extern "C" __global__ AICORE void launchTROWMINCase36(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTRowMinDNDst<uint64_t, 32, 32, 32, 32, 1>(out, src);
}

extern "C" __global__ AICORE void launchTROWMINCase37(__gm__ int64_t* out, __gm__ int64_t* src)
{
    runTRowMinDNDst<int64_t, 32, 32, 148, 145, 1, 145>(out, src);
}

extern "C" __global__ AICORE void launchTROWMINCase38(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTRowMinDNDst<uint64_t, 32, 32, 148, 145, 1, 145>(out, src);
}

extern "C" __global__ AICORE void launchTROWMINCase39(__gm__ int64_t* out, __gm__ int64_t* src)
{
    runTRowMin<int64_t, 64, 64, 16, 16, 1, 4>(out, src);
}

extern "C" __global__ AICORE void launchTROWMINCase40(__gm__ int64_t* out, __gm__ int64_t* src)
{
    runTRowMinDNDst<int64_t, 64, 64, 16, 16, 1>(out, src);
}

extern "C" __global__ AICORE void launchTROWMINCase41(__gm__ int64_t* out, __gm__ int64_t* src)
{
    runTRowMin<int64_t, 64, 64, 16, 16, 1, 4>(out, src);
}

extern "C" __global__ AICORE void launchTROWMINCase42(__gm__ int64_t* out, __gm__ int64_t* src)
{
    runTRowMinDNDst<int64_t, 64, 64, 16, 16, 1>(out, src);
}

extern "C" __global__ AICORE void launchTROWMINCase43(__gm__ int64_t* out, __gm__ int64_t* src)
{
    using SrcTile = Tile<TileType::Vec, int64_t, 64, 16, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, int64_t, 64, 4, BLayout::RowMajor, -1, -1>;
    using GuardTile = Tile<TileType::Vec, int64_t, 1, 64, BLayout::RowMajor, 1, 64>;
    using SrcGlobal = GlobalTensor<int64_t, Shape<1, 1, 1, 64, 16>, pto::Stride<1024, 1024, 1024, 16, 1>>;
    using DstGlobal = GlobalTensor<int64_t, Shape<1, 1, 1, 64, 1>, pto::Stride<64, 64, 64, 1, 1>>;
    using GuardGlobal = GlobalTensor<int64_t, Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>>;
    SrcTile srcTile(64, 16);
    SrcTile tmpTile(64, 16);
    DstTile dstTile(64, 1);
    GuardTile guardTile;
    SrcGlobal srcGlobal(src);
    DstGlobal dstGlobal(out);
    GuardGlobal guardGlobal(out + 64);
    TASSIGN<0>(srcTile);
    TASSIGN<8192>(tmpTile);
    TASSIGN<16384>(dstTile);
    // Keep the sentinel immediately after the destination's physical 64 x 4 elements.
    TASSIGN<16384 + 64 * 4 * sizeof(int64_t)>(guardTile);
    TLOAD(srcTile, srcGlobal);
    TLOAD(guardTile, guardGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TROWMIN(dstTile, srcTile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    TSTORE(guardGlobal, guardTile);
}

template <int validRows>
PTO_INTERNAL void runTRowMinDNDstTail(__gm__ int64_t* out, __gm__ int64_t* src)
{
    constexpr int outputElements = 8 + 64;
    using SrcTile = Tile<TileType::Vec, int64_t, 8, 16, BLayout::RowMajor, validRows, 16>;
    using DstTile = Tile<TileType::Vec, int64_t, 8, 1, BLayout::ColMajor, validRows, 1>;
    using OutputTile = Tile<TileType::Vec, int64_t, 1, outputElements, BLayout::RowMajor, 1, outputElements>;
    using SrcGlobal = GlobalTensor<int64_t, Shape<1, 1, 1, validRows, 16>, pto::Stride<128, 128, 128, 16, 1>>;
    using OutputGlobal = GlobalTensor<
        int64_t, Shape<1, 1, 1, 1, outputElements>,
        pto::Stride<outputElements, outputElements, outputElements, outputElements, 1>>;
    SrcTile srcTile;
    SrcTile tmpTile;
    DstTile dstTile;
    OutputTile outputTile;
    SrcGlobal srcGlobal(src);
    OutputGlobal outputGlobal(out);
    TASSIGN<0>(srcTile);
    TASSIGN<1024>(tmpTile);
    TASSIGN<2048>(dstTile);
    // This view includes inactive destination rows and the adjacent guard.
    TASSIGN<2048>(outputTile);
    TLOAD(srcTile, srcGlobal);
    TLOAD(outputTile, outputGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TROWMIN(dstTile, srcTile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(outputGlobal, outputTile);
}

extern "C" __global__ AICORE void launchTROWMINCase44(__gm__ int64_t* out, __gm__ int64_t* src)
{
    runTRowMinDNDstTail<5>(out, src);
}

extern "C" __global__ AICORE void launchTROWMINCase45(__gm__ int64_t* out, __gm__ int64_t* src)
{
    runTRowMinDNDstTail<6>(out, src);
}

extern "C" __global__ AICORE void launchTROWMINCase46(__gm__ int64_t* out, __gm__ int64_t* src)
{
    runTRowMinDNDstTail<7>(out, src);
}

extern "C" __global__ AICORE void launchTROWMINCase47(__gm__ uint32_t* out, __gm__ uint32_t* src)
{
    runTRowMin<uint32_t, 127, 127, 64, 63, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase48(__gm__ uint32_t* out, __gm__ uint32_t* src)
{
    runTRowMin<uint32_t, 63, 63, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase49(__gm__ uint32_t* out, __gm__ uint32_t* src)
{
    runTRowMin<uint32_t, 31, 31, 128, 127, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase50(__gm__ uint32_t* out, __gm__ uint32_t* src)
{
    runTRowMin<uint32_t, 15, 15, 192, 192, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase51(__gm__ uint32_t* out, __gm__ uint32_t* src)
{
    runTRowMin<uint32_t, 7, 7, 448, 447, 1>(out, src);
}

// Keep uint16_t row strides 32-byte aligned.
extern "C" __global__ AICORE void launchTROWMINCase52(__gm__ uint16_t* out, __gm__ uint16_t* src)
{
    runTRowMin<uint16_t, 128, 128, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase53(__gm__ uint16_t* out, __gm__ uint16_t* src)
{
    runTRowMin<uint16_t, 64, 64, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase54(__gm__ uint16_t* out, __gm__ uint16_t* src)
{
    runTRowMin<uint16_t, 32, 32, 128, 128, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase55(__gm__ uint16_t* out, __gm__ uint16_t* src)
{
    runTRowMin<uint16_t, 16, 16, 192, 192, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase56(__gm__ uint16_t* out, __gm__ uint16_t* src)
{
    runTRowMin<uint16_t, 8, 8, 448, 448, 1>(out, src);
}

extern "C" __global__ AICORE void launchTROWMINCase57(__gm__ float* out, __gm__ float* src)
{
    runTRowMin<float, 8, 8, 8, 1, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase58(__gm__ float* out, __gm__ float* src)
{
    runTRowMin<float, 1, 1, 8192, 8192, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase59(__gm__ float* out, __gm__ float* src)
{
    runTRowMin<float, 256, 256, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase60(__gm__ float* out, __gm__ float* src)
{
    runTRowMin<float, 8, 8, 2048, 2048, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase61(__gm__ float* out, __gm__ float* src)
{
    runTRowMin<float, 64, 64, 64, 1, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase62(__gm__ float* out, __gm__ float* src)
{
    runTRowMin<float, 63, 63, 72, 65, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase63(__gm__ float* out, __gm__ float* src)
{
    runTRowMin<float, 33, 33, 136, 129, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase64(__gm__ float* out, __gm__ float* src)
{
    runTRowMin<float, 238, 237, 80, 79, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase65(__gm__ half* out, __gm__ half* src)
{
    runTRowMin<half, 64, 64, 128, 128, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase66(__gm__ half* out, __gm__ half* src)
{
    runTRowMin<half, 32, 32, 256, 255, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase67(__gm__ half* out, __gm__ half* src)
{
    runTRowMin<half, 16, 16, 512, 512, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase68(__gm__ half* out, __gm__ half* src)
{
    runTRowMin<half, 8, 8, 1024, 1024, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase69(__gm__ half* out, __gm__ half* src)
{
    runTRowMin<half, 30, 30, 224, 24, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase70(__gm__ int8_t* out, __gm__ int8_t* src)
{
    runTRowMin<int8_t, 128, 128, 256, 256, 1, 32>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase71(__gm__ int8_t* out, __gm__ int8_t* src)
{
    runTRowMin<int8_t, 64, 64, 256, 255, 1, 32>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase72(__gm__ int8_t* out, __gm__ int8_t* src)
{
    runTRowMin<int8_t, 32, 32, 256, 256, 1, 32>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase73(__gm__ int8_t* out, __gm__ int8_t* src)
{
    runTRowMin<int8_t, 16, 16, 256, 255, 1, 32>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase74(__gm__ int8_t* out, __gm__ int8_t* src)
{
    runTRowMin<int8_t, 8, 8, 512, 512, 1, 32>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase75(__gm__ uint8_t* out, __gm__ uint8_t* src)
{
    runTRowMin<uint8_t, 128, 128, 256, 256, 1, 32>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase76(__gm__ uint8_t* out, __gm__ uint8_t* src)
{
    runTRowMin<uint8_t, 8, 8, 512, 511, 1, 32>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase77(__gm__ uint8_t* out, __gm__ uint8_t* src)
{
    runTRowMin<uint8_t, 2, 2, 2048, 2048, 1, 32>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase78(__gm__ uint32_t* out, __gm__ uint32_t* src)
{
    runTRowMin<uint32_t, 31, 31, 128, 127, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase79(__gm__ uint16_t* out, __gm__ uint16_t* src)
{
    runTRowMin<uint16_t, 32, 32, 128, 127, 1>(out, src);
}

extern "C" __global__ AICORE void launchTROWMINCase80(__gm__ float* out, __gm__ float* src)
{
    runTRowMin<float, 106, 105, 192, 191, 1, 16>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase81(__gm__ float* out, __gm__ float* src)
{
    runTRowMin<float, 1, 1, 16384, 16384, 1, 16>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase82(__gm__ float* out, __gm__ float* src)
{
    runTRowMin<float, 30, 11, 216, 216, 1, 16>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase83(__gm__ float* out, __gm__ float* src)
{
    runTRowMin<float, 64, 1, 64, 64, 1, 16>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase84(__gm__ float* out, __gm__ float* src)
{
    runTRowMin<float, 251, 250, 96, 95, 1, 16>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase85(__gm__ half* out, __gm__ half* src)
{
    runTRowMin<half, 128, 127, 320, 319, 1, 16>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase86(__gm__ int16_t* out, __gm__ int16_t* src)
{
    runTRowMin<int16_t, 200, 199, 208, 207, 1, 16>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase87(__gm__ int8_t* out, __gm__ int8_t* src)
{
    runTRowMin<int8_t, 88, 87, 896, 887, 1, 32>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase88(__gm__ uint8_t* out, __gm__ uint8_t* src)
{
    runTRowMin<uint8_t, 88, 87, 896, 887, 1, 32>(out, src);
}

template <uint32_t caseId>
void launchTROWMINTestCase(void* out, void* src, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTROWMINCase1<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 2: {
            launchTROWMINCase2<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 3: {
            launchTROWMINCase3<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 4: {
            launchTROWMINCase4<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 5: {
            launchTROWMINCase5<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 6: {
            launchTROWMINCase6<<<1, nullptr, stream>>>((half*)out, (half*)src);
            break;
        }
        case 7: {
            launchTROWMINCase7<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 8: {
            launchTROWMINCase8<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 9: {
            launchTROWMINCase9<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 10: {
            launchTROWMINCase10<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 11: {
            launchTROWMINCase11<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 12: {
            launchTROWMINCase12<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 13: {
            launchTROWMINCase13<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 14: {
            launchTROWMINCase14<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 15: {
            launchTROWMINCase15<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 16: {
            launchTROWMINCase16<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 17: {
            launchTROWMINCase17<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 18: {
            launchTROWMINCase18<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 19: {
            launchTROWMINCase19<<<1, nullptr, stream>>>((int32_t*)out, (int32_t*)src);
            break;
        }
        case 20: {
            launchTROWMINCase20<<<1, nullptr, stream>>>((int32_t*)out, (int32_t*)src);
            break;
        }
        case 21: {
            launchTROWMINCase21<<<1, nullptr, stream>>>((int32_t*)out, (int32_t*)src);
            break;
        }
        case 22: {
            launchTROWMINCase22<<<1, nullptr, stream>>>((int32_t*)out, (int32_t*)src);
            break;
        }
        case 23: {
            launchTROWMINCase23<<<1, nullptr, stream>>>((int32_t*)out, (int32_t*)src);
            break;
        }
        case 24: {
            launchTROWMINCase24<<<1, nullptr, stream>>>((int16_t*)out, (int16_t*)src);
            break;
        }
        case 25: {
            launchTROWMINCase25<<<1, nullptr, stream>>>((int16_t*)out, (int16_t*)src);
            break;
        }
        case 26: {
            launchTROWMINCase26<<<1, nullptr, stream>>>((int16_t*)out, (int16_t*)src);
            break;
        }
        case 27: {
            launchTROWMINCase27<<<1, nullptr, stream>>>((int16_t*)out, (int16_t*)src);
            break;
        }
        case 28: {
            launchTROWMINCase28<<<1, nullptr, stream>>>((int16_t*)out, (int16_t*)src);
            break;
        }
        case 29: {
            launchTROWMINCase29<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
            break;
        }
        case 30: {
            launchTROWMINCase30<<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src);
            break;
        }
        case 31: {
            launchTROWMINCase31<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
            break;
        }
        case 32: {
            launchTROWMINCase32<<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src);
            break;
        }
        case 33: {
            launchTROWMINCase33<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
            break;
        }
        case 34: {
            launchTROWMINCase34<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
            break;
        }
        case 35: {
            launchTROWMINCase35<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
            break;
        }
        case 36: {
            launchTROWMINCase36<<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src);
            break;
        }
        case 37: {
            launchTROWMINCase37<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
            break;
        }
        case 38: {
            launchTROWMINCase38<<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src);
            break;
        }
        case 39: {
            launchTROWMINCase39<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
            break;
        }
        case 40: {
            launchTROWMINCase40<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
            break;
        }
        case 41: {
            launchTROWMINCase41<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
            break;
        }
        case 42: {
            launchTROWMINCase42<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
            break;
        }
        case 43: {
            launchTROWMINCase43<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
            break;
        }
        case 44: {
            launchTROWMINCase44<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
            break;
        }
        case 45: {
            launchTROWMINCase45<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
            break;
        }
        case 46: {
            launchTROWMINCase46<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
            break;
        }
        case 47: {
            launchTROWMINCase47<<<1, nullptr, stream>>>((uint32_t*)out, (uint32_t*)src);
            break;
        }
        case 48: {
            launchTROWMINCase48<<<1, nullptr, stream>>>((uint32_t*)out, (uint32_t*)src);
            break;
        }
        case 49: {
            launchTROWMINCase49<<<1, nullptr, stream>>>((uint32_t*)out, (uint32_t*)src);
            break;
        }
        case 50: {
            launchTROWMINCase50<<<1, nullptr, stream>>>((uint32_t*)out, (uint32_t*)src);
            break;
        }
        case 51: {
            launchTROWMINCase51<<<1, nullptr, stream>>>((uint32_t*)out, (uint32_t*)src);
            break;
        }
        case 52: {
            launchTROWMINCase52<<<1, nullptr, stream>>>((uint16_t*)out, (uint16_t*)src);
            break;
        }
        case 53: {
            launchTROWMINCase53<<<1, nullptr, stream>>>((uint16_t*)out, (uint16_t*)src);
            break;
        }
        case 54: {
            launchTROWMINCase54<<<1, nullptr, stream>>>((uint16_t*)out, (uint16_t*)src);
            break;
        }
        case 55: {
            launchTROWMINCase55<<<1, nullptr, stream>>>((uint16_t*)out, (uint16_t*)src);
            break;
        }
        case 56: {
            launchTROWMINCase56<<<1, nullptr, stream>>>((uint16_t*)out, (uint16_t*)src);
            break;
        }
        case 57: {
            launchTROWMINCase57<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 58: {
            launchTROWMINCase58<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 59: {
            launchTROWMINCase59<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 60: {
            launchTROWMINCase60<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 61: {
            launchTROWMINCase61<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 62: {
            launchTROWMINCase62<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 63: {
            launchTROWMINCase63<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 64: {
            launchTROWMINCase64<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 65: {
            launchTROWMINCase65<<<1, nullptr, stream>>>((half*)out, (half*)src);
            break;
        }
        case 66: {
            launchTROWMINCase66<<<1, nullptr, stream>>>((half*)out, (half*)src);
            break;
        }
        case 67: {
            launchTROWMINCase67<<<1, nullptr, stream>>>((half*)out, (half*)src);
            break;
        }
        case 68: {
            launchTROWMINCase68<<<1, nullptr, stream>>>((half*)out, (half*)src);
            break;
        }
        case 69: {
            launchTROWMINCase69<<<1, nullptr, stream>>>((half*)out, (half*)src);
            break;
        }
        case 70: {
            launchTROWMINCase70<<<1, nullptr, stream>>>((int8_t*)out, (int8_t*)src);
            break;
        }
        case 71: {
            launchTROWMINCase71<<<1, nullptr, stream>>>((int8_t*)out, (int8_t*)src);
            break;
        }
        case 72: {
            launchTROWMINCase72<<<1, nullptr, stream>>>((int8_t*)out, (int8_t*)src);
            break;
        }
        case 73: {
            launchTROWMINCase73<<<1, nullptr, stream>>>((int8_t*)out, (int8_t*)src);
            break;
        }
        case 74: {
            launchTROWMINCase74<<<1, nullptr, stream>>>((int8_t*)out, (int8_t*)src);
            break;
        }
        case 75: {
            launchTROWMINCase75<<<1, nullptr, stream>>>((uint8_t*)out, (uint8_t*)src);
            break;
        }
        case 76: {
            launchTROWMINCase76<<<1, nullptr, stream>>>((uint8_t*)out, (uint8_t*)src);
            break;
        }
        case 77: {
            launchTROWMINCase77<<<1, nullptr, stream>>>((uint8_t*)out, (uint8_t*)src);
            break;
        }
        case 78: {
            launchTROWMINCase78<<<1, nullptr, stream>>>((uint32_t*)out, (uint32_t*)src);
            break;
        }
        case 79: {
            launchTROWMINCase79<<<1, nullptr, stream>>>((uint16_t*)out, (uint16_t*)src);
            break;
        }
        case 80: {
            launchTROWMINCase80<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 81: {
            launchTROWMINCase81<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 82: {
            launchTROWMINCase82<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 83: {
            launchTROWMINCase83<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 84: {
            launchTROWMINCase84<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 85: {
            launchTROWMINCase85<<<1, nullptr, stream>>>((half*)out, (half*)src);
            break;
        }
        case 86: {
            launchTROWMINCase86<<<1, nullptr, stream>>>((int16_t*)out, (int16_t*)src);
            break;
        }
        case 87: {
            launchTROWMINCase87<<<1, nullptr, stream>>>((int8_t*)out, (int8_t*)src);
            break;
        }
        case 88: {
            launchTROWMINCase88<<<1, nullptr, stream>>>((uint8_t*)out, (uint8_t*)src);
            break;
        }
        default: {
        }
    }
}

template void launchTROWMINTestCase<1>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<2>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<3>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<4>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<5>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<6>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<7>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<8>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<9>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<10>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<11>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<12>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<13>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<14>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<15>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<16>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<17>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<18>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<19>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<20>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<21>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<22>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<23>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<24>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<25>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<26>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<27>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<28>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<29>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<30>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<31>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<32>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<33>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<34>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<35>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<36>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<37>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<38>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<39>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<40>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<41>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<42>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<43>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<44>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<45>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<46>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<47>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<48>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<49>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<50>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<51>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<52>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<53>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<54>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<55>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<56>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<57>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<58>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<59>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<60>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<61>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<62>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<63>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<64>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<65>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<66>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<67>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<68>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<69>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<70>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<71>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<72>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<73>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<74>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<75>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<76>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<77>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<78>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<79>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<80>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<81>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<82>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<83>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<84>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<85>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<86>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<87>(void* out, void* src, aclrtStream stream);
template void launchTROWMINTestCase<88>(void* out, void* src, aclrtStream stream);
