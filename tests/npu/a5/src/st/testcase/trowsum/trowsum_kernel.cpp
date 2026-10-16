/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <acl/acl.h>

using namespace std;
using namespace pto;

template <typename T, int row, int validRow, int srcCol, int srcValidCol, int dstCol>
PTO_INTERNAL void runTRowSum(__gm__ T __out__* out, __gm__ T __in__* src)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2StrideSrc = pto::Stride<1, 1, -1, -1, 1>;
    using DynDim2StrideDst = pto::Stride<1, 1, 1, -1, -1>;

    using GlobalDataSrc = GlobalTensor<T, DynDim2Shape, DynDim2StrideSrc>;
    using GlobalDataDst = GlobalTensor<T, DynDim2Shape, DynDim2StrideDst>;
    GlobalDataSrc srcGlobal(src, DynDim2Shape(validRow, srcValidCol), DynDim2StrideSrc(row, srcCol));
    GlobalDataDst dstGlobal(out, DynDim2Shape(validRow, dstCol), DynDim2StrideDst(dstCol, row));
    using srcTileData = Tile<TileType::Vec, T, row, srcCol, BLayout::RowMajor, -1, -1>;
    using dstTileData = Tile<TileType::Vec, T, row, 16, BLayout::RowMajor, -1, -1>;
    srcTileData srcTile(validRow, srcValidCol);
    srcTileData tmpTile(validRow, srcValidCol);
    dstTileData dstTile(validRow, dstCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(tmpTile, row * srcCol * sizeof(T));
    TASSIGN(dstTile, 2 * row * srcCol * sizeof(T));

    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TROWSUM(dstTile, srcTile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
}

template <typename T, int row, int validRow, int srcCol, int srcValidCol, int dstCol, int globalSrcCol = srcCol>
PTO_INTERNAL void runTRowSumDNDst(__gm__ T* out, __gm__ T* src)
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
    TASSIGN(srcTile, 0x0);
    TASSIGN(tmpTile, row * srcCol * sizeof(T));
    TASSIGN(dstTile, 2 * row * srcCol * sizeof(T));

    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TROWSUM(dstTile, srcTile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    using dstTileDataND = Tile<TileType::Vec, T, 1, row, BLayout::RowMajor, 1, validRow>;
    dstTileDataND dstTileND;
    TRESHAPE(dstTileND, dstTile);
    TSTORE(dstGlobal, dstTileND);
}

extern "C" __global__ AICORE void launchTROWSUMCase1(__gm__ float* out, __gm__ float* src)
{
    runTRowSum<float, 127, 127, 64, 63, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase2(__gm__ float* out, __gm__ float* src)
{
    runTRowSum<float, 63, 63, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase3(__gm__ float* out, __gm__ float* src)
{
    runTRowSum<float, 31, 31, 128, 127, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase4(__gm__ float* out, __gm__ float* src)
{
    runTRowSum<float, 15, 15, 192, 192, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase5(__gm__ float* out, __gm__ float* src)
{
    runTRowSum<float, 7, 7, 448, 447, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase6(__gm__ half* out, __gm__ half* src)
{
    runTRowSum<half, 256, 256, 16, 15, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase7(__gm__ float* out, __gm__ float* src)
{
    runTRowSumDNDst<float, 64, 64, 128, 128, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase8(__gm__ float* out, __gm__ float* src)
{
    runTRowSumDNDst<float, 32, 32, 256, 256, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase9(__gm__ float* out, __gm__ float* src)
{
    runTRowSumDNDst<float, 16, 16, 512, 512, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase10(__gm__ float* out, __gm__ float* src)
{
    runTRowSumDNDst<float, 8, 8, 1024, 1024, 1>(out, src);
}

extern "C" __global__ AICORE void launchTROWSUMCase11(__gm__ int32_t* out, __gm__ int32_t* src)
{
    runTRowSum<int32_t, 127, 127, 64, 63, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase12(__gm__ int32_t* out, __gm__ int32_t* src)
{
    runTRowSum<int32_t, 63, 63, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase13(__gm__ int32_t* out, __gm__ int32_t* src)
{
    runTRowSum<int32_t, 31, 31, 128, 127, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase14(__gm__ int32_t* out, __gm__ int32_t* src)
{
    runTRowSum<int32_t, 15, 15, 192, 192, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase15(__gm__ int32_t* out, __gm__ int32_t* src)
{
    runTRowSum<int32_t, 7, 7, 448, 447, 1>(out, src);
}

// Keep int16_t row strides 32-byte aligned.
extern "C" __global__ AICORE void launchTROWSUMCase16(__gm__ int16_t* out, __gm__ int16_t* src)
{
    runTRowSum<int16_t, 128, 128, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase17(__gm__ int16_t* out, __gm__ int16_t* src)
{
    runTRowSum<int16_t, 64, 64, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase18(__gm__ int16_t* out, __gm__ int16_t* src)
{
    runTRowSum<int16_t, 32, 32, 128, 128, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase19(__gm__ int16_t* out, __gm__ int16_t* src)
{
    runTRowSum<int16_t, 16, 16, 192, 192, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase20(__gm__ int16_t* out, __gm__ int16_t* src)
{
    runTRowSum<int16_t, 8, 8, 448, 448, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase21(__gm__ int64_t* out, __gm__ int64_t* src)
{
    runTRowSum<int64_t, 4, 4, 16, 15, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase22(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTRowSum<uint64_t, 4, 4, 16, 15, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase23(__gm__ int64_t* out, __gm__ int64_t* src)
{
    runTRowSum<int64_t, 4, 4, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase24(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTRowSum<uint64_t, 4, 4, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase25(__gm__ int64_t* out, __gm__ int64_t* src)
{
    runTRowSum<int64_t, 32, 32, 32, 32, 1>(out, src);
}

extern "C" __global__ AICORE void launchTROWSUMCase26(__gm__ int64_t* out, __gm__ int64_t* src)
{
    runTRowSumDNDst<int64_t, 32, 32, 32, 32, 1>(out, src);
}

extern "C" __global__ AICORE void launchTROWSUMCase27(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTRowSumDNDst<uint64_t, 32, 32, 32, 32, 1>(out, src);
}

extern "C" __global__ AICORE void launchTROWSUMCase28(__gm__ int64_t* out, __gm__ int64_t* src)
{
    runTRowSumDNDst<int64_t, 32, 32, 148, 145, 1, 145>(out, src);
}

extern "C" __global__ AICORE void launchTROWSUMCase29(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTRowSumDNDst<uint64_t, 32, 32, 148, 145, 1, 145>(out, src);
}

template <typename T, int rows, int validRows, int cols, int validCols, int dstCols>
__global__ AICORE void launchTROWSUMGuard(__gm__ T* out, __gm__ T* src)
{
    constexpr int outputElements = rows * dstCols + 64;
    constexpr BLayout dstLayout = dstCols == 1 ? BLayout::ColMajor : BLayout::RowMajor;
    using SrcView = Tile<TileType::Vec, T, rows, cols, BLayout::RowMajor, rows, cols>;
    using SrcTile = Tile<TileType::Vec, T, rows, cols, BLayout::RowMajor, validRows, validCols>;
    using DstTile = Tile<TileType::Vec, T, rows, dstCols, dstLayout, validRows, 1>;
    using OutputView = Tile<TileType::Vec, T, 1, outputElements, BLayout::RowMajor, 1, outputElements>;
    using SrcGlobal = GlobalTensor<T, Shape<1, 1, 1, rows, cols>, pto::Stride<1, 1, 1, cols, 1>>;
    using OutputGlobal = GlobalTensor<T, Shape<1, 1, 1, 1, outputElements>, pto::Stride<1, 1, 1, outputElements, 1>>;
    SrcView srcView;
    SrcTile srcTile;
    SrcTile tmpTile;
    DstTile dstTile;
    OutputView outputView;
    SrcGlobal srcGlobal(src);
    OutputGlobal outputGlobal(out);
    TASSIGN(srcView, 0);
    TASSIGN(srcTile, 0);
    TASSIGN(tmpTile, 16384);
    TASSIGN(dstTile, 32768);
    // Read back padding, inactive rows, and the adjacent allocation guard.
    TASSIGN(outputView, 32768);
    TLOAD(srcView, srcGlobal);
    TLOAD(outputView, outputGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TROWSUM(dstTile, srcTile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(outputGlobal, outputView);
}

extern "C" __global__ AICORE void launchTROWSUMCase47(__gm__ uint32_t* out, __gm__ uint32_t* src)
{
    runTRowSum<uint32_t, 127, 127, 64, 63, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase48(__gm__ uint32_t* out, __gm__ uint32_t* src)
{
    runTRowSum<uint32_t, 63, 63, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase49(__gm__ uint32_t* out, __gm__ uint32_t* src)
{
    runTRowSum<uint32_t, 31, 31, 128, 127, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase50(__gm__ uint32_t* out, __gm__ uint32_t* src)
{
    runTRowSum<uint32_t, 15, 15, 192, 192, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase51(__gm__ uint32_t* out, __gm__ uint32_t* src)
{
    runTRowSum<uint32_t, 7, 7, 448, 447, 1>(out, src);
}

// Keep int16_t row strides 32-byte aligned.
extern "C" __global__ AICORE void launchTROWSUMCase52(__gm__ uint16_t* out, __gm__ uint16_t* src)
{
    runTRowSum<uint16_t, 128, 128, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase53(__gm__ uint16_t* out, __gm__ uint16_t* src)
{
    runTRowSum<uint16_t, 64, 64, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase54(__gm__ uint16_t* out, __gm__ uint16_t* src)
{
    runTRowSum<uint16_t, 32, 32, 128, 128, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase55(__gm__ uint16_t* out, __gm__ uint16_t* src)
{
    runTRowSum<uint16_t, 16, 16, 192, 192, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase56(__gm__ uint16_t* out, __gm__ uint16_t* src)
{
    runTRowSum<uint16_t, 8, 8, 448, 448, 1>(out, src);
}

extern "C" __global__ AICORE void launchTROWSUMCase57(__gm__ half* out, __gm__ half* src)
{
    runTRowSum<half, 8, 8, 448, 447, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase58(__gm__ half* out, __gm__ half* src)
{
    runTRowSum<half, 64, 64, 128, 128, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase59(__gm__ float* out, __gm__ float* src)
{
    runTRowSum<float, 8, 8, 8, 1, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase60(__gm__ float* out, __gm__ float* src)
{
    runTRowSum<float, 1, 1, 8192, 8192, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase61(__gm__ float* out, __gm__ float* src)
{
    runTRowSum<float, 256, 256, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase62(__gm__ float* out, __gm__ float* src)
{
    runTRowSum<float, 8, 8, 2048, 2048, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase63(__gm__ float* out, __gm__ float* src)
{
    runTRowSum<float, 64, 64, 64, 1, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase64(__gm__ float* out, __gm__ float* src)
{
    runTRowSum<float, 63, 63, 72, 65, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase65(__gm__ float* out, __gm__ float* src)
{
    runTRowSum<float, 33, 33, 136, 129, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase66(__gm__ float* out, __gm__ float* src)
{
    runTRowSum<float, 238, 237, 80, 79, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase67(__gm__ half* out, __gm__ half* src)
{
    runTRowSum<half, 64, 64, 128, 128, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase68(__gm__ half* out, __gm__ half* src)
{
    runTRowSum<half, 32, 32, 256, 255, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase69(__gm__ half* out, __gm__ half* src)
{
    runTRowSum<half, 16, 16, 512, 512, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase70(__gm__ half* out, __gm__ half* src)
{
    runTRowSum<half, 8, 8, 1024, 1024, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase71(__gm__ half* out, __gm__ half* src)
{
    runTRowSum<half, 30, 30, 224, 24, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase80(__gm__ uint32_t* out, __gm__ uint32_t* src)
{
    runTRowSum<uint32_t, 31, 31, 128, 127, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase81(__gm__ uint16_t* out, __gm__ uint16_t* src)
{
    runTRowSum<uint16_t, 32, 32, 128, 127, 1>(out, src);
}

extern "C" __global__ AICORE void launchTROWSUMCase82(__gm__ float* out, __gm__ float* src)
{
    runTRowSum<float, 106, 105, 192, 191, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase83(__gm__ float* out, __gm__ float* src)
{
    runTRowSum<float, 1, 1, 16384, 16384, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase84(__gm__ float* out, __gm__ float* src)
{
    runTRowSum<float, 30, 11, 216, 216, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase85(__gm__ float* out, __gm__ float* src)
{
    runTRowSum<float, 64, 1, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase86(__gm__ float* out, __gm__ float* src)
{
    runTRowSum<float, 251, 250, 96, 95, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase87(__gm__ half* out, __gm__ half* src)
{
    runTRowSum<half, 128, 127, 320, 319, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase88(__gm__ int16_t* out, __gm__ int16_t* src)
{
    runTRowSum<int16_t, 200, 199, 208, 207, 1>(out, src);
}

template <uint32_t caseId>
void launchTROWSUMTestCase(void* out, void* src, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTROWSUMCase1<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 2: {
            launchTROWSUMCase2<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 3: {
            launchTROWSUMCase3<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 4: {
            launchTROWSUMCase4<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 5: {
            launchTROWSUMCase5<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 6: {
            launchTROWSUMCase6<<<1, nullptr, stream>>>((half*)out, (half*)src);
            break;
        }
        case 7: {
            launchTROWSUMCase7<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 8: {
            launchTROWSUMCase8<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 9: {
            launchTROWSUMCase9<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 10: {
            launchTROWSUMCase10<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 11: {
            launchTROWSUMCase11<<<1, nullptr, stream>>>((int32_t*)out, (int32_t*)src);
            break;
        }
        case 12: {
            launchTROWSUMCase12<<<1, nullptr, stream>>>((int32_t*)out, (int32_t*)src);
            break;
        }
        case 13: {
            launchTROWSUMCase13<<<1, nullptr, stream>>>((int32_t*)out, (int32_t*)src);
            break;
        }
        case 14: {
            launchTROWSUMCase14<<<1, nullptr, stream>>>((int32_t*)out, (int32_t*)src);
            break;
        }
        case 15: {
            launchTROWSUMCase15<<<1, nullptr, stream>>>((int32_t*)out, (int32_t*)src);
            break;
        }
        case 16: {
            launchTROWSUMCase16<<<1, nullptr, stream>>>((int16_t*)out, (int16_t*)src);
            break;
        }
        case 17: {
            launchTROWSUMCase17<<<1, nullptr, stream>>>((int16_t*)out, (int16_t*)src);
            break;
        }
        case 18: {
            launchTROWSUMCase18<<<1, nullptr, stream>>>((int16_t*)out, (int16_t*)src);
            break;
        }
        case 19: {
            launchTROWSUMCase19<<<1, nullptr, stream>>>((int16_t*)out, (int16_t*)src);
            break;
        }
        case 20: {
            launchTROWSUMCase20<<<1, nullptr, stream>>>((int16_t*)out, (int16_t*)src);
            break;
        }
        case 21: {
            launchTROWSUMCase21<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
            break;
        }
        case 22: {
            launchTROWSUMCase22<<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src);
            break;
        }
        case 23: {
            launchTROWSUMCase23<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
            break;
        }
        case 24: {
            launchTROWSUMCase24<<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src);
            break;
        }
        case 25: {
            launchTROWSUMCase25<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
            break;
        }
        case 26: {
            launchTROWSUMCase26<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
            break;
        }
        case 27: {
            launchTROWSUMCase27<<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src);
            break;
        }
        case 28: {
            launchTROWSUMCase28<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
            break;
        }
        case 29: {
            launchTROWSUMCase29<<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src);
            break;
        }
        case 102: {
            launchTROWSUMGuard<int64_t, 64, 64, 16, 16, 4><<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
            break;
        }
        case 103: {
            launchTROWSUMGuard<int64_t, 8, 5, 16, 16, 1><<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
            break;
        }
        case 104: {
            launchTROWSUMGuard<int64_t, 8, 6, 16, 16, 1><<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
            break;
        }
        case 105: {
            launchTROWSUMGuard<int64_t, 8, 7, 16, 16, 1><<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
            break;
        }
        case 106: {
            launchTROWSUMGuard<uint64_t, 64, 64, 16, 16, 4><<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src);
            break;
        }
        case 107: {
            launchTROWSUMGuard<uint64_t, 8, 5, 16, 16, 1><<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src);
            break;
        }
        case 108: {
            launchTROWSUMGuard<uint64_t, 8, 6, 16, 16, 1><<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src);
            break;
        }
        case 109: {
            launchTROWSUMGuard<uint64_t, 8, 7, 16, 16, 1><<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src);
            break;
        }
        case 47: {
            launchTROWSUMCase47<<<1, nullptr, stream>>>((uint32_t*)out, (uint32_t*)src);
            break;
        }
        case 48: {
            launchTROWSUMCase48<<<1, nullptr, stream>>>((uint32_t*)out, (uint32_t*)src);
            break;
        }
        case 49: {
            launchTROWSUMCase49<<<1, nullptr, stream>>>((uint32_t*)out, (uint32_t*)src);
            break;
        }
        case 50: {
            launchTROWSUMCase50<<<1, nullptr, stream>>>((uint32_t*)out, (uint32_t*)src);
            break;
        }
        case 51: {
            launchTROWSUMCase51<<<1, nullptr, stream>>>((uint32_t*)out, (uint32_t*)src);
            break;
        }
        case 52: {
            launchTROWSUMCase52<<<1, nullptr, stream>>>((uint16_t*)out, (uint16_t*)src);
            break;
        }
        case 53: {
            launchTROWSUMCase53<<<1, nullptr, stream>>>((uint16_t*)out, (uint16_t*)src);
            break;
        }
        case 54: {
            launchTROWSUMCase54<<<1, nullptr, stream>>>((uint16_t*)out, (uint16_t*)src);
            break;
        }
        case 55: {
            launchTROWSUMCase55<<<1, nullptr, stream>>>((uint16_t*)out, (uint16_t*)src);
            break;
        }
        case 56: {
            launchTROWSUMCase56<<<1, nullptr, stream>>>((uint16_t*)out, (uint16_t*)src);
            break;
        }
        case 57: {
            launchTROWSUMCase57<<<1, nullptr, stream>>>((half*)out, (half*)src);
            break;
        }
        case 58: {
            launchTROWSUMCase58<<<1, nullptr, stream>>>((half*)out, (half*)src);
            break;
        }
        case 59: {
            launchTROWSUMCase59<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 60: {
            launchTROWSUMCase60<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 61: {
            launchTROWSUMCase61<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 62: {
            launchTROWSUMCase62<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 63: {
            launchTROWSUMCase63<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 64: {
            launchTROWSUMCase64<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 65: {
            launchTROWSUMCase65<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 66: {
            launchTROWSUMCase66<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 67: {
            launchTROWSUMCase67<<<1, nullptr, stream>>>((half*)out, (half*)src);
            break;
        }
        case 68: {
            launchTROWSUMCase68<<<1, nullptr, stream>>>((half*)out, (half*)src);
            break;
        }
        case 69: {
            launchTROWSUMCase69<<<1, nullptr, stream>>>((half*)out, (half*)src);
            break;
        }
        case 70: {
            launchTROWSUMCase70<<<1, nullptr, stream>>>((half*)out, (half*)src);
            break;
        }
        case 71: {
            launchTROWSUMCase71<<<1, nullptr, stream>>>((half*)out, (half*)src);
            break;
        }
        case 80: {
            launchTROWSUMCase80<<<1, nullptr, stream>>>((uint32_t*)out, (uint32_t*)src);
            break;
        }
        case 81: {
            launchTROWSUMCase81<<<1, nullptr, stream>>>((uint16_t*)out, (uint16_t*)src);
            break;
        }
        case 82: {
            launchTROWSUMCase82<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 83: {
            launchTROWSUMCase83<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 84: {
            launchTROWSUMCase84<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 85: {
            launchTROWSUMCase85<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 86: {
            launchTROWSUMCase86<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 87: {
            launchTROWSUMCase87<<<1, nullptr, stream>>>((half*)out, (half*)src);
            break;
        }
        case 88: {
            launchTROWSUMCase88<<<1, nullptr, stream>>>((int16_t*)out, (int16_t*)src);
            break;
        }
        default: {
        }
    }
}

template void launchTROWSUMTestCase<1>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<2>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<3>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<4>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<5>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<6>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<7>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<8>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<9>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<10>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<11>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<12>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<13>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<14>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<15>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<16>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<17>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<18>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<19>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<20>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<21>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<22>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<23>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<24>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<25>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<26>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<27>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<28>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<29>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<102>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<103>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<104>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<105>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<106>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<107>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<108>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<109>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<47>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<48>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<49>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<50>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<51>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<52>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<53>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<54>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<55>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<56>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<57>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<58>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<59>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<60>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<61>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<62>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<63>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<64>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<65>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<66>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<67>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<68>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<69>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<70>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<71>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<72>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<73>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<74>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<75>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<76>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<77>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<78>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<79>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<80>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<81>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<82>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<83>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<84>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<85>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<86>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<87>(void* out, void* src, aclrtStream stream);
template void launchTROWSUMTestCase<88>(void* out, void* src, aclrtStream stream);
