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

template <typename T, int row, int validRow, int srcCol, int srcValidCol, int dstCol>
PTO_INTERNAL void runTRowMax(__gm__ T* out, __gm__ T* src)
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
    TASSIGN<0x0>(srcTile);
    TASSIGN<row * srcCol * sizeof(T)>(tmpTile);
    TASSIGN<2 * row * srcCol * sizeof(T)>(dstTile);

    // 搬运数据
    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TROWMAX(dstTile, srcTile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
}

template <typename T, int row, int validRow, int srcCol, int srcValidCol, int dstCol>
PTO_INTERNAL void runTRowMaxDNDst(__gm__ T* out, __gm__ T* src)
{
    using ValidSrcShape = TileShape2D<T, validRow, srcValidCol>;
    using NDSrcShape = BaseShape2D<T, row, srcCol>;
    using GlobalDataSrc = GlobalTensor<T, ValidSrcShape, NDSrcShape>;
    GlobalDataSrc srcGlobal(src);

    using ValidDstShape = TileShape2D<T, dstCol, validRow>;
    using NDDstShape = BaseShape2D<T, row, dstCol>;
    using GlobalDataDst = GlobalTensor<T, ValidDstShape, NDDstShape>;
    GlobalDataDst dstGlobal(out);

    using srcTileData = Tile<TileType::Vec, T, row, srcCol, BLayout::RowMajor, row, srcCol>;
    using dstTileDataDN = Tile<TileType::Vec, T, row, 1, BLayout::ColMajor, row, 1>;
    srcTileData srcTile;
    srcTileData tmpTile;
    dstTileDataDN dstTile;
    TASSIGN<0x0>(srcTile);
    TASSIGN<row * srcCol * sizeof(T)>(tmpTile);
    TASSIGN<2 * row * srcCol * sizeof(T)>(dstTile);

    // 搬运数据
    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TROWMAX(dstTile, srcTile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    using dstTileDataND = Tile<TileType::Vec, T, 1, row, BLayout::RowMajor, 1, row>;
    dstTileDataND dstTileND;
    TRESHAPE(dstTileND, dstTile);
    TSTORE(dstGlobal, dstTileND);
}

// Half-width-tmp variant: tmp tile holds srcCol/2 columns (the pairwise-halving
// working set), for wide-row shapes where src + full-pitch tmp exceeds UB.
template <typename T, int row, int validRow, int srcCol, int srcValidCol, int dstCol, int tmpCol>
PTO_INTERNAL void runTRowMaxPackedTmp(__gm__ T* out, __gm__ T* src)
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
    using tmpTileData = Tile<TileType::Vec, T, row, tmpCol, BLayout::RowMajor, -1, -1>;
    srcTileData srcTile(validRow, srcValidCol);
    tmpTileData tmpTile(validRow, tmpCol);
    dstTileData dstTile(validRow, dstCol);
    TASSIGN<0x0>(srcTile);
    TASSIGN<row * srcCol * sizeof(T)>(tmpTile);
    TASSIGN<(row * srcCol + row * tmpCol) * sizeof(T)>(dstTile);

    // 搬运数据
    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TROWMAX(dstTile, srcTile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
}

extern "C" __global__ AICORE void launchTROWMAXCase1(__gm__ float* out, __gm__ float* src)
{
    runTRowMax<float, 127, 127, 64, 63, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMAXCase2(__gm__ float* out, __gm__ float* src)
{
    runTRowMax<float, 63, 63, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMAXCase3(__gm__ float* out, __gm__ float* src)
{
    runTRowMax<float, 31, 31, 128, 127, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMAXCase4(__gm__ float* out, __gm__ float* src)
{
    runTRowMax<float, 15, 15, 192, 192, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMAXCase5(__gm__ float* out, __gm__ float* src)
{
    runTRowMax<float, 7, 7, 448, 447, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMAXCase6(__gm__ half* out, __gm__ half* src)
{
    runTRowMax<half, 256, 256, 16, 15, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMAXCase7(__gm__ float* out, __gm__ float* src)
{
    runTRowMax<float, 30, 30, 216, 216, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMAXCase8(__gm__ float* out, __gm__ float* src)
{
    runTRowMax<float, 30, 30, 216, 24, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMAXCase9(__gm__ float* out, __gm__ float* src)
{
    runTRowMax<float, 30, 11, 216, 216, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMAXCase10(__gm__ float* out, __gm__ float* src)
{
    runTRowMax<float, 30, 11, 216, 24, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMAXCase11(__gm__ float* out, __gm__ float* src)
{
    runTRowMax<float, 238, 238, 40, 40, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMAXCase12(__gm__ float* out, __gm__ float* src)
{
    runTRowMax<float, 238, 238, 40, 16, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMAXCase13(__gm__ float* out, __gm__ float* src)
{
    runTRowMax<float, 238, 121, 40, 40, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMAXCase14(__gm__ float* out, __gm__ float* src)
{
    runTRowMax<float, 238, 121, 40, 16, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMAXCase15(__gm__ float* out, __gm__ float* src)
{
    runTRowMaxDNDst<float, 64, 64, 128, 128, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMAXCase16(__gm__ float* out, __gm__ float* src)
{
    runTRowMaxDNDst<float, 32, 32, 256, 256, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMAXCase17(__gm__ float* out, __gm__ float* src)
{
    runTRowMaxDNDst<float, 16, 16, 512, 512, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMAXCase18(__gm__ float* out, __gm__ float* src)
{
    runTRowMaxDNDst<float, 8, 8, 1024, 1024, 1>(out, src);
}

// int32 test cases
extern "C" __global__ AICORE void launchTROWMAXCase19(__gm__ int32_t* out, __gm__ int32_t* src)
{
    runTRowMax<int32_t, 127, 127, 64, 63, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMAXCase20(__gm__ int32_t* out, __gm__ int32_t* src)
{
    runTRowMax<int32_t, 63, 63, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMAXCase21(__gm__ int32_t* out, __gm__ int32_t* src)
{
    runTRowMax<int32_t, 31, 31, 128, 127, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMAXCase22(__gm__ int32_t* out, __gm__ int32_t* src)
{
    runTRowMax<int32_t, 15, 15, 192, 192, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMAXCase23(__gm__ int32_t* out, __gm__ int32_t* src)
{
    runTRowMax<int32_t, 7, 7, 448, 447, 1>(out, src);
}

// int16 test cases - need 32-byte alignment for int16_t (2 bytes), so cols must be multiple of 16
extern "C" __global__ AICORE void launchTROWMAXCase24(__gm__ int16_t* out, __gm__ int16_t* src)
{
    runTRowMax<int16_t, 128, 128, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMAXCase25(__gm__ int16_t* out, __gm__ int16_t* src)
{
    runTRowMax<int16_t, 64, 64, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMAXCase26(__gm__ int16_t* out, __gm__ int16_t* src)
{
    runTRowMax<int16_t, 32, 32, 128, 128, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMAXCase27(__gm__ int16_t* out, __gm__ int16_t* src)
{
    runTRowMax<int16_t, 16, 16, 192, 192, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMAXCase28(__gm__ int16_t* out, __gm__ int16_t* src)
{
    runTRowMax<int16_t, 8, 8, 448, 448, 1>(out, src);
}

// traversal-optimization cases (issue #1): few rows / wide rows / over-limit row pitch
// f32 4x6144: Q=96, pitch 768 blocks -> across-fallback 384, within=4
extern "C" __global__ AICORE void launchTROWMAXCase29(__gm__ float* out, __gm__ float* src)
{
    runTRowMaxPackedTmp<float, 4, 4, 6144, 6144, 1, 3072>(out, src);
}
// f32 16x2032: pitch 254 blocks (encodable), across=32 vs within=32: tie -> across
extern "C" __global__ AICORE void launchTROWMAXCase30(__gm__ float* out, __gm__ float* src)
{
    runTRowMaxPackedTmp<float, 16, 16, 2032, 2032, 1, 1016>(out, src);
}
// f32 12x2048: pitch 256 blocks > 255 -> across-fallback 384, within=12
extern "C" __global__ AICORE void launchTROWMAXCase31(__gm__ float* out, __gm__ float* src)
{
    runTRowMaxPackedTmp<float, 12, 12, 2048, 2048, 1, 1024>(out, src);
}
// f32 8x2560: pitch 320 blocks (fallback), within=8 vs across=320
extern "C" __global__ AICORE void launchTROWMAXCase32(__gm__ float* out, __gm__ float* src)
{
    runTRowMaxPackedTmp<float, 8, 8, 2560, 2560, 1, 1280>(out, src);
}
// f16 3x16384: Q=128, pitch 1024 blocks (fallback), within=3 vs across=384
extern "C" __global__ AICORE void launchTROWMAXCase33(__gm__ half* out, __gm__ half* src)
{
    runTRowMaxPackedTmp<half, 3, 3, 16384, 16384, 1, 8192>(out, src);
}
// f16 1x32768: R=1 extreme, Q=256, within=2 vs across=256
extern "C" __global__ AICORE void launchTROWMAXCase34(__gm__ half* out, __gm__ half* src)
{
    runTRowMaxPackedTmp<half, 1, 1, 32768, 32768, 1, 16384>(out, src);
}
// f32 2x12288: Q=192, pitch 768 blocks (fallback), within=2 vs across=384
extern "C" __global__ AICORE void launchTROWMAXCase35(__gm__ float* out, __gm__ float* src)
{
    runTRowMaxPackedTmp<float, 2, 2, 12288, 12288, 1, 6144>(out, src);
}
// f32 48x512: many rows, Q=8 -> model keeps across (within=48 > across=8), guard
extern "C" __global__ AICORE void launchTROWMAXCase36(__gm__ float* out, __gm__ float* src)
{
    runTRowMaxPackedTmp<float, 48, 48, 512, 512, 1, 256>(out, src);
}
// f16 3x8189 of 3x8192: Q=63 + tail 61, pitch 512 blocks (fallback), within=6 vs across=192
extern "C" __global__ AICORE void launchTROWMAXCase37(__gm__ half* out, __gm__ half* src)
{
    runTRowMaxPackedTmp<half, 3, 3, 8192, 8189, 1, 4096>(out, src);
}
// f32 4x6141 of 4x6144: Q=95 + tail 61, pitch 768 blocks (fallback), within=8 vs across=384
extern "C" __global__ AICORE void launchTROWMAXCase38(__gm__ float* out, __gm__ float* src)
{
    runTRowMaxPackedTmp<float, 4, 4, 6144, 6141, 1, 3072>(out, src);
}

// boundary / minor-gain sweep (issue #1), packed 256B/row tmp tiles
// f32 32x256: Q=4, R=32 -> within=32 > across=4: stays across (no gain by design)
extern "C" __global__ AICORE void launchTROWMAXCase39(__gm__ float* out, __gm__ float* src)
{
    runTRowMaxPackedTmp<float, 32, 32, 256, 256, 1, 64>(out, src);
}
// f32 12x1024: Q=16, R=12, pitch 128blk encodable -> within=12 < across=16: minor predicted
extern "C" __global__ AICORE void launchTROWMAXCase40(__gm__ float* out, __gm__ float* src)
{
    runTRowMaxPackedTmp<float, 12, 12, 1024, 1024, 1, 64>(out, src);
}
// f32 16x1024: Q=16, R=16 -> tie 16 vs 16: stays across
extern "C" __global__ AICORE void launchTROWMAXCase41(__gm__ float* out, __gm__ float* src)
{
    runTRowMaxPackedTmp<float, 16, 16, 1024, 1024, 1, 64>(out, src);
}
// f16 8x2048: Q=16, R=8, pitch 128blk encodable -> within=8 < across=16: minor predicted
extern "C" __global__ AICORE void launchTROWMAXCase42(__gm__ half* out, __gm__ half* src)
{
    runTRowMaxPackedTmp<half, 8, 8, 2048, 2048, 1, 128>(out, src);
}
// f32 6x4096: Q=64, R=6, pitch 512blk > 255 -> across-fallback 384 vs within=6
extern "C" __global__ AICORE void launchTROWMAXCase43(__gm__ float* out, __gm__ float* src)
{
    runTRowMaxPackedTmp<float, 6, 6, 4096, 4096, 1, 64>(out, src);
}
// f16 2x8192: Q=64, R=2, pitch 512blk > 255 -> across-fallback 128 vs within=2
extern "C" __global__ AICORE void launchTROWMAXCase44(__gm__ half* out, __gm__ half* src)
{
    runTRowMaxPackedTmp<half, 2, 2, 8192, 8192, 1, 128>(out, src);
}
// f16 8x4096: Q=32, pitch 256blk > 255 (f16 encoding-limit cliff) -> across-fallback 256 vs within=8
extern "C" __global__ AICORE void launchTROWMAXCase45(__gm__ half* out, __gm__ half* src)
{
    runTRowMaxPackedTmp<half, 8, 8, 4096, 4096, 1, 128>(out, src);
}

template <uint32_t caseId>
void launchTROWMAXTestCase(void* out, void* src, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTROWMAXCase1<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 2: {
            launchTROWMAXCase2<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 3: {
            launchTROWMAXCase3<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 4: {
            launchTROWMAXCase4<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 5: {
            launchTROWMAXCase5<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 6: {
            launchTROWMAXCase6<<<1, nullptr, stream>>>((half*)out, (half*)src);
            break;
        }
        case 7: {
            launchTROWMAXCase7<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 8: {
            launchTROWMAXCase8<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 9: {
            launchTROWMAXCase9<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 10: {
            launchTROWMAXCase10<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 11: {
            launchTROWMAXCase11<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 12: {
            launchTROWMAXCase12<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 13: {
            launchTROWMAXCase13<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 14: {
            launchTROWMAXCase14<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 15: {
            launchTROWMAXCase15<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 16: {
            launchTROWMAXCase16<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 17: {
            launchTROWMAXCase17<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 18: {
            launchTROWMAXCase18<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 19: {
            launchTROWMAXCase19<<<1, nullptr, stream>>>((int32_t*)out, (int32_t*)src);
            break;
        }
        case 20: {
            launchTROWMAXCase20<<<1, nullptr, stream>>>((int32_t*)out, (int32_t*)src);
            break;
        }
        case 21: {
            launchTROWMAXCase21<<<1, nullptr, stream>>>((int32_t*)out, (int32_t*)src);
            break;
        }
        case 22: {
            launchTROWMAXCase22<<<1, nullptr, stream>>>((int32_t*)out, (int32_t*)src);
            break;
        }
        case 23: {
            launchTROWMAXCase23<<<1, nullptr, stream>>>((int32_t*)out, (int32_t*)src);
            break;
        }
        case 24: {
            launchTROWMAXCase24<<<1, nullptr, stream>>>((int16_t*)out, (int16_t*)src);
            break;
        }
        case 25: {
            launchTROWMAXCase25<<<1, nullptr, stream>>>((int16_t*)out, (int16_t*)src);
            break;
        }
        case 26: {
            launchTROWMAXCase26<<<1, nullptr, stream>>>((int16_t*)out, (int16_t*)src);
            break;
        }
        case 27: {
            launchTROWMAXCase27<<<1, nullptr, stream>>>((int16_t*)out, (int16_t*)src);
            break;
        }
        case 28: {
            launchTROWMAXCase28<<<1, nullptr, stream>>>((int16_t*)out, (int16_t*)src);
            break;
        }
        case 29: {
            launchTROWMAXCase29<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 30: {
            launchTROWMAXCase30<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 31: {
            launchTROWMAXCase31<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 32: {
            launchTROWMAXCase32<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 33: {
            launchTROWMAXCase33<<<1, nullptr, stream>>>((half*)out, (half*)src);
            break;
        }
        case 34: {
            launchTROWMAXCase34<<<1, nullptr, stream>>>((half*)out, (half*)src);
            break;
        }
        case 35: {
            launchTROWMAXCase35<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 36: {
            launchTROWMAXCase36<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 37: {
            launchTROWMAXCase37<<<1, nullptr, stream>>>((half*)out, (half*)src);
            break;
        }
        case 38: {
            launchTROWMAXCase38<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 39: {
            launchTROWMAXCase39<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 40: {
            launchTROWMAXCase40<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 41: {
            launchTROWMAXCase41<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 42: {
            launchTROWMAXCase42<<<1, nullptr, stream>>>((half*)out, (half*)src);
            break;
        }
        case 43: {
            launchTROWMAXCase43<<<1, nullptr, stream>>>((float*)out, (float*)src);
            break;
        }
        case 44: {
            launchTROWMAXCase44<<<1, nullptr, stream>>>((half*)out, (half*)src);
            break;
        }
        case 45: {
            launchTROWMAXCase45<<<1, nullptr, stream>>>((half*)out, (half*)src);
            break;
        }
        default: {
        }
    }
}

template void launchTROWMAXTestCase<1>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<2>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<3>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<4>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<5>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<6>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<7>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<8>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<9>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<10>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<11>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<12>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<13>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<14>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<15>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<16>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<17>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<18>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<19>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<20>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<21>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<22>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<23>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<24>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<25>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<26>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<27>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<28>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<29>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<30>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<31>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<32>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<33>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<34>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<35>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<36>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<37>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<38>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<39>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<40>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<41>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<42>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<43>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<44>(void* out, void* src, aclrtStream stream);
template void launchTROWMAXTestCase<45>(void* out, void* src, aclrtStream stream);
