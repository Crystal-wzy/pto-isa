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
#include "acl/acl.h"

using namespace std;
using namespace pto;

// Mat(NZ) -> GM via TSTORE(Mat); 8/16/32-bit.
template <
    typename T, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0, int gWholeShape1,
    int gWholeShape2, int gWholeShape3, int gWholeShape4>
AICORE inline void RunTStoreMatNz2Nz(__gm__ T* out, __gm__ T* src)
{
    constexpr int gStride[5] = {
        gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4, gWholeShape2 * gWholeShape3 * gWholeShape4,
        gWholeShape3 * gWholeShape4, gWholeShape4, 1};
    constexpr int Rows = gShape2 * gShape3;
    constexpr int Cols = gShape0 * gShape1 * gShape4;

    using DynShapeDim5 = Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5, Layout::NZ>;
    using TileData = Tile<TileType::Mat, T, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;

    int validRow = gShape2 * gShape3;
    int validCol = gShape0 * gShape1 * gShape4;
    TileData srcTile(validRow, validCol);

    TASSIGN(srcTile, 0x0);

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, srcTile);
    out = dstGlobal.data();
}

// Mat(NZ in L1) -> GM ND via TSTORE(Mat); NZ->ND done in software.
// Source is a contiguous NZ global (loaded 1:1 into the NZ Mat tile); destination is a
// 2D ND global Shape<1, 1, 1, M, N> placed in a gWholeM x gWholeN buffer, so a row
// stride gap (gWholeN > N) and bottom padding (gWholeM > M) are real. M % 16 == 0, N % C0 == 0.
template <typename T, int M, int N, int gWholeM, int gWholeN>
AICORE inline void RunTStoreMatNz2Nd(__gm__ T* out, __gm__ T* src)
{
    constexpr int c0 = 32 / sizeof(T);
    using SrcShape = pto::Shape<1, N / c0, M / 16, 16, c0>;
    using SrcStride = pto::Stride<M * N, (M / 16) * 16 * c0, 16 * c0, c0, 1>;
    using SrcGlobal = GlobalTensor<T, SrcShape, SrcStride, Layout::NZ>;

    using DstShape = pto::Shape<1, 1, 1, M, N>;
    using DstStride = pto::Stride<gWholeM * gWholeN, gWholeM * gWholeN, gWholeM * gWholeN, gWholeN, 1>;
    using DstGlobal = GlobalTensor<T, DstShape, DstStride, Layout::ND>;

    SrcGlobal srcGlobal(src);
    DstGlobal dstGlobal(out);

    using TileData = Tile<TileType::Mat, T, M, N, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    TileData srcTile(M, N);
    TASSIGN(srcTile, 0x0);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, srcTile);
    out = dstGlobal.data();
}

template <
    typename T, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0, int gWholeShape1,
    int gWholeShape2, int gWholeShape3, int gWholeShape4>
__global__ AICORE void TStoreMatNz2NzKernel(__gm__ T* out, __gm__ T* src)
{
    RunTStoreMatNz2Nz<
        T, gShape0, gShape1, gShape2, gShape3, gShape4, gWholeShape0, gWholeShape1, gWholeShape2, gWholeShape3,
        gWholeShape4>(out, src);
}

template <typename T, int M, int N, int gWholeM, int gWholeN>
__global__ AICORE void TStoreMatNz2NdKernel(__gm__ T* out, __gm__ T* src)
{
    RunTStoreMatNz2Nd<T, M, N, gWholeM, gWholeN>(out, src);
}

template <
    typename T, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0, int gWholeShape1,
    int gWholeShape2, int gWholeShape3, int gWholeShape4>
void LaunchTStoreNz2Nz(T* out, T* src, void* stream)
{
    TStoreMatNz2NzKernel<
        T, gShape0, gShape1, gShape2, gShape3, gShape4, gWholeShape0, gWholeShape1, gWholeShape2, gWholeShape3,
        gWholeShape4><<<1, nullptr, stream>>>(out, src);
}

template void LaunchTStoreNz2Nz<aclFloat16, 1, 1, 2, 16, 16, 1, 1, 2, 16, 16>(
    aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTStoreNz2Nz<int16_t, 1, 2, 1, 16, 16, 2, 2, 2, 16, 16>(int16_t* out, int16_t* src, void* stream);
template void LaunchTStoreNz2Nz<int8_t, 1, 1, 2, 16, 32, 1, 2, 2, 16, 32>(int8_t* out, int8_t* src, void* stream);
template void LaunchTStoreNz2Nz<uint16_t, 2, 1, 2, 16, 16, 2, 2, 2, 16, 16>(uint16_t* out, uint16_t* src, void* stream);
template void LaunchTStoreNz2Nz<int8_t, 1, 2, 1, 16, 32, 2, 2, 2, 16, 32>(int8_t* out, int8_t* src, void* stream);
template void LaunchTStoreNz2Nz<aclFloat16, 1, 1, 1, 16, 16, 2, 1, 1, 16, 16>(
    aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTStoreNz2Nz<uint16_t, 1, 4, 1, 16, 16, 2, 4, 2, 16, 16>(uint16_t* out, uint16_t* src, void* stream);
template void LaunchTStoreNz2Nz<int16_t, 3, 1, 2, 16, 16, 4, 1, 2, 16, 16>(int16_t* out, int16_t* src, void* stream);

// gShape1=17/37 + whole>tile: non-C0-aligned width and a real K-slice dst gap.
template void LaunchTStoreNz2Nz<aclFloat16, 1, 17, 1, 16, 16, 1, 17, 2, 16, 16>(
    aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTStoreNz2Nz<int16_t, 2, 37, 2, 16, 16, 2, 37, 3, 16, 16>(int16_t* out, int16_t* src, void* stream);
template void LaunchTStoreNz2Nz<int8_t, 1, 17, 2, 16, 32, 1, 17, 4, 16, 32>(int8_t* out, int8_t* src, void* stream);

// 32-bit NZ2NZ.
template void LaunchTStoreNz2Nz<int32_t, 1, 1, 2, 16, 8, 1, 1, 2, 16, 8>(int32_t* out, int32_t* src, void* stream);
template void LaunchTStoreNz2Nz<float, 2, 1, 2, 16, 8, 2, 2, 2, 16, 8>(float* out, float* src, void* stream);

// NZ2ND: Mat(NZ, L1) -> GM ND. M % 16 == 0, N % C0 == 0.
// Contiguous dst (gWholeN == N) and gapped/padded dst (gWholeN > N / gWholeM > M) both covered.
template <typename T, int M, int N, int gWholeM, int gWholeN>
void LaunchTStoreNz2Nd(T* out, T* src, void* stream)
{
    TStoreMatNz2NdKernel<T, M, N, gWholeM, gWholeN><<<1, nullptr, stream>>>(out, src);
}

template void LaunchTStoreNz2Nd<aclFloat16, 16, 16, 16, 16>(aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTStoreNz2Nd<int16_t, 32, 32, 32, 32>(int16_t* out, int16_t* src, void* stream);
template void LaunchTStoreNz2Nd<int8_t, 32, 32, 32, 32>(int8_t* out, int8_t* src, void* stream);
template void LaunchTStoreNz2Nd<uint16_t, 16, 64, 16, 64>(uint16_t* out, uint16_t* src, void* stream);
template void LaunchTStoreNz2Nd<aclFloat16, 64, 32, 64, 32>(aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTStoreNz2Nd<int16_t, 16, 16, 16, 16>(int16_t* out, int16_t* src, void* stream);
// Non-contiguous dst: row-stride gap (gWholeN > N) and/or bottom row padding (gWholeM > M).
template void LaunchTStoreNz2Nd<aclFloat16, 32, 32, 32, 48>(aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTStoreNz2Nd<int16_t, 32, 32, 40, 32>(int16_t* out, int16_t* src, void* stream);
template void LaunchTStoreNz2Nd<int8_t, 32, 64, 32, 96>(int8_t* out, int8_t* src, void* stream);
template void LaunchTStoreNz2Nd<uint16_t, 16, 16, 24, 32>(uint16_t* out, uint16_t* src, void* stream);
// 32-bit NZ2ND.
template void LaunchTStoreNz2Nd<int32_t, 32, 32, 32, 32>(int32_t* out, int32_t* src, void* stream);
template void LaunchTStoreNz2Nd<float, 32, 32, 32, 48>(float* out, float* src, void* stream);
