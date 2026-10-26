/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>
#include <type_traits>

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

// Fill the whole declared L1 tile from the raw GM input; TSTORE may then only write the valid box back to GM.
template <typename T, typename TileData>
__tf__ AICORE inline void PrefillConvTile(TileData& tile, __gm__ T* src, uint32_t elements)
{
    __cbuf__ T* dst = (__cbuf__ T*)__cce_get_tile_ptr(tile.data());
    constexpr uint32_t blockElems = BLOCK_BYTE_SIZE / sizeof(T);
    const uint64_t blocks = (uint64_t(elements) * sizeof(T)) / BLOCK_BYTE_SIZE;
    uint64_t done = 0;
    while (done < blocks) {
        const uint64_t left = blocks - done;
        const uint16_t chunk = (left > UINT16_MAX) ? static_cast<uint16_t>(UINT16_MAX) : static_cast<uint16_t>(left);
        pto_copy_gm_to_cbuf(dst + done * blockElems, src + done * blockElems, 0, 1, chunk, 0, 0);
        done += chunk;
    }
}

template <typename T, int N, int C1, int H, int W, int vN, int vC1, int vH, int vW, bool Dynamic>
AICORE inline void RunNC1HWC0Valid(__gm__ T* out, __gm__ T* src)
{
    constexpr int C0 = 32 / sizeof(T);
    constexpr uint32_t elemCount = uint32_t(N) * C1 * H * W * C0;

    using GShape = Shape<N, C1, H, W, C0>;
    using GStride = Stride<(int64_t)C1 * H * W * C0, (int64_t)H * W * C0, (int64_t)W * C0, C0, 1>;
    using GlobalData = GlobalTensor<T, GShape, GStride, Layout::NC1HWC0>;

    using TShape = ConvTileShape<N, C1, H, W, C0>;
    using VShape = std::conditional_t<
        Dynamic, ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>, ConvTileShape<vN, vC1, vH, vW, C0>>;
    using TileData = ConvTile<TileType::Mat, T, elemCount, Layout::NC1HWC0, TShape, VShape>;

    TileData tile;
    TASSIGN(tile, 0x0);
    if constexpr (Dynamic) {
        tile.SetValidShape(int64_t(vN), int64_t(vC1), int64_t(vH), int64_t(vW), int64_t(C0));
    }
    PrefillConvTile<T, TileData>(tile, src, elemCount);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
#endif

    GlobalData dstGlobal(out);
    TSTORE(dstGlobal, tile);
}

template <typename T, int N, int D, int C1, int H, int W, int vN, int vD, int vC1, int vH, int vW>
AICORE inline void RunNDC1HWC0Valid(__gm__ T* out, __gm__ T* src)
{
    constexpr int C0 = 32 / sizeof(T);
    constexpr uint32_t elemCount = uint32_t(N) * D * C1 * H * W * C0;

    using GShape = Shape<N, D, C1, H, W>;
    using GStride =
        Stride<(int64_t)D * C1 * H * W * C0, (int64_t)C1 * H * W * C0, (int64_t)H * W * C0, (int64_t)W * C0, C0>;
    using GlobalData = GlobalTensor<T, GShape, GStride, Layout::NDC1HWC0>;

    using TShape = ConvTileShape<N, D, C1, H, W, C0>;
    using VShape = ConvTileShape<vN, vD, vC1, vH, vW, C0>;
    using TileData = ConvTile<TileType::Mat, T, elemCount, Layout::NDC1HWC0, TShape, VShape>;

    TileData tile;
    TASSIGN(tile, 0x0);
    PrefillConvTile<T, TileData>(tile, src, elemCount);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
#endif

    GlobalData dstGlobal(out);
    TSTORE(dstGlobal, tile);
}

template <typename T, int D0, int D1, int v0, int v1, int v2>
AICORE inline void RunFractalZ4DValid(__gm__ T* out, __gm__ T* src)
{
    constexpr int C0 = 32 / sizeof(T);
    constexpr int N0 = 16;
    constexpr uint32_t elemCount = uint32_t(D0) * D1 * N0 * C0;

    using GShape = Shape<1, D0, D1, N0, C0>;
    using GStride = Stride<(int64_t)D0 * D1 * N0 * C0, (int64_t)D1 * N0 * C0, (int64_t)N0 * C0, C0, 1>;
    using GlobalData = GlobalTensor<T, GShape, GStride, Layout::FRACTAL_Z>;

    using TShape = ConvTileShape<D0, D1, N0, C0>;
    using VShape = ConvTileShape<v0, v1, v2, C0>;
    using TileData = ConvTile<TileType::Mat, T, elemCount, Layout::FRACTAL_Z, TShape, VShape>;

    TileData tile;
    TASSIGN(tile, 0x0);
    PrefillConvTile<T, TileData>(tile, src, elemCount);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
#endif

    GlobalData dstGlobal(out);
    TSTORE(dstGlobal, tile);
}

template <typename T, int C1, int H, int W, int N, int vC1, int vH, int vW, int vN, bool Dynamic>
AICORE inline void RunFractalZ5DValid(__gm__ T* out, __gm__ T* src)
{
    constexpr int C0 = 32 / sizeof(T);
    constexpr uint32_t elemCount = uint32_t(C1) * H * W * N * C0;

    using GShape = Shape<C1, H, W, N, C0>;
    using GStride = Stride<(int64_t)H * W * N * C0, (int64_t)W * N * C0, (int64_t)N * C0, C0, 1>;
    using GlobalData = GlobalTensor<T, GShape, GStride, Layout::FRACTAL_Z>;

    using TShape = ConvTileShape<C1, H, W, N, C0>;
    using VShape = std::conditional_t<
        Dynamic, ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>, ConvTileShape<vC1, vH, vW, vN, C0>>;
    using TileData = ConvTile<TileType::Mat, T, elemCount, Layout::FRACTAL_Z, TShape, VShape>;

    TileData tile;
    TASSIGN(tile, 0x0);
    if constexpr (Dynamic) {
        tile.SetValidShape(int64_t(vC1), int64_t(vH), int64_t(vW), int64_t(vN), int64_t(C0));
    }
    PrefillConvTile<T, TileData>(tile, src, elemCount);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
#endif

    GlobalData dstGlobal(out);
    TSTORE(dstGlobal, tile);
}

// format = 0: NC1HWC0 static, 1: NC1HWC0 dynamic, 2: NDC1HWC0 static,
//          3: FRACTAL_Z 4D static, 4: FRACTAL_Z 5D static, 5: FRACTAL_Z 5D dynamic,
//          6: NC1HWC0 static bf16
template <typename T, int format, int s0, int s1, int s2, int s3, int s4, int v0, int v1, int v2, int v3, int v4>
__global__ AICORE void TStoreConvValidKernel(__gm__ T* out, __gm__ T* src)
{
    if constexpr (format == 0) {
        RunNC1HWC0Valid<T, s0, s1, s2, s3, v0, v1, v2, v3, false>(out, src);
    } else if constexpr (format == 1) {
        RunNC1HWC0Valid<T, s0, s1, s2, s3, v0, v1, v2, v3, true>(out, src);
    } else if constexpr (format == 2) {
        RunNDC1HWC0Valid<T, s0, s1, s2, s3, s4, v0, v1, v2, v3, v4>(out, src);
    } else if constexpr (format == 3) {
        RunFractalZ4DValid<T, s0, s1, v0, v1, v2>(out, src);
    } else if constexpr (format == 4) {
        RunFractalZ5DValid<T, s0, s1, s2, s3, v0, v1, v2, v3, false>(out, src);
    } else if constexpr (format == 5) {
        RunFractalZ5DValid<T, s0, s1, s2, s3, v0, v1, v2, v3, true>(out, src);
    } else if constexpr (format == 6) {
        RunNC1HWC0Valid<T, s0, s1, s2, s3, v0, v1, v2, v3, false>(out, src);
    }
}

template <int format, typename T, int s0, int s1, int s2, int s3, int s4, int v0, int v1, int v2, int v3, int v4>
void LaunchTStoreConvValid(T* out, T* src, void* stream)
{
    if constexpr (format == 6) {
        // bf16 and fp16 both arrive as raw 16-bit words; the format selects the device type.
        TStoreConvValidKernel<bfloat16_t, format, s0, s1, s2, s3, s4, v0, v1, v2, v3, v4>
            <<<1, nullptr, stream>>>(reinterpret_cast<bfloat16_t*>(out), reinterpret_cast<bfloat16_t*>(src));
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        // The host represents fp16 as raw 16-bit words; the kernel sees the device "half" type.
        TStoreConvValidKernel<half, format, s0, s1, s2, s3, s4, v0, v1, v2, v3, v4>
            <<<1, nullptr, stream>>>(reinterpret_cast<half*>(out), reinterpret_cast<half*>(src));
    } else {
        TStoreConvValidKernel<T, format, s0, s1, s2, s3, s4, v0, v1, v2, v3, v4><<<1, nullptr, stream>>>(out, src);
    }
}

// NC1HWC0: raw (2,3,4,5,C0=16), valid (1,2,3,4) -> per-row path (vW < srcW)
template void LaunchTStoreConvValid<0, uint16_t, 2, 3, 4, 5, 1, 1, 2, 3, 4, 1>(uint16_t*, uint16_t*, void*);
// NC1HWC0: raw (2,3,4,5,C0=16), valid (2,2,3,5) -> whole-plane path (vW == srcW)
template void LaunchTStoreConvValid<0, uint16_t, 2, 3, 4, 5, 1, 2, 2, 3, 5, 1>(uint16_t*, uint16_t*, void*);
// NC1HWC0 dynamic valid: raw (2,3,4,5,C0=8), valid (1,2,3,4)
template void LaunchTStoreConvValid<1, float, 2, 3, 4, 5, 1, 1, 2, 3, 4, 1>(float*, float*, void*);
// NDC1HWC0: raw (2,3,2,4,5,C0=16), valid (1,2,1,3,4)
template void LaunchTStoreConvValid<2, uint16_t, 2, 3, 2, 4, 5, 1, 2, 1, 3, 4>(uint16_t*, uint16_t*, void*);
// FRACTAL_Z 4D: raw [8,2,16,16], valid (6,1,10,16)
template void LaunchTStoreConvValid<3, uint16_t, 8, 2, 1, 1, 1, 6, 1, 10, 1, 1>(uint16_t*, uint16_t*, void*);
// FRACTAL_Z 5D: raw (4,2,6,16,C0=32), valid (3,1,4,10)
template void LaunchTStoreConvValid<4, int8_t, 4, 2, 6, 16, 1, 3, 1, 4, 10, 1>(int8_t*, int8_t*, void*);
// FRACTAL_Z 5D dynamic valid: raw (2,3,4,8,C0=32), valid (1,2,3,6)
template void LaunchTStoreConvValid<5, int8_t, 2, 3, 4, 8, 1, 1, 2, 3, 6, 1>(int8_t*, int8_t*, void*);
// NC1HWC0 bf16, static valid: raw (2,3,4,5,C0=16), valid (1,2,3,4)
template void LaunchTStoreConvValid<6, uint16_t, 2, 3, 4, 5, 1, 1, 2, 3, 4, 1>(uint16_t*, uint16_t*, void*);
