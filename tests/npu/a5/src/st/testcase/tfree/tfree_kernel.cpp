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
#include <pto/common/fifo.hpp>

using namespace pto;

namespace {
constexpr int M = 16;
constexpr int K = 64;
constexpr int N = 64;
constexpr int ITERATIONS = 32;
constexpr int INPUT_ELEMENTS = 2 * (M * K + K * N);
constexpr int OUTPUT_ELEMENTS = 2 * M * N;
#if defined(__DAV_VEC__) || defined(__DAV_CUBE__)
using BothPipe = TPipe<0, Direction::DIR_BOTH, K * N * sizeof(float), 2, 2, true>;
constexpr auto NO_SPLIT = TileSplitAxis::TILE_NO_SPLIT;
#endif

#ifdef __DAV_VEC__
template <int Rows>
PTO_INTERNAL void pushInput(BothPipe& pipe, __gm__ float* input)
{
    using VecND = Tile<TileType::Vec, float, Rows, K, BLayout::RowMajor, Rows, K>;
    using VecNZ = Tile<TileType::Vec, float, Rows, K, BLayout::ColMajor, Rows, K, SLayout::RowMajor, 512>;
    using Global = GlobalTensor<float, pto::Shape<1, 1, 1, Rows, K>, pto::Stride<Rows * K, Rows * K, Rows * K, K, 1>>;
    VecND nd;
    VecNZ nz;
    TASSIGN(nd, 0x8000);
    TASSIGN(nz, 0xC000);
    Global global(input);
    TLOAD(nd, global);
    pipe_barrier(PIPE_ALL);
    TMOV(nz, nd);
    pipe_barrier(PIPE_ALL);
    TPUSH<BothPipe, VecNZ, NO_SPLIT>(pipe, nz);
    pipe_barrier(PIPE_ALL);
}

PTO_INTERNAL void storeResult(BothPipe& pipe, __gm__ float* output)
{
    using Vec = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, N>;
    using Global = GlobalTensor<float, pto::Shape<1, 1, 1, M, N>, pto::Stride<M * N, M * N, M * N, N, 1>>;
    Vec tile;
    Global global(output);
    TPOP<BothPipe, Vec, NO_SPLIT>(pipe, tile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(global, tile);
    pipe_barrier(PIPE_ALL);
    TFREE<BothPipe, NO_SPLIT>(pipe);
}
#endif
} // namespace

__global__ AICORE void runTFreeDeferred(__gm__ float* output, __gm__ float* input)
{
    // Recreate the pipe in one kernel invocation so residual hardware credits
    // persist between iterations, as they do between persistent runtime tasks.
    for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
        auto* inputs = input + iteration * INPUT_ELEMENTS;
        auto* outputs = output + iteration * OUTPUT_ELEMENTS;
#ifdef __DAV_VEC__
        if (get_subblockid() == 0) {
            BothPipe pipe(nullptr, 0, 0);
            pushInput<K>(pipe, inputs);
            pushInput<M>(pipe, inputs + K * N);
            storeResult(pipe, outputs);
            pushInput<M>(pipe, inputs + K * N + M * K);
            pushInput<K>(pipe, inputs + K * N + 2 * M * K);
            storeResult(pipe, outputs + M * N);
            pipe_barrier(PIPE_ALL);
        }
#endif
#ifdef __DAV_CUBE__
        BothPipe pipe(nullptr, 0, 0);
        using MatA = Tile<TileType::Mat, float, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
        using MatB = Tile<TileType::Mat, float, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
        MatA a;
        MatB b;
        TileLeft<float, M, K, M, K> left;
        TileRight<float, K, N, K, N> right;
        TileAcc<float, M, N, M, N> acc;
        TASSIGN(left, 0);
        TASSIGN(right, 0);
        TASSIGN(acc, 0);

        // First matmul: receive B and A before releasing either FIFO entry.
        // With the old TFREE, both frees notify the boundary at pop index 2.
        TPOP<BothPipe, MatB, NO_SPLIT>(pipe, b);
        TPOP<BothPipe, MatA, NO_SPLIT>(pipe, a);
        TMOV(left, a);
        TFREE<BothPipe, NO_SPLIT>(pipe);
        TMOV(right, b);
        TFREE<BothPipe, NO_SPLIT>(pipe);
        pipe_barrier(PIPE_ALL);
        TMATMUL(acc, left, right);
        pipe_barrier(PIPE_ALL);
        TPUSH<BothPipe, decltype(acc), NO_SPLIT>(pipe, acc);
        pipe_barrier(PIPE_ALL);

        // Second matmul: immediate frees complete the four-pop reproducer.
        TPOP<BothPipe, MatA, NO_SPLIT>(pipe, a);
        TMOV(left, a);
        TFREE<BothPipe, NO_SPLIT>(pipe);
        TPOP<BothPipe, MatB, NO_SPLIT>(pipe, b);
        TMOV(right, b);
        TFREE<BothPipe, NO_SPLIT>(pipe);
        pipe_barrier(PIPE_ALL);
        TMATMUL(acc, left, right);
        pipe_barrier(PIPE_ALL);
        TPUSH<BothPipe, decltype(acc), NO_SPLIT>(pipe, acc);
        pipe_barrier(PIPE_ALL);
#endif
    }
}

void LaunchTFreeDeferred(float* output, float* input, void* stream)
{
    runTFreeDeferred<<<1, nullptr, stream>>>(output, input);
}
