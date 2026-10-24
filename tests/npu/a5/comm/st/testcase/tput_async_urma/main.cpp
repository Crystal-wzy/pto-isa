/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

#include "tput_async_urma_kernel.h"
#include "../comm_mpi.h"

// ============================================================================
// Basic correctness (URMA true async PUT on A5 3510)
// ============================================================================
TEST(TPutAsyncUrma, Vec_FloatSmall)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunPutAsyncUrmaRootPut<float, 256>(2, 2, 0, 0)));
}
TEST(TPutAsyncUrma, Vec_Int32Large)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunPutAsyncUrmaRootPut<int32_t, 4096>(2, 2, 0, 0)));
}
TEST(TPutAsyncUrma, Vec_Uint8Small)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunPutAsyncUrmaRootPut<uint8_t, 512>(2, 2, 0, 0)));
}

// ============================================================================
// Boundary scenarios
// ============================================================================
TEST(TPutAsyncUrma, Vec_Uint8_SingleChunk)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunPutAsyncUrmaRootPut<uint8_t, 64>(2, 2, 0, 0)));
}
TEST(TPutAsyncUrma, Vec_Float_ExactChunk)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunPutAsyncUrmaRootPut<float, 64>(2, 2, 0, 0)));
}

// ============================================================================
// Multi-rank broadcast
// ============================================================================
TEST(TPutAsyncUrma, Vec_FloatSmall_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE((RunPutAsyncUrmaRootPut<float, 256>(4, 4, 0, 0)));
}
// 8 ranks is not 4 ranks scaled up: the root opens a queue on each of the card's
// seven layer-0 ports, so this is the first point where picking the wrong port
// for a peer is visible.
TEST(TPutAsyncUrma, Vec_FloatSmall_8Ranks)
{
    SKIP_IF_RANKS_LT(8);
    ASSERT_TRUE((RunPutAsyncUrmaRootPut<float, 256>(8, 8, 0, 0)));
}

// ============================================================================
// Large MR boundary tests
// ============================================================================
TEST(TPutAsyncUrma, Vec_Float_MR_6MB)
{
    SKIP_IF_RANKS_LT(2);
    // 512K floats → commBytesNeeded ≈ 4MB+256B, exact alloc (no 2MB round-up)
    ASSERT_TRUE((RunPutAsyncUrmaRootPut<float, 524288>(2, 2, 0, 0)));
}
TEST(TPutAsyncUrma, Vec_Int32_MR_Over512MB)
{
    SKIP_IF_RANKS_LT(2);
    // 64M int32 → commBytesNeeded ≈ 512MB+256B, exact alloc
    ASSERT_TRUE((RunPutAsyncUrmaRootPut<int32_t, 67108864>(2, 2, 0, 0)));
}

// ============================================================================
// >256MB single transfer: one TPUT_ASYNC whose payload exceeds a single WQE's
// capacity must be auto-split by UrmaPostSend into several in-range WQEs and
// drained by one event.Wait. End-to-end value check proves the second (1MB)
// chunk lands at the right offset.
// ============================================================================
TEST(TPutAsyncUrma, Vec_Int32_Over256MB_Chunked)
{
    SKIP_IF_RANKS_LT(2);
    // 67,371,008 int32 = 257MB, past what one WQE carries → split into two.
    ASSERT_TRUE((RunPutAsyncUrmaRootPut<int32_t, 67371008>(2, 2, 0, 0)));
}

// ============================================================================
// SharedPool (Init layout SHARED_POOL; filter: TPutAsyncUrma.Pool_*). Template args
// are <T, count, nAiv, jettiesPerCore>; the pool holds nAiv * jettiesPerCore jetties
// and no test names one.
// ============================================================================
TEST(TPutAsyncUrma, Pool_OneJettyManyPeers_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE((RunPutAsyncUrmaPool<float, 256, 1, 1>(4, 4, 0, 0)));
}
TEST(TPutAsyncUrma, Pool_MultiAivDisjointRuns_2Ranks)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunPutAsyncUrmaPool<float, 256, 2, 1>(2, 2, 0, 0)));
}
TEST(TPutAsyncUrma, Pool_MultiAivDisjointRuns_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE((RunPutAsyncUrmaPool<float, 256, 2, 1>(4, 4, 0, 0)));
}
// Default-sized pool: Init asks for SHARED_POOL but omits aivCount, so the pool
// follows ACL_DEV_ATTR_VECTOR_CORE_NUM with one jetty per AIV. Two blocks then
// put on the first two runs.
TEST(TPutAsyncUrma, Pool_AutoAivCountDefault)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunPutAsyncUrmaPoolAutoAivCount(2, 2, 0, 0));
}
// 1KB / 1MB against a 4-jetty run: both still fit in one WQE, so the put stays
// on the run's first jetty. The 1MB case is the old "would have sliced at 64KB"
// regression.
TEST(TPutAsyncUrma, Pool_WideRunShortPayloadStaysOnOneJetty_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE((RunPutAsyncUrmaPool<float, 256, 2, 4>(4, 4, 0, 0)));
}
TEST(TPutAsyncUrma, Pool_WideRun1MBStaysOnOneJetty_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE((RunPutAsyncUrmaPool<int32_t, 262144, 1, 4>(4, 4, 0, 0)));
}
TEST(TPutAsyncUrma, Pool_MultiAivWideRun1MBStaysOnOneJetty_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE((RunPutAsyncUrmaPool<int32_t, 262144, 2, 4>(4, 4, 0, 0)));
}
// 512MB against a 2-jetty run: more than one WQE's worth, so the put is sliced
// and the slices round-robin the AIV's own jetties. 2 ranks keep the per-rank MR
// at 1GB (1 AIV) / 2GB (2 AIV) instead of multiplying by 4 peers.
TEST(TPutAsyncUrma, Pool_OneAivSlicesAcrossItsRun)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunPutAsyncUrmaPool<int32_t, 134217728, 1, 2>(2, 2, 0, 0)));
}
TEST(TPutAsyncUrma, Pool_MultiAivSliceAcrossOwnRuns)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunPutAsyncUrmaPool<int32_t, 134217728, 2, 2>(2, 2, 0, 0)));
}

// ============================================================================
// PerPeer with an explicit AIV width (filter: TPutAsyncUrma.PerPeer_*). Template
// args are <T, count, nAiv, jettiesPerCore>. The RootPut tests above run one
// block at the default width, so PerPeer's private-run rule -- AIV b owns
// [b * jettiesPerCore, (b + 1) * jettiesPerCore) on every peer -- is only
// checked here. queues per peer = nAiv * jettiesPerCore, and the channels a
// rank opens is that times (nRanks - 1).
// ============================================================================
TEST(TPutAsyncUrma, PerPeer_MultiAivDisjointRuns_2Ranks)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunPutAsyncUrmaPerPeer<float, 256, 2, 1>(2, 2, 0, 0)));
}
TEST(TPutAsyncUrma, PerPeer_MultiAivDisjointRuns_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE((RunPutAsyncUrmaPerPeer<float, 256, 2, 1>(4, 4, 0, 0)));
}
TEST(TPutAsyncUrma, PerPeer_MultiAivDisjointRuns_8Ranks)
{
    SKIP_IF_RANKS_LT(8);
    ASSERT_TRUE((RunPutAsyncUrmaPerPeer<float, 256, 2, 1>(8, 8, 0, 0)));
}
// 2 AIVs x 4 queues: 8 queues per peer, so 8 ranks open 56 channels and each
// block has three queues of its own plus four it must leave alone.
TEST(TPutAsyncUrma, PerPeer_WideRun_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE((RunPutAsyncUrmaPerPeer<float, 256, 2, 4>(4, 4, 0, 0)));
}
TEST(TPutAsyncUrma, PerPeer_WideRun_8Ranks)
{
    SKIP_IF_RANKS_LT(8);
    ASSERT_TRUE((RunPutAsyncUrmaPerPeer<float, 256, 2, 4>(8, 8, 0, 0)));
}
// 1MB still fits one WQE, so it stays on the run's first queue instead of
// spreading over the other three.
TEST(TPutAsyncUrma, PerPeer_WideRun1MBStaysOnOneQueue)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunPutAsyncUrmaPerPeer<int32_t, 262144, 1, 4>(2, 2, 0, 0)));
}

// ============================================================================
// All ranks send at once (filter: TPutAsyncUrma.AllToAll_*). Every test above
// has a single sender, so both directions of a link are never loaded together.
// Each sender lands in its own slot and the slot nobody writes is checked to be
// untouched, so a completion credited to the wrong queue shows up as data.
// ============================================================================
TEST(TPutAsyncUrma, AllToAll_2Ranks)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunPutAsyncUrmaAllToAll<float, 256>(2, 2, 0, 0)));
}
TEST(TPutAsyncUrma, AllToAll_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE((RunPutAsyncUrmaAllToAll<float, 256>(4, 4, 0, 0)));
}
TEST(TPutAsyncUrma, AllToAll_8Ranks)
{
    SKIP_IF_RANKS_LT(8);
    ASSERT_TRUE((RunPutAsyncUrmaAllToAll<float, 256>(8, 8, 0, 0)));
}
// 256KB per link: long enough that the seven concurrent senders overlap on the
// port rather than finishing one burst each.
TEST(TPutAsyncUrma, AllToAll_256KB_8Ranks)
{
    SKIP_IF_RANKS_LT(8);
    ASSERT_TRUE((RunPutAsyncUrmaAllToAll<int32_t, 65536>(8, 8, 0, 0)));
}

int main(int argc, char** argv)
{
    CommMpiInit(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}
