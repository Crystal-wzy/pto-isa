/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

namespace {

constexpr int kSortBlock = 32;

// Runs CPU-SIM TSORT32 on a kRows x kCols grid of (score, payload) pairs and returns
// the sorted (score, payload) pairs. Equal scores must keep their original input
// position (stable), matching A3 hardware VBS32; the payload is never a tie-breaker.
template <typename T, int kRows, int kCols>
std::vector<std::pair<T, uint32_t>> RunTSort32AndCollectPairs(
    const std::vector<T>& scores, const std::vector<uint32_t>& payloads)
{
    constexpr int kPairStride = (sizeof(T) == sizeof(uint32_t)) ? 2 : 4;
    constexpr int kPayloadOffset = (sizeof(T) == sizeof(uint32_t)) ? 1 : 2;
    using SrcTile = pto::Tile<pto::TileType::Vec, T, kRows, kCols, pto::BLayout::RowMajor>;
    using IdxTile = pto::Tile<pto::TileType::Vec, uint32_t, kRows, kCols, pto::BLayout::RowMajor>;
    using DstTile = pto::Tile<pto::TileType::Vec, T, kRows, kCols * kPairStride, pto::BLayout::RowMajor>;
    pto::NPU_MEMORY_INIT(pto::NPUArch::A2A3);
    SrcTile src;
    IdxTile idx;
    DstTile dst;
    TASSIGN_IMPL(src, 0);
    TASSIGN_IMPL(idx, kRows * kCols * sizeof(T));
    TASSIGN_IMPL(dst, kRows * kCols * (sizeof(T) + sizeof(uint32_t)));
    for (int i = 0; i < kRows * kCols; ++i) {
        src.data()[i] = scores[i];
        idx.data()[i] = payloads[i];
    }
    TSORT32_IMPL(dst, src, idx);
    std::vector<std::pair<T, uint32_t>> out;
    out.reserve(kRows * kCols);
    for (int i = 0; i < kRows * kCols; ++i) {
        uint32_t payload = 0;
        std::memcpy(&payload, &dst.data()[i * kPairStride + kPayloadOffset], sizeof(payload));
        out.emplace_back(dst.data()[i * kPairStride], payload);
    }
    return out;
}

// Reference: sort each independent 32-element block by score descending, stable.
template <typename T, int kRows, int kCols>
std::vector<std::pair<T, uint32_t>> ExpectedStableDescBlocks(
    const std::vector<T>& scores, const std::vector<uint32_t>& payloads)
{
    std::vector<std::pair<T, uint32_t>> expected;
    expected.reserve(kRows * kCols);
    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kCols; c += kSortBlock) {
            const int end = std::min(c + kSortBlock, kCols);
            std::vector<std::pair<T, uint32_t>> block;
            for (int j = c; j < end; ++j) {
                const int idx = r * kCols + j;
                block.emplace_back(scores[idx], payloads[idx]);
            }
            std::stable_sort(
                block.begin(), block.end(),
                [](const std::pair<T, uint32_t>& a, const std::pair<T, uint32_t>& b) { return a.first > b.first; });
            expected.insert(expected.end(), block.begin(), block.end());
        }
    }
    return expected;
}

template <typename T, int kRows, int kCols>
void ExpectTSort32MatchesStableOrder(const std::vector<T>& scores, const std::vector<uint32_t>& payloads)
{
    const auto actual = RunTSort32AndCollectPairs<T, kRows, kCols>(scores, payloads);
    const auto expected = ExpectedStableDescBlocks<T, kRows, kCols>(scores, payloads);
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(static_cast<float>(actual[i].first), static_cast<float>(expected[i].first));
        EXPECT_EQ(actual[i].second, expected[i].second);
    }
}

} // namespace

template <typename T0, typename T1, int kGRows, int kGCols, int kTRows, int kTCols, int validRow, int validCol>
void launchTSort32(T0* out, T0* src, T1* idx, aclrtStream stream);

class TSORT32Test : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <typename T0, typename T1, int kGRows, int kGCols, int kTRows, int kTCols, int validRow, int validCol>
bool TSort32Test()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t dstByteSize = kGRows * kGCols * 8;
    size_t srcByteSize = kGRows * kGCols * sizeof(T0);
    size_t idxByteSize = kGRows * kGCols * sizeof(T1);

    T0* dstHost = nullptr;
    T0* srcHost = nullptr;
    T1* idxHost = nullptr;
    T0* dstDevice = nullptr;
    T0* srcDevice = nullptr;
    T1* idxDevice = nullptr;

    aclrtMallocHost((void**)(&dstHost), dstByteSize);
    aclrtMallocHost((void**)(&srcHost), srcByteSize);
    aclrtMallocHost((void**)(&idxHost), idxByteSize);

    aclrtMalloc((void**)(&dstDevice), dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)(&srcDevice), srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)(&idxDevice), idxByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input0.bin", srcByteSize, srcHost, srcByteSize);
    ReadFile(GetGoldenDir() + "/input1.bin", idxByteSize, idxHost, idxByteSize);

    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(idxDevice, idxByteSize, idxHost, idxByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    PrintData(srcHost, kGRows * kGCols, INT16_T, kGCols);
    PrintData(idxHost, kGRows * kGCols, UINT32_T, kGCols);
    launchTSort32<T0, T1, kGRows, kGCols, kTRows, kTCols, validRow, validCol>(dstDevice, srcDevice, idxDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    PrintData(dstHost, kGRows * kGCols * 8 / sizeof(T0), INT16_T, kGCols * 8 / sizeof(T0));

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstByteSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFree(idxDevice);
    aclrtFree(dstHost);
    aclrtFree(srcHost);
    aclrtFree(idxHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T0> golden(dstByteSize / sizeof(T0));
    std::vector<T0> devFinal(dstByteSize / sizeof(T0));
    ReadFile(GetGoldenDir() + "/golden.bin", dstByteSize, golden.data(), dstByteSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, devFinal.data(), dstByteSize);
    return ResultCmp<T0>(golden, devFinal, 0.001f);
}

TEST_F(TSORT32Test, test0)
{
    bool res = TSort32Test<aclFloat16, uint32_t, 16, 16, 16, 16, 16, 16>();
    EXPECT_TRUE(res);
}

TEST_F(TSORT32Test, test1)
{
    bool res = TSort32Test<float, uint32_t, 8, 32, 8, 32, 8, 32>();
    EXPECT_TRUE(res);
}

TEST_F(TSORT32Test, test2)
{
    bool res = TSort32Test<float, uint32_t, 7, 32, 7, 32, 7, 32>();
    EXPECT_TRUE(res);
}

TEST_F(TSORT32Test, test3)
{
    bool res = TSort32Test<aclFloat16, uint32_t, 32, 16, 32, 16, 32, 16>();
    EXPECT_TRUE(res);
}

TEST_F(TSORT32Test, tieReversePayloadKeepsInputOrder)
{
    std::vector<float> scores(32, 1.0f);
    std::vector<uint32_t> payloads(32);
    for (int i = 0; i < 32; ++i) {
        payloads[i] = 31 - i;
    }
    ExpectTSort32MatchesStableOrder<float, 1, 32>(scores, payloads);
}

TEST_F(TSORT32Test, tieAscendingPayloadKeepsInputOrder)
{
    std::vector<float> scores(32, 1.0f);
    std::vector<uint32_t> payloads(32);
    for (int i = 0; i < 32; ++i) {
        payloads[i] = i;
    }
    ExpectTSort32MatchesStableOrder<float, 1, 32>(scores, payloads);
}

TEST_F(TSORT32Test, tiePermutedPayloadKeepsInputOrder)
{
    std::vector<float> scores(32, 1.0f);
    const std::vector<uint32_t> payloads = {3, 27, 14, 9,  31, 0,  22, 17, 5,  11, 30, 2,  19, 25, 7,  13,
                                            1, 28, 16, 21, 6,  24, 10, 15, 29, 4,  20, 12, 26, 8,  18, 23};
    ExpectTSort32MatchesStableOrder<float, 1, 32>(scores, payloads);
}

TEST_F(TSORT32Test, tieDuplicatePayloadKeepsInputOrder)
{
    std::vector<float> scores(32, 1.0f);
    std::vector<uint32_t> payloads(32);
    for (int i = 0; i < 32; ++i) {
        payloads[i] = (i % 2 == 0) ? 7 : 3;
    }
    ExpectTSort32MatchesStableOrder<float, 1, 32>(scores, payloads);
}

TEST_F(TSORT32Test, tieAcrossBlocksKeepsEachBlockInputOrder)
{
    std::vector<float> scores(2 * 64, 1.0f);
    std::vector<uint32_t> payloads(2 * 64);
    for (int i = 0; i < 64; ++i) {
        payloads[i] = 63 - i;
        payloads[64 + i] = (i % 5) * 1000;
    }
    ExpectTSort32MatchesStableOrder<float, 2, 64>(scores, payloads);
}

TEST_F(TSORT32Test, distinctScoresSortDescendingIgnoringPayload)
{
    std::vector<float> scores(32);
    std::vector<uint32_t> payloads(32);
    for (int i = 0; i < 32; ++i) {
        scores[i] = static_cast<float>(i % 7);
        payloads[i] = 31 - i;
    }
    ExpectTSort32MatchesStableOrder<float, 1, 32>(scores, payloads);
}

TEST_F(TSORT32Test, tieReversePayloadKeepsInputOrderHalf)
{
    std::vector<half> scores(32, static_cast<half>(1.0f));
    std::vector<uint32_t> payloads(32);
    for (int i = 0; i < 32; ++i) {
        payloads[i] = 31 - i;
    }
    ExpectTSort32MatchesStableOrder<half, 1, 32>(scores, payloads);
}
