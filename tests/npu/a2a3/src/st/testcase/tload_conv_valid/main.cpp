/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

constexpr int kValidSentinel = -7;

template <int format, typename T, int s0, int s1, int s2, int s3, int s4, int v0, int v1, int v2, int v3, int v4>
void LaunchTLoadConvValid(T* out, T* src, void* stream);

class TLoadConvValidTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    return "../" + suiteName + "." + caseName;
}

template <
    int format, typename T, int ElemCount, int s0, int s1, int s2, int s3, int s4, int v0, int v1, int v2, int v3,
    int v4>
void TestTload()
{
    size_t dataSize = static_cast<size_t>(ElemCount) * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T* srcHost = nullptr;
    T* dstHost = nullptr;
    T* goldHost = nullptr;
    aclrtMallocHost((void**)(&srcHost), dataSize);
    aclrtMallocHost((void**)(&dstHost), dataSize);
    aclrtMallocHost((void**)(&goldHost), dataSize);

    void* srcDevice = nullptr;
    void* dstDevice = nullptr;
    aclrtMalloc(&srcDevice, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dstDevice, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", dataSize, srcHost, dataSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dataSize, goldHost, dataSize);

    // Pre-fill the destination with the sentinel; the kernel reads it back to initialize the L1 tile.
    std::fill(dstHost, dstHost + ElemCount, static_cast<T>(kValidSentinel));

    aclrtMemcpy(srcDevice, dataSize, srcHost, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dstDevice, dataSize, dstHost, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTLoadConvValid<format, T, s0, s1, s2, s3, s4, v0, v1, v2, v3, v4>(
        static_cast<T*>(dstDevice), static_cast<T*>(srcDevice), stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(dstHost, dataSize, dstDevice, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dataSize);

    bool ret = ResultCmp(vector<T>(goldHost, goldHost + ElemCount), vector<T>(dstHost, dstHost + ElemCount), 0);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(goldHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    EXPECT_TRUE(ret);
}

// NC1HWC0, per-row path: raw (2,3,4,5,C0=16), valid (1,2,3,4).
TEST_F(TLoadConvValidTest, NC1HWC0_valid_slow_half) { TestTload<0, uint16_t, 1920, 2, 3, 4, 5, 1, 1, 2, 3, 4, 1>(); }

// NC1HWC0, whole-plane path: raw (2,3,4,5,C0=16), valid (2,2,3,5).
TEST_F(TLoadConvValidTest, NC1HWC0_valid_fast_half) { TestTload<0, uint16_t, 1920, 2, 3, 4, 5, 1, 2, 2, 3, 5, 1>(); }

// NC1HWC0 with runtime SetValidShape: raw (2,3,4,5,C0=8), valid (1,2,3,4).
TEST_F(TLoadConvValidTest, NC1HWC0_valid_dynamic_float) { TestTload<1, float, 960, 2, 3, 4, 5, 1, 1, 2, 3, 4, 1>(); }

// NC1HWC0 bf16, per-row path: raw (2,3,4,5,C0=16), valid (1,2,3,4).
TEST_F(TLoadConvValidTest, NC1HWC0_valid_bf16) { TestTload<6, uint16_t, 1920, 2, 3, 4, 5, 1, 1, 2, 3, 4, 1>(); }

// NDC1HWC0: raw (2,3,2,4,5,C0=16), valid (1,2,1,3,4).
TEST_F(TLoadConvValidTest, NDC1HWC0_valid_half) { TestTload<2, uint16_t, 3840, 2, 3, 2, 4, 5, 1, 2, 1, 3, 4>(); }

// FRACTAL_Z 4D flat view [1,8,2,16,16], valid (6,1,10,16).
TEST_F(TLoadConvValidTest, FracZ_4D_valid_half) { TestTload<3, uint16_t, 4096, 8, 2, 1, 1, 1, 6, 1, 10, 1, 1>(); }

// FRACTAL_Z 5D: raw (4,2,6,16,C0=32), valid (3,1,4,10).
TEST_F(TLoadConvValidTest, FracZ_5D_valid_int8) { TestTload<4, int8_t, 24576, 4, 2, 6, 16, 1, 3, 1, 4, 10, 1>(); }

// FRACTAL_Z 5D with runtime SetValidShape: raw (2,3,4,8,C0=32), valid (1,2,3,6).
TEST_F(TLoadConvValidTest, FracZ_5D_valid_dynamic_int8) { TestTload<5, int8_t, 6144, 2, 3, 4, 8, 1, 1, 2, 3, 6, 1>(); }
