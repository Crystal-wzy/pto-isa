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
#include "test_common.h"
#include <gtest/gtest.h>
#include <fstream>

using namespace std;
using namespace PtoTestCommon;

enum class TLoadConvCase {
    NC1HWC0Static,
    NC1HWC0DynN,
    NC1HWC0DynNGlobal,
    NDC1HWC0Static,
    FRACTAL_Z4DStatic,
    FRACTAL_Z5DStatic,
    NC1HWC0ValidStatic,
    NC1HWC0ValidDynamic,
    NDC1HWC0ValidStatic,
    FRACTAL_Z4DValidStatic,
    FRACTAL_Z5DValidStatic,
    FRACTAL_Z5DValidDynamic,
};

template <typename T, TLoadConvCase caseKey, int s0, int s1, int s2, int s3, int s4>
void LaunchTLoadConv(uint8_t* out, uint8_t* src, int64_t n, uint64_t* gLog, void* stream);

template <
    typename T, TLoadConvCase caseKey, int s0, int s1, int s2, int s3, int s4, int v0, int v1, int v2, int v3, int v4>
void LaunchTLoadConvValid(uint8_t* out, uint8_t* src);

class TLoadConvTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

inline size_t GetFileSize(const std::string& filename)
{
    std::ifstream in(filename, std::ios::binary | std::ios::ate);

    return in.is_open() ? static_cast<size_t>(in.tellg()) : 0;
}

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

#define LOGSIZE 128
#define PRINTLOG 4
#define MAXBLOCK 64

template <
    typename T, TLoadConvCase caseKey, int s0, int s1, int s2, int s3, int s4, int v0 = -1, int v1 = -1, int v2 = -1,
    int v3 = -1, int v4 = -1>
void tload_test(int64_t n = 0)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    string path = GetGoldenDir();
    size_t in_byteSize = GetFileSize(path + "/input.bin");
    size_t gold_byteSize = GetFileSize(path + "/golden.bin");

    T *srcHost, *goldHost, *dstHost;
    void *srcDevice, *dstDevice, *logDevice = nullptr;

    aclrtMallocHost((void**)&srcHost, in_byteSize);
    aclrtMallocHost((void**)&goldHost, gold_byteSize);
    aclrtMallocHost((void**)&dstHost, gold_byteSize);
    aclrtMalloc((void**)&srcDevice, in_byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&dstDevice, gold_byteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(path + "/input.bin", in_byteSize, srcHost, in_byteSize);
    ReadFile(path + "/golden.bin", gold_byteSize, goldHost, gold_byteSize);
    std::fill(dstHost, dstHost + (gold_byteSize / sizeof(*dstHost)), 0);

    aclrtMemcpy(srcDevice, in_byteSize, srcHost, in_byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dstDevice, gold_byteSize, dstHost, gold_byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

#ifdef DEBUGLOG
    aclrtMalloc((void**)&logDevice, MAXBLOCK * LOGSIZE * 8, ACL_MEM_MALLOC_HUGE_FIRST);
#endif

    if constexpr (v0 < 0) {
        LaunchTLoadConv<T, caseKey, s0, s1, s2, s3, s4>(
            (uint8_t*)dstDevice, (uint8_t*)srcDevice, n, (uint64_t*)logDevice, stream);
    } else {
        (void)n;
        LaunchTLoadConvValid<T, caseKey, s0, s1, s2, s3, s4, v0, v1, v2, v3, v4>(
            (uint8_t*)dstDevice, (uint8_t*)srcDevice);
    }
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(dstHost, gold_byteSize, dstDevice, gold_byteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    int elements = gold_byteSize / sizeof(T);
    bool ret = ResultCmp(vector<T>(goldHost, goldHost + elements), vector<T>(dstHost, dstHost + elements), 0);

    aclrtFreeHost(srcHost);
    aclrtFreeHost(goldHost);
    aclrtFreeHost(dstHost);
    aclrtFree(srcDevice);
    aclrtFree(dstDevice);
    if (logDevice)
        aclrtFree(logDevice);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    EXPECT_TRUE(ret);
}

TEST_F(TLoadConvTest, case_5HD_fused_fp16) { tload_test<half, TLoadConvCase::NC1HWC0Static, 1, 2, 4, 4, 1>(); }

TEST_F(TLoadConvTest, case_5HD_cropped_fp32) { tload_test<float, TLoadConvCase::NC1HWC0Static, 1, 4, 10, 10, 1>(); }

TEST_F(TLoadConvTest, case_FracZ_4D_fp16) { tload_test<half, TLoadConvCase::FRACTAL_Z4DStatic, 16, 2, 1, 18, 1>(); }

TEST_F(TLoadConvTest, case_FracZ_5D_small_int8)
{
    tload_test<int8_t, TLoadConvCase::FRACTAL_Z5DStatic, 4, 2, 6, 16, 1>();
}

#ifdef CPU_SIM_BFLOAT_ENABLED
TEST_F(TLoadConvTest, case_5HD_fused_bf16) { tload_test<bfloat16_t, TLoadConvCase::NC1HWC0Static, 1, 2, 4, 4, 1>(); }
#endif

TEST_F(TLoadConvTest, case_5HD_unaligned_fp16) { tload_test<half, TLoadConvCase::NC1HWC0Static, 3, 3, 5, 3, 1>(); }

TEST_F(TLoadConvTest, case_5HD_unaligned_int8) { tload_test<int8_t, TLoadConvCase::NC1HWC0Static, 3, 5, 3, 3, 1>(); }

TEST_F(TLoadConvTest, case_ND_5HD_fp16) { tload_test<half, TLoadConvCase::NDC1HWC0Static, 2, 3, 2, 4, 5>(); }

TEST_F(TLoadConvTest, case_5HD_dynN_fp16) { tload_test<half, TLoadConvCase::NC1HWC0DynN, 4, 2, 4, 4, 1>(2); }

TEST_F(TLoadConvTest, case_5HD_dynNGlobal_fp32)
{
    tload_test<float, TLoadConvCase::NC1HWC0DynNGlobal, 4, 2, 3, 5, 1>(3);
}

// ValidShape: only the valid box is loaded, the rest of the tile keeps the sentinel.
TEST_F(TLoadConvTest, case_5HD_valid_static_fp16)
{
    tload_test<half, TLoadConvCase::NC1HWC0ValidStatic, 2, 3, 4, 5, 1, 1, 2, 3, 4, 1>();
}

TEST_F(TLoadConvTest, case_5HD_valid_dynamic_fp32)
{
    tload_test<float, TLoadConvCase::NC1HWC0ValidDynamic, 2, 3, 4, 5, 1, 1, 2, 3, 4, 1>();
}

TEST_F(TLoadConvTest, case_ND_5HD_valid_static_fp16)
{
    tload_test<half, TLoadConvCase::NDC1HWC0ValidStatic, 2, 3, 2, 4, 5, 1, 2, 1, 3, 4>();
}

TEST_F(TLoadConvTest, case_FracZ_4D_valid_static_fp16)
{
    tload_test<half, TLoadConvCase::FRACTAL_Z4DValidStatic, 8, 2, 1, 1, 1, 6, 1, 10, 1, 1>();
}

TEST_F(TLoadConvTest, case_FracZ_5D_valid_static_int8)
{
    tload_test<int8_t, TLoadConvCase::FRACTAL_Z5DValidStatic, 4, 2, 6, 16, 1, 3, 1, 4, 10, 1>();
}

TEST_F(TLoadConvTest, case_FracZ_5D_valid_dynamic_int8)
{
    tload_test<int8_t, TLoadConvCase::FRACTAL_Z5DValidDynamic, 2, 3, 4, 8, 1, 1, 2, 3, 6, 1>();
}

#ifdef CPU_SIM_BFLOAT_ENABLED
TEST_F(TLoadConvTest, case_5HD_valid_bf16)
{
    tload_test<bfloat16_t, TLoadConvCase::NC1HWC0ValidStatic, 1, 2, 4, 4, 1, 1, 1, 3, 3, 1>();
}
#endif
