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
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <
    typename T, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0, int gWholeShape1,
    int gWholeShape2, int gWholeShape3, int gWholeShape4>
void LaunchTStoreNz2Nz(T* out, T* src0, void* stream);

template <typename T, int M, int N, int gWholeM, int gWholeN>
void LaunchTStoreNz2Nd(T* out, T* src0, void* stream);

class TStoreMat2GMTest : public testing::Test {
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

template <
    typename DataType, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
    int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
void test_tstore_nz2nz()
{
    size_t dataSize = gWholeShape0 * gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4 * sizeof(DataType);

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    DataType *dstHost, *srcHost;
    DataType *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), dataSize);
    aclrtMallocHost((void**)(&srcHost), dataSize);

    aclrtMalloc((void**)&dstDevice, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", dataSize, srcHost, dataSize);
    aclrtMemset(dstDevice, dataSize, 0, dataSize);
    aclrtMemset(dstHost, dataSize, 0, dataSize);

    aclrtMemcpy(dstDevice, dataSize, dstHost, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcDevice, dataSize, srcHost, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTStoreNz2Nz<
        DataType, gShape0, gShape1, gShape2, gShape3, gShape4, gWholeShape0, gWholeShape1, gWholeShape2, gWholeShape3,
        gWholeShape4>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dataSize, dstDevice, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dataSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<DataType> golden(dataSize / sizeof(DataType));
    std::vector<DataType> devFinal(dataSize / sizeof(DataType));
    ReadFile(GetGoldenDir() + "/golden.bin", dataSize, golden.data(), dataSize);
    ReadFile(GetGoldenDir() + "/output.bin", dataSize, devFinal.data(), dataSize);

    bool ret = ResultCmp<DataType>(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

// NZ2ND: src = NZ fractal of the M x N matrix (input.bin, M*N elems); dst = M x N ND
// region placed in a gWholeM x gWholeN buffer (golden.bin/output.bin, gWholeM*gWholeN elems).
// gWholeN > N gives a real row-stride gap; gWholeM > M gives bottom padding (both stay 0).
template <typename DataType, int M, int N, int gWholeM, int gWholeN>
void test_tstore_nz2nd()
{
    size_t srcSize = static_cast<size_t>(M) * N * sizeof(DataType);
    size_t dstSize = static_cast<size_t>(gWholeM) * gWholeN * sizeof(DataType);

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    DataType *dstHost, *srcHost;
    DataType *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), dstSize);
    aclrtMallocHost((void**)(&srcHost), srcSize);

    aclrtMalloc((void**)&dstDevice, dstSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcSize, srcHost, srcSize);
    aclrtMemset(dstDevice, dstSize, 0, dstSize);
    aclrtMemset(dstHost, dstSize, 0, dstSize);

    aclrtMemcpy(dstDevice, dstSize, dstHost, dstSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcDevice, srcSize, srcHost, srcSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTStoreNz2Nd<DataType, M, N, gWholeM, gWholeN>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstSize, dstDevice, dstSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<DataType> golden(dstSize / sizeof(DataType));
    std::vector<DataType> devFinal(dstSize / sizeof(DataType));
    ReadFile(GetGoldenDir() + "/golden.bin", dstSize, golden.data(), dstSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstSize, devFinal.data(), dstSize);

    bool ret = ResultCmp<DataType>(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

// Mat(NZ)->GM via TSTORE(Mat), 8/16/32-bit.
TEST_F(TStoreMat2GMTest, case1_nz2nz) { test_tstore_nz2nz<aclFloat16, 1, 1, 2, 16, 16, 1, 1, 2, 16, 16>(); }

TEST_F(TStoreMat2GMTest, case2_nz2nz) { test_tstore_nz2nz<int16_t, 1, 2, 1, 16, 16, 2, 2, 2, 16, 16>(); }

TEST_F(TStoreMat2GMTest, case3_nz2nz) { test_tstore_nz2nz<int8_t, 1, 1, 2, 16, 32, 1, 2, 2, 16, 32>(); }

TEST_F(TStoreMat2GMTest, case4_nz2nz) { test_tstore_nz2nz<uint16_t, 2, 1, 2, 16, 16, 2, 2, 2, 16, 16>(); }

TEST_F(TStoreMat2GMTest, case5_nz2nz) { test_tstore_nz2nz<int8_t, 1, 2, 1, 16, 32, 2, 2, 2, 16, 32>(); }

TEST_F(TStoreMat2GMTest, case6_nz2nz) { test_tstore_nz2nz<aclFloat16, 1, 1, 1, 16, 16, 2, 1, 1, 16, 16>(); }

TEST_F(TStoreMat2GMTest, case7_nz2nz) { test_tstore_nz2nz<uint16_t, 1, 4, 1, 16, 16, 2, 4, 2, 16, 16>(); }

TEST_F(TStoreMat2GMTest, case8_nz2nz) { test_tstore_nz2nz<int16_t, 3, 1, 2, 16, 16, 4, 1, 2, 16, 16>(); }

// gShape1=17/37 + whole>tile: non-C0-aligned width + nonzero K-slice dst gap.
TEST_F(TStoreMat2GMTest, case9_nz2nz) { test_tstore_nz2nz<aclFloat16, 1, 17, 1, 16, 16, 1, 17, 2, 16, 16>(); }

TEST_F(TStoreMat2GMTest, case10_nz2nz) { test_tstore_nz2nz<int16_t, 2, 37, 2, 16, 16, 2, 37, 3, 16, 16>(); }

TEST_F(TStoreMat2GMTest, case11_nz2nz) { test_tstore_nz2nz<int8_t, 1, 17, 2, 16, 32, 1, 17, 4, 16, 32>(); }

// 32-bit.
TEST_F(TStoreMat2GMTest, case12_nz2nz) { test_tstore_nz2nz<int32_t, 1, 1, 2, 16, 8, 1, 1, 2, 16, 8>(); }

TEST_F(TStoreMat2GMTest, case13_nz2nz) { test_tstore_nz2nz<float, 2, 1, 2, 16, 8, 2, 2, 2, 16, 8>(); }

// NZ2ND: input.bin is NZ (M*N), golden.bin is ND (gWholeM*gWholeN).
TEST_F(TStoreMat2GMTest, case1_nz2nd) { test_tstore_nz2nd<aclFloat16, 16, 16, 16, 16>(); }
TEST_F(TStoreMat2GMTest, case2_nz2nd) { test_tstore_nz2nd<int16_t, 32, 32, 32, 32>(); }
TEST_F(TStoreMat2GMTest, case3_nz2nd) { test_tstore_nz2nd<int8_t, 32, 32, 32, 32>(); }
TEST_F(TStoreMat2GMTest, case4_nz2nd) { test_tstore_nz2nd<uint16_t, 16, 64, 16, 64>(); }
TEST_F(TStoreMat2GMTest, case5_nz2nd) { test_tstore_nz2nd<aclFloat16, 64, 32, 64, 32>(); }
TEST_F(TStoreMat2GMTest, case6_nz2nd) { test_tstore_nz2nd<int16_t, 16, 16, 16, 16>(); }
// Non-contiguous dst: row-stride gap (gWholeN > N) and/or bottom row padding (gWholeM > M).
TEST_F(TStoreMat2GMTest, case7_nz2nd) { test_tstore_nz2nd<aclFloat16, 32, 32, 32, 48>(); }
TEST_F(TStoreMat2GMTest, case8_nz2nd) { test_tstore_nz2nd<int16_t, 32, 32, 40, 32>(); }
TEST_F(TStoreMat2GMTest, case9_nz2nd) { test_tstore_nz2nd<int8_t, 32, 64, 32, 96>(); }
TEST_F(TStoreMat2GMTest, case10_nz2nd) { test_tstore_nz2nd<uint16_t, 16, 16, 24, 32>(); }
// 32-bit.
TEST_F(TStoreMat2GMTest, case11_nz2nd) { test_tstore_nz2nd<int32_t, 32, 32, 32, 32>(); }
TEST_F(TStoreMat2GMTest, case12_nz2nd) { test_tstore_nz2nd<float, 32, 32, 32, 48>(); }
