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

#include <chrono>
#include <iostream>

using namespace std;
using namespace PtoTestCommon;

template <
    typename T, int format, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gShape5,
    int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
void LaunchTTRANSConv(T* out, T* src, void* stream);

template <
    typename T, int format, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gShape5, int gShape6,
    int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4, int gWholeShape5>
void LaunchTTRANSGroupConv(T* out, T* src, void* stream);

template <typename T, int srcN, int srcC, int srcH, int srcW, int dstC1HW, int dstN1, int dstN0, int dstC0>
void LaunchTTRANSConvNCHW2FZ(T* out, T* src, void* stream);

class TTRANSConvTest : public testing::Test {
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
    typename T, int format, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gShape5,
    int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
void test_ttrans()
{
    size_t srcFileSize = gShape0 * gShape1 * gShape2 * gShape3 * gShape4 * gShape5 * sizeof(T);
    size_t dstFileSize = srcFileSize;

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMallocHost((void**)(&srcHost), srcFileSize);

    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTTRANSConv<
        T, format, gShape0, gShape1, gShape2, gShape3, gShape4, gShape5, gWholeShape0, gWholeShape1, gWholeShape2,
        gWholeShape3, gWholeShape4>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstFileSize / sizeof(T));
    std::vector<T> result(dstFileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, result.data(), dstFileSize);

    bool ret = ResultCmp(golden, result, 0.001f);

    EXPECT_TRUE(ret);
}

template <
    typename T, int format, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gShape5, int gShape6,
    int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4, int gWholeShape5>
void test_ttrans_group()
{
    size_t srcFileSize = gShape0 * gShape1 * gShape2 * gShape3 * gShape4 * gShape5 * gShape6 * sizeof(T);
    size_t dstFileSize = srcFileSize;

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMallocHost((void**)(&srcHost), srcFileSize);

    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTTRANSGroupConv<
        T, format, gShape0, gShape1, gShape2, gShape3, gShape4, gShape5, gShape6, gWholeShape0, gWholeShape1,
        gWholeShape2, gWholeShape3, gWholeShape4, gWholeShape5>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstFileSize / sizeof(T));
    std::vector<T> result(dstFileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, result.data(), dstFileSize);

    bool ret = ResultCmp(golden, result, 0.001f);

    EXPECT_TRUE(ret);
}

template <typename T, int srcN, int srcC, int srcH, int srcW, int dstC1HW, int dstN1, int dstN0, int dstC0>
void test_ttrans_nchw2fz()
{
    size_t srcFileSize = srcN * srcC * srcH * srcW * sizeof(T);
    size_t dstFileSize = dstC1HW * dstN1 * dstN0 * dstC0 * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMallocHost((void**)(&srcHost), srcFileSize);

    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTTRANSConvNCHW2FZ<T, srcN, srcC, srcH, srcW, dstC1HW, dstN1, dstN0, dstC0>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstFileSize / sizeof(T));
    std::vector<T> result(dstFileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, result.data(), dstFileSize);

    bool ret = ResultCmp(golden, result, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TTRANSConvTest, float32_1_32_6_56) { test_ttrans<float, 0, 1, 4, 6, 56, 8, 1, 1, 1, 32, 6, 56>(); }

TEST_F(TTRANSConvTest, int32_1_8_1_8) { test_ttrans<int32_t, 0, 1, 1, 1, 8, 8, 1, 1, 1, 8, 1, 8>(); }

TEST_F(TTRANSConvTest, float32_5_57_4_16) { test_ttrans<float, 0, 5, 4, 4, 16, 16, 1, 1, 5, 57, 4, 16>(); }

TEST_F(TTRANSConvTest, half_1_30_2_16) { test_ttrans<aclFloat16, 0, 1, 2, 2, 16, 16, 1, 1, 1, 30, 2, 16>(); }

TEST_F(TTRANSConvTest, int16_7_53_6_16) { test_ttrans<int16_t, 0, 7, 4, 6, 16, 16, 1, 1, 7, 53, 6, 16>(); }

TEST_F(TTRANSConvTest, int8_3_64_2_64) { test_ttrans<int8_t, 0, 3, 2, 2, 64, 32, 1, 1, 3, 64, 2, 64>(); }

TEST_F(TTRANSConvTest, int8_1_63_2_128) { test_ttrans<int8_t, 0, 1, 2, 2, 128, 32, 1, 1, 1, 63, 2, 128>(); }

TEST_F(TTRANSConvTest, int8_5_58_2_16) { test_ttrans<int8_t, 0, 5, 2, 2, 16, 32, 1, 1, 5, 58, 2, 16>(); }

TEST_F(TTRANSConvTest, uint8_9_87_6_16) { test_ttrans<uint8_t, 0, 9, 3, 6, 16, 32, 1, 1, 9, 87, 6, 16>(); }

TEST_F(TTRANSConvTest, float32_1_32_6_48) { test_ttrans<float, 0, 1, 8, 6, 48, 4, 1, 1, 1, 32, 6, 48>(); }

TEST_F(TTRANSConvTest, uint16_1_26_2_16) { test_ttrans<uint16_t, 0, 1, 7, 2, 16, 4, 1, 1, 1, 26, 2, 16>(); }

TEST_F(TTRANSConvTest, int8_5_18_2_16) { test_ttrans<int8_t, 0, 5, 5, 2, 16, 4, 1, 1, 5, 18, 2, 16>(); }

// NC1HWC0 -> C1HWN0N1C0
TEST_F(TTRANSConvTest, float32_3_2_2_16_4) { test_ttrans<float, 1, 2, 2, 16, 2, 2, 4, 3, 2, 2, 16, 4>(); }

TEST_F(TTRANSConvTest, int32_37_2_3_10_8) { test_ttrans<int32_t, 1, 2, 3, 10, 3, 16, 8, 37, 2, 3, 10, 8>(); }

TEST_F(TTRANSConvTest, float16_7_2_1_8_16) { test_ttrans<aclFloat16, 1, 2, 1, 8, 1, 16, 16, 7, 2, 1, 8, 16>(); }

TEST_F(TTRANSConvTest, float16_7_2_1_8_4) { test_ttrans<aclFloat16, 1, 2, 1, 8, 1, 16, 4, 7, 2, 1, 8, 4>(); }

TEST_F(TTRANSConvTest, uint16_45_3_2_7_16) { test_ttrans<uint16_t, 1, 3, 2, 7, 3, 16, 16, 45, 3, 2, 7, 16>(); }

TEST_F(TTRANSConvTest, int8_25_5_1_6_32) { test_ttrans<int8_t, 1, 5, 1, 6, 2, 16, 32, 25, 5, 1, 6, 32>(); }

TEST_F(TTRANSConvTest, uint8_11_2_7_7_32) { test_ttrans<uint8_t, 1, 2, 7, 7, 1, 16, 32, 11, 2, 7, 7, 32>(); }

TEST_F(TTRANSConvTest, float32_1_1_32_6_56) { test_ttrans_group<float, 0, 1, 1, 4, 6, 56, 8, 1, 1, 1, 1, 32, 6, 56>(); }

TEST_F(TTRANSConvTest, int32_4_1_8_1_8) { test_ttrans_group<int32_t, 0, 4, 1, 1, 1, 8, 8, 1, 1, 4, 1, 8, 1, 8>(); }

TEST_F(TTRANSConvTest, float32_2_5_30_4_16)
{
    test_ttrans_group<float, 0, 2, 5, 2, 4, 16, 16, 1, 1, 2, 5, 30, 4, 16>();
}

TEST_F(TTRANSConvTest, half_1_1_30_2_16)
{
    test_ttrans_group<aclFloat16, 0, 1, 1, 2, 2, 16, 16, 1, 1, 1, 1, 30, 2, 16>();
}

TEST_F(TTRANSConvTest, float32_2_1_32_6_12) { test_ttrans_group<float, 0, 2, 1, 8, 6, 12, 4, 1, 1, 2, 1, 32, 6, 12>(); }

TEST_F(TTRANSConvTest, float32_1_3_2_2_16_4) { test_ttrans_group<float, 1, 1, 2, 2, 16, 2, 2, 4, 1, 3, 2, 2, 16, 4>(); }

TEST_F(TTRANSConvTest, float32_2_3_2_2_16_4) { test_ttrans_group<float, 1, 2, 2, 2, 16, 2, 2, 4, 2, 3, 2, 2, 16, 4>(); }

TEST_F(TTRANSConvTest, float32_2_4_2_2_16_4) { test_ttrans_group<float, 1, 2, 2, 2, 16, 2, 2, 4, 2, 4, 2, 2, 16, 4>(); }

TEST_F(TTRANSConvTest, float16_1_7_2_1_8_16)
{
    test_ttrans_group<aclFloat16, 1, 1, 2, 1, 8, 1, 16, 16, 1, 7, 2, 1, 8, 16>();
}

TEST_F(TTRANSConvTest, float16_4_7_2_1_8_4)
{
    test_ttrans_group<aclFloat16, 1, 4, 2, 1, 8, 1, 16, 4, 4, 7, 2, 1, 8, 4>();
}

TEST_F(TTRANSConvTest, float32_NC1HWC02NC1C0HW_0) { test_ttrans<float, 2, 1, 1, 2, 4, 8, 1, 1, 1, 2, 4, 8>(); }

TEST_F(TTRANSConvTest, float32_NC1HWC02NC1C0HW_1) { test_ttrans<float, 2, 2, 2, 2, 4, 8, 1, 2, 2, 2, 4, 8>(); }

TEST_F(TTRANSConvTest, float32_NC1HWC02NC1C0HW_2) { test_ttrans<float, 2, 2, 2, 3, 4, 8, 1, 2, 2, 3, 4, 8>(); }

TEST_F(TTRANSConvTest, float32_GNC1HWC02GNC1C0HW_0)
{
    test_ttrans_group<float, 2, 1, 1, 1, 2, 4, 8, 1, 1, 1, 1, 2, 4, 8>();
}

TEST_F(TTRANSConvTest, float32_GNC1HWC02GNC1C0HW_1)
{
    test_ttrans_group<float, 2, 2, 2, 2, 2, 4, 8, 1, 2, 2, 2, 2, 4, 8>();
}

TEST_F(TTRANSConvTest, float32_GNC1HWC02GNC1C0HW_2)
{
    test_ttrans_group<float, 2, 2, 2, 2, 3, 4, 8, 1, 2, 2, 2, 3, 4, 8>();
}

// NCHW -> FRACTAL_Z
TEST_F(TTRANSConvTest, half_7_32_2_8_FZ) { test_ttrans_nchw2fz<aclFloat16, 7, 32, 2, 8, 32, 1, 16, 16>(); }

TEST_F(TTRANSConvTest, half_4_17_4_1_FZ) { test_ttrans_nchw2fz<aclFloat16, 4, 17, 4, 1, 32, 1, 16, 16>(); }

TEST_F(TTRANSConvTest, float32_5_4_3_8_FZ) { test_ttrans_nchw2fz<float, 5, 4, 3, 8, 24, 1, 16, 8>(); }

TEST_F(TTRANSConvTest, int8_4_32_3_7_FZ) { test_ttrans_nchw2fz<int8_t, 4, 32, 3, 7, 32, 1, 16, 32>(); }

/*=========================================================================================*/
/* Valid shape 用例：与 CPU 侧同参数同语义                                                  */
/*   host 预置哨兵 -> kernel TLOAD 载入 -> TTRANS 只改 dst valid 盒 -> 盒外必须原样保留。     */
/*=========================================================================================*/

void LaunchTTRANSConvValid_NCHW2NC1HWC0(
    float* out, float* src, void* stream, const int64_t (&srcValid)[4], const int64_t (&dstValid)[5]);
void LaunchTTRANSConvValid_NC1HWC02NCHW(
    float* out, float* src, void* stream, const int64_t (&srcValid)[5], const int64_t (&dstValid)[4]);
void LaunchTTRANSConvValid_NC1HWC02C1HWN1N0C0(
    float* out, float* src, void* stream, const int64_t (&srcValid)[5], const int64_t (&dstValid)[4]);
void LaunchTTRANSConvValid_GNCHW2GNC1HWC0(
    float* out, float* src, void* stream, const int64_t (&srcValid)[5], const int64_t (&dstValid)[6]);
void LaunchTTRANSConvValid_GNC1HWC02C1HWN1N0C0(
    float* out, float* src, void* stream, const int64_t (&srcValid)[6], const int64_t (&dstValid)[4]);
void LaunchTTRANSConvValid_NCDHW2C1DHWN1N0C0(
    float* out, float* src, void* stream, const int64_t (&srcValid)[5], const int64_t (&dstValid)[4]);
void LaunchTTRANSConvValid_NCHW2C1HWN1N0C0(
    float* out, float* src, void* stream, const int64_t (&srcValid)[4], const int64_t (&dstValid)[4]);
void LaunchTTRANSConvValid_NCHW2C1DHWN1N0C0(
    float* out, float* src, void* stream, const int64_t (&srcValid)[4], const int64_t (&dstValid)[4]);

// valid 用例驱动：分配 -> 预置哨兵 -> 启动 -> 回读 -> 与 golden 逐元素比对
template <typename T, typename LaunchFn>
void test_ttrans_valid(size_t srcFileSize, size_t dstFileSize, LaunchFn launch)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMallocHost((void**)(&srcHost), srcFileSize);

    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);

    // 输出侧预置哨兵：valid 之外的字节必须原样保留
    std::fill(dstHost, dstHost + dstFileSize / sizeof(T), static_cast<T>(-12345));

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dstDevice, dstFileSize, dstHost, dstFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launch(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstFileSize / sizeof(T));
    std::vector<T> result(dstFileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, result.data(), dstFileSize);

    bool ret = ResultCmp(golden, result, 0.001f);

    EXPECT_TRUE(ret);
}

// 场景 NCHW2NC1HWC0：src raw (1, 16, 8, 16) valid (1, 16, 4, 16)；dst raw (2, 4, 8, 32, 16) valid (1, 2, 4, 16, 8)
TEST_F(TTRANSConvTest, valid_NCHW2NC1HWC0_dst_all_shrunk)
{
    constexpr size_t srcFileSize = (size_t)1 * 16 * 8 * 16 * sizeof(float);
    constexpr size_t dstFileSize = (size_t)2 * 4 * 8 * 32 * 16 * sizeof(float);
    test_ttrans_valid<float>(srcFileSize, dstFileSize, [](float* dst, float* src, void* stream) {
        const int64_t srcValid[4] = {1, 16, 4, 16};
        const int64_t dstValid[5] = {1, 2, 4, 16, 8};
        LaunchTTRANSConvValid_NCHW2NC1HWC0(dst, src, stream, srcValid, dstValid);
    });
}

// 场景 NC1HWC02NCHW：src raw (3, 5, 7, 9, 16) valid (2, 3, 4, 5, 8)；dst raw (3, 32, 7, 9) valid (2, 24, 4, 5)
TEST_F(TTRANSConvTest, valid_NC1HWC02NCHW_all_dims_differ)
{
    constexpr size_t srcFileSize = (size_t)3 * 5 * 7 * 9 * 16 * sizeof(float);
    constexpr size_t dstFileSize = (size_t)3 * 32 * 7 * 9 * sizeof(float);
    test_ttrans_valid<float>(srcFileSize, dstFileSize, [](float* dst, float* src, void* stream) {
        const int64_t srcValid[5] = {2, 3, 4, 5, 8};
        const int64_t dstValid[4] = {2, 24, 4, 5};
        LaunchTTRANSConvValid_NC1HWC02NCHW(dst, src, stream, srcValid, dstValid);
    });
}

// 场景 NC1HWC02C1HWN1N0C0：src raw (6, 3, 4, 5, 16) valid (5, 2, 2, 3, 8)；dst raw (16, 2, 24, 16) valid (12, 1, 16, 8)
TEST_F(TTRANSConvTest, valid_NC1HWC02C1HWN1N0C0_all_dims_differ)
{
    constexpr size_t srcFileSize = (size_t)6 * 3 * 4 * 5 * 16 * sizeof(float);
    constexpr size_t dstFileSize = (size_t)16 * 2 * 24 * 16 * sizeof(float);
    test_ttrans_valid<float>(srcFileSize, dstFileSize, [](float* dst, float* src, void* stream) {
        const int64_t srcValid[5] = {5, 2, 2, 3, 8};
        const int64_t dstValid[4] = {12, 1, 16, 8};
        LaunchTTRANSConvValid_NC1HWC02C1HWN1N0C0(dst, src, stream, srcValid, dstValid);
    });
}

// 场景 GNCHW2GNC1HWC0：src raw (3, 4, 10, 5, 6) valid (2, 3, 8, 4, 5)；dst raw (3, 4, 3, 5, 6, 16) valid (2, 3, 1, 4,
// 5, 8)
TEST_F(TTRANSConvTest, valid_GNCHW2GNC1HWC0_all_dims_differ)
{
    constexpr size_t srcFileSize = (size_t)3 * 4 * 10 * 5 * 6 * sizeof(float);
    constexpr size_t dstFileSize = (size_t)3 * 4 * 3 * 5 * 6 * 16 * sizeof(float);
    test_ttrans_valid<float>(srcFileSize, dstFileSize, [](float* dst, float* src, void* stream) {
        const int64_t srcValid[5] = {2, 3, 8, 4, 5};
        const int64_t dstValid[6] = {2, 3, 1, 4, 5, 8};
        LaunchTTRANSConvValid_GNCHW2GNC1HWC0(dst, src, stream, srcValid, dstValid);
    });
}

// 场景 GNC1HWC02C1HWN1N0C0：src raw (3, 4, 3, 4, 5, 16) valid (2, 3, 1, 3, 4, 8)；dst raw (28, 2, 24, 16) valid (24, 1,
// 16, 8)
TEST_F(TTRANSConvTest, valid_GNC1HWC02C1HWN1N0C0_all_dims_differ)
{
    constexpr size_t srcFileSize = (size_t)3 * 4 * 3 * 4 * 5 * 16 * sizeof(float);
    constexpr size_t dstFileSize = (size_t)28 * 2 * 24 * 16 * sizeof(float);
    test_ttrans_valid<float>(srcFileSize, dstFileSize, [](float* dst, float* src, void* stream) {
        const int64_t srcValid[6] = {2, 3, 1, 3, 4, 8};
        const int64_t dstValid[4] = {24, 1, 16, 8};
        LaunchTTRANSConvValid_GNC1HWC02C1HWN1N0C0(dst, src, stream, srcValid, dstValid);
    });
}

// 场景 NCDHW2C1DHWN1N0C0：src raw (3, 10, 3, 4, 5) valid (2, 8, 2, 3, 4)；dst raw (32, 2, 24, 16) valid (24, 1, 16, 8)
TEST_F(TTRANSConvTest, valid_NCDHW2C1DHWN1N0C0_all_dims_differ)
{
    constexpr size_t srcFileSize = (size_t)3 * 10 * 3 * 4 * 5 * sizeof(float);
    constexpr size_t dstFileSize = (size_t)32 * 2 * 24 * 16 * sizeof(float);
    test_ttrans_valid<float>(srcFileSize, dstFileSize, [](float* dst, float* src, void* stream) {
        const int64_t srcValid[5] = {2, 8, 2, 3, 4};
        const int64_t dstValid[4] = {24, 1, 16, 8};
        LaunchTTRANSConvValid_NCDHW2C1DHWN1N0C0(dst, src, stream, srcValid, dstValid);
    });
}

// 场景 NCHW2C1HWN1N0C0：src raw (8, 10, 4, 6) valid (6, 8, 3, 4)；dst raw (16, 2, 24, 16) valid (12, 1, 16, 8)
TEST_F(TTRANSConvTest, valid_NCHW2C1HWN1N0C0_all_dims_differ)
{
    constexpr size_t srcFileSize = (size_t)8 * 10 * 4 * 6 * sizeof(float);
    constexpr size_t dstFileSize = (size_t)16 * 2 * 24 * 16 * sizeof(float);
    test_ttrans_valid<float>(srcFileSize, dstFileSize, [](float* dst, float* src, void* stream) {
        const int64_t srcValid[4] = {6, 8, 3, 4};
        const int64_t dstValid[4] = {12, 1, 16, 8};
        LaunchTTRANSConvValid_NCHW2C1HWN1N0C0(dst, src, stream, srcValid, dstValid);
    });
}

// 场景 NCHW2C1DHWN1N0C0：src raw (6, 18, 6, 5) valid (5, 16, 4, 3)；dst raw (32, 2, 24, 16) valid (24, 1, 16, 8)
TEST_F(TTRANSConvTest, valid_NCHW2C1DHWN1N0C0_all_dims_differ)
{
    constexpr size_t srcFileSize = (size_t)6 * 18 * 6 * 5 * sizeof(float);
    constexpr size_t dstFileSize = (size_t)32 * 2 * 24 * 16 * sizeof(float);
    test_ttrans_valid<float>(srcFileSize, dstFileSize, [](float* dst, float* src, void* stream) {
        const int64_t srcValid[4] = {5, 16, 4, 3};
        const int64_t dstValid[4] = {24, 1, 16, 8};
        LaunchTTRANSConvValid_NCHW2C1DHWN1N0C0(dst, src, stream, srcValid, dstValid);
    });
}

/*=========================================================================================*/
/* 性能基准：同一 kernel 内重复 iters 次 TTRANS，host 侧计时（含一次预热）。                  */
/*   收缩 valid 与 legacy（valid==raw）各测一组，用于对比合并前后的路径代价。                 */
/*=========================================================================================*/

void LaunchTTRANSConvBench_NCHW2NC1HWC0(float* out, float* src, void* stream, int64_t iters);
void LaunchTTRANSConvBenchLegacy_NCHW2NC1HWC0(float* out, float* src, void* stream, int64_t iters);
void LaunchTTRANSConvBench_NC1HWC02NCHW(float* out, float* src, void* stream, int64_t iters);
void LaunchTTRANSConvBench_NC1HWC02FZ(float* out, float* src, void* stream, int64_t iters);
void LaunchTTRANSConvBench_NCHW2FZ(float* out, float* src, void* stream, int64_t iters);

template <typename T, typename LaunchFn>
void bench_ttrans_valid(size_t srcFileSize, size_t dstFileSize, int64_t iters, const char* name, LaunchFn launch)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMallocHost((void**)(&srcHost), srcFileSize);
    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // 输入取 gen_data 生成的 input.bin，输出侧预置哨兵：基准用例同时校验正确性
    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);
    std::fill(dstHost, dstHost + dstFileSize / sizeof(T), static_cast<T>(-12345));

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dstDevice, dstFileSize, dstHost, dstFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launch(dstDevice, srcDevice, stream, 1); // 预热
    aclrtSynchronizeStream(stream);

    auto t0 = std::chrono::steady_clock::now();
    launch(dstDevice, srcDevice, stream, iters);
    aclrtSynchronizeStream(stream);
    auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "[BENCH] " << name << ": " << iters << " iters, " << ms << " ms, " << (ms * 1000.0 / iters)
              << " us/iter" << std::endl;
    EXPECT_GT(ms, 0.0);

    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstFileSize / sizeof(T));
    std::vector<T> result(dstFileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, result.data(), dstFileSize);
    EXPECT_TRUE(ResultCmp(golden, result, 0.001f));
}

// B1: NCHW -> NC1HWC0，src raw (8,16,8,16) valid (8,16,6,16)；dst raw (8,2,8,16,8) valid (8,2,6,16,8)
TEST_F(TTRANSConvTest, bench_NCHW2NC1HWC0)
{
    constexpr size_t srcFileSize = (size_t)8 * 16 * 8 * 16 * sizeof(float);
    constexpr size_t dstFileSize = (size_t)8 * 2 * 8 * 16 * 8 * sizeof(float);
    bench_ttrans_valid<float>(
        srcFileSize, dstFileSize, 200, "NCHW2NC1HWC0", [](float* dst, float* src, void* stream, int64_t iters) {
            LaunchTTRANSConvBench_NCHW2NC1HWC0(dst, src, stream, iters);
        });
}

// B5: NCHW -> NC1HWC0，valid == raw（legacy 几何，用来量 legacy 路径合并前后的性能）
TEST_F(TTRANSConvTest, bench_NCHW2NC1HWC0_legacy)
{
    constexpr size_t srcFileSize = (size_t)8 * 16 * 8 * 16 * sizeof(float);
    constexpr size_t dstFileSize = (size_t)8 * 2 * 8 * 16 * 8 * sizeof(float);
    bench_ttrans_valid<float>(
        srcFileSize, dstFileSize, 200, "NCHW2NC1HWC0_legacy", [](float* dst, float* src, void* stream, int64_t iters) {
            LaunchTTRANSConvBenchLegacy_NCHW2NC1HWC0(dst, src, stream, iters);
        });
}

// B2: NC1HWC0 -> NCHW，src raw (8,2,8,16,8) valid (8,2,6,16,8)；dst raw (8,16,8,16) valid (8,16,6,16)
TEST_F(TTRANSConvTest, bench_NC1HWC02NCHW)
{
    constexpr size_t srcFileSize = (size_t)8 * 2 * 8 * 16 * 8 * sizeof(float);
    constexpr size_t dstFileSize = (size_t)8 * 16 * 8 * 16 * sizeof(float);
    bench_ttrans_valid<float>(
        srcFileSize, dstFileSize, 200, "NC1HWC02NCHW", [](float* dst, float* src, void* stream, int64_t iters) {
            LaunchTTRANSConvBench_NC1HWC02NCHW(dst, src, stream, iters);
        });
}

// B3: NC1HWC0 -> FRACTAL_Z，src raw (16,2,8,10,8) valid (13,2,6,9,8)；dst raw (160,1,16,8) valid (108,1,16,8)
TEST_F(TTRANSConvTest, bench_NC1HWC02FZ)
{
    constexpr size_t srcFileSize = (size_t)16 * 2 * 8 * 10 * 8 * sizeof(float);
    constexpr size_t dstFileSize = (size_t)160 * 1 * 16 * 8 * sizeof(float);
    bench_ttrans_valid<float>(
        srcFileSize, dstFileSize, 200, "NC1HWC02FZ", [](float* dst, float* src, void* stream, int64_t iters) {
            LaunchTTRANSConvBench_NC1HWC02FZ(dst, src, stream, iters);
        });
}

// B4: NCHW -> FRACTAL_Z，src raw (16,8,8,16) valid (13,8,6,16)；dst raw (96,1,16,8) valid (96,1,16,8)
TEST_F(TTRANSConvTest, bench_NCHW2FZ)
{
    constexpr size_t srcFileSize = (size_t)16 * 8 * 8 * 16 * sizeof(float);
    constexpr size_t dstFileSize = (size_t)96 * 1 * 16 * 8 * sizeof(float);
    bench_ttrans_valid<float>(
        srcFileSize, dstFileSize, 200, "NCHW2FZ", [](float* dst, float* src, void* stream, int64_t iters) {
            LaunchTTRANSConvBench_NCHW2FZ(dst, src, stream, iters);
        });
}
