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
#include <gtest/gtest.h>
#include <acl/acl.h>

using namespace std;
using namespace PtoTestCommon;

template <uint32_t caseId>
void launchTROWSUMTestCase(void* out, void* src, aclrtStream stream);

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

class TROWSUMTest : public testing::Test {
public:
    aclrtStream stream;
    void* dstHost;
    void* srcHost;
    void* dstDevice;
    void* srcDevice;

protected:
    void SetUp() override
    {
        aclInit(nullptr);
        aclrtSetDevice(0);
        aclrtCreateStream(&stream);
    }

    void TearDown() override
    {
        aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
    }

    template <typename T>
    bool CompareGolden(size_t dstByteSize, bool printAllEn = false)
    {
        std::vector<T> golden(dstByteSize / sizeof(T));
        std::vector<T> result(dstByteSize / sizeof(T));
        float eps = sizeof(T) == 4 ? 0.001f : 0.005f;
        ReadFile(GetGoldenDir() + "/golden.bin", dstByteSize, golden.data(), dstByteSize);
        ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, result.data(), dstByteSize);
        if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
            return ResultCmpExact(golden, result.data());
        }
        if (printAllEn) {
            return ResultCmp(golden, result, eps, 0, 1000, true);
        }
        return ResultCmp(golden, result, eps, 0, 1000, false, true);
    }

    template <
        uint32_t caseId, typename T, int row, int validRow, int srcCol, int srcValidCol, int dstCol,
        bool checkGuard = false>
    bool TRowSumTestFramework()
    {
        size_t dstByteSize = (row * dstCol + (checkGuard ? 64 : 0)) * sizeof(T);
        size_t srcByteSize = row * srcCol * sizeof(T);
        aclrtMallocHost(&dstHost, dstByteSize);
        aclrtMallocHost(&srcHost, srcByteSize);
        aclrtMalloc(&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemset(dstDevice, dstByteSize, checkGuard ? 0x5a : 0, dstByteSize);

        ReadFile(GetGoldenDir() + "/input.bin", srcByteSize, srcHost, srcByteSize);
        aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        launchTROWSUMTestCase<caseId>(dstDevice, srcDevice, stream);
        aclrtSynchronizeStream(stream);

        aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
        WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstByteSize);

        aclrtFree(dstDevice);
        aclrtFree(srcDevice);
        aclrtFreeHost(dstHost);
        aclrtFreeHost(srcHost);

        return CompareGolden<T>(dstByteSize);
    }
};

TEST_F(TROWSUMTest, case1)
{
    bool ret = TRowSumTestFramework<1, float, 127, 127, 64, 63, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case2)
{
    bool ret = TRowSumTestFramework<2, float, 63, 63, 64, 64, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case3)
{
    bool ret = TRowSumTestFramework<3, float, 31, 31, 128, 127, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case4)
{
    bool ret = TRowSumTestFramework<4, float, 15, 15, 192, 192, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case5)
{
    bool ret = TRowSumTestFramework<5, float, 7, 7, 448, 448, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case6)
{
    bool ret = TRowSumTestFramework<6, aclFloat16, 256, 256, 16, 15, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case7)
{
    bool ret = TRowSumTestFramework<7, float, 64, 64, 128, 128, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case8)
{
    bool ret = TRowSumTestFramework<8, float, 32, 32, 256, 256, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case9)
{
    bool ret = TRowSumTestFramework<9, float, 16, 16, 512, 512, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case10)
{
    bool ret = TRowSumTestFramework<10, float, 8, 8, 1024, 1024, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case11)
{
    bool ret = TRowSumTestFramework<11, int32_t, 127, 127, 64, 63, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case12)
{
    bool ret = TRowSumTestFramework<12, int32_t, 63, 63, 64, 64, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case13)
{
    bool ret = TRowSumTestFramework<13, int32_t, 31, 31, 128, 127, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case14)
{
    bool ret = TRowSumTestFramework<14, int32_t, 15, 15, 192, 192, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case15)
{
    bool ret = TRowSumTestFramework<15, int32_t, 7, 7, 448, 447, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case16)
{
    bool ret = TRowSumTestFramework<16, int16_t, 128, 128, 64, 64, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case17)
{
    bool ret = TRowSumTestFramework<17, int16_t, 64, 64, 64, 64, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case18)
{
    bool ret = TRowSumTestFramework<18, int16_t, 32, 32, 128, 128, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case19)
{
    bool ret = TRowSumTestFramework<19, int16_t, 16, 16, 192, 192, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case20)
{
    bool ret = TRowSumTestFramework<20, int16_t, 8, 8, 448, 448, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case_int64_4x16) { EXPECT_TRUE((TRowSumTestFramework<21, int64_t, 4, 4, 16, 15, 1>())); }

TEST_F(TROWSUMTest, case_uint64_4x16) { EXPECT_TRUE((TRowSumTestFramework<22, uint64_t, 4, 4, 16, 15, 1>())); }

TEST_F(TROWSUMTest, case_int64_4x64) { EXPECT_TRUE((TRowSumTestFramework<23, int64_t, 4, 4, 64, 64, 1>())); }

TEST_F(TROWSUMTest, case_uint64_4x64) { EXPECT_TRUE((TRowSumTestFramework<24, uint64_t, 4, 4, 64, 64, 1>())); }
TEST_F(TROWSUMTest, case_int64_32x32) { EXPECT_TRUE((TRowSumTestFramework<25, int64_t, 32, 32, 32, 32, 1>())); }

TEST_F(TROWSUMTest, case_int64_32x32_dndst) { EXPECT_TRUE((TRowSumTestFramework<26, int64_t, 32, 32, 32, 32, 1>())); }
TEST_F(TROWSUMTest, case_uint64_32x32_dndst) { EXPECT_TRUE((TRowSumTestFramework<27, uint64_t, 32, 32, 32, 32, 1>())); }
TEST_F(TROWSUMTest, case_int64_32x145_dndst)
{
    EXPECT_TRUE((TRowSumTestFramework<28, int64_t, 32, 32, 145, 145, 1>()));
}
TEST_F(TROWSUMTest, case_uint64_32x145_dndst)
{
    EXPECT_TRUE((TRowSumTestFramework<29, uint64_t, 32, 32, 145, 145, 1>()));
}

TEST_F(TROWSUMTest, case_int64_64x16_nddst4_guard)
{
    EXPECT_TRUE((TRowSumTestFramework<102, int64_t, 64, 64, 16, 16, 4, true>()));
}

TEST_F(TROWSUMTest, case_int64_8x16_dndst_valid5_guard)
{
    EXPECT_TRUE((TRowSumTestFramework<103, int64_t, 8, 5, 16, 16, 1, true>()));
}

TEST_F(TROWSUMTest, case_int64_8x16_dndst_valid6_guard)
{
    EXPECT_TRUE((TRowSumTestFramework<104, int64_t, 8, 6, 16, 16, 1, true>()));
}

TEST_F(TROWSUMTest, case_int64_8x16_dndst_valid7_guard)
{
    EXPECT_TRUE((TRowSumTestFramework<105, int64_t, 8, 7, 16, 16, 1, true>()));
}

TEST_F(TROWSUMTest, case_uint64_64x16_nddst4_guard)
{
    EXPECT_TRUE((TRowSumTestFramework<106, uint64_t, 64, 64, 16, 16, 4, true>()));
}

TEST_F(TROWSUMTest, case_uint64_8x16_dndst_valid5_guard)
{
    EXPECT_TRUE((TRowSumTestFramework<107, uint64_t, 8, 5, 16, 16, 1, true>()));
}

TEST_F(TROWSUMTest, case_uint64_8x16_dndst_valid6_guard)
{
    EXPECT_TRUE((TRowSumTestFramework<108, uint64_t, 8, 6, 16, 16, 1, true>()));
}

TEST_F(TROWSUMTest, case_uint64_8x16_dndst_valid7_guard)
{
    EXPECT_TRUE((TRowSumTestFramework<109, uint64_t, 8, 7, 16, 16, 1, true>()));
}

TEST_F(TROWSUMTest, case47)
{
    bool ret = TRowSumTestFramework<47, uint32_t, 127, 127, 64, 63, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case48)
{
    bool ret = TRowSumTestFramework<48, uint32_t, 63, 63, 64, 64, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case49)
{
    bool ret = TRowSumTestFramework<49, uint32_t, 31, 31, 128, 127, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case50)
{
    bool ret = TRowSumTestFramework<50, uint32_t, 15, 15, 192, 192, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case51)
{
    bool ret = TRowSumTestFramework<51, uint32_t, 7, 7, 448, 447, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case52)
{
    bool ret = TRowSumTestFramework<52, uint16_t, 128, 128, 64, 64, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case53)
{
    bool ret = TRowSumTestFramework<53, uint16_t, 64, 64, 64, 64, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case54)
{
    bool ret = TRowSumTestFramework<54, uint16_t, 32, 32, 128, 128, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case55)
{
    bool ret = TRowSumTestFramework<55, uint16_t, 16, 16, 192, 192, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case56)
{
    bool ret = TRowSumTestFramework<56, uint16_t, 8, 8, 448, 448, 1>();
    EXPECT_TRUE(ret);
}
TEST_F(TROWSUMTest, case57)
{
    bool ret = TRowSumTestFramework<57, aclFloat16, 8, 8, 448, 447, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case58)
{
    bool ret = TRowSumTestFramework<58, aclFloat16, 64, 64, 128, 128, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case_f32_1x1)
{
    bool ret = TRowSumTestFramework<59, float, 8, 8, 8, 1, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case_f32_1x8192)
{
    bool ret = TRowSumTestFramework<60, float, 1, 1, 8192, 8192, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case_f32_512x64)
{
    bool ret = TRowSumTestFramework<61, float, 256, 256, 64, 64, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case_f32_16x2048)
{
    bool ret = TRowSumTestFramework<62, float, 8, 8, 2048, 2048, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case_f32_64x64_vc1)
{
    bool ret = TRowSumTestFramework<63, float, 64, 64, 64, 1, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case_f32_63x65)
{
    bool ret = TRowSumTestFramework<64, float, 63, 63, 72, 65, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case_f32_33x129_v127)
{
    bool ret = TRowSumTestFramework<65, float, 33, 33, 136, 129, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case_f32_238x320)
{
    bool ret = TRowSumTestFramework<66, float, 238, 237, 80, 79, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case_f16_64x128)
{
    bool ret = TRowSumTestFramework<67, aclFloat16, 64, 64, 128, 128, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case_f16_32x256_v255)
{
    bool ret = TRowSumTestFramework<68, aclFloat16, 32, 32, 256, 255, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case_f16_16x512)
{
    bool ret = TRowSumTestFramework<69, aclFloat16, 16, 16, 512, 512, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case_f16_8x1024)
{
    bool ret = TRowSumTestFramework<70, aclFloat16, 8, 8, 1024, 1024, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case_f16_30x216_v24)
{
    bool ret = TRowSumTestFramework<71, aclFloat16, 30, 30, 224, 24, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case_u32_31x128_v127)
{
    bool ret = TRowSumTestFramework<80, uint32_t, 31, 31, 128, 127, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case_u16_32x128_v127)
{
    bool ret = TRowSumTestFramework<81, uint16_t, 32, 32, 128, 127, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case_f32_ubcap_106x192)
{
    bool ret = TRowSumTestFramework<82, float, 106, 105, 192, 191, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case_f32_1x16384)
{
    bool ret = TRowSumTestFramework<83, float, 1, 1, 16384, 16384, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case_f32_vrow11_30x216)
{
    bool ret = TRowSumTestFramework<84, float, 30, 11, 216, 216, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case_f32_vrow1_64x64)
{
    bool ret = TRowSumTestFramework<85, float, 64, 1, 64, 64, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case_f32_prime_251x128)
{
    bool ret = TRowSumTestFramework<86, float, 251, 250, 96, 95, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case_f16_ubcap_128x320)
{
    bool ret = TRowSumTestFramework<87, aclFloat16, 128, 127, 320, 319, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case_s16_ubcap_200x208)
{
    bool ret = TRowSumTestFramework<88, int16_t, 200, 199, 208, 207, 1>();
    EXPECT_TRUE(ret);
}
