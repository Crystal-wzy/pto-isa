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
#include <acl/acl.h>
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <uint32_t caseId>
void launchTROWMAXTestCase(void* out, void* src, aclrtStream stream);

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

class TROWMAXTest : public testing::Test {
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
    bool TRowMaxTestFramework()
    {
        constexpr size_t guardElements = checkGuard ? 64 : 0;
        size_t dstByteSize = (row * dstCol + guardElements) * sizeof(T);
        size_t srcByteSize = row * srcCol * sizeof(T);
        aclrtMallocHost(&dstHost, dstByteSize);
        aclrtMallocHost(&srcHost, srcByteSize);
        aclrtMalloc(&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemset(dstDevice, dstByteSize, 0, dstByteSize);

        ReadFile(GetGoldenDir() + "/input.bin", srcByteSize, srcHost, srcByteSize);
        aclrtMemset(dstHost, dstByteSize, checkGuard ? 0x5a : 0, dstByteSize);

        aclrtMemcpy(dstDevice, dstByteSize, dstHost, dstByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        launchTROWMAXTestCase<caseId>(dstDevice, srcDevice, stream);
        aclrtSynchronizeStream(stream);

        aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
        size_t resultByteSize = checkGuard ? dstByteSize : validRow * dstCol * sizeof(T);
        WriteFile(GetGoldenDir() + "/output.bin", dstHost, resultByteSize);

        aclrtFree(dstDevice);
        aclrtFree(srcDevice);
        aclrtFreeHost(dstHost);
        aclrtFreeHost(srcHost);

        return CompareGolden<T>(dstByteSize);
    }
};

TEST_F(TROWMAXTest, case1)
{
    bool ret = TRowMaxTestFramework<1, float, 127, 127, 64, 63, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case2)
{
    bool ret = TRowMaxTestFramework<2, float, 63, 63, 64, 64, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case3)
{
    bool ret = TRowMaxTestFramework<3, float, 31, 31, 128, 127, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case4)
{
    bool ret = TRowMaxTestFramework<4, float, 15, 15, 192, 192, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case5)
{
    bool ret = TRowMaxTestFramework<5, float, 7, 7, 448, 447, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case6)
{
    bool ret = TRowMaxTestFramework<6, aclFloat16, 256, 256, 16, 15, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case7)
{
    bool ret = TRowMaxTestFramework<7, float, 30, 30, 216, 216, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case8)
{
    bool ret = TRowMaxTestFramework<8, float, 30, 30, 216, 24, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case9)
{
    bool ret = TRowMaxTestFramework<9, float, 30, 11, 216, 216, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case10)
{
    bool ret = TRowMaxTestFramework<10, float, 30, 11, 216, 24, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case11)
{
    bool ret = TRowMaxTestFramework<11, float, 238, 238, 40, 40, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case12)
{
    bool ret = TRowMaxTestFramework<12, float, 238, 238, 40, 16, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case13)
{
    bool ret = TRowMaxTestFramework<13, float, 238, 121, 40, 40, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case14)
{
    bool ret = TRowMaxTestFramework<14, float, 238, 121, 40, 16, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case15)
{
    bool ret = TRowMaxTestFramework<15, float, 64, 64, 128, 128, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case16)
{
    bool ret = TRowMaxTestFramework<16, float, 32, 32, 256, 256, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case17)
{
    bool ret = TRowMaxTestFramework<17, float, 16, 16, 512, 512, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case18)
{
    bool ret = TRowMaxTestFramework<18, float, 8, 8, 1024, 1024, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case19)
{
    bool ret = TRowMaxTestFramework<19, int32_t, 127, 127, 64, 63, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case20)
{
    bool ret = TRowMaxTestFramework<20, int32_t, 63, 63, 64, 64, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case21)
{
    bool ret = TRowMaxTestFramework<21, int32_t, 31, 31, 128, 127, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case22)
{
    bool ret = TRowMaxTestFramework<22, int32_t, 15, 15, 192, 192, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case23)
{
    bool ret = TRowMaxTestFramework<23, int32_t, 7, 7, 448, 447, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case24)
{
    bool ret = TRowMaxTestFramework<24, int16_t, 128, 128, 64, 64, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case25)
{
    bool ret = TRowMaxTestFramework<25, int16_t, 64, 64, 64, 64, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case26)
{
    bool ret = TRowMaxTestFramework<26, int16_t, 32, 32, 128, 128, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case27)
{
    bool ret = TRowMaxTestFramework<27, int16_t, 16, 16, 192, 192, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case28)
{
    bool ret = TRowMaxTestFramework<28, int16_t, 8, 8, 448, 448, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_int64_4x16) { EXPECT_TRUE((TRowMaxTestFramework<29, int64_t, 4, 4, 16, 15, 1>())); }

TEST_F(TROWMAXTest, case_uint64_4x16) { EXPECT_TRUE((TRowMaxTestFramework<30, uint64_t, 4, 4, 16, 15, 1>())); }

TEST_F(TROWMAXTest, case_int64_4x64) { EXPECT_TRUE((TRowMaxTestFramework<31, int64_t, 4, 4, 64, 64, 1>())); }

TEST_F(TROWMAXTest, case_uint64_4x64) { EXPECT_TRUE((TRowMaxTestFramework<32, uint64_t, 4, 4, 64, 64, 1>())); }
TEST_F(TROWMAXTest, case_int64_32x32) { EXPECT_TRUE((TRowMaxTestFramework<33, int64_t, 32, 32, 32, 32, 1>())); }
TEST_F(TROWMAXTest, case_int64_1x10912) { EXPECT_TRUE((TRowMaxTestFramework<34, int64_t, 1, 1, 10912, 10912, 1>())); }

TEST_F(TROWMAXTest, case_int64_32x32_dndst) { EXPECT_TRUE((TRowMaxTestFramework<35, int64_t, 32, 32, 32, 32, 1>())); }
TEST_F(TROWMAXTest, case_uint64_32x32_dndst) { EXPECT_TRUE((TRowMaxTestFramework<36, uint64_t, 32, 32, 32, 32, 1>())); }
TEST_F(TROWMAXTest, case_int64_32x145_dndst)
{
    EXPECT_TRUE((TRowMaxTestFramework<37, int64_t, 32, 32, 145, 145, 1>()));
}
TEST_F(TROWMAXTest, case_uint64_32x145_dndst)
{
    EXPECT_TRUE((TRowMaxTestFramework<38, uint64_t, 32, 32, 145, 145, 1>()));
}

TEST_F(TROWMAXTest, case_int64_64x16_nddst4_small)
{
    EXPECT_TRUE((TRowMaxTestFramework<39, int64_t, 64, 64, 16, 16, 1>()));
}

TEST_F(TROWMAXTest, case_int64_64x16_dndst_small)
{
    EXPECT_TRUE((TRowMaxTestFramework<40, int64_t, 64, 64, 16, 16, 1>()));
}

TEST_F(TROWMAXTest, case_int64_64x16_nddst4_wide)
{
    EXPECT_TRUE((TRowMaxTestFramework<41, int64_t, 64, 64, 16, 16, 1>()));
}

TEST_F(TROWMAXTest, case_int64_64x16_dndst_wide)
{
    EXPECT_TRUE((TRowMaxTestFramework<42, int64_t, 64, 64, 16, 16, 1>()));
}

TEST_F(TROWMAXTest, case_int64_64x16_nddst4_guard)
{
    EXPECT_TRUE((TRowMaxTestFramework<43, int64_t, 64, 64, 16, 16, 1, true>()));
}

TEST_F(TROWMAXTest, case_int64_8x16_dndst_valid5_guard)
{
    EXPECT_TRUE((TRowMaxTestFramework<44, int64_t, 8, 5, 16, 16, 1, true>()));
}

TEST_F(TROWMAXTest, case_int64_8x16_dndst_valid6_guard)
{
    EXPECT_TRUE((TRowMaxTestFramework<45, int64_t, 8, 6, 16, 16, 1, true>()));
}

TEST_F(TROWMAXTest, case_int64_8x16_dndst_valid7_guard)
{
    EXPECT_TRUE((TRowMaxTestFramework<46, int64_t, 8, 7, 16, 16, 1, true>()));
}

TEST_F(TROWMAXTest, case47)
{
    bool ret = TRowMaxTestFramework<47, uint32_t, 127, 127, 64, 63, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case48)
{
    bool ret = TRowMaxTestFramework<48, uint32_t, 63, 63, 64, 64, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case49)
{
    bool ret = TRowMaxTestFramework<49, uint32_t, 31, 31, 128, 127, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case50)
{
    bool ret = TRowMaxTestFramework<50, uint32_t, 15, 15, 192, 192, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case51)
{
    bool ret = TRowMaxTestFramework<51, uint32_t, 7, 7, 448, 447, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case52)
{
    bool ret = TRowMaxTestFramework<52, uint16_t, 128, 128, 64, 64, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case53)
{
    bool ret = TRowMaxTestFramework<53, uint16_t, 64, 64, 64, 64, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case54)
{
    bool ret = TRowMaxTestFramework<54, uint16_t, 32, 32, 128, 128, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case55)
{
    bool ret = TRowMaxTestFramework<55, uint16_t, 16, 16, 192, 192, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case56)
{
    bool ret = TRowMaxTestFramework<56, uint16_t, 8, 8, 448, 448, 1>();
    EXPECT_TRUE(ret);
}
TEST_F(TROWMAXTest, case_f32_1x1)
{
    bool ret = TRowMaxTestFramework<57, float, 8, 8, 8, 1, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_f32_1x8192)
{
    bool ret = TRowMaxTestFramework<58, float, 1, 1, 8192, 8192, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_f32_512x64)
{
    bool ret = TRowMaxTestFramework<59, float, 256, 256, 64, 64, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_f32_16x2048)
{
    bool ret = TRowMaxTestFramework<60, float, 8, 8, 2048, 2048, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_f32_64x64_vc1)
{
    bool ret = TRowMaxTestFramework<61, float, 64, 64, 64, 1, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_f32_63x65)
{
    bool ret = TRowMaxTestFramework<62, float, 63, 63, 72, 65, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_f32_33x129_v127)
{
    bool ret = TRowMaxTestFramework<63, float, 33, 33, 136, 129, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_f32_238x320)
{
    bool ret = TRowMaxTestFramework<64, float, 238, 237, 80, 79, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_f16_64x128)
{
    bool ret = TRowMaxTestFramework<65, aclFloat16, 64, 64, 128, 128, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_f16_32x256_v255)
{
    bool ret = TRowMaxTestFramework<66, aclFloat16, 32, 32, 256, 255, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_f16_16x512)
{
    bool ret = TRowMaxTestFramework<67, aclFloat16, 16, 16, 512, 512, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_f16_8x1024)
{
    bool ret = TRowMaxTestFramework<68, aclFloat16, 8, 8, 1024, 1024, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_f16_30x216_v24)
{
    bool ret = TRowMaxTestFramework<69, aclFloat16, 30, 30, 224, 24, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_i8_128x64)
{
    bool ret = TRowMaxTestFramework<70, int8_t, 128, 128, 256, 256, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_i8_64x64_v63)
{
    bool ret = TRowMaxTestFramework<71, int8_t, 64, 64, 256, 255, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_i8_32x128)
{
    bool ret = TRowMaxTestFramework<72, int8_t, 32, 32, 256, 256, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_i8_16x192_v191)
{
    bool ret = TRowMaxTestFramework<73, int8_t, 16, 16, 256, 255, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_i8_8x448)
{
    bool ret = TRowMaxTestFramework<74, int8_t, 8, 8, 512, 512, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_u8_128x64)
{
    bool ret = TRowMaxTestFramework<75, uint8_t, 128, 128, 256, 256, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_u8_8x448_v447)
{
    bool ret = TRowMaxTestFramework<76, uint8_t, 8, 8, 512, 511, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_u8_2x2048)
{
    bool ret = TRowMaxTestFramework<77, uint8_t, 2, 2, 2048, 2048, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_u32_31x128_v127)
{
    bool ret = TRowMaxTestFramework<78, uint32_t, 31, 31, 128, 127, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_u16_32x128_v127)
{
    bool ret = TRowMaxTestFramework<79, uint16_t, 32, 32, 128, 127, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_f32_ubcap_106x192)
{
    bool ret = TRowMaxTestFramework<80, float, 106, 105, 192, 191, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_f32_1x16384)
{
    bool ret = TRowMaxTestFramework<81, float, 1, 1, 16384, 16384, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_f32_vrow11_30x216)
{
    bool ret = TRowMaxTestFramework<82, float, 30, 11, 216, 216, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_f32_vrow1_64x64)
{
    bool ret = TRowMaxTestFramework<83, float, 64, 1, 64, 64, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_f32_prime_251x128)
{
    bool ret = TRowMaxTestFramework<84, float, 251, 250, 96, 95, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_f16_ubcap_128x320)
{
    bool ret = TRowMaxTestFramework<85, aclFloat16, 128, 127, 320, 319, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_s16_ubcap_200x208)
{
    bool ret = TRowMaxTestFramework<86, int16_t, 200, 199, 208, 207, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_i8_ubcap_88x896)
{
    bool ret = TRowMaxTestFramework<87, int8_t, 88, 87, 896, 887, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_u8_ubcap_88x896)
{
    bool ret = TRowMaxTestFramework<88, uint8_t, 88, 87, 896, 887, 1>();
    EXPECT_TRUE(ret);
}
