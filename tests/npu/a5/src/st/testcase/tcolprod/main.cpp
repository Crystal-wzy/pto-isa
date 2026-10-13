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
#include <gtest/gtest.h>
#include <acl/acl.h>
#include <cstring>

using namespace std;
using namespace PtoTestCommon;

template <uint32_t caseId>
void launchTCOLPRODTestCase(void* out, void* src, aclrtStream stream);

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

class TCOLPRODTest : public testing::Test {
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
        ReadFile(GetGoldenDir() + "/golden.bin", dstByteSize, golden.data(), dstByteSize);
        ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, result.data(), dstByteSize);

        float eps = 0.001f;
        if (printAllEn) {
            return ResultCmp(golden, result, eps, 0, 1000, true);
        }
        return ResultCmp(golden, result, eps, 0, 1000, false, true);
    }

    // Compares raw bf16 bit patterns by widening each element to fp32 first.
    bool CompareGoldenBf16(size_t dstByteSize, bool printAllEn = false)
    {
        std::vector<uint16_t> goldenBits(dstByteSize / sizeof(uint16_t));
        std::vector<uint16_t> resultBits(dstByteSize / sizeof(uint16_t));
        ReadFile(GetGoldenDir() + "/golden.bin", dstByteSize, goldenBits.data(), dstByteSize);
        ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, resultBits.data(), dstByteSize);
        std::vector<float> golden(goldenBits.size());
        std::vector<float> result(resultBits.size());
        for (size_t i = 0; i < goldenBits.size(); ++i) {
            golden[i] = BitCastF32(static_cast<uint32_t>(goldenBits[i]) << 16);
            result[i] = BitCastF32(static_cast<uint32_t>(resultBits[i]) << 16);
        }
        if (printAllEn) {
            return ResultCmp(golden, result, 0.005f, 0, 1000, true);
        }
        return ResultCmp(golden, result, 0.005f, 0, 1000, false, true);
    }

    static float BitCastF32(uint32_t bits)
    {
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    template <
        uint32_t caseId, typename T, int srcRow, int srcValidRow, int dstRow, int col, int validCol,
        bool isBf16 = false>
    bool TCOLPRODTestFramework()
    {
        size_t dstByteSize = dstRow * col * sizeof(T);
        size_t srcByteSize = srcRow * col * sizeof(T);
        aclrtMallocHost(&dstHost, dstByteSize);
        aclrtMallocHost(&srcHost, srcByteSize);
        aclrtMalloc(&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemset(dstDevice, dstByteSize, 0, dstByteSize);

        ReadFile(GetGoldenDir() + "/input.bin", srcByteSize, srcHost, srcByteSize);
        aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        launchTCOLPRODTestCase<caseId>(dstDevice, srcDevice, stream);
        aclrtSynchronizeStream(stream);

        aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
        WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstByteSize);

        aclrtFree(dstDevice);
        aclrtFree(srcDevice);
        aclrtFreeHost(dstHost);
        aclrtFreeHost(srcHost);

        if constexpr (isBf16) {
            return CompareGoldenBf16(dstByteSize);
        } else {
            return CompareGolden<T>(dstByteSize);
        }
    }
};

TEST_F(TCOLPRODTest, case01)
{
    bool ret = TCOLPRODTestFramework<1, float, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLPRODTest, case02)
{
    bool ret = TCOLPRODTestFramework<2, float, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLPRODTest, case03)
{
    bool ret = TCOLPRODTestFramework<3, float, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLPRODTest, case41)
{
    bool ret = TCOLPRODTestFramework<41, int16_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLPRODTest, case42)
{
    bool ret = TCOLPRODTestFramework<42, int16_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLPRODTest, case43)
{
    bool ret = TCOLPRODTestFramework<43, int16_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLPRODTest, case51)
{
    bool ret = TCOLPRODTestFramework<51, uint16_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLPRODTest, case52)
{
    bool ret = TCOLPRODTestFramework<52, uint16_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLPRODTest, case53)
{
    bool ret = TCOLPRODTestFramework<53, uint16_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLPRODTest, case61)
{
    bool ret = TCOLPRODTestFramework<61, int32_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLPRODTest, case62)
{
    bool ret = TCOLPRODTestFramework<62, int32_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLPRODTest, case63)
{
    bool ret = TCOLPRODTestFramework<63, int32_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLPRODTest, case71)
{
    bool ret = TCOLPRODTestFramework<71, uint32_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLPRODTest, case72)
{
    bool ret = TCOLPRODTestFramework<72, uint32_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLPRODTest, case73)
{
    bool ret = TCOLPRODTestFramework<73, uint32_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}

// fp16 high-precision (fp32 multiplication) cases.
TEST_F(TCOLPRODTest, case81)
{
    bool ret = TCOLPRODTestFramework<81, aclFloat16, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLPRODTest, case82)
{
    bool ret = TCOLPRODTestFramework<82, aclFloat16, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLPRODTest, case83)
{
    bool ret = TCOLPRODTestFramework<83, aclFloat16, 63, 63, 1, 128, 127>();
    EXPECT_TRUE(ret);
}

// bf16 high-precision (fp32 multiplication) cases.
TEST_F(TCOLPRODTest, case91)
{
    bool ret = TCOLPRODTestFramework<91, uint16_t, 1, 1, 1, 256, 255, true>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLPRODTest, case92)
{
    bool ret = TCOLPRODTestFramework<92, uint16_t, 16, 16, 1, 128, 127, true>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLPRODTest, case93)
{
    bool ret = TCOLPRODTestFramework<93, uint16_t, 63, 63, 1, 128, 127, true>();
    EXPECT_TRUE(ret);
}
