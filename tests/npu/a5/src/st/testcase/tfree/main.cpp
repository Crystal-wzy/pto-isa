/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdlib>
#include <vector>
#include "acl/acl.h"
#include <gtest/gtest.h>

void LaunchTFreeDeferred(float* output, float* input, void* stream);

namespace {
struct Resources {
    bool initialized = false;
    bool deviceSet = false;
    int device = 0;
    aclrtStream stream = nullptr;
    float* input = nullptr;
    float* output = nullptr;

    ~Resources()
    {
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (input != nullptr) {
            aclrtFree(input);
        }
        if (output != nullptr) {
            aclrtFree(output);
        }
        if (deviceSet) {
            aclrtResetDevice(device);
        }
        if (initialized) {
            aclFinalize();
        }
    }
};
} // namespace

TEST(TFreeTest, DeferredPairAcrossRepeatedPipeLifetimes)
{
    constexpr int m = 16;
    constexpr int k = 64;
    constexpr int n = 64;
    constexpr int iterations = 32;
    constexpr int inputElements = 2 * (m * k + k * n);
    constexpr int outputElements = 2 * m * n;
    std::vector<float> input(iterations * inputElements);
    std::vector<float> golden(iterations * outputElements);
    std::vector<float> actual(golden.size());

    // Small integer inputs make the independent scalar matmul golden exact in
    // FP32, while distinct tiles and iterations detect stale FIFO payloads.
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(static_cast<int>((i * 17 + i / 7) % 5) - 2);
    }
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const float* inputs = input.data() + iteration * inputElements;
        for (int matmul = 0; matmul < 2; ++matmul) {
            const float* a = inputs + k * n + matmul * m * k;
            const float* b = inputs + matmul * (k * n + 2 * m * k);
            for (int row = 0; row < m; ++row) {
                for (int col = 0; col < n; ++col) {
                    float sum = 0;
                    for (int inner = 0; inner < k; ++inner) {
                        sum += a[row * k + inner] * b[inner * n + col];
                    }
                    golden[iteration * outputElements + matmul * m * n + row * n + col] = sum;
                }
            }
        }
    }

    Resources resources;
    const char* allocatedDevice = std::getenv("TASK_DEVICE");
    if (allocatedDevice != nullptr) {
        resources.device = std::atoi(allocatedDevice);
    }
    ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
    resources.initialized = true;
    ASSERT_EQ(aclrtSetDevice(resources.device), ACL_SUCCESS);
    resources.deviceSet = true;
    ASSERT_EQ(aclrtCreateStream(&resources.stream), ACL_SUCCESS);
    const size_t inputBytes = input.size() * sizeof(float);
    const size_t outputBytes = actual.size() * sizeof(float);
    ASSERT_EQ(
        aclrtMalloc(reinterpret_cast<void**>(&resources.input), inputBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(
        aclrtMalloc(reinterpret_cast<void**>(&resources.output), outputBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(
        aclrtMemcpy(resources.input, inputBytes, input.data(), inputBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemset(resources.output, outputBytes, 0x7e, outputBytes), ACL_SUCCESS);
    LaunchTFreeDeferred(resources.output, resources.input, resources.stream);
    ASSERT_EQ(aclrtSynchronizeStreamWithTimeout(resources.stream, 30000), ACL_SUCCESS);
    ASSERT_EQ(
        aclrtMemcpy(actual.data(), outputBytes, resources.output, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    EXPECT_EQ(actual, golden);
}
