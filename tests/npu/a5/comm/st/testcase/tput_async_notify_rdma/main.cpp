/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>

#include "../comm_mpi.h"
#include "../tput_async_rdma/tput_async_rdma_kernel.h"

// Verify one int32 SET notify, payload delivery, and adjacent signal canaries.
TEST(TPutAsyncNotifyRdma, Int32SetAndCanaries)
{
    if (CommMpiSize() != 2) {
        GTEST_SKIP() << "Requires exactly two MPI ranks";
    }
    const RdmaTestResult result = RunPutAsyncNotifyRdmaSet(2, 2, 0, 0);
    if (result == RdmaTestResult::SKIPPED) {
        GTEST_SKIP() << "RDMA runtime prerequisites are unavailable on at least one rank";
    }
    ASSERT_EQ(result, RdmaTestResult::PASSED);
}

static void CheckNotifyAdd(int32_t increment, int operations = 1, int blocks = 1)
{
    if (CommMpiSize() != 2) {
        GTEST_SKIP() << "Requires exactly two MPI ranks";
    }
    const RdmaTestResult result = RunPutAsyncNotifyRdmaAdd(2, 2, 0, 0, increment, operations, blocks);
    if (result == RdmaTestResult::SKIPPED) {
        GTEST_SKIP() << "RDMA runtime prerequisites are unavailable on at least one rank";
    }
    ASSERT_EQ(result, RdmaTestResult::PASSED);
}

TEST(TPutAsyncNotifyRdma, Int32AddAndCanaries) { CheckNotifyAdd(37); }

TEST(TPutAsyncNotifyRdma, Int32AddNegative) { CheckNotifyAdd(-3, 4); }

TEST(TPutAsyncNotifyRdma, Int32AddZero) { CheckNotifyAdd(0); }

// Four AIVs write disjoint payload slices and increment one shared signal 16 times.
TEST(TPutAsyncNotifyRdma, Int32AddMultiAiv) { CheckNotifyAdd(1, 16, 4); }

int main(int argc, char** argv)
{
    if (!CommMpiInit(&argc, &argv)) {
        return 1;
    }
    ::testing::InitGoogleTest(&argc, argv);
    const int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}
