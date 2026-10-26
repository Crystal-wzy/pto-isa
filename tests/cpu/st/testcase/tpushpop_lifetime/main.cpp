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
#include <memory>
#include <new>
#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

// This target isolates the allocation override from other instruction tests.
namespace {
thread_local bool trackAllocations = false;
thread_local std::size_t allocationCount = 0;
void* hookStorage = nullptr;
void* storageHook(uint64_t, std::size_t) { return hookStorage; }
} // namespace

void* operator new(std::size_t bytes)
{
    void* pointer = std::malloc(bytes == 0 ? 1 : bytes);
    if (pointer == nullptr) {
        throw std::bad_alloc();
    }
    if (trackAllocations) {
        ++allocationCount;
    }
    return pointer;
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete[](void* pointer) noexcept { ::operator delete(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { ::operator delete(pointer); }
// GTest also uses nothrow new. Route it through the same allocator so ASan sees
// matching allocation/deallocation families when it interposes these overloads.
void* operator new(std::size_t bytes, const std::nothrow_t&) noexcept
{
    try {
        return ::operator new(bytes);
    } catch (const std::bad_alloc&) {
        return nullptr;
    }
}
void* operator new[](std::size_t bytes, const std::nothrow_t& tag) noexcept { return ::operator new(bytes, tag); }
void operator delete(void* pointer, const std::nothrow_t&) noexcept { ::operator delete(pointer); }
void operator delete[](void* pointer, const std::nothrow_t&) noexcept { ::operator delete(pointer); }

namespace {
using namespace pto;

class TPushPopLifetimeTest : public testing::Test {
protected:
    NPUArch previousArch;
    void SetUp() override
    {
        previousArch = NPUMemoryModel::Instance().GetArch();
        NPU_MEMORY_INIT(NPUArch::A5);
        NPU_MEMORY_CLEAR();
        allocationCount = 0;
    }
    void TearDown() override
    {
        trackAllocations = false;
        cpu_sim::register_hooks(nullptr, nullptr);
        hookStorage = nullptr;
        cpu_sim::reset_execution_context();
        NPU_MEMORY_CLEAR();
        NPU_MEMORY_INIT(previousArch);
    }
};

// Raw storage ownership matches the runtime provider ABI: no typed payload
// destructor is available when the runtime retires a task's shared storage.
template <uint8_t DirectionType, bool ResetBeforeRelease, bool RawStorage>
void checkLifetime()
{
    using Vec = Tile<TileType::Vec, float8_e8m0_t, 16, 4, BLayout::RowMajor, 16, 4, SLayout::RowMajor, 32>;
    using Mat = Tile<TileType::Mat, float8_e8m0_t, 4, 16, BLayout::ColMajor, 4, 16, SLayout::ColMajor, 32>;
    using Pipe = TPipe<20, DirectionType, 256, 2, 2, true>;
    Vec source;
    Mat result;
    TASSIGN(source, 0);
    TASSIGN(result, 0);
    // Offset zero is relative to UB for Vec and L1 for Mat, so the buffers differ.
    ASSERT_NE(source.data(), result.data());
    cpu_sim::ScopedExecutionContext ctx(0, 0, 1);
    for (int i = 0; i < 64; ++i) {
        source.data()[i] = float8_e8m0_t::FromRaw(static_cast<uint8_t>(i + 127));
    }
    for (int task = 0; task < 10; ++task) {
        typename Pipe::SharedStateStorage typedStorage{};
        std::unique_ptr<void, decltype(&std::free)> rawStorage(
            RawStorage ? std::calloc(1, sizeof(typedStorage)) : nullptr, &std::free);
        if constexpr (RawStorage) {
            ASSERT_NE(rawStorage, nullptr);
            hookStorage = rawStorage.get();
        } else {
            hookStorage = &typedStorage;
        }
        cpu_sim::register_hooks(nullptr, reinterpret_cast<void*>(storageHook));
        trackAllocations = true;
        Pipe pipe(nullptr, 0, 0);
        // Exercise nonzero offsets, multiple entries and ring wraparound.
        for (int iteration = 0; iteration < 5; ++iteration) {
            pipe.prod.setAllocateStatus(true);
            pipe.prod.setRecordStatus(false);
            pipe.prod.setEntryOffset(32);
            TPUSH<Pipe, Vec, TileSplitAxis::TILE_NO_SPLIT>(pipe, source);
            pipe.prod.setAllocateStatus(false);
            pipe.prod.setRecordStatus(true);
            pipe.prod.setEntryOffset(128);
            TPUSH<Pipe, Vec, TileSplitAxis::TILE_NO_SPLIT>(pipe, source);
            pipe.cons.setWaitStatus(true);
            pipe.cons.setentryOffset(32);
            TPOP<Pipe, Mat, TileSplitAxis::TILE_NO_SPLIT>(pipe, result);
            pipe.cons.setWaitStatus(false);
            pipe.cons.setentryOffset(128);
            TPOP<Pipe, Mat, TileSplitAxis::TILE_NO_SPLIT>(pipe, result);
            TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
        }
        if constexpr (ResetBeforeRelease) {
            Pipe::reset_for_cpu_sim();
        }
        cpu_sim::register_hooks(nullptr, nullptr);
        rawStorage.reset();
        hookStorage = nullptr;
        trackAllocations = false;
        EXPECT_EQ(allocationCount, 0u) << "task=" << task;
        for (int i = 0; i < 64; ++i) {
            EXPECT_EQ(result.data()[i].RawData(), source.data()[i].RawData());
        }
    }
}

TEST_F(TPushPopLifetimeTest, raw_provider_release_without_reset)
{
    checkLifetime<Direction::DIR_V2C, false, true>();
    checkLifetime<Direction::DIR_BOTH, false, true>();
}
TEST_F(TPushPopLifetimeTest, raw_provider_release_after_reset)
{
    checkLifetime<Direction::DIR_V2C, true, true>();
    checkLifetime<Direction::DIR_BOTH, true, true>();
}
TEST_F(TPushPopLifetimeTest, typed_storage_release_without_reset)
{
    checkLifetime<Direction::DIR_V2C, false, false>();
    checkLifetime<Direction::DIR_BOTH, false, false>();
}
} // namespace
