/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ARCH_IMPL_SWITCH_HPP
#define ARCH_IMPL_SWITCH_HPP

// A6 (dav-9201) keeps dedicated TLoad/TExtract implementations; the other
// archs share the A5 headers. Per-instruction headers that internally use
// these implementations (TPush/TFillPad/TPrefetch need TLoad, TMov needs
// TExtract) include this switcher instead of carrying arch #ifdefs themselves.
// The TExtract group stays A5/A6-only to keep the kirin include surface
// unchanged (kirin units pull TLoad but never the A5 TExtract).
#if defined(PTO_NPU_ARCH_A6)
#include <pto/npu/a6/TLoad.hpp>
#else
#include <pto/npu/a5/TLoad.hpp>
#endif

#if defined(PTO_NPU_ARCH_A6)
#include <pto/npu/a6/TExtract.hpp>
#elif defined(PTO_NPU_ARCH_A5)
#include <pto/npu/a5/TExtract.hpp>
#endif

#endif
