# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

# Mat (L1) ConvTile cases run on the cube core.
pto_cube_st(tstore_conv_valid)

# UB (Vec) ConvTile cases need the vector-core instruction set, so their kernel is a second library
# linked into the same test executable.
add_library(tstore_conv_valid_ub_kernel SHARED tstore_conv_valid_kernel.cpp)
target_compile_options(tstore_conv_valid_ub_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220-vec -DMEMORY_BASE -std=c++17)
target_compile_definitions(tstore_conv_valid_ub_kernel PRIVATE PTO_CONV_TSTORE_UB_CASE=1)
target_include_directories(tstore_conv_valid_ub_kernel PRIVATE
    ${ASCEND_HOME_PATH}/pkg_inc/
    ${ASCEND_HOME_PATH}/pkg_inc/profiling/
    ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
)
target_link_options(tstore_conv_valid_ub_kernel PRIVATE --cce-fatobj-link)
target_link_libraries(tstore_conv_valid PRIVATE tstore_conv_valid_ub_kernel)
