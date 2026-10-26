#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os

import numpy as np

VALID_SENTINEL = -7

# (case, raw_shape, valid_shape, dtype). The TLOAD contract only fills the valid box; everything else must
# keep the sentinel that the kernel pre-fills into the L1 tile.
CASES = [
    ("TLoadConvValidTest.NC1HWC0_valid_slow_half", (2, 3, 4, 5, 16), (1, 2, 3, 4, 16), np.float16),
    ("TLoadConvValidTest.NC1HWC0_valid_fast_half", (2, 3, 4, 5, 16), (2, 2, 3, 5, 16), np.float16),
    ("TLoadConvValidTest.NC1HWC0_valid_dynamic_float", (2, 3, 4, 5, 8), (1, 2, 3, 4, 8), np.float32),
    ("TLoadConvValidTest.NC1HWC0_valid_bf16", (2, 3, 4, 5, 16), (1, 2, 3, 4, 16), np.uint16),
    ("TLoadConvValidTest.NDC1HWC0_valid_half", (2, 3, 2, 4, 5, 16), (1, 2, 1, 3, 4, 16), np.float16),
    ("TLoadConvValidTest.FracZ_4D_valid_half", (8, 2, 16, 16), (6, 1, 10, 16), np.float16),
    ("TLoadConvValidTest.FracZ_5D_valid_int8", (4, 2, 6, 16, 32), (3, 1, 4, 10, 32), np.int8),
    ("TLoadConvValidTest.FracZ_5D_valid_dynamic_int8", (2, 3, 4, 8, 32), (1, 2, 3, 6, 32), np.int8),
]


def gen_case(case_name, raw_shape, valid_shape, dtype):
    if not os.path.exists(case_name):
        os.makedirs(case_name)
    os.chdir(case_name)

    total = int(np.prod(raw_shape))
    data = (np.arange(total) % 100).astype(dtype).reshape(raw_shape)

    if dtype == np.float16:
        # The host represents fp16 as raw 16-bit words, so the sentinel is the word bit pattern.
        golden = np.full(raw_shape, np.uint16(VALID_SENTINEL & 0xFFFF), dtype=np.uint16).view(np.float16)
    elif dtype == np.uint16:
        # bf16 is also carried as raw 16-bit words on the host.
        golden = np.full(raw_shape, np.uint16(VALID_SENTINEL & 0xFFFF), dtype=np.uint16)
    else:
        golden = np.full(raw_shape, VALID_SENTINEL, dtype=dtype)
    valid_slice = tuple(slice(0, int(v)) for v in valid_shape)
    golden[valid_slice] = data[valid_slice]

    data.tofile("./input.bin")
    golden.tofile("./golden.bin")
    os.chdir("..")


if __name__ == "__main__":
    for case_name, raw_shape, valid_shape, dtype in CASES:
        gen_case(case_name, raw_shape, valid_shape, dtype)
