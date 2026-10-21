#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os
import numpy as np

np.random.seed(19)


def gen_golden_data_nz2nz(info):
    # Only the valid region is filled; whole-shape padding stays 0 (real dst stride gaps).
    g_shape = info.shape
    g_whole_shape = info.whole_shape
    input_arr = np.random.randint(-5, 5, size=g_whole_shape).astype(info.data_type)
    output_arr = np.zeros(g_whole_shape, dtype=info.data_type)
    valid = tuple(slice(0, s) for s in g_shape)
    output_arr[valid] = input_arr[valid]
    input_arr.tofile("./input.bin")
    output_arr.tofile("./golden.bin")


def gen_golden_data_nz2nd(info):
    # Source NZ fractal [N/c0, M/16, 16, c0]; golden = M x N written into the top-left of
    # a gWholeM x gWholeN ND buffer (rest stays 0 -> real row stride gap).
    m, n = info.shape
    whole_m, whole_n = info.whole_shape
    data_type = info.data_type
    c0 = 32 // np.dtype(data_type).itemsize
    assert m % 16 == 0 and n % c0 == 0, "NZ2ND requires M % 16 == 0 and N % c0 == 0"
    assert whole_m >= m and whole_n >= n, "NZ2ND requires whole >= shape"
    logical = np.random.randint(-5, 5, size=(m, n)).astype(data_type)
    nz = logical.reshape(m // 16, 16, n // c0, c0).transpose(2, 0, 1, 3).astype(data_type)
    golden = np.zeros((whole_m, whole_n), dtype=data_type)
    golden[0:m, 0:n] = logical
    nz.tofile("./input.bin")
    golden.tofile("./golden.bin")


def gen_golden_data_nz2nd_padded(info):
    # Padded/partial tile (matmul C geometry): the source is the full static sM x sN NZ buffer
    # (valid region M x N filled, pad zero); golden is the valid M x N top-left of the ND buffer.
    m, n = info.shape
    sm, sn = info.tile_shape
    whole_m, whole_n = info.whole_shape
    data_type = info.data_type
    c0 = 32 // np.dtype(data_type).itemsize
    inner_cols = max(c0, 16)
    assert sm % 16 == 0 and sn % inner_cols == 0, "padded tile must be 16 / inner-col aligned"
    assert sm >= m and sn >= n, "padded tile must cover the valid shape"
    assert whole_m >= m and whole_n >= n, "NZ2ND requires whole >= shape"
    logical = np.random.randint(-5, 5, size=(m, n)).astype(data_type)
    full = np.zeros((sm, sn), dtype=data_type)
    full[0:m, 0:n] = logical
    nz = full.reshape(sm // 16, 16, sn // inner_cols, inner_cols).transpose(2, 0, 1, 3).astype(data_type)
    golden = np.zeros((whole_m, whole_n), dtype=data_type)
    golden[0:m, 0:n] = logical
    nz.tofile("./input.bin")
    golden.tofile("./golden.bin")


class GlobalTensorInfo:
    def __init__(self, case_name, data_type, fmt, shape, whole_shape, tile_shape=None):
        self.case_name = case_name
        self.data_type = data_type
        self.format = fmt
        # NZ: shape / whole_shape are the 5D NZ fractal dims; NZ2ND: (M, N) / (gWholeM, gWholeN).
        self.shape = shape
        self.whole_shape = whole_shape
        # Padded/partial NZ2ND only: static tile (sM, sN) >= valid (M, N).
        self.tile_shape = tile_shape


if __name__ == "__main__":
    case_list = [
        # NZ2NZ, 8/16/32-bit (half generated as float16 = 2B, matches __gm__ half).
        GlobalTensorInfo("TStoreMat2GMTest.case1_nz2nz", np.float16, "NZ", (1, 1, 2, 16, 16), (1, 1, 2, 16, 16)),
        GlobalTensorInfo("TStoreMat2GMTest.case2_nz2nz", np.int16, "NZ", (1, 2, 1, 16, 16), (2, 2, 2, 16, 16)),
        GlobalTensorInfo("TStoreMat2GMTest.case3_nz2nz", np.int8, "NZ", (1, 1, 2, 16, 32), (1, 2, 2, 16, 32)),
        GlobalTensorInfo("TStoreMat2GMTest.case4_nz2nz", np.uint16, "NZ", (2, 1, 2, 16, 16), (2, 2, 2, 16, 16)),
        GlobalTensorInfo("TStoreMat2GMTest.case5_nz2nz", np.int8, "NZ", (1, 2, 1, 16, 32), (2, 2, 2, 16, 32)),
        GlobalTensorInfo("TStoreMat2GMTest.case6_nz2nz", np.float16, "NZ", (1, 1, 1, 16, 16), (2, 1, 1, 16, 16)),
        GlobalTensorInfo("TStoreMat2GMTest.case7_nz2nz", np.uint16, "NZ", (1, 4, 1, 16, 16), (2, 4, 2, 16, 16)),
        GlobalTensorInfo("TStoreMat2GMTest.case8_nz2nz", np.int16, "NZ", (3, 1, 2, 16, 16), (4, 1, 2, 16, 16)),
        # gShape1 = 17/37 (non-C0-aligned effective width) + whole > tile: nonzero K-slice dst gap.
        GlobalTensorInfo("TStoreMat2GMTest.case9_nz2nz", np.float16, "NZ", (1, 17, 1, 16, 16), (1, 17, 2, 16, 16)),
        GlobalTensorInfo("TStoreMat2GMTest.case10_nz2nz", np.int16, "NZ", (2, 37, 2, 16, 16), (2, 37, 3, 16, 16)),
        GlobalTensorInfo("TStoreMat2GMTest.case11_nz2nz", np.int8, "NZ", (1, 17, 2, 16, 32), (1, 17, 4, 16, 32)),
        # 32-bit (C0 = 8).
        GlobalTensorInfo("TStoreMat2GMTest.case12_nz2nz", np.int32, "NZ", (1, 1, 2, 16, 8), (1, 1, 2, 16, 8)),
        GlobalTensorInfo("TStoreMat2GMTest.case13_nz2nz", np.float32, "NZ", (2, 1, 2, 16, 8), (2, 2, 2, 16, 8)),
        # NZ2ND: M % 16 == 0, N % (32/sizeof) == 0; gWholeN > N / gWholeM > M -> real stride gap / pad.
        GlobalTensorInfo("TStoreMat2GMTest.case1_nz2nd", np.float16, "NZ2ND", (16, 16), (16, 16)),
        GlobalTensorInfo("TStoreMat2GMTest.case2_nz2nd", np.int16, "NZ2ND", (32, 32), (32, 32)),
        GlobalTensorInfo("TStoreMat2GMTest.case3_nz2nd", np.int8, "NZ2ND", (32, 32), (32, 32)),
        GlobalTensorInfo("TStoreMat2GMTest.case4_nz2nd", np.uint16, "NZ2ND", (16, 64), (16, 64)),
        GlobalTensorInfo("TStoreMat2GMTest.case5_nz2nd", np.float16, "NZ2ND", (64, 32), (64, 32)),
        GlobalTensorInfo("TStoreMat2GMTest.case6_nz2nd", np.int16, "NZ2ND", (16, 16), (16, 16)),
        GlobalTensorInfo("TStoreMat2GMTest.case7_nz2nd", np.float16, "NZ2ND", (32, 32), (32, 48)),
        GlobalTensorInfo("TStoreMat2GMTest.case8_nz2nd", np.int16, "NZ2ND", (32, 32), (40, 32)),
        GlobalTensorInfo("TStoreMat2GMTest.case9_nz2nd", np.int8, "NZ2ND", (32, 64), (32, 96)),
        GlobalTensorInfo("TStoreMat2GMTest.case10_nz2nd", np.uint16, "NZ2ND", (16, 16), (24, 32)),
        # 32-bit (C0 = 8).
        GlobalTensorInfo("TStoreMat2GMTest.case11_nz2nd", np.int32, "NZ2ND", (32, 32), (32, 32)),
        GlobalTensorInfo("TStoreMat2GMTest.case12_nz2nd", np.float32, "NZ2ND", (32, 32), (32, 48)),
        # Padded/partial tile (matmul C geometry): tile_shape = static (sM, sN), shape = valid.
        GlobalTensorInfo("TStoreMat2GMTest.case13_nz2nd", np.float16, "NZ2ND", (16, 32), (16, 32), (32, 32)),
        GlobalTensorInfo("TStoreMat2GMTest.case14_nz2nd", np.float16, "NZ2ND", (30, 60), (30, 60), (32, 64)),
        GlobalTensorInfo("TStoreMat2GMTest.case15_nz2nd", np.int8, "NZ2ND", (16, 16), (16, 16), (32, 32)),
    ]

    for info in case_list:
        if not os.path.exists(info.case_name):
            os.makedirs(info.case_name)
        original_dir = os.getcwd()
        os.chdir(info.case_name)
        if info.tile_shape is not None:
            gen_golden_data_nz2nd_padded(info)
        elif info.format == "NZ2ND":
            gen_golden_data_nz2nd(info)
        else:
            gen_golden_data_nz2nz(info)
        os.chdir(original_dir)
