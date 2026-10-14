#!/user/bin/python3
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
from enum import Enum

np.random.seed(19)


class DataFormat(Enum):
    NCHW2NC1HWC0 = 1
    NC1HWC02C1HWN1N0C0 = 2
    GNCHW2GNC1HWC0 = 3
    GNC1HWC02C1HWN1N0C0 = 4
    NC1HWC02NC1C0HW = 5
    GNC1HWC02GNC1C0HW = 6
    NCHW2C1HWN1N0C0 = 7


def nchw_to_nc1hwc0(nchw_tensor: np.ndarray, c0: int) -> np.ndarray:
    if nchw_tensor.ndim != 4:
        raise ValueError(f"The input must be a 4-dimensional NCHW tensor, current dim : {nchw_tensor.ndim}")
    if (c0 & (c0 - 1)) != 0 and c0 != 1:
        raise ValueError(f"C0 should be 8/16/32, now is : {c0}")

    n, c, h, w = nchw_tensor.shape
    c1 = (c + c0 - 1) // c0
    pad_c = c1 * c0 - c
    if pad_c > 0:
        pad_width = ((0, 0), (0, pad_c), (0, 0), (0, 0))
        nchw_padded = np.pad(nchw_tensor, pad_width, mode="constant", constant_values=0)
    else:
        nchw_padded = nchw_tensor

    nc1c0hw_tensor = nchw_padded.reshape(n, c1, c0, h, w)

    # NC1C0HW → NC1HWC0
    # origin index：0(n),1(c1),2(C0),3(h),4(w) → new index ：0,1,3,4,2
    nc1hwc0_tensor = np.transpose(nc1c0hw_tensor, axes=(0, 1, 3, 4, 2))

    return nc1hwc0_tensor


def gnchw_to_gnc1hwc0(gnchw_tensor: np.ndarray, c0: int) -> np.ndarray:
    if gnchw_tensor.ndim != 5:
        raise ValueError(f"The input must be a 5-dimensional GNCHW tensor, current dim : {gnchw_tensor.ndim}")
    if (c0 & (c0 - 1)) != 0 and c0 != 1:
        raise ValueError(f"C0 should be 8/16/32, now is : {c0}")

    g, n, c, h, w = gnchw_tensor.shape
    c1 = (c + c0 - 1) // c0
    pad_c = c1 * c0 - c
    if pad_c > 0:
        pad_width = ((0, 0), (0, 0), (0, pad_c), (0, 0), (0, 0))
        gnchw_padded = np.pad(gnchw_tensor, pad_width, mode="constant", constant_values=0)
    else:
        gnchw_padded = gnchw_tensor

    gnc1c0hw_tensor = gnchw_padded.reshape(g, n, c1, c0, h, w)

    # GNC1C0HW → GNC1HWC0
    # origin index：0(g),1(n),2(c1),3(C0),4(h),5(w) → new index ：0,1,2,4,5,3
    gnc1hwc0_tensor = np.transpose(gnc1c0hw_tensor, axes=(0, 1, 2, 4, 5, 3))

    return gnc1hwc0_tensor


def _golden_nchw2_nc1hwc0(g_info):
    """NCHW -> NC1HWC0 golden; pads channel on full tensor when needed."""
    data_type = g_info.data_type
    g_whole_shape1 = g_info.g_whole_shape1
    g_whole_shape2 = g_info.g_whole_shape2
    g_shape0 = g_info.g_shape0
    g_shape1 = g_info.g_shape1
    g_shape2 = g_info.g_shape2
    g_shape3 = g_info.g_shape3
    g_shape4 = g_info.g_shape4
    g_whole_shape3 = g_info.g_whole_shape3
    g_whole_shape4 = g_info.g_whole_shape4

    input_arr = np.random.randint(
        1, 5, size=(g_whole_shape1, g_whole_shape2, g_whole_shape3, g_whole_shape4)
    ).astype(data_type)
    golden_nchw = np.zeros(shape=(g_shape0, g_shape1 * g_shape4, g_shape2, g_shape3), dtype=data_type)
    golden_nchw = input_arr[0:g_shape0, 0:g_shape1 * g_shape4, 0:g_shape2, 0:g_shape3]
    pad_c1 = g_shape1 * g_shape4 - g_whole_shape2
    if pad_c1 > 0:
        pad_width1 = ((0, 0), (0, pad_c1), (0, 0), (0, 0))
        input_arr = np.pad(input_arr, pad_width1, mode="constant", constant_values=0)
    output_arr = nchw_to_nc1hwc0(golden_nchw, g_shape4)
    return input_arr, output_arr


def _golden_nc1hwc0_to_c1hwn1n0c0(g_info):
    """[N, C1, H, W, C0] -> [C1, H, W, N1, N0, C0]; pads N when needed."""
    data_type = g_info.data_type
    g_shape3 = g_info.g_shape3
    g_shape4 = g_info.g_shape4
    dst_n1 = g_shape3
    dst_n0 = g_shape4
    src_n = g_info.g_whole_shape0
    src_c1 = g_info.g_whole_shape1
    src_h = g_info.g_whole_shape2
    src_w = g_info.g_whole_shape3
    src_c0 = g_info.g_whole_shape4

    input_ori = np.random.randint(1, 5, size=(src_n, src_c1, src_h, src_w, src_c0)).astype(data_type)
    pad_n1 = dst_n1 * dst_n0 - g_info.g_whole_shape0
    if pad_n1 > 0:
        pad_width1 = ((0, pad_n1), (0, 0), (0, 0), (0, 0), (0, 0))
        input_arr = np.pad(input_ori, pad_width1, mode="constant", constant_values=0)
    else:
        input_arr = input_ori
    output_arr = input_arr.transpose([1, 2, 3, 0, 4])
    return input_arr, output_arr


def _golden_gnchw2_gnc1hwc0(g_info):
    """GNCHW -> GNC1HWC0 golden; pads channel on full tensor when needed."""
    data_type = g_info.data_type
    g_whole_shape1 = g_info.g_whole_shape1
    g_whole_shape2 = g_info.g_whole_shape2
    g_whole_shape3 = g_info.g_whole_shape3
    g_whole_shape4 = g_info.g_whole_shape4
    g_shape0 = g_info.g_shape0
    g_shape1 = g_info.g_shape1
    g_shape2 = g_info.g_shape2
    g_shape3 = g_info.g_shape3
    g_shape4 = g_info.g_shape4
    g_shape6 = g_info.g_shape6

    input_arr = np.random.randint(
        1, 5, size=(g_shape6, g_whole_shape1, g_whole_shape2, g_whole_shape3, g_whole_shape4)
    ).astype(data_type)
    golden_gnchw = np.zeros(shape=(g_shape6, g_shape0, g_shape1 * g_shape4, g_shape2, g_shape3), dtype=data_type)
    golden_gnchw = input_arr[0:g_shape6, 0:g_shape0, 0:g_shape1 * g_shape4, 0:g_shape2, 0:g_shape3]
    pad_c1 = g_shape1 * g_shape4 - g_whole_shape2
    if pad_c1 > 0:
        pad_width1 = ((0, 0), (0, 0), (0, pad_c1), (0, 0), (0, 0))
        input_arr = np.pad(input_arr, pad_width1, mode="constant", constant_values=0)
    output_arr = gnchw_to_gnc1hwc0(golden_gnchw, g_shape4)
    return input_arr, output_arr


def _golden_gnc1hwc0_to_c1hwn1n0c0(g_info):
    """[G, N, C1, H, W, C0] -> [G, C1, H, W, N1, N0, C0]; pads N when needed."""
    data_type = g_info.data_type
    g_shape3 = g_info.g_shape3
    g_shape4 = g_info.g_shape4
    src_g = g_info.g_whole_shape5
    src_n = g_info.g_whole_shape0
    src_c1 = g_info.g_whole_shape1
    src_h = g_info.g_whole_shape2
    src_w = g_info.g_whole_shape3
    src_c0 = g_info.g_whole_shape4
    dst_n1 = g_shape3
    dst_n0 = g_shape4

    input_ori = np.random.randint(1, 5, size=(src_g, src_n, src_c1, src_h, src_w, src_c0)).astype(data_type)
    pad_n1 = dst_n1 * dst_n0 - g_info.g_whole_shape0
    if pad_n1 > 0:
        pad_width1 = ((0, 0), (0, pad_n1), (0, 0), (0, 0), (0, 0), (0, 0))
        input_arr = np.pad(input_ori, pad_width1, mode="constant", constant_values=0)
    else:
        input_arr = input_ori
    output_arr = input_arr.transpose([0, 2, 3, 4, 1, 5])
    return input_arr, output_arr


def _golden_nc1hwc0_to_nc1c0hw(g_info):
    """[N, C1, H, W, C0] -> [N, C1, C0, H, W]; pads ??? when needed."""
    dtype = g_info.data_type
    src_n = g_info.g_whole_shape0
    src_c1 = g_info.g_whole_shape1
    src_h = g_info.g_whole_shape2
    src_w = g_info.g_whole_shape3
    src_c0 = g_info.g_whole_shape4

    input_arr = np.random.randint(1, 5, size=(src_n, src_c1, src_h, src_w, src_c0)).astype(dtype)

    # NC1HWC0 → NC1C0HW
    # origin index：0(n),1(c1),2(h),3(w),4(C0) → new index ：0,1,4,2,3
    output_arr = np.transpose(input_arr, axes=(0, 1, 4, 2, 3))
    output_arr = np.reshape(output_arr, (src_n, src_c1 * src_c0, src_h, src_w))
    return input_arr, output_arr


def _golden_gnc1hwc0_to_gnc1c0hw(g_info):
    """[G, N, C1, H, W, C0] -> [G, N, C1, C0, H, W]; pads ??? when needed."""
    dtype = g_info.data_type
    src_g = g_info.g_whole_shape0
    src_n = g_info.g_whole_shape1
    src_c1 = g_info.g_whole_shape2
    src_h = g_info.g_whole_shape3
    src_w = g_info.g_whole_shape4
    src_c0 = g_info.g_whole_shape5

    input_arr = np.random.randint(1, 5, size=(src_g, src_n, src_c1, src_h, src_w, src_c0)).astype(dtype)

    # GNC1HWC0 → GNC1C0HW
    # origin index：0(g),1(n),2(c1),3(h),4(w),5(C0) → new index ：0,1,2,5,3,4
    output_arr = np.transpose(input_arr, axes=(0, 1, 2, 5, 3, 4))
    output_arr = np.reshape(output_arr, (src_g, src_n, src_c1 * src_c0, src_h, src_w))
    return input_arr, output_arr


def _golden_nchw2c1hwn1n0c0(g_info):
    """[N, C, H, W] -> [C1*HW_padded, N1, N0, C0]; pads N, C, HW when needed."""
    dtype = g_info.data_type
    dtype_size = np.dtype(dtype).itemsize
    block_size = 32 // dtype_size

    src_n = g_info.g_whole_shape0
    src_c = g_info.g_whole_shape1
    src_h = g_info.g_whole_shape2
    src_w = g_info.g_whole_shape3
    c0 = g_info.g_shape4
    n0 = g_info.g_shape3
    n1 = g_info.g_shape2

    hw = src_h * src_w
    hw_padded = ((hw + block_size - 1) // block_size) * block_size
    c1 = (src_c + c0 - 1) // c0
    n_padded = n1 * n0

    input_arr = np.random.randint(1, 5, size=(src_n, src_c, src_h, src_w)).astype(dtype)
    input_ori = input_arr.copy()

    # Pad N
    pad_n = n_padded - src_n
    if pad_n > 0:
        input_arr = np.pad(input_arr, ((0, pad_n), (0, 0), (0, 0), (0, 0)), mode="constant", constant_values=0)

    # Pad C
    padded_c = c1 * c0
    pad_c = padded_c - src_c
    if pad_c > 0:
        input_arr = np.pad(input_arr, ((0, 0), (0, pad_c), (0, 0), (0, 0)), mode="constant", constant_values=0)

    # Reshape: (N_padded, C1, C0, H, W) -> (N_padded, C1, C0, HW)
    input_arr = input_arr.reshape(n_padded, c1, c0, src_h, src_w).reshape(n_padded, c1, c0, hw)

    # Pad HW
    pad_hw = hw_padded - hw
    if pad_hw > 0:
        input_arr = np.pad(input_arr, ((0, 0), (0, 0), (0, 0), (0, pad_hw)), mode="constant", constant_values=0)

    # (N_padded, C1, C0, HW_padded) -> transpose C0 <-> HW_padded -> (N_padded, C1, HW_padded, C0)
    input_arr = input_arr.transpose(0, 1, 3, 2)

    # Reshape to (N1, N0, C1, HW_padded, C0) -> transpose to (C1, HW_padded, N1, N0, C0)
    input_arr = input_arr.reshape(n1, n0, c1, hw_padded, c0)
    output_arr = input_arr.transpose(2, 3, 0, 1, 4)

    # Flatten to (C1*HW_padded, N1, N0, C0)
    output_arr = output_arr.reshape(c1 * hw_padded, n1, n0, c0)
    return input_ori, output_arr



def _golden_valid(g_info):
    """Valid shape 用例：raw 只表达每维 stride 与容量，valid 决定处理域。
    写入范围 = dst valid 盒；盒内没有源对应的位置（C0 / N0 尾巴）补 0；盒外保持哨兵 -12345。"""
    dt = g_info.data_type
    kind = g_info.valid_kind
    sr, sv = tuple(g_info.src_raw_shape), tuple(g_info.src_valid_shape)
    dr, dv = tuple(g_info.dst_raw_shape), tuple(g_info.dst_valid_shape)
    c0 = 32 // np.dtype(dt).itemsize
    n0 = 16
    sent = np.array(-12345, dtype=dt)
    inp = np.random.randint(1, 5, size=sr).astype(dt)

    if kind in ("NCHW2NC1HWC0",):
        n, c, h, w = sr
        vn, vc, vh, vw = sv
        dvn, dvc1, dvh, dvw, dvc0 = dv
        assert dvc0 == c0 and dvc1 == (vc + c0 - 1) // c0 and (dvn, dvh, dvw) == (vn, vh, vw)
        box = inp[:vn, :vc, :vh, :vw]
        pad = dvc1 * c0 - vc
        if pad > 0:
            box = np.pad(box, ((0, 0), (0, pad), (0, 0), (0, 0)), mode='constant')
        conv = box.reshape(vn, dvc1, c0, vh, vw).transpose(0, 1, 3, 4, 2)
    elif kind == "NC1HWC02NCHW":
        vn, vc1, vh, vw, vc0 = sv
        dvn, dvc, dvh, dvw = dv
        assert vc0 == c0 and (dvn, dvc, dvh, dvw) == (vn, vc1 * vc0, vh, vw)
        box = inp[:vn, :vc1, :vh, :vw, :vc0]
        conv = box.transpose(0, 1, 4, 2, 3).reshape(vn, vc1 * vc0, vh, vw)
    elif kind == "NC1HWC02C1HWN1N0C0":
        vn, vc1, vh, vw, vc0 = sv
        d0, dn1, dn0, dc0 = dv
        assert dc0 == vc0 == c0 and dn0 == n0
        dn1_ = (vn + dn0 - 1) // dn0
        assert (d0, dn1, dn0, dc0) == (vc1 * vh * vw, dn1_, dn0, vc0)
        box = inp[:vn, :vc1, :vh, :vw, :vc0]
        pad = dn1_ * dn0 - vn
        if pad > 0:
            box = np.pad(box, ((0, pad), (0, 0), (0, 0), (0, 0), (0, 0)), mode='constant')
        conv = box.reshape(dn1_, dn0, vc1, vh, vw, vc0).transpose(2, 3, 4, 0, 1, 5).reshape(
            vc1 * vh * vw, dn1_, dn0, vc0
        )
    elif kind == "GNCHW2GNC1HWC0":
        vg, vn, vc, vh, vw = sv
        dgv, dnv, dc1v, dhv, dwv, dc0v = dv
        assert dc0v == c0 and dc1v == (vc + c0 - 1) // c0
        assert (dgv, dnv, dhv, dwv) == (vg, vn, vh, vw)
        box = inp[:vg, :vn, :vc, :vh, :vw]
        pad = dc1v * c0 - vc
        if pad > 0:
            box = np.pad(box, ((0, 0), (0, 0), (0, pad), (0, 0), (0, 0)), mode='constant')
        conv = box.reshape(vg, vn, dc1v, c0, vh, vw).transpose(0, 1, 2, 4, 5, 3)
    elif kind == "GNC1HWC02C1HWN1N0C0":
        vg, vn, vc1, vh, vw, vc0 = sv
        d0, dn1, dn0, dc0 = dv
        assert dc0 == vc0 == c0 and dn0 == n0
        dn1_ = (vn + dn0 - 1) // dn0
        assert (d0, dn1, dn0, dc0) == (vg * vc1 * vh * vw, dn1_, dn0, vc0)
        box = inp[:vg, :vn, :vc1, :vh, :vw, :vc0]
        pad = dn1_ * dn0 - vn
        if pad > 0:
            box = np.pad(box, ((0, 0), (0, pad), (0, 0), (0, 0), (0, 0), (0, 0)), mode='constant')
        conv = box.reshape(vg, dn1_, dn0, vc1, vh, vw, vc0).transpose(0, 3, 4, 5, 1, 2, 6).reshape(
            vg * vc1 * vh * vw, dn1_, dn0, vc0
        )
    elif kind == "NCDHW2C1DHWN1N0C0":
        vn, vc, vd, vh, vw = sv
        d0, dn1, dn0, dc0 = dv
        assert dc0 == c0 and dn0 == n0
        vc1 = (vc + dc0 - 1) // dc0
        dn1_ = (vn + dn0 - 1) // dn0
        assert (d0, dn1, dn0, dc0) == (vd * vc1 * vh * vw, dn1_, dn0, dc0)
        box = inp[:vn, :vc, :vd, :vh, :vw]
        pad = vc1 * dc0 - vc
        if pad > 0:
            box = np.pad(box, ((0, 0), (0, pad), (0, 0), (0, 0), (0, 0)), mode='constant')
        box = box.reshape(vn, vc1, dc0, vd, vh, vw)
        pad = dn1_ * dn0 - vn
        if pad > 0:
            box = np.pad(box, ((0, pad), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)), mode='constant')
        conv = box.reshape(dn1_, dn0, vc1, dc0, vd, vh, vw).transpose(4, 2, 5, 6, 0, 1, 3).reshape(
            vd * vc1 * vh * vw, dn1_, dn0, dc0
        )
    elif kind in ("NCHW2C1HWN1N0C0", "NCHW2C1DHWN1N0C0"):
        vn, vc, vh, vw = sv
        d0, dn1, dn0, dc0 = dv
        assert dc0 == c0 and dn0 == n0
        vc1 = (vc + dc0 - 1) // dc0
        dn1_ = (vn + dn0 - 1) // dn0
        assert (d0, dn1, dn0, dc0) == (vc1 * vh * vw, dn1_, dn0, dc0)
        box = inp[:vn, :vc, :vh, :vw]
        pad = vc1 * dc0 - vc
        if pad > 0:
            box = np.pad(box, ((0, 0), (0, pad), (0, 0), (0, 0)), mode='constant')
        box = box.reshape(vn, vc1, dc0, vh, vw)
        pad = dn1_ * dn0 - vn
        if pad > 0:
            box = np.pad(box, ((0, pad), (0, 0), (0, 0), (0, 0), (0, 0)), mode='constant')
        conv = box.reshape(dn1_, dn0, vc1, dc0, vh, vw).transpose(2, 4, 5, 0, 1, 3).reshape(
            vc1 * vh * vw, dn1_, dn0, dc0
        )
    else:
        raise ValueError(f"unknown valid kind: {kind}")

    output_arr = np.full(dr, sent, dtype=dt)
    output_arr[tuple(slice(0, v) for v in dv)] = conv
    return inp, output_arr


def gen_golden_data(g_info):
    if getattr(g_info, "src_valid_shape", None) is not None:
        input_arr, output_arr = _golden_valid(g_info)
        input_arr.tofile("./input.bin")
        output_arr.tofile("./golden.bin")
        return
    shape1 = g_info.shape
    if shape1 == DataFormat["NCHW2NC1HWC0"].value:
        input_arr, output_arr = _golden_nchw2_nc1hwc0(g_info)
    elif shape1 == DataFormat["NC1HWC02C1HWN1N0C0"].value:
        input_arr, output_arr = _golden_nc1hwc0_to_c1hwn1n0c0(g_info)
    elif shape1 == DataFormat["GNCHW2GNC1HWC0"].value:
        input_arr, output_arr = _golden_gnchw2_gnc1hwc0(g_info)
    elif shape1 == DataFormat["GNC1HWC02C1HWN1N0C0"].value:
        input_arr, output_arr = _golden_gnc1hwc0_to_c1hwn1n0c0(g_info)
    elif shape1 == DataFormat["NC1HWC02NC1C0HW"].value:
        input_arr, output_arr = _golden_nc1hwc0_to_nc1c0hw(g_info)
    elif shape1 == DataFormat["GNC1HWC02GNC1C0HW"].value:
        input_arr, output_arr = _golden_gnc1hwc0_to_gnc1c0hw(g_info)
    elif shape1 == DataFormat["NCHW2C1HWN1N0C0"].value:
        input_arr, output_arr = _golden_nchw2c1hwn1n0c0(g_info)
    else:
        data_type = g_info.data_type
        g_shape3 = g_info.g_shape3
        g_shape4 = g_info.g_shape4
        input_arr = np.random.randint(1, 5, [g_shape3, g_shape4]).astype(data_type)
        output_arr = np.zeros([g_shape3, g_shape4]).astype(data_type)

    input_arr.tofile("./input.bin")  # already pad N
    output_arr.tofile("./golden.bin")


class TTRANSParams:
    def __init__(
        self,
        case_name,
        data_type,
        shape,
        g_shape0=1,
        g_shape1=1,
        g_shape2=1,
        g_shape3=1,
        g_shape4=1,
        g_shape5=1,
        g_whole_shape0=1,
        g_whole_shape1=1,
        g_whole_shape2=1,
        g_whole_shape3=1,
        g_whole_shape4=1,
        g_shape6=1,
        g_whole_shape5=1,
        src_raw_shape=None,
        dst_raw_shape=None,
        src_valid_shape=None,
        dst_valid_shape=None,
        valid_kind=None,
    ):
        self.case_name = case_name
        self.data_type = data_type
        self.shape = shape
        self.g_shape0 = g_shape0
        self.g_shape1 = g_shape1
        self.g_shape2 = g_shape2
        self.g_shape3 = g_shape3
        self.g_shape4 = g_shape4
        self.g_shape5 = g_shape5
        self.g_shape6 = g_shape6
        self.g_whole_shape0 = g_whole_shape0
        self.g_whole_shape1 = g_whole_shape1
        self.g_whole_shape2 = g_whole_shape2
        self.g_whole_shape3 = g_whole_shape3
        self.g_whole_shape4 = g_whole_shape4
        self.g_whole_shape5 = g_whole_shape5
        self.src_raw_shape = src_raw_shape
        self.dst_raw_shape = dst_raw_shape
        self.src_valid_shape = src_valid_shape
        self.dst_valid_shape = dst_valid_shape
        self.valid_kind = valid_kind


if __name__ == "__main__":
    case_params_list = [
        # N, C1, H, W, C0, 1 <- 1, N, C, H, W
        TTRANSParams(
            "TTRANSConvTest.float32_1_32_6_56",
            np.float32,
            DataFormat["NCHW2NC1HWC0"].value,
            1,
            4,
            6,
            56,
            8,
            1,
            1,
            1,
            32,
            6,
            56,
        ),
        TTRANSParams(
            "TTRANSConvTest.int32_1_8_1_8", np.int32, DataFormat["NCHW2NC1HWC0"].value, 1, 1, 1, 8, 8, 1, 1, 1, 8, 1, 8
        ),
        TTRANSParams(
            "TTRANSConvTest.float32_5_57_4_16",
            np.float32,
            DataFormat["NCHW2NC1HWC0"].value,
            5,
            4,
            4,
            16,
            16,
            1,
            1,
            5,
            57,
            4,
            16,
        ),
        TTRANSParams(
            "TTRANSConvTest.half_1_30_2_16",
            np.float16,
            DataFormat["NCHW2NC1HWC0"].value,
            1,
            2,
            2,
            16,
            16,
            1,
            1,
            1,
            30,
            2,
            16,
        ),
        TTRANSParams(
            "TTRANSConvTest.int16_7_53_6_16",
            np.int16,
            DataFormat["NCHW2NC1HWC0"].value,
            7,
            4,
            6,
            16,
            16,
            1,
            1,
            7,
            53,
            6,
            16,
        ),
        TTRANSParams(
            "TTRANSConvTest.int8_3_64_2_64",
            np.int8,
            DataFormat["NCHW2NC1HWC0"].value,
            3,
            2,
            2,
            64,
            32,
            1,
            1,
            3,
            64,
            2,
            64,
        ),
        TTRANSParams(
            "TTRANSConvTest.int8_1_63_2_128",
            np.int8,
            DataFormat["NCHW2NC1HWC0"].value,
            1,
            2,
            2,
            128,
            32,
            1,
            1,
            1,
            63,
            2,
            128,
        ),
        TTRANSParams(
            "TTRANSConvTest.int8_5_58_2_16",
            np.int8,
            DataFormat["NCHW2NC1HWC0"].value,
            5,
            2,
            2,
            16,
            32,
            1,
            1,
            5,
            58,
            2,
            16,
        ),
        TTRANSParams(
            "TTRANSConvTest.uint8_9_87_6_16",
            np.uint8,
            DataFormat["NCHW2NC1HWC0"].value,
            9,
            3,
            6,
            16,
            32,
            1,
            1,
            9,
            87,
            6,
            16,
        ),
        TTRANSParams(
            "TTRANSConvTest.float32_1_32_6_48",
            np.float32,
            DataFormat["NCHW2NC1HWC0"].value,
            1,
            8,
            6,
            48,
            4,
            1,
            1,
            1,
            32,
            6,
            48,
        ),
        TTRANSParams(
            "TTRANSConvTest.uint16_1_26_2_16",
            np.uint16,
            DataFormat["NCHW2NC1HWC0"].value,
            1,
            7,
            2,
            16,
            4,
            1,
            1,
            1,
            26,
            2,
            16,
        ),
        TTRANSParams(
            "TTRANSConvTest.int8_5_18_2_16",
            np.int8,
            DataFormat["NCHW2NC1HWC0"].value,
            5,
            5,
            2,
            16,
            4,
            1,
            1,
            5,
            18,
            2,
            16,
        ),
        # C1, H, W, N1, N0, C0 <- N, C1, H, W, C0
        TTRANSParams(
            "TTRANSConvTest.float32_3_2_2_16_4",
            np.float32,
            DataFormat["NC1HWC02C1HWN1N0C0"].value,
            2,
            2,
            16,
            2,
            2,
            4,
            3,
            2,
            2,
            16,
            4,
        ),
        TTRANSParams(
            "TTRANSConvTest.int32_37_2_3_10_8",
            np.int32,
            DataFormat["NC1HWC02C1HWN1N0C0"].value,
            2,
            3,
            10,
            3,
            16,
            8,
            37,
            2,
            3,
            10,
            8,
        ),
        TTRANSParams(
            "TTRANSConvTest.float16_7_2_1_8_16",
            np.float16,
            DataFormat["NC1HWC02C1HWN1N0C0"].value,
            2,
            1,
            8,
            1,
            16,
            16,
            7,
            2,
            1,
            8,
            16,
        ),
        TTRANSParams(
            "TTRANSConvTest.float16_7_2_1_8_4",
            np.float16,
            DataFormat["NC1HWC02C1HWN1N0C0"].value,
            2,
            1,
            8,
            1,
            16,
            4,
            7,
            2,
            1,
            8,
            4,
        ),
        TTRANSParams(
            "TTRANSConvTest.uint16_45_3_2_7_16",
            np.uint16,
            DataFormat["NC1HWC02C1HWN1N0C0"].value,
            3,
            2,
            7,
            3,
            16,
            16,
            45,
            3,
            2,
            7,
            16,
        ),
        TTRANSParams(
            "TTRANSConvTest.int8_25_5_1_6_32",
            np.int8,
            DataFormat["NC1HWC02C1HWN1N0C0"].value,
            5,
            1,
            6,
            2,
            16,
            32,
            25,
            5,
            1,
            6,
            32,
        ),
        TTRANSParams(
            "TTRANSConvTest.uint8_11_2_7_7_32",
            np.uint8,
            DataFormat["NC1HWC02C1HWN1N0C0"].value,
            2,
            7,
            7,
            1,
            16,
            32,
            11,
            2,
            7,
            7,
            32,
        ),
        # G, N, C1, H, W, C0, 1 <- G, N, C, H, W
        TTRANSParams(
            "TTRANSConvTest.float32_1_1_32_6_56",
            np.float32,
            DataFormat["GNCHW2GNC1HWC0"].value,
            1,
            4,
            6,
            56,
            8,
            1,
            1,
            1,
            32,
            6,
            56,
            1,
            1,
        ),
        TTRANSParams(
            "TTRANSConvTest.int32_4_1_8_1_8",
            np.int32,
            DataFormat["GNCHW2GNC1HWC0"].value,
            1,
            1,
            1,
            8,
            8,
            1,
            1,
            1,
            8,
            1,
            8,
            4,
            4,
        ),
        TTRANSParams(
            "TTRANSConvTest.float32_2_5_30_4_16",
            np.float32,
            DataFormat["GNCHW2GNC1HWC0"].value,
            5,
            2,
            4,
            16,
            16,
            1,
            1,
            5,
            30,
            4,
            16,
            2,
            2,
        ),
        TTRANSParams(
            "TTRANSConvTest.half_1_1_30_2_16",
            np.float16,
            DataFormat["GNCHW2GNC1HWC0"].value,
            1,
            2,
            2,
            16,
            16,
            1,
            1,
            1,
            30,
            2,
            16,
            1,
            1,
        ),
        TTRANSParams(
            "TTRANSConvTest.float32_2_1_32_6_12",
            np.float32,
            DataFormat["GNCHW2GNC1HWC0"].value,
            1,
            8,
            6,
            12,
            4,
            1,
            1,
            1,
            32,
            6,
            12,
            2,
            2,
        ),
        # G, C1, H, W, N1, N0, C0 <- G, N, C1, H, W, C0
        TTRANSParams(
            "TTRANSConvTest.float32_1_3_2_2_16_4",
            np.float32,
            DataFormat["GNC1HWC02C1HWN1N0C0"].value,
            2,
            2,
            16,
            2,
            2,
            4,
            3,
            2,
            2,
            16,
            4,
            1,
            1,
        ),
        TTRANSParams(
            "TTRANSConvTest.float32_2_3_2_2_16_4",
            np.float32,
            DataFormat["GNC1HWC02C1HWN1N0C0"].value,
            2,
            2,
            16,
            2,
            2,
            4,
            3,
            2,
            2,
            16,
            4,
            2,
            2,
        ),
        TTRANSParams(
            "TTRANSConvTest.float32_2_4_2_2_16_4",
            np.float32,
            DataFormat["GNC1HWC02C1HWN1N0C0"].value,
            2,
            2,
            16,
            2,
            2,
            4,
            4,
            2,
            2,
            16,
            4,
            2,
            2,
        ),
        TTRANSParams(
            "TTRANSConvTest.float16_1_7_2_1_8_16",
            np.float16,
            DataFormat["GNC1HWC02C1HWN1N0C0"].value,
            2,
            1,
            8,
            1,
            16,
            16,
            7,
            2,
            1,
            8,
            16,
            1,
            1,
        ),
        TTRANSParams(
            "TTRANSConvTest.float16_4_7_2_1_8_4",
            np.float16,
            DataFormat["GNC1HWC02C1HWN1N0C0"].value,
            2,
            1,
            8,
            1,
            16,
            4,
            7,
            2,
            1,
            8,
            4,
            4,
            4,
        ),
        TTRANSParams(
            "TTRANSConvTest.float32_NC1HWC02NC1C0HW_0",
            np.float32,
            DataFormat["NC1HWC02NC1C0HW"].value,
            1,
            1,
            2,
            4,
            8,
            1,
            1,
            1,
            2,
            4,
            8
        ),
        TTRANSParams(
            "TTRANSConvTest.float32_NC1HWC02NC1C0HW_1",
            np.float32,
            DataFormat["NC1HWC02NC1C0HW"].value,
            2,
            2,
            2,
            2,
            4,
            8,
            2,
            2,
            2,
            4,
            8
        ),
        TTRANSParams(
            "TTRANSConvTest.float32_NC1HWC02NC1C0HW_2",
            np.float32,
            DataFormat["NC1HWC02NC1C0HW"].value,
            2,
            2,
            2,
            3,
            4,
            8,
            2,
            2,
            3,
            4,
            8
        ),
        TTRANSParams(
            "TTRANSConvTest.float32_GNC1HWC02GNC1C0HW_0",
            np.float32,
            DataFormat["GNC1HWC02GNC1C0HW"].value,
            1,
            1,
            1,
            2,
            4,
            8,
            1,
            1,
            1,
            2,
            4,
            1,
            8
        ),
        TTRANSParams(
            "TTRANSConvTest.float32_GNC1HWC02GNC1C0HW_1",
            np.float32,
            DataFormat["GNC1HWC02GNC1C0HW"].value,
            2,
            2,
            2,
            2,
            4,
            8,
            2,
            2,
            2,
            2,
            4,
            1,
            8
        ),
        TTRANSParams(
            "TTRANSConvTest.float32_GNC1HWC02GNC1C0HW_2",
            np.float32,
            DataFormat["GNC1HWC02GNC1C0HW"].value,
            2,
            2,
            2,
            3,
            4,
            8,
            2,
            2,
            2,
            3,
            4,
            1,
            8
        ),
        # NCHW -> FRACTAL_Z
        # g_shape2=N1, g_shape3=N0, g_shape4=C0 <- g_whole_shape0=N, g_whole_shape1=C, g_whole_shape2=H, g_whole_shape3=W
        TTRANSParams(
            "TTRANSConvTest.half_7_32_2_8_FZ",
            np.float16,
            DataFormat["NCHW2C1HWN1N0C0"].value,
            1,
            1,
            1,
            16,
            16,
            1,
            7,
            32,
            2,
            8,
            1,
        ),
        TTRANSParams(
            "TTRANSConvTest.half_4_17_4_1_FZ",
            np.float16,
            DataFormat["NCHW2C1HWN1N0C0"].value,
            1,
            1,
            1,
            16,
            16,
            1,
            4,
            17,
            4,
            1,
            1,
        ),
        TTRANSParams(
            "TTRANSConvTest.float32_5_4_3_8_FZ",
            np.float32,
            DataFormat["NCHW2C1HWN1N0C0"].value,
            1,
            1,
            1,
            16,
            8,
            1,
            5,
            4,
            3,
            8,
            1,
        ),
        TTRANSParams(
            "TTRANSConvTest.int8_4_32_3_7_FZ",
            np.int8,
            DataFormat["NCHW2C1HWN1N0C0"].value,
            1,
            1,
            1,
            16,
            32,
            1,
            4,
            32,
            3,
            7,
            1,
        ),

        # --- Valid shape 用例：与 CPU 侧同参数同语义（两侧 raw 与 valid 每一维都不同）---
        TTRANSParams(
            "TTRANSConvTest.valid_NCHW2NC1HWC0_dst_all_shrunk", np.float32, 0,
            src_raw_shape=(1, 16, 8, 16), dst_raw_shape=(2, 4, 8, 32, 16),
            src_valid_shape=(1, 16, 4, 16), dst_valid_shape=(1, 2, 4, 16, 8),
            valid_kind="NCHW2NC1HWC0",
        ),        TTRANSParams(
            "TTRANSConvTest.valid_NC1HWC02NCHW_all_dims_differ", np.float32, 0,
            src_raw_shape=(3, 5, 7, 9, 16), dst_raw_shape=(3, 32, 7, 9),
            src_valid_shape=(2, 3, 4, 5, 8), dst_valid_shape=(2, 24, 4, 5),
            valid_kind="NC1HWC02NCHW",
        ),        TTRANSParams(
            "TTRANSConvTest.valid_NC1HWC02C1HWN1N0C0_all_dims_differ", np.float32, 0,
            src_raw_shape=(6, 3, 4, 5, 16), dst_raw_shape=(16, 2, 24, 16),
            src_valid_shape=(5, 2, 2, 3, 8), dst_valid_shape=(12, 1, 16, 8),
            valid_kind="NC1HWC02C1HWN1N0C0",
        ),        TTRANSParams(
            "TTRANSConvTest.valid_GNCHW2GNC1HWC0_all_dims_differ", np.float32, 0,
            src_raw_shape=(3, 4, 10, 5, 6), dst_raw_shape=(3, 4, 3, 5, 6, 16),
            src_valid_shape=(2, 3, 8, 4, 5), dst_valid_shape=(2, 3, 1, 4, 5, 8),
            valid_kind="GNCHW2GNC1HWC0",
        ),        TTRANSParams(
            "TTRANSConvTest.valid_GNC1HWC02C1HWN1N0C0_all_dims_differ", np.float32, 0,
            src_raw_shape=(3, 4, 3, 4, 5, 16), dst_raw_shape=(28, 2, 24, 16),
            src_valid_shape=(2, 3, 1, 3, 4, 8), dst_valid_shape=(24, 1, 16, 8),
            valid_kind="GNC1HWC02C1HWN1N0C0",
        ),        TTRANSParams(
            "TTRANSConvTest.valid_NCDHW2C1DHWN1N0C0_all_dims_differ", np.float32, 0,
            src_raw_shape=(3, 10, 3, 4, 5), dst_raw_shape=(32, 2, 24, 16),
            src_valid_shape=(2, 8, 2, 3, 4), dst_valid_shape=(24, 1, 16, 8),
            valid_kind="NCDHW2C1DHWN1N0C0",
        ),        TTRANSParams(
            "TTRANSConvTest.valid_NCHW2C1HWN1N0C0_all_dims_differ", np.float32, 0,
            src_raw_shape=(8, 10, 4, 6), dst_raw_shape=(16, 2, 24, 16),
            src_valid_shape=(6, 8, 3, 4), dst_valid_shape=(12, 1, 16, 8),
            valid_kind="NCHW2C1HWN1N0C0",
        ),        TTRANSParams(
            "TTRANSConvTest.valid_NCHW2C1DHWN1N0C0_all_dims_differ", np.float32, 0,
            src_raw_shape=(6, 18, 6, 5), dst_raw_shape=(32, 2, 24, 16),
            src_valid_shape=(5, 16, 4, 3), dst_valid_shape=(24, 1, 16, 8),
            valid_kind="NCHW2C1DHWN1N0C0",
        ),

        # --- 快路径基准用例：几何对 32B 块对齐，能走向量快路径（同时校验正确性）---
        TTRANSParams(
            "TTRANSConvTest.bench_NCHW2NC1HWC0", np.float32, 0,
            src_raw_shape=(8, 16, 8, 16), dst_raw_shape=(8, 2, 8, 16, 8),
            src_valid_shape=(8, 16, 6, 16), dst_valid_shape=(8, 2, 6, 16, 8),
            valid_kind="NCHW2NC1HWC0",
        ),
        TTRANSParams(
            "TTRANSConvTest.bench_NCHW2NC1HWC0_legacy", np.float32, 0,
            src_raw_shape=(8, 16, 8, 16), dst_raw_shape=(8, 2, 8, 16, 8),
            src_valid_shape=(8, 16, 8, 16), dst_valid_shape=(8, 2, 8, 16, 8),
            valid_kind="NCHW2NC1HWC0",
        ),
        TTRANSParams(
            "TTRANSConvTest.bench_NC1HWC02NCHW", np.float32, 0,
            src_raw_shape=(8, 2, 8, 16, 8), dst_raw_shape=(8, 16, 8, 16),
            src_valid_shape=(8, 2, 6, 16, 8), dst_valid_shape=(8, 16, 6, 16),
            valid_kind="NC1HWC02NCHW",
        ),
        TTRANSParams(
            "TTRANSConvTest.bench_NC1HWC02FZ", np.float32, 0,
            src_raw_shape=(16, 2, 8, 10, 8), dst_raw_shape=(160, 1, 16, 8),
            src_valid_shape=(13, 2, 6, 9, 8), dst_valid_shape=(108, 1, 16, 8),
            valid_kind="NC1HWC02C1HWN1N0C0",
        ),
        TTRANSParams(
            "TTRANSConvTest.bench_NCHW2FZ", np.float32, 0,
            src_raw_shape=(16, 8, 8, 16), dst_raw_shape=(96, 1, 16, 8),
            src_valid_shape=(13, 8, 6, 16), dst_valid_shape=(96, 1, 16, 8),
            valid_kind="NCHW2C1HWN1N0C0",
        ),
    ]

    for case_params in case_params_list:
        case_name = case_params.case_name
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden_data(case_params)
        os.chdir(original_dir)
