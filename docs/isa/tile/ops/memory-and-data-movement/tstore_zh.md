# TSTORE

## 指令示意图

![TSTORE tile operation](../../../../figures/isa/TSTORE.svg)

## 简介

`TSTORE` 把 Tile 中的数据写回 `GlobalTensor`（GM）。在普通写回之外，它还支持原子写入，以及当前实现中仅针对 `TileType::Acc` 暴露的量化写回重载。

## 数学语义

地址计算取决于 `GlobalTensor` 的 shape / stride 和 Tile 布局。用带基址偏移的二维视角表示时：

$$ \mathrm{dst}_{r_0 + i,\; c_0 + j} = \mathrm{src}_{i,j} $$

真正的写回范围由源 Tile 的 valid region 决定。

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../../../../assembly/PTO-AS_zh.md)。

同步形式：

```text
tstore %t1, %sv_out[%c0, %c0]
```

### AS Level 1（SSA）

```text
pto.tstore %src, %mem : (!pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### AS Level 2（DPS）

```text
pto.tstore ins(%src : !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp` 和 `include/pto/common/constants.hpp`：

```cpp
template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
          typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, WaitEvents &... events);

template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
          typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, uint64_t preQuantScalar, WaitEvents &... events);

template <typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone,
          typename... WaitEvents>
PTO_INST RecordEvent TSTORE_FP(GlobalData &dst, TileData &src, FpTileData &fp, WaitEvents &... events);
```

`preQuantScalar` 和 `TSTORE_FP` 这两类量化写回重载，在当前 A2/A3 与 A5 backend 上只对 `TileType::Acc` 合法；它们不是通用 vec-tile 量化写回接口。

## 约束

### 通用约束

- 写回大小由 `src.GetValidRow()` / `src.GetValidCol()` 决定。
- 目标 `GlobalTensor` 的 shape 必须足以容纳这次写回。

### A2/A3 实现检查

- 源 Tile 的位置类型必须是 `TileType::Vec`、`TileType::Mat` 或 `TileType::Acc`。
- 运行时要求：所有 `dst.GetShape(dim)` 以及 `src.GetValidRow()/GetValidCol()` 都必须大于 `0`。
- 对 `TileType::Vec` / `TileType::Mat`：
  - `TileData::DType` 必须属于：
    `int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`int64_t`、`uint64_t`、`half`、`bfloat16_t`、`float`
  - `sizeof(TileData::DType)` 必须等于 `sizeof(GlobalData::DType)`
  - 布局必须匹配 ND / DN / NZ，或满足单行 / 单列特殊情形
  - 对 `int64_t/uint64_t`，仅支持 `ND -> ND` 与 `DN -> DN`
  - A2/A3 不提供原生的 vec 量化写回路径；若要做 `vec -> GM` 的类型转换或量化，应先显式生成转换后的 vec Tile，再执行同 dtype 的 `TSTORE`
- 对 `TileType::Acc`：
  - 目标布局必须为 ND 或 NZ
  - 源 dtype 必须为 `int32_t` 或 `float`
  - 不量化时，目标 dtype 必须为 `__gm__ int32_t/float/half/bfloat16_t`
  - 静态 shape 约束：
    `1 <= TileData::Cols <= 4095`
    若为 ND，`1 <= TileData::Rows <= 8192`
    若为 NZ，`1 <= TileData::Rows <= 65535` 且 `TileData::Cols % 16 == 0`
  - 运行时要求：`1 <= src.GetValidCol() <= 4095`

### A5 实现检查

- 源 Tile 的位置类型必须为 `TileType::Vec` 或 `TileType::Acc`（A5 不支持 `Mat` store）。
- 对 `TileType::Vec`：
  - `sizeof(TileData::DType)` 必须等于 `sizeof(GlobalData::DType)`
  - `TileData::DType` 必须属于：
    `int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`int64_t`、`uint64_t`、
    `half`、`bfloat16_t`、`float`、`float8_e4m3_t`、`float8_e5m2_t`、`hifloat8_t`、
    `float4_e1m2x2_t`、`float4_e2m1x2_t`
  - 布局必须匹配 ND / DN / NZ，或满足单行 / 单列特殊情形
  - 还存在额外的对齐要求，例如 ND 时 row-major 宽度字节数应为 32 的倍数；DN 时 column-major 高度字节数应为 32 的倍数
- 对 `TileType::Acc`：
  - 目标布局必须为 ND 或 NZ
  - 源 dtype 必须为 `int32_t` 或 `float`
  - 不量化时，目标 dtype 必须为 `__gm__ int32_t/float/half/bfloat16_t`
  - 静态 shape 约束与 A2/A3 一致
  - `AtomicAdd` 还会进一步限制目标 dtype 必须属于支持的原子类型

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_auto(__gm__ T* out) {
  using TileT = Tile<TileType::Vec, T, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor gout(out);
  TileT t;
  TSTORE(gout, t);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_manual(__gm__ T* out) {
  using TileT = Tile<TileType::Vec, T, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor gout(out);
  TileT t;
  TASSIGN(t, 0x1000);
  TSTORE<TileT, GTensor, AtomicType::AtomicAdd>(gout, t);
}
```

## 相关页面

- [内存与数据搬运指令集](../../memory-and-data-movement_zh.md)
- [一致性基线](../../../memory-model/consistency-baseline_zh.md)
- [布局参考](../../../state-and-types/layout_zh.md)
