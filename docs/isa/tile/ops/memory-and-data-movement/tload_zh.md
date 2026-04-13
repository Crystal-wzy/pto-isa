# TLOAD

## 指令示意图

![TLOAD tile operation](../../../../figures/isa/TLOAD.svg)

## 简介

`TLOAD` 把 `GlobalTensor`（GM 视图）中的数据装入 Tile。它是 tile 数据路径进入本地缓冲的基本入口，因此除了元素值本身，合法的 shape、layout、valid region 和目标 profile 也都会影响这条指令能否成立。

## 数学语义

地址计算取决于 `GlobalTensor` 的 shape / stride 以及 Tile 的布局。用带基址偏移的二维视角表示时：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{r_0 + i,\; c_0 + j} $$

其中 `r_0`、`c_0` 由 `GlobalTensor` 的当前视图和 tile 的装载位置决定。真正的传输大小由目标 Tile 的 valid region 决定，而不是由 Tile 的物理矩形尺寸决定。

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../../../../assembly/PTO-AS_zh.md)。

同步形式：

```text
%t0 = tload %sv[%c0, %c0] : (!pto.memref<...>, index, index) -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tload %mem : !pto.partition_tensor_view<MxNxdtype> ->
!pto.tile<loc, dtype, rows, cols, blayout, slayout, fractal, pad>
```

### AS Level 2（DPS）

```text
pto.tload ins(%mem : !pto.partition_tensor_view<MxNxdtype>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TLOAD(TileData &dst, GlobalData &src, WaitEvents &... events);
```

## 约束

### 通用约束

- `sizeof(TileData::DType)` 必须与 `sizeof(GlobalData::DType)` 一致。
- 实际传输大小取 `dst.GetValidRow()` 与 `dst.GetValidCol()`。
- `src.GetShape(dim)` 以及 `dst.GetValidRow()/GetValidCol()` 在运行时都必须大于 `0`。

### A2/A3 实现检查

- `TileData::DType` 必须属于：
  `int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`int64_t`、`uint64_t`、`half`、`bfloat16_t`、`float`。
- 目标 Tile 的位置类型必须是 `TileType::Vec` 或 `TileType::Mat`。
- `TileType::Vec` 仅支持布局完全匹配的装载：
  `ND -> ND`、`DN -> DN`、`NZ -> NZ`。
- `TileType::Mat` 还支持 `ND -> NZ` 与 `DN -> ZN`。
- 对 `ND -> NZ` 或 `DN -> ZN`：
  `GlobalData::staticShape[0..2] == 1`，且 `TileData::SFractalSize == 512`。
- 对 `int64_t/uint64_t`，仅支持 `ND -> ND` 与 `DN -> DN`。

### A5 实现检查

- `sizeof(TileData::DType)` 必须为 `1`、`2`、`4` 或 `8` 字节，并与 `GlobalData::DType` 大小一致。
- 对 `int64_t/uint64_t`，`TileData::PadVal` 必须是 `PadValue::Null` 或 `PadValue::Zero`。
- `TileType::Vec` 仅支持以下布局对：
  - ND + row-major + `SLayout::NoneBox`
  - DN + col-major + `SLayout::NoneBox`
  - NZ + `SLayout::RowMajor`
- 对编译期已知 shape 的 row-major ND->ND：
  - `TileData::ValidCol` 必须等于 `GlobalData::staticShape[4]`
  - `TileData::ValidRow` 必须等于 `GlobalData::staticShape[0..3]` 的乘积
- `TileType::Mat` 额外受 `TLoadCubeCheck` 限制，只允许特定 ND / DN / NZ 转换并受 L1 大小约束。
- A5 的 `TileType::Mat` 还覆盖 MX 格式装载：
  - `MX_A_ZZ/MX_A_ND/MX_A_DN -> ZZ`
  - `MX_B_NN/MX_B_ND/MX_B_DN -> NN`
- 对 `MX_A_ZZ/MX_B_NN`：
  `GlobalData::staticShape[3] == 16` 且 `GlobalData::staticShape[4] == 2`。
- 对 `MX_A_ND/MX_A_DN/MX_B_ND/MX_B_DN`：
  `GlobalData::staticShape[0] == 1`、`GlobalData::staticShape[1] == 1`、`GlobalData::staticShape[4] == 2`。
- 对 scaleA，`dst.GetValidCol() % 2 == 0`；
  对 scaleB，`dst.GetValidRow() % 2 == 0`。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_auto(__gm__ T* in) {
  using TileT = Tile<TileType::Vec, T, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor gin(in);
  TileT t;
  TLOAD(t, gin);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_manual(__gm__ T* in) {
  using TileT = Tile<TileType::Vec, T, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor gin(in);
  TileT t;
  TASSIGN(t, 0x1000);
  TLOAD(t, gin);
}
```

## 相关页面

- [内存与数据搬运指令集](../../memory-and-data-movement_zh.md)
- [GlobalTensor 与数据搬运](../../../programming-model/globaltensor-and-data-movement_zh.md)
- [布局参考](../../../state-and-types/layout_zh.md)
