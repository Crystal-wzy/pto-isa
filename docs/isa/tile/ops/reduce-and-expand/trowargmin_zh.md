# TROWARGMIN

## 指令示意图

![TROWARGMIN tile operation](../../../../figures/isa/TROWARGMIN.svg)

## 简介

`TROWARGMIN` 对输入 Tile 的每一行做“取最小值位置”的归约，结果返回**列索引**。它通常用在后续仍然需要知道“最优元素落在哪一列”的场景里，而不是只要最小值本身的场景。

输出 Tile 只保留一列，因此第 `i` 行的结果 `dst[i, 0]` 表示：源 Tile 第 `i` 行的最小元素位于哪一列。

## 数学语义

设：

- `R = src.GetValidRow()`
- `C = src.GetValidCol()`

对 `0 <= i < R`：

$$ \mathrm{dst}_{i,0} = \underset{0 \le j < C}{\operatorname{argmin}} \; \mathrm{src}_{i,j} $$

如果一行里有多个相同的最小值，具体选择哪一个索引由实现决定。可移植代码不应依赖某个固定的 tie-breaking 结果。

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../../../../assembly/PTO-AS_zh.md)。

同步形式：

```text
%dst = trowargmin %src : !pto.tile<...> -> !pto.tile<...>
```

在 lowering 过程中，backend 可能会引入临时 scratch Tile；C++ 内建接口因此显式接收 `tmp` 操作数。

### AS Level 1（SSA）

```text
%dst = pto.trowargmin %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.trowargmin ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWARGMIN(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events);
```

## 约束

### 通用约束

- `src` 和 `dst` 都必须是 `TileType::Vec`。
- `src` 必须是标准 ND 布局：row-major 且非分形（`BLayout::RowMajor`、`SLayout::NoneBox`）。
- `dst` 必须是以下两类之一：
  - ND 布局，或
  - 单列 DN 布局（`Cols == 1`）。
- `dst` 的元素类型必须是 `uint32_t` 或 `int32_t`。
- `src` 的元素类型必须属于：
  `half`、`float`、`int32_t`、`int16_t`。
- 运行时要求：
  - `src.GetValidRow() != 0`
  - `src.GetValidCol() != 0`
  - `dst.GetValidRow() == src.GetValidRow()`

### A2/A3 实现检查

- 当 `src.GetValidCol() <= elementPerRepeat` 时，A2/A3 可以直接完成行内索引归约，此时 `tmp` 不参与计算。
- 当 `src.GetValidCol() > elementPerRepeat` 时，A2/A3 会使用 `tmp` 做分阶段归约：
  - `validCol <= elementPerRepeat^2` 时走一阶段暂存；
  - `validCol > elementPerRepeat^2` 时走两阶段暂存。
- 因为暂存是按“每一行”展开的，`tmp` 至少应覆盖与 `src` 同样的有效行数。

### A5 与 CPU 实现

- A5 和 CPU 的接口也保留了 `tmp` 参数，但当前实现并不会实际使用它。
- 这意味着 `tmp` 在这些 profile 里主要是接口兼容用，而不是算法必需输入。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, uint32_t, 16, 1, BLayout::ColMajor>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TROWARGMIN(dst, src, tmp);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, uint32_t, 16, 1, BLayout::ColMajor>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(tmp, 0x3000);
  TROWARGMIN(dst, src, tmp);
}
```

## 相关页面

- [归约与扩展指令集](../../reduce-and-expand_zh.md)
- [TROWMIN](./trowmin_zh.md)
- [布局参考](../../../state-and-types/layout_zh.md)
