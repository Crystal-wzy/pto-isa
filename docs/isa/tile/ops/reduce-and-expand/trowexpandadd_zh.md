# TROWEXPANDADD

## 指令示意图

![TROWEXPANDADD tile operation](../../../../figures/isa/TROWEXPANDADD.svg)

## 简介

`TROWEXPANDADD` 把一个“按行给出的标量向量”广播到整行，再和 `src0` 做逐元素加法。它适合表达 softmax 前后的按行偏移、量化参数按行展开等模式。

## 数学语义

设 `R = dst.GetValidRow()`、`C = dst.GetValidCol()`。记 `s_i` 为第 `i` 行对应的广播标量，则：

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} + s_i $$

其中 `s_i` 并不总是以同一种布局存放；不同 backend 允许的 `src1` 形态略有区别，但语义上都表示“每行一个标量”。

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../../../../assembly/PTO-AS_zh.md)。

同步形式：

```text
%dst = trowexpandadd %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.trowexpandadd %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.trowexpandadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDADD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDADD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp,
                                   WaitEvents &... events);
```

## 约束

- `dst/src0/src1` 的元素类型必须一致，且当前实现只支持 `half` 或 `float`。
- `dst` 必须是 row-major Tile。
- `src0` 或 `src1` 其中之一必须和 `dst` 拥有相同的 valid shape，并作为逐元素主输入。
- 另一侧则承担“每行一个标量”的角色。

### 无 `tmp` 形式

- A2/A3 / A5 当前允许广播侧 `src1` 使用两种形态之一：
  - row-major，且每行提供 `32 / sizeof(T)` 字节的数据
  - 非 row-major 单列向量（`validCol == 1`）

### 带 `tmp` 形式

- 带 `tmp` 的重载通常对应更窄的广播输入形态，当前 backend 更偏向使用单列广播向量。

### CPU 模拟器

- CPU 会按“每行取一个标量”的抽象语义执行：
  - 如果 `src1` 是单行，则取第 `r` 列
  - 如果 `src1` 是单列，则取第 `r` 行

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using MatT = Tile<TileType::Vec, float, 16, 16>;
  using RowBiasT = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  MatT src0, dst;
  RowBiasT src1;
  TROWEXPANDADD(dst, src0, src1);
}
```

## 相关页面

- [TROWEXPANDMAX](./trowexpandmax_zh.md)
- [TROWEXPANDMIN](./trowexpandmin_zh.md)
- [归约与扩展指令集](../../reduce-and-expand_zh.md)
