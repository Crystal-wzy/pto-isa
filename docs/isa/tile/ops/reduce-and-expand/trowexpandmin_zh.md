# TROWEXPANDMIN

## 指令示意图

![TROWEXPANDMIN tile operation](../../../../figures/isa/TROWEXPANDMIN.svg)

## 简介

`TROWEXPANDMIN` 把一个“按行给出的标量向量”广播到整行，再和 `src0` 做逐元素最小值。它和 `TROWEXPANDMAX` 对称，常用于按行上界裁剪或稳定化步骤。

## 数学语义

设 `R = dst.GetValidRow()`、`C = dst.GetValidCol()`。记 `s_i` 为第 `i` 行对应的广播标量，则：

$$ \mathrm{dst}_{i,j} = \min(\mathrm{src0}_{i,j}, s_i) $$

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../../../../assembly/PTO-AS_zh.md)。

同步形式：

```text
%dst = trowexpandmin %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.trowexpandmin %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.trowexpandmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMIN(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMIN(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp,
                                   WaitEvents &... events);
```

## 约束

- `dst/src0/src1` 的元素类型必须一致，且当前实现只支持 `half` 或 `float`。
- `dst` 必须是 row-major Tile。
- `src0` 或 `src1` 其中之一必须与 `dst` 具有相同的 valid shape。
- 另一侧承担“每行一个标量”的广播角色。

广播侧允许的具体形态与 `TROWEXPANDADD` 相同：不带 `tmp` 时可接受单列广播或一行 32B 数据块；带 `tmp` 时更偏向单列广播。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using MatT = Tile<TileType::Vec, float, 16, 16>;
  using RowBiasT = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  MatT src0, dst;
  RowBiasT src1;
  TROWEXPANDMIN(dst, src0, src1);
}
```

## 相关页面

- [TROWEXPANDMAX](./trowexpandmax_zh.md)
- [TROWEXPANDADD](./trowexpandadd_zh.md)
- [归约与扩展指令集](../../reduce-and-expand_zh.md)
