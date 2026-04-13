# TROWEXPANDEXPDIF

## 指令示意图

![TROWEXPANDEXPDIF tile operation](../../../../figures/isa/TROWEXPANDEXPDIF.svg)

## 简介

`TROWEXPANDEXPDIF` 先做按行广播减法，再对结果逐元素取指数。它常用于类似 softmax 的按行稳定化路径：先减去某个行基准，再做 `exp`。

## 数学语义

设 `R = dst.GetValidRow()`、`C = dst.GetValidCol()`。记 `s_i` 为第 `i` 行对应的广播标量，则：

$$ \mathrm{dst}_{i,j} = \exp(\mathrm{src0}_{i,j} - s_i) $$

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../../../../assembly/PTO-AS_zh.md)。

同步形式：

```text
%dst = trowexpandexpdif %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.trowexpandexpdif %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.trowexpandexpdif ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDEXPDIF(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1,
                                      WaitEvents &... events);

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDEXPDIF(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp,
                                      WaitEvents &... events);
```

## 约束

- `dst/src0/src1` 的元素类型必须一致，且当前实现只支持 `half` 或 `float`。
- `dst` 必须是 row-major Tile。
- `src0` 或 `src1` 其中之一必须与 `dst` 具有相同的 valid shape。
- 另一侧承担“每行一个标量”的广播角色。

### backend 行为

- A2/A3 的实现实际上就是：
  1. 先执行 `TROWEXPANDSUB`
  2. 再对 `dst` 执行 `TEXP`
- A5 则有专门的 `vexpdif` / `vsub + vexp` 路径，但外部语义一致。
- CPU 模拟器按抽象逐元素语义执行。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using MatT = Tile<TileType::Vec, float, 16, 16>;
  using RowBiasT = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  MatT src0, dst;
  RowBiasT src1;
  TROWEXPANDEXPDIF(dst, src0, src1);
}
```

## 相关页面

- [TROWEXPANDADD](./trowexpandadd_zh.md)
- [TEXP](../../../TEXP_zh.md)
- [归约与扩展指令集](../../reduce-and-expand_zh.md)
