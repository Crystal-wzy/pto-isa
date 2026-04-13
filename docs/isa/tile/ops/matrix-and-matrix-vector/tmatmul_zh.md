# TMATMUL

## 指令示意图

![TMATMUL tile operation](../../../../figures/isa/TMATMUL.svg)

## 简介

`TMATMUL` 是 PTO tile 路径里最核心的矩阵乘法指令之一：它从 `Left` / `Right` Tile 读取操作数，在 `Acc` Tile 中生成新的乘加结果。

这条指令的语义是“用当前 `aMatrix` 和 `bMatrix` 计算一个新的输出块”。如果你要在已有累加器内容上继续累加，应改用 `TMATMUL_ACC`；这也是 `TMATMUL` 和 `TMATMUL_ACC` 分开的原因。

## 数学语义

设：

- `M = aMatrix.GetValidRow()`
- `K = aMatrix.GetValidCol()`
- `N = bMatrix.GetValidCol()`

对 `0 <= i < M` 和 `0 <= j < N`：

$$ \mathrm{C}_{i,j} = \sum_{k=0}^{K-1} \mathrm{A}_{i,k} \cdot \mathrm{B}_{k,j} $$

这里的有效计算域由 `aMatrix` 和 `bMatrix` 的 valid region 决定，而不是单纯由静态 Tile 尺寸决定。

累加器内部的精确实现细节，例如某些 target 上的特殊舍入或 profile 限定的数据类型子集，不属于这条通用语义本身，而是目标 profile 的约束。

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../../../../assembly/PTO-AS_zh.md)。

同步形式：

```text
%acc = tmatmul %a, %b : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%c = pto.tmatmul %a, %b : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tmatmul ins(%a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, WaitEvents &... events);

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, WaitEvents &... events);
```

带 `AccPhase` 的模板重载不会改变矩阵乘法本身的算术含义；它只是在目标实现侧附带 unit-flag 相关的选择。

## 约束

### 通用约束

- 静态 shape 必须满足：
  - `TileLeft::Rows == TileRes::Rows`
  - `TileLeft::Cols == TileRight::Rows`
  - `TileRight::Cols == TileRes::Cols`
- Tile 位置必须满足：
  - 左操作数为 `TileType::Left`
  - 右操作数为 `TileType::Right`
  - 结果为 `TileType::Acc`
- 运行时 `m/k/n` 均必须落在 `[1, 4095]`。
- `TMATMUL` 依赖 cube 路径认可的 fractal/layout 组合；Left / Right / Acc 不是可随意替换的通用 Tile 位置。

### A2/A3 实现检查

- A2/A3 支持的 `(CType, AType, BType)` 组合是：
  - `(int32_t, int8_t, int8_t)`
  - `(float, half, half)`
  - `(float, float, float)`
  - `(float, bfloat16_t, bfloat16_t)`
- 对 `float + float + float` 路径，backend 还会根据输入对齐信息决定某些内部实现细节，但不改变外部矩阵乘法语义。

### A5 实现检查

- A5 上，累加器类型必须是 `int32_t` 或 `float`。
- 当累加器为 `int32_t` 时，左右输入都必须是 `int8_t`。
- 当累加器为 `float` 时，当前实现支持以下输入族：
  - `half / half`
  - `bfloat16_t / bfloat16_t`
  - `float / float`
  - 若干 fp8 组合：`float8_e4m3_t` / `float8_e5m2_t`
  - `hifloat8_t / hifloat8_t`
- A5 的 Left / Right / Acc 还要求固定的 fractal 方向：
  - Left：`Loc == Left`，非 row-major，`SFractal == RowMajor`
  - Right：`Loc == Right`，row-major，`SFractal == ColMajor`
  - Acc：`Loc == Acc`，非 row-major，`SFractal == RowMajor`
- 某些具体 A5 目标可能比通用 A5 实现更窄。例如部分具体芯片只开放其中一部分 dtype 组合；这类收窄以目标 profile 为准。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using A = TileLeft<half, 16, 16>;
  using B = TileRight<half, 16, 16>;
  using C = TileAcc<float, 16, 16>;
  A a;
  B b;
  C c;
  TMATMUL(c, a, b);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using A = TileLeft<half, 16, 16>;
  using B = TileRight<half, 16, 16>;
  using C = TileAcc<float, 16, 16>;
  A a;
  B b;
  C c;
  TASSIGN(a, 0x1000);
  TASSIGN(b, 0x2000);
  TASSIGN(c, 0x3000);
  TMATMUL(c, a, b);
}
```

## 相关页面

- [矩阵与矩阵向量指令集](../../matrix-and-matrix-vector_zh.md)
- [TMATMUL_ACC](./tmatmul-acc_zh.md)
- [TMOV](../layout-and-rearrangement/tmov_zh.md)
