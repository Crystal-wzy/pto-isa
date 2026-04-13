# TMATMUL_ACC

## 指令示意图

![TMATMUL_ACC tile operation](../../../../figures/isa/TMATMUL_ACC.svg)

## 简介

`TMATMUL_ACC` 是“带累加器输入”的矩阵乘法版本。和 `TMATMUL` 相比，它不会把输出块当成从零开始的新结果，而是把已有累加器内容视为初值，再叠加一轮 `A * B`。

在分块 GEMM 里，这通常就是 K 维循环中间阶段真正要用的形式。

## 数学语义

设：

- `M = aMatrix.GetValidRow()`
- `K = aMatrix.GetValidCol()`
- `N = bMatrix.GetValidCol()`

则：

$$ \mathrm{C}_{\text{out}, i,j} = \mathrm{C}_{\text{in}, i,j} + \sum_{k=0}^{K-1} \mathrm{A}_{i,k} \cdot \mathrm{B}_{k,j} $$

这就是 `TMATMUL_ACC` 与 `TMATMUL` 的唯一核心差别：前者保留已有累加器值，后者把结果视为新生成块。

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../../../../assembly/PTO-AS_zh.md)。

同步形式：

```text
%acc1 = tmatmul.acc %acc0, %a, %b : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%c_out = pto.tmatmul.acc %c_in, %a, %b : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tmatmul.acc ins(%c_in, %a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c_out : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_ACC(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix,
                                 WaitEvents &... events);

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_ACC(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix,
                                 WaitEvents &... events);

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight,
          typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_ACC(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, WaitEvents &... events);
```

最后一个便捷重载等价于把同一个累加器 Tile 同时作为输入和输出。

## 约束

### 通用约束

- `TMATMUL` 的所有 shape、位置、dtype 和 target-profile 约束，在这里同样成立。
- `m/k/n` 仍然取自 `aMatrix.GetValidRow()`、`aMatrix.GetValidCol()` 和 `bMatrix.GetValidCol()`。
- 最稳妥、最可移植的用法，是让 `cInMatrix` 与 `cOutMatrix` 使用同一种 `Acc` Tile 类型，并且在需要时直接传同一个对象。

### 实现说明

- CPU 模拟器按接口字面语义工作：`cInMatrix` 作为显式输入累加器参与计算，结果写入 `cOutMatrix`。
- 当前 NPU backend 的实现路径则更接近“对 `cOutMatrix` 本身继续累加”：
  - A2/A3、A5 和 Kirin9030 的 `TMATMUL_ACC_IMPL` 都只把 `cOutMatrix` 传给底层 `mad` 路径
  - 也就是说，它们不会先把 `cInMatrix` 拷贝到 `cOutMatrix`
- 因而在实际代码里，最常见也最安全的写法就是直接使用共享累加器重载：

```cpp
TMATMUL_ACC(acc, aTile, bTile);
```

如果你显式传入不同的 `cInMatrix` / `cOutMatrix`，CPU 和 NPU backend 当前可能表现不同。

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
  C c0, c1;
  TMATMUL_ACC(c1, c0, a, b);
}
```

### 共享累加器写法

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_accumulate() {
  using A = TileLeft<half, 16, 16>;
  using B = TileRight<half, 16, 16>;
  using C = TileAcc<float, 16, 16>;
  A a;
  B b;
  C acc;
  TMATMUL_ACC(acc, a, b);
}
```

## 相关页面

- [TMATMUL](./tmatmul_zh.md)
- [TMATMUL_BIAS](./tmatmul-bias_zh.md)
- [矩阵与矩阵向量指令集](../../matrix-and-matrix-vector_zh.md)
