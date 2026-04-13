# 矩阵与矩阵-向量指令集

该指令集覆盖 GEMV、matmul 及其累加、bias 和 MX 变体。它们依赖 `Mat`、`Left`、`Right`、`Acc` 等特定 tile role，而不是通用 `Vec` tile。

## 操作

| 操作 | 说明 |
| --- | --- |
| `pto.tgemv` / `tgemv_acc` / `tgemv_bias` / `tgemv_mx` | 矩阵-向量乘及其变体 |
| `pto.tmatmul` / `tmatmul_acc` / `tmatmul_bias` / `tmatmul_mx` | 矩阵乘及其变体 |

## 机制

### Matmul

$$ \mathrm{C}_{i,j} = \sum_k \mathrm{A}_{i,k} \times \mathrm{B}_{k,j} $$

`_acc` 变体在现有累加器上继续累加，`_bias` 变体在结果中加入 bias。

### GEMV

矩阵-向量乘是 matmul 的特例，其中右操作数退化为列向量或行向量。

### MX 变体

MX 变体依赖 `Left` / `Right` 角色和目标 profile 的专用格式支持，常用于 int8 输入配合更宽的累加器。

## 输出形状

| 操作 | 输入 | 输出 |
| --- | --- | --- |
| GEMV | `(M,K)` × `(K,1)` | `(M,1)` |
| Matmul | `(M,K)` × `(K,N)` | `(M,N)` |

## 目标 Profile 支持

| 特性 | CPU | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| 普通 GEMV / matmul | Software/Emulated | Hardware | Hardware |
| MX format | No | No | Yes |
| FP8 / 更激进低精度形式 | No | No | Yes |

## 约束

- 左、右、累加器 tile role 必须匹配
- 维度必须满足 matmul 兼容关系
- `_acc` 变体要求输出 tile 可作为累加器继续参与运算
- `_bias` 变体要求 bias 的 shape 与输出兼容
- MX 变体要求 profile 和布局都支持对应格式

## 不允许的情形

- 使用不兼容的 tile role 组合
- K 维或输出 shape 不匹配
- 在 CPU / A2 / A3 上使用 A5 专属 MX 路径

## 相关页面

- [Tile 指令族](../instruction-families/tile-families_zh.md)
- [Tile 指令集](../instruction-surfaces/tile-instructions_zh.md)
- [位置意图与合法性](../state-and-types/location-intent-and-legality_zh.md)
