# 二元向量指令集

二元向量指令对两个 `!pto.vreg<NxT>` 输入做 lane 级运算，并产生一个向量结果。它们是 `pto.v*` 中最核心的一组算术与逻辑操作。

## 常见操作

- `pto.vadd` / `pto.vsub`
- `pto.vmul` / `pto.vdiv`
- `pto.vmax` / `pto.vmin`
- `pto.vand` / `pto.vor` / `pto.vxor`
- `pto.vshl` / `pto.vshr`
- `pto.vaddc` / `pto.vsubc`

## 操作数模型

- `%lhs` / `%rhs`：两个源向量寄存器
- `%mask`：控制参与 lane 的谓词
- `%result`：目标向量寄存器

除非某条指令另有说明，`%lhs`、`%rhs` 和 `%result` 的向量宽度与元素类型必须一致。

## 执行模型

这组操作在 `PIPE_V` 上执行。常见的数据路径是：

```text
DMA -> UB -> vlds -> vreg binary op -> vsts -> UB -> DMA
```

在 A2/A3 上，这类操作通常以 `get_buf / rls_buf` 加 `set_flag / wait_flag` 的 producer-consumer 模式与 DMA 配合；在 A5 上则由原生向量流水线执行。

## 机制

对每个激活 lane：

$$ \mathrm{dst}_i = f(\mathrm{lhs}_i, \mathrm{rhs}_i) $$

`%mask` 为非激活的 lane 决定是否参与。域外或未激活 lane 的精确结果，必须以具体指令文档为准。

## 约束

- 两个输入与结果的类型必须兼容
- 掩码宽度必须匹配向量宽度
- 位运算和移位类只对相应整数类型合法
- 某些操作会在 A5 / A2A3 之间有不同的支持子集或性能特征

## 不允许的情形

- 用宽度不匹配的谓词驱动二元向量运算
- 把未文档化的 masked-lane 行为当成稳定契约
- 在不支持的类型组合上使用位移或 carry 变体

## 相关页面

- [向量指令族](./vector-families_zh.md)
- [向量指令集](../instruction-surfaces/vector-instructions_zh.md)
