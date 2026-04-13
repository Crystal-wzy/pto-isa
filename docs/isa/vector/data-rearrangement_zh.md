# 数据重排指令集

数据重排指令在寄存器内部或寄存器之间重排数据，而不经过 DMA 或 tile 搬运。它们包括 interleave、deinterleave、slide、compress、expand、pack 和 unpack。

## 常见操作

- `pto.vintlv` / `pto.vdintlv`
- `pto.vslide` / `pto.vshift`
- `pto.vsqz` / `pto.vusqz`
- `pto.vpack` / `pto.vsunpack` / `pto.vzunpack`
- `pto.vperm`
- `pto.vintlvv2` / `pto.vdintlvv2`

## 操作数模型

- `%lhs` / `%rhs`：双输入重排
- `%src`：单输入重排
- `%result`：单个结果向量
- 某些操作会返回成对结果，如 interleave / deinterleave

## 机制

### Interleave / Deinterleave

把两个向量按 even/odd 交织，或把交织后的数据拆分回两路。

### Slide / Shift

在逻辑拼接窗口上抽取一个长度为 `N` 的滑动结果，或在单源向量上做零填充 shift。

### Compress / Expand

按 mask 压紧有效 lane，或把压紧结果再展开到固定寄存器形状。

### Pack / Unpack / Permute

把更宽或更窄的 lane 重新组合，或按索引置换顺序。

## 约束

- 成对输出的顺序必须被保留
- 某些重排只对特定元素宽度合法
- 压缩与展开需要与 mask 语义一致
- `vslide` 的源顺序和提取偏移不能被 lowering 改写

## 不允许的情形

- 把成对输出的顺序当作可交换
- 混淆 zero-fill、preserve 或 wrap 类 shift 语义
- 在未文档化的元素宽度上使用特定 pack / unpack 变体

## 相关页面

- [向量指令族](./vector-families_zh.md)
- [向量指令集](../instruction-surfaces/vector-instructions_zh.md)
