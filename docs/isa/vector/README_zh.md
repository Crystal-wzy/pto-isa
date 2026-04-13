# 向量 ISA 参考

`pto.v*` 是 PTO ISA 的向量微指令集。它直接暴露向量流水线、向量寄存器、谓词和向量可见 UB 搬运。

## 组织方式

向量参考按指令族组织，具体 per-op 页面位于 `vector/ops/` 下。

## 指令族

- 向量加载存储
- 谓词与物化
- 一元向量操作
- 二元向量操作
- 向量-标量操作
- 转换操作
- 归约操作
- 比较与选择
- 数据重排
- SFU 与 DSA

## 共享约束

- 向量宽度由元素类型决定
- 谓词宽度必须匹配向量宽度
- 对齐、分布和部分高级形式依赖目标 profile
- 向量层没有 tile 级 valid region 语义

## 相关页面

- [向量指令集](../instruction-surfaces/vector-instructions_zh.md)
- [向量指令族](../instruction-families/vector-families_zh.md)
- [指令描述格式](../reference/format-of-instruction-descriptions_zh.md)
