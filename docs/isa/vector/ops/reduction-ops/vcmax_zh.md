# pto.vcmax

对向量做最大值归约。

## 语法

```mlir
%result = pto.vcmax %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

A5 已文档化的支持形式：`i16` 到 `i32`，以及 `f16`、`f32`。

## 关键约束

- group reduction 必须保留 VLane 级语义。
- 归约算子与元素类型必须匹配。

## 相关页面

- [归约指令集](../../reduction-ops_zh.md)
